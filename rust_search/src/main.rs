use std::{borrow::Cow, collections::HashMap, net::SocketAddr, ops::Not, sync::Arc};

use actix_web::{
    get,
    http::header::ContentType,
    main, middleware,
    web::{Data, Query},
    App, HttpResponse, HttpServer,
};
use ndarray::{Array, Axis, CowArray};
use ort::{tensor::OrtOwnedTensor, Environment, Session, SessionBuilder, Value as OrtValue};
use qdrant_client::{
    prelude::*,
    qdrant::{
        condition::ConditionOneOf, r#match::MatchValue, value::Kind, Condition, FieldCondition,
        Filter, LookupLocation, Match, PointId, RecommendPoints, RecommendResponse, SearchResponse,
        Value,
    },
};
use rust_tokenizers::tokenizer::{BertTokenizer, Tokenizer, TruncationStrategy};
use serde::Deserialize;

const MODEL_PATH: &str = "all-MiniLM-L6-v2.onnx";
const VOCAB_PATH: &str = "vocab.txt";
const SPECIAL_TOKEN_PATH: &str = "special_tokens_map.json";
const COLLECTION_NAME: &str = "site";
const SEARCH_LIMIT: u64 = 5;
const TEXT_LIMIT: usize = 80;

fn take_n_chars(text: &str, limit: usize) -> Cow<'_, str> {
    if text.len() < limit {
        text.into()
    } else if let Some((i, _)) = text.char_indices().nth(limit) {
        Cow::Owned(format!("{}...", &text[..i]))
    } else {
        text.into()
    }
}

fn highlight(result: &mut String, text: &str, q: &str) {
    let mut split = text.splitn(2, q);
    if let (Some(before), Some(after)) = (split.next(), split.next()) {
        result.extend([before, "<b>", q, "</b>", after]);
    } else {
        result.push_str(text);
    }
}

fn add_point(result: &mut String, payload: &HashMap<String, Value>, q: &str) {
    result.push_str("{\"payload\":");
    result.push_str(&serde_json::to_string(payload).unwrap()); // should not be able to fail
    result.push_str(",\"highlight\":\"");
    if let Some(Kind::StringValue(text)) = &payload.get("text").and_then(|v| v.kind.as_ref()) {
        highlight(result, &take_n_chars(text, TEXT_LIMIT), q);
    }
    result.push_str("\"}");
}

fn prefix_to_id(prefix: &str) -> PointId {
    let len = prefix.len();
    PointId::from(u64::from_le_bytes(if len < 8 {
        let mut result = [0_u8; 8];
        result[..len].copy_from_slice(prefix.as_bytes());
        result
    } else {
        prefix.as_bytes()[..8].try_into().unwrap()
    }))
}

#[derive(Deserialize)]
struct Search {
    q: String,
    #[serde(default)]
    section: String,
}

#[get("/api/search")]
async fn query(
    context: Data<(BertTokenizer, Session, QdrantClient)>,
    search: Query<Search>,
) -> HttpResponse {
    let Search { q, section } = search.into_inner();
    let (tokenizer, session, qdrant) = context.get_ref();
    let mut points = if q.len() < 5 {
        let filter = if section.is_empty() {
            None
        } else {
            Some(Filter::all([Condition::matches(
                "sections",
                MatchValue::Keyword(section.clone()),
            )]))
        };
        match qdrant
            .recommend(&RecommendPoints {
                collection_name: COLLECTION_NAME.to_string(),
                positive: vec![prefix_to_id(&q)],
                filter,
                limit: SEARCH_LIMIT,
                with_payload: Some(true.into()),
                lookup_from: Some(LookupLocation {
                    collection_name: "prefix-cache".to_string(),
                    vector_name: None,
                }),
                ..Default::default()
            })
            .await
        {
            Ok(RecommendResponse { result, .. }) => result,
            Err(_) => Vec::new(),
        }
    } else {
        Vec::new()
    };
    if points.is_empty() {
        // tokenize
        let mut encoding = tokenizer.encode(&q, None, 512, &TruncationStrategy::LongestFirst, 1);
        let alloc = session.allocator();
        let token_ids = std::mem::take(&mut encoding.token_ids);
        let shape = (1, token_ids.len());
        let token_ids = Array::from_shape_vec(shape, token_ids).unwrap();
        let attentions = Array::from_elem(shape, 1_i64);
        let type_ids = Array::from_elem(shape, 0_i64);
        // embed
        let output: OrtOwnedTensor<f32, _> = session
            .run(vec![
                OrtValue::from_array(alloc, &CowArray::from(token_ids.into_dyn())).unwrap(),
                OrtValue::from_array(alloc, &CowArray::from(attentions.into_dyn())).unwrap(),
                OrtValue::from_array(alloc, &CowArray::from(type_ids.into_dyn())).unwrap(),
            ])
            .unwrap()[0]
            .try_extract()
            .unwrap();
        let pooled = output.view().mean_axis(Axis(1)).unwrap();
        let vector = pooled.as_slice().unwrap().to_vec();
        // query qdrant
        match qdrant
            .search_points(&SearchPoints {
                collection_name: COLLECTION_NAME.to_string(),
                vector,
                filter: section.is_empty().not().then(|| Filter {
                    must: vec![Condition {
                        condition_one_of: Some(ConditionOneOf::Field(FieldCondition {
                            key: "sections".into(),
                            r#match: Some(Match {
                                match_value: Some(MatchValue::Keyword(section)),
                            }),
                            ..Default::default()
                        })),
                    }],
                    ..Default::default()
                }),
                limit: SEARCH_LIMIT,
                with_payload: Some(true.into()),
                ..Default::default()
            })
            .await
        {
            Ok(SearchResponse { result, .. }) => points = result,
            Err(e) => return HttpResponse::InternalServerError().body(e.to_string()),
        }
    };
    let mut result = String::from("[");
    let mut points = points.into_iter();
    if let Some(first) = points.next() {
        add_point(&mut result, &first.payload, &q);
        for point in points {
            result.push(',');
            add_point(&mut result, &point.payload, &q);
        }
    }
    result.push(']');
    HttpResponse::Ok()
        .insert_header(ContentType::json())
        .body(result)
}

#[main]
async fn main() -> std::io::Result<()> {
    env_logger::init();
    let uri = std::env::var("SERVICE_URL").map_or("127.0.0.1:5497".into(), Cow::Owned);
    let qdrant_url = std::env::var("QDRANT_URL").map_or("http://localhost:6334".into(), Cow::Owned);
    let api_key = std::env::var("QDRANT_API_KEY");
    let addr: SocketAddr = uri.parse().expect("malformed URI");
    let tokenizer = BertTokenizer::from_file_with_special_token_mapping(
        VOCAB_PATH,
        true,
        false,
        SPECIAL_TOKEN_PATH,
    )
    .unwrap();
    let env = Arc::new(Environment::builder().build().unwrap());
    let session = SessionBuilder::new(&env)
        .unwrap()
        .with_model_from_file(MODEL_PATH)
        .unwrap();
    let mut config = QdrantClientConfig::from_url(&qdrant_url);
    if let Ok(key) = &api_key {
        config.set_api_key(key);
    }
    let qdrant = QdrantClient::new(Some(config)).unwrap();
    qdrant.health_check().await.unwrap();
    let context = Data::new((tokenizer, session, qdrant));
    let server = HttpServer::new(move || {
        App::new()
            .app_data(context.clone())
            .wrap(middleware::Logger::default())
            .service(query)
    });
    server.bind(addr)?.run().await
}
