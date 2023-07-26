use std::{
    borrow::Cow, collections::HashMap, fmt::Write, io, net::SocketAddr, ops::Not, sync::Arc,
};

use actix_web::{
    error::InternalError,
    get,
    http::StatusCode,
    main, middleware,
    web::{Data, Query},
    App, HttpServer, Responder,
};
use ndarray::{Array, Axis, CowArray};
use ort::{tensor::OrtOwnedTensor, Environment, Session, SessionBuilder, Value as OrtValue};
use qdrant_client::{
    prelude::*,
    qdrant::{
        condition::ConditionOneOf, r#match::MatchValue, value::Kind, Condition, FieldCondition,
        Filter, ListValue, LookupLocation, Match, RecommendPoints, RecommendResponse, ScrollPoints,
        ScrollResponse, SearchResponse, Struct, Value,
    },
};
use rust_tokenizers::tokenizer::{BertTokenizer, Tokenizer, TruncationStrategy};
use rustls::{Certificate, PrivateKey, ServerConfig};
use rustls_pemfile::Item;
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

// Ignored as writing to a `String` is always Ok
#[allow(unused_must_use)]
fn write_map(result: &mut String, fields: &HashMap<String, Value>) {
    result.push('{');
    let mut fields_iter = fields.into_iter();
    if let Some((key, value)) = fields_iter.next() {
        write!(result, "\"{}\":", key);
        write_value(result, value);
        for (key, value) in fields_iter {
            write!(result, ",\"{}\":", key);
            write_value(result, value);
        }
    }
    result.push('}');
}

// Ignored as writing to a `String` is always Ok
#[allow(unused_must_use)]
fn write_value(result: &mut String, value: &Value) {
    match &value.kind {
        Some(Kind::DoubleValue(v)) => {
            write!(result, "{}", v);
        }
        Some(Kind::IntegerValue(v)) => {
            write!(result, "{}", v);
        }
        Some(Kind::StringValue(v)) => {
            write!(result, "\"{}\"", v);
        }
        Some(Kind::BoolValue(v)) => {
            write!(result, "{}", v);
        }
        Some(Kind::StructValue(Struct { fields })) => write_map(result, fields),
        Some(Kind::ListValue(ListValue { values })) => {
            result.push('[');
            let mut values_iter = values.into_iter();
            if let Some(value) = values_iter.next() {
                write_value(result, value);
                for value in values_iter {
                    result.push(',');
                    write_value(result, value);
                }
            }
            result.push(']');
        }
        _ => result.push_str("null"),
    }
}

fn add_point(result: &mut String, payload: &HashMap<String, Value>, q: &str) {
    result.push_str("{\"payload:");
    write_map(result, &payload);
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
    section: String,
}

#[get("/search")]
async fn query(
    context: Data<(BertTokenizer, Session, QdrantClient)>,
    search: Query<Search>,
) -> impl Responder {
    let Search { q, section } = search.into_inner();
    let (tokenizer, session, qdrant) = context.get_ref();
    let mut result = String::from("[");
    let points = if q.len() < 5 {
        let mut must = vec![Condition::matches("text", q.clone())];
        if !section.is_empty() {
            must.push(Condition::matches("sections", section));
        }
        match qdrant
            .recommend(&RecommendPoints {
                collection_name: COLLECTION_NAME.to_string(),
                positive: vec![prefix_to_id(&q)],
                filter: Some(Filter {
                    must,
                    ..Default::default()
                }),
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
            Err(e) => return Err(InternalError::new(e, StatusCode::BAD_GATEWAY)),
        }
    } else {
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
            Ok(SearchResponse { result, .. }) => result,
            Err(e) => return Err(InternalError::new(e, StatusCode::BAD_GATEWAY)),
        }
    };
    let mut points = points.into_iter();
    if let Some(first) = points.next() {
        add_point(&mut result, &first.payload, &q);
        for point in points {
            result.push(',');
            add_point(&mut result, &point.payload, &q);
        }
    }
    result.push(']');
    Ok(result)
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
    (if uri.starts_with("https://") {
        let file = std::fs::File::open(
            &*std::env::var("CERTS").map_or(".cacerts.pem".into(), Cow::Owned),
        )?;
        let mut read = io::BufReader::new(file);
        let mut cert = Vec::new();
        let mut key = None;
        while let Some(item) = rustls_pemfile::read_one(&mut read)? {
            match item {
                Item::X509Certificate(data) => cert.push(Certificate(data)),
                Item::RSAKey(data) | Item::PKCS8Key(data) | Item::ECKey(data) => {
                    if let Some(_) = key {
                        return Err(io::Error::new(
                            io::ErrorKind::Other,
                            "multiple private keys found",
                        ));
                    } else {
                        key = Some(PrivateKey(data));
                    }
                }
                _ => {}
            }
        }
        let key = key.expect("missing private key");
        server.bind_rustls(
            addr,
            ServerConfig::builder()
                .with_safe_defaults()
                .with_no_client_auth()
                .with_single_cert(cert, key)
                .unwrap(),
        )?
    } else {
        server.bind(addr)?
    })
    .run()
    .await
}
