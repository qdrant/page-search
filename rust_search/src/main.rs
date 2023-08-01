mod common;

use std::{borrow::Cow, net::SocketAddr, sync::Arc};
use std::collections::{HashMap, HashSet};

use actix_web::{
    get,
    http::header::ContentType,
    main, middleware,
    web::{Data, Query},
    App, HttpResponse, HttpServer,
};
use ort::{Environment, Session, SessionBuilder};
use qdrant_client::{
    prelude::*,
    qdrant::{r#match::MatchValue, value::Kind, Condition,
             Filter, PointId, RecommendPoints,
    },
};
use qdrant_client::prelude::point_id::PointIdOptions;
use qdrant_client::qdrant::{BatchResult, LookupLocation, RecommendBatchPoints, ScoredPoint, SearchBatchPoints};
use regex::Regex;
use rust_tokenizers::tokenizer::BertTokenizer;
use serde::{Deserialize, Serialize};
use crate::common::{COLLECTION_NAME, get_embedding, get_qdrant_url, MODEL_PATH, PREFIX_COLLECTION_NAME};

const VOCAB_PATH: &str = "vocab.txt";
const SPECIAL_TOKEN_PATH: &str = "special_tokens_map.json";
const SEARCH_LIMIT: u64 = 5;
const TEXT_LIMIT: usize = 80;


/// Postprocess search response
///
/// - Highlight matching query in text using `<b>` tag on word boundaries
/// - Limit text length to `TEXT_LIMIT` characters
fn post_process_response_text(text: &str, query: &str) -> String {
    let text = if text.chars().count() > TEXT_LIMIT {
        format!("{}...", &text.chars().take(TEXT_LIMIT).collect::<String>())
    } else {
        text.to_string()
    };

    let escaped_query = regex::escape(query);
    let pattern = format!(r"(^|\W)({})($|\W)", escaped_query);
    let re = Regex::new(&pattern).expect("Failed to compile regex");

    let before = "<b>";
    let after = "</b>";

    let highlighted_text = re.replace_all(&text, |caps: &regex::Captures| {
        format!("{}{}{}{}{}", &caps[1], before, &caps[2], after, &caps[3])
    });

    highlighted_text.to_string()
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

#[derive(Serialize)]
struct ResponseItem {
    pub payload: HashMap<String, Value>,
    pub highlight: String,
}

// There are 4 levels of filtering priority:
// 1. Search with text match in header
// 2. Search with text match in body
// 3. Search with no text match in header
// 4. Search with no text match

// We perform all 4 searches in parallel and then merge results

fn get_list_of_title_tags() -> Vec<String> {
    vec![
        "h1".to_string(),
        "h2".to_string(),
        "h3".to_string(),
        "h4".to_string(),
        "h5".to_string(),
        "h6".to_string(),
    ]
}

fn get_list_of_body_tags() -> Vec<String> {
    vec![
        "p".to_string(),
        "li".to_string(),
    ]
}

fn get_title_text_filter(query: &str) -> Vec<Condition> {
    vec![
        Condition::matches("tag", get_list_of_title_tags()),
        Condition::matches("text", MatchValue::Text(query.to_string())),
    ]
}

fn get_body_text_filter(query: &str) -> Vec<Condition> {
    vec![
        Condition::matches("tag", get_list_of_body_tags()),
        Condition::matches("text", MatchValue::Text(query.to_string())),
    ]
}

fn get_title_filter() -> Vec<Condition> {
    vec![
        Condition::matches("tag", get_list_of_title_tags()),
    ]
}


fn get_recommend_request(query: &str, conditions: impl IntoIterator<Item=Condition>) -> RecommendPoints {
    RecommendPoints {
        collection_name: COLLECTION_NAME.to_string(),
        positive: vec![prefix_to_id(query)],
        filter: Some(Filter::must(conditions)),
        limit: SEARCH_LIMIT,
        with_payload: Some(true.into()),
        lookup_from: Some(LookupLocation {
            collection_name: PREFIX_COLLECTION_NAME.to_string(),
            vector_name: None,
        }),
        ..Default::default()
    }
}

fn get_search_request(vector: &[f32], conditions: impl IntoIterator<Item=Condition>) -> SearchPoints {
    SearchPoints {
        collection_name: COLLECTION_NAME.to_string(),
        vector: vector.to_vec(),
        filter: Some(Filter::must(conditions)),
        limit: SEARCH_LIMIT,
        with_payload: Some(true.into()),
        ..Default::default()
    }
}

fn point_id_to_hash(id: PointId) -> String {
    match id.point_id_options {
        None => "".to_string(),
        Some(PointIdOptions::Num(num)) => format!("{}", num),
        Some(PointIdOptions::Uuid(uuid)) => uuid,
    }
}

fn merge_results(results: Vec<BatchResult>) -> Vec<ScoredPoint> {
    let mut seen = HashSet::new();
    let mut res = vec![];
    for batch_result in results {
        for point in batch_result.result {
            let hashable_id = point.id.clone().map(point_id_to_hash).unwrap_or_default();

            if seen.contains(&hashable_id) {
                continue;
            }
            seen.insert(hashable_id);
            res.push(point);
        }
        if res.len() >= SEARCH_LIMIT as usize {
            break;
        }
    }
    res.into_iter().take(SEARCH_LIMIT as usize).collect()
}

async fn recommend_request(client: &QdrantClient, section_condition: Option<Condition>, query: &str) -> Vec<ScoredPoint> {
    let mut title_text_filter = get_title_text_filter(query);
    let mut body_text_filter = get_body_text_filter(query);
    let mut title_filter = get_title_filter();
    let mut no_text_filter = vec![];

    if let Some(section_condition) = section_condition {
        title_text_filter.push(section_condition.clone());
        body_text_filter.push(section_condition.clone());
        title_filter.push(section_condition.clone());
        no_text_filter.push(section_condition);
    }

    match client.recommend_batch(
        &RecommendBatchPoints {
            collection_name: COLLECTION_NAME.to_string(),
            recommend_points: vec![
                get_recommend_request(query, title_text_filter),
                get_recommend_request(query, body_text_filter),
                get_recommend_request(query, title_filter),
                get_recommend_request(query, no_text_filter),
            ],
            read_consistency: None,
        }
    ).await {
        Ok(response) => {
            merge_results(response.result)
        }
        Err(_) => { // TODO: distinguish between 404 and other errors
            vec![]
        }
    }
}

async fn search_request(client: &QdrantClient, section_condition: Option<Condition>, query: &str, vector: Vec<f32>) -> Result<Vec<ScoredPoint>, HttpResponse> {
    let mut title_text_filter = get_title_text_filter(query);
    let mut body_text_filter = get_body_text_filter(query);
    let mut title_filter = get_title_filter();
    let mut no_text_filter = vec![];

    if let Some(section_condition) = section_condition {
        title_text_filter.push(section_condition.clone());
        body_text_filter.push(section_condition.clone());
        title_filter.push(section_condition.clone());
        no_text_filter.push(section_condition);
    }

    match client.search_batch_points(
        &SearchBatchPoints {
            collection_name: COLLECTION_NAME.to_string(),
            search_points: vec![
                get_search_request(&vector, title_text_filter),
                get_search_request(&vector, body_text_filter),
                get_search_request(&vector, title_filter),
                get_search_request(&vector, no_text_filter),
            ],
            read_consistency: None,
        }
    ).await {
        Ok(response) => {
            Ok(merge_results(response.result))
        }
        Err(e) => Err(HttpResponse::InternalServerError().body(e.to_string()))
    }
}

#[get("/api/search")]
async fn query_handler(
    context: Data<(BertTokenizer, Session, QdrantClient)>,
    search: Query<Search>,
) -> HttpResponse {
    let Search { q, section } = search.into_inner();

    println!("Query: {}", q);

    let (tokenizer, session, qdrant) = context.get_ref();

    let section_condition = if section.is_empty() {
        None
    } else {
        Some(Condition::matches(
            "sections",
            MatchValue::Keyword(section.clone()),
        ))
    };

    let mut points = if q.len() < 5 {
        recommend_request(qdrant, section_condition.clone(), &q).await
    } else {
        Vec::new()
    };
    if points.is_empty() {
        let vector = get_embedding(tokenizer, session, &q);
        match search_request(qdrant, section_condition, &q, vector).await
        {
            Ok(search_response) => points = search_response,
            Err(e) => return e,
        }
    };

    // Postprocess search results
    let response: Vec<_> = points
        .into_iter()
        .map(|point| {
            let highlight = if let Some(Kind::StringValue(text)) = &point.payload.get("text").and_then(|v| v.kind.as_ref()) {
                post_process_response_text(text, &q)
            } else {
                "".to_string()
            };

            ResponseItem {
                payload: point.payload,
                highlight,
            }
        }).collect();

    HttpResponse::Ok()
        .insert_header(ContentType::json())
        .body(
            serde_json::to_string(&response)
                .expect("Failed to serialize response"),
        )
}

#[main]
async fn main() -> std::io::Result<()> {
    env_logger::init();
    let uri = std::env::var("SERVICE_URL").map_or("0.0.0.0:8005".into(), Cow::Owned);
    let qdrant_url = get_qdrant_url();
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
            .service(query_handler)
    });
    server.bind(addr)?.run().await
}
