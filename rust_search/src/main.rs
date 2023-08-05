mod common;

use std::collections::HashMap;
use std::time::Instant;
use std::{borrow::Cow, net::SocketAddr, sync::Arc};

use crate::common::{
    get_embedding, get_qdrant_url, COLLECTION_NAME, MODEL_PATH, PREFIX_COLLECTION_NAME,
};
use actix_cors::Cors;
use actix_web::{
    get,
    http::header::ContentType,
    main, middleware,
    web::{Data, Query},
    App, HttpResponse, HttpServer,
};
use futures::StreamExt;
use ort::{Environment, Session, SessionBuilder};
use qdrant_client::qdrant::{
    BatchResult, LookupLocation, RecommendBatchPoints, ScoredPoint, SearchBatchPoints,
};
use qdrant_client::{
    prelude::*,
    qdrant::{r#match::MatchValue, value::Kind, Condition, Filter, PointId, RecommendPoints},
};
use rust_tokenizers::tokenizer::BertTokenizer;
use serde::{Deserialize, Serialize};

const VOCAB_PATH: &str = "vocab.txt";
const SPECIAL_TOKEN_PATH: &str = "special_tokens_map.json";
const SEARCH_LIMIT: u64 = 5;
const TEXT_LIMIT: usize = 80;

/// Postprocess search response
///
/// - Highlight matching query in text using `<b>` tag on word boundaries
/// - Limit text length to `TEXT_LIMIT` characters
fn post_process_response_text(text: &str, query: &str) -> String {
    // avoid counting chars if the length is lower than our limit
    let (text, ellipsis) = if text.len() < TEXT_LIMIT {
        (text, "")
    } else if let Some((n, _)) = text.char_indices().nth(TEXT_LIMIT) {
        // text is longer so cut at limit and keep ellipsis to add later
        (&text[..n], "...")
    } else {
        (text, "")
    };

    let escaped_query = regex::escape(query);
    let pattern = format!(r"\b({})\b", escaped_query);

    let mut regex_builder = regex::RegexBuilder::new(&pattern);
    regex_builder.case_insensitive(true).unicode(true);

    let re = regex_builder.build().expect("Failed to compile regex");

    let highlighted_text = re.replace_all(text, |caps: &regex::Captures| {
        format!("<b>{}</b>", &caps[0])
    });

    highlighted_text.to_string() + ellipsis
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

#[derive(Serialize)]
struct Response {
    pub result: Vec<ResponseItem>,
    pub time: f64,
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

fn create_filters(query: &str, section: Option<Condition>) -> [Filter; 4] {
    let query_cond = Condition::matches("text", MatchValue::Text(query.to_string()));
    let title_cond = Condition::matches("tag", get_list_of_title_tags());
    let mut title_text_filter = Filter::must([query_cond.clone(), title_cond.clone()]);
    let mut body_text_filter = Filter::must([query_cond.clone()]);
    let mut title_filter = Filter::must([title_cond.clone()]);
    let mut no_text_filter = Filter::must_not([query_cond.clone(), title_cond.clone()]);
    body_text_filter.must_not.push(title_cond);
    title_filter.must_not.push(query_cond);

    if let Some(section_condition) = section {
        title_text_filter.must.push(section_condition.clone());
        body_text_filter.must.push(section_condition.clone());
        title_filter.must.push(section_condition.clone());
        no_text_filter.must.push(section_condition);
    }

    [
        title_text_filter,
        body_text_filter,
        title_filter,
        no_text_filter,
    ]
}

fn get_recommend_request(query: &str, filter: Filter) -> RecommendPoints {
    RecommendPoints {
        collection_name: COLLECTION_NAME.to_string(),
        positive: vec![prefix_to_id(query)],
        filter: Some(filter),
        limit: SEARCH_LIMIT,
        with_payload: Some(true.into()),
        lookup_from: Some(LookupLocation {
            collection_name: PREFIX_COLLECTION_NAME.to_string(),
            vector_name: None,
        }),
        ..Default::default()
    }
}

fn get_search_request(vector: &[f32], filter: Filter) -> SearchPoints {
    SearchPoints {
        collection_name: COLLECTION_NAME.to_string(),
        vector: vector.to_vec(),
        filter: Some(filter),
        limit: SEARCH_LIMIT,
        with_payload: Some(true.into()),
        ..Default::default()
    }
}

fn merge_results(results: Vec<BatchResult>) -> Vec<ScoredPoint> {
    results
        .into_iter()
        .flat_map(|r| r.result)
        .take(SEARCH_LIMIT as usize)
        .collect()
}

async fn recommend_request(
    client: &QdrantClient,
    section_condition: Option<Condition>,
    query: &str,
) -> Result<Vec<ScoredPoint>, HttpResponse> {
    let [title_text_filter, body_text_filter, title_filter, no_text_filter] =
        create_filters(query, section_condition);

    match client
        .recommend_batch(&RecommendBatchPoints {
            collection_name: COLLECTION_NAME.to_string(),
            recommend_points: vec![
                get_recommend_request(query, title_text_filter),
                get_recommend_request(query, body_text_filter),
                get_recommend_request(query, title_filter),
                get_recommend_request(query, no_text_filter),
            ],
            read_consistency: None,
        })
        .await
    {
        Ok(response) => {
            log::debug!("Recommend Qdrant time: {:?}", response.time);
            Ok(merge_results(response.result))
        }
        Err(_) => {
            // TODO: distinguish between 404 and other errors
            Ok(vec![])
        }
    }
}

async fn search_request(
    client: &QdrantClient,
    section_condition: Option<Condition>,
    query: &str,
    vector: Vec<f32>,
) -> Result<Vec<ScoredPoint>, HttpResponse> {
    let [title_text_filter, body_text_filter, title_filter, no_text_filter] =
        create_filters(query, section_condition);

    match client
        .search_batch_points(&SearchBatchPoints {
            collection_name: COLLECTION_NAME.to_string(),
            search_points: vec![
                get_search_request(&vector, title_text_filter),
                get_search_request(&vector, body_text_filter),
                get_search_request(&vector, title_filter),
                get_search_request(&vector, no_text_filter),
            ],
            read_consistency: None,
        })
        .await
    {
        Ok(response) => {
            log::debug!("Search Qdrant time: {:?}", response.time);
            Ok(merge_results(response.result))
        }
        Err(e) => Err(HttpResponse::InternalServerError().body(e.to_string())),
    }
}

async fn search_or_recommend(
    client: &QdrantClient,
    tokenizer: &BertTokenizer,
    session: &Session,
    section_condition: Option<Condition>,
    query: &str,
    do_recommend: bool,
) -> Result<Vec<ScoredPoint>, HttpResponse> {
    if do_recommend {
        recommend_request(client, section_condition, query).await
    } else {
        let vector = get_embedding(tokenizer, session, query);
        search_request(client, section_condition, query, vector).await
    }
}

#[get("/api/search")]
async fn query_handler(
    context: Data<(BertTokenizer, Session, QdrantClient)>,
    search: Query<Search>,
) -> HttpResponse {
    let time_start = Instant::now();

    let Search { q, section } = search.into_inner();

    log::info!("Query: {}", q);

    let (tokenizer, session, qdrant) = context.get_ref();

    let section_condition = if section.is_empty() {
        None
    } else {
        Some(Condition::matches(
            "sections",
            MatchValue::Keyword(section.clone()),
        ))
    };

    let mut query_stream = vec![];

    if q.len() < 5 {
        query_stream.push(search_or_recommend(
            qdrant,
            tokenizer,
            session,
            section_condition.clone(),
            &q,
            true,
        ));
    }

    query_stream.push(search_or_recommend(
        qdrant,
        tokenizer,
        session,
        section_condition.clone(),
        &q,
        false,
    ));

    let mut search_stream = futures::stream::iter(query_stream).buffer_unordered(2);

    let mut points = vec![];
    while let Some(result) = search_stream.next().await {
        log::debug!("response in {:?}", time_start.elapsed());
        match result {
            Ok(response) => {
                if !response.is_empty() {
                    points.extend(response);
                    break;
                }
            }
            Err(err) => return err,
        }
    }

    // Postprocess search results
    let response_items: Vec<_> = points
        .into_iter()
        .map(|point| {
            let highlight = if let Some(Kind::StringValue(text)) =
                &point.payload.get("text").and_then(|v| v.kind.as_ref())
            {
                post_process_response_text(text, &q)
            } else {
                "".to_string()
            };

            ResponseItem {
                payload: point.payload,
                highlight,
            }
        })
        .collect();

    HttpResponse::Ok().insert_header(ContentType::json()).body(
        serde_json::to_string(&Response {
            result: response_items,
            time: time_start.elapsed().as_micros() as f64 / 1_000_000.0,
        })
        .expect("Failed to serialize response"),
    )
}

#[main]
async fn main() -> std::io::Result<()> {
    let mut log_builder = env_logger::Builder::new();
    log_builder.parse_filters(&std::env::var("SERVICE_LOG_LEVEL").unwrap_or("info".into()));
    log_builder.init();

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
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header();

        App::new()
            .app_data(context.clone())
            .wrap(cors)
            .wrap(middleware::Logger::default())
            .service(query_handler)
    });
    server.bind(addr)?.run().await
}
