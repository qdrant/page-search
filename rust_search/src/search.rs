use std::collections::{HashMap, HashSet};
use std::time::Instant;

use actix_web::{
    get,
    http::header::ContentType,
    web::{Data, Query},
    HttpResponse,
};
use futures::StreamExt;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::r#match::MatchValue;
use qdrant_client::qdrant::value::Kind;
use qdrant_client::qdrant::{
    BatchResult, Condition, Filter, LookupLocationBuilder, PointId, QueryBatchPointsBuilder,
    QueryPoints, QueryPointsBuilder, RecommendInput, ScoredPoint, Value, VectorInput,
};
use qdrant_client::Qdrant;
use serde::{Deserialize, Serialize};

use crate::common::{COLLECTION_NAME, PREFIX_COLLECTION_NAME};
use crate::embedding::QueryEncoder;

/// Number of results `/api/search` returns.
pub const SEARCH_LIMIT: u64 = 5;
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
    let pattern = format!(r"(^|\W)({})(.*)", escaped_query);

    let mut regex_builder = regex::RegexBuilder::new(&pattern);
    regex_builder.case_insensitive(true).unicode(true);

    let re = regex_builder.build().expect("Failed to compile regex");

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
    #[serde(default)]
    partition: Option<String>,
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

fn get_list_of_body_tags() -> Vec<String> {
    vec!["p".to_string(), "li".to_string()]
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
    vec![Condition::matches("tag", get_list_of_title_tags())]
}

fn get_recommend_query(
    query: &str,
    conditions: impl IntoIterator<Item = Condition>,
    limit: u64,
) -> QueryPoints {
    QueryPointsBuilder::new(COLLECTION_NAME)
        .query(RecommendInput {
            positive: vec![prefix_to_id(query).into()],
            ..Default::default()
        })
        .filter(Filter::must(conditions))
        .limit(limit)
        .with_payload(true)
        .lookup_from(LookupLocationBuilder::new(PREFIX_COLLECTION_NAME).build())
        .build()
}

fn get_search_query(
    vector: VectorInput,
    conditions: impl IntoIterator<Item = Condition>,
    limit: u64,
) -> QueryPoints {
    QueryPointsBuilder::new(COLLECTION_NAME)
        .query(vector)
        .filter(Filter::must(conditions))
        .limit(limit)
        .with_payload(true)
        .build()
}

fn point_id_to_hash(id: PointId) -> String {
    match id.point_id_options {
        None => "".to_string(),
        Some(PointIdOptions::Num(num)) => format!("{}", num),
        Some(PointIdOptions::Uuid(uuid)) => uuid,
    }
}

fn merge_results(results: Vec<BatchResult>, limit: u64) -> Vec<ScoredPoint> {
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
        if res.len() >= limit as usize {
            break;
        }
    }
    res.into_iter().take(limit as usize).collect()
}

/// The four filters described above, each narrowed by the section and partition, if any.
fn priority_filters(
    query: &str,
    section_condition: Option<Condition>,
    partition_condition: Option<Condition>,
) -> [Vec<Condition>; 4] {
    let mut filters = [
        get_title_text_filter(query),
        get_body_text_filter(query),
        get_title_filter(),
        vec![],
    ];
    for condition in [section_condition, partition_condition]
        .into_iter()
        .flatten()
    {
        for filter in &mut filters {
            filter.push(condition.clone());
        }
    }
    filters
}

async fn recommend_request(
    client: &Qdrant,
    section_condition: Option<Condition>,
    partition_condition: Option<Condition>,
    query: &str,
    limit: u64,
) -> anyhow::Result<Vec<ScoredPoint>> {
    let queries = priority_filters(query, section_condition, partition_condition)
        .into_iter()
        .map(|filter| get_recommend_query(query, filter, limit))
        .collect::<Vec<_>>();

    match client
        .query_batch(QueryBatchPointsBuilder::new(COLLECTION_NAME, queries))
        .await
    {
        Ok(response) => {
            log::debug!("Recommend Qdrant time: {:?}", response.time);
            Ok(merge_results(response.result, limit))
        }
        Err(_) => {
            // TODO: distinguish between 404 and other errors
            Ok(vec![])
        }
    }
}

async fn search_request(
    client: &Qdrant,
    section_condition: Option<Condition>,
    partition_condition: Option<Condition>,
    query: &str,
    vector: VectorInput,
    limit: u64,
) -> anyhow::Result<Vec<ScoredPoint>> {
    let queries = priority_filters(query, section_condition, partition_condition)
        .into_iter()
        .map(|filter| get_search_query(vector.clone(), filter, limit))
        .collect::<Vec<_>>();

    let response = client
        .query_batch(QueryBatchPointsBuilder::new(COLLECTION_NAME, queries))
        .await?;
    log::debug!("Search Qdrant time: {:?}", response.time);
    Ok(merge_results(response.result, limit))
}

async fn search_or_recommend(
    client: &Qdrant,
    encoder: &QueryEncoder,
    section_condition: Option<Condition>,
    partition_condition: Option<Condition>,
    query: &str,
    do_recommend: bool,
    limit: u64,
) -> anyhow::Result<Vec<ScoredPoint>> {
    if do_recommend {
        recommend_request(client, section_condition, partition_condition, query, limit).await
    } else {
        let vector = encoder.encode(query).await?;
        search_request(
            client,
            section_condition,
            partition_condition,
            query,
            vector,
            limit,
        )
        .await
    }
}

/// A condition matching any of `values` in `key`, or `None` when there are no values.
fn any_of(key: &str, values: &[String]) -> Option<Condition> {
    match values {
        [] => None,
        [value] => Some(Condition::matches(key, value.clone())),
        _ => Some(Condition::matches(key, values.to_vec())),
    }
}

/// Split a comma-separated query parameter, dropping empty items.
pub fn split_list(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Search the site, as `/api/search` does, returning up to `limit` matching
/// headings and passages.
///
/// `sections` and `partitions` narrow the search to any of the given values;
/// empty slices don't filter.
pub async fn search(
    client: &Qdrant,
    encoder: &QueryEncoder,
    query: &str,
    sections: &[String],
    partitions: &[String],
    limit: u64,
) -> anyhow::Result<Vec<ScoredPoint>> {
    let section_condition = any_of("sections", sections);
    let partition_condition = any_of("partition", partitions);

    let mut query_stream = vec![];

    if query.len() < 5 {
        query_stream.push(search_or_recommend(
            client,
            encoder,
            section_condition.clone(),
            partition_condition.clone(),
            query,
            true,
            limit,
        ));
    }

    query_stream.push(search_or_recommend(
        client,
        encoder,
        section_condition.clone(),
        partition_condition.clone(),
        query,
        false,
        limit,
    ));

    let mut search_stream = futures::stream::iter(query_stream).buffer_unordered(2);

    while let Some(result) = search_stream.next().await {
        let response = result?;
        if !response.is_empty() {
            return Ok(response);
        }
    }
    Ok(vec![])
}

#[get("/api/search")]
pub async fn query_handler(
    context: Data<Qdrant>,
    encoder: Data<QueryEncoder>,
    search: Query<Search>,
) -> HttpResponse {
    let time_start = Instant::now();

    let Search {
        q,
        section,
        partition,
    } = search.into_inner();

    log::info!("Query: {}", q);

    let sections = split_list(&section);
    let partitions = partition.as_deref().map(split_list).unwrap_or_default();

    let points = match self::search(
        context.get_ref(),
        &encoder,
        &q,
        &sections,
        &partitions,
        SEARCH_LIMIT,
    )
    .await
    {
        Ok(points) => points,
        Err(e) => return HttpResponse::InternalServerError().body(e.to_string()),
    };
    log::debug!("response in {:?}", time_start.elapsed());

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
