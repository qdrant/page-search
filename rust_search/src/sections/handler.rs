use actix_web::web::{Data, Query};
use actix_web::{get, HttpRequest, HttpResponse};
use std::collections::BTreeSet;

use qdrant_client::qdrant::r#match::MatchValue;
use qdrant_client::qdrant::{
    value::Kind, Condition, Document, Filter, PayloadIncludeSelector, PointId, QueryPointsBuilder,
    ScoredPoint, ScrollPointsBuilder, VectorInput,
};
use qdrant_client::Qdrant;
use serde::Deserialize;

use super::models::{Section, SectionSearchResult, slugify_heading};

const SECTION_COLLECTION_NAME: &str = "sections";

fn sections_exact_limit() -> u64 {
    std::env::var("SECTIONS_EXACT_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100)
}

fn sections_search_limit() -> u64 {
    std::env::var("SECTIONS_SEARCH_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10)
}

/// Batch size used when scrolling through points to enumerate sub-pages.
fn sublinks_scroll_batch() -> u32 {
    std::env::var("SECTIONS_SCROLL_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000)
}

fn parse_sections(points: Vec<ScoredPoint>) -> Vec<Section> {
    points
        .into_iter()
        .filter_map(|p| Section::from_payload(p.payload))
        .collect()
}

fn build_conditions(path: &str, query: Option<&str>, section: Option<&str>) -> Vec<Condition> {
    let mut conditions = if section.is_none() && query.is_some() {
        vec![Condition::matches(
            "parent_pages",
            MatchValue::Keyword(path.to_string()),
        )]
    } else {
        vec![Condition::matches(
            "page",
            MatchValue::Keyword(path.to_string()),
        )]
    };

    if let Some(s) = section {
        conditions.push(Condition::matches(
            "parent_sections",
            MatchValue::Keyword(s.to_string()),
        ));
    }

    conditions
}

async fn query_by_filter(
    client: &Qdrant,
    conditions: Vec<Condition>,
    limit: u64,
) -> anyhow::Result<Vec<ScoredPoint>> {
    let result = client
        .query(
            QueryPointsBuilder::new(SECTION_COLLECTION_NAME)
                .filter(Filter::must(conditions))
                .limit(limit)
                .with_payload(true),
        )
        .await?;
    Ok(result.result)
}

const NEURAL_ENCODER: &str = "sentence-transformers/all-MiniLM-L6-v2";

async fn query_by_document(
    client: &Qdrant,
    query: &str,
    conditions: Vec<Condition>,
) -> anyhow::Result<Vec<ScoredPoint>> {
    let result = client
        .query(
            QueryPointsBuilder::new(SECTION_COLLECTION_NAME)
                .query(VectorInput::from(Document::new(query, NEURAL_ENCODER)))
                .filter(Filter::must(conditions))
                .limit(sections_search_limit())
                .with_payload(true),
        )
        .await?;
    Ok(result.result)
}

async fn search_by_query(
    client: &Qdrant,
    query: &str,
    conditions: Vec<Condition>,
    exact_limit: u64,
) -> anyhow::Result<SectionSearchResult> {
    // Try exact slug match first
    let slug = slugify_heading(query);
    let mut exact_conditions = conditions.clone();
    exact_conditions.push(Condition::matches(
        "slug",
        MatchValue::Keyword(slug),
    ));

    let points = query_by_filter(client, exact_conditions, exact_limit).await?;
    if !points.is_empty() {
        return Ok(SectionSearchResult {
            sections: parse_sections(points),
            sublinks: None,
        });
    }

    // Fallback to server-side vector search
    let points = query_by_document(client, query, conditions).await?;
    Ok(SectionSearchResult {
        sections: parse_sections(points),
        sublinks: None,
    })
}

/// Scroll every point under `path` to enumerate its distinct sub-pages in full (issue #30).
async fn fetch_sublinks(client: &Qdrant, path: &str) -> anyhow::Result<Vec<String>> {
    let filter = Filter {
        must: vec![Condition::matches(
            "parent_pages",
            MatchValue::Keyword(path.to_string()),
        )],
        must_not: vec![Condition::matches(
            "page",
            MatchValue::Keyword(path.to_string()),
        )],
        ..Default::default()
    };

    let batch = sublinks_scroll_batch();
    let mut pages: BTreeSet<String> = BTreeSet::new();
    let mut offset: Option<PointId> = None;

    loop {
        let mut builder = ScrollPointsBuilder::new(SECTION_COLLECTION_NAME)
            .filter(filter.clone())
            .limit(batch)
            .with_payload(PayloadIncludeSelector {
                fields: vec!["page".to_string()],
            })
            .with_vectors(false);
        if let Some(o) = offset.take() {
            builder = builder.offset(o);
        }

        let response = client.scroll(builder).await?;

        for point in response.result {
            if let Some(Kind::StringValue(page)) =
                point.payload.get("page").and_then(|v| v.kind.clone())
            {
                pages.insert(page);
            }
        }

        match response.next_page_offset {
            Some(next) => offset = Some(next),
            None => break,
        }
    }

    Ok(pages.into_iter().collect())
}

async fn browse_sections(
    client: &Qdrant,
    path: &str,
    section: Option<&str>,
    conditions: Vec<Condition>,
    exact_limit: u64,
) -> anyhow::Result<Option<SectionSearchResult>> {
    let points = query_by_filter(client, conditions, exact_limit).await?;
    let sections = parse_sections(points);

    let sublinks = if section.is_none() {
        Some(fetch_sublinks(client, path).await?)
    } else {
        None
    };

    let is_empty = sections.is_empty()
        && sublinks.as_ref().is_none_or(|s| s.is_empty());

    if is_empty {
        return Ok(None);
    }

    Ok(Some(SectionSearchResult { sections, sublinks }))
}

async fn search_sections(
    client: &Qdrant,
    query: Option<&str>,
    path: &str,
    section: Option<&str>,
    limit: Option<u64>,
) -> anyhow::Result<Option<SectionSearchResult>> {
    let clean_path = path.trim_matches('/');
    let conditions = build_conditions(clean_path, query, section);
    let exact_limit = limit.unwrap_or_else(sections_exact_limit);

    match query {
        Some(q) => Ok(Some(
            search_by_query(client, q, conditions, exact_limit).await?,
        )),
        None => browse_sections(client, clean_path, section, conditions, exact_limit).await,
    }
}

#[derive(Deserialize)]
struct MdSearch {
    q: Option<String>,
    s: Option<String>,
    limit: Option<u64>,
}

#[get("/md/{path:.*}")]
pub async fn md_handler(
    path: actix_web::web::Path<String>,
    req: HttpRequest,
    query: Query<MdSearch>,
    qdrant: Data<Qdrant>,
) -> HttpResponse {
    let path_str = path.into_inner();
    let qdrant = qdrant.get_ref();

    let result = search_sections(
        qdrant,
        query.q.as_deref(),
        &path_str,
        query.s.as_deref(),
        query.limit,
    )
    .await;

    match result {
        Ok(Some(section_result)) => {
            log::info!("sections={}", section_result.sections.len());

            let conn = req.connection_info();
            let base_url = format!("{}://{}", conn.scheme(), conn.host());
            let request_path = req.uri().path();
            let request_query = req.uri().query();

            let markdown = section_result.to_markdown(request_path, request_query, &base_url);
            HttpResponse::Ok()
                .content_type("text/markdown; charset=utf-8")
                .body(markdown)
        }
        Ok(None) => {
            HttpResponse::NotFound().body("Page or section not found")
        }
        Err(e) => {
            log::error!("Section search error: {}", e);
            HttpResponse::InternalServerError().body(e.to_string())
        }
    }
}
