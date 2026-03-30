use actix_web::web::{Data, Query};
use actix_web::{get, HttpRequest, HttpResponse};
use qdrant_client::qdrant::r#match::MatchValue;
use qdrant_client::qdrant::{
    Condition, Document, FacetCountsBuilder, Filter, QueryPointsBuilder, ScoredPoint, VectorInput,
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
) -> anyhow::Result<SectionSearchResult> {
    // Try exact slug match first
    let slug = slugify_heading(query);
    let mut exact_conditions = conditions.clone();
    exact_conditions.push(Condition::matches(
        "slug",
        MatchValue::Keyword(slug),
    ));

    let points = query_by_filter(client, exact_conditions, sections_exact_limit()).await?;
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

async fn fetch_sublinks(client: &Qdrant, path: &str) -> anyhow::Result<Vec<String>> {
    let facet_result = client
        .facet(
            FacetCountsBuilder::new(SECTION_COLLECTION_NAME, "page")
                .filter(Filter {
                    must: vec![Condition::matches(
                        "parent_pages",
                        MatchValue::Keyword(path.to_string()),
                    )],
                    must_not: vec![Condition::matches(
                        "page",
                        MatchValue::Keyword(path.to_string()),
                    )],
                    ..Default::default()
                })
                .limit(sections_exact_limit()),
        )
        .await?;

    let mut links: Vec<String> = facet_result
        .hits
        .into_iter()
        .filter_map(|hit| {
            hit.value.and_then(|v| match v.variant? {
                qdrant_client::qdrant::facet_value::Variant::StringValue(s) => Some(s),
                _ => None,
            })
        })
        .collect();
    links.sort();
    Ok(links)
}

async fn browse_sections(
    client: &Qdrant,
    path: &str,
    section: Option<&str>,
    conditions: Vec<Condition>,
) -> anyhow::Result<Option<SectionSearchResult>> {
    let points = query_by_filter(client, conditions, sections_exact_limit()).await?;
    let sections = parse_sections(points);

    let sublinks = if section.is_none() {
        Some(fetch_sublinks(client, path).await?)
    } else {
        None
    };

    let is_empty = sections.is_empty()
        && sublinks.as_ref().map_or(true, |s| s.is_empty());

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
) -> anyhow::Result<Option<SectionSearchResult>> {
    let clean_path = path.trim_matches('/');
    let conditions = build_conditions(clean_path, query, section);

    match query {
        Some(q) => Ok(Some(search_by_query(client, q, conditions).await?)),
        None => browse_sections(client, clean_path, section, conditions).await,
    }
}

#[derive(Deserialize)]
struct MdSearch {
    q: Option<String>,
    s: Option<String>,
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
