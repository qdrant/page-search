use actix_web::web::{Data, Query};
use actix_web::{get, HttpRequest, HttpResponse};
use std::collections::BTreeSet;

use qdrant_client::qdrant::r#match::MatchValue;
use qdrant_client::qdrant::{
    facet_value, value::Kind, Condition, FacetCountsBuilder, Filter, PayloadIncludeSelector,
    PointId, QueryPointsBuilder, ScoredPoint, ScrollPointsBuilder,
};
use qdrant_client::Qdrant;
use serde::Deserialize;

use super::models::{Section, SectionSearchResult, slugify_heading};
use crate::embedding::QueryEncoder;
use crate::similar_paths::{path_words, similar_paths};

const SECTION_COLLECTION_NAME: &str = "sections";
/// Most pages suggested for a missing path.
const MAX_SUGGESTIONS: usize = 5;
/// Above the number of pages, so the facet returns all of them.
const MAX_PAGES: u64 = 100_000;

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

async fn query_by_document(
    client: &Qdrant,
    encoder: &QueryEncoder,
    query: &str,
    conditions: Vec<Condition>,
) -> anyhow::Result<Vec<ScoredPoint>> {
    let result = client
        .query(
            QueryPointsBuilder::new(SECTION_COLLECTION_NAME)
                .query(encoder.encode(query).await?)
                .filter(Filter::must(conditions))
                .limit(sections_search_limit())
                .with_payload(true),
        )
        .await?;
    Ok(result.result)
}

async fn search_by_query(
    client: &Qdrant,
    encoder: &QueryEncoder,
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
    let points = query_by_document(client, encoder, query, conditions).await?;
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

/// Sections of the page at `path`: those matching `query` if given, otherwise
/// all of them (or those under `section`) plus the page's sub-pages.
pub async fn search_sections(
    client: &Qdrant,
    encoder: &QueryEncoder,
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
            search_by_query(client, encoder, q, conditions, exact_limit).await?,
        )),
        None => browse_sections(client, clean_path, section, conditions, exact_limit).await,
    }
}

/// Every page in the sections collection.
async fn all_pages(client: &Qdrant) -> anyhow::Result<Vec<String>> {
    let response = client
        .facet(FacetCountsBuilder::new(SECTION_COLLECTION_NAME, "page").limit(MAX_PAGES))
        .await?;
    Ok(response
        .hits
        .into_iter()
        .filter_map(|hit| match hit.value?.variant? {
            facet_value::Variant::StringValue(page) => Some(page),
            _ => None,
        })
        .collect())
}

/// Pages with sections about the words of `path`, best first.
async fn pages_about(
    client: &Qdrant,
    encoder: &QueryEncoder,
    path: &str,
) -> anyhow::Result<Vec<String>> {
    let query = path_words(path).join(" ");
    if query.is_empty() {
        return Ok(vec![]);
    }
    let mut pages: Vec<String> = Vec::new();
    for section in parse_sections(query_by_document(client, encoder, &query, vec![]).await?) {
        if !pages.contains(&section.page) {
            pages.push(section.page);
        }
    }
    Ok(pages)
}

/// Existing pages to suggest for a missing page or section.
enum Suggestions {
    /// The page exists, the section doesn't.
    WholePage(String),
    /// Similar pages, then pages about the words of the path.
    Pages(Vec<String>),
}

async fn suggest_pages(
    client: &Qdrant,
    encoder: &QueryEncoder,
    path: &str,
    section: Option<&str>,
) -> Suggestions {
    // Accept `<page>/index.md` and `<page>.md`.
    let path = path.trim_matches('/');
    let path = path
        .strip_suffix("/index.md")
        .or_else(|| path.strip_suffix(".md"))
        .unwrap_or(path);
    let pages = all_pages(client).await.unwrap_or_else(|e| {
        log::warn!("No suggestions from the page list: {e}");
        vec![]
    });
    if pages.iter().any(|page| page == path) {
        return match section {
            Some(_) => Suggestions::WholePage(path.to_string()),
            None => Suggestions::Pages(vec![path.to_string()]),
        };
    }
    let mut suggestions: Vec<String> =
        similar_paths(path, pages.iter().map(String::as_str), MAX_SUGGESTIONS)
            .into_iter()
            .map(str::to_string)
            .collect();
    match pages_about(client, encoder, path).await {
        Ok(found) => suggestions.extend(found),
        Err(e) => log::warn!("No suggestions from search: {e}"),
    }
    let mut seen = BTreeSet::new();
    suggestions.retain(|page| seen.insert(page.clone()));
    suggestions.truncate(MAX_SUGGESTIONS);
    Suggestions::Pages(suggestions)
}

/// Markdown body of a `/md` 404.
fn not_found_markdown(request_path: &str, base_url: &str, suggestions: &Suggestions) -> String {
    match suggestions {
        Suggestions::WholePage(page) => format!(
            "Section not found: {request_path}\n\n\
             The page exists. Read all its sections: {base_url}/md/{page}\n"
        ),
        Suggestions::Pages(pages) if pages.is_empty() => {
            format!("Page or section not found: {request_path}\n")
        }
        Suggestions::Pages(pages) => {
            let links: Vec<String> = pages
                .iter()
                .map(|page| format!("- {base_url}/md/{page}"))
                .collect();
            format!(
                "Page or section not found: {request_path}\n\n\
                 These existing pages may be the one you meant:\n\n{}\n",
                links.join("\n")
            )
        }
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
    encoder: Data<QueryEncoder>,
) -> HttpResponse {
    let path_str = path.into_inner();
    let qdrant = qdrant.get_ref();

    let result = search_sections(
        qdrant,
        &encoder,
        query.q.as_deref(),
        &path_str,
        query.s.as_deref(),
        query.limit,
    )
    .await;

    // The connection info must not be held across an await.
    let base_url = {
        let conn = req.connection_info();
        format!("{}://{}", conn.scheme(), conn.host())
    };
    let request_path = req.uri().path();

    match result {
        Ok(Some(section_result)) => {
            log::info!("sections={}", section_result.sections.len());

            let request_query = req.uri().query();

            let markdown = section_result.to_markdown(request_path, request_query, &base_url);
            HttpResponse::Ok()
                .content_type("text/markdown; charset=utf-8")
                .body(markdown)
        }
        Ok(None) => {
            let suggestions = suggest_pages(qdrant, &encoder, &path_str, query.s.as_deref()).await;
            HttpResponse::NotFound()
                .content_type("text/markdown; charset=utf-8")
                .body(not_found_markdown(request_path, &base_url, &suggestions))
        }
        Err(e) => {
            log::error!("Section search error: {}", e);
            HttpResponse::InternalServerError().body(e.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = "https://search.qdrant.tech";

    #[test]
    fn not_found_lists_suggested_pages() {
        let suggestions = Suggestions::Pages(vec![
            "documentation/search/filtering".to_string(),
            "documentation/guides/filtering-tips".to_string(),
        ]);
        assert_eq!(
            not_found_markdown("/md/documentation/concepts/filter", BASE, &suggestions),
            "Page or section not found: /md/documentation/concepts/filter\n\n\
             These existing pages may be the one you meant:\n\n\
             - https://search.qdrant.tech/md/documentation/search/filtering\n\
             - https://search.qdrant.tech/md/documentation/guides/filtering-tips\n"
        );
    }

    #[test]
    fn not_found_without_suggestions() {
        assert_eq!(
            not_found_markdown("/md/x", BASE, &Suggestions::Pages(vec![])),
            "Page or section not found: /md/x\n"
        );
    }

    #[test]
    fn missing_section_points_to_the_whole_page() {
        let suggestions = Suggestions::WholePage("documentation/search/filtering".to_string());
        let markdown = not_found_markdown("/md/documentation/search/filtering", BASE, &suggestions);
        assert!(markdown.starts_with("Section not found"), "{markdown}");
        assert!(
            markdown.contains("https://search.qdrant.tech/md/documentation/search/filtering\n"),
            "{markdown}"
        );
    }
}
