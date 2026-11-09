use actix_web::web::{Data, Query};
use actix_web::{get, HttpRequest, HttpResponse};
use std::collections::BTreeSet;

use qdrant_client::qdrant::r#match::MatchValue;
use qdrant_client::qdrant::{
    value::Kind, Condition, Filter, PayloadIncludeSelector, PointId, QueryPointsBuilder,
    ScoredPoint, ScrollPointsBuilder,
};
use qdrant_client::Qdrant;
use serde::Deserialize;

use super::models::{Section, SectionSearchResult, slugify_heading};
use crate::common::SECTION_COLLECTION_NAME;
use crate::embedding::QueryEncoder;
use crate::similar_paths::{suggest, Suggestion};

/// Most pages suggested for a missing path.
const MAX_SUGGESTIONS: usize = 5;
/// Most sections listed for a missing section.
const MAX_LISTED_SECTIONS: u32 = 500;

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

/// A heading of a page, to link to with `?s=`.
#[derive(Debug, PartialEq)]
struct Heading {
    title: String,
    slug: String,
    level: i64,
}

/// The headings of `page` in the order they appear, as Qdrant orders them by
/// the `line` index.
async fn page_headings(client: &Qdrant, page: &str) -> anyhow::Result<Vec<Heading>> {
    let response = client
        .scroll(
            ScrollPointsBuilder::new(SECTION_COLLECTION_NAME)
                .filter(Filter::must([Condition::matches(
                    "page",
                    MatchValue::Keyword(page.to_string()),
                )]))
                .order_by("line")
                .limit(MAX_LISTED_SECTIONS)
                .with_payload(PayloadIncludeSelector {
                    fields: ["title", "slug", "level"].map(String::from).to_vec(),
                })
                .with_vectors(false),
        )
        .await?;
    Ok(response
        .result
        .into_iter()
        .filter_map(|point| {
            let text = |key: &str| match point.payload.get(key)?.kind.as_ref()? {
                Kind::StringValue(value) if !value.is_empty() => Some(value.clone()),
                _ => None,
            };
            let level = match point.payload.get("level")?.kind.as_ref()? {
                Kind::IntegerValue(level) => *level,
                Kind::DoubleValue(level) => *level as i64,
                _ => return None,
            };
            Some(Heading {
                title: text("title")?,
                slug: text("slug")?,
                level,
            })
        })
        .collect())
}

/// Existing pages to suggest for a missing page or section.
enum Suggestions {
    /// The page exists, the section doesn't: the page's headings.
    WholePage {
        page: String,
        headings: Vec<Heading>,
    },
    /// Pages that resemble the missing one.
    Pages(Vec<Suggestion>),
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
    // Without a section, the page would have been found if it had sections.
    if section.is_some() {
        match page_headings(client, path).await {
            Ok(headings) if !headings.is_empty() => {
                return Suggestions::WholePage {
                    page: path.to_string(),
                    headings,
                }
            }
            Ok(_) => {}
            Err(e) => log::warn!("Could not list the sections of {path}: {e}"),
        }
    }
    Suggestions::Pages(
        suggest(client, Some(encoder), path, &[], MAX_SUGGESTIONS)
            .await
            .unwrap_or_else(|e| {
                log::warn!("No suggestions from the page index: {e}");
                vec![]
            }),
    )
}

/// Markdown body of a `/md` 404. `request` is the path and query.
fn not_found_markdown(request: &str, base_url: &str, suggestions: &Suggestions) -> String {
    match suggestions {
        Suggestions::WholePage { page, headings } => {
            let top = headings.iter().map(|h| h.level).min().unwrap_or(1);
            let rows: Vec<String> = headings
                .iter()
                .map(|h| {
                    format!(
                        "{}- [{}]({base_url}/md/{page}?s={})",
                        "  ".repeat((h.level - top) as usize),
                        h.title,
                        h.slug
                    )
                })
                .collect();
            format!(
                "Section not found: {request}\n\n\
                 The page has these sections:\n\n{}\n\n\
                 Read the whole page: {base_url}/md/{page}\n",
                rows.join("\n")
            )
        }
        Suggestions::Pages(pages) if pages.is_empty() => {
            format!("Page or section not found: {request}\n")
        }
        Suggestions::Pages(pages) => {
            let links: Vec<String> = pages
                .iter()
                .map(|s| match s.title.as_str() {
                    "" => format!("- {base_url}/md/{}", s.page),
                    title => format!("- [{title}]({base_url}/md/{})", s.page),
                })
                .collect();
            format!(
                "Page or section not found: {request}\n\n\
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
                .body(not_found_markdown(
                    req.uri()
                        .path_and_query()
                        .map_or(request_path, |pq| pq.as_str()),
                    &base_url,
                    &suggestions,
                ))
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

    fn suggestion(page: &str, title: &str) -> Suggestion {
        Suggestion {
            page: page.to_string(),
            title: title.to_string(),
        }
    }

    #[test]
    fn not_found_lists_suggested_pages() {
        let suggestions = Suggestions::Pages(vec![
            suggestion("documentation/search/filtering", "Filtering"),
            suggestion("documentation/guides", ""),
        ]);
        assert_eq!(
            not_found_markdown("/md/documentation/concepts/filter", BASE, &suggestions),
            "Page or section not found: /md/documentation/concepts/filter\n\n\
             These existing pages may be the one you meant:\n\n\
             - [Filtering](https://search.qdrant.tech/md/documentation/search/filtering)\n\
             - https://search.qdrant.tech/md/documentation/guides\n"
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
    fn missing_section_lists_the_sections_of_the_page() {
        let heading = |title: &str, slug: &str, level| Heading {
            title: title.to_string(),
            slug: slug.to_string(),
            level,
        };
        let suggestions = Suggestions::WholePage {
            page: "documentation/search/filtering".to_string(),
            headings: vec![
                heading("Filtering", "filtering", 1),
                heading("Filtering Clauses", "filtering-clauses", 2),
                heading("Must", "must", 3),
                heading("Filtering Conditions", "filtering-conditions", 2),
            ],
        };
        assert_eq!(
            not_found_markdown(
                "/md/documentation/search/filtering?s=old-heading",
                BASE,
                &suggestions,
            ),
            "Section not found: /md/documentation/search/filtering?s=old-heading\n\n\
             The page has these sections:\n\n\
             - [Filtering](https://search.qdrant.tech/md/documentation/search/filtering?s=filtering)\n\
             \x20\x20- [Filtering Clauses](https://search.qdrant.tech/md/documentation/search/filtering?s=filtering-clauses)\n\
             \x20\x20\x20\x20- [Must](https://search.qdrant.tech/md/documentation/search/filtering?s=must)\n\
             \x20\x20- [Filtering Conditions](https://search.qdrant.tech/md/documentation/search/filtering?s=filtering-conditions)\n\n\
             Read the whole page: https://search.qdrant.tech/md/documentation/search/filtering\n"
        );
    }
}
