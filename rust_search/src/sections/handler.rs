use actix_web::http::header::LOCATION;
use actix_web::http::StatusCode;
use actix_web::web::{Data, Query};
use actix_web::{get, HttpRequest, HttpResponse};
use qdrant_client::qdrant::r#match::MatchValue;
use qdrant_client::qdrant::{
    Condition, Document, FacetCountsBuilder, Filter, QueryPointsBuilder, ScoredPoint, VectorInput,
};
use qdrant_client::Qdrant;
use serde::Deserialize;

use super::models::{Section, SectionSearchResult, slugify_heading};
use crate::redirects::{self, RedirectStore, Target};

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

fn base_url(req: &HttpRequest) -> String {
    let conn = req.connection_info();
    format!("{}://{}", conn.scheme(), conn.host())
}

fn not_found() -> HttpResponse {
    HttpResponse::NotFound().body("Page or section not found")
}

fn render_markdown(result: &SectionSearchResult, req: &HttpRequest) -> HttpResponse {
    let markdown = result.to_markdown(req.uri().path(), req.uri().query(), &base_url(req));
    HttpResponse::Ok()
        .content_type("text/markdown; charset=utf-8")
        .body(markdown)
}

/// Build the `/md/`-prefixed location for a resolved target.
///
/// The request's own query string is preserved. Failing that, a fragment on the
/// target becomes `?s=`, which is how this mirror spells a section — see
/// `links::rewrite_links`. (The spec suggests dropping target fragments as
/// meaningless for a markdown fetch; here they are not, so they are translated.)
fn redirect_location(
    target_path: &str,
    fragment: Option<String>,
    request_query: Option<&str>,
) -> String {
    let path = format!("/md/{}", target_path.trim_matches('/'));
    match (request_query, fragment) {
        (Some(query), _) if !query.is_empty() => format!("{}?{}", path, query),
        (_, Some(fragment)) => format!("{}?s={}", path, fragment),
        _ => path,
    }
}

/// Consult the redirect table for a path the mirror holds no document for.
///
/// Hooking in on a miss is what makes `force` handling unnecessary: Netlify
/// skips a non-forced rule when a real file exists at that path, so answering
/// from the table only when there is no document reproduces that semantic
/// exactly. A forced rule (`301!`) pointing at an *internal* path would have to
/// be consulted before serving a document instead — all three forced rules are
/// external today, so this is where that check would go if that ever changes.
async fn redirect_for_miss(
    qdrant: &Qdrant,
    redirects: &RedirectStore,
    captured_path: &str,
    req: &HttpRequest,
    query: &MdSearch,
) -> HttpResponse {
    let rules = redirects.rules();
    let lookup = redirects::lookup_path(captured_path);

    let Some(resolution) = redirects::resolve(&rules, &lookup) else {
        return not_found();
    };
    let status = StatusCode::from_u16(resolution.status).unwrap_or(StatusCode::MOVED_PERMANENTLY);
    let is_rewrite = resolution.is_rewrite();

    let (target_path, fragment) = match resolution.target {
        Target::External(url) => {
            log::info!("redirect {} -> {} (external, not mirrored)", lookup, url);
            return HttpResponse::build(status)
                .insert_header((LOCATION, url))
                .finish();
        }
        Target::Internal { path, fragment } => (path, fragment),
    };

    // The table can route a path to a document that does not exist. Redirecting
    // into a dead end is worse than the 404 it replaces — the client burns a
    // round trip, lands nowhere, and gets no signal that the table was at fault.
    // This costs one filter-only Qdrant query (no embedding), and only on
    // requests that were already going to 404.
    let target = match search_sections(qdrant, None, &target_path, None).await {
        Ok(Some(target)) => target,
        Ok(None) => {
            log::warn!(
                "redirect {} -> {} suppressed: the mirror holds no such document",
                lookup,
                target_path
            );
            return not_found();
        }
        Err(e) => {
            log::error!("Section search error: {}", e);
            return HttpResponse::InternalServerError().body(e.to_string());
        }
    };

    // A `200` rule is a rewrite: Netlify proxies the target and the URL does not
    // change, so answering a redirect for one would diverge from the live site.
    // Every rule is `301` today; this keeps a future `200!` honest.
    if is_rewrite {
        log::info!("rewrite {} -> {}", lookup, target_path);
        if query.q.is_none() && query.s.is_none() {
            return render_markdown(&target, req);
        }
        return match search_sections(
            qdrant,
            query.q.as_deref(),
            &target_path,
            query.s.as_deref(),
        )
        .await
        {
            Ok(Some(result)) => render_markdown(&result, req),
            Ok(None) => not_found(),
            Err(e) => {
                log::error!("Section search error: {}", e);
                HttpResponse::InternalServerError().body(e.to_string())
            }
        };
    }

    let location = redirect_location(&target_path, fragment, req.uri().query());
    log::info!("redirect {} -> {} ({})", lookup, location, status.as_u16());
    HttpResponse::build(status)
        .insert_header((LOCATION, format!("{}{}", base_url(req), location)))
        .finish()
}

#[get("/md/{path:.*}")]
pub async fn md_handler(
    path: actix_web::web::Path<String>,
    req: HttpRequest,
    query: Query<MdSearch>,
    qdrant: Data<Qdrant>,
    redirects: Data<RedirectStore>,
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
            render_markdown(&section_result, &req)
        }
        Ok(None) => {
            redirect_for_miss(qdrant, redirects.get_ref(), &path_str, &req, &query).await
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

    #[test]
    fn location_gets_the_md_prefix() {
        assert_eq!(
            redirect_location("/documentation/guides/quickstart/", None, None),
            "/md/documentation/guides/quickstart"
        );
    }

    #[test]
    fn location_keeps_the_request_query() {
        assert_eq!(
            redirect_location("/documentation/guides/", None, Some("q=setup")),
            "/md/documentation/guides?q=setup"
        );
    }

    #[test]
    fn location_translates_a_target_fragment_to_a_section_query() {
        assert_eq!(
            redirect_location(
                "/documentation/manage-data/points/",
                Some("scroll-points".into()),
                None
            ),
            "/md/documentation/manage-data/points?s=scroll-points"
        );
    }

    #[test]
    fn the_request_query_outranks_a_target_fragment() {
        assert_eq!(
            redirect_location("/documentation/points/", Some("scroll-points".into()), Some("s=upsert-points")),
            "/md/documentation/points?s=upsert-points"
        );
    }

    #[test]
    fn location_of_the_mirror_root() {
        assert_eq!(redirect_location("/", None, None), "/md/");
    }
}
