//! The admin panel's JSON API, for signed-in admins only.

use std::time::{Duration, Instant};

use actix_web::web::{self, Data, Path, Query};
use actix_web::{get, HttpResponse};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::auth::Admin;
use super::config::AdminConfig;
use qdrant_client::Qdrant;

use crate::request_log::queries::TrafficFilter;
use crate::request_log::snippet_queries::{self, SnippetFilter, Sort, SortBy, Source};
use crate::request_log::{now_ms, queries, ReplayToken, RequestFilter, RequestLog, REPLAY_HEADER};
use crate::snippets;

/// The longest stretch the panel shows, matching the log's retention.
const MAX_DAYS: u32 = 90;
/// Replayed responses longer than this are cut.
const MAX_REPLAY_BODY: usize = 2 << 20;

/// Where replays go: this service, over loopback.
#[derive(Clone)]
pub struct ReplayTarget {
    /// For example `http://127.0.0.1:8005`.
    pub base_url: String,
    /// `Host` header for `/mcp`, one the MCP server accepts.
    pub mcp_host: String,
    pub token: ReplayToken,
}

fn since(days: Option<u32>) -> i64 {
    let days = days.unwrap_or(30).clamp(1, MAX_DAYS);
    now_ms() - i64::from(days) * 24 * 3600 * 1000
}

/// Runs a query on a fresh read-only connection off the async workers.
async fn read<T: Send + 'static>(
    log: &RequestLog,
    query: impl FnOnce(&rusqlite::Connection) -> rusqlite::Result<T> + Send + 'static,
) -> actix_web::Result<T> {
    let log = log.clone();
    web::block(move || query(&log.reader()?))
        .await?
        .map_err(|e| {
            log::error!("Request log query failed: {e}");
            actix_web::error::ErrorInternalServerError("Request log query failed")
        })
}

#[get("/api/me")]
pub async fn me(admin: Admin) -> HttpResponse {
    HttpResponse::Ok().json(json!({ "email": admin.email }))
}

#[derive(Deserialize)]
pub struct Range {
    days: Option<u32>,
    endpoint: Option<String>,
    #[serde(default)]
    hide_protocol: bool,
    #[serde(default)]
    hide_noise: bool,
    #[serde(default)]
    errors: bool,
}

/// What the traffic view leaves out, from the query parameters
/// `hide_protocol`, `hide_noise` and `errors`.
macro_rules! traffic_filter {
    ($q:expr) => {
        TrafficFilter {
            hide_protocol: $q.hide_protocol,
            hide_noise: $q.hide_noise,
            errors_only: $q.errors,
        }
    };
}

#[get("/api/stats")]
pub async fn stats(
    _: Admin,
    log: Data<RequestLog>,
    q: Query<Range>,
) -> actix_web::Result<HttpResponse> {
    let since = since(q.days);
    let traffic = traffic_filter!(q);
    let stats = read(&log, move |c| queries::stats(c, since, traffic)).await?;
    Ok(HttpResponse::Ok().json(stats))
}

#[get("/api/top")]
pub async fn top(
    _: Admin,
    log: Data<RequestLog>,
    q: Query<Range>,
) -> actix_web::Result<HttpResponse> {
    let since = since(q.days);
    let traffic = traffic_filter!(q);
    let endpoint = q.into_inner().endpoint.filter(|e| !e.is_empty());
    let top = read(&log, move |c| {
        queries::top_queries(c, since, endpoint.as_deref(), traffic, 50)
    })
    .await?;
    Ok(HttpResponse::Ok().json(top))
}

#[derive(Deserialize)]
pub struct ListQuery {
    days: Option<u32>,
    endpoint: Option<String>,
    contains: Option<String>,
    before: Option<i64>,
    #[serde(default)]
    hide_protocol: bool,
    #[serde(default)]
    hide_noise: bool,
    #[serde(default)]
    errors: bool,
}

#[get("/api/requests")]
pub async fn requests(
    _: Admin,
    log: Data<RequestLog>,
    q: Query<ListQuery>,
) -> actix_web::Result<HttpResponse> {
    let q = q.into_inner();
    let filter = RequestFilter {
        since: since(q.days),
        endpoint: q.endpoint.filter(|e| !e.is_empty()),
        contains: q.contains.filter(|c| !c.is_empty()),
        before_id: q.before,
        limit: 50,
        traffic: traffic_filter!(q),
    };
    let list = read(&log, move |c| queries::list(c, &filter)).await?;
    Ok(HttpResponse::Ok().json(list))
}

#[get("/api/requests/{id}")]
pub async fn request_detail(
    _: Admin,
    log: Data<RequestLog>,
    id: Path<i64>,
) -> actix_web::Result<HttpResponse> {
    let id = id.into_inner();
    match read(&log, move |c| queries::get(c, id)).await? {
        Some(record) => Ok(HttpResponse::Ok().json(record)),
        None => Ok(HttpResponse::NotFound().finish()),
    }
}

#[derive(Serialize)]
struct Replay {
    status: u16,
    content_type: Option<String>,
    duration_ms: f64,
    body: String,
    truncated: bool,
}

/// Sends a logged request to the service again, to show what it returns now.
#[get("/api/requests/{id}/replay")]
pub async fn replay(
    _: Admin,
    config: Data<AdminConfig>,
    log: Data<RequestLog>,
    target: Data<ReplayTarget>,
    http: Data<reqwest::Client>,
    id: Path<i64>,
) -> actix_web::Result<HttpResponse> {
    let id = id.into_inner();
    let Some(record) = read(&log, move |c| queries::get(c, id)).await? else {
        return Ok(HttpResponse::NotFound().finish());
    };

    let mut url = format!("{}{}", target.base_url, record.path);
    if !record.query_string.is_empty() {
        url = format!("{url}?{}", record.query_string);
    }
    let method = reqwest::Method::from_bytes(record.method.as_bytes())
        .map_err(actix_web::error::ErrorBadRequest)?;
    let mut request = http
        .request(method, url)
        .timeout(Duration::from_secs(60))
        .header(REPLAY_HEADER, &target.token.0);
    if record.endpoint.starts_with("mcp:") {
        request = request
            .header(reqwest::header::HOST, &target.mcp_host)
            .header(
                reqwest::header::ACCEPT,
                "application/json, text/event-stream",
            )
            .header(reqwest::header::CONTENT_TYPE, "application/json");
    } else if let Some((scheme, host)) = config.public_url.split_once("://") {
        // The service builds links from the host it was called at; make
        // them point at the public address, not at loopback.
        request = request
            .header("X-Forwarded-Proto", scheme)
            .header("X-Forwarded-Host", host);
    }
    if let Some(body) = record.body {
        request = request.body(body);
    }

    let start = Instant::now();
    let response = request
        .send()
        .await
        .map_err(|e| actix_web::error::ErrorBadGateway(format!("Replay failed: {e}")))?;
    let status = response.status().as_u16();
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let bytes = response
        .bytes()
        .await
        .map_err(|e| actix_web::error::ErrorBadGateway(format!("Replay failed: {e}")))?;
    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
    let truncated = bytes.len() > MAX_REPLAY_BODY;
    let body = String::from_utf8_lossy(&bytes[..bytes.len().min(MAX_REPLAY_BODY)]).into_owned();

    Ok(HttpResponse::Ok().json(Replay {
        status,
        content_type,
        duration_ms,
        body,
        truncated,
    }))
}

// ---- Snippet search reports ----

#[derive(Deserialize)]
pub struct SnippetQuery {
    days: Option<u32>,
    /// `rest` or `mcp`.
    source: Option<String>,
    /// The language as requested; `-` for searches without one.
    language: Option<String>,
    contains: Option<String>,
    #[serde(default)]
    errors: bool,
    /// For the request list: `time`, `source`, `language`, `query`,
    /// `status` or `duration`.
    sort: Option<String>,
    /// For the request list: `asc` or `desc`, the default.
    order: Option<String>,
    /// For the request list: how many requests to skip.
    #[serde(default)]
    offset: u32,
}

/// Searches without a language, in the `language` parameter.
const NO_LANGUAGE: &str = "-";

impl SnippetQuery {
    fn sort(&self) -> Sort {
        Sort {
            by: self
                .sort
                .as_deref()
                .and_then(SortBy::parse)
                .unwrap_or_default(),
            ascending: self.order.as_deref() == Some("asc"),
        }
    }

    fn filter(self) -> SnippetFilter {
        SnippetFilter {
            since: since(self.days),
            source: self.source.as_deref().and_then(Source::parse),
            language: self.language.filter(|l| !l.is_empty()).map(|l| {
                if l == NO_LANGUAGE {
                    String::new()
                } else {
                    l
                }
            }),
            contains: self.contains.filter(|c| !c.is_empty()),
            errors_only: self.errors,
        }
    }
}

#[get("/api/snippets/stats")]
pub async fn snippet_stats(
    _: Admin,
    log: Data<RequestLog>,
    q: Query<SnippetQuery>,
) -> actix_web::Result<HttpResponse> {
    let filter = q.into_inner().filter();
    let report = read(&log, move |c| snippet_queries::stats(c, &filter)).await?;
    Ok(HttpResponse::Ok().json(report))
}

/// Searches by requested language and source, whatever the other filters.
#[get("/api/snippets/languages")]
pub async fn snippet_languages(
    _: Admin,
    log: Data<RequestLog>,
    q: Query<Range>,
) -> actix_web::Result<HttpResponse> {
    let since = since(q.days);
    let rows = read(&log, move |c| snippet_queries::languages(c, since)).await?;
    Ok(HttpResponse::Ok().json(rows))
}

#[get("/api/snippets/top")]
pub async fn snippet_top(
    _: Admin,
    log: Data<RequestLog>,
    q: Query<SnippetQuery>,
) -> actix_web::Result<HttpResponse> {
    let filter = q.into_inner().filter();
    let rows = read(&log, move |c| snippet_queries::top_queries(c, &filter, 50)).await?;
    Ok(HttpResponse::Ok().json(rows))
}

#[get("/api/snippets/requests")]
pub async fn snippet_requests(
    _: Admin,
    log: Data<RequestLog>,
    q: Query<SnippetQuery>,
) -> actix_web::Result<HttpResponse> {
    let q = q.into_inner();
    let (sort, offset) = (q.sort(), q.offset);
    let filter = q.filter();
    let list = read(&log, move |c| {
        snippet_queries::list(c, &filter, sort, offset, 50)
    })
    .await?;
    Ok(HttpResponse::Ok().json(list))
}

fn qdrant_or_unavailable(qdrant: Option<Data<Qdrant>>) -> actix_web::Result<Data<Qdrant>> {
    qdrant.ok_or_else(|| actix_web::error::ErrorServiceUnavailable("Qdrant is not configured"))
}

fn qdrant_error(e: anyhow::Error) -> actix_web::Error {
    log::error!("Snippet debugging query failed: {e}");
    actix_web::error::ErrorBadGateway(format!("Qdrant query failed: {e}"))
}

/// What the snippet collection holds, to compare with what's searched for.
#[get("/api/snippets/index")]
pub async fn snippet_index(
    _: Admin,
    qdrant: Option<Data<Qdrant>>,
) -> actix_web::Result<HttpResponse> {
    let qdrant = qdrant_or_unavailable(qdrant)?;
    let overview = snippets::index_overview(&qdrant)
        .await
        .map_err(qdrant_error)?;
    Ok(HttpResponse::Ok().json(overview))
}

#[derive(Deserialize)]
pub struct DebugSearch {
    query: String,
    language: Option<String>,
    limit: Option<u64>,
    /// Normalize the language as the MCP server does.
    #[serde(default)]
    normalize: bool,
}

#[derive(Serialize)]
struct DebugSearchResult {
    /// The language the index was filtered by, if any.
    searched_as: Option<String>,
    duration_ms: f64,
    #[serde(flatten)]
    breakdown: snippets::SearchBreakdown,
}

/// Runs a snippet search, and each half of the hybrid search on its own.
#[get("/api/snippets/search")]
pub async fn snippet_search(
    _: Admin,
    qdrant: Option<Data<Qdrant>>,
    q: Query<DebugSearch>,
) -> actix_web::Result<HttpResponse> {
    let qdrant = qdrant_or_unavailable(qdrant)?;
    let q = q.into_inner();
    let query = q.query.trim();
    if query.is_empty() {
        return Err(actix_web::error::ErrorBadRequest("The query is empty"));
    }
    let searched_as = q
        .language
        .map(|l| {
            if q.normalize {
                snippets::normalize_language(&l)
            } else {
                l
            }
        })
        .filter(|l| !l.is_empty());
    let limit = q.limit.unwrap_or(5).clamp(1, 50);
    let start = Instant::now();
    let breakdown = snippets::search_breakdown(&qdrant, query, searched_as.as_deref(), limit)
        .await
        .map_err(qdrant_error)?;
    Ok(HttpResponse::Ok().json(DebugSearchResult {
        searched_as,
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        breakdown,
    }))
}
