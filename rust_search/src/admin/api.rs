//! The admin panel's JSON API, for signed-in admins only.

use std::time::{Duration, Instant};

use actix_web::web::{self, Data, Path, Query};
use actix_web::{get, HttpResponse};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::auth::Admin;
use crate::request_log::{now_ms, queries, ReplayToken, RequestFilter, RequestLog, REPLAY_HEADER};

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
}

#[get("/api/stats")]
pub async fn stats(
    _: Admin,
    log: Data<RequestLog>,
    q: Query<Range>,
) -> actix_web::Result<HttpResponse> {
    let since = since(q.days);
    let stats = read(&log, move |c| queries::stats(c, since)).await?;
    Ok(HttpResponse::Ok().json(stats))
}

#[get("/api/top")]
pub async fn top(
    _: Admin,
    log: Data<RequestLog>,
    q: Query<Range>,
) -> actix_web::Result<HttpResponse> {
    let since = since(q.days);
    let endpoint = q.into_inner().endpoint.filter(|e| !e.is_empty());
    let top = read(&log, move |c| {
        queries::top_queries(c, since, endpoint.as_deref(), 50)
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
