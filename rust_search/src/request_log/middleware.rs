use std::time::Instant;

use actix_web::body::MessageBody;
use actix_web::dev::{ServiceRequest, ServiceResponse};
use actix_web::http::{header, Method};
use actix_web::middleware::Next;
use actix_web::web::{Bytes, BytesMut, Data};
use actix_web::{Error, HttpMessage, HttpResponse};
use futures::StreamExt;
use serde_json::Value;

use super::store::{now_ms, RequestLog, RequestRecord};

/// Largest MCP request body the log reads; bigger ones are refused.
const MAX_MCP_BODY: usize = 1 << 20;

/// Header the admin panel's replays carry, so they don't count as traffic.
pub const REPLAY_HEADER: &str = "x-admin-replay";

/// A secret per process that marks the admin panel's own replays.
#[derive(Clone)]
pub struct ReplayToken(pub String);

impl ReplayToken {
    pub fn random() -> Self {
        use rand::Rng;
        let bytes: [u8; 16] = rand::thread_rng().gen();
        Self(bytes.iter().map(|b| format!("{b:02x}")).collect())
    }
}

/// What a request was for, from its path and, for MCP, its JSON-RPC body.
/// `None` for requests that aren't logged.
fn classify(
    method: &Method,
    path: &str,
    query_string: &str,
    body: Option<&[u8]>,
) -> Option<(String, Option<String>)> {
    let param = |names: &[&str]| {
        url::form_urlencoded::parse(query_string.as_bytes())
            .find(|(k, _)| names.contains(&k.as_ref()))
            .map(|(_, v)| v.into_owned())
    };
    if path == "/api/search" {
        Some(("search".into(), param(&["q"])))
    } else if let Some(page) = path.strip_prefix("/md") {
        Some(("md".into(), param(&["q"]).or(Some(page.to_string()))))
    } else if path == "/skills" || path.starts_with("/skills/") {
        Some(("skills".into(), param(&["q", "query"])))
    } else if path == "/snippets/search" {
        Some(("snippets".into(), param(&["query"])))
    } else if path == "/mcp" || path.starts_with("/mcp/") {
        if method != Method::POST {
            return Some((format!("mcp:{method}"), None));
        }
        let json: Value = body.and_then(|b| serde_json::from_slice(b).ok())?;
        // A batch is logged by its first message.
        let message = json.as_array().and_then(|a| a.first()).unwrap_or(&json);
        let rpc_method = message.get("method")?.as_str()?;
        if rpc_method.starts_with("notifications/") {
            return None;
        }
        if rpc_method == "tools/call" {
            let params = message.get("params");
            let tool = params
                .and_then(|p| p.get("name"))
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let args = params.and_then(|p| p.get("arguments"));
            let query = ["query", "page", "section"]
                .iter()
                .find_map(|k| args?.get(*k)?.as_str())
                .map(str::to_string);
            Some((format!("mcp:{tool}"), query))
        } else {
            Some((format!("mcp:{rpc_method}"), None))
        }
    } else {
        None
    }
}

/// Middleware recording every API and MCP request in the request log.
pub async fn log_requests(
    mut req: ServiceRequest,
    next: Next<impl MessageBody + 'static>,
) -> Result<ServiceResponse<impl MessageBody>, Error> {
    let log = req.app_data::<Data<RequestLog>>().cloned();
    let is_replay = match (
        req.app_data::<Data<ReplayToken>>(),
        req.headers().get(REPLAY_HEADER),
    ) {
        (Some(token), Some(value)) => value.as_bytes() == token.0.as_bytes(),
        _ => false,
    };
    let Some(log) = log.filter(|_| !is_replay) else {
        return Ok(next.call(req).await?.map_into_left_body());
    };

    let start = Instant::now();
    let ts = now_ms();
    let method = req.method().clone();
    let path = req.path().to_string();
    let query_string = req.query_string().to_string();
    let user_agent = req
        .headers()
        .get(header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);

    // The MCP request says what it does in its body: read it, then hand it on.
    let mut body = None;
    if method == Method::POST && path.starts_with("/mcp") {
        let mut payload = req.take_payload();
        let mut bytes = BytesMut::new();
        while let Some(chunk) = payload.next().await {
            bytes.extend_from_slice(&chunk?);
            if bytes.len() > MAX_MCP_BODY {
                let response = HttpResponse::PayloadTooLarge().finish();
                return Ok(req.into_response(response).map_into_right_body());
            }
        }
        let bytes: Bytes = bytes.freeze();
        let (_, mut replayed) = actix_http::h1::Payload::create(true);
        replayed.unread_data(bytes.clone());
        req.set_payload(replayed.into());
        body = Some(bytes);
    }

    let Some((endpoint, query)) = classify(&method, &path, &query_string, body.as_deref()) else {
        return Ok(next.call(req).await?.map_into_left_body());
    };

    let response = next.call(req).await?;
    log.record(RequestRecord {
        id: 0,
        ts,
        endpoint,
        query,
        method: method.to_string(),
        path,
        query_string,
        body: body.map(|b| String::from_utf8_lossy(&b).into_owned()),
        status: response.status().as_u16(),
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        user_agent,
    });
    Ok(response.map_into_left_body())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_endpoints() {
        let get = Method::GET;
        assert_eq!(
            classify(&get, "/api/search", "q=hnsw&section=", None),
            Some(("search".into(), Some("hnsw".into())))
        );
        assert_eq!(
            classify(&get, "/md/documentation/concepts/", "", None),
            Some(("md".into(), Some("/documentation/concepts/".into())))
        );
        assert_eq!(
            classify(
                &get,
                "/snippets/search",
                "query=upsert&language=python",
                None
            ),
            Some(("snippets".into(), Some("upsert".into())))
        );
        assert_eq!(classify(&get, "/favicon.ico", "", None), None);
        assert_eq!(classify(&get, "/admin", "", None), None);
    }

    #[test]
    fn classifies_mcp_calls() {
        let post = Method::POST;
        let call = br#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"search_docs","arguments":{"query":"filtering"}}}"#;
        assert_eq!(
            classify(&post, "/mcp", "", Some(call)),
            Some(("mcp:search_docs".into(), Some("filtering".into())))
        );
        let init = br#"{"jsonrpc":"2.0","id":0,"method":"initialize","params":{}}"#;
        assert_eq!(
            classify(&post, "/mcp", "", Some(init)),
            Some(("mcp:initialize".into(), None))
        );
        let note = br#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
        assert_eq!(classify(&post, "/mcp", "", Some(note)), None);
    }
}
