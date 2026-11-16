//! The admin panel over HTTP, with Google sign-in faked by a mock server and
//! the service's endpoints by echo handlers.

use std::net::TcpListener;
use std::time::Duration;

use actix_web::{middleware, web, App, HttpRequest, HttpResponse, HttpServer};
use reqwest::{redirect::Policy, StatusCode};
use serde_json::{json, Value};
use wiremock::matchers::{body_string_contains, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use rmcp_actix_web::transport::{LocalSessionManager, StreamableHttpService};

use super::*;
use crate::docs_mcp::config::Config;
use crate::docs_mcp::{Backend, DocsServer, FixedBackend};
use crate::request_log::{self, ReplayToken};

struct Harness {
    base: String,
    google: MockServer,
    client: reqwest::Client,
    db: std::path::PathBuf,
}

impl Drop for Harness {
    fn drop(&mut self) {
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", self.db.display()));
        }
    }
}

async fn echo_search(req: HttpRequest) -> HttpResponse {
    HttpResponse::Ok().json(json!({ "query": req.query_string() }))
}

async fn echo_mcp(body: String) -> HttpResponse {
    HttpResponse::Ok()
        .content_type("application/json")
        .body(body)
}

impl Harness {
    async fn start() -> Self {
        let google = MockServer::start().await;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let base = format!("http://{addr}");
        let db = std::env::temp_dir().join(format!("admin-test-{}.sqlite", rand::random::<u64>()));

        let config = AdminConfig {
            client_id: "client".into(),
            client_secret: "secret".into(),
            session_key: "a session key that is long enough to derive from".into(),
            public_url: base.clone(),
            allowed_domains: vec!["qdrant.com".into()],
            allowed_emails: vec!["friend@example.com".into()],
            auth_url: format!("{}/auth", google.uri()),
            token_url: format!("{}/token", google.uri()),
            userinfo_url: format!("{}/userinfo", google.uri()),
        };
        config.validate().unwrap();
        let log = RequestLog::open(&db, 90).unwrap();
        let token = ReplayToken::random();
        let panel = AdminPanel::new(
            config,
            log.clone(),
            ReplayTarget {
                base_url: base.clone(),
                mcp_host: "localhost".into(),
                token: token.clone(),
            },
        );
        let log = web::Data::new(log);
        let token = web::Data::new(token);
        // The real docs MCP server, with nothing to search, at /mcp/docs.
        let docs = DocsServer::new(
            Config {
                sources: vec!["documentation".into()],
            },
            Backend::Fixed(std::sync::Arc::new(FixedBackend::default())),
        );
        let server = HttpServer::new(move || {
            let docs = docs.clone();
            let mcp = StreamableHttpService::builder()
                .service_factory(std::sync::Arc::new(move || Ok(docs.clone())))
                .session_manager(std::sync::Arc::new(LocalSessionManager::default()))
                .stateful_mode(false)
                .json_response(true)
                .allowed_hosts(vec!["127.0.0.1".to_string()])
                .build();
            App::new()
                .app_data(log.clone())
                .app_data(token.clone())
                .wrap(middleware::from_fn(request_log::log_requests))
                .route("/api/search", web::get().to(echo_search))
                .route("/snippets/search", web::get().to(echo_search))
                .route("/mcp", web::post().to(echo_mcp))
                .service(mcp.scope_with_path("/mcp/docs"))
                .configure(|cfg| panel.configure(cfg))
        })
        .workers(1)
        .listen(listener)
        .unwrap()
        .run();
        actix_web::rt::spawn(server);

        let client = reqwest::Client::builder()
            .redirect(Policy::none())
            .build()
            .unwrap();
        Self {
            base,
            google,
            client,
            db,
        }
    }

    async fn get(&self, path: &str, cookie: Option<&str>) -> reqwest::Response {
        let mut request = self.client.get(format!("{}{path}", self.base));
        if let Some(cookie) = cookie {
            request = request.header("cookie", cookie);
        }
        request.send().await.unwrap()
    }

    async fn json(&self, path: &str, cookie: &str) -> Value {
        let response = self.get(path, Some(cookie)).await;
        assert_eq!(response.status(), StatusCode::OK, "{path}");
        response.json().await.unwrap()
    }

    /// Signs in as the account Google's fake userinfo returns, and gives the
    /// callback's response.
    async fn sign_in(&self, userinfo: Value) -> reqwest::Response {
        Mock::given(method("POST"))
            .and(path("/token"))
            .and(body_string_contains("code=the-code"))
            .and(body_string_contains("client_secret=secret"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"access_token": "tok"})))
            .mount(&self.google)
            .await;
        Mock::given(method("GET"))
            .and(path("/userinfo"))
            .and(header("authorization", "Bearer tok"))
            .respond_with(ResponseTemplate::new(200).set_body_json(userinfo))
            .mount(&self.google)
            .await;

        let login = self.get("/admin/login", None).await;
        assert_eq!(login.status(), StatusCode::FOUND);
        let location = url::Url::parse(login.headers()["location"].to_str().unwrap()).unwrap();
        assert!(location.as_str().starts_with(&self.google.uri()));
        let param = |name: &str| {
            location
                .query_pairs()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v.into_owned())
        };
        assert_eq!(
            param("redirect_uri"),
            Some(format!("{}/admin/auth/callback", self.base))
        );
        assert_eq!(param("hd").as_deref(), Some("qdrant.com"));
        let state = param("state").unwrap();

        self.get(
            &format!("/admin/auth/callback?code=the-code&state={state}"),
            Some(&cookie_of(&login)),
        )
        .await
    }
}

fn cookie_of(response: &reqwest::Response) -> String {
    let set_cookie = response.headers()["set-cookie"].to_str().unwrap();
    set_cookie.split(';').next().unwrap().to_string()
}

async fn wait_for_total(h: &Harness, cookie: &str, total: i64) -> Value {
    for _ in 0..50 {
        let stats = h.json("/admin/api/stats?days=7", cookie).await;
        if stats["total"] == total {
            return stats;
        }
        actix_web::rt::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("the request log never reached {total} requests");
}

#[actix_web::test]
async fn signed_out_visitors_are_sent_to_sign_in() {
    let h = Harness::start().await;
    let page = h.get("/admin", None).await;
    assert_eq!(page.status(), StatusCode::FOUND);
    assert_eq!(page.headers()["location"], "/admin/login");
    assert_eq!(
        h.get("/admin/api/stats", None).await.status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        h.get("/admin/api/requests/1/replay", None).await.status(),
        StatusCode::UNAUTHORIZED
    );
}

#[actix_web::test]
async fn callback_needs_the_state_it_was_sent() {
    let h = Harness::start().await;
    let login = h.get("/admin/login", None).await;
    let callback = h
        .get(
            "/admin/auth/callback?code=c&state=forged",
            Some(&cookie_of(&login)),
        )
        .await;
    assert_eq!(callback.status(), StatusCode::BAD_REQUEST);
    // Without the cookie the state can't be checked either.
    let callback = h
        .get("/admin/auth/callback?code=c&state=forged", None)
        .await;
    assert_eq!(callback.status(), StatusCode::BAD_REQUEST);
}

#[actix_web::test]
async fn other_accounts_are_refused() {
    for userinfo in [
        json!({"email": "someone@gmail.com", "email_verified": true}),
        // The domain is only trusted from Google's hosted-domain claim.
        json!({"email": "someone@qdrant.com", "email_verified": true}),
        json!({"email": "someone@qdrant.com", "email_verified": false, "hd": "qdrant.com"}),
    ] {
        let h = Harness::start().await;
        let callback = h.sign_in(userinfo.clone()).await;
        assert_eq!(callback.status(), StatusCode::FORBIDDEN, "{userinfo}");
        // Whatever cookie comes back doesn't open the API.
        if callback.headers().contains_key("set-cookie") {
            let me = h.get("/admin/api/me", Some(&cookie_of(&callback))).await;
            assert_eq!(me.status(), StatusCode::UNAUTHORIZED);
        }
    }
}

#[actix_web::test]
async fn admins_see_stats_and_replay_requests() {
    let h = Harness::start().await;
    let callback = h
        .sign_in(json!({"email": "Ann@qdrant.com", "email_verified": true, "hd": "qdrant.com"}))
        .await;
    assert_eq!(callback.status(), StatusCode::FOUND);
    assert_eq!(callback.headers()["location"], "/admin");
    let cookie = cookie_of(&callback);
    assert_eq!(
        h.json("/admin/api/me", &cookie).await["email"],
        "ann@qdrant.com"
    );
    let page = h.get("/admin", Some(&cookie)).await;
    assert_eq!(page.status(), StatusCode::OK);

    // Traffic: two searches and an MCP tool call. The admin requests above
    // aren't logged.
    for q in ["hnsw", "hnsw"] {
        h.get(&format!("/api/search?q={q}&section="), None).await;
    }
    let call = json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "search_docs", "arguments": {"query": "filters"}}});
    h.client
        .post(format!("{}/mcp", h.base))
        .json(&call)
        .send()
        .await
        .unwrap();

    let stats = wait_for_total(&h, &cookie, 3).await;
    let endpoints: Vec<_> = stats["endpoints"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            (
                e["endpoint"].as_str().unwrap(),
                e["count"].as_i64().unwrap(),
            )
        })
        .collect();
    assert_eq!(endpoints, [("search", 2), ("mcp:search_docs", 1)]);
    let groups: Vec<_> = stats["daily"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d["group"].as_str().unwrap())
        .collect();
    assert_eq!(groups, ["mcp", "search"]);

    let top = h
        .json("/admin/api/top?days=7&endpoint=search", &cookie)
        .await;
    assert_eq!(top[0]["query"], "hnsw");
    assert_eq!(top[0]["count"], 2);

    let mcp = h.json("/admin/api/requests?endpoint=mcp", &cookie).await;
    assert_eq!(mcp.as_array().unwrap().len(), 1);
    assert_eq!(mcp[0]["query"], "filters");
    let found = h.json("/admin/api/requests?contains=HNS", &cookie).await;
    assert_eq!(found.as_array().unwrap().len(), 2);

    // A replay runs the request again and returns the service's answer.
    let replay = h
        .json(
            &format!("/admin/api/requests/{}/replay", top[0]["last_id"]),
            &cookie,
        )
        .await;
    assert_eq!(replay["status"], 200);
    let body: Value = serde_json::from_str(replay["body"].as_str().unwrap()).unwrap();
    assert_eq!(body["query"], "q=hnsw&section=");

    let replay = h
        .json(
            &format!("/admin/api/requests/{}/replay", mcp[0]["id"]),
            &cookie,
        )
        .await;
    let body: Value = serde_json::from_str(replay["body"].as_str().unwrap()).unwrap();
    assert_eq!(body, call);

    // Replays aren't traffic.
    actix_web::rt::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(h.json("/admin/api/stats?days=7", &cookie).await["total"], 3);

    // Signing out ends the session.
    let logout = h
        .client
        .post(format!("{}/admin/logout", h.base))
        .header("cookie", &cookie)
        .send()
        .await
        .unwrap();
    let cleared = cookie_of(&logout);
    assert_eq!(
        h.get("/admin/api/me", Some(&cleared)).await.status(),
        StatusCode::UNAUTHORIZED
    );
}

#[actix_web::test]
async fn admins_see_snippet_search_reports() {
    let h = Harness::start().await;
    let callback = h
        .sign_in(json!({"email": "ann@qdrant.com", "email_verified": true, "hd": "qdrant.com"}))
        .await;
    let cookie = cookie_of(&callback);
    assert_eq!(
        h.get("/admin/api/snippets/stats", None).await.status(),
        StatusCode::UNAUTHORIZED
    );

    for language in ["python", "python", "py"] {
        h.get(
            &format!("/snippets/search?query=upsert&language={language}"),
            None,
        )
        .await;
    }
    for arguments in [
        json!({"query": "filters", "language": "ts"}),
        json!({"query": "filters"}),
    ] {
        let call = json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "search_snippets", "arguments": arguments}});
        h.client
            .post(format!("{}/mcp", h.base))
            .json(&call)
            .send()
            .await
            .unwrap();
    }
    h.get("/api/search?q=upsert", None).await;
    wait_for_total(&h, &cookie, 6).await;

    let stats = h.json("/admin/api/snippets/stats?days=7", &cookie).await;
    assert_eq!(stats["total"], 5);
    assert_eq!(stats["without_language"], 1);
    let languages = h
        .json("/admin/api/snippets/languages?days=7", &cookie)
        .await;
    let languages: Vec<_> = languages
        .as_array()
        .unwrap()
        .iter()
        .map(|l| {
            (
                l["language"].as_str().unwrap(),
                l["source"].as_str().unwrap(),
                l["searched_as"].as_str().unwrap(),
                l["count"].as_i64().unwrap(),
            )
        })
        .collect();
    assert_eq!(languages[0], ("python", "rest", "python", 2));
    assert!(languages.contains(&("ts", "mcp", "typescript", 1)));
    assert!(languages.contains(&("", "mcp", "", 1)));

    let mcp = h
        .json("/admin/api/snippets/stats?days=7&source=mcp", &cookie)
        .await;
    assert_eq!(mcp["total"], 2);

    let none = h
        .json("/admin/api/snippets/requests?language=-", &cookie)
        .await;
    assert_eq!(none.as_array().unwrap().len(), 1);
    assert_eq!(none[0]["endpoint"], "mcp:search_snippets");
    assert_eq!(none[0]["language"], "");

    let top = h.json("/admin/api/snippets/top?source=rest", &cookie).await;
    assert_eq!(top[0]["query"], "upsert");
    assert_eq!(top[0]["language"], "python");
    assert_eq!(top[0]["count"], 2);
    let detail = h
        .json(
            &format!("/admin/api/requests/{}", top[0]["last_id"]),
            &cookie,
        )
        .await;
    assert_eq!(detail["language"], "python");

    // Without Qdrant, the index and the debugger are unavailable.
    assert_eq!(
        h.get("/admin/api/snippets/index", Some(&cookie))
            .await
            .status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        h.get("/admin/api/snippets/search?query=upsert", Some(&cookie))
            .await
            .status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
}

impl Harness {
    /// Sends a JSON-RPC message to the real docs MCP server.
    async fn mcp(&self, message: Value) -> Value {
        self.client
            .post(format!("{}/mcp/docs", self.base))
            .header("accept", "application/json, text/event-stream")
            .json(&message)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap()
    }
}

#[actix_web::test]
async fn noise_and_protocol_requests_can_be_left_out() {
    let h = Harness::start().await;
    let callback = h
        .sign_in(json!({"email": "ann@qdrant.com", "email_verified": true, "hd": "qdrant.com"}))
        .await;
    let cookie = cookie_of(&callback);

    let init = json!({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}});
    h.mcp(init).await;
    // A tool the server doesn't have: a JSON-RPC error, with HTTP 200.
    let unknown = h
        .mcp(json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "search", "arguments": {"query": "filters"}}}))
        .await;
    assert_eq!(unknown["error"]["code"], -32602, "{unknown}");
    // A tool that fails: a tool error, with HTTP 200.
    let missing = h
        .mcp(json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "get_page", "arguments": {"page": "/documentation/nope/"}}}))
        .await;
    assert_eq!(missing["result"]["isError"], true, "{missing}");
    // A scanner, and a real search.
    h.get("/api/search?q=%2E%2E%2F%2E%2E%2Fetc%2Fpasswd", None)
        .await;
    h.get("/api/search?q=hnsw", None).await;
    wait_for_total(&h, &cookie, 5).await;

    let requests = h.json("/admin/api/requests", &cookie).await;
    let logged: Vec<_> = requests
        .as_array()
        .unwrap()
        .iter()
        .rev()
        .map(|r| {
            (
                r["endpoint"].as_str().unwrap(),
                r["category"].as_str().unwrap(),
                r["status"].as_i64().unwrap(),
                r["rpc_error"]
                    .as_str()
                    .map(|e| e.split(':').next().unwrap().to_string()),
            )
        })
        .collect();
    assert_eq!(
        logged,
        [
            ("mcp:initialize", "protocol", 200, None),
            (
                "mcp:search",
                "unknown",
                200,
                Some("-32602 tool not found".into())
            ),
            ("mcp:get_page", "use", 200, Some("tool error".into())),
            ("search", "scanner", 200, None),
            ("search", "use", 200, None),
        ]
    );

    let all = h.json("/admin/api/stats?days=7", &cookie).await;
    assert_eq!(all["total"], 5);
    assert_eq!(
        (all["protocol"].as_i64(), all["noise"].as_i64()),
        (Some(1), Some(2))
    );
    assert_eq!(all["noise_errors"], 1);
    // Both MCP calls failed, though with HTTP 200.
    assert_eq!(all["errors"], 2);
    assert_eq!(all["mcp_tool_calls"], 1);
    assert_eq!(all["mcp_sessions"], 1);

    let clean = h
        .json(
            "/admin/api/stats?days=7&hide_protocol=true&hide_noise=true",
            &cookie,
        )
        .await;
    assert_eq!(clean["total"], 2);
    assert_eq!(clean["errors"], 1);
    let errors = h
        .json(
            "/admin/api/requests?hide_protocol=true&hide_noise=true&errors=true",
            &cookie,
        )
        .await;
    assert_eq!(errors.as_array().unwrap().len(), 1);
    assert_eq!(errors[0]["endpoint"], "mcp:get_page");
    let top = h
        .json("/admin/api/top?days=7&hide_noise=true", &cookie)
        .await;
    assert!(
        top.as_array()
            .unwrap()
            .iter()
            .all(|t| t["query"] != "filters"),
        "{top}"
    );
}
