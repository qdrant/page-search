//! The admin panel over HTTP, with Google sign-in faked by a mock server and
//! the service's endpoints by echo handlers.

use std::net::TcpListener;
use std::time::Duration;

use actix_web::{middleware, web, App, HttpRequest, HttpResponse, HttpServer};
use reqwest::{redirect::Policy, StatusCode};
use serde_json::{json, Value};
use wiremock::matchers::{body_string_contains, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::*;
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
        let server = HttpServer::new(move || {
            App::new()
                .app_data(log.clone())
                .app_data(token.clone())
                .wrap(middleware::from_fn(request_log::log_requests))
                .route("/api/search", web::get().to(echo_search))
                .route("/mcp", web::post().to(echo_mcp))
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
