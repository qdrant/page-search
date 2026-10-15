use std::{borrow::Cow, net::SocketAddr, sync::Arc};

use actix_cors::Cors;
use actix_web::{main, middleware, web::Data, App, HttpServer};
use qdrant_client::Qdrant;
use rmcp_actix_web::transport::{LocalSessionManager, StreamableHttpService};
use rust_search::admin::{AdminConfig, AdminPanel, ReplayTarget};
use rust_search::common::get_qdrant_url;
use rust_search::docs_mcp::{config::Config, DocsServer, SearchBackend};
use rust_search::embedding::QueryEncoder;
use rust_search::request_log::{self, ReplayToken, RequestLog};
use rust_search::{search, sections, skills, snippets};

/// Hosts the docs MCP server answers by default: this service's public name,
/// and loopback for local testing.
const DEFAULT_MCP_ALLOWED_HOSTS: &str = "search.qdrant.tech,localhost,127.0.0.1";
/// Where the request log is kept, relative to the working directory. The
/// container mounts a volume at `/service/data`.
const DEFAULT_REQUEST_LOG_PATH: &str = "data/requests.sqlite";
const DEFAULT_REQUEST_LOG_RETENTION_DAYS: u32 = 90;

#[main]
async fn main() -> std::io::Result<()> {
    let mut log_builder = env_logger::Builder::new();
    log_builder.parse_filters(&std::env::var("SERVICE_LOG_LEVEL").unwrap_or("info".into()));
    log_builder.init();

    let uri = std::env::var("SERVICE_URL").map_or("0.0.0.0:8005".into(), Cow::Owned);
    let qdrant_url = get_qdrant_url();
    let api_key = std::env::var("QDRANT_API_KEY");
    let addr: SocketAddr = uri.parse().expect("malformed URI");
    let mut builder = Qdrant::from_url(&qdrant_url);
    if let Ok(key) = &api_key {
        builder = builder.api_key(key.clone());
    }
    let qdrant = builder.build().unwrap();
    qdrant.health_check().await.unwrap();
    let qdrant = Arc::new(qdrant);
    let encoder = QueryEncoder::from_env().expect("invalid QUERY_EMBEDDING");

    // The docs MCP server at /mcp shares the Qdrant client and query encoder.
    let docs_config = Config::from_env().expect("invalid DOCS_MCP_* setting");
    let mcp_allowed_hosts: Vec<String> = std::env::var("MCP_ALLOWED_HOSTS")
        .unwrap_or(DEFAULT_MCP_ALLOWED_HOSTS.into())
        .split(',')
        .map(|h| h.trim().to_string())
        .filter(|h| !h.is_empty())
        .collect();
    // An empty list would turn the Host check off rather than fall back to
    // the default, leaving /mcp open to DNS rebinding.
    assert!(!mcp_allowed_hosts.is_empty(), "MCP_ALLOWED_HOSTS is empty");
    log::info!(
        "Docs MCP serving {:?} at /mcp for hosts {mcp_allowed_hosts:?}",
        docs_config.sources
    );
    let docs_mcp = DocsServer::new(
        docs_config,
        SearchBackend::Qdrant {
            client: qdrant.clone(),
            encoder: encoder.clone(),
        },
    );

    // The request log, and the admin panel that shows it. The service runs
    // without them rather than not at all.
    let request_log_path =
        std::env::var("REQUEST_LOG_PATH").unwrap_or(DEFAULT_REQUEST_LOG_PATH.into());
    let retention_days = std::env::var("REQUEST_LOG_RETENTION_DAYS")
        .ok()
        .map(|d| d.parse().expect("invalid REQUEST_LOG_RETENTION_DAYS"))
        .unwrap_or(DEFAULT_REQUEST_LOG_RETENTION_DAYS);
    let request_log = match RequestLog::open(&request_log_path, retention_days) {
        Ok(log) => {
            log::info!("Logging requests to {request_log_path} for {retention_days} days");
            Some(log)
        }
        Err(e) => {
            log::error!("Not logging requests: can't open {request_log_path}: {e}");
            None
        }
    };
    let replay_token = ReplayToken::random();
    let admin = match (AdminConfig::from_env(), &request_log) {
        (Ok(config), Some(log)) => {
            let host = if addr.ip().is_unspecified() {
                match addr {
                    SocketAddr::V4(_) => "127.0.0.1".to_string(),
                    SocketAddr::V6(_) => "[::1]".to_string(),
                }
            } else {
                match addr {
                    SocketAddr::V4(a) => a.ip().to_string(),
                    SocketAddr::V6(a) => format!("[{}]", a.ip()),
                }
            };
            let target = ReplayTarget {
                base_url: format!("http://{host}:{}", addr.port()),
                mcp_host: mcp_allowed_hosts[0].clone(),
                token: replay_token.clone(),
            };
            log::info!("Admin panel at /admin for {}", config.public_url);
            Some(AdminPanel::new(config, log.clone(), target))
        }
        (Err(reason), _) => {
            log::warn!("Admin panel is off: {reason}");
            None
        }
        (_, None) => {
            log::warn!("Admin panel is off: there is no request log");
            None
        }
    };

    let qdrant = Data::from(qdrant);
    let encoder = Data::new(encoder);
    let request_log = request_log.map(Data::new);
    let replay_token = Data::new(replay_token);
    let server = HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header();

        let docs_mcp = docs_mcp.clone();
        // Stateless with JSON responses: every request stands alone, so it
        // works through proxies and across replicas.
        let mcp = StreamableHttpService::builder()
            .service_factory(Arc::new(move || Ok(docs_mcp.clone())))
            .session_manager(Arc::new(LocalSessionManager::default()))
            .stateful_mode(false)
            .json_response(true)
            .allowed_hosts(mcp_allowed_hosts.clone())
            .build();

        let mut app = App::new()
            .app_data(qdrant.clone())
            .app_data(encoder.clone())
            .app_data(replay_token.clone());
        if let Some(log) = &request_log {
            app = app.app_data(log.clone());
        }
        app.wrap(middleware::from_fn(request_log::log_requests))
            .wrap(cors)
            // Actix's default Logger format plus one appended field (client IP from the
            // reverse-proxy hop). Field order is load-bearing for a downstream parser: append
            // new fields, never reorder or insert, or that parser breaks.
            .wrap(middleware::Logger::new(
                "%a \"%r\" %s %b \"%{Referer}i\" \"%{User-Agent}i\" %T \"%{X-Forwarded-For}i\"",
            ))
            .service(search::query_handler)
            .service(sections::md_handler)
            .service(skills::skills_handler)
            .service(snippets::search_handler)
            .service(mcp.scope_with_path("/mcp"))
            .configure(|cfg| match &admin {
                Some(admin) => admin.configure(cfg),
                None => AdminPanel::configure_disabled(cfg),
            })
    });
    server.bind(addr)?.run().await
}
