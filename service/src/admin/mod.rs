//! The admin panel at `/admin`: request statistics from the request log, and
//! replays of logged requests to show what the service returns for them now.
//! Its snippet view reports on snippet searches and runs them with each half
//! of the hybrid search shown on its own, with the Qdrant client the app
//! shares, if there is one.
//! Its feedback view lists what agents reported with the docs MCP server's
//! `submit_feedback` tool, for admins to dismiss once dealt with.
//! Google accounts on the allow lists sign in with OAuth.

mod api;
mod auth;
pub mod config;

use actix_session::config::{CookieContentSecurity, PersistentSession};
use actix_session::storage::CookieSessionStore;
use actix_session::SessionMiddleware;
use actix_web::cookie::{time, Key, SameSite};
use actix_web::web::{self, Data, ServiceConfig};
use actix_web::{HttpRequest, HttpResponse};

pub use api::ReplayTarget;
pub use config::AdminConfig;

use crate::feedback::FeedbackStore;
use crate::request_log::RequestLog;

const PAGE: &str = include_str!("page.html");

/// Everything the admin panel needs, built once and shared by the workers.
#[derive(Clone)]
pub struct AdminPanel {
    config: Data<AdminConfig>,
    key: Key,
    log: Data<RequestLog>,
    target: Data<ReplayTarget>,
    http: Data<reqwest::Client>,
    /// `None` when feedback isn't kept: the feedback API answers 503.
    feedback: Option<Data<FeedbackStore>>,
}

impl AdminPanel {
    pub fn new(
        config: AdminConfig,
        log: RequestLog,
        target: ReplayTarget,
        feedback: Option<FeedbackStore>,
    ) -> Self {
        let key = Key::derive_from(config.session_key.as_bytes());
        Self {
            config: Data::new(config),
            key,
            log: Data::new(log),
            target: Data::new(target),
            http: Data::new(reqwest::Client::new()),
            feedback: feedback.map(Data::new),
        }
    }

    /// Registers `/admin` and everything under it.
    pub fn configure(&self, cfg: &mut ServiceConfig) {
        let session = SessionMiddleware::builder(CookieSessionStore::default(), self.key.clone())
            .cookie_name("site_search_admin".into())
            .cookie_path("/admin".into())
            .cookie_secure(self.config.secure_cookies())
            .cookie_http_only(true)
            // Lax, so the cookie comes along when Google redirects back.
            .cookie_same_site(SameSite::Lax)
            .cookie_content_security(CookieContentSecurity::Private)
            .session_lifecycle(
                PersistentSession::default()
                    .session_ttl(time::Duration::hours(auth::SESSION_HOURS)),
            )
            .build();
        let mut scope = web::scope("/admin");
        if let Some(feedback) = &self.feedback {
            scope = scope.app_data(feedback.clone());
        }
        cfg.service(
            scope
                .wrap(session)
                .app_data(self.config.clone())
                .app_data(self.log.clone())
                .app_data(self.target.clone())
                .app_data(self.http.clone())
                .service(web::resource(["", "/"]).get(page))
                .service(auth::login)
                .service(auth::callback)
                .service(auth::logout)
                .service(api::me)
                .service(api::stats)
                .service(api::top)
                .service(api::requests)
                .service(api::request_detail)
                .service(api::replay)
                .service(api::snippet_stats)
                .service(api::snippet_languages)
                .service(api::snippet_top)
                .service(api::snippet_requests)
                .service(api::snippet_index)
                .service(api::snippet_search)
                .service(api::feedback_list)
                .service(api::feedback_pages)
                .service(api::feedback_dismiss),
        );
    }

    /// `/admin` when the panel isn't configured.
    pub fn configure_disabled(cfg: &mut ServiceConfig) {
        cfg.service(web::scope("/admin").default_service(web::to(|| async {
            HttpResponse::ServiceUnavailable().body("The admin panel is not configured.")
        })));
    }
}

async fn page(req: HttpRequest, config: Data<AdminConfig>) -> HttpResponse {
    auth::page_or_sign_in(&req, &config, PAGE)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
