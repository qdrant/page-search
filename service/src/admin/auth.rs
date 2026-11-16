//! Google sign-in (OAuth 2.0 authorization code flow) and the admin session.

use std::future::{ready, Ready};

use actix_session::{Session, SessionExt};
use actix_web::http::header;
use actix_web::web::{Data, Query};
use actix_web::{dev::Payload, get, post, FromRequest, HttpRequest, HttpResponse};
use rand::Rng;
use serde::Deserialize;

use super::config::AdminConfig;
use crate::request_log::now_ms;

const STATE_KEY: &str = "oauth_state";
const EMAIL_KEY: &str = "email";
const DOMAIN_KEY: &str = "hd";
const SIGNED_IN_KEY: &str = "signed_in_at";
/// Sign-ins last this long, checked on the server: the cookie store alone
/// can't expire a session.
pub const SESSION_HOURS: i64 = 12;

/// A signed-in admin. Handlers taking it answer 401 to everyone else.
pub struct Admin {
    pub email: String,
}

fn signed_in(session: &Session, config: &AdminConfig) -> Option<Admin> {
    let email: String = session.get(EMAIL_KEY).ok()??;
    let at: i64 = session.get(SIGNED_IN_KEY).ok()??;
    let fresh = now_ms() - at < SESSION_HOURS * 3600 * 1000;
    let domain: Option<String> = session.get(DOMAIN_KEY).ok()?;
    // The allow lists may have changed since the sign-in.
    (fresh && config.allows(&email, domain.as_deref())).then_some(Admin { email })
}

impl FromRequest for Admin {
    type Error = actix_web::Error;
    type Future = Ready<Result<Self, Self::Error>>;

    fn from_request(req: &HttpRequest, _: &mut Payload) -> Self::Future {
        let admin = req
            .app_data::<Data<AdminConfig>>()
            .and_then(|config| signed_in(&req.get_session(), config));
        ready(admin.ok_or_else(|| actix_web::error::ErrorUnauthorized("Sign in first")))
    }
}

/// The page, for admins, or a redirect to sign in.
pub fn page_or_sign_in(
    req: &HttpRequest,
    config: &AdminConfig,
    page: &'static str,
) -> HttpResponse {
    match signed_in(&req.get_session(), config) {
        Some(_) => HttpResponse::Ok()
            .content_type("text/html; charset=utf-8")
            .insert_header((header::CACHE_CONTROL, "no-store"))
            .body(page),
        None => redirect("/admin/login"),
    }
}

fn redirect(location: &str) -> HttpResponse {
    HttpResponse::Found()
        .insert_header((header::LOCATION, location))
        .finish()
}

fn message_page(status: actix_web::http::StatusCode, message: &str) -> HttpResponse {
    let message = message
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    HttpResponse::build(status)
        .content_type("text/html; charset=utf-8")
        .body(format!(
            "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>\
             <title>Site Search Admin</title>\
             <body style='font-family:system-ui,sans-serif;max-width:32rem;margin:4rem auto;padding:0 16px'>\
             <h1 style='font-size:1.25rem'>Site Search Admin</h1><p>{message}</p>\
             <p><a href='/admin/login'>Sign in with Google</a></p></body>"
        ))
}

#[get("/login")]
pub async fn login(config: Data<AdminConfig>, session: Session) -> HttpResponse {
    let state: String = rand::thread_rng()
        .gen::<[u8; 24]>()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if session.insert(STATE_KEY, &state).is_err() {
        return HttpResponse::InternalServerError().finish();
    }
    let mut url = url::Url::parse(&config.auth_url).expect("invalid auth URL");
    url.query_pairs_mut()
        .append_pair("client_id", &config.client_id)
        .append_pair("redirect_uri", &config.redirect_uri())
        .append_pair("response_type", "code")
        .append_pair("scope", "openid email")
        .append_pair("state", &state)
        .append_pair("prompt", "select_account");
    // Only a hint for Google's account chooser; the callback checks the domain.
    if let [domain] = config.allowed_domains.as_slice() {
        url.query_pairs_mut().append_pair("hd", domain);
    }
    redirect(url.as_str())
}

#[derive(Deserialize)]
pub struct Callback {
    code: Option<String>,
    state: Option<String>,
    error: Option<String>,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
}

#[derive(Deserialize)]
struct UserInfo {
    email: Option<String>,
    #[serde(default)]
    email_verified: bool,
    hd: Option<String>,
}

#[get("/auth/callback")]
pub async fn callback(
    config: Data<AdminConfig>,
    http: Data<reqwest::Client>,
    session: Session,
    query: Query<Callback>,
) -> HttpResponse {
    use actix_web::http::StatusCode;

    let expected: Option<String> = session.remove_as(STATE_KEY).and_then(Result::ok);
    let Callback { code, state, error } = query.into_inner();
    if let Some(error) = error {
        return message_page(
            StatusCode::UNAUTHORIZED,
            &format!("Google sign-in failed: {error}."),
        );
    }
    let (Some(code), Some(state), Some(expected)) = (code, state, expected) else {
        return message_page(
            StatusCode::BAD_REQUEST,
            "The sign-in expired. Please try again.",
        );
    };
    if state != expected {
        return message_page(
            StatusCode::BAD_REQUEST,
            "The sign-in expired. Please try again.",
        );
    }

    let user = match fetch_user(&config, &http, &code).await {
        Ok(user) => user,
        Err(e) => {
            log::warn!("Admin sign-in failed: {e}");
            return message_page(
                StatusCode::BAD_GATEWAY,
                "Couldn't confirm the sign-in with Google.",
            );
        }
    };
    let Some(email) = user.email.filter(|_| user.email_verified) else {
        return message_page(
            StatusCode::FORBIDDEN,
            "This Google account has no verified email address.",
        );
    };
    if !config.allows(&email, user.hd.as_deref()) {
        log::warn!("Admin sign-in refused for {email}");
        return message_page(
            StatusCode::FORBIDDEN,
            &format!("{email} may not use the admin panel."),
        );
    }

    log::info!("Admin signed in: {email}");
    session.renew();
    if session.insert(EMAIL_KEY, email.to_lowercase()).is_err()
        || session.insert(DOMAIN_KEY, user.hd).is_err()
        || session.insert(SIGNED_IN_KEY, now_ms()).is_err()
    {
        return HttpResponse::InternalServerError().finish();
    }
    redirect("/admin")
}

async fn fetch_user(
    config: &AdminConfig,
    http: &reqwest::Client,
    code: &str,
) -> anyhow::Result<UserInfo> {
    let redirect_uri = config.redirect_uri();
    let token: TokenResponse = http
        .post(&config.token_url)
        .form(&[
            ("code", code),
            ("client_id", &config.client_id),
            ("client_secret", &config.client_secret),
            ("redirect_uri", &redirect_uri),
            ("grant_type", "authorization_code"),
        ])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    Ok(http
        .get(&config.userinfo_url)
        .bearer_auth(token.access_token)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}

#[post("/logout")]
pub async fn logout(session: Session) -> HttpResponse {
    session.purge();
    message_page(actix_web::http::StatusCode::OK, "You are signed out.")
}
