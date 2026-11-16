use std::env;

/// Default public address of the service, where Google sends admins back to.
const DEFAULT_PUBLIC_URL: &str = "https://search.qdrant.tech";

/// Google sign-in for the admin panel, from the environment.
#[derive(Clone, Debug)]
pub struct AdminConfig {
    pub client_id: String,
    pub client_secret: String,
    /// At least 32 bytes; the session cookie's keys are derived from it.
    pub session_key: String,
    /// Scheme and host the service is reached at, without a trailing slash.
    pub public_url: String,
    /// Google Workspace domains whose accounts may sign in.
    pub allowed_domains: Vec<String>,
    /// Accounts that may sign in whatever their domain.
    pub allowed_emails: Vec<String>,
    pub auth_url: String,
    pub token_url: String,
    pub userinfo_url: String,
}

fn list(name: &str) -> Vec<String> {
    env::var(name)
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

impl AdminConfig {
    /// The settings, or why the admin panel stays off.
    pub fn from_env() -> Result<Self, String> {
        let required = |name: &str| {
            env::var(name)
                .ok()
                .filter(|v| !v.trim().is_empty())
                .ok_or_else(|| format!("{name} is not set"))
        };
        let config = Self {
            client_id: required("GOOGLE_OAUTH_CLIENT_ID")?,
            client_secret: required("GOOGLE_OAUTH_CLIENT_SECRET")?,
            session_key: required("ADMIN_SESSION_KEY")?,
            public_url: env::var("ADMIN_PUBLIC_URL")
                .unwrap_or(DEFAULT_PUBLIC_URL.into())
                .trim_end_matches('/')
                .to_string(),
            allowed_domains: list("ADMIN_ALLOWED_DOMAINS"),
            allowed_emails: list("ADMIN_ALLOWED_EMAILS"),
            auth_url: "https://accounts.google.com/o/oauth2/v2/auth".into(),
            token_url: "https://oauth2.googleapis.com/token".into(),
            userinfo_url: "https://openidconnect.googleapis.com/v1/userinfo".into(),
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.session_key.len() < 32 {
            return Err("ADMIN_SESSION_KEY is shorter than 32 bytes".into());
        }
        // Without a list, any Google account could sign in.
        if self.allowed_domains.is_empty() && self.allowed_emails.is_empty() {
            return Err("neither ADMIN_ALLOWED_DOMAINS nor ADMIN_ALLOWED_EMAILS is set".into());
        }
        Ok(())
    }

    pub fn redirect_uri(&self) -> String {
        format!("{}/admin/auth/callback", self.public_url)
    }

    pub fn secure_cookies(&self) -> bool {
        self.public_url.starts_with("https://")
    }

    /// Whether a verified Google account may use the admin panel. `hosted_domain`
    /// is Google's `hd` claim: the Workspace the account belongs to, which an
    /// email address alone doesn't prove.
    pub fn allows(&self, email: &str, hosted_domain: Option<&str>) -> bool {
        let email = email.to_lowercase();
        self.allowed_emails.contains(&email)
            || hosted_domain.map(str::to_lowercase).is_some_and(|hd| {
                self.allowed_domains.contains(&hd) && email.ends_with(&format!("@{hd}"))
            })
    }
}
