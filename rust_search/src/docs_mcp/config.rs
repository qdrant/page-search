//! Settings for the docs MCP server, read from `DOCS_MCP_*` environment variables.

use std::time::Duration;

/// Top-level sections of qdrant.tech the server serves. Each is a URL prefix
/// (`/documentation/`, `/articles/`, ...) and a value of the `sections`
/// payload field in the `site` collection.
pub const SOURCES: [&str; 4] = ["documentation", "articles", "blog", "course"];

/// Parse a comma-separated list of sources, keeping the given order and
/// dropping duplicates.
pub fn parse_sources(value: &str) -> Result<Vec<String>, String> {
    let mut sources: Vec<String> = Vec::new();
    for source in value.split(',') {
        let source = source.trim().trim_matches('/');
        if !source.is_empty() && !sources.iter().any(|s| s == source) {
            sources.push(source.to_string());
        }
    }
    let unknown: Vec<&str> = sources
        .iter()
        .map(String::as_str)
        .filter(|s| !SOURCES.contains(s))
        .collect();
    if !unknown.is_empty() {
        return Err(format!(
            "Unknown source(s): {}. Choose from: {}.",
            unknown.join(", "),
            SOURCES.join(", ")
        ));
    }
    if sources.is_empty() {
        return Err(format!(
            "No sources given. Choose from: {}.",
            SOURCES.join(", ")
        ));
    }
    Ok(sources)
}

#[derive(Clone, Debug)]
pub struct Config {
    /// Where pages are read from. Point it at a local `hugo serve`
    /// (http://localhost:1313) or a Netlify deploy preview to test unreleased
    /// docs. Search still describes whatever was last indexed.
    pub content_base_url: String,
    /// Which parts of the site the tools serve. All of them by default.
    pub sources: Vec<String>,
    /// How long to keep the page index before fetching it again.
    pub manifest_ttl: Duration,
    /// How long to wait for any request to the site.
    pub http_timeout: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            content_base_url: "https://qdrant.tech".to_string(),
            sources: SOURCES.iter().map(|s| s.to_string()).collect(),
            manifest_ttl: Duration::from_secs(300),
            http_timeout: Duration::from_secs(15),
        }
    }
}

impl Config {
    /// Read `DOCS_MCP_BASE_URL`, `DOCS_MCP_SOURCES` (comma-separated),
    /// `DOCS_MCP_MANIFEST_TTL`, and `DOCS_MCP_HTTP_TIMEOUT` (seconds), falling
    /// back to the defaults.
    pub fn from_env() -> Result<Self, String> {
        let defaults = Self::default();
        let seconds = |name: &str, default: Duration| -> Result<Duration, String> {
            match std::env::var(name) {
                Ok(value) => value
                    .parse::<f64>()
                    .map(Duration::from_secs_f64)
                    .map_err(|_| format!("{name} must be a number of seconds, not {value:?}")),
                Err(_) => Ok(default),
            }
        };
        Ok(Self {
            content_base_url: std::env::var("DOCS_MCP_BASE_URL")
                .map(|url| url.trim_end_matches('/').to_string())
                .unwrap_or(defaults.content_base_url),
            sources: match std::env::var("DOCS_MCP_SOURCES") {
                Ok(value) => parse_sources(&value)?,
                Err(_) => defaults.sources,
            },
            manifest_ttl: seconds("DOCS_MCP_MANIFEST_TTL", defaults.manifest_ttl)?,
            http_timeout: seconds("DOCS_MCP_HTTP_TIMEOUT", defaults.http_timeout)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sources() {
        assert_eq!(
            parse_sources(" blog, /documentation/ ,blog").unwrap(),
            ["blog", "documentation"]
        );
        assert!(parse_sources("docs").is_err());
        assert!(parse_sources(" , ").is_err());
    }
}
