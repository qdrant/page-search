//! Settings for the docs MCP server, read from `DOCS_MCP_*` environment variables.

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
    /// Which parts of the site the tools serve. All of them by default.
    pub sources: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            sources: SOURCES.iter().map(|s| s.to_string()).collect(),
        }
    }
}

impl Config {
    /// Read `DOCS_MCP_SOURCES` (comma-separated), falling back to all sources.
    pub fn from_env() -> Result<Self, String> {
        Ok(Self {
            sources: match std::env::var("DOCS_MCP_SOURCES") {
                Ok(value) => parse_sources(&value)?,
                Err(_) => Self::default().sources,
            },
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
