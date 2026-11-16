//! What kind of traffic a logged request is: real use of the service, MCP
//! protocol overhead, or noise, so the admin panel can leave out what
//! doesn't say anything about how the service is used.

use std::collections::HashSet;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::docs_mcp::DocsServer;

/// Bumped whenever [`categorize`] changes, to categorize the log again.
pub const CATEGORY_VERSION: i64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Category {
    /// A search, a page, or an MCP tool call.
    Use,
    /// MCP sessions and discovery: `initialize`, `tools/list`, `ping`, the
    /// `GET` that opens an event stream, and the like.
    Protocol,
    /// A call to an MCP tool or method this server doesn't have.
    Unknown,
    /// A scanner probing for weaknesses: paths like `/.env` or `wp-admin`,
    /// injection payloads, or a scanner's user agent.
    Scanner,
}

impl Category {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Use => "use",
            Self::Protocol => "protocol",
            Self::Unknown => "unknown",
            Self::Scanner => "scanner",
        }
    }
}

/// MCP methods that open, describe, or keep up a session.
const PROTOCOL_METHODS: &[&str] = &[
    "initialize",
    "ping",
    "tools/list",
    "prompts/list",
    "resources/list",
    "resources/templates/list",
    "logging/setLevel",
    "completion/complete",
];

/// Things in a path or a query that only scanners send.
const SCANNER_PATTERNS: &[&str] = &[
    "../",
    "..\\",
    "/.env",
    "/.git",
    "/.aws",
    "/.ssh",
    "/.htaccess",
    "/.htpasswd",
    "/.ds_store",
    "wp-admin",
    "wp-login",
    "wp-content",
    "wp-includes",
    "xmlrpc.php",
    "phpmyadmin",
    "phpinfo",
    "cgi-bin",
    "/etc/passwd",
    "win.ini",
    "<script",
    "javascript:",
    "${jndi:",
    "union select",
    "union all select",
    "' or '1'='1",
    "\" or \"1\"=\"1",
    "sleep(",
    "waitfor delay",
];

/// File types scanners ask for at the end of a path.
const SCANNER_EXTENSIONS: &[&str] = &[
    ".php", ".asp", ".aspx", ".jsp", ".cgi", ".env", ".sql", ".bak", ".old", ".swp", ".ini",
];

/// User agents of scanners, in lowercase.
const SCANNER_AGENTS: &[&str] = &[
    "zgrab",
    "masscan",
    "nuclei",
    "sqlmap",
    "nikto",
    "nmap",
    "censysinspect",
    "expanse",
    "l9explore",
    "l9tcpid",
    "internetmeasurement",
    "modatscanner",
    "wpscan",
    "dirbuster",
    "gobuster",
];

fn known_tools() -> &'static HashSet<String> {
    static TOOLS: OnceLock<HashSet<String>> = OnceLock::new();
    TOOLS.get_or_init(|| DocsServer::tool_names().into_iter().collect())
}

/// What a request is, from what the log keeps of it.
pub struct Request<'a> {
    pub endpoint: &'a str,
    pub path: &'a str,
    pub query_string: &'a str,
    pub query: Option<&'a str>,
    pub user_agent: Option<&'a str>,
    /// The JSON-RPC error of an MCP response, as [`super::rpc_error`] gives it.
    pub rpc_error: Option<&'a str>,
}

pub fn categorize(r: &Request) -> Category {
    if is_scanner(r) {
        return Category::Scanner;
    }
    let Some(name) = r.endpoint.strip_prefix("mcp:") else {
        return Category::Use;
    };
    // The server's own answer is the surest sign of an unknown tool.
    if r.rpc_error.is_some_and(|e| {
        e.starts_with("-32601 ") || (e.starts_with("-32602 ") && e.contains("tool not found"))
    }) {
        return Category::Unknown;
    }
    if known_tools().contains(name) {
        Category::Use
    } else if PROTOCOL_METHODS.contains(&name) || is_http_method(name) {
        Category::Protocol
    } else {
        Category::Unknown
    }
}

/// `mcp:GET` and the like: a request to `/mcp` that isn't a JSON-RPC POST.
fn is_http_method(name: &str) -> bool {
    !name.is_empty() && name.bytes().all(|b| b.is_ascii_uppercase())
}

fn is_scanner(r: &Request) -> bool {
    if let Some(agent) = r.user_agent {
        let agent = agent.to_lowercase();
        if SCANNER_AGENTS.iter().any(|a| agent.contains(a)) {
            return true;
        }
    }
    let decoded_query: String = url::form_urlencoded::parse(r.query_string.as_bytes())
        .map(|(k, v)| format!("{k}={v} "))
        .collect();
    let path = percent_decode(r.path).to_lowercase();
    let text = format!("{path} {decoded_query} {}", r.query.unwrap_or_default()).to_lowercase();
    if SCANNER_PATTERNS.iter().any(|p| text.contains(p)) {
        return true;
    }
    let last_segment = path
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or_default();
    SCANNER_EXTENSIONS.iter().any(|e| last_segment.ends_with(e))
}

/// `%2e%2e` as `..`, so encoded paths match the patterns too.
fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex = |b: u8| (b as char).to_digit(16);
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(h), Some(l)) = (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(endpoint: &str, path: &str, query: Option<&str>) -> Request<'static> {
        Request {
            endpoint: Box::leak(endpoint.to_string().into_boxed_str()),
            path: Box::leak(path.to_string().into_boxed_str()),
            query_string: "",
            query: query.map(|q| &*Box::leak(q.to_string().into_boxed_str())),
            user_agent: Some("claude-code/2.1"),
            rpc_error: None,
        }
    }

    #[test]
    fn mcp_calls_by_what_the_server_has() {
        assert_eq!(
            categorize(&request("mcp:search_docs", "/mcp", Some("filters"))),
            Category::Use
        );
        assert_eq!(
            categorize(&request("mcp:get_page", "/mcp", None)),
            Category::Use
        );
        assert_eq!(
            categorize(&request("mcp:initialize", "/mcp", None)),
            Category::Protocol
        );
        assert_eq!(
            categorize(&request("mcp:tools/list", "/mcp", None)),
            Category::Protocol
        );
        assert_eq!(
            categorize(&request("mcp:GET", "/mcp", None)),
            Category::Protocol
        );
        assert_eq!(
            categorize(&request("mcp:search", "/mcp", None)),
            Category::Unknown
        );
        assert_eq!(
            categorize(&request("mcp:sampling/createMessage", "/mcp", None)),
            Category::Unknown
        );
        let rejected = Request {
            rpc_error: Some("-32602 tool not found"),
            ..request("mcp:search_docs", "/mcp", None)
        };
        assert_eq!(categorize(&rejected), Category::Unknown);
        // A tool's own error is no unknown tool.
        let failed = Request {
            rpc_error: Some("tool error: Page not found"),
            ..request("mcp:get_page", "/mcp", None)
        };
        assert_eq!(categorize(&failed), Category::Use);
    }

    #[test]
    fn scanners_by_what_they_ask_for() {
        for (endpoint, path, query) in [
            ("md", "/md/.env", Some("/.env")),
            (
                "md",
                "/md/wp-admin/install.php",
                Some("/wp-admin/install.php"),
            ),
            ("md", "/md/documentation/%2e%2e/%2e%2e/etc/passwd", None),
            ("md", "/md/index.php", Some("/index.php")),
            ("search", "/api/search", Some("x' OR '1'='1")),
            ("search", "/api/search", Some("${jndi:ldap://x}")),
            ("mcp:get_page", "/mcp", Some("../../.git/config")),
        ] {
            assert_eq!(
                categorize(&request(endpoint, path, query)),
                Category::Scanner,
                "{path} {query:?}"
            );
        }
        let agent = Request {
            user_agent: Some("Mozilla/5.0 zgrab/0.x"),
            ..request("search", "/api/search", Some("hnsw"))
        };
        assert_eq!(categorize(&agent), Category::Scanner);
    }

    #[test]
    fn real_searches_and_pages_are_use() {
        for (endpoint, path, query) in [
            (
                "md",
                "/md/documentation/concepts/filtering/",
                Some("/documentation/concepts/filtering/"),
            ),
            (
                "md",
                "/md/documentation/foo/bar/baz",
                Some("/documentation/foo/bar/baz"),
            ),
            ("md", "/md/documentation/guides/installation/", None),
            ("search", "/api/search", Some("environment variables")),
            ("search", "/api/search", Some("admin api key")),
            ("search", "/api/search", Some("select payload fields")),
            ("snippets", "/snippets/search", Some("upsert points")),
            ("skills", "/skills", Some("deploy on kubernetes")),
        ] {
            assert_eq!(
                categorize(&request(endpoint, path, query)),
                Category::Use,
                "{path} {query:?}"
            );
        }
    }
}
