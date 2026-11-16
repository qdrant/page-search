//! A log of the requests to the service, kept for the admin panel.

pub mod category;
mod middleware;
pub mod queries;
pub mod snippet_queries;
mod store;

/// The endpoint of `/snippets/search` in the log.
pub const SNIPPETS_ENDPOINT: &str = "snippets";
/// The endpoint of the docs MCP server's `search_snippets` tool in the log.
pub const MCP_SNIPPETS_ENDPOINT: &str = "mcp:search_snippets";

/// SQL that holds for a failed request: an HTTP error, or an MCP call
/// answered with an error, which comes with HTTP 200.
pub const IS_ERROR: &str = "(status >= 400 OR rpc_error IS NOT NULL)";

pub use middleware::{log_requests, ReplayToken, REPLAY_HEADER};
pub use queries::RequestFilter;
pub use store::{now_ms, RequestLog, RequestRecord};
