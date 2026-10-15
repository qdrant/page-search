//! Search over the content of qdrant.tech, shared by the binaries: the HTTP
//! search service, which also serves the docs MCP server, and the collection
//! setup tools. The service also logs its requests for an admin panel.

pub mod admin;
pub mod common;
pub mod docs_mcp;
pub mod embedding;
pub mod request_log;
pub mod search;
pub mod sections;
pub mod skills;
pub mod snippets;
