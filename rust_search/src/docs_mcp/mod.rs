//! The docs MCP server: lets AI assistants search and read the content
//! published on qdrant.tech. The search service mounts it at
//! `search.qdrant.tech/mcp`.

pub mod config;
mod server;
pub mod site;

pub use server::{DocsServer, SearchBackend};
