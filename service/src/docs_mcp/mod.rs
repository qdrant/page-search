//! The docs MCP server: lets AI assistants search and read the content
//! published on qdrant.tech, as indexed in Qdrant. The search service mounts it at
//! `search.qdrant.tech/mcp`.

pub mod config;
pub mod library;
mod server;

#[cfg(test)]
pub(crate) use server::tests::FixedBackend;
pub use server::{Backend, DocsServer};
