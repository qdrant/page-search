//! Connecting to Qdrant, and the collections this project keeps there.

use qdrant_client::qdrant::PointId;
use qdrant_client::Qdrant;

/// A point per line of text of the site, searched by `/api/search`.
pub const SITE_COLLECTION_NAME: &str = "site";
/// Embeddings of word prefixes, for `/api/search` queries shorter than a word.
pub const PREFIX_COLLECTION_NAME: &str = "prefix-cache";
/// A point per section of a page, read by `/md` and the docs MCP server.
pub const SECTION_COLLECTION_NAME: &str = "sections";
/// A point per page, for suggesting pages in place of missing ones.
pub const PAGE_COLLECTION_NAME: &str = "pages";
/// A point per skill of skills.qdrant.tech, searched by `/skills`.
pub const SKILLS_COLLECTION_NAME: &str = "skills";
/// A point per code example, searched by `/snippets/search`.
pub const SNIPPET_COLLECTION_NAME: &str = "snippet-search";

/// `QDRANT_URL`, or the gRPC port of `QDRANT_HOST` over HTTPS, or a local Qdrant.
pub fn qdrant_url() -> String {
    match std::env::var("QDRANT_URL") {
        Ok(url) => url,
        Err(_) => match std::env::var("QDRANT_HOST") {
            Ok(host) => format!("https://{}:6334", host),
            Err(_) => "http://localhost:6334".to_string(),
        },
    }
}

/// A client for [`qdrant_url`], with `QDRANT_API_KEY` if it's set.
pub fn client_from_env() -> anyhow::Result<Qdrant> {
    let mut builder = Qdrant::from_url(&qdrant_url());
    if let Ok(key) = std::env::var("QDRANT_API_KEY") {
        builder = builder.api_key(key);
    }
    Ok(builder.build()?)
}

/// ID of a prefix in [`PREFIX_COLLECTION_NAME`]: its first 8 bytes, zero-padded.
pub fn prefix_to_id(prefix: &str) -> PointId {
    let len = prefix.len();
    PointId::from(u64::from_le_bytes(if len < 8 {
        let mut result = [0_u8; 8];
        result[..len].copy_from_slice(prefix.as_bytes());
        result
    } else {
        prefix.as_bytes()[..8].try_into().unwrap()
    }))
}
