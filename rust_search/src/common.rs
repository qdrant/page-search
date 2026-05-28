// Allow unused code, as not all submodules use all functions
#![allow(dead_code)]

pub const COLLECTION_NAME: &str = "site";
pub const PREFIX_COLLECTION_NAME: &str = "prefix-cache";

pub fn get_qdrant_url() -> String {
    match std::env::var("QDRANT_URL") {
        Ok(url) => url,
        Err(_) => match std::env::var("QDRANT_HOST") {
            Ok(host) => format!("https://{}:6334", host),
            Err(_) => "http://localhost:6334".to_string(),
        },
    }
}
