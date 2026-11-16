use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

const FUSION_BASE_URL: &str = "https://fusion.qdrant.tech";

static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

fn shared_client() -> &'static reqwest::Client {
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .pool_idle_timeout(std::time::Duration::from_secs(60))
            .build()
            .expect("reqwest client")
    })
}

#[derive(Serialize)]
struct TextRequest<'a> {
    text: &'a str,
}

#[derive(Deserialize)]
struct ClassifyResponse {
    score: i32,
}

#[derive(Deserialize)]
struct ExtractResponse {
    tokens: Vec<String>,
}

fn api_key() -> Option<String> {
    std::env::var("AUTO_FUSION_API_KEY").ok().filter(|k| !k.is_empty())
}

pub async fn classify(query: &str) -> anyhow::Result<i32> {
    let key = api_key().ok_or_else(|| anyhow::anyhow!("AUTO_FUSION_API_KEY not set"))?;
    let resp: ClassifyResponse = shared_client()
        .post(format!("{FUSION_BASE_URL}/v1/classify"))
        .bearer_auth(key)
        .json(&TextRequest { text: query })
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    Ok(resp.score)
}

pub async fn extract(query: &str) -> anyhow::Result<Vec<String>> {
    let key = api_key().ok_or_else(|| anyhow::anyhow!("AUTO_FUSION_API_KEY not set"))?;
    let resp: ExtractResponse = shared_client()
        .post(format!("{FUSION_BASE_URL}/v1/extract"))
        .bearer_auth(key)
        .json(&TextRequest { text: query })
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    Ok(resp.tokens)
}

pub enum Strategy {
    Dense,
    Hybrid,
    Bm25,
}

impl From<i32> for Strategy {
    fn from(score: i32) -> Self {
        match score {
            0..=2 => Strategy::Dense,
            3..=6 => Strategy::Hybrid,
            _ => Strategy::Bm25,
        }
    }
}
