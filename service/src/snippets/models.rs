use std::collections::HashMap;

use qdrant_client::qdrant::Value;
use qdrant_client::Payload;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct SourceInfo {
    pub url: String,
    pub hash: String,
    pub lines: Option<(i64, i64)>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SnippetContext {
    pub before: String,
    pub after: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Snippet {
    pub code: String,
    pub language: String,
    pub version: String,
    pub revision: i64,
    pub package_name: String,
    pub source: SourceInfo,
    pub context: SnippetContext,
    pub description: Option<String>,
}

impl Snippet {
    pub fn from_payload(payload: HashMap<String, Value>) -> Option<Self> {
        Payload::from(payload).deserialize().ok()
    }
}

pub struct SnippetSearchResult(pub Vec<Snippet>);

impl SnippetSearchResult {
    pub fn to_json(&self) -> String {
        let values: Vec<serde_json::Value> = self
            .0
            .iter()
            .map(|s| serde_json::to_value(s).unwrap_or_default())
            .collect();
        serde_json::json!({ "result": values }).to_string()
    }

    pub fn to_markdown(&self) -> String {
        if self.0.is_empty() {
            return "No snippets found.".to_string();
        }
        let mut blocks: Vec<String> = Vec::new();
        for (i, s) in self.0.iter().enumerate() {
            blocks.push(format!("## Snippet {}\n", i + 1));
            blocks.push(format!(
                "*{}* (v{}) — {}\n",
                s.package_name, s.version, s.source.url
            ));
            if let Some(ref desc) = s.description {
                blocks.push(format!("{}\n", desc));
            }
            blocks.push(format!("```{}\n{}\n```\n", s.language, s.code));
        }
        blocks.join("\n")
    }
}
