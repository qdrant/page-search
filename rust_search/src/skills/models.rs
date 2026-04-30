use std::collections::HashMap;

use qdrant_client::qdrant::Value;
use qdrant_client::Payload;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub content: String,
    pub url: String,
    pub page: String,
    pub parent_pages: Vec<String>,
}

impl Skill {
    pub fn from_payload(payload: HashMap<String, Value>) -> Option<Self> {
        Payload::from(payload).deserialize().ok()
    }

    pub fn to_markdown(&self) -> String {
        format!(
            "---\nname: {}\ndescription: {}\n---\n{}",
            self.name, self.description, self.content
        )
    }
}

pub struct SkillSearchResult(pub Vec<Skill>);

impl SkillSearchResult {
    pub fn to_markdown(&self) -> String {
        self.0
            .iter()
            .map(|s| s.to_markdown())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}
