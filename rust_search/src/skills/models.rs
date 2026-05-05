use std::collections::HashMap;

use qdrant_client::qdrant::Value;
use qdrant_client::Payload;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
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

pub struct SkillSearchResult {
    pub skills: Vec<Skill>,
    pub strategy: Option<String>,
}

impl SkillSearchResult {
    pub fn with_strategy(skills: Vec<Skill>, strategy: impl Into<String>) -> Self {
        Self { skills, strategy: Some(strategy.into()) }
    }

    pub fn len(&self) -> usize {
        self.skills.len()
    }

    pub fn to_markdown(&self) -> String {
        if self.skills.is_empty() {
            return "<!-- no skills found -->\n".to_string();
        }
        self.skills
            .iter()
            .map(|s| s.to_markdown())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}
