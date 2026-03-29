use std::collections::HashMap;

use qdrant_client::qdrant::Value;
use qdrant_client::Payload;
use regex::Regex;
use serde::Deserialize;

pub fn slugify_heading(title: &str) -> String {
    let s = title.to_lowercase();
    let s = s.trim();
    let re_spaces = Regex::new(r"[\s\-_]+").unwrap();
    let s = re_spaces.replace_all(s, "-");
    let re_nonword = Regex::new(r"[^\w\-]").unwrap();
    re_nonword.replace_all(&s, "").to_string()
}

#[derive(Debug, Deserialize)]
pub struct Section {
    pub title: String,
    pub slug: String,
    pub content: String,
    pub url: String,
    pub page: String,
    pub parent_sections: Vec<String>,
    pub parent_pages: Vec<String>,
    pub level: i64,
    pub line: i64,
}

impl Section {
    pub fn from_payload(payload: HashMap<String, Value>) -> Option<Self> {
        Payload::from(payload).deserialize().ok()
    }
}

pub struct SectionSearchResult {
    pub sections: Vec<Section>,
    pub sublinks: Option<Vec<String>>,
}

impl SectionSearchResult {
    /// Render the result as markdown, mirroring the Python implementation.
    ///
    /// `request_path` is the full request path (e.g. `/md/documentation/guides`).
    /// `request_query` is the raw query string (e.g. `q=foo&s=bar`), if any.
    /// `base_url` is `{scheme}://{host}` (e.g. `https://example.com`).
    pub fn to_markdown(
        &self,
        request_path: &str,
        request_query: Option<&str>,
        base_url: &str,
    ) -> String {
        let mut sections: Vec<&Section> = self.sections.iter().collect();
        sections.sort_by(|a, b| (&a.url, a.line).cmp(&(&b.url, b.line)));

        // Compute up_url
        let stripped = request_path.trim_matches('/');
        let segments: Vec<&str> = stripped.split('/').collect();

        let has_section_query = request_query.map(|q| q.contains("s=")).unwrap_or(false);

        let (up_path, up_query): (String, Option<String>) =
            if has_section_query
                && !sections.is_empty()
                && sections[0].parent_sections.len() > 1
            {
                // Navigate to parent section
                let parent_idx = sections[0].parent_sections.len() - 2;
                let parent_section = &sections[0].parent_sections[parent_idx];
                (
                    format!("/{}", stripped),
                    Some(format!("s={}", parent_section)),
                )
            } else {
                // Navigate one path segment up
                let up_segments = &segments[..segments.len().saturating_sub(1)];
                (format!("/{}", up_segments.join("/")), None)
            };

        let up_url = if let Some(q) = up_query {
            format!("{}{}?{}", base_url, up_path, q)
        } else {
            format!("{}{}", base_url, up_path)
        };

        let sections_text: String = sections
            .iter()
            .map(|s| s.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        let mut result = format!("Read one level up: {}\n\n{}", up_url, sections_text);

        if let Some(ref sublinks) = self.sublinks {
            if !sublinks.is_empty() {
                result.push_str("\n## Subsites to Search\n");
                for sub in sublinks {
                    // sublinks are page paths like "documentation/guides/something"
                    // prefix with /md/ to match our routing
                    result.push_str(&format!("\n{}/md/{}", base_url, sub));
                }
            }
        }

        result
    }
}
