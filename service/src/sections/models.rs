use std::collections::HashMap;

use qdrant_client::qdrant::Value;
use qdrant_client::Payload;
use regex::Regex;
use serde::Deserialize;

use super::links::{rebase_service_links, rewrite_links};

pub fn slugify_heading(title: &str) -> String {
    let s = title.to_lowercase();
    let s = s.trim();
    let re_spaces = Regex::new(r"[\s\-_]+").unwrap();
    let s = re_spaces.replace_all(s, "-");
    let re_nonword = Regex::new(r"[^\w\-]").unwrap();
    re_nonword.replace_all(&s, "").to_string()
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
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
            .map(|s| rewrite_links(&s.content))
            .collect::<Vec<_>>()
            .join("\n");
        let sections_text = rebase_service_links(&sections_text, base_url);

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

#[cfg(test)]
mod tests {
    use super::*;

    fn bulk_upload_page() -> SectionSearchResult {
        SectionSearchResult {
            sections: vec![Section {
                title: "Bulk Upload".into(),
                slug: "bulk-upload".into(),
                content: "> Explore Qdrant's agent skills catalog at https://skills.qdrant.tech/\n\
                          > Search the documentation at https://search.qdrant.tech/md/documentation?q=your+query+here\n\
                          > Use this file for an index of the main pages: https://qdrant.tech/llms.txt\n\n\
                          # Bulk Upload\n\n\
                          [Upsert points in batches](/documentation/manage-data/points/#upload-points)."
                    .into(),
                url: "documentation/manage-data/bulk-upload".into(),
                page: "documentation/manage-data/bulk-upload".into(),
                parent_sections: vec!["bulk-upload".into()],
                parent_pages: vec![],
                level: 1,
                line: 0,
            }],
            sublinks: None,
        }
    }

    #[test]
    fn page_links_use_the_requested_host() {
        for base in ["https://search.qdrant.tech", "https://skills.qdrant.tech"] {
            assert_eq!(
                bulk_upload_page().to_markdown(
                    "/md/documentation/manage-data/bulk-upload",
                    None,
                    base
                ),
                format!(
                    "Read one level up: {base}/md/documentation/manage-data\n\n\
                     > Explore Qdrant's agent skills catalog at https://skills.qdrant.tech/\n\
                     > Search the documentation at {base}/md/documentation?q=your+query+here\n\
                     > Use this file for an index of the main pages: https://qdrant.tech/llms.txt\n\n\
                     # Bulk Upload\n\n\
                     [Upsert points in batches](/md/documentation/manage-data/points?s=upload-points)."
                )
            );
        }
    }
}
