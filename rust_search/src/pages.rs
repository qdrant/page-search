//! What the `pages` collection knows about each page besides its vectors.
//!
//! `index_pages` takes it from the site's page index, `/docs-manifest.json`,
//! when it builds the collection. Everything that lists or describes pages
//! reads it from Qdrant, not from the site.

use serde::{Deserialize, Serialize};

/// Payload field that is `false` for a section that only groups pages in the
/// sidebar and has no page of its own.
pub const RENDERED_FIELD: &str = "rendered";

/// A page's metadata, stored in its payload as these fields.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(default)]
pub struct PageMeta {
    pub description: String,
    /// `page`, or `section` for a page that lists the pages below it.
    pub kind: String,
    /// The part of the documentation the page belongs to, such as `develop`.
    /// Empty outside the documentation.
    pub partition: String,
    /// Order among sibling pages in the sidebar, lowest first.
    #[serde(deserialize_with = "any_number")]
    pub weight: i64,
    /// Old paths of the page, written like `page` (see [`page_key`]).
    pub aliases: Vec<String>,
    /// Publication date, `YYYY-MM-DD`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub date: Option<String>,
    /// Date of the last change, `YYYY-MM-DD`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lastmod: Option<String>,
    /// Where the page is hosted, if not on qdrant.tech.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_url: Option<String>,
    /// See [`RENDERED_FIELD`].
    pub rendered: bool,
}

impl Default for PageMeta {
    /// A page the page index doesn't list.
    fn default() -> Self {
        Self {
            description: String::new(),
            kind: "page".to_string(),
            partition: String::new(),
            weight: 0,
            aliases: Vec::new(),
            date: None,
            lastmod: None,
            external_url: None,
            rendered: true,
        }
    }
}

/// Hugo writes some weights as floats, such as `1000.0`.
fn any_number<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<i64, D::Error> {
    Ok(f64::deserialize(deserializer)? as i64)
}

/// A path written like the `page` payload field: below the site root, without
/// slashes at either end, such as `documentation/search/filtering`.
pub fn page_key(path: &str) -> String {
    path.trim_matches('/').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_manifest_fields_and_fills_in_the_rest() {
        let meta: PageMeta = serde_json::from_value(serde_json::json!({
            "description": "Filter points",
            "weight": 1000.0,
            "aliases": ["/documentation/filtering"],
            "date": "2024-01-10",
            "url": "https://qdrant.tech/documentation/search/filtering/",
        }))
        .unwrap();
        assert_eq!(meta.weight, 1000);
        assert_eq!(meta.kind, "page");
        assert!(meta.rendered);
        assert_eq!(meta.date.as_deref(), Some("2024-01-10"));
    }

    #[test]
    fn leaves_missing_dates_out_of_the_payload() {
        let value = serde_json::to_value(PageMeta::default()).unwrap();
        assert!(value.get("date").is_none());
        assert_eq!(value[RENDERED_FIELD], true);
    }

    #[test]
    fn writes_paths_without_slashes_at_the_ends() {
        assert_eq!(page_key("/documentation/faq/"), "documentation/faq");
        assert_eq!(page_key("articles/x"), "articles/x");
    }
}
