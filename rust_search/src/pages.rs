//! Pages as Qdrant stores them: their metadata in the `pages` collection,
//! and their Markdown in the sections of the `sections` collection.
//!
//! `index_pages` takes the metadata from the site's page index,
//! `/docs-manifest.json`, when it builds the collection. Everything that
//! lists, describes, or reads pages gets them from Qdrant, not from the site.

use qdrant_client::qdrant::r#match::MatchValue;
use qdrant_client::qdrant::{
    Condition, Filter, PayloadIncludeSelector, PointId, ScrollPointsBuilder,
};
use qdrant_client::{Payload, Qdrant};
use serde::{Deserialize, Serialize};

use crate::qdrant::{PAGE_COLLECTION_NAME, SECTION_COLLECTION_NAME};

/// Pages with a code example in every client language, linked from the
/// documentation. They can be read, but aren't listed or suggested: the page
/// index doesn't list them, so they aren't in the `pages` collection.
pub const SNIPPET_PAGES: &str = "documentation/snippets";
/// Most sections read for one page.
const MAX_SECTIONS: u32 = 10_000;

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
    /// The values of the fields an entry of the page index leaves out.
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

/// Whether `page`, written like the `page` payload field, is a snippet page.
pub fn is_snippet_page(page: &str) -> bool {
    page == SNIPPET_PAGES || page.starts_with(&format!("{SNIPPET_PAGES}/"))
}

/// A point of the `pages` collection without its vectors.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct StoredPage {
    /// See [`page_key`].
    pub page: String,
    #[serde(default)]
    pub title: String,
    #[serde(flatten)]
    pub meta: PageMeta,
}

/// Every page of the `pages` collection.
pub async fn all_pages(client: &Qdrant) -> anyhow::Result<Vec<StoredPage>> {
    let mut pages = Vec::new();
    let mut offset: Option<PointId> = None;
    loop {
        let mut scroll = ScrollPointsBuilder::new(PAGE_COLLECTION_NAME)
            .limit(1000)
            .with_payload(true)
            .with_vectors(false);
        if let Some(offset) = offset.take() {
            scroll = scroll.offset(offset);
        }
        let response = client.scroll(scroll).await?;
        for point in response.result {
            pages.push(Payload::from(point.payload).deserialize()?);
        }
        match response.next_page_offset {
            Some(next) => offset = Some(next),
            None => break,
        }
    }
    Ok(pages)
}

/// The Markdown of `page`, written like the `page` payload field: its
/// sections in order. `None` if it has no sections.
pub async fn page_markdown(client: &Qdrant, page: &str) -> anyhow::Result<Option<String>> {
    let response = client
        .scroll(
            ScrollPointsBuilder::new(SECTION_COLLECTION_NAME)
                .filter(Filter::must([Condition::matches(
                    "page",
                    MatchValue::Keyword(page.to_string()),
                )]))
                .order_by("line")
                .limit(MAX_SECTIONS)
                .with_payload(PayloadIncludeSelector {
                    fields: vec!["content".to_string()],
                })
                .with_vectors(false),
        )
        .await?;
    let sections: Vec<String> = response
        .result
        .into_iter()
        .filter_map(|point| {
            #[derive(Deserialize)]
            struct Content {
                content: String,
            }
            let content: Content = Payload::from(point.payload).deserialize().ok()?;
            Some(content.content)
        })
        .collect();
    // Each section is the lines from its heading to the next one.
    Ok((!sections.is_empty()).then(|| sections.join("\n")))
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
    fn reads_stored_pages() {
        let stored: StoredPage = serde_json::from_value(serde_json::json!({
            "page": "documentation/faq",
            "title": "FAQ",
            "source": "documentation",
            "ancestors": ["documentation"],
            "kind": "section",
            "rendered": false,
        }))
        .unwrap();
        assert_eq!(stored.title, "FAQ");
        assert_eq!(stored.meta.kind, "section");
        assert!(!stored.meta.rendered);
    }

    #[test]
    fn recognizes_snippet_pages() {
        assert!(is_snippet_page(
            "documentation/snippets/query-points/hybrid-rrf"
        ));
        assert!(!is_snippet_page("documentation/snippets-guide"));
        assert!(!is_snippet_page("documentation/search"));
    }

    #[test]
    fn writes_paths_without_slashes_at_the_ends() {
        assert_eq!(page_key("/documentation/faq/"), "documentation/faq");
        assert_eq!(page_key("articles/x"), "articles/x");
    }
}
