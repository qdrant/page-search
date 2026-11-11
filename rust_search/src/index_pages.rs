//! Build the `pages` collection from the pages of the `sections` collection
//! and the metadata of the site's page index, `/docs-manifest.json`. Run it
//! after `site_search.sections`.
//!
//! The collection suggests pages in place of missing ones, and holds what the
//! page index says about each page (see [`rust_search::pages::PageMeta`]), so
//! the service never reads the page index itself. `DOCS_MANIFEST_URL` sets
//! where it is read from.
//!
//! `QUERY_EMBEDDING` chooses how page meanings are embedded, as for queries:
//! by Qdrant Cloud on upsert (the default) or, with `local`, in this process.

use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::{ensure, Result};
use qdrant_client::qdrant::value::Kind;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, Distance, FieldType, Modifier,
    PayloadIncludeSelector, PointId, ScrollPointsBuilder, SparseVectorParamsBuilder,
    SparseVectorsConfigBuilder, UpsertPointsBuilder, VectorParamsBuilder, VectorsConfigBuilder,
};
use qdrant_client::Qdrant;
use rust_search::embedding::QueryEncoder;
use rust_search::pages::{page_key, PageMeta, RENDERED_FIELD};
use rust_search::qdrant::{client_from_env, PAGE_COLLECTION_NAME, SECTION_COLLECTION_NAME};
use rust_search::similar_paths::{page_point, PageText, MEANING_VECTOR, SLUG_VECTOR, WORDS_VECTOR};
use serde::Deserialize;
use tokio::main;

/// Size of [`rust_search::embedding::NEURAL_ENCODER`] vectors.
const MEANING_SIZE: u64 = 384;
const BATCH: usize = 256;
const DEFAULT_MANIFEST_URL: &str = "https://qdrant.tech/docs-manifest.json";

/// An entry of the site's page index.
#[derive(Deserialize)]
struct ManifestEntry {
    /// Such as `/documentation/search/filtering/`.
    path: String,
    #[serde(default)]
    title: String,
    #[serde(flatten)]
    meta: PageMeta,
}

/// The site's page index, from `DOCS_MANIFEST_URL`.
async fn manifest() -> Result<Vec<ManifestEntry>> {
    #[derive(Deserialize)]
    struct Manifest {
        pages: Vec<ManifestEntry>,
    }
    let url = std::env::var("DOCS_MANIFEST_URL").unwrap_or(DEFAULT_MANIFEST_URL.into());
    let manifest: Manifest = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()?
        .get(&url)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    ensure!(!manifest.pages.is_empty(), "{url} lists no pages");
    Ok(manifest.pages)
}

/// The pages of the `sections` collection with their metadata from the page
/// index, whose titles replace those of the first headings, and the sections
/// the page index lists that have no page of their own.
///
/// Pages the page index lists that have no sections can't be read, so they
/// are left out, and returned separately.
fn with_metadata(
    pages: Vec<PageText>,
    manifest: Vec<ManifestEntry>,
) -> (Vec<(PageText, PageMeta)>, Vec<String>) {
    let mut listed: BTreeMap<String, ManifestEntry> = manifest
        .into_iter()
        .map(|entry| (page_key(&entry.path), entry))
        .collect();
    let mut found = Vec::with_capacity(pages.len());
    for mut page in pages {
        let meta = match listed.remove(&page.page) {
            Some(entry) => {
                if !entry.title.is_empty() {
                    page.title = entry.title;
                }
                entry.meta
            }
            None => PageMeta::default(),
        };
        found.push((page, meta));
    }
    let mut unreadable = Vec::new();
    for (page, entry) in listed {
        if entry.meta.rendered {
            unreadable.push(page);
            continue;
        }
        let text = PageText {
            page,
            title: entry.title,
            headings: Vec::new(),
        };
        found.push((text, entry.meta));
    }
    for (_, meta) in &mut found {
        for alias in &mut meta.aliases {
            *alias = page_key(alias);
        }
    }
    (found, unreadable)
}

#[derive(Default)]
struct Page {
    /// Line and title of the first level-1 heading.
    title: Option<(i64, String)>,
    /// Level-2 headings by line.
    headings: BTreeMap<i64, String>,
}

fn string(point: &qdrant_client::qdrant::RetrievedPoint, key: &str) -> Option<String> {
    match point.payload.get(key)?.kind.as_ref()? {
        Kind::StringValue(value) => Some(value.clone()),
        _ => None,
    }
}

fn integer(point: &qdrant_client::qdrant::RetrievedPoint, key: &str) -> Option<i64> {
    match point.payload.get(key)?.kind.as_ref()? {
        Kind::IntegerValue(value) => Some(*value),
        Kind::DoubleValue(value) => Some(*value as i64),
        _ => None,
    }
}

/// Every page that has sections, with its title and headings.
async fn section_pages(client: &Qdrant) -> Result<Vec<PageText>> {
    let mut pages: BTreeMap<String, Page> = BTreeMap::new();
    let mut offset: Option<PointId> = None;
    loop {
        let mut scroll = ScrollPointsBuilder::new(SECTION_COLLECTION_NAME)
            .limit(1000)
            .with_payload(PayloadIncludeSelector {
                fields: ["page", "title", "level", "line"]
                    .map(String::from)
                    .to_vec(),
            })
            .with_vectors(false);
        if let Some(offset) = offset.take() {
            scroll = scroll.offset(offset);
        }
        let response = client.scroll(scroll).await?;
        for point in &response.result {
            let Some(path) = string(point, "page") else {
                continue;
            };
            let page = pages.entry(path).or_default();
            let title = string(point, "title").unwrap_or_default();
            let line = integer(point, "line").unwrap_or_default();
            match integer(point, "level") {
                _ if title.is_empty() => {}
                Some(1) if page.title.as_ref().is_none_or(|(l, _)| line < *l) => {
                    page.title = Some((line, title));
                }
                Some(2) => {
                    page.headings.insert(line, title);
                }
                _ => {}
            }
        }
        match response.next_page_offset {
            Some(next) => offset = Some(next),
            None => break,
        }
    }
    Ok(pages
        .into_iter()
        .map(|(page, found)| PageText {
            title: found.title.map(|(_, title)| title).unwrap_or_default(),
            headings: found.headings.into_values().collect(),
            page,
        })
        .collect())
}

#[main]
async fn main() -> Result<()> {
    let client = client_from_env()?;
    let encoder = QueryEncoder::from_env()?;

    let pages = section_pages(&client).await?;
    // Keep the current pages rather than replace them with nothing.
    ensure!(!pages.is_empty(), "{SECTION_COLLECTION_NAME} has no pages");
    println!("{} pages found", pages.len());
    let (pages, unreadable) = with_metadata(pages, manifest().await?);
    if !unreadable.is_empty() {
        println!(
            "{} pages of the page index have no sections, left out: {}",
            unreadable.len(),
            unreadable.join(", ")
        );
    }

    if client.collection_exists(PAGE_COLLECTION_NAME).await? {
        client.delete_collection(PAGE_COLLECTION_NAME).await?;
    }
    let mut dense = VectorsConfigBuilder::default();
    dense.add_named_vector_params(
        MEANING_VECTOR,
        VectorParamsBuilder::new(MEANING_SIZE, Distance::Cosine),
    );
    let mut sparse = SparseVectorsConfigBuilder::default();
    // Unit-length n-gram vectors: the dot product is their cosine similarity.
    sparse.add_named_vector_params(SLUG_VECTOR, SparseVectorParamsBuilder::default());
    // BM25 needs IDF from the collection.
    sparse.add_named_vector_params(
        WORDS_VECTOR,
        SparseVectorParamsBuilder::default().modifier(Modifier::Idf),
    );
    client
        .create_collection(
            CreateCollectionBuilder::new(PAGE_COLLECTION_NAME)
                .vectors_config(dense)
                .sparse_vectors_config(sparse),
        )
        .await?;
    for field in ["page", "source", "ancestors", "aliases"] {
        client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(
                    PAGE_COLLECTION_NAME,
                    field,
                    FieldType::Keyword,
                )
                .wait(true),
            )
            .await?;
    }

    client
        .create_field_index(
            CreateFieldIndexCollectionBuilder::new(
                PAGE_COLLECTION_NAME,
                RENDERED_FIELD,
                FieldType::Bool,
            )
            .wait(true),
        )
        .await?;

    for (n, chunk) in pages.chunks(BATCH).enumerate() {
        let meanings = encoder
            .embed_documents(chunk.iter().map(|(page, _)| page.meaning()).collect())
            .await?;
        let points: Vec<_> = chunk
            .iter()
            .zip(meanings)
            .enumerate()
            .map(|(i, ((page, meta), meaning))| {
                page_point((n * BATCH + i) as u64, page, meta, meaning)
            })
            .collect();
        client
            .upsert_points(UpsertPointsBuilder::new(PAGE_COLLECTION_NAME, points).wait(true))
            .await?;
    }
    println!("{} pages indexed", pages.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(path: &str, title: &str) -> PageText {
        PageText {
            page: path.to_string(),
            title: title.to_string(),
            headings: vec!["Usage".to_string()],
        }
    }

    fn entry(value: serde_json::Value) -> ManifestEntry {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn adds_metadata_from_the_page_index() {
        let (pages, unreadable) = with_metadata(
            vec![
                page("documentation/search/filtering", "Filtering points"),
                page("pricing", "Pricing"),
            ],
            vec![
                entry(serde_json::json!({
                    "path": "/documentation/search/filtering/",
                    "title": "Filtering",
                    "partition": "develop",
                    "weight": 30,
                    "aliases": ["/documentation/filtering", "/documentation/concepts/filtering/"],
                })),
                entry(serde_json::json!({
                    "path": "/documentation/faq/",
                    "title": "FAQ",
                    "kind": "section",
                    "rendered": false,
                })),
                entry(serde_json::json!({"path": "/blog/no-sections/", "title": "Lost"})),
            ],
        );

        let (filtering, meta) = &pages[0];
        assert_eq!(filtering.title, "Filtering");
        assert_eq!(filtering.headings, ["Usage"]);
        assert_eq!(meta.partition, "develop");
        assert_eq!(meta.weight, 30);
        assert_eq!(
            meta.aliases,
            [
                "documentation/filtering",
                "documentation/concepts/filtering"
            ]
        );

        // Pages the page index doesn't list keep their own title.
        let (pricing, meta) = &pages[1];
        assert_eq!(pricing.title, "Pricing");
        assert_eq!(meta, &PageMeta::default());

        let (faq, meta) = &pages[2];
        assert_eq!(faq.page, "documentation/faq");
        assert_eq!(faq.title, "FAQ");
        assert!(!meta.rendered);

        assert_eq!(pages.len(), 3);
        assert_eq!(unreadable, ["blog/no-sections"]);
    }
}
