//! Build the `pages` collection from the site's page index,
//! `/docs-manifest.json`, which lists every page that has a Markdown version.
//!
//! The collection suggests pages in place of missing ones, and holds what the
//! page index says about each page (see [`rust_search::pages::PageMeta`]), so
//! the service never reads the page index itself. `DOCS_MANIFEST_URL` sets
//! where it is read from.
//!
//! `QUERY_EMBEDDING` chooses how page meanings are embedded, as for queries:
//! by Qdrant Cloud on upsert (the default) or, with `local`, in this process.

use std::time::Duration;

use anyhow::{ensure, Result};
use qdrant_client::qdrant::{
    CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, Distance, FieldType, Modifier,
    SparseVectorParamsBuilder, SparseVectorsConfigBuilder, UpsertPointsBuilder,
    VectorParamsBuilder, VectorsConfigBuilder,
};
use rust_search::embedding::QueryEncoder;
use rust_search::pages::{page_key, PageMeta, RENDERED_FIELD};
use rust_search::qdrant::{client_from_env, PAGE_COLLECTION_NAME};
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

/// The pages of the page index, with paths and aliases written like the
/// `page` payload field.
fn pages(manifest: Vec<ManifestEntry>) -> Vec<PageText> {
    manifest
        .into_iter()
        .map(|entry| {
            let mut meta = entry.meta;
            for alias in &mut meta.aliases {
                *alias = page_key(alias);
            }
            PageText {
                page: page_key(&entry.path),
                title: entry.title,
                meta,
            }
        })
        .collect()
}

#[main]
async fn main() -> Result<()> {
    let client = client_from_env()?;
    let encoder = QueryEncoder::from_env()?;

    // Read before the current pages are deleted, so they stay if this fails.
    let pages = pages(manifest().await?);
    println!("{} pages found", pages.len());

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
            .embed_documents(chunk.iter().map(PageText::meaning).collect())
            .await?;
        let points: Vec<_> = chunk
            .iter()
            .zip(meanings)
            .enumerate()
            .map(|(i, (page, meaning))| page_point((n * BATCH + i) as u64, page, meaning))
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

    fn entry(value: serde_json::Value) -> ManifestEntry {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn reads_pages_from_the_page_index() {
        let pages = pages(vec![
            entry(serde_json::json!({
                "path": "/documentation/search/filtering/",
                "title": "Filtering",
                "description": "Filter points by payload",
                "partition": "develop",
                "weight": 30,
                "aliases": ["/documentation/filtering", "/documentation/concepts/filtering/"],
                "markdown_url": "https://qdrant.tech/documentation/search/filtering/index.md",
            })),
            entry(serde_json::json!({
                "path": "/documentation/faq/",
                "title": "FAQ",
                "kind": "section",
                "rendered": false,
            })),
        ]);

        let filtering = &pages[0];
        assert_eq!(filtering.page, "documentation/search/filtering");
        assert_eq!(filtering.title, "Filtering");
        assert_eq!(
            filtering.meaning(),
            "Filtering. filtering. Filter points by payload"
        );
        assert_eq!(filtering.meta.partition, "develop");
        assert_eq!(filtering.meta.weight, 30);
        assert_eq!(
            filtering.meta.aliases,
            [
                "documentation/filtering",
                "documentation/concepts/filtering"
            ]
        );
        assert!(filtering.meta.rendered);

        let faq = &pages[1];
        assert_eq!(faq.page, "documentation/faq");
        assert!(!faq.meta.rendered);
        assert_eq!(pages.len(), 2);
    }
}
