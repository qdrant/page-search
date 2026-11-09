//! Build the `pages` collection, which suggests pages in place of missing
//! ones, from the pages of the `sections` collection. Run it after
//! `site_search.sections`.
//!
//! `QUERY_EMBEDDING` chooses how page meanings are embedded, as for queries:
//! by Qdrant Cloud on upsert (the default) or, with `local`, in this process.

use std::collections::BTreeMap;

use anyhow::{ensure, Result};
use qdrant_client::qdrant::value::Kind;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, Distance, FieldType, Modifier,
    PayloadIncludeSelector, PointId, ScrollPointsBuilder, SparseVectorParamsBuilder,
    SparseVectorsConfigBuilder, UpsertPointsBuilder, VectorParamsBuilder, VectorsConfigBuilder,
};
use qdrant_client::Qdrant;
use rust_search::common::{get_qdrant_url, PAGE_COLLECTION_NAME, SECTION_COLLECTION_NAME};
use rust_search::embedding::QueryEncoder;
use rust_search::similar_paths::{page_point, PageText, MEANING_VECTOR, SLUG_VECTOR, WORDS_VECTOR};
use tokio::main;

/// Size of [`rust_search::embedding::NEURAL_ENCODER`] vectors.
const MEANING_SIZE: u64 = 384;
const BATCH: usize = 256;

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
    let mut builder = Qdrant::from_url(&get_qdrant_url());
    if let Ok(key) = std::env::var("QDRANT_API_KEY") {
        builder = builder.api_key(key);
    }
    let client = builder.build()?;
    let encoder = QueryEncoder::from_env()?;

    let pages = section_pages(&client).await?;
    // Keep the current pages rather than replace them with nothing.
    ensure!(!pages.is_empty(), "{SECTION_COLLECTION_NAME} has no pages");
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
    for field in ["page", "source", "ancestors"] {
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
