//! Build the `pages` collection, which suggests pages in place of missing
//! ones, from the pages of the `sections` collection. Run it after
//! `site_search.sections`.

use std::collections::BTreeSet;

use anyhow::{ensure, Result};
use qdrant_client::qdrant::value::Kind;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, FieldType, Modifier,
    PayloadIncludeSelector, PointId, ScrollPointsBuilder, SparseVectorParamsBuilder,
    SparseVectorsConfigBuilder, UpsertPointsBuilder,
};
use qdrant_client::Qdrant;
use rust_search::common::{get_qdrant_url, PAGE_COLLECTION_NAME, SECTION_COLLECTION_NAME};
use rust_search::similar_paths::{page_point, SLUG_VECTOR};
use tokio::main;

/// Every page that has sections.
async fn section_pages(client: &Qdrant) -> Result<BTreeSet<String>> {
    let mut pages = BTreeSet::new();
    let mut offset: Option<PointId> = None;
    loop {
        let mut scroll = ScrollPointsBuilder::new(SECTION_COLLECTION_NAME)
            .limit(1000)
            .with_payload(PayloadIncludeSelector {
                fields: vec!["page".to_string()],
            })
            .with_vectors(false);
        if let Some(offset) = offset.take() {
            scroll = scroll.offset(offset);
        }
        let response = client.scroll(scroll).await?;
        for point in response.result {
            if let Some(Kind::StringValue(page)) =
                point.payload.get("page").and_then(|v| v.kind.clone())
            {
                pages.insert(page);
            }
        }
        match response.next_page_offset {
            Some(next) => offset = Some(next),
            None => break,
        }
    }
    Ok(pages)
}

#[main]
async fn main() -> Result<()> {
    let mut builder = Qdrant::from_url(&get_qdrant_url());
    if let Ok(key) = std::env::var("QDRANT_API_KEY") {
        builder = builder.api_key(key);
    }
    let client = builder.build()?;

    let pages = section_pages(&client).await?;
    // Keep the current pages rather than replace them with nothing.
    ensure!(!pages.is_empty(), "{SECTION_COLLECTION_NAME} has no pages");
    println!("{} pages found", pages.len());

    if client.collection_exists(PAGE_COLLECTION_NAME).await? {
        client.delete_collection(PAGE_COLLECTION_NAME).await?;
    }
    let mut sparse_vectors = SparseVectorsConfigBuilder::default();
    // IDF makes trigrams that are on few pages count more.
    sparse_vectors.add_named_vector_params(
        SLUG_VECTOR,
        SparseVectorParamsBuilder::default().modifier(Modifier::Idf),
    );
    client
        .create_collection(
            CreateCollectionBuilder::new(PAGE_COLLECTION_NAME)
                .sparse_vectors_config(sparse_vectors),
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

    let points: Vec<_> = pages
        .iter()
        .enumerate()
        .map(|(id, page)| page_point(id as u64, page))
        .collect();
    for chunk in points.chunks(1024) {
        client
            .upsert_points(
                UpsertPointsBuilder::new(PAGE_COLLECTION_NAME, chunk.to_vec()).wait(true),
            )
            .await?;
    }
    println!("{} pages indexed", points.len());
    Ok(())
}
