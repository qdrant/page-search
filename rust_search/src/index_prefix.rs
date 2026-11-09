use anyhow::Result;
use itertools::Itertools;
use qdrant_client::qdrant::{
    vectors_config::Config, Memory, OptimizersConfigDiff, PointStruct, VectorParams, Vectors,
    VectorsConfig,
};
use qdrant_client::qdrant::{CreateCollection, Distance, Document, UpsertPointsBuilder, Value};
use rust_search::embedding::NEURAL_ENCODER;
use rust_search::qdrant::{client_from_env, prefix_to_id, PREFIX_COLLECTION_NAME};
use std::collections::{HashMap, HashSet};
use std::io::Write;
use tokio::main;

fn n_chars(word: &str, n: usize) -> &str {
    if word.len() <= n {
        word
    } else {
        &word[..word.char_indices().nth(n).map_or(0, |(i, _)| i)]
    }
}

#[main]
async fn main() -> Result<()> {
    // Get word prefixes
    let words = std::fs::read_to_string("words.txt")?;
    let mut prefixes = HashSet::new();
    for word in words.lines() {
        for n in 1..6 {
            prefixes.insert(n_chars(word, n));
        }
    }
    prefixes.remove("");
    println!("{} prefixes found", prefixes.len());

    // embed all word prefixes
    let id = &mut 1_u64;
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    let points = prefixes.into_iter().map(|prefix| {
        let vector = Vectors::from(Document::new(prefix, NEURAL_ENCODER));
        if (*id).is_multiple_of(100) {
            write!(stdout, "{id}").unwrap();
        } else {
            write!(stdout, ".").unwrap();
        }
        stdout.flush().unwrap();
        let payload = vec![("prefix".to_string(), prefix.into())]
            .into_iter()
            .collect::<HashMap<_, Value>>();

        PointStruct {
            id: Some(prefix_to_id(prefix)),
            vectors: Some(vector),
            payload,
        }
    });

    // store the word prefixes with embedding
    let qdrant_client = client_from_env()?;

    if !qdrant_client
        .collection_exists(PREFIX_COLLECTION_NAME)
        .await?
    {
        qdrant_client
            .create_collection(CreateCollection {
                collection_name: PREFIX_COLLECTION_NAME.into(),
                vectors_config: Some(VectorsConfig {
                    config: Some(Config::Params(VectorParams {
                        size: 384,
                        distance: Distance::Cosine as i32,
                        memory: Some(Memory::Cold as i32),
                        ..Default::default()
                    })),
                }),
                optimizers_config: Some(OptimizersConfigDiff {
                    indexing_threshold: Some(0), // disable indexing
                    ..Default::default()
                }),
                ..Default::default()
            })
            .await?;
    }
    for p in &points.chunks(1024) {
        let p: Vec<_> = p.collect();

        let request = UpsertPointsBuilder::new(PREFIX_COLLECTION_NAME, p);

        qdrant_client.upsert_points(request).await?;
    }

    Ok(())
}
