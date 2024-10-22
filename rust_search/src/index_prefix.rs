mod common;

use crate::common::{get_embedding, get_qdrant_url, PREFIX_COLLECTION_NAME};
use anyhow::Result;
use itertools::Itertools;
use ort::{Environment, SessionBuilder};
use qdrant_client::config::QdrantConfig;
use qdrant_client::qdrant::{
    vectors_config::Config, OptimizersConfigDiff, PointId, PointStruct, VectorParams, Vectors,
    VectorsConfig,
};
use qdrant_client::qdrant::{CreateCollection, Distance, UpsertPointsBuilder, Value};
use qdrant_client::Qdrant;
use rust_tokenizers::tokenizer::BertTokenizer;
use std::collections::{HashMap, HashSet};
use std::{io::Write, sync::Arc};
use tokio::main;

const MODEL_PATH: &str = "all-MiniLM-L6-v2.onnx";
const VOCAB_PATH: &str = "vocab.txt";
const SPECIAL_TOKEN_PATH: &str = "special_tokens_map.json";

fn prefix_to_id(prefix: &str) -> PointId {
    let len = prefix.len();
    PointId::from(u64::from_le_bytes(if len < 8 {
        let mut result = [0_u8; 8];
        result[..len].copy_from_slice(prefix.as_bytes());
        result
    } else {
        prefix.as_bytes()[..8].try_into().unwrap()
    }))
}

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
    let tokenizer = BertTokenizer::from_file_with_special_token_mapping(
        VOCAB_PATH,
        true,
        false,
        SPECIAL_TOKEN_PATH,
    )
    .unwrap();
    let env = Arc::new(Environment::builder().build()?);
    let session = SessionBuilder::new(&env)?.with_model_from_file(MODEL_PATH)?;
    let id = &mut 1_u64;
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    let points = prefixes.into_iter().map(|prefix| {
        let vector = get_embedding(&tokenizer, &session, prefix);
        if *id % 100 == 0 {
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
            vectors: Some(Vectors::from(vector)),
            payload,
        }
    });

    // store the word prefixes with embedding
    let qdrant_url = get_qdrant_url();
    let mut config = QdrantConfig::from_url(&qdrant_url);
    if let Ok(key) = std::env::var("QDRANT_API_KEY") {
        config.set_api_key(&key);
    }
    let qdrant_client = Qdrant::new(config)?;

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
                        on_disk: Some(true),
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
