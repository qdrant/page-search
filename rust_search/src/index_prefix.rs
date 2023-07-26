use anyhow::Result;
use ndarray::{Array, Axis, CowArray};
use ort::{tensor::OrtOwnedTensor, Environment, SessionBuilder, Value as OrtValue};
use qdrant_client::prelude::*;
use qdrant_client::qdrant::{
    vectors_config::Config, PointId, PointStruct, VectorParams, Vectors, VectorsConfig,
};
use rust_tokenizers::tokenizer::{BertTokenizer, Tokenizer, TruncationStrategy};
use std::{io::Write, sync::Arc};
use tokio::main;

const MODEL_PATH: &str = "all-MiniLM-L6-v2.onnx";
const VOCAB_PATH: &str = "vocab.txt";
const SPECIAL_TOKEN_PATH: &str = "special_tokens_map.json";
const COLLECTION_NAME: &str = "prefix-cache";

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
        &word[..word.char_indices().nth(n).map(|(i, _)| i).unwrap_or(0)]
    }
}

#[main]
async fn main() -> Result<()> {
    // Get word prefixes
    let words = std::fs::read_to_string("words.txt")?;
    let mut prefixes = std::collections::HashSet::new();
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
    let alloc = session.allocator();
    let id = &mut 1_u64;
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    let mut points = prefixes.into_iter().map(|prefix| {
        let mut encoding =
            tokenizer.encode(&prefix, None, 512, &TruncationStrategy::LongestFirst, 1);
        let token_ids = std::mem::take(&mut encoding.token_ids);
        let shape = (1, token_ids.len());
        let token_ids = Array::from_shape_vec(shape, token_ids).unwrap();
        let attentions = Array::from_elem(shape, 1_i64);
        let type_ids = Array::from_elem(shape, 0_i64);
        // embed
        let output: OrtOwnedTensor<f32, _> = session
            .run(vec![
                OrtValue::from_array(alloc, &CowArray::from(token_ids.into_dyn())).unwrap(),
                OrtValue::from_array(alloc, &CowArray::from(attentions.into_dyn())).unwrap(),
                OrtValue::from_array(alloc, &CowArray::from(type_ids.into_dyn())).unwrap(),
            ])
            .unwrap()[0]
            .try_extract()
            .unwrap();
        let pooled = output.view().mean_axis(Axis(1)).unwrap();
        let vector = pooled.as_slice().unwrap().to_vec();
        if *id % 100 == 0 {
            write!(stdout, "{id}").unwrap();
        } else {
            write!(stdout, ".").unwrap();
        }
        stdout.flush().unwrap();
        PointStruct {
            id: Some(prefix_to_id(prefix)),
            vectors: Some(Vectors::from(vector)),
            payload: Default::default(),
        }
    });

    // store the word prefixes with embedding
    let qdrant_url = std::env::var("QDRANT_URL").unwrap_or("http://localhost:6334".to_string());
    let mut config = QdrantClientConfig::from_url(&qdrant_url);
    if let Ok(key) = std::env::var("QDRANT_API_KEY") {
        config.set_api_key(&key);
    }
    let qdrant_client = QdrantClient::new(Some(config))?;

    if !qdrant_client.has_collection(COLLECTION_NAME).await? {
        qdrant_client
            .create_collection(&CreateCollection {
                collection_name: COLLECTION_NAME.into(),
                vectors_config: Some(VectorsConfig {
                    config: Some(Config::Params(VectorParams {
                        size: 384,
                        distance: Distance::Cosine as i32,
                        ..Default::default()
                    })),
                }),
                ..Default::default()
            })
            .await?;
    }
    loop {
        let p = (&mut points).take(1024).collect::<Vec<_>>();
        if p.is_empty() {
            break;
        }
        qdrant_client
            .upsert_points(COLLECTION_NAME, p, None)
            .await?;
    }
    Ok(())
}
