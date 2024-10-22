mod common;

use crate::common::{get_embedding, get_qdrant_url, COLLECTION_NAME, MODEL_PATH};
use anyhow::Result;
use ort::{Environment, SessionBuilder};
use qdrant_client::{
    config::QdrantConfig,
    qdrant::{
        vectors_config::Config, CreateCollection, Distance, PointId, PointStruct, UpsertPointsBuilder, Value, VectorParams, Vectors, VectorsConfig
    },
    Qdrant,
};
use rust_tokenizers::tokenizer::BertTokenizer;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Write},
    sync::Arc,
};
use tokio::main;

const SITE_DATA: &str = "../page-search/data/abstracts.jsonl";
const VOCAB_PATH: &str = "vocab.txt";
const SPECIAL_TOKEN_PATH: &str = "special_tokens_map.json";

#[main]
async fn main() -> Result<()> {
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
    let site_file = File::open(SITE_DATA)?;
    let site_reader = BufReader::new(site_file);
    let mut points = site_reader.lines().map(move |line| {
        let payload: HashMap<String, Value> = serde_json::from_str(&line.unwrap()).unwrap();
        let text = payload.get("text").and_then(Value::as_str).unwrap();

        let vector = get_embedding(&tokenizer, &session, text);

        if *id % 100 == 0 {
            write!(stdout, "{id}").unwrap();
        } else {
            write!(stdout, ".").unwrap();
        }
        stdout.flush().unwrap();
        PointStruct {
            id: Some(PointId::from(std::mem::replace(id, *id + 1))),
            payload,
            vectors: Some(Vectors::from(vector)),
        }
    });

    // store the word prefixes with embedding
    let qdrant_url = get_qdrant_url();
    let mut config = QdrantConfig::from_url(&qdrant_url);
    if let Ok(key) = std::env::var("QDRANT_API_KEY") {
        config.set_api_key(&key);
    }
    let qdrant_client = Qdrant::new(config)?;

    if !qdrant_client.collection_exists(COLLECTION_NAME).await? {
        qdrant_client
            .create_collection(CreateCollection {
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
        let request = UpsertPointsBuilder::new(COLLECTION_NAME, p);

        qdrant_client.upsert_points(request).await?;
    }
    Ok(())
}
