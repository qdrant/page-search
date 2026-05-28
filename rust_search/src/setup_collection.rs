mod common;

use crate::common::{get_qdrant_url, COLLECTION_NAME};
use anyhow::Result;
use qdrant_client::{
    qdrant::{
        vectors_config::Config, CreateCollection, Distance, Document, PointId, PointStruct,
        UpsertPointsBuilder, Value, VectorParams, Vectors, VectorsConfig,
    },
    Qdrant,
};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Write},
};
use tokio::main;

const SITE_DATA: &str = "../page-search/data/abstracts.jsonl";

const NEURAL_ENCODER: &str = "sentence-transformers/all-MiniLM-L6-v2";

#[main]
async fn main() -> Result<()> {
    // embed all word prefixes
    let id = &mut 1_u64;
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    let site_file = File::open(SITE_DATA)?;
    let site_reader = BufReader::new(site_file);
    let mut points = site_reader.lines().map(move |line| {
        let payload: HashMap<String, Value> = serde_json::from_str(&line.unwrap()).unwrap();
        let text = payload.get("text").and_then(Value::as_str).unwrap();

        let vector = Vectors::from(Document::new(text, NEURAL_ENCODER));

        if (*id).is_multiple_of(100) {
            write!(stdout, "{id}").unwrap();
        } else {
            write!(stdout, ".").unwrap();
        }
        stdout.flush().unwrap();
        PointStruct {
            id: Some(PointId::from(std::mem::replace(id, *id + 1))),
            payload,
            vectors: Some(vector),
        }
    });

    // store the word prefixes with embedding
    let qdrant_url = get_qdrant_url();
    let mut builder = Qdrant::from_url(&qdrant_url);
    if let Ok(key) = std::env::var("QDRANT_API_KEY") {
        builder = builder.api_key(key);
    }
    let qdrant_client = builder.build()?;

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
