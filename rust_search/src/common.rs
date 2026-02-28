// Allow unused code, as not all submodules use all functions
#![allow(dead_code)]

use ndarray::{Array, Axis, CowArray};
use ort::tensor::OrtOwnedTensor;
use ort::Session;
use ort::Value as OrtValue;

use rust_tokenizers::tokenizer::{BertTokenizer, Tokenizer, TruncationStrategy};

pub const COLLECTION_NAME: &str = "site";
pub const PREFIX_COLLECTION_NAME: &str = "prefix-cache";
pub const MODEL_PATH: &str = "all-MiniLM-L6-v2.onnx";

pub fn get_qdrant_url() -> String {
    match std::env::var("QDRANT_URL") {
        Ok(url) => url,
        Err(_) => match std::env::var("QDRANT_HOST") {
            Ok(host) => format!("https://{}:6334", host),
            Err(_) => "http://localhost:6334".to_string(),
        },
    }
}

pub fn get_embedding(tokenizer: &BertTokenizer, session: &Session, query: &str) -> Vec<f32> {
    // tokenize
    let mut encoding = tokenizer.encode(query, None, 512, &TruncationStrategy::LongestFirst, 1);
    let alloc = session.allocator();
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
    pooled.as_slice().unwrap().to_vec()
}
