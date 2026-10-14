//! Turning query text into the vector Qdrant searches with.

use qdrant_client::qdrant::{Document, VectorInput};

/// Model the `site` and `sections` collections are indexed with.
pub const NEURAL_ENCODER: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// How queries are embedded.
///
/// Production sends the text to Qdrant Cloud, which embeds it with
/// [`NEURAL_ENCODER`]. A local Qdrant can't do that, so builds with the
/// `local-embeddings` feature can embed in-process with the same model instead.
#[derive(Clone)]
pub enum QueryEncoder {
    Cloud,
    #[cfg(feature = "local-embeddings")]
    Local(std::sync::Arc<std::sync::Mutex<fastembed::TextEmbedding>>),
}

impl QueryEncoder {
    /// Chosen by `QUERY_EMBEDDING`: `cloud` (the default) or `local`.
    pub fn from_env() -> anyhow::Result<Self> {
        match std::env::var("QUERY_EMBEDDING").as_deref() {
            Err(_) | Ok("cloud") => Ok(Self::Cloud),
            #[cfg(feature = "local-embeddings")]
            Ok("local") => Self::local(),
            #[cfg(not(feature = "local-embeddings"))]
            Ok("local") => anyhow::bail!(
                "QUERY_EMBEDDING=local needs a build with `--features local-embeddings`"
            ),
            Ok(other) => anyhow::bail!("Unknown QUERY_EMBEDDING {other:?}, use cloud or local"),
        }
    }

    /// Load the model, downloading it on first use into `FASTEMBED_CACHE_DIR`
    /// (default `.fastembed_cache`).
    #[cfg(feature = "local-embeddings")]
    pub fn local() -> anyhow::Result<Self> {
        use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

        let mut options = TextInitOptions::new(EmbeddingModel::AllMiniLML6V2);
        if let Ok(dir) = std::env::var("FASTEMBED_CACHE_DIR") {
            options = options.with_cache_dir(dir.into());
        }
        let model = TextEmbedding::try_new(options)?;
        Ok(Self::Local(std::sync::Arc::new(std::sync::Mutex::new(
            model,
        ))))
    }

    pub async fn encode(&self, text: &str) -> anyhow::Result<VectorInput> {
        match self {
            Self::Cloud => Ok(VectorInput::from(Document::new(text, NEURAL_ENCODER))),
            #[cfg(feature = "local-embeddings")]
            Self::Local(model) => {
                let model = model.clone();
                let text = text.to_string();
                // Inference is CPU-bound; keep it off the async workers.
                let mut vectors =
                    tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<Vec<f32>>> {
                        let mut model = model
                            .lock()
                            .map_err(|_| anyhow::anyhow!("embedding model lock poisoned"))?;
                        Ok(model.embed([text], None)?)
                    })
                    .await??;
                Ok(VectorInput::new_dense(vectors.remove(0)))
            }
        }
    }
}
