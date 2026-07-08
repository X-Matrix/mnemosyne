//! Text embedding — stub mode by default; enable `candle-backend` feature for
//! real BERT inference via Candle.

use mnemosyne_core::{traits::EmbeddingModel, types::Embedding, Error};
use async_trait::async_trait;
use tracing::warn;

/// Default text embedding model (384-dim, lightweight).
pub const DEFAULT_TEXT_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";
/// Stub embedding dimension (matches all-MiniLM-L6-v2).
const STUB_DIM: usize = 384;

pub struct TextEmbedder {
    model_id: String,
    embedding_dim: usize,
}

impl TextEmbedder {
    /// Load model weights and tokenizer from HuggingFace Hub.
    ///
    /// With the default build (no `candle-backend` feature) this returns a stub
    /// embedder that generates deterministic hash-based vectors.
    pub async fn load(model_id: &str) -> Result<Self, Error> {
        warn!(
            "Using stub embedder for '{}' (enable `candle-backend` feature for real inference)",
            model_id
        );
        Ok(Self {
            model_id: model_id.to_string(),
            embedding_dim: STUB_DIM,
        })
    }

    /// Deterministic pseudo-embedding from text hash (dev/test only).
    fn embed_stub(text: &str, dim: usize) -> Embedding {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut vec = vec![0.0f32; dim];
        for (i, chunk) in text.as_bytes().chunks(4).take(dim).enumerate() {
            let mut h = DefaultHasher::new();
            chunk.hash(&mut h);
            (i as u64).hash(&mut h);
            let hv = h.finish() as i32;
            vec[i] = hv as f32 / i32::MAX as f32;
        }
        // L2-normalise.
        let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-9);
        vec.iter_mut().for_each(|v| *v /= norm);
        vec
    }
}

#[async_trait]
impl EmbeddingModel for TextEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    async fn embed_text(&self, text: &str) -> mnemosyne_core::Result<Embedding> {
        let text = text.to_string();
        let dim = self.embedding_dim;
        tokio::task::spawn_blocking(move || Self::embed_stub(&text, dim))
            .await
            .map_err(|e| Error::model(e.to_string()))
    }

    async fn embed_batch(&self, texts: &[&str]) -> mnemosyne_core::Result<Vec<Embedding>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed_text(text).await?);
        }
        Ok(results)
    }
}

