//! Text embedding with optional candle BERT backend.
//!
//! - **Default (stub)**: deterministic hash-based vectors — no native deps.
//! - **`candle-backend` feature**: real BERT inference via Candle + HuggingFace.

use async_trait::async_trait;
use mnemosyne_core::{traits::EmbeddingModel, types::Embedding, Error};
use std::sync::Arc;

#[cfg(feature = "candle-backend")]
use super::bert_impl::BertEmbedder;
#[cfg(not(feature = "candle-backend"))]
use tracing::warn;

/// Default model identifier.
pub const DEFAULT_TEXT_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";
#[cfg(not(feature = "candle-backend"))]
const STUB_DIM: usize = 384;

// ── Backend enum ─────────────────────────────────────────────────────────────

enum Backend {
    #[cfg(not(feature = "candle-backend"))]
    Stub { dim: usize },
    #[cfg(feature = "candle-backend")]
    Bert(Box<BertEmbedder>),
}

impl Backend {
    fn embed(&self, text: &str) -> Result<Vec<f32>, Error> {
        match self {
            #[cfg(not(feature = "candle-backend"))]
            Self::Stub { dim } => Ok(stub_embed(text, *dim)),
            #[cfg(feature = "candle-backend")]
            Self::Bert(bert) => bert.embed(text),
        }
    }

    #[cfg(not(feature = "candle-backend"))]
    fn dim(&self) -> usize {
        match self {
            Self::Stub { dim } => *dim,
        }
    }
}

// SAFETY: Backend::Stub is trivially safe.
// Backend::Bert is safe because BertEmbedder explicitly implements Send+Sync.
unsafe impl Send for Backend {}
unsafe impl Sync for Backend {}

// ── TextEmbedder ─────────────────────────────────────────────────────────────

pub struct TextEmbedder {
    model_id: String,
    embedding_dim: usize,
    backend: Arc<Backend>,
}

impl TextEmbedder {
    pub async fn load(model_id: &str) -> Result<Self, Error> {
        #[cfg(feature = "candle-backend")]
        {
            let bert = BertEmbedder::load(model_id).await?;
            let dim = bert.dim;
            Ok(Self {
                model_id: model_id.to_string(),
                embedding_dim: dim,
                backend: Arc::new(Backend::Bert(Box::new(bert))),
            })
        }

        #[cfg(not(feature = "candle-backend"))]
        {
            warn!(
                "Stub embedder active for '{}'. Enable `candle-backend` feature for real BERT.",
                model_id
            );
            Ok(Self {
                model_id: model_id.to_string(),
                embedding_dim: STUB_DIM,
                backend: Arc::new(Backend::Stub { dim: STUB_DIM }),
            })
        }
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
        let backend = Arc::clone(&self.backend);
        tokio::task::spawn_blocking(move || backend.embed(&text))
            .await
            .map_err(|e| Error::model(e.to_string()))?
    }

    async fn embed_batch(&self, texts: &[&str]) -> mnemosyne_core::Result<Vec<Embedding>> {
        let mut results = Vec::with_capacity(texts.len());
        for t in texts {
            results.push(self.embed_text(t).await?);
        }
        Ok(results)
    }
}

impl TextEmbedder {
    /// Returns true if this embedder supports sparse (lexical) encoding.
    /// Only true for BGE-M3 when `sparse_linear.pt` was successfully loaded.
    pub fn has_sparse(&self) -> bool {
        #[cfg(feature = "candle-backend")]
        {
            match &*self.backend {
                Backend::Bert(b) => b.has_sparse(),
            }
        }
        #[cfg(not(feature = "candle-backend"))]
        false
    }

    /// Compute BGE-M3 sparse (lexical) weights. Returns `{token_id: weight}`.
    /// Errors if no sparse_linear head is loaded.
    pub fn embed_sparse(
        &self,
        text: &str,
    ) -> mnemosyne_core::Result<std::collections::HashMap<u32, f32>> {
        #[cfg(feature = "candle-backend")]
        {
            let text = text.to_string();
            let backend = Arc::clone(&self.backend);
            match &*backend {
                Backend::Bert(b) => b.embed_sparse(&text),
            }
        }
        #[cfg(not(feature = "candle-backend"))]
        Err(Error::model(
            "sparse embedding requires candle-backend".to_string(),
        ))
    }

    /// Compute dense + sparse embeddings in a **single forward pass**.
    ///
    /// Significantly faster than calling `embed_text` + `embed_sparse` separately.
    pub fn embed_combined(
        &self,
        text: &str,
    ) -> mnemosyne_core::Result<(
        mnemosyne_core::types::Embedding,
        Option<std::collections::HashMap<u32, f32>>,
    )> {
        #[cfg(feature = "candle-backend")]
        {
            let text = text.to_string();
            let backend = Arc::clone(&self.backend);
            match &*backend {
                Backend::Bert(b) => b.embed_combined(&text),
            }
        }
        #[cfg(not(feature = "candle-backend"))]
        {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let dense = stub_embed(
                text,
                match &*self.backend {
                    Backend::Stub { dim } => *dim,
                },
            );
            Ok((dense, None))
        }
    }

    /// Compute dense + sparse embeddings for a **batch** of texts.
    ///
    /// Internally splits into mini-batches of `batch_size` and forwards each
    /// mini-batch in a single pass, then reassembles results in input order.
    /// Falls back to sequential `embed_combined` when `candle-backend` is off.
    pub fn embed_batch_combined(
        &self,
        texts: &[String],
        batch_size: usize,
    ) -> mnemosyne_core::Result<
        Vec<(
            mnemosyne_core::types::Embedding,
            Option<std::collections::HashMap<u32, f32>>,
        )>,
    > {
        #[cfg(feature = "candle-backend")]
        {
            let backend = Arc::clone(&self.backend);
            match &*backend {
                Backend::Bert(b) => b.embed_batch_combined(texts, batch_size),
            }
        }
        #[cfg(not(feature = "candle-backend"))]
        {
            // Stub: delegate to sequential embed_combined
            texts.iter().map(|t| self.embed_combined(t)).collect()
        }
    }
}

// ── Stub embedding ────────────────────────────────────────────────────────────
#[cfg(not(feature = "candle-backend"))]
fn stub_embed(text: &str, dim: usize) -> Vec<f32> {
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
    let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-9);
    vec.iter_mut().for_each(|v| *v /= norm);
    vec
}
