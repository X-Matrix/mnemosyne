//! Approximate Nearest Neighbour (ANN) index using HNSW.
//!
//! Built on top of [`instant_distance`] (pure-Rust HNSW).
//! Used by [`super::hybrid::HybridIndex`] for fast vector search when the
//! number of embeddings exceeds `ANN_THRESHOLD`.
//!
//! Falls back to brute-force cosine for small indexes.

use instant_distance::{Builder, HnswMap, Search};
use mnemosyne_core::Error;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// Switch to HNSW once we have more than this many embeddings.
const ANN_THRESHOLD: usize = 2_000;

// ── Newtype so we can implement `instant_distance::Point` ────────────────────

#[derive(Clone)]
struct EmbVec(Vec<f32>);

impl instant_distance::Point for EmbVec {
    fn distance(&self, other: &Self) -> f32 {
        // Cosine distance = 1 – cosine similarity.
        // Assumes L2-normalised embeddings ⟹ dot product == cosine similarity.
        let dot: f32 = self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum();
        1.0 - dot.clamp(-1.0, 1.0)
    }
}

// ── AnnIndex ─────────────────────────────────────────────────────────────────

/// Immutable HNSW snapshot.  Rebuild when stale.
pub struct AnnIndex {
    hnsw: HnswMap<EmbVec, String>,
}

impl AnnIndex {
    /// Build an HNSW index from `(chunk_id, embedding)` pairs.
    pub fn build(pairs: &[(String, Vec<f32>)]) -> Self {
        info!("Building ANN index from {} embeddings", pairs.len());
        let points: Vec<EmbVec> = pairs.iter().map(|(_, e)| EmbVec(e.clone())).collect();
        let values: Vec<String> = pairs.iter().map(|(id, _)| id.clone()).collect();
        let hnsw = Builder::default().build(points, values);
        Self { hnsw }
    }

    /// Return the top-`k` nearest chunk IDs with their distances.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        let qv = EmbVec(query.to_vec());
        let mut s = Search::default();
        self.hnsw
            .search(&qv, &mut s)
            .take(k)
            .map(|item| (item.value.clone(), item.distance))
            .collect()
    }
}

// ── AnnCache — lazy-rebuilt, shared across async tasks ───────────────────────

/// Cache entry: embedding dimension + index.
type CacheEntry = Option<(usize, Arc<AnnIndex>)>;

/// Thread-safe, lazily-built ANN index.
///
/// The cached index is keyed by embedding dimension so that BERT (384-d)
/// and CLIP (512-d) searches never share the same index.
#[derive(Clone)]
pub struct AnnCache {
    inner: Arc<RwLock<CacheEntry>>,
}

impl AnnCache {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(None)),
        }
    }

    /// Discard the cached index (triggers a rebuild on next search).
    pub async fn invalidate(&self) {
        *self.inner.write().await = None;
    }

    /// Return the cached index for `dim`-dimensional embeddings, rebuilding
    /// if necessary.  Caches are dimension-specific: a CLIP-512 query never
    /// reuses a BERT-384 index.
    pub async fn get_or_build(
        &self,
        dim: usize,
        load_fn: impl std::future::Future<Output = Result<Vec<(String, Vec<f32>)>, Error>>,
    ) -> Result<Option<Arc<AnnIndex>>, Error> {
        // Fast-path: already built for this dimension.
        if let Some((cached_dim, ref idx)) = *self.inner.read().await {
            if cached_dim == dim {
                return Ok(Some(Arc::clone(idx)));
            }
            // Wrong dimension → fall through and rebuild.
        }

        // Slow-path: (re)build.
        let pairs = load_fn.await?;
        if pairs.len() < ANN_THRESHOLD {
            return Ok(None); // Too small — use brute force.
        }

        let idx = Arc::new(AnnIndex::build(&pairs));
        *self.inner.write().await = Some((dim, Arc::clone(&idx)));
        info!("ANN index ready ({} embeddings, dim={})", pairs.len(), dim);
        Ok(Some(idx))
    }
}

impl Default for AnnCache {
    fn default() -> Self {
        Self::new()
    }
}
