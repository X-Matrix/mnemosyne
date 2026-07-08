//! Hybrid (vector + FTS5) search index for Mnemosyne.
//!
//! [`HybridIndex`] implements [`mnemosyne_core::traits::SearchIndex`] and
//! fuses vector-similarity results with FTS5 BM25 results using
//! **Reciprocal Rank Fusion** (RRF).

pub mod cosine;
pub mod hybrid;
pub mod rrf;

pub use hybrid::HybridIndex;

pub type Result<T> = std::result::Result<T, mnemosyne_core::Error>;
