use crate::{
    types::{Embedding, IndexedChunk, ParsedContent, SearchQuery, SearchResult},
    Result,
};
use async_trait::async_trait;
use std::path::Path;

/// Implemented by every file-type specific parser.
///
/// Parsers are registered in `ParserRegistry` (see `mnemosyne-parser`)
/// and dispatched based on file extension.
#[async_trait]
pub trait FileParser: Send + Sync {
    /// File extensions this parser handles (lowercase, without leading dot).
    fn supported_extensions(&self) -> &[&'static str];

    /// Parse `path` into one or more content chunks.
    ///
    /// Large files should be split into multiple chunks to fit model
    /// context windows and improve retrieval granularity.
    async fn parse(&self, path: &Path) -> Result<Vec<ParsedContent>>;
}

/// Implemented by every embedding model backend.
#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    /// Stable identifier such as `"sentence-transformers/all-MiniLM-L6-v2"`.
    fn model_id(&self) -> &str;

    /// Output dimension of the embedding vectors.
    fn embedding_dim(&self) -> usize;

    /// Embed a single text string.
    async fn embed_text(&self, text: &str) -> Result<Embedding>;

    /// Embed multiple texts in one batched forward pass.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>>;
}

/// Implemented by the search index backend.
#[async_trait]
pub trait SearchIndex: Send + Sync {
    /// Insert or update a chunk (and its embedding) in the index.
    async fn upsert(&self, chunk: &IndexedChunk) -> Result<()>;

    /// Remove all chunks belonging to a file.
    async fn remove_file(&self, file_id: &str) -> Result<()>;

    /// Pure cosine-similarity vector search.
    async fn vector_search(
        &self,
        embedding: &Embedding,
        limit: usize,
    ) -> Result<Vec<SearchResult>>;

    /// Pure BM25 / FTS5 keyword search.
    async fn keyword_search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>>;

    /// Hybrid search: fuses vector and keyword results with RRF.
    async fn hybrid_search(
        &self,
        query: &SearchQuery,
        query_embedding: &Embedding,
    ) -> Result<Vec<SearchResult>>;
}
