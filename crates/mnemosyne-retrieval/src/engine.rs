use chrono::{TimeZone, Utc};
use mnemosyne_core::{
    traits::{EmbeddingModel, SearchIndex},
    types::{
        FileRecord, FileType, IndexStats, IndexedChunk, ParsedContent, SearchQuery, SearchResult,
    },
    Error,
};
use mnemosyne_index::HybridIndex;
use mnemosyne_model::ModelRegistry;
use mnemosyne_parser::ParserRegistry;
use mnemosyne_storage::{Database, FileRepo};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info, warn};
use uuid::Uuid;
use walkdir::WalkDir;

pub struct SearchEngine {
    pub(crate) db: Database,
    pub(crate) index: Arc<HybridIndex>,
    pub(crate) parsers: Arc<ParserRegistry>,
    pub(crate) models: Arc<ModelRegistry>,
    pub(crate) text_model_id: String,
}

impl SearchEngine {
    pub fn new(
        db: Database,
        index: Arc<HybridIndex>,
        parsers: Arc<ParserRegistry>,
        models: Arc<ModelRegistry>,
        text_model_id: String,
    ) -> Self {
        Self { db, index, parsers, models, text_model_id }
    }

    pub fn builder() -> crate::builder::SearchEngineBuilder {
        crate::builder::SearchEngineBuilder::new()
    }

    // ── Indexing ─────────────────────────────────────────────────────────────

    /// Recursively index all supported files under `dir`.
    pub async fn index_directory(&self, dir: impl AsRef<Path>) -> Result<IndexStats, Error> {
        let dir = dir.as_ref().to_path_buf();
        info!("Indexing directory: {}", dir.display());

        let mut stats = IndexStats::default();

        let entries: Vec<_> = WalkDir::new(&dir)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .filter(|e| self.parsers.is_supported(e.path()))
            .collect();

        info!("Found {} supported files", entries.len());

        for entry in entries {
            let path = entry.path().to_path_buf();
            match self.index_file(&path).await {
                Ok(true) => {
                    let ext = path
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("?")
                        .to_string();
                    let ft = format!("{:?}", FileType::from_extension(&ext));
                    *stats.files_by_type.entry(ft).or_default() += 1;
                    stats.total_files += 1;
                }
                Ok(false) => debug!("Skipped (unchanged): {}", path.display()),
                Err(e) => warn!("Failed to index {}: {}", path.display(), e),
            }
        }

        // Count chunks.
        let db = self.db.clone();
        stats.total_chunks = tokio::task::spawn_blocking(move || {
            mnemosyne_storage::ChunkRepo::new(&db).count()
        })
        .await
        .map_err(|e| Error::storage(e.to_string()))??;

        info!(
            "Indexing complete: {} files, {} chunks",
            stats.total_files, stats.total_chunks
        );
        Ok(stats)
    }

    /// Index a single file. Returns `Ok(true)` if indexed, `Ok(false)` if skipped.
    pub async fn index_file(&self, path: &Path) -> Result<bool, Error> {
        // Compute content hash for change detection.
        let raw = tokio::fs::read(path).await.map_err(Error::Io)?;
        let hash = hex::encode(Sha256::digest(&raw));

        let meta = tokio::fs::metadata(path).await.map_err(Error::Io)?;
        let size = meta.len();
        let modified_at = meta
            .modified()
            .ok()
            .and_then(|t| {
                t.duration_since(std::time::UNIX_EPOCH)
                    .ok()
                    .map(|d| chrono::Utc.timestamp_opt(d.as_secs() as i64, 0).single())
            })
            .flatten();

        // Skip if hash unchanged.
        let path_str = path.to_string_lossy().to_string();
        let db_clone = self.db.clone();
        let hash_clone = hash.clone();
        let path_str_clone = path_str.clone();

        let existing = tokio::task::spawn_blocking(move || {
            FileRepo::new(&db_clone).find_by_path(&path_str_clone)
        })
        .await
        .map_err(|e| Error::storage(e.to_string()))??;

        if let Some(ref rec) = existing {
            if rec.content_hash.as_deref() == Some(&hash_clone) {
                return Ok(false);
            }
        }

        let file_id = existing
            .as_ref()
            .map(|r| r.id.clone())
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_string();
        let file_type = FileType::from_extension(&ext);

        let file_record = FileRecord {
            id: file_id.clone(),
            path: path.to_path_buf(),
            file_type,
            size,
            modified_at,
            indexed_at: Some(chrono::Utc::now()),
            content_hash: Some(hash),
        };

        // Parse content.
        let chunks_content = self.parsers.parse(path).await?;
        if chunks_content.is_empty() {
            return Ok(false);
        }

        // Generate embeddings.
        let model = self.models.get_text_embedder(&self.text_model_id).await?;

        let texts: Vec<&str> = chunks_content.iter().map(|c| c.as_text()).collect();
        let embeddings = model.embed_batch(&texts).await?;

        // Persist file record.
        {
            let db = self.db.clone();
            let fr = file_record.clone();
            tokio::task::spawn_blocking(move || FileRepo::new(&db).upsert(&fr))
                .await
                .map_err(|e| Error::storage(e.to_string()))??;
        }

        // Persist chunks + embeddings.
        for (i, (content, embedding)) in
            chunks_content.into_iter().zip(embeddings.into_iter()).enumerate()
        {
            let chunk_id = format!("{file_id}:{i}");
            let chunk = IndexedChunk {
                chunk_id,
                file_id: file_id.clone(),
                chunk_index: i,
                content,
                embedding: Some(embedding),
            };
            self.index.upsert(&chunk).await?;
        }

        Ok(true)
    }

    // ── Search ────────────────────────────────────────────────────────────────

    /// Execute a search query and return ranked results.
    pub async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>, Error> {
        if query.text.trim().is_empty() {
            return Ok(vec![]);
        }

        let model = self.models.get_text_embedder(&self.text_model_id).await?;
        let query_embedding = model.embed_text(&query.text).await?;

        use mnemosyne_core::types::SearchMode;
        let results = match &query.mode {
            SearchMode::Vector => {
                self.index
                    .vector_search(&query_embedding, query.limit)
                    .await?
            }
            SearchMode::Keyword => {
                self.index
                    .keyword_search(&query.text, query.limit)
                    .await?
            }
            SearchMode::Hybrid => {
                self.index
                    .hybrid_search(&query, &query_embedding)
                    .await?
            }
        };

        Ok(results)
    }

    // ── Utilities ─────────────────────────────────────────────────────────────

    pub async fn get_stats(&self) -> Result<IndexStats, Error> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let file_repo = FileRepo::new(&db);
            let chunk_repo = mnemosyne_storage::ChunkRepo::new(&db);
            Ok(IndexStats {
                total_files: file_repo.count()?,
                total_chunks: chunk_repo.count()?,
                files_by_type: Default::default(),
                index_size_bytes: 0,
            })
        })
        .await
        .map_err(|e| Error::storage(e.to_string()))?
    }

    pub async fn list_files(
        &self,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<FileRecord>, Error> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || FileRepo::new(&db).list(limit, offset))
            .await
            .map_err(|e| Error::storage(e.to_string()))?
    }

    pub async fn remove_file(&self, file_id: &str) -> Result<(), Error> {
        self.index.remove_file(file_id).await
    }
}


