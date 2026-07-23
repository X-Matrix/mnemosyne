use crate::ignore::IgnoreConfig;
use chrono::TimeZone;
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

/// Default CLIP model used when `clip-backend` feature is enabled.
pub const DEFAULT_CLIP_MODEL: &str = "openai/clip-vit-base-patch32";
/// Default Whisper model used when `whisper-backend` feature is enabled.
pub const DEFAULT_WHISPER_MODEL: &str = "openai/whisper-tiny";

pub struct SearchEngine {
    pub(crate) db: Database,
    pub(crate) index: Arc<HybridIndex>,
    pub(crate) parsers: Arc<ParserRegistry>,
    pub(crate) models: Arc<ModelRegistry>,
    pub(crate) text_model_id: String,
    pub(crate) vision_model_id: String,
    pub(crate) audio_model_id: String,
    pub(crate) ignore_config: Arc<IgnoreConfig>,
}

impl SearchEngine {
    pub fn new(
        db: Database,
        index: Arc<HybridIndex>,
        parsers: Arc<ParserRegistry>,
        models: Arc<ModelRegistry>,
        text_model_id: String,
        vision_model_id: String,
        audio_model_id: String,
        ignore_config: Arc<IgnoreConfig>,
    ) -> Self {
        Self {
            db,
            index,
            parsers,
            models,
            text_model_id,
            vision_model_id,
            audio_model_id,
            ignore_config,
        }
    }

    /// Expose the current ignore configuration.
    pub fn ignore_config(&self) -> &IgnoreConfig {
        &self.ignore_config
    }

    pub fn builder() -> crate::builder::SearchEngineBuilder {
        crate::builder::SearchEngineBuilder::new()
    }

    /// Expose the underlying database (e.g. for test assertions).
    pub fn db(&self) -> &mnemosyne_storage::Database {
        &self.db
    }

    /// Return the currently active text-embedding model ID.
    pub fn get_text_model(&self) -> &str {
        &self.text_model_id
    }
    /// Return the currently active vision-embedding model ID.
    pub fn get_vision_model(&self) -> &str {
        &self.vision_model_id
    }
    /// Return the currently active audio-transcription model ID.
    pub fn get_audio_model(&self) -> &str {
        &self.audio_model_id
    }

    /// Switch the active text-embedding model at runtime.
    /// **Existing embeddings are incompatible — re-index required.**
    pub fn set_text_model(&mut self, id: impl Into<String>) {
        let old = std::mem::replace(&mut self.text_model_id, id.into());
        if old != self.text_model_id {
            tracing::info!("Text model switched: {} → {}", old, self.text_model_id);
        }
    }
    /// Switch the active vision-embedding (CLIP) model at runtime.
    pub fn set_vision_model(&mut self, id: impl Into<String>) {
        let old = std::mem::replace(&mut self.vision_model_id, id.into());
        if old != self.vision_model_id {
            tracing::info!("Vision model switched: {} → {}", old, self.vision_model_id);
        }
    }
    /// Switch the active audio-transcription (Whisper) model at runtime.
    pub fn set_audio_model(&mut self, id: impl Into<String>) {
        let old = std::mem::replace(&mut self.audio_model_id, id.into());
        if old != self.audio_model_id {
            tracing::info!("Audio model switched: {} → {}", old, self.audio_model_id);
        }
    }

    /// Enable or disable force-HNSW mode (bypasses the 2000-embedding threshold).
    pub async fn set_force_hnsw(&self, force: bool) {
        self.index.set_force_hnsw(force).await;
    }

    /// Return the current force-HNSW setting.
    pub fn get_force_hnsw(&self) -> bool {
        self.index.get_force_hnsw()
    }

    // ── Indexing ─────────────────────────────────────────────────────────────

    /// Recursively index all supported files under `dir`.
    pub async fn index_directory(&self, dir: impl AsRef<Path>) -> Result<IndexStats, Error> {
        let dir = dir.as_ref().to_path_buf();
        info!("Indexing directory: {}", dir.display());

        let mut stats = IndexStats::default();

        // Walk directory, logging every error and unsupported file so we can diagnose issues
        let mut total_walked = 0usize;
        let mut walkdir_errors = 0usize;
        let mut unsupported_count = 0usize;

        let ignore = Arc::clone(&self.ignore_config);
        let root = dir.clone();
        let entries: Vec<_> = WalkDir::new(&dir)
            .follow_links(true)
            .into_iter()
            .filter_entry(move |e| {
                // Prune ignored directories early so we never descend into them.
                if e.file_type().is_dir() {
                    let is_root = e.path() == root;
                    !ignore.should_skip_dir(e.path(), is_root)
                } else {
                    true
                }
            })
            .filter_map(|e| match e {
                Ok(entry) => {
                    total_walked += 1;
                    Some(entry)
                }
                Err(err) => {
                    walkdir_errors += 1;
                    warn!("[index_directory] WalkDir error: {}", err);
                    None
                }
            })
            .filter(|e| e.file_type().is_file())
            .filter(|e| {
                let supported = self.parsers.is_supported(e.path());
                if !supported {
                    let ext = e.path().extension().and_then(|x| x.to_str()).unwrap_or("");
                    if !ext.is_empty() {
                        debug!(
                            "[index_directory] unsupported extension '{}': {}",
                            ext,
                            e.path().display()
                        );
                        unsupported_count += 1;
                    }
                }
                supported
            })
            .collect();

        info!(
            "[index_directory] walk complete: walked={} walkdir_errors={} unsupported={} supported={}",
            total_walked, walkdir_errors, unsupported_count, entries.len()
        );

        if entries.is_empty() {
            // Check if we can read the directory at all — helps surface TCC denials.
            match std::fs::read_dir(&dir) {
                Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                    return Err(Error::parse(format!(
                        "Permission denied reading '{}'. \
                         Please grant access in: System Settings → Privacy & Security → Files and Folders",
                        dir.display()
                    )));
                }
                Err(e) => {
                    warn!("[index_directory] read_dir failed: {}", e);
                }
                Ok(_) => {
                    info!("[index_directory] read_dir OK but no supported files found — check extensions");
                }
            }
        }

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
        stats.total_chunks =
            tokio::task::spawn_blocking(move || mnemosyne_storage::ChunkRepo::new(&db).count())
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

        #[cfg(feature = "whisper-backend")]
        let is_audio = matches!(file_type, FileType::Audio);

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
        #[allow(unused_mut)]
        let mut chunks_content = self.parsers.parse(path).await?;
        if chunks_content.is_empty() {
            return Ok(false);
        }

        // For audio files, replace the parser stub with a real Whisper transcript
        // when the whisper-backend feature is compiled in.
        #[cfg(feature = "whisper-backend")]
        {
            if is_audio {
                match self.transcribe_audio(path).await {
                    Ok(transcript) if !transcript.trim().is_empty() => {
                        chunks_content =
                            vec![mnemosyne_core::types::ParsedContent::AudioTranscript {
                                transcript,
                                language: None,
                            }];
                    }
                    Ok(_) => {}
                    Err(e) => tracing::warn!("Whisper failed, keeping stub: {e}"),
                }
            }
        }

        // Generate embeddings — route to CLIP for images, Whisper transcript text for audio,
        // text embedder for everything else.
        //
        // For BGE-M3 (has sparse_linear), use embed_combined() to compute dense +
        // sparse in a SINGLE forward pass per text chunk instead of two passes.
        // This halves computation time compared to embed_chunks + embed_sparse.
        #[cfg(feature = "candle-backend")]
        let (embeddings, sparse_map) = {
            let text_model = self.models.get_text_embedder(&self.text_model_id).await?;
            if text_model.has_sparse() {
                // Combined path: one forward pass → (dense, Option<sparse>)
                let mut dense_vecs: Vec<mnemosyne_core::types::Embedding> =
                    Vec::with_capacity(chunks_content.len());
                let mut sparse_vecs: Vec<Option<std::collections::HashMap<u32, f32>>> =
                    Vec::with_capacity(chunks_content.len());

                for chunk in &chunks_content {
                    let text_opt = match chunk {
                        mnemosyne_core::types::ParsedContent::Text { text } => Some(text.clone()),
                        mnemosyne_core::types::ParsedContent::AudioTranscript {
                            transcript,
                            ..
                        } => Some(transcript.clone()),
                        _ => None,
                    };

                    if let Some(text) = text_opt {
                        let m = text_model.clone();
                        let (dense, sparse) =
                            tokio::task::spawn_blocking(move || m.embed_combined(&text))
                                .await
                                .map_err(|e| Error::model(e.to_string()))??;
                        dense_vecs.push(dense);
                        sparse_vecs.push(sparse);
                    } else {
                        // Non-text chunk (image, video): dense via normal path, no sparse.
                        let emb = self.embed_single_chunk(path, chunk).await?;
                        dense_vecs.push(emb);
                        sparse_vecs.push(None);
                    }
                }
                (dense_vecs, Some(sparse_vecs))
            } else {
                // Standard path: dense only.
                let embs = self.embed_chunks(path, &chunks_content).await?;
                (
                    embs,
                    None::<Vec<Option<std::collections::HashMap<u32, f32>>>>,
                )
            }
        };

        #[cfg(not(feature = "candle-backend"))]
        let (embeddings, sparse_map) = {
            let embs = self.embed_chunks(path, &chunks_content).await?;
            (
                embs,
                None::<Vec<Option<std::collections::HashMap<u32, f32>>>>,
            )
        };

        // Persist file record.
        {
            let db = self.db.clone();
            let fr = file_record.clone();
            tokio::task::spawn_blocking(move || FileRepo::new(&db).upsert(&fr))
                .await
                .map_err(|e| Error::storage(e.to_string()))??;
        }

        // Persist chunks + embeddings + optional sparse embeddings.
        for (i, (content, embedding)) in chunks_content.into_iter().zip(embeddings).enumerate() {
            let chunk_id = format!("{file_id}:{i}");

            // ── 1. Upsert the chunk first (document_chunks row must exist before
            //       sparse_embeddings foreign key can reference it).
            let chunk = IndexedChunk {
                chunk_id: chunk_id.clone(),
                file_id: file_id.clone(),
                chunk_index: i,
                content,
                embedding: Some(embedding),
            };
            self.index.upsert(&chunk).await?;

            // ── 2. Now store sparse embedding (foreign key satisfied).
            if let Some(ref svecs) = sparse_map {
                if let Some(Some(sparse)) = svecs.get(i) {
                    let db = self.db.clone();
                    let cid = chunk_id.clone();
                    let mid = self.text_model_id.clone();
                    let sv = sparse.clone();
                    tokio::task::spawn_blocking(move || {
                        mnemosyne_storage::SparseEmbeddingRepo::new(&db).upsert(&cid, &mid, &sv)
                    })
                    .await
                    .map_err(|e| Error::storage(e.to_string()))??;
                }
            }
        }

        Ok(true)
    }

    // ── Search ────────────────────────────────────────────────────────────────

    /// Execute a search query and return ranked results.
    ///
    /// When `clip-backend` is compiled in, the query is also embedded with the
    /// CLIP text encoder so that image chunks (stored as 512-d CLIP vectors) are
    /// included in Vector and Hybrid searches alongside text results.
    /// Keyword mode is excluded — FTS5 on image captions has no semantic value.
    pub async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>, Error> {
        if query.text.trim().is_empty() {
            return Ok(vec![]);
        }

        // ── Text (BERT) embedding for text / audio / PDF chunks ───────────────
        let bert_model = self.models.get_text_embedder(&self.text_model_id).await?;
        let bert_embedding = bert_model.embed_text(&query.text).await?;

        // ── Sparse (lexical) embedding for BGE-M3 keyword path ────────────────
        // When sparse_linear is loaded, use lexical dot-product instead of FTS5.
        let query_sparse: Option<std::collections::HashMap<u32, f32>> = if bert_model.has_sparse() {
            bert_model.embed_sparse(&query.text).ok()
        } else {
            None
        };

        use mnemosyne_core::types::SearchMode;

        // Minimum cosine similarity for pure-vector results.
        // Below this the match is considered noise (< 45 % semantic overlap).
        // If the entire result set is below the floor (corpus mismatch), we
        // still return the top FLOOR_FALLBACK items so the UI isn't empty.
        const MIN_VECTOR_SCORE: f32 = 0.45;
        const FLOOR_FALLBACK: usize = 5;

        let mut results = match &query.mode {
            SearchMode::Vector => {
                // Fetch extra candidates so we have room to filter.
                let inner = (query.limit * 3).max(30);
                let raw = self.index.vector_search(&bert_embedding, inner).await?;
                let above: Vec<SearchResult> = raw
                    .iter()
                    .filter(|r| r.score >= MIN_VECTOR_SCORE)
                    .take(query.limit)
                    .cloned()
                    .collect();
                if above.is_empty() {
                    // Nothing confident — fall back to top-N so UI stays useful.
                    raw.into_iter().take(FLOOR_FALLBACK).collect()
                } else {
                    above
                }
            }
            SearchMode::Keyword => {
                if let Some(ref sparse) = query_sparse {
                    self.index
                        .sparse_keyword_search(sparse, query.limit)
                        .await?
                } else {
                    self.index.keyword_search(&query.text, query.limit).await?
                }
            }
            SearchMode::Hybrid => {
                if let Some(ref sparse) = query_sparse {
                    // Sparse-hybrid: replace the FTS5 arm with sparse dot-product.
                    let inner = query.limit * 3;
                    let (vec_res, kw_res) = tokio::join!(
                        self.index.vector_search(&bert_embedding, inner),
                        self.index.sparse_keyword_search(sparse, inner),
                    );
                    let vec_results = vec_res?;
                    let kw_results = kw_res?;
                    // RRF fusion (mirrors HybridIndex::hybrid_search).
                    let vec_ranked: Vec<(String, f32)> = vec_results
                        .iter()
                        .map(|r| (format!("{}:{}", r.file_record.id, r.chunk_index), r.score))
                        .collect();
                    let kw_ranked: Vec<(String, f32)> = kw_results
                        .iter()
                        .map(|r| (format!("{}:{}", r.file_record.id, r.chunk_index), r.score))
                        .collect();
                    let fused = mnemosyne_index::rrf::fuse(
                        &vec_ranked,
                        &kw_ranked,
                        query.limit,
                        query.vector_weight,
                        query.keyword_weight,
                    );
                    let mut result_map: std::collections::HashMap<String, SearchResult> =
                        std::collections::HashMap::new();
                    for r in vec_results.into_iter().chain(kw_results) {
                        let key = format!("{}:{}", r.file_record.id, r.chunk_index);
                        result_map.entry(key).or_insert(r);
                    }
                    fused
                        .into_iter()
                        .filter_map(|(key, score)| {
                            let mut r = result_map.remove(&key)?;
                            r.score = score;
                            r.match_type = mnemosyne_core::types::MatchType::Hybrid;
                            Some(r)
                        })
                        .collect()
                } else {
                    self.index.hybrid_search(&query, &bert_embedding).await?
                }
            }
        };

        // ── Apply file-type filter ─────────────────────────────────────────────
        // Retain only results whose file type is in query.file_types.
        if let Some(ref types) = query.file_types {
            if !types.is_empty() {
                results.retain(|r| types.contains(&r.file_record.file_type));
            }
        }

        // ── CLIP text embedding for image chunks (clip-backend only) ──────────
        // Images are stored with 512-dim CLIP vectors, which live in a completely
        // separate embedding space from the 384-dim BERT space used for text /
        // audio / PDF chunks.  The BERT-based hybrid_search therefore never sees
        // images regardless of the query.  We fix this by running an additional
        // CLIP text→image search and merging its results.
        //
        // Modes:
        //   Vector  — included; merge & re-sort; threshold 0.63 (cosine ≥ 0.26).
        //   Hybrid  — included but carefully controlled:
        //     • Strict threshold 0.75 (cosine ≥ 0.50) — random screenshots score
        //       ~0.65 for most queries and must be excluded.
        //     • Hard cap at 5 images — prevents images flooding text results.
        //     • No re-sort — RRF text scores (~0.016) and CLIP cosine scores
        //       (~0.65+) are on incompatible scales; re-sorting always puts images
        //       first regardless of relevance.  Images are appended after text.
        //   Keyword — excluded; FTS5 on captions has no semantic value.
        #[cfg(feature = "clip-backend")]
        if matches!(&query.mode, SearchMode::Vector | SearchMode::Hybrid) {
            if let Ok(clip_text_emb) = self.embed_text_with_clip(&query.text).await {
                let is_hybrid = matches!(&query.mode, SearchMode::Hybrid);

                // Typical CLIP text-image cosine similarity:
                //   unrelated pairs  → 0.05-0.15
                //   somewhat related → 0.20-0.40
                //   strong match     → 0.40-0.70
                // Scores here are cosine similarities (after L2→cosine conversion
                // in EmbeddingRepo::vector_knn).
                const CLIP_MIN_VECTOR: f32 = 0.15; // permissive — user chose vector
                const CLIP_MIN_HYBRID: f32 = 0.25; // exclude clearly unrelated images
                const CLIP_MAX_HYBRID: usize = 8; // cap images appended in hybrid

                let clip_min_score = if is_hybrid {
                    CLIP_MIN_HYBRID
                } else {
                    CLIP_MIN_VECTOR
                };
                let clip_fetch = if is_hybrid {
                    CLIP_MAX_HYBRID * 4
                } else {
                    query.limit * 2
                };

                let mut clip_results = self
                    .index
                    .vector_search(&clip_text_emb, clip_fetch)
                    .await
                    .unwrap_or_default();

                debug!(
                    "CLIP raw results: {} before threshold {:.2}",
                    clip_results.len(),
                    clip_min_score
                );
                if let Some(top) = clip_results.first() {
                    debug!(
                        "CLIP top score: {:.4} ({})",
                        top.score,
                        top.file_record
                            .path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("?")
                    );
                }

                clip_results.retain(|r| r.score >= clip_min_score);

                // Apply the same file-type filter to CLIP results.
                if let Some(ref types) = query.file_types {
                    if !types.is_empty() {
                        clip_results.retain(|r| types.contains(&r.file_record.file_type));
                    }
                }

                // Deduplicate against existing results.
                let seen: std::collections::HashSet<String> = results
                    .iter()
                    .map(|r| format!("{}:{}", r.file_record.id, r.chunk_index))
                    .collect();
                clip_results
                    .retain(|r| !seen.contains(&format!("{}:{}", r.file_record.id, r.chunk_index)));

                if is_hybrid {
                    // Merge CLIP image results with text/PDF results and sort by
                    // score globally.  Both RRF hybrid scores and CLIP cosine
                    // similarities are normalised to [0, 1], so a single sort gives
                    // correct relevance ordering (e.g. a 69% image beats a 1% text).
                    clip_results.truncate(CLIP_MAX_HYBRID);
                    results.extend(clip_results);
                    results.sort_by(|a, b| {
                        b.score
                            .partial_cmp(&a.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    results.truncate(query.limit);
                } else {
                    // Vector mode: merge and sort by cosine score (same scale).
                    results.extend(clip_results);
                    results.sort_by(|a, b| {
                        b.score
                            .partial_cmp(&a.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    results.truncate(query.limit);
                }
            }
        }

        Ok(results)
    }

    // ── Utilities ─────────────────────────────────────────────────────────────

    pub async fn get_stats(&self) -> Result<IndexStats, Error> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let file_repo = FileRepo::new(&db);
            let chunk_repo = mnemosyne_storage::ChunkRepo::new(&db);

            // Count files per FileType from the database.
            let mut files_by_type: std::collections::HashMap<String, u64> =
                std::collections::HashMap::new();
            {
                let conn = db.conn.lock().unwrap();
                let mut stmt = conn
                    .prepare("SELECT file_type, COUNT(*) FROM files GROUP BY file_type")
                    .map_err(|e| Error::storage(e.to_string()))?;
                let rows = stmt
                    .query_map([], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
                    })
                    .map_err(|e| Error::storage(e.to_string()))?;
                for row in rows.flatten() {
                    let (ft_json, count) = row;
                    // file_type is stored as serde_json string e.g. "\"text\""
                    let ft: FileType = serde_json::from_str(&ft_json).unwrap_or(FileType::Unknown);
                    // Use Debug format ("Text", "Image", …) — matches frontend keys
                    *files_by_type.entry(format!("{:?}", ft)).or_default() += count as u64;
                }
            }
            // Note: PDF files now have FileType::Pdf (not Text), so they appear
            // in their own bucket automatically from the GROUP BY query above.

            Ok(IndexStats {
                total_files: file_repo.count()?,
                total_chunks: chunk_repo.count()?,
                files_by_type,
                index_size_bytes: 0,
            })
        })
        .await
        .map_err(|e| Error::storage(e.to_string()))?
    }

    pub async fn list_files(&self, limit: usize, offset: usize) -> Result<Vec<FileRecord>, Error> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || FileRepo::new(&db).list(limit, offset))
            .await
            .map_err(|e| Error::storage(e.to_string()))?
    }

    /// Count files whose path starts with `dir_path` (direct children + subdirs).
    pub async fn count_files_in_dir(&self, dir_path: &str) -> Result<u64, Error> {
        let db = self.db.clone();
        let prefix = format!("{}/", dir_path.trim_end_matches('/'));
        tokio::task::spawn_blocking(move || FileRepo::new(&db).count_by_prefix(&prefix))
            .await
            .map_err(|e| Error::storage(e.to_string()))?
    }

    pub async fn remove_file(&self, file_id: &str) -> Result<(), Error> {
        self.index.remove_file(file_id).await
    }

    /// Remove a file from the index by its filesystem path.
    ///
    /// Returns `Ok(true)` if the file was found and removed, `Ok(false)` if it
    /// was not in the index (already clean).
    pub async fn remove_file_by_path(&self, path: &Path) -> Result<bool, Error> {
        let db = self.db.clone();
        let path_str = path.to_string_lossy().to_string();

        let record =
            tokio::task::spawn_blocking(move || FileRepo::new(&db).find_by_path(&path_str))
                .await
                .map_err(|e| Error::storage(e.to_string()))??;

        match record {
            Some(rec) => {
                self.index.remove_file(&rec.id).await?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    // ── Embedding routing ─────────────────────────────────────────────────────

    /// Generate embeddings for a batch of parsed content chunks.
    ///
    /// Routing logic:
    /// - `Image` chunks  → CLIP vision encoder (when `clip-backend` feature is on)
    /// - `AudioTranscript` → text embedder on the transcript text
    /// - Everything else → text embedder
    async fn embed_chunks(
        &self,
        file_path: &Path,
        chunks: &[ParsedContent],
    ) -> Result<Vec<mnemosyne_core::types::Embedding>, Error> {
        #[cfg(not(feature = "clip-backend"))]
        let _ = file_path; // only used when clip-backend is compiled in
        let mut embeddings = Vec::with_capacity(chunks.len());

        for chunk in chunks {
            let emb = match chunk {
                // ── Image: CLIP vision embedding (clip-backend) or caption text ──
                ParsedContent::Image { caption, .. } => {
                    // `caption` is used only when clip-backend is NOT active;
                    // suppress the warning for the clip-backend build.
                    #[cfg(feature = "clip-backend")]
                    let _ = &caption;
                    #[cfg(feature = "clip-backend")]
                    {
                        let clip = self.models.get_clip_embedder(&self.vision_model_id).await?;
                        let fp = file_path.to_path_buf();
                        tokio::task::spawn_blocking(move || clip.embed_image(&fp))
                            .await
                            .map_err(|e| Error::model(e.to_string()))?
                            .map_err(|e| {
                                tracing::warn!("CLIP failed: {e}");
                                e
                            })?
                    }
                    #[cfg(not(feature = "clip-backend"))]
                    {
                        let model = self.models.get_text_embedder(&self.text_model_id).await?;
                        model.embed_text(caption).await?
                    }
                }

                // ── Audio: embed transcript text ──────────────────────────────
                ParsedContent::AudioTranscript { transcript, .. } => {
                    let model = self.models.get_text_embedder(&self.text_model_id).await?;
                    model.embed_text(transcript).await?
                }

                // ── Video keyframe description ────────────────────────────────
                ParsedContent::VideoKeyframe { description, .. } => {
                    let model = self.models.get_text_embedder(&self.text_model_id).await?;
                    model.embed_text(description).await?
                }

                // ── Text: standard path ───────────────────────────────────────
                ParsedContent::Text { text } => {
                    let model = self.models.get_text_embedder(&self.text_model_id).await?;
                    model.embed_text(text).await?
                }
            };
            embeddings.push(emb);
        }
        Ok(embeddings)
    }

    // ── CLIP image embedding (requires clip-backend feature) ──────────────────

    /// Embed a single chunk (image / video / audio-stub) using the normal routing.
    /// Used by the combined-path when a non-text chunk is encountered.
    async fn embed_single_chunk(
        &self,
        file_path: &Path,
        chunk: &mnemosyne_core::types::ParsedContent,
    ) -> Result<mnemosyne_core::types::Embedding, Error> {
        let chunks = std::slice::from_ref(chunk);
        let mut embs = self.embed_chunks(file_path, chunks).await?;
        embs.pop()
            .ok_or_else(|| Error::model("no embedding produced".to_string()))
    }

    /// Embed an image file using the CLIP vision encoder.
    /// Falls back to filename-based text embedding if feature is not enabled.
    // `return` inside #[cfg] blocks is flagged by clippy but is semantically
    // required to exit the function before the not(feature) fallback branch.
    #[allow(clippy::needless_return)]
    pub async fn embed_image(
        &self,
        path: &Path,
    ) -> Result<mnemosyne_core::types::Embedding, Error> {
        #[cfg(feature = "clip-backend")]
        {
            let clip = self.models.get_clip_embedder(&self.vision_model_id).await?;
            let path = path.to_path_buf();
            return tokio::task::spawn_blocking(move || clip.embed_image(&path))
                .await
                .map_err(|e| Error::model(e.to_string()))?;
        }
        // Fallback: text embedding on the filename (no CLIP compiled).
        #[cfg(not(feature = "clip-backend"))]
        {
            let fallback = path.file_name().and_then(|n| n.to_str()).unwrap_or("image");
            let model = self.models.get_text_embedder(&self.text_model_id).await?;
            model.embed_text(fallback).await
        }
    }

    /// Encode a text query with the CLIP text encoder to produce a 512-dim embedding
    /// that can be compared against CLIP image embeddings stored in the index.
    ///
    /// This enables text-to-image semantic search when `clip-backend` is compiled in.
    /// CLIP's text and image encoders share the same latent space, so a text query
    /// like "a red car" will return semantically similar images.
    #[cfg(feature = "clip-backend")]
    async fn embed_text_with_clip(
        &self,
        text: &str,
    ) -> Result<mnemosyne_core::types::Embedding, Error> {
        let clip = self.models.get_clip_embedder(&self.vision_model_id).await?;
        let text = text.to_string();
        tokio::task::spawn_blocking(move || clip.embed_text(&text))
            .await
            .map_err(|e| Error::model(e.to_string()))?
    }

    /// Transcribe an audio file using Whisper and embed the transcript.
    #[allow(clippy::needless_return)]
    pub async fn transcribe_audio(&self, path: &Path) -> Result<String, Error> {
        #[cfg(feature = "whisper-backend")]
        {
            let whisper = self.models.get_whisper(&self.audio_model_id).await?;
            let path = path.to_path_buf();
            return tokio::task::spawn_blocking(move || whisper.transcribe(&path))
                .await
                .map_err(|e| Error::model(e.to_string()))?;
        }
        // Fallback (whisper-backend not compiled): return filename.
        #[cfg(not(feature = "whisper-backend"))]
        Ok(path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("audio")
            .to_string())
    }

    /// List all models registered in the local model registry.
    pub async fn list_models(
        &self,
    ) -> Result<Vec<mnemosyne_storage::model_repo::ModelRecord>, Error> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || mnemosyne_storage::ModelRepo::new(&db).list())
            .await
            .map_err(|e| Error::storage(e.to_string()))?
    }

    /// Download a model from HuggingFace Hub and register it locally.
    /// - `proxy_url`   – forwarded to the HTTP client (empty/None = no system proxy).
    /// - `hf_endpoint` – optional mirror URL (e.g. `"https://hf-mirror.com"`).
    pub async fn download_model(
        &self,
        model_id: &str,
        proxy_url: Option<&str>,
        hf_endpoint: Option<&str>,
    ) -> Result<(), Error> {
        use mnemosyne_model::ModelDownloader;

        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let cache_dir = std::path::PathBuf::from(home)
            .join(".mnemosyne")
            .join("models");
        tokio::fs::create_dir_all(&cache_dir)
            .await
            .map_err(Error::Io)?;

        let downloader = ModelDownloader::new(cache_dir);
        let local_path = downloader
            .download(model_id, proxy_url, hf_endpoint)
            .await?;

        let db = self.db.clone();
        let model_id = model_id.to_string();
        let path_str = local_path.to_string_lossy().to_string();
        tokio::task::spawn_blocking(move || {
            mnemosyne_storage::ModelRepo::new(&db).register(&model_id, &path_str, None)
        })
        .await
        .map_err(|e| Error::storage(e.to_string()))??;

        Ok(())
    }
}
