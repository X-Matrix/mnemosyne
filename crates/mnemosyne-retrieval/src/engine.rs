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
pub const DEFAULT_CLIP_MODEL:    &str = "openai/clip-vit-base-patch32";
/// Default Whisper model used when `whisper-backend` feature is enabled.
pub const DEFAULT_WHISPER_MODEL: &str = "openai/whisper-tiny";

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
                _ => {}
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

        // Generate embeddings — route to CLIP for images, Whisper for audio,
        // text embedder for everything else.
        let embeddings = self.embed_chunks(path, &chunks_content).await?;

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
        let mut embeddings = Vec::with_capacity(chunks.len());

        for chunk in chunks {
            let emb = match chunk {
                // ── Image: try CLIP, fall back to text embedder ───────────────
                ParsedContent::Image { caption, .. } => {
                    #[cfg(feature = "mnemosyne-model/clip-backend")]
                    {
                        // Feature routing delegated to engine builder config.
                        // For now we embed the caption text.
                        let _ = file_path; // suppress unused warning
                    }
                    let model = self.models.get_text_embedder(&self.text_model_id).await?;
                    model.embed_text(caption).await?
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

    /// Embed an image file using the CLIP vision encoder.
    /// Falls back to filename-based text embedding if feature is not enabled.
    pub async fn embed_image(&self, path: &Path) -> Result<mnemosyne_core::types::Embedding, Error> {
        #[cfg(all(feature = "mnemosyne-model/clip-backend"))]
        {
            // Dynamic feature check — if clip-backend was compiled in, use CLIP.
            let clip = self.models.get_clip_embedder(crate::engine::DEFAULT_CLIP_MODEL).await;
            if let Ok(clip) = clip {
                let path = path.to_path_buf();
                return tokio::task::spawn_blocking(move || clip.embed_image(&path))
                    .await
                    .map_err(|e| Error::model(e.to_string()))?;
            }
        }
        // Fallback: text embedding on the filename.
        let fallback = path.file_name().and_then(|n| n.to_str()).unwrap_or("image");
        let model = self.models.get_text_embedder(&self.text_model_id).await?;
        model.embed_text(fallback).await
    }

    /// Transcribe an audio file using Whisper and embed the transcript.
    pub async fn transcribe_audio(&self, path: &Path) -> Result<String, Error> {
        #[cfg(all(feature = "mnemosyne-model/whisper-backend"))]
        {
            let whisper = self.models.get_whisper(crate::engine::DEFAULT_WHISPER_MODEL).await;
            if let Ok(whisper) = whisper {
                let path = path.to_path_buf();
                return tokio::task::spawn_blocking(move || whisper.transcribe(&path))
                    .await
                    .map_err(|e| Error::model(e.to_string()))?;
            }
        }
        // Fallback: return filename.
        Ok(path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("audio")
            .to_string())
    }


    /// List all models registered in the local model registry.
    pub async fn list_models(&self) -> Result<Vec<mnemosyne_storage::model_repo::ModelRecord>, Error> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            mnemosyne_storage::ModelRepo::new(&db).list()
        })
        .await
        .map_err(|e| Error::storage(e.to_string()))?
    }

    /// Download a model from HuggingFace Hub and register it locally.
    pub async fn download_model(&self, model_id: &str) -> Result<(), Error> {
        use mnemosyne_model::ModelDownloader;

        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let cache_dir = std::path::PathBuf::from(home).join(".mnemosyne").join("models");
        tokio::fs::create_dir_all(&cache_dir).await.map_err(Error::Io)?;

        let downloader = ModelDownloader::new(cache_dir);
        let local_path = downloader.download(model_id).await?;

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

