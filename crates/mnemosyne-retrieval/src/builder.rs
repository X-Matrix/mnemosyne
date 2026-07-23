use crate::engine::{SearchEngine, DEFAULT_BATCH_SIZE};
use crate::ignore::IgnoreConfig;
use mnemosyne_core::Error;
use mnemosyne_index::HybridIndex;
use mnemosyne_model::{ModelRegistry, DEFAULT_TEXT_MODEL};
use mnemosyne_parser::ParserRegistry;
use mnemosyne_storage::Database;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Default vision-embedding (CLIP) model.
pub const DEFAULT_VISION_MODEL: &str = "openai/clip-vit-base-patch32";
/// Default audio-transcription (Whisper) model.
pub const DEFAULT_AUDIO_MODEL: &str = "openai/whisper-tiny";

/// Fluent builder for [`SearchEngine`].
pub struct SearchEngineBuilder {
    db_path: Option<PathBuf>,
    text_model_id: String,
    vision_model_id: String,
    audio_model_id: String,
    ignore_config: IgnoreConfig,
    batch_size: usize,
}

impl SearchEngineBuilder {
    pub fn new() -> Self {
        Self {
            db_path: None,
            text_model_id: DEFAULT_TEXT_MODEL.to_string(),
            vision_model_id: DEFAULT_VISION_MODEL.to_string(),
            audio_model_id: DEFAULT_AUDIO_MODEL.to_string(),
            ignore_config: IgnoreConfig::default(),
            batch_size: DEFAULT_BATCH_SIZE,
        }
    }

    pub fn db_path(mut self, path: impl AsRef<Path>) -> Self {
        self.db_path = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn text_model(mut self, model_id: impl Into<String>) -> Self {
        self.text_model_id = model_id.into();
        self
    }
    pub fn vision_model(mut self, model_id: impl Into<String>) -> Self {
        self.vision_model_id = model_id.into();
        self
    }
    pub fn audio_model(mut self, model_id: impl Into<String>) -> Self {
        self.audio_model_id = model_id.into();
        self
    }

    /// Override the ignore configuration (directories to skip during indexing).
    /// Defaults to [`IgnoreConfig::default`] which covers the most common cases.
    pub fn ignore_config(mut self, config: IgnoreConfig) -> Self {
        self.ignore_config = config;
        self
    }

    /// Set the number of texts to embed per forward pass (default: [`DEFAULT_BATCH_SIZE`]).
    ///
    /// Larger values improve throughput (especially on Metal GPU) but require
    /// more memory.  Tune based on available RAM / VRAM and average chunk length.
    pub fn batch_size(mut self, n: usize) -> Self {
        self.batch_size = n.max(1);
        self
    }

    pub async fn build(self) -> Result<SearchEngine, Error> {
        let db_path = self.db_path.unwrap_or_else(default_db_path);

        if let Some(parent) = db_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(Error::Io)?;
        }

        let db = tokio::task::spawn_blocking({
            let db_path = db_path.clone();
            move || Database::open(&db_path)
        })
        .await
        .map_err(|e| Error::storage(e.to_string()))??;

        let index = Arc::new(HybridIndex::new(db.clone()));
        let parsers = Arc::new(ParserRegistry::with_defaults());
        let models = Arc::new(ModelRegistry::new());

        Ok(SearchEngine::new(
            db,
            index,
            parsers,
            models,
            self.text_model_id,
            self.vision_model_id,
            self.audio_model_id,
            Arc::new(self.ignore_config),
            self.batch_size,
        ))
    }
}

impl Default for SearchEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

fn default_db_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".mnemosyne").join("db.sqlite")
}
