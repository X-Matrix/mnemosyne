use crate::engine::SearchEngine;
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
}

impl SearchEngineBuilder {
    pub fn new() -> Self {
        Self {
            db_path: None,
            text_model_id: DEFAULT_TEXT_MODEL.to_string(),
            vision_model_id: DEFAULT_VISION_MODEL.to_string(),
            audio_model_id: DEFAULT_AUDIO_MODEL.to_string(),
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
