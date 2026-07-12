use crate::TextEmbedder;
use mnemosyne_core::{traits::EmbeddingModel, Error};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[cfg(feature = "clip-backend")]
use crate::ClipEmbedder;
#[cfg(feature = "whisper-backend")]
use crate::WhisperTranscriber;

/// Thread-safe cache of loaded embedding models.
pub struct ModelRegistry {
    text_models: RwLock<HashMap<String, Arc<TextEmbedder>>>,
    #[cfg(feature = "clip-backend")]
    clip_models: RwLock<HashMap<String, Arc<ClipEmbedder>>>,
    #[cfg(feature = "whisper-backend")]
    whisper_models: RwLock<HashMap<String, Arc<WhisperTranscriber>>>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            text_models: RwLock::new(HashMap::new()),
            #[cfg(feature = "clip-backend")]
            clip_models: RwLock::new(HashMap::new()),
            #[cfg(feature = "whisper-backend")]
            whisper_models: RwLock::new(HashMap::new()),
        }
    }

    /// Load (or return cached) a text embedding model.
    pub async fn get_text_embedder(&self, model_id: &str) -> Result<Arc<TextEmbedder>, Error> {
        {
            let g = self.text_models.read().unwrap();
            if let Some(m) = g.get(model_id) {
                return Ok(Arc::clone(m));
            }
        }
        let m = Arc::new(TextEmbedder::load(model_id).await?);
        self.text_models
            .write()
            .unwrap()
            .insert(model_id.to_string(), Arc::clone(&m));
        Ok(m)
    }

    /// Load (or return cached) a CLIP image embedder.
    #[cfg(feature = "clip-backend")]
    pub async fn get_clip_embedder(&self, model_id: &str) -> Result<Arc<ClipEmbedder>, Error> {
        {
            let g = self.clip_models.read().unwrap();
            if let Some(m) = g.get(model_id) {
                return Ok(Arc::clone(m));
            }
        }
        let m = Arc::new(ClipEmbedder::load(model_id).await?);
        self.clip_models
            .write()
            .unwrap()
            .insert(model_id.to_string(), Arc::clone(&m));
        Ok(m)
    }

    /// Load (or return cached) a Whisper audio transcriber.
    #[cfg(feature = "whisper-backend")]
    pub async fn get_whisper(&self, model_id: &str) -> Result<Arc<WhisperTranscriber>, Error> {
        {
            let g = self.whisper_models.read().unwrap();
            if let Some(m) = g.get(model_id) {
                return Ok(Arc::clone(m));
            }
        }
        let m = Arc::new(WhisperTranscriber::load(model_id).await?);
        self.whisper_models
            .write()
            .unwrap()
            .insert(model_id.to_string(), Arc::clone(&m));
        Ok(m)
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
