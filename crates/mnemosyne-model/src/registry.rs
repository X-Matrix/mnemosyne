use crate::TextEmbedder;
use mnemosyne_core::{traits::EmbeddingModel, Error};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Thread-safe cache of loaded embedding models.
pub struct ModelRegistry {
    text_models: RwLock<HashMap<String, Arc<TextEmbedder>>>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            text_models: RwLock::new(HashMap::new()),
        }
    }

    /// Load (or return cached) a text embedding model by its HuggingFace model ID.
    pub async fn get_text_embedder(
        &self,
        model_id: &str,
    ) -> Result<Arc<TextEmbedder>, Error> {
        // Fast path: already loaded.
        {
            let guard = self.text_models.read().unwrap();
            if let Some(model) = guard.get(model_id) {
                return Ok(Arc::clone(model));
            }
        }

        // Slow path: load the model.
        let model = Arc::new(TextEmbedder::load(model_id).await?);
        {
            let mut guard = self.text_models.write().unwrap();
            guard.insert(model_id.to_string(), Arc::clone(&model));
        }
        Ok(model)
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
