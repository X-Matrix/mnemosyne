use mnemosyne_core::Error;
use std::path::PathBuf;
use tracing::info;

/// Downloads model files from HuggingFace Hub using `hf-hub`.
pub struct ModelDownloader {
    cache_dir: PathBuf,
}

impl ModelDownloader {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }

    /// Download a model and return the local directory path.
    pub async fn download(&self, model_id: &str) -> Result<PathBuf, Error> {
        use hf_hub::api::tokio::Api;

        info!("Downloading model: {}", model_id);

        let api = Api::new()
            .map_err(|e| Error::model(format!("hf-hub init failed: {e}")))?;

        // Download config.json, tokenizer.json, and model weights.
        let repo = api.model(model_id.to_string());
        let config_path = repo
            .get("config.json")
            .await
            .map_err(|e| Error::model(format!("failed to download config.json: {e}")))?;

        // Return the parent directory of the downloaded config file.
        let model_dir = config_path
            .parent()
            .ok_or_else(|| Error::model("unexpected cache path".to_string()))?
            .to_path_buf();

        info!("Model cached at: {}", model_dir.display());
        Ok(model_dir)
    }
}
