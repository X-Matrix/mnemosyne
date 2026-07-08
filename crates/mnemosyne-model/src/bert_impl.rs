//! Real BERT embedding via HuggingFace Candle.
//!
//! Compiled only when the `candle-backend` feature is enabled.
//! Default model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim).

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use hf_hub::api::tokio::Api;
use mnemosyne_core::Error;
use tokenizers::Tokenizer;
use tracing::info;

pub struct BertEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    pub dim: usize,
}

impl BertEmbedder {
    /// Download model from HuggingFace Hub and load into memory (CPU).
    pub async fn load(model_id: &str) -> Result<Self, Error> {
        info!("Loading BERT model '{}' via Candle", model_id);

        let api = Api::new().map_err(|e| Error::model(e.to_string()))?;
        let repo = api.model(model_id.to_string());

        let config_path = repo
            .get("config.json")
            .await
            .map_err(|e| Error::model(format!("config.json: {e}")))?;

        let tokenizer_path = repo
            .get("tokenizer.json")
            .await
            .map_err(|e| Error::model(format!("tokenizer.json: {e}")))?;

        let weights_path = repo
            .get("model.safetensors")
            .await
            .or_else(|_| async { repo.get("pytorch_model.bin").await })
            .await
            .map_err(|e| Error::model(format!("weights: {e}")))?;

        let config: BertConfig = {
            let json = std::fs::read_to_string(&config_path).map_err(Error::Io)?;
            serde_json::from_str(&json).map_err(|e| Error::model(format!("parse config: {e}")))?
        };
        let dim = config.hidden_size;

        let device = Device::Cpu;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)
                .map_err(|e| Error::model(format!("load weights: {e}")))?
        };

        let model =
            BertModel::load(vb, &config).map_err(|e| Error::model(format!("build model: {e}")))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| Error::model(format!("tokenizer: {e}")))?;

        info!("BERT '{}' ready (dim={dim}, device=CPU)", model_id);
        Ok(Self { model, tokenizer, device, dim })
    }

    /// Encode a single text string into a normalised embedding.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, Error> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| Error::model(e.to_string()))?;

        let make = |data: &[u32]| {
            Tensor::new(data, &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| Error::model(e.to_string()))
        };

        let ids   = make(encoding.get_ids())?;
        let types = make(encoding.get_type_ids())?;
        let mask  = make(encoding.get_attention_mask())?;

        let output = self
            .model
            .forward(&ids, &types, Some(&mask))
            .map_err(|e| Error::model(e.to_string()))?;

        // Mean-pool over token dimension, then L2-normalise.
        let pooled = output
            .mean(1)
            .and_then(|t| t.squeeze(0))
            .map_err(|e| Error::model(e.to_string()))?;

        let vec: Vec<f32> = pooled.to_vec1().map_err(|e| Error::model(e.to_string()))?;
        let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-9);
        Ok(vec.into_iter().map(|v| v / norm).collect())
    }
}

// SAFETY: candle CPU tensors are read-only after construction; no interior
// mutability. Safe to share across threads.
unsafe impl Send for BertEmbedder {}
unsafe impl Sync for BertEmbedder {}
