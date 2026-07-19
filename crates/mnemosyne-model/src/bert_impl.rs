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
use tracing::{debug, info};

pub struct BertEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    pub dim: usize,
}

/// Return the local model directory `~/.mnemosyne/models/{model_id}` if it exists.
fn local_model_dir(model_id: &str) -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let dir = std::path::PathBuf::from(home)
        .join(".mnemosyne/models")
        .join(model_id);
    if dir.is_dir() {
        Some(dir)
    } else {
        None
    }
}

impl BertEmbedder {
    /// Load from local cache (`~/.mnemosyne/models/`) or download from HuggingFace Hub.
    pub async fn load(model_id: &str) -> Result<Self, Error> {
        info!("Loading BERT model '{}' via Candle", model_id);

        // ── Local cache: ~/.mnemosyne/models/{model_id}/ ──────────────────────
        let (config_path, tokenizer_path, weights_path) =
            if let Some(dir) = local_model_dir(model_id) {
                let weights = if dir.join("model.safetensors").exists() {
                    dir.join("model.safetensors")
                } else {
                    dir.join("pytorch_model.bin")
                };
                let cfg = dir.join("config.json");
                let tok = dir.join("tokenizer.json");
                if cfg.exists() && tok.exists() && weights.exists() {
                    info!("BERT: loading from local cache {}", dir.display());
                    (cfg, tok, weights)
                } else {
                    download_bert_files(model_id).await?
                }
            } else {
                download_bert_files(model_id).await?
            };

        let config: BertConfig = {
            let json = std::fs::read_to_string(&config_path).map_err(Error::Io)?;
            serde_json::from_str(&json).map_err(|e| Error::model(format!("parse config: {e}")))?
        };
        let dim = config.hidden_size;

        let device = Device::Cpu;

        let vb = load_var_builder(&weights_path, DTYPE, &device)?;

        let model =
            BertModel::load(vb, &config).map_err(|e| Error::model(format!("build model: {e}")))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| Error::model(format!("tokenizer: {e}")))?;

        info!("BERT '{}' ready (dim={dim}, device=CPU)", model_id);
        Ok(Self {
            model,
            tokenizer,
            device,
            dim,
        })
    }

    /// Encode a single text string into a normalised embedding.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, Error> {
        let preview: String = text.chars().take(80).collect();
        let truncated = text.chars().count() > 80;
        debug!(
            "BERT embed: input={:?}{} ({} chars)",
            preview,
            if truncated { "…" } else { "" },
            text.chars().count(),
        );

        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| Error::model(e.to_string()))?;

        let n_tokens = encoding.get_ids().len();
        debug!("BERT tokenised: {} tokens", n_tokens);

        let make = |data: &[u32]| {
            Tensor::new(data, &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| Error::model(e.to_string()))
        };

        let ids   = make(encoding.get_ids())?;
        let types = make(encoding.get_type_ids())?;
        let mask  = make(encoding.get_attention_mask())?;

        let t0 = std::time::Instant::now();
        let output = self
            .model
            .forward(&ids, &types, Some(&mask))
            .map_err(|e| Error::model(e.to_string()))?;
        debug!("BERT forward pass: {:?}", t0.elapsed());

        // Mean-pool over token dimension, then L2-normalise.
        let pooled = output
            .mean(1)
            .and_then(|t| t.squeeze(0))
            .map_err(|e| Error::model(e.to_string()))?;

        let vec: Vec<f32> = pooled.to_vec1().map_err(|e| Error::model(e.to_string()))?;
        let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-9);
        debug!(
            "BERT output: dim={}, L2_norm_before={:.6}",
            vec.len(),
            norm,
        );
        Ok(vec.into_iter().map(|v| v / norm).collect())
    }
}

// SAFETY: candle CPU tensors are read-only after construction; no interior
// mutability. Safe to share across threads.
unsafe impl Send for BertEmbedder {}
unsafe impl Sync for BertEmbedder {}

// ── Weight loading helper ──────────────────────────────────────────────────

fn load_var_builder(
    path: &std::path::Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>, Error> {
    let is_sf = path.extension().and_then(|e| e.to_str()) == Some("safetensors");
    if is_sf {
        unsafe {
            VarBuilder::from_mmaped_safetensors(&[path], dtype, device)
                .map_err(|e| Error::model(e.to_string()))
        }
    } else {
        VarBuilder::from_pth(path, dtype, device).map_err(|e| Error::model(e.to_string()))
    }
}

// ── HuggingFace Hub fallback ──────────────────────────────────────────────────

async fn download_bert_files(
    model_id: &str,
) -> Result<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf), Error> {
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
    let weights_path = match repo.get("model.safetensors").await {
        Ok(p) => p,
        Err(_) => repo
            .get("pytorch_model.bin")
            .await
            .map_err(|e| Error::model(format!("weights: {e}")))?,
    };
    Ok((config_path, tokenizer_path, weights_path))
}
