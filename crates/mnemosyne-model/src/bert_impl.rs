//! Real BERT / XLM-RoBERTa text embedding via HuggingFace Candle.
//!
//! Compiled only when the `candle-backend` feature is enabled.
//!
//! Supported architectures and pooling strategies:
//! - `bert`        (e.g. sentence-transformers/all-MiniLM-L6-v2) — mean pooling
//! - `xlm-roberta` (e.g. BAAI/bge-m3)                           — CLS pooling

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use hf_hub::api::tokio::Api;
use mnemosyne_core::Error;
use tokenizers::Tokenizer;
use tracing::{debug, info};

// ── Pooling strategy ──────────────────────────────────────────────────────────

/// How to aggregate per-token hidden states into a single embedding vector.
pub enum Pooling {
    /// Arithmetic mean of all non-padding token states.
    /// Standard for sentence-transformers models (e.g. all-MiniLM-L6-v2).
    Mean,
    /// Use only the `[CLS]` / `<s>` token (position 0).
    /// Recommended for BGE and most RoBERTa-based retrieval models.
    Cls,
}

// ── Embedder ──────────────────────────────────────────────────────────────────

pub struct BertEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    pub dim: usize,
    pooling: Pooling,
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
    ///
    /// Pooling strategy and weight-key prefix are auto-detected from `model_type`
    /// in the model's `config.json`:
    /// - `"bert"`        → mean pooling, root weight prefix
    /// - `"xlm-roberta"` / `"roberta"` → CLS pooling, `roberta.` weight prefix
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

        // ── Detect architecture from config ───────────────────────────────────
        let config_json_str = std::fs::read_to_string(&config_path).map_err(Error::Io)?;
        let config_json: serde_json::Value = serde_json::from_str(&config_json_str)
            .map_err(|e| Error::model(format!("parse config json: {e}")))?;
        let model_type = config_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("bert")
            .to_string();

        let config: BertConfig = serde_json::from_str(&config_json_str)
            .map_err(|e| Error::model(format!("parse config: {e}")))?;
        let dim = config.hidden_size;

        // ── Pooling and weight-prefix selection ───────────────────────────────
        let is_roberta = matches!(model_type.as_str(), "xlm-roberta" | "roberta");
        let pooling = if is_roberta {
            info!("BERT: model_type={model_type}, using CLS pooling");
            Pooling::Cls
        } else {
            Pooling::Mean
        };

        let device = Device::Cpu;
        let vb = load_var_builder(&weights_path, DTYPE, &device)?;

        // XLM-RoBERTa safetensors store weights under the `roberta.*` namespace.
        let model = if is_roberta {
            BertModel::load(vb.pp("roberta"), &config)
                .map_err(|e| Error::model(format!("build model (roberta): {e}")))?
        } else {
            BertModel::load(vb, &config)
                .map_err(|e| Error::model(format!("build model: {e}")))?
        };

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| Error::model(format!("tokenizer: {e}")))?;

        info!(
            "BERT '{}' ready (type={}, dim={dim}, pooling={}, device=CPU)",
            model_id,
            model_type,
            if is_roberta { "cls" } else { "mean" }
        );
        Ok(Self {
            model,
            tokenizer,
            device,
            dim,
            pooling,
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

        // Pool the sequence output into a single vector, then L2-normalise.
        let pooled = match self.pooling {
            Pooling::Mean => {
                // Arithmetic mean over all token positions.
                output
                    .mean(1)
                    .and_then(|t| t.squeeze(0))
                    .map_err(|e| Error::model(e.to_string()))?
            }
            Pooling::Cls => {
                // CLS / <s> token sits at sequence position 0.
                // output: (1, seq_len, hidden) → narrow to (1, 1, hidden)
                //         → squeeze seq dim → (1, hidden) → squeeze batch → (hidden,)
                output
                    .narrow(1, 0, 1)
                    .and_then(|t| t.squeeze(1))
                    .and_then(|t| t.squeeze(0))
                    .map_err(|e| Error::model(e.to_string()))?
            }
        };

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
