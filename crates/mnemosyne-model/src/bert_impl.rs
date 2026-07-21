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
use std::collections::HashMap;
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
    /// Sparse projection weight for BGE-M3 lexical retrieval.
    /// Shape: [1, hidden_size].  `None` for models without sparse_linear.pt.
    sparse_linear: Option<Tensor>,
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

        // Both BERT and XLM-RoBERTa based models (e.g. BAAI/bge-m3) store their
        // weights WITHOUT a model-level prefix in the file — keys start directly
        // with `embeddings.*` and `encoder.*` at the root.  The only difference
        // that matters for loading is the pooling strategy.
        let model =
            BertModel::load(vb, &config).map_err(|e| Error::model(format!("build model: {e}")))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| Error::model(format!("tokenizer: {e}")))?;

        // ── Load sparse_linear.pt for BGE-M3 (lexical retrieval head) ─────────
        // Shape: [1, hidden_size] stored in fp16.  Non-fatal if absent.
        let sparse_linear = if is_roberta {
            local_model_dir(model_id)
                .map(|dir| dir.join("sparse_linear.pt"))
                .filter(|p| p.exists())
                .and_then(|p| {
                    // The file is a pytorch state_dict with key "weight".
                    // Stored in fp16; load and immediately cast to f32.
                    let vb = load_var_builder(&p, DType::F16, &device).ok()?;
                    let w = vb.get(&[1, dim], "weight").ok()?;
                    let w = w.to_dtype(DType::F32).ok()?;
                    info!("BGE-M3 sparse_linear loaded (shape [1, {dim}])");
                    Some(w)
                })
        } else {
            None
        };

        info!(
            "BERT '{}' ready (type={}, dim={dim}, pooling={}, sparse={}, device=CPU)",
            model_id,
            model_type,
            if is_roberta { "cls" } else { "mean" },
            sparse_linear.is_some(),
        );
        Ok(Self {
            model,
            tokenizer,
            device,
            dim,
            pooling,
            sparse_linear,
        })
    }

    /// Returns true if this embedder supports sparse (lexical) encoding.
    pub fn has_sparse(&self) -> bool {
        self.sparse_linear.is_some()
    }

    /// Compute BGE-M3 lexical weights for `text`.
    ///
    /// Returns a map of `token_id → importance_weight` (non-negative).  For each
    /// unique input token the weight is the max ReLU-activated score across all
    /// positions where that token appears.
    ///
    /// Algorithm (matches the Python FlagEmbedding reference):
    ///   hidden_states [seq, H] → sparse_linear → [seq, 1] → squeeze → ReLU
    ///   → aggregate by max per token_id
    pub fn embed_sparse(&self, text: &str) -> Result<HashMap<u32, f32>, Error> {
        let w = self
            .sparse_linear
            .as_ref()
            .ok_or_else(|| Error::model("sparse_linear not loaded".to_string()))?;

        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| Error::model(e.to_string()))?;

        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        // ── Debug: input ──────────────────────────────────────────────────────
        let preview: String = text.chars().take(60).collect();
        debug!(
            "sparse embed: input={:?}{} ({} tokens)",
            preview,
            if text.chars().count() > 60 { "…" } else { "" },
            token_ids.len(),
        );
        if tracing::enabled!(tracing::Level::DEBUG) {
            // Print token pieces so we can see how text is segmented.
            if let Some(tokens) = encoding.get_tokens().get(..token_ids.len().min(20)) {
                debug!("sparse tokens: {:?}", tokens);
            }
        }

        let make = |data: &[u32]| {
            Tensor::new(data, &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| Error::model(e.to_string()))
        };

        let ids = make(&token_ids)?;
        let types = make(encoding.get_type_ids())?;
        let mask = make(encoding.get_attention_mask())?;

        let t0 = std::time::Instant::now();

        // Forward pass → all hidden states [1, seq_len, hidden_size]
        let hidden = self
            .model
            .forward(&ids, &types, Some(&mask))
            .map_err(|e| Error::model(e.to_string()))?;

        // [1, seq_len, H] → [seq_len, H]
        let hidden = hidden
            .squeeze(0)
            .map_err(|e| Error::model(e.to_string()))?;

        // [seq_len, H] × [H, 1] → [seq_len, 1] → [seq_len]
        let scores = hidden
            .matmul(&w.t().map_err(|e| Error::model(e.to_string()))?)
            .map_err(|e| Error::model(e.to_string()))?
            .squeeze(1)
            .map_err(|e| Error::model(e.to_string()))?
            .relu()
            .map_err(|e| Error::model(e.to_string()))?;

        let scores_vec: Vec<f32> = scores.to_vec1().map_err(|e| Error::model(e.to_string()))?;

        debug!("sparse forward: {:?}", t0.elapsed());

        // Aggregate: max weight per unique token_id.
        let mut lexical: HashMap<u32, f32> = HashMap::new();
        for (&tid, &score) in token_ids.iter().zip(scores_vec.iter()) {
            if score > 0.0 {
                lexical.entry(tid).and_modify(|e| *e = e.max(score)).or_insert(score);
            }
        }

        // ── Debug: output ─────────────────────────────────────────────────────
        let nonzero = lexical.len();
        let max_w = lexical.values().cloned().fold(0.0f32, f32::max);
        debug!(
            "sparse output: {} non-zero token weights, max={:.4}",
            nonzero, max_w
        );
        if tracing::enabled!(tracing::Level::DEBUG) {
            // Show top-5 weighted tokens (token_id + weight).
            let mut sorted: Vec<(u32, f32)> = lexical.iter().map(|(&k, &v)| (k, v)).collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top: Vec<_> = sorted
                .iter()
                .take(5)
                .map(|(tid, w)| {
                    let piece = self
                        .tokenizer
                        .id_to_token(*tid)
                        .unwrap_or_else(|| format!("[{tid}]"));
                    format!("{:?}:{:.4}", piece, w)
                })
                .collect();
            debug!("sparse top-5: [{}]", top.join(", "));
        }

        Ok(lexical)
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

        let ids = make(encoding.get_ids())?;
        let types = make(encoding.get_type_ids())?;
        let mask = make(encoding.get_attention_mask())?;

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
        debug!("BERT output: dim={}, L2_norm_before={:.6}", vec.len(), norm,);
        Ok(vec.into_iter().map(|v| v / norm).collect())
    }

    /// Encode text and return **both** the dense embedding and (if available)
    /// the sparse lexical weights in a **single forward pass**.
    ///
    /// Using this instead of calling `embed()` + `embed_sparse()` separately
    /// halves the computation time for BGE-M3 models.
    pub fn embed_combined(
        &self,
        text: &str,
    ) -> Result<(Vec<f32>, Option<HashMap<u32, f32>>), Error> {
        let preview: String = text.chars().take(80).collect();
        let truncated = text.chars().count() > 80;
        debug!(
            "BERT embed_combined: input={:?}{} ({} chars)",
            preview,
            if truncated { "…" } else { "" },
            text.chars().count(),
        );

        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| Error::model(e.to_string()))?;

        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        debug!("BERT tokenised: {} tokens", token_ids.len());

        let make = |data: &[u32]| {
            Tensor::new(data, &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| Error::model(e.to_string()))
        };

        let ids   = make(&token_ids)?;
        let types = make(encoding.get_type_ids())?;
        let mask  = make(encoding.get_attention_mask())?;

        let t0 = std::time::Instant::now();
        // ── Single forward pass ───────────────────────────────────────────────
        let output = self
            .model
            .forward(&ids, &types, Some(&mask))
            .map_err(|e| Error::model(e.to_string()))?;
        debug!("BERT forward pass: {:?}", t0.elapsed());

        // ── Dense embedding (pool + L2 normalise) ─────────────────────────────
        let pooled = match self.pooling {
            Pooling::Mean => output
                .mean(1)
                .and_then(|t| t.squeeze(0))
                .map_err(|e| Error::model(e.to_string()))?,
            Pooling::Cls => output
                .narrow(1, 0, 1)
                .and_then(|t| t.squeeze(1))
                .and_then(|t| t.squeeze(0))
                .map_err(|e| Error::model(e.to_string()))?,
        };
        let dense_raw: Vec<f32> = pooled.to_vec1().map_err(|e| Error::model(e.to_string()))?;
        let norm = dense_raw.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-9);
        debug!("BERT output: dim={}, L2_norm_before={:.6}", dense_raw.len(), norm);
        let dense: Vec<f32> = dense_raw.into_iter().map(|v| v / norm).collect();

        // ── Sparse weights (reuse hidden states if sparse_linear is loaded) ───
        let sparse: Option<HashMap<u32, f32>> = if let Some(w) = &self.sparse_linear {
            // output: [1, seq_len, H] → [seq_len, H]
            let hidden = output
                .squeeze(0)
                .map_err(|e| Error::model(e.to_string()))?;
            // [seq_len, H] × [H, 1] → [seq_len, 1] → [seq_len] → ReLU
            let scores = hidden
                .matmul(&w.t().map_err(|e| Error::model(e.to_string()))?)
                .map_err(|e| Error::model(e.to_string()))?
                .squeeze(1)
                .map_err(|e| Error::model(e.to_string()))?
                .relu()
                .map_err(|e| Error::model(e.to_string()))?;
            let scores_vec: Vec<f32> =
                scores.to_vec1().map_err(|e| Error::model(e.to_string()))?;

            let mut lexical: HashMap<u32, f32> = HashMap::new();
            for (&tid, &score) in token_ids.iter().zip(scores_vec.iter()) {
                if score > 0.0 {
                    lexical
                        .entry(tid)
                        .and_modify(|e| *e = e.max(score))
                        .or_insert(score);
                }
            }

            let nonzero = lexical.len();
            let max_w = lexical.values().cloned().fold(0.0f32, f32::max);
            debug!(
                "sparse output: {} non-zero token weights, max={:.4}",
                nonzero, max_w
            );
            Some(lexical)
        } else {
            None
        };

        Ok((dense, sparse))
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
