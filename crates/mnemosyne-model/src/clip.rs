//! CLIP image + text embedding via HuggingFace Candle.
//!
//! Compiled only with the `clip-backend` feature.
//!
//! Supported models:
//! - `openai/clip-vit-base-patch32`        (512-dim, English text)
//! - `OFA-Sys/chinese-clip-vit-base-patch16` (512-dim, Chinese / multilingual)

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::chinese_clip::{self as ch_clip, ChineseClipConfig};
use candle_transformers::models::clip::{self, ClipConfig};
use hf_hub::api::tokio::Api;
use mnemosyne_core::Error;
use std::path::Path;
use tokenizers::{
    models::wordpiece::WordPiece,
    normalizers::bert::BertNormalizer,
    pre_tokenizers::bert::BertPreTokenizer,
    processors::bert::BertProcessing,
    Model, // needed for WordPiece::get_vocab()
    Tokenizer,
};
use tracing::info;

/// Output embedding dimension for both model variants.
pub const CLIP_DIM: usize = 512;

// ── OpenAI CLIP constants ─────────────────────────────────────────────────────

/// Max sequence length for OpenAI CLIP text encoder (ViT-B/32).
const CLIP_MAX_SEQ: usize = 77;

/// Normalization constants shared by both variants (ImageNet / CLIP training).
const CLIP_MEAN: [f32; 3] = [0.481_454_7, 0.457_827_5, 0.408_210_7];
const CLIP_STD:  [f32; 3] = [0.268_629_5, 0.261_302_6, 0.275_777_1];

// ── Chinese CLIP constants ────────────────────────────────────────────────────

/// Text context window used during Chinese CLIP training.
const CH_CLIP_MAX_SEQ: usize = 52;

// ── Internal variant ──────────────────────────────────────────────────────────

struct OpenAiState {
    model: clip::ClipModel,
    tokenizer: Option<Tokenizer>,
    device: Device,
}

struct ChineseState {
    model: ch_clip::ChineseClipModel,
    tokenizer: Tokenizer,
    device: Device,
    /// Token ID used for padding (0 in standard BERT / Chinese BERT vocab).
    pad_id: u32,
}

enum ClipVariant {
    OpenAi(OpenAiState),
    Chinese(ChineseState),
}

// ── Public embedder ───────────────────────────────────────────────────────────

pub struct ClipEmbedder {
    variant: ClipVariant,
}

/// Return the local model directory `~/.mnemosyne/models/{model_id}` if it exists.
fn local_model_dir(model_id: &str) -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let dir = std::path::PathBuf::from(home)
        .join(".mnemosyne/models")
        .join(model_id);
    if dir.is_dir() { Some(dir) } else { None }
}

impl ClipEmbedder {
    /// Load from local cache (`~/.mnemosyne/models/`) or download from HuggingFace Hub.
    ///
    /// Model variant is auto-detected from `model_id`:
    /// - IDs containing `"chinese-clip"` or starting with `"OFA-Sys/"` → Chinese CLIP
    /// - Everything else → OpenAI CLIP
    pub async fn load(model_id: &str) -> Result<Self, Error> {
        info!("Loading CLIP model '{}'", model_id);
        if is_chinese_clip(model_id) {
            Self::load_chinese(model_id).await
        } else {
            Self::load_openai(model_id).await
        }
    }

    // ── OpenAI CLIP loader ────────────────────────────────────────────────────

    async fn load_openai(model_id: &str) -> Result<Self, Error> {
        let device = Device::Cpu;

        let weights_path = match local_model_dir(model_id) {
            Some(dir) => {
                let sf = dir.join("model.safetensors");
                let pt = dir.join("pytorch_model.bin");
                if sf.exists() {
                    info!("CLIP: loading from local cache {}", dir.display());
                    sf
                } else if pt.exists() {
                    info!("CLIP: loading from local cache {}", dir.display());
                    pt
                } else {
                    download_weights(model_id).await?
                }
            }
            None => download_weights(model_id).await?,
        };

        let config = ClipConfig::vit_base_patch32();
        let vb = load_var_builder(&weights_path, DType::F32, &device)?;
        let model =
            clip::ClipModel::new(vb, &config).map_err(|e| Error::model(e.to_string()))?;

        let tokenizer = local_model_dir(model_id).and_then(|dir| {
            let p = dir.join("tokenizer.json");
            p.exists().then(|| Tokenizer::from_file(&p).ok()).flatten()
        });

        info!(
            "CLIP '{}' loaded (dim={CLIP_DIM}, device=CPU, text_enc={})",
            model_id,
            tokenizer.is_some()
        );
        Ok(Self {
            variant: ClipVariant::OpenAi(OpenAiState { model, tokenizer, device }),
        })
    }

    // ── Chinese CLIP loader ───────────────────────────────────────────────────

    async fn load_chinese(model_id: &str) -> Result<Self, Error> {
        let device = Device::Cpu;

        // ── Weights ───────────────────────────────────────────────────────────
        let weights_path = match local_model_dir(model_id) {
            Some(dir) => {
                let sf = dir.join("model.safetensors");
                let pt = dir.join("pytorch_model.bin");
                if sf.exists() {
                    info!("Chinese CLIP: loading from local cache {}", dir.display());
                    sf
                } else if pt.exists() {
                    info!("Chinese CLIP: loading from local cache {}", dir.display());
                    pt
                } else {
                    download_weights(model_id).await?
                }
            }
            None => download_weights(model_id).await?,
        };

        // ── Tokenizer (Chinese BERT vocab) ────────────────────────────────────
        // OFA-Sys/chinese-clip-vit-base-patch16 provides `vocab.txt` (standard
        // Chinese BERT vocabulary) rather than `tokenizer.json`.  We build the
        // tokenizer programmatically from vocab.txt when tokenizer.json is absent.
        let tokenizer = load_chinese_clip_tokenizer(model_id).await?;

        // PAD token is `[PAD]` (ID 0 in Chinese BERT vocab).
        let pad_id = tokenizer
            .get_vocab(true)
            .get("[PAD]")
            .copied()
            .unwrap_or(0u32);

        // ── Model ─────────────────────────────────────────────────────────────
        let config = ChineseClipConfig::clip_vit_base_patch16();
        let vb = load_var_builder(&weights_path, DType::F32, &device)?;
        let model = ch_clip::ChineseClipModel::new(vb, &config)
            .map_err(|e| Error::model(format!("Chinese CLIP model: {e}")))?;

        info!(
            "Chinese CLIP '{}' loaded (dim={CLIP_DIM}, device=CPU)",
            model_id
        );
        Ok(Self {
            variant: ClipVariant::Chinese(ChineseState {
                model,
                tokenizer,
                device,
                pad_id,
            }),
        })
    }

    // ── Public interface ──────────────────────────────────────────────────────

    /// Produce a 512-dim L2-normalised image embedding.
    pub fn embed_image(&self, image_path: &Path) -> Result<Vec<f32>, Error> {
        match &self.variant {
            ClipVariant::OpenAi(s) => embed_image_openai(&s.model, &s.device, image_path),
            ClipVariant::Chinese(s) => embed_image_chinese(&s.model, &s.device, image_path),
        }
    }

    /// Produce a 512-dim L2-normalised text embedding in the same space as `embed_image`.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>, Error> {
        match &self.variant {
            ClipVariant::OpenAi(s) => embed_text_openai(s, text),
            ClipVariant::Chinese(s) => embed_text_chinese(s, text),
        }
    }
}

// SAFETY: candle CPU tensors are read-only after load; no interior mutability.
unsafe impl Send for ClipEmbedder {}
unsafe impl Sync for ClipEmbedder {}

// ── OpenAI CLIP inference ─────────────────────────────────────────────────────

fn embed_image_openai(model: &clip::ClipModel, device: &Device, path: &Path) -> Result<Vec<f32>, Error> {
    let pixels = preprocess_image(path)?;
    let tensor = Tensor::from_vec(pixels, (1usize, 3, 224, 224), device)
        .map_err(|e| Error::model(e.to_string()))?;
    let features = model
        .get_image_features(&tensor)
        .map_err(|e| Error::model(e.to_string()))?;
    let vec: Vec<f32> = features
        .squeeze(0)
        .map_err(|e| Error::model(e.to_string()))?
        .to_vec1()
        .map_err(|e| Error::model(e.to_string()))?;
    Ok(l2_normalize(vec))
}

fn embed_text_openai(state: &OpenAiState, text: &str) -> Result<Vec<f32>, Error> {
    let tokenizer = state
        .tokenizer
        .as_ref()
        .ok_or_else(|| Error::model("CLIP tokenizer not loaded".to_string()))?;

    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| Error::model(e.to_string()))?;
    let ids: Vec<u32> = encoding
        .get_ids()
        .iter()
        .copied()
        .take(CLIP_MAX_SEQ - 2)
        .collect();

    // OpenAI CLIP special tokens: SOT = 49406, EOT = 49407.
    let sot: u32 = 49406;
    let eot: u32 = 49407;
    let mut seq = vec![sot];
    seq.extend_from_slice(&ids);
    seq.push(eot);
    seq.resize(CLIP_MAX_SEQ, 0u32);

    let input = Tensor::new(seq.as_slice(), &state.device)
        .and_then(|t| t.unsqueeze(0))
        .map_err(|e| Error::model(e.to_string()))?;
    let features = state
        .model
        .get_text_features(&input)
        .map_err(|e| Error::model(e.to_string()))?;
    let vec: Vec<f32> = features
        .squeeze(0)
        .map_err(|e| Error::model(e.to_string()))?
        .to_vec1()
        .map_err(|e| Error::model(e.to_string()))?;
    Ok(l2_normalize(vec))
}

// ── Chinese CLIP inference ────────────────────────────────────────────────────

fn embed_image_chinese(
    model: &ch_clip::ChineseClipModel,
    device: &Device,
    path: &Path,
) -> Result<Vec<f32>, Error> {
    // Use Chinese-CLIP-specific preprocessing (Triangle filter, resize_to_fill)
    // to match the reference implementation exactly.
    let pixels = preprocess_image_chinese(path)?;
    let tensor = Tensor::from_vec(pixels, (1usize, 3, 224, 224), device)
        .map_err(|e| Error::model(e.to_string()))?;
    let features = model
        .get_image_features(&tensor)
        .map_err(|e| Error::model(e.to_string()))?;
    let vec: Vec<f32> = features
        .squeeze(0)
        .map_err(|e| Error::model(e.to_string()))?
        .to_vec1()
        .map_err(|e| Error::model(e.to_string()))?;
    Ok(l2_normalize(vec))
}

fn embed_text_chinese(state: &ChineseState, text: &str) -> Result<Vec<f32>, Error> {
    // BERT-style tokenisation: [CLS] tokens... [SEP]
    let encoding = state
        .tokenizer
        .encode(text, true)
        .map_err(|e| Error::model(e.to_string()))?;

    // Truncate to CH_CLIP_MAX_SEQ then pad to that length.
    let raw_ids:   Vec<u32> = encoding.get_ids().iter().copied().take(CH_CLIP_MAX_SEQ).collect();
    // type_ids from tokenizer (all-zero for single-segment input, matches reference)
    let raw_types: Vec<u32> = encoding.get_type_ids().iter().copied().take(CH_CLIP_MAX_SEQ).collect();
    let raw_mask:  Vec<u32> = encoding.get_attention_mask().iter().copied().take(CH_CLIP_MAX_SEQ).collect();
    let seq_len = raw_ids.len();

    let mut ids   = raw_ids;
    let mut types = raw_types;
    let mut mask  = raw_mask;
    ids.resize(CH_CLIP_MAX_SEQ,  state.pad_id);
    types.resize(CH_CLIP_MAX_SEQ, 0u32);
    mask.resize(CH_CLIP_MAX_SEQ,  0u32);

    let make = |v: Vec<u32>| {
        Tensor::new(v.as_slice(), &state.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| Error::model(e.to_string()))
    };
    let input_ids   = make(ids)?;
    let token_types = make(types)?;
    let attn_mask   = make(mask)?;

    tracing::debug!(
        "Chinese CLIP text: {:?}… ({} tokens padded to {})",
        text.chars().take(40).collect::<String>(),
        seq_len,
        CH_CLIP_MAX_SEQ
    );

    let features = state
        .model
        .get_text_features(&input_ids, Some(&token_types), Some(&attn_mask))
        .map_err(|e| Error::model(e.to_string()))?;

    let vec: Vec<f32> = features
        .squeeze(0)
        .map_err(|e| Error::model(e.to_string()))?
        .to_vec1()
        .map_err(|e| Error::model(e.to_string()))?;
    Ok(l2_normalize(vec))
}

// ── Model-ID helpers ──────────────────────────────────────────────────────────

fn is_chinese_clip(model_id: &str) -> bool {
    let id = model_id.to_lowercase();
    id.contains("chinese-clip") || id.starts_with("ofa-sys/")
}

// ── Weight / tokenizer loading helpers ───────────────────────────────────────

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

// ── Chinese CLIP tokenizer helpers ───────────────────────────────────────────

/// Load (or download + build) the tokenizer for a Chinese CLIP model.
///
/// OFA-Sys/chinese-clip-vit-base-patch16 provides only `vocab.txt` (no
/// `tokenizer.json`).  We build the tokenizer programmatically when needed.
///
/// Priority:
/// 1. Local `tokenizer.json` → load directly.
/// 2. Local `vocab.txt`      → build BertWordPiece tokenizer.
/// 3. Download `tokenizer.json` from Hub (may return 404 → skip).
/// 4. Download `vocab.txt`   → build BertWordPiece tokenizer.
async fn load_chinese_clip_tokenizer(model_id: &str) -> Result<Tokenizer, Error> {
    if let Some(dir) = local_model_dir(model_id) {
        let json = dir.join("tokenizer.json");
        if json.exists() {
            return Tokenizer::from_file(&json)
                .map_err(|e| Error::model(format!("tokenizer.json: {e}")));
        }
        let vocab = dir.join("vocab.txt");
        if vocab.exists() {
            return build_tokenizer_from_vocab(&vocab);
        }
    }
    // Try tokenizer.json from Hub (silently skip 404)
    if let Ok(p) = download_file(model_id, "tokenizer.json").await {
        if let Ok(tok) = Tokenizer::from_file(&p) {
            return Ok(tok);
        }
    }
    // Fall back to vocab.txt
    let vocab_path = download_file(model_id, "vocab.txt").await?;
    build_tokenizer_from_vocab(&vocab_path)
}

/// Build a Chinese BERT WordPiece tokenizer from a raw `vocab.txt` file.
///
/// Matches the standard `bert-base-chinese` tokenizer configuration:
/// - `handle_chinese_chars = true`  (space-pad CJK chars before splitting)
/// - `lowercase = false`            (Chinese CLIP does NOT lowercase)
/// - `[CLS]` / `[SEP]` post-processing
fn build_tokenizer_from_vocab(vocab_path: &std::path::Path) -> Result<Tokenizer, Error> {
    let vocab_str = vocab_path
        .to_str()
        .ok_or_else(|| Error::model("vocab path is not valid UTF-8".to_string()))?;

    let model = WordPiece::from_file(vocab_str)
        .unk_token("[UNK]".to_string())
        .build()
        .map_err(|e| Error::model(format!("WordPiece from vocab.txt: {e}")))?;

    let vocab = model.get_vocab();
    let cls_id = vocab
        .get("[CLS]")
        .copied()
        .ok_or_else(|| Error::model("[CLS] not found in vocab.txt".to_string()))?;
    let sep_id = vocab
        .get("[SEP]")
        .copied()
        .ok_or_else(|| Error::model("[SEP] not found in vocab.txt".to_string()))?;

    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_normalizer(Some(BertNormalizer::new(true, true, None, false)));
    tokenizer.with_pre_tokenizer(Some(BertPreTokenizer));
    tokenizer.with_post_processor(Some(BertProcessing::new(
        ("[SEP]".to_string(), sep_id),
        ("[CLS]".to_string(), cls_id),
    )));

    info!("Built Chinese BERT tokenizer from vocab.txt ({} tokens)", vocab.len());
    Ok(tokenizer)
}

// ── Weight / tokenizer download helpers ──────────────────────────────────────

async fn download_weights(model_id: &str) -> Result<std::path::PathBuf, Error> {
    let api  = Api::new().map_err(|e| Error::model(e.to_string()))?;
    let repo = api.model(model_id.to_string());
    match repo.get("model.safetensors").await {
        Ok(p) => Ok(p),
        Err(_) => repo
            .get("pytorch_model.bin")
            .await
            .map_err(|e| Error::model(format!("weights: {e}"))),
    }
}

/// Download any single file from a HuggingFace model repo.
async fn download_file(model_id: &str, filename: &str) -> Result<std::path::PathBuf, Error> {
    let api  = Api::new().map_err(|e| Error::model(e.to_string()))?;
    let repo = api.model(model_id.to_string());
    repo.get(filename)
        .await
        .map_err(|e| Error::model(format!("{filename}: {e}")))
}

// ── Shared image preprocessing (224×224, same normalization for both models) ──

fn preprocess_image(path: &Path) -> Result<Vec<f32>, Error> {
    use image::imageops::FilterType;

    // ── OpenAI CLIP ──────────────────────────────────────────────────────────
    // Official CLIP preprocessing: resize & crop to 224×224 with bicubic
    // (CatmullRom is the `image` crate's closest equivalent).
    let img = image::open(path)
        .map_err(|e| Error::parse(e.to_string()))?
        .resize_to_fill(224, 224, FilterType::CatmullRom)
        .to_rgb8();

    let mut chw = vec![0.0f32; 3 * 224 * 224];
    for y in 0..224usize {
        for x in 0..224usize {
            let p = img.get_pixel(x as u32, y as u32);
            for c in 0..3usize {
                let v = p[c] as f32 / 255.0;
                chw[c * 224 * 224 + y * 224 + x] = (v - CLIP_MEAN[c]) / CLIP_STD[c];
            }
        }
    }
    Ok(chw)
}

/// Image preprocessing for Chinese CLIP.
///
/// Matches the reference implementation exactly:
/// `resize_to_fill(224, 224, Triangle)` → HWC raw bytes → CHW f32 → CLIP normalize.
fn preprocess_image_chinese(path: &Path) -> Result<Vec<f32>, Error> {
    use image::imageops::FilterType;

    // Chinese CLIP reference uses Triangle (bilinear) filter + resize_to_fill.
    let img = image::open(path)
        .map_err(|e| Error::parse(e.to_string()))?
        .resize_to_fill(224, 224, FilterType::Triangle)
        .to_rgb8();

    let mut chw = vec![0.0f32; 3 * 224 * 224];
    for y in 0..224usize {
        for x in 0..224usize {
            let p = img.get_pixel(x as u32, y as u32);
            for c in 0..3usize {
                let v = p[c] as f32 / 255.0;
                chw[c * 224 * 224 + y * 224 + x] = (v - CLIP_MEAN[c]) / CLIP_STD[c];
            }
        }
    }
    Ok(chw)
}

fn l2_normalize(mut vec: Vec<f32>) -> Vec<f32> {
    let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-9);
    vec.iter_mut().for_each(|v| *v /= norm);
    vec
}
