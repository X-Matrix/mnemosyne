//! CLIP image embedding via HuggingFace Candle.
//!
//! Compiled only with the `clip-backend` feature.
//! Model: `openai/clip-vit-base-patch32` (512-dim, 224×224 input).

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip::{self, ClipConfig};
use hf_hub::api::tokio::Api;
use mnemosyne_core::Error;
use std::path::Path;
use tracing::info;

/// Output embedding dimension for CLIP ViT-B/32.
pub const CLIP_DIM: usize = 512;

/// CLIP normalization constants (ImageNet-style used by OpenAI CLIP).
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD:  [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

pub struct ClipEmbedder {
    model:  clip::ClipModel,
    device: Device,
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
    pub async fn load(model_id: &str) -> Result<Self, Error> {
        info!("Loading CLIP model '{}'", model_id);
        let device = Device::Cpu;

        // ── Local cache: ~/.mnemosyne/models/{model_id}/ ──────────────────────
        let weights_path = if let Some(dir) = local_model_dir(model_id) {
            let sf = dir.join("model.safetensors");
            let pt = dir.join("pytorch_model.bin");
            if sf.exists() {
                info!("CLIP: loading from local cache {}", dir.display());
                sf
            } else if pt.exists() {
                info!("CLIP: loading from local cache {}", dir.display());
                pt
            } else {
                download_clip_weights(model_id).await?
            }
        } else {
            download_clip_weights(model_id).await?
        };

        // Use the candle-transformers built-in config for ViT-B/32.
        // ClipConfig does not implement serde::Deserialize so we cannot parse
        // the JSON directly; the static constructor encodes the correct values.
        let config = ClipConfig::vit_base_patch32();

        let vb = load_var_builder(&weights_path, DType::F32, &device)?;
        let model = clip::ClipModel::new(vb, &config)
            .map_err(|e| Error::model(e.to_string()))?;

        info!("CLIP '{}' loaded (dim={CLIP_DIM}, device=CPU)", model_id);
        Ok(Self { model, device })
    }

    /// Produce a 512-dim L2-normalised embedding for `image_path`.
    pub fn embed_image(&self, image_path: &Path) -> Result<Vec<f32>, Error> {
        let pixels = preprocess_image(image_path)?;
        let tensor = Tensor::from_vec(pixels, (1usize, 3, 224, 224), &self.device)
            .map_err(|e| Error::model(e.to_string()))?;

        let features = self.model.get_image_features(&tensor)
            .map_err(|e| Error::model(e.to_string()))?;

        let vec: Vec<f32> = features.squeeze(0)
            .map_err(|e| Error::model(e.to_string()))?
            .to_vec1()
            .map_err(|e| Error::model(e.to_string()))?;

        Ok(l2_normalize(vec))
    }
}

// SAFETY: candle CPU tensors are read-only after load; no interior mutability.
unsafe impl Send for ClipEmbedder {}
unsafe impl Sync for ClipEmbedder {}

// ── Weight loading helper ──────────────────────────────────────────────────

/// Load a VarBuilder that handles both `.safetensors` (mmap) and
/// `pytorch_model.bin` (PyTorch pickle) weight files.
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
        VarBuilder::from_pth(path, dtype, device)
            .map_err(|e| Error::model(e.to_string()))
    }
}

// ── HuggingFace Hub fallback ──────────────────────────────────────────────────

async fn download_clip_weights(model_id: &str) -> Result<std::path::PathBuf, Error> {
    let api = Api::new().map_err(|e| Error::model(e.to_string()))?;
    let repo = api.model(model_id.to_string());
    match repo.get("model.safetensors").await {
        Ok(p) => Ok(p),
        Err(_) => repo.get("pytorch_model.bin").await
            .map_err(|e| Error::model(format!("weights: {e}"))),
    }
}

// ── Image preprocessing ───────────────────────────────────────────────────────

/// Resize image to 224×224, normalize, and return CHW f32 layout.
fn preprocess_image(path: &Path) -> Result<Vec<f32>, Error> {
    use image::imageops::FilterType;

    let img = image::open(path)
        .map_err(|e| Error::parse(e.to_string()))?
        .resize_exact(224, 224, FilterType::CatmullRom)
        .to_rgb8();

    // Convert HWC → CHW and apply CLIP normalization.
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
