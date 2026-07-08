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

impl ClipEmbedder {
    /// Download and load the CLIP model from HuggingFace Hub.
    pub async fn load(model_id: &str) -> Result<Self, Error> {
        info!("Loading CLIP model '{}'", model_id);
        let device = Device::Cpu;

        let api = Api::new().map_err(|e| Error::model(e.to_string()))?;
        let repo = api.model(model_id.to_string());

        let config_path = repo.get("config.json").await
            .map_err(|e| Error::model(format!("config.json: {e}")))?;
        let weights_path = repo.get("model.safetensors").await
            .or_else(|_| async { repo.get("pytorch_model.bin").await })
            .await
            .map_err(|e| Error::model(format!("weights: {e}")))?;

        let config: ClipConfig = {
            let s = std::fs::read_to_string(&config_path).map_err(Error::Io)?;
            serde_json::from_str(&s).map_err(|e| Error::model(e.to_string()))?
        };

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .map_err(|e| Error::model(e.to_string()))?
        };
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
