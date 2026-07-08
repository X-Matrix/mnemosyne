//! Whisper speech-to-text via HuggingFace Candle.
//!
//! Compiled only with the `whisper-backend` feature.
//! Requires 16 kHz mono WAV input (other formats auto-resampled).
//! Default model: `openai/whisper-tiny`.

use candle_core::{Device, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, audio, Config};
use hf_hub::api::tokio::Api;
use hound::WavReader;
use mnemosyne_core::Error;
use std::path::Path;
use tokenizers::Tokenizer;
use tracing::info;

const SOT_TOKEN:           u32 = 50258;
const EOT_TOKEN:           u32 = 50257;
const TRANSCRIBE_TOKEN:    u32 = 50359;
const NO_TIMESTAMPS_TOKEN: u32 = 50363;
const ENGLISH_TOKEN:       u32 = 50259;
const SAMPLE_RATE:         u32 = 16_000;
const MAX_DECODE_TOKENS:   usize = 448;
const N_FFT:               usize = 400;

pub struct WhisperTranscriber {
    model:     m::Whisper,
    config:    Config,
    tokenizer: Tokenizer,
    device:    Device,
}

impl WhisperTranscriber {
    pub async fn load(model_id: &str) -> Result<Self, Error> {
        info!("Loading Whisper model '{}'", model_id);
        let device = Device::Cpu;

        let api = Api::new().map_err(|e| Error::model(e.to_string()))?;
        let repo = api.model(model_id.to_string());

        let config_path = repo.get("config.json").await
            .map_err(|e| Error::model(format!("config.json: {e}")))?;
        let tokenizer_path = repo.get("tokenizer.json").await
            .map_err(|e| Error::model(format!("tokenizer.json: {e}")))?;
        let weights_path = repo.get("model.safetensors").await
            .or_else(|_| async { repo.get("pytorch_model.bin").await })
            .await
            .map_err(|e| Error::model(format!("weights: {e}")))?;

        let config: Config = {
            let s = std::fs::read_to_string(&config_path).map_err(Error::Io)?;
            serde_json::from_str(&s).map_err(|e| Error::model(e.to_string()))?
        };

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[weights_path],
                candle_core::DType::F32,
                &device,
            )
            .map_err(|e| Error::model(e.to_string()))?
        };
        let model = m::Whisper::load(&vb, config.clone())
            .map_err(|e| Error::model(e.to_string()))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| Error::model(e.to_string()))?;

        info!("Whisper '{}' ready (device=CPU)", model_id);
        Ok(Self { model, config, tokenizer, device })
    }

    /// Transcribe a WAV file to a text string.
    pub fn transcribe(&self, path: &Path) -> Result<String, Error> {
        let pcm = load_wav_mono_16k(path)?;
        let mel_filters = compute_mel_filter_bytes(self.config.num_mel_bins);
        let mel = audio::pcm_to_mel(&self.config, &pcm, &mel_filters)
            .map_err(|e| Error::model(e.to_string()))?;

        let mel_len = mel.len();
        let n_mels = self.config.num_mel_bins;
        let mel_tensor = Tensor::from_vec(mel, (1usize, n_mels, mel_len / n_mels), &self.device)
            .map_err(|e| Error::model(e.to_string()))?;

        let audio_features = self.model.encoder.forward(&mel_tensor, true)
            .map_err(|e| Error::model(e.to_string()))?;

        // Greedy decode
        let mut tokens = vec![SOT_TOKEN, ENGLISH_TOKEN, TRANSCRIBE_TOKEN, NO_TIMESTAMPS_TOKEN];
        let mut text_tokens: Vec<u32> = Vec::new();

        for _ in 0..MAX_DECODE_TOKENS {
            let input = Tensor::new(tokens.as_slice(), &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| Error::model(e.to_string()))?;

            let logits = self.model.decoder.forward(&input, &audio_features, true)
                .map_err(|e| Error::model(e.to_string()))?;

            let seq_len = logits.dim(1).map_err(|e| Error::model(e.to_string()))?;
            let next_token = logits
                .narrow(1, seq_len - 1, 1)
                .and_then(|t| t.squeeze(1))
                .and_then(|t| t.argmax(D::Minus1))
                .and_then(|t| t.to_scalar::<u32>())
                .map_err(|e| Error::model(e.to_string()))?;

            if next_token == EOT_TOKEN { break; }
            tokens.push(next_token);
            text_tokens.push(next_token);
        }

        self.tokenizer.decode(&text_tokens, true)
            .map_err(|e| Error::model(e.to_string()))
    }
}

unsafe impl Send for WhisperTranscriber {}
unsafe impl Sync for WhisperTranscriber {}

// ── Mel filter bank (computed at runtime, librosa-compatible) ─────────────────

fn hz_to_mel(hz: f32) -> f32 { 2595.0 * (1.0 + hz / 700.0).log10() }
fn mel_to_hz(mel: f32) -> f32 { 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0) }

/// Compute mel filter bank as raw f32-LE bytes.
fn compute_mel_filter_bytes(n_mels: usize) -> Vec<u8> {
    let sr = SAMPLE_RATE as f32;
    let n_fft = N_FFT;
    let n_freqs = n_fft / 2 + 1;
    let fmin = 0.0f32;
    let fmax = sr / 2.0;

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // n_mels+2 equally-spaced mel-scale centre frequencies
    let mel_pts: Vec<f32> = (0..=(n_mels + 1))
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();
    let hz_pts: Vec<f32> = mel_pts.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_pts: Vec<usize> = hz_pts.iter()
        .map(|&f| ((f / sr) * (n_fft + 2) as f32).floor() as usize)
        .collect();

    let mut filters = vec![0.0f32; n_mels * n_freqs];
    for m in 0..n_mels {
        let lo  = bin_pts[m];
        let mid = bin_pts[m + 1];
        let hi  = bin_pts[m + 2];
        for f in lo..hi.min(n_freqs) {
            filters[m * n_freqs + f] = if f <= mid {
                (f - lo) as f32 / (mid - lo).max(1) as f32
            } else {
                (hi - f) as f32 / (hi - mid).max(1) as f32
            };
        }
    }

    filters.iter().flat_map(|v| v.to_le_bytes()).collect()
}

// ── WAV loading ───────────────────────────────────────────────────────────────

fn load_wav_mono_16k(path: &Path) -> Result<Vec<f32>, Error> {
    let mut reader = WavReader::open(path).map_err(|e| Error::parse(e.to_string()))?;
    let spec = reader.spec();
    let ch = spec.channels as usize;

    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>()
            .map(|s| s.map_err(|e| Error::parse(e.to_string())))
            .collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1i32 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale)
                         .map_err(|e| Error::parse(e.to_string())))
                .collect::<Result<_, _>>()?
        }
    };

    let mono: Vec<f32> = if ch == 1 { raw }
    else { raw.chunks_exact(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect() };

    if spec.sample_rate == SAMPLE_RATE { return Ok(mono); }

    let ratio = spec.sample_rate as f64 / SAMPLE_RATE as f64;
    let n_out = (mono.len() as f64 / ratio) as usize;
    Ok((0..n_out).map(|i| {
        let src = i as f64 * ratio;
        let lo = src.floor() as usize;
        let hi = (lo + 1).min(mono.len() - 1);
        let t = (src - lo as f64) as f32;
        mono[lo] * (1.0 - t) + mono[hi] * t
    }).collect())
}
