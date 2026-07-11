//! Whisper speech-to-text via HuggingFace Candle.
//!
//! Compiled only with the `whisper-backend` feature.
//! Accepts any audio format supported by symphonia (MP3, WAV, FLAC, OGG, AAC/M4A).
//! Audio is decoded and resampled to 16 kHz mono before inference.
//! Default model: `openai/whisper-tiny`.

use candle_core::{Device, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, audio, model as whisper_model, Config};
use hf_hub::api::tokio::Api;
use mnemosyne_core::Error;
use std::path::Path;
use std::sync::Mutex;
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
    // Wrapped in Mutex because encoder/decoder `forward` take `&mut self`.
    model:     Mutex<whisper_model::Whisper>,
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
        let weights_path = match repo.get("model.safetensors").await {
            Ok(p) => p,
            Err(_) => repo.get("pytorch_model.bin").await
                .map_err(|e| Error::model(format!("weights: {e}")))?,
        };

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
        let model = whisper_model::Whisper::load(&vb, config.clone())
            .map_err(|e| Error::model(e.to_string()))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| Error::model(e.to_string()))?;

        info!("Whisper '{}' ready (device=CPU)", model_id);
        Ok(Self { model: Mutex::new(model), config, tokenizer, device })
    }

    /// Transcribe an audio file (MP3, WAV, FLAC, OGG, M4A, …) to a text string.
    ///
    /// The file is decoded and resampled to 16 kHz mono by symphonia before
    /// being fed to the Whisper encoder.
    pub fn transcribe(&self, path: &Path) -> Result<String, Error> {
        let pcm          = decode_audio_mono_16k(path)?;
        // pcm_to_mel returns Vec<f32> directly (no Result)
        let mel_filters  = compute_mel_filters_f32(self.config.num_mel_bins);
        let mel: Vec<f32> = audio::pcm_to_mel(&self.config, &pcm, &mel_filters);

        let mel_len = mel.len();
        let n_mels  = self.config.num_mel_bins;
        let mel_tensor = Tensor::from_vec(mel, (1usize, n_mels, mel_len / n_mels), &self.device)
            .map_err(|e| Error::model(e.to_string()))?;

        let mut model = self.model.lock().map_err(|e| Error::model(e.to_string()))?;
        let audio_features = model.encoder.forward(&mel_tensor, true)
            .map_err(|e| Error::model(e.to_string()))?;

        // Greedy decode
        let mut tokens = vec![SOT_TOKEN, ENGLISH_TOKEN, TRANSCRIBE_TOKEN, NO_TIMESTAMPS_TOKEN];
        let mut text_tokens: Vec<u32> = Vec::new();

        for _ in 0..MAX_DECODE_TOKENS {
            let input = Tensor::new(tokens.as_slice(), &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| Error::model(e.to_string()))?;

            let logits = model.decoder.forward(&input, &audio_features, true)
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

// ── Audio loading (symphonia) ─────────────────────────────────────────────────

/// Decode any symphonia-supported audio file to a 16 kHz mono f32 PCM vector.
///
/// Supports: MP3, WAV, FLAC, OGG/Vorbis, OGG/Opus, AAC/M4A, and more.
fn decode_audio_mono_16k(path: &Path) -> Result<Vec<f32>, Error> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path).map_err(Error::Io)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| Error::parse(format!("symphonia probe: {e}")))?;

    let mut format = probed.format;
    let track = format.default_track()
        .ok_or_else(|| Error::parse("no audio track found".to_string()))?;

    let track_id    = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
    let channels    = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| Error::parse(format!("symphonia decoder: {e}")))?;

    let mut raw_mono: Vec<f32>              = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) if p.track_id() == track_id => p,
            Ok(_)  => continue,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(_) => break,
        };

        let decoded = match decoder.decode(&packet) {
            Ok(d)  => d,
            Err(_) => continue,
        };

        if sample_buf.is_none() {
            let spec = *decoded.spec();
            sample_buf = Some(SampleBuffer::<f32>::new(decoded.capacity() as u64, spec));
        }

        if let Some(buf) = sample_buf.as_mut() {
            buf.copy_interleaved_ref(decoded);
            let samples = buf.samples();
            if channels == 1 {
                raw_mono.extend_from_slice(samples);
            } else {
                for frame in samples.chunks_exact(channels) {
                    raw_mono.push(frame.iter().sum::<f32>() / channels as f32);
                }
            }
        }
    }

    if raw_mono.is_empty() {
        return Err(Error::parse("audio file produced no samples".to_string()));
    }

    // Resample to 16 kHz using linear interpolation.
    if sample_rate == SAMPLE_RATE {
        return Ok(raw_mono);
    }
    let ratio = sample_rate as f64 / SAMPLE_RATE as f64;
    let n_out  = (raw_mono.len() as f64 / ratio) as usize;
    Ok((0..n_out).map(|i| {
        let src = i as f64 * ratio;
        let lo  = src.floor() as usize;
        let hi  = (lo + 1).min(raw_mono.len() - 1);
        let t   = (src - lo as f64) as f32;
        raw_mono[lo] * (1.0 - t) + raw_mono[hi] * t
    }).collect())
}

// ── Mel filter bank (computed at runtime, librosa-compatible) ─────────────────

fn hz_to_mel(hz: f32) -> f32 { 2595.0 * (1.0 + hz / 700.0).log10() }
fn mel_to_hz(mel: f32) -> f32 { 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0) }

/// Compute mel filter bank as a flat f32 array (n_mels × n_freqs).
/// Same layout expected by `audio::pcm_to_mel`.
fn compute_mel_filters_f32(n_mels: usize) -> Vec<f32> {
    let sr = SAMPLE_RATE as f32;
    let n_fft = N_FFT;
    let n_freqs = n_fft / 2 + 1;
    let fmin = 0.0f32;
    let fmax = sr / 2.0;

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

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
    filters
}


