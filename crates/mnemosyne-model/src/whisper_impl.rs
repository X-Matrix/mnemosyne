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
const SAMPLE_RATE:         u32   = 16_000;
/// Whisper processes audio in 30-second windows.
/// Matches the encoder's positional embedding capacity: N_FRAMES/2 = 1500 positions.
const N_FRAMES:             usize = 3000;   // 30 s × 100 fps
/// Max new tokens to generate per 30-second chunk.
/// Must be ≤ max_target_positions(448) − 4 initial special tokens = 444.
const MAX_DECODE_TOKENS:    usize = 400;
const N_FFT:               usize = 400;

pub struct WhisperTranscriber {
    // Wrapped in Mutex because encoder/decoder `forward` take `&mut self`.
    model:     Mutex<whisper_model::Whisper>,
    config:    Config,
    tokenizer: Tokenizer,
    device:    Device,
}

/// Return the local model directory `~/.mnemosyne/models/{model_id}` if it exists.
fn local_model_dir(model_id: &str) -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let dir = std::path::PathBuf::from(home)
        .join(".mnemosyne/models")
        .join(model_id);
    if dir.is_dir() { Some(dir) } else { None }
}

impl WhisperTranscriber {
    pub async fn load(model_id: &str) -> Result<Self, Error> {
        info!("Loading Whisper model '{}'", model_id);
        let device = Device::Cpu;

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
                    info!("Whisper: loading from local cache {}", dir.display());
                    (cfg, tok, weights)
                } else {
                    download_whisper_files(model_id).await?
                }
            } else {
                download_whisper_files(model_id).await?
            };

        let config: Config = {
            let s = std::fs::read_to_string(&config_path).map_err(Error::Io)?;
            serde_json::from_str(&s).map_err(|e| Error::model(e.to_string()))?
        };

        let vb = load_var_builder(&weights_path, candle_core::DType::F32, &device)?;
        let model = whisper_model::Whisper::load(&vb, config.clone())
            .map_err(|e| Error::model(e.to_string()))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| Error::model(e.to_string()))?;

        info!("Whisper '{}' ready (device=CPU)", model_id);
        Ok(Self { model: Mutex::new(model), config, tokenizer, device })
    }

    /// Transcribe an audio file to text by processing it in 30-second chunks.
    ///
    /// Each chunk is encoded independently with Whisper's audio encoder,
    /// then decoded using KV-cache-aware greedy search (O(n) vs the naive
    /// O(n²) approach of re-encoding all tokens on each step).
    /// The per-chunk and full transcripts are written to the INFO log.
    pub fn transcribe(&self, path: &Path) -> Result<String, Error> {
        let pcm = decode_audio_mono_16k(path)?;
        if pcm.is_empty() {
            return Ok(String::new());
        }

        let duration_s = pcm.len() as f64 / SAMPLE_RATE as f64;
        // Number of raw PCM samples that correspond to one 30-second Whisper window.
        // HOP_LENGTH = 160, so N_FRAMES × 160 = 3000 × 160 = 480_000 samples.
        const CHUNK_SAMPLES: usize = N_FRAMES * 160;
        let n_chunks = (pcm.len() + CHUNK_SAMPLES - 1) / CHUNK_SAMPLES;

        info!(
            "Whisper: '{}' {:.1} s → {} chunk(s)",
            path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
            duration_s,
            n_chunks,
        );

        let mel_filters = compute_mel_filters_f32(self.config.num_mel_bins);
        let n_mels = self.config.num_mel_bins;
        // Safety cap: total token sequence must not exceed decoder positional
        // embedding size (max_target_positions, 448 for base/tiny).
        let max_new = (self.config.max_target_positions.saturating_sub(4))
            .min(MAX_DECODE_TOKENS);

        let mut model = self.model.lock().map_err(|e| Error::model(e.to_string()))?;
        let mut parts: Vec<String> = Vec::new();

        for (ci, pcm_chunk) in pcm.chunks(CHUNK_SAMPLES).enumerate() {
            // ── Build mel tensor (trim candle's internal padding to N_FRAMES) ──
            let mel = audio::pcm_to_mel(&self.config, pcm_chunk, &mel_filters);
            let actual_frames = mel.len() / n_mels;
            let keep_frames   = actual_frames.min(N_FRAMES);

            let mel_trimmed: Vec<f32> = (0..n_mels)
                .flat_map(|m| {
                    mel[m * actual_frames..m * actual_frames + keep_frames]
                        .iter().copied()
                })
                .collect();

            let mel_tensor =
                Tensor::from_vec(mel_trimmed, (1usize, n_mels, keep_frames), &self.device)
                    .map_err(|e| Error::model(e.to_string()))?;

            let mel_tensor = if keep_frames < N_FRAMES {
                let pad = Tensor::zeros(
                    (1, n_mels, N_FRAMES - keep_frames),
                    candle_core::DType::F32,
                    &self.device,
                ).map_err(|e| Error::model(e.to_string()))?;
                Tensor::cat(&[&mel_tensor, &pad], 2)
                    .map_err(|e| Error::model(e.to_string()))?
            } else {
                mel_tensor
            };

            // ── Encode audio ──
            let audio_features = model.encoder.forward(&mel_tensor, true)
                .map_err(|e| Error::model(e.to_string()))?;

            // ── KV-cache greedy decode ──────────────────────────────────
            // Pass all initial tokens in one forward call (flush_kv_cache=true),
            // then feed only the latest token each step (flush_kv_cache=false).
            // Reduces decoder work from O(n²) to O(n).
            let init: &[u32] =
                &[SOT_TOKEN, ENGLISH_TOKEN, TRANSCRIBE_TOKEN, NO_TIMESTAMPS_TOKEN];
            let init_t = Tensor::new(init, &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| Error::model(e.to_string()))?;

            let logits0 = model.decoder.forward(&init_t, &audio_features, true)
                .map_err(|e| Error::model(e.to_string()))?;
            let mut last_tok = next_token_from_logits(&logits0, init.len() - 1)?;

            let mut text_tokens: Vec<u32> = Vec::new();
            for _ in 0..max_new {
                if last_tok == EOT_TOKEN { break; }
                text_tokens.push(last_tok);

                let step_t = Tensor::new(&[last_tok][..], &self.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| Error::model(e.to_string()))?;
                let logits = model.decoder.forward(&step_t, &audio_features, false)
                    .map_err(|e| Error::model(e.to_string()))?;
                last_tok = next_token_from_logits(&logits, 0)?;
            }

            let chunk_text = self.tokenizer.decode(&text_tokens, true)
                .map_err(|e| Error::model(e.to_string()))?;
            let chunk_text = chunk_text.trim().to_string();

            info!("Whisper chunk {}/{}: \"{}\"", ci + 1, n_chunks, chunk_text);
            if !chunk_text.is_empty() {
                parts.push(chunk_text);
            }
        }

        let transcript = parts.join(" ");
        info!(
            "Whisper '{}' transcript: {}",
            path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
            transcript,
        );
        Ok(transcript)
    }
}

unsafe impl Send for WhisperTranscriber {}
unsafe impl Sync for WhisperTranscriber {}

// ── Decode helpers ──────────────────────────────────────────────────

/// Extract the greedy next-token prediction from logits at sequence position `pos`.
///
/// `logits`: shape `(1, seq_len, vocab_size)`
/// `pos`:    0-indexed position in the sequence (usually `seq_len - 1` or `0`)
fn next_token_from_logits(logits: &Tensor, pos: usize) -> Result<u32, Error> {
    logits
        .narrow(1, pos, 1)                 // (1, 1, vocab)
        .and_then(|t| t.squeeze(1))        // (1, vocab)
        .and_then(|t| t.argmax(D::Minus1)) // (1,)
        .and_then(|t| t.squeeze(0))        // ()
        .and_then(|t| t.to_scalar::<u32>())
        .map_err(|e| Error::model(e.to_string()))
}

// ── Weight loading helper ──────────────────────────────────────────────────

fn load_var_builder(
    path: &std::path::Path,
    dtype: candle_core::DType,
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

async fn download_whisper_files(
    model_id: &str,
) -> Result<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf), Error> {
    let api = Api::new().map_err(|e| Error::model(e.to_string()))?;
    let repo = api.model(model_id.to_string());
    let config_path    = repo.get("config.json").await
        .map_err(|e| Error::model(format!("config.json: {e}")))?;
    let tokenizer_path = repo.get("tokenizer.json").await
        .map_err(|e| Error::model(format!("tokenizer.json: {e}")))?;
    let weights_path = match repo.get("model.safetensors").await {
        Ok(p) => p,
        Err(_) => repo.get("pytorch_model.bin").await
            .map_err(|e| Error::model(format!("weights: {e}")))?,
    };
    Ok((config_path, tokenizer_path, weights_path))
}

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


