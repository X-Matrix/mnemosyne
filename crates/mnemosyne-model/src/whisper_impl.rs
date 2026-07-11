//! Whisper speech-to-text via HuggingFace Candle.
//!
//! Compiled only with the `whisper-backend` feature.
//! Accepts any audio format supported by symphonia (MP3, WAV, FLAC, OGG, AAC/M4A).
//! Audio is decoded and resampled to 16 kHz mono before inference.
//! Default model: `openai/whisper-tiny`.

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{audio, model as whisper_model, Config};
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
/// Whisper-base generates ~4 tokens/second; 80 tokens covers ~20 s of speech.
/// Kept deliberately small to bound O(n²) self-attention decode cost on CPU.
const MAX_DECODE_TOKENS:    usize = 80;
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

    /// Transcribe an audio file to text.
    ///
    /// Implementation mirrors the official candle Whisper example:
    ///  1. Pre-computed Slaney mel filters loaded from embedded bytes.
    ///  2. Mel computed once for the FULL signal, then sliced into 30-s windows
    ///     with `narrow()` to avoid chunk-boundary artefacts.
    ///  3. Greedy decode: `decoder.forward()` returns hidden states;
    ///     `decoder.final_linear()` projects them to vocab logits.
    ///  4. KV-cache flush only on the FIRST decode step per segment
    ///     (`flush_kv_cache = step == 0`).
    pub fn transcribe(&self, path: &Path) -> Result<String, Error> {
        let pcm = decode_audio_mono_16k(path)?;
        if pcm.is_empty() { return Ok(String::new()); }

        let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        let duration_s = pcm.len() as f64 / SAMPLE_RATE as f64;

        // Slaney-normalised mel filter bank (pre-computed, embedded at compile time).
        let mel_filters = load_mel_filters(self.config.num_mel_bins)?;

        // Compute mel for the FULL signal at once, then slice with narrow().
        let mel_vec = audio::pcm_to_mel(&self.config, &pcm, &mel_filters);
        let total_frames = mel_vec.len() / self.config.num_mel_bins;
        let n_mels = self.config.num_mel_bins;

        let mel = Tensor::from_vec(
            mel_vec, (1usize, n_mels, total_frames), &self.device,
        ).map_err(|e| Error::model(e.to_string()))?;

        // pcm_to_mel adds internal padding, so total_frames ≥ duration_s×100 fps.
        let n_segs = (total_frames + N_FRAMES - 1) / N_FRAMES;
        info!("Whisper: '{}' {:.1}s → {} segment(s)", filename, duration_s, n_segs);

        let max_new = (self.config.max_target_positions.saturating_sub(4))
            .min(MAX_DECODE_TOKENS);

        let mut model = self.model.lock().map_err(|e| Error::model(e.to_string()))?;
        let mut parts: Vec<String> = Vec::new();

        let mut seek = 0usize;
        let mut seg_idx = 0usize;
        while seek < total_frames {
            let seg_len = N_FRAMES.min(total_frames - seek);
            seg_idx += 1;

            // Slice and pad to N_FRAMES.
            let mel_seg = mel.narrow(2, seek, seg_len)
                .map_err(|e| Error::model(e.to_string()))?;
            let mel_seg = if seg_len < N_FRAMES {
                let pad = Tensor::zeros(
                    (1, n_mels, N_FRAMES - seg_len),
                    candle_core::DType::F32, &self.device,
                ).map_err(|e| Error::model(e.to_string()))?;
                Tensor::cat(&[&mel_seg, &pad], 2)
                    .map_err(|e| Error::model(e.to_string()))?
            } else { mel_seg };

            seek += seg_len;

            // Encode audio segment.
            let audio_features = model.encoder.forward(&mel_seg, true)
                .map_err(|e| Error::model(e.to_string()))?;

            // Greedy decode.
            // Per reference: flush_kv_cache = (step == 0) — cache is primed on the
            // first step and extended for subsequent steps within the same segment.
            let mut tokens: Vec<u32> = vec![
                SOT_TOKEN, ENGLISH_TOKEN, TRANSCRIBE_TOKEN, NO_TIMESTAMPS_TOKEN,
            ];

            'decode: for step in 0..max_new {
                let tokens_t = Tensor::new(tokens.as_slice(), &self.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| Error::model(e.to_string()))?;

                // forward() returns hidden states, NOT logits.
                let ys = model.decoder.forward(&tokens_t, &audio_features, step == 0)
                    .map_err(|e| Error::model(e.to_string()))?;

                // Project last hidden state to vocab logits via final_linear().
                let seq_len = ys.dim(1).map_err(|e| Error::model(e.to_string()))?;
                let hidden = ys.narrow(1, seq_len - 1, 1)  // (1, 1, d_model)
                    .map_err(|e| Error::model(e.to_string()))?;
                let logits = model.decoder.final_linear(&hidden)  // (1, 1, vocab)
                    .map_err(|e| Error::model(e.to_string()))?;

                // Greedy argmax.
                let v: Vec<f32> = logits.flatten_all()
                    .and_then(|t| t.to_vec1::<f32>())
                    .map_err(|e| Error::model(e.to_string()))?;
                let next_token = v.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(EOT_TOKEN);

                if next_token == EOT_TOKEN
                    || tokens.len() >= self.config.max_target_positions
                {
                    break 'decode;
                }
                tokens.push(next_token);

                // Repetition guard: same bigram 4+ times in last 16 tokens.
                if tokens.len() >= 16 {
                    let tail = &tokens[tokens.len() - 16..];
                    let last = (tail[tail.len()-2], tail[tail.len()-1]);
                    if tail.windows(2).filter(|w| (w[0], w[1]) == last).count() >= 4 {
                        break 'decode;
                    }
                }
            }

            let seg_text = self.tokenizer.decode(&tokens[4..], true)  // skip special tokens
                .map_err(|e| Error::model(e.to_string()))?;
            let seg_text = seg_text.trim().to_string();

            info!("Whisper segment {}/{}: \"{}\"", seg_idx, n_segs, seg_text);
            if !seg_text.is_empty() { parts.push(seg_text); }
        }

        let transcript = parts.join(" ");
        info!("Whisper '{}': {}", filename, transcript);
        Ok(transcript)
    }
}

unsafe impl Send for WhisperTranscriber {}
unsafe impl Sync for WhisperTranscriber {}

// ── Mel filter bank ──────────────────────────────────────────────────

/// Load the pre-computed Slaney-normalised mel filter bank from embedded bytes.
///
/// Layout: little-endian f32 values, row-major (n_mels × N_FFT/2+1).
/// Same binary files used by the official candle Whisper example.
fn load_mel_filters(n_mels: usize) -> Result<Vec<f32>, Error> {
    let bytes: &[u8] = match n_mels {
        80  => include_bytes!("melfilters.bytes"),
        128 => include_bytes!("melfilters128.bytes"),
        _   => return Err(Error::model(format!("no mel filter bank for n_mels={n_mels}"))),
    };
    Ok(bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
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



