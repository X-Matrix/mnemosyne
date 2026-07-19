//! HuggingFace model file downloader.
//!
//! Bypasses `hf-hub`'s API and downloads directly via reqwest so we have
//! full control over URL construction and TLS configuration.
//!
//! Files are cached at `~/.mnemosyne/models/<org>/<name>/` so they are only
//! fetched once.

use mnemosyne_core::Error;
use std::path::{Path, PathBuf};
use tracing::info;

/// Base URL for HuggingFace Hub.
/// Can be overridden by passing an explicit endpoint or via the `HF_ENDPOINT`
/// environment variable (env var takes priority over the built-in default).
fn compute_endpoint(override_url: Option<&str>) -> String {
    if let Some(url) = override_url {
        let url = url.trim();
        if !url.is_empty() {
            return url.trim_end_matches('/').to_string();
        }
    }
    // Check env var; if not set, use direct HuggingFace (no mirror by default).
    std::env::var("HF_ENDPOINT")
        .unwrap_or_else(|_| "https://huggingface.co".to_string())
        .trim_end_matches('/')
        .to_string()
}

/// Files to download for each model type.
const TEXT_MODEL_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
];

const TEXT_MODEL_FILES_FALLBACK: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "pytorch_model.bin",
];

/// Chinese CLIP models (OFA-Sys/chinese-clip-*) use `vocab.txt` instead of
/// `tokenizer.json`, and do not have `tokenizer_config.json`.
/// Available files: config.json, vocab.txt, pytorch_model.bin
const CHINESE_CLIP_FILES: &[&str] = &[
    "config.json",
    "vocab.txt",
    "pytorch_model.bin",
];

/// Returns true for Chinese CLIP model IDs.
fn is_chinese_clip(model_id: &str) -> bool {
    let id = model_id.to_lowercase();
    id.contains("chinese-clip") || id.starts_with("ofa-sys/")
}

/// Downloads model files from HuggingFace Hub into the local cache.
pub struct ModelDownloader {
    cache_dir: PathBuf,
}

impl ModelDownloader {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }

    /// Download all required files for `model_id` and return the local dir.
    ///
    /// - `proxy_url`   – explicit proxy (e.g. `"socks5://127.0.0.1:7890"`).
    ///   `None` / empty → no system proxy.
    /// - `hf_endpoint` – HuggingFace mirror base URL (e.g. `"https://hf-mirror.com"`).
    ///   `None` / empty → use `HF_ENDPOINT` env var or fall back to `huggingface.co`.
    pub async fn download(
        &self,
        model_id: &str,
        proxy_url: Option<&str>,
        hf_endpoint: Option<&str>,
    ) -> Result<PathBuf, Error> {
        info!("Downloading model: {model_id}");

        let model_dir = self
            .cache_dir
            .join(model_id.replace('/', std::path::MAIN_SEPARATOR_STR));
        tokio::fs::create_dir_all(&model_dir)
            .await
            .map_err(Error::Io)?;

        let endpoint = compute_endpoint(hf_endpoint);

        // Build HTTP client: explicit proxy OR no_proxy (bypass misconfigured system proxy).
        let mut builder = reqwest::Client::builder().user_agent("mnemosyne/0.1");
        match proxy_url {
            Some(url) if !url.trim().is_empty() => {
                let proxy = reqwest::Proxy::all(url.trim())
                    .map_err(|e| Error::model(format!("proxy config: {e}")))?;
                builder = builder.proxy(proxy);
                info!("Using proxy: {}", url.trim());
            }
            _ => {
                // Disable system proxy — prevents "Connection refused" when
                // macOS proxy settings point at a local agent that isn't running.
                builder = builder.no_proxy();
            }
        }
        let client = builder
            .build()
            .map_err(|e| Error::model(format!("http client: {e}")))?;

        // Chinese CLIP models use vocab.txt instead of tokenizer.json.
        // All other models try safetensors first, then pytorch bin.
        let result = if is_chinese_clip(model_id) {
            self.download_files(model_id, CHINESE_CLIP_FILES, &model_dir, &endpoint, &client)
                .await
        } else {
            let r = self
                .download_files(model_id, TEXT_MODEL_FILES, &model_dir, &endpoint, &client)
                .await;
            if r.is_err() {
                self.download_files(
                    model_id,
                    TEXT_MODEL_FILES_FALLBACK,
                    &model_dir,
                    &endpoint,
                    &client,
                )
                .await
            } else {
                r
            }
        };

        result?;

        info!("Model '{}' cached at {}", model_id, model_dir.display());
        Ok(model_dir)
    }

    /// Download a list of files; stop on first error.
    async fn download_files(
        &self,
        model_id: &str,
        files: &[&str],
        model_dir: &Path,
        endpoint: &str,
        client: &reqwest::Client,
    ) -> Result<(), Error> {
        for filename in files {
            let dest = model_dir.join(filename);
            if dest.exists() {
                info!("Cached: {filename}");
                continue;
            }

            let url = format!("{endpoint}/{model_id}/resolve/main/{filename}");
            info!("GET {url}");

            let resp = client
                .get(&url)
                .send()
                .await
                .map_err(|e| Error::model(format!("GET {filename}: {e}")))?;

            if !resp.status().is_success() {
                return Err(Error::model(format!("{filename}: HTTP {}", resp.status())));
            }

            let bytes = resp
                .bytes()
                .await
                .map_err(|e| Error::model(format!("read {filename}: {e}")))?;

            tokio::fs::write(&dest, &bytes).await.map_err(Error::Io)?;

            info!("Saved {filename} ({} bytes)", bytes.len());
        }
        Ok(())
    }
}
