//! File-system watcher for automatic incremental indexing.
//!
//! Uses `notify-debouncer-mini` with a Tokio channel bridge so event
//! processing runs inside the Tokio runtime.

use mnemosyne_core::Error;
use notify::RecursiveMode;
use notify_debouncer_mini::{new_debouncer, DebounceEventResult};
use std::{path::Path, sync::Arc, time::Duration};
use tracing::{debug, info, warn};

use crate::engine::SearchEngine;

/// A running file-system watcher. Drop to stop watching.
pub struct FileWatcher {
    _debouncer: notify_debouncer_mini::Debouncer<notify::RecommendedWatcher>,
}

impl FileWatcher {
    /// Start watching `dir` and automatically re-index changed files.
    /// Must be called within a Tokio runtime.
    pub async fn watch(dir: impl AsRef<Path>, engine: Arc<SearchEngine>) -> Result<Self, Error> {
        let dir = dir.as_ref().to_path_buf();
        info!("Starting file watcher on: {}", dir.display());

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<DebounceEventResult>();

        let mut debouncer = new_debouncer(Duration::from_millis(500), move |result| {
            let _ = tx.send(result);
        })
        .map_err(|e| Error::Other(anyhow::anyhow!("watcher init: {e}")))?;

        debouncer
            .watcher()
            .watch(&dir, RecursiveMode::Recursive)
            .map_err(|e| Error::Other(anyhow::anyhow!("watch start: {e}")))?;

        tokio::spawn(async move {
            while let Some(result) = rx.recv().await {
                let events = match result {
                    Ok(evts) => evts,
                    Err(errs) => {
                        warn!("Watcher error: {errs}");
                        continue;
                    }
                };
                for event in events {
                    let path = event.path.clone();
                    if !is_indexable(&path) {
                        continue;
                    }
                    debug!("File changed: {}", path.display());
                    match engine.index_file(&path).await {
                        Ok(true) => info!("Re-indexed: {}", path.display()),
                        Ok(false) => debug!("Unchanged: {}", path.display()),
                        Err(e) => warn!("Re-index failed for {}: {e}", path.display()),
                    }
                }
            }
        });

        Ok(Self {
            _debouncer: debouncer,
        })
    }
}

fn is_indexable(path: &Path) -> bool {
    if path
        .components()
        .any(|c| c.as_os_str().to_str().is_some_and(|s| s.starts_with('.')))
    {
        return false;
    }
    if !path.is_file() {
        return false;
    }
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    matches!(
        ext.as_str(),
        "txt"
            | "md"
            | "markdown"
            | "csv"
            | "json"
            | "toml"
            | "yaml"
            | "yml"
            | "log"
            | "rs"
            | "py"
            | "js"
            | "ts"
            | "go"
            | "sql"
            | "html"
            | "css"
            | "jpg"
            | "jpeg"
            | "png"
            | "bmp"
            | "gif"
            | "webp"
            | "mp3"
            | "wav"
            | "flac"
            | "mp4"
            | "avi"
            | "mov"
    )
}
