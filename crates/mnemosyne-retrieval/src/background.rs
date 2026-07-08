//! Background periodic indexer.
//!
//! Combines a periodic full rescan with real-time `FileWatcher` so the index
//! is always fresh.

use mnemosyne_core::Error;
use std::{path::PathBuf, sync::Arc, time::Duration};
use tokio::task::JoinHandle;
use tracing::{info, warn};

use crate::{engine::SearchEngine, watcher::FileWatcher};

/// Spawns background tasks that keep the index up to date.
///
/// # Behaviour
/// 1. Immediately performs a full scan of all `directories`.
/// 2. Starts a `FileWatcher` on each directory for real-time incremental updates.
/// 3. Repeats the full scan every `rescan_interval`.
pub struct BackgroundIndexer {
    engine: Arc<SearchEngine>,
    directories: Vec<PathBuf>,
    rescan_interval: Duration,
}

impl BackgroundIndexer {
    pub fn new(
        engine: Arc<SearchEngine>,
        directories: Vec<PathBuf>,
        rescan_interval: Duration,
    ) -> Self {
        Self { engine, directories, rescan_interval }
    }

    /// Start the background indexer.
    ///
    /// Returns the join handle and a list of `FileWatcher` handles.
    /// Drop the `FileWatcher`s to stop real-time watching.
    /// Call `.abort()` on the join handle to stop periodic scanning.
    pub async fn start(self) -> Result<(JoinHandle<()>, Vec<FileWatcher>), Error> {
        // Start real-time watchers first.
        let mut watchers = Vec::new();
        for dir in &self.directories {
            match FileWatcher::watch(dir, Arc::clone(&self.engine)).await {
                Ok(w) => watchers.push(w),
                Err(e) => warn!("Could not watch {}: {e}", dir.display()),
            }
        }

        let engine = self.engine;
        let dirs = self.directories;
        let interval = self.rescan_interval;

        let handle = tokio::spawn(async move {
            loop {
                for dir in &dirs {
                    info!("Background rescan: {}", dir.display());
                    match engine.index_directory(dir).await {
                        Ok(stats) => info!(
                            "Rescan done: {} new files in {}",
                            stats.total_files,
                            dir.display()
                        ),
                        Err(e) => warn!("Rescan failed for {}: {e}", dir.display()),
                    }
                }
                tokio::time::sleep(interval).await;
            }
        });

        Ok((handle, watchers))
    }
}
