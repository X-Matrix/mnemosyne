use mnemosyne_retrieval::{watcher::FileWatcher, SearchEngine};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

/// Shared application state managed by Tauri.
pub struct AppState {
    pub engine: Arc<RwLock<Option<SearchEngine>>>,
    /// Active file watchers (indexed by directory path string).
    pub watchers: Arc<Mutex<Vec<FileWatcher>>>,
}

impl AppState {
    pub fn new(engine: SearchEngine) -> Self {
        Self {
            engine: Arc::new(RwLock::new(Some(engine))),
            watchers: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn empty() -> Self {
        Self {
            engine: Arc::new(RwLock::new(None)),
            watchers: Arc::new(Mutex::new(Vec::new())),
        }
    }
}
