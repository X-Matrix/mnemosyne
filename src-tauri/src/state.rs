use mnemosyne_retrieval::{watcher::FileWatcher, SearchEngine};
use std::{
    collections::HashMap,
    sync::Arc,
};
use tokio::sync::{Mutex, RwLock};

/// Progress of a background indexing run.
#[derive(Debug, Clone, serde::Serialize)]
pub struct IndexProgress {
    pub path: String,
    pub running: bool,
    pub new_files: u64,
    pub error: Option<String>,
}

/// Shared application state managed by Tauri.
pub struct AppState {
    pub engine:   Arc<RwLock<Option<SearchEngine>>>,
    pub watchers: Arc<Mutex<Vec<FileWatcher>>>,
    /// Currently active background indexing jobs (path → progress).
    pub indexing: Arc<Mutex<HashMap<String, IndexProgress>>>,
}

impl AppState {
    pub fn new(engine: SearchEngine) -> Self {
        Self {
            engine:   Arc::new(RwLock::new(Some(engine))),
            watchers: Arc::new(Mutex::new(Vec::new())),
            indexing: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn empty() -> Self {
        Self {
            engine:   Arc::new(RwLock::new(None)),
            watchers: Arc::new(Mutex::new(Vec::new())),
            indexing: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}
