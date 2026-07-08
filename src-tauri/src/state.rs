use mnemosyne_retrieval::SearchEngine;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Shared application state managed by Tauri.
pub struct AppState {
    pub engine: Arc<RwLock<Option<SearchEngine>>>,
}

impl AppState {
    pub fn new(engine: SearchEngine) -> Self {
        Self {
            engine: Arc::new(RwLock::new(Some(engine))),
        }
    }

    pub fn empty() -> Self {
        Self {
            engine: Arc::new(RwLock::new(None)),
        }
    }
}
