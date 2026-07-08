use mnemosyne_retrieval::SearchEngine;

/// Shared application state injected into every route handler.
pub struct AppState {
    pub engine: SearchEngine,
}

impl AppState {
    pub fn new(engine: SearchEngine) -> Self {
        Self { engine }
    }
}
