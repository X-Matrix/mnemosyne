use mnemosyne_retrieval::{watcher::FileWatcher, SearchEngine};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};
use tokio::sync::{Mutex, RwLock};

/// A single captured log entry.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LogEntry {
    pub ts:      String,
    pub level:   String,
    pub target:  String,
    pub message: String,
}

/// Shared circular log buffer (max 500 entries).
pub type LogBuf = Arc<std::sync::Mutex<VecDeque<LogEntry>>>;

pub fn new_log_buf() -> LogBuf {
    Arc::new(std::sync::Mutex::new(VecDeque::with_capacity(500)))
}

/// Progress of a background indexing run.
#[derive(Debug, Clone, serde::Serialize)]
pub struct IndexProgress {
    pub path:      String,
    pub running:   bool,
    pub new_files: u64,
    pub error:     Option<String>,
}

/// Shared application state managed by Tauri.
pub struct AppState {
    pub engine:     Arc<RwLock<Option<SearchEngine>>>,
    pub watchers:   Arc<Mutex<Vec<FileWatcher>>>,
    pub indexing:   Arc<Mutex<HashMap<String, IndexProgress>>>,
    pub log_buffer: LogBuf,
    pub api_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    pub api_port:   Arc<Mutex<Option<u16>>>,
}

impl AppState {
    pub fn empty_with_log(log_buffer: LogBuf) -> Self {
        Self {
            engine:     Arc::new(RwLock::new(None)),
            watchers:   Arc::new(Mutex::new(Vec::new())),
            indexing:   Arc::new(Mutex::new(HashMap::new())),
            log_buffer,
            api_handle: Arc::new(Mutex::new(None)),
            api_port:   Arc::new(Mutex::new(None)),
        }
    }
}
