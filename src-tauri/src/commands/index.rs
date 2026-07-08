use crate::state::AppState;
use mnemosyne_core::types::{FileRecord, IndexStats};
use mnemosyne_retrieval::watcher::FileWatcher;
use serde::Serialize;
use std::sync::Arc;
use tauri::State;

#[derive(Debug, Serialize)]
pub struct CommandError {
    message: String,
}

impl From<mnemosyne_core::Error> for CommandError {
    fn from(e: mnemosyne_core::Error) -> Self {
        Self { message: e.to_string() }
    }
}

/// Open the native folder picker and return the selected path (or null).
#[tauri::command]
pub fn pick_directory(app: tauri::AppHandle) -> Option<String> {
    use tauri_plugin_dialog::DialogExt;
    app.dialog()
        .file()
        .blocking_pick_folder()
        .and_then(|fp| fp.into_path().ok())
        .map(|p| p.to_string_lossy().to_string())
}

#[tauri::command]
pub async fn index_directory(
    state: State<'_, AppState>,
    path: String,
) -> Result<IndexStats, CommandError> {
    let lock = state.engine.read().await;
    let engine = lock
        .as_ref()
        .ok_or_else(|| CommandError { message: "engine not ready".into() })?;
    engine.index_directory(&path).await.map_err(Into::into)
}

/// Start watching a directory for changes and auto-reindex.
#[tauri::command]
pub async fn watch_directory(
    state: State<'_, AppState>,
    path: String,
) -> Result<(), CommandError> {
    let engine_guard = state.engine.read().await;
    let engine = engine_guard
        .as_ref()
        .ok_or_else(|| CommandError { message: "engine not ready".into() })?;

    // We need Arc<SearchEngine> for the watcher — wrap via a local Arc.
    // NOTE: This creates a short-lived Arc just for the watcher setup.
    // In production, consider storing Arc<SearchEngine> in state directly.
    let engine_arc = Arc::new(
        mnemosyne_retrieval::SearchEngine::builder()
            .build()
            .await
            .map_err(|e| CommandError { message: e.to_string() })?,
    );

    let watcher = FileWatcher::watch(&path, engine_arc)
        .await
        .map_err(|e| CommandError { message: e.to_string() })?;

    state.watchers.lock().await.push(watcher);
    Ok(())
}

/// Stop all active watchers.
#[tauri::command]
pub async fn stop_watching(state: State<'_, AppState>) -> Result<(), CommandError> {
    state.watchers.lock().await.clear();
    Ok(())
}

#[tauri::command]
pub async fn get_stats(
    state: State<'_, AppState>,
) -> Result<IndexStats, CommandError> {
    let lock = state.engine.read().await;
    let engine = lock
        .as_ref()
        .ok_or_else(|| CommandError { message: "engine not ready".into() })?;
    engine.get_stats().await.map_err(Into::into)
}

#[tauri::command]
pub async fn list_files(
    state: State<'_, AppState>,
    limit: Option<usize>,
    offset: Option<usize>,
) -> Result<Vec<FileRecord>, CommandError> {
    let lock = state.engine.read().await;
    let engine = lock
        .as_ref()
        .ok_or_else(|| CommandError { message: "engine not ready".into() })?;
    engine
        .list_files(limit.unwrap_or(50), offset.unwrap_or(0))
        .await
        .map_err(Into::into)
}

#[tauri::command]
pub async fn remove_file(
    state: State<'_, AppState>,
    id: String,
) -> Result<(), CommandError> {
    let lock = state.engine.read().await;
    let engine = lock
        .as_ref()
        .ok_or_else(|| CommandError { message: "engine not ready".into() })?;
    engine.remove_file(&id).await.map_err(Into::into)
}
