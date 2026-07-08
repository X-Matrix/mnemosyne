use crate::state::AppState;
use mnemosyne_core::types::{FileRecord, IndexStats};
use serde::Serialize;
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
