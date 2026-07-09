use crate::state::AppState;
use mnemosyne_storage::model_repo::ModelRecord;
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
pub async fn download_model(
    state: State<'_, AppState>,
    model_id: String,
) -> Result<(), CommandError> {
    let lock = state.engine.read().await;
    let engine = lock
        .as_ref()
        .ok_or_else(|| CommandError { message: "engine not ready".into() })?;
    engine.download_model(&model_id).await.map_err(Into::into)
}

#[tauri::command]
pub async fn list_models(
    state: State<'_, AppState>,
) -> Result<Vec<ModelRecord>, CommandError> {
    let lock = state.engine.read().await;
    let engine = lock
        .as_ref()
        .ok_or_else(|| CommandError { message: "engine not ready".into() })?;
    engine.list_models().await.map_err(Into::into)
}
