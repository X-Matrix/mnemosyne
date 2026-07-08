use crate::state::AppState;
use mnemosyne_core::types::SearchQuery;
use serde::Serialize;
use tauri::State;

#[derive(Debug, Serialize)]
pub struct SearchError {
    message: String,
}

impl From<mnemosyne_core::Error> for SearchError {
    fn from(e: mnemosyne_core::Error) -> Self {
        Self { message: e.to_string() }
    }
}

#[tauri::command]
pub async fn search_files(
    state: State<'_, AppState>,
    query: SearchQuery,
) -> Result<serde_json::Value, SearchError> {
    let lock = state.engine.read().await;
    let engine = lock
        .as_ref()
        .ok_or_else(|| SearchError { message: "engine not ready".into() })?;

    let results = engine.search(query).await?;
    Ok(serde_json::to_value(results).unwrap_or_default())
}
