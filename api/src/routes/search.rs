use crate::{ApiError, AppState};
use axum::{extract::State, Json};
use mnemosyne_core::types::SearchQuery;
use std::sync::Arc;

/// `POST /api/search`
///
/// Body: `SearchQuery` JSON
pub async fn search(
    State(state): State<Arc<AppState>>,
    Json(query): Json<SearchQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if query.text.trim().is_empty() {
        return Err(ApiError::bad_request("query text must not be empty"));
    }
    let results = state.engine.search(query).await?;
    Ok(Json(serde_json::to_value(results)?))
}
