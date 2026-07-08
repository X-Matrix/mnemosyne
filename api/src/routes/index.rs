use crate::{ApiError, AppState};
use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Deserialize)]
pub struct IndexRequest {
    pub path: String,
}

/// `POST /api/index`
///
/// Body: `{ "path": "/absolute/path/to/directory" }`
pub async fn index_directory(
    State(state): State<Arc<AppState>>,
    Json(req): Json<IndexRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let stats = state.engine.index_directory(&req.path).await?;
    Ok(Json(serde_json::to_value(stats)?))
}

/// `GET /api/stats`
pub async fn stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let stats = state.engine.get_stats().await?;
    Ok(Json(serde_json::to_value(stats)?))
}
