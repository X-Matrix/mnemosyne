use crate::{ApiError, AppState};
use axum::{
    extract::{Path, Query, State},
    Json,
};
use serde::Deserialize;
use std::sync::Arc;

#[derive(Deserialize)]
pub struct ListParams {
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    offset: usize,
}

fn default_limit() -> usize {
    50
}

/// `GET /api/files?limit=50&offset=0`
pub async fn list_files(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ListParams>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let files = state.engine.list_files(params.limit, params.offset).await?;
    Ok(Json(serde_json::to_value(files)?))
}

/// `DELETE /api/files/:id`
pub async fn remove_file(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    state.engine.remove_file(&id).await?;
    Ok(Json(serde_json::json!({ "deleted": id })))
}
