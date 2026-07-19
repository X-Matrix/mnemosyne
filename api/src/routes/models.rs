use crate::{ApiError, AppState};
use axum::{extract::State, Json};
use serde::Deserialize;
use std::sync::Arc;

#[derive(Deserialize)]
pub struct DownloadRequest {
    pub model_id: String,
}

/// `GET /api/models`
pub async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let models = state.engine.list_models().await?;
    Ok(Json(serde_json::to_value(models)?))
}

/// `POST /api/models/download`
///
/// Body: `{ "model_id": "sentence-transformers/all-MiniLM-L6-v2" }`
pub async fn download_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DownloadRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    state.engine.download_model(&req.model_id, None, None).await?;
    Ok(Json(serde_json::json!({
        "model_id": req.model_id,
        "status": "downloaded"
    })))
}
