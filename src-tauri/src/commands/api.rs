use crate::state::AppState;
use axum::{
    extract::State as AxumState,
    response::Html,
    routing::{delete, get, post},
    Json, Router,
};
use mnemosyne_core::types::SearchQuery;
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, sync::Arc};
use tauri::State;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;

type EngineRef = Arc<RwLock<Option<mnemosyne_retrieval::SearchEngine>>>;

/// Status returned to the frontend.
#[derive(Debug, Serialize)]
pub struct ApiStatus {
    pub running: bool,
    pub port: Option<u16>,
    pub url: Option<String>,
}

// ── Tauri commands ─────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn start_api_server(state: State<'_, AppState>, port: u16) -> Result<ApiStatus, String> {
    // Abort any existing server
    {
        let mut h = state.api_handle.lock().await;
        if let Some(handle) = h.take() {
            handle.abort();
        }
    }

    let engine = Arc::clone(&state.engine);
    // Use tokio directly to get a JoinHandle we can store
    let handle = tokio::task::spawn(async move {
        if let Err(e) = serve(engine, port).await {
            tracing::error!("REST API server error on :{port}: {e}");
        }
    });

    *state.api_handle.lock().await = Some(handle);
    *state.api_port.lock().await = Some(port);

    tracing::info!("REST API started on http://localhost:{port}");
    Ok(ApiStatus {
        running: true,
        port: Some(port),
        url: Some(format!("http://localhost:{port}")),
    })
}

#[tauri::command]
pub async fn stop_api_server(state: State<'_, AppState>) -> Result<(), String> {
    if let Some(h) = state.api_handle.lock().await.take() {
        h.abort();
    }
    *state.api_port.lock().await = None;
    tracing::info!("REST API stopped");
    Ok(())
}

#[tauri::command]
pub async fn get_api_status(state: State<'_, AppState>) -> Result<ApiStatus, String> {
    let port = *state.api_port.lock().await;
    let running = state.api_handle.lock().await.is_some();
    Ok(ApiStatus {
        running,
        port,
        url: port.map(|p| format!("http://localhost:{p}")),
    })
}

// ── Axum server ────────────────────────────────────────────────────────────────

async fn serve(engine: EngineRef, port: u16) -> anyhow::Result<()> {
    let app = Router::new()
        // Meta
        .route("/health", get(health))
        .route("/api/docs", get(swagger_ui))
        .route("/api/docs/openapi.json", get(openapi_json))
        // Data
        .route("/api/stats", get(api_stats))
        .route("/api/search", post(api_search))
        .route("/api/files", get(api_files))
        .route("/api/files/{id}", delete(api_remove_file)) // fixed: was :id (Axum 0.7 syntax)
        .route("/api/index", post(api_index))
        .layer(CorsLayer::permissive())
        .with_state(engine);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

async fn openapi_json() -> (axum::http::StatusCode, axum::http::HeaderMap, String) {
    let spec = r#"{
  "openapi": "3.0.3",
  "info": { "title": "Mnemosyne API", "version": "0.1.0",
             "description": "Local file semantic search REST API" },
  "paths": {
    "/health": {
      "get": { "summary": "Health check", "tags": ["meta"],
        "responses": { "200": { "description": "Service is up" } } }
    },
    "/api/stats": {
      "get": { "summary": "Index statistics", "tags": ["index"],
        "responses": { "200": { "description": "IndexStats" } } }
    },
    "/api/search": {
      "post": { "summary": "Semantic / keyword search", "tags": ["search"],
        "requestBody": { "required": true,
          "content": { "application/json": { "schema": {
            "type": "object",
            "properties": {
              "text":           { "type": "string", "example": "机器学习" },
              "mode":           { "type": "string", "enum": ["Hybrid","Vector","Keyword"], "default": "Hybrid" },
              "limit":          { "type": "integer", "default": 20 },
              "vector_weight":  { "type": "number", "default": 0.7 },
              "keyword_weight": { "type": "number", "default": 0.3 }
            }, "required": ["text"] } } } },
        "responses": { "200": { "description": "Array of SearchResult" } } }
    },
    "/api/files": {
      "get": { "summary": "List indexed files", "tags": ["files"],
        "parameters": [
          { "name": "limit", "in": "query", "schema": { "type": "integer", "default": 200 } },
          { "name": "offset", "in": "query", "schema": { "type": "integer", "default": 0 } }
        ],
        "responses": { "200": { "description": "Array of FileRecord" } } }
    },
    "/api/files/{id}": {
      "delete": { "summary": "Remove file from index", "tags": ["files"],
        "parameters": [ { "name": "id", "in": "path", "required": true,
                          "schema": { "type": "string" } } ],
        "responses": { "200": { "description": "Deleted" } } }
    },
    "/api/index": {
      "post": { "summary": "Index a directory", "tags": ["index"],
        "requestBody": { "required": true,
          "content": { "application/json": { "schema": {
            "type": "object", "properties": { "path": { "type": "string" } },
            "required": ["path"] } } } },
        "responses": { "200": { "description": "IndexStats" } } }
    }
  }
}"#;
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(
        axum::http::header::CONTENT_TYPE,
        "application/json".parse().unwrap(),
    );
    (axum::http::StatusCode::OK, headers, spec.to_string())
}

async fn swagger_ui() -> axum::response::Html<String> {
    axum::response::Html(
        r#"<!DOCTYPE html>
<html>
<head><title>Mnemosyne API Docs</title>
  <meta charset="utf-8"/>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist/swagger-ui.css">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist/swagger-ui-bundle.js"></script>
  <script>
    SwaggerUIBundle({ url: '/api/docs/openapi.json', dom_id: '#swagger-ui',
      presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset] });
  </script>
</body></html>"#
            .to_string(),
    )
}

async fn api_stats(AxumState(engine): AxumState<EngineRef>) -> Json<serde_json::Value> {
    let g = engine.read().await;
    match g.as_ref() {
        Some(eng) => match eng.get_stats().await {
            Ok(s) => Json(serde_json::to_value(s).unwrap_or_default()),
            Err(e) => Json(serde_json::json!({"error": e.to_string()})),
        },
        None => Json(serde_json::json!({"error": "engine not ready"})),
    }
}

async fn api_search(
    AxumState(engine): AxumState<EngineRef>,
    Json(query): Json<SearchQuery>,
) -> Json<serde_json::Value> {
    let g = engine.read().await;
    match g.as_ref() {
        Some(eng) => match eng.search(query).await {
            Ok(r) => Json(serde_json::to_value(r).unwrap_or_default()),
            Err(e) => Json(serde_json::json!({"error": e.to_string()})),
        },
        None => Json(serde_json::json!({"error": "engine not ready"})),
    }
}

async fn api_files(AxumState(engine): AxumState<EngineRef>) -> Json<serde_json::Value> {
    let g = engine.read().await;
    match g.as_ref() {
        Some(eng) => match eng.list_files(200, 0).await {
            Ok(f) => Json(serde_json::to_value(f).unwrap_or_default()),
            Err(e) => Json(serde_json::json!({"error": e.to_string()})),
        },
        None => Json(serde_json::json!({"error": "engine not ready"})),
    }
}

async fn api_remove_file(
    AxumState(engine): AxumState<EngineRef>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Json<serde_json::Value> {
    let g = engine.read().await;
    match g.as_ref() {
        Some(eng) => match eng.remove_file(&id).await {
            Ok(_) => Json(serde_json::json!({"ok": true})),
            Err(e) => Json(serde_json::json!({"error": e.to_string()})),
        },
        None => Json(serde_json::json!({"error": "engine not ready"})),
    }
}

#[derive(Deserialize)]
struct IndexReq {
    path: String,
}

async fn api_index(
    AxumState(engine): AxumState<EngineRef>,
    Json(req): Json<IndexReq>,
) -> Json<serde_json::Value> {
    let g = engine.read().await;
    match g.as_ref() {
        Some(eng) => match eng.index_directory(&req.path).await {
            Ok(s) => Json(serde_json::to_value(s).unwrap_or_default()),
            Err(e) => Json(serde_json::json!({"error": e.to_string()})),
        },
        None => Json(serde_json::json!({"error": "engine not ready"})),
    }
}
