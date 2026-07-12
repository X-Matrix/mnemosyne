mod error;
mod routes;
mod state;

pub use error::ApiError;
pub use state::AppState;

use axum::{
    routing::{delete, get, post},
    Router,
};
use std::{net::SocketAddr, sync::Arc};
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::info;
use tracing_subscriber::EnvFilter;

/// Start the HTTP API server.
///
/// Reads `MNEMOSYNE_DB` (database path) and `MNEMOSYNE_PORT` (default 8080)
/// from environment variables.
pub async fn run() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,mnemosyne=debug")),
        )
        .init();

    let db_path = std::env::var("MNEMOSYNE_DB").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        format!("{home}/.mnemosyne/db.sqlite")
    });

    let port: u16 = std::env::var("MNEMOSYNE_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    info!("Initialising SearchEngine (db: {db_path})");
    let engine = mnemosyne_retrieval::SearchEngine::builder()
        .db_path(&db_path)
        .build()
        .await?;

    let state = Arc::new(AppState::new(engine));

    let app = Router::new()
        // Search
        .route("/api/search", post(routes::search::search))
        // Indexing
        .route("/api/index", post(routes::index::index_directory))
        .route("/api/stats", get(routes::index::stats))
        // Files
        .route("/api/files", get(routes::files::list_files))
        .route("/api/files/{id}", delete(routes::files::remove_file))
        // Models
        .route("/api/models", get(routes::models::list_models))
        .route("/api/models/download", post(routes::models::download_model))
        // Health
        .route("/health", get(|| async { "ok" }))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("Mnemosyne API listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
