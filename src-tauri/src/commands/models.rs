use crate::state::AppState;
use mnemosyne_storage::model_repo::ModelRecord;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
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

// ── Persisted active-models config ────────────────────────────────────────────
// Stored as JSON: ~/.mnemosyne/.active_models.json
// { "text": "...", "vision": "...", "audio": "..." }

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct ActiveModelsConfig {
    pub text:   Option<String>,
    pub vision: Option<String>,
    pub audio:  Option<String>,
}

fn active_models_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".mnemosyne").join(".active_models.json")
}

/// Read persisted active-model preferences.
/// Falls back to legacy single-file for the `text` slot if needed.
pub fn read_persisted_models() -> ActiveModelsConfig {
    if let Ok(raw) = std::fs::read_to_string(active_models_path()) {
        if let Ok(cfg) = serde_json::from_str::<ActiveModelsConfig>(&raw) {
            return cfg;
        }
    }
    // Legacy migration: read old ~/.mnemosyne/.active_model file for text slot
    let legacy = {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        PathBuf::from(home).join(".mnemosyne").join(".active_model")
    };
    let text = std::fs::read_to_string(&legacy)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    ActiveModelsConfig { text, ..Default::default() }
}

fn write_persisted_models(cfg: &ActiveModelsConfig) {
    let p = active_models_path();
    if let Some(dir) = p.parent() { let _ = std::fs::create_dir_all(dir); }
    if let Ok(json) = serde_json::to_string_pretty(cfg) {
        let _ = std::fs::write(&p, json);
    }
}

/// All three currently active model IDs (returned to the frontend).
#[derive(Debug, Serialize)]
pub struct ActiveModels {
    pub text:   String,
    pub vision: String,
    pub audio:  String,
}

// ── Commands ──────────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn download_model(
    state: State<'_, AppState>,
    model_id: String,
    proxy_url: Option<String>,
) -> Result<(), CommandError> {
    let lock = state.engine.read().await;
    let engine = lock
        .as_ref()
        .ok_or_else(|| CommandError { message: "engine not ready".into() })?;
    engine.download_model(&model_id, proxy_url.as_deref()).await.map_err(Into::into)
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

/// Return all three active model IDs from the running engine.
#[tauri::command]
pub async fn get_active_models(state: State<'_, AppState>) -> Result<ActiveModels, CommandError> {
    let lock = state.engine.read().await;
    let engine = lock
        .as_ref()
        .ok_or_else(|| CommandError { message: "engine not ready".into() })?;
    Ok(ActiveModels {
        text:   engine.get_text_model().to_string(),
        vision: engine.get_vision_model().to_string(),
        audio:  engine.get_audio_model().to_string(),
    })
}

/// Switch the active model for a specific category.
///
/// `category` must be one of `"text"`, `"vision"`, or `"audio"`.
/// Only the corresponding engine slot is updated; the other two are untouched.
#[tauri::command]
pub async fn switch_model(
    state: State<'_, AppState>,
    model_id: String,
    category: String,
) -> Result<(), CommandError> {
    {
        let mut lock = state.engine.write().await;
        let engine = lock
            .as_mut()
            .ok_or_else(|| CommandError { message: "engine not ready".into() })?;
        match category.as_str() {
            "text"   => engine.set_text_model(&model_id),
            "vision" => engine.set_vision_model(&model_id),
            "audio"  => engine.set_audio_model(&model_id),
            other    => return Err(CommandError {
                message: format!("unknown model category: '{other}'. Expected text/vision/audio"),
            }),
        }
    }
    // Persist — read current, patch, write back
    let mut cfg = read_persisted_models();
    match category.as_str() {
        "text"   => cfg.text   = Some(model_id.clone()),
        "vision" => cfg.vision = Some(model_id.clone()),
        "audio"  => cfg.audio  = Some(model_id.clone()),
        _ => {}
    }
    write_persisted_models(&cfg);
    tracing::info!("Active {} model switched to: {}", category, model_id);
    Ok(())
}
