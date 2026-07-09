use crate::state::{AppState, IndexProgress};
use mnemosyne_core::types::{FileRecord, IndexStats};
use mnemosyne_retrieval::watcher::FileWatcher;
use serde::Serialize;
use std::sync::Arc;
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

/// Open the native folder picker and return the selected path (or null).
///
/// Uses the async callback-based API so the dialog is shown on the main
/// thread (required on macOS) while the Tauri command awaits on a channel.
#[tauri::command]
pub async fn pick_directory(app: tauri::AppHandle) -> Option<String> {
    use tauri_plugin_dialog::DialogExt;

    tracing::info!("[pick_directory] opening folder picker dialog");

    let (tx, rx) = tokio::sync::oneshot::channel::<Option<String>>();

    app.dialog().file().pick_folder(move |folder| {
        tracing::info!("[pick_directory] callback fired, folder={:?}", folder);
        let path = match folder {
            None => {
                tracing::info!("[pick_directory] user cancelled or dialog returned None");
                None
            }
            Some(fp) => {
                tracing::debug!("[pick_directory] FilePath value: {:?}", fp);
                match fp.into_path() {
                    Ok(p) => {
                        let s = p.to_string_lossy().to_string();
                        tracing::info!("[pick_directory] resolved path: {}", s);
                        Some(s)
                    }
                    Err(e) => {
                        tracing::error!("[pick_directory] into_path() failed: {:?}", e);
                        None
                    }
                }
            }
        };
        if let Err(e) = tx.send(path) {
            tracing::error!("[pick_directory] oneshot send failed: {:?}", e);
        }
    });

    tracing::info!("[pick_directory] awaiting callback…");
    let result = rx.await.ok().flatten();
    tracing::info!("[pick_directory] returning: {:?}", result);
    result
}

/// Start indexing a directory **in the background** and return immediately.
/// Poll `get_indexing_status` to track progress.
#[tauri::command]
pub async fn index_directory_bg(
    state: State<'_, AppState>,
    path: String,
) -> Result<(), CommandError> {
    tracing::info!("[index_directory_bg] called with path={:?}", path);

    // Verify path exists before we even try
    let meta = std::fs::metadata(&path);
    match &meta {
        Ok(m) if m.is_dir() => tracing::info!("[index_directory_bg] path is a valid directory"),
        Ok(m) => tracing::warn!("[index_directory_bg] path exists but is NOT a directory (is_file={})", m.is_file()),
        Err(e) => {
            tracing::error!("[index_directory_bg] path metadata error: {} — {}", path, e);
            return Err(CommandError { message: format!("路径无法访问: {e}") });
        }
    }

    let engine_guard = state.engine.read().await;
    let engine_ready = engine_guard.as_ref().is_some();
    tracing::info!("[index_directory_bg] engine ready={}", engine_ready);
    let _engine = engine_guard
        .as_ref()
        .ok_or_else(|| { tracing::error!("[index_directory_bg] engine not ready!"); CommandError { message: "engine not ready".into() } })?;
    drop(engine_guard);

    // Mark as running
    {
        let mut map = state.indexing.lock().await;
        map.insert(path.clone(), IndexProgress {
            path: path.clone(),
            running: true,
            new_files: 0,
            error: None,
        });
    }

    // Clone needed handles for the spawned task
    let indexing_map = Arc::clone(&state.indexing);
    let engine_ref = Arc::clone(&state.engine);
    let path_clone = path.clone();

    tracing::info!("[index_directory_bg] spawning background task for: {}", path);
    tauri::async_runtime::spawn(async move {
        tracing::info!("[index_bg_task] starting index_directory for: {}", path_clone);
        let result = {
            let guard = engine_ref.read().await;
            if let Some(eng) = guard.as_ref() {
                eng.index_directory(&path_clone).await
            } else {
                tracing::error!("[index_bg_task] engine gone when task executed!");
                Err(mnemosyne_core::Error::storage("engine gone".to_string()))
            }
        };

        let mut map = indexing_map.lock().await;
        match result {
            Ok(stats) => {
                tracing::info!("[index_bg_task] completed: {} files indexed for {}", stats.total_files, path_clone);
                if let Some(p) = map.get_mut(&path_clone) {
                    p.running   = false;
                    p.new_files = stats.total_files;
                }
            }
            Err(e) => {
                tracing::error!("[index_bg_task] FAILED for {}: {}", path_clone, e);
                if let Some(p) = map.get_mut(&path_clone) {
                    p.running = false;
                    p.error   = Some(e.to_string());
                }
            }
        }
    });

    Ok(())
}

/// Get the status of all active/recent indexing jobs.
#[tauri::command]
pub async fn get_indexing_status(
    state: State<'_, AppState>,
) -> Result<Vec<IndexProgress>, CommandError> {
    let map = state.indexing.lock().await;
    Ok(map.values().cloned().collect())
}

/// Synchronous index (kept for backward-compat, blocks until done).
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

/// Start watching a directory for changes and auto-reindex.
#[tauri::command]
pub async fn watch_directory(
    state: State<'_, AppState>,
    path: String,
) -> Result<(), CommandError> {
    let engine_arc = {
        let guard = state.engine.read().await;
        if guard.is_none() {
            return Err(CommandError { message: "engine not ready".into() });
        }
        // Build a fresh engine for the watcher
        mnemosyne_retrieval::SearchEngine::builder()
            .build()
            .await
            .map(Arc::new)
            .map_err(|e| CommandError { message: e.to_string() })?
    };

    let watcher = FileWatcher::watch(&path, engine_arc)
        .await
        .map_err(|e| CommandError { message: e.to_string() })?;

    state.watchers.lock().await.push(watcher);
    Ok(())
}

/// Stop all active watchers.
#[tauri::command]
pub async fn stop_watching(state: State<'_, AppState>) -> Result<(), CommandError> {
    state.watchers.lock().await.clear();
    Ok(())
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


