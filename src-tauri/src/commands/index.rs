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
        Self {
            message: e.to_string(),
        }
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
        Ok(m) => tracing::warn!(
            "[index_directory_bg] path exists but is NOT a directory (is_file={})",
            m.is_file()
        ),
        Err(e) => {
            tracing::error!("[index_directory_bg] path metadata error: {} — {}", path, e);
            return Err(CommandError {
                message: format!("路径无法访问: {e}"),
            });
        }
    }

    let engine_guard = state.engine.read().await;
    let engine_ready = engine_guard.as_ref().is_some();
    tracing::info!("[index_directory_bg] engine ready={}", engine_ready);
    let _engine = engine_guard.as_ref().ok_or_else(|| {
        tracing::error!("[index_directory_bg] engine not ready!");
        CommandError {
            message: "engine not ready".into(),
        }
    })?;
    drop(engine_guard);

    // Mark as running
    {
        let mut map = state.indexing.lock().await;
        map.insert(
            path.clone(),
            IndexProgress {
                path: path.clone(),
                running: true,
                new_files: 0,
                error: None,
            },
        );
    }

    // Clone needed handles for the spawned task
    let indexing_map = Arc::clone(&state.indexing);
    let engine_ref = Arc::clone(&state.engine);
    let path_clone = path.clone();

    tracing::info!(
        "[index_directory_bg] spawning background task for: {}",
        path
    );
    tauri::async_runtime::spawn(async move {
        tracing::info!(
            "[index_bg_task] starting index_directory for: {}",
            path_clone
        );
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
                tracing::info!(
                    "[index_bg_task] completed: {} files indexed for {}",
                    stats.total_files,
                    path_clone
                );
                if let Some(p) = map.get_mut(&path_clone) {
                    p.running = false;
                    p.new_files = stats.total_files;
                }
            }
            Err(e) => {
                tracing::error!("[index_bg_task] FAILED for {}: {}", path_clone, e);
                if let Some(p) = map.get_mut(&path_clone) {
                    p.running = false;
                    p.error = Some(e.to_string());
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
    let engine = lock.as_ref().ok_or_else(|| CommandError {
        message: "engine not ready".into(),
    })?;
    engine.index_directory(&path).await.map_err(Into::into)
}

/// Start watching a directory for changes and auto-reindex.
#[tauri::command]
pub async fn watch_directory(state: State<'_, AppState>, path: String) -> Result<(), CommandError> {
    let engine_arc = {
        let guard = state.engine.read().await;
        if guard.is_none() {
            return Err(CommandError {
                message: "engine not ready".into(),
            });
        }
        // Build a fresh engine for the watcher
        mnemosyne_retrieval::SearchEngine::builder()
            .build()
            .await
            .map(Arc::new)
            .map_err(|e| CommandError {
                message: e.to_string(),
            })?
    };

    let watcher = FileWatcher::watch(&path, engine_arc)
        .await
        .map_err(|e| CommandError {
            message: e.to_string(),
        })?;

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
pub async fn get_stats(state: State<'_, AppState>) -> Result<IndexStats, CommandError> {
    let lock = state.engine.read().await;
    let engine = lock.as_ref().ok_or_else(|| CommandError {
        message: "engine not ready".into(),
    })?;
    engine.get_stats().await.map_err(Into::into)
}

#[tauri::command]
pub async fn list_files(
    state: State<'_, AppState>,
    limit: Option<usize>,
    offset: Option<usize>,
) -> Result<Vec<FileRecord>, CommandError> {
    let lock = state.engine.read().await;
    let engine = lock.as_ref().ok_or_else(|| CommandError {
        message: "engine not ready".into(),
    })?;
    engine
        .list_files(limit.unwrap_or(50), offset.unwrap_or(0))
        .await
        .map_err(Into::into)
}

#[tauri::command]
pub async fn remove_file(state: State<'_, AppState>, id: String) -> Result<(), CommandError> {
    let lock = state.engine.read().await;
    let engine = lock.as_ref().ok_or_else(|| CommandError {
        message: "engine not ready".into(),
    })?;
    engine.remove_file(&id).await.map_err(Into::into)
}

/// Wipe every indexed record from the database.
///
/// Deletes all rows from `files` (which cascades to `document_chunks`,
/// `embeddings`, and the FTS5 table via triggers), then drops any
/// `embedding_vec_*` virtual tables created by sqlite-vec so that a
/// subsequent re-index starts completely clean.
#[tauri::command]
pub async fn clear_index(state: State<'_, AppState>) -> Result<(), String> {
    let lock = state.engine.read().await;
    let engine = lock
        .as_ref()
        .ok_or_else(|| "engine not ready".to_string())?;

    let conn = engine.db().conn.clone();

    tokio::task::spawn_blocking(move || -> Result<(), String> {
        let conn = conn.lock().map_err(|e| e.to_string())?;

        // Cascade deletes chunks + embeddings; FTS5 triggers clean themselves.
        conn.execute("DELETE FROM files", [])
            .map_err(|e| e.to_string())?;

        // Drop vec0 virtual tables so they are rebuilt cleanly on next search.
        // Table names are always "embedding_vec_{usize}" — no injection risk.
        let tables: Vec<String> = conn
            .prepare(
                "SELECT name FROM sqlite_master \
                 WHERE type='table' AND name LIKE 'embedding_vec_%'",
            )
            .and_then(|mut stmt| {
                stmt.query_map([], |r| r.get::<_, String>(0))
                    .map(|rows| rows.flatten().collect())
            })
            .unwrap_or_default();

        for t in &tables {
            let _ = conn.execute(&format!("DROP TABLE IF EXISTS \"{t}\""), []);
        }

        tracing::info!(
            "Index cleared: files/chunks/embeddings deleted, {} vec0 tables dropped",
            tables.len()
        );
        Ok(())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Return a preview payload for the given file path.
///
/// - Text files  → `{ type: "text",       content, ext }`
/// - Image files → `{ type: "image",      data_url, size }` or `{ type: "image_large", size }`
/// - Others      → `{ type: "binary",     file_type, size, ext }`
#[tauri::command]
pub async fn preview_file(path: String) -> Result<serde_json::Value, CommandError> {
    use mnemosyne_core::types::FileType;
    use std::path::Path;

    let p = Path::new(&path);
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let ft = FileType::from_extension(&ext);

    // ── PDF ───────────────────────────────────────────────────────────────────
    // PDFs are classified as FileType::Text; handle before the match so the
    // text branch doesn't try to decode them as UTF-8.
    if ext == "pdf" {
        let meta = tokio::fs::metadata(p).await.map_err(|e| CommandError {
            message: e.to_string(),
        })?;
        const MAX_PDF: u64 = 20 * 1024 * 1024; // 20 MB
        if meta.len() > MAX_PDF {
            return Ok(serde_json::json!({
                "type": "pdf_large",
                "size": meta.len(),
                "ext":  "pdf"
            }));
        }
        let data = tokio::fs::read(p).await.map_err(|e| CommandError {
            message: e.to_string(),
        })?;
        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&data);
        return Ok(serde_json::json!({
            "type":     "pdf",
            "b64":      b64,
            "size":     data.len(),
            "ext":      "pdf"
        }));
    }

    match ft {
        FileType::Text => {
            let raw = tokio::fs::read(p).await.map_err(|e| CommandError {
                message: e.to_string(),
            })?;
            // Try UTF-8; fall back to lossy
            let content: String = String::from_utf8(raw.clone())
                .unwrap_or_else(|_| String::from_utf8_lossy(&raw).into_owned());
            let preview: String = content.chars().take(12_000).collect();
            Ok(serde_json::json!({ "type": "text", "content": preview, "ext": ext }))
        }

        FileType::Image => {
            let meta = tokio::fs::metadata(p).await.map_err(|e| CommandError {
                message: e.to_string(),
            })?;

            const MAX_INLINE: u64 = 6 * 1024 * 1024; // 6 MB
            if meta.len() > MAX_INLINE {
                return Ok(
                    serde_json::json!({ "type": "image_large", "size": meta.len(), "ext": ext }),
                );
            }

            let data = tokio::fs::read(p).await.map_err(|e| CommandError {
                message: e.to_string(),
            })?;
            let mime = match ext.as_str() {
                "jpg" | "jpeg" => "image/jpeg",
                "png" => "image/png",
                "gif" => "image/gif",
                "webp" => "image/webp",
                "bmp" => "image/bmp",
                "svg" => "image/svg+xml",
                _ => "image/jpeg",
            };
            use base64::Engine as _;
            let b64 = base64::engine::general_purpose::STANDARD.encode(&data);
            Ok(serde_json::json!({
                "type":     "image",
                "data_url": format!("data:{mime};base64,{b64}"),
                "size":     data.len(),
                "ext":      ext
            }))
        }

        _ => {
            // ── Audio ─────────────────────────────────────────────────────────
            if matches!(ft, FileType::Audio) {
                let meta = tokio::fs::metadata(p).await.map_err(|e| CommandError {
                    message: e.to_string(),
                })?;
                const MAX_AUDIO: u64 = 50 * 1024 * 1024; // 50 MB
                if meta.len() > MAX_AUDIO {
                    return Ok(serde_json::json!({
                        "type": "audio_large",
                        "size": meta.len(),
                        "ext":  ext
                    }));
                }
                let data = tokio::fs::read(p).await.map_err(|e| CommandError {
                    message: e.to_string(),
                })?;
                let mime = match ext.as_str() {
                    "mp3" => "audio/mpeg",
                    "wav" => "audio/wav",
                    "flac" => "audio/flac",
                    "ogg" | "oga" => "audio/ogg",
                    "aac" => "audio/aac",
                    "m4a" => "audio/mp4",
                    "opus" => "audio/opus",
                    "wma" => "audio/x-ms-wma",
                    _ => "audio/mpeg",
                };
                use base64::Engine as _;
                let b64 = base64::engine::general_purpose::STANDARD.encode(&data);
                return Ok(serde_json::json!({
                    "type":     "audio",
                    "data_url": format!("data:{mime};base64,{b64}"),
                    "size":     data.len(),
                    "ext":      ext,
                    "mime":     mime
                }));
            }

            let meta = tokio::fs::metadata(p).await.map_err(|e| CommandError {
                message: e.to_string(),
            })?;
            Ok(serde_json::json!({
                "type":      "binary",
                "file_type": format!("{ft:?}"),
                "size":      meta.len(),
                "ext":       ext
            }))
        }
    }
}

// ── HNSW force-enable toggle ──────────────────────────────────────────────────

/// Enable / disable force-HNSW mode.
/// When enabled, the HNSW index is built and used even when the number of
/// stored embeddings is below the normal 2 000-entry threshold.
#[tauri::command]
pub async fn set_force_hnsw(state: State<'_, AppState>, force: bool) -> Result<(), String> {
    let guard = state.engine.read().await;
    if let Some(engine) = guard.as_ref() {
        engine.set_force_hnsw(force).await;
        tracing::info!("force_hnsw set to {force}");
    }
    Ok(())
}

/// Return the current force-HNSW setting.
#[tauri::command]
pub async fn get_force_hnsw(state: State<'_, AppState>) -> Result<bool, String> {
    let guard = state.engine.read().await;
    Ok(guard.as_ref().is_some_and(|e| e.get_force_hnsw()))
}
