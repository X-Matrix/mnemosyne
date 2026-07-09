mod commands;
mod state;

use state::AppState;
use tauri::Manager;
use tracing_subscriber::EnvFilter;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    eprintln!(">>> Mnemosyne starting (stderr logging active)");

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,mnemosyne=debug")),
        )
        .init();

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            // ── CRITICAL: manage() MUST be called synchronously during setup ──
            // Calling it from a spawned task causes State<AppState> to panic.
            app.manage(AppState::empty());

            // Grab the engine Arc so we can populate it from the async task.
            let engine_lock = app.state::<AppState>().engine.clone();
            let indexing_map = app.state::<AppState>().indexing.clone();

            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            let db_path = std::path::PathBuf::from(&home)
                .join(".mnemosyne")
                .join("db.sqlite");

            tracing::info!("Using database: {}", db_path.display());

            tauri::async_runtime::spawn(async move {
                match mnemosyne_retrieval::SearchEngine::builder()
                    .db_path(&db_path)
                    .build()
                    .await
                {
                    Ok(engine) => {
                        let mut guard = engine_lock.write().await;
                        *guard = Some(engine);
                        tracing::info!("SearchEngine ready (db: {})", db_path.display());
                    }
                    Err(e) => {
                        tracing::error!("Failed to initialise SearchEngine: {e}");
                    }
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::search::search_files,
            commands::index::index_directory,
            commands::index::index_directory_bg,
            commands::index::get_indexing_status,
            commands::index::pick_directory,
            commands::index::watch_directory,
            commands::index::stop_watching,
            commands::index::get_stats,
            commands::index::list_files,
            commands::index::remove_file,
            commands::models::download_model,
            commands::models::list_models,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Mnemosyne");
}

