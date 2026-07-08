mod commands;
mod state;

use state::AppState;
use tauri::Manager;
use tracing_subscriber::EnvFilter;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Initialise logging.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,mnemosyne=debug")),
        )
        .init();

    tauri::Builder::default()
        .setup(|app| {
            let data_dir = app
                .path()
                .app_data_dir()
                .expect("failed to get app data dir");

            // Initialise SearchEngine asynchronously on the Tokio runtime
            // and store it in managed state.
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let db_path = data_dir.join("mnemosyne.sqlite");
                match mnemosyne_retrieval::SearchEngine::builder()
                    .db_path(&db_path)
                    .build()
                    .await
                {
                    Ok(engine) => {
                        handle.manage(AppState::new(engine));
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
            commands::index::get_stats,
            commands::index::list_files,
            commands::index::remove_file,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Mnemosyne");
}
