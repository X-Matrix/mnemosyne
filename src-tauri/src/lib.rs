mod commands;
mod state;

use state::{AppState, LogBuf, LogEntry};
use tauri::Manager;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

// ── Custom log-capture tracing Layer ─────────────────────────────────────────

struct LogCaptureLayer {
    buffer: LogBuf,
}

impl<S> tracing_subscriber::Layer<S> for LogCaptureLayer
where
    S: tracing::Subscriber,
{
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let meta = event.metadata();
        let level = meta.level().to_string().to_uppercase();
        let target = meta.target().to_string();

        let mut message = String::new();
        struct Visitor<'a>(&'a mut String);
        impl<'a> tracing::field::Visit for Visitor<'a> {
            fn record_str(&mut self, field: &tracing::field::Field, val: &str) {
                if field.name() == "message" {
                    self.0.push_str(val);
                }
            }
            fn record_debug(&mut self, field: &tracing::field::Field, val: &dyn std::fmt::Debug) {
                if field.name() == "message" && self.0.is_empty() {
                    self.0.push_str(&format!("{val:?}"));
                }
            }
        }
        event.record(&mut Visitor(&mut message));

        let ts = chrono::Local::now().format("%H:%M:%S%.3f").to_string();
        let entry = LogEntry {
            ts,
            level,
            target,
            message,
        };

        if let Ok(mut buf) = self.buffer.lock() {
            if buf.len() >= 500 {
                buf.pop_front();
            }
            buf.push_back(entry);
        }
    }
}

// ── Application entry-point ───────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    eprintln!(">>> Mnemosyne starting");

    let log_buf = state::new_log_buf();

    // Layered subscriber: fmt (stderr) + capture (in-memory)
    tracing_subscriber::registry()
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,mnemosyne=debug")),
        )
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .with(LogCaptureLayer {
            buffer: Arc::clone(&log_buf),
        })
        .init();

    use std::sync::Arc;

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(move |app| {
            app.manage(AppState::empty_with_log(Arc::clone(&log_buf)));

            let engine_lock = app.state::<AppState>().engine.clone();

            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            let db_path = std::path::PathBuf::from(&home)
                .join(".mnemosyne")
                .join("db.sqlite");

            // Read persisted active models for all three categories.
            let initial = commands::models::read_persisted_models();
            if let Some(ref m) = initial.text {
                tracing::info!("Restoring text model: {m}");
            }
            if let Some(ref m) = initial.vision {
                tracing::info!("Restoring vision model: {m}");
            }
            if let Some(ref m) = initial.audio {
                tracing::info!("Restoring audio model: {m}");
            }

            tracing::info!("Using database: {}", db_path.display());

            tauri::async_runtime::spawn(async move {
                let mut builder = mnemosyne_retrieval::SearchEngine::builder().db_path(&db_path);
                if let Some(m) = initial.text {
                    builder = builder.text_model(m);
                }
                if let Some(m) = initial.vision {
                    builder = builder.vision_model(m);
                }
                if let Some(m) = initial.audio {
                    builder = builder.audio_model(m);
                }
                match builder.build().await {
                    Ok(engine) => {
                        *engine_lock.write().await = Some(engine);
                        tracing::info!("SearchEngine ready (db: {})", db_path.display());
                    }
                    Err(e) => tracing::error!("Failed to initialise SearchEngine: {e}"),
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
            commands::index::preview_file,
            commands::models::download_model,
            commands::models::list_models,
            commands::models::switch_model,
            commands::models::get_active_models,
            commands::logs::get_logs,
            commands::logs::clear_logs,
            commands::api::start_api_server,
            commands::api::stop_api_server,
            commands::api::get_api_status,
            commands::index::set_force_hnsw,
            commands::index::get_force_hnsw,
            commands::index::clear_index,
            commands::index::count_files_in_dir,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Mnemosyne");
}
