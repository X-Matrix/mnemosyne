use anyhow::Result;
use clap::{Parser, Subcommand};
use mnemosyne_core::types::{SearchMode, SearchQuery};
use mnemosyne_retrieval::SearchEngine;
use tracing_subscriber::EnvFilter;

/// Mnemosyne — intelligent local file search.
#[derive(Parser)]
#[command(name = "mnemosyne", version, about, long_about = None)]
struct Cli {
    /// Path to the SQLite database (default: ~/.mnemosyne/db.sqlite).
    #[arg(long, global = true)]
    db: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Recursively index a directory.
    Index {
        /// Directory to index.
        path: String,
    },

    /// Search indexed files.
    Search {
        /// Query text.
        query: String,

        /// Maximum number of results.
        #[arg(short, long, default_value = "10")]
        limit: usize,

        /// Search mode: vector | keyword | hybrid.
        #[arg(short, long, default_value = "hybrid")]
        mode: String,

        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },

    /// List indexed files.
    List {
        /// Number of entries to show.
        #[arg(short, long, default_value = "20")]
        limit: usize,

        /// Offset for pagination.
        #[arg(short, long, default_value = "0")]
        offset: usize,
    },

    /// Show index statistics.
    Stats,

    /// Remove a file from the index.
    Remove {
        /// File ID.
        id: String,
    },

    /// Start the REST API server.
    Serve {
        /// TCP port to listen on.
        #[arg(short, long, default_value = "8080")]
        port: u16,
    },

    /// Watch a directory and auto-index changes.
    Watch {
        /// Directory to watch.
        path: String,
    },

    /// Download a model from HuggingFace Hub.
    ModelDownload {
        /// HuggingFace model ID (e.g. sentence-transformers/all-MiniLM-L6-v2)
        model_id: String,
    },

    /// List downloaded models.
    Models,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("warn,mnemosyne=info")),
        )
        .init();

    let cli = Cli::parse();
    let db_path = cli.db.clone(); // save before move into builder

    let mut builder = SearchEngine::builder();
    if let Some(ref db) = db_path {
        builder = builder.db_path(db);
    }
    let engine = builder.build().await?;

    match cli.command {
        Commands::Index { path } => {
            println!("Indexing: {path}");
            let stats = engine.index_directory(&path).await?;
            println!("Done: {} files indexed, {} total chunks", stats.total_files, stats.total_chunks);
            for (ft, count) in &stats.files_by_type {
                println!("  {ft}: {count}");
            }
        }

        Commands::Search { query, limit, mode, json } => {
            let search_mode = match mode.as_str() {
                "vector"  => SearchMode::Vector,
                "keyword" => SearchMode::Keyword,
                _         => SearchMode::Hybrid,
            };
            let results = engine
                .search(SearchQuery {
                    text: query,
                    limit,
                    mode: search_mode,
                    ..Default::default()
                })
                .await?;

            if json {
                println!("{}", serde_json::to_string_pretty(&results)?);
            } else {
                for (i, r) in results.iter().enumerate() {
                    println!(
                        "{}. [score={:.3}] {}",
                        i + 1,
                        r.score,
                        r.file_record.path.display()
                    );
                    if let Some(snippet) = &r.snippet {
                        println!("   {}", snippet.chars().take(120).collect::<String>());
                    }
                }
            }
        }

        Commands::List { limit, offset } => {
            let files = engine.list_files(limit, offset).await?;
            for f in &files {
                println!("{} — {:?} — {}", f.id, f.file_type, f.path.display());
            }
            println!("({} files)", files.len());
        }

        Commands::Stats => {
            let stats = engine.get_stats().await?;
            println!("Files:  {}", stats.total_files);
            println!("Chunks: {}", stats.total_chunks);
        }

        Commands::Remove { id } => {
            engine.remove_file(&id).await?;
            println!("Removed: {id}");
        }

        Commands::Serve { port } => {
            std::env::set_var("MNEMOSYNE_PORT", port.to_string());
            if let Some(db) = db_path {
                std::env::set_var("MNEMOSYNE_DB", db);
            }
            drop(engine); // API server creates its own engine
            mnemosyne_api::run().await?;
        }

        Commands::Watch { path } => {
            use mnemosyne_retrieval::watcher::FileWatcher;
            use std::sync::Arc;
            println!("Watching: {path} (Ctrl-C to stop)");
            let engine = Arc::new(engine);
            let _w = FileWatcher::watch(&path, Arc::clone(&engine)).await?;
            tokio::signal::ctrl_c().await?;
            println!("Stopped.");
        }

        Commands::ModelDownload { model_id } => {
            println!("Downloading model: {model_id}");
            engine.download_model(&model_id).await?;
            println!("Done.");
        }

        Commands::Models => {
            let models = engine.list_models().await?;
            if models.is_empty() {
                println!("No models downloaded yet.");
            } else {
                for m in &models {
                    println!("  {} → {}", m.model_id, m.local_path);
                }
            }
        }
    }

    Ok(())
}
