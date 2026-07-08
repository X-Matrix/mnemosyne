//! Search engine orchestration for Mnemosyne.
//!
//! [`SearchEngine`] is the single entry point for both indexing and searching:
//!
//! ```no_run
//! use mnemosyne_retrieval::SearchEngine;
//! use mnemosyne_core::types::SearchQuery;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let engine = SearchEngine::builder()
//!     .db_path("~/.mnemosyne/db.sqlite")
//!     .build()
//!     .await?;
//!
//! engine.index_directory("/home/user/Documents").await?;
//!
//! let results = engine.search(SearchQuery {
//!     text: "machine learning papers".into(),
//!     ..Default::default()
//! }).await?;
//! # Ok(())
//! # }
//! ```

pub mod builder;
pub mod engine;
pub mod indexer;
pub mod watcher;

pub use builder::SearchEngineBuilder;
pub use engine::SearchEngine;

pub type Result<T> = std::result::Result<T, mnemosyne_core::Error>;
