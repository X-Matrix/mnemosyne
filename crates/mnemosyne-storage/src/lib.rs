//! SQLite-backed persistence for Mnemosyne.
//!
//! All database access goes through [`Database`].

pub mod chunk_repo;
pub mod db;
pub mod embedding_repo;
pub mod file_repo;
pub mod model_repo;

pub use chunk_repo::ChunkRepo;
pub use db::Database;
pub use embedding_repo::EmbeddingRepo;
pub use file_repo::FileRepo;
pub use model_repo::ModelRepo;

pub type Result<T> = std::result::Result<T, mnemosyne_core::Error>;
