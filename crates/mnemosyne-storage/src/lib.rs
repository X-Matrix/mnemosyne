//! SQLite-backed persistence for Mnemosyne.
//!
//! All database access goes through [`Database`].

pub mod db;
pub mod file_repo;
pub mod chunk_repo;
pub mod embedding_repo;
pub mod model_repo;

pub use db::Database;
pub use file_repo::FileRepo;
pub use chunk_repo::ChunkRepo;
pub use embedding_repo::EmbeddingRepo;
pub use model_repo::ModelRepo;

pub type Result<T> = std::result::Result<T, mnemosyne_core::Error>;
