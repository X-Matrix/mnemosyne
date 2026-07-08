//! Model loading, inference, and download management for Mnemosyne.
//!
//! Embedding models are loaded lazily and cached in a [`ModelRegistry`].
//!
//! # Supported backends
//! - **Text**: `sentence-transformers/all-MiniLM-L6-v2` via Candle BERT
//! - **Image**: CLIP *(planned)*
//! - **Audio**: Whisper *(planned)*

pub mod downloader;
pub mod registry;
pub mod text;

pub use downloader::ModelDownloader;
pub use registry::ModelRegistry;
pub use text::{TextEmbedder, DEFAULT_TEXT_MODEL};

pub type Result<T> = std::result::Result<T, mnemosyne_core::Error>;
