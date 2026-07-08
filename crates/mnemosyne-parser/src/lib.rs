//! Pluggable file content parsers for Mnemosyne.
//!
//! Parsers implement [`mnemosyne_core::traits::FileParser`] and are
//! registered in [`ParserRegistry`] by extension.
//!
//! # Built-in parsers
//! - [`TextParser`] — plain text, Markdown, CSV, source code, …
//! - [`ImageParser`] — stub (CLIP integration planned)
//! - [`AudioParser`] — stub (Whisper integration planned)
//! - [`VideoParser`] — stub (frame extraction planned)

pub mod audio;
pub mod image;
#[cfg(feature = "pdf")]
pub mod pdf;
pub mod registry;
pub mod text;
pub mod video;

pub use audio::AudioParser;
pub use image::ImageParser;
#[cfg(feature = "pdf")]
pub use pdf::PdfParser;
pub use registry::ParserRegistry;
pub use text::TextParser;
pub use video::VideoParser;

pub type Result<T> = std::result::Result<T, mnemosyne_core::Error>;
