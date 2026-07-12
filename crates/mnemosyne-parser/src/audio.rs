//! Audio parser.
//!
//! With the `whisper-backend` feature (mnemosyne-model): real speech-to-text.
//! Default: returns filename + stub transcript placeholder.

use async_trait::async_trait;
use mnemosyne_core::{traits::FileParser, types::ParsedContent, Error, Result};
use std::path::Path;
use tracing::debug;

pub struct AudioParser;

#[async_trait]
impl FileParser for AudioParser {
    fn supported_extensions(&self) -> &[&'static str] {
        &["mp3", "wav", "flac", "aac", "ogg", "m4a", "wma", "opus"]
    }

    async fn parse(&self, path: &Path) -> Result<Vec<ParsedContent>> {
        debug!("AudioParser: {}", path.display());
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Whisper transcription is invoked by the retrieval engine directly
        // using the model registry; the parser only extracts what it can
        // without a model (filename + format info).
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("?");
        Ok(vec![ParsedContent::AudioTranscript {
            transcript: format!("Audio ({ext}): {filename}"),
            language: None,
        }])
    }
}
