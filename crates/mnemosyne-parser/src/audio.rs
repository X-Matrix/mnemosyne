//! Audio parser stub.
//!
//! Full implementation will use OpenAI Whisper (via Candle) for speech-to-text
//! transcription. Currently returns a placeholder.

use mnemosyne_core::{traits::FileParser, types::ParsedContent, Error, Result};
use async_trait::async_trait;
use std::path::Path;
use tracing::debug;

pub struct AudioParser;

#[async_trait]
impl FileParser for AudioParser {
    fn supported_extensions(&self) -> &[&'static str] {
        &["mp3", "wav", "flac", "aac", "ogg", "m4a", "wma", "opus"]
    }

    async fn parse(&self, path: &Path) -> Result<Vec<ParsedContent>> {
        debug!("AudioParser (stub): {}", path.display());

        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        // TODO: integrate Whisper for real transcription.
        Ok(vec![ParsedContent::AudioTranscript {
            transcript: format!("Audio file: {filename}"),
            language: None,
        }])
    }
}
