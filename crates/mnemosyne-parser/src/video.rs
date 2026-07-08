//! Video parser stub.
//!
//! Full implementation will extract keyframes using ffmpeg bindings,
//! then caption them with CLIP. Currently returns a placeholder.

use mnemosyne_core::{traits::FileParser, types::ParsedContent, Error, Result};
use async_trait::async_trait;
use std::path::Path;
use tracing::debug;

pub struct VideoParser;

#[async_trait]
impl FileParser for VideoParser {
    fn supported_extensions(&self) -> &[&'static str] {
        &["mp4", "avi", "mov", "mkv", "webm", "flv", "wmv", "m4v"]
    }

    async fn parse(&self, path: &Path) -> Result<Vec<ParsedContent>> {
        debug!("VideoParser (stub): {}", path.display());

        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        // TODO: extract keyframes + CLIP captioning.
        Ok(vec![ParsedContent::VideoKeyframe {
            timestamp_secs: 0.0,
            description: format!("Video file: {filename}"),
        }])
    }
}
