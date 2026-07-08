//! Image parser stub.
//!
//! Full implementation will use the CLIP model for image captioning
//! and tag generation. Currently returns a placeholder description.

use mnemosyne_core::{traits::FileParser, types::ParsedContent, Error, Result};
use async_trait::async_trait;
use std::path::Path;
use tracing::debug;

pub struct ImageParser;

#[async_trait]
impl FileParser for ImageParser {
    fn supported_extensions(&self) -> &[&'static str] {
        &["jpg", "jpeg", "png", "bmp", "gif", "webp", "tiff", "tif", "heic", "heif"]
    }

    async fn parse(&self, path: &Path) -> Result<Vec<ParsedContent>> {
        debug!("ImageParser (stub): {}", path.display());

        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        // TODO: integrate CLIP model for real captioning.
        Ok(vec![ParsedContent::Image {
            caption: format!("Image file: {filename}"),
            tags: vec![],
        }])
    }
}
