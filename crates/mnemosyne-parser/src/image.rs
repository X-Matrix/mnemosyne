//! Image parser.
//!
//! With the `image-meta` feature (default): reads image dimensions via the
//! `image` crate without decoding pixels.
//! Without the feature: returns a placeholder description.

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
        debug!("ImageParser: {}", path.display());

        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        #[cfg(feature = "image-meta")]
        {
            let path = path.to_path_buf();
            let caption = tokio::task::spawn_blocking(move || {
                match image::image_dimensions(&path) {
                    Ok((w, h)) => format!(
                        "Image: {} ({}×{} pixels)",
                        path.file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("unknown"),
                        w, h
                    ),
                    Err(_) => format!("Image file: {}", filename),
                }
            })
            .await
            .map_err(|e| Error::parse(e.to_string()))?;

            return Ok(vec![ParsedContent::Image {
                caption,
                tags: vec![],
            }]);
        }

        #[cfg(not(feature = "image-meta"))]
        Ok(vec![ParsedContent::Image {
            caption: format!("Image file: {filename}"),
            tags: vec![],
        }])
    }
}
