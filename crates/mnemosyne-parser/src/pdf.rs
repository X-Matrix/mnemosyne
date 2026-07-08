//! PDF text extraction.
//!
//! Requires the `pdf` feature (enabled by default).
//! Uses `pdf-extract` which is a pure-Rust PDF text extractor.

use mnemosyne_core::{traits::FileParser, types::ParsedContent, Error, Result};
use async_trait::async_trait;
use std::path::Path;
use tracing::{debug, warn};

const CHUNK_SIZE: usize = 1500;
const CHUNK_OVERLAP: usize = 150;

/// Extracts text content from PDF files.
pub struct PdfParser;

#[async_trait]
impl FileParser for PdfParser {
    fn supported_extensions(&self) -> &[&'static str] {
        &["pdf"]
    }

    async fn parse(&self, path: &Path) -> Result<Vec<ParsedContent>> {
        debug!("PdfParser: {}", path.display());
        let path = path.to_path_buf();

        let text = tokio::task::spawn_blocking(move || {
            pdf_extract::extract_text(&path)
        })
        .await
        .map_err(|e| Error::parse(e.to_string()))?
        .unwrap_or_else(|e| {
            warn!("PDF text extraction failed: {e}");
            String::new()
        });

        let text = text.trim().to_string();
        if text.is_empty() {
            return Ok(vec![]);
        }

        Ok(split_text(&text, CHUNK_SIZE, CHUNK_OVERLAP)
            .into_iter()
            .map(|chunk| ParsedContent::Text { text: chunk })
            .collect())
    }
}

fn split_text(text: &str, size: usize, overlap: usize) -> Vec<String> {
    if text.len() <= size {
        return vec![text.to_string()];
    }
    let chars: Vec<char> = text.chars().collect();
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + size).min(chars.len());
        chunks.push(chars[start..end].iter().collect());
        if end == chars.len() { break; }
        start += size - overlap;
    }
    chunks
}
