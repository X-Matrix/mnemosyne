//! PDF text extraction.
//!
//! Requires the `pdf` feature (enabled by default).
//! Uses `pdf-extract` which is a pure-Rust PDF text extractor.
//!
//! ## Chinese/CJK PDFs
//! `pdf-extract` panics on Identity-H/V encoded fonts (common in CJK PDFs).
//! We catch the panic and fall back to indexing with the filename so the
//! document is still discoverable by search.

use mnemosyne_core::{traits::FileParser, types::ParsedContent, Error, Result};
use async_trait::async_trait;
use std::path::Path;
use tracing::{debug, warn};

const CHUNK_SIZE:    usize = 1500;
const CHUNK_OVERLAP: usize = 150;

pub struct PdfParser;

#[async_trait]
impl FileParser for PdfParser {
    fn supported_extensions(&self) -> &[&'static str] {
        &["pdf"]
    }

    async fn parse(&self, path: &Path) -> Result<Vec<ParsedContent>> {
        debug!("PdfParser: {}", path.display());

        // Save filename/dir info before moving path into the closure.
        let stem: String = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("document")
            .to_string();
        let parent_dirs: Vec<String> = path
            .ancestors()
            .skip(1)
            .take(2)
            .filter_map(|p| p.file_name()?.to_str().map(str::to_string))
            .collect();

        let path_for_closure = path.to_path_buf();

        // Run extraction in a blocking thread; catch any panic so CJK PDFs
        // (Identity-H font encoding) don't kill the worker thread.
        let text = tokio::task::spawn_blocking(move || -> String {
            match std::panic::catch_unwind(|| pdf_extract::extract_text(&path_for_closure)) {
                Ok(Ok(t)) => t,
                Ok(Err(e)) => {
                    warn!("PDF text error: {e}");
                    String::new()
                }
                Err(payload) => {
                    let msg = payload
                        .downcast_ref::<String>()
                        .map(String::as_str)
                        .or_else(|| payload.downcast_ref::<&str>().copied())
                        .unwrap_or("unknown panic");
                    warn!("PDF parser panicked (CJK font?): {msg}");
                    String::new()
                }
            }
        })
        .await
        .unwrap_or_default();

        let text = text.trim().to_string();
        if !text.is_empty() {
            return Ok(split_text(&text, CHUNK_SIZE, CHUNK_OVERLAP)
                .into_iter()
                .map(|chunk| ParsedContent::Text { text: chunk })
                .collect());
        }

        // ── Fallback: index file stem + parent dirs ──────────────────────────
        // When body text can't be extracted (CJK font, encrypted, etc.) we still
        // index the filename so the PDF is discoverable by name/path search.
        // e.g. "AI自动化研发平台商业计划书.pdf" → "AI自动化研发平台商业计划书 Downloads"
        let fallback = if parent_dirs.is_empty() {
            stem
        } else {
            format!("{} {}", stem, parent_dirs.join(" "))
        };

        Ok(vec![ParsedContent::Text { text: fallback }])
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
