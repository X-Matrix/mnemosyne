use async_trait::async_trait;
use mnemosyne_core::{traits::FileParser, types::ParsedContent, Error, Result};
use std::path::Path;
use tracing::debug;

use crate::chunking::{filter_quality, Chunker, CodeStrategy, MarkdownStrategy, ProseStrategy};

/// Plain-text, Markdown, CSV and source-code parser.
///
/// Chunking is delegated to [`crate::chunking`] strategies so that PDF, audio
/// transcripts, and other formats can reuse the same splitting logic.
pub struct TextParser;

#[async_trait]
impl FileParser for TextParser {
    fn supported_extensions(&self) -> &[&'static str] {
        &[
            "txt", "md", "markdown", "csv", "json", "xml", "html", "htm", "rst",
            "toml", "yaml", "yml", "log", "ini", "conf", "py", "rs", "js", "ts",
            "go", "java", "c", "cpp", "h", "css", "sh", "bat", "sql",
        ]
    }

    async fn parse(&self, path: &Path) -> Result<Vec<ParsedContent>> {
        debug!("TextParser: {}", path.display());

        let raw = tokio::fs::read(path).await.map_err(Error::Io)?;
        let text = match String::from_utf8(raw.clone()) {
            Ok(s) => s,
            Err(_) => String::from_utf8_lossy(&raw).into_owned(),
        };

        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let raw_chunks: Vec<String> = match ext.as_str() {
            "md" | "markdown" => Chunker::new(MarkdownStrategy).chunk(&text),
            "py" | "rs" | "js" | "ts" | "go" | "java" | "c" | "cpp" | "h"
            | "css" | "sh" | "sql" => Chunker::new(CodeStrategy).chunk(&text),
            _ => Chunker::new(ProseStrategy).chunk(&text),
        };

        // Remove symbol-heavy / escape-table fragments that produce noisy
        // embeddings and pollute search results (e.g. markdown escape tables).
        let chunks = filter_quality(raw_chunks);

        Ok(chunks
            .into_iter()
            .map(|c| ParsedContent::Text { text: c })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunking::{Chunker, ProseStrategy, CHUNK_MAX};

    #[test]
    fn parse_empty_returns_empty() {
        let chunks: Vec<String> = Chunker::no_overlap(ProseStrategy).chunk("   ");
        assert!(chunks.is_empty());
    }

    #[test]
    fn parse_long_text_respects_max() {
        let text = "This is a sentence. ".repeat(200);
        let chunks = Chunker::new(ProseStrategy).chunk(&text);
        for c in &chunks {
            assert!(c.chars().count() <= CHUNK_MAX);
        }
    }
}
