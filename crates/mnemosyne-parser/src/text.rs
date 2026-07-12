use async_trait::async_trait;
use mnemosyne_core::{traits::FileParser, types::ParsedContent, Error, Result};
use std::path::Path;
use tracing::debug;

/// Chunk size in characters for splitting large text files.
const CHUNK_SIZE: usize = 1500;
/// Overlap between consecutive chunks.
const CHUNK_OVERLAP: usize = 150;

/// Parses plain-text, Markdown, CSV, and source-code files.
pub struct TextParser;

#[async_trait]
impl FileParser for TextParser {
    fn supported_extensions(&self) -> &[&'static str] {
        &[
            "txt", "md", "markdown", "csv", "json", "xml", "html", "htm", "rst", "toml", "yaml",
            "yml", "log", "ini", "conf", "py", "rs", "js", "ts", "go", "java", "c", "cpp", "h",
            "css", "sh", "bat", "sql",
        ]
    }

    async fn parse(&self, path: &Path) -> Result<Vec<ParsedContent>> {
        debug!("TextParser: {}", path.display());

        let raw = tokio::fs::read(path).await.map_err(|e| Error::Io(e))?;

        // Attempt UTF-8; fall back to lossy conversion.
        let text = match String::from_utf8(raw.clone()) {
            Ok(s) => s,
            Err(_) => String::from_utf8_lossy(&raw).into_owned(),
        };

        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        let chunks = split_into_chunks(&text, CHUNK_SIZE, CHUNK_OVERLAP);
        Ok(chunks
            .into_iter()
            .map(|c| ParsedContent::Text { text: c })
            .collect())
    }
}

/// Split `text` into overlapping chunks of up to `size` characters.
fn split_into_chunks(text: &str, size: usize, overlap: usize) -> Vec<String> {
    if text.len() <= size {
        return vec![text.to_string()];
    }

    let chars: Vec<char> = text.chars().collect();
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < chars.len() {
        let end = (start + size).min(chars.len());
        let chunk: String = chars[start..end].iter().collect();
        chunks.push(chunk);

        if end == chars.len() {
            break;
        }
        start += size - overlap;
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_short_text() {
        let chunks = split_into_chunks("hello world", 1500, 150);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "hello world");
    }

    #[test]
    fn test_split_long_text() {
        let text = "a".repeat(4000);
        let chunks = split_into_chunks(&text, 1500, 150);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(c.len() <= 1500);
        }
    }
}
