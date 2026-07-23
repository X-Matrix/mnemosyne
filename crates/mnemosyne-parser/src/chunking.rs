//! Semantic text chunking strategies.
//!
//! All strategies implement [`ChunkStrategy`], which takes a string and
//! returns a `Vec<String>` of non-overlapping (before overlap is applied)
//! logical chunks.
//!
//! # Available strategies
//! | Strategy | Best for |
//! |---|---|
//! | [`ProseStrategy`] | Plain text, general prose |
//! | [`MarkdownStrategy`] | Markdown files (splits on headings) |
//! | [`CodeStrategy`] | Source code (splits on blank-line blocks) |
//!
//! # Example
//! ```no_run
//! use mnemosyne_parser::chunking::{Chunker, MarkdownStrategy};
//! let chunks = Chunker::new(MarkdownStrategy).chunk("# Heading\n\nContent.");
//! ```

/// Soft target for chunk size in characters.
pub const CHUNK_TARGET: usize = 1200;
/// Hard upper limit — no chunk exceeds this.
pub const CHUNK_MAX: usize = 1800;
/// Sentences to carry over from the previous chunk for context continuity.
pub const OVERLAP_SENTENCES: usize = 2;
/// A chunk must have at least this many characters to be worth indexing.
/// Set low so CJK-dense sentences (1 char ≈ 1 word) are never rejected by
/// length alone; the [`MIN_WORD_CHARS`] check handles noise.
pub const MIN_CHUNK_LEN: usize = 30;
/// Minimum "word-characters" in runs of ≥ 2 consecutive informative chars.
/// Prevents symbol-table / escape-sequence snippets from polluting the index.
pub const MIN_WORD_CHARS: usize = 30;

// ── Public trait ──────────────────────────────────────────────────────────────

/// A pluggable text-splitting strategy.
///
/// Implementors receive the raw text and return a list of chunks.
/// The [`Chunker`] wrapper optionally applies sentence-overlap on top.
pub trait ChunkStrategy: Send + Sync {
    /// Split `text` into semantic chunks.
    ///
    /// Each returned string should be non-empty.  The implementation must
    /// ensure no chunk exceeds [`CHUNK_MAX`] characters.
    fn split(&self, text: &str) -> Vec<String>;
}

// ── Chunker facade ────────────────────────────────────────────────────────────

/// Wraps a [`ChunkStrategy`] and applies sentence overlap after splitting.
pub struct Chunker<S: ChunkStrategy> {
    strategy: S,
    with_overlap: bool,
}

impl<S: ChunkStrategy> Chunker<S> {
    /// Create a `Chunker` that applies overlap (default).
    pub fn new(strategy: S) -> Self {
        Self {
            strategy,
            with_overlap: true,
        }
    }

    /// Create a `Chunker` that does NOT apply overlap (useful for testing).
    pub fn no_overlap(strategy: S) -> Self {
        Self {
            strategy,
            with_overlap: false,
        }
    }

    /// Split `text` into chunks, optionally with sentence overlap.
    pub fn chunk(&self, text: &str) -> Vec<String> {
        let mut chunks = self.strategy.split(text);
        chunks.retain(|c| !c.trim().is_empty());
        if self.with_overlap {
            chunks = apply_overlap(chunks);
        }
        chunks
    }
}

// ── Concrete strategies ───────────────────────────────────────────────────────

/// General prose strategy: paragraph → sentence → word → raw character cascade.
pub struct ProseStrategy;

impl ChunkStrategy for ProseStrategy {
    fn split(&self, text: &str) -> Vec<String> {
        let paragraphs = split_on_blank_lines(text);
        merge_and_split(paragraphs, CHUNK_TARGET, CHUNK_MAX)
    }
}

/// Markdown strategy: split on heading boundaries first.
/// Sections that exceed `CHUNK_MAX` are further split by prose rules.
pub struct MarkdownStrategy;

impl ChunkStrategy for MarkdownStrategy {
    fn split(&self, text: &str) -> Vec<String> {
        let mut sections: Vec<String> = Vec::new();
        let mut current = String::new();

        for line in text.lines() {
            let trimmed = line.trim_start();
            let is_heading = trimmed.starts_with('#')
                && trimmed[1..].starts_with(|c: char| c == '#' || c == ' ');
            if is_heading && !current.trim().is_empty() {
                sections.push(current.trim().to_string());
                current = String::new();
            }
            current.push_str(line);
            current.push('\n');
        }
        if !current.trim().is_empty() {
            sections.push(current.trim().to_string());
        }

        // If no headings found, fall back to prose.
        if sections.is_empty() {
            return ProseStrategy.split(text);
        }

        // Over-large sections are recursively split.
        sections
            .into_iter()
            .flat_map(|s| {
                if s.chars().count() > CHUNK_MAX {
                    ProseStrategy.split(&s)
                } else {
                    vec![s]
                }
            })
            .collect()
    }
}

/// Source-code strategy: split on blank lines between top-level blocks.
pub struct CodeStrategy;

impl ChunkStrategy for CodeStrategy {
    fn split(&self, text: &str) -> Vec<String> {
        let blocks = split_on_blank_lines(text);
        merge_and_split(blocks, CHUNK_TARGET, CHUNK_MAX)
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Split on runs of ≥2 consecutive newlines (paragraph breaks).
fn split_on_blank_lines(text: &str) -> Vec<&str> {
    let mut parts: Vec<&str> = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        if bytes[i] == b'\n' {
            let mut j = i + 1;
            while j < len && bytes[j] == b'\n' {
                j += 1;
            }
            if j > i + 1 {
                let part = text[start..i].trim();
                if !part.is_empty() {
                    parts.push(part);
                }
                start = j;
                i = j;
                continue;
            }
        }
        i += 1;
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        parts.push(tail);
    }
    if parts.is_empty() {
        parts.push(text.trim());
    }
    parts
}

/// Merge small blocks / split large blocks to stay near `target` characters.
fn merge_and_split(blocks: Vec<&str>, target: usize, max_size: usize) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();

    for block in blocks {
        let block_len = block.chars().count();
        let current_len = current.chars().count();

        if current_len == 0 {
            current.push_str(block);
        } else if current_len + block_len + 2 <= target {
            current.push_str("\n\n");
            current.push_str(block);
        } else {
            flush(&mut chunks, &mut current, max_size);
            current = block.to_string();
        }
    }
    flush(&mut chunks, &mut current, max_size);
    chunks
}

fn flush(chunks: &mut Vec<String>, current: &mut String, max_size: usize) {
    if current.is_empty() {
        return;
    }
    if current.chars().count() > max_size {
        chunks.extend(split_at_sentences(current, max_size));
    } else {
        chunks.push(current.clone());
    }
    current.clear();
}

/// Split at sentence boundaries keeping each chunk ≤ `max_size` chars.
pub fn split_at_sentences(text: &str, max_size: usize) -> Vec<String> {
    let sentences = tokenise_sentences(text);
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();

    for sentence in sentences {
        let s_len = sentence.chars().count();
        let c_len = current.chars().count();
        if c_len == 0 {
            if s_len > max_size {
                chunks.extend(split_at_words(&sentence, max_size));
            } else {
                current.push_str(&sentence);
            }
        } else if c_len + s_len <= max_size {
            current.push_str(&sentence);
        } else {
            chunks.push(current.trim().to_string());
            if s_len > max_size {
                chunks.extend(split_at_words(&sentence, max_size));
                current = String::new();
            } else {
                current = sentence;
            }
        }
    }
    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }
    chunks
}

/// Lightweight sentence tokeniser for `.?!。？！`.
/// English sentence endings require trailing whitespace; Chinese endings do not.
pub fn tokenise_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;
    while i < len {
        let c = chars[i];
        current.push(c);
        let is_chinese_end = matches!(c, '。' | '？' | '！');
        let is_ascii_end = matches!(c, '.' | '?' | '!');
        if is_chinese_end {
            // Chinese punctuation always ends a sentence.
            sentences.push(current.clone());
            current = String::new();
        } else if is_ascii_end {
            // ASCII punctuation only ends a sentence before whitespace or EOT.
            let next = chars.get(i + 1).copied();
            if next.is_none() || next.map(|n| n.is_whitespace()).unwrap_or(false) {
                while let Some(&ws) = chars.get(i + 1) {
                    if ws.is_whitespace() && ws != '\n' {
                        current.push(ws);
                        i += 1;
                    } else {
                        break;
                    }
                }
                sentences.push(current.clone());
                current = String::new();
            }
        }
        i += 1;
    }
    if !current.trim().is_empty() {
        sentences.push(current);
    }
    if sentences.is_empty() {
        sentences.push(text.to_string());
    }
    sentences
}

/// Split at word boundaries keeping each part ≤ `max_size` chars.
pub fn split_at_words(text: &str, max_size: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let w_len = word.chars().count();
        let c_len = current.chars().count();
        if c_len == 0 {
            if w_len >= max_size {
                chunks.extend(split_raw(word, max_size));
            } else {
                current.push_str(word);
            }
        } else if c_len + 1 + w_len <= max_size {
            current.push(' ');
            current.push_str(word);
        } else {
            chunks.push(current.clone());
            current = word.to_string();
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Raw character split (last resort — no semantic awareness).
pub fn split_raw(text: &str, max_size: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    chars.chunks(max_size).map(|c| c.iter().collect()).collect()
}

/// Prepend the last `OVERLAP_SENTENCES` sentences of chunk N to chunk N+1.
/// The prepended tail is capped so the result never exceeds `CHUNK_MAX`.
pub fn apply_overlap(chunks: Vec<String>) -> Vec<String> {
    if chunks.len() <= 1 {
        return chunks;
    }
    let mut result = Vec::with_capacity(chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        if i == 0 {
            result.push(chunk.clone());
            continue;
        }
        let tail = last_n_sentences(&chunks[i - 1], OVERLAP_SENTENCES);
        let tail = tail.trim();
        // Only prepend if the combined length stays within CHUNK_MAX.
        if tail.is_empty() || tail.chars().count() + 1 + chunk.chars().count() > CHUNK_MAX {
            result.push(chunk.clone());
        } else {
            let mut with_ctx = tail.to_string();
            with_ctx.push('\n');
            with_ctx.push_str(chunk);
            result.push(with_ctx);
        }
    }
    result
}

/// Return the last `n` sentences from `text` joined as a single string.
pub fn last_n_sentences(text: &str, n: usize) -> String {
    tokenise_sentences(text)
        .iter()
        .rev()
        .take(n)
        .rev()
        .cloned()
        .collect::<Vec<_>>()
        .join("")
}

// ── Chunk quality filter ──────────────────────────────────────────────────────

/// Return `true` for a character that conveys semantic information:
/// Latin letters, digits, CJK, Japanese kana, Korean hangul.
fn is_informative(c: char) -> bool {
    c.is_alphabetic()
        || c.is_ascii_digit()
        || ('\u{4E00}'..='\u{9FFF}').contains(&c) // CJK unified
        || ('\u{3040}'..='\u{30FF}').contains(&c) // hiragana + katakana
        || ('\u{AC00}'..='\u{D7AF}').contains(&c) // hangul syllables
}

/// Count characters that belong to *runs* of ≥ 2 consecutive informative chars.
///
/// A single stray letter in the middle of symbols (e.g. the `r` in `\r\n`) does
/// not count; only genuine "word tokens" do.
fn word_chars(text: &str) -> usize {
    let mut total = 0usize;
    let mut run = 0usize;
    for c in text.chars() {
        if is_informative(c) {
            run += 1;
        } else {
            if run >= 2 {
                total += run;
            }
            run = 0;
        }
    }
    if run >= 2 {
        total += run;
    }
    total
}

/// Return `true` if `text` has enough natural-language content to be worth
/// embedding and indexing.
///
/// Filters out:
/// * Very short fragments (< [`MIN_CHUNK_LEN`] chars after trimming).
/// * Symbol / escape-sequence tables where almost no "word" characters exist
///   (e.g. `html2markdown 转义 \`\\\` \`\[\` \`\]\` \`\=\`` — only 19 word
///   chars).
///
/// Deliberately **not** filtering source code: a function body easily exceeds
/// [`MIN_WORD_CHARS`] via identifiers and keywords.
pub fn is_quality_chunk(text: &str) -> bool {
    let stripped = text.trim();
    stripped.chars().count() >= MIN_CHUNK_LEN && word_chars(stripped) >= MIN_WORD_CHARS
}

/// Remove low-quality chunks from a list (see [`is_quality_chunk`]).
pub fn filter_quality(chunks: Vec<String>) -> Vec<String> {
    chunks.into_iter().filter(|c| is_quality_chunk(c)).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prose_short_text_single_chunk() {
        let chunks = Chunker::no_overlap(ProseStrategy).chunk("Hello world.");
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn prose_long_does_not_exceed_max() {
        let para = "This is a sentence. ".repeat(200);
        let chunks = Chunker::new(ProseStrategy).chunk(&para);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(
                c.chars().count() <= CHUNK_MAX,
                "chunk too long: {}",
                c.chars().count()
            );
        }
    }

    #[test]
    fn markdown_splits_on_headings() {
        let md = "# Section 1\n\nContent one.\n\n## Section 2\n\nContent two.\n";
        let chunks = Chunker::no_overlap(MarkdownStrategy).chunk(md);
        assert!(chunks.len() >= 2);
        assert!(chunks[0].contains("Section 1"));
        assert!(chunks.iter().any(|c| c.contains("Section 2")));
    }

    #[test]
    fn chinese_sentence_boundary() {
        let text = "这是第一句话。这是第二句话。这是第三句话。";
        let sentences = tokenise_sentences(text);
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn no_empty_chunks() {
        let text = "\n\n\nHello.\n\n\nWorld.\n\n\n";
        let chunks = Chunker::new(ProseStrategy).chunk(text);
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(!c.trim().is_empty());
        }
    }

    #[test]
    fn quality_filter_rejects_escape_table() {
        // The exact snippet that triggered the bug report.
        let bad = "## 转义字符\nhtml2markdown 转义 `\\\\` `\\[` `\\]` `\\=`";
        assert!(!is_quality_chunk(bad), "escape table should be filtered");
    }

    #[test]
    fn quality_filter_keeps_prose() {
        let good = "Mnemosyne 是一个本地优先的智能文件搜索引擎，支持语义向量检索与关键词混合搜索。";
        assert!(
            is_quality_chunk(good),
            "Chinese prose should pass quality check"
        );
    }

    #[test]
    fn quality_filter_keeps_code_body() {
        let code = r#"pub fn is_quality_chunk(text: &str) -> bool {
    let stripped = text.trim();
    stripped.chars().count() >= MIN_CHUNK_LEN && word_chars(stripped) >= MIN_WORD_CHARS
}"#;
        assert!(
            is_quality_chunk(code),
            "Rust function body should pass quality check"
        );
    }

    #[test]
    fn filter_quality_removes_short_chunks() {
        let chunks = vec![
            "OK".to_string(),
            "This sentence is long enough to pass the quality filter easily.".to_string(),
        ];
        let filtered = filter_quality(chunks);
        assert_eq!(filtered.len(), 1);
        assert!(filtered[0].contains("long enough"));
    }

    #[test]
    fn overlap_adds_sentences_from_prev_chunk() {
        let long = "First sentence. Second sentence. Third sentence. ".repeat(30);
        let chunks = Chunker::new(ProseStrategy).chunk(&long);
        if chunks.len() > 1 {
            let tail = last_n_sentences(&chunks[0], OVERLAP_SENTENCES);
            // The second chunk must contain the tail of the first chunk.
            assert!(chunks[1].contains(tail.trim()));
        }
    }
}
