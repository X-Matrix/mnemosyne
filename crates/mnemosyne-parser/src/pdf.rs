//! PDF text extraction.
//!
//! Requires the `pdf` feature (enabled by default).
//!
//! Extraction strategy (tried in order):
//!  1. `pdf-extract`  — fast, handles most Latin/ASCII PDFs.
//!  2. `lopdf`        — native Rust CJK extractor; parses each page's content
//!                      stream and decodes glyphs via the font's ToUnicode CMap.
//!                      Handles Identity-H / Identity-V encoded CJK PDFs without
//!                      requiring any external programs or C libraries.
//!  3. Filename       — last-resort fallback for encrypted / damaged PDFs.

use async_trait::async_trait;
use mnemosyne_core::{traits::FileParser, types::ParsedContent, Error, Result};
use std::path::Path;
use tracing::{debug, warn};

const CHUNK_SIZE: usize = 1500;
const CHUNK_OVERLAP: usize = 150;

pub struct PdfParser;

#[async_trait]
impl FileParser for PdfParser {
    fn supported_extensions(&self) -> &[&'static str] {
        &["pdf"]
    }

    async fn parse(&self, path: &Path) -> Result<Vec<ParsedContent>> {
        debug!("PdfParser: {}", path.display());

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

        // ── Primary: pdf-extract (silent panic, good for Latin PDFs) ─────────
        let text = tokio::task::spawn_blocking(move || -> String {
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let result = std::panic::catch_unwind(|| pdf_extract::extract_text(&path_for_closure));
            std::panic::set_hook(prev);
            match result {
                Ok(Ok(t)) => t,
                Ok(Err(e)) => {
                    warn!("pdf-extract error: {e}");
                    String::new()
                }
                Err(_) => String::new(), // CJK / unsupported encoding
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

        // ── Secondary: lopdf ToUnicode CMap (native CJK support) ─────────────
        let path2 = path.to_path_buf();
        let lopdf_text =
            tokio::task::spawn_blocking(move || cjk::extract(&path2).unwrap_or_default())
                .await
                .unwrap_or_default();

        let lopdf_text = lopdf_text.trim().to_string();
        if !lopdf_text.is_empty() {
            debug!("PdfParser: lopdf extracted {} chars", lopdf_text.len());
            return Ok(split_text(&lopdf_text, CHUNK_SIZE, CHUNK_OVERLAP)
                .into_iter()
                .map(|chunk| ParsedContent::Text { text: chunk })
                .collect());
        }

        // ── Fallback: filename + parent directories ───────────────────────────
        let fallback = if parent_dirs.is_empty() {
            stem
        } else {
            format!("{} {}", stem, parent_dirs.join(" "))
        };
        Ok(vec![ParsedContent::Text { text: fallback }])
    }
}

// ── Native CJK extractor using lopdf ─────────────────────────────────────────

mod cjk {
    use lopdf::{Document, Object, ObjectId};
    use std::collections::HashMap;

    /// Extract text from a PDF using lopdf + ToUnicode CMap decoding.
    /// Returns an empty String if the PDF cannot be parsed or contains no text.
    pub fn extract(
        path: &std::path::Path,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let doc = Document::load(path)?;
        let mut all_text = String::new();

        for (_, page_id) in doc.get_pages() {
            // ── Build per-page font → ToUnicode maps ──────────────────────────
            // We collect (name → stream ObjectId) while the page-fonts borrow is
            // live, then dereference the streams after the borrow is released.
            let font_stream_ids: Vec<(Vec<u8>, ObjectId)> = {
                let fonts = doc.get_page_fonts(page_id);
                fonts
                    .iter()
                    .filter_map(|(name, dict)| {
                        let obj = dict.get(b"ToUnicode").ok()?;
                        let id = obj.as_reference().ok()?;
                        Some((name.clone(), id))
                    })
                    .collect()
            }; // `fonts` borrow ends here — safe to call doc.get_object() again

            let font_maps: HashMap<Vec<u8>, HashMap<Vec<u8>, String>> = font_stream_ids
                .into_iter()
                .filter_map(|(name, stream_id)| {
                    let content = match doc.get_object(stream_id).ok()? {
                        Object::Stream(s) => s.decompressed_content().ok()?,
                        _ => return None,
                    };
                    let cmap = parse_to_unicode_cmap(&String::from_utf8_lossy(&content));
                    if cmap.is_empty() {
                        None
                    } else {
                        Some((name, cmap))
                    }
                })
                .collect();

            // ── Parse content stream and extract text ─────────────────────────
            let content_bytes = match doc.get_page_content(page_id) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let content = match lopdf::content::Content::decode(&content_bytes) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let mut current_font: Option<Vec<u8>> = None;
            let mut page_text = String::new();

            for op in &content.operations {
                match op.operator.as_ref() {
                    // Set current font
                    "Tf" => {
                        if let Some(Object::Name(name)) = op.operands.first() {
                            current_font = Some(name.clone());
                        }
                    }
                    // Show string
                    "Tj" => {
                        if let Some(Object::String(bytes, _)) = op.operands.first() {
                            page_text.push_str(&decode(bytes, current_font.as_deref(), &font_maps));
                        }
                    }
                    // Show array of strings / kerning adjustments
                    "TJ" => {
                        if let Some(Object::Array(arr)) = op.operands.first() {
                            for item in arr {
                                if let Object::String(bytes, _) = item {
                                    page_text.push_str(&decode(
                                        bytes,
                                        current_font.as_deref(),
                                        &font_maps,
                                    ));
                                }
                            }
                        }
                    }
                    // Next-line show string
                    "'" => {
                        page_text.push('\n');
                        if let Some(Object::String(bytes, _)) = op.operands.first() {
                            page_text.push_str(&decode(bytes, current_font.as_deref(), &font_maps));
                        }
                    }
                    // Next-line with word/char spacing show string
                    "\"" => {
                        page_text.push('\n');
                        if let Some(Object::String(bytes, _)) = op.operands.get(2) {
                            page_text.push_str(&decode(bytes, current_font.as_deref(), &font_maps));
                        }
                    }
                    // Text line movement → treat as whitespace
                    "Td" | "TD" | "Tm" => page_text.push(' '),
                    "T*" => page_text.push('\n'),
                    _ => {}
                }
            }

            let trimmed = page_text.trim().to_string();
            if !trimmed.is_empty() {
                all_text.push_str(&trimmed);
                all_text.push('\n');
            }
        }

        Ok(all_text.trim().to_string())
    }

    // ── ToUnicode CMap parser ─────────────────────────────────────────────────

    /// Parse a ToUnicode CMap and return `char_code_bytes → Unicode_string`.
    fn parse_to_unicode_cmap(cmap: &str) -> HashMap<Vec<u8>, String> {
        #[derive(PartialEq)]
        enum State {
            None,
            BfChar,
            BfRange,
        }
        let mut map = HashMap::new();
        let mut state = State::None;

        for line in cmap.lines() {
            let line = line.trim();
            match line {
                l if l.ends_with("beginbfchar") => state = State::BfChar,
                "endbfchar" => state = State::None,
                l if l.ends_with("beginbfrange") => state = State::BfRange,
                "endbfrange" => state = State::None,
                _ => match state {
                    State::BfChar => parse_bfchar(line, &mut map),
                    State::BfRange => parse_bfrange(line, &mut map),
                    State::None => {}
                },
            }
        }
        map
    }

    fn hex_bytes(token: &str) -> Option<Vec<u8>> {
        let inner = token.trim().strip_prefix('<')?.strip_suffix('>')?;
        if inner.len() % 2 != 0 {
            return None;
        }
        (0..inner.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&inner[i..i + 2], 16).ok())
            .collect()
    }

    fn utf16be_to_string(bytes: &[u8]) -> Option<String> {
        if bytes.len() % 2 != 0 {
            return None;
        }
        let units: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|b| u16::from_be_bytes([b[0], b[1]]))
            .collect();
        String::from_utf16(&units).ok()
    }

    /// Parse one `<src> <dst>` line inside a `beginbfchar` block.
    fn parse_bfchar(line: &str, map: &mut HashMap<Vec<u8>, String>) {
        let mut tokens = line.split_whitespace();
        let src = tokens.next().and_then(hex_bytes);
        let dst = tokens.next().and_then(hex_bytes);
        if let (Some(s), Some(d)) = (src, dst) {
            if let Some(ch) = utf16be_to_string(&d) {
                map.insert(s, ch);
            }
        }
    }

    /// Parse one `<start> <end> <dst>` line inside a `beginbfrange` block.
    /// The dst may be a hex string (contiguous range) or an array `[<d1> <d2>…]`.
    fn parse_bfrange(line: &str, map: &mut HashMap<Vec<u8>, String>) {
        let mut tokens = line.split_whitespace();
        let start_bytes = tokens.next().and_then(hex_bytes);
        let end_bytes = tokens.next().and_then(hex_bytes);
        let (Some(start), Some(end)) = (start_bytes, end_bytes) else {
            return;
        };

        // Remaining tokens may be a hex string or an array
        let rest: String = tokens.collect::<Vec<_>>().join(" ");
        let rest = rest.trim();

        let code_width = start.len();
        let start_code = bytes_to_u32(&start);
        let end_code = bytes_to_u32(&end);

        if rest.starts_with('[') {
            // Array form: each entry maps to successive source codes
            let entries: Vec<&str> = rest
                .trim_start_matches('[')
                .trim_end_matches(']')
                .split_whitespace()
                .collect();
            for (i, entry) in entries.iter().enumerate() {
                let code = start_code + i as u32;
                if code > end_code {
                    break;
                }
                if let Some(dst) = hex_bytes(entry).and_then(|b| utf16be_to_string(&b)) {
                    map.insert(u32_to_bytes(code, code_width), dst);
                }
            }
        } else if let Some(dst_bytes) = hex_bytes(rest) {
            // Contiguous range: dst_start + i maps to src_start + i
            let dst_u16 = bytes_to_u32(&dst_bytes);
            for code in start_code..=end_code {
                let unicode_cp = dst_u16 + (code - start_code);
                if let Some(ch) = char::from_u32(unicode_cp) {
                    map.insert(u32_to_bytes(code, code_width), ch.to_string());
                }
            }
        }
    }

    fn bytes_to_u32(bytes: &[u8]) -> u32 {
        bytes.iter().fold(0u32, |acc, &b| (acc << 8) | b as u32)
    }

    fn u32_to_bytes(code: u32, width: usize) -> Vec<u8> {
        (0..width)
            .rev()
            .map(|shift| ((code >> (shift * 8)) & 0xFF) as u8)
            .collect()
    }

    // ── Glyph-code decoder ────────────────────────────────────────────────────

    /// Decode a PDF string byte sequence using the current font's ToUnicode map.
    fn decode(
        bytes: &[u8],
        font_name: Option<&[u8]>,
        maps: &HashMap<Vec<u8>, HashMap<Vec<u8>, String>>,
    ) -> String {
        let Some(name) = font_name else {
            return String::from_utf8_lossy(bytes).to_string();
        };
        let Some(font_map) = maps.get(name) else {
            return String::from_utf8_lossy(bytes).to_string();
        };
        if font_map.is_empty() {
            return String::from_utf8_lossy(bytes).to_string();
        }

        // Detect whether this font uses 1- or 2-byte character codes.
        let two_byte = font_map.keys().any(|k| k.len() == 2);

        let mut result = String::new();
        let mut i = 0;
        while i < bytes.len() {
            if two_byte && i + 1 < bytes.len() {
                if let Some(s) = font_map.get(&bytes[i..i + 2]) {
                    result.push_str(s);
                    i += 2;
                    continue;
                }
            }
            // 1-byte fallback (or when 2-byte miss occurs at end of buffer)
            if let Some(s) = font_map.get(&bytes[i..i + 1]) {
                result.push_str(s);
            }
            i += 1;
        }
        result
    }
}

// ── Text chunking ─────────────────────────────────────────────────────────────

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
        if end == chars.len() {
            break;
        }
        start += size - overlap;
    }
    chunks
}
