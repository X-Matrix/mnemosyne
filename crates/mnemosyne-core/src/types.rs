use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Category of a file based on its extension.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FileType {
    Text,
    Image,
    Audio,
    Video,
    Unknown,
}

impl FileType {
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "txt" | "md" | "markdown" | "csv" | "json" | "xml" | "html" | "htm" | "rst"
            | "toml" | "yaml" | "yml" | "log" | "ini" | "conf" | "py" | "rs" | "js" | "ts"
            | "go" | "java" | "c" | "cpp" | "h" | "css" | "sh" | "bat" => Self::Text,

            "jpg" | "jpeg" | "png" | "bmp" | "gif" | "webp" | "tiff" | "tif" | "svg"
            | "heic" | "heif" => Self::Image,

            "mp3" | "wav" | "flac" | "aac" | "ogg" | "m4a" | "wma" | "opus" => Self::Audio,

            "mp4" | "avi" | "mov" | "mkv" | "webm" | "flv" | "wmv" | "m4v" => Self::Video,

            _ => Self::Unknown,
        }
    }
}

/// Metadata record for a file tracked by Mnemosyne.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRecord {
    pub id: String,
    pub path: PathBuf,
    pub file_type: FileType,
    pub size: u64,
    pub modified_at: Option<DateTime<Utc>>,
    pub indexed_at: Option<DateTime<Utc>>,
    pub content_hash: Option<String>,
}

/// Parsed / extracted content from a file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ParsedContent {
    Text {
        text: String,
    },
    Image {
        caption: String,
        tags: Vec<String>,
    },
    AudioTranscript {
        transcript: String,
        language: Option<String>,
    },
    VideoKeyframe {
        timestamp_secs: f32,
        description: String,
    },
}

impl ParsedContent {
    /// Return the textual representation suitable for embedding.
    pub fn as_text(&self) -> &str {
        match self {
            Self::Text { text } => text.as_str(),
            Self::Image { caption, .. } => caption.as_str(),
            Self::AudioTranscript { transcript, .. } => transcript.as_str(),
            Self::VideoKeyframe { description, .. } => description.as_str(),
        }
    }
}

/// Dense vector embedding produced by an embedding model.
pub type Embedding = Vec<f32>;

/// A document chunk with optional embedding, stored in the index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedChunk {
    pub chunk_id: String,
    pub file_id: String,
    pub chunk_index: usize,
    pub content: ParsedContent,
    pub embedding: Option<Embedding>,
}

/// User query to the search engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub text: String,
    pub file_types: Option<Vec<FileType>>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
    #[serde(default)]
    pub mode: SearchMode,
}

fn default_limit() -> usize {
    20
}

impl Default for SearchQuery {
    fn default() -> Self {
        Self {
            text: String::new(),
            file_types: None,
            limit: 20,
            offset: 0,
            mode: SearchMode::Hybrid,
        }
    }
}

/// How to combine vector and keyword signals.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    Vector,
    Keyword,
    #[default]
    Hybrid,
}

/// A single result returned from a search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub file_record: FileRecord,
    pub score: f32,
    pub snippet: Option<String>,
    pub match_type: MatchType,
    pub chunk_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MatchType {
    Vector,
    Keyword,
    Hybrid,
}

/// High-level statistics about the index.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexStats {
    pub total_files: u64,
    pub total_chunks: u64,
    pub files_by_type: std::collections::HashMap<String, u64>,
    pub index_size_bytes: u64,
}
