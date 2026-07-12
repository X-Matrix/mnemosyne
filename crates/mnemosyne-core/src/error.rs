use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error: {message}")]
    Parse { message: String },

    #[error("Model error: {message}")]
    Model { message: String },

    #[error("Storage error: {message}")]
    Storage { message: String },

    #[error("Index error: {message}")]
    Index { message: String },

    #[error("Unsupported file type: {extension}")]
    UnsupportedFileType { extension: String },

    #[error("Model not loaded: {model_id}")]
    ModelNotLoaded { model_id: String },

    #[error("File not found: {path}")]
    FileNotFound { path: String },

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl Error {
    pub fn parse(msg: impl Into<String>) -> Self {
        Self::Parse {
            message: msg.into(),
        }
    }

    pub fn model(msg: impl Into<String>) -> Self {
        Self::Model {
            message: msg.into(),
        }
    }

    pub fn storage(msg: impl Into<String>) -> Self {
        Self::Storage {
            message: msg.into(),
        }
    }

    pub fn index(msg: impl Into<String>) -> Self {
        Self::Index {
            message: msg.into(),
        }
    }
}
