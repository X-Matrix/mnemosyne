use crate::{AudioParser, ImageParser, TextParser, VideoParser};
use mnemosyne_core::{traits::FileParser, types::ParsedContent, Error, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Dispatches files to the correct [`FileParser`] based on extension.
///
/// New parsers can be registered at runtime with [`ParserRegistry::register`].
pub struct ParserRegistry {
    parsers: HashMap<String, Arc<dyn FileParser>>,
}

impl ParserRegistry {
    /// Build a registry pre-loaded with all built-in parsers.
    pub fn with_defaults() -> Self {
        let mut registry = Self {
            parsers: HashMap::new(),
        };
        registry.register(Arc::new(TextParser));
        registry.register(Arc::new(ImageParser));
        registry.register(Arc::new(AudioParser));
        registry.register(Arc::new(VideoParser));
        #[cfg(feature = "pdf")]
        registry.register(Arc::new(crate::PdfParser));
        registry
    }

    /// Register a parser for all extensions it declares.
    pub fn register(&mut self, parser: Arc<dyn FileParser>) {
        for ext in parser.supported_extensions() {
            self.parsers.insert(ext.to_string(), Arc::clone(&parser));
        }
    }

    /// Return the parser for the given file, if one is registered.
    pub fn get_for_file(&self, path: &Path) -> Option<Arc<dyn FileParser>> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())?
            .to_lowercase();
        self.parsers.get(&ext).map(Arc::clone)
    }

    /// Parse a file, dispatching to the appropriate parser.
    pub async fn parse(&self, path: &Path) -> Result<Vec<ParsedContent>> {
        let parser = self.get_for_file(path).ok_or_else(|| {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("?")
                .to_string();
            Error::UnsupportedFileType { extension: ext }
        })?;
        parser.parse(path).await
    }

    /// Returns `true` if a parser is registered for this file's extension.
    pub fn is_supported(&self, path: &Path) -> bool {
        self.get_for_file(path).is_some()
    }
}
