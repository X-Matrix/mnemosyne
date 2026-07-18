//! Directory / path ignore rules for indexing and file watching.
//!
//! [`IgnoreConfig`] holds a set of directory *names* (not full paths) that
//! should be excluded.  Matching is done against each path component so that
//! `node_modules` is skipped wherever it appears in the tree.

use std::collections::HashSet;
use std::path::Path;

/// Directories that are virtually never useful to index.
pub const DEFAULT_IGNORED_DIRS: &[&str] = &[
    // ── Version control ──────────────────────────────────────────────────────
    ".git",
    ".svn",
    ".hg",
    ".bzr",
    // ── JS / Node ────────────────────────────────────────────────────────────
    "node_modules",
    "bower_components",
    "jspm_packages",
    ".npm",
    ".yarn",
    ".pnp",
    // ── Python ───────────────────────────────────────────────────────────────
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".tox",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".eggs",
    "*.egg-info",   // kept for documentation; matched by name prefix check
    // ── Rust ─────────────────────────────────────────────────────────────────
    "target",
    // ── Java / Kotlin / Android ───────────────────────────────────────────────
    ".gradle",
    ".m2",
    "build",
    // ── iOS / macOS ──────────────────────────────────────────────────────────
    "Pods",
    "DerivedData",
    ".build",       // Swift PM
    // ── Frontend build output ────────────────────────────────────────────────
    "dist",
    "out",
    ".next",
    ".nuxt",
    ".output",
    ".cache",
    ".parcel-cache",
    ".turbo",
    ".svelte-kit",
    // ── Test / coverage ──────────────────────────────────────────────────────
    "coverage",
    ".nyc_output",
    // ── IDE / editor ─────────────────────────────────────────────────────────
    ".idea",
    ".vscode",
    ".vs",
    ".eclipse",
    // ── OS artefacts ─────────────────────────────────────────────────────────
    "$RECYCLE.BIN",
    "System Volume Information",
    ".Spotlight-V100",
    ".Trashes",
    ".fseventsd",
    // ── Misc ─────────────────────────────────────────────────────────────────
    "vendor",       // Go / PHP vendor dirs (not the same as Rust workspace vendor)
    "__generated__",
    ".terraform",
    ".vagrant",
    "tmp",
    "temp",
    "log",
    "logs",
];

/// Rules controlling which directories are skipped during indexing.
///
/// # Matching rules
/// - Any path **component** equal to an entry in `ignored_dirs` is skipped.
/// - Any **directory** whose name starts with `'.'` is skipped unless it is
///   the root of the scan itself.
/// - Plain files whose name starts with `'.'` are **not** skipped — only
///   directories.
#[derive(Debug, Clone)]
pub struct IgnoreConfig {
    ignored_dirs: HashSet<String>,
}

impl Default for IgnoreConfig {
    fn default() -> Self {
        Self {
            ignored_dirs: DEFAULT_IGNORED_DIRS
                .iter()
                .filter(|s| !s.contains('*')) // skip glob entries (doc-only)
                .map(|s| s.to_string())
                .collect(),
        }
    }
}

impl IgnoreConfig {
    /// Create a config with the built-in default ignore list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an empty config (nothing is ignored).
    pub fn empty() -> Self {
        Self {
            ignored_dirs: HashSet::new(),
        }
    }

    /// Add extra directory names to ignore.
    pub fn with_extra(mut self, dirs: impl IntoIterator<Item = impl Into<String>>) -> Self {
        for d in dirs {
            self.ignored_dirs.insert(d.into());
        }
        self
    }

    /// Remove entries from the ignore list (allow previously-ignored dirs).
    pub fn allow(mut self, dirs: impl IntoIterator<Item = impl Into<String>>) -> Self {
        for d in dirs {
            self.ignored_dirs.remove(&d.into());
        }
        self
    }

    /// Returns `true` if this **directory entry** should be skipped.
    ///
    /// Used with `WalkDir::filter_entry` to prune entire subtrees.
    /// Pass `is_root = true` for the top-level scan directory so it is never
    /// filtered even if its name starts with `'.'`.
    pub fn should_skip_dir(&self, path: &Path, is_root: bool) -> bool {
        if is_root {
            return false;
        }
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with('.') {
                return true;
            }
            if self.ignored_dirs.contains(name) {
                return true;
            }
        }
        false
    }

    /// Returns `true` if a **file path** (as received from the watcher)
    /// should be skipped because one of its ancestor components is ignored.
    pub fn should_skip_path(&self, path: &Path) -> bool {
        for component in path.components() {
            if let std::path::Component::Normal(name) = component {
                if let Some(s) = name.to_str() {
                    if s.starts_with('.') {
                        return true;
                    }
                    if self.ignored_dirs.contains(s) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Expose the current set for display / serialisation.
    pub fn ignored_dirs(&self) -> &HashSet<String> {
        &self.ignored_dirs
    }
}
