use mnemosyne_core::Error;
use rusqlite::Connection;
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use tracing::{info, warn};

/// Thread-safe wrapper around a single SQLite connection.
#[derive(Clone)]
pub struct Database {
    pub conn: Arc<Mutex<Connection>>,
    /// True when the sqlite-vector extension was successfully loaded and verified.
    pub sqlite_vector_loaded: Arc<AtomicBool>,
}

impl Database {
    pub fn open(path: &Path) -> Result<Self, Error> {
        let conn = Connection::open(path).map_err(|e| Error::storage(e.to_string()))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
            .map_err(|e| Error::storage(e.to_string()))?;
        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
            sqlite_vector_loaded: Arc::new(AtomicBool::new(false)),
        };
        db.try_load_sqlite_vector();
        db.migrate()?;
        Ok(db)
    }

    pub fn open_in_memory() -> Result<Self, Error> {
        let conn = Connection::open_in_memory().map_err(|e| Error::storage(e.to_string()))?;
        conn.execute_batch("PRAGMA foreign_keys=ON;")
            .map_err(|e| Error::storage(e.to_string()))?;
        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
            sqlite_vector_loaded: Arc::new(AtomicBool::new(false)),
        };
        db.migrate()?;
        Ok(db)
    }

    /// Returns true if the sqlite-vector extension is loaded and functional.
    pub fn sqlite_vector_loaded(&self) -> bool {
        self.sqlite_vector_loaded.load(Ordering::Acquire)
    }

    /// Try to load the sqlite-vec extension from well-known locations.
    ///
    /// The extension is `asg017/sqlite-vec` (`vec0.dylib` on macOS).
    /// Locations tried (in order):
    ///  1. `~/.mnemosyne/lib/vec0.dylib`   (macOS)
    ///  2. `~/.mnemosyne/lib/vec0.so`      (Linux)
    ///  3. `./vec0.dylib` / `./vec0.so`    (current directory)
    ///
    /// Silently skips if not found — falls back to Rust-side HNSW / cosine.
    fn try_load_sqlite_vector(&self) {
        let (ext_name, ext_name_alt) = if cfg!(target_os = "macos") {
            ("vec0.dylib", "vec0.dylib")
        } else if cfg!(target_os = "windows") {
            ("vec0.dll", "vec0.dll")
        } else {
            ("vec0.so", "vec0.so")
        };

        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let candidates = [
            format!("{home}/.mnemosyne/lib/{ext_name}"),
            format!("./{ext_name_alt}"),
        ];

        let conn = self.conn.lock().unwrap();
        // rusqlite requires enabling extension loading first.
        // SAFETY: we call this only during database initialisation.
        unsafe {
            if conn.load_extension_enable().is_err() {
                return;
            }
        }
        for path in &candidates {
            if !std::path::Path::new(path).exists() {
                continue;
            }
            let result = unsafe { conn.load_extension(path, None) };
            match result {
                Ok(_) => {
                    info!("sqlite-vector extension loaded from {path}");
                    // Verify the extension is functional by calling vec_version().
                    match conn.query_row("SELECT vec_version()", [], |r| r.get::<_, String>(0)) {
                        Ok(ver) => {
                            info!("sqlite-vector {ver} ready — KNN search enabled");
                            self.sqlite_vector_loaded.store(true, Ordering::Release);
                        }
                        Err(e) => warn!("sqlite-vector loaded but vec_version() failed: {e}"),
                    }
                    break;
                }
                Err(e) => warn!("Failed to load sqlite-vector from {path}: {e}"),
            }
        }
        let _ = conn.load_extension_disable();
    }

    fn migrate(&self) -> Result<(), Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(SCHEMA_SQL)
            .map_err(|e| Error::storage(format!("migration failed: {e}")))?;
        Ok(())
    }
}

const SCHEMA_SQL: &str = r#"
-- ── Files ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS files (
    id           TEXT PRIMARY KEY,
    path         TEXT NOT NULL UNIQUE,
    file_type    TEXT NOT NULL,
    size         INTEGER NOT NULL DEFAULT 0,
    modified_at  INTEGER,          -- Unix epoch seconds (UTC)
    indexed_at   INTEGER,
    content_hash TEXT
);

-- ── Content chunks ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS document_chunks (
    id          TEXT PRIMARY KEY,
    file_id     TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    kind        TEXT NOT NULL,    -- 'text'|'image'|'audio_transcript'|'video_keyframe'
    content     TEXT NOT NULL,
    UNIQUE(file_id, chunk_index)
);

-- ── FTS5 full-text index ──────────────────────────────────────────────────────
-- trigram tokenizer: splits every 3-char window, enables CJK substring search
-- without a Chinese-specific segmenter.  Queries of >= 3 chars work perfectly.
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
    content,
    content='document_chunks',
    content_rowid='rowid',
    tokenize='trigram'
);

-- Keep FTS in sync via triggers.
CREATE TRIGGER IF NOT EXISTS fts_chunks_ai
    AFTER INSERT ON document_chunks BEGIN
        INSERT INTO fts_chunks(rowid, content) VALUES (new.rowid, new.content);
    END;

CREATE TRIGGER IF NOT EXISTS fts_chunks_ad
    AFTER DELETE ON document_chunks BEGIN
        INSERT INTO fts_chunks(fts_chunks, rowid, content)
            VALUES ('delete', old.rowid, old.content);
    END;

CREATE TRIGGER IF NOT EXISTS fts_chunks_au
    AFTER UPDATE ON document_chunks BEGIN
        INSERT INTO fts_chunks(fts_chunks, rowid, content)
            VALUES ('delete', old.rowid, old.content);
        INSERT INTO fts_chunks(rowid, content) VALUES (new.rowid, new.content);
    END;

-- ── Embeddings ────────────────────────────────────────────────────────────────
-- Stored as raw f32 little-endian bytes until sqlite-vector is available.
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id  TEXT PRIMARY KEY REFERENCES document_chunks(id) ON DELETE CASCADE,
    model_id  TEXT NOT NULL,
    embedding BLOB NOT NULL
);

-- ── Downloaded model registry ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_registry (
    model_id      TEXT PRIMARY KEY,
    local_path    TEXT NOT NULL,
    version       TEXT,
    downloaded_at INTEGER
);
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_in_memory_runs_migrations() {
        let db = Database::open_in_memory().expect("db should open");
        let conn = db.conn.lock().unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }
}
