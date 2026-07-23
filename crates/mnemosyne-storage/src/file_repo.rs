use crate::Database;
use chrono::{TimeZone, Utc};
use mnemosyne_core::{
    types::{FileRecord, FileType},
    Error,
};
use rusqlite::params;
use std::path::PathBuf;

pub struct FileRepo<'a> {
    pub db: &'a Database,
}

impl<'a> FileRepo<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    /// Insert or update a file record, returning the **id that is stored** in the
    /// database after the operation.
    ///
    /// If the path already exists under a *different* id (e.g. because a previous
    /// lookup missed due to a unicode-normalisation mismatch on macOS), the
    /// existing id is preserved so that associated chunks remain valid.  The
    /// caller should use the returned id when creating chunk records.
    pub fn upsert(&self, record: &FileRecord) -> Result<String, Error> {
        let conn = self.db.conn.lock().unwrap();
        let path = record.path.to_string_lossy().to_string();

        // Use COALESCE to prefer the id already stored for this path.
        // This avoids "UNIQUE constraint failed: files.path" when find_by_path
        // previously returned None (e.g. unicode NFD/NFC mismatch on macOS).
        conn.execute(
            r#"INSERT INTO files (id, path, file_type, size, modified_at, indexed_at, content_hash)
               SELECT
                   COALESCE((SELECT id FROM files WHERE path = ?2), ?1),
                   ?2, ?3, ?4, ?5, ?6, ?7
               ON CONFLICT(id) DO UPDATE SET
                   path         = excluded.path,
                   file_type    = excluded.file_type,
                   size         = excluded.size,
                   modified_at  = excluded.modified_at,
                   indexed_at   = excluded.indexed_at,
                   content_hash = excluded.content_hash"#,
            params![
                record.id,
                path,
                serde_json::to_string(&record.file_type).unwrap_or_default(),
                record.size as i64,
                record.modified_at.map(|t| t.timestamp()),
                record.indexed_at.map(|t| t.timestamp()),
                record.content_hash,
            ],
        )
        .map_err(|e| Error::storage(e.to_string()))?;

        // Return the id that is now in the DB for this path (may differ from
        // record.id when a path conflict was silently resolved above).
        conn.query_row(
            "SELECT id FROM files WHERE path = ?1",
            params![path],
            |row| row.get::<_, String>(0),
        )
        .map_err(|e| Error::storage(e.to_string()))
    }

    pub fn get(&self, id: &str) -> Result<Option<FileRecord>, Error> {
        let conn = self.db.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT id, path, file_type, size, modified_at, indexed_at, content_hash
                 FROM files WHERE id = ?1",
            )
            .map_err(|e| Error::storage(e.to_string()))?;

        let mut rows = stmt
            .query_map(params![id], row_to_file_record)
            .map_err(|e| Error::storage(e.to_string()))?;

        rows.next()
            .transpose()
            .map_err(|e| Error::storage(e.to_string()))
    }

    pub fn find_by_path(&self, path: &str) -> Result<Option<FileRecord>, Error> {
        let conn = self.db.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT id, path, file_type, size, modified_at, indexed_at, content_hash
                 FROM files WHERE path = ?1",
            )
            .map_err(|e| Error::storage(e.to_string()))?;

        let mut rows = stmt
            .query_map(params![path], row_to_file_record)
            .map_err(|e| Error::storage(e.to_string()))?;

        rows.next()
            .transpose()
            .map_err(|e| Error::storage(e.to_string()))
    }

    pub fn list(&self, limit: usize, offset: usize) -> Result<Vec<FileRecord>, Error> {
        let conn = self.db.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT id, path, file_type, size, modified_at, indexed_at, content_hash
                 FROM files ORDER BY indexed_at DESC LIMIT ?1 OFFSET ?2",
            )
            .map_err(|e| Error::storage(e.to_string()))?;

        let rows = stmt
            .query_map(params![limit as i64, offset as i64], row_to_file_record)
            .map_err(|e| Error::storage(e.to_string()))?;

        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| Error::storage(e.to_string()))
    }

    pub fn delete(&self, id: &str) -> Result<(), Error> {
        let conn = self.db.conn.lock().unwrap();
        conn.execute("DELETE FROM files WHERE id = ?1", params![id])
            .map_err(|e| Error::storage(e.to_string()))?;
        Ok(())
    }

    pub fn count(&self) -> Result<u64, Error> {
        let conn = self.db.conn.lock().unwrap();
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))
            .map_err(|e| Error::storage(e.to_string()))?;
        Ok(n as u64)
    }

    /// Count files whose path starts with `prefix` (e.g. `/home/user/docs/`).
    pub fn count_by_prefix(&self, prefix: &str) -> Result<u64, Error> {
        let conn = self.db.conn.lock().unwrap();
        let pattern = format!("{prefix}%");
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM files WHERE path LIKE ?1",
                params![pattern],
                |r| r.get(0),
            )
            .map_err(|e| Error::storage(e.to_string()))?;
        Ok(n as u64)
    }
}

fn row_to_file_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<FileRecord> {
    let file_type_str: String = row.get(2)?;
    let file_type: FileType = serde_json::from_str(&file_type_str).unwrap_or(FileType::Unknown);

    let modified_at: Option<i64> = row.get(4)?;
    let indexed_at: Option<i64> = row.get(5)?;

    Ok(FileRecord {
        id: row.get(0)?,
        path: PathBuf::from(row.get::<_, String>(1)?),
        file_type,
        size: row.get::<_, i64>(3)? as u64,
        modified_at: modified_at.map(|ts| Utc.timestamp_opt(ts, 0).single().unwrap_or_default()),
        indexed_at: indexed_at.map(|ts| Utc.timestamp_opt(ts, 0).single().unwrap_or_default()),
        content_hash: row.get(6)?,
    })
}
