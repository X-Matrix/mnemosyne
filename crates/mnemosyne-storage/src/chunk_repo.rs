use crate::Database;
use mnemosyne_core::{types::ParsedContent, Error};
use rusqlite::params;

/// Row returned by `fts_search`: (chunk_id, file_id, chunk_index, content, bm25_score)
pub type FtsRow = (String, String, i64, String, f64);

pub struct ChunkRepo<'a> {
    pub db: &'a Database,
}

#[derive(Debug)]
pub struct ChunkRow {
    pub id: String,
    pub file_id: String,
    pub chunk_index: usize,
    pub content: ParsedContent,
    pub rowid: i64,
}

impl<'a> ChunkRepo<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    pub fn upsert(
        &self,
        chunk_id: &str,
        file_id: &str,
        chunk_index: usize,
        content: &ParsedContent,
    ) -> Result<i64, Error> {
        let conn = self.db.conn.lock().unwrap();
        let kind = match content {
            ParsedContent::Text { .. } => "text",
            ParsedContent::Image { .. } => "image",
            ParsedContent::AudioTranscript { .. } => "audio_transcript",
            ParsedContent::VideoKeyframe { .. } => "video_keyframe",
        };
        let text = content.as_text().to_owned();
        conn.execute(
            r#"INSERT INTO document_chunks (id, file_id, chunk_index, kind, content)
               VALUES (?1, ?2, ?3, ?4, ?5)
               ON CONFLICT(file_id, chunk_index) DO UPDATE SET
                   id      = excluded.id,
                   kind    = excluded.kind,
                   content = excluded.content"#,
            params![chunk_id, file_id, chunk_index as i64, kind, text],
        )
        .map_err(|e| Error::storage(e.to_string()))?;

        let rowid = conn.last_insert_rowid();
        Ok(rowid)
    }

    pub fn get_by_file(&self, file_id: &str) -> Result<Vec<ChunkRow>, Error> {
        let conn = self.db.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT id, file_id, chunk_index, kind, content, rowid
                 FROM document_chunks WHERE file_id = ?1 ORDER BY chunk_index",
            )
            .map_err(|e| Error::storage(e.to_string()))?;

        let rows = stmt
            .query_map(params![file_id], |row| {
                let kind: String = row.get(3)?;
                let text: String = row.get(4)?;
                let content = match kind.as_str() {
                    "image" => ParsedContent::Image {
                        caption: text,
                        tags: vec![],
                    },
                    "audio_transcript" => ParsedContent::AudioTranscript {
                        transcript: text,
                        language: None,
                    },
                    "video_keyframe" => ParsedContent::VideoKeyframe {
                        timestamp_secs: 0.0,
                        description: text,
                    },
                    _ => ParsedContent::Text { text },
                };
                Ok(ChunkRow {
                    id: row.get(0)?,
                    file_id: row.get(1)?,
                    chunk_index: row.get::<_, i64>(2)? as usize,
                    content,
                    rowid: row.get(5)?,
                })
            })
            .map_err(|e| Error::storage(e.to_string()))?;

        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| Error::storage(e.to_string()))
    }

    pub fn fts_search(&self, query: &str, limit: usize) -> Result<Vec<FtsRow>, Error> {
        let conn = self.db.conn.lock().unwrap();
        let char_count = query.chars().count();

        // FTS5 trigram tokenizer requires >= 3 characters.
        // For shorter queries we fall back to a LIKE scan.
        if char_count >= 3 {
            let mut stmt = conn
                .prepare(
                    r#"SELECT dc.id, dc.file_id, dc.chunk_index, dc.content,
                              bm25(fts_chunks) AS bm25_score
                       FROM fts_chunks
                       JOIN document_chunks dc ON dc.rowid = fts_chunks.rowid
                       WHERE fts_chunks MATCH ?1
                       ORDER BY bm25_score
                       LIMIT ?2"#,
                )
                .map_err(|e| Error::storage(e.to_string()))?;

            let rows: Vec<(String, String, i64, String, f64)> = stmt
                .query_map(params![query, limit as i64], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, f64>(4)?,
                    ))
                })
                .map_err(|e| Error::storage(e.to_string()))?
                .filter_map(|r| r.ok())
                .collect();
            Ok(rows)
        } else {
            // Short query (1-2 chars): LIKE scan on raw content.
            let pattern = format!("%{query}%");
            let mut stmt = conn
                .prepare(
                    "SELECT id, file_id, chunk_index, content, -1.0 \
                     FROM document_chunks WHERE content LIKE ?1 LIMIT ?2",
                )
                .map_err(|e| Error::storage(e.to_string()))?;

            let rows: Vec<(String, String, i64, String, f64)> = stmt
                .query_map(params![pattern, limit as i64], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, f64>(4)?,
                    ))
                })
                .map_err(|e| Error::storage(e.to_string()))?
                .filter_map(|r| r.ok())
                .collect();
            Ok(rows)
        }
    }

    pub fn count(&self) -> Result<u64, Error> {
        let conn = self.db.conn.lock().unwrap();
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM document_chunks", [], |r| r.get(0))
            .map_err(|e| Error::storage(e.to_string()))?;
        Ok(n as u64)
    }
}
