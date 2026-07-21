use crate::Database;
use mnemosyne_core::Error;
use rusqlite::{params, Connection};
use tracing::info;

/// Row returned by embedding queries:
/// (chunk_id, file_id, chunk_index, content, embedding)
pub type EmbeddingRow = (String, String, i64, String, Vec<f32>);

/// Row returned by sqlite-vector KNN queries:
/// (chunk_id, file_id, chunk_index, content, cosine_distance)
pub type KnnRow = (String, String, i64, String, f32);

pub struct EmbeddingRepo<'a> {
    pub db: &'a Database,
}

impl<'a> EmbeddingRepo<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    /// Store an embedding for a chunk. `embedding` is a slice of f32 values.
    /// When sqlite-vector is loaded, also keeps the corresponding vec0 table in sync.
    pub fn upsert(&self, chunk_id: &str, model_id: &str, embedding: &[f32]) -> Result<(), Error> {
        let bytes = f32_slice_to_bytes(embedding);
        let conn = self.db.conn.lock().unwrap();
        conn.execute(
            r#"INSERT INTO embeddings (chunk_id, model_id, embedding)
               VALUES (?1, ?2, ?3)
               ON CONFLICT(chunk_id) DO UPDATE SET
                   model_id  = excluded.model_id,
                   embedding = excluded.embedding"#,
            params![chunk_id, model_id, bytes],
        )
        .map_err(|e| Error::storage(e.to_string()))?;

        // Keep vec0 in sync when sqlite-vector is available.
        if self.db.sqlite_vector_loaded() {
            let dim = embedding.len();
            let rowid: i64 = conn
                .query_row(
                    "SELECT rowid FROM embeddings WHERE chunk_id = ?1",
                    params![chunk_id],
                    |r| r.get(0),
                )
                .map_err(|e| Error::storage(e.to_string()))?;
            ensure_vec0_for_dim(&conn, dim)?;
            conn.execute(
                &format!(
                    "INSERT OR REPLACE INTO embedding_vec_{dim}(rowid, embedding) \
                     VALUES (?1, ?2)"
                ),
                params![rowid, bytes],
            )
            .map_err(|e| Error::storage(e.to_string()))?;
        }

        Ok(())
    }

    /// Retrieve the embedding for a chunk.
    pub fn get(&self, chunk_id: &str) -> Result<Option<Vec<f32>>, Error> {
        let conn = self.db.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT embedding FROM embeddings WHERE chunk_id = ?1")
            .map_err(|e| Error::storage(e.to_string()))?;

        let mut rows = stmt
            .query_map(params![chunk_id], |row| row.get::<_, Vec<u8>>(0))
            .map_err(|e| Error::storage(e.to_string()))?;

        if let Some(bytes) = rows.next() {
            let bytes = bytes.map_err(|e| Error::storage(e.to_string()))?;
            Ok(Some(bytes_to_f32_vec(&bytes)))
        } else {
            Ok(None)
        }
    }

    /// Return all (chunk_id, file_id, chunk_index, content, embedding) for vector search.
    pub fn all_with_metadata(&self) -> Result<Vec<EmbeddingRow>, Error> {
        let conn = self.db.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT e.chunk_id, dc.file_id, dc.chunk_index, dc.content, e.embedding
                 FROM embeddings e
                 JOIN document_chunks dc ON dc.id = e.chunk_id",
            )
            .map_err(|e| Error::storage(e.to_string()))?;

        let rows: Vec<_> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, Vec<u8>>(4)?,
                ))
            })
            .map_err(|e| Error::storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .map(|(cid, fid, cidx, content, bytes)| {
                let emb = bytes_to_f32_vec(&bytes);
                (cid, fid, cidx, content, emb)
            })
            .collect();

        Ok(rows)
    }

    /// Same as `all_with_metadata` but only returns embeddings whose dimension
    /// matches `dim`.  Required to avoid BERT (384-d) ↔ CLIP (512-d) mismatch.
    pub fn all_with_metadata_by_dim(&self, dim: usize) -> Result<Vec<EmbeddingRow>, Error> {
        let byte_len = (dim * 4) as i64; // each f32 = 4 bytes
        let conn = self.db.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT e.chunk_id, dc.file_id, dc.chunk_index, dc.content, e.embedding
                 FROM embeddings e
                 JOIN document_chunks dc ON dc.id = e.chunk_id
                 WHERE LENGTH(e.embedding) = ?1",
            )
            .map_err(|e| Error::storage(e.to_string()))?;

        let rows: Vec<_> = stmt
            .query_map(rusqlite::params![byte_len], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, Vec<u8>>(4)?,
                ))
            })
            .map_err(|e| Error::storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .map(|(cid, fid, cidx, content, bytes)| {
                (cid, fid, cidx, content, bytes_to_f32_vec(&bytes))
            })
            .collect();

        Ok(rows)
    }

    // ── sqlite-vector helpers ─────────────────────────────────────────────────

    /// Ensure the `embedding_vec_{dim}` vec0 virtual table exists and is
    /// populated from the BLOB embeddings table.  Safe to call on every search:
    /// skips repopulation when counts already match.
    ///
    /// Should only be called when `db.sqlite_vector_loaded()` is true.
    pub fn sync_to_vec0(&self, dim: usize) -> Result<(), Error> {
        let conn = self.db.conn.lock().unwrap();
        ensure_vec0_for_dim(&conn, dim)?;

        let byte_len = (dim * 4) as i64;
        let table = format!("embedding_vec_{dim}");

        let emb_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM embeddings WHERE LENGTH(embedding) = ?1",
                params![byte_len],
                |r| r.get(0),
            )
            .map_err(|e| Error::storage(e.to_string()))?;

        // Query vec0 table count.  If the shadow table is missing (sqlite-vec
        // internal corruption / database was replaced without the extension),
        // DROP and recreate the vec0 table so it starts clean.
        let vec_count: i64 = match conn
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
        {
            Ok(n) => n,
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("rowids") || msg.contains("no such table") {
                    tracing::warn!(
                        "vec0 table {table} is corrupt (missing shadow table): {msg} — recreating"
                    );
                    conn.execute(&format!("DROP TABLE IF EXISTS \"{table}\""), [])
                        .map_err(|e| Error::storage(e.to_string()))?;
                    ensure_vec0_for_dim(&conn, dim)?;
                    0
                } else {
                    return Err(Error::storage(msg));
                }
            }
        };

        if emb_count == vec_count {
            return Ok(()); // already in sync
        }

        info!("Syncing {emb_count} embeddings (dim={dim}) into {table} (had {vec_count})");

        // Full repopulation from the authoritative BLOB store.
        conn.execute(&format!("DELETE FROM {table}"), [])
            .map_err(|e| Error::storage(e.to_string()))?;

        let mut stmt = conn
            .prepare("SELECT rowid, embedding FROM embeddings WHERE LENGTH(embedding) = ?1")
            .map_err(|e| Error::storage(e.to_string()))?;

        let rows: Vec<(i64, Vec<u8>)> = stmt
            .query_map(params![byte_len], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?))
            })
            .map_err(|e| Error::storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        for (rowid, bytes) in rows {
            conn.execute(
                &format!("INSERT OR REPLACE INTO {table}(rowid, embedding) VALUES (?1, ?2)"),
                params![rowid, bytes],
            )
            .map_err(|e| Error::storage(e.to_string()))?;
        }

        Ok(())
    }

    /// Run a KNN search via the sqlite-vector vec0 virtual table.
    ///
    /// Returns up to `limit` rows ordered by ascending cosine distance.
    /// Call `sync_to_vec0` first to ensure the vec0 table is populated.
    pub fn vector_knn(&self, query: &[f32], limit: usize) -> Result<Vec<KnnRow>, Error> {
        let dim = query.len();
        let query_bytes = f32_slice_to_bytes(query);
        let table = format!("embedding_vec_{dim}");
        let conn = self.db.conn.lock().unwrap();

        // Step 1 — KNN search: vec0 returns (rowid, distance) pairs.
        let knn_sql = format!(
            "SELECT rowid, distance FROM {table} \
             WHERE embedding MATCH ?1 \
             ORDER BY distance \
             LIMIT ?2"
        );
        let hits: Vec<(i64, f64)> = {
            let mut stmt = conn
                .prepare(&knn_sql)
                .map_err(|e| Error::storage(e.to_string()))?;
            let collected: Vec<(i64, f64)> = stmt
                .query_map(params![query_bytes, limit as i64], |r| {
                    Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?))
                })
                .map_err(|e| Error::storage(e.to_string()))?
                .filter_map(|r| r.ok())
                .collect();
            collected
        };

        // Step 2 — resolve metadata for each vec0 rowid.
        let mut results = Vec::with_capacity(hits.len());
        for (rowid, l2_dist) in hits {
            // vec0 uses L2 distance for float vectors.
            // For L2-normalised unit vectors:
            //   L2² = 2 * (1 – cos_similarity)
            //   → cos_distance = 1 – cos_similarity = L2² / 2
            // Converting to cosine distance keeps the semantics consistent
            // with the score formula used by callers: score = 1 – cos_distance.
            let cosine_dist = ((l2_dist as f32 * l2_dist as f32) / 2.0).clamp(0.0, 2.0);
            let row: Option<(String, String, i64, String)> = conn
                .query_row(
                    "SELECT e.chunk_id, dc.file_id, dc.chunk_index, dc.content
                     FROM embeddings e
                     JOIN document_chunks dc ON dc.id = e.chunk_id
                     WHERE e.rowid = ?1",
                    params![rowid],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
                )
                .ok();
            if let Some((cid, fid, cidx, content)) = row {
                results.push((cid, fid, cidx, content, cosine_dist));
            }
        }

        Ok(results)
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// Create `embedding_vec_{dim}` as a vec0 virtual table if it doesn't exist.
fn ensure_vec0_for_dim(conn: &Connection, dim: usize) -> Result<(), Error> {
    // dim is always a usize (numeric), so the format string is injection-safe.
    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS embedding_vec_{dim} \
         USING vec0(embedding float[{dim}]);"
    ))
    .map_err(|e| Error::storage(format!("failed to create vec0 table dim={dim}: {e}")))
}
