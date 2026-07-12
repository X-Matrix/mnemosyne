use crate::Database;
use mnemosyne_core::Error;
use rusqlite::params;

pub struct EmbeddingRepo<'a> {
    pub db: &'a Database,
}

impl<'a> EmbeddingRepo<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    /// Store an embedding for a chunk. `embedding` is a slice of f32 values.
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
    pub fn all_with_metadata(&self) -> Result<Vec<(String, String, i64, String, Vec<f32>)>, Error> {
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
    pub fn all_with_metadata_by_dim(
        &self,
        dim: usize,
    ) -> Result<Vec<(String, String, i64, String, Vec<f32>)>, Error> {
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
