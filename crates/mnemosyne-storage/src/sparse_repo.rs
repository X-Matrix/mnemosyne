//! Storage and retrieval for BGE-M3 sparse (lexical) embeddings.

use crate::Database;
use mnemosyne_core::Error;
use rusqlite::params;
use std::collections::HashMap;

pub struct SparseEmbeddingRepo<'a> {
    pub db: &'a Database,
}

impl<'a> SparseEmbeddingRepo<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    /// Persist `{token_id: weight}` for a chunk.
    pub fn upsert(
        &self,
        chunk_id: &str,
        model_id: &str,
        weights: &HashMap<u32, f32>,
    ) -> Result<(), Error> {
        let json = serde_json::to_string(weights)
            .map_err(|e| Error::storage(format!("sparse json: {e}")))?;
        let conn = self.db.conn.lock().unwrap();
        conn.execute(
            r#"INSERT INTO sparse_embeddings (chunk_id, model_id, data)
               VALUES (?1, ?2, ?3)
               ON CONFLICT(chunk_id) DO UPDATE SET
                   model_id = excluded.model_id,
                   data     = excluded.data"#,
            params![chunk_id, model_id, json],
        )
        .map_err(|e| Error::storage(e.to_string()))?;
        Ok(())
    }

    /// Compute dot-product matching score between `query` and all stored sparse
    /// embeddings.  Returns `(chunk_id, score)` pairs sorted descending.
    ///
    /// Only chunks with a non-zero intersection with the query are returned.
    pub fn search(
        &self,
        query: &HashMap<u32, f32>,
        limit: usize,
    ) -> Result<Vec<(String, f32)>, Error> {
        if query.is_empty() {
            return Ok(vec![]);
        }

        let conn = self.db.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT chunk_id, data FROM sparse_embeddings")
            .map_err(|e| Error::storage(e.to_string()))?;

        let mut scored: Vec<(String, f32)> = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| Error::storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .filter_map(|(chunk_id, json)| {
                let doc: HashMap<u32, f32> = serde_json::from_str(&json).ok()?;
                // Sparse dot product
                let score: f32 = query
                    .iter()
                    .filter_map(|(tid, qw)| doc.get(tid).map(|dw| qw * dw))
                    .sum();
                if score > 0.0 {
                    Some((chunk_id, score))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        Ok(scored)
    }
}
