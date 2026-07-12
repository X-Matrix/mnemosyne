use crate::{ann::AnnCache, cosine, rrf};
use mnemosyne_core::{
    traits::SearchIndex,
    types::{Embedding, IndexedChunk, MatchType, SearchQuery, SearchResult},
    Error,
};
use mnemosyne_storage::{ChunkRepo, Database, EmbeddingRepo, FileRepo};
use async_trait::async_trait;
use rusqlite::params;

/// Hybrid search index backed by SQLite + optional HNSW ANN.
pub struct HybridIndex {
    db: Database,
    ann: AnnCache,
}

impl HybridIndex {
    pub fn new(db: Database) -> Self {
        Self { db, ann: AnnCache::new() }
    }
}

#[async_trait]
impl SearchIndex for HybridIndex {
    async fn upsert(&self, chunk: &IndexedChunk) -> mnemosyne_core::Result<()> {
        let db = self.db.clone();
        let chunk = chunk.clone();
        tokio::task::spawn_blocking(move || -> mnemosyne_core::Result<()> {
            let chunk_repo = ChunkRepo::new(&db);
            chunk_repo.upsert(
                &chunk.chunk_id,
                &chunk.file_id,
                chunk.chunk_index,
                &chunk.content,
            )?;
            if let Some(ref embedding) = chunk.embedding {
                let emb_repo = EmbeddingRepo::new(&db);
                emb_repo.upsert(&chunk.chunk_id, "default", embedding)?;
            }
            Ok(())
        })
        .await
        .map_err(|e| Error::index(e.to_string()))??;

        // Invalidate ANN cache so it rebuilds on next search.
        self.ann.invalidate().await;
        Ok::<(), mnemosyne_core::Error>(())
    }

    async fn remove_file(&self, file_id: &str) -> mnemosyne_core::Result<()> {
        let db = self.db.clone();
        let file_id = file_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = db.conn.lock().unwrap();
            conn.execute("DELETE FROM files WHERE id = ?1", params![file_id])
                .map_err(|e| Error::storage(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(|e| Error::index(e.to_string()))?
    }

    async fn vector_search(
        &self,
        query_embedding: &Embedding,
        limit: usize,
    ) -> mnemosyne_core::Result<Vec<SearchResult>> {
        let db = self.db.clone();
        let query_emb = query_embedding.clone();
        let query_dim = query_emb.len(); // BERT=384, CLIP=512 — never mix!

        // ── Try ANN path first ────────────────────────────────────────────────
        // Build (or reuse) an index containing ONLY same-dimensional embeddings.
        let db_for_load = db.clone();
        if let Some(ann_idx) = self
            .ann
            .get_or_build(query_dim, async move {
                Ok(EmbeddingRepo::new(&db_for_load)
                    .all_with_metadata_by_dim(query_dim)?
                    .into_iter()
                    .map(|(cid, _fid, _cidx, _content, emb)| (cid, emb))
                    .collect())
            })
            .await?
        {
            let file_repo = mnemosyne_storage::FileRepo::new(&db);
            let hits = ann_idx.search(&query_emb, limit);
            let mut results = Vec::with_capacity(hits.len());

            // Resolve FileRecord and snippet for ANN hits.
            let conn = db.conn.lock().unwrap();
            for (chunk_id, dist) in hits {
                let score = 1.0 - dist; // distance → similarity
                let row: Option<(String, i64, String)> = {
                    let mut st = conn
                        .prepare("SELECT file_id, chunk_index, content FROM document_chunks WHERE id = ?1")
                        .map_err(|e| Error::storage(e.to_string()))?;
                    st.query_row(rusqlite::params![chunk_id], |r| {
                        Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, String>(2)?))
                    })
                    .ok()
                };
                if let Some((fid, cidx, content)) = row {
                    if let Some(fr) = file_repo.get(&fid)? {
                        results.push(SearchResult {
                            file_record: fr,
                            score,
                            snippet: Some(content.chars().take(200).collect()),
                            match_type: MatchType::Vector,
                            chunk_index: cidx as usize,
                        });
                    }
                }
            }
            return Ok(results);
        }

        // ── Brute-force fallback ──────────────────────────────────────────────
        tokio::task::spawn_blocking(move || -> mnemosyne_core::Result<Vec<SearchResult>> {
            // Only load embeddings with the same dimension as the query.
            let all_rows = EmbeddingRepo::new(&db).all_with_metadata_by_dim(query_dim)?;
            let file_repo = FileRepo::new(&db);

            let mut scored: Vec<(String, String, usize, String, f32)> = all_rows
                .into_iter()
                .map(|(cid, fid, cidx, content, emb)| {
                    let sim = cosine::cosine_similarity(&query_emb, &emb);
                    (cid, fid, cidx as usize, content, cosine::similarity_to_score(sim))
                })
                .collect();

            scored.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(limit);

            let mut results = Vec::with_capacity(scored.len());
            for (_, file_id, chunk_index, content, score) in scored {
                if let Some(fr) = file_repo.get(&file_id)? {
                    results.push(SearchResult {
                        file_record: fr,
                        score,
                        snippet: Some(content.chars().take(200).collect()),
                        match_type: MatchType::Vector,
                        chunk_index,
                    });
                }
            }
            Ok(results)
        })
        .await
        .map_err(|e| Error::index(e.to_string()))?
    }

    async fn keyword_search(
        &self,
        query: &str,
        limit: usize,
    ) -> mnemosyne_core::Result<Vec<SearchResult>> {
        let db = self.db.clone();
        let query = query.to_string();

        tokio::task::spawn_blocking(move || -> mnemosyne_core::Result<Vec<SearchResult>> {
            // Step 1: FTS search via storage API.
            let kw_rows = ChunkRepo::new(&db).fts_search(&query, limit)?;
            let file_repo = FileRepo::new(&db);

            // Step 2: resolve FileRecord.
            let mut results = Vec::with_capacity(kw_rows.len());
            for (_, file_id, chunk_index, content, bm25) in kw_rows {
                if let Some(fr) = file_repo.get(&file_id)? {
                    let score = (-bm25 as f32).max(0.0) / 10.0;
                    results.push(SearchResult {
                        file_record: fr,
                        score,
                        snippet: Some(content.chars().take(200).collect()),
                        match_type: MatchType::Keyword,
                        chunk_index: chunk_index as usize,
                    });
                }
            }
            Ok(results)
        })
        .await
        .map_err(|e| Error::index(e.to_string()))?
    }

    async fn hybrid_search(
        &self,
        query: &SearchQuery,
        query_embedding: &Embedding,
    ) -> mnemosyne_core::Result<Vec<SearchResult>> {
        let inner_limit = query.limit * 3;

        let (vec_res, kw_res) = tokio::join!(
            self.vector_search(query_embedding, inner_limit),
            self.keyword_search(&query.text, inner_limit),
        );

        let vec_results = vec_res?;
        let kw_results = kw_res?;

        let vec_ranked: Vec<(String, f32)> = vec_results
            .iter()
            .map(|r| (format!("{}:{}", r.file_record.id, r.chunk_index), r.score))
            .collect();
        let kw_ranked: Vec<(String, f32)> = kw_results
            .iter()
            .map(|r| (format!("{}:{}", r.file_record.id, r.chunk_index), r.score))
            .collect();

        let fused = rrf::fuse(&vec_ranked, &kw_ranked, query.limit,
            query.vector_weight, query.keyword_weight);

        let mut result_map: std::collections::HashMap<String, SearchResult> =
            std::collections::HashMap::new();
        for r in vec_results.into_iter().chain(kw_results.into_iter()) {
            let key = format!("{}:{}", r.file_record.id, r.chunk_index);
            result_map.entry(key).or_insert(r);
        }

        let final_results: Vec<SearchResult> = fused
            .into_iter()
            .filter_map(|(key, score)| {
                let mut r = result_map.remove(&key)?;
                r.score = score;
                r.match_type = MatchType::Hybrid;
                Some(r)
            })
            .collect();

        Ok(final_results)
    }
}


