//! Reciprocal Rank Fusion (RRF) for merging result lists.
//!
//! Formula: `score(d) = Σ 1 / (k + rank(d))`  where k = 60 (standard).

use std::collections::HashMap;

const RRF_K: f32 = 60.0;

/// Merge two ranked lists of `(chunk_id, score)` using weighted RRF.
///
/// Formula: `score(d) = v_w/( k+rank_vec(d) ) + k_w/(k+rank_kw(d))`
/// `v_w` / `k_w` are the caller-supplied weights (default 1.0 each).
pub fn fuse(
    vector_results:  &[(String, f32)],
    keyword_results: &[(String, f32)],
    limit:          usize,
    vector_weight:  f32,
    keyword_weight: f32,
) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f32> = HashMap::new();

    for (rank, (id, _)) in vector_results.iter().enumerate() {
        *scores.entry(id.clone()).or_default() +=
            vector_weight / (RRF_K + rank as f32 + 1.0);
    }

    for (rank, (id, _)) in keyword_results.iter().enumerate() {
        *scores.entry(id.clone()).or_default() +=
            keyword_weight / (RRF_K + rank as f32 + 1.0);
    }

    let mut ranked: Vec<(String, f32)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(limit);
    ranked
}
