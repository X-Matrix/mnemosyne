//! Score-aware Reciprocal Rank Fusion (RRF) for merging result lists.
//!
//! Classic RRF uses only rank positions and discards original scores, producing
//! tiny values like 3% that are not meaningful to users.  This implementation
//! multiplies each rank term by the document's original similarity score so
//! that the final output is normalised to \[0, 1\] and reflects the underlying
//! relevance.
//!
//! Formula:
//!   raw(d)  = Σ weight_i * score_i(d) / (k + rank_i(d))
//!   ceil    = Σ weight_i / (k + 1)          ← theoretical max (rank=1, score=1)
//!   hybrid  = clamp(raw / ceil, 0, 1)
//!
//! Example: vector=0.47, keyword=0.09, both rank-1, weights=1.0:
//!   raw  = 0.47/61 + 0.09/61 = 0.00918
//!   ceil = 2.0/61             = 0.03279
//!   hybrid = 0.00918 / 0.03279 ≈ 0.28  (28 %)

use std::collections::HashMap;

const RRF_K: f32 = 60.0;

/// Merge two ranked lists of `(chunk_id, score)` using score-aware weighted RRF.
///
/// The returned scores are normalised to `(0, 1]` so they can be displayed as
/// meaningful percentages.  A document that ranks first with score 1.0 in every
/// input list receives a normalised score of 1.0; a document that appears only
/// in the vector list is penalised relative to one that matches both.
pub fn fuse(
    vector_results: &[(String, f32)],
    keyword_results: &[(String, f32)],
    limit: usize,
    vector_weight: f32,
    keyword_weight: f32,
) -> Vec<(String, f32)> {
    let mut raw: HashMap<String, f32> = HashMap::new();

    for (rank, (id, score)) in vector_results.iter().enumerate() {
        *raw.entry(id.clone()).or_default() += vector_weight * score / (RRF_K + rank as f32 + 1.0);
    }
    for (rank, (id, score)) in keyword_results.iter().enumerate() {
        *raw.entry(id.clone()).or_default() += keyword_weight * score / (RRF_K + rank as f32 + 1.0);
    }

    // Theoretical ceiling: rank=1 (denominator = k+1) with score=1.0 in every list.
    let ceil = (vector_weight + keyword_weight) / (RRF_K + 1.0);
    // Guard against degenerate calls (both weights zero).
    let ceil = if ceil > 0.0 { ceil } else { 1.0 };

    let mut ranked: Vec<(String, f32)> = raw
        .into_iter()
        .map(|(id, s)| (id, (s / ceil).clamp(0.0, 1.0)))
        .collect();

    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(limit);
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(s: &str) -> String {
        s.to_string()
    }

    #[test]
    fn score_aware_gives_interpretable_percentage() {
        // vector=47%, keyword=9%, both rank-1, equal weights → expect ~28%
        let vec = vec![(id("a"), 0.47_f32)];
        let kw = vec![(id("a"), 0.09_f32)];
        let result = fuse(&vec, &kw, 10, 1.0, 1.0);
        assert_eq!(result.len(), 1);
        let score = result[0].1;
        // Should be (0.47+0.09)/61 / (2/61) = 0.56/2 = 0.28
        assert!(
            (score - 0.28).abs() < 0.01,
            "expected ~0.28, got {score:.4}"
        );
    }

    #[test]
    fn perfect_match_reaches_one() {
        let vec = vec![(id("a"), 1.0_f32)];
        let kw = vec![(id("a"), 1.0_f32)];
        let result = fuse(&vec, &kw, 10, 1.0, 1.0);
        assert!(
            (result[0].1 - 1.0).abs() < 1e-6,
            "rank-1 perfect-match must be 1.0"
        );
    }

    #[test]
    fn only_in_vector_is_penalised() {
        let vec = vec![(id("a"), 1.0_f32)];
        let kw: Vec<(String, f32)> = vec![];
        let result = fuse(&vec, &kw, 10, 1.0, 1.0);
        let score = result[0].1;
        // 1.0/61 / (2/61) = 0.5  — penalised for missing keyword signal
        assert!((score - 0.5).abs() < 0.01, "expected ~0.50, got {score:.4}");
    }

    #[test]
    fn higher_score_beats_better_rank() {
        // "b" is rank-1 with 0.30; "a" is rank-2 with 0.90 → "a" should win
        let vec = vec![(id("b"), 0.30_f32), (id("a"), 0.90_f32)];
        let kw: Vec<(String, f32)> = vec![];
        let result = fuse(&vec, &kw, 10, 1.0, 1.0);
        assert_eq!(
            result[0].0, "a",
            "high-score rank-2 must beat low-score rank-1"
        );
    }

    #[test]
    fn order_is_descending() {
        let vec = vec![(id("a"), 0.9), (id("b"), 0.5), (id("c"), 0.2)];
        let kw = vec![(id("b"), 0.8), (id("a"), 0.3)];
        let result = fuse(&vec, &kw, 10, 1.0, 1.0);
        for w in result.windows(2) {
            assert!(w[0].1 >= w[1].1, "scores must be non-increasing");
        }
    }
}
