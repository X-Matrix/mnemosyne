//! Cosine similarity helper.

/// Compute the cosine similarity between two L2-normalised vectors.
///
/// Both vectors must have the same length. Returns a value in `[-1, 1]`.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "embedding dimension mismatch");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Score to rank: higher is better.
///
/// Since embeddings are L2-normalised, cosine similarity equals dot product.
/// We shift to `[0, 1]` range: `(sim + 1) / 2`.
pub fn similarity_to_score(sim: f32) -> f32 {
    (sim + 1.0) / 2.0
}
