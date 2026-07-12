//! Integration tests for the full search pipeline.
//!
//! These tests use an in-memory (temp-dir) database and verify:
//! - Files are indexed correctly
//! - FTS keyword search returns relevant results
//! - Vector search works with the stub embedder
//! - Change detection (same hash → skip)

use mnemosyne_core::types::{SearchMode, SearchQuery};
use mnemosyne_retrieval::SearchEngine;
use std::io::Write;
use tempfile::TempDir;

/// Helper: create engine pointing to a temp database.
async fn make_engine(tmp: &TempDir) -> SearchEngine {
    SearchEngine::builder()
        .db_path(tmp.path().join("test.sqlite"))
        .build()
        .await
        .expect("SearchEngine::build")
}

/// Helper: write a text file into a temp directory.
fn write_file(dir: &TempDir, name: &str, content: &str) -> std::path::PathBuf {
    let path = dir.path().join(name);
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(content.as_bytes()).unwrap();
    path
}

// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_index_and_stats() {
    let tmp = TempDir::new().unwrap();
    let engine = make_engine(&tmp).await;

    write_file(&tmp, "hello.txt", "Hello world, this is a test document.");
    write_file(
        &tmp,
        "rust.md",
        "Rust is a systems programming language focused on safety.",
    );

    let stats = engine
        .index_directory(tmp.path())
        .await
        .expect("index_directory");

    assert_eq!(stats.total_files, 2, "expected 2 files indexed");
    assert!(stats.total_chunks >= 2, "expected at least 2 chunks");
}

#[tokio::test]
async fn test_keyword_search() {
    let tmp = TempDir::new().unwrap();
    let engine = make_engine(&tmp).await;

    write_file(
        &tmp,
        "doc1.txt",
        "The quick brown fox jumps over the lazy dog.",
    );
    write_file(
        &tmp,
        "doc2.txt",
        "Rust programming language memory safety features.",
    );
    write_file(
        &tmp,
        "doc3.txt",
        "Machine learning and deep learning neural networks.",
    );

    engine.index_directory(tmp.path()).await.unwrap();

    let results = engine
        .search(SearchQuery {
            text: "Rust memory safety".to_string(),
            mode: SearchMode::Keyword,
            limit: 5,
            ..Default::default()
        })
        .await
        .expect("search");

    assert!(!results.is_empty(), "keyword search should return results");
    let top = &results[0];
    assert!(
        top.file_record.path.file_name().unwrap() == "doc2.txt",
        "top result should be doc2.txt, got: {:?}",
        top.file_record.path.file_name()
    );
}

#[tokio::test]
async fn test_vector_search_returns_results() {
    let tmp = TempDir::new().unwrap();
    let engine = make_engine(&tmp).await;

    write_file(&tmp, "a.txt", "apple banana cherry");
    write_file(&tmp, "b.txt", "dog cat mouse");

    engine.index_directory(tmp.path()).await.unwrap();

    let results = engine
        .search(SearchQuery {
            text: "fruits like apple".to_string(),
            mode: SearchMode::Vector,
            limit: 5,
            ..Default::default()
        })
        .await
        .expect("vector search");

    assert!(!results.is_empty(), "vector search should return results");
}

#[tokio::test]
async fn test_hybrid_search() {
    let tmp = TempDir::new().unwrap();
    let engine = make_engine(&tmp).await;

    for i in 0..5 {
        write_file(
            &tmp,
            &format!("file{i}.txt"),
            &format!("Document {i} about topic {i} with unique content identifier {i}."),
        );
    }

    engine.index_directory(tmp.path()).await.unwrap();

    let results = engine
        .search(SearchQuery {
            text: "unique content identifier".to_string(),
            mode: SearchMode::Hybrid,
            limit: 5,
            ..Default::default()
        })
        .await
        .expect("hybrid search");

    assert!(!results.is_empty(), "hybrid search should return results");
}

#[tokio::test]
async fn test_incremental_index_skips_unchanged() {
    let tmp = TempDir::new().unwrap();
    let engine = make_engine(&tmp).await;

    write_file(&tmp, "stable.txt", "This content does not change.");

    // First index.
    let stats1 = engine.index_directory(tmp.path()).await.unwrap();
    assert_eq!(stats1.total_files, 1);

    // Second index — same file, should be skipped (total_files = 0 new).
    let stats2 = engine.index_directory(tmp.path()).await.unwrap();
    assert_eq!(stats2.total_files, 0, "unchanged file should be skipped");
}

#[tokio::test]
async fn test_list_and_remove() {
    let tmp = TempDir::new().unwrap();
    let engine = make_engine(&tmp).await;

    write_file(&tmp, "x.txt", "content x");
    write_file(&tmp, "y.txt", "content y");

    engine.index_directory(tmp.path()).await.unwrap();

    let files = engine.list_files(10, 0).await.unwrap();
    assert_eq!(files.len(), 2);

    let id = files[0].id.clone();
    engine.remove_file(&id).await.unwrap();

    let files_after = engine.list_files(10, 0).await.unwrap();
    assert_eq!(files_after.len(), 1);
}

#[tokio::test]
async fn test_stats_counts() {
    let tmp = TempDir::new().unwrap();
    let engine = make_engine(&tmp).await;

    // Create a large file that will be split into multiple chunks.
    let large = "word ".repeat(1000); // 5000 chars, > 1500 chunk size
    write_file(&tmp, "large.txt", &large);

    engine.index_directory(tmp.path()).await.unwrap();

    let stats = engine.get_stats().await.unwrap();
    assert_eq!(stats.total_files, 1);
    assert!(
        stats.total_chunks >= 3,
        "large file should produce multiple chunks"
    );
}
