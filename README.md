<div align="center">
  <img src="src-tauri/icons/128x128.png" alt="Mnemosyne" width="80" />
  <h1>Mnemosyne</h1>
  <p><strong>An intelligent, local-first file search and analysis system.</strong></p>
  <p>
    <a href="https://github.com/mnemosyne-proj/mnemosyne/actions"><img src="https://github.com/mnemosyne-proj/mnemosyne/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <img src="https://img.shields.io/badge/Rust-1.75+-orange?logo=rust" alt="Rust">
    <img src="https://img.shields.io/badge/Tauri-2.x-blue?logo=tauri" alt="Tauri">
    <img src="https://img.shields.io/badge/SQLite-3.x-green?logo=sqlite" alt="SQLite">
    <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT">
  </p>
</div>

---

## About

Mnemosyne provides intelligent search and analysis for your local files, featuring:

- **Hybrid Search**: Combines vector-based semantic search with FTS5 keyword search, using Reciprocal Rank Fusion (RRF) to merge and rank results.
- **Multi-Format Support**: Handles text (txt, md, csv, json...), PDFs, images, audio, and video files.
- **Multiple UIs**: Access it via a desktop GUI (Tauri 2), a command-line interface (CLI), or a REST API.
- **Real-time Updates**: A file watcher automatically detects changes and performs incremental indexing.
- **Zero External Services**: All data is stored locally in a single SQLite file, requiring no separate database server.

---

## Features

| Feature | Status |
|---|---|
| Text File Indexing (txt/md/csv/json/py/rs...) | ✅ Fully Implemented |
| PDF Text Extraction | ✅ Fully Implemented |
| Image Metadata Extraction (Dimensions) | ✅ Fully Implemented |
| Audio/Video File Indexing | ✅ Stub (Path + Metadata) |
| FTS5 Keyword Search (BM25) | ✅ Fully Implemented |
| Vector Similarity Search | ✅ Cosine Similarity (Brute-force) |
| RRF Hybrid Search | ✅ Fully Implemented |
| Incremental Indexing (SHA-256 Change Detection) | ✅ Fully Implemented |
| Real-time File Watcher | ✅ Fully Implemented |
| Background Periodic Rescan | ✅ Fully Implemented |
| REST API Server | ✅ Fully Implemented |
| CLI Tool | ✅ Fully Implemented |
| Desktop GUI (Tauri 2) | ✅ Skeleton Implemented |
| Real BERT Inference (via `candle`) | ✅ `candle-backend` feature |
| CLIP Image Embeddings | ✅ `clip-backend` feature |
| Whisper Audio Transcription | ✅ `whisper-backend` feature |
| ANN Vector Search (via `sqlite-vector`) | ✅ Dynamic Loading + HNSW Fallback |

---

## Architecture

```
┌─────────────────────────────────────────────┐
│                 UI Layer                    │
│  [Tauri2 GUI]  [CLI]  [REST API / axum]     │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│         mnemosyne-retrieval                  │
│   SearchEngine (Indexing + Search Facade)    │
│   BackgroundIndexer │ FileWatcher            │
└──┬────────────────┬────────────────┬────────┘
   │                │                │
┌──▼───┐   ┌────────▼──┐   ┌────────▼──────┐
│model │   │  parser   │   │    index      │
│BERT  │   │Text/PDF/  │   │HybridIndex    │
│stub  │   │Image/...  │   │Vector + FTS5  │
└──────┘   └───────────┘   └──────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │   storage        │
                          │SQLite (bundled)  │
                          │files/chunks/embs │
                          └─────────────────┘
```

### Crate Dependency Graph

```
mnemosyne-core          ← Base Types/Traits/Errors
  ← mnemosyne-storage   ← SQLite CRUD Operations
  ← mnemosyne-model     ← Text Embeddings (stub/candle)
  ← mnemosyne-parser    ← File Parsing Plugins
  ← mnemosyne-index     ← Hybrid Search Engine
    ← mnemosyne-retrieval  ← SearchEngine Facade
      ← cli / api / src-tauri
```

---

## Installation

### Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Rust | ≥ 1.75 | `rustup update stable` |
| Tauri CLI | 2.x | `cargo install tauri-cli` |
| Node.js | ≥ 18 | For building the GUI frontend |
| Git LFS | latest | `git lfs install` |

On macOS, you also need Xcode Command Line Tools:
```bash
xcode-select --install
```

### Building

```bash
git clone <repo-url> mnemosyne
cd mnemosyne

# Pull LFS files
git lfs pull

# Build CLI + REST API (fastest, no Node.js required)
cargo build --release -p mnemosyne-cli -p mnemosyne-api

# Build Desktop GUI
npm install
cargo tauri build

# Enable full ML backend (BERT + CLIP + Whisper)
cargo build --release -p mnemosyne-cli \
  --features "mnemosyne-model/full"

# Enable features individually
cargo build --release -p mnemosyne-cli \
  --features "mnemosyne-model/candle-backend"   # BERT only

cargo build --release -p mnemosyne-cli \
  --features "mnemosyne-model/clip-backend"     # BERT + CLIP

cargo build --release -p mnemosyne-cli \
  --features "mnemosyne-model/whisper-backend"  # BERT + Whisper
```

Build artifacts are located in `target/release/`:
- `mnemosyne` — The CLI tool
- `mnemosyne-server` — The REST API server

---

## Usage

### CLI

```bash
# Index a directory (recursively)
mnemosyne index ~/Documents

# Search (hybrid mode)
mnemosyne search "Rust memory safety"

# Keyword-only search
mnemosyne search "async trait" --mode keyword --limit 20

# Vector-only search
mnemosyne search "machine learning papers" --mode vector

# Output as JSON
mnemosyne search "tokio runtime" --json

# Watch a directory for real-time updates
mnemosyne watch ~/Documents

# View statistics
mnemosyne stats

# List indexed files
mnemosyne list --limit 50

# Remove a file record
mnemosyne remove <file-id>

# Start the REST API server (defaults to port 8080)
mnemosyne serve --port 8080

# Specify a custom database path
mnemosyne --db /path/to/db.sqlite index ~/Documents
```

### REST API

Start the server:
```bash
# Configure via environment variables
MNEMOSYNE_PORT=8080 MNEMOSYNE_DB=~/.mnemosyne/db.sqlite mnemosyne-server
```

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/search` | Search for files |
| `POST` | `/api/index` | Index a directory |
| `GET` | `/api/stats` | Get statistics |
| `GET` | `/api/files` | List files (?limit&offset) |
| `DELETE`| `/api/files/:id` | Delete a file record |
| `GET` | `/api/models` | List downloaded models |
| `POST` | `/api/models/download`| Download a model |
| `GET` | `/health` | Health check |

**Example (cURL):**
```bash
# Hybrid search
curl -X POST http://localhost:8080/api/search \
  -H "Content-Type: application/json" \
  -d '{"text": "async programming", "mode": "hybrid", "limit": 10}'

# Index a directory
curl -X POST http://localhost:8080/api/index \
  -H "Content-Type: application/json" \
  -d '{"path": "/home/user/Documents"}'

# Get stats
curl http://localhost:8080/api/stats
```

**SearchQuery Payload:**
```json
{
  "text": "Your query text",
  "mode": "hybrid",      // "hybrid" | "vector" | "keyword"
  "limit": 20,
  "offset": 0,
  "file_types": null     // null for all, or ["text","image","audio","video"]
}
```

### Desktop GUI

```bash
npm run dev   # Development mode (hot-reloading)
npm run build # Build the application
```

GUI Features:
- Search bar (press Enter to search, with mode switching)
- "Index Directory" button to add new paths
- Left panel: Real-time stats and file type filters
- Results list: File path, relevant snippets, and similarity scores

---

## Configuration

Mnemosyne is configured via environment variables and command-line arguments. No config file is used.

| Variable | Default | Description |
|---|---|---|
| `MNEMOSYNE_DB` | `~/.mnemosyne/db.sqlite` | Database path |
| `MNEMOSYNE_PORT` | `8080` | API server port |
| `RUST_LOG` | `warn,mnemosyne=info` | Logging level |

---

## Embedding Models

### Default (Stub Mode)

No model download is required. This mode uses deterministic, hash-based pseudo-vectors. It's suitable for development and testing but does **not** perform true semantic search.

### Text BERT (`candle-backend`)

Enable at compile time:
```bash
cargo build --release -p mnemosyne-cli --features mnemosyne-model/candle-backend
```

The model (~90 MB) is downloaded automatically on first run:
```bash
mnemosyne model-download sentence-transformers/all-MiniLM-L6-v2
```

### Image CLIP (`clip-backend`)

```bash
cargo build --release -p mnemosyne-cli --features mnemosyne-model/clip-backend
mnemosyne model-download openai/clip-vit-base-patch32
```

After indexing an image directory, you can search using natural language:
```bash
mnemosyne index ~/Pictures
mnemosyne search "sunset over mountains"
```

### Audio Whisper (`whisper-backend`)

```bash
cargo build --release -p mnemosyne-cli --features mnemosyne-model/whisper-backend
mnemosyne model-download openai/whisper-tiny
```

Supports WAV files (16-bit or float, automatically resampled to 16 kHz).

```bash
mnemosyne index ~/Recordings   # Transcribes and indexes automatically
mnemosyne search "project meeting notes"
```

### ANN Vector Search (sqlite-vector + HNSW)

When the index size exceeds **2,000 chunks**, the system automatically switches to HNSW (Hierarchical Navigable Small World) for approximate nearest neighbor search, improving search speed from O(n) to O(log n).

Optionally, install the [sqlite-vector](https://github.com/sqliteai/sqlite-vector) extension to enable vector operations on the SQL side:

```bash
# macOS
curl -L https://github.com/sqliteai/sqlite-vector/releases/latest/download/sqlite_vector.dylib \
  -o ~/.mnemosyne/lib/sqlite_vector.dylib
```

The system detects and loads it automatically at startup.

**Recommended Models:**

| Use Case | Model | Dimensions | Size |
|---|---|---|---|
| Text (Lightweight) | `sentence-transformers/all-MiniLM-L6-v2` | 384 | 90 MB |
| Text (High Quality)| `sentence-transformers/all-mpnet-base-v2` | 768 | 420 MB |
| Multilingual | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | 470 MB |
| Image | `openai/clip-vit-base-patch32` | 512 | 350 MB |
| Audio | `openai/whisper-tiny` | — | 75 MB |
| Audio (High Quality)| `openai/whisper-base` | — | 140 MB |

---

## Data Storage

All data is stored in a single SQLite file (in WAL mode):

```
~/.mnemosyne/db.sqlite
  ├── files              # File metadata + SHA-256 hash
  ├── document_chunks    # Content chunks (1500 chars, 150-word overlap)
  ├── fts_chunks         # FTS5 full-text index (BM25, unicode61 tokenizer)
  ├── embeddings         # Vectors (f32 little-endian BLOB)
  └── model_registry     # Records of downloaded models
```

---

## Development

```bash
# Run unit and integration tests (11 tests, including end-to-end pipeline)
cargo test -p mnemosyne-storage -p mnemosyne-retrieval

# Check all crates without a full compile
cargo check --workspace

# Format code
cargo fmt --all

# Lint
cargo clippy --workspace -- -D warnings
```

**Integration Test Coverage:**
- File indexing and statistics
- FTS5 keyword search (BM25 ranking validation)
- Vector search result validation
- Hybrid RRF search
- Incremental indexing (skips identical hashes)
- File listing and deletion
- Large file chunking

---

## Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| Language | Rust 1.75+ | Memory safety, zero-cost abstractions |
| GUI Framework | Tauri 2.x | Native WebView, cross-platform |
| Async Runtime | Tokio 1.x | Compatible with Tauri, rich ecosystem |
| Database | SQLite (rusqlite bundled) | No external service dependency |
| Full-Text Search | SQLite FTS5 | Built-in, BM25, unicode61 |
| Vector Storage | SQLite BLOB | Simple initial implementation; migratable |
| Embedding Models | Candle (Optional) | Pure Rust, CPU/Metal, no Python dependency |
| HTTP Framework | Axum 0.8 | Tower ecosystem, CORS, tracing |
| PDF Parsing | pdf-extract | Pure Rust |
| Image Parsing | image crate | Pure Rust, for reading dimensions |
| File Watching | notify + debouncer | Cross-platform, 500ms debounce |
| Error Handling | thiserror + anyhow | `thiserror` for libraries, `anyhow` for apps |

---

## Roadmap

- [ ] **CLIP Image Embeddings** — Semantic search for images based on visual content.
- [ ] **Whisper Audio Transcription** — Transcribe MP3/WAV to text for searching.
- [ ] **Video Keyframe Extraction** — Use ffmpeg + CLIP for frame descriptions.
- [ ] **`sqlite-vector` Integration** — ANN search for larger-scale indexes.
- [ ] **Multilingual Tokenizer** — Add `jieba` for Chinese FTS5 tokenization.
- [ ] **Incremental Sync Daemon** — A background service that runs on system startup.
- [ ] **Search Result Highlighting** — Pinpoint and highlight matched terms in context.

---

## License

MIT License — see the [LICENSE](LICENSE) file for details.
