<div align="center">
  <img src="src-tauri/icons/128x128.png" alt="Mnemosyne" width="80" />
  <h1>Mnemosyne</h1>
  <p><strong>An intelligent, local-first file search and analysis system.</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Rust-1.75+-orange?logo=rust" alt="Rust">
    <img src="https://img.shields.io/badge/Tauri-2.x-blue?logo=tauri" alt="Tauri">
    <img src="https://img.shields.io/badge/SQLite-3.x-green?logo=sqlite" alt="SQLite">
    <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT">
  </p>
</div>

---

## About

Mnemosyne provides intelligent search and analysis for your local files:

- **Hybrid Search**: Vector semantic search + FTS5 keyword search, fused with Reciprocal Rank Fusion (RRF).
- **Multi-Format**: Text (txt/md/csv/json...), PDF, images, audio, video.
- **Multilingual Models**: BGE-M3 (100+ languages) and Chinese CLIP for Chinese text-to-image search.
- **Multiple UIs**: Desktop GUI (Tauri 2), CLI, REST API.
- **Real-time Updates**: File watcher detects changes and updates the index incrementally. **File deletions are automatically removed from the index.**
- **Smart Directory Filtering**: Automatically skips `node_modules`, `.git`, `target`, and 60+ other irrelevant directories.
- **Zero External Services**: All data stored in a single SQLite file.

---

## Features

| Feature | Status |
|---|---|
| Text File Indexing (txt/md/csv/json/py/rs...) | ✅ Fully Implemented |
| PDF Text Extraction | ✅ Fully Implemented |
| CLIP Image Embeddings | ✅ `clip-backend` feature |
| Whisper Audio Transcription | ✅ `whisper-backend` feature |
| FTS5 Keyword Search (BM25) | ✅ Fully Implemented |
| sqlite-vec KNN Vector Search (HNSW) | ✅ Dynamic Runtime Loading |
| Pure Rust HNSW Fallback | ✅ Fully Implemented |
| RRF Hybrid Search | ✅ Fully Implemented |
| Incremental Indexing (SHA-256 Change Detection) | ✅ Fully Implemented |
| Real-time File Watcher (incl. deletion sync) | ✅ Fully Implemented |
| Directory Ignore Filtering (60+ rules) | ✅ Fully Implemented |
| REST API Server (with Swagger UI) | ✅ Fully Implemented |
| CLI Tool | ✅ Fully Implemented |
| Desktop GUI (Tauri 2) | ✅ Full Implementation |
| BERT Text Embeddings (via `candle`) | ✅ `candle-backend` feature |
| BGE-M3 Multilingual Embeddings (1024-dim) | ✅ `candle-backend` feature |
| OpenAI CLIP (English image search) | ✅ `clip-backend` feature |
| Chinese CLIP (Chinese image search) | ✅ `clip-backend` feature |
| Apple Silicon Metal GPU Acceleration | ✅ `metal-backend` (auto-detected) |

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
│   SearchEngine · BackgroundIndexer           │
│   FileWatcher · IgnoreConfig                 │
└──┬────────────────┬────────────────┬────────┘
   │                │                │
┌──▼──────────┐  ┌──▼───────┐  ┌────▼──────────┐
│   model     │  │  parser  │  │    index       │
│BERT/BGE-M3  │  │Text/PDF/ │  │ HybridIndex    │
│CLIP/Whisper │  │Image/... │  │ Vector + FTS5  │
└─────────────┘  └──────────┘  └──────┬────────┘
                                        │
                               ┌────────▼────────┐
                               │    storage       │
                               │ SQLite + WAL     │
                               │ sqlite-vec (KNN) │
                               └─────────────────┘
```

### Crate Dependency Graph

```
mnemosyne-core          ← Base types / traits / errors
  ← mnemosyne-storage   ← SQLite CRUD + sqlite-vec integration
  ← mnemosyne-model     ← Text / image / audio embeddings (Candle / Metal)
  ← mnemosyne-parser    ← File parsing plugins
  ← mnemosyne-index     ← Hybrid search engine
    ← mnemosyne-retrieval  ← SearchEngine facade + IgnoreConfig
      ← cli / api / src-tauri
```

---

## Installation

### Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Rust | ≥ 1.75 | `rustup update stable` |
| Tauri CLI | 2.x | `cargo install tauri-cli` (GUI only) |
| Node.js | ≥ 18 | GUI frontend build only |

On macOS, you also need Xcode Command Line Tools:
```bash
xcode-select --install
```

### Building with build.sh (recommended)

```bash
git clone <repo-url> mnemosyne && cd mnemosyne

# Full ML backend + auto-download sqlite-vec
# Apple Silicon automatically enables Metal GPU acceleration
./scripts/build.sh --full --sqlite-vector

# Common options
./scripts/build.sh --candle              # BERT text only
./scripts/build.sh --full --no-metal     # Disable Metal, force CPU
./scripts/build.sh --full --gui          # Also build desktop GUI
./scripts/build.sh --dev                 # Debug profile (faster compile)
```

**build.sh options:**

| Option | Description |
|---|---|
| `--release` / `--dev` | Release (default) or Debug profile |
| `--candle` / `--clip` / `--whisper` / `--full` | Enable ML backends |
| `--gui` | Also build Tauri desktop app |
| `--sqlite-vector` | Auto-download sqlite-vec to `~/.mnemosyne/lib/` |
| `--no-metal` | Suppress Metal auto-detection on Apple Silicon |

### Manual Cargo Build

```bash
# Full backend (add --features metal-backend for Apple Silicon)
cargo build --release -p mnemosyne-cli \
  --features "candle-backend,clip-backend,whisper-backend"
```

Build artifacts in `target/release/`: `mnemosyne` (CLI), `mnemosyne-server` (API)

---

## Usage

### CLI

```bash
# Index a directory (auto-skips node_modules/.git/etc.)
mnemosyne index ~/Documents

# Hybrid search
mnemosyne search "Rust memory safety"

# Keyword / vector search
mnemosyne search "async trait" --mode keyword --limit 20
mnemosyne search "machine learning papers" --mode vector

# Download embedding models
mnemosyne model-download BAAI/bge-m3                               # Multilingual (recommended)
mnemosyne model-download OFA-Sys/chinese-clip-vit-base-patch16    # Chinese image search
mnemosyne model-download openai/clip-vit-base-patch32             # English image search
mnemosyne model-download openai/whisper-tiny                      # Audio transcription

# Real-time watching (file deletions are auto-removed from index)
mnemosyne watch ~/Documents

# Stats / list / remove
mnemosyne stats
mnemosyne list --limit 50
mnemosyne remove <file-id>

# Start REST API
mnemosyne serve --port 8080
```

### REST API

```bash
curl -X POST http://localhost:8080/api/search \
  -H "Content-Type: application/json" \
  -d '{"text": "machine learning", "mode": "hybrid", "limit": 10}'
```

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/search` | Hybrid / vector / keyword search |
| `POST` | `/api/index` | Index a directory |
| `GET` | `/api/stats` | Statistics |
| `GET` | `/api/files` | List files |
| `DELETE` | `/api/files/:id` | Delete file record |
| `GET` | `/api/models` | Downloaded models |
| `POST` | `/api/models/download` | Download a model |
| `GET` | `/api/docs` | Swagger UI |
| `GET` | `/health` | Health check |

### Desktop GUI Features

- Hybrid / vector / keyword search mode; adjustable weights and score threshold
- Directory management (add, re-index, real-time watching)
- File browser and preview (text, PDF, images, audio)
- Model management (download, switch BERT/CLIP/Whisper models)
- HuggingFace mirror configuration (direct / hf-mirror.com / custom)
- Proxy configuration
- Built-in REST API server toggle (with Swagger UI link)
- Real-time log viewer
- **Clear index database** (requires typing a random obscure confirmation word to prevent accidents)

---

## Embedding Models

### Text Models (`candle-backend`)

| Model | Dim | Size | Languages |
|---|---|---|---|
| `BAAI/bge-m3` ⭐ | 1024 | 2.1 GB | Chinese / English / 100+ languages, CLS pooling |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 87 MB | English, lightweight |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | 458 MB | 50 languages |
| `sentence-transformers/all-mpnet-base-v2` | 768 | 418 MB | English, high quality |

### Vision Models (`clip-backend`)

| Model | Dim | Size | Text Language |
|---|---|---|---|
| `OFA-Sys/chinese-clip-vit-base-patch16` ⭐ | 512 | 718 MB | **Chinese** (Chinese CLIP) |
| `openai/clip-vit-base-patch32` | 512 | 579 MB | English (OpenAI CLIP) |

### Audio Models (`whisper-backend`)

| Model | Size | Use Case |
|---|---|---|
| `openai/whisper-tiny` | 147 MB | Speed priority |
| `openai/whisper-base` | 280 MB | Balanced quality/speed |
| `openai/whisper-small` | ~490 MB | High quality, multilingual |

### sqlite-vec KNN Acceleration

Install [asg017/sqlite-vec](https://github.com/asg017/sqlite-vec) to enable HNSW KNN search:

```bash
# Automatic via build.sh
./scripts/build.sh --sqlite-vector

# Manual (macOS arm64)
mkdir -p ~/.mnemosyne/lib
curl -fsSL https://github.com/asg017/sqlite-vec/releases/latest/download/sqlite-vec-0.1.9-loadable-macos-aarch64.tar.gz \
  | tar -xzO vec0.dylib > ~/.mnemosyne/lib/vec0.dylib
```

Automatically detected at startup; silently falls back to pure Rust HNSW if absent.

---

## Directory Filtering (IgnoreConfig)

The following directories are automatically skipped during indexing:

| Category | Includes |
|---|---|
| Version control | `.git` `.svn` `.hg` |
| JS/Node | `node_modules` `bower_components` `.npm` `.yarn` |
| Python | `__pycache__` `.venv` `venv` `.tox` `.pytest_cache` |
| Rust | `target` |
| Java/Android | `.gradle` `.m2` `build` |
| iOS/Swift | `Pods` `DerivedData` |
| Frontend build | `dist` `out` `.next` `.nuxt` `.cache` `.parcel-cache` |
| IDE | `.idea` `.vscode` `.vs` |
| OS artefacts | `.Spotlight-V100` `.Trashes` `$RECYCLE.BIN` |

Customizable via `IgnoreConfig` in `mnemosyne-retrieval`.

---

## Data Storage

```
~/.mnemosyne/
  db.sqlite
    ├── files              # File metadata + SHA-256 hash
    ├── document_chunks    # Content chunks
    ├── fts_chunks         # FTS5 full-text index (BM25)
    ├── embeddings         # Vector BLOBs (f32 little-endian)
    ├── embedding_vec_384  # sqlite-vec HNSW (BERT 384-dim)
    ├── embedding_vec_512  # sqlite-vec HNSW (CLIP 512-dim)
    ├── embedding_vec_1024 # sqlite-vec HNSW (BGE-M3 1024-dim)
    └── model_registry     # Downloaded model records
  models/                  # Local model cache
  lib/vec0.dylib           # sqlite-vec extension (optional)
```

---

## Development

```bash
cargo test -p mnemosyne-storage -p mnemosyne-retrieval
cargo check --workspace
cargo fmt --all
cargo clippy --workspace -- -D warnings
```

---

## Tech Stack

| Component | Choice |
|---|---|
| Language | Rust 1.75+ |
| GUI | Tauri 2.x + HTML/CSS/JS |
| Async runtime | Tokio 1.x |
| Database | SQLite (rusqlite bundled + WAL) |
| Full-text search | FTS5 (BM25, trigram tokenizer) |
| Vector KNN | sqlite-vec (HNSW) + Rust HNSW fallback |
| Embedding inference | HuggingFace Candle (CPU / Metal GPU) |
| Text models | BERT / BGE-M3 (XLM-RoBERTa architecture) |
| Vision models | OpenAI CLIP / Chinese CLIP |
| Audio models | Whisper (via Candle) |
| HTTP | Axum 0.8 + Tower |
| File watching | notify-debouncer-mini (500ms debounce) |
