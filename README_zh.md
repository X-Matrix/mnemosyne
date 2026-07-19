<div align="center">
  <img src="src-tauri/icons/128x128.png" alt="Mnemosyne" width="80" />
  <h1>Mnemosyne</h1>
  <p><strong>一个智能的、本地优先的文件搜索与分析系统。</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Rust-1.75+-orange?logo=rust" alt="Rust">
    <img src="https://img.shields.io/badge/Tauri-2.x-blue?logo=tauri" alt="Tauri">
    <img src="https://img.shields.io/badge/SQLite-3.x-green?logo=sqlite" alt="SQLite">
    <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT">
  </p>
</div>

---

## 关于

Mnemosyne 为本地文件提供智能搜索与分析：

- **混合检索**：向量语义搜索 + FTS5 关键词搜索，RRF 融合排名。
- **多格式支持**：文本（txt/md/csv/json...）、PDF、图像、音频、视频。
- **多语言模型**：BGE-M3（中英文等 100+ 语言）和 Chinese CLIP（中文图片检索）。
- **多种交互界面**：桌面 GUI（Tauri 2）、CLI、REST API。
- **实时更新**：监听文件变更并增量索引，**文件删除自动同步移除**。
- **智能目录过滤**：自动跳过 `node_modules`、`.git`、`target` 等 60+ 无关目录。
- **零外部服务**：所有数据存储于单个 SQLite 文件。

---

## 功能

| 功能 | 状态 |
|---|---|
| 文本文件索引 (txt/md/csv/json/py/rs...) | ✅ 完全实现 |
| PDF 文本提取 | ✅ 完全实现 |
| CLIP 图像嵌入 | ✅ `clip-backend` 功能 |
| Whisper 音频转录 | ✅ `whisper-backend` 功能 |
| FTS5 关键词搜索 (BM25) | ✅ 完全实现 |
| sqlite-vec KNN 向量搜索（HNSW） | ✅ 运行时动态加载 |
| 纯 Rust HNSW（sqlite-vec 回退） | ✅ 完全实现 |
| RRF 混合搜索 | ✅ 完全实现 |
| 增量索引（SHA-256 变更检测） | ✅ 完全实现 |
| 实时文件监听（含删除同步） | ✅ 完全实现 |
| 目录忽略过滤（60+ 规则） | ✅ 完全实现 |
| REST API 服务器（含 Swagger UI） | ✅ 完全实现 |
| CLI 工具 | ✅ 完全实现 |
| 桌面 GUI (Tauri 2) | ✅ 完整实现 |
| BERT 文本嵌入（via `candle`） | ✅ `candle-backend` 功能 |
| BGE-M3 多语言嵌入（1024 维） | ✅ `candle-backend` 功能 |
| OpenAI CLIP（英文图片检索） | ✅ `clip-backend` 功能 |
| Chinese CLIP（中文图片检索） | ✅ `clip-backend` 功能 |
| Apple Silicon Metal GPU 加速 | ✅ `metal-backend`（自动检测） |

---

## 架构

```
┌─────────────────────────────────────────────┐
│                 UI 层                       │
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

### Crate 依赖图

```
mnemosyne-core          ← 基础类型/特征/错误
  ← mnemosyne-storage   ← SQLite CRUD + sqlite-vec 集成
  ← mnemosyne-model     ← 文本/图像/音频嵌入（Candle/Metal）
  ← mnemosyne-parser    ← 文件解析（插件化）
  ← mnemosyne-index     ← 混合检索引擎
    ← mnemosyne-retrieval  ← SearchEngine 门面 + IgnoreConfig
      ← cli / api / src-tauri
```

---

## 安装

### 先决条件

| 工具 | 版本 | 说明 |
|---|---|---|
| Rust | ≥ 1.75 | `rustup update stable` |
| Tauri CLI | 2.x | `cargo install tauri-cli`（仅 GUI） |
| Node.js | ≥ 18 | 仅 GUI 前端构建时需要 |

macOS 还需要 Xcode 命令行工具：
```bash
xcode-select --install
```

### 使用 build.sh 构建（推荐）

```bash
git clone <repo-url> mnemosyne && cd mnemosyne

# 完整 ML 后端 + 自动下载 sqlite-vec
# Apple Silicon 自动启用 Metal GPU 加速
./scripts/build.sh --full --sqlite-vector

# 常用选项
./scripts/build.sh --candle              # 仅 BERT 文本嵌入
./scripts/build.sh --full --no-metal     # 关闭 Metal，强制 CPU
./scripts/build.sh --full --gui          # 额外构建桌面 GUI
./scripts/build.sh --dev                 # Debug profile（编译更快）
```

**build.sh 选项：**

| 选项 | 说明 |
|---|---|
| `--release` / `--dev` | Release（默认）或 Debug profile |
| `--candle` / `--clip` / `--whisper` / `--full` | 按需启用 ML 后端 |
| `--gui` | 额外构建 Tauri 桌面应用 |
| `--sqlite-vector` | 自动下载 sqlite-vec 到 `~/.mnemosyne/lib/` |
| `--no-metal` | 禁止 Apple Silicon 自动启用 Metal GPU |

### 手动 Cargo 构建

```bash
# 完整后端（Apple Silicon 加 --features metal-backend）
cargo build --release -p mnemosyne-cli \
  --features "candle-backend,clip-backend,whisper-backend"
```

构建产物位于 `target/release/`：`mnemosyne`（CLI）、`mnemosyne-server`（API）

---

## 使用

### CLI

```bash
# 索引目录（自动跳过 node_modules/.git 等）
mnemosyne index ~/Documents

# 混合搜索
mnemosyne search "Rust 内存安全"

# 关键词 / 向量搜索
mnemosyne search "机器学习" --mode keyword
mnemosyne search "机器学习" --mode vector

# 下载嵌入模型
mnemosyne model-download BAAI/bge-m3                               # 中英文推荐
mnemosyne model-download OFA-Sys/chinese-clip-vit-base-patch16    # 中文图片
mnemosyne model-download openai/clip-vit-base-patch32             # 英文图片
mnemosyne model-download openai/whisper-tiny                      # 音频转录

# 实时监听（文件删除会自动移除索引）
mnemosyne watch ~/Documents

# 统计 / 列表 / 删除
mnemosyne stats
mnemosyne list --limit 50
mnemosyne remove <file-id>

# 启动 REST API
mnemosyne serve --port 8080
```

### REST API

```bash
curl -X POST http://localhost:8080/api/search \
  -H "Content-Type: application/json" \
  -d '{"text": "机器学习", "mode": "hybrid", "limit": 10}'
```

| 方法 | 路径 | 描述 |
|---|---|---|
| `POST` | `/api/search` | 混合/向量/关键词搜索 |
| `POST` | `/api/index` | 索引目录 |
| `GET` | `/api/stats` | 统计信息 |
| `GET` | `/api/files` | 文件列表 |
| `DELETE` | `/api/files/:id` | 删除文件记录 |
| `GET` | `/api/models` | 已下载模型 |
| `POST` | `/api/models/download` | 下载模型 |
| `GET` | `/api/docs` | Swagger UI |
| `GET` | `/health` | 健康检查 |

### 桌面 GUI 功能

- 混合 / 向量 / 关键词搜索模式；可调节权重与阈值
- 索引目录管理（添加、重新索引、实时监听）
- 文件浏览与预览（文本、PDF、图像、音频）
- 模型管理（下载、切换 BERT/CLIP/Whisper 模型）
- HuggingFace 镜像源配置（直连 / hf-mirror.com / 自定义）
- 代理配置
- REST API 服务器内置开关（含 Swagger UI 链接）
- 运行日志实时查看（含 DEBUG 级别切换）
- **清空索引数据库**（需输入随机验证词确认，防误操作）

---

## 嵌入模型

### 文本模型（`candle-backend`）

| 模型 | 维度 | 大小 | 语言 |
|---|---|---|---|
| `BAAI/bge-m3` ⭐ | 1024 | 570 MB | 中/英/100+ 语言，CLS 池化 |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 90 MB | 英文，轻量快速 |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | 470 MB | 50 种语言 |
| `sentence-transformers/all-mpnet-base-v2` | 768 | 420 MB | 英文，高质量 |

### 视觉模型（`clip-backend`）

| 模型 | 维度 | 大小 | 文本语言 |
|---|---|---|---|
| `OFA-Sys/chinese-clip-vit-base-patch16` ⭐ | 512 | 380 MB | 中文，Chinese CLIP |
| `openai/clip-vit-base-patch32` | 512 | 340 MB | 英文，OpenAI CLIP |

### 音频模型（`whisper-backend`）

| 模型 | 大小 | 适用场景 |
|---|---|---|
| `openai/whisper-tiny` | 78 MB | 速度优先 |
| `openai/whisper-base` | 148 MB | 质量与速度平衡 |
| `openai/whisper-small` | 488 MB | 高质量，中英文 |

### sqlite-vec KNN 加速

```bash
# 通过 build.sh 自动安装
./scripts/build.sh --sqlite-vector

# 手动安装（macOS arm64）
mkdir -p ~/.mnemosyne/lib
curl -fsSL https://github.com/asg017/sqlite-vec/releases/latest/download/sqlite-vec-0.1.9-loadable-macos-aarch64.tar.gz \
  | tar -xzO vec0.dylib > ~/.mnemosyne/lib/vec0.dylib
```

启动时自动检测；若不存在则回退到纯 Rust HNSW（> 2000 条）或暴力余弦（≤ 2000 条）。

---

## 目录过滤（IgnoreConfig）

索引时自动跳过以下类别（可通过代码自定义）：

| 类别 | 包含 |
|---|---|
| 版本控制 | `.git` `.svn` `.hg` |
| JS/Node | `node_modules` `bower_components` `.npm` `.yarn` |
| Python | `__pycache__` `.venv` `venv` `.tox` `.pytest_cache` |
| Rust | `target` |
| Java/Android | `.gradle` `.m2` `build` |
| iOS/Swift | `Pods` `DerivedData` |
| 前端构建 | `dist` `out` `.next` `.nuxt` `.cache` `.parcel-cache` |
| IDE | `.idea` `.vscode` `.vs` |
| OS 生成 | `.Spotlight-V100` `.Trashes` `$RECYCLE.BIN` |

---

## 数据存储

```
~/.mnemosyne/
  db.sqlite
    ├── files              # 文件元数据 + SHA-256 哈希
    ├── document_chunks    # 内容分块
    ├── fts_chunks         # FTS5 全文索引（BM25）
    ├── embeddings         # 向量 BLOB（f32 LE）
    ├── embedding_vec_384  # sqlite-vec HNSW（BERT 384 维）
    ├── embedding_vec_512  # sqlite-vec HNSW（CLIP 512 维）
    ├── embedding_vec_1024 # sqlite-vec HNSW（BGE-M3 1024 维）
    └── model_registry     # 已下载模型
  models/                  # 本地模型缓存
  lib/vec0.dylib           # sqlite-vec 扩展（可选）
```

---

## 开发

```bash
cargo test -p mnemosyne-storage -p mnemosyne-retrieval
cargo check --workspace
cargo fmt --all
cargo clippy --workspace -- -D warnings
```

---

## 技术栈

| 组件 | 选择 |
|---|---|
| 语言 | Rust 1.75+ |
| GUI | Tauri 2.x + HTML/CSS/JS |
| 异步运行时 | Tokio 1.x |
| 数据库 | SQLite（rusqlite bundled + WAL）|
| 全文搜索 | FTS5（BM25，trigram 分词器）|
| 向量 KNN | sqlite-vec（HNSW）+ Rust HNSW 回退 |
| 嵌入推理 | HuggingFace Candle（CPU / Metal GPU）|
| 文本模型 | BERT / BGE-M3（XLM-RoBERTa 架构）|
| 图像模型 | OpenAI CLIP / Chinese CLIP |
| 音频模型 | Whisper（via Candle）|
| HTTP | Axum 0.8 + Tower |
| 文件监听 | notify-debouncer-mini（500ms 防抖）|
