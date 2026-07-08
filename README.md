<div align="center">
  <h1>🧠 Mnemosyne</h1>
  <p><strong>本地文件智能搜索与分析系统</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Rust-1.75+-orange?logo=rust" alt="Rust">
    <img src="https://img.shields.io/badge/Tauri-2.x-blue?logo=tauri" alt="Tauri">
    <img src="https://img.shields.io/badge/SQLite-3.x-green?logo=sqlite" alt="SQLite">
    <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT">
  </p>
</div>

---

## 简介

Mnemosyne 对本地文件进行智能化的搜索和分析，提供：

- **混合检索**：向量语义搜索 + FTS5 关键词搜索，通过 RRF 融合排名
- **多格式支持**：文本（txt/md/csv/json…）、PDF、图片、音频、视频
- **三种 UI**：桌面 GUI（Tauri2）、命令行 CLI、REST API
- **实时更新**：文件监听器自动检测变更并增量索引
- **零外部服务**：所有数据本地 SQLite，无需独立数据库服务

---

## 功能特性

| 功能 | 状态 |
|------|------|
| 文本文件索引（txt/md/csv/json/py/rs…） | ✅ 完整实现 |
| PDF 文本提取 | ✅ 完整实现 |
| 图片尺寸元数据提取 | ✅ 完整实现 |
| 音频/视频文件记录 | ✅ 存根（路径+元数据） |
| FTS5 关键词搜索（BM25） | ✅ 完整实现 |
| 向量相似度搜索 | ✅ 余弦相似度（brute-force） |
| RRF 混合搜索 | ✅ 完整实现 |
| 增量索引（SHA-256 变更检测） | ✅ 完整实现 |
| 文件实时监听（FileWatcher） | ✅ 完整实现 |
| 后台定期重扫（BackgroundIndexer） | ✅ 完整实现 |
| REST API 服务器 | ✅ 完整实现 |
| CLI 工具 | ✅ 完整实现 |
| 桌面 GUI（Tauri2） | ✅ 骨架实现 |
| 真实 BERT 推理（candle） | ✅ `candle-backend` feature |
| CLIP 图像 Embedding | ✅ `clip-backend` feature |
| Whisper 音频转录 | ✅ `whisper-backend` feature |
| sqlite-vector ANN 检索 | ✅ 动态扩展加载 + HNSW fallback |

---

## 架构概览

```
┌─────────────────────────────────────────────┐
│              UI 层                           │
│  [Tauri2 GUI]  [CLI]  [REST API / axum]     │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│         mnemosyne-retrieval                  │
│   SearchEngine (索引 + 搜索编排门面)          │
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

### Crate 依赖图

```
mnemosyne-core          ← 基础类型/Trait/Error
  ← mnemosyne-storage   ← SQLite CRUD
  ← mnemosyne-model     ← 文本嵌入（stub/candle）
  ← mnemosyne-parser    ← 文件解析插件
  ← mnemosyne-index     ← 混合检索引擎
    ← mnemosyne-retrieval  ← SearchEngine 门面
      ← cli / api / src-tauri
```

---

## 安装

### 前置条件

| 工具 | 版本 | 说明 |
|------|------|------|
| Rust | ≥ 1.75 | `rustup update stable` |
| Tauri CLI | 2.x | `cargo install tauri-cli` |
| Node.js | ≥ 18 | GUI 前端构建 |

macOS 还需要 Xcode Command Line Tools：
```bash
xcode-select --install
```

### 构建

```bash
git clone <repo-url> mnemosyne
cd mnemosyne

# CLI + REST API（最快，无需 Node.js）
cargo build --release -p mnemosyne-cli -p mnemosyne-api

# 桌面 GUI
npm install
cargo tauri build

# 启用完整 ML 后端（BERT + CLIP + Whisper）
cargo build --release -p mnemosyne-cli \
  --features "mnemosyne-model/full"

# 单独启用某个功能
cargo build --release -p mnemosyne-cli \
  --features "mnemosyne-model/candle-backend"   # 仅 BERT

cargo build --release -p mnemosyne-cli \
  --features "mnemosyne-model/clip-backend"     # BERT + CLIP

cargo build --release -p mnemosyne-cli \
  --features "mnemosyne-model/whisper-backend"  # BERT + Whisper
```

构建产物位于 `target/release/`：
- `mnemosyne` — CLI 工具
- `mnemosyne-server` — REST API 服务器

---

## 使用

### CLI

```bash
# 索引一个目录（递归）
mnemosyne index ~/Documents

# 搜索（混合模式）
mnemosyne search "Rust 内存安全"

# 仅关键词搜索
mnemosyne search "async trait" --mode keyword --limit 20

# 仅向量搜索
mnemosyne search "机器学习论文" --mode vector

# 输出 JSON
mnemosyne search "tokio runtime" --json

# 实时监听目录（文件变更自动重索引）
mnemosyne watch ~/Documents

# 查看统计
mnemosyne stats

# 列出已索引文件
mnemosyne list --limit 50

# 删除文件记录
mnemosyne remove <file-id>

# 启动 REST API 服务（默认 8080）
mnemosyne serve --port 8080

# 指定数据库路径
mnemosyne --db /path/to/db.sqlite index ~/Documents
```

### REST API

启动服务器：
```bash
# 环境变量配置
MNEMOSYNE_PORT=8080 MNEMOSYNE_DB=~/.mnemosyne/db.sqlite mnemosyne-server
```

**端点一览：**

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/search` | 搜索文件 |
| `POST` | `/api/index` | 索引目录 |
| `GET` | `/api/stats` | 统计信息 |
| `GET` | `/api/files` | 列出文件（?limit&offset） |
| `DELETE` | `/api/files/:id` | 删除文件记录 |
| `GET` | `/api/models` | 已下载模型列表 |
| `POST` | `/api/models/download` | 下载模型 |
| `GET` | `/health` | 健康检查 |

**搜索示例（curl）：**
```bash
# 混合搜索
curl -X POST http://localhost:8080/api/search \
  -H "Content-Type: application/json" \
  -d '{"text": "异步编程", "mode": "hybrid", "limit": 10}'

# 索引目录
curl -X POST http://localhost:8080/api/index \
  -H "Content-Type: application/json" \
  -d '{"path": "/home/user/Documents"}'

# 统计
curl http://localhost:8080/api/stats
```

**SearchQuery 结构：**
```json
{
  "text": "查询文本",
  "mode": "hybrid",      // "hybrid" | "vector" | "keyword"
  "limit": 20,
  "offset": 0,
  "file_types": null     // null = 全部，或 ["text","image","audio","video"]
}
```

### 桌面 GUI

```bash
npm run dev   # 开发模式（热重载）
npm run build # 打包应用
```

GUI 功能：
- 搜索框（Enter 执行，支持切换搜索模式）
- 点击「索引目录」输入路径并建立索引
- 左侧面板：实时统计 + 文件类型过滤
- 结果列表：文件路径、相关片段、相似度分数

---

## 配置

Mnemosyne 通过环境变量和命令行参数配置，无配置文件。

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MNEMOSYNE_DB` | `~/.mnemosyne/db.sqlite` | 数据库路径 |
| `MNEMOSYNE_PORT` | `8080` | API 服务端口 |
| `RUST_LOG` | `warn,mnemosyne=info` | 日志级别 |

---

## 嵌入模型

### 默认（Stub 模式）

无需下载模型，使用基于哈希的确定性伪向量。适合开发调试，**不能**进行真正的语义搜索。

### 文本 BERT（`candle-backend`）

编译时启用：
```bash
cargo build --release -p mnemosyne-cli --features mnemosyne-model/candle-backend
```

首次运行自动下载模型（~90 MB）：
```bash
mnemosyne model-download sentence-transformers/all-MiniLM-L6-v2
```

### 图像 CLIP（`clip-backend`）

```bash
cargo build --release -p mnemosyne-cli --features mnemosyne-model/clip-backend
mnemosyne model-download openai/clip-vit-base-patch32
```

索引图片目录后即可用自然语言搜索：
```bash
mnemosyne index ~/Pictures
mnemosyne search "sunset over mountains"
```

### 音频 Whisper（`whisper-backend`）

```bash
cargo build --release -p mnemosyne-cli --features mnemosyne-model/whisper-backend
mnemosyne model-download openai/whisper-tiny
```

支持 WAV（16-bit 或 float，任意采样率自动重采样至 16 kHz）。

```bash
mnemosyne index ~/Recordings   # 自动转录后索引
mnemosyne search "project meeting notes"
```

### ANN 向量检索（sqlite-vector + HNSW）

当索引规模超过 **2 000 个分块**时，系统自动切换为 HNSW 近似最近邻搜索（`instant-distance`），搜索速度从 O(n) 提升到 O(log n)。

可选：安装 [sqlite-vector](https://github.com/sqliteai/sqlite-vector) 扩展以启用 SQL 侧向量运算：

```bash
# macOS
curl -L https://github.com/sqliteai/sqlite-vector/releases/latest/download/sqlite_vector.dylib \
  -o ~/.mnemosyne/lib/sqlite_vector.dylib
```

系统启动时自动检测并加载，无需额外配置。

**推荐模型：**

| 用途 | 模型 | 维度 | 磁盘 |
|------|------|------|------|
| 文本（轻量） | `sentence-transformers/all-MiniLM-L6-v2` | 384 | 90 MB |
| 文本（高质量） | `sentence-transformers/all-mpnet-base-v2` | 768 | 420 MB |
| 多语言 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | 470 MB |
| 图像 | `openai/clip-vit-base-patch32` | 512 | 350 MB |
| 音频 | `openai/whisper-tiny` | — | 75 MB |
| 音频（高质量） | `openai/whisper-base` | — | 140 MB |

---

## 数据存储

所有数据存储在单个 SQLite 文件（WAL 模式）：

```
~/.mnemosyne/db.sqlite
  ├── files              # 文件元数据 + SHA-256 哈希
  ├── document_chunks    # 内容分块（1500字符/块，150字重叠）
  ├── fts_chunks         # FTS5 全文索引（BM25，unicode61 分词）
  ├── embeddings         # 向量（f32 小端序 BLOB）
  └── model_registry     # 已下载模型记录
```

---

## 开发

```bash
# 单元测试 + 集成测试（11 个测试，含端到端 pipeline 验证）
cargo test -p mnemosyne-storage -p mnemosyne-retrieval

# 检查所有 crate（无需完整编译）
cargo check --workspace

# 代码格式化
cargo fmt --all

# Lint
cargo clippy --workspace -- -D warnings
```

**集成测试覆盖：**
- 文件索引与统计
- FTS5 关键词搜索（BM25 排名验证）
- 向量搜索结果返回
- 混合 RRF 搜索
- 增量索引（相同哈希跳过）
- 文件列表与删除
- 大文件多块分割

---

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| 语言 | Rust 1.75+ | 内存安全，零成本抽象 |
| GUI 框架 | Tauri 2.x | 原生 WebView，跨平台 |
| 异步运行时 | Tokio 1.x | 与 Tauri 兼容，生态完善 |
| 数据库 | SQLite (rusqlite bundled) | 无外部服务依赖 |
| 全文检索 | SQLite FTS5 | 内置，BM25，unicode61 |
| 向量存储 | SQLite BLOB | 初始实现；可迁移至 sqlite-vector |
| 嵌入模型 | Candle (可选) | 纯 Rust，CPU/Metal，无 Python |
| HTTP 框架 | Axum 0.8 | Tower 生态，CORS，tracing |
| PDF 解析 | pdf-extract | 纯 Rust |
| 图像解析 | image crate | 纯 Rust，读取尺寸 |
| 文件监听 | notify + debouncer | 跨平台，500ms 防抖 |
| 错误处理 | thiserror + anyhow | 库用 thiserror，应用用 anyhow |

---

## 路线图

- [ ] **CLIP 图像 Embedding** — 基于视觉内容的图片语义搜索
- [ ] **Whisper 音频转录** — MP3/WAV 语音转文字后检索
- [ ] **视频关键帧提取** — ffmpeg + CLIP 帧描述
- [ ] **sqlite-vector 集成** — ANN 近似最近邻，支持更大规模索引
- [ ] **多语言分词优化** — jieba 中文分词 for FTS5
- [ ] **增量同步守护进程** — 系统启动自动运行的后台服务
- [ ] **搜索结果高亮** — 命中词精确定位与上下文高亮

---

## 许可证

MIT License — 详见 [LICENSE](LICENSE) 文件。
