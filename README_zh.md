<div align="center">
  <img src="src-tauri/icons/128x128.png" alt="Mnemosyne" width="80" />
  <h1>Mnemosyne</h1>
  <p><strong>一个智能的、本地优先的文件搜索与分析系统。</strong></p>
  <p>
    <a href="https://github.com/mnemosyne-proj/mnemosyne/actions"><img src="https://github.com/mnemosyne-proj/mnemosyne/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <img src="https://img.shields.io/badge/Rust-1.75+-orange?logo=rust" alt="Rust">
    <img src="https://img.shields.io/badge/Tauri-2.x-blue?logo=tauri" alt="Tauri">
    <img src="https://img.shields.io/badge/SQLite-3.x-green?logo=sqlite" alt="SQLite">
    <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT">
  </p>
</div>

---

## 关于

Mnemosyne 为您的本地文件提供智能搜索和分析，其特点包括：

- **混合搜索**：结合了基于向量的语义搜索和 FTS5 关键字搜索，使用倒数排序融合（RRF）来合并和排序结果。
- **多格式支持**：处理文本（txt、md、csv、json...）、PDF、图像、音频和视频文件。
- **多种用户界面**：通过桌面 GUI（Tauri 2）、命令行界面（CLI）或 REST API 进行访问。
- **实时更新**：文件观察程序会自动检测更改并执行增量索引。
- **零外部服务**：所有数据都本地存储在单个 SQLite 文件中，无需单独的数据库服务器。

---

## 功能

| 功能 | 状态 |
|---|---|
| 文本文件索引 (txt/md/csv/json/py/rs...) | ✅ 完全实现 |
| PDF 文本提取 | ✅ 完全实现 |
| 图像元数据提取（尺寸） | ✅ 完全实现 |
| 音频/视频文件索引 | ✅ 存根（路径 + 元数据） |
| FTS5 关键字搜索 (BM25) | ✅ 完全实现 |
| 向量相似性搜索 | ✅ 余弦相似度（暴力搜索） |
| RRF 混合搜索 | ✅ 完全实现 |
| 增量索引（SHA-256 变更检测） | ✅ 完全实现 |
| 实时文件观察程序 | ✅ 完全实现 |
| 后台定期重新扫描 | ✅ 完全实现 |
| REST API 服务器 | ✅ 完全实现 |
| CLI 工具 | ✅ 完全实现 |
| 桌面 GUI (Tauri 2) | ✅ 骨架实现 |
| 真实 BERT 推理（通过 `candle`） | ✅ `candle-backend` 功能 |
| CLIP 图像嵌入 | ✅ `clip-backend` 功能 |
| Whisper 音频转录 | ✅ `whisper-backend` 功能 |
| ANN 向量搜索（通过 `sqlite-vector`） | ✅ 动态加载 + HNSW 后备 |

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
│   SearchEngine (索引 + 搜索门面)             │
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
                          │SQLite (捆绑)     │
                          │files/chunks/embs │
                          └─────────────────┘
```

### Crate 依赖图

```
mnemosyne-core          ← 基础类型/特征/错误
  ← mnemosyne-storage   ← SQLite CRUD 操作
  ← mnemosyne-model     ← 文本嵌入 (stub/candle)
  ← mnemosyne-parser    ← 文件解析插件
  ← mnemosyne-index     ← 混合搜索引擎
    ← mnemosyne-retrieval  ← SearchEngine 门面
      ← cli / api / src-tauri
```

---

## 安装

### 先决条件

| 工具 | 版本 | 注意 |
|---|---|---|
| Rust | ≥ 1.75 | `rustup update stable` |
| Tauri CLI | 2.x | `cargo install tauri-cli` |
| Node.js | ≥ 18 | 用于构建 GUI 前端 |
| Git LFS | 最新 | `git lfs install` |

在 macOS 上，您还需要 Xcode 命令行工具：
```bash
xcode-select --install
```

### 构建

```bash
git clone <repo-url> mnemosyne
cd mnemosyne

# 拉取 LFS 文件
git lfs pull

# 构建 CLI + REST API（最快，无需 Node.js）
cargo build --release -p mnemosyne-cli -p mnemosyne-api

# 构建桌面 GUI
npm install
cargo tauri build

# 启用完整的 ML 后端（BERT + CLIP + Whisper）
cargo build --release -p mnemosyne-cli \
  --features "mnemosyne-model/full"

# 单独启用功能
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

# 仅关键字搜索
mnemosyne search "async trait" --mode keyword --limit 20

# 仅向量搜索
mnemosyne search "机器学习论文" --mode vector

# 输出为 JSON
mnemosyne search "tokio runtime" --json

# 观察目录以进行实时更新
mnemosyne watch ~/Documents

# 查看统计信息
mnemosyne stats

# 列出已索引的文件
mnemosyne list --limit 50

# 删除文件记录
mnemosyne remove <file-id>

# 启动 REST API 服务器（默认为 8080 端口）
mnemosyne serve --port 8080

# 指定自定义数据库路径
mnemosyne --db /path/to/db.sqlite index ~/Documents
```

### REST API

启动服务器：
```bash
# 通过环境变量配置
MNEMOSYNE_PORT=8080 MNEMOSYNE_DB=~/.mnemosyne/db.sqlite mnemosyne-server
```

**端点：**

| 方法 | 路径 | 描述 |
|---|---|---|
| `POST` | `/api/search` | 搜索文件 |
| `POST` | `/api/index` | 索引目录 |
| `GET` | `/api/stats` | 获取统计信息 |
| `GET` | `/api/files` | 列出文件 (?limit&offset) |
| `DELETE`| `/api/files/:id` | 删除文件记录 |
| `GET` | `/api/models` | 列出已下载的模型 |
| `POST` | `/api/models/download`| 下载模型 |
| `GET` | `/health` | 健康检查 |

**示例 (cURL):**
```bash
# 混合搜索
curl -X POST http://localhost:8080/api/search \
  -H "Content-Type: application/json" \
  -d '{"text": "异步编程", "mode": "hybrid", "limit": 10}'

# 索引目录
curl -X POST http://localhost:8080/api/index \
  -H "Content-Type: application/json" \
  -d '{"path": "/home/user/Documents"}'

# 获取统计信息
curl http://localhost:8080/api/stats
```

**SearchQuery 负载：**
```json
{
  "text": "您的查询文本",
  "mode": "hybrid",      // "hybrid" | "vector" | "keyword"
  "limit": 20,
  "offset": 0,
  "file_types": null     // null 表示所有，或 ["text","image","audio","video"]
}
```

### 桌面 GUI

```bash
npm run dev   # 开发模式（热重载）
npm run build # 构建应用程序
```

GUI 功能：
- 搜索栏（按 Enter 搜索，支持模式切换）
- “索引目录”按钮以添加新路径
- 左侧面板：实时统计和文件类型过滤器
- 结果列表：文件路径、相关片段和相似度分数

---

## 配置

Mnemosyne 通过环境变量和命令行参数进行配置。不使用配置文件。

| 变量 | 默认值 | 描述 |
|---|---|---|
| `MNEMOSYNE_DB` | `~/.mnemosyne/db.sqlite` | 数据库路径 |
| `MNEMOSYNE_PORT` | `8080` | API 服务器端口 |
| `RUST_LOG` | `warn,mnemosyne=info` | 日志级别 |

---

## 嵌入模型

### 默认（存根模式）

无需下载模型。此模式使用确定性的、基于哈希的伪向量。它适用于开发和测试，但**不**执行真正的语义搜索。

### 文本 BERT (`candle-backend`)

在编译时启用：
```bash
cargo build --release -p mnemosyne-cli --features mnemosyne-model/candle-backend
```

模型（约 90 MB）在首次运行时自动下载：
```bash
mnemosyne model-download sentence-transformers/all-MiniLM-L6-v2
```

### 图像 CLIP (`clip-backend`)

```bash
cargo build --release -p mnemosyne-cli --features mnemosyne-model/clip-backend
mnemosyne model-download openai/clip-vit-base-patch32
```

索引图像目录后，您可以使用自然语言进行搜索：
```bash
mnemosyne index ~/Pictures
mnemosyne search "日落下的山脉"
```

### 音频 Whisper (`whisper-backend`)

```bash
cargo build --release -p mnemosyne-cli --features mnemosyne-model/whisper-backend
mnemosyne model-download openai/whisper-tiny
```

支持 WAV 文件（16 位或浮点，自动重采样到 16 kHz）。

```bash
mnemosyne index ~/Recordings   # 自动转录和索引
mnemosyne search "项目会议记录"
```

### ANN 向量搜索 (sqlite-vector + HNSW)

当索引大小超过 **2,000 个块**时，系统会自动切换到 HNSW进行近似最近邻搜索，将搜索速度从 O(n) 提高到 O(log n)。

或者，安装 [sqlite-vector](https://github.com/sqliteai/sqlite-vector) 扩展以在 SQL 端启用向量操作：

```bash
# macOS
curl -L https://github.com/sqliteai/sqlite-vector/releases/latest/download/sqlite_vector.dylib \
  -o ~/.mnemosyne/lib/sqlite_vector.dylib
```

系统在启动时会自动检测并加载它。

**推荐模型：**

| 用例 | 模型 | 维度 | 大小 |
|---|---|---|---|
| 文本（轻量级） | `sentence-transformers/all-MiniLM-L6-v2` | 384 | 90 MB |
| 文本（高质量）| `sentence-transformers/all-mpnet-base-v2` | 768 | 420 MB |
| 多语言 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | 470 MB |
| 图像 | `openai/clip-vit-base-patch32` | 512 | 350 MB |
| 音频 | `openai/whisper-tiny` | — | 75 MB |
| 音频（高质量）| `openai/whisper-base` | — | 140 MB |

---

## 数据存储

所有数据都存储在单个 SQLite 文件中（在 WAL 模式下）：

```
~/.mnemosyne/db.sqlite
  ├── files              # 文件元数据 + SHA-256 哈希
  ├── document_chunks    # 内容块（1500 字符，150 字重叠）
  ├── fts_chunks         # FTS5 全文索引（BM25, unicode61 分词器）
  ├── embeddings         # 向量（f32 小端字节序 BLOB）
  └── model_registry     # 已下载模型的记录
```

---

## 开发

```bash
# 运行单元和集成测试（11 个测试，包括端到端流水线）
cargo test -p mnemosyne-storage -p mnemosyne-retrieval

# 检查所有 crate 而不进行完整编译
cargo check --workspace

# 格式化代码
cargo fmt --all

# Lint
cargo clippy --workspace -- -D warnings
```

**集成测试覆盖范围：**
- 文件索引和统计
- FTS5 关键字搜索（BM25 排名验证）
- 向量搜索结果验证
- 混合 RRF 搜索
- 增量索引（跳过相同的哈希）
- 文件列表和删除
- 大文件分块

---

## 技术栈

| 组件 | 选择 | 理由 |
|---|---|---|
| 语言 | Rust 1.75+ | 内存安全，零成本抽象 |
| GUI 框架 | Tauri 2.x | 原生 WebView，跨平台 |
| 异步运行时 | Tokio 1.x | 与 Tauri 兼容，生态系统丰富 |
| 数据库 | SQLite (rusqlite 捆绑) | 无外部服务依赖 |
| 全文搜索 | SQLite FTS5 | 内置，BM25，unicode61 |
| 向量存储 | SQLite BLOB | 简单的初始实现；可迁移 |
| 嵌入模型 | Candle (可选) | 纯 Rust，CPU/Metal，无 Python 依赖 |
| HTTP 框架 | Axum 0.8 | Tower 生态系统，CORS，tracing |
| PDF 解析 | pdf-extract | 纯 Rust |
| 图像解析 | image crate | 纯 Rust，用于读取尺寸 |
| 文件观察 | notify + debouncer | 跨平台，500ms 防抖 |
| 错误处理 | thiserror + anyhow | `thiserror` 用于库，`anyhow` 用于应用 |

---

## 路线图

- [ ] **CLIP 图像嵌入** — 基于视觉内容的图像语义搜索。
- [ ] **Whisper 音频转录** — 将 MP3/WAV 转录为文本以供搜索。
- [ ] **视频关键帧提取** — 使用 ffmpeg + CLIP 进行帧描述。
- [ ] **`sqlite-vector` 集成** — 用于更大规模索引的 ANN 搜索。
- [ ] **多语言分词器** — 为 FTS5 添加 `jieba` 中文分词。
- [ ] **增量同步守护进程** — 在系统启动时运行的后台服务。
- [ ] **搜索结果高亮** — 在上下文中精确定位和高亮匹配的术语。

---

## 许可证

MIT 许可证 — 详情请参阅 [LICENSE](LICENSE) 文件。
