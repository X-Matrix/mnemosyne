# 架构常见问题解答

> 基于对当前代码库的实际 review，回答三个核心架构问题。

---

## Q1：如果增加新的 Embedding 模型，能不能快速做到？

**结论：可以，且架构对此有充分预留。** 但根据模型类别不同，所需工作量有差异。

### 架构支撑

系统在 `mnemosyne-core` 中定义了 `EmbeddingModel` trait（[crates/mnemosyne-core/src/traits.rs](../crates/mnemosyne-core/src/traits.rs)）：

```rust
pub trait EmbeddingModel: Send + Sync {
    fn model_id(&self) -> &str;
    fn embedding_dim(&self) -> usize;
    async fn embed_text(&self, text: &str) -> Result<Embedding>;
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>>;
}
```

实现该 trait 的任意结构体都可以被注册进 `ModelRegistry`。

### 当前已有三类模型槽位

| 槽位 | Feature Flag | 实现 | 用途 |
|------|-------------|------|------|
| text_model | `candle-backend` | BERT（all-MiniLM-L6-v2） | 文本 / 音频 transcript |
| vision_model | `clip-backend` | CLIP ViT | 图像 |
| audio_model | `whisper-backend` | Whisper | 语音转文字 |

每个槽位都有对应的 stub 实现（无 feature 时走哈希向量），所以模型是可选依赖。

### 新增模型的步骤（以新增 text embedding 模型为例）

1. **在 `mnemosyne-model/src/` 下新建实现文件**，用 `candle` 或其他推理框架实现推理逻辑。
2. **让新结构体实现 `EmbeddingModel` trait**（仅需实现 `model_id`、`embedding_dim`、`embed_text`、`embed_batch` 四个方法）。
3. **在 `ModelRegistry` 中添加对应的 `HashMap<String, Arc<NewEmbedder>>` 缓存字段**（见 [crates/mnemosyne-model/src/registry.rs](../crates/mnemosyne-model/src/registry.rs)），并添加 `get_new_embedder()` 懒加载方法。
4. **在 `SearchEngine` 中添加新的 `model_id` 字段**（见 [crates/mnemosyne-retrieval/src/engine.rs](../crates/mnemosyne-retrieval/src/engine.rs)），暴露 `get_xxx_model()` / `set_xxx_model()` 方法。
5. **在 `embed_chunks()` 的路由逻辑中**，为新文件类型或新 chunk 类型添加分支。

### 注意事项（当前代码的实际限制）

- **向量维度不可混用**：`HybridIndex` 在 `vector_search` 里用 `query_dim` 过滤（`BERT=384, CLIP=512 — never mix!`）。切换同一槽位的模型后，**历史 embedding 与新模型维度不兼容，必须重新索引**。`SearchEngine::set_text_model()` 的注释中已明确提示这一点。
- **Cargo feature 门控**：新模型最好通过 feature flag 可选引入，避免给不需要该模型的编译目标增加依赖体积。
- **模型下载**：`ModelDownloader`（基于 `hf-hub`）已支持从 HuggingFace Hub 拉取，本地缓存路径为 `~/.mnemosyne/models/{model_id}/`，新模型只需遵循同样的目录结构即可复用下载逻辑。

---

## Q2：如果文件本身不是文本，而是 mp3 / pdf / 图片，是如何构建索引的？

**结论：通过"解析 → 转化为文本/向量 → 嵌入统一索引"的流水线，所有文件类型最终都被归一化为可检索的向量+文本。**

### 整体流水线

```
文件路径
  │
  ▼
ParserRegistry.parse(path)          ← 按扩展名路由到对应解析器
  │
  ▼
Vec<ParsedContent>                  ← 解析结果（可能包含多个 chunk）
  │
  ▼
SearchEngine::embed_chunks(path, chunks)   ← 按 chunk 类型路由到模型
  │
  ├─ Image    → CLIP 视觉编码器（clip-backend）或 caption 文本嵌入
  ├─ Audio    → Whisper 转录 → 文本嵌入（BERT）
  └─ Text/PDF → 文本嵌入（BERT）
  │
  ▼
Vec<Embedding>
  │
  ▼
HybridIndex.upsert(chunk)           ← 写入 FTS5 + embedding blob
```

### 三种非文本文件的具体处理

#### MP3 / 音频（wav, flac, aac, ogg, m4a, opus）

1. `AudioParser::parse()` 返回一个占位 `ParsedContent::AudioTranscript`，内容为文件名 + 格式（stub）。
2. 在 `engine.rs::index_file()` 中，如果编译了 `whisper-backend` feature，会调用 `transcribe_audio()` 替换 stub 内容，用 Whisper 模型做真实语音识别，得到转录文本。
3. `embed_chunks()` 对 `AudioTranscript` chunk 调用 **BERT 文本嵌入**，将转录文本转化为向量。
4. 转录文本同时写入 FTS5 全文索引，支持关键词搜索。

> 当前状态：Whisper 模型推理框架（`WhisperTranscriber`、`symphonia` 解码）已实现，属于 Phase 4-5 待集成阶段，stub 路径可正常运行。

#### PDF

1. PDF 通过 `PdfParser`（需启用 `pdf` feature，依赖 `pdf-extract` crate）解析为若干 `ParsedContent::Text` chunk。
2. 之后与普通文本文件完全相同：BERT 嵌入 + FTS5 写入。
3. 统计时 PDF 单独计入 `"Pdf"` 类型（从 `"Text"` 桶中分离），方便前端侧边栏区分展示。

#### 图片（jpg, jpeg, png, bmp, webp, tiff, gif, heic）

1. `ImageParser::parse()` 读取图像尺寸（需 `image-meta` feature），返回 `ParsedContent::Image { caption, tags }`，caption 形如 `"Image: photo.jpg (1920×1080 pixels)"`。
2. `embed_chunks()` 对 `Image` chunk 的路由：
   - **有 `clip-backend`**：调用 CLIP 视觉编码器对原始图片像素编码，生成 512 维向量。
   - **无 `clip-backend`**：对 caption 文本做 BERT 嵌入（降级）。
3. CLIP 向量写入 embedding 表；caption 文本写入 FTS5（可关键词搜索文件名/尺寸）。

### ParserRegistry 的扩展性

```rust
pub fn register(&mut self, parser: Arc<dyn FileParser>) { ... }
```

只需实现 `FileParser` trait（`supported_extensions()` + `parse()`），调用一次 `register()` 即可新增文件格式支持，无需修改其他代码。

---

## Q3：给出 query 后，是如何同时 query 到文本和 mp3 / pdf / 图片的？

**结论：所有文件类型在索引时已被归一化到同一个索引（FTS5 + embedding 表），查询时统一检索，通过"双塔"策略在向量空间对齐文本与图像。**

### 查询入口

`SearchEngine::search(query)` 是唯一入口（[crates/mnemosyne-retrieval/src/engine.rs](../crates/mnemosyne-retrieval/src/engine.rs)），支持三种模式：

| 模式 | 说明 |
|------|------|
| `Vector` | 纯向量 cosine 检索 |
| `Keyword` | 纯 FTS5 BM25 关键词检索 |
| `Hybrid` | 向量 + 关键词 → RRF 融合排名 |

### 文本/音频/PDF 的检索（统一 BERT 空间）

```
query.text
  │
  ▼
BERT text embedder → query_embedding (384-dim)
  │
  ├─ vector_search(query_embedding)   → 匹配所有 384-dim chunks（文本、PDF、音频 transcript）
  └─ keyword_search(query.text)       → FTS5 全文匹配（同一张 fts_chunks 表）
```

文本、PDF、音频文件的内容在索引时都被转化为 BERT 向量（384 维），因此一次向量检索可以同时命中三种类型。

### 图片的检索（CLIP 双塔）

图片在索引时用 CLIP **视觉编码器**（512 维）；检索时用 CLIP **文本编码器**对 query 编码（同为 512 维）。CLIP 的两个塔共享同一个语义空间，使文字查询能与图片向量直接比较。

```
query.text
  │
  ▼
CLIP text encoder → clip_text_embedding (512-dim)
  │
  ▼
vector_search(clip_text_embedding)   → 仅命中 512-dim 的 Image chunks
```

代码实现（`engine.rs` search 方法）：

```rust
// 先用 BERT 检索文本/音频/PDF
let bert_embedding = bert_model.embed_text(&query.text).await?;
let mut results = self.index.vector_search(&bert_embedding, limit).await?;

// 仅在 Vector 模式下追加 CLIP 图像结果（Hybrid/Keyword 下跳过，避免噪声）
#[cfg(feature = "clip-backend")]
if matches!(&query.mode, SearchMode::Vector) {
    let clip_text_emb = self.embed_text_with_clip(&query.text).await?;
    let mut clip_results = self.index.vector_search(&clip_text_emb, limit * 2).await?;
    // 噪声过滤：cosine < 0.26 (score < 0.63) 的图片结果丢弃
    clip_results.retain(|r| r.score >= CLIP_MIN_SCORE);
    // 合并，去重，按 score 重排
    results.extend(clip_results);
    results.sort_by(|a, b| b.score.partial_cmp(&a.score)...);
    results.truncate(limit);
}
```

`HybridIndex::vector_search` 内部按 `query_dim` 过滤，384-dim 查询只匹配 BERT 向量，512-dim 查询只匹配 CLIP 向量，不会混淆。

### 文件类型过滤

`SearchQuery` 支持 `file_types: Option<Vec<FileType>>` 字段，在合并结果后按类型过滤：

```rust
if let Some(ref types) = query.file_types {
    results.retain(|r| types.contains(&r.file_record.file_type));
}
```

用户可以只看图片、只看音频，或不加过滤同时返回所有类型。

### 完整检索流程图

```
用户 query: "a red sunset"
          │
          ├─ BERT(384d) ─────────────────────────────────────────┐
          │     ↓                                                 │
          │  vector_search(384d) ───→ 文本 chunk matches          │
          │  keyword_search       ───→ FTS5 matches               ├─→ RRF 融合 → 最终排名
          │                                                        │
          └─ CLIP text encoder(512d) ─→ vector_search(512d) ─────→ 图片 chunk matches
                                        (仅 Vector 模式，score≥0.63)
```

---

## 总结：设计符合度评估

| 问题 | 架构符合度 | 备注 |
|------|-----------|------|
| 新增 Embedding 模型 | ✅ 高 | trait 抽象 + feature flag + ModelRegistry 缓存，步骤清晰 |
| 非文本文件索引 | ✅ 高 | ParserRegistry 统一解析路由，embed_chunks 按内容类型分流 |
| 跨类型统一查询 | ✅ 高 | BERT/CLIP 双塔共存，同一 HybridIndex，维度隔离不混淆 |

**唯一需注意的实际缺口**：Whisper 和 CLIP 推理（Phase 4-5）尚未集成到 release 构建（feature flag 默认关闭），当前 stub 路径可运行但无真实语义能力。架构设计本身已完备支持这两条路径。
