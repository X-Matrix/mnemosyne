# 搜索策略设计说明

> 记录混合检索、CLIP 图像搜索及文件类型过滤的关键决策，供后续迭代参考。

---

## 1. 三种搜索模式

| 模式 | 触发条件 | 向量来源 | FTS5 | 图像 CLIP |
|------|----------|----------|------|-----------|
| **向量** | `SearchMode::Vector` | BERT 384-dim | ❌ | ✅（宽松阈值） |
| **混合** | `SearchMode::Hybrid` | BERT 384-dim | ✅（RRF 融合） | ✅（严格阈值） |
| **关键词** | `SearchMode::Keyword` | ❌ | ✅ | ❌ |

---

## 2. CLIP 图像搜索策略

### 为什么图像需要独立的搜索通道

- 文本/PDF/音频使用 **BERT 384-dim** 向量空间
- 图像使用 **CLIP 512-dim** 向量空间

两者维度不同、语义空间完全独立。`hybrid_search` 内部的 `vector_search` 按维度过滤，BERT 查询永远看不到 CLIP 向量，反之亦然。因此图像必须通过单独的 CLIP 文本→图像搜索通道注入。

### 各模式的 CLIP 参数

```
Vector 模式
  threshold = 0.63  (cosine ≥ 0.26)
  cap       = limit × 2（无硬性上限）
  merge     = 与 BERT 结果合并后按 cosine 分数重排

Hybrid 模式
  threshold = 0.75  (cosine ≥ 0.50)
  cap       = 5 张（硬性上限）
  merge     = 追加到文本结果之后，不重排
              （避免 RRF/cosine 量级不匹配问题，见下节）

Keyword 模式
  不启用 CLIP（FTS5 对图像 caption 无语义价值）
```

### 为什么 Hybrid 模式不重排

RRF 和 CLIP cosine 分数量级完全不同：

| 来源 | 计算公式 | 典型值 |
|------|----------|--------|
| RRF 文本/PDF | `1 / (60 + rank)` | 0.016 |
| CLIP 图像 cosine | `(cosine + 1) / 2` | 0.63–0.85 |

合并后统一按分数排序，CLIP 图像永远排在文本结果之前，与查询相关度无关。因此 Hybrid 模式不重排，直接将图像追加到文本结果末尾。

### 为什么 Hybrid 模式 threshold = 0.75

- 随机桌面截图对任意文字查询的 CLIP cosine 约 **0.30**（分数约 0.65）
- `threshold = 0.63` 会让数百张无关截图通过，淹没文本结果
- `threshold = 0.75`（cosine ≥ 0.50）排除噪声，保留真正相关的图像
  （如查询「Call Me By Your Name」对应电影海报的 cosine 通常 ≥ 0.55）

---

## 3. PDF 文件类型设计

### 历史问题

PDF 曾被归类为 `FileType::Text`，导致：
- 无法通过 `SearchQuery::file_types = [Pdf]` 后端过滤，只能客户端按文件名 `.pdf` 过滤
- 统计数据需要额外的 SQL 查询 hack 才能单独计数 PDF

### 当前方案

`FileType::Pdf` 是独立枚举变体（序列化为 `"pdf"`）：

```rust
pub enum FileType { Text, Pdf, Image, Audio, Video, Unknown }
```

- `FileType::from_extension("pdf")` → `FileType::Pdf`
- 后端可直接 `file_types: ["pdf"]` 过滤
- `get_stats` 不再需要按路径扩展名计数的 hack
- PDF 内容仍通过 `ParsedContent::Text { text }` 进入 BERT 索引（解析层不变）

---

## 4. 文件类型过滤的端到端流程

```
用户点击侧边栏 [PDF] 过滤按钮
  │
  ▼
GUI: doSearch() 发送 file_types: ["pdf"]
  │
  ▼
Tauri → SearchEngine::search()
  │
  ├─ 执行 BERT hybrid_search / vector_search / keyword_search
  │
  ├─ filter: results.retain(|r| r.file_record.file_type == FileType::Pdf)
  │
  └─ 返回仅含 PDF 的结果列表
```

`image`、`audio`、`video`、`text`、`pdf` 过滤全部由后端处理，无客户端后处理。

---

## 5. 搜索参数完整说明

`SearchQuery` 结构（序列化后传给 Tauri invoke）：

```typescript
{
  text:            string,      // 查询文本
  mode:            "hybrid" | "vector" | "keyword",
  limit:           number,      // 返回条数（默认 20）
  offset:          number,      // 分页偏移
  file_types:      string[] | null, // ["text","pdf","image","audio","video"] 或 null=全部
  vector_weight:   number,      // 向量结果 RRF 权重（默认 1.0）
  keyword_weight:  number,      // 关键词结果 RRF 权重（默认 1.0）
}
```

---

## 6. 待优化项

| 问题 | 描述 | 优先级 |
|------|------|--------|
| RRF 分数归一化 | RRF (~0.016) 与 BERT cosine (0.5-1.0) 量级不匹配，目前 Hybrid 通过"不重排"规避 | P2 |
| CLIP 阈值自动校准 | 目前 0.75 是经验值，可根据实际数据分布自动校准 | P3 |
| PDF 向量质量 | 依赖 `pdf-extract` 提取文本；复杂排版/扫描件质量差 | P2 |
| 视频帧 CLIP | 视频目前无 embedding（stub），Phase 6 目标 | P3 |
