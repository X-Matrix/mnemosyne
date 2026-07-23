---
title: "Mnemosyne：给本地文件装上可检索的记忆"
date: 2026-07-19
tags: [Rust, 本地优先, 语义搜索, Agent, Candle, SQLite, 多模态]
description: "一个纯 Rust 实现的本地多模态文件检索系统，试图解决 Agent 外部记忆、多模态索引和轻量推理三个问题。"
---

## 那个没被填上的空格

大多数 Agent 框架谈记忆，讲的是对话历史和短期上下文。RAG 系统谈知识库，解决的是外部文档的批量入库。

但有一块地方一直空着：**你机器上那几万个文件**。

它们不是 PDF 知识库，也不是对话日志。它们是截图、录音、代码草稿、会议记录、图片，散落在各个目录里，没有结构，没有标签，没有嵌入。每次你想找什么，要么靠 Spotlight 的文件名搜索碰运气，要么打开 Finder 翻半天。

这是 Agent 系统现在解决不了的：**把用户本地文件变成可语义检索的长期外部记忆**。

Mnemosyne 是我为这个问题写的答案。

---

## 四个约束

项目开始时给自己定了四条约束。

**第一，多模态从一开始就要支持。** 本地文件的现实是混合的，txt 和 png 和 mp3 共存于同一个文件夹。一个只做文本索引的方案不够用。

**第二，脱离 Python 和 PyTorch。** 模型推理一定要能部署在没有 Python 环境的机器上，冷启动不能靠几百 MB 的运行时。这把工具链锁定到了 Rust，推理层锁定到了 HuggingFace 的 [Candle](https://github.com/huggingface/candle)。

**第三，零外部服务。** 不起 Qdrant，不起 Elasticsearch，不起 PostgreSQL。索引、向量、全文搜索全部压进一个 SQLite 文件。用户的数据就在 `~/.mnemosyne/db.sqlite`，随时可以删。

**第四，同时服务 GUI 用户和 Agent。** 人需要一个可以点击、可以预览的界面；Agent 需要一个 REST API。两个接口，同一套核心。

---

## 方案

系统是六个 Rust crate 的层叠，依赖方向单向：

```
mnemosyne-core
  ← mnemosyne-storage   (SQLite CRUD + sqlite-vec)
  ← mnemosyne-model     (Candle 推理)
  ← mnemosyne-parser    (多格式解析)
  ← mnemosyne-index     (HybridIndex)
    ← mnemosyne-retrieval  (SearchEngine 门面)
      ← cli / api / src-tauri
```

### 存储

所有持久化压进单个 SQLite 文件，WAL 模式，无网络依赖：

- `files` — 文件路径、类型、大小、SHA-256 哈希
- `document_chunks` — 内容分块，带 FTS5 触发器自动维护全文索引
- `embeddings` — f32 小端字节序 BLOB，每条记录关联一个 chunk
- `embedding_vec_N` — sqlite-vec 的 vec0 虚拟表，按维度分表（384/512/1024）

向量检索走 [sqlite-vec](https://github.com/asg017/sqlite-vec)，在 SQLite 内部实现 HNSW。它是一个运行时动态加载的 `.dylib`，不存在时系统回退到纯 Rust 的 `instant-distance` 实现。切换对上层透明，同一套 SQL 接口。

变更检测用 SHA-256 哈希，文件内容未变则跳过重新嵌入，避免无效的推理开销。

### 解析

解析层设计为插件注册表，每种文件类型对应一个 `Parser` 实现：

| 类型 | 解析策略 |
|---|---|
| 文本（txt/md/json/py/rs...） | 按字符滑动窗口分块，1500 字符，150 字重叠 |
| PDF | pdf-extract 提取文本后分块 |
| 图像 | 提取尺寸元数据 + CLIP 视觉编码器生成 512 维嵌入 |
| 音频 | symphonia 解码 → Whisper 转录 → 文本嵌入 |

每种类型输出统一的 `ParsedContent` 枚举，后续嵌入管道不感知格式差异。图像的嵌入维度（512）和文本嵌入维度（384/1024）不同，存入不同的 vec0 虚拟表，检索时分别查询再合并。

### 模型

推理层完全基于 [Candle](https://github.com/huggingface/candle)，无 Python 运行时，无 ONNX Runtime。三类模型槽位：

**文本嵌入**：支持 BERT 架构的所有模型。`config.json` 里的 `model_type` 字段决定池化策略：`bert` 用均值池化，`xlm-roberta`（如 BGE-M3）用 CLS 池化。BGE-M3（BAAI）输出 1024 维，支持中英文等 100+ 语言，是目前的默认推荐。

**图像嵌入**：OpenAI CLIP 走 `clip::ClipModel`，Chinese CLIP（OFA-Sys）走 `chinese_clip::ChineseClipModel`，两者共用同一个 `ClipEmbedder` 结构体，通过 model_id 前缀自动分发。图像预处理按各自原始实现对齐：OpenAI CLIP 用 CatmullRom 滤波器，Chinese CLIP 用 Triangle 滤波器，均做 `resize_to_fill` 保持宽高比。

**音频转录**：Whisper 推理，symphonia 负责音频解码，支持 MP3/WAV/FLAC/OGG，内部统一重采样到 16kHz。

模型权重缓存在 `~/.mnemosyne/models/`，按需加载，ModelRegistry 持有弱引用缓存，不用的模型不常驻内存。Apple Silicon 上编译时加 `metal-backend` feature 即可启用 Metal GPU 加速，build.sh 在 arm64 macOS 上自动附加。

### 检索

一次搜索并发走两条路径，最终通过 RRF 融合：

```
query
  ├─ BERT/BGE-M3 编码 → vec0 KNN → 向量结果（余弦相似度）
  └─ FTS5 BM25        → 关键词结果（sigmoid 归一化）
               ↓
            RRF 融合
               ↓
         [如果 clip-backend 可用]
               ↓
  CLIP 文本编码 → vec0 KNN（512/1024 维表）→ 图像结果
               ↓
         按余弦相似度全局排序
```

RRF 公式：$\text{score}(d) = \sum_r \frac{w_r}{k + \text{rank}_r(d)}$，权重可在 GUI 或 API 里调节。

向量得分是真实余弦相似度（vec0 返回 L2 距离，内部换算 $\cos = 1 - \text{L2}^2/2$）。关键词得分做 sigmoid 归一化：$s = |bm25| / (|bm25| + 10)$，避免 BM25 原始值超过 1.0 的歧义。

混合结果和图像结果统一按得分降序排列，不区分文件类型。

### 目录过滤

索引时用 `WalkDir::filter_entry` 在进入目录前剪枝，默认跳过 60+ 类目录（`node_modules`、`.git`、`target`、`__pycache__` 等）。过滤规则在 `IgnoreConfig` 里硬编码，支持运行时通过 `.with_extra()` / `.allow()` 扩展。文件监听器同样检查路径组件，从 watcher 事件中过滤来自忽略目录的变更。

---

## 现在能做什么

索引一个目录，只需要一行：

```bash
mnemosyne index ~/Documents
```

之后可以用中文自然语言搜索，也可以搜图片，也可以混搜：

```bash
mnemosyne search "机器学习相关的会议记录"
mnemosyne search "大楼" --mode vector   # 图片也会出现
```

文件有变动，监听进程会自动重建索引；文件被删除，索引里的记录也随之消失。

GUI 里可以切换嵌入模型，配置 HuggingFace 镜像源，查看实时日志，预览 PDF 和图片，甚至内嵌播放音频。

REST API 暴露了搜索、索引、文件管理和模型下载接口，一行 curl 就能调用：

```bash
curl -X POST http://localhost:8080/api/search \
  -d '{"text": "项目复盘", "mode": "hybrid", "limit": 10}'
```

---

## 还没做完的事

几个问题现在还没有好答案。

**中文全文检索的颗粒度。** FTS5 的 trigram 分词器可以处理中文，但颗粒度粗，召回率不如专门的分词器。jieba 或 jieba-rs 是候选，但要嵌进 SQLite 自定义分词器需要一些工程量。

**视频内容理解。** 视频文件现在只索引文件名和元数据，关键帧提取和 CLIP 描述还没有做。

**Agent 工具协议。** REST API 已经可以被 Agent 调用，但还没有适配 MCP（Model Context Protocol）或 function calling 的标准格式。这层包装不复杂，但需要做。

**增量重嵌入。** 切换嵌入模型后，旧的向量需要重建。现在的方案是手动重新索引，没有自动迁移。

---

## 一个小注脚

项目名字来自希腊神话里的记忆女神 Mnemosyne。她是缪斯的母亲，也是遗忘的对立面。

给本地文件建索引这件事，本质上是在和遗忘对抗。你拍了那么多截图，写了那么多草稿，录了那么多会议，最后大多沉在硬盘里，永远不会再被看见。

这个项目试图让它们重新可寻。
