---
type: fetched_content
source: https://docs.siliconflow.cn/cn/api-reference/embeddings/create-embeddings
title: 创建嵌入请求 - SiliconFlow
fetched_at: 2026-02-24T00:00:00Z
status: partial
author:
published_date:
knowledge_point: Milvus 2.6 Embedding Functions 深入
content_type: documentation
fetch_tool: Grok-mcp web-fetch
priority: high
word_count: 1350
---

# 创建嵌入请求 - SiliconFlow

## 元信息
- **来源**：https://docs.siliconflow.cn/cn/api-reference/embeddings/create-embeddings
- **作者**：
- **发布日期**：
- **抓取时间**：2026-02-24T00:00:00Z

## 内容摘要
该文档描述 SiliconFlow 的 Embeddings API（POST `/v1/embeddings`）：认证方式（Bearer Token）、请求参数（model/input/encoding_format/dimensions）、批量输入支持（input 可为字符串数组）、不同模型最大输入 token 限制（512/8192/32768）与响应结构（data 列表 + usage token 统计）。

---

## 正文内容

## Create Embeddings

### cURL 示例

```bash
curl --request POST \
  --url https://api.siliconflow.cn/v1/embeddings \
  --header 'Authorization: Bearer <token>' \
  --header 'Content-Type: application/json' \
  --data '
{
  "model": "BAAI/bge-large-zh-v1.5",
  "input": "Silicon flow embedding online: fast, affordable, and high-quality embedding services. come try it out!"
}
'
```

### 响应示例

```json
{
  "object": "list",
  "model": "<string>",
  "data": [
    {
      "object": "embedding",
      "embedding": [123],
      "index": 123
    }
  ],
  "usage": {
    "prompt_tokens": 123,
    "completion_tokens": 123,
    "total_tokens": 123
  }
}
```

### 认证

`Authorization: Bearer <token>`（header，required）

### 请求体（application/json）

- `model` (string, required)
  - 模型名称；可用模型列表见 Models。
- `input` (string | array, required)
  - 单条文本或批量文本数组；也可为 token 数组。
  - 输入不得为空字符串，且不得超过模型最大 token 限制。
  - 最大输入 tokens：
    - `BAAI/bge-large-zh-v1.5` / `BAAI/bge-large-en-v1.5` / `netease-youdao/bce-embedding-base_v1`: 512
    - `BAAI/bge-m3` / `Pro/BAAI/bge-m3`: 8192
    - `Qwen/Qwen3-Embedding-*`: 32768
- `encoding_format` (enum, default: float)
  - `float` 或 `base64`
- `dimensions` (integer)
  - 仅 `Qwen/Qwen3` 系列支持，且有允许的离散取值范围。

### 响应（200）

- header：`x-siliconcloud-trace-id`
- body：
  - `object`: "list"
  - `model`: 模型名
  - `data`: embeddings 列表（每项含 `embedding` 与 `index`）
  - `usage`: token 统计

---

## 关键信息提取

### 技术要点
- 支持批量：`input` 可以是字符串数组；但文档未给出“最大 batch 条数”，实际受 token/QPS/RPM/TPM 等限制约束。
- 不同模型 token 上限差异很大（512 → 8192 → 32768）。
- Qwen3 embedding 允许通过 `dimensions` 控制输出维度（离散取值）。

### 代码示例
- 见 cURL 示例。

### 相关链接
- https://docs.siliconflow.cn/cn/api-reference/embeddings/create-embeddings
- https://cloud.siliconflow.cn/models

---

## 抓取质量评估
- **完整性**：部分（页面存在重复块与侧边导航；按“激进清洗”保留核心 API 参考信息）
- **可用性**：高
- **时效性**：最新
