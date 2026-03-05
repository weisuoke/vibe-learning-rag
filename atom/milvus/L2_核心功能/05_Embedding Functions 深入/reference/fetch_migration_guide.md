---
type: fetched_content
source: https://github.com/milvus-io/milvus/discussions/44016
title: Does milvus support adding embedding functions to already created collections? · milvus-io/milvus · Discussion #44016
fetched_at: 2026-02-24T00:00:00Z
status: partial
author:
published_date:
knowledge_point: Milvus 2.6 Embedding Functions 深入
content_type: discussion
fetch_tool: Grok-mcp web-fetch
priority: high
word_count: 170
---

# Does milvus support adding embedding functions to already created collections?

## 元信息
- **来源**：https://github.com/milvus-io/milvus/discussions/44016
- **作者**：
- **发布日期**：
- **抓取时间**：2026-02-24T00:00:00Z

## 内容摘要
该讨论询问：已有 schema/index 且手动写入向量的数据集合，能否“后加”Embedding Function 以免手工上传向量。讨论结论偏向当前不支持，并提出未来可能新增 add_function/delete_function 或通过 modify collection 实现。

---

## 正文内容

I have a collection with a schema and index already created, and I've manually uploaded some vector data. Can I add an embedding function to this collection so that I don't need to manually upload vector data?

probably you can't. unless you are already using same embedding models as milvus supported

What models does milvus support? After I created a collection using the Python SDK, I couldn't find a way to update the Embedding Function for the collection.

Another idea is, is it possible to write a self-generated vector into a Collection with an Embedding Function?

unfortunately right now there is no support like it. but it's not hard to add one. @junjiejiangjjj please take this into consideration. you will need to write your own functions. that's the goal of embedding function project. Right now it's just the first step so user defined function is not supported. right now all the supported model is listed here we will gonna to support more models on Zilliz clou!

We can consider adding two interfaces, add_function and delete_function , to implement related logic

maybe we can simply do motify collection instead of adding new interface?

Currently, the interfaces for modifying a collection are alter_collection_field and alter_collection_properties , but neither can be used to modify a function, so a new interface still needs to be added.

@junjiejiangjjj The alter_collection_field API only works when connected to the default database. I’m connected with a non-default DB, yet the call still targets default.

don't think so. you need to use database before alter collection

---

## 关键信息提取

### 技术要点
- 当前（讨论时点）不支持对已存在 collection 追加/修改 embedding function。
- 可能需要新增接口：`add_function` / `delete_function`，或扩展现有修改 collection 的能力。
- 相关 API 背景：`alter_collection_field` / `alter_collection_properties` 无法修改 function；且存在非 default DB 连接时 alter 行为的疑问。

### 代码示例
- 无

### 相关链接
- https://github.com/milvus-io/milvus/discussions/44016

---

## 抓取质量评估
- **完整性**：部分（GitHub 页面渲染导致作者/时间戳未完整提取）
- **可用性**：中
- **时效性**：较新
