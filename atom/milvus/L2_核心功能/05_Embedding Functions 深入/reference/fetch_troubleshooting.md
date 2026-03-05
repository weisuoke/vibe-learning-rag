---
type: fetched_content
source: https://github.com/milvus-io/milvus/issues/32716
title: [Bug]: docker crash when inserts more data · Issue #32716 · milvus-io/milvus
fetched_at: 2026-02-24T00:00:00Z
status: success
author: KuroKienDinh
published_date:
knowledge_point: Milvus 2.6 Embedding Functions 深入
content_type: discussion
fetch_tool: Grok-mcp web-fetch
priority: high
word_count: 330
---

# [Bug]: docker crash when inserts more data

## 元信息
- **来源**：https://github.com/milvus-io/milvus/issues/32716
- **作者**：KuroKienDinh
- **发布日期**：
- **抓取时间**：2026-02-24T00:00:00Z

## 内容摘要
该 Issue 描述：在 Windows 上以 Docker standalone GPU 方式部署 Milvus 2.4.0，插入到 10M entities（768 维，IVF_SQ8，batchsize=10000）时容器崩溃，表现为 CPU 100% 且 RAM 不足；Issue 贴中 Steps/Logs 为空，并被标注 triage/needs-information。

---

## 正文内容

### Environment

- Milvus version: 2.4.0
- Deployment mode: standalone
- MQ type: docker
- SDK version: 2.4.0
- OS: window
- CPU/Memory: 64g ram
- GPU: 24g ram
- Others: docker 50gb/60gb ram

### Current Behavior

When I insert upto 10M entities, docker crash then milvus disconnect. As I check because of cpu usage 100% and there no available RAM.

I use IVF_SQ8 index, each vectors 768 dimension.

I install milvus docker gpu version.

I use batchsize insert 10000 entities one time.

I think cpu and ram won't increase when we insert data?

### Expected Behavior

Cpu and ram shouldn't OOM because only 10M entities

### Steps To Reproduce

(empty)

### Milvus Log

(empty)

---

## 关键信息提取

### 技术要点
- 该报告缺少复现步骤与日志，无法直接给出结论；但指向“大批量插入 + 资源限制/配置”导致的 OOM/崩溃。
- triage/needs-information 表明需要补充：容器资源限制、日志、具体 insert 参数与索引/flush/compaction 状态。

### 代码示例
- 无

### 相关链接
- https://github.com/milvus-io/milvus/issues/32716

---

## 抓取质量评估
- **完整性**：完整（原 Issue 本身 Steps/Logs 为空）
- **可用性**：中
- **时效性**：较新
