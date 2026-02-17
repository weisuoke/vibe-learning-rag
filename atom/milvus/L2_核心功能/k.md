# L2_核心功能 知识点列表

> 掌握 Milvus 2.6 的核心功能，理解 2026 年生产环境必备技能

---

## 知识点清单

1. **向量索引类型** - 理解 FLAT、IVF_FLAT、HNSW、GPU CAGRA、RaBitQ 等索引类型及其适用场景
2. **相似度度量** - 掌握 L2、IP、COSINE 等距离度量方式的选择
3. **混合检索（向量+全文）** - 掌握向量检索与 BM25 全文搜索的混合策略（2026 年核心技能）
4. **标量过滤与 JSON 索引** - 实现向量检索与标量条件的混合查询，使用 JSON Path Index
5. **Embedding Functions 深入** - 掌握 Milvus 2.6 内置 Embedding 能力，简化 RAG 开发流程

---

## 学习顺序建议

```
向量索引类型 → 相似度度量 → 混合检索 → 标量过滤与JSON索引 → Embedding Functions深入
    ↓              ↓           ↓              ↓                    ↓
  性能优化      精确匹配    2026核心技能   复杂元数据过滤      简化开发流程
```

---

## 前置知识

- ✅ L1_快速入门（所有知识点）
- ✅ 向量检索原理
- ✅ BM25 全文搜索原理（推荐）

---

## 学习目标

完成本层级学习后，你将能够：
- ✅ 根据数据规模选择合适的索引类型（包括 GPU CAGRA、RaBitQ）
- ✅ 根据 Embedding 模型选择度量方式
- ✅ 实现向量+BM25 混合检索（2026 年生产环境标准）
- ✅ 使用 JSON Path Index 实现复杂元数据过滤
- ✅ 理解索引参数对性能的影响
- ✅ 使用 Milvus 2.6 Embedding Functions 简化 RAG 开发
- ✅ 配置多种 Embedding 提供商（OpenAI、Cohere、Bedrock 等）

---

## 2026 核心特性

### Milvus 2.6 新特性
- **Embedding Functions**: Data-in, Data-out 模式，无需外部预处理
- **RaBitQ 量化**: 72% 内存节省 + 4x 性能提升
- **JSON Path Index**: 100x 嵌套 JSON 查询性能提升

### Milvus 2.5 核心特性
- **BM25 全文搜索**: 4x 快于 Elasticsearch
- **混合检索**: 向量+全文搜索成为标准模式

### Milvus 2.4 核心特性
- **GPU CAGRA 索引**: GPU 加速向量检索
- **稀疏向量支持**: 原生支持稀疏向量索引

---

## 预计学习时间

- 快速入门：4-5小时（核心3个知识点：索引、混合检索、Embedding Functions）
- 完整学习：10-12小时（全部5个知识点）

---

## 2026 学习重点

1. **混合检索是核心技能**：不再是高级特性，而是生产环境标准
2. **Embedding Functions 简化开发**：无需外部 Embedding 预处理
3. **成本优化索引**：RaBitQ 在十亿级规模下的应用
4. **GPU 加速**：CAGRA 索引在高性能场景的应用

---

**开始学习：** [01_向量索引类型](./01_向量索引类型/)
