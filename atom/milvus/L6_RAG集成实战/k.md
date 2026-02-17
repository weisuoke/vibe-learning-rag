# L6_RAG集成实战 知识点列表

> 将 Milvus 2.6 应用于实际 RAG 系统，掌握 2026 年端到端的集成实践

---

## 知识点清单

1. **文档问答系统实现** - 使用 Embedding Functions 构建完整的文档问答 RAG 系统（Data-in, Data-out 模式）
2. **多租户知识库** - 实现多租户隔离的企业级知识库系统（支持 100K collections）
3. **Agentic RAG 实现** - 使用 LangGraph 构建智能决策的 Agentic RAG 系统（2026 年生产标准）
4. **Milvus 与 LangChain 集成** - 使用 LangChain 框架快速构建 RAG 应用（集成 Embedding Functions）
5. **Milvus 与 LlamaIndex 集成** - 使用 LlamaIndex 实现高级 RAG 模式（Agentic RAG）

---

## 学习顺序建议

```
文档问答系统实现 → 多租户知识库 → Agentic RAG实现 → LangChain集成 → LlamaIndex集成
       ↓                 ↓                ↓                ↓              ↓
   基础RAG          企业应用        2026生产标准      快速开发        高级模式
```

---

## 前置知识

- ✅ L1_快速入门（所有知识点）
- ✅ L2_核心功能（所有知识点）
- ✅ L3_高级特性（分区管理、混合检索、稀疏向量）
- ✅ atom/rag/L3_RAG核心流程（所有知识点）
- 🔶 atom/rag/L5_框架与落地（LangChain、LlamaIndex）

---

## 学习目标

完成本层级学习后，你将能够：
- ✅ 使用 Embedding Functions 从零构建完整的文档问答系统（无需外部 Embedding 预处理）
- ✅ 实现向量+BM25 混合检索的 RAG 系统
- ✅ 实现多租户数据隔离和权限控制（支持 100K collections）
- ✅ 使用 LangGraph 构建 Agentic RAG 系统（智能决策、查询改写、文档相关性评分）
- ✅ 优化大规模向量检索性能（RaBitQ 量化、热冷分层）
- ✅ 使用 LangChain 快速集成 Milvus 2.6
- ✅ 使用 LlamaIndex 实现高级 RAG 模式
- ✅ 处理实际生产环境中的各种挑战
- ✅ 评估和优化 RAG 系统质量

---

## 2026 核心特性

### Milvus 2.6 RAG 特性
- **Embedding Functions**: Data-in, Data-out 模式，简化 RAG 开发流程
- **混合检索**: 向量+BM25 成为标准模式
- **100K Collections**: 支持大规模多租户 SaaS 应用
- **RaBitQ 量化**: 十亿级规模成本优化

### 2026 RAG 模式
- **Agentic RAG**: 使用 LangGraph 构建智能决策系统
- **查询改写**: 自动优化查询质量
- **文档相关性评分**: 自动过滤无关文档
- **工具调用**: 集成多种工具的复杂 RAG 系统

---

## 预计学习时间

- 快速入门：6-8小时（知识点1-2：基础 RAG + 多租户）
- 完整学习：15-18小时（全部5个知识点）

---

## 核心应用场景

### 场景1：企业文档问答（使用 Embedding Functions）
```
原始文本 → Milvus 自动 Embedding → 向量+BM25 混合检索 → LLM 生成答案
```

### 场景2：多租户 SaaS 知识库
- 租户隔离（Partition 或 Collection）
- 支持 100K collections
- 权限控制
- 数据安全

### 场景3：Agentic RAG（2026 年生产标准）
```
用户查询 → Agent 决策（是否检索） → 查询改写 → 混合检索 → 文档相关性评分 → LLM 生成
```

### 场景4：电商商品推荐
- 商品向量化（图片+文本）
- 多向量检索
- 个性化推荐

### 场景5：代码搜索引擎
- 代码向量化
- 语义代码搜索
- 代码片段推荐

---

## 2026 vs 传统 RAG 对比

### 传统 RAG 流程（5步）
```
1. 加载文档
2. 调用外部 Embedding API
3. 插入向量到 Milvus
4. 向量检索
5. LLM 生成
```

### 2026 Milvus 2.6 RAG 流程（3步）
```
1. 插入原始文本到 Milvus（自动 Embedding）
2. 混合检索（向量+BM25）
3. LLM 生成
```

### 2026 Agentic RAG 流程（智能决策）
```
1. Agent 决策：是否需要检索？
2. 查询改写：优化查询质量
3. 混合检索：向量+BM25
4. 文档相关性评分：过滤无关文档
5. LLM 生成：基于高质量上下文
```

---

## 2026 学习重点

1. **Embedding Functions 是核心**：无需外部 Embedding 预处理
2. **混合检索是标准**：向量+BM25 不再是可选项
3. **Agentic RAG 是生产标准**：智能决策、查询改写、文档评分
4. **成本优化**：RaBitQ 量化、热冷分层在十亿级规模的应用
5. **多租户大规模**：100K collections 的 SaaS 应用

---

## RAG 质量优化策略

| 策略 | 目标 | 实现方式 |
|------|------|----------|
| 查询改写 | 提升检索质量 | LLM 改写、HyDE |
| 混合检索 | 平衡语义和关键词 | 向量+BM25 |
| ReRank | 提升相关性 | Cohere ReRank |
| 文档评分 | 过滤无关文档 | LLM 评分 |
| Agentic 决策 | 智能检索 | LangGraph |

---

**开始学习：** [01_文档问答系统实现](./01_文档问答系统实现/)
