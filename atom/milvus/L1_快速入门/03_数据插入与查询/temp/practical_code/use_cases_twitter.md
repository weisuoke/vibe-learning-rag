---
source: Grok-mcp web search results
query: Milvus 2.6 Data-in Data-out pattern 2026
platform: Twitter
fetched_at: 2026-02-21
---

# Twitter/X Discussions: Milvus 2.6 Data-in Data-out Pattern

## Search Results

### 1. Introducing Milvus 2.6: Affordable Vector Search at Billion Scale
**URL**: https://milvus.io/blog/introduce-milvus-2-6-built-for-scale-designed-to-reduce-costs.md
**Description**: Milvus 2.6 引入 'Data-In, Data-Out' 体验，通过 Function 接口集成第三方嵌入模型，实现原始文本直接插入和自然语言查询，简化向量转换流程。

### 2. Introducing the Embedding Function: How Milvus 2.6 Streamlines Vectorization and Semantic Search
**URL**: https://milvus.io/blog/data-in-and-data-out-in-milvus-2-6.md
**Description**: 详细解释 Milvus 2.6 的 Data-in, Data-out 特性，基于新 Function 模块内置嵌入生成，支持直接插入原始数据并自动向量化，连接 OpenAI、Hugging Face 等模型。

### 3. Milvus 2.6 Preview: 72% Memory Reduction Without Compromising Recall and 4x Faster Than Elasticsearch
**URL**: https://milvus.io/blog/milvus-26-preview-72-memory-reduction-without-compromising-recall-and-4x-faster-than-elasticsearch.md
**Description**: Milvus 2.6 预览介绍 Data-In, Data-Out 功能，通过 Function 接口直接处理原始文本到向量搜索，简化嵌入管道。

### 4. Release Notes | Milvus Documentation
**URL**: https://milvus.io/docs/release_notes.md
**Description**: Milvus 官方发布说明，突出 2.6 版本引入 'Data-in, Data-Out' 能力，直接集成第三方嵌入模型简化 AI 应用开发。

### 5. Milvus 2.6.x GA on Zilliz Cloud, Making Vector Search Faster
**URL**: https://zilliz.com/blog/milvus-2-6-ga-on-zilliz-cloud
**Description**: Zilliz 云上 Milvus 2.6 GA 发布，强调 Data in, data out 体验优化向量搜索工作流。

## Key Insights from Community Discussions

### Data-in, Data-out Pattern Benefits
1. **Simplified Workflow**: 3步流程替代传统5步流程
   - 传统: 准备数据 → 生成嵌入 → 插入向量 → 查询向量 → 映射结果
   - 新模式: 插入原始数据 → 查询原始数据 → 获取结果

2. **Reduced Infrastructure**: 无需维护独立的嵌入服务
3. **Consistency**: 插入和查询使用相同的嵌入模型
4. **Developer Experience**: 更简单的API，更少的代码

### Technical Implementation
- **Function Module**: 核心实现机制
- **Provider Integration**: 统一接口支持多提供商
- **Automatic Vectorization**: 透明的向量化过程
- **Configuration Management**: 集中式凭证管理

### Use Cases
- **RAG Applications**: 简化RAG管道构建
- **Semantic Search**: 直接文本搜索
- **Document Q&A**: 文档问答系统
- **Knowledge Base**: 知识库检索

### Community Feedback
- Positive reception for simplified workflow
- Interest in multi-provider support
- Questions about performance impact
- Requests for more embedding model options
