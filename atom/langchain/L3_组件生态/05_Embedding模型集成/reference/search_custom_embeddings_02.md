---
type: search_result
search_query: custom embeddings implementation LangChain 2025 2026 tutorial
search_engine: grok-mcp
searched_at: 2026-02-25
knowledge_point: Embedding模型集成
---

# 搜索结果：自定义 Embeddings 实现教程

## 搜索摘要

搜索关键词：custom embeddings implementation LangChain 2025 2026 tutorial
平台：GitHub, Reddit
结果数量：8个

## 相关链接

1. [自定义RAG代理与LangChain嵌入对比](https://github.com/vmayakumar/rag-agents)
   - 全面指南对比自定义嵌入实现与LangChain LangGraph RAG代理，含嵌入管理器代码示例

2. [LangChain RAG产品推荐教程](https://github.com/maryamariyan/langchain-rag-tutorial)
   - 演示自定义替换Embeddings模型，从HuggingFace切换OpenAI实现灵活RAG向量搜索

3. [GenAI LangChain 2026生产RAG](https://github.com/AdilShamim8/GenAI-Roadmap-with-Notes-Using-LangChain)
   - 2026生产级RAG实现笔记，使用text-embedding-3-large嵌入模型与Chroma存储

4. [bRAG-langchain高级嵌入教程](https://github.com/bragai/bRAG-langchain)
   - 笔记本详解多嵌入模型、索引技术与高级自定义RAG构建方法

5. [LangChain嵌入缓存设计实践](https://github.com/ksmooi/agentic_ai_lab)
   - LangChain实践教程，覆盖嵌入缓存设计和文档索引优化RAG检索

6. [2025 LangChain向量嵌入指南](https://github.com/FareedKhan-dev/contextual-engineering-guide)
   - 上下文工程管道教程，将工具描述转为向量嵌入支持LangChain代理实现

7. [LangChain官方嵌入接口实现](https://github.com/langchain-ai/langchain)
   - 核心框架提供Embeddings标准接口，支持开发者自定义嵌入模型集成

8. [LangChain 2025复杂性社区讨论](https://www.reddit.com/r/LocalLLaMA/comments/1iudao8/langchain_is_still_a_rabbit_hole_in_2025/)
   - Reddit讨论2025 LangChain嵌入与自定义工作流构建的实践经验

## 关键信息提取

### 1. 自定义 Embeddings 实现模式

**继承 Embeddings 基类**：
```python
from langchain_core.embeddings import Embeddings

class CustomEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 实现批量嵌入
        pass

    def embed_query(self, text: str) -> list[float]:
        # 实现单个查询嵌入
        pass
```

### 2. 多提供商切换

**HuggingFace → OpenAI**：
- 使用统一的 Embeddings 接口
- 只需更换实例化的类
- 保持代码其他部分不变

### 3. 嵌入管理器模式

**集中管理多个嵌入模型**：
- 支持动态切换
- 统一的缓存策略
- 性能监控

### 4. 2026 生产级实践

**关键要点**：
- 使用 text-embedding-3-large
- 集成 Chroma 向量存储
- 实现嵌入缓存
- 监控和日志

### 5. 高级自定义技术

**多嵌入模型**：
- 不同字段使用不同模型
- 混合嵌入策略
- 加权融合

**索引技术**：
- 分层索引
- 动态索引更新
- 索引压缩

### 6. 上下文工程

**工具描述向量化**：
- 将工具描述转为嵌入
- 支持 Agent 工具选择
- 提高工具调用准确性

## 待抓取链接（高优先级）

1. https://github.com/vmayakumar/rag-agents - 自定义嵌入对比
2. https://github.com/maryamariyan/langchain-rag-tutorial - 模型切换教程
3. https://github.com/bragai/bRAG-langchain - 高级嵌入教程
4. https://github.com/ksmooi/agentic_ai_lab - 缓存设计实践
5. https://github.com/FareedKhan-dev/contextual-engineering-guide - 上下文工程
6. https://www.reddit.com/r/LocalLLaMA/comments/1iudao8/langchain_is_still_a_rabbit_hole_in_2025/ - 社区讨论

## 排除链接

- https://github.com/langchain-ai/langchain - 官方仓库（已通过源码分析）
- https://github.com/AdilShamim8/GenAI-Roadmap-with-Notes-Using-LangChain - 已在搜索1中出现
