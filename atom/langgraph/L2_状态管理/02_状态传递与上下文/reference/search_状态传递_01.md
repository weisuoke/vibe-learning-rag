---
type: search_result
search_query: langgraph state passing context management 2025 2026
search_engine: grok-mcp
searched_at: 2026-02-26
knowledge_point: 02_状态传递与上下文
platform: GitHub
---

# 搜索结果：LangGraph 状态传递与上下文管理（GitHub）

## 搜索摘要
搜索关键词：langgraph state passing context management 2025 2026
平台：GitHub
结果数量：8 个

## 相关链接

1. [LangGraph官方文档 - Context overview](https://docs.langchain.com/oss/python/concepts/context)
   - 介绍LangGraph中动态运行时上下文
   - 通过state对象管理可变数据
   - 与内存、工具上下文的区别和使用方式

2. [LangGraph Graph API overview](https://docs.langchain.com/oss/python/langgraph/graph-api)
   - 详细说明LangGraph核心组件
   - State作为共享数据结构
   - 消息传递机制和runtime context的定义与访问方法

3. [State Management in LangGraph: The Foundation of Reliable AI Workflows](https://medium.com/algomart/state-management-in-langgraph-the-foundation-of-reliable-ai-workflows-db98dd1499ca)
   - 深入解析LangGraph状态管理原理
   - state作为状态机核心
   - 节点接收并更新state，实现可靠上下文传递与工作流

4. [Context Engineering - LangChain Blog](https://blog.langchain.com/context-engineering-for-agents)
   - 探讨上下文工程技术
   - 利用LangGraph state对象实现写、选、压缩上下文
   - 支持短长期内存和选择性暴露

5. [LangGraph GitHub仓库](https://github.com/langchain-ai/langgraph)
   - LangGraph官方开源仓库
   - 用于构建弹性语言代理作为图
   - 支持状态管理、持久化和长运行代理

6. [FareedKhan-dev/contextual-engineering-guide GitHub](https://github.com/FareedKhan-dev/contextual-engineering-guide)
   - 使用LangChain和LangGraph实现上下文工程管道示例
   - 包括从state选择上下文并传递给下游LLM调用

7. [LangGraph State: The Engine Behind Smarter AI Workflows](https://www.cloudthat.com/resources/blog/langgraph-state-the-engine-behind-smarter-ai-workflows)
   - 解释LangGraph中state作为上下文保持器
   - 实现节点间信息共享、上下文保存和复杂决策

8. [Context Engineering with LangGraph: Why State Management Matters](https://www.linkedin.com/pulse/context-engineering-langgraph-why-state-management-matters-mainkar-2relf)
   - 强调LangGraph状态管理优于传统上下文窗口
   - 提供外部存储、选择性检索和避免上下文污染的最佳实践

## 关键信息提取

### 1. State 作为上下文保持器
- State 是 LangGraph 中的核心概念
- 作为节点间信息共享的载体
- 支持上下文保存和复杂决策

### 2. Runtime Context 的作用
- 动态运行时上下文
- 与 state 的区别：state 可变，context 不可变
- 适合传递配置、依赖等

### 3. 上下文工程技术
- 写（Write）：节点向 state 写入信息
- 选（Select）：选择性暴露上下文
- 压缩（Compress）：压缩上下文以节省 token

### 4. 状态管理优势
- 外部存储：不依赖 LLM 的上下文窗口
- 选择性检索：只检索需要的信息
- 避免上下文污染：隔离不同类型的信息

### 5. 与传统上下文窗口的对比
- 传统：所有信息都在 prompt 中
- LangGraph：信息存储在 state 中，按需检索

## 需要深入抓取的链接

### 高优先级（2025-2026 年资料）
1. https://medium.com/algomart/state-management-in-langgraph-the-foundation-of-reliable-ai-workflows-db98dd1499ca
   - 原因：深入解析状态管理原理
   - 内容类型：技术博客
   - 预期内容：状态管理的实现细节和最佳实践

2. https://blog.langchain.com/context-engineering-for-agents
   - 原因：官方博客，权威性高
   - 内容类型：技术文章
   - 预期内容：上下文工程的实践指南

3. https://www.cloudthat.com/resources/blog/langgraph-state-the-engine-behind-smarter-ai-workflows
   - 原因：详细解释 state 的作用
   - 内容类型：技术博客
   - 预期内容：state 的工作原理和应用场景

4. https://www.linkedin.com/pulse/context-engineering-langgraph-why-state-management-matters-mainkar-2relf
   - 原因：最佳实践和对比分析
   - 内容类型：技术文章
   - 预期内容：状态管理的优势和实践建议

### 中优先级
5. https://github.com/FareedKhan-dev/contextual-engineering-guide
   - 原因：实践示例代码
   - 内容类型：GitHub 仓库
   - 预期内容：上下文工程的代码示例

## 排除的链接
- https://docs.langchain.com/oss/python/concepts/context - 官方文档（已通过 Context7 获取）
- https://docs.langchain.com/oss/python/langgraph/graph-api - 官方文档（已通过 Context7 获取）
- https://github.com/langchain-ai/langgraph - 源码仓库（直接读取本地源码）
