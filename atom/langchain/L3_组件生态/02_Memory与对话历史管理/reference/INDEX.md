# 资料索引

生成时间：2026-02-24

## 概览
- 总文件数：5
- 源码分析：1 个
- Context7 文档：1 个
- 搜索结果：3 个
- 抓取内容：0 个（待抓取）

## 按知识点分类

### 传统 Memory 系统（已弃用）
#### 源码分析
- [source_memory_01.md](source_memory_01.md) - LangChain Memory 核心架构

#### Context7 文档
- [context7_langchain_01.md](context7_langchain_01.md) - Memory 迁移指南

### 现代对话历史管理
#### Context7 文档
- [context7_langchain_01.md](context7_langchain_01.md) - RunnableWithMessageHistory、Session 管理

#### 搜索结果
- [search_memory_01.md](search_memory_01.md) - LangChain Memory 2025-2026 趋势
- [search_memory_02.md](search_memory_02.md) - RunnableWithMessageHistory 最佳实践

### LangGraph 内存管理
#### Context7 文档
- [context7_langchain_01.md](context7_langchain_01.md) - Checkpointer、SummarizationNode

#### 搜索结果
- [search_memory_01.md](search_memory_01.md) - LangGraph 内存管理
- [search_memory_03.md](search_memory_03.md) - LangGraph Checkpointer 教程

### LangMem SDK（2025新特性）
#### 搜索结果
- [search_memory_01.md](search_memory_01.md) - LangMem SDK 发布
- [search_memory_03.md](search_memory_03.md) - 长期内存管理

### 持久化存储
#### Context7 文档
- [context7_langchain_01.md](context7_langchain_01.md) - CockroachDB、PostgreSQL、Redis

#### 搜索结果
- [search_memory_02.md](search_memory_02.md) - PostgreSQL 异步集成
- [search_memory_03.md](search_memory_03.md) - Redis/PostgreSQL/SQLite Checkpointer

### 生产环境最佳实践
#### 搜索结果
- [search_memory_01.md](search_memory_01.md) - 生产级最佳实践
- [search_memory_02.md](search_memory_02.md) - 生产环境架构
- [search_memory_03.md](search_memory_03.md) - 生产环境最佳实践

## 按文件类型分类

### 源码分析（1 个）
1. [source_memory_01.md](source_memory_01.md) - LangChain Memory 核心架构
   - BaseChatMemory 基类
   - 12种 Memory 类型
   - 对话历史存储抽象
   - 上下文窗口管理策略
   - **重要发现**：所有 Memory 类已被弃用

### Context7 文档（1 个）
1. [context7_langchain_01.md](context7_langchain_01.md) - LangChain Memory 与对话历史管理
   - Memory 迁移指南
   - InMemoryChatMessageHistory
   - RunnableWithMessageHistory
   - LangGraph Checkpointer
   - 持久化存储
   - 最佳实践

### 搜索结果（3 个）
1. [search_memory_01.md](search_memory_01.md) - LangChain Memory 2025-2026
   - LangMem SDK
   - 长对话历史管理
   - LangGraph 内存管理
   - 生产级最佳实践

2. [search_memory_02.md](search_memory_02.md) - RunnableWithMessageHistory 最佳实践
   - 官方源码
   - 生产环境架构
   - 内存限制策略
   - 异步集成
   - LangGraph 集成

3. [search_memory_03.md](search_memory_03.md) - LangGraph Checkpointer 教程
   - Checkpointer 核心概念
   - 长期内存管理
   - 多用户多会话
   - 持久化存储
   - 生产环境实践

## 质量评估
- 高质量资料：5 个（所有资料）
- 中等质量资料：0 个
- 低质量资料：0 个

## 覆盖度分析
- 传统 Memory 系统：✓ 完全覆盖（源码 + Context7）
- 现代对话历史管理：✓ 完全覆盖（Context7 + 网络）
- LangGraph 内存管理：✓ 完全覆盖（Context7 + 网络）
- LangMem SDK：✓ 完全覆盖（网络）
- 持久化存储：✓ 完全覆盖（Context7 + 网络）
- 生产环境实践：✓ 完全覆盖（网络）

## 核心发现

### 1. 重要警告：Memory 类已弃用
- 所有传统 Memory 类（ConversationBufferMemory 等）已被标记为 deprecated
- 自 0.3.1 版本起弃用，将在 1.0.0 版本移除
- 不支持原生工具调用，会静默失败
- 官方迁移指南：https://python.langchain.com/docs/versions/migrating_memory/

### 2. 现代方法：RunnableWithMessageHistory
- 使用 InMemoryChatMessageHistory 和 RunnableWithMessageHistory
- Session 管理模式成为核心
- 支持多种持久化存储（Redis、PostgreSQL、SQLite）
- 生产环境需要内存限制和异步集成

### 3. LangGraph 成为主流
- Checkpointer 机制实现状态持久化
- SummarizationNode 自动总结对话历史
- 支持多用户多会话管理
- Thread ID 隔离不同对话

### 4. LangMem SDK（2025新特性）
- 2025年2月发布
- 提供长期记忆管理
- 语义知识提取
- 跨会话记忆维护

### 5. 生产环境关键点
- 限制交互次数而非全数据库存储
- 使用持久化存储（Redis、PostgreSQL）
- 实现内存限制和总结机制
- 异步操作和性能优化

## 待抓取链接统计

根据搜索结果中标注的"待抓取链接"，共有以下链接需要进一步抓取：

### 高优先级（2025-2026 最新资料）
1. https://github.com/langchain-ai/langmem
2. https://github.com/orgs/community/discussions/172736
3. https://www.reddit.com/r/LangChain/comments/1khxspe/managing_conversation_history_with_langgraph/
4. https://www.reddit.com/r/LangChain/comments/1o0l2wj/best_practices_for_building_productionlevel/
5. https://github.com/langchain-ai/langchain/issues/23716
6. https://www.reddit.com/r/LangChain/comments/1ergfbf/chat_memory_history_in_production_architectures/
7. https://github.com/langchain-ai/langchain-postgres/issues/122
8. https://github.com/FareedKhan-dev/langgraph-long-memory
9. https://github.com/dipanjanS/mastering-intelligent-agents-langgraph-workshop-dhs2025/...
10. https://github.com/girijesh-ai/langgraph-ai-assistant-tutorial

### 中优先级
11. https://www.reddit.com/r/LangChain/comments/1jnxnvt/how_do_you_manage_conversation_history_with_files/
12. https://www.reddit.com/r/LangChain/comments/1d4n1ci/limiting_memory_in_runnablewithmessagehistory/
13. https://www.reddit.com/r/LangChain/comments/1dposfd/integration_issues_with_langgraph/
14. https://github.com/redis-developer/langgraph-redis
15. https://github.com/leslieo2/LangGraph-Mastery-Playbook
16. https://www.reddit.com/r/LangChain/comments/1hzj9ir/any_guides_or_directions_for_postgres_checkpointer/

**注意**：这些链接已在搜索结果文件中标注，可以在阶段二进行抓取。
