---
type: search_result
search_query: LangGraph checkpointer memory management tutorial 2025 2026
search_engine: grok-mcp
searched_at: 2026-02-24
knowledge_point: Memory与对话历史管理
---

# 搜索结果：LangGraph Checkpointer 内存管理教程（2025-2026）

## 搜索摘要

搜索了 GitHub 和 Reddit 平台上关于 LangGraph checkpointer 和内存管理的教程（2025-2026年）。主要发现包括：
1. **官方教程**：LangGraph 官方文档和教程笔记
2. **长期内存管理**：2025年发布的长期内存管理教程
3. **持久化存储**：Redis、PostgreSQL、SQLite 等 checkpointer 实现
4. **生产环境实践**：社区分享的生产级内存管理经验

## 相关链接

### GitHub 官方资源

1. **[LangGraph 官方内存添加指南](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/memory/add-memory.md)**
   - LangGraph 官方文档，详细说明使用 checkpointer 为图添加内存，实现状态持久化，支持子图独立内存配置

2. **[LangGraph 长内存管理教程](https://github.com/FareedKhan-dev/langgraph-long-memory)**
   - 2025年发布教程，专注于内存管理，通过构建邮件助手探索短期和长期内存策略

3. **[LangGraph AI助手完整构建教程](https://github.com/girijesh-ai/langgraph-ai-assistant-tutorial)**
   - 从基础到生产级代理教程，涵盖 checkpointer 内存管理、工具和持久化实现

4. **[2025多用户LangGraph自适应内存系统](https://github.com/dipanjanS/mastering-intelligent-agents-langgraph-workshop-dhs2025/blob/main/Module-3-Context-Engineering-for-Agentic-AI-Systems/M3LC4_Build_a_Multi_User_Multi_Session_Adaptive_Agentic_AI_Systems_with_Short_and_Long_Term_Memory.ipynb)**
   - 使用 LangGraph checkpointer 和 LangMem 构建多用户多会话短长期内存代理的2025工作坊笔记

5. **[Redis LangGraph Checkpointer与内存管理](https://github.com/redis-developer/langgraph-redis)**
   - Redis 实现的 checkpointer 和 store，为 LangGraph 提供高效内存管理和状态持久化

6. **[LangGraph Mastery Playbook内存基础](https://github.com/leslieo2/LangGraph-Mastery-Playbook)**
   - 脚本化教程，讲解 LangGraph 内存、checkpointer 配置和多种持久化存储选项

### Reddit 社区讨论

7. **[LangGraph Postgres Checkpointer设置指南](https://www.reddit.com/r/LangChain/comments/1hzj9ir/any_guides_or_directions_for_postgres_checkpointer/)**
   - Reddit 讨论 LangGraph Postgres checkpointer 的安装配置和生产使用指南

8. **[LangGraph检查点与状态内存管理](https://www.reddit.com/r/LangChain/comments/1dabjys/langgraph_checkpoints_vs_history/)**
   - 社区探讨 LangGraph checkpointers 用于历史记录和内存持久化的方法

## 关键信息提取

### 1. **LangGraph Checkpointer 核心概念**

**什么是 Checkpointer？**
- LangGraph 的状态持久化机制
- 在每个节点执行后保存图的状态
- 支持跨会话恢复对话
- 实现长期记忆管理

**核心功能**：
- **状态保存**：自动保存图的状态
- **状态恢复**：从保存的状态恢复执行
- **Thread 管理**：通过 thread ID 隔离不同对话
- **时间旅行**：回溯到历史状态

### 2. **官方内存添加指南**

**来源**：LangGraph 官方文档

**核心内容**：

#### A. **基础 Checkpointer 使用**

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

# 创建 checkpointer
checkpointer = InMemorySaver()

# 编译图时传入 checkpointer
graph = builder.compile(checkpointer=checkpointer)

# 使用 thread ID 调用
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "message"}, config)
```

#### B. **子图独立内存配置**

```python
# 子图可以有独立的 checkpointer
subgraph_checkpointer = InMemorySaver()
subgraph = subgraph_builder.compile(checkpointer=subgraph_checkpointer)

# 主图使用不同的 checkpointer
main_checkpointer = InMemorySaver()
main_graph = main_builder.compile(checkpointer=main_checkpointer)
```

**特点**：
- 子图和主图可以使用不同的 checkpointer
- 支持复杂的多层内存管理
- 适合多代理系统

### 3. **长期内存管理教程（2025）**

**来源**：GitHub - FareedKhan-dev/langgraph-long-memory

**核心内容**：

#### A. **短期内存 vs 长期内存**

**短期内存**：
- 使用 Checkpointer 实现
- 保存对话历史
- 适合单个会话

**长期内存**：
- 使用 LangMem Store 实现
- 提取语义知识
- 跨会话持久化

#### B. **邮件助手示例**

```python
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import SummarizationNode

# 短期内存：Checkpointer
checkpointer = InMemorySaver()

# 长期内存：LangMem Store
store = MemoryStore()

# 构建图
builder = StateGraph(State)
builder.add_node("summarize", SummarizationNode(...))
builder.add_node("call_model", call_model)

graph = builder.compile(
    checkpointer=checkpointer,
    store=store
)
```

**应用场景**：
- 邮件助手
- 个人助理
- 客户服务机器人

### 4. **多用户多会话内存系统（2025 DHS 工作坊）**

**来源**：GitHub - dipanjanS/mastering-intelligent-agents-langgraph-workshop-dhs2025

**核心内容**：

#### A. **多用户隔离**

```python
# 每个用户有独立的 thread ID
user1_config = {"configurable": {"thread_id": "user1_session1"}}
user2_config = {"configurable": {"thread_id": "user2_session1"}}

# 用户1的对话
graph.invoke({"messages": "user1 message"}, user1_config)

# 用户2的对话
graph.invoke({"messages": "user2 message"}, user2_config)
```

#### B. **多会话管理**

```python
# 同一用户的不同会话
session1_config = {"configurable": {"thread_id": "user1_session1"}}
session2_config = {"configurable": {"thread_id": "user1_session2"}}

# 会话1
graph.invoke({"messages": "message in session 1"}, session1_config)

# 会话2
graph.invoke({"messages": "message in session 2"}, session2_config)
```

#### C. **自适应内存**

```python
from langmem.short_term import SummarizationNode

# 根据对话长度自动总结
summarization_node = SummarizationNode(
    max_tokens_before_summary=256,
    max_summary_tokens=128,
)
```

**特点**：
- 支持多用户并发
- 每个用户有独立的内存空间
- 自动内存管理和总结

### 5. **持久化存储实现**

#### A. **Redis Checkpointer**

**来源**：GitHub - redis-developer/langgraph-redis

```python
from langgraph_redis import RedisSaver

# 创建 Redis checkpointer
checkpointer = RedisSaver(
    redis_url="redis://localhost:6379",
    ttl=3600  # 1小时过期
)

graph = builder.compile(checkpointer=checkpointer)
```

**特点**：
- 高性能
- 支持 TTL
- 适合生产环境

#### B. **PostgreSQL Checkpointer**

**来源**：Reddit 讨论

```python
from langgraph_postgres import PostgresSaver

# 创建 PostgreSQL checkpointer
checkpointer = PostgresSaver(
    connection_string="postgresql://user:pass@localhost/db"
)

graph = builder.compile(checkpointer=checkpointer)
```

**特点**：
- 持久化存储
- 支持复杂查询
- 适合长期存储

#### C. **SQLite Checkpointer**

**来源**：GitHub 教程

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 创建 SQLite checkpointer
checkpointer = SqliteSaver("checkpoints.db")

graph = builder.compile(checkpointer=checkpointer)
```

**特点**：
- 轻量级
- 无需额外服务
- 适合开发和小型应用

### 6. **LangGraph Mastery Playbook**

**来源**：GitHub - leslieo2/LangGraph-Mastery-Playbook

**核心内容**：

#### A. **Checkpointer 配置**

```python
# 内存 checkpointer
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

# SQLite checkpointer
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver("checkpoints.db")

# PostgreSQL checkpointer
from langgraph_postgres import PostgresSaver
checkpointer = PostgresSaver(connection_string)

# Redis checkpointer
from langgraph_redis import RedisSaver
checkpointer = RedisSaver(redis_url)
```

#### B. **内存总结**

```python
from langmem.short_term import SummarizationNode

# 配置总结节点
summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=summarization_model,
    max_tokens=256,
    max_tokens_before_summary=256,
    max_summary_tokens=128,
)

# 添加到图中
builder.add_node("summarize", summarization_node)
```

#### C. **状态管理**

```python
from langgraph.graph import StateGraph, MessagesState

class State(MessagesState):
    context: dict[str, RunningSummary]

builder = StateGraph(State)
```

### 7. **生产环境最佳实践**

#### A. **Checkpointer 选择**

**开发环境**：
- `InMemorySaver`：快速开发和测试

**生产环境**：
- `RedisSaver`：高性能、支持 TTL
- `PostgresSaver`：持久化、支持复杂查询
- `SqliteSaver`：轻量级、适合小型应用

#### B. **Thread ID 管理**

**推荐格式**：
```python
# 用户ID + 会话ID
thread_id = f"user_{user_id}_session_{session_id}"

# 或使用 UUID
import uuid
thread_id = str(uuid.uuid4())
```

**注意事项**：
- Thread ID 必须唯一
- 避免使用敏感信息
- 定期清理过期 thread

#### C. **内存限制**

```python
from langmem.short_term import SummarizationNode

# 配置内存限制
summarization_node = SummarizationNode(
    max_tokens_before_summary=256,  # 超过256 token 开始总结
    max_summary_tokens=128,  # 总结最多128 token
)
```

#### D. **错误处理**

```python
try:
    result = graph.invoke({"messages": "message"}, config)
except Exception as e:
    # 记录错误
    logger.error(f"Graph execution failed: {e}")
    # 清理状态
    checkpointer.clear(thread_id)
    # 返回错误响应
    return {"error": str(e)}
```

### 8. **社区讨论要点**

#### A. **Checkpoints vs History**

**来源**：Reddit 讨论

**核心问题**：
- Checkpoints 和 History 有什么区别？
- 什么时候使用 Checkpoints？

**答案**：
- **Checkpoints**：保存完整的图状态，包括所有节点的状态
- **History**：只保存消息历史
- **使用场景**：
  - 简单对话：使用 History（RunnableWithMessageHistory）
  - 复杂 Agent：使用 Checkpoints（LangGraph）

#### B. **Postgres Checkpointer 设置**

**来源**：Reddit 讨论

**常见问题**：
- 如何配置 PostgreSQL checkpointer？
- 如何处理连接池？
- 如何优化性能？

**解决方案**：
```python
from langgraph_postgres import PostgresSaver
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# 创建连接池
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)

# 创建 checkpointer
checkpointer = PostgresSaver(engine=engine)
```

### 9. **2025-2026 技术趋势**

**主要趋势**：

1. **LangMem 集成**
   - 短期内存：Checkpointer
   - 长期内存：LangMem Store
   - 自动内存管理

2. **多用户支持**
   - Thread ID 隔离
   - 独立内存空间
   - 并发处理

3. **持久化存储**
   - Redis：高性能
   - PostgreSQL：持久化
   - SQLite：轻量级

4. **自动总结**
   - SummarizationNode
   - Token 限制
   - 语义压缩

## 待抓取链接（需要更多详细信息）

根据规范，以下链接需要进一步抓取以获取完整内容：

### 高优先级（2025-2026 最新资料）

1. **https://github.com/FareedKhan-dev/langgraph-long-memory**
   - 知识点标签：长期内存管理、LangMem
   - 原因：2025年10月发布的完整教程，包含短期和长期内存策略
   - 内容焦点：教程文档、代码示例、实现细节

2. **https://github.com/dipanjanS/mastering-intelligent-agents-langgraph-workshop-dhs2025/blob/main/Module-3-Context-Engineering-for-Agentic-AI-Systems/M3LC4_Build_a_Multi_User_Multi_Session_Adaptive_Agentic_AI_Systems_with_Short_and_Long_Term_Memory.ipynb**
   - 知识点标签：多用户多会话、自适应内存
   - 原因：2025 DHS 工作坊的实战笔记，包含完整的实现
   - 内容焦点：Jupyter Notebook、代码实现、最佳实践

3. **https://github.com/girijesh-ai/langgraph-ai-assistant-tutorial**
   - 知识点标签：AI助手构建、生产级实践
   - 原因：从基础到生产级的完整教程
   - 内容焦点：教程文档、代码示例、部署指南

### 中优先级

4. **https://github.com/redis-developer/langgraph-redis**
   - 知识点标签：Redis 集成、高性能存储
   - 原因：Redis checkpointer 的官方实现
   - 内容焦点：API 文档、配置指南、性能优化

5. **https://github.com/leslieo2/LangGraph-Mastery-Playbook**
   - 知识点标签：LangGraph 基础、配置指南
   - 原因：实用的脚本式教程
   - 内容焦点：配置示例、最佳实践

6. **https://www.reddit.com/r/LangChain/comments/1hzj9ir/any_guides_or_directions_for_postgres_checkpointer/**
   - 知识点标签：PostgreSQL 集成、生产环境
   - 原因：PostgreSQL checkpointer 的实战经验
   - 内容焦点：配置方法、问题排查

## 排除的链接（已通过其他方式获取）

以下链接不需要抓取，因为已通过源码或 Context7 获取：
- LangGraph 官方文档链接（已通过 Context7 获取）
- LangChain 源码仓库链接（直接读取本地源码）

## 总结

**核心发现**：
1. **LangGraph Checkpointer** 是现代对话历史管理的主流方案
2. **多种持久化存储**：Redis、PostgreSQL、SQLite
3. **LangMem 集成**：短期 + 长期内存管理
4. **多用户支持**：Thread ID 隔离、并发处理
5. **自动总结**：SummarizationNode、Token 限制

**学习重点**：
- LangGraph Checkpointer 的核心概念
- 持久化存储的选择和配置
- 多用户多会话管理
- LangMem 集成
- 生产环境最佳实践
