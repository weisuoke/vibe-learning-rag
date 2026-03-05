# 核心概念：LangGraph 内存管理与 LangMem

> **推荐方案**：本文档讲解 LangGraph 的 Checkpointer 机制和 LangMem SDK，适用于复杂 Agent 系统和长期记忆管理。

---

## 一、LangGraph Checkpointer 概述

### 1.1 什么是 Checkpointer

**Checkpointer** 是 LangGraph 的核心状态持久化机制，用于保存和恢复图的完整状态。

**核心设计**：
```
StateGraph → Checkpointer → Persistent Storage
     ↓              ↓
  Nodes        Save/Load State
     ↓              ↓
  Edges      [InMemory, Redis, PostgreSQL, ...]
```

**与 RunnableWithMessageHistory 的区别**：
- **RunnableWithMessageHistory**：只保存消息历史
- **Checkpointer**：保存完整图状态（包括消息、节点状态、边状态等）

### 1.2 为什么需要 Checkpointer

**适用场景**：
1. **复杂 Agent 系统**：多节点工作流，需要保存完整状态
2. **长时间运行任务**：任务可能跨多个会话
3. **故障恢复**：系统崩溃后可以从检查点恢复
4. **时间旅行**：回溯到历史状态进行调试
5. **多用户多会话**：通过 Thread ID 隔离不同对话

**核心优势**：
- 保存完整图状态，不仅是消息历史
- 支持跨会话恢复
- 支持时间旅行（回溯到历史状态）
- 支持多用户多会话管理

[来源: reference/search_memory_03.md | LangGraph Checkpointer 教程]

---

## 二、Checkpointer 核心机制

### 2.1 核心接口

**BaseCheckpointSaver**：
```python
class BaseCheckpointSaver:
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        """保存检查点"""
        ...

    def get(
        self,
        config: RunnableConfig,
    ) -> Optional[Checkpoint]:
        """获取检查点"""
        ...

    def list(
        self,
        config: RunnableConfig,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Checkpoint]:
        """列出检查点"""
        ...
```

**核心概念**：
- **Checkpoint**：图的完整状态快照
- **Thread ID**：唯一标识一个对话线程
- **Checkpoint ID**：唯一标识一个检查点
- **Metadata**：检查点的元数据（时间戳、用户信息等）

[来源: reference/search_memory_03.md | LangGraph Checkpointer 核心概念]

### 2.2 工作流程

**保存检查点**：
```python
# 1. 创建 StateGraph
builder = StateGraph(State)

# 2. 添加节点
builder.add_node("node1", node1_func)
builder.add_node("node2", node2_func)

# 3. 添加边
builder.add_edge(START, "node1")
builder.add_edge("node1", "node2")
builder.add_edge("node2", END)

# 4. 编译图（配置 Checkpointer）
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 5. 调用图（传递 thread_id）
config = {"configurable": {"thread_id": "thread_123"}}
result = graph.invoke({"messages": "Hello"}, config)

# 6. Checkpointer 自动保存状态
# - 每个节点执行后保存检查点
# - 包含完整图状态
```

**恢复检查点**：
```python
# 1. 使用相同的 thread_id 调用
config = {"configurable": {"thread_id": "thread_123"}}
result = graph.invoke({"messages": "Continue"}, config)

# 2. Checkpointer 自动加载历史状态
# - 从上次检查点恢复
# - 继续执行
```

[来源: reference/search_memory_03.md | LangGraph Checkpointer 工作流程]

### 2.3 Thread ID 管理

**Thread ID 设计**：
```python
# 方案1：用户ID + 会话ID
thread_id = f"user_{user_id}_session_{session_id}"

# 方案2：UUID
import uuid
thread_id = str(uuid.uuid4())

# 方案3：时间戳 + 用户ID
import time
thread_id = f"{user_id}_{int(time.time())}"
```

**多用户隔离**：
```python
# 用户1的对话
config_user1 = {"configurable": {"thread_id": "user_alice_thread_1"}}
graph.invoke({"messages": "Hi, I'm Alice"}, config_user1)

# 用户2的对话
config_user2 = {"configurable": {"thread_id": "user_bob_thread_1"}}
graph.invoke({"messages": "Hi, I'm Bob"}, config_user2)

# 用户1的第二轮对话
graph.invoke({"messages": "What's my name?"}, config_user1)
# 输出: "Your name is Alice."

# 用户2的第二轮对话
graph.invoke({"messages": "What's my name?"}, config_user2)
# 输出: "Your name is Bob."
```

[来源: reference/search_memory_03.md | Thread ID 管理]

---

## 三、Checkpointer 实现

### 3.1 InMemorySaver

**特点**：
- 内存存储
- 简单快速
- 不持久化
- 适合开发和测试

**基础用法**：
```python
from langgraph.checkpoint.memory import MemorySaver

# 创建 Checkpointer
checkpointer = MemorySaver()

# 编译图
graph = builder.compile(checkpointer=checkpointer)

# 使用
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "Hello"}, config)
```

[来源: reference/search_memory_03.md | InMemorySaver]

### 3.2 RedisSaver

**特点**：
- 高性能
- 支持 TTL
- 分布式
- 适合生产环境

**基础用法**：
```python
from langgraph.checkpoint.redis import RedisSaver
from redis import Redis

# 创建 Redis 客户端
redis_client = Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True,
)

# 创建 Checkpointer
checkpointer = RedisSaver(redis_client)

# 编译图
graph = builder.compile(checkpointer=checkpointer)

# 使用
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "Hello"}, config)
```

**高级配置**：
```python
from redis import ConnectionPool

# 创建连接池
pool = ConnectionPool(
    host="localhost",
    port=6379,
    db=0,
    max_connections=50,
    decode_responses=True,
)

redis_client = Redis(connection_pool=pool)
checkpointer = RedisSaver(redis_client)
```

[来源: reference/search_memory_03.md | RedisSaver]

### 3.3 PostgresSaver

**特点**：
- 持久化
- 支持复杂查询
- 事务支持
- 适合长期存储

**基础用法**：
```python
from langgraph.checkpoint.postgres import PostgresSaver

# 创建 Checkpointer
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost/db"
)

# 编译图
graph = builder.compile(checkpointer=checkpointer)

# 使用
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "Hello"}, config)
```

**异步集成**：
```python
import asyncpg
from langgraph.checkpoint.postgres import AsyncPostgresSaver

# 创建异步连接池
pool = await asyncpg.create_pool(
    "postgresql://user:pass@localhost/db",
    min_size=10,
    max_size=50,
)

# 创建 Checkpointer
checkpointer = AsyncPostgresSaver(pool)

# 编译图
graph = builder.compile(checkpointer=checkpointer)

# 异步调用
config = {"configurable": {"thread_id": "1"}}
result = await graph.ainvoke({"messages": "Hello"}, config)
```

[来源: reference/search_memory_03.md | PostgresSaver]

### 3.4 SqliteSaver

**特点**：
- 轻量级
- 文件存储
- 零配置
- 适合小型应用

**基础用法**：
```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 创建 Checkpointer
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# 编译图
graph = builder.compile(checkpointer=checkpointer)

# 使用
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "Hello"}, config)
```

[来源: reference/context7_langchain_01.md | LangChain 官方文档]

---

## 四、SummarizationNode

### 4.1 什么是 SummarizationNode

**SummarizationNode** 是 LangGraph 提供的自动总结节点，用于压缩对话历史。

**核心设计**：
```
Messages → SummarizationNode → Summary + Recent Messages
    ↓              ↓
 [Msg1, Msg2,   LLM Summarize
  Msg3, ...]        ↓
                [Summary, Msg3, ...]
```

**适用场景**：
- 长对话历史
- 上下文窗口限制
- 需要保留语义信息

[来源: reference/search_memory_02.md | SummarizationNode]

### 4.2 基础用法

**创建 SummarizationNode**：
```python
from langgraph.prebuilt import SummarizationNode
from langchain_openai import ChatOpenAI

# 创建总结节点
summarization_node = SummarizationNode(
    model=ChatOpenAI(model="gpt-4"),
    max_tokens_before_summary=2000,  # 超过2000 tokens时触发总结
    max_summary_tokens=500,           # 总结最多500 tokens
)

# 添加到图中
builder = StateGraph(State)
builder.add_node("summarize", summarization_node)
builder.add_edge(START, "summarize")
builder.add_edge("summarize", "chat")
builder.add_edge("chat", END)

# 编译图
graph = builder.compile(checkpointer=checkpointer)
```

**工作流程**：
1. 检查消息历史的 Token 数量
2. 如果超过 `max_tokens_before_summary`，触发总结
3. 使用 LLM 总结旧消息
4. 保留总结 + 最近的消息
5. 继续执行

[来源: reference/search_memory_02.md | SummarizationNode 用法]

### 4.3 自定义总结策略

**自定义总结 Prompt**：
```python
from langchain_core.prompts import ChatPromptTemplate

# 自定义总结 Prompt
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the following conversation in 3-5 sentences:"),
    ("placeholder", "{messages}"),
])

summarization_node = SummarizationNode(
    model=ChatOpenAI(model="gpt-4"),
    max_tokens_before_summary=2000,
    max_summary_tokens=500,
    summary_prompt=summary_prompt,
)
```

**自定义总结逻辑**：
```python
def custom_summarize(messages):
    # 自定义总结逻辑
    # 例如：只总结用户消息
    user_messages = [m for m in messages if isinstance(m, HumanMessage)]
    summary = summarize_llm.invoke(user_messages)
    return summary

summarization_node = SummarizationNode(
    model=ChatOpenAI(model="gpt-4"),
    max_tokens_before_summary=2000,
    max_summary_tokens=500,
    summarize_func=custom_summarize,
)
```

[来源: reference/search_memory_02.md | 自定义总结策略]

---

## 五、LangMem SDK

### 5.1 什么是 LangMem

**LangMem SDK** 是 LangChain 于 2025年2月发布的长期记忆管理库。

**核心特性**：
1. **短期记忆**：使用 Checkpointer 管理对话历史
2. **长期记忆**：使用 LangMem Store 管理跨会话记忆
3. **语义知识提取**：自动从对话中提取关键信息
4. **跨会话记忆维护**：在不同会话间共享记忆
5. **自动优化代理提示**：根据记忆优化 Prompt

**核心设计**：
```
StateGraph → Checkpointer (短期) + LangMem Store (长期)
     ↓              ↓                    ↓
  Nodes      [InMemorySaver, ...]  [语义知识提取, 跨会话记忆]
```

[来源: reference/search_memory_01.md | LangMem SDK]

### 5.2 安装和配置

**安装**：
```bash
pip install langmem
```

**基础配置**：
```python
from langmem import MemoryStore
from langgraph.checkpoint.memory import MemorySaver

# 创建短期记忆（Checkpointer）
checkpointer = MemorySaver()

# 创建长期记忆（LangMem Store）
store = MemoryStore()

# 编译图
graph = builder.compile(
    checkpointer=checkpointer,
    store=store,
)
```

[来源: reference/search_memory_01.md | LangMem 安装]

### 5.3 语义知识提取

**自动提取关键信息**：
```python
from langmem import MemoryStore
from langchain_openai import ChatOpenAI

# 创建 Store（配置提取模型）
store = MemoryStore(
    extraction_model=ChatOpenAI(model="gpt-4"),
    extraction_prompt="Extract key facts from the conversation:",
)

# 编译图
graph = builder.compile(
    checkpointer=checkpointer,
    store=store,
)

# 使用
config = {"configurable": {"thread_id": "1", "user_id": "alice"}}
graph.invoke({"messages": "I live in San Francisco"}, config)

# LangMem 自动提取：
# - 用户: alice
# - 居住地: San Francisco
```

**提取的信息类型**：
- 用户偏好
- 个人信息
- 历史事件
- 关键决策
- 重要关系

[来源: reference/search_memory_01.md | 语义知识提取]

### 5.4 跨会话记忆维护

**跨会话共享记忆**：
```python
# 会话1：用户提供信息
config1 = {"configurable": {"thread_id": "thread_1", "user_id": "alice"}}
graph.invoke({"messages": "I love pizza"}, config1)

# 会话2：使用相同的 user_id
config2 = {"configurable": {"thread_id": "thread_2", "user_id": "alice"}}
graph.invoke({"messages": "What food do I like?"}, config2)
# 输出: "You love pizza."

# LangMem 自动：
# - 从 Store 中检索 user_id=alice 的记忆
# - 注入到当前会话的上下文中
```

**记忆检索策略**：
```python
from langmem import MemoryStore

store = MemoryStore(
    retrieval_strategy="semantic",  # 语义检索
    top_k=5,                         # 检索前5条记忆
    similarity_threshold=0.7,        # 相似度阈值
)
```

[来源: reference/search_memory_01.md | 跨会话记忆]

### 5.5 自动优化代理提示

**根据记忆优化 Prompt**：
```python
from langmem import MemoryStore

store = MemoryStore(
    auto_optimize_prompt=True,  # 自动优化 Prompt
)

# 编译图
graph = builder.compile(
    checkpointer=checkpointer,
    store=store,
)

# 使用
config = {"configurable": {"thread_id": "1", "user_id": "alice"}}
graph.invoke({"messages": "Help me plan a trip"}, config)

# LangMem 自动：
# - 从 Store 中检索 user_id=alice 的记忆
# - 根据记忆优化 Prompt
# - 例如：添加"用户喜欢披萨，居住在旧金山"到 Prompt
```

[来源: reference/search_memory_01.md | 自动优化 Prompt]

---

## 六、生产环境最佳实践

### 6.1 Checkpointer 选择

**决策树**：
```
需要持久化？
├─ 否 → InMemorySaver（开发/测试）
└─ 是
   ├─ 高性能 → RedisSaver
   ├─ 长期存储 → PostgresSaver
   └─ 轻量级 → SqliteSaver
```

### 6.2 Thread ID 管理

**推荐格式**：
```python
# 格式：{user_id}_{session_id}_{timestamp}
thread_id = f"{user_id}_{session_id}_{int(time.time())}"
```

**清理策略**：
```python
# 定期清理旧 Thread
def cleanup_old_threads(checkpointer, days=30):
    cutoff = time.time() - (days * 24 * 60 * 60)
    # 删除超过30天的 Thread
    checkpointer.delete_threads_before(cutoff)
```

### 6.3 内存限制

**使用 SummarizationNode**：
```python
summarization_node = SummarizationNode(
    model=ChatOpenAI(model="gpt-4"),
    max_tokens_before_summary=2000,
    max_summary_tokens=500,
)
```

**手动限制**：
```python
def limit_messages(state):
    messages = state["messages"]
    if len(messages) > 20:
        # 只保留最近20条消息
        state["messages"] = messages[-20:]
    return state

builder.add_node("limit", limit_messages)
```

### 6.4 错误处理

**Checkpointer 错误处理**：
```python
try:
    result = graph.invoke({"messages": "Hello"}, config)
except Exception as e:
    logger.error(f"Checkpointer error: {e}")
    # 降级到内存存储
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    result = graph.invoke({"messages": "Hello"}, config)
```

### 6.5 监控和日志

**关键指标**：
- Thread 数量
- Checkpoint 数量
- 存储大小
- 响应时间
- 错误率

**日志记录**：
```python
import logging

logger = logging.getLogger(__name__)

def log_checkpoint(config, checkpoint):
    logger.info(f"Saved checkpoint for thread {config['configurable']['thread_id']}")
    logger.info(f"Checkpoint size: {len(str(checkpoint))} bytes")
```

---

## 七、对比总结

### 7.1 三种方案对比

| 维度 | RunnableWithMessageHistory | LangGraph Checkpointer | LangMem SDK |
|------|---------------------------|----------------------|-------------|
| **状态管理** | 仅消息历史 | 完整图状态 | 短期+长期记忆 |
| **适用场景** | 简单对话系统 | 复杂 Agent 系统 | 长期记忆管理 |
| **持久化** | 支持 | 支持 | 支持 |
| **跨会话** | 通过 Session ID | 通过 Thread ID | 通过 User ID |
| **语义提取** | 不支持 | 不支持 | 支持 |
| **时间旅行** | 不支持 | 支持 | 不支持 |
| **复杂度** | 低 | 中 | 高 |

### 7.2 选择建议

**简单对话系统**：
- 使用 RunnableWithMessageHistory
- 适合：聊天机器人、简单问答

**复杂 Agent 系统**：
- 使用 LangGraph Checkpointer
- 适合：多节点工作流、需要状态管理

**长期记忆管理**：
- 使用 LangMem SDK
- 适合：个人助理、跨会话记忆

---

## 数据来源

- [来源: reference/context7_langchain_01.md | LangChain 官方文档]
- [来源: reference/search_memory_01.md | LangMem SDK 2025]
- [来源: reference/search_memory_02.md | SummarizationNode]
- [来源: reference/search_memory_03.md | LangGraph Checkpointer 教程]

---

**版本**：v1.0
**最后更新**：2026-02-24
**维护者**：Claude Code
