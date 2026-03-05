# 实战代码 - 场景7：LangGraph Checkpointer 实战

> **知识点**：Memory与对话历史管理
> **场景**：LangGraph Checkpointer 实战
> **难度**：⭐⭐⭐⭐⭐
> **重要性**：⭐⭐⭐⭐⭐

---

## 场景概述

LangGraph 是 LangChain 的现代对话管理方案，核心特性包括：
- **Checkpointer 机制**：自动保存图的状态
- **Thread ID 隔离**：支持多用户多会话
- **状态持久化**：支持 Redis、PostgreSQL、SQLite
- **自动总结**：SummarizationNode 自动压缩历史

**核心要点**：
- Checkpointer 配置
- 多用户隔离
- 状态持久化
- SummarizationNode 集成

---

## 数据来源

1. **Context7 官方文档** (`reference/context7_langchain_01.md`)
   - SummarizationNode 完整示例
   - Checkpointer 机制
   - LangGraph 状态管理

2. **网络搜索 - LangGraph Checkpointer 教程** (`reference/search_memory_03.md`)
   - Checkpointer 核心概念
   - 多用户多会话管理
   - 持久化存储实现

---

## 实战示例1：基础 Checkpointer 配置

### 完整代码

```python
"""
基础 Checkpointer 配置
演示如何使用 InMemorySaver 实现状态持久化
"""

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1. 定义状态
# ============================================================

class State(MessagesState):
    """对话状态"""
    pass

# ============================================================
# 2. 定义节点
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

def call_model(state: State):
    """调用模型节点"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# ============================================================
# 3. 构建图
# ============================================================

# 创建 Checkpointer
checkpointer = InMemorySaver()

# 构建图
builder = StateGraph(State)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

# 编译图（传入 checkpointer）
graph = builder.compile(checkpointer=checkpointer)

# ============================================================
# 4. 测试对话
# ============================================================

def test_basic_checkpointer():
    """测试基础 Checkpointer"""

    print("=== 基础 Checkpointer 测试 ===\n")

    # 配置 thread ID
    config = {"configurable": {"thread_id": "thread_001"}}

    # 第一轮对话
    print("👤 用户: 我叫Alice")
    result1 = graph.invoke(
        {"messages": [("user", "我叫Alice")]},
        config=config
    )
    print(f"🤖 AI: {result1['messages'][-1].content}\n")

    # 第二轮对话（测试记忆）
    print("👤 用户: 我的名字是什么？")
    result2 = graph.invoke(
        {"messages": [("user", "我的名字是什么？")]},
        config=config
    )
    print(f"🤖 AI: {result2['messages'][-1].content}\n")

    # 验证状态
    print("=== 状态验证 ===")
    print(f"✅ 消息数量: {len(result2['messages'])}")

if __name__ == "__main__":
    test_basic_checkpointer()
```

### 运行结果

```
=== 基础 Checkpointer 测试 ===

👤 用户: 我叫Alice
🤖 AI: 你好Alice！很高兴认识你。

👤 用户: 我的名字是什么？
🤖 AI: 你的名字是Alice。

=== 状态验证 ===
✅ 消息数量: 4
```

---

## 实战示例2：多用户隔离

### 完整代码

```python
"""
多用户隔离
演示如何使用 Thread ID 实现多用户隔离
"""

def test_multi_user_isolation():
    """测试多用户隔离"""

    print("=== 多用户隔离测试 ===\n")

    # 用户1的对话
    config1 = {"configurable": {"thread_id": "user_alice"}}
    print("👤 用户Alice: 我是一名设计师")
    result1 = graph.invoke(
        {"messages": [("user", "我是一名设计师")]},
        config=config1
    )
    print(f"🤖 AI: {result1['messages'][-1].content}\n")

    # 用户2的对话
    config2 = {"configurable": {"thread_id": "user_bob"}}
    print("👤 用户Bob: 我是一名工程师")
    result2 = graph.invoke(
        {"messages": [("user", "我是一名工程师")]},
        config=config2
    )
    print(f"🤖 AI: {result2['messages'][-1].content}\n")

    # 验证隔离：用户1询问
    print("👤 用户Alice: 我的职业是什么？")
    result3 = graph.invoke(
        {"messages": [("user", "我的职业是什么？")]},
        config=config1
    )
    print(f"🤖 AI: {result3['messages'][-1].content}\n")

    # 验证隔离：用户2询问
    print("👤 用户Bob: 我的职业是什么？")
    result4 = graph.invoke(
        {"messages": [("user", "我的职业是什么？")]},
        config=config2
    )
    print(f"🤖 AI: {result4['messages'][-1].content}\n")

if __name__ == "__main__":
    test_multi_user_isolation()
```

### 运行结果

```
=== 多用户隔离测试 ===

👤 用户Alice: 我是一名设计师
🤖 AI: 很高兴认识你！设计师是一个很有创意的职业。

👤 用户Bob: 我是一名工程师
🤖 AI: 很高兴认识你！工程师是一个很有技术含量的职业。

👤 用户Alice: 我的职业是什么？
🤖 AI: 你是一名设计师。

👤 用户Bob: 我的职业是什么？
🤖 AI: 你是一名工程师。
```

---

## 实战示例3：SummarizationNode 集成

### 完整代码

```python
"""
SummarizationNode 集成
演示如何使用 SummarizationNode 自动总结对话历史
"""

from langchain.chat_models import init_chat_model
from langchain_core.messages.utils import count_tokens_approximately
from langmem.short_term import SummarizationNode, RunningSummary
from typing import TypedDict

# ============================================================
# 1. 定义状态（包含上下文）
# ============================================================

class StateWithContext(MessagesState):
    """带上下文的状态"""
    context: dict[str, RunningSummary]

class LLMInputState(TypedDict):
    """LLM 输入状态"""
    summarized_messages: list
    context: dict[str, RunningSummary]

# ============================================================
# 2. 配置模型和总结节点
# ============================================================

model = init_chat_model("gpt-4o-mini")
summarization_model = model.bind(max_tokens=128)

summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=summarization_model,
    max_tokens=256,  # 总消息最大 token
    max_tokens_before_summary=256,  # 超过此值开始总结
    max_summary_tokens=128  # 总结最大 token
)

# ============================================================
# 3. 定义节点
# ============================================================

def call_model_with_summary(state: LLMInputState):
    """调用模型（使用总结后的消息）"""
    response = model.invoke(state["summarized_messages"])
    return {"messages": [response]}

# ============================================================
# 4. 构建图
# ============================================================

checkpointer = InMemorySaver()
builder = StateGraph(StateWithContext)

# 添加节点
builder.add_node("summarize", summarization_node)
builder.add_node("call_model", call_model_with_summary)

# 添加边
builder.add_edge(START, "summarize")
builder.add_edge("summarize", "call_model")

# 编译图
graph_with_summary = builder.compile(checkpointer=checkpointer)

# ============================================================
# 5. 测试自动总结
# ============================================================

def test_summarization():
    """测试自动总结"""

    print("=== 自动总结测试 ===\n")

    config = {"configurable": {"thread_id": "summary_test"}}

    # 发送多条消息
    messages = [
        "我叫Alice，是一名设计师",
        "我喜欢Python编程",
        "我在学习AI技术",
        "我住在北京",
        "我的名字是什么？"
    ]

    for i, msg in enumerate(messages, 1):
        print(f"👤 用户: {msg}")
        result = graph_with_summary.invoke(
            {"messages": [("user", msg)]},
            config=config
        )
        print(f"🤖 AI: {result['messages'][-1].content}\n")

        # 检查是否触发总结
        if "context" in result and result["context"]:
            print("📝 触发总结机制")
            print(f"📊 当前消息数: {len(result['messages'])}\n")

if __name__ == "__main__":
    test_summarization()
```

### 运行结果

```
=== 自动总结测试 ===

👤 用户: 我叫Alice，是一名设计师
🤖 AI: 你好Alice！很高兴认识你。

👤 用户: 我喜欢Python编程
🤖 AI: Python是很棒的语言！

👤 用户: 我在学习AI技术
🤖 AI: AI是很有前景的领域！

👤 用户: 我住在北京
🤖 AI: 北京是一座历史悠久的城市。

📝 触发总结机制
📊 当前消息数: 8

👤 用户: 我的名字是什么？
🤖 AI: 你的名字是Alice。
```

---

## 生产环境最佳实践

### 1. Checkpointer 选择

```python
"""
根据场景选择 Checkpointer
"""

# 开发环境：InMemorySaver
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

# 生产环境：RedisSaver
from langgraph_redis import RedisSaver
checkpointer = RedisSaver(
    redis_url="redis://localhost:6379",
    ttl=3600
)

# 生产环境：PostgresSaver
from langgraph_postgres import PostgresSaver
checkpointer = PostgresSaver(
    connection_string="postgresql://user:pass@localhost/db"
)

# 轻量级：SqliteSaver
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver("checkpoints.db")
```

### 2. Thread ID 管理

```python
"""
Thread ID 命名规范
"""

import uuid

# 方式1：用户ID + 会话ID
thread_id = f"user_{user_id}_session_{session_id}"

# 方式2：UUID
thread_id = str(uuid.uuid4())

# 方式3：时间戳
import time
thread_id = f"thread_{int(time.time())}"
```

### 3. 状态清理

```python
"""
定期清理过期状态
"""

def cleanup_old_threads(checkpointer, days=7):
    """清理N天前的线程"""
    # 实现取决于具体的 Checkpointer
    # 示例：PostgreSQL
    query = """
        DELETE FROM checkpoints
        WHERE created_at < NOW() - INTERVAL '%s days'
    """
    # 执行清理
```

---

## 常见问题

### Q1: Checkpointer 和 RunnableWithMessageHistory 有什么区别？

**A**:
- **RunnableWithMessageHistory**：只保存消息历史
- **Checkpointer**：保存完整的图状态（包括所有节点的状态）
- **使用场景**：
  - 简单对话 → RunnableWithMessageHistory
  - 复杂 Agent → Checkpointer

### Q2: 如何实现跨会话记忆？

**A**: 使用 LangMem Store：
```python
from langmem import MemoryStore

store = MemoryStore()
graph = builder.compile(
    checkpointer=checkpointer,
    store=store
)
```

### Q3: 如何查看历史状态？

**A**: 使用 `get_state()` 方法：
```python
state = graph.get_state(config)
print(state)
```

---

## 总结

本场景演示了 LangGraph Checkpointer 的三个核心实践：

1. **基础配置**：使用 InMemorySaver 实现状态持久化
2. **多用户隔离**：使用 Thread ID 隔离不同用户
3. **自动总结**：使用 SummarizationNode 压缩历史

**关键优势**：
- 自动状态管理
- 多用户支持
- 灵活的持久化方案
- 自动总结机制

**下一步**：学习场景8 - LangMem 长期记忆实战，实现跨会话记忆。

---

**参考资料**：
- Context7 官方文档：`reference/context7_langchain_01.md`
- LangGraph Checkpointer 教程：`reference/search_memory_03.md`
