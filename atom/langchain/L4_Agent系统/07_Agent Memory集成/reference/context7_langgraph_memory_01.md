---
type: context7_documentation
library: langgraph
version: 0.x (2025-2026)
fetched_at: 2026-03-06
knowledge_point: 07_Agent Memory集成
context7_query: memory persistence checkpointer state management long term memory
---

# Context7 文档：LangGraph Memory & Persistence

## 文档来源
- 库名称：LangGraph
- 版本：latest (2025-2026)
- 官方文档链接：https://docs.langchain.com/oss/python/langgraph

## 关键信息提取

### 1. LangGraph 双层记忆架构

LangGraph 将记忆分为两层：

#### Checkpointer（短期记忆 / 线程内）
- 保持**同一 thread 内**的对话状态
- 每个图步骤自动保存检查点
- 类似"游戏存档" - 可以回溯到任意步骤
- 适用：会话内的对话历史

#### Store（长期记忆 / 跨线程）
- 在**不同 thread 之间**共享数据
- 如用户偏好、学习的事实
- 需要显式读写
- 适用：跨会话的用户画像、持久知识

### 2. Checkpointer 集成示例

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

checkpointer = InMemorySaver()

builder = StateGraph(MessagesState)
# ... add nodes and edges ...
graph = builder.compile(checkpointer=checkpointer)

# 同一 thread_id 保持对话
config = {"configurable": {"thread_id": "session-1"}}
graph.invoke({"messages": [...]}, config)
```

### 3. Store 长期记忆 + 语义搜索

```python
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.runtime import Runtime

# 创建带语义搜索的 Store
embeddings = init_embeddings("openai:text-embedding-3-small")
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
    }
)

# 存储用户记忆
store.put(("user_123", "memories"), "1", {"text": "I love pizza"})
store.put(("user_123", "memories"), "2", {"text": "I am a plumber"})

# 在节点中使用语义搜索检索记忆
async def chat(state: MessagesState, runtime: Runtime):
    items = await runtime.store.asearch(
        ("user_123", "memories"),
        query=state["messages"][-1].content,
        limit=2
    )
    memories = "\n".join(item.value["text"] for item in items)
    # ... 注入到 system prompt 中
```

### 4. Checkpointer + Store 组合使用

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

# 短期记忆（线程内）
checkpointer = InMemorySaver()

# 长期记忆（跨线程）
store = InMemoryStore(index={"embed": embeddings, "dims": 1536})

# 同时编译
graph = builder.compile(checkpointer=checkpointer, store=store)
```

### 5. context_schema 用户上下文

```python
from dataclasses import dataclass

@dataclass
class Context:
    user_id: str

builder = StateGraph(MessagesState, context_schema=Context)
graph = builder.compile(checkpointer=checkpointer, store=store)
```

### 6. Functional API 短期记忆

```python
@entrypoint(checkpointer=checkpointer)
def my_workflow(number: int, *, previous: Any = None) -> int:
    previous = previous or 0
    return number + previous

config = {"configurable": {"thread_id": "some_thread_id"}}
my_workflow.invoke(1, config)  # 1
my_workflow.invoke(2, config)  # 3 (previous=1)
```

### 7. 生产级 Checkpointer

| Checkpointer | 后端 | 适用场景 |
|---------------|------|----------|
| InMemorySaver | 内存 | 开发/测试 |
| SqliteSaver | SQLite | 轻量生产 |
| PostgresSaver | PostgreSQL | 企业生产 |
| RedisSaver | Redis | 高性能/低延迟 |
