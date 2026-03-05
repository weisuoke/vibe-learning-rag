---
type: context7_documentation
library: langgraph
version: latest (2026)
fetched_at: 2026-02-27
knowledge_point: 10_状态持久化准备
context7_query: "state persistence checkpointing serialization"
---

# Context7 文档：LangGraph 持久化与序列化

## 文档来源
- 库名称：langgraph
- 官方文档链接：https://docs.langchain.com/oss/python/langgraph/persistence

## 1. 持久化概述

编译 StateGraph 时配置 checkpointer，图会在每个节点执行后自动保存状态（checkpoint）。支持：
- 中断后恢复执行
- 时间旅行调试
- 人机协作工作流
- 持久化长时间运行的 Agent

使用持久化需要传入 thread_id：
```python
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"foo": "", "bar": []}, config)
```

## 2. Checkpoint 数据结构

```python
checkpoint = {
    "v": 4,                          # 版本
    "ts": "2024-07-31T20:14:19...",  # 时间戳
    "id": "1ef4f797-8335-...",       # 唯一 ID
    "channel_values": {              # 实际状态值
        "my_key": "meow",
        "node": "node"
    },
    "channel_versions": {            # 每个通道的版本
        "__start__": 2,
        "my_key": 3,
    },
    "versions_seen": {               # 每个节点看到的版本
        "__input__": {},
        "__start__": {"__start__": 1},
    }
}
```

## 3. 序列化器协议

`langgraph_checkpoint` 定义了：
- `SerializerProtocol` - 自定义序列化器接口
- `JsonPlusSerializer` - 默认实现，处理 LangChain/LangGraph 原语、datetime、enum 等

加密持久化：
```python
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

serde = EncryptedSerializer.from_pycryptodome_aes()  # 读取 LANGGRAPH_AES_KEY
checkpointer = SqliteSaver(sqlite3.connect("checkpoint.db"), serde=serde)
```

## 4. 可用的 Checkpointer 实现

### InMemorySaver（开发/测试）
```python
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
```

### SqliteSaver（同步，文件存储）
```python
from langgraph.checkpoint.sqlite import SqliteSaver
with SqliteSaver.from_conn_string(":memory:") as checkpointer:
    checkpointer.put(write_config, checkpoint, {}, {})
    checkpointer.get(read_config)
    list(checkpointer.list(read_config))
```

### AsyncPostgresSaver（生产环境）
```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    await checkpointer.aput(write_config, checkpoint, {}, {})
```

## 5. 状态检索

### 通过 Graph API
```python
state = graph.get_state(config)
print(state.values['messages'])
```

### 通过 Checkpointer API
```python
checkpointer.get_tuple(config)  # 返回 CheckpointTuple
```

### 列出所有 checkpoint
```python
for checkpoint in memory.list(config):
    print(f"Checkpoint: {checkpoint.config['configurable']['checkpoint_id']}")
```

## 6. Store 与 Checkpointer 集成

```python
graph = builder.compile(checkpointer=checkpointer, store=store)
```
Checkpointer 处理每线程状态持久化；Store 处理跨线程共享数据。

## 7. 完整工作示例

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]

def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}

workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"foo": "", "bar": []}, config)
```
