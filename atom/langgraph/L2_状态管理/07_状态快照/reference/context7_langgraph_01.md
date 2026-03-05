---
type: context7_documentation
library: langgraph
version: latest (2026)
fetched_at: 2026-02-27
knowledge_point: 07_状态快照
context7_query: state snapshots get_state get_state_history checkpoint time travel
---

# Context7 文档：LangGraph 状态快照与时间旅行

## 文档来源
- 库名称：LangGraph (Python)
- Context7 ID: /websites/langchain_oss_python_langgraph
- 官方文档链接：https://docs.langchain.com/oss/python/langgraph/persistence

## 关键信息提取

### 1. 获取当前状态快照 (get_state)

```python
# 获取最新状态快照
config = {"configurable": {"thread_id": "1"}}
graph.get_state(config)

# 获取特定 checkpoint_id 的状态快照
config = {
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": "1ef663ba-28fe-6528-8002-5a559208592c"
    }
}
graph.get_state(config)
```

返回 `StateSnapshot` 对象，包含消息历史、配置、元数据和时间戳。

### 2. 获取状态历史 (get_state_history)

```python
# 状态按逆时间顺序返回
states = list(graph.get_state_history(config))

for state in states:
    print(state.next)
    print(state.config["configurable"]["checkpoint_id"])
    print()
```

`get_state_history(config)` 返回 `StateSnapshot` 对象列表，每个代表图在特定时间点的状态。最新的 StateSnapshot 排在最前面。

### 3. 更新状态 (update_state)

```python
# 基于选定的历史状态创建新 checkpoint
new_config = graph.update_state(
    selected_state.config,
    values={"topic": "chickens"}
)
```

`update_state` 方法基于之前的状态创建新 checkpoint。新 checkpoint 关联到同一 thread 但有不同的 checkpoint_id。

更新时遵循 reducer 规则：
```python
from typing import Annotated
from operator import add

class State(TypedDict):
    foo: int                          # 无 reducer → 覆盖
    bar: Annotated[list[str], add]    # 有 reducer → 追加

graph.update_state(config, {"foo": 2, "bar": ["b"]})
# foo 被覆盖为 2，bar 追加 ["b"]
```

### 4. 时间旅行 (Time Travel)

#### 重放 (Replay)
```python
# 使用特定 checkpoint_id 重放执行
config = {
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": "0c62ca34-ac19-445d-bbb0-5b4984975b2a"
    }
}
graph.invoke(None, config=config)
```

LangGraph 智能地重放 checkpoint 之前已执行的步骤，然后重新执行之后的步骤。

#### 分叉 (Fork)
1. 获取历史状态
2. 使用 `update_state()` 修改状态
3. 从新 checkpoint 继续执行

### 5. Checkpointer 能力总结

Checkpointers 提供三大能力：
1. **持久化** - 跨交互保持状态
2. **时间旅行** - 重放和调试过去的执行
3. **分叉** - 从特定 checkpoint 探索替代执行路径
