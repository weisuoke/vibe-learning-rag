---
type: fetched_content
source: https://www.reddit.com/r/LangChain/comments/1hxt5t7/help_me_understand_state_reducers_in_langgraph
title: Help Me Understand State Reducers in LangGraph
fetched_at: 2026-02-27
status: partial (reconstructed from search)
author: Reddit community
knowledge_point: 08_状态转换函数
fetch_tool: grok-mcp
---

# Help Me Understand State Reducers in LangGraph

## 核心问题

用户在使用 LangGraph 时对 reducer 的工作机制有疑问：

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
    some_list: list[str]
    some_counter: int
```

## 社区回答要点

### Reducer 签名
```python
def reducer(old_value, new_value) -> merged_value:
    ...
```

### 常见 Reducer 模式

1. **消息追加**: `add_messages`
2. **列表拼接**: `operator.add`
3. **集合并集**: `operator.or_`
4. **取最大值**:
```python
def take_max(old, new):
    if new is None:
        return old
    return max(old, new) if old is not None else new
```

5. **计数器递增**:
```python
def increment(old: int, new: int | None) -> int:
    return (old or 0) + (new or 1)
```

### 关键行为
- 如果节点不返回某个 key → 不变，reducer 不被调用
- Reducer 接收 (previous_persisted_value, value_returned_by_node)
- 单个元素追加需要包装成列表: `{"some_list": [single_item]}`
