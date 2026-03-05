---
type: context7_documentation
library: langgraph
version: latest (2026)
fetched_at: 2026-02-27
knowledge_point: 08_状态转换函数
context7_query: state management reducers state transition functions
---

# Context7 文档：LangGraph 状态转换与 Reducer

## 文档来源
- 库名称：LangGraph
- Context7 ID：/websites/langchain_oss_python_langgraph
- 官方文档链接：https://docs.langchain.com/oss/python/langgraph/graph-api

## 关键信息提取

### 1. 自定义 Reducer 定义

使用 `Annotated` 类型注解绑定 reducer 函数到状态字段：

```python
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: int                          # 无 reducer，默认覆盖
    bar: Annotated[list[str], add]    # 有 reducer，列表追加
```

### 2. 自定义 Reducer 函数

```python
def add(left, right):
    """自定义 reducer：合并两个值"""
    return left + right
```

### 3. 消息列表管理

```python
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

### 4. Command API - 状态更新 + 控制流

```python
from langgraph.types import Command
from typing import Literal

def my_node(state: State) -> Command[Literal["my_other_node"]]:
    return Command(
        update={"foo": "bar"},
        goto="my_other_node"
    )
```

### 5. update_state 外部更新

```python
# 尊重 reducer 函数
graph.update_state(config, {"foo": 2, "bar": ["b"]})
# foo 被覆盖为 2（无 reducer）
# bar 被追加 "b"（有 add reducer）
```

### 6. 子图状态传递

使用 `Annotated` + `operator.add` 在父图和子图之间传递状态：

```python
import operator
from typing_extensions import Annotated
from typing import TypedDict

class State(TypedDict):
    foo: Annotated[str, operator.add]
```
