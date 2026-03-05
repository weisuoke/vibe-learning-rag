# 实战代码 - 场景5：Optional字段与默认值

> 完整可运行的 Optional 字段与默认值处理示例

## 场景描述

在 LangGraph 中，状态字段可能是可选的，需要正确处理 Optional 字段和默认值。本场景演示如何使用 NotRequired、Optional 和默认值机制。

**应用场景**：
- 可选配置参数
- 渐进式状态构建
- 向后兼容的 Schema 演化
- 条件性数据处理

## 核心知识点

1. **NotRequired vs Optional**：语义区别
2. **默认值设置**：TypedDict、Pydantic、Dataclass
3. **None 值处理**：Reducer 中的 None 处理
4. **字段验证**：可选字段的验证逻辑

## NotRequired vs Optional

### 语义区别

```python
from typing import TypedDict, Optional
from typing_extensions import NotRequired

# NotRequired: 字段可以不存在
class State1(TypedDict):
    required_field: str
    optional_field: NotRequired[int]

# Optional: 字段必须存在，但值可以是 None
class State2(TypedDict):
    required_field: str
    optional_field: Optional[int]

# 使用示例
state1: State1 = {"required_field": "value"}  # OK: optional_field 不存在
state2: State2 = {"required_field": "value", "optional_field": None}  # OK: 值是 None
```

### 推荐使用 NotRequired

```python
from typing import TypedDict, Annotated
from typing_extensions import NotRequired
import operator

class State(TypedDict):
    # 必需字段
    query: str
    messages: Annotated[list, operator.add]

    # 可选字段（推荐）
    metadata: NotRequired[dict]
    context: NotRequired[str]
    debug_info: NotRequired[list]
```

## 完整代码：TypedDict 默认值

```python
from typing import TypedDict, Annotated
from typing_extensions import NotRequired
import operator
from langgraph.graph import StateGraph, START, END

# 1. 定义状态 Schema
class State(TypedDict):
    # 必需字段
    query: str
    messages: Annotated[list, operator.add]

    # 可选字段
    metadata: NotRequired[dict]
    context: NotRequired[str]
    count: NotRequired[int]

# 2. 定义节点
def process_query(state: State) -> dict:
    """处理查询"""
    print(f"Processing query: {state['query']}")

    # 安全访问可选字段
    metadata = state.get('metadata', {})
    context = state.get('context', '')
    count = state.get('count', 0)

    print(f"Metadata: {metadata}")
    print(f"Context: {context}")
    print(f"Count: {count}")

    return {
        "messages": [f"Processed: {state['query']}"],
        "count": count + 1
    }

def add_metadata(state: State) -> dict:
    """添加元数据"""
    return {
        "metadata": {"timestamp": "2026-02-25", "source": "api"}
    }

def add_context(state: State) -> dict:
    """添加上下文"""
    return {
        "context": "This is additional context"
    }

# 3. 构建图
graph = StateGraph(State)

graph.add_node("process", process_query)
graph.add_node("add_metadata", add_metadata)
graph.add_node("add_context", add_context)

graph.add_edge(START, "process")
graph.add_edge("process", "add_metadata")
graph.add_edge("add_metadata", "add_context")
graph.add_edge("add_context", END)

app = graph.compile()

# 4. 执行（只提供必需字段）
if __name__ == "__main__":
    result = app.invoke({
        "query": "What is LangGraph?",
        "messages": []
    })

    print("\n=== Final State ===")
    print(f"Query: {result['query']}")
    print(f"Messages: {result['messages']}")
    print(f"Metadata: {result.get('metadata', {})}")
    print(f"Context: {result.get('context', '')}")
    print(f"Count: {result.get('count', 0)}")
```

## 完整代码：Pydantic 默认值

```python
from pydantic import BaseModel, Field
from typing import Annotated
import operator
from langgraph.graph import StateGraph, START, END

# 1. 定义状态 Schema（Pydantic）
class State(BaseModel):
    # 必需字段
    query: str
    messages: Annotated[list, operator.add] = Field(default_factory=list)

    # 可选字段（带默认值）
    metadata: dict = Field(default_factory=dict)
    context: str = ""
    count: int = 0

# 2. 定义节点
def process_query(state: State) -> dict:
    """处理查询"""
    print(f"Processing query: {state.query}")
    print(f"Metadata: {state.metadata}")
    print(f"Context: {state.context}")
    print(f"Count: {state.count}")

    return {
        "messages": [f"Processed: {state.query}"],
        "count": state.count + 1
    }

def add_metadata(state: State) -> dict:
    """添加元数据"""
    return {
        "metadata": {"timestamp": "2026-02-25", "source": "api"}
    }

# 3. 构建图
graph = StateGraph(State)

graph.add_node("process", process_query)
graph.add_node("add_metadata", add_metadata)

graph.add_edge(START, "process")
graph.add_edge("process", "add_metadata")
graph.add_edge("add_metadata", END)

app = graph.compile()

# 4. 执行（只提供必需字段）
if __name__ == "__main__":
    result = app.invoke({"query": "What is LangGraph?"})

    print("\n=== Final State ===")
    print(f"Query: {result['query']}")
    print(f"Messages: {result['messages']}")
    print(f"Metadata: {result['metadata']}")
    print(f"Context: {result['context']}")
    print(f"Count: {result['count']}")
```

## Reducer 中的 None 处理

### 问题：Optional 字段的 Reducer

```python
from typing import TypedDict, Annotated, Optional
import operator

# 问题：如果 messages 是 None 怎么办？
class State(TypedDict):
    messages: Annotated[Optional[list], operator.add]

# operator.add 无法处理 None
# None + [1, 2] -> TypeError
```

### 解决方案：自定义 Reducer

```python
from typing import TypedDict, Annotated, Optional

def safe_add(a: Optional[list], b: list) -> list:
    """安全的列表追加"""
    if a is None:
        return b
    return a + b

class State(TypedDict):
    messages: Annotated[Optional[list], safe_add]

# 使用示例
def node1(state: State) -> dict:
    return {"messages": ["hello"]}

def node2(state: State) -> dict:
    return {"messages": ["world"]}

# 初始状态可以是 None
result = app.invoke({"messages": None})
# messages 会自动变成 ["hello", "world"]
```

## 完整代码：Safe Reducer

```python
from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, START, END

# 1. 定义 Safe Reducer
def safe_list_add(a: Optional[list], b: list) -> list:
    """安全的列表追加"""
    if a is None:
        return b
    return a + b

def safe_dict_merge(a: Optional[dict], b: dict) -> dict:
    """安全的字典合并"""
    if a is None:
        return b
    return {**a, **b}

# 2. 定义状态 Schema
class State(TypedDict):
    query: str
    messages: Annotated[Optional[list], safe_list_add]
    metadata: Annotated[Optional[dict], safe_dict_merge]

# 3. 定义节点
def node1(state: State) -> dict:
    return {
        "messages": ["Message from node1"],
        "metadata": {"node1": True}
    }

def node2(state: State) -> dict:
    return {
        "messages": ["Message from node2"],
        "metadata": {"node2": True}
    }

# 4. 构建图
graph = StateGraph(State)

graph.add_node("node1", node1)
graph.add_node("node2", node2)

graph.add_edge(START, "node1")
graph.add_edge("node1", "node2")
graph.add_edge("node2", END)

app = graph.compile()

# 5. 执行（初始值为 None）
if __name__ == "__main__":
    result = app.invoke({
        "query": "test",
        "messages": None,
        "metadata": None
    })

    print("\n=== Final State ===")
    print(f"Messages: {result['messages']}")
    print(f"Metadata: {result['metadata']}")
```

## 字段验证

### Pydantic 验证

```python
from pydantic import BaseModel, Field, field_validator

class State(BaseModel):
    query: str
    count: int = 0
    metadata: dict = Field(default_factory=dict)

    @field_validator('count')
    @classmethod
    def validate_count(cls, v):
        if v < 0:
            raise ValueError('count must be non-negative')
        return v

    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v):
        if len(v) > 10:
            raise ValueError('metadata cannot have more than 10 keys')
        return v
```

## 最佳实践

### 1. 优先使用 NotRequired

```python
# 推荐
class State(TypedDict):
    required: str
    optional: NotRequired[int]

# 不推荐
class State(TypedDict):
    required: str
    optional: Optional[int]  # 必须提供，但可以是 None
```

### 2. 为可选字段提供默认值

```python
# Pydantic
class State(BaseModel):
    count: int = 0
    metadata: dict = Field(default_factory=dict)

# 访问时使用 get()
def node(state: State) -> dict:
    count = state.get('count', 0)  # TypedDict
    count = state.count  # Pydantic（自动使用默认值）
```

### 3. Reducer 处理 None

```python
def safe_reducer(a: Optional[T], b: T) -> T:
    if a is None:
        return b
    return combine(a, b)
```

### 4. 文档化可选字段

```python
class State(TypedDict):
    query: str  # 必需：用户查询
    messages: Annotated[list, operator.add]  # 必需：消息列表
    metadata: NotRequired[dict]  # 可选：额外元数据
    context: NotRequired[str]  # 可选：上下文信息
```

## 常见陷阱

### 陷阱1：混淆 NotRequired 和 Optional

```python
# 错误：使用 Optional
class State(TypedDict):
    field: Optional[int]

state: State = {}  # TypeError: 缺少 field

# 正确：使用 NotRequired
class State(TypedDict):
    field: NotRequired[int]

state: State = {}  # OK
```

### 陷阱2：忘记处理 None

```python
# 错误：Reducer 不处理 None
class State(TypedDict):
    messages: Annotated[Optional[list], operator.add]

# operator.add 无法处理 None

# 正确：使用 Safe Reducer
def safe_add(a: Optional[list], b: list) -> list:
    return b if a is None else a + b

class State(TypedDict):
    messages: Annotated[Optional[list], safe_add]
```

### 陷阱3：可变默认值

```python
# 错误：可变默认值
class State(TypedDict):
    metadata: dict = {}  # 危险！所有实例共享

# 正确：使用 default_factory
from pydantic import BaseModel, Field

class State(BaseModel):
    metadata: dict = Field(default_factory=dict)
```

## 参考资料

- 核心概念：`03_核心概念_04_Field默认值与验证.md`
- typing_extensions：`reference/context7_typing_extensions_01.md`
- Pydantic：`reference/context7_pydantic_01.md`
