# 实战代码 - 场景8：复杂状态Schema设计

> 完整可运行的复杂状态 Schema 设计示例

## 场景描述

在实际应用中，状态 Schema 可能需要处理复杂的数据结构、多层嵌套、动态字段等。本场景演示如何设计和实现复杂的状态 Schema。

**应用场景**：
- 多步骤工作流
- 复杂业务逻辑
- 多数据源集成
- 动态配置系统

## 核心知识点

1. **嵌套状态结构**：多层数据嵌套
2. **动态字段扩展**：运行时添加字段
3. **Schema 演化**：向后兼容的 Schema 更新
4. **类型安全**：复杂类型的类型检查

## 完整代码：多层嵌套状态

```python
from typing import TypedDict, Annotated, List, Dict
from typing_extensions import NotRequired
import operator
from langgraph.graph import StateGraph, START, END

# 1. 定义嵌套状态结构
class Message(TypedDict):
    role: str
    content: str
    timestamp: float

class UserProfile(TypedDict):
    user_id: str
    name: str
    preferences: Dict[str, str]

class SearchResult(TypedDict):
    source: str
    content: str
    score: float

class State(TypedDict):
    # 用户信息
    user: UserProfile

    # 消息历史
    messages: Annotated[List[Message], operator.add]

    # 搜索结果
    search_results: Annotated[List[SearchResult], operator.add]

    # 元数据
    metadata: Annotated[Dict[str, any], operator.or_]

    # 可选字段
    context: NotRequired[str]
    debug_info: NotRequired[Dict[str, any]]

# 2. 定义节点
def process_user(state: State) -> dict:
    """处理用户信息"""
    user = state['user']
    print(f"Processing user: {user['name']}")

    return {
        "messages": [{
            "role": "system",
            "content": f"Welcome {user['name']}",
            "timestamp": time.time()
        }],
        "metadata": {"user_processed": True}
    }

def search_data(state: State) -> dict:
    """搜索数据"""
    query = state['messages'][-1]['content'] if state['messages'] else ""

    return {
        "search_results": [
            {
                "source": "database",
                "content": f"Result for: {query}",
                "score": 0.95
            }
        ],
        "metadata": {"search_completed": True}
    }

def generate_response(state: State) -> dict:
    """生成响应"""
    results = state['search_results']
    user = state['user']

    response = f"Found {len(results)} results for {user['name']}"

    return {
        "messages": [{
            "role": "assistant",
            "content": response,
            "timestamp": time.time()
        }]
    }

# 3. 构建图
graph = StateGraph(State)

graph.add_node("process_user", process_user)
graph.add_node("search", search_data)
graph.add_node("generate", generate_response)

graph.add_edge(START, "process_user")
graph.add_edge("process_user", "search")
graph.add_edge("search", "generate")
graph.add_edge("generate", END)

app = graph.compile()

# 4. 执行
if __name__ == "__main__":
    import time

    result = app.invoke({
        "user": {
            "user_id": "user_123",
            "name": "Alice",
            "preferences": {"lang": "en", "theme": "dark"}
        },
        "messages": [{
            "role": "user",
            "content": "Search for LangGraph tutorials",
            "timestamp": time.time()
        }],
        "search_results": [],
        "metadata": {}
    })

    print("\n=== Final State ===")
    print(f"User: {result['user']['name']}")
    print(f"Messages: {len(result['messages'])}")
    print(f"Search results: {len(result['search_results'])}")
    print(f"Metadata: {result['metadata']}")
```

## 完整代码：动态字段扩展

```python
from typing import TypedDict, Annotated, Any, Dict
import operator
from langgraph.graph import StateGraph, START, END

# 1. 定义基础状态
class BaseState(TypedDict):
    query: str
    results: Annotated[list, operator.add]
    metadata: Annotated[Dict[str, Any], operator.or_]

# 2. 动态扩展状态
def extend_state(state: BaseState, extensions: Dict[str, Any]) -> dict:
    """动态扩展状态"""
    return {
        "metadata": {
            **state.get('metadata', {}),
            "extensions": extensions
        }
    }

# 3. 定义节点
def dynamic_node(state: BaseState) -> dict:
    """动态添加字段"""
    # 根据条件动态添加字段
    extensions = {}

    if "advanced" in state['query']:
        extensions['advanced_mode'] = True
        extensions['extra_data'] = {"key": "value"}

    return extend_state(state, extensions)

# 4. 构建图
graph = StateGraph(BaseState)
graph.add_node("dynamic", dynamic_node)
graph.add_edge(START, "dynamic")
graph.add_edge("dynamic", END)

app = graph.compile()

# 5. 执行
if __name__ == "__main__":
    result = app.invoke({
        "query": "advanced search",
        "results": [],
        "metadata": {}
    })

    print(f"Metadata: {result['metadata']}")
```

## 完整代码：Schema 演化

```python
from typing import TypedDict, Annotated
from typing_extensions import NotRequired
import operator
from langgraph.graph import StateGraph, START, END

# 1. V1 Schema
class StateV1(TypedDict):
    messages: Annotated[list, operator.add]
    user_id: str

# 2. V2 Schema（向后兼容）
class StateV2(TypedDict):
    messages: Annotated[list, operator.add]
    user_id: str
    # 新增字段（可选）
    session_id: NotRequired[str]
    metadata: NotRequired[dict]

# 3. 迁移函数
def migrate_v1_to_v2(state_v1: StateV1) -> StateV2:
    """从 V1 迁移到 V2"""
    return {
        **state_v1,
        "session_id": f"session_{state_v1['user_id']}",
        "metadata": {"version": "v2"}
    }

# 4. 兼容节点
def compatible_node(state: StateV2) -> dict:
    """兼容 V1 和 V2 的节点"""
    # 安全访问新字段
    session_id = state.get('session_id', 'default_session')
    metadata = state.get('metadata', {})

    return {
        "messages": [f"Session: {session_id}"],
        "metadata": {**metadata, "processed": True}
    }

# 5. 构建图
graph = StateGraph(StateV2)
graph.add_node("process", compatible_node)
graph.add_edge(START, "process")
graph.add_edge("process", END)

app = graph.compile()

# 6. 执行（V1 数据）
if __name__ == "__main__":
    # V1 数据
    state_v1: StateV1 = {
        "messages": ["hello"],
        "user_id": "user_123"
    }

    # 迁移到 V2
    state_v2 = migrate_v1_to_v2(state_v1)

    # 执行
    result = app.invoke(state_v2)

    print(f"Messages: {result['messages']}")
    print(f"Session ID: {result.get('session_id')}")
    print(f"Metadata: {result.get('metadata')}")
```

## 完整代码：多 Schema 架构

```python
from typing import TypedDict, Annotated
from typing_extensions import NotRequired
import operator
from langgraph.graph import StateGraph, START, END

# 1. 定义多个 Schema
class InputState(TypedDict):
    """输入 Schema"""
    query: str
    user_id: str

class PrivateState(TypedDict):
    """私有 Schema"""
    query: str
    user_id: str
    internal_data: dict
    cache: dict

class OutputState(TypedDict):
    """输出 Schema"""
    response: str
    metadata: dict

class OverallState(TypedDict):
    """整体 Schema"""
    query: str
    user_id: str
    internal_data: NotRequired[dict]
    cache: NotRequired[dict]
    response: NotRequired[str]
    metadata: NotRequired[dict]

# 2. 定义节点
def input_node(state: InputState) -> dict:
    """输入节点（只能访问 InputState）"""
    return {
        "internal_data": {"query_length": len(state['query'])},
        "cache": {}
    }

def process_node(state: PrivateState) -> dict:
    """处理节点（可以访问 PrivateState）"""
    query = state['query']
    internal_data = state['internal_data']

    return {
        "response": f"Processed: {query}",
        "metadata": {"internal": internal_data}
    }

def output_node(state: OverallState) -> dict:
    """输出节点（返回 OutputState）"""
    return {
        "response": state.get('response', ''),
        "metadata": state.get('metadata', {})
    }

# 3. 构建图
graph = StateGraph(OverallState)

graph.add_node("input", input_node)
graph.add_node("process", process_node)
graph.add_node("output", output_node)

graph.add_edge(START, "input")
graph.add_edge("input", "process")
graph.add_edge("process", "output")
graph.add_edge("output", END)

app = graph.compile()

# 4. 执行
if __name__ == "__main__":
    result = app.invoke({
        "query": "What is LangGraph?",
        "user_id": "user_123"
    })

    print(f"Response: {result.get('response')}")
    print(f"Metadata: {result.get('metadata')}")
```

## 设计模式

### 1. 分层状态模式

```python
# 应用层
class ApplicationState(TypedDict):
    user_input: str
    final_output: str

# 业务层
class BusinessState(TypedDict):
    user_input: str
    processed_data: dict
    final_output: str

# 数据层
class DataState(TypedDict):
    user_input: str
    processed_data: dict
    db_results: list
    final_output: str
```

### 2. 事件驱动模式

```python
class Event(TypedDict):
    type: str
    data: dict
    timestamp: float

class EventDrivenState(TypedDict):
    events: Annotated[List[Event], operator.add]
    current_state: str
    metadata: dict
```

### 3. 管道模式

```python
class PipelineStage(TypedDict):
    name: str
    input: dict
    output: dict
    status: str

class PipelineState(TypedDict):
    stages: Annotated[List[PipelineStage], operator.add]
    current_stage: int
    final_result: NotRequired[dict]
```

## 最佳实践

1. **保持扁平化**：避免过度嵌套
2. **使用 NotRequired**：标记可选字段
3. **Schema 版本化**：支持向后兼容
4. **类型注解完整**：确保类型安全
5. **文档化设计**：说明 Schema 设计意图

## 常见陷阱

### 陷阱1：过度嵌套

```python
# 错误：过度嵌套
class BadState(TypedDict):
    data: Dict[str, Dict[str, Dict[str, Any]]]

# 正确：扁平化
class GoodState(TypedDict):
    data_level1: dict
    data_level2: dict
    data_level3: dict
```

### 陷阱2：不兼容的 Schema 演化

```python
# 错误：删除必需字段
class StateV1(TypedDict):
    field1: str
    field2: str

class StateV2(TypedDict):
    field1: str
    # field2 被删除了！

# 正确：使用 NotRequired
class StateV2(TypedDict):
    field1: str
    field2: NotRequired[str]  # 标记为可选
```

### 陷阱3：忘记类型注解

```python
# 错误：缺少类型注解
class BadState(TypedDict):
    data: dict  # 太泛化

# 正确：明确类型
class GoodState(TypedDict):
    data: Dict[str, List[str]]  # 明确类型
```

## 参考资料

- 核心概念：`03_核心概念_01_TypedDict状态定义.md`
- 多 Schema：`03_核心概念_05_多Schema架构.md`
- 生产实践：`03_核心概念_10_生产环境最佳实践.md`
