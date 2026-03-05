# 核心概念05：多Schema架构

> 使用多个 Schema 定义输入、输出和整体状态

## 概念定义

**多Schema架构允许为图定义不同的状态视图：整体状态(state_schema)、输入状态(input_schema)和输出状态(output_schema)，实现状态的分层管理和接口隔离。**

## 为什么需要多Schema

### 问题场景

```python
# 单一 Schema 的问题
class State(TypedDict):
    # 输入字段
    query: str
    user_id: str

    # 内部字段
    intermediate_results: list
    debug_info: dict

    # 输出字段
    answer: str

# 问题：
# 1. 用户需要提供所有字段吗？
# 2. 内部字段应该暴露给用户吗？
# 3. 输出应该包含所有字段吗？
```

**多Schema架构解决了状态的可见性和接口设计问题。**

## 三种Schema

### 1. State Schema（整体状态）

**定义**：图内部使用的完整状态。

```python
from typing import TypedDict, Annotated
import operator

class OverallState(TypedDict):
    # 输入字段
    query: str
    user_id: str

    # 内部字段
    intermediate_results: Annotated[list, operator.add]
    debug_info: dict

    # 输出字段
    answer: str
```

**用途**：
- 节点之间传递数据
- 存储中间计算结果
- 保存调试信息

### 2. Input Schema（输入状态）

**定义**：用户调用图时需要提供的字段。

```python
class InputState(TypedDict):
    query: str
    user_id: str
```

**用途**：
- 定义图的输入接口
- 验证用户输入
- 文档化API

### 3. Output Schema（输出状态）

**定义**：图执行完成后返回给用户的字段。

```python
class OutputState(TypedDict):
    answer: str
    user_id: str  # 可选：返回用户ID
```

**用途**：
- 定义图的输出接口
- 隐藏内部实现细节
- 控制返回数据

## 基础用法

### 1. 创建多Schema图

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph

# 1. 定义三个 Schema
class OverallState(TypedDict):
    query: str
    user_id: str
    intermediate_results: Annotated[list, operator.add]
    answer: str

class InputState(TypedDict):
    query: str
    user_id: str

class OutputState(TypedDict):
    answer: str

# 2. 创建图
graph = StateGraph(
    state_schema=OverallState,
    input_schema=InputState,
    output_schema=OutputState
)
```

### 2. 节点使用整体状态

```python
def process_node(state: OverallState) -> dict:
    """节点使用整体状态"""
    query = state["query"]
    user_id = state["user_id"]

    # 可以访问和修改所有字段
    return {
        "intermediate_results": [{"step": "process", "data": "..."}],
        "answer": f"Processed: {query}"
    }

graph.add_node("process", process_node)
```

### 3. 用户只需提供输入字段

```python
# 用户调用
result = graph.invoke({
    "query": "What is LangGraph?",
    "user_id": "user123"
    # 不需要提供 intermediate_results 和 answer
})

# 输出只包含 OutputState 定义的字段
print(result)
# {"answer": "Processed: What is LangGraph?"}
```

## 高级模式

### 1. 输入输出字段重叠

```python
class OverallState(TypedDict):
    messages: Annotated[list, operator.add]
    user_id: str
    context: str

class InputState(TypedDict):
    messages: list  # 用户提供初始消息
    user_id: str

class OutputState(TypedDict):
    messages: list  # 返回完整对话历史
```

### 2. 输出包含部分输入

```python
class InputState(TypedDict):
    query: str
    user_id: str
    session_id: str

class OutputState(TypedDict):
    answer: str
    user_id: str  # 返回用户ID
    session_id: str  # 返回会话ID
```

### 3. 嵌套Schema

```python
class InternalState(TypedDict):
    cache: dict
    metrics: dict

class OverallState(InternalState):
    query: str
    answer: str

class InputState(TypedDict):
    query: str

class OutputState(TypedDict):
    answer: str
```

## 在LangGraph中的应用

### 1. 聊天机器人

```python
from typing import TypedDict, Annotated, List
import operator

# 整体状态
class ChatOverallState(TypedDict):
    # 输入
    messages: Annotated[List[dict], operator.add]
    user_id: str

    # 内部
    context: str
    retrieved_docs: list

    # 输出
    response: str

# 输入状态
class ChatInputState(TypedDict):
    messages: List[dict]
    user_id: str

# 输出状态
class ChatOutputState(TypedDict):
    messages: List[dict]  # 包含用户消息和AI回复
    response: str

# 创建图
graph = StateGraph(
    state_schema=ChatOverallState,
    input_schema=ChatInputState,
    output_schema=ChatOutputState
)

# 节点
def retrieve_node(state: ChatOverallState) -> dict:
    # 检索相关文档
    docs = retrieve(state["messages"][-1]["content"])
    return {"retrieved_docs": docs}

def generate_node(state: ChatOverallState) -> dict:
    # 生成回复
    response = llm.invoke(
        messages=state["messages"],
        context=state["retrieved_docs"]
    )
    return {
        "messages": [{"role": "assistant", "content": response}],
        "response": response
    }

graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_edge("retrieve", "generate")
graph.set_entry_point("retrieve")
graph.set_finish_point("generate")

# 使用
result = graph.invoke({
    "messages": [{"role": "user", "content": "Hello"}],
    "user_id": "user123"
})
# 输出：{"messages": [...], "response": "..."}
# 不包含 retrieved_docs
```

### 2. 数据处理管道

```python
# 整体状态
class PipelineOverallState(TypedDict):
    # 输入
    raw_data: list
    config: dict

    # 内部
    processed_data: Annotated[list, operator.add]
    errors: Annotated[list, operator.add]
    stats: dict

    # 输出
    final_data: list
    summary: dict

# 输入状态
class PipelineInputState(TypedDict):
    raw_data: list
    config: dict

# 输出状态
class PipelineOutputState(TypedDict):
    final_data: list
    summary: dict

graph = StateGraph(
    state_schema=PipelineOverallState,
    input_schema=PipelineInputState,
    output_schema=PipelineOutputState
)
```

### 3. 工作流系统

```python
# 整体状态
class WorkflowOverallState(TypedDict):
    # 输入
    task_id: str
    task_config: dict

    # 内部
    steps: Annotated[list, operator.add]
    retry_count: Annotated[int, operator.add]
    internal_state: dict

    # 输出
    status: str
    result: dict
    error: str

# 输入状态
class WorkflowInputState(TypedDict):
    task_id: str
    task_config: dict

# 输出状态
class WorkflowOutputState(TypedDict):
    task_id: str
    status: str
    result: dict
    error: str

graph = StateGraph(
    state_schema=WorkflowOverallState,
    input_schema=WorkflowInputState,
    output_schema=WorkflowOutputState
)
```

## Schema继承

### 1. 基础继承

```python
# 基础状态
class BaseState(TypedDict):
    user_id: str
    timestamp: float

# 输入继承
class InputState(BaseState):
    query: str

# 整体继承
class OverallState(BaseState):
    query: str
    intermediate: list
    answer: str

# 输出继承
class OutputState(BaseState):
    answer: str
```

### 2. 组合继承

```python
class UserInfo(TypedDict):
    user_id: str
    user_name: str

class QueryInfo(TypedDict):
    query: str
    query_type: str

class OverallState(UserInfo, QueryInfo):
    answer: str
    internal_data: dict

class InputState(UserInfo, QueryInfo):
    pass

class OutputState(UserInfo):
    answer: str
```

## 最佳实践

### 1. 最小输入原则

```python
# ✓ 只要求必需字段
class InputState(TypedDict):
    query: str

# ❌ 要求过多字段
class InputState(TypedDict):
    query: str
    user_id: str
    session_id: str
    metadata: dict
```

### 2. 最小输出原则

```python
# ✓ 只返回必要信息
class OutputState(TypedDict):
    answer: str

# ❌ 返回内部实现细节
class OutputState(TypedDict):
    answer: str
    intermediate_results: list
    debug_info: dict
```

### 3. 清晰的命名

```python
# ✓ 清晰的命名
class ChatOverallState(TypedDict):
    pass

class ChatInputState(TypedDict):
    pass

class ChatOutputState(TypedDict):
    pass

# ❌ 模糊的命名
class State1(TypedDict):
    pass

class State2(TypedDict):
    pass
```

### 4. 文档化Schema

```python
class InputState(TypedDict):
    """
    聊天机器人输入状态

    Attributes:
        messages: 用户消息列表
        user_id: 用户唯一标识符
    """
    messages: list
    user_id: str
```

## 常见错误

### 1. 输入输出不一致

```python
# ❌ 错误：输出字段不在整体状态中
class OverallState(TypedDict):
    query: str
    answer: str

class OutputState(TypedDict):
    result: str  # ❌ result 不在 OverallState 中

# ✓ 正确
class OutputState(TypedDict):
    answer: str  # ✓ answer 在 OverallState 中
```

### 2. 忘记定义输入字段

```python
# ❌ 错误：输入字段不在整体状态中
class OverallState(TypedDict):
    answer: str

class InputState(TypedDict):
    query: str  # ❌ query 不在 OverallState 中

# ✓ 正确
class OverallState(TypedDict):
    query: str
    answer: str
```

### 3. 暴露内部字段

```python
# ❌ 错误：输出包含内部字段
class OutputState(TypedDict):
    answer: str
    debug_info: dict  # ❌ 不应暴露调试信息

# ✓ 正确
class OutputState(TypedDict):
    answer: str
```

## 调试技巧

### 1. 验证Schema一致性

```python
def validate_schemas(overall, input_schema, output_schema):
    """验证Schema一致性"""
    overall_fields = set(overall.__annotations__.keys())
    input_fields = set(input_schema.__annotations__.keys())
    output_fields = set(output_schema.__annotations__.keys())

    # 检查输入字段
    if not input_fields.issubset(overall_fields):
        missing = input_fields - overall_fields
        print(f"Input fields not in overall: {missing}")

    # 检查输出字段
    if not output_fields.issubset(overall_fields):
        missing = output_fields - overall_fields
        print(f"Output fields not in overall: {missing}")

validate_schemas(OverallState, InputState, OutputState)
```

### 2. 打印Schema信息

```python
def print_schema_info(graph):
    """打印Schema信息"""
    print("Overall State:", graph.state_schema.__annotations__)
    print("Input State:", graph.input_schema.__annotations__)
    print("Output State:", graph.output_schema.__annotations__)

print_schema_info(graph)
```

## 总结

**多Schema架构提供了状态的分层管理**：
- State Schema：内部完整状态
- Input Schema：用户输入接口
- Output Schema：用户输出接口

**核心优势**：
- 接口隔离：隐藏内部实现
- 清晰的API：明确输入输出
- 灵活性：支持复杂状态管理

**最佳实践**：
- 最小输入输出原则
- 清晰的命名和文档
- 验证Schema一致性

## 参考资料

- TypedDict基础：`03_核心概念_01_TypedDict状态定义.md`
- 实战应用：`07_实战代码_场景3_多Schema架构实战.md`
- StateGraph初始化：`03_核心概念_08_StateGraph初始化.md`
