# 核心概念08：StateGraph初始化

> StateGraph的创建和配置

## 概念定义

**StateGraph初始化是创建图实例的过程，包括定义状态Schema、配置输入输出Schema、设置节点和边，最终编译成可执行的图。**

## 基础初始化

### 1. 最简单的初始化

```python
from typing import TypedDict
from langgraph.graph import StateGraph

class State(TypedDict):
    messages: list

# 创建图
graph = StateGraph(State)

# 添加节点
def my_node(state: State) -> dict:
    return {"messages": ["hello"]}

graph.add_node("my_node", my_node)

# 设置入口和出口
graph.set_entry_point("my_node")
graph.set_finish_point("my_node")

# 编译
app = graph.compile()
```

### 2. 完整初始化

```python
from typing import TypedDict, Annotated
import operator

class OverallState(TypedDict):
    messages: Annotated[list, operator.add]
    user_id: str

class InputState(TypedDict):
    messages: list
    user_id: str

class OutputState(TypedDict):
    messages: list

# 创建图（多Schema）
graph = StateGraph(
    state_schema=OverallState,
    input_schema=InputState,
    output_schema=OutputState
)
```

## 初始化参数

### 1. state_schema（必需）

```python
# 定义整体状态
class State(TypedDict):
    field1: str
    field2: int

graph = StateGraph(state_schema=State)
```

### 2. input_schema（可选）

```python
# 定义输入接口
class InputState(TypedDict):
    field1: str

graph = StateGraph(
    state_schema=State,
    input_schema=InputState
)
```

### 3. output_schema（可选）

```python
# 定义输出接口
class OutputState(TypedDict):
    field1: str

graph = StateGraph(
    state_schema=State,
    output_schema=OutputState
)
```

### 4. config_schema（可选）

```python
from pydantic import BaseModel

class ConfigSchema(BaseModel):
    max_iterations: int = 10
    timeout: float = 30.0

graph = StateGraph(
    state_schema=State,
    config_schema=ConfigSchema
)
```

## 添加节点

### 1. 基础节点

```python
def my_node(state: State) -> dict:
    return {"field": "value"}

graph.add_node("my_node", my_node)
```

### 2. 异步节点

```python
async def async_node(state: State) -> dict:
    result = await async_operation()
    return {"field": result}

graph.add_node("async_node", async_node)
```

### 3. 类方法节点

```python
class MyProcessor:
    def process(self, state: State) -> dict:
        return {"field": "processed"}

processor = MyProcessor()
graph.add_node("processor", processor.process)
```

## 添加边

### 1. 普通边

```python
# 固定连接
graph.add_edge("node1", "node2")
```

### 2. 条件边

```python
def route(state: State) -> str:
    if state["count"] > 10:
        return "high"
    return "low"

graph.add_conditional_edges(
    "decision_node",
    route,
    {
        "high": "high_handler",
        "low": "low_handler"
    }
)
```

### 3. 入口和出口

```python
# 设置入口点
graph.set_entry_point("start_node")

# 设置出口点
graph.set_finish_point("end_node")
```

## 编译图

### 1. 基础编译

```python
app = graph.compile()
```

### 2. 带Checkpointer编译

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
```

### 3. 带中断编译

```python
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_review"],
    interrupt_after=["critical_step"]
)
```

## 完整示例

### 1. 聊天机器人初始化

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph

class ChatState(TypedDict):
    messages: Annotated[list, operator.add]
    user_id: str

# 创建图
graph = StateGraph(ChatState)

# 添加节点
def user_input(state: ChatState) -> dict:
    return {"messages": [{"role": "user", "content": "Hello"}]}

def llm_response(state: ChatState) -> dict:
    messages = state["messages"]
    response = "Hi there!"
    return {"messages": [{"role": "assistant", "content": response}]}

graph.add_node("user_input", user_input)
graph.add_node("llm_response", llm_response)

# 添加边
graph.add_edge("user_input", "llm_response")

# 设置入口和出口
graph.set_entry_point("user_input")
graph.set_finish_point("llm_response")

# 编译
app = graph.compile()

# 运行
result = app.invoke({"messages": [], "user_id": "user123"})
```

### 2. 工作流初始化

```python
class WorkflowState(TypedDict):
    task_id: str
    status: str
    steps: Annotated[list, operator.add]

graph = StateGraph(WorkflowState)

def init_task(state: WorkflowState) -> dict:
    return {"status": "initialized", "steps": ["init"]}

def process_task(state: WorkflowState) -> dict:
    return {"status": "processing", "steps": ["process"]}

def complete_task(state: WorkflowState) -> dict:
    return {"status": "completed", "steps": ["complete"]}

graph.add_node("init", init_task)
graph.add_node("process", process_task)
graph.add_node("complete", complete_task)

graph.add_edge("init", "process")
graph.add_edge("process", "complete")

graph.set_entry_point("init")
graph.set_finish_point("complete")

app = graph.compile()
```

## 初始化模式

### 1. 构建器模式

```python
def create_chat_graph():
    """工厂函数创建图"""
    graph = StateGraph(ChatState)

    # 添加节点
    graph.add_node("input", user_input)
    graph.add_node("llm", llm_response)

    # 添加边
    graph.add_edge("input", "llm")

    # 设置入口出口
    graph.set_entry_point("input")
    graph.set_finish_point("llm")

    return graph.compile()

app = create_chat_graph()
```

### 2. 配置驱动

```python
def create_graph_from_config(config: dict):
    """从配置创建图"""
    graph = StateGraph(config["state_schema"])

    # 添加节点
    for node_name, node_func in config["nodes"].items():
        graph.add_node(node_name, node_func)

    # 添加边
    for source, target in config["edges"]:
        graph.add_edge(source, target)

    graph.set_entry_point(config["entry"])
    graph.set_finish_point(config["finish"])

    return graph.compile()
```

## 最佳实践

### 1. 清晰的结构

```python
# ✓ 清晰的初始化流程
def create_graph():
    # 1. 定义状态
    class State(TypedDict):
        pass

    # 2. 创建图
    graph = StateGraph(State)

    # 3. 添加节点
    graph.add_node("node1", func1)

    # 4. 添加边
    graph.add_edge("node1", "node2")

    # 5. 设置入口出口
    graph.set_entry_point("node1")
    graph.set_finish_point("node2")

    # 6. 编译
    return graph.compile()
```

### 2. 验证配置

```python
def validate_graph(graph):
    """验证图配置"""
    # 检查是否有入口点
    if not graph.entry_point:
        raise ValueError("No entry point set")

    # 检查是否有出口点
    if not graph.finish_point:
        raise ValueError("No finish point set")

    # 检查节点连接
    for node in graph.nodes:
        if not graph.has_edge_to(node):
            print(f"Warning: {node} has no incoming edges")
```

### 3. 错误处理

```python
def safe_compile(graph):
    """安全编译图"""
    try:
        validate_graph(graph)
        return graph.compile()
    except Exception as e:
        print(f"Compilation failed: {e}")
        return None
```

## 总结

**StateGraph初始化的核心步骤**：
1. 定义状态Schema
2. 创建StateGraph实例
3. 添加节点和边
4. 设置入口和出口
5. 编译成可执行图

**关键点**：
- state_schema是必需的
- input/output_schema是可选的
- 编译前必须设置入口和出口
- 支持多种初始化模式

## 参考资料

- TypedDict基础：`03_核心概念_01_TypedDict状态定义.md`
- 多Schema架构：`03_核心概念_05_多Schema架构.md`
- 实战应用：`07_实战代码_场景1_基础TypedDict状态定义.md`
