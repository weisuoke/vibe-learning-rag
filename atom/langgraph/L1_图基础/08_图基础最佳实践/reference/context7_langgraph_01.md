---
type: context7_documentation
library: LangGraph
library_id: /websites/langchain_oss_python_langgraph
version: main (2026-02-17)
fetched_at: 2026-02-25
knowledge_point: 08_图基础最佳实践
context7_query: common mistakes anti patterns pitfalls, error handling retry policy performance optimization, graph design best practices node granularity state design
---

# Context7 文档：LangGraph 图基础最佳实践

## 文档来源
- 库名称：LangGraph
- Library ID：/websites/langchain_oss_python_langgraph
- 版本：main (最后更新: 2026-02-17)
- 官方文档链接：https://docs.langchain.com/oss/python/langgraph/
- 总代码片段数：900
- 信任评分：10/10
- 基准评分：86.9

## 关键信息提取

### 1. 常见反模式 (Anti-Patterns)

#### 反模式1：在 Interrupt 前追加列表

**问题描述**：
在 interrupt 之前向列表追加数据会导致每次恢复时重复添加条目。

**错误示例**：
```python
def node_a(state: State):
    # ❌ Bad: appending to a list before interrupt
    # This will add duplicate entries on each resume
    db.append_to_history(state["user_id"], "approval_requested")

    approved = interrupt("Approve this change?")

    return {"approved": approved}
```

**最佳实践**：
- ✅ 将副作用封装在 task 函数中
- ✅ 使用 @task 装饰器确保副作用只执行一次
- ✅ 在 interrupt 之后执行副作用

**正确示例**：
```python
from langgraph.func import task

@task
def write_to_file():
    with open("output.txt", "w") as f:
        f.write("Side effect executed")

@entrypoint(checkpointer=checkpointer)
def my_workflow(inputs: dict) -> int:
    # The side effect is now encapsulated in a task.
    write_to_file().result()
    value = interrupt("question")
    return value
```

#### 反模式2：非确定性工作流

**问题描述**：
使用当前时间等非确定性因素决定执行路径会导致恢复时行为不一致。

**错误示例**：
```python
from langgraph.func import entrypoint

@entrypoint(checkpointer=checkpointer)
def my_workflow(inputs: dict) -> int:
    t0 = inputs["t0"]
    t1 = time.time()  # [!code highlight] 非确定性

    delta_t = t1 - t0

    if delta_t > 1:
        result = slow_task(1).result()
        value = interrupt("question")
    else:
        result = slow_task(2).result()
        value = interrupt("question")

    return {
        "result": result,
        "value": value
    }
```

**最佳实践**：
- ✅ 避免在工作流中使用 time.time()
- ✅ 将时间戳作为输入参数传递
- ✅ 确保工作流的执行路径是确定的

#### 反模式3：抑制意外错误

**问题描述**：
捕获并抑制错误会隐藏关键问题,使调试困难。

**错误示例**：
```python
def send_reply(state: EmailAgentState):
    try:
        email_service.send(state["draft_response"])
    except Exception:
        pass  # ❌ Bad: silently ignoring errors
```

**正确示例**：
```python
def send_reply(state: EmailAgentState):
    try:
        email_service.send(state["draft_response"])
    except Exception:
        raise  # ✅ Surface unexpected errors
```

**最佳实践**：
- ✅ 让意外错误向上传播
- ✅ 只捕获预期的特定错误
- ✅ 使用 RetryPolicy 处理临时性错误

### 2. 错误处理最佳实践

#### RetryPolicy 配置

**基础配置**：
```python
from langgraph.types import RetryPolicy

# 为特定节点配置重试策略
retry_policy = RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,
    retry_on=ValueError  # 只重试 ValueError
)

workflow.add_node(
    "search_documentation",
    search_documentation,
    retry_policy=retry_policy
)
```

**数据库和模型节点的自定义重试**：
```python
import sqlite3
from langgraph.types import RetryPolicy

# 数据库查询节点：重试 sqlite3.OperationalError
builder.add_node(
    "query_database",
    query_database,
    retry_policy=RetryPolicy(retry_on=sqlite3.OperationalError),
)

# 模型调用节点：最多重试 5 次
builder.add_node(
    "model",
    call_model,
    retry_policy=RetryPolicy(max_attempts=5)
)
```

**Task 级别的重试策略**：
```python
from langgraph.func import task
from langgraph.types import RetryPolicy

# 配置 RetryPolicy 重试 ValueError
retry_policy = RetryPolicy(retry_on=ValueError)

@task(retry_policy=retry_policy)
def get_info():
    global attempts
    attempts += 1

    if attempts < 2:
        raise ValueError('Failure')
    return "OK"
```

**最佳实践**：
- ✅ 为 API 调用、数据库查询、LLM 交互配置重试策略
- ✅ 默认 retry_on 函数会重试大多数异常,但排除常见编程错误
- ✅ 对于 HTTP 请求,默认重试 5xx 状态码
- ✅ 使用 retry_on 参数自定义重试条件

### 3. 状态管理最佳实践

#### 状态定义

**TypedDict 定义**：
```python
from typing_extensions import TypedDict

class WorkflowState(TypedDict):
    user_input: str
    search_results: list
    generated_response: str
    validation_status: str
```

**节点访问和修改状态**：
```python
def search_node(state):
    # Access shared state
    results = search(state["user_input"])
    return {"search_results": results}

def validation_node(state):
    # Access results from previous node
    is_valid = validate(state["generated_response"])
    return {"validation_status": "valid" if is_valid else "invalid"}
```

**最佳实践**：
- ✅ 节点应该返回状态更新,而不是直接修改状态
- ✅ 返回部分状态更新,不需要返回完整状态
- ✅ 使用 TypedDict 或 Pydantic 定义状态类型

#### 状态更新与 Reducer

**使用 Reducer 聚合状态**：
```python
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]  # 使用 add reducer

# 初始状态: {"foo": 1, "bar": ["a"]}
graph.update_state(config, {"foo": 2, "bar": ["b"]})
# 结果: {"foo": 2, "bar": ["a", "b"]}  # bar 使用 add reducer 追加
```

**并行执行与 Reducer**：
```python
import operator
from typing import Annotated
from typing_extensions import TypedDict

class State(TypedDict):
    # operator.add reducer 使其只追加
    aggregate: Annotated[list, operator.add]

def a(state: State):
    return {"aggregate": ["A"]}

def b(state: State):
    return {"aggregate": ["B"]}

def c(state: State):
    return {"aggregate": ["C"]}

# 并行执行 b 和 c
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "d")
builder.add_edge("c", "d")

# 结果: aggregate = ["A", "B", "C"]
```

**最佳实践**：
- ✅ 使用 Annotated[type, reducer] 定义需要聚合的字段
- ✅ operator.add 适合列表追加
- ✅ 没有 reducer 的字段会被覆盖
- ✅ 有 reducer 的字段会聚合多个节点的更新

### 4. 副作用处理最佳实践

#### 封装副作用

**问题**：
副作用(写文件、发送邮件)在工作流恢复时会重复执行。

**解决方案**：
```python
from langgraph.func import task

@task
def write_to_file():
    with open("output.txt", "w") as f:
        f.write("Side effect executed")

@entrypoint(checkpointer=checkpointer)
def my_workflow(inputs: dict) -> int:
    # 副作用封装在 task 中,只执行一次
    write_to_file().result()
    value = interrupt("question")
    return value
```

**错误示例**：
```python
@entrypoint(checkpointer=checkpointer)
def my_workflow(inputs: dict) -> int:
    # ❌ 这段代码在恢复工作流时会再次执行
    with open("output.txt", "w") as f:
        f.write("Side effect executed")
    value = interrupt("question")
    return value
```

**最佳实践**：
- ✅ 将副作用封装在 @task 函数中
- ✅ 副作用应该在 interrupt 之后执行
- ✅ 使用 checkpointer 确保副作用只执行一次

### 5. 图设计最佳实践

#### 节点定义

**节点函数规范**：
```python
from langchain.messages import AIMessage

def node(state: State):
    messages = state["messages"]
    new_message = AIMessage("Hello!")
    # ✅ 返回状态更新,不要直接修改状态
    return {"messages": messages + [new_message], "extra_field": 10}
```

**最佳实践**：
- ✅ 节点函数应该返回状态更新字典
- ✅ 不要直接修改输入的 state
- ✅ 返回部分状态更新,不需要返回完整状态
- ✅ 节点函数应该是纯函数或幂等的

#### 图结构设计

**并行执行**：
```python
builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)
builder.add_node(c)
builder.add_node(d)

# 从 a 扇出到 b 和 c (并行执行)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("a", "c")

# 从 b 和 c 扇入到 d
builder.add_edge("b", "d")
builder.add_edge("c", "d")
builder.add_edge("d", END)

graph = builder.compile()
```

**最佳实践**：
- ✅ 使用扇出模式实现并行执行
- ✅ 使用 Reducer 聚合并行节点的结果
- ✅ 确保并行节点之间没有依赖关系

## 总结

从 Context7 官方文档中提取的核心最佳实践:

1. **反模式避免**: 不要在 interrupt 前执行副作用、避免非确定性工作流、不要抑制意外错误
2. **错误处理**: 使用 RetryPolicy 处理临时性错误,让意外错误向上传播
3. **状态管理**: 使用 TypedDict 定义状态,使用 Reducer 聚合并行更新,节点返回部分状态更新
4. **副作用处理**: 将副作用封装在 @task 函数中,确保只执行一次
5. **图设计**: 节点函数应该是纯函数,使用扇出/扇入模式实现并行执行
