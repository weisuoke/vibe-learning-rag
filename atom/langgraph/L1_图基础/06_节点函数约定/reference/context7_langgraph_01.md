---
type: context7_documentation
library: langgraph
version: main
fetched_at: 2026-02-25
knowledge_point: 节点函数约定
context7_query: node function definition parameters return value async error handling
---

# Context7 文档：LangGraph 节点函数

## 文档来源
- 库名称：LangGraph
- 版本：main
- 官方文档链接：https://github.com/langchain-ai/langgraph

## 关键信息提取

### 1. 节点函数基础定义

**节点函数签名**：
```python
def node_function(state: dict) -> dict:
    """
    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updated state with new/modified keys
    """
    return {"key": "value"}
```

**来源**：LangGraph RAG 示例

### 2. 节点函数返回值约定

**部分状态更新**：
- 节点函数只需返回需要更新的状态字段
- 不需要返回完整的状态对象
- 返回的字段会与现有状态合并

**示例**：
```python
def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
```

### 3. 条件路由函数

**路由函数签名**：
```python
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations == max_iterations:
        print("---DECISION: FINISH---")
        return "end"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        return "generate"
```

**关键点**：
- 返回字符串表示下一个节点名称
- 用于 `add_conditional_edges` 的路由函数

### 4. 错误处理节点

**错误检查模式**：
```python
def code_check(state: GraphState):
    """
    Check code

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """
    print("---CHECKING CODE---")

    # State
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]

    # Get solution components
    imports = code_solution.imports
    code = code_solution.code

    # Check imports
    try:
        exec(imports)
    except Exception as e:
        print("---CODE IMPORT CHECK: FAILED---")
        error_message = [
            (
                "user",
                f"Your solution failed the import test. Here is the error: {e}. Reflect on this error and your prior attempt to solve the problem. (1) State what you think went wrong with the prior solution and (2) try to solve this problem again. Return the FULL SOLUTION. Use the code tool to structure the output with a prefix, imports, and code block:",
            )
        ]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes"
        }

    # Check execution
    try:
        combined_code = f"{imports}\n{code}"
        print(f"CODE TO TEST: {combined_code}")
        # Use a shared scope for exec
        global_scope = {}
        exec(combined_code, global_scope)
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        error_message = [
            (
                "user",
                f"Your solution failed the code execution test: {e}) Reflect on this error and your prior attempt to solve the problem. (1) State what you think went wrong with the prior solution and (2) try to solve this problem again. Return the FULL SOLUTION. Use the code tool to structure the output with a prefix, imports, and code block:",
            )
        ]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes"
        }

    # No errors
    print("---NO CODE TEST FAILURES---")
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no"
    }
```

**关键点**：
- 使用 try-except 捕获异常
- 在状态中添加错误标志
- 返回错误信息供后续节点处理

### 5. RetryPolicy 配置

**重试策略配置**：
```python
from langgraph.graph import START, END, StateGraph
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict
import random

class State(TypedDict):
    result: str
    attempts: int

# Custom retry condition
def retry_on_rate_limit(error: Exception) -> bool:
    return "rate limit" in str(error).lower()

def unreliable_api_call(state: State) -> dict:
    """Simulate an unreliable API that sometimes fails."""
    attempts = state.get("attempts", 0) + 1
    if random.random() < 0.5:  # 50% chance of failure
        raise Exception("Rate limit exceeded")
    return {"result": "Success!", "attempts": attempts}

builder = StateGraph(State)
builder.add_node(
    "api_call",
    unreliable_api_call,
    retry_policy=RetryPolicy(
        initial_interval=0.5,      # Start with 0.5 second delay
        backoff_factor=2.0,        # Double delay each retry
        max_interval=10.0,         # Max 10 seconds between retries
        max_attempts=5,            # Try up to 5 times
        jitter=True,               # Add randomness to prevent thundering herd
        retry_on=retry_on_rate_limit  # Custom retry condition
    )
)
builder.add_edge(START, "api_call")
builder.add_edge("api_call", END)

graph = builder.compile()
result = graph.invoke({"result": "", "attempts": 0})
print(f"Result: {result['result']}, Attempts: {result['attempts']}")
```

**RetryPolicy 参数**：
- `initial_interval`: 初始重试间隔（秒）
- `backoff_factor`: 退避因子（指数退避）
- `max_interval`: 最大重试间隔（秒）
- `max_attempts`: 最大重试次数
- `jitter`: 是否添加随机抖动
- `retry_on`: 重试条件（异常类型或自定义函数）

### 6. Human-in-the-loop 模式

**interrupt 函数使用**：
```python
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

class FormState(TypedDict):
    name: str
    email: str
    confirmed: bool

def collect_info(state: FormState) -> dict:
    """Collect user information with interrupts."""
    name = interrupt("Please enter your name:")
    email = interrupt("Please enter your email:")
    return {"name": name, "email": email}

def confirm_info(state: FormState) -> dict:
    """Confirm collected information."""
    confirmed = interrupt(
        f"Confirm details - Name: {state['name']}, Email: {state['email']} (yes/no)"
    )
    return {"confirmed": confirmed == "yes"}

builder = StateGraph(FormState)
builder.add_node("collect", collect_info)
builder.add_node("confirm", confirm_info)
builder.add_edge(START, "collect")
builder.add_edge("collect", "confirm")
builder.add_edge("confirm", END)
```

**关键点**：
- `interrupt()` 函数暂停图执行
- 需要 checkpointer 保存状态
- 使用 `Command(resume=...)` 恢复执行

## 总结

LangGraph 节点函数的核心约定：

1. **输入**：接收 state 字典
2. **输出**：返回部分状态更新（字典）或路由决策（字符串）
3. **错误处理**：使用 try-except + RetryPolicy
4. **异步支持**：自动检测 async 函数
5. **人机交互**：使用 interrupt 函数暂停执行
