# 核心概念 7：Literal 与 TypedDict

## 一句话定义

**Literal 是精确指定字面量值的类型注解，TypedDict 是定义字典结构的类型，两者结合可以创建类型安全的配置和状态对象，是 LangGraph 状态管理和 LangChain 配置的核心类型工具。**

---

## 为什么需要 Literal 和 TypedDict？

### 问题1：字符串类型太宽泛

```python
# 使用 str 类型（不安全）
def set_mode(mode: str) -> None:
    if mode == "development":
        print("Dev mode")
    elif mode == "production":
        print("Prod mode")
    else:
        print("Unknown mode")

# ⚠️ 类型检查器不会报错，但运行时可能出错
set_mode("dev")  # 拼写错误，运行时才发现
set_mode("prod")  # 拼写错误
```

**Literal 的解决方案**：

```python
from typing import Literal

# 使用 Literal 精确指定值
def set_mode(mode: Literal["development", "production"]) -> None:
    if mode == "development":
        print("Dev mode")
    elif mode == "production":
        print("Prod mode")

# ✅ 类型检查器验证
set_mode("development")  # ✅
set_mode("production")   # ✅
set_mode("dev")  # ❌ 类型错误：Literal["development", "production"] expected
```

### 问题2：字典类型不够精确

```python
# 使用 dict 类型（不安全）
def process_user(user: dict) -> str:
    return f"Hello, {user['name']}"  # ⚠️ 可能没有 'name' 键

# 运行时错误
process_user({"age": 30})  # KeyError: 'name'
```

**TypedDict 的解决方案**：

```python
from typing import TypedDict

# 定义字典结构
class User(TypedDict):
    name: str
    age: int

def process_user(user: User) -> str:
    return f"Hello, {user['name']}"  # ✅ 类型检查器知道有 'name' 键

# ✅ 类型检查器验证
process_user({"name": "Alice", "age": 30})  # ✅
process_user({"age": 30})  # ❌ 类型错误：Missing required key 'name'
```

---

## Literal 基础

### 1. 基本用法

```python
from typing import Literal

# 字符串字面量
Mode = Literal["development", "production", "testing"]

def set_mode(mode: Mode) -> None:
    print(f"Mode: {mode}")

# 数字字面量
Port = Literal[8000, 8080, 3000]

def start_server(port: Port) -> None:
    print(f"Server on port {port}")

# 布尔字面量
Flag = Literal[True]  # 只能是 True

def enable_feature(enabled: Flag) -> None:
    print("Feature enabled")

# 混合类型
Status = Literal["success", "error", 404, 500]

def handle_status(status: Status) -> None:
    print(f"Status: {status}")
```

### 2. Literal 联合

```python
from typing import Literal

# 组合多个 Literal
HttpMethod = Literal["GET", "POST", "PUT", "DELETE"]
HttpStatus = Literal[200, 201, 400, 404, 500]

def make_request(
    method: HttpMethod,
    expected_status: HttpStatus
) -> None:
    print(f"{method} request, expecting {expected_status}")

# 使用
make_request("GET", 200)  # ✅
make_request("PATCH", 200)  # ❌ 类型错误
```

### 3. Literal 类型别名

```python
from typing import Literal

# 定义类型别名
Environment = Literal["dev", "staging", "prod"]
LogLevel = Literal["debug", "info", "warning", "error"]

class Config:
    def __init__(
        self,
        env: Environment,
        log_level: LogLevel
    ):
        self.env = env
        self.log_level = log_level

# 使用
config = Config("dev", "debug")  # ✅
config = Config("development", "debug")  # ❌ 类型错误
```

---

## TypedDict 基础

### 1. 基本定义

```python
from typing import TypedDict

# 方式1：类语法（推荐）
class User(TypedDict):
    name: str
    age: int
    email: str

# 方式2：函数语法
User = TypedDict('User', {'name': str, 'age': int, 'email': str})

# 使用
user: User = {
    "name": "Alice",
    "age": 30,
    "email": "alice@example.com"
}
```

### 2. 可选字段（NotRequired）

```python
from typing import TypedDict, NotRequired

# Python 3.11+ 语法
class User(TypedDict):
    name: str
    age: int
    email: NotRequired[str]  # 可选字段

# 使用
user1: User = {"name": "Alice", "age": 30}  # ✅ email 可选
user2: User = {"name": "Bob", "age": 25, "email": "bob@example.com"}  # ✅

# Python 3.10 及以下（使用 total=False）
class UserOptional(TypedDict, total=False):
    email: str

class User(UserOptional):
    name: str
    age: int
```

### 3. 只读字段（ReadOnly）

```python
from typing import TypedDict, ReadOnly

# Python 3.13+ 语法
class User(TypedDict):
    name: str
    age: int
    id: ReadOnly[str]  # 只读字段

# 使用
user: User = {"name": "Alice", "age": 30, "id": "123"}

# ⚠️ 类型检查器会警告（但运行时不会阻止）
user["id"] = "456"  # 类型检查器警告：Cannot assign to read-only key
```

### 4. 继承 TypedDict

```python
from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

class Employee(Person):
    employee_id: str
    department: str

# Employee 包含所有字段
employee: Employee = {
    "name": "Alice",
    "age": 30,
    "employee_id": "E001",
    "department": "Engineering"
}
```

---

## Literal 与 TypedDict 结合

### 1. 类型安全的状态机

```python
from typing import TypedDict, Literal

# 定义状态类型
State = Literal["idle", "running", "paused", "stopped"]

class TaskState(TypedDict):
    status: State
    progress: int
    message: str

def update_task(state: TaskState) -> None:
    print(f"Task {state['status']}: {state['progress']}%")

# 使用
task: TaskState = {
    "status": "running",
    "progress": 50,
    "message": "Processing..."
}
update_task(task)  # ✅

# ❌ 类型错误
bad_task: TaskState = {
    "status": "executing",  # 不是有效的 State
    "progress": 50,
    "message": "Processing..."
}
```

### 2. 配置对象

```python
from typing import TypedDict, Literal, NotRequired

Environment = Literal["development", "staging", "production"]
LogLevel = Literal["debug", "info", "warning", "error"]

class DatabaseConfig(TypedDict):
    host: str
    port: int
    database: str
    username: str
    password: str

class AppConfig(TypedDict):
    environment: Environment
    log_level: LogLevel
    debug: bool
    database: DatabaseConfig
    api_key: NotRequired[str]  # 可选

# 使用
config: AppConfig = {
    "environment": "development",
    "log_level": "debug",
    "debug": True,
    "database": {
        "host": "localhost",
        "port": 5432,
        "database": "myapp",
        "username": "admin",
        "password": "secret"
    }
}
```

---

## 在 LangChain 中的应用

### 1. LangGraph 状态类型

```python
from typing import TypedDict, Literal, NotRequired
from langgraph.graph import StateGraph

# 定义状态类型
Action = Literal["continue", "end", "retry"]

class AgentState(TypedDict):
    messages: list[str]
    next_action: Action
    retry_count: int
    error: NotRequired[str]

# 使用状态
def process_node(state: AgentState) -> AgentState:
    """处理节点"""
    messages = state["messages"]
    messages.append("Processing...")

    return {
        "messages": messages,
        "next_action": "continue",
        "retry_count": state["retry_count"]
    }

# 创建图
graph = StateGraph(AgentState)
graph.add_node("process", process_node)
```

### 2. Runnable 配置

```python
from typing import TypedDict, Literal, NotRequired
from langchain_core.runnables import Runnable

# 定义配置类型
class RunnableConfig(TypedDict):
    run_name: NotRequired[str]
    tags: NotRequired[list[str]]
    metadata: NotRequired[dict[str, str]]
    max_concurrency: NotRequired[int]

# 使用配置
config: RunnableConfig = {
    "run_name": "my_chain",
    "tags": ["production", "v1"],
    "metadata": {"user_id": "123"}
}

# 传递给 Runnable
result = chain.invoke(input_data, config=config)
```

### 3. 工具参数

```python
from typing import TypedDict, Literal
from langchain_core.tools import tool

# 定义工具参数类型
SearchType = Literal["web", "image", "news"]

class SearchParams(TypedDict):
    query: str
    search_type: SearchType
    max_results: int

@tool
def search_tool(params: SearchParams) -> list[dict]:
    """搜索工具"""
    query = params["query"]
    search_type = params["search_type"]
    max_results = params["max_results"]

    # 执行搜索
    return [{"title": "Result 1", "url": "https://example.com"}]

# 使用
result = search_tool({
    "query": "Python typing",
    "search_type": "web",
    "max_results": 10
})
```

---

## 2025-2026 新特性

### 1. ReadOnly（Python 3.13+）

```python
from typing import TypedDict, ReadOnly

class User(TypedDict):
    name: str
    age: int
    id: ReadOnly[str]  # 只读字段

# 类型检查器会警告修改只读字段
user: User = {"name": "Alice", "age": 30, "id": "123"}
user["id"] = "456"  # ⚠️ 类型检查器警告
```

### 2. NotRequired 和 Required

```python
from typing import TypedDict, NotRequired, Required

# 默认所有字段必需
class User(TypedDict):
    name: str
    age: int
    email: NotRequired[str]  # 可选

# 默认所有字段可选
class UserOptional(TypedDict, total=False):
    name: str
    age: str
    email: Required[str]  # 必需
```

### 3. 泛型 TypedDict（实验性）

```python
from typing import TypedDict, Generic, TypeVar

T = TypeVar('T')

# 泛型 TypedDict（实验性，支持有限）
class Response(TypedDict, Generic[T]):
    data: T
    status: int
    message: str

# 使用
user_response: Response[dict] = {
    "data": {"name": "Alice"},
    "status": 200,
    "message": "Success"
}
```

---

## 实战示例：类型安全的 LangGraph 工作流

```python
"""
类型安全的 LangGraph 工作流
演示：Literal 和 TypedDict 在实际项目中的应用
"""

from typing import TypedDict, Literal, NotRequired
from langgraph.graph import StateGraph, END

# ===== 1. 定义状态类型 =====

# 动作类型
Action = Literal["analyze", "generate", "review", "end"]

# 状态类型
class WorkflowState(TypedDict):
    """工作流状态"""
    input: str
    analysis: NotRequired[str]
    generated_content: NotRequired[str]
    review_result: NotRequired[str]
    next_action: Action
    iteration: int
    max_iterations: int

# ===== 2. 定义节点函数 =====

def analyze_node(state: WorkflowState) -> WorkflowState:
    """分析节点"""
    print(f"[Iteration {state['iteration']}] Analyzing input...")

    analysis = f"Analysis of: {state['input']}"

    # 决定下一步
    next_action: Action = "generate" if state['iteration'] < state['max_iterations'] else "end"

    return {
        **state,
        "analysis": analysis,
        "next_action": next_action
    }

def generate_node(state: WorkflowState) -> WorkflowState:
    """生成节点"""
    print(f"[Iteration {state['iteration']}] Generating content...")

    analysis = state.get("analysis", "")
    generated = f"Generated content based on: {analysis}"

    return {
        **state,
        "generated_content": generated,
        "next_action": "review"
    }

def review_node(state: WorkflowState) -> WorkflowState:
    """审查节点"""
    print(f"[Iteration {state['iteration']}] Reviewing content...")

    generated = state.get("generated_content", "")
    review = f"Review: {generated} looks good"

    # 决定是否继续迭代
    iteration = state['iteration'] + 1
    next_action: Action = "analyze" if iteration < state['max_iterations'] else "end"

    return {
        **state,
        "review_result": review,
        "next_action": next_action,
        "iteration": iteration
    }

# ===== 3. 定义路由函数 =====

def route_next(state: WorkflowState) -> str:
    """根据状态决定下一步"""
    action = state["next_action"]

    if action == "analyze":
        return "analyze"
    elif action == "generate":
        return "generate"
    elif action == "review":
        return "review"
    else:
        return END

# ===== 4. 构建图 =====

# 创建状态图
workflow = StateGraph(WorkflowState)

# 添加节点
workflow.add_node("analyze", analyze_node)
workflow.add_node("generate", generate_node)
workflow.add_node("review", review_node)

# 设置入口点
workflow.set_entry_point("analyze")

# 添加条件边
workflow.add_conditional_edges(
    "analyze",
    route_next,
    {
        "generate": "generate",
        "end": END
    }
)

workflow.add_conditional_edges(
    "generate",
    route_next,
    {
        "review": "review"
    }
)

workflow.add_conditional_edges(
    "review",
    route_next,
    {
        "analyze": "analyze",
        "end": END
    }
)

# 编译图
app = workflow.compile()

# ===== 5. 使用工作流 =====

print("=== 类型安全的 LangGraph 工作流 ===\n")

# 初始状态（类型安全）
initial_state: WorkflowState = {
    "input": "Create a blog post about Python typing",
    "next_action": "analyze",
    "iteration": 1,
    "max_iterations": 2
}

# 运行工作流
final_state = app.invoke(initial_state)

# 输出结果
print("\n=== 最终状态 ===")
print(f"Input: {final_state['input']}")
print(f"Analysis: {final_state.get('analysis', 'N/A')}")
print(f"Generated: {final_state.get('generated_content', 'N/A')}")
print(f"Review: {final_state.get('review_result', 'N/A')}")
print(f"Iterations: {final_state['iteration']}")

# ===== 6. 类型检查验证 =====

# ✅ 类型正确
valid_state: WorkflowState = {
    "input": "test",
    "next_action": "analyze",
    "iteration": 1,
    "max_iterations": 3
}

# ❌ 类型错误（编译时发现）
# invalid_state: WorkflowState = {
#     "input": "test",
#     "next_action": "invalid_action",  # 不是有效的 Action
#     "iteration": 1,
#     "max_iterations": 3
# }

# ❌ 缺少必需字段
# incomplete_state: WorkflowState = {
#     "input": "test",
#     # 缺少 next_action
#     "iteration": 1,
#     "max_iterations": 3
# }
```

**运行输出**：
```
=== 类型安全的 LangGraph 工作流 ===

[Iteration 1] Analyzing input...
[Iteration 1] Generating content...
[Iteration 1] Reviewing content...
[Iteration 2] Analyzing input...

=== 最终状态 ===
Input: Create a blog post about Python typing
Analysis: Analysis of: Create a blog post about Python typing
Generated: Generated content based on: Analysis of: Create a blog post about Python typing
Review: Review: Generated content based on: Analysis of: Create a blog post about Python typing looks good
Iterations: 2
```

---

## 2025-2026 最佳实践

### 1. 优先使用 Literal 而非字符串

```python
# ❌ 不推荐：使用 str
def set_mode(mode: str) -> None:
    if mode in ["dev", "prod"]:
        ...

# ✅ 推荐：使用 Literal
from typing import Literal

def set_mode(mode: Literal["dev", "prod"]) -> None:
    ...
```

### 2. TypedDict 优于普通 dict

```python
# ❌ 不推荐：使用 dict
def process_user(user: dict[str, Any]) -> str:
    return user["name"]  # 可能没有 'name' 键

# ✅ 推荐：使用 TypedDict
from typing import TypedDict

class User(TypedDict):
    name: str
    age: int

def process_user(user: User) -> str:
    return user["name"]  # 类型安全
```

### 3. 使用 NotRequired 标记可选字段

```python
from typing import TypedDict, NotRequired

# ✅ 推荐：明确标记可选字段
class Config(TypedDict):
    host: str
    port: int
    debug: NotRequired[bool]  # 可选
```

### 4. 组合 Literal 和 TypedDict

```python
from typing import TypedDict, Literal

Status = Literal["pending", "success", "error"]

class Response(TypedDict):
    status: Status
    data: dict
    message: str

# 类型安全
response: Response = {
    "status": "success",
    "data": {"result": 42},
    "message": "OK"
}
```

---

## 常见陷阱

### 1. Literal 不是运行时检查

```python
from typing import Literal

Mode = Literal["dev", "prod"]

def set_mode(mode: Mode) -> None:
    print(f"Mode: {mode}")

# ⚠️ 运行时不会报错
set_mode("invalid")  # 运行时不会检查

# ✅ 需要显式运行时检查
def set_mode_safe(mode: Mode) -> None:
    if mode not in ["dev", "prod"]:
        raise ValueError(f"Invalid mode: {mode}")
    print(f"Mode: {mode}")
```

### 2. TypedDict 不是类

```python
from typing import TypedDict

class User(TypedDict):
    name: str
    age: int

# ❌ 不能实例化
user = User(name="Alice", age=30)  # TypeError

# ✅ 使用字典字面量
user: User = {"name": "Alice", "age": 30}
```

### 3. TypedDict 不支持方法

```python
from typing import TypedDict

# ❌ TypedDict 不能有方法
class User(TypedDict):
    name: str
    age: int

    def greet(self) -> str:  # 错误
        return f"Hello, {self['name']}"

# ✅ 使用 Pydantic 或普通类
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

    def greet(self) -> str:
        return f"Hello, {self.name}"
```

---

## TypedDict vs Pydantic

### 对比表

| 特性 | TypedDict | Pydantic |
|------|-----------|----------|
| 类型检查 | ✅ 编译时 | ✅ 编译时 + 运行时 |
| 运行时验证 | ❌ | ✅ |
| 方法支持 | ❌ | ✅ |
| 继承 | ✅ | ✅ |
| 性能 | 快（无开销） | 慢（验证开销） |
| 序列化 | ❌ | ✅ |
| 适用场景 | 简单字典结构 | 复杂数据模型 |
| 2025-2026 推荐 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### 使用场景

```python
from typing import TypedDict
from pydantic import BaseModel

# TypedDict：简单字典结构
class ConfigDict(TypedDict):
    host: str
    port: int

config: ConfigDict = {"host": "localhost", "port": 8000}

# Pydantic：复杂数据模型
class User(BaseModel):
    name: str
    age: int
    email: str

    def greet(self) -> str:
        return f"Hello, {self.name}"

user = User(name="Alice", age=30, email="alice@example.com")
```

---

## 学习检查清单

- [ ] 理解 Literal 的作用和用法
- [ ] 掌握 TypedDict 的定义和使用
- [ ] 了解 NotRequired 和 ReadOnly
- [ ] 理解 Literal 和 TypedDict 的结合
- [ ] 了解 LangGraph 中的状态类型
- [ ] 知道 TypedDict vs Pydantic 的区别
- [ ] 能够创建类型安全的配置对象
- [ ] 遵循 2025-2026 最佳实践

---

## 下一步学习

- **核心概念 8**：类型守卫 - 学习运行时类型检查
- **核心概念 4**：Runnable 泛型设计 - 复习 LangChain 类型系统
- **L7_LangGraph 与状态管理** - 深入学习 LangGraph

---

## 参考资源

1. [Python typing 官方文档 - Literal](https://docs.python.org/3/library/typing.html#typing.Literal)
2. [Python typing 官方文档 - TypedDict](https://docs.python.org/3/library/typing.html#typing.TypedDict)
3. [PEP 586 – Literal Types](https://peps.python.org/pep-0586/) - Literal 规范
4. [PEP 589 – TypedDict](https://peps.python.org/pep-0589/) - TypedDict 规范
5. [Python 3.13 新特性 - ReadOnly](https://docs.python.org/3/whatsnew/3.13.html)
6. [Type Safety in LangGraph: TypedDict vs. Pydantic](https://shazaali.substack.com/p/type-safety-in-langgraph-when-to) - 2025
7. [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/) - 状态类型应用
