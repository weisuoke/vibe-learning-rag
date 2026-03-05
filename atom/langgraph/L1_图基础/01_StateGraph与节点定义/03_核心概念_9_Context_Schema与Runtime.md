# 核心概念 9：Context Schema 与 Runtime

> **来源**：源码分析 (state.py:141-250, _node.py:194-197) + Context7 官方文档

---

## 概念定义

**Context Schema 与 Runtime 是 LangGraph 提供的运行时上下文注入机制，允许节点访问不可变的配置数据和运行时信息。**

---

## 详细解释

### 什么是 Context Schema？

**Context Schema** 是在创建 StateGraph 时定义的**不可变上下文数据结构**，用于存储运行时配置、环境变量、全局参数等不需要在节点间传递和修改的数据。

**关键特征**：
- 使用 TypedDict 定义结构
- 在图编译时固定，运行时不可修改
- 所有节点共享相同的上下文
- 通过 `Runtime[ContextT]` 注入到节点函数

**与 State 的区别**：

| 特性 | State | Context |
|------|-------|---------|
| 可变性 | 可变（节点可以更新） | 不可变（只读） |
| 传递方式 | 节点间传递 | 运行时注入 |
| 用途 | 工作流数据 | 配置和环境 |
| 定义位置 | `state_schema` | `context_schema` |

### 什么是 Runtime？

**Runtime[ContextT]** 是 LangGraph 提供的运行时对象，封装了上下文数据和运行时信息，通过依赖注入传递给节点函数。

**Runtime 对象包含**：
- `context`: 上下文数据（ContextT 类型）
- `config`: RunnableConfig 配置
- `store`: BaseStore 存储接口（可选）
- `writer`: StreamWriter 流写入器（可选）

---

## 源码分析

### 1. StateGraph 构造函数中的 context_schema

**源码位置**：`state.py:197-250`

```python
class StateGraph(Generic[StateT, ContextT, InputT, OutputT]):
    def __init__(
        self,
        state_schema: type[StateT],
        context_schema: type[ContextT] | None = None,  # 可选的上下文 Schema
        *,
        input_schema: type[InputT] | None = None,
        output_schema: type[OutputT] | None = None,
        **kwargs: Unpack[DeprecatedKwargs],
    ) -> None:
        # 初始化内部数据结构
        self.nodes = {}
        self.edges = set()
        self.branches = defaultdict(dict)
        self.schemas = {}
        self.channels = {}
        self.managed = {}
        self.compiled = False
        self.waiting_edges = set()

        # 设置 schema
        self.state_schema = state_schema
        self.input_schema = cast(type[InputT], input_schema or state_schema)
        self.output_schema = cast(type[OutputT], output_schema or state_schema)
        self.context_schema = context_schema  # 保存上下文 Schema

        # 添加 schema 到内部字典
        self._add_schema(self.state_schema)
        self._add_schema(self.input_schema, allow_managed=False)
        self._add_schema(self.output_schema, allow_managed=False)
```

**关键点**：
- `context_schema` 是可选参数（默认 None）
- 与 `state_schema` 分离，职责明确
- 不会被添加到 channels（因为不参与状态传递）

### 2. 节点函数协议：_NodeWithRuntime

**源码位置**：`_node.py:194-197`

```python
class _NodeWithRuntime(Protocol[NodeInputT_contra, ContextT]):
    """节点函数协议：接受 state 和 runtime 参数"""
    def __call__(
        self, state: NodeInputT_contra, *, runtime: Runtime[ContextT]
    ) -> Any: ...
```

**关键点**：
- `runtime` 是关键字参数（keyword-only）
- 泛型 `ContextT` 与 StateGraph 的 `ContextT` 对应
- 返回值可以是任意类型（通常是部分状态更新）

### 3. Runtime 对象的使用

**源码示例**：`state.py:141-180`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

# 定义 State
class State(TypedDict):
    x: list

# 定义 Context
class Context(TypedDict):
    r: float  # 运行时参数

# 创建图，指定 context_schema
graph = StateGraph(state_schema=State, context_schema=Context)

# 节点函数：通过 runtime 访问上下文
def node(state: State, *, runtime: Runtime[Context]) -> dict:
    # 从 runtime.context 获取上下文数据
    r = runtime.context.get("r", 1.0)
    x = state["x"][-1]
    next_value = x * r * (1 - x)
    return {"x": next_value}

# 添加节点
graph.add_node("A", node)
graph.set_entry_point("A")
graph.set_finish_point("A")

# 编译图
compiled = graph.compile()

# 执行时传入 context 参数
step1 = compiled.invoke({"x": [0.5]}, context={"r": 3.0})
# 输出: {'x': [0.5, 0.75]}
```

---

## 代码示例

### 示例 1：基础 Context 使用

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime

# ===== 1. 定义 State 和 Context =====
class State(TypedDict):
    message: str
    count: int

class Context(TypedDict):
    max_count: int  # 最大计数限制
    prefix: str     # 消息前缀

# ===== 2. 创建图 =====
graph = StateGraph(state_schema=State, context_schema=Context)

# ===== 3. 定义节点函数 =====
def process_node(state: State, *, runtime: Runtime[Context]) -> dict:
    """处理节点：使用上下文中的配置"""
    # 从 runtime.context 获取配置
    max_count = runtime.context.get("max_count", 10)
    prefix = runtime.context.get("prefix", "")

    # 处理逻辑
    new_count = state["count"] + 1
    new_message = f"{prefix} {state['message']} (count: {new_count})"

    return {
        "message": new_message,
        "count": new_count
    }

def check_node(state: State, *, runtime: Runtime[Context]) -> dict:
    """检查节点：根据上下文判断是否继续"""
    max_count = runtime.context.get("max_count", 10)

    if state["count"] >= max_count:
        return {"message": f"{state['message']} - DONE"}
    return {}

# ===== 4. 构建图 =====
graph.add_node("process", process_node)
graph.add_node("check", check_node)
graph.add_edge(START, "process")
graph.add_edge("process", "check")
graph.add_edge("check", END)

# ===== 5. 编译并执行 =====
compiled = graph.compile()

# 执行时传入 context
result = compiled.invoke(
    {"message": "Hello", "count": 0},
    context={"max_count": 3, "prefix": "[INFO]"}
)

print(result)
# 输出: {'message': '[INFO] Hello (count: 1) - DONE', 'count': 3}
```

### 示例 2：多环境配置

```python
from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime

# ===== 1. 定义环境配置 Context =====
class EnvContext(TypedDict):
    environment: Literal["dev", "staging", "prod"]
    api_endpoint: str
    timeout: int
    retry_count: int

# ===== 2. 定义 State =====
class APIState(TypedDict):
    request: str
    response: str | None
    error: str | None

# ===== 3. 创建图 =====
graph = StateGraph(state_schema=APIState, context_schema=EnvContext)

# ===== 4. 定义节点函数 =====
def call_api_node(state: APIState, *, runtime: Runtime[EnvContext]) -> dict:
    """调用 API：根据环境使用不同配置"""
    env = runtime.context.get("environment", "dev")
    endpoint = runtime.context.get("api_endpoint", "http://localhost")
    timeout = runtime.context.get("timeout", 30)

    print(f"[{env}] Calling API: {endpoint}")
    print(f"Timeout: {timeout}s")

    # 模拟 API 调用
    try:
        # 实际应用中这里会调用真实 API
        response = f"Response from {endpoint}"
        return {"response": response, "error": None}
    except Exception as e:
        return {"response": None, "error": str(e)}

def retry_node(state: APIState, *, runtime: Runtime[EnvContext]) -> dict:
    """重试节点：根据环境配置重试次数"""
    retry_count = runtime.context.get("retry_count", 3)

    if state["error"] and retry_count > 0:
        print(f"Retrying... (remaining: {retry_count})")
        # 实际应用中这里会重新调用 API
        return {"error": None}
    return {}

# ===== 5. 构建图 =====
graph.add_node("call_api", call_api_node)
graph.add_node("retry", retry_node)
graph.add_edge(START, "call_api")
graph.add_edge("call_api", "retry")
graph.add_edge("retry", END)

# ===== 6. 编译图 =====
compiled = graph.compile()

# ===== 7. 不同环境执行 =====

# 开发环境
dev_result = compiled.invoke(
    {"request": "GET /users", "response": None, "error": None},
    context={
        "environment": "dev",
        "api_endpoint": "http://localhost:8000",
        "timeout": 10,
        "retry_count": 1
    }
)

# 生产环境
prod_result = compiled.invoke(
    {"request": "GET /users", "response": None, "error": None},
    context={
        "environment": "prod",
        "api_endpoint": "https://api.example.com",
        "timeout": 30,
        "retry_count": 5
    }
)
```

### 示例 3：LLM 配置注入

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langchain_openai import ChatOpenAI

# ===== 1. 定义 LLM 配置 Context =====
class LLMContext(TypedDict):
    model_name: str
    temperature: float
    max_tokens: int
    api_key: str

# ===== 2. 定义 State =====
class ChatState(TypedDict):
    messages: list[str]
    response: str | None

# ===== 3. 创建图 =====
graph = StateGraph(state_schema=ChatState, context_schema=LLMContext)

# ===== 4. 定义节点函数 =====
def llm_node(state: ChatState, *, runtime: Runtime[LLMContext]) -> dict:
    """LLM 节点：使用上下文中的 LLM 配置"""
    # 从 runtime.context 获取 LLM 配置
    model_name = runtime.context.get("model_name", "gpt-3.5-turbo")
    temperature = runtime.context.get("temperature", 0.7)
    max_tokens = runtime.context.get("max_tokens", 1000)

    # 创建 LLM 实例
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # 调用 LLM
    messages = state["messages"]
    response = llm.invoke(messages[-1])

    return {"response": response.content}

# ===== 5. 构建图 =====
graph.add_node("llm", llm_node)
graph.add_edge(START, "llm")
graph.add_edge("llm", END)

# ===== 6. 编译并执行 =====
compiled = graph.compile()

# 使用不同的 LLM 配置
result = compiled.invoke(
    {"messages": ["What is LangGraph?"], "response": None},
    context={
        "model_name": "gpt-4",
        "temperature": 0.3,
        "max_tokens": 500,
        "api_key": "sk-..."
    }
)
```

---

## 在实际应用中的使用

### 使用场景 1：环境配置管理

**问题**：不同环境（开发、测试、生产）需要不同的配置，但不希望在 State 中传递这些配置。

**解决方案**：使用 Context Schema 存储环境配置。

```python
class EnvContext(TypedDict):
    environment: str
    database_url: str
    api_keys: dict[str, str]
    feature_flags: dict[str, bool]

# 节点函数可以根据环境做不同处理
def node(state: State, *, runtime: Runtime[EnvContext]) -> dict:
    env = runtime.context.get("environment")
    if env == "prod":
        # 生产环境逻辑
        pass
    else:
        # 开发环境逻辑
        pass
```

### 使用场景 2：全局参数注入

**问题**：多个节点需要访问相同的全局参数（如 LLM 配置、超时设置），但不希望在每个节点间传递。

**解决方案**：使用 Context Schema 存储全局参数。

```python
class GlobalContext(TypedDict):
    llm_model: str
    llm_temperature: float
    timeout: int
    max_retries: int

# 所有节点都可以访问这些全局参数
def node1(state: State, *, runtime: Runtime[GlobalContext]) -> dict:
    model = runtime.context.get("llm_model")
    # 使用 model

def node2(state: State, *, runtime: Runtime[GlobalContext]) -> dict:
    timeout = runtime.context.get("timeout")
    # 使用 timeout
```

### 使用场景 3：用户会话信息

**问题**：需要在整个工作流中访问用户信息（如用户 ID、权限），但这些信息不应该被节点修改。

**解决方案**：使用 Context Schema 存储用户会话信息。

```python
class UserContext(TypedDict):
    user_id: str
    username: str
    permissions: list[str]
    session_id: str

# 节点可以根据用户权限做不同处理
def node(state: State, *, runtime: Runtime[UserContext]) -> dict:
    user_id = runtime.context.get("user_id")
    permissions = runtime.context.get("permissions", [])

    if "admin" in permissions:
        # 管理员逻辑
        pass
    else:
        # 普通用户逻辑
        pass
```

### 使用场景 4：A/B 测试配置

**问题**：需要在不同用户群体中测试不同的算法或参数，但不希望修改代码。

**解决方案**：使用 Context Schema 存储实验配置。

```python
class ExperimentContext(TypedDict):
    experiment_id: str
    variant: str  # "A" or "B"
    parameters: dict[str, any]

# 节点根据实验变体执行不同逻辑
def node(state: State, *, runtime: Runtime[ExperimentContext]) -> dict:
    variant = runtime.context.get("variant", "A")

    if variant == "A":
        # 算法 A
        result = algorithm_a(state)
    else:
        # 算法 B
        result = algorithm_b(state)

    return result
```

---

## 最佳实践

### 1. Context 应该存储什么？

**✅ 适合放在 Context 中**：
- 环境配置（dev/staging/prod）
- API 密钥和凭证
- 全局参数（超时、重试次数）
- LLM 配置（模型名称、温度）
- 用户会话信息（用户 ID、权限）
- 实验配置（A/B 测试变体）
- 不可变的元数据

**❌ 不适合放在 Context 中**：
- 需要在节点间传递和修改的数据
- 工作流的中间结果
- 动态变化的状态
- 大量数据（应该放在 State 或外部存储）

### 2. Context 的命名规范

```python
# ✅ 好的命名
class AppContext(TypedDict):
    environment: str
    api_endpoint: str

class LLMContext(TypedDict):
    model_name: str
    temperature: float

# ❌ 不好的命名
class Context(TypedDict):  # 太泛化
    data: dict  # 不明确

class Config(TypedDict):  # 容易与 RunnableConfig 混淆
    settings: dict
```

### 3. 类型安全

```python
# ✅ 使用 TypedDict 定义明确的类型
class Context(TypedDict):
    max_count: int
    prefix: str

# ✅ 使用 Literal 限制可选值
from typing import Literal

class Context(TypedDict):
    environment: Literal["dev", "staging", "prod"]

# ✅ 使用 NotRequired 标记可选字段
from typing_extensions import NotRequired

class Context(TypedDict):
    required_field: str
    optional_field: NotRequired[int]
```

### 4. 默认值处理

```python
def node(state: State, *, runtime: Runtime[Context]) -> dict:
    # ✅ 使用 get() 提供默认值
    max_count = runtime.context.get("max_count", 10)

    # ❌ 直接访问可能导致 KeyError
    max_count = runtime.context["max_count"]
```

---

## 常见问题

### Q1: Context 和 State 有什么区别？

**A**:
- **State**：可变的工作流数据，节点可以更新，在节点间传递
- **Context**：不可变的配置数据，节点只能读取，运行时注入

### Q2: 可以在节点中修改 Context 吗？

**A**: 不可以。Context 是只读的，节点函数无法修改 Context 的内容。如果需要修改数据，应该放在 State 中。

### Q3: Context 在哪里传入？

**A**: 在调用 `invoke()`, `stream()`, `ainvoke()` 等方法时，通过 `context` 参数传入：

```python
result = compiled.invoke(
    {"x": 0.5},  # State 初始值
    context={"r": 3.0}  # Context 数据
)
```

### Q4: 如果不定义 context_schema 会怎样？

**A**: 如果不定义 `context_schema`，节点函数就不能使用 `runtime: Runtime[ContextT]` 参数。但可以使用其他节点协议（如 `_NodeWithConfig`）。

### Q5: Runtime 还包含什么？

**A**: Runtime 对象除了 `context` 外，还包含：
- `config`: RunnableConfig 配置
- `store`: BaseStore 存储接口
- `writer`: StreamWriter 流写入器

---

## 总结

**Context Schema 与 Runtime 的核心价值**：
1. **分离关注点**：配置与数据分离，State 专注于工作流数据，Context 专注于配置
2. **类型安全**：通过 TypedDict 和泛型提供完整的类型检查
3. **不可变性**：Context 只读，避免意外修改配置
4. **依赖注入**：通过 Runtime 对象优雅地注入依赖
5. **多环境支持**：轻松支持开发、测试、生产等多环境配置

**一句话总结**：Context Schema 定义不可变的运行时配置，通过 Runtime 对象注入到节点函数，实现配置与数据的分离，提供类型安全的依赖注入机制。
