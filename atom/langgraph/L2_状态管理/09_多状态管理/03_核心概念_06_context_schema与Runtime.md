# 核心概念6：context_schema 与 Runtime

> context_schema 定义不可变的运行时上下文——它不是状态，不会被 checkpoint，不会被节点修改，而是在调用时一次性传入的"环境配置"。

---

## 引用来源

**源码分析**:
- `libs/langgraph/langgraph/graph/state.py` — StateGraph 构造函数中的 context_schema 参数
- `libs/langgraph/langgraph/pregel/main.py` — Runtime 对象的注入机制

**官方文档**:
- Context7 LangGraph 文档 (2026-02-27)

---

## 1. 为什么需要 context_schema？

### 核心问题：有些数据不该放在状态里

在 LangGraph 中，State 是图的核心数据载体。但有些数据放在 State 里不合适：

```python
class State(TypedDict):
    query: str
    result: str
    # ❌ 以下数据不应该放在 State 中
    api_key: str          # 敏感信息，不该被 checkpoint 持久化
    db_connection: Any    # 运行时资源，无法序列化
    user_id: str          # 调用级别的配置，不该被节点修改
    feature_flags: dict   # 环境配置，不该参与状态流转
```

这些数据的共同特点：

| 特点 | State | 这些数据 |
|------|-------|----------|
| 可变性 | 节点可以修改 | 不应该被修改 |
| 持久化 | 被 checkpoint 保存 | 不应该被保存 |
| 生命周期 | 跨多次调用 | 单次调用有效 |
| 来源 | 图内部产生 | 外部传入 |

**context_schema 就是为这类数据设计的——定义一个不可变的、不被持久化的运行时上下文。**

### 一句话定义

**context_schema 定义了调用图时传入的不可变上下文数据的结构，通过 Runtime 对象在节点中访问，不参与状态管理和 checkpoint。**

### 双重类比

**前端类比：React 的 Context API**

```javascript
// React Context：全局只读配置，组件不能修改
const ThemeContext = React.createContext({ theme: "dark" });

function MyComponent() {
  const { theme } = useContext(ThemeContext);  // 只读访问
  // theme 不能被组件修改，只能由 Provider 设置
}
```

React Context 提供全局只读数据，组件可以读取但不能修改。LangGraph 的 context_schema 做的事情一模一样——节点可以读取 context，但不能修改它。

**日常生活类比：考试时的考场规则**

考试时，考场规则（不能交头接耳、不能使用手机）是"上下文"——每个考生都能看到，但没人能修改它。考生的答题卡才是"状态"——每个人都在修改自己的答案。context_schema 就是考场规则，State 就是答题卡。

---

## 2. context_schema 的本质：不创建 Channel

### 与其他三种 Schema 的根本区别

StateGraph 接受四种 Schema 参数：

```python
def __init__(
    self,
    state_schema: type[StateT],        # 创建 Channel ✅
    context_schema: type[ContextT],     # 不创建 Channel ❌
    *,
    input_schema: type[InputT],         # 创建 Channel ✅
    output_schema: type[OutputT],       # 创建 Channel ✅
) -> None:
```

[来源: sourcecode/langgraph/libs/langgraph/langgraph/graph/state.py]

关键区别：

| Schema | 创建 Channel？ | 参与状态流转？ | 被 Checkpoint？ | 节点可修改？ |
|--------|---------------|---------------|-----------------|-------------|
| state_schema | 是 | 是 | 是 | 是 |
| input_schema | 是 | 是（入口） | 是 | 是 |
| output_schema | 是 | 是（出口） | 是 | 是 |
| context_schema | 否 | 否 | 否 | 否 |

**context_schema 是四种 Schema 中唯一不创建 Channel 的。** 它的字段不会出现在图的 Channel 注册表中，不会被 Reducer 处理，不会被 Checkpoint 保存。

### 源码层面的证据

在 `_add_schema()` 方法中，只有 state_schema、input_schema、output_schema 会调用 `_get_channels()` 来创建 Channel。context_schema 走的是完全不同的路径——它的类型信息被保存下来，但不参与 Channel 系统。

```python
# 简化的逻辑
class StateGraph:
    def __init__(self, state_schema, context_schema=None, *,
                 input_schema=None, output_schema=None):
        # 这三个创建 Channel
        self._add_schema(state_schema)
        if input_schema:
            self._add_schema(input_schema)
        if output_schema:
            self._add_schema(output_schema)

        # context_schema 不创建 Channel，只保存类型信息
        self.context_schema = context_schema
```

---

## 3. 基本用法：Runtime 对象

### 完整示例

```python
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from typing_extensions import TypedDict


# 第一步：定义 context_schema
class ContextSchema(TypedDict):
    my_runtime_value: str


# 第二步：定义 State（正常的状态）
class State(TypedDict):
    my_state_value: int


# 第三步：节点函数接收 Runtime 参数
def node(state: State, runtime: Runtime[ContextSchema]):
    """节点通过 runtime.context 访问上下文"""
    if runtime.context["my_runtime_value"] == "a":
        return {"my_state_value": 1}
    elif runtime.context["my_runtime_value"] == "b":
        return {"my_state_value": 2}
    else:
        return {"my_state_value": 0}


# 第四步：构建图时传入 context_schema
builder = StateGraph(State, context_schema=ContextSchema)
builder.add_node(node)
builder.add_edge(START, "node")
builder.add_edge("node", END)
graph = builder.compile()


# 第五步：调用时通过 context 参数传入上下文
result_a = graph.invoke(
    {"my_state_value": 0},
    context={"my_runtime_value": "a"}  # 传入上下文
)
print(result_a)  # {'my_state_value': 1}

result_b = graph.invoke(
    {"my_state_value": 0},
    context={"my_runtime_value": "b"}  # 不同的上下文
)
print(result_b)  # {'my_state_value': 2}
```

### 关键步骤

1. **定义 ContextSchema** — 用 TypedDict 声明上下文的结构
2. **传给 StateGraph** — `StateGraph(State, context_schema=ContextSchema)`
3. **节点中访问** — 函数签名加 `runtime: Runtime[ContextSchema]`，通过 `runtime.context` 读取
4. **调用时传入** — `graph.invoke(input, context={...})`

---

## 4. Runtime 对象详解

### Runtime 的泛型参数

```python
from langgraph.runtime import Runtime

# Runtime[ContextSchema] 表示这个 Runtime 携带 ContextSchema 类型的上下文
def my_node(state: State, runtime: Runtime[ContextSchema]):
    value = runtime.context["my_runtime_value"]  # 类型安全的访问
```

`Runtime` 是一个泛型类，泛型参数就是 context_schema 的类型。这提供了类型检查支持——IDE 能自动补全 `runtime.context` 的字段。

### Runtime 的注入机制

LangGraph 在执行节点函数时，会检查函数签名：

```python
# LangGraph 内部逻辑（简化）
def execute_node(node_func, state, config, context):
    sig = inspect.signature(node_func)
    params = sig.parameters

    kwargs = {"state": state}

    # 如果函数签名中有 runtime 参数，注入 Runtime 对象
    if "runtime" in params:
        kwargs["runtime"] = Runtime(context=context)

    # 如果函数签名中有 config 参数，注入 config
    if "config" in params:
        kwargs["config"] = config

    return node_func(**kwargs)
```

这意味着 `runtime` 参数是可选的——如果节点不需要上下文，不写这个参数就行。

### 节点函数的多种签名

```python
# 签名1：只接收状态
def node_basic(state: State):
    return {"value": state["value"] + 1}

# 签名2：状态 + config
def node_with_config(state: State, config: dict):
    thread_id = config["configurable"]["thread_id"]
    return {"value": state["value"] + 1}

# 签名3：状态 + runtime（推荐的新方式）
def node_with_runtime(state: State, runtime: Runtime[ContextSchema]):
    api_key = runtime.context["api_key"]
    return {"value": state["value"] + 1}

# 签名4：状态 + config + runtime（全部都要）
def node_full(state: State, config: dict, runtime: Runtime[ContextSchema]):
    thread_id = config["configurable"]["thread_id"]
    api_key = runtime.context["api_key"]
    return {"value": state["value"] + 1}
```

---

## 5. State vs Context：核心对比

### 概念对比

```
┌─────────────────────────────────────────────────┐
│                  图的执行                        │
│                                                  │
│  Context（不可变）          State（可变）         │
│  ┌──────────────┐          ┌──────────────┐     │
│  │ api_key      │          │ query        │     │
│  │ user_id      │  ──读──→ │ result       │     │
│  │ model_name   │          │ messages     │     │
│  └──────────────┘          └──────────────┘     │
│  调用时传入，全程不变        节点读写，持续变化     │
│  不被 checkpoint            每步 checkpoint      │
│  不参与 Channel 系统        通过 Channel 流转     │
└─────────────────────────────────────────────────┘
```

### 详细对比表

| 维度 | State | Context |
|------|-------|---------|
| 定义方式 | `state_schema` | `context_schema` |
| 传入方式 | `graph.invoke(input)` | `graph.invoke(input, context=...)` |
| 访问方式 | `state["field"]` | `runtime.context["field"]` |
| 可变性 | 节点可以修改（返回更新） | 不可变，节点只能读取 |
| 持久化 | 被 Checkpoint 保存 | 不被 Checkpoint 保存 |
| Channel | 每个字段对应一个 Channel | 不创建 Channel |
| Reducer | 支持（Annotated） | 不支持 |
| 生命周期 | 跨 checkpoint 持续存在 | 单次 invoke 有效 |
| 典型内容 | 查询、结果、消息历史 | API 密钥、用户ID、配置 |

### 一个关键区别的例子

```python
# 假设用户中途恢复了一个被中断的图执行
# State 会从 checkpoint 恢复
# Context 需要重新传入

# 第一次调用（被中断）
graph.invoke(
    {"query": "什么是RAG？"},
    context={"api_key": "sk-xxx", "user_id": "user_123"},
    config={"configurable": {"thread_id": "t1"}}
)

# 恢复执行时，State 自动从 checkpoint 恢复
# 但 Context 必须重新传入！
graph.invoke(
    None,  # State 从 checkpoint 恢复，不需要传入
    context={"api_key": "sk-xxx", "user_id": "user_123"},  # 必须重新传
    config={"configurable": {"thread_id": "t1"}}
)
```

---

## 6. 替代已废弃的 config_schema

### 历史背景

在 context_schema 出现之前，LangGraph 使用 `config_schema` 来传递运行时配置：

```python
# ❌ 旧方式（已废弃）：通过 config_schema
class ConfigSchema(TypedDict):
    model_name: str

builder = StateGraph(State, config_schema=ConfigSchema)

def node(state: State, config: dict):
    model = config["configurable"]["model_name"]  # 从 config 中取
    ...

graph.invoke(input, config={"configurable": {"model_name": "gpt-4"}})
```

```python
# ✅ 新方式：通过 context_schema + Runtime
class ContextSchema(TypedDict):
    model_name: str

builder = StateGraph(State, context_schema=ContextSchema)

def node(state: State, runtime: Runtime[ContextSchema]):
    model = runtime.context["model_name"]  # 从 runtime.context 中取
    ...

graph.invoke(input, context={"model_name": "gpt-4"})
```

### 为什么要替换？

| 问题 | config_schema | context_schema |
|------|--------------|----------------|
| 访问方式 | `config["configurable"]["key"]` 嵌套太深 | `runtime.context["key"]` 简洁直观 |
| 类型安全 | config 是 dict，无类型提示 | Runtime[T] 有完整类型支持 |
| 语义清晰 | config 混合了框架配置和用户数据 | context 明确是用户的运行时数据 |
| 与框架配置混淆 | thread_id 等框架配置也在 config 中 | context 与框架配置完全分离 |

---

## 7. 实际使用场景

### 场景1：多模型切换

```python
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from typing_extensions import TypedDict


class ModelContext(TypedDict):
    model_name: str
    temperature: float
    max_tokens: int


class State(TypedDict):
    query: str
    answer: str


def generate(state: State, runtime: Runtime[ModelContext]):
    """根据上下文选择不同的模型配置"""
    model = runtime.context["model_name"]
    temp = runtime.context["temperature"]
    max_tok = runtime.context["max_tokens"]

    # 实际调用 LLM（这里用模拟）
    answer = f"[{model}, temp={temp}] 回答: {state['query']}"
    return {"answer": answer}


builder = StateGraph(State, context_schema=ModelContext)
builder.add_node("generate", generate)
builder.add_edge(START, "generate")
builder.add_edge("generate", END)
graph = builder.compile()

# 同一个图，不同的模型配置
result_fast = graph.invoke(
    {"query": "什么是RAG？", "answer": ""},
    context={"model_name": "gpt-4o-mini", "temperature": 0.0, "max_tokens": 100}
)

result_quality = graph.invoke(
    {"query": "什么是RAG？", "answer": ""},
    context={"model_name": "gpt-4o", "temperature": 0.7, "max_tokens": 2000}
)
```

### 场景2：多租户隔离

```python
class TenantContext(TypedDict):
    tenant_id: str
    api_key: str
    collection_name: str  # 每个租户有自己的向量集合


class RAGState(TypedDict):
    query: str
    docs: list[str]
    answer: str


def retrieve(state: RAGState, runtime: Runtime[TenantContext]):
    """根据租户上下文检索对应的集合"""
    collection = runtime.context["collection_name"]
    tenant = runtime.context["tenant_id"]
    # 从租户专属的集合中检索
    docs = [f"[{tenant}/{collection}] doc1", f"[{tenant}/{collection}] doc2"]
    return {"docs": docs}


def generate(state: RAGState, runtime: Runtime[TenantContext]):
    """使用租户的 API key 调用 LLM"""
    api_key = runtime.context["api_key"]
    # 使用租户的 API key 调用模型
    return {"answer": f"基于 {len(state['docs'])} 篇文档的回答"}


builder = StateGraph(RAGState, context_schema=TenantContext)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
rag = builder.compile()

# 租户A 的请求
rag.invoke(
    {"query": "产品价格？", "docs": [], "answer": ""},
    context={
        "tenant_id": "tenant_a",
        "api_key": "sk-tenant-a-xxx",
        "collection_name": "tenant_a_docs"
    }
)

# 租户B 的请求
rag.invoke(
    {"query": "退货政策？", "docs": [], "answer": ""},
    context={
        "tenant_id": "tenant_b",
        "api_key": "sk-tenant-b-yyy",
        "collection_name": "tenant_b_docs"
    }
)
```

### 场景3：特性开关（Feature Flags）

```python
class FeatureContext(TypedDict):
    enable_rerank: bool
    enable_query_rewrite: bool
    enable_cache: bool
    debug_mode: bool


class State(TypedDict):
    query: str
    docs: list[str]
    answer: str


def retrieve(state: State, runtime: Runtime[FeatureContext]):
    """根据特性开关决定检索策略"""
    docs = ["doc1", "doc2", "doc3"]  # 基础检索

    if runtime.context["enable_rerank"]:
        docs = rerank(docs)  # 启用重排序

    if runtime.context["debug_mode"]:
        print(f"检索到 {len(docs)} 篇文档")

    return {"docs": docs}


def maybe_rewrite_query(state: State, runtime: Runtime[FeatureContext]):
    """条件性地改写查询"""
    if runtime.context["enable_query_rewrite"]:
        return {"query": f"改写后: {state['query']}"}
    return {}  # 不改写，返回空更新
```

---

## 8. 子图中的 context 传递

context 会自动传递给子图，子图可以定义自己的 context_schema：

```python
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from typing_extensions import TypedDict


# 子图也可以有自己的 context_schema
class SubContext(TypedDict):
    sub_config: str


class SubState(TypedDict):
    value: str


def sub_node(state: SubState, runtime: Runtime[SubContext]):
    config_val = runtime.context.get("sub_config", "default")
    return {"value": f"{state['value']} [{config_val}]"}


sub_builder = StateGraph(SubState, context_schema=SubContext)
sub_builder.add_node("sub_node", sub_node)
sub_builder.add_edge(START, "sub_node")
sub_builder.add_edge("sub_node", END)
sub = sub_builder.compile()


# 父图
class ParentContext(TypedDict):
    parent_config: str


class ParentState(TypedDict):
    value: str


def parent_node(state: ParentState, runtime: Runtime[ParentContext]):
    return {"value": f"parent [{runtime.context['parent_config']}]"}


parent_builder = StateGraph(ParentState, context_schema=ParentContext)
parent_builder.add_node("parent_node", parent_node)
parent_builder.add_node("sub", sub)
parent_builder.add_edge(START, "parent_node")
parent_builder.add_edge("parent_node", "sub")
parent_builder.add_edge("sub", END)
parent = parent_builder.compile()

# 调用时传入的 context 会被传递给所有层级
result = parent.invoke(
    {"value": "start"},
    context={
        "parent_config": "p_val",
        "sub_config": "s_val"
    }
)
```

---

## 9. 常见误区

### 误区1：把应该是 State 的数据放进 Context

```python
# ❌ 错误：查询结果应该是 State，因为它会被节点修改
class BadContext(TypedDict):
    query: str      # 可能被改写 → 应该是 State
    docs: list[str] # 会被检索填充 → 应该是 State
    api_key: str    # 不变的配置 → 适合 Context

# ✅ 正确：只把不可变的配置放进 Context
class GoodContext(TypedDict):
    api_key: str
    model_name: str
    tenant_id: str
```

### 误区2：试图在节点中修改 Context

```python
# ❌ 错误：Context 是不可变的
def bad_node(state: State, runtime: Runtime[ContextSchema]):
    runtime.context["api_key"] = "new_key"  # 这不会生效！
    return {"value": 1}

# ✅ 正确：只读取 Context
def good_node(state: State, runtime: Runtime[ContextSchema]):
    key = runtime.context["api_key"]  # 只读
    return {"value": 1}
```

### 误区3：期望 Context 被 Checkpoint 保存

```python
# Context 不会被 checkpoint 保存
# 恢复执行时必须重新传入

# 第一次调用
graph.invoke(input, context={"key": "value"}, config=thread_config)

# 恢复时，context 必须重新传入
# ❌ 错误：以为 context 会自动恢复
graph.invoke(None, config=thread_config)

# ✅ 正确：重新传入 context
graph.invoke(None, context={"key": "value"}, config=thread_config)
```

---

## 10. 判断数据该放哪里

```
这个数据会被节点修改吗？
├─ 是 → State（state_schema）
└─ 否 → 这个数据需要被 checkpoint 保存吗？
    ├─ 是 → State（state_schema，但节点不修改它）
    └─ 否 → 这个数据是敏感信息或运行时资源吗？
        ├─ 是 → Context（context_schema）
        └─ 否 → 这个数据每次调用都不同吗？
            ├─ 是 → Context（context_schema）
            └─ 否 → 考虑硬编码或环境变量
```

### 快速参考

| 数据类型 | 放哪里 | 原因 |
|----------|--------|------|
| 用户查询 | State | 可能被改写，需要 checkpoint |
| 检索结果 | State | 被节点填充，需要 checkpoint |
| 对话历史 | State | 持续增长，需要 checkpoint |
| API 密钥 | Context | 敏感信息，不该被持久化 |
| 模型名称 | Context | 调用级配置，不该被修改 |
| 用户ID | Context | 调用级标识，不该被修改 |
| 数据库连接 | Context | 运行时资源，无法序列化 |
| 特性开关 | Context | 调用级配置，不该被修改 |
| Thread ID | Config | 框架级配置，用 config 传递 |

---

## 小结

| 要点 | 说明 |
|------|------|
| 本质 | 不创建 Channel 的 Schema，定义不可变运行时上下文 |
| 访问方式 | 节点签名加 `runtime: Runtime[T]`，通过 `runtime.context` 读取 |
| 传入方式 | `graph.invoke(input, context={...})` |
| 不可变 | 节点只能读取，不能修改 |
| 不持久化 | 不被 Checkpoint 保存，恢复执行时需重新传入 |
| 替代 | 替代已废弃的 config_schema，语义更清晰、类型更安全 |
| 典型用途 | API 密钥、模型配置、租户ID、特性开关、数据库连接 |
| 子图传递 | context 自动传递给子图 |
