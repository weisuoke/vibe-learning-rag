# 核心概念 6：TypedDict vs Pydantic BaseModel 选型策略

## 概述

LangGraph 支持两种状态定义方式：`TypedDict`（轻量类型提示）和 `Pydantic BaseModel`（运行时验证）。两者不是非此即彼的关系——生产环境中最常见的做法是**混合使用**：边界处用 Pydantic 做严格验证，内部状态用 TypedDict 保持轻量。选型的核心判断标准只有一个：**这个数据是否来自不可信的外部？**

[来源: reference/search_状态验证_01.md | 社区最佳实践]

---

## 本质区别：一句话说清

- **TypedDict**：写给 IDE 和开发者看的"类型注释"，运行时就是普通字典，不做任何检查
- **Pydantic BaseModel**：写给程序看的"数据合同"，运行时强制执行类型验证、转换和约束

```python
from typing_extensions import TypedDict
from pydantic import BaseModel

# TypedDict：运行时不检查
class StateA(TypedDict):
    count: int

a: StateA = {"count": "不是数字"}  # 运行时不报错，IDE 会警告

# Pydantic：运行时强制检查
class StateB(BaseModel):
    count: int

b = StateB(count="42")     # ✅ 自动转换为 int(42)
b = StateB(count="abc")    # ❌ ValidationError
```

---

## 全面对比表

| 特性 | TypedDict | Pydantic BaseModel |
|------|-----------|-------------------|
| 运行时验证 | ❌ 无 | ✅ 完整验证 |
| 类型转换 | ❌ 无 | ✅ 自动转换（如 `"42"` → `42`） |
| 自定义验证器 | ❌ 不支持 | ✅ `field_validator` / `model_validator` |
| 性能开销 | 极低（普通字典） | 有一定开销（首次验证） |
| IDE 支持 | ✅ 类型提示 | ✅ 类型提示 + 自动补全 + 文档 |
| 序列化 | 需手动处理 | ✅ `model_dump()` / `model_dump_json()` |
| 默认值 | 有限支持（`total=False`） | ✅ 完整支持（直接赋值） |
| 嵌套模型 | 手动处理 | ✅ 自动递归验证 |
| Reducer 支持 | ✅ `Annotated[list, add]` | ⚠️ 支持但需注意兼容性 |
| 数据访问方式 | `state["key"]`（字典） | `state.key`（属性） |
| 继承自 | `dict` | `BaseModel`（非 dict） |

[来源: reference/search_状态验证_01.md | 社区最佳实践]

---

## TypedDict 示例：轻量内部状态

```python
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END

# TypedDict 定义：简洁、零开销
class AgentState(TypedDict):
    query: str
    response: str
    step_count: int
    messages: Annotated[list, add]  # 支持 Reducer

def process(state: AgentState):
    # 访问方式：字典风格
    query = state["query"]
    return {
        "response": f"处理了: {query}",
        "step_count": state["step_count"] + 1,
        "messages": [f"步骤 {state['step_count'] + 1} 完成"],
    }

builder = StateGraph(AgentState)
builder.add_node("process", process)
builder.add_edge(START, "process")
builder.add_edge("process", END)
graph = builder.compile()

result = graph.invoke({
    "query": "什么是 RAG？",
    "response": "",
    "step_count": 0,
    "messages": [],
})
print(result)
```

**TypedDict 的优势场景**：
- 节点之间的内部状态传递，数据来源可信
- 需要 Reducer 函数聚合多个节点的输出
- 性能敏感的高频调用场景
- 简单的数据搬运，不需要复杂校验

---

## Pydantic BaseModel 示例：严格边界验证

```python
from pydantic import BaseModel, Field, field_validator
from langgraph.graph import StateGraph, START, END

# Pydantic 定义：带验证、带约束、带默认值
class AgentState(BaseModel):
    query: str = Field(min_length=1, max_length=500, description="用户查询")
    response: str = ""
    step_count: int = Field(default=0, ge=0)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    @field_validator('query')
    @classmethod
    def clean_query(cls, v: str) -> str:
        """自动清洗用户输入：去除首尾空白"""
        return v.strip()

    @field_validator('response')
    @classmethod
    def truncate_response(cls, v: str) -> str:
        """防止响应过长"""
        if len(v) > 10000:
            return v[:10000] + "..."
        return v

def process(state: AgentState):
    # 访问方式：属性风格
    query = state.query
    return {
        "response": f"处理了: {query}",
        "step_count": state.step_count + 1,
    }

builder = StateGraph(AgentState)
builder.add_node("process", process)
builder.add_edge(START, "process")
builder.add_edge("process", END)
graph = builder.compile()

# ✅ 正常调用
result = graph.invoke({"query": "什么是 RAG？"})

# ✅ 自动类型转换：字符串 "0.9" → float 0.9
result = graph.invoke({"query": "测试", "temperature": "0.9"})

# ❌ 验证失败：空查询
try:
    graph.invoke({"query": ""})
except Exception as e:
    print(f"验证拦截: {e}")

# ❌ 验证失败：temperature 超出范围
try:
    graph.invoke({"query": "测试", "temperature": 5.0})
except Exception as e:
    print(f"验证拦截: {e}")
```

**Pydantic 的优势场景**：
- 处理不可信的外部输入（用户请求、API 调用）
- LLM 输出需要格式验证（结构化输出解析）
- 需要自动类型转换（前端传来的字符串数字）
- 需要复杂的业务规则验证（字段间依赖关系）
- 需要序列化/反序列化（API 响应、日志记录）

---

## 选型决策树

面对一个新的 LangGraph 项目，按以下决策树选择：

```
这个状态数据需要运行时验证吗？
│
├── 是 → 使用 Pydantic BaseModel
│   │
│   ├── 仅需要验证外部输入？
│   │   └── 是 → input_schema 用 Pydantic，内部状态用 TypedDict
│   │
│   ├── 全程都需要验证？
│   │   └── 是 → state_schema 用 Pydantic
│   │
│   └── 需要验证输出格式？
│       └── 是 → output_schema 用 Pydantic
│
└── 否 → 使用 TypedDict
    │
    ├── 需要 Reducer 聚合？
    │   └── 是 → TypedDict + Annotated[type, reducer]
    │
    ├── 需要高性能？
    │   └── 是 → 纯 TypedDict
    │
    └── 简单数据传递？
        └── 是 → 纯 TypedDict
```

**一句话总结**：不确定就用 TypedDict，碰到外部数据就上 Pydantic。

---

## 混合架构：生产环境推荐模式

这是社区共识最强的模式——**边界用 Pydantic，内部用 TypedDict**。LangGraph 的 `StateGraph` 构造函数天然支持这种分离：

```python
from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END

# ========== 输入验证：Pydantic（严格） ==========
class InputState(BaseModel):
    """外部输入，需要严格验证"""
    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    @field_validator('query')
    @classmethod
    def clean_query(cls, v: str) -> str:
        return v.strip()

# ========== 内部状态：TypedDict（轻量） ==========
class InternalState(TypedDict):
    """内部流转，不需要重复验证"""
    query: str
    max_results: int
    temperature: float
    results: Annotated[list, add]   # Reducer：多个节点的结果自动合并
    answer: str

# ========== 输出验证：Pydantic（规范） ==========
class OutputState(BaseModel):
    """对外输出，确保格式规范"""
    answer: str
    sources: list[str] = []

# ========== 节点定义 ==========
def search_node(state: InternalState):
    query = state["query"]  # TypedDict 用字典访问
    return {
        "results": [
            {"title": f"关于 {query} 的文档", "url": "https://example.com"}
        ]
    }

def generate_node(state: InternalState):
    context = str(state["results"])
    return {
        "answer": f"基于检索结果的回答: {context}",
    }

# ========== 构建图 ==========
builder = StateGraph(
    InternalState,              # 内部状态 schema
    input=InputState,           # 输入 schema（Pydantic 验证）
    output=OutputState,         # 输出 schema（Pydantic 格式化）
)
builder.add_node("search", search_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "search")
builder.add_edge("search", "generate")
builder.add_edge("generate", END)
graph = builder.compile()

# 使用
result = graph.invoke({"query": "什么是向量数据库？"})
print(result)
# 输出只包含 OutputState 定义的字段：answer 和 sources
```

### 混合架构的三层职责

| 层 | Schema 类型 | 职责 | 为什么 |
|----|------------|------|--------|
| 输入层 | Pydantic `InputState` | 验证外部输入 | 用户数据不可信，必须校验 |
| 内部层 | TypedDict `InternalState` | 节点间数据流转 | 内部数据可信，追求性能 |
| 输出层 | Pydantic `OutputState` | 规范输出格式 | 对外接口需要稳定契约 |

[来源: reference/search_状态验证_01.md]

---

## 源码视角：LangGraph 如何区分两者

理解底层机制能帮你预判行为。关键在 `state.py` 的 `_pick_mapper` 函数：

```python
# langgraph/graph/state.py（简化版）
def _pick_mapper(state_keys, schema):
    if state_keys == ["__root__"]:
        return None                          # 根模型，不转换
    if isclass(schema) and issubclass(schema, dict):
        return None                          # TypedDict 继承 dict，不转换
    return partial(_coerce_state, schema)    # Pydantic BaseModel，需要转换

def _coerce_state(schema, input):
    return schema(**input)                   # 触发 Pydantic 验证！
```

**核心逻辑**：
- `TypedDict` 继承自 `dict` → `issubclass(schema, dict)` 为 `True` → 不做转换
- `Pydantic BaseModel` 不继承 `dict` → 返回 `_coerce_state` → 调用 `schema(**input)` 触发验证

这就是为什么 TypedDict 零开销而 Pydantic 有验证开销的根本原因。

[来源: sourcecode/langgraph/libs/langgraph/langgraph/graph/state.py]

---

## 性能对比

| 指标 | TypedDict | Pydantic BaseModel |
|------|-----------|-------------------|
| 状态创建 | ~0 开销（普通字典） | 首次验证有开销 |
| 节点间传递 | 直接传字典 | 经过 `_coerce_state` 转换 |
| 内存占用 | 字典大小 | 对象实例 + 元数据 |
| LangGraph 优化 | 无需优化 | LRU 缓存动态模型创建 |

**实际影响**：
- 对于大多数 RAG 应用，性能差异可以忽略不计
- 瓶颈通常在 LLM API 调用（秒级）和向量检索（毫秒级），而非状态验证（微秒级）
- 只有在极高频调用（每秒数千次状态更新）时才需要考虑性能差异

```python
# 性能差异的直觉感受
from typing_extensions import TypedDict
from pydantic import BaseModel

class StateT(TypedDict):
    query: str
    count: int

class StateP(BaseModel):
    query: str
    count: int = 0

# TypedDict：~0.1 微秒
data = {"query": "test", "count": 1}

# Pydantic：~1-5 微秒（首次），后续有缓存
state = StateP(**data)

# LLM API 调用：~500-2000 毫秒
# 结论：状态验证的开销在 RAG 场景中完全可以忽略
```

[来源: sourcecode/langgraph/libs/langgraph/langgraph/_internal/_pydantic.py]

---

## 何时必须用 Pydantic

以下场景中，TypedDict 无法满足需求，必须使用 Pydantic：

### 1. 处理不可信的外部输入

```python
from pydantic import BaseModel, Field, field_validator

class UserInput(BaseModel):
    """用户通过 API 提交的查询"""
    query: str = Field(min_length=1, max_length=1000)
    language: str = Field(default="zh", pattern=r"^(zh|en|ja)$")

    @field_validator('query')
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        # 去除潜在的注入内容
        return v.strip().replace('\x00', '')
```

### 2. LLM 输出需要格式验证

```python
class LLMOutput(BaseModel):
    """LLM 结构化输出，需要验证格式"""
    answer: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[str] = Field(default_factory=list, max_length=10)
```

### 3. 需要自动类型转换

```python
class APIState(BaseModel):
    """前端传来的数据，类型可能不精确"""
    page: int = 1           # 前端可能传 "1"（字符串），Pydantic 自动转换
    score: float = 0.0      # 前端可能传 "0.95"，自动转换
    active: bool = True     # 前端可能传 "true"，自动转换
```

### 4. 需要复杂的业务规则验证

```python
from pydantic import BaseModel, model_validator

class SearchConfig(BaseModel):
    start_date: str
    end_date: str
    max_results: int = 10

    @model_validator(mode='after')
    def validate_date_range(self):
        if self.start_date > self.end_date:
            raise ValueError('start_date 必须早于 end_date')
        return self
```

---

## 何时应该用 TypedDict

以下场景中，TypedDict 是更好的选择：

### 1. 内部节点间的状态传递

```python
from typing_extensions import TypedDict

class InternalState(TypedDict):
    """节点间传递的中间数据，来源可信"""
    query: str
    embeddings: list[float]
    retrieved_docs: list[dict]
    generated_answer: str
```

### 2. 需要 Reducer 函数

```python
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class ChatState(TypedDict):
    """对话状态，消息需要累积"""
    messages: Annotated[list, add]      # 多个节点的消息自动合并
    tool_calls: Annotated[list, add]    # 工具调用记录累积
    current_step: str
```

TypedDict + `Annotated` 是 LangGraph 中 Reducer 的标准写法。Pydantic 也支持 `Annotated`，但 TypedDict 更简洁直接。

### 3. 性能敏感场景

```python
# 高频调用的子图状态，用 TypedDict 减少开销
class ToolState(TypedDict):
    tool_name: str
    tool_input: dict
    tool_output: str
```

### 4. 简单的数据传递

```python
# 只是搬运数据，不需要验证
class PassthroughState(TypedDict):
    data: dict
    metadata: dict
```

---

## 社区共识总结

基于 GitHub Issues、技术博客和社区讨论，以下是被广泛认可的最佳实践：

| 原则 | 说明 |
|------|------|
| 内部状态用 TypedDict | 轻量、无运行时开销、Reducer 支持好 |
| 边界处用 Pydantic | 输入输出、外部集成需要严格验证 |
| 生产环境混合使用 | `StateGraph(InternalState, input=InputState, output=OutputState)` |
| 保持状态简洁 | 不要把所有数据都塞进状态，只放节点间需要共享的 |
| 统一团队规范 | 同一项目中保持一致的选型标准 |
| 考虑错误处理 | Pydantic 验证失败会抛 `ValidationError`，需要 try-except |

### 常见误区

- **误区 1**："用了 Pydantic 就万事大吉" → 验证只在 `_coerce_state` 时触发，不是每次状态更新都验证
- **误区 2**："TypedDict 完全没有类型安全" → IDE 和 mypy 仍然会做静态检查
- **误区 3**："两者不能混用" → LangGraph 的 `input`/`output` 参数就是为混合使用设计的
- **误区 4**："Pydantic 性能差" → 在 RAG 场景中，LLM 调用才是瓶颈，验证开销可忽略

[来源: reference/search_状态验证_01.md]

---

## 速查卡片

```
┌─────────────────────────────────────────────────┐
│         TypedDict vs Pydantic 选型速查            │
├─────────────────────────────────────────────────┤
│                                                 │
│  数据来自外部（用户/API/LLM）？                    │
│    → Pydantic BaseModel                         │
│                                                 │
│  数据来自内部节点？                                │
│    → TypedDict                                  │
│                                                 │
│  需要 Reducer？                                  │
│    → TypedDict + Annotated                      │
│                                                 │
│  需要自动类型转换？                                │
│    → Pydantic BaseModel                         │
│                                                 │
│  生产环境？                                       │
│    → 混合架构：                                   │
│      StateGraph(TypedDict,                      │
│                 input=Pydantic,                  │
│                 output=Pydantic)                 │
│                                                 │
└─────────────────────────────────────────────────┘
```

<!-- APPEND_MARKER_6 -->
