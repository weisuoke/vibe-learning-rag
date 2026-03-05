# 核心概念 1：Pydantic BaseModel 状态定义

## 概述

Pydantic `BaseModel` 是 Python 生态中最流行的数据验证库的核心类。在 LangGraph 中，你可以用它替代 `TypedDict` 来定义状态 schema，从而获得**运行时类型验证**、**自动类型转换**和**自定义验证器**三大能力。简单说：TypedDict 只是"写给 IDE 看的注释"，而 Pydantic BaseModel 是"写给程序看的合同"。

[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

---

## 为什么需要 Pydantic BaseModel？

### TypedDict 的局限

TypedDict 是 Python 标准库提供的类型提示工具，LangGraph 默认使用它定义状态：

```python
from typing_extensions import TypedDict

class AgentState(TypedDict):
    query: str
    response: str
    confidence: float
```

看起来很好，但有一个关键问题：**TypedDict 不做任何运行时检查**。

```python
# TypedDict 不会阻止你传入错误类型
state = AgentState(query=123, response=None, confidence="高")  # 不报错！
```

IDE 会画红线提醒你，但程序运行时完全不管。这在 RAG 系统中很危险——如果用户传入了错误格式的查询，系统会在后续节点中以难以排查的方式崩溃。

### Pydantic BaseModel 的解决方案

Pydantic BaseModel 在运行时强制执行类型检查：

```python
from pydantic import BaseModel

class AgentState(BaseModel):
    query: str
    response: str = ""
    confidence: float = 0.0

# ✅ 正常使用
state = AgentState(query="什么是 RAG？")
print(state.query)        # "什么是 RAG？"
print(state.confidence)   # 0.0

# ✅ 自动类型转换
state = AgentState(query="测试", confidence="0.95")
print(state.confidence)   # 0.95（字符串自动转为 float）

# ❌ 无法转换时直接报错
try:
    state = AgentState(query="测试", confidence="不是数字")
except Exception as e:
    print(f"验证失败: {e}")
    # ValidationError: 1 validation error for AgentState
    # confidence - Input should be a valid number
```

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

### 对比总结

| 特性 | TypedDict | Pydantic BaseModel |
|------|-----------|-------------------|
| 静态类型提示 | ✅ 支持 | ✅ 支持 |
| 运行时验证 | ❌ 无 | ✅ 有 |
| 自动类型转换 | ❌ 无 | ✅ 有（如 str → int） |
| 默认值 | 需要 `total=False` | 直接赋值 |
| 自定义验证器 | ❌ 无 | ✅ field_validator / model_validator |
| 嵌套模型验证 | ❌ 无 | ✅ 递归验证 |
| 性能开销 | 几乎为零 | 有少量开销 |
| 推荐场景 | 内部状态、高性能 | 输入边界、需要严格验证 |

[来源: reference/search_状态验证_01.md | 社区最佳实践]

---

## 基础用法：在 LangGraph 中使用 BaseModel

### 最小完整示例

```python
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END

# 1. 用 Pydantic BaseModel 定义状态
class AgentState(BaseModel):
    query: str
    response: str = ""
    confidence: float = 0.0

# 2. 定义节点函数
def process_node(state: AgentState):
    return {"response": f"处理了: {state.query}", "confidence": 0.95}

# 3. 构建图
builder = StateGraph(AgentState)
builder.add_node("process", process_node)
builder.add_edge(START, "process")
builder.add_edge("process", END)
graph = builder.compile()

# 4. 运行
result = graph.invoke({"query": "什么是 RAG？"})
print(result)
# {'query': '什么是 RAG？', 'response': '处理了: 什么是 RAG？', 'confidence': 0.95}
```

**关键点**：
- `StateGraph(AgentState)` 接受 Pydantic BaseModel 作为 schema
- 节点函数接收的 `state` 参数是 `AgentState` 实例，可以用 `.query` 访问属性
- 节点函数返回的是普通字典，LangGraph 会自动合并到状态中
- 调用 `graph.invoke()` 时传入的字典会经过 Pydantic 验证

[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

---

## 源码解析：LangGraph 如何触发 Pydantic 验证

理解底层机制能帮你预判行为、排查问题。LangGraph 通过两个关键函数将 Pydantic 验证接入图的执行流程。

### `_pick_mapper`：决定是否需要类型转换

```python
# langgraph/graph/state.py (简化版)
from functools import partial
from inspect import isclass

def _pick_mapper(state_keys, schema):
    # 情况1：根模型，不需要转换
    if state_keys == ["__root__"]:
        return None
    # 情况2：schema 是 dict 的子类（如 TypedDict），不需要转换
    if isclass(schema) and issubclass(schema, dict):
        return None
    # 情况3：其他情况（如 Pydantic BaseModel），需要转换
    return partial(_coerce_state, schema)
```

**逻辑很清晰**：
- TypedDict 继承自 `dict`，走第二个分支，不做转换
- Pydantic BaseModel 不继承 `dict`，走第三个分支，返回转换函数

### `_coerce_state`：实际触发验证的地方

```python
# langgraph/graph/state.py
def _coerce_state(schema, input):
    return schema(**input)  # 这里触发 Pydantic 验证！
```

就这一行代码。当 `schema` 是 `AgentState`（Pydantic BaseModel）时，`schema(**input)` 等价于 `AgentState(**input)`，这会触发 Pydantic 的完整验证流程：

1. 检查必填字段是否存在
2. 对每个字段执行类型验证和转换
3. 运行 `field_validator` 装饰器
4. 运行 `model_validator` 装饰器

如果任何一步失败，Pydantic 会抛出 `ValidationError`。

[来源: sourcecode/langgraph/libs/langgraph/langgraph/graph/state.py]

### 验证触发时机

```
graph.invoke({"query": "什么是 RAG？"})
    ↓
LangGraph 内部调用 _pick_mapper(state_keys, AgentState)
    ↓
返回 partial(_coerce_state, AgentState)
    ↓
执行 _coerce_state(AgentState, {"query": "什么是 RAG？"})
    ↓
等价于 AgentState(query="什么是 RAG？")
    ↓
Pydantic 验证通过 → 创建 AgentState 实例
    ↓
传入第一个节点
```

---

## 嵌套模型：复杂状态的结构化定义

实际的 RAG 系统状态往往不是扁平的。Pydantic 支持嵌套模型，让你用组合的方式构建复杂状态。

```python
from pydantic import BaseModel

# 定义子模型
class SearchResult(BaseModel):
    title: str
    url: str
    score: float

# 在状态中嵌套使用
class RAGState(BaseModel):
    query: str
    results: list[SearchResult] = []
    answer: str = ""

# 使用时，嵌套模型也会被验证
state = RAGState(
    query="什么是向量数据库？",
    results=[
        {"title": "Milvus 入门", "url": "https://milvus.io", "score": 0.95},
        {"title": "向量检索原理", "url": "https://example.com", "score": 0.87},
    ]
)

# Pydantic 自动将字典转换为 SearchResult 实例
print(type(state.results[0]))  # <class 'SearchResult'>
print(state.results[0].title)  # "Milvus 入门"

# 嵌套验证也生效
try:
    RAGState(
        query="测试",
        results=[{"title": "缺少字段"}]  # 缺少 url 和 score
    )
except Exception as e:
    print(f"嵌套验证失败: {e}")
```

**在 LangGraph 中使用嵌套模型**：

```python
from langgraph.graph import StateGraph, START, END

class SearchResult(BaseModel):
    title: str
    url: str
    score: float

class RAGState(BaseModel):
    query: str
    results: list[SearchResult] = []
    answer: str = ""

def search_node(state: RAGState):
    # 模拟检索结果
    return {
        "results": [
            SearchResult(title="RAG 架构", url="https://example.com/rag", score=0.92)
        ]
    }

def generate_node(state: RAGState):
    context = "\n".join(r.title for r in state.results)
    return {"answer": f"基于 {context} 的回答"}

builder = StateGraph(RAGState)
builder.add_node("search", search_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "search")
builder.add_edge("search", "generate")
builder.add_edge("generate", END)
graph = builder.compile()

result = graph.invoke({"query": "什么是 RAG？"})
print(result)
```

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

---

## Field 约束：精细化字段验证

Pydantic 的 `Field` 函数让你为字段添加各种约束条件，不需要写自定义验证器就能完成常见的验证需求。

```python
from pydantic import BaseModel, Field

class StrictState(BaseModel):
    # 字符串长度约束
    query: str = Field(min_length=1, max_length=500, description="用户查询")

    # 数值范围约束
    max_results: int = Field(default=5, ge=1, le=20, description="最大返回结果数")

    # 浮点数范围约束
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="生成温度")

    # 正则表达式约束
    model_name: str = Field(default="gpt-4", pattern=r"^(gpt|claude|gemini)-.*$")
```

**验证效果**：

```python
# ✅ 正常
state = StrictState(query="什么是 RAG？")

# ❌ 查询为空
try:
    StrictState(query="")
except Exception as e:
    print("空查询被拦截")

# ❌ max_results 超出范围
try:
    StrictState(query="测试", max_results=100)
except Exception as e:
    print("结果数超限被拦截")

# ❌ temperature 超出范围
try:
    StrictState(query="测试", temperature=3.0)
except Exception as e:
    print("温度超限被拦截")
```

### 常用 Field 约束速查

| 约束 | 适用类型 | 说明 | 示例 |
|------|---------|------|------|
| `min_length` | str, list | 最小长度 | `Field(min_length=1)` |
| `max_length` | str, list | 最大长度 | `Field(max_length=500)` |
| `ge` | int, float | 大于等于 | `Field(ge=0)` |
| `le` | int, float | 小于等于 | `Field(le=100)` |
| `gt` | int, float | 大于 | `Field(gt=0)` |
| `lt` | int, float | 小于 | `Field(lt=1.0)` |
| `pattern` | str | 正则匹配 | `Field(pattern=r"^gpt-.*$")` |
| `default` | 任意 | 默认值 | `Field(default=5)` |

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

---

## 验证范围说明：一个关键的设计选择

这是使用 Pydantic BaseModel 时最容易踩的坑，必须理解清楚。

### 验证只在入口触发

Pydantic 验证**仅在图的首个节点输入时触发**，节点之间的状态传递不会重新触发完整验证。

```python
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

class MyState(BaseModel):
    value: int = Field(ge=0, le=100)  # 约束：0-100

def node_a(state: MyState):
    # 这里返回 200，超出了 Field 约束
    # 但不会报错！因为节点间传递不触发验证
    return {"value": 200}

def node_b(state: MyState):
    print(f"node_b 收到: {state.value}")  # 200，没有被拦截
    return {"value": state.value}

builder = StateGraph(MyState)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)
graph = builder.compile()

# 入口验证生效：value=50 通过
result = graph.invoke({"value": 50})
print(result["value"])  # 200（node_a 的返回值，未被验证拦截）

# 入口验证生效：value=200 会被拦截
try:
    graph.invoke({"value": 200})  # ❌ ValidationError
except Exception as e:
    print(f"入口验证拦截: {e}")
```

### 为什么这样设计？

这是 LangGraph 的**性能优化选择**：

1. **性能考虑**：每次节点间传递都做完整验证会显著降低性能
2. **信任内部逻辑**：节点是开发者自己写的代码，应该保证输出正确
3. **灵活性**：允许中间状态暂时超出约束范围

### 如果需要节点间验证怎么办？

在节点函数内部手动验证：

```python
def strict_node(state: MyState):
    result = {"value": some_computation()}

    # 手动验证
    validated = MyState(**{**state.model_dump(), **result})
    return result
```

[来源: sourcecode/langgraph/libs/langgraph/langgraph/graph/state.py]

---

## 实际应用场景

### 场景 1：RAG 系统中验证查询格式

```python
from pydantic import BaseModel, Field, field_validator

class RAGQueryState(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    language: str = Field(default="zh", pattern=r"^(zh|en|ja)$")
    top_k: int = Field(default=5, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def query_not_only_spaces(cls, v: str) -> str:
        if v.strip() == "":
            raise ValueError("查询不能只包含空格")
        return v.strip()
```

### 场景 2：Agent 系统中验证工具调用参数

```python
from pydantic import BaseModel, Field
from typing import Literal

class ToolCallState(BaseModel):
    tool_name: Literal["search", "calculator", "weather"]
    tool_input: dict = Field(default_factory=dict)
    max_retries: int = Field(default=3, ge=1, le=10)
    timeout: float = Field(default=30.0, gt=0, le=300.0)
```

### 场景 3：多步推理中验证中间状态

```python
from pydantic import BaseModel, Field

class ReasoningState(BaseModel):
    question: str = Field(min_length=1)
    current_step: int = Field(default=0, ge=0)
    max_steps: int = Field(default=5, ge=1, le=20)
    thoughts: list[str] = Field(default_factory=list)
    final_answer: str = ""
```

[来源: reference/search_状态验证_01.md | 社区最佳实践]

---

## 常见误区

### 误区 1："用了 Pydantic 就全程都有验证" ❌

**事实**：验证只在 `graph.invoke()` 的入口触发，节点之间不会重新验证。这是上面详细讲过的设计选择。

### 误区 2："Pydantic BaseModel 和 TypedDict 可以混用" ❌

**事实**：一个 `StateGraph` 只能使用一种 schema 类型。如果用了 `BaseModel`，所有节点都基于这个 BaseModel 工作。不要在同一个图中混用两种定义方式。

### 误区 3："BaseModel 状态的性能开销很大" ❌

**事实**：Pydantic v2 使用 Rust 编写的核心验证引擎，性能开销很小。对于大多数 RAG 应用，瓶颈在 LLM 调用和向量检索，状态验证的开销可以忽略不计。

[来源: reference/search_状态验证_01.md | 社区讨论]

---

## 总结

### 核心要点

1. **Pydantic BaseModel** 为 LangGraph 状态提供运行时验证，TypedDict 只有静态提示
2. **`_coerce_state`** 是触发验证的关键函数，通过 `schema(**input)` 调用 Pydantic
3. **嵌套模型** 支持复杂状态结构，Pydantic 会递归验证
4. **Field 约束** 提供声明式的字段验证（长度、范围、正则等）
5. **验证范围有限**：仅在图的入口触发，节点间不重新验证

### 选型建议

- **原型开发 / 内部状态**：用 TypedDict，轻量快速
- **生产环境 / 外部输入**：用 Pydantic BaseModel，安全可靠
- **混合策略**：对外接口的图用 BaseModel，内部子图用 TypedDict

### 下一步

在理解了 Pydantic BaseModel 状态定义后，下一个核心概念将深入讲解 **field_validator 与 model_validator 自定义验证器**，包括：
- 字段级验证 vs 模型级验证
- `mode='before'` 和 `mode='after'` 的区别
- 在 LangGraph 中的实际验证模式

---

**参考资料**：
- [LangGraph 源码 - state.py](sourcecode/langgraph/libs/langgraph/langgraph/graph/state.py)
- [LangGraph 官方文档](reference/context7_langgraph_01.md)
- [Pydantic 官方文档](reference/context7_pydantic_01.md)
- [社区最佳实践](reference/search_状态验证_01.md)
