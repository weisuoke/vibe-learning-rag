# 核心概念 1：状态 Schema 设计原则

> Schema 是状态管理的地基。地基歪了，上面盖什么都会塌。好的 Schema 设计让节点职责清晰、类型安全、易于演进。

---

## 引用来源

**源码分析**:
- `libs/langgraph/langgraph/graph/state.py` — `_get_channel()` 通道类型选择、`_add_schema()` Schema 注册
- `libs/langgraph/langgraph/channels/binop.py` — `BinaryOperatorAggregate` Reducer 通道
- `libs/langgraph/langgraph/channels/last_value.py` — `LastValue` 默认覆盖通道

[来源: sourcecode/langgraph]

**官方文档**:
- Context7 LangGraph 文档 (2026-02-27)

---

## 概述

**状态 Schema 设计原则是指在 LangGraph 中定义状态结构时，如何选择类型系统（TypedDict vs Pydantic）、如何命名字段、如何绑定 Reducer、以及如何通过 Input/Output Schema 分离来控制图的对外接口。**

Schema 是所有最佳实践的基础——Reducer 选型、不可变性、模块化、持久化，全都建立在一个设计良好的 Schema 之上。

---

## 1. TypedDict vs Pydantic 选型

LangGraph 支持两种主流的 Schema 定义方式。选错了不会报错，但会在项目规模增长后带来痛苦。

### 1.1 TypedDict 方式

TypedDict 是 Python 标准库的类型提示工具，**零运行时开销**，只在静态检查时生效。

```python
from typing import Annotated, TypedDict
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage


class ChatState(TypedDict):
    """简单聊天机器人的状态 — TypedDict 方式"""
    # 消息列表，使用 add_messages reducer 追加合并
    messages: Annotated[list[AnyMessage], add_messages]
    # 用户 ID，无 reducer → 默认 LastValue 覆盖
    user_id: str
    # 当前轮次，无 reducer → 默认 LastValue 覆盖
    turn_count: int
```

**特点**：
- 运行时就是普通 `dict`，没有额外的实例化开销
- 类型检查依赖 mypy / pyright 等外部工具
- 不会自动验证数据——传入错误类型不会报错
- LangGraph 内部通过 `_get_channel()` 解析字段注解来决定通道类型

**源码原理**：`_get_channel()` 遇到无注解字段时创建 `LastValue` 通道，遇到 `Annotated[T, reducer]` 时创建 `BinaryOperatorAggregate` 通道。TypedDict 的字段在运行时通过 `get_type_hints()` 提取。

### 1.2 Pydantic BaseModel 方式

Pydantic 提供**运行时验证**，每次状态更新都会检查类型和约束。

```python
from pydantic import BaseModel, Field, field_validator
from typing import Annotated
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage


class StrictChatState(BaseModel):
    """带验证的聊天状态 — Pydantic 方式"""
    messages: Annotated[list[AnyMessage], add_messages] = Field(
        default_factory=list,
        description="对话消息历史"
    )
    user_id: str = Field(
        ...,  # 必填
        min_length=1,
        description="用户唯一标识"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,  # >= 0
        le=2.0,  # <= 2
        description="LLM 生成温度"
    )

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        """确保 user_id 不含空格"""
        if " " in v:
            raise ValueError("user_id 不能包含空格")
        return v
```

**特点**：
- 运行时自动验证类型、范围、格式
- 支持嵌套模型、自定义验证器
- 节点接收的是验证后的 Pydantic 对象（可用 `.` 访问属性）
- 节点返回的是普通字典（不需要返回完整模型）
- 有实例化开销，每次状态更新都会触发验证

**源码原理**：LangGraph 内部使用 `_pick_mapper` 实现惰性求值——只在节点真正需要 Pydantic 对象时才做类型转换，避免不必要的性能损耗。

### 1.3 选型决策表

| 场景 | 推荐 | 原因 |
|------|------|------|
| 原型开发 / 快速验证 | TypedDict | 零开销，改起来快 |
| 字段少于 5 个的简单图 | TypedDict | 杀鸡不用牛刀 |
| 需要运行时数据验证 | Pydantic | 自动拦截非法数据 |
| 接收外部用户输入 | Pydantic | 防御性编程必备 |
| 嵌套复杂数据结构 | Pydantic | 嵌套模型天然支持 |
| 需要 JSON Schema 文档 | Pydantic | `model_json_schema()` 一键生成 |
| 高频调用、性能敏感 | TypedDict | 无验证开销 |
| 团队协作、长期维护 | Pydantic | 类型约束即文档 |

**经验法则**：原型用 TypedDict 快速迭代，上生产前切换到 Pydantic 加固。

---

## 2. 字段设计规范

### 2.1 命名规范

好的字段名是自文档化的——看名字就知道存什么、怎么更新。

```python
# ❌ 反面示例：含糊、缩写、易混淆
class BadState(TypedDict):
    msgs: list          # msgs 是什么？messages？
    q: str              # query？question？queue？
    res: str            # result？response？resource？
    data: dict          # 万能垃圾桶
    flag: bool          # 什么 flag？
    tmp: str            # 临时的什么？


# ✅ 正面示例：描述性、无歧义、一致
class GoodState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str                    # 用户的原始查询
    search_results: list[dict]         # 检索返回的文档列表
    generated_answer: str              # LLM 生成的回答
    is_retrieval_needed: bool          # 是否需要检索
    retry_count: int                   # 当前重试次数
```

**命名三原则**：
1. **用全称不用缩写**：`messages` 而非 `msgs`，`temperature` 而非 `temp`
2. **前缀表示来源/阶段**：`user_query`、`search_results`、`generated_answer`
3. **布尔字段用 `is_` / `has_` / `should_` 开头**：`is_retrieval_needed`

### 2.2 类型注解最佳实践

```python
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage
import operator


class WellTypedState(TypedDict):
    # ✅ 使用 Annotated 绑定 Reducer
    messages: Annotated[list[AnyMessage], add_messages]

    # ✅ 累积列表用 operator.add
    retrieved_chunks: Annotated[list[str], operator.add]

    # ✅ 可选字段用 Optional + 不提供默认值
    # （TypedDict 中 Optional 表示"值可以是 None"，不是"字段可以不传"）
    error_message: Optional[str]

    # ✅ 简单覆盖字段不加 Annotated
    current_step: str

    # ❌ 避免：用 Any 类型（丧失类型安全）
    # metadata: Any

    # ❌ 避免：用可变默认值（TypedDict 不支持默认值，这是 Pydantic 的事）
    # items: list = []
```

**关键区分**：
- **无 Annotated** → `LastValue` 通道 → 后写入的覆盖先写入的
- **`Annotated[T, reducer]`** → `BinaryOperatorAggregate` 通道 → 通过 reducer 函数合并

### 2.3 默认值设计

TypedDict 本身不支持默认值，但 LangGraph 在通道层面处理了这个问题：

```python
# TypedDict：通过 total=False 标记可选字段
class FlexibleState(TypedDict, total=False):
    """total=False 表示所有字段都是可选的"""
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str
    context: str


# Pydantic：直接用 Field 设置默认值
class DefaultState(BaseModel):
    """Pydantic 天然支持默认值"""
    messages: Annotated[list[AnyMessage], add_messages] = Field(
        default_factory=list
    )
    user_query: str = ""
    max_retries: int = Field(default=3, ge=1, le=10)
    context: Optional[str] = None
```

---

## 3. Schema 演进策略

生产系统的 Schema 不可能一成不变。关键是**如何在不破坏已有检查点的前提下演进**。

### 3.1 向后兼容的字段添加（安全操作）

```python
# === v1：初始版本 ===
class StateV1(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str


# === v2：添加新字段（向后兼容）===
class StateV2(TypedDict, total=False):
    # 保留原有字段
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str
    # 新增字段：使用 total=False 或 Optional 确保旧检查点能加载
    search_results: list[dict]       # 旧检查点没有这个字段，不会报错
    confidence_score: Optional[float] # None 表示"尚未计算"
```

**原则**：新字段必须是可选的（`Optional` 或 `total=False`），这样旧检查点反序列化时不会因为缺少字段而崩溃。

### 3.2 不兼容变更的处理

当你需要修改字段类型或删除字段时，不能直接改——旧检查点会炸。

```python
# === 策略：版本化 Schema + 适配层 ===

class StateV3(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str
    # v2 的 search_results: list[dict] 改为结构化类型
    search_results: list[dict]          # 保留旧字段（过渡期）
    structured_results: list[dict]      # 新字段，使用新结构

    # 版本标记，用于运行时判断
    schema_version: int


def migrate_node(state: dict) -> dict:
    """迁移节点：将旧格式转换为新格式"""
    version = state.get("schema_version", 1)

    if version < 3:
        # 将旧的 search_results 转换为新的 structured_results
        old_results = state.get("search_results", [])
        structured = [
            {"content": r.get("text", ""), "score": r.get("score", 0.0)}
            for r in old_results
        ]
        return {
            "structured_results": structured,
            "schema_version": 3,
        }

    return {}  # 已经是最新版本，无需迁移
```

**演进三步法**：
1. **添加新字段**（保留旧字段）→ 新旧共存
2. **迁移数据**（在节点中转换格式）→ 逐步切换
3. **移除旧字段**（确认所有检查点已迁移后）→ 清理完成

---

## 4. Input/Output Schema 分离

### 4.1 为什么需要分离

默认情况下，`StateGraph` 的输入和输出使用同一个 Schema。但在生产环境中，这会带来两个问题：

1. **信息泄露**：内部中间状态（如 `retrieved_chunks`、`debug_info`）暴露给调用者
2. **接口混乱**：调用者不知道该传哪些字段，哪些是内部使用的

```
调用者视角：
  输入：user_query, user_id
  输出：answer, sources

内部状态（不应暴露）：
  retrieved_chunks, reranked_results, prompt_template,
  llm_response_raw, retry_count, error_log ...
```

### 4.2 实现方式

LangGraph 的 `StateGraph` 构造函数支持 `input` 和 `output` 参数：

```python
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, add_messages, START, END
from langchain_core.messages import AnyMessage, HumanMessage


# === 1. 定义三个 Schema ===

class InputSchema(TypedDict):
    """对外输入接口：调用者只需要提供这些"""
    user_query: str
    user_id: str


class OutputSchema(TypedDict):
    """对外输出接口：调用者只能看到这些"""
    answer: str
    sources: list[str]


class InternalState(TypedDict, total=False):
    """完整内部状态：包含输入 + 输出 + 中间数据"""
    # 输入字段（与 InputSchema 重叠）
    user_query: str
    user_id: str
    # 中间字段（对外不可见）
    retrieved_chunks: Annotated[list[str], lambda a, b: a + b]
    reranked_results: list[dict]
    prompt_text: str
    # 输出字段（与 OutputSchema 重叠）
    answer: str
    sources: list[str]


# === 2. 创建图时指定 input/output ===

graph = StateGraph(
    InternalState,          # 完整内部状态
    input=InputSchema,      # 输入过滤：只接受这些字段
    output=OutputSchema,    # 输出过滤：只返回这些字段
)


# === 3. 节点可以访问完整内部状态 ===

def retrieve_node(state: InternalState) -> dict:
    """检索节点：读取 user_query，写入 retrieved_chunks"""
    query = state["user_query"]
    # 模拟检索
    chunks = [f"关于 '{query}' 的文档片段 1", f"关于 '{query}' 的文档片段 2"]
    return {"retrieved_chunks": chunks}


def generate_node(state: InternalState) -> dict:
    """生成节点：读取 retrieved_chunks，写入 answer + sources"""
    chunks = state.get("retrieved_chunks", [])
    return {
        "answer": f"基于 {len(chunks)} 个文档片段生成的回答",
        "sources": [f"doc_{i}" for i in range(len(chunks))],
    }


graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()

# === 4. 调用时只传 InputSchema 的字段 ===
result = app.invoke({"user_query": "什么是 RAG？", "user_id": "user_123"})

# result 只包含 OutputSchema 的字段：
# {"answer": "基于 2 个文档片段生成的回答", "sources": ["doc_0", "doc_1"]}
# retrieved_chunks、reranked_results 等中间状态不会泄露
```

**源码原理**：`StateGraph` 在编译时通过 `_pick_mapper` 对输入和输出做字段过滤。输入时只提取 `InputSchema` 中定义的字段写入通道；输出时只从通道中读取 `OutputSchema` 中定义的字段返回给调用者。

### 4.3 Schema 分离的设计原则

```
┌─────────────────────────────────────────────┐
│              InternalState                   │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
│  │  Input   │  │ Internal │  │  Output   │ │
│  │  Schema  │  │  Fields  │  │  Schema   │ │
│  │          │  │          │  │           │ │
│  │ user_id  │  │ chunks   │  │ answer    │ │
│  │ query    │  │ scores   │  │ sources   │ │
│  │          │  │ prompt   │  │           │ │
│  └──────────┘  └──────────┘  └───────────┘ │
│       ↑                            ↓        │
│    调用者传入                   调用者接收     │
└─────────────────────────────────────────────┘
```

**三条规则**：
1. `InputSchema` 的字段必须是 `InternalState` 的子集
2. `OutputSchema` 的字段必须是 `InternalState` 的子集
3. `InputSchema` 和 `OutputSchema` 可以有重叠（如 `user_id` 既是输入也是输出）

---

## 5. 在 LangGraph 中的应用

### 5.1 RAG 工作流的 Schema 设计实例

把上面所有原则综合起来，看一个完整的 RAG 工作流 Schema 设计：

```python
from typing import Annotated, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, field_validator
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage
import operator


# === 生产级 RAG 状态设计 ===

class RAGInputSchema(BaseModel):
    """输入 Schema：Pydantic 验证外部输入"""
    query: str = Field(..., min_length=1, max_length=2000)
    user_id: str = Field(..., pattern=r"^[a-zA-Z0-9_-]+$")
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


class RAGOutputSchema(TypedDict):
    """输出 Schema：TypedDict 足够（已经过内部验证）"""
    answer: str
    sources: list[str]
    confidence: float


class RAGInternalState(TypedDict, total=False):
    """内部状态：TypedDict 保持轻量"""
    # --- 输入字段 ---
    query: str
    user_id: str
    top_k: int

    # --- 检索阶段 ---
    retrieved_chunks: Annotated[list[dict], operator.add]
    retrieval_scores: list[float]

    # --- 重排序阶段 ---
    reranked_chunks: list[dict]

    # --- 生成阶段 ---
    prompt_text: str
    messages: Annotated[list[AnyMessage], add_messages]

    # --- 输出字段 ---
    answer: str
    sources: list[str]
    confidence: float

    # --- 流程控制 ---
    current_step: str
    retry_count: int
    schema_version: int
```

### 5.2 设计决策总结

| 决策点 | 选择 | 理由 |
|--------|------|------|
| InputSchema 类型 | Pydantic | 外部输入必须验证 |
| OutputSchema 类型 | TypedDict | 内部已验证，无需二次开销 |
| InternalState 类型 | TypedDict | 高频更新，性能优先 |
| 新字段策略 | `total=False` | 向后兼容旧检查点 |
| 列表字段 | `Annotated + reducer` | 多节点并行追加不丢数据 |
| 覆盖字段 | 无 Annotated | 简单直接，后写入的生效 |

---

## 小结

状态 Schema 设计是 LangGraph 最佳实践的第一步，也是最重要的一步。核心要点：

1. **TypedDict vs Pydantic**：原型用 TypedDict 快，生产用 Pydantic 稳；也可以混合使用（输入用 Pydantic 验证，内部用 TypedDict 提速）
2. **字段命名**：描述性全称 + 来源前缀 + 布尔字段 `is_/has_` 前缀
3. **Annotated 绑定 Reducer**：需要累积的字段用 `Annotated[T, reducer]`，需要覆盖的字段不加注解
4. **Schema 演进**：新字段用 Optional，不兼容变更走"添加→迁移→移除"三步法
5. **Input/Output 分离**：对外暴露最小接口，内部状态不泄露

一句话：**Schema 设计得好，后面的 Reducer、持久化、模块化都会顺理成章；Schema 设计得烂，后面每一步都在还债。**
