# 核心概念 01：Input/Output Schema 分离

> 通过为图定义独立的输入 Schema 和输出 Schema，精确控制图的边界——外部只看到你想暴露的字段，内部状态完全隐藏。

---

## 引用来源

**源码分析**:
- `libs/langgraph/langgraph/graph/state.py` — StateGraph 构造函数、`_add_schema()` 合并机制

**官方文档**:
- Context7 LangGraph 文档 (2026-02-27)

---

## 一句话定义

**Input/Output Schema 分离是 LangGraph 的图边界控制机制：`input_schema` 决定图接受哪些字段，`output_schema` 决定图返回哪些字段，内部节点可以自由读写所有字段。**

---

## 为什么需要这个？

### 问题场景

假设你在构建一个 RAG 系统，内部状态包含：

```python
class State(TypedDict):
    question: str           # 用户输入
    retrieved_docs: list    # 检索到的文档（中间产物）
    rerank_scores: list     # 重排序分数（中间产物）
    debug_info: dict        # 调试信息（内部使用）
    answer: str             # 最终回答
```

如果不做分离，调用方需要：
1. 传入所有字段（包括不需要的 `retrieved_docs`、`rerank_scores`）
2. 收到所有字段（包括不想暴露的 `debug_info`）

这就像一个 REST API 把所有内部字段都暴露给前端——既不安全，也不优雅。

### 解决方案

```python
class InputState(TypedDict):
    question: str       # 调用方只需传这一个字段

class OutputState(TypedDict):
    answer: str         # 调用方只收到这一个字段

# 内部状态包含所有字段
class OverallState(TypedDict):
    question: str
    retrieved_docs: list
    rerank_scores: list
    debug_info: dict
    answer: str

builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
```

现在调用方只需要 `graph.invoke({"question": "什么是RAG？"})`，只收到 `{"answer": "RAG是..."}`。

---

## 双重类比：先建立直觉

### 前端类比：React 组件的 Props 与内部 State

```
React 组件                              LangGraph 图
─────────────                           ─────────────
// Props = 外部传入（InputSchema）       class InputState(TypedDict):
interface Props {                           question: str
  question: string
}                                       class OutputState(TypedDict):
                                            answer: str
// 内部 State = 不暴露给外部
const [docs, setDocs] = useState([])    class OverallState(TypedDict):
const [scores, setScores] = useState([])    question: str
                                            retrieved_docs: list
// return = 外部看到的（OutputSchema）       rerank_scores: list
return <div>{answer}</div>                  answer: str
```

- `Props` 对应 `input_schema`：外部传入的参数
- 内部 `useState` 对应 `OverallState` 中的私有字段：外部看不到
- `return JSX` 对应 `output_schema`：只暴露最终结果

### 日常生活类比：餐厅点餐

- **菜单**（InputSchema）：顾客只需要说"来一份宫保鸡丁"
- **厨房内部**（OverallState）：切菜、腌制、炒制、调味——顾客不需要知道
- **上菜**（OutputSchema）：顾客只收到一盘成品菜

你不会让顾客填一张表格写明"鸡肉切丁大小、油温多少度、花椒放几粒"——那是厨房内部的事。

---

## 详细解释

### 1. StateGraph 构造函数的四个 Schema 参数

```python
# 来源: libs/langgraph/langgraph/graph/state.py
def __init__(
    self,
    state_schema: type[StateT],           # 内部完整状态（必填）
    context_schema: type[ContextT] | None = None,  # 运行时上下文（可选）
    *,
    input_schema: type[InputT] | None = None,      # 输入 Schema（可选）
    output_schema: type[OutputT] | None = None,     # 输出 Schema（可选）
) -> None:
```

**默认行为**：
- 不指定 `input_schema` → 默认等于 `state_schema`（接受所有字段）
- 不指定 `output_schema` → 默认等于 `state_schema`（返回所有字段）
- 这就是为什么之前的简单示例不需要关心这个——默认就是"全进全出"

### 2. 三个 Schema 的职责划分

```
                    ┌─────────────────────────────────────┐
                    │           LangGraph 图               │
                    │                                     │
  InputState ──────→│  START 节点只写入 InputState 的字段   │
  (图的入口)        │                                     │
                    │  内部节点可以读写 OverallState       │──────→ OutputState
                    │  的所有字段                          │        (图的出口)
                    │                                     │
                    │  END 节点只返回 OutputState 的字段    │
                    └─────────────────────────────────────┘
```

关键规则：
- **START 节点**：只将输入数据写入 `input_schema` 定义的字段
- **普通节点**：可以读写 `state_schema`（OverallState）的所有字段
- **END 节点**：只返回 `output_schema` 定义的字段

### 3. 多 Schema 合并机制

所有 Schema 的字段最终合并到同一个 Channel 池中。`_add_schema()` 方法负责这个合并：

```python
# 来源: libs/langgraph/langgraph/graph/state.py（简化版）
def _add_schema(self, schema: type) -> None:
    """将一个 Schema 的字段注册到统一的 Channel 池"""
    channels, managed, type_hints = _get_channels(schema)

    for key, channel in channels.items():
        if key in self.channels:
            # 同名字段已存在 → 检查兼容性
            if isinstance(channel, LastValue):
                pass  # LastValue 与任何 Channel 兼容
            elif isinstance(self.channels[key], LastValue):
                self.channels[key] = channel  # 用更具体的 Channel 替换
            elif channel != self.channels[key]:
                raise ValueError(f"Channel '{key}' 定义冲突")
        else:
            self.channels[key] = channel  # 新字段，直接注册

    self.schemas[schema] = type_hints
```

**兼容性规则**：
- 同名字段在不同 Schema 中必须兼容
- `LastValue`（无 Reducer）与任何 Channel 兼容
- 两个不同的 Reducer 绑定到同名字段 → 报错

### 4. 常见的继承模式

最推荐的做法是让 `OverallState` 继承 `InputState` 和 `OutputState`：

```python
class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str

# OverallState 继承两者，确保字段名一致
class OverallState(InputState, OutputState):
    # 额外的内部字段
    retrieved_docs: list[str]
    debug_info: dict
```

这样做的好处：
- 字段名自动一致，不会出现拼写错误
- 类型检查器可以验证字段类型兼容性
- 代码意图清晰：OverallState 是 InputState + OutputState + 私有字段

---

## 代码示例

### 示例 1：基础 Input/Output 分离

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# ========== 定义三层 Schema ==========
class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str

class OverallState(InputState, OutputState):
    """内部状态：包含输入、输出和中间字段"""
    pass

# ========== 定义节点 ==========
def answer_node(state: InputState):
    """接收问题，返回回答"""
    return {"answer": "bye", "question": state["question"]}

# ========== 构建图 ==========
builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
builder.add_node(answer_node)
builder.add_edge(START, "answer_node")
builder.add_edge("answer_node", END)

graph = builder.compile()

# ========== 执行 ==========
result = graph.invoke({"question": "hi"})
print(result)
# {'answer': 'bye'}  ← 只有 OutputState 的字段！question 被过滤掉了
```

注意：`answer_node` 返回了 `question` 和 `answer` 两个字段，但最终输出只包含 `answer`——因为 `output_schema=OutputState` 只定义了 `answer`。

### 示例 2：四 Schema 模式（含 PrivateState）

这是一个更复杂的模式，展示了四种 Schema 如何协作：

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# ========== 四种 Schema ==========
class InputState(TypedDict):
    """图的入口：调用方传入的字段"""
    user_input: str

class OutputState(TypedDict):
    """图的出口：调用方收到的字段"""
    graph_output: str

class OverallState(TypedDict):
    """内部完整状态：所有字段的超集"""
    foo: str
    user_input: str
    graph_output: str

class PrivateState(TypedDict):
    """节点私有状态：只在特定节点之间传递"""
    bar: str

# ========== 节点函数 ==========
def node_1(state: InputState) -> OverallState:
    """第一个节点：读取输入，写入内部字段"""
    return {"foo": state["user_input"] + " name"}

def node_2(state: OverallState) -> PrivateState:
    """第二个节点：读取内部字段，写入私有字段"""
    return {"bar": state["foo"] + " is"}

def node_3(state: PrivateState) -> OutputState:
    """第三个节点：读取私有字段，写入输出字段"""
    return {"graph_output": state["bar"] + " Lance"}

# ========== 构建图 ==========
builder = StateGraph(
    OverallState,
    input_schema=InputState,
    output_schema=OutputState,
)
builder.add_node(node_1)
builder.add_node(node_2)
builder.add_node(node_3)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

graph = builder.compile()

# ========== 执行 ==========
result = graph.invoke({"user_input": "My"})
print(result)
# {'graph_output': 'My name is Lance'}
```

**数据流追踪**：

```
invoke({"user_input": "My"})
  ↓ START 只写入 InputState 的字段
  ↓
┌──────────────────────────────────────────────┐
│ Channels: {user_input: "My", foo: ?, graph_output: ?, bar: ?} │
└──────────────────────────────────────────────┘
  ↓
node_1(InputState) → {"foo": "My name"}
  ↓ 写入 foo Channel
  ↓
┌──────────────────────────────────────────────┐
│ Channels: {user_input: "My", foo: "My name", ...}            │
└──────────────────────────────────────────────┘
  ↓
node_2(OverallState) → {"bar": "My name is"}
  ↓ 写入 bar Channel
  ↓
┌──────────────────────────────────────────────┐
│ Channels: {user_input: "My", foo: "My name", bar: "My name is", ...} │
└──────────────────────────────────────────────┘
  ↓
node_3(PrivateState) → {"graph_output": "My name is Lance"}
  ↓ 写入 graph_output Channel
  ↓
┌──────────────────────────────────────────────┐
│ Channels: {..., graph_output: "My name is Lance"}            │
└──────────────────────────────────────────────┘
  ↓ END 只返回 OutputState 的字段
  ↓
{"graph_output": "My name is Lance"}
```

### 示例 3：RAG 场景的 Schema 分离

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ========== Schema 定义 ==========
class RAGInput(TypedDict):
    """用户只需传入问题"""
    question: str

class RAGOutput(TypedDict):
    """用户只收到回答"""
    answer: str

class RAGState(TypedDict):
    """内部完整状态"""
    question: str
    answer: str
    # 以下字段对调用方不可见
    chunks: list[str]
    rerank_scores: list[float]
    prompt_tokens: int
    completion_tokens: int

# ========== 节点 ==========
def retrieve(state: RAGState) -> dict:
    # 模拟检索
    return {"chunks": [f"文档片段关于: {state['question']}"]}

def rerank(state: RAGState) -> dict:
    # 模拟重排序
    return {"rerank_scores": [0.95], "chunks": state["chunks"][:3]}

def generate(state: RAGState) -> dict:
    context = "\n".join(state["chunks"])
    answer = f"基于检索结果回答: {context}"
    return {
        "answer": answer,
        "prompt_tokens": 150,
        "completion_tokens": 50,
    }

# ========== 构建图 ==========
builder = StateGraph(RAGState, input_schema=RAGInput, output_schema=RAGOutput)
builder.add_node("retrieve", retrieve)
builder.add_node("rerank", rerank)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "rerank")
builder.add_edge("rerank", "generate")
builder.add_edge("generate", END)

graph = builder.compile()

# ========== 调用 ==========
result = graph.invoke({"question": "什么是向量数据库？"})
print(result)
# {'answer': '基于检索结果回答: 文档片段关于: 什么是向量数据库？'}
# 注意：chunks、rerank_scores、prompt_tokens、completion_tokens 都不会出现在输出中
```

---

## 源码解析

### 关键源码 1：Schema 注册流程

当你创建 `StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)` 时，构造函数内部会依次注册三个 Schema：

```python
# 来源: libs/langgraph/langgraph/graph/state.py（简化版）
class StateGraph:
    def __init__(self, state_schema, *, input_schema=None, output_schema=None):
        # 1. 注册主 Schema
        self._add_schema(state_schema)

        # 2. 注册 input_schema（默认等于 state_schema）
        self.input_schema = input_schema or state_schema
        if input_schema:
            self._add_schema(input_schema)

        # 3. 注册 output_schema（默认等于 state_schema）
        self.output_schema = output_schema or state_schema
        if output_schema:
            self._add_schema(output_schema)
```

### 关键源码 2：输入过滤（START 节点）

START 节点在写入时，只写入 `input_schema` 定义的字段：

```python
# 来源: libs/langgraph/langgraph/graph/state.py（简化版）
# 编译时，START 节点的 ChannelWrite 只包含 input_schema 的字段
input_channels = list(self.schemas[self.input_schema].keys())
# START → ChannelWrite(input_channels)
```

### 关键源码 3：输出过滤（END 节点）

图执行完毕后，只返回 `output_schema` 定义的字段：

```python
# 来源: libs/langgraph/langgraph/graph/state.py（简化版）
# 编译时，输出 ChannelRead 只读取 output_schema 的字段
output_channels = list(self.schemas[self.output_schema].keys())
# END → ChannelRead(output_channels)
```

### Channel 池的统一视图

无论定义了多少个 Schema，所有字段最终都在同一个 Channel 池中：

```
InputState:   {question}           ──┐
OutputState:  {answer}             ──┼──→ Channel 池: {question, answer, chunks, rerank_scores, ...}
OverallState: {question, answer,   ──┘
               chunks, rerank_scores,
               prompt_tokens, completion_tokens}
```

节点在执行时，通过 ChannelRead 从这个统一的 Channel 池中读取数据，通过 ChannelWrite 写入数据。Schema 只是一个"视图"——决定了哪些 Channel 对外可见。

---

## 在 LangGraph 开发中的应用

### 场景 1：API 服务化

当你把 LangGraph 图包装成 FastAPI 接口时，Input/Output Schema 直接对应请求体和响应体：

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/ask")
async def ask(request: RAGInput) -> RAGOutput:
    result = graph.invoke(request)
    return result
```

`input_schema` 和 `output_schema` 天然就是 API 的契约。

### 场景 2：子图作为黑盒

当一个图被另一个图作为子图使用时，Input/Output Schema 定义了子图的"接口"：

```python
# 父图只需要知道子图接受 question、返回 answer
# 不需要关心子图内部有多少中间状态
parent_builder.add_node("rag", rag_graph)  # rag_graph 的 I/O Schema 就是接口
```

### 场景 3：安全隔离

防止敏感的内部状态泄露给调用方：

```python
class InternalState(TypedDict):
    api_key: str          # 不应暴露
    raw_response: dict    # 不应暴露
    user_query: str
    final_answer: str

class SafeOutput(TypedDict):
    final_answer: str     # 只暴露最终结果

builder = StateGraph(InternalState, output_schema=SafeOutput)
```

---

## 常见问题

### Q1：不指定 input_schema 和 output_schema 会怎样？

默认等于 `state_schema`。图接受所有字段、返回所有字段——和之前的简单用法完全一样。

### Q2：InputState 和 OutputState 可以有重叠字段吗？

可以。比如 `question` 同时出现在 InputState 和 OutputState 中，那么输出也会包含 `question`。

### Q3：节点可以写入不在 OverallState 中的字段吗？

不行。所有字段必须在某个已注册的 Schema 中定义。写入未注册的字段会被忽略或报错。

### Q4：PrivateState 需要注册到 StateGraph 吗？

PrivateState 的字段必须存在于 `state_schema`（OverallState）中。PrivateState 本身只是节点函数的类型注解，用于限制节点"看到"的字段子集——它不需要单独注册到 StateGraph。

---

## 小结

1. **三个 Schema 各司其职**：`input_schema` 控制入口、`output_schema` 控制出口、`state_schema` 定义完整内部状态
2. **默认行为**：不指定时，input/output 都等于 state_schema（全进全出）
3. **统一 Channel 池**：所有 Schema 的字段合并到同一个 Channel 池，Schema 只是"视图"
4. **推荐继承模式**：`OverallState(InputState, OutputState)` 确保字段名一致
5. **核心价值**：像 API 一样定义图的边界——隐藏内部实现，只暴露必要接口

---

**版本信息**
- 文档版本: v1.0
- 最后更新: 2026-02-27
- LangGraph 版本: main (2026-02-27)

**引用来源**
- LangGraph 源码：`libs/langgraph/langgraph/graph/state.py`
- Context7 官方文档（2026-02-27）
