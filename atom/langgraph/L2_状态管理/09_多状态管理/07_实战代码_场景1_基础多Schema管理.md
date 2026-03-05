# 实战代码 - 场景1：基础多Schema管理

## 场景说明

本场景演示 LangGraph 多状态管理的核心用法——通过四种 Schema 协作，为图定义清晰的输入边界、输出边界、内部状态和节点私有状态：

1. **InputState**：图的入口白名单——外部调用者只能传入这些字段
2. **OutputState**：图的出口白名单——外部调用者只能拿到这些字段
3. **OverallState**：图的内部完整状态——所有节点共享的字段池
4. **PrivateState**：节点间的私有状态——只在特定节点之间传递，不暴露给外部

这是多状态管理的最基础模式——理解"四种 Schema 各司其职"，是构建复杂工作流的前提。

[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]
[来源: reference/source_多状态管理_01.md | LangGraph 源码分析]

---

## 核心知识点

| 知识点 | 说明 |
|--------|------|
| `StateGraph(state_schema, input_schema, output_schema)` | 构造函数的三个 Schema 参数 |
| InputState | 控制图的输入接口，外部只能传入这些字段 |
| OutputState | 控制图的输出接口，`invoke` 只返回这些字段 |
| OverallState | 图的内部完整状态，所有 Channel 的超集 |
| PrivateState | 节点函数的参数/返回类型注解，实现节点级状态分组 |
| 节点级类型推断 | LangGraph 自动从函数签名推断节点的 input_schema |

---

## 完整代码

```python
"""
LangGraph 多状态管理 - 基础多 Schema 管理
演示：Input/Output/Overall/Private Schema 四种角色协作

核心知识点：
- InputState 控制图的输入边界
- OutputState 控制图的输出边界
- OverallState 作为内部完整状态
- PrivateState 实现节点级状态分组
- 节点函数的类型注解决定它"看到"哪些字段

运行环境：Python 3.13+, langgraph
安装依赖：uv add langgraph
运行方式：python examples/langgraph/multi_schema_basic.py
"""

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


# ============================================================
# 示例1：四种 Schema 协作
# ============================================================

print("=" * 60)
print("=== 示例1：四种 Schema 协作 ===")
print("=" * 60)


# ----- 1. 定义四种 Schema -----

class InputState(TypedDict):
    """图的输入接口 - 外部调用者只能传入 user_input"""
    user_input: str


class OutputState(TypedDict):
    """图的输出接口 - 外部调用者只能拿到 graph_output"""
    graph_output: str


class OverallState(TypedDict):
    """图的内部完整状态 - 包含所有字段

    注意：OverallState 不需要继承 InputState/OutputState，
    但必须包含它们的字段（同名同类型），
    LangGraph 通过字段名匹配来建立映射关系。
    """
    user_input: str      # 来自 InputState
    graph_output: str    # 来自 OutputState
    foo: str             # 内部中间字段，外部不可见


class PrivateState(TypedDict):
    """节点间的私有状态

    这个 Schema 不在 StateGraph 构造函数中出现，
    而是通过节点函数的参数/返回类型注解来使用。
    LangGraph 会自动从 OverallState 的 Channel 中
    提取 PrivateState 需要的字段。
    """
    bar: str


# ----- 2. 定义节点函数 -----

def node_1(state: InputState) -> OverallState:
    """节点1：读取 InputState，写入 OverallState

    - 参数类型 InputState → 只能看到 user_input
    - 返回类型 OverallState → 可以写入 foo、user_input、graph_output
    - 实际只写入 foo（部分更新）
    """
    print(f"  node_1 收到: {state}")
    return {"foo": state["user_input"] + " name"}


def node_2(state: OverallState) -> PrivateState:
    """节点2：读取 OverallState，写入 PrivateState

    - 参数类型 OverallState → 能看到所有内部字段
    - 返回类型 PrivateState → 写入 bar 字段
    - bar 会被添加到 OverallState 的 Channel 中
    """
    print(f"  node_2 收到: {state}")
    return {"bar": state["foo"] + " is"}


def node_3(state: PrivateState) -> OutputState:
    """节点3：读取 PrivateState，写入 OutputState

    - 参数类型 PrivateState → 只能看到 bar
    - 返回类型 OutputState → 写入 graph_output
    """
    print(f"  node_3 收到: {state}")
    return {"graph_output": state["bar"] + " Lance"}


# ----- 3. 构建图 -----

builder = StateGraph(
    OverallState,                    # 内部完整状态
    input_schema=InputState,         # 入口白名单
    output_schema=OutputState,       # 出口白名单
)

builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

graph = builder.compile()


# ----- 4. 执行并观察 -----

print("\n--- 执行图 ---")
result = graph.invoke({"user_input": "My"})

print(f"\n最终输出: {result}")
print(f"输出类型: {type(result)}")
print(f"输出的键: {list(result.keys())}")
print("注意：输出只包含 OutputState 的字段，foo 和 bar 都被隐藏了！")


# ============================================================
# 示例2：继承方式定义 Schema
# ============================================================

print()
print("=" * 60)
print("=== 示例2：继承方式定义 Schema ===")
print("=" * 60)


class InputState2(TypedDict):
    """输入：用户问题"""
    question: str


class OutputState2(TypedDict):
    """输出：最终回答"""
    answer: str


class OverallState2(InputState2, OutputState2):
    """内部状态：继承输入和输出，再加内部字段

    使用继承的好处：
    - 自动包含 InputState2 和 OutputState2 的字段
    - 不需要手动重复声明 question 和 answer
    - 类型检查更严格
    """
    intermediate: str    # 内部中间结果


def analyze_node(state: InputState2) -> dict:
    """分析节点：只读取问题"""
    print(f"  analyze 收到: {state}")
    question = state["question"]
    return {"intermediate": f"分析结果: {question} → 关键词提取完成"}


def answer_node(state: OverallState2) -> dict:
    """回答节点：读取问题和中间结果，生成回答"""
    print(f"  answer 收到: {state}")
    return {
        "answer": f"回答: 基于 [{state['intermediate']}] 生成的答案",
    }


builder2 = StateGraph(
    OverallState2,
    input_schema=InputState2,
    output_schema=OutputState2,
)

builder2.add_node("analyze", analyze_node)
builder2.add_node("answer", answer_node)

builder2.add_edge(START, "analyze")
builder2.add_edge("analyze", "answer")
builder2.add_edge("answer", END)

graph2 = builder2.compile()

print("\n--- 执行图 ---")
result2 = graph2.invoke({"question": "什么是 RAG？"})
print(f"\n最终输出: {result2}")
print("注意：intermediate 字段不在输出中！")


# ============================================================
# 示例3：stream 观察每个节点的状态变化
# ============================================================

print()
print("=" * 60)
print("=== 示例3：stream 观察状态流转 ===")
print("=" * 60)

print("\n--- 逐步观察示例1的图 ---")
for step in graph.stream({"user_input": "My"}):
    for node_name, node_output in step.items():
        print(f"\n  节点 [{node_name}] 返回:")
        for key, value in node_output.items():
            print(f"    {key}: {value!r}")


# ============================================================
# 示例4：验证输入边界
# ============================================================

print()
print("=" * 60)
print("=== 示例4：验证输入边界 ===")
print("=" * 60)

# 传入 InputState 之外的字段
print("\n--- 传入额外字段 foo ---")
try:
    result_extra = graph.invoke({"user_input": "My", "foo": "extra"})
    print(f"结果: {result_extra}")
    print("说明：额外字段被静默忽略，不会报错")
except Exception as e:
    print(f"错误: {type(e).__name__}: {e}")

# 缺少必需字段
print("\n--- 缺少必需字段 user_input ---")
try:
    result_missing = graph.invoke({})
    print(f"结果: {result_missing}")
except Exception as e:
    print(f"错误: {type(e).__name__}: {e}")


# ============================================================
# 示例5：对比——不使用 input/output schema
# ============================================================

print()
print("=" * 60)
print("=== 示例5：对比——不使用 input/output schema ===")
print("=" * 60)


class SimpleState(TypedDict):
    """单一状态：所有字段对外可见"""
    user_input: str
    foo: str
    bar: str
    graph_output: str


def simple_node_1(state: SimpleState) -> dict:
    return {"foo": state["user_input"] + " name"}


def simple_node_2(state: SimpleState) -> dict:
    return {"bar": state["foo"] + " is"}


def simple_node_3(state: SimpleState) -> dict:
    return {"graph_output": state["bar"] + " Lance"}


simple_builder = StateGraph(SimpleState)  # 没有 input/output schema
simple_builder.add_node("node_1", simple_node_1)
simple_builder.add_node("node_2", simple_node_2)
simple_builder.add_node("node_3", simple_node_3)
simple_builder.add_edge(START, "node_1")
simple_builder.add_edge("node_1", "node_2")
simple_builder.add_edge("node_2", "node_3")
simple_builder.add_edge("node_3", END)

simple_graph = simple_builder.compile()

simple_result = simple_graph.invoke({
    "user_input": "My",
    "foo": "",
    "bar": "",
    "graph_output": "",
})

print(f"\n无 Schema 分离的输出: {simple_result}")
print(f"输出的键: {list(simple_result.keys())}")
print("问题：所有内部字段（foo, bar）都暴露给了调用者！")

print(f"\n有 Schema 分离的输出: {result}")
print(f"输出的键: {list(result.keys())}")
print("优势：只暴露 graph_output，内部实现细节被隐藏")


# ============================================================
# 示例6：RAG 场景的多 Schema 设计
# ============================================================

print()
print("=" * 60)
print("=== 示例6：RAG 场景的多 Schema 设计 ===")
print("=" * 60)


class RAGInput(TypedDict):
    """RAG 输入：用户只需要传查询"""
    query: str


class RAGOutput(TypedDict):
    """RAG 输出：用户只拿到回答和来源"""
    answer: str
    sources: list[str]


class RAGInternal(TypedDict):
    """RAG 内部状态：包含所有中间数据"""
    query: str
    answer: str
    sources: list[str]
    # 以下是内部中间状态，对外不可见
    chunks: list[str]
    rerank_scores: list[float]
    prompt: str


def retrieve_node(state: RAGInput) -> dict:
    """检索节点：只需要 query"""
    query = state["query"]
    # 模拟向量检索
    chunks = [
        f"文档片段1: {query}相关内容A",
        f"文档片段2: {query}相关内容B",
        f"文档片段3: {query}相关内容C",
    ]
    scores = [0.95, 0.87, 0.72]
    print(f"  retrieve: 检索到 {len(chunks)} 个片段")
    return {"chunks": chunks, "rerank_scores": scores}


def generate_node(state: RAGInternal) -> dict:
    """生成节点：需要 query + chunks"""
    query = state["query"]
    chunks = state["chunks"]
    scores = state["rerank_scores"]

    # 模拟 Prompt 构建
    prompt = f"基于以下文档回答问题：\n问题：{query}\n文档：{chunks[0]}"

    # 模拟 LLM 生成
    answer = f"根据检索结果，{query}的答案是..."
    sources = [f"doc_{i}" for i, s in enumerate(scores) if s > 0.8]

    print(f"  generate: 使用 {len(sources)} 个高质量来源生成回答")
    return {
        "prompt": prompt,
        "answer": answer,
        "sources": sources,
    }


rag_builder = StateGraph(
    RAGInternal,
    input_schema=RAGInput,
    output_schema=RAGOutput,
)

rag_builder.add_node("retrieve", retrieve_node)
rag_builder.add_node("generate", generate_node)

rag_builder.add_edge(START, "retrieve")
rag_builder.add_edge("retrieve", "generate")
rag_builder.add_edge("generate", END)

rag_graph = rag_builder.compile()

print("\n--- 执行 RAG 图 ---")
rag_result = rag_graph.invoke({"query": "什么是向量数据库"})
print(f"\nRAG 输出: {rag_result}")
print(f"输出的键: {list(rag_result.keys())}")
print("注意：chunks、rerank_scores、prompt 等内部字段全部被隐藏！")
print("调用者只看到 answer 和 sources，这就是多 Schema 管理的价值。")
```

---

## 预期输出

```text
============================================================
=== 示例1：四种 Schema 协作 ===
============================================================

--- 执行图 ---
  node_1 收到: {'user_input': 'My'}
  node_2 收到: {'user_input': 'My', 'graph_output': '', 'foo': 'My name'}
  node_3 收到: {'bar': 'My name is'}

最终输出: {'graph_output': 'My name is Lance'}
输出类型: <class 'dict'>
输出的键: ['graph_output']
注意：输出只包含 OutputState 的字段，foo 和 bar 都被隐藏了！

============================================================
=== 示例2：继承方式定义 Schema ===
============================================================

--- 执行图 ---
  analyze 收到: {'question': '什么是 RAG？'}
  answer 收到: {'question': '什么是 RAG？', 'answer': '', 'intermediate': '分析结果: 什么是 RAG？ → 关键词提取完成'}

最终输出: {'answer': '回答: 基于 [分析结果: 什么是 RAG？ → 关键词提取完成] 生成的答案'}
注意：intermediate 字段不在输出中！

============================================================
=== 示例3：stream 观察状态流转 ===
============================================================

--- 逐步观察示例1的图 ---

  节点 [node_1] 返回:
    foo: 'My name'

  节点 [node_2] 返回:
    bar: 'My name is'

  节点 [node_3] 返回:
    graph_output: 'My name is Lance'

============================================================
=== 示例4：验证输入边界 ===
============================================================

--- 传入额外字段 foo ---
结果: {'graph_output': 'My name is Lance'}
说明：额外字段被静默忽略，不会报错

--- 缺少必需字段 user_input ---
错误: KeyError: 'user_input'

============================================================
=== 示例5：对比——不使用 input/output schema ===
============================================================

无 Schema 分离的输出: {'user_input': 'My', 'foo': 'My name', 'bar': 'My name is', 'graph_output': 'My name is Lance'}
输出的键: ['user_input', 'foo', 'bar', 'graph_output']
问题：所有内部字段（foo, bar）都暴露给了调用者！

有 Schema 分离的输出: {'graph_output': 'My name is Lance'}
输出的键: ['graph_output']
优势：只暴露 graph_output，内部实现细节被隐藏

============================================================
=== 示例6：RAG 场景的多 Schema 设计 ===
============================================================

--- 执行 RAG 图 ---
  retrieve: 检索到 3 个片段
  generate: 使用 2 个高质量来源生成回答

RAG 输出: {'answer': '根据检索结果，什么是向量数据库的答案是...', 'sources': ['doc_0', 'doc_1']}
输出的键: ['answer', 'sources']
注意：chunks、rerank_scores、prompt 等内部字段全部被隐藏！
调用者只看到 answer 和 sources，这就是多 Schema 管理的价值。
```

---

## 代码解析

### 1. 四种 Schema 的角色分工

```
外部调用者                    图的内部
                    ┌─────────────────────────────┐
                    │                             │
InputState ──────►  │  OverallState（完整状态）     │  ──────► OutputState
{user_input}        │  {user_input, foo,          │          {graph_output}
                    │   graph_output, bar}         │
                    │                             │
                    │  PrivateState（节点私有）     │
                    │  {bar} ← 只在 node_2→node_3  │
                    └─────────────────────────────┘
```

- **InputState**：图的"前门"，只允许 `user_input` 进入
- **OutputState**：图的"后门"，只允许 `graph_output` 出去
- **OverallState**：图的"内部仓库"，所有字段都在这里
- **PrivateState**：节点间的"悄悄话"，通过函数类型注解实现

### 2. 节点级类型推断机制

```python
def node_1(state: InputState) -> OverallState:
    #         ↑ 参数类型注解                ↑ 返回类型注解
    #         LangGraph 自动推断：          LangGraph 知道：
    #         这个节点只需要 user_input      这个节点可以写入 OverallState 的字段
```

LangGraph 源码中的关键逻辑：
- 检查节点函数第一个参数的类型注解
- 如果是带 `__annotations__` 的类（TypedDict），自动作为节点的 `input_schema`
- 节点只会收到该 Schema 中声明的字段

### 3. PrivateState 的工作原理

PrivateState 并不是 StateGraph 构造函数的参数，而是通过节点函数的类型注解来使用：

```python
# node_2 返回 PrivateState → bar 字段被写入 OverallState 的 Channel
def node_2(state: OverallState) -> PrivateState:
    return {"bar": state["foo"] + " is"}

# node_3 参数是 PrivateState → 只能看到 bar 字段
def node_3(state: PrivateState) -> OutputState:
    return {"graph_output": state["bar"] + " Lance"}
```

关键点：`bar` 字段虽然不在 OverallState 的定义中，但 LangGraph 会自动为它创建 Channel。所有通过节点返回值写入的字段，都会被注册到图的 Channel 池中。

### 4. 继承 vs 独立定义

```python
# 方式1：独立定义（字段名必须匹配）
class InputState(TypedDict):
    user_input: str

class OverallState(TypedDict):
    user_input: str      # 必须和 InputState 同名同类型
    foo: str

# 方式2：继承（自动包含父类字段）
class OverallState2(InputState2, OutputState2):
    intermediate: str    # 只需要声明额外字段
```

两种方式都可以，继承方式更简洁，但独立定义更灵活（允许 OverallState 中的字段类型与 InputState 不完全一致）。

---

## 关键要点

1. **四种 Schema 各司其职**：InputState 管入口、OutputState 管出口、OverallState 管内部、PrivateState 管节点间私有通信
2. **节点函数的类型注解决定可见性**：参数类型决定"能看到什么"，返回类型决定"能写入什么"
3. **输出过滤是自动的**：`invoke` 的返回值自动只包含 OutputState 的字段，无需手动过滤
4. **PrivateState 不在构造函数中**：它通过节点函数的类型注解隐式工作
5. **额外输入字段被静默忽略**：传入 InputState 之外的字段不会报错，但也不会生效

---

## 扩展思考

1. **如果 OverallState 中缺少 InputState 的某个字段会怎样？** 图编译时不会报错，但运行时该字段不会被传入内部状态，节点读取时会得到默认值或报 KeyError。

2. **PrivateState 的字段如果和 OverallState 的字段重名会怎样？** 它们共享同一个 Channel。这意味着 PrivateState 的 `bar` 和 OverallState 的 `bar`（如果有的话）是同一个字段。

3. **能否为不同节点定义不同的 PrivateState？** 可以。每个节点函数可以有不同的参数类型注解，LangGraph 会为每个节点独立推断 input_schema。

4. **在 RAG 系统中，哪些字段适合放在 OutputState？** 只有最终用户需要看到的字段：answer、sources、confidence_score 等。embedding_cache、raw_chunks、rerank_scores 等中间数据应该留在 OverallState 内部。

---

**版本信息**
- 文档版本: v1.0
- 最后更新: 2026-02-27
- LangGraph 版本: main (2026-02-17)

**引用来源**
- LangGraph 源码：`libs/langgraph/langgraph/graph/state.py`
- Context7 官方文档（2026-02-27）
