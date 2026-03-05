# 实战代码 - 场景1：生产级状态 Schema 设计

## 场景说明

本场景演示如何为 RAG 多轮对话系统设计一套生产级的状态 Schema。你将看到：

1. 使用 `TypedDict + Annotated` 定义核心内部状态
2. 消息字段绑定 `add_messages` reducer 实现追加合并
3. 检索结果使用自定义 reducer（带上限控制，防止状态膨胀）
4. 配置项使用 LastValue 语义（无注解，后写入覆盖）
5. 通过 `input` / `output` Schema 分离对外接口，隐藏内部中间状态

**核心原则：** 好的 Schema 设计让节点职责清晰、类型安全、接口最小化。输入用 InputSchema 过滤，输出用 OutputSchema 过滤，内部状态对调用者完全不可见。

[来源: 03_核心概念_1_状态Schema设计原则.md]

---

## 设计思路

```
调用者 ──→ InputSchema(messages) ──→ ┌──────────────────────┐
                                      │    RAGChatState       │
                                      │  messages (add)       │
                                      │  retrieved_docs (cap) │
                                      │  current_query (last) │
                                      │  turn_count (last)    │
                                      │  config (last)        │
                                      │  answer (last)        │
                                      └──────────────────────┘
调用者 ←── OutputSchema(messages, answer) ←──┘
```

- `messages`：用 `add_messages` 追加，保留完整对话历史
- `retrieved_docs`：自定义 reducer，只保留最近 N 条，防止无限膨胀
- `current_query` / `turn_count` / `config`：无注解 → LastValue 覆盖更新
- `answer`：每轮生成新回答，覆盖即可

---

## 完整代码

```python
"""
生产级 RAG 多轮对话系统 - 状态 Schema 设计
演示：TypedDict + Annotated + 自定义 Reducer + Input/Output Schema 分离

核心知识点：
- add_messages reducer：消息列表追加合并
- 自定义 capped reducer：带上限的列表累积
- LastValue 语义：无注解字段后写入覆盖
- Input/Output Schema：对外接口最小化

运行环境：Python 3.13+, langgraph
安装依赖：uv add langgraph
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage


# ============================================================
# 1. 自定义 Reducer —— 带上限的文档列表
# ============================================================

MAX_DOCS = 5  # 最多保留 5 条检索结果

def capped_docs_reducer(existing: list, new: list) -> list:
    """带上限的文档列表 reducer

    为什么需要上限？
    多轮对话中每轮都会检索新文档，如果无限累积，
    状态会越来越大，序列化/反序列化开销也会增长。
    只保留最近 MAX_DOCS 条，旧的自动淘汰。
    """
    combined = existing + new
    return combined[-MAX_DOCS:]  # 只保留最新的 N 条


# ============================================================
# 2. 核心状态 Schema —— 完整内部状态
# ============================================================

class RAGChatState(TypedDict, total=False):
    """RAG 多轮对话的完整内部状态

    字段设计决策：
    - messages: Annotated + add_messages → 追加合并，保留对话历史
    - retrieved_docs: Annotated + capped_reducer → 累积但有上限
    - current_query: 无注解 → LastValue，每轮覆盖为最新查询
    - turn_count: 无注解 → LastValue，每轮覆盖为最新轮次
    - config: 无注解 → LastValue，配置变更直接覆盖
    - answer: 无注解 → LastValue，每轮生成新回答
    """
    messages: Annotated[list, add_messages]
    retrieved_docs: Annotated[list[dict], capped_docs_reducer]
    current_query: str
    turn_count: int
    config: dict       # 系统配置：top_k、temperature 等
    answer: str


# ============================================================
# 3. Input/Output Schema —— 对外接口最小化
# ============================================================

class InputSchema(TypedDict):
    """调用者只需要传消息列表"""
    messages: Annotated[list, add_messages]


class OutputSchema(TypedDict):
    """调用者只能看到消息和回答，看不到 retrieved_docs 等内部状态"""
    messages: Annotated[list, add_messages]
    answer: str


# ============================================================
# 4. 节点实现 —— 模拟 RAG 管道
# ============================================================

def extract_query(state: RAGChatState) -> dict:
    """从最新消息中提取用户查询，更新轮次计数"""
    messages = state.get("messages", [])
    # 取最后一条消息的内容作为当前查询
    last_msg = messages[-1] if messages else None
    query = last_msg.content if last_msg else ""
    turn = state.get("turn_count", 0) + 1

    print(f"[extract_query] 第 {turn} 轮, 查询: '{query}'")
    return {
        "current_query": query,
        "turn_count": turn,
        "config": {"top_k": 3, "temperature": 0.7},
    }


def retrieve(state: RAGChatState) -> dict:
    """模拟文档检索 —— 根据查询返回相关文档"""
    query = state.get("current_query", "")
    top_k = state.get("config", {}).get("top_k", 3)

    # 模拟检索结果（实际项目中这里调用向量数据库）
    mock_docs = [
        {"content": f"[文档A] 关于'{query}'的核心概念解释", "score": 0.95},
        {"content": f"[文档B] '{query}'的实际应用案例", "score": 0.87},
        {"content": f"[文档C] '{query}'的常见误区", "score": 0.78},
    ][:top_k]

    print(f"[retrieve] 检索到 {len(mock_docs)} 篇文档")
    # 返回的 retrieved_docs 会经过 capped_docs_reducer 合并
    return {"retrieved_docs": mock_docs}


def generate(state: RAGChatState) -> dict:
    """模拟回答生成 —— 基于检索文档生成回答"""
    docs = state.get("retrieved_docs", [])
    query = state.get("current_query", "")
    turn = state.get("turn_count", 0)

    # 拼接上下文（实际项目中这里调用 LLM）
    context = " | ".join(d["content"] for d in docs[-3:])
    answer = f"[第{turn}轮回答] 关于'{query}': {context}"

    print(f"[generate] 生成回答, 使用 {len(docs)} 篇文档")
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


# ============================================================
# 5. 构建图 —— 使用 Input/Output Schema 分离
# ============================================================

print("=" * 55)
print("=== 构建 RAG 多轮对话图 ===")
print("=" * 55)

builder = StateGraph(
    RAGChatState,           # 完整内部状态
    input=InputSchema,      # 输入过滤：只接受 messages
    output=OutputSchema,    # 输出过滤：只返回 messages + answer
)

builder.add_node("extract_query", extract_query)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

builder.add_edge(START, "extract_query")
builder.add_edge("extract_query", "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()
print("图编译完成: extract_query -> retrieve -> generate\n")


# ============================================================
# 6. 运行演示 —— 模拟两轮对话
# ============================================================

# --- 第 1 轮 ---
print("=" * 55)
print("=== 第 1 轮对话 ===")
print("=" * 55)

result1 = graph.invoke({
    "messages": [HumanMessage(content="什么是 RAG？")],
})

print(f"\n返回的键: {list(result1.keys())}")
print(f"answer: {result1['answer']}")
print(f"messages 数量: {len(result1['messages'])}")
# 注意：retrieved_docs、turn_count、config 不在返回结果中！
print(f"retrieved_docs 是否泄露: {'retrieved_docs' in result1}")

# --- 第 2 轮 ---
print(f"\n{'=' * 55}")
print("=== 第 2 轮对话 ===")
print("=" * 55)

result2 = graph.invoke({
    "messages": [HumanMessage(content="RAG 有哪些优化策略？")],
})

print(f"\nanswer: {result2['answer']}")
print(f"messages 数量: {len(result2['messages'])}")
print(f"retrieved_docs 是否泄露: {'retrieved_docs' in result2}")


# ============================================================
# 7. 验证 capped_docs_reducer 的效果
# ============================================================

print(f"\n{'=' * 55}")
print("=== 验证 capped_docs_reducer ===")
print("=" * 55)

existing = [{"content": f"旧文档{i}", "score": 0.5} for i in range(4)]
new = [{"content": f"新文档{i}", "score": 0.9} for i in range(3)]
merged = capped_docs_reducer(existing, new)

print(f"已有文档: {len(existing)} 篇")
print(f"新增文档: {len(new)} 篇")
print(f"合并后 (上限={MAX_DOCS}): {len(merged)} 篇")
print(f"保留的文档: {[d['content'] for d in merged]}")
```

---

## 预期输出

```text
=======================================================
=== 构建 RAG 多轮对话图 ===
=======================================================
图编译完成: extract_query -> retrieve -> generate

=======================================================
=== 第 1 轮对话 ===
=======================================================
[extract_query] 第 1 轮, 查询: '什么是 RAG？'
[retrieve] 检索到 3 篇文档
[generate] 生成回答, 使用 3 篇文档

返回的键: ['messages', 'answer']
answer: [第1轮回答] 关于'什么是 RAG？': [文档A] 关于'什么是 RAG？'的核心概念解释 | ...
messages 数量: 2
retrieved_docs 是否泄露: False

=======================================================
=== 第 2 轮对话 ===
=======================================================
[extract_query] 第 1 轮, 查询: 'RAG 有哪些优化策略？'
[retrieve] 检索到 3 篇文档
[generate] 生成回答, 使用 3 篇文档

answer: [第1轮回答] 关于'RAG 有哪些优化策略？': [文档A] 关于'RAG 有哪些优化策略？'的核心概念解释 | ...
messages 数量: 2
retrieved_docs 是否泄露: False

=======================================================
=== 验证 capped_docs_reducer ===
=======================================================
已有文档: 4 篇
新增文档: 3 篇
合并后 (上限=5): 5 篇
保留的文档: ['旧文档2', '旧文档3', '新文档0', '新文档1', '新文档2']
```

---

## 设计要点总结

### 关键决策说明

| 决策 | 选择 | 理由 |
|------|------|------|
| 内部状态类型 | TypedDict | 高频更新，零运行时开销 |
| messages reducer | `add_messages` | LangGraph 标准做法，自动处理消息 ID 去重 |
| retrieved_docs reducer | 自定义 `capped_docs_reducer` | 防止多轮对话中文档列表无限膨胀 |
| current_query / config | 无注解 (LastValue) | 每轮覆盖为最新值，不需要历史 |
| Input Schema | 只暴露 `messages` | 调用者不需要知道内部有 config、turn_count 等字段 |
| Output Schema | 只暴露 `messages` + `answer` | 隐藏 retrieved_docs 等中间状态，防止信息泄露 |
| `total=False` | 所有字段可选 | 向后兼容，新增字段不会破坏已有检查点 |

### 三个核心模式

1. **追加模式** (`Annotated[list, add_messages]`)：消息历史只增不减，保留完整对话上下文
2. **有限累积模式** (`Annotated[list, capped_reducer]`)：累积但有上限，平衡信息保留与性能
3. **覆盖模式** (无注解)：每次写入直接覆盖，适合配置项和临时变量

---

## 学习检查清单

- [ ] 能用 `Annotated[list, add_messages]` 定义消息累积字段
- [ ] 能编写自定义 reducer 控制列表上限
- [ ] 理解无注解字段默认使用 LastValue（覆盖更新）
- [ ] 能通过 `input` / `output` 参数分离对外接口
- [ ] 理解 `total=False` 对 Schema 演进的意义
- [ ] 能验证 Output Schema 确实隐藏了内部状态

---

## 下一步

掌握了生产级 Schema 设计后，进入 **场景2**，学习：
- Reducer 选型与组合策略
- 多种 reducer 在同一个 Schema 中的协作
- 状态大小监控与自动清理
