# 核心概念 2：Reducer 选型与设计模式

> Reducer 是状态更新的核心机制——选对 Reducer，状态管理就成功了一半；选错了，并发 Bug 会让你怀疑人生。

---

## 引用来源

**源码分析**:
- `libs/langgraph/langgraph/channels/binop.py` — BinaryOperatorAggregate（Reducer 执行引擎）
- `libs/langgraph/langgraph/graph/state.py` — `_get_channel()` 通道选择逻辑
- `libs/langgraph/langgraph/types.py` — Overwrite 类型定义

**官方文档**:
- Context7 LangGraph 文档 (2026-02-27)

---

## 概述

**Reducer 决定了"多个节点同时更新同一个字段时，最终值是什么"。**

LangGraph 中每个状态字段都对应一个 Channel，而 Channel 的行为由 Reducer 决定。没有 Reducer 注解的字段默认"后写覆盖"（LastValue），有 Reducer 注解的字段按你定义的规则合并。

选型的核心问题只有一个：**这个字段在多节点更新时，应该覆盖、累积、还是按自定义逻辑合并？**

---

## 1. 内置 Reducer 类型

LangGraph 提供三种开箱即用的 Reducer 模式，覆盖 90% 的场景。

### 1.1 LastValue（默认覆盖）

**无注解时的默认行为：后写入的值直接覆盖前值。**

```python
from typing import TypedDict

class State(TypedDict):
    current_query: str       # 无注解 → LastValue
    status: str              # 无注解 → LastValue
    temperature: float       # 无注解 → LastValue
```

**工作原理：**

```
Node A 返回 {"status": "searching"}
Node B 返回 {"status": "done"}
→ 最终 status = "done"（B 的值覆盖 A）
```

**适用场景：**
- 配置项（`model_name`、`temperature`）
- 当前状态标记（`status`、`current_step`）
- 单一来源的值（只有一个节点会更新的字段）

**注意：** 如果两个并行节点同时更新 LastValue 字段，结果取决于执行顺序——这通常意味着你的设计有问题。LastValue 字段应该只被一个节点更新，或者你不关心谁的值"赢"。

```python
# ✅ 好的 LastValue 用法：只有一个节点更新
def router_node(state: State) -> dict:
    return {"status": "routing"}  # 只有 router 更新 status

# ❌ 坏的 LastValue 用法：多个并行节点更新同一字段
def node_a(state: State) -> dict:
    return {"result": "from_a"}  # 并行时谁赢不确定

def node_b(state: State) -> dict:
    return {"result": "from_b"}  # 并行时谁赢不确定
```

### 1.2 add_messages（消息累积）

**LangGraph 内置的消息列表 Reducer，专为聊天场景设计。**

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
```

**核心能力：**

```python
# 能力1：自动追加新消息
# 现有: [HumanMessage("hi")]
# 更新: [AIMessage("hello")]
# 结果: [HumanMessage("hi"), AIMessage("hello")]

# 能力2：按 ID 去重更新（相同 ID 的消息会被替换）
# 现有: [AIMessage(id="msg1", content="draft")]
# 更新: [AIMessage(id="msg1", content="final")]
# 结果: [AIMessage(id="msg1", content="final")]  ← 替换而非追加

# 能力3：支持 RemoveMessage 删除特定消息
from langchain_core.messages import RemoveMessage
# 更新: [RemoveMessage(id="msg1")]
# 结果: 删除 id="msg1" 的消息
```

**为什么不直接用 `operator.add`？**

`add_messages` 比简单的列表拼接多了 ID 去重和消息删除能力。在多轮对话中，你可能需要修改已发送的消息（比如流式输出时逐步更新 AI 回复），这时 ID 去重就是刚需。

### 1.3 operator.add（列表/数值追加）

**Python 标准库的加法运算符，用于简单累积。**

```python
import operator
from typing import TypedDict, Annotated

class State(TypedDict):
    # 数值累加
    total_tokens: Annotated[int, operator.add]
    # 列表拼接
    retrieved_docs: Annotated[list[str], operator.add]
    # 字符串拼接（少用）
    log: Annotated[str, operator.add]
```

**工作原理：**

```python
# 数值累加
# Node A: {"total_tokens": 150}
# Node B: {"total_tokens": 200}
# 结果: total_tokens = 350

# 列表拼接
# Node A: {"retrieved_docs": ["doc1", "doc2"]}
# Node B: {"retrieved_docs": ["doc3"]}
# 结果: retrieved_docs = ["doc1", "doc2", "doc3"]
```

**并发安全性分析：**

| 类型 | 结合律 | 交换律 | 并发安全 |
|------|--------|--------|----------|
| `int` + `operator.add` | ✅ | ✅ | 完全安全 |
| `list` + `operator.add` | ✅ | ❌ | 顺序不确定 |
| `str` + `operator.add` | ✅ | ❌ | 顺序不确定 |

列表拼接不满足交换律（`["a"] + ["b"] != ["b"] + ["a"]`），但如果你只关心"收集了哪些元素"而不关心顺序，实践中是安全的。

---

## 2. 自定义 Reducer 设计

当内置 Reducer 不够用时，你需要自定义。

### 2.1 Reducer 函数签名

```python
def my_reducer(existing: T, new: T) -> T:
    """
    参数:
        existing: 当前状态中的值
        new: 节点返回的更新值
    返回:
        合并后的新值
    """
    ...
```

**框架内部调用方式（源码）：**

```python
# 来源: libs/langgraph/langgraph/channels/binop.py
class BinaryOperatorAggregate:
    def update(self, values: Sequence[Value]) -> bool:
        if not values:
            return False  # 空更新 → 无操作
        if self.value is MISSING:
            self.value = values[0]  # 首次赋值
            values = values[1:]
        for value in values:
            # 这里调用你的 reducer 函数
            self.value = self.operator(self.value, value)
        return True
```

关键点：`update()` 接收一个**序列**，所有并行节点的更新在一次调用中按顺序应用。这保证了原子性——不存在"处理了一半"的中间状态。

### 2.2 常见自定义模式

#### 模式 1：去重累积

```python
def dedup_add(existing: list[str], new: list[str]) -> list[str]:
    """累积但去重，保持插入顺序"""
    seen = set(existing)
    result = list(existing)
    for item in new:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

class State(TypedDict):
    sources: Annotated[list[str], dedup_add]

# Node A: {"sources": ["wiki", "arxiv"]}
# Node B: {"sources": ["arxiv", "github"]}  ← "arxiv" 已存在
# 结果: ["wiki", "arxiv", "github"]  ← 自动去重
```

#### 模式 2：带上限的列表

```python
def capped_add(existing: list[str], new: list[str], max_size: int = 100) -> list[str]:
    """累积但限制最大长度，超出时丢弃最旧的"""
    combined = existing + new
    if len(combined) > max_size:
        return combined[-max_size:]  # 保留最新的
    return combined

# 用 functools.partial 固定参数
from functools import partial
keep_last_50 = partial(capped_add, max_size=50)

class State(TypedDict):
    recent_queries: Annotated[list[str], keep_last_50]
```

#### 模式 3：条件更新（只接受更高分）

```python
def higher_score_wins(existing: float, new: float) -> float:
    """只有更高的分数才能更新"""
    return max(existing, new)

class State(TypedDict):
    best_score: Annotated[float, higher_score_wins]

# Node A: {"best_score": 0.85}
# Node B: {"best_score": 0.92}
# 结果: best_score = 0.92  ← max 满足结合律+交换律，并发安全
```

#### 模式 4：字典深度合并

```python
def deep_merge(existing: dict, new: dict) -> dict:
    """深度合并字典，new 的值优先"""
    result = {**existing}
    for key, value in new.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

class State(TypedDict):
    metadata: Annotated[dict, deep_merge]

# Node A: {"metadata": {"retrieval": {"method": "vector"}}}
# Node B: {"metadata": {"retrieval": {"top_k": 5}, "model": "gpt-4"}}
# 结果: {"retrieval": {"method": "vector", "top_k": 5}, "model": "gpt-4"}
```

### 2.3 Reducer 设计三原则

**原则 1：确定性（纯函数）** — 相同输入永远相同输出

```python
# ❌ 非确定性：random.shuffle / datetime.now()
# ✅ 确定性：sorted(set(existing + new))
```

**原则 2：无副作用** — 不写文件、不发请求、不打日志

```python
# ❌ 有副作用
def bad(current: int, new: int) -> int:
    open("log.txt", "a").write(f"{current}+{new}\n")  # 副作用！
    return current + new

# ✅ 纯计算，日志放在节点中
def good(current: int, new: int) -> int:
    return current + new
```

**原则 3：性能意识** — 避免 O(n^2) 操作

```python
# ❌ list 的 `in` 是 O(n)，大列表时 O(n²)
def slow(existing: list, new: list) -> list:
    return [x for x in existing + new if existing.count(x) <= 1]

# ✅ 用 set 加速查找，O(n)
def fast(existing: list, new: list) -> list:
    seen = set(existing)
    result = list(existing)
    for item in new:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
```

---

## 3. Overwrite 机制

### 3.1 什么是 Overwrite

**Overwrite 是一个特殊包装器，让你绕过 Reducer 直接替换字段值。**

正常流程中，有 Reducer 注解的字段永远走 Reducer 逻辑（比如 `operator.add` 只会累加，不会重置）。但有时你确实需要"清零重来"——这就是 Overwrite 的用途。

**源码实现：**

```python
# 来源: libs/langgraph/langgraph/channels/binop.py
def _get_overwrite(value: Any) -> tuple[bool, Any]:
    """检查值是否是 Overwrite 包装"""
    if isinstance(value, Overwrite):
        return True, value.value  # 是 → 绕过 reducer，直接用内部值
    return False, value           # 否 → 正常走 reducer
```

### 3.2 使用场景与代码示例

```python
from langgraph.types import Overwrite
from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    messages: Annotated[list[str], operator.add]
    error_count: Annotated[int, operator.add]

# 场景1：状态重置 — 清空消息历史
def reset_node(state: State) -> dict:
    return {
        "messages": Overwrite([]),   # 绕过 operator.add，直接替换为空列表
        "error_count": Overwrite(0)  # 绕过 operator.add，直接替换为 0
    }

# 场景2：错误恢复 — 只保留最近5条
def error_recovery_node(state: State) -> dict:
    return {"messages": Overwrite(state["messages"][-5:])}

# 场景3：对话裁剪 — 防止 context window 溢出
def trim_node(state: State) -> dict:
    if len(state["messages"]) > 50:
        return {"messages": Overwrite(state["messages"][-20:])}
    return {}
```

### 3.3 注意事项

**Overwrite 是一把双刃剑：**

| 优点 | 风险 |
|------|------|
| 提供状态重置能力 | 破坏 Reducer 的一致性保证 |
| 解决"只增不减"问题 | 并发时可能丢失其他节点的更新 |
| 错误恢复的最后手段 | 过度使用说明 Schema 设计有问题 |

**使用原则：**
1. 优先考虑能否通过 Reducer 本身解决（比如带上限的列表）
2. Overwrite 只用于"重置"和"恢复"等非常规操作
3. 不要在并行节点中使用 Overwrite——一个节点 Overwrite，另一个节点正常更新，后者的更新会丢失

---

## 4. Reducer 选型决策

### 4.1 决策流程

```
这个字段会被多个节点更新吗？
├── 否 → 用 LastValue（无注解）
└── 是 → 多个更新应该怎么合并？
    ├── 后写覆盖（只要最新值）→ LastValue
    ├── 数值累加 → operator.add
    ├── 列表追加 → operator.add
    ├── 消息列表（需要ID去重）→ add_messages
    ├── 集合并集 → 自定义 union_reducer
    ├── 取最大/最小值 → 自定义 max/min reducer
    ├── 去重列表 → 自定义 dedup_add
    ├── 字典合并 → 自定义 deep_merge
    └── 其他复杂逻辑 → 自定义 reducer
```

### 4.2 选型对照表

| 数据类型 | 业务语义 | 推荐 Reducer | 并发安全 | 说明 |
|----------|----------|-------------|----------|------|
| `str` | 当前状态 | LastValue | ✅ 单写者 | `status`、`current_model` |
| `int` | 计数/累加 | `operator.add` | ✅ | `token_count`、`retry_count` |
| `float` | 最优分数 | `max` | ✅ | `best_score` |
| `bool` | 任一为真 | `lambda a, b: a or b` | ✅ | `has_error` |
| `list` | 聊天消息 | `add_messages` | ✅ | 支持 ID 去重和删除 |
| `list` | 收集结果 | `operator.add` | ⚠️ 顺序不定 | `retrieved_docs` |
| `list` | 去重收集 | 自定义 `dedup_add` | ⚠️ 顺序不定 | `visited_urls` |
| `set` | 标签集合 | 自定义 `union` | ✅ | `tags` |
| `dict` | 配置合并 | 自定义 `deep_merge` | ⚠️ 键冲突 | `metadata` |
| `dict` | 当前配置 | LastValue | ✅ 单写者 | `config` |

### 4.3 Reducer 组合模式

实际项目中，一个 State 通常混合使用多种 Reducer：

```python
import operator
from typing import TypedDict, Annotated
from langgraph.graph import add_messages
from functools import partial

# ── 自定义 Reducer ──

def dedup_add(existing: list[str], new: list[str]) -> list[str]:
    seen = set(existing)
    result = list(existing)
    for item in new:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def capped_list(existing: list, new: list, max_size: int = 50) -> list:
    combined = existing + new
    return combined[-max_size:] if len(combined) > max_size else combined

keep_recent_50 = partial(capped_list, max_size=50)

# ── 生产级 RAG 状态 ──

class RAGState(TypedDict):
    # LastValue：单写者字段
    query: str                                              # 当前查询
    status: str                                             # 流程状态

    # add_messages：聊天消息（ID 去重）
    messages: Annotated[list, add_messages]

    # operator.add：数值累加
    total_tokens: Annotated[int, operator.add]
    search_count: Annotated[int, operator.add]

    # 自定义：去重收集
    sources: Annotated[list[str], dedup_add]

    # 自定义：带上限的日志
    debug_log: Annotated[list[str], keep_recent_50]

    # 自定义：取最高分
    best_relevance: Annotated[float, max]
```

---

## 5. 在 LangGraph 中的应用

**RAG 多路检索——Reducer 选型的经典场景：**

```python
import operator
from typing import TypedDict, Annotated
from dataclasses import dataclass

@dataclass(frozen=True)
class SearchResult:
    content: str
    source: str
    score: float

def merge_results(
    existing: list[SearchResult], new: list[SearchResult]
) -> list[SearchResult]:
    """合并检索结果：去重 + 按分数排序"""
    seen = set()
    merged = []
    for doc in existing + new:
        key = (doc.source, doc.content[:100])
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    return sorted(merged, key=lambda d: d.score, reverse=True)

class MultiRetrievalState(TypedDict):
    query: str                                                    # LastValue
    results: Annotated[list[SearchResult], merge_results]         # 自定义合并
    search_count: Annotated[int, operator.add]                    # 数值累加
    errors: Annotated[list[str], operator.add]                    # 列表拼接

# 向量检索、关键词检索、图谱检索并行执行
def vector_search(state: MultiRetrievalState) -> dict:
    return {"results": [SearchResult("向量结果", "vector_db", 0.95)], "search_count": 1}

def keyword_search(state: MultiRetrievalState) -> dict:
    return {"results": [SearchResult("关键词结果", "elastic", 0.88)], "search_count": 1}

# 并行完成后，框架自动调用各字段的 Reducer：
# results → merge_results 合并去重排序
# search_count → operator.add 累加为 2
# errors → operator.add 拼接错误列表
```

---

## 小结

**Reducer 选型的核心决策树：**

1. 单写者 → 不加注解（LastValue）
2. 多写者 + 简单累加 → `operator.add`
3. 多写者 + 消息列表 → `add_messages`
4. 多写者 + 复杂合并 → 自定义 Reducer
5. 需要重置 → Overwrite（谨慎使用）

**三条铁律：**
- Reducer 必须是**纯函数**（无副作用、确定性）
- 并发场景下 Reducer 应满足**结合律 + 交换律**
- Overwrite 只用于**重置和恢复**，不要滥用

**一句话记忆：**

> Reducer 是状态字段的"合并策略"——选对了，多节点并发更新自动合并；选错了，数据丢失还查不出原因。
