# 核心概念3：状态合并与 Reducer

> 并行节点执行后如何合并状态？Reducer 函数是 LangGraph 状态管理的核心机制

---

## 什么是 Reducer？

**Reducer** 是 LangGraph 中用于合并并行节点状态更新的函数。

当多个节点并行执行并返回状态更新时，Reducer 函数决定如何将这些更新合并到图的全局状态中。

**核心问题**：
```python
# 并行节点 A 返回：{"results": ["A"]}
# 并行节点 B 返回：{"results": ["B"]}
# 并行节点 C 返回：{"results": ["C"]}

# 问题：最终状态的 results 应该是什么？
# - 覆盖模式：["C"] (最后一个覆盖前面的)
# - 追加模式：["A", "B", "C"] (所有结果合并)
```

**Reducer 的作用**：定义合并策略，避免状态覆盖问题。

---

## 为什么需要 Reducer？

### 核心问题：并行更新冲突

在并行执行场景中，多个节点可能同时更新同一个状态字段：

```python
# 场景：三个节点并行生成笑话
def joke_node_1(state):
    return {"jokes": ["Joke 1"]}

def joke_node_2(state):
    return {"jokes": ["Joke 2"]}

def joke_node_3(state):
    return {"jokes": ["Joke 3"]}

# 问题：如果没有 Reducer，最终 jokes 只会保留一个
# 期望：jokes = ["Joke 1", "Joke 2", "Joke 3"]
```

**传统状态更新的问题**：
- **覆盖模式**：后执行的节点会覆盖先执行的节点的结果
- **丢失数据**：并行节点的部分结果会丢失
- **不确定性**：最终结果取决于节点执行顺序

**Reducer 的价值**：
1. **避免覆盖**：确保所有并行节点的结果都被保留
2. **定义策略**：明确指定如何合并多个更新
3. **类型安全**：通过类型注解强制执行合并策略

---

## Reducer 的工作原理

### 1. 基本机制

**定义**：Reducer 是一个接受两个参数的函数，返回合并后的值。

```python
def reducer(current_value, new_value):
    """
    current_value: 当前状态中的值
    new_value: 节点返回的新值
    返回: 合并后的值
    """
    return merged_value
```

**执行流程**：
```
1. 并行节点执行完毕
   ↓
2. 收集所有节点的状态更新
   ↓
3. 对每个状态字段，使用 Reducer 合并更新
   ↓
4. 更新图的全局状态
```

### 2. 使用 `Annotated` 语法

**语法**：
```python
from typing import Annotated
import operator

class State(TypedDict):
    # 使用 Annotated 指定 Reducer
    results: Annotated[list, operator.add]
    #       ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^
    #       类型              Reducer 函数
```

**含义**：
- `list`：状态字段的类型
- `operator.add`：合并策略（列表追加）

**效果**：
```python
# 当前状态：{"results": ["A"]}
# 节点返回：{"results": ["B"]}
# 合并后：  {"results": ["A", "B"]}  (使用 operator.add)
```

---

## 内置 Reducer

### 1. `operator.add` - 列表追加/数值相加

**用途**：最常用的 Reducer，支持列表追加和数值相加。

**列表追加**：
```python
from typing import Annotated
import operator
from typing_extensions import TypedDict

class State(TypedDict):
    messages: Annotated[list, operator.add]

# 当前状态：{"messages": ["Hello"]}
# 节点返回：{"messages": ["World"]}
# 合并后：  {"messages": ["Hello", "World"]}
```

**数值相加**：
```python
class State(TypedDict):
    total: Annotated[int, operator.add]

# 当前状态：{"total": 10}
# 节点返回：{"total": 5}
# 合并后：  {"total": 15}
```

**完整示例**：
```python
from langgraph.graph import StateGraph, START, END
from typing import Annotated
import operator
from typing_extensions import TypedDict

class State(TypedDict):
    aggregate: Annotated[list, operator.add]

def node_a(state: State):
    return {"aggregate": ["A"]}

def node_b(state: State):
    return {"aggregate": ["B"]}

def node_c(state: State):
    return {"aggregate": ["C"]}

builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_node("c", node_c)

# Fan-out: a -> b, c
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", END)
builder.add_edge("c", END)

graph = builder.compile()

result = graph.invoke({"aggregate": []})
print(result["aggregate"])
# 输出：['A', 'B', 'C']
```

---

### 2. `operator.mul` - 数值相乘

**用途**：数值相乘。

```python
class State(TypedDict):
    product: Annotated[int, operator.mul]

# 当前状态：{"product": 2}
# 节点返回：{"product": 3}
# 合并后：  {"product": 6}
```

---

### 3. 默认行为（无 Reducer）

**如果不指定 Reducer**：使用覆盖模式（最后一个更新生效）。

```python
class State(TypedDict):
    value: str  # 没有 Annotated，使用默认覆盖模式

# 当前状态：{"value": "A"}
# 节点返回：{"value": "B"}
# 合并后：  {"value": "B"}  (覆盖)
```

---

## 自定义 Reducer 函数

### 1. 基本自定义 Reducer

**定义自定义合并逻辑**：

```python
def merge_unique(current: list, new: list) -> list:
    """合并列表并去重"""
    return list(set(current + new))

class State(TypedDict):
    tags: Annotated[list, merge_unique]

# 当前状态：{"tags": ["python", "ai"]}
# 节点返回：{"tags": ["ai", "ml"]}
# 合并后：  {"tags": ["python", "ai", "ml"]}  (去重)
```

---

### 2. 复杂合并策略

**示例：合并字典列表**：

```python
def merge_results(current: list[dict], new: list[dict]) -> list[dict]:
    """合并结果列表，按 score 排序并保留前 10 个"""
    all_results = current + new
    # 按 score 降序排序
    sorted_results = sorted(all_results, key=lambda x: x.get("score", 0), reverse=True)
    # 保留前 10 个
    return sorted_results[:10]

class State(TypedDict):
    search_results: Annotated[list[dict], merge_results]

# 当前状态：{"search_results": [{"doc": "A", "score": 0.9}]}
# 节点返回：{"search_results": [{"doc": "B", "score": 0.95}]}
# 合并后：  {"search_results": [{"doc": "B", "score": 0.95}, {"doc": "A", "score": 0.9}]}
```

---

### 3. 条件合并

**示例：根据条件选择合并策略**：

```python
def smart_merge(current: list, new: list) -> list:
    """智能合并：如果新列表为空，保留当前值"""
    if not new:
        return current
    if not current:
        return new
    return current + new

class State(TypedDict):
    items: Annotated[list, smart_merge]
```

---

### 4. 带元数据的合并

**示例：合并时添加时间戳**：

```python
from datetime import datetime

def merge_with_timestamp(current: list[dict], new: list[dict]) -> list[dict]:
    """合并时为每个新项添加时间戳"""
    timestamp = datetime.now().isoformat()
    new_with_ts = [{"timestamp": timestamp, **item} for item in new]
    return current + new_with_ts

class State(TypedDict):
    events: Annotated[list[dict], merge_with_timestamp]

# 节点返回：{"events": [{"type": "click"}]}
# 合并后：  {"events": [{"timestamp": "2026-02-27T...", "type": "click"}]}
```

---

## 完整实战示例：Map-Reduce 笑话生成

```python
"""
Map-Reduce 笑话生成示例
演示：使用 Reducer 合并并行生成的笑话
"""

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing import Annotated
import operator
from typing_extensions import TypedDict

# ===== 1. 定义状态（使用 Reducer）=====
class OverallState(TypedDict):
    """全局状态"""
    topic: str
    subjects: list[str]
    # 使用 operator.add 作为 Reducer，自动追加笑话
    jokes: Annotated[list[str], operator.add]
    best_joke: str

class JokeState(TypedDict):
    """单个笑话生成任务的状态"""
    subject: str

# ===== 2. 定义节点 =====
def generate_topics(state: OverallState):
    """生成笑话主题列表"""
    print(f"[Generate Topics] Topic: {state['topic']}")
    # 根据主题生成多个子主题
    subjects = ["cats", "dogs", "birds"]
    return {"subjects": subjects}

def generate_joke(state: JokeState):
    """为单个主题生成笑话"""
    subject = state["subject"]
    print(f"[Generate Joke] Subject: {subject}")

    # 模拟笑话生成
    joke_map = {
        "cats": "Why don't cats play poker? Too many cheetahs!",
        "dogs": "Why do dogs run in circles? It's hard to run in squares!",
        "birds": "Why do birds fly south? It's too far to walk!"
    }

    joke = joke_map.get(subject, f"Generic joke about {subject}")

    # 返回笑话（会被 Reducer 自动追加到 jokes 列表）
    return {"jokes": [joke]}

def continue_to_jokes(state: OverallState):
    """动态创建并行任务"""
    # 为每个主题创建一个 Send 对象
    return [
        Send("generate_joke", {"subject": s})
        for s in state["subjects"]
    ]

def select_best_joke(state: OverallState):
    """选择最佳笑话"""
    print(f"[Select Best] All jokes: {state['jokes']}")

    # 简单选择第一个笑话作为最佳
    best = state["jokes"][0] if state["jokes"] else "No jokes generated"

    return {"best_joke": best}

# ===== 3. 构建图 =====
builder = StateGraph(OverallState)

# 添加节点
builder.add_node("generate_topics", generate_topics)
builder.add_node("generate_joke", generate_joke)
builder.add_node("select_best", select_best_joke)

# 添加边
builder.add_edge(START, "generate_topics")

# 动态并行边：根据 subjects 数量动态创建并行任务
builder.add_conditional_edges(
    "generate_topics",
    continue_to_jokes,
    ["generate_joke"]
)

# Fan-in 边
builder.add_edge("generate_joke", "select_best")
builder.add_edge("select_best", END)

# 编译图
graph = builder.compile()

# ===== 4. 执行 =====
print("=== Map-Reduce 笑话生成示例 ===\n")

result = graph.invoke({
    "topic": "animals",
    "subjects": [],
    "jokes": [],
    "best_joke": ""
})

print("\n=== 最终结果 ===")
print(f"主题: {result['topic']}")
print(f"子主题: {result['subjects']}")
print(f"所有笑话: {result['jokes']}")
print(f"最佳笑话: {result['best_joke']}")
```

**运行输出**：
```
=== Map-Reduce 笑话生成示例 ===

[Generate Topics] Topic: animals
[Generate Joke] Subject: cats
[Generate Joke] Subject: dogs
[Generate Joke] Subject: birds
[Select Best] All jokes: ['Why don't cats play poker? Too many cheetahs!', 'Why do dogs run in circles? It's hard to run in squares!', 'Why do birds fly south? It's too far to walk!']

=== 最终结果 ===
主题: animals
子主题: ['cats', 'dogs', 'birds']
所有笑话: ['Why don't cats play poker? Too many cheetahs!', 'Why do dogs run in circles? It's hard to run in squares!', 'Why do birds fly south? It's too far to walk!']
最佳笑话: Why don't cats play poker? Too many cheetahs!
```

**关键点**：
1. `jokes: Annotated[list[str], operator.add]` 定义了追加策略
2. 每个 `generate_joke` 节点返回 `{"jokes": [joke]}`
3. Reducer 自动将所有笑话合并到 `jokes` 列表
4. 无需手动管理状态合并逻辑

---

## Reducer 的高级用法

### 1. 多字段 Reducer

**不同字段使用不同 Reducer**：

```python
class State(TypedDict):
    # 列表追加
    messages: Annotated[list, operator.add]
    # 数值相加
    total_cost: Annotated[float, operator.add]
    # 覆盖模式（默认）
    status: str
    # 自定义合并
    metadata: Annotated[dict, merge_metadata]

def merge_metadata(current: dict, new: dict) -> dict:
    """合并元数据字典"""
    return {**current, **new}
```

---

### 2. 条件 Reducer

**根据状态条件选择合并策略**：

```python
def conditional_merge(current: list, new: list) -> list:
    """根据列表长度选择合并策略"""
    if len(current) > 100:
        # 如果列表太长，只保留最新的 100 个
        return (current + new)[-100:]
    else:
        # 否则追加
        return current + new

class State(TypedDict):
    history: Annotated[list, conditional_merge]
```

---

### 3. 带验证的 Reducer

**合并前验证数据**：

```python
def validated_merge(current: list, new: list) -> list:
    """合并前验证数据"""
    # 过滤掉无效数据
    valid_new = [item for item in new if item is not None and item != ""]
    return current + valid_new

class State(TypedDict):
    valid_items: Annotated[list, validated_merge]
```

---

## 常见误区

### 误区1：Reducer 会被每个节点调用 ❌

**错误理解**：每个节点执行时都会调用 Reducer。

**正确理解**：Reducer 只在合并并行节点的结果时调用。

```python
# Reducer 调用时机：
# 1. 并行节点 A 返回 {"results": ["A"]}
# 2. 并行节点 B 返回 {"results": ["B"]}
# 3. 合并时调用 Reducer: reducer(["A"], ["B"]) -> ["A", "B"]
```

---

### 误区2：Reducer 可以访问完整状态 ❌

**错误理解**：Reducer 函数可以访问图的完整状态。

**正确理解**：Reducer 只接收当前值和新值两个参数。

```python
# 错误：Reducer 不能访问其他状态字段
def wrong_reducer(current, new, state):  # ❌ 没有 state 参数
    return current + new

# 正确：Reducer 只接收两个参数
def correct_reducer(current, new):  # ✅
    return current + new
```

---

### 误区3：所有字段都需要 Reducer ❌

**错误理解**：所有状态字段都必须定义 Reducer。

**正确理解**：只有需要合并的字段才需要 Reducer，其他字段使用默认覆盖模式。

```python
class State(TypedDict):
    # 需要合并的字段
    results: Annotated[list, operator.add]

    # 不需要合并的字段（使用默认覆盖模式）
    status: str
    current_step: int
```

---

### 误区4：Reducer 会改变节点返回值 ❌

**错误理解**：Reducer 会修改节点的返回值。

**正确理解**：Reducer 只影响状态合并，不改变节点返回值。

```python
def node(state):
    # 节点返回的值不会被 Reducer 修改
    return {"results": ["A"]}

# Reducer 只在合并时使用
def reducer(current, new):
    return current + new
```

---

## 与前端开发的类比

| LangGraph 概念 | 前端类比 | 说明 |
|----------------|----------|------|
| Reducer | Redux reducer | 定义状态更新逻辑 |
| `operator.add` | `[...prev, ...new]` | 数组展开合并 |
| 状态合并 | `Object.assign()` | 对象合并 |
| `Annotated` | TypeScript 类型注解 | 类型 + 元数据 |
| 并行更新 | 多个 dispatch 同时触发 | 并发状态更新 |

**前端示例**：
```javascript
// Redux reducer 类似于 LangGraph Reducer
function messagesReducer(state = [], action) {
  switch (action.type) {
    case 'ADD_MESSAGE':
      // 类似于 operator.add
      return [...state, action.payload];
    case 'REPLACE_MESSAGE':
      // 类似于默认覆盖模式
      return [action.payload];
    default:
      return state;
  }
}

// 并行 dispatch
Promise.all([
  dispatch({ type: 'ADD_MESSAGE', payload: 'A' }),
  dispatch({ type: 'ADD_MESSAGE', payload: 'B' }),
  dispatch({ type: 'ADD_MESSAGE', payload: 'C' })
]);
// 最终 state: ['A', 'B', 'C']
```

---

## 日常生活类比

**Reducer 就像餐厅的订单汇总**：

1. **并行点单**：
   - 三个服务员同时接收不同桌的订单
   - 服务员 A：["汉堡"]
   - 服务员 B：["薯条"]
   - 服务员 C：["可乐"]

2. **Reducer（汇总规则）**：
   - **追加模式**（`operator.add`）：所有订单合并 → ["汉堡", "薯条", "可乐"]
   - **覆盖模式**（默认）：只保留最后一个 → ["可乐"]
   - **自定义**：去重合并 → ["汉堡", "薯条", "可乐"]（如果有重复则去重）

3. **最终订单**：
   - 厨房收到合并后的完整订单
   - 开始准备所有菜品

**没有 Reducer 的问题**：
- 厨房只能看到最后一个服务员的订单
- 其他订单会丢失
- 客人收不到完整的菜品

---

## 性能考虑

### 1. Reducer 的性能开销

**问题**：频繁的列表追加可能导致性能问题。

**优化**：
```python
# 不推荐：每次追加都创建新列表
def slow_reducer(current: list, new: list) -> list:
    return current + new  # O(n) 复制

# 推荐：使用 extend（如果可变）
def fast_reducer(current: list, new: list) -> list:
    result = current.copy()
    result.extend(new)  # O(k) 追加
    return result
```

### 2. 大数据量合并

**问题**：合并大量数据时内存压力大。

**优化**：
```python
def chunked_merge(current: list, new: list) -> list:
    """分块合并，限制总大小"""
    max_size = 10000
    combined = current + new
    if len(combined) > max_size:
        # 只保留最新的 max_size 个
        return combined[-max_size:]
    return combined
```

### 3. 复杂对象合并

**问题**：合并复杂对象时计算开销大。

**优化**：
```python
def efficient_merge(current: list[dict], new: list[dict]) -> list[dict]:
    """使用字典去重，避免重复计算"""
    # 使用 id 作为键去重
    merged = {item["id"]: item for item in current}
    merged.update({item["id"]: item for item in new})
    return list(merged.values())
```

---

## 最佳实践

### 1. 明确合并策略

**为每个需要合并的字段明确指定 Reducer**：

```python
class State(TypedDict):
    # 明确：使用追加模式
    messages: Annotated[list, operator.add]

    # 明确：使用覆盖模式（默认）
    status: str

    # 明确：使用自定义合并
    results: Annotated[list, merge_unique]
```

### 2. 保持 Reducer 简单

**Reducer 应该是纯函数，无副作用**：

```python
# 好：纯函数
def good_reducer(current, new):
    return current + new

# 坏：有副作用
def bad_reducer(current, new):
    # ❌ 不要在 Reducer 中修改外部状态
    global total_count
    total_count += len(new)
    return current + new
```

### 3. 使用类型注解

**使用 `Annotated` 明确类型和 Reducer**：

```python
from typing import Annotated

# 好：类型和 Reducer 都明确
class State(TypedDict):
    items: Annotated[list[str], operator.add]

# 坏：类型不明确
class State(TypedDict):
    items: Annotated[list, operator.add]  # list 的元素类型不明确
```

### 4. 测试 Reducer

**为自定义 Reducer 编写测试**：

```python
def test_merge_unique():
    current = ["a", "b"]
    new = ["b", "c"]
    result = merge_unique(current, new)
    assert result == ["a", "b", "c"]
    assert len(result) == len(set(result))  # 确保去重
```

---

## 总结

**Reducer 是 LangGraph 状态管理的核心机制**：

1. **作用**：
   - 定义并行节点状态更新的合并策略
   - 避免状态覆盖问题
   - 确保所有并行结果都被保留

2. **使用方式**：
   - `Annotated[type, reducer]` 语法
   - 内置 Reducer：`operator.add`、`operator.mul`
   - 自定义 Reducer：定义任意合并逻辑

3. **关键特性**：
   - 纯函数：只接收当前值和新值
   - 类型安全：通过 `Annotated` 强制执行
   - 灵活：支持任意自定义合并逻辑

4. **最佳实践**：
   - 明确指定合并策略
   - 保持 Reducer 简单
   - 使用类型注解
   - 编写测试

5. **性能优化**：
   - 避免频繁复制大对象
   - 限制合并后的数据大小
   - 使用高效的数据结构

---

## 参考资料

**官方文档**：
- [LangGraph State Management](https://docs.langchain.com/oss/python/langgraph/use-graph-api)
- [Parallel Execution with Reducers](https://docs.langchain.com/oss/python/langgraph/use-graph-api)

**源码**：
- `langgraph/types.py` - Send 类定义
- `langgraph/graph/state.py` - 状态管理实现
- `langgraph/pregel/main.py` - Pregel 算法

**社区资源**：
- [Why does LangGraph merge state from parallel branches?](https://forum.langchain.com/t/question-why-does-langgraph-merge-state-from-parallel-branches-instead-of-branch-isolation/602)
- [Best practices for parallel nodes](https://forum.langchain.com/t/best-practices-for-parallel-nodes-fanouts/1900)

---

**下一步学习**：
- 核心概念4：Bulk Synchronous Parallel 模型
- 实战代码：Map-Reduce 工作流
- 实战代码：并行 LLM 调用
