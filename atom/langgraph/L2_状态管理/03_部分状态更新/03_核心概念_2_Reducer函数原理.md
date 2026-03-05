# Reducer函数原理

> LangGraph 状态管理的核心机制：如何使用 Reducer 函数实现增量状态更新

---

## 1. 【30字核心】

**Reducer 函数是定义状态字段如何合并更新的二元操作符，通过 BinaryOperatorAggregate 实现增量更新而非覆盖。**

[来源: reference/source_部分状态更新_01.md]

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理，从源头思考问题

### Reducer 函数的第一性原理

#### 1. 最基础的定义

**Reducer = 接收两个值（当前值 + 新值），返回合并后的值的函数**

```python
def reducer(current_value, new_value):
    return merged_value
```

仅此而已！没有更基础的了。

[来源: reference/source_部分状态更新_01.md]

#### 2. 为什么需要 Reducer？

**核心问题：如何在不丢失历史信息的情况下更新状态？**

在状态化工作流中，节点之间需要传递和累积信息。如果每次更新都覆盖旧值，历史信息就会丢失。Reducer 提供了一种声明式的方式来定义"如何合并"。

[来源: reference/context7_langgraph_01.md]

#### 3. Reducer 的三层价值

##### 价值1：保留历史信息

不是简单覆盖，而是智能合并。例如消息列表追加而非替换。

##### 价值2：声明式更新策略

在状态定义时就明确更新规则，节点函数无需关心合并逻辑。

##### 价值3：类型安全

通过 Annotated 类型注解，编译时就能检查更新策略的正确性。

[来源: reference/search_部分状态更新_01.md]

#### 4. 从第一性原理推导 LangGraph 应用

**推理链：**
```
1. 工作流需要在节点间传递状态
   ↓
2. 某些状态需要累积（如消息历史、计数器）
   ↓
3. 需要一种机制定义"如何累积"
   ↓
4. Reducer 函数提供了这种机制
   ↓
5. BinaryOperatorAggregate 类实现了 Reducer
   ↓
6. Annotated 类型注解让 Reducer 声明式定义
```

[来源: reference/source_部分状态更新_01.md]

#### 5. 一句话总结第一性原理

**Reducer 是定义状态合并规则的二元函数，让状态更新从覆盖变为智能累积。**

---

## 3. 【核心概念】

### 核心概念1：BinaryOperatorAggregate 类

**LangGraph 使用 BinaryOperatorAggregate 类实现 Reducer 机制**

```python
# 源码简化版本
class BinaryOperatorAggregate:
    def __init__(self, typ: type, operator: Callable):
        self.typ = typ
        self.operator = operator  # 二元操作符函数
        self.value = typ()  # 初始化空值

    def update(self, new_value):
        # 使用 operator 合并当前值和新值
        self.value = self.operator(self.value, new_value)
        return self.value
```

[来源: reference/source_部分状态更新_01.md]

**在 LangGraph 中的使用：**

```python
from typing import Annotated
import operator

class State(TypedDict):
    # 使用 operator.add 作为 Reducer
    messages: Annotated[list[str], operator.add]
    counter: Annotated[int, operator.add]
```

**工作原理：**
1. 节点返回 `{"messages": ["new msg"]}`
2. BinaryOperatorAggregate 调用 `operator.add(current_messages, ["new msg"])`
3. 结果是列表追加而非替换

[来源: reference/context7_langgraph_01.md]

---

### 核心概念2：operator.add Reducer

**最常用的 Reducer：使用 Python 标准库的 operator.add**

```python
import operator
from typing import Annotated, TypedDict

class State(TypedDict):
    # 列表追加
    messages: Annotated[list[str], operator.add]
    # 数值累加
    total: Annotated[int, operator.add]
```

**operator.add 的行为：**
- 列表：`[1, 2] + [3] = [1, 2, 3]` （追加）
- 数值：`5 + 3 = 8` （累加）
- 字符串：`"hello" + " world" = "hello world"` （拼接）

[来源: reference/context7_langgraph_01.md]

**实际示例：**

```python
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
import operator

class State(TypedDict):
    messages: Annotated[list[str], operator.add]
    counter: Annotated[int, operator.add]

def node_a(state: State):
    return {"messages": ["A"], "counter": 1}

def node_b(state: State):
    return {"messages": ["B"], "counter": 1}

# 构建图
builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)

graph = builder.compile()

# 执行
result = graph.invoke({"messages": [], "counter": 0})
print(result)
# {'messages': ['A', 'B'], 'counter': 2}
```

[来源: reference/context7_langgraph_01.md]

---

### 核心概念3：自定义 Reducer 函数

**可以定义自己的 Reducer 函数实现复杂合并逻辑**

```python
from typing import Annotated, TypedDict

def merge_dicts(current: dict, new: dict) -> dict:
    """自定义 Reducer：深度合并字典"""
    result = current.copy()
    result.update(new)
    return result

class State(TypedDict):
    metadata: Annotated[dict, merge_dicts]

def node(state: State):
    return {"metadata": {"new_key": "new_value"}}
```

[来源: reference/search_部分状态更新_01.md]

**更复杂的示例：去重列表**

```python
def unique_append(current: list, new: list) -> list:
    """追加新元素，但保持唯一性"""
    result = current.copy()
    for item in new:
        if item not in result:
            result.append(item)
    return result

class State(TypedDict):
    tags: Annotated[list[str], unique_append]
```

---

## 4. 【最小可用】

掌握以下内容，就能开始使用 Reducer 函数：

### 4.1 使用 operator.add 实现列表追加

```python
from typing import Annotated, TypedDict
import operator

class State(TypedDict):
    items: Annotated[list, operator.add]
```

### 4.2 使用 operator.add 实现数值累加

```python
class State(TypedDict):
    count: Annotated[int, operator.add]
```

### 4.3 理解 Reducer 的调用时机

Reducer 在节点返回部分状态时自动调用，无需手动触发。

**这些知识足以：**
- 实现消息历史累积
- 实现计数器累加
- 理解 LangGraph 状态更新机制
- 为后续学习 add_messages 和自定义 Reducer 打基础

---

## 5. 【双重类比】

### 类比1：Reducer 函数

**前端类比：** Redux reducer

Redux 中的 reducer 接收当前 state 和 action，返回新 state。LangGraph 的 Reducer 接收当前值和新值，返回合并后的值。

**日常生活类比：** 银行账户余额更新

存款时不是覆盖余额，而是累加：`新余额 = 当前余额 + 存款金额`

```python
# LangGraph Reducer
balance: Annotated[int, operator.add]

# 节点返回
return {"balance": 100}  # 累加 100，而非设置为 100
```

---

### 类比2：BinaryOperatorAggregate

**前端类比：** Array.reduce()

JavaScript 的 `reduce()` 方法也是二元操作：`(accumulator, currentValue) => newAccumulator`

**日常生活类比：** 购物车总价计算

每添加一件商品，总价 = 当前总价 + 商品价格

```python
# LangGraph
total: Annotated[float, operator.add]

# 每个节点添加商品
return {"total": item_price}
```

---

### 类比3：Annotated 类型注解

**前端类比：** TypeScript 装饰器

TypeScript 装饰器为类或方法添加元数据，Annotated 为类型添加 Reducer 元数据。

**日常生活类比：** 快递包裹标签

标签上注明"易碎品-轻拿轻放"，Annotated 注明"此字段-使用 operator.add 合并"

```python
# LangGraph
messages: Annotated[list, operator.add]
#         ^^^^^^^^^^^^^^^^^^^^^^^^
#         类型 + 处理规则
```

---

## 6. 【反直觉点】

### 误区1：Reducer 会自动应用到所有字段 ❌

**为什么错？**
- 只有使用 `Annotated[type, reducer]` 定义的字段才会使用 Reducer
- 没有 Reducer 的字段使用覆盖策略

**为什么人们容易这样错？**
看到某些字段使用 Reducer，误以为这是全局行为。

**正确理解：**

```python
class State(TypedDict):
    messages: Annotated[list, operator.add]  # 使用 Reducer
    counter: int  # 覆盖策略

def node(state: State):
    return {"messages": ["new"], "counter": 5}

# messages 追加，counter 覆盖
```

[来源: reference/context7_langgraph_01.md]

---

### 误区2：Reducer 可以修改当前值 ❌

**为什么错？**
- Reducer 应该是纯函数，返回新值而非修改旧值
- 修改旧值会导致状态不可预测

**为什么人们容易这样错？**
习惯了命令式编程的 mutation 思维。

**正确理解：**

```python
# ❌ 错误：修改当前值
def bad_reducer(current: list, new: list) -> list:
    current.extend(new)  # 修改了 current
    return current

# ✅ 正确：返回新值
def good_reducer(current: list, new: list) -> list:
    return current + new  # 返回新列表
```

[来源: reference/search_部分状态更新_01.md]

---

### 误区3：operator.add 只能用于数值 ❌

**为什么错？**
- `operator.add` 对应 Python 的 `+` 操作符
- 支持任何实现了 `__add__` 方法的类型

**为什么人们容易这样错？**
"add" 这个词让人联想到数学加法。

**正确理解：**

```python
# 列表
[1, 2] + [3] = [1, 2, 3]

# 字符串
"hello" + " world" = "hello world"

# 元组
(1, 2) + (3,) = (1, 2, 3)

# 数值
5 + 3 = 8
```

---

## 7. 【实战代码】

```python
"""
Reducer 函数原理实战示例
演示：operator.add、自定义 Reducer、Reducer 行为对比
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
import operator

# ===== 1. 基础 Reducer：operator.add =====
print("=== 场景1：使用 operator.add 实现列表追加 ===")

class BasicState(TypedDict):
    messages: Annotated[list[str], operator.add]
    counter: Annotated[int, operator.add]

def node1(state: BasicState):
    print(f"Node1 接收: {state}")
    return {"messages": ["Hello"], "counter": 1}

def node2(state: BasicState):
    print(f"Node2 接收: {state}")
    return {"messages": ["World"], "counter": 1}

builder = StateGraph(BasicState)
builder.add_node("node1", node1)
builder.add_node("node2", node2)
builder.add_edge(START, "node1")
builder.add_edge("node1", "node2")
builder.add_edge("node2", END)

graph = builder.compile()
result = graph.invoke({"messages": [], "counter": 0})
print(f"最终结果: {result}")
# 输出: {'messages': ['Hello', 'World'], 'counter': 2}

# ===== 2. 自定义 Reducer：去重追加 =====
print("\n=== 场景2：自定义 Reducer 实现去重追加 ===")

def unique_append(current: list, new: list) -> list:
    """追加新元素，但保持唯一性"""
    result = current.copy()
    for item in new:
        if item not in result:
            result.append(item)
    return result

class UniqueState(TypedDict):
    tags: Annotated[list[str], unique_append]

def add_tags_1(state: UniqueState):
    return {"tags": ["python", "ai"]}

def add_tags_2(state: UniqueState):
    return {"tags": ["python", "ml"]}  # python 重复

builder2 = StateGraph(UniqueState)
builder2.add_node("add1", add_tags_1)
builder2.add_node("add2", add_tags_2)
builder2.add_edge(START, "add1")
builder2.add_edge("add1", "add2")
builder2.add_edge("add2", END)

graph2 = builder2.compile()
result2 = graph2.invoke({"tags": []})
print(f"去重后的标签: {result2}")
# 输出: {'tags': ['python', 'ai', 'ml']}

# ===== 3. Reducer vs 覆盖对比 =====
print("\n=== 场景3：Reducer 与覆盖策略对比 ===")

class MixedState(TypedDict):
    with_reducer: Annotated[list[int], operator.add]
    without_reducer: list[int]

def update_both(state: MixedState):
    return {
        "with_reducer": [1, 2],
        "without_reducer": [1, 2]
    }

builder3 = StateGraph(MixedState)
builder3.add_node("update", update_both)
builder3.add_edge(START, "update")
builder3.add_edge("update", END)

graph3 = builder3.compile()
result3 = graph3.invoke({
    "with_reducer": [0],
    "without_reducer": [0]
})
print(f"with_reducer (追加): {result3['with_reducer']}")
# 输出: [0, 1, 2]
print(f"without_reducer (覆盖): {result3['without_reducer']}")
# 输出: [1, 2]

print("\n✅ 所有示例运行完成")
```

**运行输出示例：**
```
=== 场景1：使用 operator.add 实现列表追加 ===
Node1 接收: {'messages': [], 'counter': 0}
Node2 接收: {'messages': ['Hello'], 'counter': 1}
最终结果: {'messages': ['Hello', 'World'], 'counter': 2}

=== 场景2：自定义 Reducer 实现去重追加 ===
去重后的标签: {'tags': ['python', 'ai', 'ml']}

=== 场景3：Reducer 与覆盖策略对比 ===
with_reducer (追加): [0, 1, 2]
without_reducer (覆盖): [1, 2]

✅ 所有示例运行完成
```

[来源: reference/source_部分状态更新_01.md, reference/context7_langgraph_01.md]

---

## 8. 【面试必问】

### 问题："LangGraph 的 Reducer 函数是什么？它解决了什么问题？"

**普通回答（❌ 不出彩）：**
"Reducer 是用来合并状态的函数，可以让状态累加而不是覆盖。"

**出彩回答（✅ 推荐）：**

> **Reducer 函数有三层含义：**
>
> 1. **机制层面**：Reducer 是接收当前值和新值、返回合并后值的二元函数，通过 BinaryOperatorAggregate 类实现。
>
> 2. **设计层面**：Reducer 提供了声明式的状态更新策略，在状态定义时就明确"如何合并"，节点函数无需关心合并逻辑。
>
> 3. **应用层面**：Reducer 解决了状态化工作流中的信息累积问题，例如消息历史追加、计数器累加、标签去重等场景。
>
> **与 Redux reducer 的区别**：Redux reducer 处理 action 到 state 的转换，LangGraph Reducer 处理值到值的合并。
>
> **在实际工作中的应用**：在构建多步推理 Agent 时，使用 `Annotated[list, operator.add]` 定义消息历史字段，每个节点只需返回新消息，框架自动追加到历史中，避免手动管理列表拼接逻辑。

**为什么这个回答出彩？**
1. ✅ 分层次解释（机制/设计/应用）
2. ✅ 对比相关概念（Redux reducer）
3. ✅ 结合实际应用场景（多步推理 Agent）
4. ✅ 展示对框架设计的深度理解

---

## 9. 【化骨绵掌】

### 卡片1：Reducer 的本质

**一句话：** Reducer 是 `(current, new) -> merged` 的纯函数

**举例：**
```python
def reducer(current: list, new: list) -> list:
    return current + new
```

**应用：** 所有 Reducer 都遵循这个签名

---

### 卡片2：BinaryOperatorAggregate 类

**一句话：** LangGraph 使用此类包装 Reducer 函数

**举例：**
```python
aggregate = BinaryOperatorAggregate(list, operator.add)
aggregate.update([1, 2])  # 内部调用 operator.add
```

**应用：** 理解框架内部实现

---

### 卡片3：operator.add 的多态性

**一句话：** operator.add 对应 `+` 操作符，支持多种类型

**举例：**
- 列表：`[1] + [2] = [1, 2]`
- 数值：`1 + 2 = 3`
- 字符串：`"a" + "b" = "ab"`

**应用：** 一个 Reducer 适用多种场景

---

### 卡片4：Annotated 类型注解

**一句话：** 使用 `Annotated[type, reducer]` 为字段指定 Reducer

**举例：**
```python
messages: Annotated[list, operator.add]
```

**应用：** 声明式定义更新策略

---

### 卡片5：自定义 Reducer

**一句话：** 可以定义任意二元函数作为 Reducer

**举例：**
```python
def merge(a: dict, b: dict) -> dict:
    return {**a, **b}

metadata: Annotated[dict, merge]
```

**应用：** 实现复杂合并逻辑

---

### 卡片6：Reducer 的纯函数要求

**一句话：** Reducer 不应修改输入值，应返回新值

**举例：**
```python
# ✅ 正确
def good(a, b): return a + b

# ❌ 错误
def bad(a, b): a.extend(b); return a
```

**应用：** 保证状态可预测性

---

### 卡片7：Reducer vs 覆盖

**一句话：** 有 Reducer 的字段累积，无 Reducer 的字段覆盖

**举例：**
```python
with_reducer: Annotated[list, operator.add]  # 累积
without: list  # 覆盖
```

**应用：** 根据需求选择策略

---

### 卡片8：Reducer 的调用时机

**一句话：** 节点返回部分状态时，框架自动调用 Reducer

**举例：**
```python
def node(state):
    return {"messages": ["new"]}  # 自动调用 Reducer
```

**应用：** 无需手动触发

---

### 卡片9：常见 Reducer 模式

**一句话：** 列表追加、数值累加、字典合并是最常见的三种模式

**举例：**
```python
list_field: Annotated[list, operator.add]
count: Annotated[int, operator.add]
meta: Annotated[dict, merge_dicts]
```

**应用：** 覆盖 90% 的使用场景

---

### 卡片10：Reducer 与 add_messages

**一句话：** add_messages 是专门用于消息列表的高级 Reducer

**举例：**
```python
from langgraph.graph.message import add_messages

messages: Annotated[list, add_messages]
```

**应用：** 下一个知识点详细学习

---

## 10. 【一句话总结】

**Reducer 函数是 LangGraph 状态管理的核心机制，通过 BinaryOperatorAggregate 类和 Annotated 类型注解，实现声明式的增量状态更新，让状态累积而非覆盖。**

---

**版本：** v1.0
**创建时间：** 2026-02-26
**知识点层级：** L2_状态管理 > 03_部分状态更新 > 核心概念2
**维护者：** Claude Code

**参考资料：**
- reference/source_部分状态更新_01.md - LangGraph 源码分析
- reference/context7_langgraph_01.md - Context7 官方文档
- reference/search_部分状态更新_01.md - 社区最佳实践
