# Annotated字段定义

> LangGraph 状态管理的声明式语法：如何使用 Annotated 类型注解定义字段的更新策略

---

## 1. 【30字核心】

**Annotated 是 Python 类型注解，通过 `Annotated[type, reducer]` 语法为状态字段声明式定义更新策略。**

[来源: reference/context7_langgraph_01.md]

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理，从源头思考问题

### Annotated 字段定义的第一性原理

#### 1. 最基础的定义

**Annotated = 类型 + 元数据的组合**

```python
from typing import Annotated

# 基本形式
Annotated[类型, 元数据]

# 在 LangGraph 中
Annotated[list, operator.add]
#         ^^^^  ^^^^^^^^^^^^^
#         类型  Reducer 元数据
```

仅此而已！没有更基础的了。

[来源: reference/context7_langgraph_01.md]

#### 2. 为什么需要 Annotated？

**核心问题：如何在类型定义时就明确更新策略？**

在状态化工作流中，不同字段需要不同的更新策略：
- 消息列表需要追加
- 计数器需要累加
- 配置需要覆盖

Annotated 让我们在定义状态时就声明这些策略，而不是在每个节点函数中重复实现。

[来源: reference/search_部分状态更新_01.md]

#### 3. Annotated 的三层价值

##### 价值1：声明式定义

在状态定义时就明确更新规则，代码更清晰。

##### 价值2：类型安全

编译时就能检查类型和 Reducer 的兼容性。

##### 价值3：关注点分离

节点函数只需返回新值，不需要关心如何合并。

[来源: reference/search_部分状态更新_01.md]

#### 4. 从第一性原理推导 LangGraph 应用

**推理链：**
```
1. 状态字段需要不同的更新策略
   ↓
2. 需要一种方式在定义时就声明策略
   ↓
3. Python 的 Annotated 提供了类型 + 元数据的能力
   ↓
4. LangGraph 使用 Annotated 的元数据存储 Reducer
   ↓
5. 框架在状态更新时读取 Reducer 并应用
   ↓
6. 实现声明式的状态更新策略
```

[来源: reference/source_部分状态更新_01.md]

#### 5. 一句话总结第一性原理

**Annotated 是类型与元数据的组合，让状态更新策略在定义时就声明式确定。**

---

## 3. 【核心概念】

### 核心概念1：Annotated 基本语法

**使用 Annotated 为类型添加 Reducer 元数据**

```python
from typing import Annotated, TypedDict
import operator

class State(TypedDict):
    # 语法：Annotated[类型, Reducer函数]
    messages: Annotated[list[str], operator.add]
    counter: Annotated[int, operator.add]
    config: str  # 没有 Annotated，使用覆盖策略
```

[来源: reference/context7_langgraph_01.md]

**工作原理：**
1. LangGraph 解析状态定义
2. 检测 Annotated 类型
3. 提取 Reducer 函数（第二个参数）
4. 在状态更新时应用 Reducer

**实际示例：**

```python
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
import operator

class State(TypedDict):
    items: Annotated[list[str], operator.add]
    total: Annotated[int, operator.add]
    status: str  # 覆盖策略

def node_a(state: State):
    return {"items": ["A"], "total": 1, "status": "processing"}

def node_b(state: State):
    return {"items": ["B"], "total": 2, "status": "done"}

builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)

graph = builder.compile()
result = graph.invoke({"items": [], "total": 0, "status": "init"})

print(result)
# {
#   'items': ['A', 'B'],      # 追加
#   'total': 3,               # 累加
#   'status': 'done'          # 覆盖
# }
```

[来源: reference/context7_langgraph_01.md]

---

### 核心概念2：常用 Reducer 模式

**LangGraph 中最常用的 Reducer 函数**

#### 模式1：operator.add（列表追加）

```python
from typing import Annotated
import operator

class State(TypedDict):
    messages: Annotated[list[str], operator.add]

# 节点返回
return {"messages": ["new message"]}
# 结果：追加到列表末尾
```

#### 模式2：operator.add（数值累加）

```python
class State(TypedDict):
    counter: Annotated[int, operator.add]

# 节点返回
return {"counter": 5}
# 结果：当前值 + 5
```

#### 模式3：add_messages（消息列表）

```python
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 节点返回
return {"messages": [HumanMessage(content="Hello")]}
# 结果：智能追加，支持消息 ID 去重
```

[来源: reference/context7_langgraph_01.md]

#### 模式4：自定义 Reducer

```python
def merge_dicts(current: dict, new: dict) -> dict:
    """深度合并字典"""
    result = current.copy()
    result.update(new)
    return result

class State(TypedDict):
    metadata: Annotated[dict, merge_dicts]
```

[来源: reference/search_部分状态更新_01.md]

---

### 核心概念3：Annotated 与覆盖策略对比

**有 Annotated 和没有 Annotated 的字段行为完全不同**

```python
from typing import Annotated, TypedDict
import operator

class State(TypedDict):
    # 有 Annotated：使用 Reducer
    with_reducer: Annotated[list[int], operator.add]

    # 没有 Annotated：覆盖策略
    without_reducer: list[int]

def node(state: State):
    return {
        "with_reducer": [3, 4],
        "without_reducer": [3, 4]
    }

# 初始状态
initial = {
    "with_reducer": [1, 2],
    "without_reducer": [1, 2]
}

# 更新后
# with_reducer: [1, 2, 3, 4]  # 追加
# without_reducer: [3, 4]     # 覆盖
```

[来源: reference/source_部分状态更新_01.md]

**选择策略的原则：**
- 需要累积历史信息 → 使用 Annotated + Reducer
- 只需要最新值 → 不使用 Annotated（覆盖）

---

## 4. 【最小可用】

掌握以下内容，就能开始使用 Annotated 字段定义：

### 4.1 基本语法

```python
from typing import Annotated, TypedDict
import operator

class State(TypedDict):
    field: Annotated[type, reducer]
```

### 4.2 列表追加

```python
messages: Annotated[list[str], operator.add]
```

### 4.3 数值累加

```python
counter: Annotated[int, operator.add]
```

### 4.4 覆盖策略（不使用 Annotated）

```python
status: str  # 直接定义类型，不使用 Annotated
```

**这些知识足以：**
- 定义状态字段的更新策略
- 实现消息历史累积
- 实现计数器累加
- 理解 Annotated 与覆盖策略的区别

---

## 5. 【双重类比】

### 类比1：Annotated 类型注解

**前端类比：** TypeScript 装饰器

TypeScript 装饰器为类或方法添加元数据，Annotated 为类型添加 Reducer 元数据。

```typescript
// TypeScript 装饰器
@Component({
  selector: 'app-root',
  template: '<h1>Hello</h1>'
})
class AppComponent {}
```

```python
# LangGraph Annotated
messages: Annotated[list, operator.add]
#         ^^^^^^^^^^^^^^^^^^^^^^^^
#         类型 + 元数据
```

**日常生活类比：** 快递包裹标签

包裹上的标签注明"易碎品-轻拿轻放"，Annotated 注明"此字段-使用 operator.add 合并"。

---

### 类比2：Reducer 选择

**前端类比：** React setState 的合并策略

React 的 setState 可以传入对象（合并）或函数（自定义逻辑），Annotated 的 Reducer 类似于自定义合并逻辑。

```javascript
// React setState
this.setState(prevState => ({
  count: prevState.count + 1  // 自定义合并
}))
```

```python
# LangGraph Annotated
counter: Annotated[int, operator.add]  # 自定义合并
```

**日常生活类比：** 银行账户操作规则

存款账户标注"累加"，临时密码标注"覆盖"。

---

### 类比3：声明式 vs 命令式

**前端类比：** CSS vs JavaScript 样式

CSS 声明式定义样式，JavaScript 命令式修改样式。Annotated 是声明式，手动合并是命令式。

```css
/* CSS 声明式 */
.button { color: blue; }
```

```javascript
// JavaScript 命令式
element.style.color = 'blue';
```

```python
# LangGraph 声明式
messages: Annotated[list, operator.add]

# 命令式（不推荐）
def node(state):
    state["messages"].extend(["new"])  # 直接修改
```

**日常生活类比：** 自动洗衣机 vs 手洗

自动洗衣机设定程序（声明式），手洗每步操作（命令式）。

---

## 6. 【反直觉点】

### 误区1：Annotated 会改变字段的类型 ❌

**为什么错？**
- Annotated 只是添加元数据，不改变运行时类型
- 字段的实际类型仍然是 Annotated 的第一个参数

**为什么人们容易这样错？**
看到 `Annotated[list, operator.add]` 以为创建了新类型。

**正确理解：**

```python
from typing import Annotated, get_type_hints
import operator

class State(TypedDict):
    messages: Annotated[list[str], operator.add]

# 运行时类型仍然是 list[str]
hints = get_type_hints(State, include_extras=True)
print(hints['messages'])  # Annotated[list[str], <built-in function add>]

# 实际使用时就是普通列表
state: State = {"messages": ["hello"]}
state["messages"].append("world")  # 正常的列表操作
```

[来源: reference/context7_langgraph_01.md]

---

### 误区2：所有字段都应该使用 Annotated ❌

**为什么错？**
- 只有需要累积的字段才使用 Annotated
- 需要覆盖的字段不应该使用 Annotated

**为什么人们容易这样错？**
看到某些字段使用 Annotated，误以为这是最佳实践。

**正确理解：**

```python
class State(TypedDict):
    # ✅ 需要累积：使用 Annotated
    messages: Annotated[list, operator.add]

    # ✅ 需要覆盖：不使用 Annotated
    current_step: str
    is_complete: bool

    # ❌ 错误：状态标志不应该累积
    # status: Annotated[str, operator.add]  # 会导致字符串拼接
```

[来源: reference/search_部分状态更新_01.md]

---

### 误区3：Annotated 的 Reducer 可以是任意函数 ❌

**为什么错？**
- Reducer 必须是二元函数：`(current, new) -> merged`
- 必须是纯函数，不能有副作用

**为什么人们容易这样错？**
以为 Annotated 的第二个参数可以是任意元数据。

**正确理解：**

```python
# ❌ 错误：不是二元函数
def bad_reducer(value):
    return value + 1

# ❌ 错误：有副作用
def bad_reducer2(current, new):
    print("Updating...")  # 副作用
    return current + new

# ✅ 正确：纯二元函数
def good_reducer(current: list, new: list) -> list:
    return current + new

class State(TypedDict):
    items: Annotated[list, good_reducer]
```

---

## 7. 【实战代码】

```python
"""
Annotated 字段定义实战示例
演示：基本语法、常用模式、策略对比
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
import operator

# ===== 1. 基本 Annotated 语法 =====
print("=== 场景1：Annotated 基本语法 ===")

class BasicState(TypedDict):
    # 列表追加
    logs: Annotated[list[str], operator.add]
    # 数值累加
    count: Annotated[int, operator.add]
    # 覆盖策略
    status: str

def log_node(state: BasicState):
    return {
        "logs": [f"Step {state['count'] + 1}"],
        "count": 1,
        "status": "processing"
    }

builder = StateGraph(BasicState)
builder.add_node("log", log_node)
builder.add_edge(START, "log")
builder.add_edge("log", END)

graph = builder.compile()
result = graph.invoke({"logs": [], "count": 0, "status": "init"})
print(f"结果: {result}")
# {'logs': ['Step 1'], 'count': 1, 'status': 'processing'}

# ===== 2. add_messages Reducer =====
print("\n=== 场景2：使用 add_messages 管理对话 ===")

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

def user_node(state: ChatState):
    return {"messages": [HumanMessage(content="Hello")]}

def ai_node(state: ChatState):
    return {"messages": [AIMessage(content="Hi there!")]}

builder2 = StateGraph(ChatState)
builder2.add_node("user", user_node)
builder2.add_node("ai", ai_node)
builder2.add_edge(START, "user")
builder2.add_edge("user", "ai")
builder2.add_edge("ai", END)

graph2 = builder2.compile()
result2 = graph2.invoke({"messages": []})
print(f"对话历史: {[m.content for m in result2['messages']]}")
# ['Hello', 'Hi there!']

# ===== 3. 自定义 Reducer =====
print("\n=== 场景3：自定义 Reducer 实现去重 ===")

def unique_add(current: list, new: list) -> list:
    """去重追加"""
    result = current.copy()
    for item in new:
        if item not in result:
            result.append(item)
    return result

class UniqueState(TypedDict):
    tags: Annotated[list[str], unique_add]

def add_tags(state: UniqueState):
    return {"tags": ["python", "ai"]}

def add_more_tags(state: UniqueState):
    return {"tags": ["python", "ml"]}  # python 重复

builder3 = StateGraph(UniqueState)
builder3.add_node("add1", add_tags)
builder3.add_node("add2", add_more_tags)
builder3.add_edge(START, "add1")
builder3.add_edge("add1", "add2")
builder3.add_edge("add2", END)

graph3 = builder3.compile()
result3 = graph3.invoke({"tags": []})
print(f"去重后的标签: {result3['tags']}")
# ['python', 'ai', 'ml']

# ===== 4. Annotated vs 覆盖对比 =====
print("\n=== 场景4：Annotated 与覆盖策略对比 ===")

class MixedState(TypedDict):
    accumulated: Annotated[list[str], operator.add]
    overwritten: list[str]

def update_both(state: MixedState):
    return {
        "accumulated": ["new"],
        "overwritten": ["new"]
    }

builder4 = StateGraph(MixedState)
builder4.add_node("update", update_both)
builder4.add_edge(START, "update")
builder4.add_edge("update", END)

graph4 = builder4.compile()
result4 = graph4.invoke({
    "accumulated": ["old"],
    "overwritten": ["old"]
})
print(f"accumulated (追加): {result4['accumulated']}")  # ['old', 'new']
print(f"overwritten (覆盖): {result4['overwritten']}")  # ['new']

print("\n✅ 所有示例运行完成")
```

**运行输出示例：**
```
=== 场景1：Annotated 基本语法 ===
结果: {'logs': ['Step 1'], 'count': 1, 'status': 'processing'}

=== 场景2：使用 add_messages 管理对话 ===
对话历史: ['Hello', 'Hi there!']

=== 场景3：自定义 Reducer 实现去重 ===
去重后的标签: ['python', 'ai', 'ml']

=== 场景4：Annotated 与覆盖策略对比 ===
accumulated (追加): ['old', 'new']
overwritten (覆盖): ['new']

✅ 所有示例运行完成
```

[来源: reference/source_部分状态更新_01.md, reference/context7_langgraph_01.md]

---

## 8. 【面试必问】

### 问题："LangGraph 中 Annotated 字段定义的作用是什么？"

**普通回答（❌ 不出彩）：**
"Annotated 用来定义字段的更新方式，可以让字段累加而不是覆盖。"

**出彩回答（✅ 推荐）：**

> **Annotated 字段定义有三层含义：**
>
> 1. **语法层面**：Annotated 是 Python 3.9+ 的类型注解特性，通过 `Annotated[type, metadata]` 语法为类型添加元数据。在 LangGraph 中，元数据是 Reducer 函数。
>
> 2. **设计层面**：Annotated 实现了声明式的状态更新策略。在状态定义时就明确"如何合并"，而不是在每个节点函数中重复实现合并逻辑，体现了关注点分离原则。
>
> 3. **应用层面**：Annotated 让不同字段可以有不同的更新策略。例如消息列表使用 `operator.add` 追加，计数器使用 `operator.add` 累加，状态标志不使用 Annotated 直接覆盖。
>
> **与 TypeScript 装饰器的类比**：TypeScript 装饰器为类添加元数据，Annotated 为类型添加元数据，都是在定义时就声明行为。
>
> **在实际工作中的应用**：在构建多步推理 Agent 时，使用 `Annotated[list, add_messages]` 定义消息历史字段，框架自动处理消息追加和去重，节点函数只需返回新消息，代码更简洁可维护。

**为什么这个回答出彩？**
1. ✅ 分层次解释（语法/设计/应用）
2. ✅ 对比相关概念（TypeScript 装饰器）
3. ✅ 结合实际应用场景（多步推理 Agent）
4. ✅ 展示对声明式编程的理解

---

## 9. 【化骨绵掌】

### 卡片1：Annotated 本质

**一句话：** Annotated 是类型 + 元数据的组合

**举例：**
```python
Annotated[list, operator.add]
#         ^^^^  ^^^^^^^^^^^^^
#         类型  元数据
```

**应用：** 为类型添加额外信息

---

### 卡片2：基本语法

**一句话：** `Annotated[type, reducer]` 定义字段更新策略

**举例：**
```python
messages: Annotated[list[str], operator.add]
```

**应用：** 状态字段定义

---

### 卡片3：operator.add 多态性

**一句话：** operator.add 对应 `+` 操作符，支持多种类型

**举例：**
- 列表：`[1] + [2] = [1, 2]`
- 数值：`1 + 2 = 3`

**应用：** 一个 Reducer 适用多种场景

---

### 卡片4：add_messages Reducer

**一句话：** 专门用于消息列表的高级 Reducer

**举例：**
```python
from langgraph.graph.message import add_messages
messages: Annotated[list, add_messages]
```

**应用：** 对话历史管理

---

### 卡片5：覆盖策略

**一句话：** 不使用 Annotated 的字段使用覆盖策略

**举例：**
```python
status: str  # 直接覆盖
```

**应用：** 只需要最新值的字段

---

### 卡片6：自定义 Reducer

**一句话：** 可以定义任意二元函数作为 Reducer

**举例：**
```python
def merge(a, b): return a + b
field: Annotated[list, merge]
```

**应用：** 实现复杂合并逻辑

---

### 卡片7：声明式 vs 命令式

**一句话：** Annotated 是声明式，手动合并是命令式

**举例：**
```python
# 声明式
messages: Annotated[list, operator.add]

# 命令式
state["messages"].extend(new_messages)
```

**应用：** 声明式更清晰可维护

---

### 卡片8：类型安全

**一句话：** Annotated 提供编译时类型检查

**举例：**
```python
# 类型检查器会验证 Reducer 与类型的兼容性
messages: Annotated[list, operator.add]  # ✅
messages: Annotated[int, operator.add]   # ✅
```

**应用：** 提前发现类型错误

---

### 卡片9：选择策略原则

**一句话：** 需要累积用 Annotated，需要覆盖不用

**举例：**
```python
history: Annotated[list, operator.add]  # 累积
status: str  # 覆盖
```

**应用：** 根据业务需求选择

---

### 卡片10：与 Pydantic 集成

**一句话：** Annotated 可以与 Pydantic 模型结合使用

**举例：**
```python
from pydantic import BaseModel

class State(BaseModel):
    messages: Annotated[list, operator.add]
```

**应用：** 类型验证 + 状态管理

---

## 10. 【一句话总结】

**Annotated 是 Python 类型注解特性，通过 `Annotated[type, reducer]` 语法为 LangGraph 状态字段声明式定义更新策略，实现关注点分离和代码简洁性。**

---

**版本：** v1.0
**创建时间：** 2026-02-26
**知识点层级：** L2_状态管理 > 03_部分状态更新 > 核心概念3
**维护者：** Claude Code

**参考资料：**
- reference/source_部分状态更新_01.md - LangGraph 源码分析
- reference/context7_langgraph_01.md - Context7 官方文档
- reference/search_部分状态更新_01.md - 社区最佳实践
