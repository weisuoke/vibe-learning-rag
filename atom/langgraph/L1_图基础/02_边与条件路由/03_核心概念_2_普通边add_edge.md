# 03_核心概念_2_普通边add_edge

## 概念定义

**普通边（Normal Edge）是 LangGraph 中的固定路由边，通过 `add_edge` 方法添加，用于连接两个节点并建立确定的执行顺序。**

---

## 数据来源

本文档基于以下资料生成：
- [来源: reference/source_edges_01_add_edge.md] - add_edge 方法源码分析
- [来源: reference/context7_langgraph_01_edges_routing.md] - LangGraph 官方文档

---

## 核心要点

### 什么是普通边？

普通边是最基础的边类型，用于建立节点之间的固定连接关系。一旦添加，执行路径就确定了，不会根据状态值改变。

### 普通边的三个特点

1. **路由固定**：起始节点总是连接到同一个结束节点
2. **执行确定**：没有分支逻辑，按顺序执行
3. **简单高效**：实现简单，执行效率高

---

## 详细解释

### 1. 方法签名

```python
def add_edge(
    self,
    start_key: str | list[str],  # 起始节点（可以是单个或多个）
    end_key: str                  # 结束节点
) -> Self:
    """添加有向边从起始节点到结束节点

    当提供单个起始节点时，图会等待该节点完成后执行结束节点。
    当提供多个起始节点时，图会等待所有起始节点完成后才执行结束节点。

    Args:
        start_key: 起始节点的键（可以是单个字符串或字符串列表）
        end_key: 结束节点的键

    Returns:
        Self: StateGraph 实例，支持链式调用

    Raises:
        ValueError: 如果 start_key 是 'END' 或节点不存在
    """
```

[来源: reference/source_edges_01_add_edge.md]

---

### 2. 单起始节点边

#### 定义

单起始节点边是最常见的边类型，从一个节点连接到另一个节点。

#### 语法

```python
graph.add_edge(start_key, end_key)
```

#### 执行语义

- 等待 `start_key` 节点完成
- 传递状态到 `end_key` 节点
- 触发 `end_key` 节点执行

#### 示例：线性流程

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    value: int
    message: str

def step1(state: State) -> dict:
    """第一步：处理输入"""
    print(f"Step 1: value={state['value']}")
    return {"value": state["value"] * 2}

def step2(state: State) -> dict:
    """第二步：继续处理"""
    print(f"Step 2: value={state['value']}")
    return {"value": state["value"] + 10}

def step3(state: State) -> dict:
    """第三步：生成结果"""
    print(f"Step 3: value={state['value']}")
    return {"message": f"Final value: {state['value']}"}

# 构建图
graph = StateGraph(State)

# 添加节点
graph.add_node("step1", step1)
graph.add_node("step2", step2)
graph.add_node("step3", step3)

# 添加普通边：线性流程
graph.add_edge(START, "step1")
graph.add_edge("step1", "step2")
graph.add_edge("step2", "step3")
graph.add_edge("step3", END)

# 编译并执行
app = graph.compile()
result = app.invoke({"value": 5, "message": ""})
print(result)
# 输出：
# Step 1: value=5
# Step 2: value=10
# Step 3: value=20
# {'value': 20, 'message': 'Final value: 20'}
```

**执行流程**：
```
START → step1 → step2 → step3 → END
```

[来源: reference/context7_langgraph_01_edges_routing.md]

---

### 3. 多起始节点边（等待边）

#### 定义

多起始节点边用于等待多个节点都完成后才执行结束节点，实现 AND 逻辑。

#### 语法

```python
graph.add_edge([start_key1, start_key2, ...], end_key)
```

#### 执行语义

- 等待所有起始节点完成（AND 逻辑）
- 合并所有起始节点的状态更新
- 传递合并后的状态到结束节点
- 触发结束节点执行

#### 示例：并行汇聚

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    data: list[int]
    sum: int
    product: int
    max_value: int
    result: str

def compute_sum(state: State) -> dict:
    """计算总和"""
    total = sum(state["data"])
    print(f"Sum: {total}")
    return {"sum": total}

def compute_product(state: State) -> dict:
    """计算乘积"""
    result = 1
    for x in state["data"]:
        result *= x
    print(f"Product: {result}")
    return {"product": result}

def compute_max(state: State) -> dict:
    """计算最大值"""
    max_val = max(state["data"])
    print(f"Max: {max_val}")
    return {"max_value": max_val}

def merge_results(state: State) -> dict:
    """合并所有结果"""
    result = f"Sum={state['sum']}, Product={state['product']}, Max={state['max_value']}"
    print(f"Merged: {result}")
    return {"result": result}

# 构建图
graph = StateGraph(State)

# 添加节点
graph.add_node("sum", compute_sum)
graph.add_node("product", compute_product)
graph.add_node("max", compute_max)
graph.add_node("merge", merge_results)

# 并行执行三个计算节点
graph.add_edge(START, "sum")
graph.add_edge(START, "product")
graph.add_edge(START, "max")

# 等待边：等待三个节点都完成
graph.add_edge(["sum", "product", "max"], "merge")

graph.add_edge("merge", END)

# 编译并执行
app = graph.compile()
result = app.invoke({
    "data": [2, 3, 4],
    "sum": 0,
    "product": 0,
    "max_value": 0,
    "result": ""
})
print(result)
# 输出：
# Sum: 9
# Product: 24
# Max: 4
# Merged: Sum=9, Product=24, Max=4
# {'data': [2, 3, 4], 'sum': 9, 'product': 24, 'max_value': 4, 'result': 'Sum=9, Product=24, Max=4'}
```

**执行流程**：
```
START → sum ↘
START → product → merge → END
START → max ↗
```

[来源: reference/source_edges_01_add_edge.md]

---

### 4. 边的存储结构

#### 单起始节点边的存储

**数据结构**：
```python
self.edges: set[tuple[str, str]]
```

**示例**：
```python
{
    ("START", "step1"),
    ("step1", "step2"),
    ("step2", "END")
}
```

#### 多起始节点边的存储

**数据结构**：
```python
self.waiting_edges: set[tuple[tuple[str, ...], str]]
```

**示例**：
```python
{
    (("sum", "product", "max"), "merge")
}
```

[来源: reference/source_edges_01_add_edge.md]

---

### 5. 验证规则

#### 规则 1：END 不能作为起始节点

**原因**：END 是图的终止标记，没有后续节点。

**示例**：
```python
# ❌ 错误
graph.add_edge(END, "node1")
# ValueError: END cannot be a start node

# ✅ 正确
graph.add_edge("node1", END)
```

#### 规则 2：START 不能作为结束节点

**原因**：START 是图的入口标记，不能被其他节点指向。

**示例**：
```python
# ❌ 错误
graph.add_edge("node1", START)
# ValueError: START cannot be an end node

# ✅ 正确
graph.add_edge(START, "node1")
```

#### 规则 3：节点必须已添加

**原因**：边连接的节点必须存在。

**示例**：
```python
# ❌ 错误：node2 未添加
graph.add_node("node1", func1)
graph.add_edge("node1", "node2")
# ValueError: Need to add_node `node2` first

# ✅ 正确：先添加节点
graph.add_node("node1", func1)
graph.add_node("node2", func2)
graph.add_edge("node1", "node2")
```

#### 规则 4：非 StateGraph 不允许重复边

**原因**：非 StateGraph 可能没有状态合并机制。

**示例**：
```python
from langgraph.graph import Graph

# ❌ 错误：非 StateGraph 的重复边
graph = Graph()
graph.add_edge("node1", "node2")
graph.add_edge("node1", "node3")
# ValueError: Already found path for node 'node1'

# ✅ 正确：StateGraph 支持多边
graph = StateGraph(State)
graph.add_edge("node1", "node2")
graph.add_edge("node1", "node3")  # 合法
```

[来源: reference/source_edges_01_add_edge.md]

---

### 6. 快捷方法

#### set_entry_point

**定义**：设置图的入口节点。

**等价于**：`add_edge(START, key)`

**示例**：
```python
# 方式 1：使用 add_edge
graph.add_edge(START, "entry_node")

# 方式 2：使用快捷方法
graph.set_entry_point("entry_node")
```

#### set_finish_point

**定义**：设置图的结束节点。

**等价于**：`add_edge(key, END)`

**示例**：
```python
# 方式 1：使用 add_edge
graph.add_edge("final_node", END)

# 方式 2：使用快捷方法
graph.set_finish_point("final_node")
```

---

### 7. 链式调用

`add_edge` 方法返回 `Self`，支持链式调用。

**示例**：
```python
graph = StateGraph(State)
graph.add_node("node1", func1)
graph.add_node("node2", func2)
graph.add_node("node3", func3)

# 链式调用
graph.add_edge(START, "node1") \
     .add_edge("node1", "node2") \
     .add_edge("node2", "node3") \
     .add_edge("node3", END)
```

---

## 双重类比

### 前端类比

#### 类比 1：普通边 = React Router 固定路由

**前端概念**：
```javascript
<Route path="/home" component={Home} />
<Route path="/about" component={About} />
```

**LangGraph 概念**：
```python
graph.add_edge("home", "about")
```

**相似性**：
- 都是固定的连接关系
- 路径确定，不会改变
- 没有条件判断

#### 类比 2：等待边 = Promise.all()

**前端概念**：
```javascript
Promise.all([
  fetchUser(),
  fetchPosts(),
  fetchComments()
]).then(([user, posts, comments]) => {
  render(user, posts, comments);
});
```

**LangGraph 概念**：
```python
graph.add_edge(["fetch_user", "fetch_posts", "fetch_comments"], "render")
```

**相似性**：
- 都是等待多个操作完成
- 实现 AND 逻辑
- 所有操作完成后才继续

### 日常生活类比

#### 类比 1：普通边 = 工厂流水线

**日常场景**：汽车生产线
- 零件从工序 A 传送到工序 B
- 路径固定，按顺序执行
- 每个工序完成后自动进入下一个

**LangGraph 概念**：
```python
graph.add_edge("assemble", "paint")
graph.add_edge("paint", "inspect")
```

#### 类比 2：等待边 = 会议等待

**日常场景**：会议开始前等待所有人到齐
- 等待所有参会者到达
- 所有人到齐后才开始
- 实现 AND 逻辑

**LangGraph 概念**：
```python
graph.add_edge(["person1", "person2", "person3"], "start_meeting")
```

---

## 实际应用场景

### 场景 1：数据处理管道

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class DataState(TypedDict):
    raw_data: str
    cleaned_data: str
    transformed_data: str
    result: str

def load_data(state: DataState) -> dict:
    """加载数据"""
    raw = "raw data from source"
    print(f"Loaded: {raw}")
    return {"raw_data": raw}

def clean_data(state: DataState) -> dict:
    """清洗数据"""
    cleaned = state["raw_data"].strip().lower()
    print(f"Cleaned: {cleaned}")
    return {"cleaned_data": cleaned}

def transform_data(state: DataState) -> dict:
    """转换数据"""
    transformed = state["cleaned_data"].replace(" ", "_")
    print(f"Transformed: {transformed}")
    return {"transformed_data": transformed}

def save_result(state: DataState) -> dict:
    """保存结果"""
    result = f"Saved: {state['transformed_data']}"
    print(result)
    return {"result": result}

# 构建数据处理管道
graph = StateGraph(DataState)

graph.add_node("load", load_data)
graph.add_node("clean", clean_data)
graph.add_node("transform", transform_data)
graph.add_node("save", save_result)

# 线性管道
graph.add_edge(START, "load")
graph.add_edge("load", "clean")
graph.add_edge("clean", "transform")
graph.add_edge("transform", "save")
graph.add_edge("save", END)

app = graph.compile()
result = app.invoke({
    "raw_data": "",
    "cleaned_data": "",
    "transformed_data": "",
    "result": ""
})
print(result)
```

**关键点**：
- 使用普通边构建线性数据处理管道
- 每个步骤按顺序执行
- 状态在节点间传递

[来源: reference/context7_langgraph_01_edges_routing.md]

---

### 场景 2：并行特征提取

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class FeatureState(TypedDict):
    text: str
    word_count: int
    char_count: int
    sentence_count: int
    summary: str

def count_words(state: FeatureState) -> dict:
    """统计单词数"""
    count = len(state["text"].split())
    print(f"Words: {count}")
    return {"word_count": count}

def count_chars(state: FeatureState) -> dict:
    """统计字符数"""
    count = len(state["text"])
    print(f"Chars: {count}")
    return {"char_count": count}

def count_sentences(state: FeatureState) -> dict:
    """统计句子数"""
    count = state["text"].count(".") + state["text"].count("!") + state["text"].count("?")
    print(f"Sentences: {count}")
    return {"sentence_count": count}

def summarize(state: FeatureState) -> dict:
    """生成摘要"""
    summary = f"Text has {state['word_count']} words, {state['char_count']} chars, {state['sentence_count']} sentences"
    print(f"Summary: {summary}")
    return {"summary": summary}

# 构建特征提取图
graph = StateGraph(FeatureState)

graph.add_node("words", count_words)
graph.add_node("chars", count_chars)
graph.add_node("sentences", count_sentences)
graph.add_node("summarize", summarize)

# 并行提取特征
graph.add_edge(START, "words")
graph.add_edge(START, "chars")
graph.add_edge(START, "sentences")

# 等待所有特征提取完成
graph.add_edge(["words", "chars", "sentences"], "summarize")

graph.add_edge("summarize", END)

app = graph.compile()
result = app.invoke({
    "text": "Hello world. This is a test. How are you?",
    "word_count": 0,
    "char_count": 0,
    "sentence_count": 0,
    "summary": ""
})
print(result)
```

**关键点**：
- 使用普通边实现并行特征提取
- 使用等待边汇聚所有特征
- 提高处理效率

[来源: reference/context7_langgraph_01_edges_routing.md]

---

## 核心洞察

### 洞察 1：普通边是最基础的控制流

普通边是 LangGraph 中最简单、最基础的控制流机制。所有复杂的工作流都是在普通边的基础上构建的。

### 洞察 2：等待边实现并行汇聚

等待边（多起始节点边）是实现并行处理和汇聚的关键机制，可以显著提高工作流的执行效率。

### 洞察 3：链式调用提高代码可读性

`add_edge` 支持链式调用，可以让图的构建代码更加简洁和易读。

---

## 常见误区

### 误区 1：混淆单起始节点和多起始节点

❌ **错误理解**：以为多次调用 `add_edge` 就是多起始节点

```python
# 这不是多起始节点边，而是两条独立的边
graph.add_edge("node1", "merge")
graph.add_edge("node2", "merge")
```

✅ **正确理解**：多起始节点边需要使用列表

```python
# 这才是多起始节点边（等待边）
graph.add_edge(["node1", "node2"], "merge")
```

### 误区 2：忘记添加节点就添加边

❌ **错误**：
```python
graph.add_edge("node1", "node2")  # node1 和 node2 都未添加
```

✅ **正确**：
```python
graph.add_node("node1", func1)
graph.add_node("node2", func2)
graph.add_edge("node1", "node2")
```

### 误区 3：在非 StateGraph 中使用多边

❌ **错误**：
```python
from langgraph.graph import Graph
graph = Graph()
graph.add_edge("node1", "node2")
graph.add_edge("node1", "node3")  # 错误：重复边
```

✅ **正确**：
```python
graph = StateGraph(State)
graph.add_edge("node1", "node2")
graph.add_edge("node1", "node3")  # 正确：StateGraph 支持多边
```

---

## 参考资料

- [来源: reference/source_edges_01_add_edge.md] - add_edge 源码分析
- [来源: reference/context7_langgraph_01_edges_routing.md] - LangGraph 官方文档

---

**记住**：普通边是 LangGraph 的基础，用于建立固定的执行顺序。单起始节点边用于线性流程，多起始节点边用于并行汇聚。
