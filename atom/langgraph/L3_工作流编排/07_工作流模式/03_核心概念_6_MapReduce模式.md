# 07_工作流模式 - 核心概念6：Map-Reduce 模式

> LangGraph L3_工作流编排 | 核心概念 6/6

---

## 一句话定义

**Map-Reduce 模式是通过 Send API 在运行时动态创建任意数量的并行任务，处理完后归约汇总结果的工作流模式。**

---

## 模式原理

### 核心思想

普通并行模式的分支数量在编译时就确定了（写死在图结构里）。Map-Reduce 模式的分支数量是**运行时动态决定**的——有多少数据就创建多少并行任务。

```
普通并行（编译时确定3个分支）：
START → ┌→ Worker_A ─┐
        ├→ Worker_B ─┤→ Aggregate → END
        └→ Worker_C ─┘

Map-Reduce（运行时动态N个分支）：
START → Splitter → ┌→ Worker(item_1) ─┐
                   ├→ Worker(item_2) ─┤
                   ├→ Worker(item_3) ─┤→ Reducer → END
                   ├→ ...            ─┤
                   └→ Worker(item_N) ─┘
```

### 前端类比

就像 `Array.map().reduce()`：

```javascript
// 前端 Map-Reduce
const results = items
  .map(item => processItem(item))    // Map: 对每个元素并行处理
  .reduce((acc, r) => merge(acc, r)) // Reduce: 归约合并结果
```

### 日常生活类比

老师批改试卷：把一摞试卷分给多个助教（Map），每人批改几份，最后汇总成绩单（Reduce）。试卷数量每次考试都不同，但流程一样。

---

## 核心特征

| 特征 | 说明 |
|------|------|
| **Send API** | 运行时动态创建并行任务的核心机制 |
| **动态并行度** | 分支数量由数据决定，不是图结构决定 |
| **独立状态** | 每个 Worker 收到独立的输入，互不干扰 |
| **Reducer 归约** | 通过 `operator.add` 等 reducer 汇总所有结果 |

---

## Send API 详解

`Send` 是 LangGraph 实现动态并行的核心原语：

```python
from langgraph.types import Send

# Send(目标节点名, 该节点的输入)
Send("worker", {"item": "apple"})
```

在条件边的路由函数中返回 `Send` 列表，LangGraph 会为每个 `Send` 创建一个独立的任务：

```python
def fan_out(state):
    # 返回 Send 列表 → 动态创建 N 个并行任务
    return [Send("worker", {"item": item}) for item in state["items"]]
```

---

## LangGraph 实现

### 完整示例：动态主题笑话生成

```python
"""
Map-Reduce 模式实战：
1. 生成一组主题（Splitter）
2. 为每个主题并行生成笑话（Map/Worker）
3. 选出最佳笑话（Reduce）
"""
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# ===== 1. 定义状态 =====
class OverallState(TypedDict):
    """全局状态"""
    topic: str
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]  # reducer: 追加合并
    best_joke: str


class WorkerState(TypedDict):
    """Worker 节点的独立输入状态"""
    subject: str


# ===== 2. 定义节点 =====
def generate_subjects(state: OverallState):
    """Splitter：根据主题生成子主题列表"""
    topic = state["topic"]
    # 实际项目中用 LLM 生成
    subjects = {
        "动物": ["狮子", "企鹅", "大象"],
        "编程": ["Python", "JavaScript", "Bug"],
        "食物": ["火锅", "披萨", "寿司"],
    }
    result = subjects.get(topic, ["主题1", "主题2", "主题3"])
    print(f"[Splitter] 生成 {len(result)} 个子主题: {result}")
    return {"subjects": result}


def generate_joke(state: WorkerState):
    """Worker：为单个子主题生成笑话"""
    subject = state["subject"]
    # 实际项目中用 LLM 生成
    joke_map = {
        "狮子": "为什么狮子不吃快餐？因为它们抓不住！",
        "企鹅": "企鹅为什么不怕冷？因为它们穿着燕尾服！",
        "大象": "大象为什么不用电脑？因为它们怕鼠标！",
    }
    joke = joke_map.get(subject, f"关于{subject}的笑话")
    print(f"[Worker] {subject} → {joke}")
    return {"jokes": [joke]}


def select_best(state: OverallState):
    """Reducer：选出最佳笑话"""
    jokes = state["jokes"]
    # 实际项目中用 LLM 评估
    best = max(jokes, key=len)  # 简单示例：选最长的
    print(f"[Reducer] 从 {len(jokes)} 个笑话中选出最佳")
    return {"best_joke": best}


# ===== 3. 定义动态路由（核心！）=====
def continue_to_jokes(state: OverallState):
    """动态创建 N 个并行 Worker 任务"""
    return [
        Send("generate_joke", {"subject": subject})
        for subject in state["subjects"]
    ]


# ===== 4. 构建图 =====
builder = StateGraph(OverallState)

builder.add_node("generate_subjects", generate_subjects)
builder.add_node("generate_joke", generate_joke)
builder.add_node("select_best", select_best)

builder.add_edge(START, "generate_subjects")

# 关键：用条件边 + Send 实现动态并行
builder.add_conditional_edges(
    "generate_subjects",
    continue_to_jokes,
    ["generate_joke"],  # 声明可能的目标节点
)

builder.add_edge("generate_joke", "select_best")
builder.add_edge("select_best", END)

graph = builder.compile()

# ===== 5. 执行 =====
result = graph.invoke({"topic": "动物"})
print(f"\n所有笑话: {result['jokes']}")
print(f"最佳笑话: {result['best_joke']}")
```

**运行输出：**
```
[Splitter] 生成 3 个子主题: ['狮子', '企鹅', '大象']
[Worker] 狮子 → 为什么狮子不吃快餐？因为它们抓不住！
[Worker] 企鹅 → 企鹅为什么不怕冷？因为它们穿着燕尾服！
[Worker] 大象 → 大象为什么不用电脑？因为它们怕鼠标！
[Reducer] 从 3 个笑话中选出最佳

所有笑话: ['为什么狮子不吃快餐？...', '企鹅为什么不怕冷？...', '大象为什么不用电脑？...']
最佳笑话: 企鹅为什么不怕冷？因为它们穿着燕尾服！
```

---

### Worker 状态隔离

每个 Worker 收到的是**独立的状态副本**，互不干扰：

```python
# Send 为每个 Worker 创建独立输入
Send("generate_joke", {"subject": "狮子"})   # Worker 1 只看到 "狮子"
Send("generate_joke", {"subject": "企鹅"})   # Worker 2 只看到 "企鹅"
Send("generate_joke", {"subject": "大象"})   # Worker 3 只看到 "大象"
```

Worker 的输出通过 `Annotated[list, operator.add]` reducer 自动合并到全局状态。

---

### Functional API 实现

```python
from langgraph.func import task, entrypoint


@task
def split_topics(topic: str) -> list[str]:
    """拆分为子主题"""
    return ["狮子", "企鹅", "大象"]


@task
def make_joke(subject: str) -> str:
    """为单个主题生成笑话"""
    return f"关于{subject}的笑话..."


@task
def pick_best(jokes: list[str]) -> str:
    """选出最佳"""
    return max(jokes, key=len)


@entrypoint()
def map_reduce_jokes(topic: str) -> str:
    subjects = split_topics(topic).result()

    # Map：为每个主题创建并行任务
    joke_futures = [make_joke(s) for s in subjects]

    # Reduce：等待所有结果，选出最佳
    jokes = [f.result() for f in joke_futures]
    return pick_best(jokes).result()
```

---

## 与普通并行模式的区别

| 维度 | 普通并行 | Map-Reduce |
|------|---------|------------|
| **分支数量** | 编译时确定 | 运行时动态 |
| **实现方式** | 多条 `add_edge` | `Send` API |
| **Worker 状态** | 共享全局状态 | 独立输入状态 |
| **适用场景** | 固定的并行任务 | 数据驱动的批量处理 |
| **灵活性** | 低（改图结构才能改分支数） | 高（数据决定分支数） |

```python
# 普通并行：写死3个分支
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("a", "d")

# Map-Reduce：动态N个分支
def fan_out(state):
    return [Send("worker", {"item": i}) for i in state["items"]]
builder.add_conditional_edges("splitter", fan_out, ["worker"])
```

---

## 适用场景

| 场景 | Splitter | Worker | Reducer |
|------|----------|--------|---------|
| **批量文档处理** | 拆分文档列表 | 处理单个文档 | 合并处理结果 |
| **多主题内容生成** | 规划章节 | 写单个章节 | 组装完整文章 |
| **批量翻译** | 拆分段落 | 翻译单段 | 拼接译文 |
| **RAG 多查询** | 分解复合问题 | 检索单个子问题 | 综合答案 |
| **数据清洗** | 分批数据 | 清洗单批 | 合并结果 |

---

## 优缺点分析

| 优点 | 缺点 |
|------|------|
| 处理动态数量的并行任务 | Send API 学习成本 |
| Worker 状态隔离，互不干扰 | 需要设计 Worker 独立状态 |
| 天然适合批量处理 | 大量 Send 可能导致资源压力 |
| 与数据量自动适配 | 调试时需要追踪多个 Worker |

---

## 在 RAG 中的应用

### 复合问题分解检索

```python
# 用户问："比较 Python 和 JavaScript 在 AI 领域的应用"
# Splitter：分解为子问题
# Worker：分别检索每个子问题
# Reducer：综合所有检索结果生成答案

def decompose_query(state):
    return [
        Send("retrieve", {"query": "Python 在 AI 领域的应用"}),
        Send("retrieve", {"query": "JavaScript 在 AI 领域的应用"}),
        Send("retrieve", {"query": "Python vs JavaScript AI 对比"}),
    ]
```

---

## 小结

Map-Reduce 模式的本质是**用 Send API 实现数据驱动的动态并行**。与普通并行的关键区别：分支数量不是写死在图里的，而是由运行时数据决定。适合所有"对一批数据做相同处理"的场景。记住三步：Split（拆分）→ Map（并行处理）→ Reduce（归约汇总）。

[来源: sourcecode/langgraph + Context7 官方文档]
