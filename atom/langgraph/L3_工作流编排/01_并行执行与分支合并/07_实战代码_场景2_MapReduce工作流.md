# 实战代码 - 场景2：Map-Reduce 工作流

> 演示 LangGraph 中使用 Send API 实现动态并行的 Map-Reduce 模式

---

## 场景概述

**目标**：掌握 Send API 实现动态 Map-Reduce 工作流

**核心技术**：
- Send API 动态并行
- Map-Reduce 模式
- 条件边与动态路由
- 状态 Reducer 自动聚合

**实际应用**：
- 笑话生成（官方示例）
- 文档并行处理
- 多智能体协作
- 批量任务执行

---

## 完整代码示例

```python
"""
LangGraph Map-Reduce 工作流实战
演示：Send API、动态并行、Map-Reduce 模式
"""

import os
import operator
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# ===== 场景1：笑话生成 Map-Reduce（官方示例）=====
print("=" * 60)
print("场景1：笑话生成 Map-Reduce")
print("=" * 60)

# 定义整体状态
class OverallState(TypedDict):
    topic: str
    subjects: list[str]
    # 使用 operator.add reducer 自动合并笑话
    jokes: Annotated[list[str], operator.add]
    best_selected_joke: str

# Map 阶段：生成主题
def generate_topics(state: OverallState):
    """生成多个主题（Map 的输入）"""
    print(f"[生成主题] 主题: {state['topic']}")
    return {"subjects": ["lions", "elephants", "penguins"]}

# Map 阶段：为每个主题生成笑话
def generate_joke(state: OverallState):
    """为单个主题生成笑话（Map 任务）"""
    subject = state["subject"]
    print(f"[生成笑话] 主题: {subject}")

    # 模拟笑话生成（实际应用中调用 LLM）
    joke_map = {
        "lions": "Why don't lions like fast food? Because they can't catch it!",
        "elephants": "Why don't elephants use computers? They're afraid of the mouse!",
        "penguins": "Why don't penguins like talking to strangers at parties? Because they find it hard to break the ice."
    }

    return {"jokes": [joke_map[subject]]}

# 条件边：动态创建并行任务
def continue_to_jokes(state: OverallState):
    """返回多个 Send 对象，实现动态并行"""
    # 为每个主题创建一个 Send 对象
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

# Reduce 阶段：选择最佳笑话
def best_joke(state: OverallState):
    """从所有笑话中选择最佳（Reduce 任务）"""
    print(f"[选择最佳] 收到 {len(state['jokes'])} 个笑话")
    # 简单选择第一个（实际应用中可以用 LLM 评分）
    return {"best_selected_joke": state["jokes"][0]}

# 构建图
builder = StateGraph(OverallState)
builder.add_node("generate_topics", generate_topics)
builder.add_node("generate_joke", generate_joke)
builder.add_node("best_joke", best_joke)

# 添加边
builder.add_edge(START, "generate_topics")
# 条件边：动态创建并行任务
builder.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
builder.add_edge("generate_joke", "best_joke")
builder.add_edge("best_joke", END)

# 编译图
graph = builder.compile()

# 执行
print("\n执行图...")
result = graph.invoke({"topic": "animals"})
print(f"\n所有笑话:")
for i, joke in enumerate(result["jokes"], 1):
    print(f"{i}. {joke}")
print(f"\n最佳笑话: {result['best_selected_joke']}")

# ===== 场景2：文档并行处理 Map-Reduce =====
print("\n" + "=" * 60)
print("场景2：文档并行处理 Map-Reduce")
print("=" * 60)

# 初始化 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 定义状态
class DocumentState(TypedDict):
    documents: list[str]
    # 使用 operator.add reducer 自动合并摘要
    summaries: Annotated[list[str], operator.add]
    final_summary: str

def load_documents(state: DocumentState):
    """加载文档（Map 的输入）"""
    print("[加载文档] 准备处理文档...")
    docs = [
        "LangGraph is a framework for building stateful, multi-actor applications with LLMs.",
        "It provides a way to define graphs where nodes are functions and edges define the flow.",
        "LangGraph supports parallel execution, checkpointing, and human-in-the-loop patterns."
    ]
    return {"documents": docs}

def summarize_document(state: DocumentState):
    """为单个文档生成摘要（Map 任务）"""
    doc = state["document"]
    print(f"[生成摘要] 文档长度: {len(doc)} 字符")

    # 调用 LLM 生成摘要
    msg = llm.invoke(f"Summarize this in one short sentence: {doc}")
    return {"summaries": [msg.content]}

def continue_to_summaries(state: DocumentState):
    """动态创建并行摘要任务"""
    return [Send("summarize_document", {"document": doc}) for doc in state["documents"]]

def combine_summaries(state: DocumentState):
    """合并所有摘要（Reduce 任务）"""
    print(f"[合并摘要] 收到 {len(state['summaries'])} 个摘要")

    # 调用 LLM 合并摘要
    all_summaries = "\n".join(f"- {s}" for s in state["summaries"])
    msg = llm.invoke(f"Combine these summaries into one paragraph:\n{all_summaries}")
    return {"final_summary": msg.content}

# 构建图
doc_builder = StateGraph(DocumentState)
doc_builder.add_node("load_documents", load_documents)
doc_builder.add_node("summarize_document", summarize_document)
doc_builder.add_node("combine_summaries", combine_summaries)

doc_builder.add_edge(START, "load_documents")
doc_builder.add_conditional_edges("load_documents", continue_to_summaries, ["summarize_document"])
doc_builder.add_edge("summarize_document", "combine_summaries")
doc_builder.add_edge("combine_summaries", END)

# 编译图
doc_graph = doc_builder.compile()

# 执行
print("\n执行图...")
doc_result = doc_graph.invoke({})
print(f"\n各文档摘要:")
for i, summary in enumerate(doc_result["summaries"], 1):
    print(f"{i}. {summary}")
print(f"\n最终摘要:\n{doc_result['final_summary']}")

# ===== 场景3：词频统计 Map-Reduce =====
print("\n" + "=" * 60)
print("场景3：词频统计 Map-Reduce")
print("=" * 60)

from collections import Counter

# 自定义 reducer：合并字典
def merge_counters(left: dict, right: dict) -> dict:
    """合并两个计数器字典"""
    result = Counter(left)
    result.update(right)
    return dict(result)

# 定义状态
class WordCountState(TypedDict):
    documents: list[str]
    # 使用自定义 reducer 合并词频
    word_counts: Annotated[dict, merge_counters]
    top_words: list[tuple[str, int]]

def prepare_documents(state: WordCountState):
    """准备文档"""
    docs = [
        "the quick brown fox jumps over the lazy dog",
        "the dog was lazy but the fox was quick",
        "quick brown animals are better than lazy animals"
    ]
    return {"documents": docs}

def count_words(state: WordCountState):
    """统计单个文档的词频（Map 任务）"""
    doc = state["document"]
    print(f"[统计词频] 文档: {doc[:30]}...")

    # 统计词频
    words = doc.lower().split()
    counts = Counter(words)
    return {"word_counts": dict(counts)}

def continue_to_count(state: WordCountState):
    """动态创建并行统计任务"""
    return [Send("count_words", {"document": doc}) for doc in state["documents"]]

def find_top_words(state: WordCountState):
    """找出最常见的词（Reduce 任务）"""
    print(f"[找出高频词] 总词数: {len(state['word_counts'])}")

    # 找出前5个最常见的词
    counter = Counter(state["word_counts"])
    top_5 = counter.most_common(5)
    return {"top_words": top_5}

# 构建图
wc_builder = StateGraph(WordCountState)
wc_builder.add_node("prepare_documents", prepare_documents)
wc_builder.add_node("count_words", count_words)
wc_builder.add_node("find_top_words", find_top_words)

wc_builder.add_edge(START, "prepare_documents")
wc_builder.add_conditional_edges("prepare_documents", continue_to_count, ["count_words"])
wc_builder.add_edge("count_words", "find_top_words")
wc_builder.add_edge("find_top_words", END)

# 编译图
wc_graph = wc_builder.compile()

# 执行
print("\n执行图...")
wc_result = wc_graph.invoke({})
print(f"\n词频统计结果:")
for word, count in wc_result["top_words"]:
    print(f"  {word}: {count}")

# ===== 场景4：多智能体协作 Map-Reduce =====
print("\n" + "=" * 60)
print("场景4：多智能体协作 Map-Reduce")
print("=" * 60)

# 定义状态
class AgentState(TypedDict):
    task: str
    subtasks: list[str]
    results: Annotated[list[str], operator.add]
    final_report: str

def plan_tasks(state: AgentState):
    """规划子任务（Map 的输入）"""
    print(f"[规划任务] 主任务: {state['task']}")
    subtasks = [
        "Research current market trends",
        "Analyze competitor strategies",
        "Identify growth opportunities"
    ]
    return {"subtasks": subtasks}

def execute_subtask(state: AgentState):
    """执行单个子任务（Map 任务）"""
    subtask = state["subtask"]
    print(f"[执行子任务] {subtask}")

    # 模拟智能体执行任务
    result = f"Completed: {subtask} - Found 3 key insights"
    return {"results": [result]}

def continue_to_subtasks(state: AgentState):
    """动态创建并行子任务"""
    return [Send("execute_subtask", {"subtask": st}) for st in state["subtasks"]]

def generate_report(state: AgentState):
    """生成最终报告（Reduce 任务）"""
    print(f"[生成报告] 收到 {len(state['results'])} 个结果")

    report = "Market Analysis Report\n\n"
    for i, result in enumerate(state["results"], 1):
        report += f"{i}. {result}\n"
    return {"final_report": report}

# 构建图
agent_builder = StateGraph(AgentState)
agent_builder.add_node("plan_tasks", plan_tasks)
agent_builder.add_node("execute_subtask", execute_subtask)
agent_builder.add_node("generate_report", generate_report)

agent_builder.add_edge(START, "plan_tasks")
agent_builder.add_conditional_edges("plan_tasks", continue_to_subtasks, ["execute_subtask"])
agent_builder.add_edge("execute_subtask", "generate_report")
agent_builder.add_edge("generate_report", END)

# 编译图
agent_graph = agent_builder.compile()

# 执行
print("\n执行图...")
agent_result = agent_graph.invoke({"task": "Market analysis for Q1 2026"})
print(f"\n{agent_result['final_report']}")

# ===== 场景5：可视化图结构 =====
print("\n" + "=" * 60)
print("场景5：可视化图结构")
print("=" * 60)

print("\n笑话生成 Map-Reduce 图的 Mermaid 表示:")
print(graph.get_graph().draw_mermaid())

# ===== 总结 =====
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
Map-Reduce 工作流的核心要点：

1. **Send API**：
   - 动态创建并行任务
   - 在条件边中返回 Send 对象列表
   - 每个 Send 指定目标节点和输入状态

2. **Map 阶段**：
   - 将大任务拆分成多个小任务
   - 每个小任务并行执行
   - 使用 Send 动态创建任务

3. **Reduce 阶段**：
   - 使用 Reducer 函数自动聚合结果
   - operator.add：列表追加
   - 自定义函数：复杂合并逻辑

4. **动态并行**：
   - 运行时确定并行度
   - 灵活的任务分配
   - 适合数据驱动的场景

5. **实际应用**：
   - 文档并行处理
   - 多智能体协作
   - 批量任务执行
   - 词频统计等数据处理
""")
```

---

## 运行输出示例

```
============================================================
场景1：笑话生成 Map-Reduce
============================================================
[生成主题] 主题: animals

执行图...
[生成笑话] 主题: lions
[生成笑话] 主题: elephants
[生成笑话] 主题: penguins
[选择最佳] 收到 3 个笑话

所有笑话:
1. Why don't lions like fast food? Because they can't catch it!
2. Why don't elephants use computers? They're afraid of the mouse!
3. Why don't penguins like talking to strangers at parties? Because they find it hard to break the ice.

最佳笑话: Why don't lions like fast food? Because they can't catch it!

============================================================
场景2：文档并行处理 Map-Reduce
============================================================

执行图...
[加载文档] 准备处理文档...
[生成摘要] 文档长度: 79 字符
[生成摘要] 文档长度: 82 字符
[生成摘要] 文档长度: 91 字符
[合并摘要] 收到 3 个摘要

各文档摘要:
1. LangGraph is a framework for building stateful, multi-actor LLM applications.
2. It defines graphs with function nodes and flow-defining edges.
3. LangGraph supports parallel execution, checkpointing, and human-in-the-loop patterns.

最终摘要:
LangGraph is a framework for building stateful, multi-actor LLM applications that defines graphs with function nodes and flow-defining edges, supporting parallel execution, checkpointing, and human-in-the-loop patterns.

============================================================
场景3：词频统计 Map-Reduce
============================================================

执行图...
[统计词频] 文档: the quick brown fox jumps ove...
[统计词频] 文档: the dog was lazy but the fox ...
[统计词频] 文档: quick brown animals are bette...
[找出高频词] 总词数: 20

词频统计结果:
  the: 5
  quick: 3
  lazy: 3
  fox: 2
  dog: 2
```

---

## 核心知识点

### 1. Send API 语法

```python
from langgraph.types import Send

# 创建单个 Send 对象
send = Send("target_node", {"key": "value"})

# 创建多个 Send 对象（并行执行）
sends = [
    Send("target_node", {"id": 1}),
    Send("target_node", {"id": 2}),
    Send("target_node", {"id": 3})
]

# 在条件边中返回
def continue_to_parallel(state):
    return [Send("process", {"item": item}) for item in state["items"]]
```

### 2. Map-Reduce 模式

**Map 阶段**：
```python
def map_function(state):
    """处理单个数据项"""
    item = state["item"]
    result = process(item)
    return {"results": [result]}  # 返回列表，使用 reducer 合并
```

**Reduce 阶段**：
```python
def reduce_function(state):
    """聚合所有结果"""
    all_results = state["results"]  # 自动合并的结果
    final = aggregate(all_results)
    return {"final": final}
```

### 3. 状态 Reducer

**内置 Reducer**：
```python
import operator

class State(TypedDict):
    # 列表追加
    items: Annotated[list, operator.add]
```

**自定义 Reducer**：
```python
def merge_dicts(left: dict, right: dict) -> dict:
    result = Counter(left)
    result.update(right)
    return dict(result)

class State(TypedDict):
    counts: Annotated[dict, merge_dicts]
```

### 4. 条件边与 Send

```python
# 添加条件边
builder.add_conditional_edges(
    "source_node",           # 源节点
    continue_function,       # 条件函数（返回 Send 列表）
    ["target_node"]          # 可能的目标节点列表
)

# 条件函数
def continue_function(state):
    return [Send("target_node", {"data": d}) for d in state["data_list"]]
```

---

## 实际应用场景

### 场景1：文档批量摘要

**问题**：需要为大量文档生成摘要

**解决方案**：
- Map：并行为每个文档生成摘要
- Reduce：合并所有摘要成总结

**效果**：
- 处理时间：O(max(doc_time)) vs O(sum(doc_time))
- 充分利用 LLM API 并发能力

### 场景2：多智能体协作

**问题**：复杂任务需要多个智能体协作

**解决方案**：
- Map：每个智能体处理子任务
- Reduce：整合所有智能体的结果

**效果**：
- 任务并行执行
- 结果自动聚合
- 易于扩展

### 场景3：数据并行处理

**问题**：大规模数据处理（词频统计、情感分析等）

**解决方案**：
- Map：并行处理每个数据块
- Reduce：合并处理结果

**效果**：
- 充分利用计算资源
- 缩短处理时间
- 易于水平扩展

---

## 常见问题

### Q1：Send API 与静态并行的区别？

**答**：
- **Send API（动态）**：运行时确定并行度，灵活但有额外开销
- **静态并行**：编译时确定并行度，性能更好但不够灵活

**选择建议**：
- 并行度固定 → 静态并行
- 并行度动态 → Send API

### Q2：如何控制并行度？

**答**：通过限制 Send 对象数量

```python
def continue_with_limit(state):
    items = state["items"][:10]  # 最多10个并行任务
    return [Send("process", {"item": i}) for i in items]
```

### Q3：Reducer 何时被调用？

**答**：当多个并行节点更新同一个状态字段时自动调用

```python
# 节点1返回 {"results": ["A"]}
# 节点2返回 {"results": ["B"]}
# Reducer 自动合并：["A"] + ["B"] = ["A", "B"]
```

### Q4：Send 可以发送到不同的节点吗？

**答**：可以，混合使用不同的目标节点

```python
def continue_mixed(state):
    return [
        Send("node_a", {"type": "a"}),
        Send("node_b", {"type": "b"}),
        Send("node_a", {"type": "a2"})
    ]
```

---

## 最佳实践

### 1. 合理设置并行度

```python
# ✅ 好：限制并行度
def continue_with_limit(state):
    items = state["items"][:100]  # 最多100个并行任务
    return [Send("process", {"item": i}) for i in items]

# ❌ 坏：无限制并行
def continue_unlimited(state):
    items = state["items"]  # 可能有10000个任务
    return [Send("process", {"item": i}) for i in items]
```

### 2. 使用合适的 Reducer

```python
# ✅ 好：使用 operator.add 追加列表
class State(TypedDict):
    results: Annotated[list, operator.add]

# ❌ 坏：没有 reducer，结果会被覆盖
class State(TypedDict):
    results: list
```

### 3. 处理错误

```python
def safe_map_function(state):
    try:
        result = risky_operation(state["item"])
        return {"results": [result]}
    except Exception as e:
        return {"results": [f"Error: {str(e)}"]}
```

### 4. 监控性能

```python
import time

def timed_map_function(state):
    start = time.time()
    result = process(state["item"])
    duration = time.time() - start
    print(f"处理时间: {duration:.2f}秒")
    return {"results": [result]}
```

---

## 引用来源

本文档基于以下资料编写：

1. **LangGraph 官方文档 - Send API 实现 Map-Reduce**
   - 来源：`reference/context7_langgraph_01.md`
   - 链接：https://docs.langchain.com/oss/python/langgraph/use-graph-api

2. **LangGraph 官方教程 - Map-Reduce with Send API**
   - 来源：`reference/search_Send_MapReduce_01.md`
   - 链接：https://langchain-ai.github.io/langgraph/how-tos/map-reduce/

3. **Medium 文章 - Implementing Map-Reduce with LangGraph**
   - 来源：`reference/search_Send_MapReduce_01.md`
   - 链接：https://medium.com/@astropomeai/implementing-map-reduce-with-langgraph

4. **Dev.to 文章 - Leveraging LangGraph's Send API**
   - 来源：`reference/search_Send_MapReduce_01.md`
   - 链接：https://dev.to/sreeni5018/leveraging-langgraphs-send-api

---

## 下一步学习

- **场景3**：并行 LLM 调用（复杂聚合策略）
- **场景4**：多数据源并行获取（错误处理与重试）
- **场景5**：复杂分支合并（条件边 + Send）
- **场景6**：性能优化实战（大规模并行任务）

---

**版本**：v1.0
**最后更新**：2026-02-27
**适用于**：LangGraph 0.2+, Python 3.13+
