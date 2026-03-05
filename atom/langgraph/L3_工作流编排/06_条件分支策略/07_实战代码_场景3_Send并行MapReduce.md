# 实战代码 - 场景3：Send 并行 Map-Reduce

> 使用 Send API 实现并行文档摘要生成，多个文档同时处理后聚合结果，掌握 LangGraph 中最核心的并行处理模式。

---

## 场景说明

**业务场景：批量文档摘要系统**

输入一批文档，系统并行为每篇文档生成摘要，最后将所有摘要汇总成一份报告。这是 Send API 最经典的应用——Map-Reduce 模式。

**为什么用 Send？**
- 文档数量在运行前未知（可能 3 篇，也可能 30 篇）
- 每篇文档的处理完全独立，天然适合并行
- 需要将所有结果聚合到一个统一的报告中

**核心模式：**

```
                        Map 阶段                      Reduce 阶段
                   ┌─→ summarize(doc1) ──┐
输入文档列表 ──→ ──┼─→ summarize(doc2) ──┼──→ generate_report ──→ 输出
                   └─→ summarize(doc3) ──┘
```

**本场景覆盖的 Send API 核心要点：**
1. OverallState 和 DocumentState 分离（状态隔离）
2. 条件边返回 Send 列表（动态并行）
3. `Annotated[list, operator.add]` reducer 聚合结果
4. 最终汇总节点消费聚合数据

---

## 完整可运行代码

### 基础版：文档摘要 Map-Reduce

```python
"""
条件分支策略 - 实战场景3：Send 并行 Map-Reduce
演示：批量文档摘要系统

核心模式：
- Map: 条件边返回 Send 列表，每个文档分发到独立的摘要节点
- Reduce: operator.add 自动聚合所有摘要，汇总节点生成报告
"""

import operator
from typing import TypedDict, Annotated
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END


# ===== 1. 定义状态（关键：主图状态 vs 子任务状态分离） =====

class OverallState(TypedDict):
    """主图状态：包含全局信息"""
    documents: list[str]                                    # 输入：待处理的文档列表
    summaries: Annotated[list[str], operator.add]           # 聚合：所有摘要（自动合并）
    final_report: str                                       # 输出：最终汇总报告


class DocumentState(TypedDict):
    """子任务状态：只包含单个文档需要的信息

    关键设计：与 OverallState 完全独立
    - 子任务不需要知道有多少文档
    - 子任务不需要访问其他文档的摘要
    - Send 的 arg 就是 DocumentState 的完整输入
    """
    document: str                                           # 单个文档内容


# ===== 2. 定义 Map 函数（条件边路由） =====

def distribute_documents(state: OverallState):
    """
    Map 阶段：将每个文档分发到独立的摘要节点

    这个函数作为条件边的路由函数：
    - 返回 Send 列表（不是字符串！）
    - 每个 Send 创建一个并行任务
    - 每个 Send 携带独立的 DocumentState
    """
    print("\n" + "=" * 60)
    print("📤 Map 阶段：分发文档到并行摘要任务")
    print("=" * 60)
    print(f"📊 文档总数: {len(state['documents'])}")

    sends = []
    for i, doc in enumerate(state["documents"]):
        print(f"  ➡️  分发文档 {i + 1}: {doc[:30]}...")
        sends.append(
            Send("summarize", {"document": doc})  # 每个 Send 指向同一个节点，但携带不同数据
        )

    print(f"✅ 创建了 {len(sends)} 个并行任务")
    return sends


# ===== 3. 定义处理节点 =====

def summarize(state: DocumentState) -> dict:
    """
    处理单个文档，生成摘要

    注意：
    - 接收的是 DocumentState，不是 OverallState
    - 返回的 key 必须匹配 OverallState 中的 reducer 字段
    - 返回 {"summaries": [单个摘要]} —— 列表形式，因为 reducer 是 operator.add
    """
    doc = state["document"]

    # 模拟摘要生成（实际应用中调用 LLM）
    word_count = len(doc)
    if word_count > 50:
        summary = f"[摘要] {doc[:40]}...（共{word_count}字）"
    else:
        summary = f"[摘要] {doc}（共{word_count}字）"

    print(f"  📝 生成摘要: {summary}")

    # 关键：返回列表，因为 summaries 的 reducer 是 operator.add
    return {"summaries": [summary]}


def generate_report(state: OverallState) -> dict:
    """
    Reduce 阶段：汇总所有摘要，生成最终报告

    此时 state["summaries"] 已经包含所有并行任务的结果
    （由 operator.add reducer 自动聚合）
    """
    print("\n" + "=" * 60)
    print("📊 Reduce 阶段：汇总所有摘要")
    print("=" * 60)

    report = "=" * 40 + "\n"
    report += "    文档摘要报告\n"
    report += "=" * 40 + "\n\n"
    report += f"处理文档数: {len(state['summaries'])}\n\n"

    for i, summary in enumerate(state["summaries"], 1):
        report += f"{i}. {summary}\n"

    report += "\n" + "=" * 40

    print(f"✅ 报告生成完成，包含 {len(state['summaries'])} 条摘要")
    return {"final_report": report}


# ===== 4. 构建图 =====

builder = StateGraph(OverallState)

# 添加节点
builder.add_node("summarize", summarize)
builder.add_node("report", generate_report)

# 关键：从 START 使用条件边，路由函数返回 Send 列表
builder.add_conditional_edges(START, distribute_documents)

# 所有 summarize 实例完成后 → report
builder.add_edge("summarize", "report")

# report → END
builder.add_edge("report", END)

# 编译
graph = builder.compile()


# ===== 5. 运行测试 =====

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 Send 并行 Map-Reduce：批量文档摘要系统")
    print("=" * 70)

    result = graph.invoke({
        "documents": [
            "Python是一种高级编程语言，以简洁优雅著称，广泛应用于Web开发、数据科学、人工智能等领域",
            "LangGraph是一个用于构建状态化多Agent应用的框架，基于LangChain生态，支持循环、分支、持久化等高级特性",
            "RAG是检索增强生成的缩写，通过将外部知识库的检索结果注入到LLM的上下文中，显著提升生成质量和准确性",
            "向量数据库是存储和检索高维向量的专用数据库，支持相似度搜索，是RAG系统的核心基础设施",
        ],
        "summaries": [],
        "final_report": ""
    })

    print("\n" + result["final_report"])
```

---

### 执行流程解析

```
1. invoke({"documents": [doc1, doc2, doc3, doc4], ...})
   │
2. START → 条件边调用 distribute_documents()
   │
3. 返回 [Send("summarize", {"document": doc1}),
   │      Send("summarize", {"document": doc2}),
   │      Send("summarize", {"document": doc3}),
   │      Send("summarize", {"document": doc4})]
   │
4. 并行执行（4 个 summarize 实例同时运行）：
   │  ├─ summarize({"document": doc1}) → {"summaries": ["摘要1"]}
   │  ├─ summarize({"document": doc2}) → {"summaries": ["摘要2"]}
   │  ├─ summarize({"document": doc3}) → {"summaries": ["摘要3"]}
   │  └─ summarize({"document": doc4}) → {"summaries": ["摘要4"]}
   │
5. Reduce 自动聚合（operator.add）：
   │  summaries = ["摘要1"] + ["摘要2"] + ["摘要3"] + ["摘要4"]
   │
6. report 节点：读取聚合后的 summaries，生成 final_report
   │
7. → END
```

---

## 进阶：多级 Map-Reduce

### 场景：摘要 → 评分 → 排名

真实系统中，Map-Reduce 往往不止一层。比如：先并行生成摘要，再并行评估质量，最后排名汇总。

```python
"""
进阶：多级 Map-Reduce（两层并行）
流程：文档 → 并行摘要 → 并行评分 → 排名汇总
"""

import operator
from typing import TypedDict, Annotated
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END


# ===== 状态定义 =====

class PipelineState(TypedDict):
    """主图状态"""
    documents: list[str]
    summaries: Annotated[list[dict], operator.add]    # 第一层聚合：摘要
    evaluations: Annotated[list[dict], operator.add]  # 第二层聚合：评分
    ranking: str                                       # 最终排名

class DocState(TypedDict):
    """摘要子任务状态"""
    document: str
    doc_index: int

class EvalState(TypedDict):
    """评分子任务状态"""
    summary: str
    doc_index: int


# ===== 第一层 Map：分发文档 =====

def distribute_to_summarize(state: PipelineState):
    """第一层 Map：为每个文档创建摘要任务"""
    print("\n📤 第一层 Map：分发文档到摘要节点")
    return [
        Send("summarize_doc", {"document": doc, "doc_index": i})
        for i, doc in enumerate(state["documents"])
    ]

def summarize_doc(state: DocState) -> dict:
    """生成单个文档摘要"""
    doc = state["document"]
    idx = state["doc_index"]
    summary = doc[:30] + "..." if len(doc) > 30 else doc
    print(f"  📝 文档{idx}: 生成摘要")
    return {
        "summaries": [{"index": idx, "text": summary, "length": len(doc)}]
    }


# ===== 第二层 Map：分发摘要到评分 =====

def distribute_to_evaluate(state: PipelineState):
    """第二层 Map：为每个摘要创建评分任务"""
    print("\n📤 第二层 Map：分发摘要到评分节点")
    return [
        Send("evaluate_summary", {"summary": s["text"], "doc_index": s["index"]})
        for s in state["summaries"]
    ]

def evaluate_summary(state: EvalState) -> dict:
    """评估单个摘要的质量"""
    summary = state["summary"]
    idx = state["doc_index"]

    # 模拟评分逻辑（实际应用中调用 LLM）
    score = min(len(summary) / 10, 10.0)  # 简单的长度评分
    quality = "优秀" if score > 7 else "良好" if score > 4 else "一般"

    print(f"  ⭐ 文档{idx}: 评分 {score:.1f} ({quality})")
    return {
        "evaluations": [{"index": idx, "score": score, "quality": quality}]
    }


# ===== Reduce：排名汇总 =====

def rank_results(state: PipelineState) -> dict:
    """汇总所有评分，生成排名"""
    print("\n📊 Reduce：生成最终排名")

    # 按评分排序
    sorted_evals = sorted(state["evaluations"], key=lambda x: x["score"], reverse=True)

    ranking = "=== 文档质量排名 ===\n"
    for rank, ev in enumerate(sorted_evals, 1):
        summary = state["summaries"][ev["index"]]
        ranking += f"第{rank}名 | 评分: {ev['score']:.1f} | {ev['quality']} | {summary['text']}\n"

    return {"ranking": ranking}


# ===== 构建两层 Map-Reduce 图 =====

builder = StateGraph(PipelineState)

# 节点
builder.add_node("summarize_doc", summarize_doc)
builder.add_node("evaluate_summary", evaluate_summary)
builder.add_node("rank", rank_results)

# 第一层 Map-Reduce
builder.add_conditional_edges(START, distribute_to_summarize)
builder.add_conditional_edges("summarize_doc", distribute_to_evaluate)

# 第二层 Reduce
builder.add_edge("evaluate_summary", "rank")
builder.add_edge("rank", END)

pipeline = builder.compile()


# ===== 测试 =====

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 多级 Map-Reduce：摘要 → 评分 → 排名")
    print("=" * 70)

    result = pipeline.invoke({
        "documents": [
            "短文档",
            "这是一篇中等长度的文档，包含了一些有价值的信息和观点",
            "这是一篇非常详细的长文档，深入探讨了人工智能在医疗领域的应用前景，包括影像诊断、药物研发、个性化治疗方案等多个方面的最新进展",
        ],
        "summaries": [],
        "evaluations": [],
        "ranking": ""
    })

    print("\n" + result["ranking"])
```

### 多级 Map-Reduce 执行流程

```
START
  │
  ├─ 第一层 Map: distribute_to_summarize()
  │    ├─ summarize_doc(doc0) → summaries += [摘要0]
  │    ├─ summarize_doc(doc1) → summaries += [摘要1]
  │    └─ summarize_doc(doc2) → summaries += [摘要2]
  │
  ├─ 第二层 Map: distribute_to_evaluate()
  │    ├─ evaluate_summary(摘要0) → evaluations += [评分0]
  │    ├─ evaluate_summary(摘要1) → evaluations += [评分1]
  │    └─ evaluate_summary(摘要2) → evaluations += [评分2]
  │
  └─ Reduce: rank_results()
       └─ 读取 evaluations + summaries → 生成 ranking
           │
           END
```

---

## 关键模式总结

### 模式1：状态隔离的三要素

```python
# 1. 主图状态：包含输入 + 聚合字段
class OverallState(TypedDict):
    inputs: list[str]
    results: Annotated[list[str], operator.add]  # 必须用 reducer

# 2. 子任务状态：只包含单个任务需要的数据
class TaskState(TypedDict):
    single_input: str

# 3. Send 连接两者
def distribute(state: OverallState):
    return [Send("process", {"single_input": x}) for x in state["inputs"]]
```

### 模式2：reducer 是 Map-Reduce 的粘合剂

```python
# ✅ 正确：返回列表，reducer 自动合并
def process(state: TaskState) -> dict:
    return {"results": ["单个结果"]}  # 列表形式

# ❌ 错误：返回非列表，operator.add 无法合并
def process(state: TaskState) -> dict:
    return {"results": "单个结果"}  # 字符串，会报错
```

### 模式3：条件边串联实现多级 Map-Reduce

```python
# 第一层 Map
builder.add_conditional_edges(START, distribute_to_step1)

# 第二层 Map（从第一层的处理节点出发）
builder.add_conditional_edges("step1_node", distribute_to_step2)

# 最终 Reduce
builder.add_edge("step2_node", "final_reduce")
```

---

## 注意事项与常见错误

### 错误1：忘记初始化 reducer 字段

```python
# ❌ 错误：没有初始化 summaries
result = graph.invoke({
    "documents": ["doc1", "doc2"]
})
# 可能报错：KeyError: 'summaries'

# ✅ 正确：初始化为空列表
result = graph.invoke({
    "documents": ["doc1", "doc2"],
    "summaries": [],       # 必须初始化
    "final_report": ""
})
```

### 错误2：子任务返回值格式不匹配 reducer

```python
# ❌ 错误：返回字符串，但 reducer 期望列表
def summarize(state: DocumentState) -> dict:
    return {"summaries": "一个摘要"}  # operator.add 无法合并字符串和列表

# ✅ 正确：返回列表
def summarize(state: DocumentState) -> dict:
    return {"summaries": ["一个摘要"]}  # 列表 + 列表 = 合并
```

### 错误3：Send 的 arg 结构与子任务状态不匹配

```python
# ❌ 错误：Send 传递的 key 与 DocumentState 不匹配
Send("summarize", {"doc": "内容"})  # key 是 "doc"

class DocumentState(TypedDict):
    document: str  # 期望 key 是 "document"

# ✅ 正确：key 必须匹配
Send("summarize", {"document": "内容"})
```

### 错误4：在子任务中尝试访问主图状态

```python
# ❌ 错误：子任务无法访问主图的 documents
def summarize(state: DocumentState) -> dict:
    all_docs = state["documents"]  # KeyError! DocumentState 没有这个字段

# ✅ 正确：子任务只能访问自己的状态
def summarize(state: DocumentState) -> dict:
    doc = state["document"]  # 只访问 DocumentState 的字段
```

---

## RAG 应用场景

Send 并行 Map-Reduce 在 RAG 系统中的典型应用：

| 场景 | Map 阶段 | Reduce 阶段 |
|------|---------|------------|
| 多文档问答 | 并行检索每个文档的相关片段 | 合并所有片段，生成统一回答 |
| 批量 Embedding | 并行为每个 chunk 生成向量 | 收集所有向量，批量写入向量库 |
| 多源检索 | 同时查询多个知识库 | 合并去重，ReRank 排序 |
| 质量评估 | 并行评估每个检索结果的相关性 | 汇总评分，过滤低质量结果 |

---

## 学习检查清单

- [ ] 理解 OverallState 和子任务 State 为什么要分离
- [ ] 掌握 `Annotated[list, operator.add]` 的聚合机制
- [ ] 能写出完整的 Map（Send 分发）→ Reduce（汇总）流程
- [ ] 理解多级 Map-Reduce 的串联方式
- [ ] 知道常见的 4 个错误及其解决方法

---

**文档版本**: v1.0
**最后更新**: 2026-03-01
**来源**: reference/fetch_map_reduce_01.md, reference/source_conditional_branching_01.md
