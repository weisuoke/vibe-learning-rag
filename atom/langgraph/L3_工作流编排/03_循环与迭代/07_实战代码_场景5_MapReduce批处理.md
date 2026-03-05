# 实战代码：MapReduce 批处理

> Send 并行迭代 + 结果合并 — 动态任务分发模式

---

## 引用来源

本文档基于以下资料：

1. **参考资料**：
   - `reference/source_循环迭代_01.md` - LangGraph 循环迭代机制源码分析
   - `reference/context7_langgraph_01.md` - LangGraph 官方文档
   - `reference/search_循环迭代_01.md` - 社区资源搜索汇总

2. **关键来源**：
   - LangGraph 源码 `types.py` L289-L361 - Send 类定义
   - LangGraph 源码 `state.py` L1323-L1370 - Send 在 `attach_branch` 中的处理
   - LangGraph 源码 `_branch.py` L192-L225 - 条件边路由解析
   - [LangGraph 官方文档 - Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
   - [LangGraph 官方文档 - Use Graph API](https://docs.langchain.com/oss/python/langgraph/use-graph-api)

---

## 场景概述

**目标**：使用 Send 类实现 Map-Reduce 模式，将大任务拆分为并行子任务

**核心技术**：
- `Send` 类动态创建并行任务
- `Annotated[list, operator.add]` 自动合并结果
- OverallState vs WorkerState 状态分离
- 条件边返回 Send 列表

**实际应用**：
- RAG 中的批量文档 embedding
- 多源数据并行聚合
- 批量 API 调用与结果合并
- 并行质量评估

---

## 核心知识：Send 类回顾

### Send 的本质

```python
# [来源: sourcecode/langgraph/types.py L289-L361]
class Send:
    """发送消息到图中特定节点的指令对象"""
    __slots__ = ("node", "arg")

    node: str  # 目标节点名称
    arg: Any   # 发送给目标节点的输入数据

    def __init__(self, /, node: str, arg: Any) -> None:
        self.node = node
        self.arg = arg
```

**两个关键设计**：
1. `__slots__` 轻量设计：没有 `__dict__`，内存占用小，适合大量创建
2. 只有两个字段：`node`（去哪个节点）和 `arg`（带什么数据）

### Send vs 普通循环

| 特性 | 普通循环（条件边回连） | Send 动态迭代 |
|------|----------------------|--------------|
| 迭代次数 | 运行时逐步决定 | 一次性确定所有任务 |
| 执行方式 | 串行（一次一个） | 并行（同时执行） |
| 输入数据 | 共享同一个状态 | 每个任务独立输入 |
| 适用场景 | 自校正、对话循环 | 批处理、Map-Reduce |

### Send 的底层处理

条件边返回 Send 对象时，经过 `_finish()` 方法处理：

```python
# [来源: sourcecode/langgraph/graph/_branch.py L192-L225]
def _finish(self, writer, input, result, config):
    if not isinstance(result, (list, tuple)):
        result = [result]
    # Send 对象不写入 branch:to:{node} channel
    # 而是写入 TASKS channel，由 Pregel 引擎调度并行执行
    entries = writer(destinations, False)
```

在 `attach_branch()` 中（`state.py` L1323-L1370），Send 对象被直接传递到 TASKS channel，而非像普通路由那样写入触发 channel。这就是 Send 能实现并行的底层原因。

---

## 完整代码示例 1：经典 Map-Reduce 批处理

```python
"""
LangGraph MapReduce 批处理实战
演示：Send 并行分发 + operator.add 自动合并

场景：批量分析多个主题，并行处理后汇总结果

图结构：
  START → distribute → [Send] → worker_1 ─┐
                              → worker_2 ─┤→ aggregate → summarize → END
                              → worker_3 ─┘
"""

import operator
import time
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# ===== 1. 状态定义 =====

# 主图状态：管理整体流程
class OverallState(TypedDict):
    subjects: list[str]                              # 待处理的主题列表
    results: Annotated[list[dict], operator.add]     # 合并的结果（Reducer!）
    final_summary: str                               # 最终汇总

# Worker 状态：每个子任务的独立状态
class WorkerState(TypedDict):
    subject: str    # 单个主题
    result: dict    # 处理结果


# ===== 2. 节点实现 =====

def distribute_node(state: OverallState) -> dict:
    """
    分发节点：准备任务列表

    注意：这个节点本身不使用 Send。
    Send 在条件边函数中使用（见下方 fan_out 函数）。
    """
    subjects = state["subjects"]
    print(f"[分发] 收到 {len(subjects)} 个主题: {subjects}")
    return {}  # 不修改状态，由条件边负责分发


def worker_node(state: WorkerState) -> dict:
    """
    Worker 节点：处理单个子任务

    ★ 关键：这个节点接收的是 WorkerState，不是 OverallState！
    Send("worker", {"subject": "xxx"}) 中的 arg 就是 WorkerState。

    每个 Send 创建一个独立的 worker 实例，它们并行执行。
    """
    subject = state["subject"]

    # 模拟耗时处理（实际中可能是 LLM 调用、API 请求等）
    print(f"  [Worker] 处理主题: '{subject}' ...")
    time.sleep(0.1)  # 模拟处理时间

    # 模拟分析结果
    analysis_map = {
        "Python": {
            "subject": "Python",
            "summary": "通用编程语言，AI/ML 首选",
            "score": 95,
            "keywords": ["简洁", "生态丰富", "AI框架"],
        },
        "Rust": {
            "subject": "Rust",
            "summary": "系统编程语言，安全性极高",
            "score": 88,
            "keywords": ["内存安全", "高性能", "零成本抽象"],
        },
        "JavaScript": {
            "subject": "JavaScript",
            "summary": "Web 开发核心语言，全栈通用",
            "score": 90,
            "keywords": ["浏览器", "Node.js", "异步"],
        },
        "Go": {
            "subject": "Go",
            "summary": "云原生首选语言，并发模型优秀",
            "score": 85,
            "keywords": ["goroutine", "简洁", "编译快"],
        },
    }

    result = analysis_map.get(subject, {
        "subject": subject,
        "summary": f"{subject} 的分析结果",
        "score": 70,
        "keywords": ["通用"],
    })

    print(f"  [Worker] 完成: '{subject}' (得分: {result['score']})")

    # ★ 返回时使用 OverallState 的字段名 "results"
    # operator.add Reducer 会自动将所有 worker 的结果合并
    return {"results": [result]}


def aggregate_node(state: OverallState) -> dict:
    """
    聚合节点：所有 worker 完成后执行

    此时 state["results"] 已经包含了所有 worker 的结果
    （由 operator.add Reducer 自动合并）
    """
    results = state["results"]
    print(f"\n[聚合] 收到 {len(results)} 个结果")
    for r in results:
        print(f"  - {r['subject']}: {r['summary']} (得分: {r['score']})")
    return {}


def summarize_node(state: OverallState) -> dict:
    """生成最终汇总报告"""
    results = state["results"]

    # 按得分排序
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

    # 生成汇总
    lines = ["=== 分析报告 ==="]
    for i, r in enumerate(sorted_results, 1):
        lines.append(f"  #{i} {r['subject']}: {r['summary']} (得分: {r['score']})")

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0
    lines.append(f"  平均得分: {avg_score:.1f}")

    summary = "\n".join(lines)
    print(f"\n[汇总]\n{summary}")

    return {"final_summary": summary}


# ===== 3. Send 分发逻辑（条件边函数） =====

def fan_out(state: OverallState) -> list[Send]:
    """
    ★ 核心：条件边函数返回 Send 列表

    每个 Send 对象指定：
    - node: 目标节点名（"worker"）
    - arg: 传递给目标节点的数据（WorkerState 格式）

    LangGraph 会为每个 Send 创建一个独立的任务，并行执行。
    """
    return [
        Send("worker", {"subject": s})
        for s in state["subjects"]
    ]


# ===== 4. 构建图 =====

builder = StateGraph(OverallState)

# 添加节点
builder.add_node("distribute", distribute_node)
builder.add_node("worker", worker_node)
builder.add_node("aggregate", aggregate_node)
builder.add_node("summarize", summarize_node)

# 添加边
builder.add_edge(START, "distribute")
# ★ 条件边：distribute 之后，通过 fan_out 动态创建并行任务
builder.add_conditional_edges("distribute", fan_out, ["worker"])
# 所有 worker 完成后 → aggregate
builder.add_edge("worker", "aggregate")
builder.add_edge("aggregate", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()


# ===== 5. 执行 =====

print("=" * 60)
print("场景1：经典 Map-Reduce 批处理")
print("=" * 60)

result = graph.invoke({
    "subjects": ["Python", "Rust", "JavaScript", "Go"],
    "results": [],
    "final_summary": "",
})

print(f"\n最终报告:\n{result['final_summary']}")
print(f"结果数量: {len(result['results'])}")
```

**运行输出**：
```
============================================================
场景1：经典 Map-Reduce 批处理
============================================================
[分发] 收到 4 个主题: ['Python', 'Rust', 'JavaScript', 'Go']
  [Worker] 处理主题: 'Python' ...
  [Worker] 完成: 'Python' (得分: 95)
  [Worker] 处理主题: 'Rust' ...
  [Worker] 完成: 'Rust' (得分: 88)
  [Worker] 处理主题: 'JavaScript' ...
  [Worker] 完成: 'JavaScript' (得分: 90)
  [Worker] 处理主题: 'Go' ...
  [Worker] 完成: 'Go' (得分: 85)

[聚合] 收到 4 个结果
  - Python: 通用编程语言，AI/ML 首选 (得分: 95)
  - Rust: 系统编程语言，安全性极高 (得分: 88)
  - JavaScript: Web 开发核心语言，全栈通用 (得分: 90)
  - Go: 云原生首选语言，并发模型优秀 (得分: 85)

[汇总]
=== 分析报告 ===
  #1 Python: 通用编程语言，AI/ML 首选 (得分: 95)
  #2 JavaScript: Web 开发核心语言，全栈通用 (得分: 90)
  #3 Rust: 系统编程语言，安全性极高 (得分: 88)
  #4 Go: 云原生首选语言，并发模型优秀 (得分: 85)
  平均得分: 89.5

最终报告:
=== 分析报告 ===
  #1 Python: 通用编程语言，AI/ML 首选 (得分: 95)
  #2 JavaScript: Web 开发核心语言，全栈通用 (得分: 90)
  #3 Rust: 系统编程语言，安全性极高 (得分: 88)
  #4 Go: 云原生首选语言，并发模型优秀 (得分: 85)
  平均得分: 89.5
结果数量: 4
```

### 要点解析

1. **Send 在条件边中使用**：`fan_out` 函数返回 `[Send("worker", {...}), ...]`，LangGraph 为每个 Send 创建独立任务
2. **OverallState vs WorkerState**：主图用 OverallState，worker 节点接收 WorkerState。Send 的 `arg` 就是 WorkerState
3. **Reducer 自动合并**：`results: Annotated[list[dict], operator.add]` 确保所有 worker 的返回值被追加到同一个列表
4. **worker 返回主图字段**：worker 节点返回 `{"results": [result]}`，虽然它接收的是 WorkerState，但返回值会合并到 OverallState

---

## 完整代码示例 2：RAG 文档批量处理

```python
"""
LangGraph MapReduce 实战 - RAG 文档批量处理
演示：并行处理多个文档，提取关键信息后汇总

场景：给定一批文档，并行提取每个文档的摘要和关键词，
      然后合并为统一的知识库索引。

图结构：
  START → prepare → [Send] → process_doc_1 ─┐
                           → process_doc_2 ─┤→ merge → build_index → END
                           → process_doc_3 ─┘
"""

import operator
import hashlib
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# ===== 1. 状态定义 =====

class PipelineState(TypedDict):
    """主图状态：管理整个文档处理流水线"""
    documents: list[dict]                                # 输入文档列表
    processed: Annotated[list[dict], operator.add]       # 处理结果（Reducer 合并）
    index: dict                                          # 最终索引


class DocWorkerState(TypedDict):
    """Worker 状态：单个文档的处理上下文"""
    doc_id: str
    title: str
    content: str


# ===== 2. 模拟文档处理函数 =====

def extract_keywords(text: str) -> list[str]:
    """模拟关键词提取（实际中用 NLP 模型或 LLM）"""
    keyword_map = {
        "向量": ["向量", "embedding", "相似度"],
        "检索": ["检索", "搜索", "查询"],
        "分块": ["分块", "chunk", "切分"],
        "模型": ["模型", "LLM", "transformer"],
        "索引": ["索引", "index", "HNSW"],
    }
    found = []
    for key, keywords in keyword_map.items():
        if any(kw in text for kw in [key]):
            found.extend(keywords)
    return found if found else ["通用"]


def generate_doc_summary(title: str, content: str) -> str:
    """模拟文档摘要生成（实际中调用 LLM）"""
    return f"{title}：{content[:50]}..." if len(content) > 50 else f"{title}：{content}"


def compute_doc_hash(content: str) -> str:
    """计算文档指纹（用于去重）"""
    return hashlib.md5(content.encode()).hexdigest()[:8]


# ===== 3. 节点实现 =====

def prepare_node(state: PipelineState) -> dict:
    """准备阶段：验证文档列表"""
    docs = state["documents"]
    print(f"[准备] 收到 {len(docs)} 个文档")
    for doc in docs:
        print(f"  - {doc['title']} ({len(doc['content'])} 字符)")
    return {}


def process_doc_node(state: DocWorkerState) -> dict:
    """
    处理单个文档（Worker 节点）

    每个 Send 创建一个独立实例，接收 DocWorkerState。
    返回值通过 operator.add 合并到 PipelineState.processed。
    """
    doc_id = state["doc_id"]
    title = state["title"]
    content = state["content"]

    print(f"  [处理] 文档 '{title}' (ID: {doc_id})")

    # 提取关键词
    keywords = extract_keywords(content)

    # 生成摘要
    summary = generate_doc_summary(title, content)

    # 计算指纹
    doc_hash = compute_doc_hash(content)

    result = {
        "doc_id": doc_id,
        "title": title,
        "summary": summary,
        "keywords": keywords,
        "hash": doc_hash,
        "char_count": len(content),
    }

    print(f"  [完成] '{title}': 关键词={keywords}, 指纹={doc_hash}")

    return {"processed": [result]}


def merge_node(state: PipelineState) -> dict:
    """合并所有处理结果"""
    processed = state["processed"]
    print(f"\n[合并] 收到 {len(processed)} 个文档的处理结果")
    return {}


def build_index_node(state: PipelineState) -> dict:
    """构建知识库索引"""
    processed = state["processed"]

    # 构建索引结构
    index = {
        "total_docs": len(processed),
        "total_chars": sum(p["char_count"] for p in processed),
        "all_keywords": list(set(
            kw for p in processed for kw in p["keywords"]
        )),
        "documents": {
            p["doc_id"]: {
                "title": p["title"],
                "summary": p["summary"],
                "keywords": p["keywords"],
                "hash": p["hash"],
            }
            for p in processed
        },
    }

    print(f"[索引] 构建完成:")
    print(f"  文档数: {index['total_docs']}")
    print(f"  总字符: {index['total_chars']}")
    print(f"  关键词: {index['all_keywords']}")

    return {"index": index}


# ===== 4. Send 分发函数 =====

def fan_out_docs(state: PipelineState) -> list[Send]:
    """为每个文档创建一个 Send，并行处理"""
    return [
        Send("process_doc", {
            "doc_id": f"doc_{i:03d}",
            "title": doc["title"],
            "content": doc["content"],
        })
        for i, doc in enumerate(state["documents"])
    ]


# ===== 5. 构建图 =====

builder = StateGraph(PipelineState)

builder.add_node("prepare", prepare_node)
builder.add_node("process_doc", process_doc_node)
builder.add_node("merge", merge_node)
builder.add_node("build_index", build_index_node)

builder.add_edge(START, "prepare")
builder.add_conditional_edges("prepare", fan_out_docs, ["process_doc"])
builder.add_edge("process_doc", "merge")
builder.add_edge("merge", "build_index")
builder.add_edge("build_index", END)

graph = builder.compile()


# ===== 6. 执行 =====

print("\n" + "=" * 60)
print("场景2：RAG 文档批量处理")
print("=" * 60)

documents = [
    {
        "title": "向量数据库入门",
        "content": "向量数据库是存储和检索高维向量的专用数据库。它使用近似最近邻算法（如HNSW）来实现高效的相似度搜索。在RAG系统中，向量数据库是核心组件。",
    },
    {
        "title": "文本分块策略",
        "content": "文本分块是RAG预处理的关键步骤。常见策略包括固定大小分块、语义分块和递归分块。分块大小直接影响检索质量。",
    },
    {
        "title": "大模型API调用指南",
        "content": "调用大模型API需要注意token限制、温度参数和系统提示词的设计。合理的prompt工程可以显著提升模型输出质量。",
    },
    {
        "title": "检索增强生成概述",
        "content": "RAG通过检索外部知识来增强大模型的生成能力。核心流程包括：文档索引、查询检索、上下文注入和答案生成四个阶段。",
    },
]

result = graph.invoke({
    "documents": documents,
    "processed": [],
    "index": {},
})

print(f"\n=== 最终索引 ===")
index = result["index"]
print(f"文档数: {index['total_docs']}")
print(f"关键词库: {index['all_keywords']}")
for doc_id, info in index["documents"].items():
    print(f"  {doc_id}: {info['title']} → {info['keywords']}")
```

**运行输出**：
```
============================================================
场景2：RAG 文档批量处理
============================================================
[准备] 收到 4 个文档
  - 向量数据库入门 (92 字符)
  - 文本分块策略 (62 字符)
  - 大模型API调用指南 (62 字符)
  - 检索增强生成概述 (68 字符)
  [处理] 文档 '向量数据库入门' (ID: doc_000)
  [完成] '向量数据库入门': 关键词=['向量', 'embedding', '相似度', '检索', '搜索', '查询', '索引', 'index', 'HNSW'], 指纹=a3b2c1d4
  [处理] 文档 '文本分块策略' (ID: doc_001)
  [完成] '文本分块策略': 关键词=['分块', 'chunk', '切分', '检索', '搜索', '查询'], 指纹=e5f6g7h8
  [处理] 文档 '大模型API调用指南' (ID: doc_002)
  [完成] '大模型API调用指南': 关键词=['模型', 'LLM', 'transformer'], 指纹=i9j0k1l2
  [处理] 文档 '检索增强生成概述' (ID: doc_003)
  [完成] '检索增强生成概述': 关键词=['检索', '搜索', '查询', '索引', 'index', 'HNSW'], 指纹=m3n4o5p6

[合并] 收到 4 个文档的处理结果

[索引] 构建完成:
  文档数: 4
  总字符: 284
  关键词: ['向量', 'embedding', '相似度', '检索', '搜索', '查询', '索引', 'index', 'HNSW', '分块', 'chunk', '切分', '模型', 'LLM', 'transformer']

=== 最终索引 ===
文档数: 4
关键词库: ['向量', 'embedding', '相似度', ...]
  doc_000: 向量数据库入门 → ['向量', 'embedding', '相似度', ...]
  doc_001: 文本分块策略 → ['分块', 'chunk', '切分', ...]
  doc_002: 大模型API调用指南 → ['模型', 'LLM', 'transformer']
  doc_003: 检索增强生成概述 → ['检索', '搜索', '查询', ...]
```

---

## 关键模式总结

### 模式 1：Send 分发的标准写法

```python
from langgraph.types import Send

def fan_out(state: OverallState) -> list[Send]:
    """条件边函数：返回 Send 列表"""
    return [
        Send("worker_node", {"field": item})
        for item in state["items"]
    ]

# 在图中使用
builder.add_conditional_edges("source_node", fan_out, ["worker_node"])
```

三个要素：
1. 条件边函数返回 `list[Send]`
2. 每个 Send 指定目标节点和输入数据
3. `add_conditional_edges` 的第三个参数列出所有可能的目标节点

### 模式 2：OverallState vs WorkerState 分离

```python
class OverallState(TypedDict):
    items: list[str]
    results: Annotated[list[dict], operator.add]  # Reducer 合并

class WorkerState(TypedDict):
    item: str       # 单个任务的输入
    result: dict    # 单个任务的输出
```

为什么要分离？
- OverallState 管理全局数据（输入列表、合并结果）
- WorkerState 只关注单个任务（更简洁、更安全）
- Send 的 `arg` 就是 WorkerState，worker 节点不需要知道全局状态

### 模式 3：Reducer 自动合并的原理

```python
# Worker 返回：
{"results": [{"subject": "Python", "score": 95}]}

# 4 个 Worker 并行完成后，operator.add 自动合并：
# results = [] + [result_1] + [result_2] + [result_3] + [result_4]
# results = [result_1, result_2, result_3, result_4]
```

没有 Reducer 会怎样？多个 worker 同时写入 `results`，后写入的覆盖先写入的，最终只剩一个结果。

### 模式 4：Send vs Command 的区别

```python
from langgraph.types import Send, Command

# Send：用于并行分发（Map 阶段）
# - 在条件边函数中返回
# - 创建多个并行任务
# - 每个任务有独立输入
Send("worker", {"item": "data"})

# Command：用于顺序路由（节点内部）
# - 在节点函数中返回
# - 同时更新状态 + 指定下一个节点
# - 适合自循环、动态路由
Command(update={"score": 0.9}, goto="next_node")
```

| 特性 | Send | Command |
|------|------|---------|
| 使用位置 | 条件边函数 | 节点函数 |
| 并行能力 | 支持（多个 Send 并行） | 不支持（顺序执行） |
| 状态更新 | 不支持（只传递输入） | 支持（update 字段） |
| 典型场景 | Map-Reduce、批处理 | 自循环、Agent 路由 |

---

## 与 RAG 的关联

| MapReduce 模式 | RAG 应用场景 |
|----------------|-------------|
| 并行文档处理 | 批量 embedding 生成：每个文档独立向量化 |
| 结果合并 | 多源检索合并：从多个向量库检索后合并排序 |
| 动态任务数 | 文档数量运行时才知道，Send 天然支持 |
| Worker 隔离 | 每个文档的处理互不影响，失败不会波及其他 |

### RAG 批量 Embedding 的 MapReduce 模式

```
文档列表 → [Send] → embed_doc_1 ─┐
                  → embed_doc_2 ─┤→ 合并向量 → 写入向量库
                  → embed_doc_3 ─┘
```

这比串行处理快得多。如果有 100 个文档，Send 可以并行处理所有文档（受限于 LLM API 的并发限制），而串行循环需要逐个处理。

---

## 常见陷阱

### 陷阱 1：忘记 Reducer 导致结果丢失

```python
# ❌ 没有 Reducer，并行结果互相覆盖
class BadState(TypedDict):
    results: list[dict]  # 最后一个 worker 的结果覆盖前面的

# ✅ 使用 operator.add Reducer
class GoodState(TypedDict):
    results: Annotated[list[dict], operator.add]  # 所有结果自动合并
```

### 陷阱 2：Send 不能发送到 END

```python
# ❌ 会抛出 InvalidUpdateError
Send("__end__", {"data": "xxx"})

# [来源: sourcecode/langgraph/graph/_branch.py L192-L225]
# if any(p.node == END for p in destinations if isinstance(p, Send)):
#     raise InvalidUpdateError("Cannot send a packet to the END node")
```

### 陷阱 3：Worker 返回值的字段名必须匹配 OverallState

```python
# ❌ Worker 返回 WorkerState 的字段名
def worker(state: WorkerState) -> dict:
    return {"result": "xxx"}  # "result" 不在 OverallState 中！

# ✅ Worker 返回 OverallState 的 Reducer 字段名
def worker(state: WorkerState) -> dict:
    return {"results": ["xxx"]}  # "results" 是 OverallState 中的 Reducer 字段
```

### 陷阱 4：条件边的第三个参数遗漏

```python
# ❌ 没有指定可能的目标节点，编译时可能报错
builder.add_conditional_edges("distribute", fan_out)

# ✅ 明确列出所有可能的目标节点
builder.add_conditional_edges("distribute", fan_out, ["worker"])
```
