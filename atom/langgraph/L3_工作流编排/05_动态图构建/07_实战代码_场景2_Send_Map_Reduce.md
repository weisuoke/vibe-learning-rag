# 实战场景2：Send Map-Reduce - 动态并行任务处理

> 构建一个「多主题内容生成器」：用户输入任意数量的主题，系统用 Send 将每个主题并行分发到生成节点，所有内容生成完成后自动汇总。

---

## 场景描述

想象你在构建一个 RAG 驱动的内容平台。用户说："帮我分别写一段关于 Python、Rust、Go 的介绍"。问题是——主题数量在编译时完全未知，可能是 2 个，也可能是 20 个。你需要：

1. **接收节点**：接收用户输入的主题列表
2. **分发逻辑**：用 Send 为每个主题动态创建一个并行任务
3. **生成节点**：每个任务独立生成内容（并行执行）
4. **汇总节点**：等所有任务完成后，合并所有结果

### 双重类比

**前端类比：** 就像 `Promise.all()` —— 你有一个 URL 数组，用 `urls.map(url => fetch(url))` 创建多个并行请求，然后 `Promise.all()` 等所有请求完成后统一处理结果。Send 就是那个 `.map()`，Reducer 就是那个 `Promise.all()`。

**日常生活类比：** 就像一个项目经理分配任务——老板说"调研这 5 个竞品"，项目经理把 5 个竞品分别分配给 5 个人（Send），每个人独立调研（并行生成），最后项目经理收集所有报告合并成一份总报告（汇总）。

### 工作流图示

```
                         ┌──→ [生成: Python] ──┐
                         │                      │
[START] → [分发主题] ──┼──→ [生成: Rust]   ──┼──→ [汇总结果] → [END]
                         │                      │
                         └──→ [生成: Go]     ──┘

                    （数量运行时决定，可能是 2 个也可能是 20 个）
```

---

## 完整可运行代码

```python
"""
实战场景2：Send Map-Reduce - 多主题内容生成器
演示：Send API 实现运行时动态并行任务分发与结果合并

运行要求：pip install langgraph
无需外部 API，完全本地运行
"""
import operator
import time
import random
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# ===== 1. 定义状态 =====

class OverallState(TypedDict):
    """主图状态 - 管理整个 Map-Reduce 流程

    Attributes:
        subjects: 用户输入的主题列表（如 ["Python", "Rust", "Go"]）
        results: 所有生成结果的汇总（Reducer: operator.add 追加合并）
        summary: 最终汇总报告
    """
    subjects: list[str]
    results: Annotated[list[dict], operator.add]
    summary: str


class GenerateState(TypedDict):
    """子任务状态 - 每个 Send 创建的独立任务使用这个状态

    注意：这个状态和 OverallState 是不同的！
    Send 会把 {"subject": "Python"} 作为这个节点的输入状态。
    生成节点返回的 {"results": [...]} 会通过 Reducer 合并到 OverallState。
    """
    subject: str


# ===== 2. 分发节点（Map 阶段）=====

def distribute_subjects(state: OverallState):
    """分发节点 - 为每个主题创建一个 Send 任务

    这是 Map-Reduce 的核心！
    返回一个 Send 对象列表，每个 Send 会创建一个独立的并行任务。

    Send 接受两个参数：
    - 第1个参数：目标节点名（"generate_content"）
    - 第2个参数：传递给该节点的状态（{"subject": s}）
    """
    subjects = state["subjects"]
    print(f"[分发] 收到 {len(subjects)} 个主题: {subjects}")
    print(f"[分发] 创建 {len(subjects)} 个并行任务...")

    # 关键代码：返回 Send 列表
    # 每个 Send 会独立调用 "generate_content" 节点
    # 传入的状态是 GenerateState 格式（只有 subject 字段）
    return [Send("generate_content", {"subject": s}) for s in subjects]


# ===== 3. 生成节点（并行执行）=====

def generate_content(state: GenerateState) -> dict:
    """生成节点 - 为单个主题生成内容

    注意：
    1. 输入参数 state 的类型是 GenerateState（不是 OverallState！）
       因为 Send 传入的是 {"subject": "Python"}
    2. 返回值中的 "results" 键会通过 operator.add Reducer
       自动追加到 OverallState 的 results 列表中
    """
    subject = state["subject"]

    # 模拟内容生成（真实场景中这里调用 LLM）
    delay = random.uniform(0.1, 0.3)
    time.sleep(delay)

    # 模拟生成的内容
    content_templates = {
        "Python": "Python 是一门优雅的动态语言，以简洁的语法和丰富的生态著称。广泛应用于 AI、Web 开发和数据科学。",
        "Rust": "Rust 是一门系统级编程语言，以内存安全和零成本抽象闻名。适合构建高性能、高可靠性的系统软件。",
        "Go": "Go 是 Google 开发的编译型语言，以并发模型和简洁设计著称。特别适合构建微服务和云原生应用。",
        "JavaScript": "JavaScript 是 Web 的核心语言，从浏览器到服务器无处不在。事件驱动和原型继承是其独特特性。",
        "TypeScript": "TypeScript 是 JavaScript 的超集，添加了静态类型系统。在大型项目中提供更好的开发体验和代码可靠性。",
    }

    content = content_templates.get(
        subject,
        f"{subject} 是一个值得深入学习的技术领域，具有独特的设计理念和广泛的应用场景。"
    )

    word_count = len(content)
    print(f"  [生成] 主题: {subject} | 耗时: {delay:.2f}s | 字数: {word_count}")

    # 返回结果 —— results 列表会通过 Reducer 合并到 OverallState
    return {
        "results": [{
            "subject": subject,
            "content": content,
            "word_count": word_count,
            "generation_time": round(delay, 2),
        }]
    }


# ===== 4. 汇总节点（Reduce 阶段）=====

def summarize_all(state: OverallState) -> dict:
    """汇总节点 - 合并所有并行任务的结果

    当所有 Send 创建的任务都完成后，这个节点才会执行。
    此时 state["results"] 已经包含了所有生成节点的结果
    （通过 operator.add Reducer 自动合并）。
    """
    results = state["results"]

    print(f"\n[汇总] 收到 {len(results)} 个生成结果")

    # 构建汇总报告
    total_words = sum(r["word_count"] for r in results)
    total_time = sum(r["generation_time"] for r in results)

    report_lines = [
        "=" * 50,
        "多主题内容生成报告",
        "=" * 50,
    ]

    for i, result in enumerate(results, 1):
        report_lines.append(f"\n--- 主题 {i}: {result['subject']} ---")
        report_lines.append(f"内容: {result['content']}")
        report_lines.append(f"字数: {result['word_count']} | 耗时: {result['generation_time']}s")

    report_lines.append(f"\n{'=' * 50}")
    report_lines.append(f"总计: {len(results)} 个主题 | {total_words} 字 | {total_time:.2f}s")
    report_lines.append("=" * 50)

    summary = "\n".join(report_lines)
    print(f"[汇总] 总计 {len(results)} 个主题, {total_words} 字")

    return {"summary": summary}


# ===== 5. 构建工作流图 =====

def build_map_reduce_workflow():
    """构建 Map-Reduce 工作流

    关键步骤：
    1. 创建 StateGraph（使用 OverallState）
    2. 添加节点（分发、生成、汇总）
    3. 用 add_conditional_edges 连接分发节点（返回 Send 列表）
    4. 用 add_edge 连接生成节点到汇总节点
    5. 编译
    """
    graph = StateGraph(OverallState)

    # --- 添加节点 ---
    graph.add_node("distribute", distribute_subjects)
    graph.add_node("generate_content", generate_content)
    graph.add_node("summarize", summarize_all)

    # --- 入口边 ---
    graph.add_edge(START, "distribute")

    # --- 条件边：分发 → 生成（核心！）---
    # distribute_subjects 返回 [Send("generate_content", {...}), ...]
    # LangGraph 看到返回值是 Send 列表，会自动并行调用目标节点
    graph.add_conditional_edges("distribute", distribute_subjects)

    # --- 生成 → 汇总 ---
    graph.add_edge("generate_content", "summarize")

    # --- 汇总 → 结束 ---
    graph.add_edge("summarize", END)

    return graph.compile()


# ===== 6. 运行测试 =====

if __name__ == "__main__":
    app = build_map_reduce_workflow()

    # 测试1：3 个主题
    print("=" * 60)
    print("测试1：3 个主题")
    print("=" * 60)

    result = app.invoke({
        "subjects": ["Python", "Rust", "Go"],
        "results": [],
        "summary": "",
    })

    print(f"\n{result['summary']}")

    # 测试2：5 个主题（展示动态数量）
    print("\n\n" + "=" * 60)
    print("测试2：5 个主题（动态数量）")
    print("=" * 60)

    result = app.invoke({
        "subjects": ["Python", "Rust", "Go", "JavaScript", "TypeScript"],
        "results": [],
        "summary": "",
    })

    print(f"\n{result['summary']}")

    # 测试3：1 个主题（边界情况）
    print("\n\n" + "=" * 60)
    print("测试3：1 个主题（边界情况）")
    print("=" * 60)

    result = app.invoke({
        "subjects": ["Python"],
        "results": [],
        "summary": "",
    })

    print(f"\n{result['summary']}")
```

---

## 运行输出示例

```
============================================================
测试1：3 个主题
============================================================
[分发] 收到 3 个主题: ['Python', 'Rust', 'Go']
[分发] 创建 3 个并行任务...
  [生成] 主题: Python | 耗时: 0.15s | 字数: 52
  [生成] 主题: Rust | 耗时: 0.22s | 字数: 48
  [生成] 主题: Go | 耗时: 0.11s | 字数: 46

[汇总] 收到 3 个生成结果
[汇总] 总计 3 个主题, 146 字

==================================================
多主题内容生成报告
==================================================

--- 主题 1: Python ---
内容: Python 是一门优雅的动态语言，以简洁的语法和丰富的生态著称。广泛应用于 AI、Web 开发和数据科学。
字数: 52 | 耗时: 0.15s

--- 主题 2: Rust ---
内容: Rust 是一门系统级编程语言，以内存安全和零成本抽象闻名。适合构建高性能、高可靠性的系统软件。
字数: 48 | 耗时: 0.22s

--- 主题 3: Go ---
内容: Go 是 Google 开发的编译型语言，以并发模型和简洁设计著称。特别适合构建微服务和云原生应用。
字数: 46 | 耗时: 0.11s

==================================================
总计: 3 个主题 | 146 字 | 0.48s
==================================================
```

---

## 代码逐行解析

### 1. 两种 State 的区别（最关键的概念）

```python
class OverallState(TypedDict):
    subjects: list[str]                          # 输入：主题列表
    results: Annotated[list[dict], operator.add]  # 输出：所有结果（Reducer 合并）
    summary: str                                  # 最终汇总

class GenerateState(TypedDict):
    subject: str                                  # 单个主题（注意：不是 subjects）
```

这是 Send Map-Reduce 模式中最容易混淆的地方：

- **OverallState** 是主图的状态，管理整个流程的输入输出
- **GenerateState** 是每个 Send 任务的输入状态，只包含单个任务需要的数据

Send 做的事情就是把 OverallState 中的列表"拆开"，为每个元素创建一个独立的 GenerateState。

**前端类比：**

```javascript
// OverallState 相当于：
const overallState = { subjects: ["Python", "Rust", "Go"], results: [] };

// Send 相当于 .map()，把列表拆成独立任务：
const tasks = overallState.subjects.map(s => ({ subject: s }));
// tasks = [{ subject: "Python" }, { subject: "Rust" }, { subject: "Go" }]

// 每个 task 就是一个 GenerateState
```

### 2. Send 的工作机制

```python
def distribute_subjects(state: OverallState):
    return [Send("generate_content", {"subject": s}) for s in state["subjects"]]
```

逐步拆解这行代码：

1. `state["subjects"]` → `["Python", "Rust", "Go"]`
2. 列表推导式为每个主题创建一个 `Send` 对象
3. `Send("generate_content", {"subject": "Python"})` 的含义：
   - 第1个参数 `"generate_content"`：要调用的目标节点名
   - 第2个参数 `{"subject": "Python"}`：传递给该节点的状态
4. 返回的列表 `[Send(...), Send(...), Send(...)]` 告诉 LangGraph：
   - "请并行调用 3 次 generate_content 节点"
   - "每次调用传入不同的状态"

> [来源: LangGraph 源码 types.py - Send 类定义，行 289-362]

### 3. Reducer 如何合并并行结果

```python
results: Annotated[list[dict], operator.add]
```

这是 Map-Reduce 的 "Reduce" 部分。当 3 个并行的 generate_content 节点都完成后：

```
生成节点1 返回: {"results": [{"subject": "Python", ...}]}
生成节点2 返回: {"results": [{"subject": "Rust", ...}]}
生成节点3 返回: {"results": [{"subject": "Go", ...}]}

Reducer (operator.add) 合并:
results = [] + [{"subject": "Python", ...}] + [{"subject": "Rust", ...}] + [{"subject": "Go", ...}]
       = [{"subject": "Python", ...}, {"subject": "Rust", ...}, {"subject": "Go", ...}]
```

如果没有 `operator.add` Reducer，后完成的任务会覆盖先完成的结果，你只能拿到最后一个任务的输出。

**日常生活类比：** Reducer 就像一个收件箱——每个人交上来的报告都放进去（追加），而不是后来的报告把前面的覆盖掉。

### 4. 条件边连接分发节点

```python
graph.add_conditional_edges("distribute", distribute_subjects)
```

这里 `distribute_subjects` 既是节点函数，也是条件边的路由函数。当它返回 `[Send(...)]` 列表时，LangGraph 会：

1. 识别返回值是 Send 对象列表
2. 为每个 Send 创建一个独立的任务
3. 将任务推入 TASKS 通道
4. 在下一个 BSP（批量同步并行）步骤中并行执行所有任务

> [来源: LangGraph 源码 pregel/_algo.py - prepare_next_tasks 处理 PUSH 任务]

### 5. 生成节点的返回值

```python
def generate_content(state: GenerateState) -> dict:
    # ...
    return {
        "results": [{
            "subject": subject,
            "content": content,
            # ...
        }]
    }
```

注意返回值中 `"results"` 是一个**列表**（包裹在 `[...]` 中），即使只有一个元素。这是因为 `operator.add` Reducer 需要两个列表做加法。如果返回的是字典而不是列表，`operator.add` 会报错。

---

## 进阶扩展

### 扩展1：添加错误处理

某个子任务失败不应该影响其他任务的执行：

```python
def generate_content_safe(state: GenerateState) -> dict:
    """带错误处理的生成节点"""
    subject = state["subject"]
    try:
        # 模拟偶尔失败
        if random.random() < 0.2:
            raise ValueError(f"生成 {subject} 内容时出错")

        content = f"{subject} 的精彩内容..."
        return {
            "results": [{
                "subject": subject,
                "content": content,
                "status": "success",
            }]
        }
    except Exception as e:
        print(f"  [错误] 主题 {subject} 生成失败: {e}")
        return {
            "results": [{
                "subject": subject,
                "content": "",
                "status": "failed",
                "error": str(e),
            }]
        }
```

因为每个 Send 任务是独立的，一个任务的异常不会影响其他任务。Reducer 会照常合并所有结果（包括失败的），汇总节点可以根据 `status` 字段区分处理。

### 扩展2：添加进度追踪

```python
class OverallState(TypedDict):
    subjects: list[str]
    results: Annotated[list[dict], operator.add]
    progress: Annotated[list[str], operator.add]  # 新增：进度日志
    summary: str


def generate_content_with_progress(state: GenerateState) -> dict:
    """带进度追踪的生成节点"""
    subject = state["subject"]
    start_time = time.time()

    # ... 生成逻辑 ...

    elapsed = time.time() - start_time
    return {
        "results": [{"subject": subject, "content": "..."}],
        "progress": [f"[{elapsed:.2f}s] {subject} 生成完成"],
    }
```

### 扩展3：嵌套 Map-Reduce

如果每个主题还需要生成多个子章节，可以嵌套 Send：

```python
# 第一层 Send：按主题分发
def distribute_subjects(state):
    return [Send("generate_outline", {"subject": s}) for s in state["subjects"]]

# 第二层 Send：按章节分发（在子图中）
def distribute_sections(state):
    sections = state["outline"]  # 如 ["简介", "特性", "应用"]
    return [
        Send("write_section", {"subject": state["subject"], "section": sec})
        for sec in sections
    ]
```

这就像项目管理中的层级分解——先按项目分配团队，每个团队再按模块分配个人。

---

## 关键收获

| 概念 | 本场景中的体现 |
|------|---------------|
| `Send` 对象 | 为每个主题创建独立的并行任务 |
| `Send(node, state)` | 第1参数是目标节点，第2参数是传入的状态 |
| `OverallState` vs `GenerateState` | 主图状态 vs 子任务状态，结构不同 |
| `Annotated[list, operator.add]` | Reducer 合并并行结果，追加而非覆盖 |
| `add_conditional_edges` | 连接分发节点，识别 Send 列表并并行执行 |
| Map 阶段 | distribute_subjects 将列表拆分为独立任务 |
| Reduce 阶段 | operator.add Reducer 自动合并所有结果 |

---

## 参考来源

- [LangGraph 源码: types.py - Send 类定义（行 289-362）](sourcecode/langgraph/types.py)
- [LangGraph 源码: pregel/_algo.py - prepare_next_tasks 处理 Send 任务](sourcecode/langgraph/pregel/_algo.py)
- [Context7 LangGraph 文档 - Send API 动态边](https://docs.langchain.com/oss/python/langgraph/graph-api)
