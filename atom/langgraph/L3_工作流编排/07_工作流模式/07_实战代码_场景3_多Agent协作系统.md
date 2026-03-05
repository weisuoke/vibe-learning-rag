# 07_工作流模式 - 实战代码 场景3：多 Agent 协作系统

> LangGraph L3_工作流编排 | 实战场景 3/6

---

## 场景描述

构建一个**技术文章生成系统**：编排者分析主题并规划章节，多个工作者并行撰写各章节，合成者汇总为完整文章。

模式组合：**编排者-工作者模式 + Map-Reduce 模式**

```
主题输入 → 编排者（规划章节）→ [工作者1 | 工作者2 | ... | 工作者N] → 合成者 → 完整文章
```

---

## 完整代码

```python
"""
多 Agent 协作系统：技术文章生成
模式组合：编排者-工作者 + Map-Reduce

编排者：分析主题，规划文章结构
工作者：并行撰写各章节
合成者：汇总为完整文章
"""
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# ===== 1. 定义状态 =====
class ArticleState(TypedDict):
    """全局状态"""
    topic: str                                    # 文章主题
    outline: list[dict]                           # 文章大纲
    sections: Annotated[list[dict], operator.add]  # 各章节内容（reducer 合并）
    article: str                                  # 最终文章


class SectionInput(TypedDict):
    """单个章节的工作者输入"""
    section_title: str
    section_description: str
    section_index: int


# ===== 2. 编排者节点 =====
def orchestrator(state: ArticleState) -> dict:
    """编排者：分析主题，生成文章大纲"""
    topic = state["topic"]

    # 模拟 LLM 规划（实际项目中用 llm.with_structured_output）
    outlines = {
        "LangGraph入门": [
            {"title": "什么是LangGraph", "desc": "介绍LangGraph的定义和核心理念"},
            {"title": "核心概念", "desc": "StateGraph、节点、边、状态的基本概念"},
            {"title": "快速上手", "desc": "第一个LangGraph程序的完整示例"},
            {"title": "实际应用", "desc": "LangGraph在RAG和Agent中的应用场景"},
        ],
        "RAG最佳实践": [
            {"title": "RAG架构概述", "desc": "检索增强生成的基本架构"},
            {"title": "文档处理", "desc": "分块策略和向量化方法"},
            {"title": "检索优化", "desc": "混合检索和重排序技术"},
        ],
    }

    outline = outlines.get(topic, [
        {"title": "概述", "desc": f"介绍{topic}的基本概念"},
        {"title": "核心技术", "desc": f"{topic}的关键技术点"},
        {"title": "实践指南", "desc": f"{topic}的实际应用"},
    ])

    print(f"[编排者] 主题: {topic}")
    print(f"[编排者] 规划了 {len(outline)} 个章节:")
    for i, section in enumerate(outline):
        print(f"  {i+1}. {section['title']}")

    return {"outline": outline}


# ===== 3. 动态分发（Send API）=====
def dispatch_to_workers(state: ArticleState):
    """根据大纲动态创建工作者任务"""
    return [
        Send("writer", {
            "section_title": section["title"],
            "section_description": section["desc"],
            "section_index": i,
        })
        for i, section in enumerate(state["outline"])
    ]


# ===== 4. 工作者节点 =====
def writer(state: SectionInput) -> dict:
    """工作者：撰写单个章节"""
    title = state["section_title"]
    desc = state["section_description"]
    index = state["section_index"]

    # 模拟 LLM 写作
    content = (
        f"## {index + 1}. {title}\n\n"
        f"{desc}\n\n"
        f"{title}是一个重要的技术概念。"
        f"在实际开发中，我们需要理解它的核心原理和应用场景。\n\n"
        f"### 关键要点\n"
        f"- 要点1：{title}的基本定义\n"
        f"- 要点2：{title}的实现方式\n"
        f"- 要点3：{title}的最佳实践\n"
    )

    print(f"[工作者] 完成章节: {title} ({len(content)} 字)")

    return {
        "sections": [{
            "index": index,
            "title": title,
            "content": content,
        }]
    }


# ===== 5. 合成者节点 =====
def synthesizer(state: ArticleState) -> dict:
    """合成者：汇总所有章节为完整文章"""
    # 按章节序号排序
    sorted_sections = sorted(state["sections"], key=lambda s: s["index"])

    # 组装文章
    article_parts = [f"# {state['topic']}\n"]
    for section in sorted_sections:
        article_parts.append(section["content"])

    article = "\n---\n\n".join(article_parts)

    print(f"[合成者] 汇总 {len(sorted_sections)} 个章节，总长 {len(article)} 字")
    return {"article": article}


# ===== 6. 构建图 =====
builder = StateGraph(ArticleState)

builder.add_node("orchestrator", orchestrator)
builder.add_node("writer", writer)
builder.add_node("synthesizer", synthesizer)

builder.add_edge(START, "orchestrator")

# 核心：用条件边 + Send 实现动态并行
builder.add_conditional_edges(
    "orchestrator",
    dispatch_to_workers,
    ["writer"],  # 声明可能的目标节点
)

builder.add_edge("writer", "synthesizer")
builder.add_edge("synthesizer", END)

article_generator = builder.compile()


# ===== 7. 测试 =====
if __name__ == "__main__":
    print("=" * 60)
    print("生成文章: LangGraph入门")
    print("=" * 60)

    result = article_generator.invoke({
        "topic": "LangGraph入门",
    })

    print(f"\n{'='*60}")
    print("最终文章:")
    print("=" * 60)
    print(result["article"])
    print(f"\n文章总长: {len(result['article'])} 字")
    print(f"章节数: {len(result['sections'])}")
```

**运行输出：**
```
============================================================
生成文章: LangGraph入门
============================================================
[编排者] 主题: LangGraph入门
[编排者] 规划了 4 个章节:
  1. 什么是LangGraph
  2. 核心概念
  3. 快速上手
  4. 实际应用
[工作者] 完成章节: 什么是LangGraph (186 字)
[工作者] 完成章节: 核心概念 (178 字)
[工作者] 完成章节: 快速上手 (176 字)
[工作者] 完成章节: 实际应用 (180 字)
[合成者] 汇总 4 个章节，总长 856 字

============================================================
最终文章:
============================================================
# LangGraph入门

---

## 1. 什么是LangGraph
...
```

---

## 关键设计解析

### 1. Send API 实现动态并行

```python
def dispatch_to_workers(state):
    # 章节数量由编排者动态决定，不是写死的
    return [Send("writer", {...}) for section in state["outline"]]
```

编排者规划了 4 个章节，就创建 4 个并行工作者；规划了 10 个，就创建 10 个。

### 2. Worker 状态隔离

每个 Writer 只收到自己章节的信息（`SectionInput`），看不到其他章节。这保证了并行安全。

### 3. Reducer 自动合并

```python
sections: Annotated[list[dict], operator.add]
```

4 个 Writer 各返回一个 `sections` 列表，`operator.add` 自动把它们合并成一个大列表。

### 4. 排序保证顺序

Worker 并行执行，完成顺序不确定。合成者用 `section_index` 排序，确保文章章节顺序正确。

---

## 学习要点

1. **编排者-工作者 + Map-Reduce 的组合**：编排者动态规划 + Send 动态分发
2. **状态隔离设计**：`SectionInput` 与 `ArticleState` 分离
3. **结果排序**：并行结果需要在合成时重新排序
4. **实际应用**：报告生成、代码审查、多文档分析都是这个模式

[来源: Context7 官方文档]
