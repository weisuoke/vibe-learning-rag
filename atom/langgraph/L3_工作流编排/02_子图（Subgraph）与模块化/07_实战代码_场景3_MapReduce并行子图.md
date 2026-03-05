# 实战代码 - 场景3：Map-Reduce 并行子图

## 场景说明

本示例展示如何使用 `Send` + 子图实现完整的 **Map-Reduce 并行处理模式**。

**实际应用场景：多主题内容生成系统**
- 输入：一组主题 `["Python", "Rust", "Go"]`
- Map 阶段：使用 `Send` 将每个主题动态分发到子图
- 子图处理：为每个主题经过「编辑 -> 生成 -> 润色」三步流水线
- Reduce 阶段：通过 `Annotated[list, operator.add]` 自动收集所有结果

**核心技术点：**
- `Send` 动态分发实现 fan-out
- 子图 `input_schema` / `output_schema` 定义接口契约
- `operator.add` reducer 自动聚合并行结果
- 子图编译后作为节点添加到父图

---

## 架构流程图

```
                    ┌─────────────┐
                    │    START    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  准备主题    │  prepare_topics
                    └──────┬──────┘
                           │
                    条件边: dispatch_to_subgraph
                    返回 Send 列表 (fan-out)
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   Send("generate",  Send("generate",  Send("generate",
    {subject:         {subject:         {subject:
     "Python"})        "Rust"})          "Go"})
          │                │                │
          ▼                ▼                ▼
   ┌────────────┐  ┌────────────┐  ┌────────────┐
   │  子图实例1  │  │  子图实例2  │  │  子图实例3  │
   │            │  │            │  │            │
   │ edit_topic │  │ edit_topic │  │ edit_topic │
   │     │      │  │     │      │  │     │      │
   │     ▼      │  │     ▼      │  │     ▼      │
   │ generate   │  │ generate   │  │ generate   │
   │     │      │  │     │      │  │     │      │
   │     ▼      │  │     ▼      │  │     ▼      │
   │  polish    │  │  polish    │  │  polish    │
   └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
          │                │                │
          └────────────────┼────────────────┘
                           │
                    Reducer: operator.add
                    自动合并 articles 列表
                           │
                    ┌──────▼──────┐
                    │  summarize  │  汇总所有文章
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │     END     │
                    └─────────────┘
```

---

## 完整可运行代码

```python
"""
LangGraph Map-Reduce 并行子图实战
场景：多主题内容生成系统

演示：
1. Send 动态分发多个主题到子图（Map 阶段）
2. 子图内部三步流水线：编辑 → 生成 → 润色
3. operator.add reducer 自动聚合结果（Reduce 阶段）
4. 子图使用 input_schema / output_schema 定义接口
"""

from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# ============================================================
# 第一部分：定义状态结构
# ============================================================

# --- 父图状态 ---
class OverallState(TypedDict):
    """父图的整体状态"""
    subjects: list[str]                                    # 输入：要处理的主题列表
    articles: Annotated[list[dict], operator.add]          # 输出：所有文章（reducer 自动合并）
    summary: str                                           # 最终汇总


# --- 子图内部状态 ---
class ArticleState(TypedDict):
    """子图的内部工作状态（比输入/输出更丰富）"""
    subject: str          # 主题名称
    outline: str          # 编辑阶段产出的大纲
    draft: str            # 生成阶段产出的草稿
    final_text: str       # 润色阶段产出的最终文本


# --- 子图输入 Schema ---
class ArticleInput(TypedDict):
    """子图接受的输入（Send 传入的数据结构）"""
    subject: str          # 只需要主题名称


# --- 子图输出 Schema ---
class ArticleOutput(TypedDict):
    """子图返回给父图的输出"""
    articles: list[dict]  # 返回文章列表（与父图 articles 键名一致，触发 reducer）


# ============================================================
# 第二部分：子图节点函数
# ============================================================

def edit_topic(state: ArticleState) -> dict:
    """
    编辑节点：根据主题生成写作大纲

    这是子图的第一步，接收主题并规划文章结构。
    实际项目中可以调用 LLM 生成大纲。
    """
    subject = state["subject"]
    print(f"  [编辑] 为 '{subject}' 规划大纲...")

    # 模拟生成大纲（不依赖外部 API）
    outline = (
        f"《{subject} 技术概览》\n"
        f"  1. {subject} 的起源与设计哲学\n"
        f"  2. {subject} 的核心特性\n"
        f"  3. {subject} 的典型应用场景\n"
        f"  4. {subject} 的未来展望"
    )

    print(f"  [编辑] '{subject}' 大纲完成")
    return {"outline": outline}


def generate_content(state: ArticleState) -> dict:
    """
    生成节点：根据大纲生成文章草稿

    这是子图的第二步，基于大纲写出完整草稿。
    实际项目中可以调用 LLM 生成内容。
    """
    subject = state["subject"]
    outline = state["outline"]
    print(f"  [生成] 为 '{subject}' 撰写草稿...")

    # 模拟生成草稿内容
    draft = (
        f"# {subject} 技术概览\n\n"
        f"## 起源与设计哲学\n"
        f"{subject} 诞生的初衷是解决特定的工程问题。"
        f"它的设计哲学强调简洁性和实用性。\n\n"
        f"## 核心特性\n"
        f"{subject} 提供了现代编程语言的关键特性，"
        f"包括类型系统、并发模型和丰富的标准库。\n\n"
        f"## 典型应用场景\n"
        f"{subject} 广泛应用于 Web 开发、系统编程、"
        f"数据科学等领域。\n\n"
        f"## 未来展望\n"
        f"{subject} 社区活跃，生态系统持续壮大，"
        f"未来将在更多领域发挥作用。"
    )

    print(f"  [生成] '{subject}' 草稿完成 ({len(draft)} 字符)")
    return {"draft": draft}


def polish_article(state: ArticleState) -> dict:
    """
    润色节点：对草稿进行最终润色

    这是子图的最后一步，输出最终文章。
    注意：返回值包含 articles 键，与父图状态对应，
    触发 operator.add reducer 进行结果聚合。
    """
    subject = state["subject"]
    draft = state["draft"]
    print(f"  [润色] 为 '{subject}' 执行最终润色...")

    # 模拟润色：添加元数据和格式化
    final_text = (
        f"{'='*50}\n"
        f"{draft}\n"
        f"{'='*50}\n"
        f"[作者: AI Writer | 主题: {subject} | 状态: 已发布]"
    )

    print(f"  [润色] '{subject}' 最终文章完成")

    # 关键：返回 articles 列表，与父图 OverallState 的 articles 键对应
    # operator.add reducer 会自动将所有子图的返回合并到一个列表中
    return {
        "articles": [{
            "subject": subject,
            "text": final_text,
            "word_count": len(final_text),
        }]
    }


# ============================================================
# 第三部分：构建子图
# ============================================================

def build_article_subgraph():
    """
    构建文章生成子图

    关键参数：
    - state_schema=ArticleState    子图内部工作状态
    - input_schema=ArticleInput    子图接受的输入（由 Send 传入）
    - output_schema=ArticleOutput  子图返回给父图的输出
    """
    subgraph_builder = StateGraph(
        ArticleState,
        input=ArticleInput,        # 定义子图入口接受的数据结构
        output=ArticleOutput,      # 定义子图出口返回的数据结构
    )

    # 添加三步流水线节点
    subgraph_builder.add_node("edit_topic", edit_topic)
    subgraph_builder.add_node("generate_content", generate_content)
    subgraph_builder.add_node("polish_article", polish_article)

    # 线性流程：编辑 → 生成 → 润色
    subgraph_builder.add_edge(START, "edit_topic")
    subgraph_builder.add_edge("edit_topic", "generate_content")
    subgraph_builder.add_edge("generate_content", "polish_article")
    subgraph_builder.add_edge("polish_article", END)

    # 编译子图
    return subgraph_builder.compile()


# ============================================================
# 第四部分：父图节点函数
# ============================================================

def prepare_topics(state: OverallState) -> dict:
    """
    准备节点：预处理主题列表

    在实际项目中，这里可以做主题去重、验证等操作。
    """
    subjects = state["subjects"]
    print(f"\n{'='*60}")
    print(f"准备处理 {len(subjects)} 个主题: {subjects}")
    print(f"{'='*60}")
    return {}  # 不修改状态，保持原样传递


def dispatch_to_subgraph(state: OverallState) -> list[Send]:
    """
    分发函数（条件边）：为每个主题创建一个 Send 对象

    这是 Map 阶段的核心：
    - 遍历所有主题
    - 为每个主题创建 Send("generate_article", {"subject": topic})
    - LangGraph 会为每个 Send 并行执行子图
    """
    sends = []
    for subject in state["subjects"]:
        print(f"  -> 分发主题: '{subject}' 到子图")
        sends.append(
            Send(
                "generate_article",    # 目标节点名（父图中注册的子图节点）
                {"subject": subject},  # 传给子图的输入（匹配 ArticleInput）
            )
        )
    print(f"共分发 {len(sends)} 个并行任务\n")
    return sends


def summarize_results(state: OverallState) -> dict:
    """
    汇总节点：对所有文章结果进行汇总

    这是 Reduce 之后的处理步骤。
    此时 state["articles"] 已经由 reducer 自动合并了所有子图的输出。
    """
    articles = state["articles"]
    print(f"\n{'='*60}")
    print(f"汇总阶段：收到 {len(articles)} 篇文章")
    print(f"{'='*60}")

    # 生成汇总
    summary_lines = [f"本次共生成 {len(articles)} 篇技术文章："]
    total_words = 0
    for i, article in enumerate(articles, 1):
        summary_lines.append(
            f"  {i}. {article['subject']} ({article['word_count']} 字符)"
        )
        total_words += article["word_count"]
    summary_lines.append(f"  总计: {total_words} 字符")

    summary = "\n".join(summary_lines)
    print(summary)

    return {"summary": summary}


# ============================================================
# 第五部分：构建并运行父图
# ============================================================

def build_main_graph():
    """构建完整的 Map-Reduce 父图"""

    # 编译子图
    article_subgraph = build_article_subgraph()

    # 构建父图
    builder = StateGraph(OverallState)

    # 添加节点
    builder.add_node("prepare_topics", prepare_topics)
    builder.add_node("generate_article", article_subgraph)   # 子图作为节点
    builder.add_node("summarize_results", summarize_results)

    # 定义边
    builder.add_edge(START, "prepare_topics")

    # 关键：条件边返回 Send 列表，实现 fan-out 到子图
    builder.add_conditional_edges("prepare_topics", dispatch_to_subgraph)

    # 所有子图执行完毕后（reducer 合并结果后），进入汇总节点
    builder.add_edge("generate_article", "summarize_results")
    builder.add_edge("summarize_results", END)

    return builder.compile()


# ============================================================
# 第六部分：运行示例
# ============================================================

if __name__ == "__main__":
    # 构建图
    graph = build_main_graph()

    # 准备输入
    initial_state = {
        "subjects": ["Python", "Rust", "Go"],
    }

    print("=" * 60)
    print("Map-Reduce 并行子图 - 多主题内容生成系统")
    print("=" * 60)

    # --- 方式1：invoke 直接执行 ---
    print("\n>>> 方式1：invoke 执行")
    result = graph.invoke(initial_state)

    print(f"\n--- 最终结果 ---")
    print(f"汇总: {result['summary']}")
    print(f"文章数: {len(result['articles'])}")
    for article in result["articles"]:
        print(f"\n主题: {article['subject']}")
        print(f"字数: {article['word_count']}")
        print(article["text"][:100] + "...")

    # --- 方式2：stream 流式执行，观察每一步 ---
    print("\n\n" + "=" * 60)
    print(">>> 方式2：stream 流式执行")
    print("=" * 60)

    for step in graph.stream(initial_state, stream_mode="updates"):
        # step 格式: {node_name: state_update}
        for node_name, update in step.items():
            print(f"\n[流式] 节点 '{node_name}' 输出:")
            if "articles" in update:
                for a in update["articles"]:
                    print(f"  文章: {a['subject']} ({a['word_count']} 字符)")
            elif "summary" in update:
                print(f"  汇总: {update['summary'][:80]}...")

    # --- 方式3：stream + subgraphs=True 监控子图内部 ---
    print("\n\n" + "=" * 60)
    print(">>> 方式3：stream + subgraphs=True 监控子图内部")
    print("=" * 60)

    for namespace, step in graph.stream(
        initial_state,
        stream_mode="updates",
        subgraphs=True,
    ):
        # namespace: () 表示父图, ("generate_article:xxx",) 表示子图
        level = "父图" if namespace == () else f"子图 {namespace}"
        for node_name, update in step.items():
            print(f"  [{level}] 节点 '{node_name}' -> keys: {list(update.keys())}")
```

---

## 运行输出示例

```
============================================================
Map-Reduce 并行子图 - 多主题内容生成系统
============================================================

>>> 方式1：invoke 执行

============================================================
准备处理 3 个主题: ['Python', 'Rust', 'Go']
============================================================
  -> 分发主题: 'Python' 到子图
  -> 分发主题: 'Rust' 到子图
  -> 分发主题: 'Go' 到子图
共分发 3 个并行任务

  [编辑] 为 'Python' 规划大纲...
  [编辑] 'Python' 大纲完成
  [生成] 为 'Python' 撰写草稿...
  [生成] 'Python' 草稿完成 (296 字符)
  [润色] 为 'Python' 执行最终润色...
  [润色] 'Python' 最终文章完成

  [编辑] 为 'Rust' 规划大纲...
  [编辑] 'Rust' 大纲完成
  [生成] 为 'Rust' 撰写草稿...
  [生成] 'Rust' 草稿完成 (284 字符)
  [润色] 为 'Rust' 执行最终润色...
  [润色] 'Rust' 最终文章完成

  [编辑] 为 'Go' 规划大纲...
  [编辑] 'Go' 大纲完成
  [生成] 为 'Go' 撰写草稿...
  [生成] 'Go' 草稿完成 (270 字符)
  [润色] 为 'Go' 执行最终润色...
  [润色] 'Go' 最终文章完成

============================================================
汇总阶段：收到 3 篇文章
============================================================
本次共生成 3 篇技术文章：
  1. Python (396 字符)
  2. Rust (384 字符)
  3. Go (370 字符)
  总计: 1150 字符

--- 最终结果 ---
汇总: 本次共生成 3 篇技术文章：
  1. Python (396 字符)
  2. Rust (384 字符)
  3. Go (370 字符)
  总计: 1150 字符
文章数: 3

主题: Python
字数: 396
==================================================
# Python 技术概览

## 起源与设计哲学
Python 诞生的初衷是...

>>> 方式3：stream + subgraphs=True 监控子图内部
============================================================
  [父图] 节点 'prepare_topics' -> keys: []
  [子图 ('generate_article:abc123',)] 节点 'edit_topic' -> keys: ['outline']
  [子图 ('generate_article:abc123',)] 节点 'generate_content' -> keys: ['draft']
  [子图 ('generate_article:abc123',)] 节点 'polish_article' -> keys: ['articles']
  [子图 ('generate_article:def456',)] 节点 'edit_topic' -> keys: ['outline']
  [子图 ('generate_article:def456',)] 节点 'generate_content' -> keys: ['draft']
  [子图 ('generate_article:def456',)] 节点 'polish_article' -> keys: ['articles']
  [子图 ('generate_article:ghi789',)] 节点 'edit_topic' -> keys: ['outline']
  [子图 ('generate_article:ghi789',)] 节点 'generate_content' -> keys: ['draft']
  [子图 ('generate_article:ghi789',)] 节点 'polish_article' -> keys: ['articles']
  [父图] 节点 'generate_article' -> keys: ['articles']
  [父图] 节点 'summarize_results' -> keys: ['summary']
```

---

## 关键机制解析

### 1. Send 分发 = Map 阶段

```python
def dispatch_to_subgraph(state: OverallState) -> list[Send]:
    # 每个 Send 创建一个独立的子图执行实例
    return [
        Send("generate_article", {"subject": s})
        for s in state["subjects"]
    ]
```

- `Send` 的第一个参数是父图中注册的节点名
- 第二个参数是传给子图的输入数据，必须匹配 `ArticleInput`
- 返回 N 个 Send = 并行创建 N 个子图实例

### 2. Reducer 合并 = Reduce 阶段

```python
class OverallState(TypedDict):
    articles: Annotated[list[dict], operator.add]  # 关键！
```

- 每个子图返回 `{"articles": [单个结果]}`
- `operator.add` 会自动将所有子图的 `articles` 列表拼接
- 最终 `state["articles"]` 包含所有结果

### 3. 子图接口契约

```python
subgraph = StateGraph(
    ArticleState,           # 内部状态（丰富）
    input=ArticleInput,     # 入口（精简）
    output=ArticleOutput,   # 出口（精简）
)
```

- `input` 定义 Send 可以传什么进来
- `output` 定义什么数据能流回父图
- 内部状态 `ArticleState` 可以比输入/输出更丰富

---

## RAG 开发中的应用

| 场景 | Map 阶段 | 子图处理 | Reduce 阶段 |
|------|----------|----------|-------------|
| 多文档检索 | 将 query 分发到多个知识库 | 每个库独立检索 | 合并并重排序结果 |
| 并行评估 | 将答案分发到多个评估器 | 独立评估（准确性、相关性、完整性） | 汇总评估分数 |
| 多模型投票 | 将 prompt 分发到多个 LLM | 每个 LLM 独立生成 | 投票选出最佳答案 |
| 多语言处理 | 将文档按语言分发 | 各语言独立处理 | 统一格式输出 |

---

## 常见陷阱

1. **Send 的 arg 必须匹配子图 input_schema**：如果子图定义了 `input=ArticleInput`，Send 传入的字典必须包含 `ArticleInput` 的所有键
2. **Reducer 是必需的**：如果没有 `Annotated[list, operator.add]`，多个子图返回同一个键时只有最后一个会保留
3. **子图 output_schema 的键必须与父图状态键一致**：子图返回 `{"articles": [...]}` 要求父图也有 `articles` 键
4. **Send 只能在条件边函数中返回**：不能在普通节点函数中使用 Send
