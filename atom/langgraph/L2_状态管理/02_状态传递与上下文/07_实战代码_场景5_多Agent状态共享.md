# 实战代码 - 场景5：多Agent状态共享

## 场景描述

在多 Agent 系统中，多个 Agent 需要协作完成复杂任务。每个 Agent 负责不同的职责，但需要共享某些状态信息以实现协调。本场景演示如何在 LangGraph 中设计和实现多 Agent 状态共享机制。

**实际应用场景**：
- 研究助手系统：搜索 Agent、分析 Agent、写作 Agent 协作完成研究报告
- 客服系统：路由 Agent、查询 Agent、回复 Agent 协作处理用户请求
- 数据处理管道：提取 Agent、转换 Agent、加载 Agent 协作处理数据

**核心挑战**：
1. 如何设计共享状态 Schema 避免冲突
2. 如何实现 Agent 间的数据传递
3. 如何避免状态竞争和覆盖
4. 如何保持状态的一致性

## 核心概念

### 1. 共享状态设计模式

**模式一：全局共享状态**
- 所有 Agent 读写同一个大状态对象
- 优点：简单直接
- 缺点：容易冲突，难以维护

**模式二：命名空间隔离**
- 每个 Agent 有自己的状态命名空间
- 优点：避免冲突，职责清晰
- 缺点：需要显式传递共享数据

**模式三：混合模式**（推荐）
- 共享字段 + Agent 专属字段
- 优点：兼顾协作和隔离
- 缺点：需要仔细设计 Schema

### 2. Reducer 的作用

在多 Agent 系统中，Reducer 决定了状态更新的合并策略：
- **覆盖模式**：后写入的值覆盖前值（默认）
- **追加模式**：使用 `operator.add` 或 `add_messages` 累积值
- **自定义模式**：实现自定义合并逻辑

## 完整代码示例

```python
"""
多 Agent 状态共享实战示例
演示：研究助手系统 - 搜索、分析、写作 Agent 协作
"""

import os
from typing import TypedDict, Annotated, Literal
from operator import add
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 加载环境变量
load_dotenv()

# ===== 1. 状态 Schema 设计 =====

class ResearchState(TypedDict):
    """
    研究助手系统的共享状态

    设计原则：
    - 共享字段：所有 Agent 都需要的信息（messages, topic, final_report）
    - Agent 专属字段：各 Agent 的输出结果（search_results, analysis, draft）
    - 使用 Annotated 指定 Reducer 策略
    """
    # 共享字段 - 对话历史（追加模式）
    messages: Annotated[list, add_messages]

    # 共享字段 - 研究主题（覆盖模式）
    topic: str

    # Agent 专属字段 - 搜索 Agent 的输出
    search_results: list[str]

    # Agent 专属字段 - 分析 Agent 的输出
    analysis: str

    # Agent 专属字段 - 写作 Agent 的输出
    draft: str

    # 共享字段 - 最终报告（覆盖模式）
    final_report: str

    # 元数据 - 当前步骤（用于调试）
    current_step: str


# ===== 2. Agent 节点定义 =====

def search_agent(state: ResearchState) -> dict:
    """
    搜索 Agent：根据主题搜索相关信息

    职责：
    - 读取 topic
    - 模拟搜索过程
    - 写入 search_results
    """
    print(f"\n[搜索 Agent] 开始搜索主题: {state['topic']}")

    # 模拟搜索结果（实际应用中调用搜索 API）
    search_results = [
        f"关于 {state['topic']} 的研究论文 A",
        f"关于 {state['topic']} 的技术博客 B",
        f"关于 {state['topic']} 的开源项目 C",
    ]

    print(f"[搜索 Agent] 找到 {len(search_results)} 条结果")

    # 返回部分状态更新
    return {
        "search_results": search_results,
        "current_step": "search_completed",
        "messages": [AIMessage(content=f"搜索完成，找到 {len(search_results)} 条相关信息")]
    }


def analysis_agent(state: ResearchState) -> dict:
    """
    分析 Agent：分析搜索结果

    职责：
    - 读取 search_results
    - 进行分析
    - 写入 analysis
    """
    print(f"\n[分析 Agent] 开始分析 {len(state['search_results'])} 条搜索结果")

    # 模拟分析过程（实际应用中调用 LLM）
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    analysis_prompt = f"""
    请分析以下关于 "{state['topic']}" 的搜索结果：

    {chr(10).join(f"{i+1}. {result}" for i, result in enumerate(state['search_results']))}

    请提供：
    1. 主要发现（3-5 点）
    2. 关键趋势
    3. 潜在问题

    保持简洁，每点不超过一句话。
    """

    response = llm.invoke([SystemMessage(content=analysis_prompt)])
    analysis = response.content

    print(f"[分析 Agent] 分析完成，生成 {len(analysis)} 字符的分析报告")

    return {
        "analysis": analysis,
        "current_step": "analysis_completed",
        "messages": [AIMessage(content="分析完成，已生成分析报告")]
    }


def writing_agent(state: ResearchState) -> dict:
    """
    写作 Agent：基于分析结果撰写报告

    职责：
    - 读取 topic, search_results, analysis
    - 撰写报告
    - 写入 final_report
    """
    print(f"\n[写作 Agent] 开始撰写关于 '{state['topic']}' 的报告")

    # 模拟写作过程（实际应用中调用 LLM）
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    writing_prompt = f"""
    请基于以下信息撰写一份简短的研究报告：

    主题：{state['topic']}

    搜索结果：
    {chr(10).join(f"- {result}" for result in state['search_results'])}

    分析报告：
    {state['analysis']}

    要求：
    1. 结构清晰（引言、主体、结论）
    2. 长度控制在 200-300 字
    3. 突出关键发现
    """

    response = llm.invoke([SystemMessage(content=writing_prompt)])
    final_report = response.content

    print(f"[写作 Agent] 报告撰写完成，共 {len(final_report)} 字符")

    return {
        "final_report": final_report,
        "current_step": "writing_completed",
        "messages": [AIMessage(content="报告撰写完成")]
    }


# ===== 3. 构建多 Agent 工作流 =====

def build_research_workflow() -> StateGraph:
    """构建研究助手工作流"""

    # 创建状态图
    workflow = StateGraph(ResearchState)

    # 添加 Agent 节点
    workflow.add_node("search", search_agent)
    workflow.add_node("analysis", analysis_agent)
    workflow.add_node("writing", writing_agent)

    # 定义执行顺序（线性流程）
    workflow.add_edge(START, "search")
    workflow.add_edge("search", "analysis")
    workflow.add_edge("analysis", "writing")
    workflow.add_edge("writing", END)

    return workflow.compile()


# ===== 4. 运行示例 =====

def main():
    """主函数：演示多 Agent 状态共享"""

    print("=" * 60)
    print("多 Agent 状态共享实战示例")
    print("场景：研究助手系统 - 搜索、分析、写作 Agent 协作")
    print("=" * 60)

    # 构建工作流
    app = build_research_workflow()

    # 初始输入
    initial_state = {
        "topic": "LangGraph 状态管理最佳实践",
        "messages": [HumanMessage(content="请研究 LangGraph 状态管理的最佳实践")],
        "search_results": [],
        "analysis": "",
        "draft": "",
        "final_report": "",
        "current_step": "initialized"
    }

    print(f"\n初始状态：")
    print(f"  主题: {initial_state['topic']}")
    print(f"  当前步骤: {initial_state['current_step']}")

    # 执行工作流
    print("\n开始执行工作流...")
    final_state = app.invoke(initial_state)

    # 输出最终结果
    print("\n" + "=" * 60)
    print("执行完成！最终状态：")
    print("=" * 60)

    print(f"\n主题: {final_state['topic']}")
    print(f"当前步骤: {final_state['current_step']}")

    print(f"\n搜索结果 ({len(final_state['search_results'])} 条):")
    for i, result in enumerate(final_state['search_results'], 1):
        print(f"  {i}. {result}")

    print(f"\n分析报告:")
    print(f"  {final_state['analysis'][:200]}...")

    print(f"\n最终报告:")
    print(f"  {final_state['final_report'][:300]}...")

    print(f"\n消息历史 ({len(final_state['messages'])} 条):")
    for msg in final_state['messages']:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        print(f"  [{role}] {msg.content}")


# ===== 5. 高级示例：并行 Agent 协作 =====

def build_parallel_research_workflow() -> StateGraph:
    """
    构建并行研究工作流

    场景：搜索 Agent 完成后，分析和总结 Agent 并行执行
    """

    def summarize_agent(state: ResearchState) -> dict:
        """总结 Agent：生成搜索结果摘要"""
        print(f"\n[总结 Agent] 开始总结搜索结果")

        summary = f"共找到 {len(state['search_results'])} 条结果，涵盖论文、博客和开源项目"

        return {
            "messages": [AIMessage(content=f"总结完成：{summary}")]
        }

    def combine_results(state: ResearchState) -> dict:
        """合并 Agent：合并分析和总结结果"""
        print(f"\n[合并 Agent] 合并分析和总结结果")

        # 读取分析和总结结果
        analysis = state.get('analysis', '')

        # 合并结果
        combined = f"分析报告：\n{analysis}\n\n"

        return {
            "final_report": combined,
            "current_step": "combined"
        }

    workflow = StateGraph(ResearchState)

    # 添加节点
    workflow.add_node("search", search_agent)
    workflow.add_node("analysis", analysis_agent)
    workflow.add_node("summarize", summarize_agent)
    workflow.add_node("combine", combine_results)

    # 定义执行顺序
    workflow.add_edge(START, "search")

    # 并行执行分析和总结
    workflow.add_edge("search", "analysis")
    workflow.add_edge("search", "summarize")

    # 等待两个并行任务完成后合并
    workflow.add_edge("analysis", "combine")
    workflow.add_edge("summarize", "combine")

    workflow.add_edge("combine", END)

    return workflow.compile()


if __name__ == "__main__":
    main()

    print("\n\n" + "=" * 60)
    print("高级示例：并行 Agent 协作")
    print("=" * 60)

    # 运行并行工作流
    parallel_app = build_parallel_research_workflow()

    initial_state = {
        "topic": "LangGraph 并行执行模式",
        "messages": [HumanMessage(content="请研究 LangGraph 的并行执行模式")],
        "search_results": [],
        "analysis": "",
        "draft": "",
        "final_report": "",
        "current_step": "initialized"
    }

    print(f"\n执行并行工作流...")
    final_state = parallel_app.invoke(initial_state)

    print(f"\n最终报告:")
    print(final_state['final_report'])
```

## 运行输出示例

```
============================================================
多 Agent 状态共享实战示例
场景：研究助手系统 - 搜索、分析、写作 Agent 协作
============================================================

初始状态：
  主题: LangGraph 状态管理最佳实践
  当前步骤: initialized

开始执行工作流...

[搜索 Agent] 开始搜索主题: LangGraph 状态管理最佳实践
[搜索 Agent] 找到 3 条结果

[分析 Agent] 开始分析 3 条搜索结果
[分析 Agent] 分析完成，生成 456 字符的分析报告

[写作 Agent] 开始撰写关于 'LangGraph 状态管理最佳实践' 的报告
[写作 Agent] 报告撰写完成，共 287 字符

============================================================
执行完成！最终状态：
============================================================

主题: LangGraph 状态管理最佳实践
当前步骤: writing_completed

搜索结果 (3 条):
  1. 关于 LangGraph 状态管理最佳实践 的研究论文 A
  2. 关于 LangGraph 状态管理最佳实践 的技术博客 B
  3. 关于 LangGraph 状态管理最佳实践 的开源项目 C

分析报告:
  主要发现：
1. TypedDict 是定义状态 Schema 的推荐方式
2. Reducer 函数控制状态更新策略
3. 命名空间隔离避免 Agent 间冲突
...

最终报告:
  # LangGraph 状态管理最佳实践研究报告

## 引言
LangGraph 作为构建多 Agent 系统的框架，其状态管理机制是确保 Agent 协作的核心。本报告基于最新研究和实践，总结了状态管理的关键要点。

## 主体
根据分析，LangGraph 状态管理的最佳实践包括...

消息历史 (4 条):
  [用户] 请研究 LangGraph 状态管理的最佳实践
  [助手] 搜索完成，找到 3 条相关信息
  [助手] 分析完成，已生成分析报告
  [助手] 报告撰写完成
```

## 核心要点总结

### 1. 状态 Schema 设计原则

**DO（推荐）**：
- ✅ 使用 TypedDict 定义清晰的状态结构
- ✅ 区分共享字段和 Agent 专属字段
- ✅ 使用 Annotated 指定 Reducer 策略
- ✅ 为调试添加元数据字段（如 current_step）

**DON'T（避免）**：
- ❌ 所有字段都用追加模式（会导致状态膨胀）
- ❌ Agent 间随意覆盖彼此的字段
- ❌ 缺少类型注解（难以维护）

### 2. Agent 协作模式

**线性协作**：
```
搜索 → 分析 → 写作
```
- 适用场景：步骤有明确依赖关系
- 优点：逻辑清晰，易于调试
- 缺点：执行时间较长

**并行协作**：
```
        ┌→ 分析 ┐
搜索 →  │       ├→ 合并
        └→ 总结 ┘
```
- 适用场景：多个 Agent 可独立执行
- 优点：提高执行效率
- 缺点：需要合并逻辑

### 3. 状态冲突避免策略

**策略一：命名空间隔离**
```python
class State(TypedDict):
    agent_a_data: dict  # Agent A 专属
    agent_b_data: dict  # Agent B 专属
    shared_data: dict   # 共享数据
```

**策略二：只读共享**
```python
# Agent 只读取共享字段，不修改
def agent(state: State) -> dict:
    topic = state['topic']  # 只读
    return {"agent_result": process(topic)}  # 只写自己的字段
```

**策略三：显式传递**
```python
# 通过 messages 传递数据
return {
    "messages": [AIMessage(content=json.dumps(data))]
}
```

## 引用来源

本文档基于以下资料编写：

1. **Reddit 讨论：多 Agent 状态管理**
   - 来源：https://www.reddit.com/r/LangGraph/comments/1n867pe/managing_shared_state_in_langgraph_multiagent
   - 文件：`reference/fetch_状态传递_07.md`
   - 核心内容：共享状态设计、并发冲突避免、namespaced state

2. **Reddit 讨论：上下文管理**
   - 来源：https://www.reddit.com/r/LangChain/comments/1kz912z/context_management_using_state
   - 文件：`reference/fetch_状态传递_05.md`
   - 核心内容：共享内存、状态读写、持久化

3. **LangChain 官方博客：上下文工程**
   - 来源：https://blog.langchain.com/context-engineering-for-agents
   - 文件：`reference/fetch_状态传递_02.md`
   - 核心内容：Write、Select、Compress、Isolate 策略

4. **Context7 官方文档**
   - 来源：LangGraph 官方文档
   - 文件：`reference/context7_langgraph_01.md`
   - 核心内容：多状态 Schema、Runtime Context、节点参数类型

5. **GitHub 实践示例**
   - 来源：https://github.com/FareedKhan-dev/contextual-engineering-guide
   - 文件：`reference/fetch_状态传递_08.md`
   - 核心内容：StateGraph、Scratchpad、代码示例

## 扩展阅读

- LangGraph 官方文档：Multi-Agent Systems
- Anthropic 博客：Building Multi-Agent Research System
- LangChain 博客：Context Engineering for Agents

---

**文档版本**：v1.0
**生成时间**：2026-02-26
**知识点**：L2_状态管理 > 02_状态传递与上下文 > 场景5_多Agent状态共享
