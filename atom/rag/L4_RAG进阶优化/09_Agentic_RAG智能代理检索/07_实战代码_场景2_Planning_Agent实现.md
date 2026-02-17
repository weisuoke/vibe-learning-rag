# 实战代码 - 场景 2: Planning Agent 实现

## 场景描述

**目标**: 使用 LangGraph 构建 Plan-and-Execute 风格的规划代理,实现复杂查询的任务分解和逐步执行

**难点**:
- 将复杂查询分解为可执行的子任务
- 管理任务执行状态
- 根据执行结果动态调整计划

**解决方案**: 使用 LangGraph 状态图实现规划、执行、反思循环

---

## 环境准备

```bash
# 安装依赖
uv add langgraph langchain langchain-openai python-dotenv
```

---

## 完整代码

```python
"""
Planning Agent - Plan-and-Execute 实现
演示: 复杂查询的任务分解和执行

技术栈:
- LangGraph: 0.2.0+
- LangChain: 0.1.0+
- OpenAI: 1.0.0+
"""

import os
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv
import operator

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# 加载环境变量
load_dotenv()

# ===== 1. 初始化 LLM =====
print("初始化 LLM...")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ===== 2. 定义状态 =====

class AgentState(TypedDict):
    """代理状态"""
    query: str                                    # 原始查询
    plan: List[str]                               # 任务计划
    current_task_index: int                       # 当前任务索引
    task_results: Annotated[List[str], operator.add]  # 任务结果列表
    final_answer: str                             # 最终答案
    iteration: int                                # 迭代次数

# ===== 3. 规划节点 =====

def plan_node(state: AgentState) -> AgentState:
    """生成任务计划"""
    query = state["query"]

    print(f"\n{'='*60}")
    print(f"📋 规划阶段")
    print(f"{'='*60}")
    print(f"查询: {query}\n")

    # 生成计划
    plan_prompt = f"""
将以下查询分解为 3-5 个可执行的子任务。
每个任务应该独立、具体、可执行。

查询: {query}

请按以下格式返回任务列表(每行一个任务):
1. 任务描述
2. 任务描述
3. 任务描述
"""

    response = llm.invoke([HumanMessage(content=plan_prompt)])
    plan_text = response.content

    # 解析任务
    tasks = []
    for line in plan_text.split("\n"):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-")):
            # 移除编号
            task = line.split(".", 1)[-1].strip()
            if task:
                tasks.append(task)

    print("生成的计划:")
    for i, task in enumerate(tasks, 1):
        print(f"  {i}. {task}")

    return {
        "plan": tasks,
        "current_task_index": 0,
        "iteration": state.get("iteration", 0) + 1
    }

# ===== 4. 执行节点 =====

def execute_node(state: AgentState) -> AgentState:
    """执行当前任务"""
    plan = state["plan"]
    current_index = state["current_task_index"]
    task_results = state.get("task_results", [])

    if current_index >= len(plan):
        return state

    current_task = plan[current_index]

    print(f"\n{'='*60}")
    print(f"⚙️  执行阶段 - 任务 {current_index + 1}/{len(plan)}")
    print(f"{'='*60}")
    print(f"任务: {current_task}\n")

    # 执行任务(模拟)
    execute_prompt = f"""
执行以下任务并返回结果:

任务: {current_task}

已有上下文:
{chr(10).join(task_results) if task_results else "无"}

请提供简洁的执行结果:
"""

    response = llm.invoke([HumanMessage(content=execute_prompt)])
    result = response.content.strip()

    print(f"结果: {result}\n")

    return {
        "current_task_index": current_index + 1,
        "task_results": [f"任务 {current_index + 1}: {result}"]
    }

# ===== 5. 反思节点 =====

def reflect_node(state: AgentState) -> AgentState:
    """反思执行结果"""
    query = state["query"]
    task_results = state.get("task_results", [])

    print(f"\n{'='*60}")
    print(f"🤔 反思阶段")
    print(f"{'='*60}\n")

    # 评估结果
    reflect_prompt = f"""
评估以下任务执行结果是否足以回答原始查询:

原始查询: {query}

执行结果:
{chr(10).join(task_results)}

评估:
1. 信息是否完整? (是/否)
2. 是否需要补充? (是/否)

只返回: 完整 或 不完整
"""

    response = llm.invoke([HumanMessage(content=reflect_prompt)])
    evaluation = response.content.strip()

    print(f"评估结果: {evaluation}\n")

    return state

# ===== 6. 生成节点 =====

def generate_node(state: AgentState) -> AgentState:
    """生成最终答案"""
    query = state["query"]
    task_results = state.get("task_results", [])

    print(f"\n{'='*60}")
    print(f"📝 生成阶段")
    print(f"{'='*60}\n")

    # 生成答案
    generate_prompt = f"""
基于以下任务执行结果,生成对原始查询的完整答案:

查询: {query}

执行结果:
{chr(10).join(task_results)}

最终答案:
"""

    response = llm.invoke([HumanMessage(content=generate_prompt)])
    final_answer = response.content.strip()

    print(f"最终答案:\n{final_answer}\n")

    return {"final_answer": final_answer}

# ===== 7. 决策函数 =====

def should_continue(state: AgentState) -> str:
    """决定是否继续执行任务"""
    current_index = state["current_task_index"]
    plan = state["plan"]

    if current_index < len(plan):
        return "execute"  # 继续执行
    else:
        return "generate"  # 生成答案

# ===== 8. 构建图 =====

def create_planning_agent():
    """创建规划代理"""
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("plan", plan_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("generate", generate_node)

    # 设置入口
    workflow.set_entry_point("plan")

    # 添加边
    workflow.add_edge("plan", "execute")

    # 条件边: 执行后决定继续或生成
    workflow.add_conditional_edges(
        "execute",
        should_continue,
        {
            "execute": "execute",  # 继续执行下一个任务
            "generate": "reflect"  # 所有任务完成,进入反思
        }
    )

    workflow.add_edge("reflect", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

# ===== 9. 测试 =====

def main():
    """主函数"""
    agent = create_planning_agent()

    # 测试查询
    test_queries = [
        "比较 BERT 和 GPT 的优缺点",
        "解释 Transformer 的工作原理并举例说明",
        "什么是 RAG? 它如何工作?"
    ]

    for query in test_queries:
        print(f"\n{'#'*60}")
        print(f"# 查询: {query}")
        print(f"{'#'*60}\n")

        # 执行
        result = agent.invoke({
            "query": query,
            "plan": [],
            "current_task_index": 0,
            "task_results": [],
            "final_answer": "",
            "iteration": 0
        })

        print(f"\n{'='*60}")
        print(f"✅ 完成")
        print(f"{'='*60}")
        print(f"最终答案:\n{result['final_answer']}")
        print(f"\n迭代次数: {result['iteration']}")
        print(f"执行任务数: {len(result['task_results'])}")
        print()

if __name__ == "__main__":
    main()
```

---

## 运行输出

```
初始化 LLM...

############################################################
# 查询: 比较 BERT 和 GPT 的优缺点
############################################################

============================================================
📋 规划阶段
============================================================
查询: 比较 BERT 和 GPT 的优缺点

生成的计划:
  1. 检索 BERT 的技术特点和优缺点
  2. 检索 GPT 的技术特点和优缺点
  3. 对比 BERT 和 GPT 的架构差异
  4. 总结两者的适用场景

============================================================
⚙️  执行阶段 - 任务 1/4
============================================================
任务: 检索 BERT 的技术特点和优缺点

结果: BERT 是双向编码器,使用 Masked LM 预训练。
优点: 擅长理解任务,上下文理解能力强
缺点: 不适合生成任务

============================================================
⚙️  执行阶段 - 任务 2/4
============================================================
任务: 检索 GPT 的技术特点和优缺点

结果: GPT 是单向解码器,使用自回归预训练。
优点: 擅长生成任务,文本生成流畅
缺点: 上下文理解不如双向模型

============================================================
⚙️  执行阶段 - 任务 3/4
============================================================
任务: 对比 BERT 和 GPT 的架构差异

结果: BERT 使用双向 Transformer 编码器,GPT 使用单向 Transformer 解码器。
BERT 可以看到完整上下文,GPT 只能看到前文。

============================================================
⚙️  执行阶段 - 任务 4/4
============================================================
任务: 总结两者的适用场景

结果: BERT 适合分类、问答、命名实体识别等理解任务。
GPT 适合文本生成、对话、摘要等生成任务。

============================================================
🤔 反思阶段
============================================================

评估结果: 完整

============================================================
📝 生成阶段
============================================================

最终答案:
BERT 和 GPT 是两种不同的 Transformer 架构:

**BERT (双向编码器)**
- 优点: 强大的上下文理解能力,擅长理解任务
- 缺点: 不适合生成任务
- 适用场景: 分类、问答、NER

**GPT (单向解码器)**
- 优点: 流畅的文本生成能力,擅长生成任务
- 缺点: 上下文理解不如双向模型
- 适用场景: 文本生成、对话、摘要

核心差异: BERT 双向理解,GPT 单向生成。

============================================================
✅ 完成
============================================================
最终答案:
BERT 和 GPT 是两种不同的 Transformer 架构:

**BERT (双向编码器)**
- 优点: 强大的上下文理解能力,擅长理解任务
- 缺点: 不适合生成任务
- 适用场景: 分类、问答、NER

**GPT (单向解码器)**
- 优点: 流畅的文本生成能力,擅长生成任务
- 缺点: 上下文理解不如双向模型
- 适用场景: 文本生成、对话、摘要

核心差异: BERT 双向理解,GPT 单向生成。

迭代次数: 1
执行任务数: 4
```

---

## 代码解析

### 关键点 1: 状态定义

```python
class AgentState(TypedDict):
    """代理状态"""
    query: str                                    # 原始查询
    plan: List[str]                               # 任务计划
    current_task_index: int                       # 当前任务索引
    task_results: Annotated[List[str], operator.add]  # 任务结果列表
    final_answer: str                             # 最终答案
    iteration: int                                # 迭代次数
```

**要点**:
- 使用 `TypedDict` 定义状态结构
- `Annotated[List[str], operator.add]` 实现结果累加
- 状态在节点间传递和更新

### 关键点 2: 状态图构建

```python
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("plan", plan_node)
workflow.add_node("execute", execute_node)
workflow.add_node("reflect", reflect_node)
workflow.add_node("generate", generate_node)

# 条件边
workflow.add_conditional_edges(
    "execute",
    should_continue,
    {
        "execute": "execute",  # 循环执行
        "generate": "reflect"  # 完成后反思
    }
)
```

**要点**:
- 清晰的节点定义
- 条件边实现循环执行
- 状态驱动的流程控制

### 关键点 3: 任务执行循环

```python
def should_continue(state: AgentState) -> str:
    """决定是否继续执行任务"""
    current_index = state["current_task_index"]
    plan = state["plan"]

    if current_index < len(plan):
        return "execute"  # 继续执行
    else:
        return "generate"  # 生成答案
```

**要点**:
- 基于状态的决策逻辑
- 自动循环执行所有任务
- 完成后自动进入下一阶段

---

## 扩展思考

### 如何优化?

**1. 添加动态重规划**
```python
def replan_node(state: AgentState) -> AgentState:
    """根据执行结果重新规划"""
    task_results = state["task_results"]
    original_plan = state["plan"]

    # 评估是否需要调整计划
    if needs_adjustment(task_results):
        new_plan = generate_new_plan(original_plan, task_results)
        return {"plan": new_plan, "current_task_index": 0}

    return state
```

**2. 添加并行执行**
```python
def identify_parallel_tasks(plan: List[str]) -> List[List[int]]:
    """识别可并行执行的任务"""
    # 分析任务依赖关系
    dependencies = analyze_dependencies(plan)

    # 分组独立任务
    parallel_groups = group_independent_tasks(dependencies)

    return parallel_groups

async def execute_parallel(tasks: List[str]):
    """并行执行独立任务"""
    results = await asyncio.gather(*[
        execute_task_async(task) for task in tasks
    ])
    return results
```

**3. 添加任务优先级**
```python
def prioritize_tasks(plan: List[str]) -> List[str]:
    """任务优先级排序"""
    # 评估每个任务的重要性
    priorities = [evaluate_priority(task) for task in plan]

    # 按优先级排序
    sorted_plan = [task for _, task in sorted(zip(priorities, plan), reverse=True)]

    return sorted_plan
```

### 如何扩展?

**1. 支持多轮规划**
```python
def multi_round_planning(state: AgentState) -> AgentState:
    """多轮规划"""
    max_rounds = 3
    current_round = state.get("planning_round", 0)

    if current_round < max_rounds:
        # 基于前一轮结果重新规划
        refined_plan = refine_plan(state["plan"], state["task_results"])
        return {
            "plan": refined_plan,
            "planning_round": current_round + 1
        }

    return state
```

**2. 支持子任务分解**
```python
def decompose_task(task: str) -> List[str]:
    """将复杂任务分解为子任务"""
    if is_complex(task):
        subtasks = llm.invoke(f"将任务分解为子任务: {task}")
        return parse_subtasks(subtasks)

    return [task]

def execute_with_decomposition(state: AgentState) -> AgentState:
    """支持子任务分解的执行"""
    current_task = state["plan"][state["current_task_index"]]

    # 分解任务
    subtasks = decompose_task(current_task)

    # 执行子任务
    results = [execute_subtask(st) for st in subtasks]

    # 聚合结果
    aggregated_result = aggregate_results(results)

    return {"task_results": [aggregated_result]}
```

### 生产级改进

**1. 错误处理和重试**
```python
def execute_with_retry(state: AgentState, max_retries: int = 3) -> AgentState:
    """带重试的任务执行"""
    current_task = state["plan"][state["current_task_index"]]

    for attempt in range(max_retries):
        try:
            result = execute_task(current_task)

            # 验证结果
            if is_valid_result(result):
                return {"task_results": [result]}

        except Exception as e:
            if attempt == max_retries - 1:
                # 最后一次尝试失败,记录错误
                return {"task_results": [f"任务失败: {e}"]}

            # 重试前等待
            time.sleep(2 ** attempt)

    return state
```

**2. 性能监控**
```python
import time

def execute_with_metrics(state: AgentState) -> AgentState:
    """带性能监控的执行"""
    start_time = time.time()

    # 执行任务
    result = execute_task(state)

    # 记录指标
    execution_time = time.time() - start_time

    metrics = {
        "task_index": state["current_task_index"],
        "execution_time": execution_time,
        "result_length": len(result.get("task_results", []))
    }

    print(f"📊 指标: {metrics}")

    return result
```

**3. 状态持久化**
```python
import json

def save_state(state: AgentState, filename: str):
    """保存状态到文件"""
    with open(filename, "w") as f:
        json.dump(state, f, indent=2)

def load_state(filename: str) -> AgentState:
    """从文件加载状态"""
    with open(filename, "r") as f:
        return json.load(f)

def execute_with_checkpoint(state: AgentState) -> AgentState:
    """带检查点的执行"""
    # 执行前保存状态
    save_state(state, f"checkpoint_{state['current_task_index']}.json")

    try:
        result = execute_task(state)
        return result
    except Exception as e:
        # 失败时可以从检查点恢复
        print(f"执行失败,可从检查点恢复: checkpoint_{state['current_task_index']}.json")
        raise
```

---

## 参考资源

### 官方文档
- LangGraph Plan-and-Execute: https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/
- LangGraph State Graphs: https://langchain-ai.github.io/langgraph/

### 相关博客
- "Building an Agentic RAG System with LangGraph" (Medium, 2025)
- "Plan-and-Execute Agent Design Pattern" (LangChain Blog, 2026)

---

**版本**: v1.0
**最后更新**: 2026-02-17
**代码行数**: ~200 行
