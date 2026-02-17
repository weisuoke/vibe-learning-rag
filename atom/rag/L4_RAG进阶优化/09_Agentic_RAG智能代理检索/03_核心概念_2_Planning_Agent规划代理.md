# 核心概念 2: Planning Agent 规划代理

## 一句话定义

**Planning Agent 是任务分解专家,将复杂查询拆解为可执行的子任务序列,在 Agentic RAG 中实现多步骤推理和迭代检索。**

---

## 详细解释

### 什么是 Planning Agent?

Planning Agent 是 Agentic RAG 的"战略规划师",负责:
- **任务分解**: 将复杂查询拆分为子任务
- **执行规划**: 确定子任务的执行顺序
- **动态调整**: 根据执行结果重新规划

**核心价值**: 让 RAG 系统能够处理需要多步推理的复杂查询。

### 为什么需要 Planning Agent?

传统 RAG 的局限:
```python
# 传统 RAG: 一次检索,一次生成
query = "比较 2022 和 2023 年的营收增长率,并分析原因"
results = retriever.search(query)  # 一次检索
answer = llm.generate(results)     # 一次生成
# 问题: 无法分步处理复杂逻辑
```

**复杂查询需要多步骤**:
1. 检索 2022 年营收数据
2. 检索 2023 年营收数据
3. 计算增长率
4. 检索相关分析报告
5. 综合生成答案

### Planning Agent 如何工作?

**Plan-and-Execute 模式**:
```
复杂查询
    ↓
[规划阶段] → 生成任务列表
    ↓
[执行阶段] → 逐个执行任务
    ↓
[反思阶段] → 评估结果,重新规划
    ↓
最终答案
```

---

## 核心原理

### 原理图解

```
┌─────────────────────────────────────────┐
│       Planning Agent 工作流程           │
├─────────────────────────────────────────┤
│                                         │
│  查询: "比较 BERT 和 GPT 的优缺点"      │
│       ↓                                 │
│  [规划器 Planner]                       │
│   生成计划:                             │
│   1. 检索 BERT 的技术特点               │
│   2. 检索 GPT 的技术特点                │
│   3. 对比两者的优缺点                   │
│   4. 生成综合分析                       │
│       ↓                                 │
│  [执行器 Executor]                      │
│   执行任务 1 → 结果 1                   │
│   执行任务 2 → 结果 2                   │
│   执行任务 3 → 结果 3                   │
│   执行任务 4 → 最终答案                 │
│       ↓                                 │
│  [反思器 Reflector]                     │
│   评估: 答案是否完整?                   │
│   决策: 需要补充信息? → 重新规划        │
│                                         │
└─────────────────────────────────────────┘
```

### 工作流程

**Step 1: 规划 (Planning)**
```python
def plan(query: str) -> List[Task]:
    """生成任务计划"""
    prompt = f"""
    将以下查询分解为可执行的子任务:
    查询: {query}

    要求:
    - 每个任务独立可执行
    - 任务之间有逻辑顺序
    - 最后一个任务是综合答案

    任务列表:
    """
    tasks = llm.predict(prompt)
    return parse_tasks(tasks)
```

**Step 2: 执行 (Execution)**
```python
def execute(tasks: List[Task]) -> List[Result]:
    """执行任务列表"""
    results = []
    for task in tasks:
        result = execute_task(task)
        results.append(result)
    return results
```

**Step 3: 反思 (Reflection)**
```python
def reflect(results: List[Result]) -> Decision:
    """评估结果并决策"""
    prompt = f"""
    评估以下执行结果:
    {results}

    问题:
    1. 答案是否完整?
    2. 是否需要补充信息?
    3. 下一步行动?

    决策:
    """
    decision = llm.predict(prompt)
    return decision
```

### 关键技术

**1. Plan-and-Solve (2023)**
```python
# 受 Plan-and-Solve 论文启发
def plan_and_solve(query: str):
    # Step 1: 生成计划
    plan = generate_plan(query)

    # Step 2: 执行计划
    results = []
    for step in plan:
        result = execute_step(step)
        results.append(result)

    # Step 3: 综合答案
    answer = synthesize(results)
    return answer
```

**2. ReAct + Planning (2025)**
```python
# 结合 ReAct 的推理和规划
def react_planning(query: str):
    plan = generate_plan(query)

    for step in plan:
        # Thought: 思考
        thought = think(step)

        # Action: 行动
        action = decide_action(thought)
        observation = execute_action(action)

        # Reflection: 反思
        if not is_satisfactory(observation):
            plan = replan(plan, observation)

    return final_answer
```

**3. LangGraph 状态图 (2026)**
```python
from langgraph.graph import StateGraph

# 定义状态图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("plan", plan_node)
workflow.add_node("execute", execute_node)
workflow.add_node("reflect", reflect_node)

# 添加边
workflow.add_edge("plan", "execute")
workflow.add_conditional_edges(
    "reflect",
    should_continue,
    {
        "continue": "plan",  # 重新规划
        "end": END           # 结束
    }
)
```

---

## 手写实现

```python
"""
Planning Agent 从零实现
演示: Plan-and-Execute 模式
"""

from typing import List, Dict
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== 1. 数据结构 =====
class Task:
    def __init__(self, description: str, task_id: int):
        self.description = description
        self.task_id = task_id
        self.result = None

class Plan:
    def __init__(self, tasks: List[Task]):
        self.tasks = tasks
        self.current_index = 0

# ===== 2. 规划器 =====
def generate_plan(query: str) -> Plan:
    """生成任务计划"""
    prompt = f"""
    将查询分解为 3-5 个子任务,每行一个任务:

    查询: {query}

    子任务:
    1.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    # 解析任务
    content = response.choices[0].message.content
    task_lines = [line.strip() for line in content.split("\n") if line.strip()]

    tasks = []
    for i, line in enumerate(task_lines):
        # 移除编号
        description = line.split(".", 1)[-1].strip()
        tasks.append(Task(description, i + 1))

    return Plan(tasks)

# ===== 3. 执行器 =====
def execute_task(task: Task, context: List[str]) -> str:
    """执行单个任务"""
    prompt = f"""
    执行以下任务:
    任务: {task.description}

    上下文:
    {chr(10).join(context)}

    结果:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content

# ===== 4. 反思器 =====
def reflect_on_results(query: str, results: List[str]) -> Dict:
    """评估结果质量"""
    prompt = f"""
    原始查询: {query}

    执行结果:
    {chr(10).join(results)}

    评估:
    1. 答案是否完整? (是/否)
    2. 是否需要补充? (是/否)
    3. 建议:

    只返回 JSON: {{"complete": true/false, "need_more": true/false, "suggestion": "..."}}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    import json
    return json.loads(response.choices[0].message.content)

# ===== 5. Planning Agent =====
class PlanningAgent:
    """规划代理"""

    def __init__(self):
        self.max_iterations = 3

    def run(self, query: str) -> str:
        """执行 Plan-and-Execute"""
        print(f"\n{'='*50}")
        print(f"查询: {query}")
        print(f"{'='*50}\n")

        # Step 1: 生成计划
        print("📋 生成计划...")
        plan = generate_plan(query)

        for i, task in enumerate(plan.tasks, 1):
            print(f"  {i}. {task.description}")

        # Step 2: 执行计划
        print("\n⚙️  执行任务...")
        context = []

        for task in plan.tasks:
            print(f"\n  执行任务 {task.task_id}: {task.description}")
            result = execute_task(task, context)
            task.result = result
            context.append(f"任务 {task.task_id} 结果: {result}")
            print(f"  ✓ 完成")

        # Step 3: 反思
        print("\n🤔 反思结果...")
        results = [task.result for task in plan.tasks]
        reflection = reflect_on_results(query, results)

        print(f"  完整性: {'✓' if reflection['complete'] else '✗'}")
        print(f"  需要补充: {'是' if reflection['need_more'] else '否'}")

        # Step 4: 生成最终答案
        print("\n📝 生成最终答案...")
        final_answer = self._synthesize_answer(query, results)

        return final_answer

    def _synthesize_answer(self, query: str, results: List[str]) -> str:
        """综合生成最终答案"""
        prompt = f"""
        基于以下执行结果,回答原始查询:

        查询: {query}

        执行结果:
        {chr(10).join(results)}

        最终答案:
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content

# ===== 6. 测试 =====
if __name__ == "__main__":
    agent = PlanningAgent()

    test_queries = [
        "比较 BERT 和 GPT 的优缺点",
        "解释 Transformer 的工作原理并举例说明",
    ]

    for query in test_queries:
        answer = agent.run(query)
        print(f"\n{'='*50}")
        print(f"最终答案:\n{answer}")
        print(f"{'='*50}\n")
```

---

## 在 RAG 中的应用

### 应用场景 1: 多文档对比分析

**问题**: "比较 2022 和 2023 年的财报数据"

**Planning Agent 方案**:
```python
def compare_financial_reports(query: str):
    # 生成计划
    plan = [
        "检索 2022 年财报数据",
        "检索 2023 年财报数据",
        "提取关键指标",
        "计算增长率",
        "生成对比分析"
    ]

    # 执行计划
    results = {}
    results["2022"] = retrieve("2022年财报")
    results["2023"] = retrieve("2023年财报")
    results["metrics"] = extract_metrics(results)
    results["growth"] = calculate_growth(results["metrics"])

    # 生成答案
    return synthesize_comparison(results)
```

### 应用场景 2: 复杂技术问题

**问题**: "如何优化 RAG 系统的检索性能?"

**Planning Agent 方案**:
```python
def optimize_rag_performance(query: str):
    # 生成计划
    plan = [
        "检索当前性能瓶颈",
        "检索优化方案",
        "评估方案可行性",
        "生成实施建议"
    ]

    # 执行计划
    bottlenecks = retrieve("RAG 性能瓶颈")
    solutions = retrieve("RAG 优化方案")
    evaluation = evaluate_solutions(solutions, bottlenecks)
    recommendations = generate_recommendations(evaluation)

    return recommendations
```

### 应用场景 3: 研究助手

**问题**: "总结 2025 年 Agentic RAG 的研究进展"

**Planning Agent 方案**:
```python
def research_summary(query: str):
    # 生成计划
    plan = [
        "检索 2025 年相关论文",
        "提取核心创新点",
        "分类整理",
        "生成综述"
    ]

    # 执行计划
    papers = retrieve("2025 Agentic RAG 论文")
    innovations = extract_innovations(papers)
    categorized = categorize(innovations)
    summary = generate_summary(categorized)

    return summary
```

---

## 主流框架实现

### LangGraph 实现 (推荐)

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class AgentState(TypedDict):
    query: str
    plan: List[str]
    results: List[str]
    final_answer: str

def plan_node(state: AgentState):
    """规划节点"""
    query = state["query"]
    plan = generate_plan(query)
    return {"plan": plan.tasks}

def execute_node(state: AgentState):
    """执行节点"""
    plan = state["plan"]
    results = []
    for task in plan:
        result = execute_task(task, results)
        results.append(result)
    return {"results": results}

def reflect_node(state: AgentState):
    """反思节点"""
    reflection = reflect_on_results(state["query"], state["results"])
    return {"reflection": reflection}

def should_continue(state: AgentState):
    """决策函数"""
    if state.get("reflection", {}).get("complete"):
        return "synthesize"
    return "replan"

# 构建图
workflow = StateGraph(AgentState)
workflow.add_node("plan", plan_node)
workflow.add_node("execute", execute_node)
workflow.add_node("reflect", reflect_node)

workflow.set_entry_point("plan")
workflow.add_edge("plan", "execute")
workflow.add_edge("execute", "reflect")
workflow.add_conditional_edges(
    "reflect",
    should_continue,
    {
        "synthesize": END,
        "replan": "plan"
    }
)

app = workflow.compile()
```

### LangChain 实现

```python
from langchain.agents import AgentExecutor, create_plan_and_execute_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

# 定义工具
tools = [
    Tool(
        name="Search",
        func=vector_search,
        description="搜索相关文档"
    ),
    Tool(
        name="Calculate",
        func=calculator,
        description="执行计算"
    )
]

# 创建 Plan-and-Execute Agent
llm = ChatOpenAI(model="gpt-4o")
agent = create_plan_and_execute_agent(llm, tools)

# 执行
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = executor.run("比较 2022 和 2023 年营收")
```

### LlamaIndex 实现

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool

# 创建查询引擎工具
query_engine = index.as_query_engine()
query_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="search",
    description="搜索文档"
)

# 创建 ReAct Agent (带规划能力)
agent = ReActAgent.from_tools(
    [query_tool],
    llm=llm,
    verbose=True
)

# 执行
response = agent.chat("比较 BERT 和 GPT")
```

---

## 最佳实践 (2025-2026)

### 性能优化

**1. 并行执行独立任务**
```python
import asyncio

async def parallel_execute(tasks: List[Task]):
    """并行执行独立任务"""
    # 识别独立任务
    independent_tasks = identify_independent(tasks)

    # 并行执行
    results = await asyncio.gather(*[
        execute_task_async(task)
        for task in independent_tasks
    ])

    return results
```

**2. 缓存中间结果**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def execute_task_cached(task_description: str):
    """缓存任务执行结果"""
    return execute_task(task_description)
```

### 成本控制

**1. 限制规划深度**
```python
def generate_plan(query: str, max_tasks: int = 5):
    """限制任务数量"""
    prompt = f"分解为最多 {max_tasks} 个子任务: {query}"
    # ...
```

**2. 使用小模型规划**
```python
# 规划用小模型
planner_llm = ChatOpenAI(model="gpt-4o-mini")

# 执行用大模型
executor_llm = ChatOpenAI(model="gpt-4o")
```

### 错误处理

**1. 任务失败重试**
```python
def execute_with_retry(task: Task, max_retries: int = 3):
    """任务失败重试"""
    for attempt in range(max_retries):
        try:
            return execute_task(task)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"重试 {attempt + 1}/{max_retries}")
```

**2. 动态重新规划**
```python
def adaptive_planning(query: str):
    """自适应规划"""
    plan = generate_plan(query)

    for task in plan.tasks:
        result = execute_task(task)

        # 根据结果调整计划
        if not is_satisfactory(result):
            plan = replan(plan, result)

    return plan
```

---

## 常见问题

### 问题 1: 规划太复杂导致效率低?

**解决方案**:
```python
# 1. 限制规划深度
def simple_plan(query: str):
    """简化规划"""
    prompt = f"""
    将查询分解为 3 个核心步骤:
    {query}

    步骤:
    """
    return generate_plan(prompt)

# 2. 快速路径
def fast_path_check(query: str):
    """检查是否需要规划"""
    if is_simple_query(query):
        return direct_answer(query)
    else:
        return planning_agent(query)
```

### 问题 2: 如何评估规划质量?

**评估指标**:
```python
def evaluate_plan(plan: Plan, query: str):
    """评估规划质量"""
    metrics = {
        "completeness": check_completeness(plan, query),
        "efficiency": count_redundant_tasks(plan),
        "feasibility": check_feasibility(plan)
    }

    score = (
        metrics["completeness"] * 0.5 +
        (1 - metrics["efficiency"]) * 0.3 +
        metrics["feasibility"] * 0.2
    )

    return score, metrics
```

### 问题 3: 规划失败怎么办?

**回退策略**:
```python
def robust_planning(query: str):
    """鲁棒规划"""
    try:
        # 尝试规划
        plan = generate_plan(query)
        return execute_plan(plan)
    except Exception as e:
        # 回退到简单模式
        print(f"规划失败: {e}, 使用简单模式")
        return simple_rag(query)
```

---

## 参考资源

### 论文
- "Plan-and-Solve Prompting" (arXiv 2305.04091, 2023)
- "Agentic RAG: A Survey" (arXiv 2501.09136, 2025)
- "ReAct: Synergizing Reasoning and Acting" (arXiv 2210.03629, 2022)

### 博客
- LangGraph: "Plan-and-Execute Agent" (2026)
  https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/
- "Building an Agentic RAG System with LangGraph" (Medium, 2025)
- Vellum AI: "Agentic Workflows Guide" (2026)

### 框架文档
- LangGraph State Graphs: https://langchain-ai.github.io/langgraph/
- LangChain Plan-and-Execute: https://python.langchain.com/docs/modules/agents/
- LlamaIndex Workflow Agents: https://docs.llamaindex.ai/

---

**版本**: v1.0
**最后更新**: 2026-02-17
**字数**: ~450 行
