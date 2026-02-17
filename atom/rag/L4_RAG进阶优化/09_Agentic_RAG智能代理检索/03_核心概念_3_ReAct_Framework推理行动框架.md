# 核心概念 3: ReAct Framework 推理行动框架

## 一句话定义

**ReAct Framework 是"推理(Reasoning) + 行动(Acting)"的循环框架,让 AI 代理在思考和执行之间交替迭代,在 Agentic RAG 中实现自主决策和动态检索。**

---

## 详细解释

### 什么是 ReAct Framework?

ReAct 是 2022 年由 Yao 等人提出的代理框架,核心思想:
- **Reasoning**: AI 先思考下一步该做什么
- **Acting**: 基于思考执行具体行动
- **Observing**: 观察行动结果
- **Reflecting**: 反思结果并决定下一步

**核心价值**: 让 AI 像人类专家一样"边想边做",而非一次性生成答案。

### 为什么需要 ReAct Framework?

传统 RAG 的问题:
```python
# 传统 RAG: 一次检索,一次生成
query = "Transformer 的注意力机制如何工作?"
docs = retriever.search(query)  # 一次检索
answer = llm.generate(docs)     # 一次生成
# 问题: 如果检索结果不够,无法补充
```

**复杂查询需要迭代**:
```python
# ReAct: 思考 → 行动 → 观察 → 反思 → 循环
query = "Transformer 的注意力机制如何工作?"

# Thought 1: 先检索基础概念
action_1 = "搜索 Transformer 基础"
observation_1 = retriever.search(action_1)

# Thought 2: 结果不够详细,需要更具体的信息
action_2 = "搜索 Self-Attention 计算公式"
observation_2 = retriever.search(action_2)

# Thought 3: 现在信息足够了,可以生成答案
answer = llm.generate(observation_1 + observation_2)
```

### ReAct Framework 如何工作?

**核心循环**:
```
用户查询
    ↓
[Thought] 思考: 我需要什么信息?
    ↓
[Action] 行动: 执行检索/工具调用
    ↓
[Observation] 观察: 获得结果
    ↓
[Reflection] 反思: 结果是否足够?
    ↓
    ├─ 是 → 生成最终答案
    └─ 否 → 回到 [Thought]
```

---

## 核心原理

### 原理图解

```
┌─────────────────────────────────────────┐
│         ReAct 循环示例                  │
├─────────────────────────────────────────┤
│                                         │
│  查询: "比较 BERT 和 GPT 的优缺点"      │
│                                         │
│  [Thought 1]                            │
│  我需要先了解 BERT 的特点               │
│       ↓                                 │
│  [Action 1]                             │
│  search("BERT 技术特点")                │
│       ↓                                 │
│  [Observation 1]                        │
│  BERT 是双向编码器,擅长理解任务...      │
│       ↓                                 │
│  [Thought 2]                            │
│  现在需要了解 GPT 的特点                │
│       ↓                                 │
│  [Action 2]                             │
│  search("GPT 技术特点")                 │
│       ↓                                 │
│  [Observation 2]                        │
│  GPT 是单向解码器,擅长生成任务...       │
│       ↓                                 │
│  [Thought 3]                            │
│  信息足够了,可以对比分析                │
│       ↓                                 │
│  [Action 3]                             │
│  generate_answer()                      │
│       ↓                                 │
│  最终答案: BERT vs GPT 对比分析         │
│                                         │
└─────────────────────────────────────────┘
```

### 工作流程

**Step 1: Thought (思考)**
```python
def think(query: str, context: List[str]) -> str:
    """思考下一步行动"""
    prompt = f"""
    查询: {query}
    已知信息: {context}

    思考: 下一步我应该做什么?
    """
    thought = llm.predict(prompt)
    return thought
```

**Step 2: Action (行动)**
```python
def act(thought: str) -> Dict:
    """基于思考执行行动"""
    # 解析思考,决定行动类型
    if "搜索" in thought:
        action = {"type": "search", "query": extract_query(thought)}
    elif "计算" in thought:
        action = {"type": "calculate", "expression": extract_expr(thought)}
    elif "生成答案" in thought:
        action = {"type": "finish", "answer": generate_answer()}

    return action
```

**Step 3: Observation (观察)**
```python
def observe(action: Dict) -> str:
    """执行行动并观察结果"""
    if action["type"] == "search":
        result = retriever.search(action["query"])
    elif action["type"] == "calculate":
        result = calculator.run(action["expression"])
    elif action["type"] == "finish":
        result = action["answer"]

    return result
```

**Step 4: Reflection (反思)**
```python
def reflect(observation: str, query: str) -> bool:
    """反思是否需要继续"""
    prompt = f"""
    原始查询: {query}
    当前结果: {observation}

    问题: 信息是否足够回答查询? (是/否)
    """
    decision = llm.predict(prompt)
    return "是" in decision
```

### 关键技术

**1. Prompt 设计 (2022 原版)**
```python
REACT_PROMPT = """
你是一个问答助手。使用以下格式回答问题:

Question: 用户的问题
Thought: 你应该思考下一步做什么
Action: 执行的行动 [search/calculate/finish]
Action Input: 行动的输入
Observation: 行动的结果
... (重复 Thought/Action/Observation)
Thought: 我现在知道最终答案了
Final Answer: 最终答案

Question: {query}
"""
```

**2. 工具集成 (2025 增强)**
```python
from langchain.agents import Tool

tools = [
    Tool(
        name="Search",
        func=vector_search,
        description="搜索相关文档,输入查询字符串"
    ),
    Tool(
        name="Calculator",
        func=calculator,
        description="执行数学计算,输入表达式"
    ),
    Tool(
        name="WebSearch",
        func=web_search,
        description="搜索实时信息,输入查询"
    )
]
```

**3. 自我纠错 (2026 最新)**
```python
def self_correction_react(query: str, max_retries: int = 3):
    """带自我纠错的 ReAct"""
    for attempt in range(max_retries):
        try:
            result = react_loop(query)

            # 验证结果
            if is_valid(result):
                return result

            # 自我纠错
            feedback = f"结果不正确,原因: {validate_error(result)}"
            query = f"{query}\n反馈: {feedback}"

        except Exception as e:
            continue

    return "无法生成满意答案"
```

---

## 手写实现

```python
"""
ReAct Framework 从零实现
演示: Thought → Action → Observation 循环
"""

from typing import List, Dict, Optional
from openai import OpenAI
import os
import re

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== 1. 工具定义 =====
class SearchTool:
    """搜索工具(模拟)"""
    def run(self, query: str) -> str:
        # 模拟搜索结果
        knowledge_base = {
            "bert": "BERT 是双向编码器,使用 Masked LM 预训练,擅长理解任务",
            "gpt": "GPT 是单向解码器,使用自回归预训练,擅长生成任务",
            "transformer": "Transformer 使用 Self-Attention 机制,并行处理序列"
        }

        for key, value in knowledge_base.items():
            if key in query.lower():
                return value

        return "未找到相关信息"

class CalculatorTool:
    """计算工具"""
    def run(self, expression: str) -> str:
        try:
            result = eval(expression)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {e}"

# ===== 2. ReAct Agent =====
class ReActAgent:
    """ReAct 代理"""

    def __init__(self):
        self.search_tool = SearchTool()
        self.calculator_tool = CalculatorTool()
        self.max_iterations = 5

    def run(self, query: str) -> str:
        """执行 ReAct 循环"""
        print(f"\n{'='*50}")
        print(f"查询: {query}")
        print(f"{'='*50}\n")

        context = []
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            print(f"--- 迭代 {iteration} ---\n")

            # Step 1: Thought
            thought = self._think(query, context)
            print(f"💭 Thought: {thought}\n")

            # 检查是否完成
            if "finish" in thought.lower() or "最终答案" in thought:
                final_answer = self._generate_final_answer(query, context)
                print(f"✅ Final Answer: {final_answer}\n")
                return final_answer

            # Step 2: Action
            action = self._parse_action(thought)
            print(f"⚡ Action: {action['type']}({action['input']})\n")

            # Step 3: Observation
            observation = self._execute_action(action)
            print(f"👁️  Observation: {observation}\n")

            # 保存上下文
            context.append({
                "thought": thought,
                "action": action,
                "observation": observation
            })

        return "达到最大迭代次数,无法生成答案"

    def _think(self, query: str, context: List[Dict]) -> str:
        """思考下一步"""
        context_str = "\n".join([
            f"Thought: {c['thought']}\nAction: {c['action']}\nObservation: {c['observation']}"
            for c in context
        ])

        prompt = f"""
你是一个问答助手。使用以下格式:

Thought: 思考下一步做什么
Action: search(查询) 或 calculate(表达式) 或 finish

已有上下文:
{context_str}

原始查询: {query}

下一步 Thought:
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return response.choices[0].message.content.strip()

    def _parse_action(self, thought: str) -> Dict:
        """解析行动"""
        # 提取 search(...)
        search_match = re.search(r'search\((.*?)\)', thought, re.IGNORECASE)
        if search_match:
            return {"type": "search", "input": search_match.group(1).strip('"\'').strip()}

        # 提取 calculate(...)
        calc_match = re.search(r'calculate\((.*?)\)', thought, re.IGNORECASE)
        if calc_match:
            return {"type": "calculate", "input": calc_match.group(1).strip()}

        # 默认搜索
        return {"type": "search", "input": thought}

    def _execute_action(self, action: Dict) -> str:
        """执行行动"""
        if action["type"] == "search":
            return self.search_tool.run(action["input"])
        elif action["type"] == "calculate":
            return self.calculator_tool.run(action["input"])
        else:
            return "未知行动类型"

    def _generate_final_answer(self, query: str, context: List[Dict]) -> str:
        """生成最终答案"""
        observations = "\n".join([c["observation"] for c in context])

        prompt = f"""
基于以下信息回答问题:

问题: {query}

信息:
{observations}

答案:
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

# ===== 3. 测试 =====
if __name__ == "__main__":
    agent = ReActAgent()

    test_queries = [
        "什么是 BERT?",
        "比较 BERT 和 GPT",
        "Transformer 的核心机制是什么?"
    ]

    for query in test_queries:
        answer = agent.run(query)
        print(f"\n{'='*50}\n")
```

---

## 在 RAG 中的应用

### 应用场景 1: 迭代检索

**问题**: 初次检索结果不够完整

**ReAct 方案**:
```python
def iterative_rag(query: str):
    """迭代式 RAG"""
    context = []

    # Thought 1: 先检索基础信息
    docs_1 = retriever.search(query)
    context.extend(docs_1)

    # Reflection: 信息是否足够?
    if not is_sufficient(docs_1, query):
        # Thought 2: 需要更具体的信息
        refined_query = refine_query(query, docs_1)
        docs_2 = retriever.search(refined_query)
        context.extend(docs_2)

    # Generate answer
    return llm.generate(context)
```

### 应用场景 2: 多工具协作

**问题**: 需要结合检索和计算

**ReAct 方案**:
```python
def multi_tool_rag(query: str):
    """多工具 RAG"""
    # Thought: 先检索数据
    if "数据" in query:
        data = retriever.search(query)

    # Thought: 需要计算
    if "计算" in query or "增长率" in query:
        result = calculator.run(extract_formula(data))

    # Thought: 生成答案
    return llm.generate(f"数据: {data}, 计算: {result}")
```

### 应用场景 3: 自我验证

**问题**: 生成的答案可能不准确

**ReAct 方案**:
```python
def self_verify_rag(query: str):
    """自我验证 RAG"""
    # Generate initial answer
    answer = rag_pipeline(query)

    # Verify answer
    verification = verify_answer(answer, query)

    if not verification["correct"]:
        # Re-search with feedback
        feedback = verification["reason"]
        refined_query = f"{query} (注意: {feedback})"
        answer = rag_pipeline(refined_query)

    return answer
```

---

## 主流框架实现

### LangChain 实现 (推荐)

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain import hub

# 定义工具
tools = [
    Tool(
        name="Search",
        func=vector_search,
        description="搜索相关文档"
    ),
    Tool(
        name="Calculator",
        func=calculator,
        description="执行数学计算"
    )
]

# 获取 ReAct Prompt
prompt = hub.pull("hwchase17/react")

# 创建 ReAct Agent
llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_react_agent(llm, tools, prompt)

# 执行
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

result = executor.invoke({"input": "比较 BERT 和 GPT"})
```

### LangGraph 实现

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

# 定义状态
class AgentState(TypedDict):
    input: str
    agent_outcome: Union[AgentAction, AgentFinish]
    intermediate_steps: List[Tuple[AgentAction, str]]

# 定义节点
def run_agent(state: AgentState):
    """运行代理(Thought + Action)"""
    agent_outcome = agent.invoke(state)
    return {"agent_outcome": agent_outcome}

def execute_tools(state: AgentState):
    """执行工具(Observation)"""
    agent_action = state["agent_outcome"]
    output = tool_executor.invoke(agent_action)
    return {"intermediate_steps": [(agent_action, output)]}

# 构建图
workflow = StateGraph(AgentState)
workflow.add_node("agent", run_agent)
workflow.add_node("tools", execute_tools)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    lambda x: "continue" if isinstance(x["agent_outcome"], AgentAction) else "end",
    {
        "continue": "tools",
        "end": END
    }
)
workflow.add_edge("tools", "agent")

app = workflow.compile()
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

# 创建 ReAct Agent
agent = ReActAgent.from_tools(
    [query_tool],
    llm=llm,
    verbose=True,
    max_iterations=5
)

# 执行
response = agent.chat("比较 BERT 和 GPT")
print(response)
```

---

## 最佳实践 (2025-2026)

### 性能优化

**1. 限制迭代次数**
```python
# 避免无限循环
agent = ReActAgent(max_iterations=5)
```

**2. 缓存工具结果**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query: str):
    return retriever.search(query)
```

**3. 并行工具调用**
```python
import asyncio

async def parallel_tools(actions: List[Dict]):
    """并行执行多个工具"""
    tasks = [execute_tool_async(action) for action in actions]
    return await asyncio.gather(*tasks)
```

### 成本控制

**1. 使用小模型思考**
```python
# Thought 用小模型
thought_llm = ChatOpenAI(model="gpt-4o-mini")

# Final Answer 用大模型
answer_llm = ChatOpenAI(model="gpt-4o")
```

**2. 早停策略**
```python
def early_stop_react(query: str):
    """早停策略"""
    for i in range(max_iterations):
        thought = think(query, context)

        # 如果置信度高,提前结束
        if confidence(thought) > 0.9:
            return generate_answer(context)

        # 继续迭代
        action = act(thought)
        observation = observe(action)
        context.append(observation)
```

### 错误处理

**1. 工具调用失败**
```python
def safe_tool_call(tool: Tool, input: str):
    """安全的工具调用"""
    try:
        return tool.run(input)
    except Exception as e:
        return f"工具调用失败: {e}, 请尝试其他方法"
```

**2. 无限循环检测**
```python
def detect_loop(context: List[Dict]):
    """检测重复行动"""
    recent_actions = [c["action"] for c in context[-3:]]

    if len(recent_actions) == 3 and len(set(recent_actions)) == 1:
        raise Exception("检测到重复行动,停止循环")
```

---

## 常见问题

### 问题 1: ReAct 太慢怎么办?

**原因**: 每次迭代都调用 LLM

**解决方案**:
```python
# 1. 减少迭代次数
agent = ReActAgent(max_iterations=3)

# 2. 使用规则优先
def fast_react(query: str):
    # 简单查询直接回答
    if is_simple(query):
        return direct_answer(query)

    # 复杂查询用 ReAct
    return react_agent(query)
```

### 问题 2: 如何提高 ReAct 准确率?

**解决方案**:
```python
# 1. 改进 Prompt
BETTER_PROMPT = """
你是专家助手。严格按照以下格式:

Thought: 详细思考下一步(必须具体)
Action: 明确的行动[search/calculate/finish]
Action Input: 精确的输入

示例:
Thought: 我需要了解 BERT 的预训练方法
Action: search
Action Input: BERT 预训练 Masked LM
"""

# 2. 添加示例
FEW_SHOT_EXAMPLES = [
    {
        "query": "什么是 Transformer?",
        "thought": "需要搜索 Transformer 基础概念",
        "action": "search(Transformer 架构)",
        "observation": "Transformer 使用 Self-Attention..."
    }
]
```

### 问题 3: ReAct vs Planning Agent 如何选择?

**对比**:
```python
# ReAct: 边想边做,灵活但可能低效
react_agent(query)  # 动态决策,适合不确定任务

# Planning: 先规划再执行,高效但不灵活
planning_agent(query)  # 预先规划,适合明确任务

# 选择建议:
# - 探索性任务 → ReAct
# - 明确任务 → Planning
# - 复杂任务 → ReAct + Planning 混合
```

---

## 参考资源

### 论文
- "ReAct: Synergizing Reasoning and Acting in Language Models" (arXiv 2210.03629, 2022)
- "Reflexion: Language Agents with Verbal Reinforcement Learning" (arXiv 2303.11366, 2023)

### 博客
- IBM: "What is Agentic RAG?" (2026) - ReAct 在 RAG 中的应用
  https://www.ibm.com/think/topics/agentic-rag
- "ReAct Framework Explained" (Medium, 2026)
  https://medium.com/@linz07m/react-reasoning-and-acting-framework-03e71aff1877
- LangChain: "ReAct Agent" (2026)
  https://python.langchain.com/docs/modules/agents/agent_types/react

### 框架文档
- LangChain ReAct: https://python.langchain.com/docs/modules/agents/
- LangGraph ReAct: https://langchain-ai.github.io/langgraph/
- LlamaIndex ReAct: https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/

---

**版本**: v1.0
**最后更新**: 2026-02-17
**字数**: ~450 行
