# Agent推理与决策 - 实战代码：思维链增强 Agent

> 用 Chain-of-Thought（CoT）技术让 Agent 的推理更深入、工具选择更准确

---

## 场景说明

**目标：** 通过思维链（CoT）技术增强 Agent 的推理质量，对比有无 CoT 的效果差异。

**你将学到：**
1. Zero-Shot CoT：一句话触发深度推理
2. Few-Shot CoT：用示例引导推理模式
3. 自定义 ReAct Prompt 加入 CoT 指令
4. 对比有无 CoT 的推理质量差异
5. CoT 在 RAG Agent 中的实际应用

**核心概念：**
- **Chain-of-Thought（CoT）**：让 LLM 展示推理过程，而不是直接给答案
- **Zero-Shot CoT**：只需添加 "Let's think step by step" 就能触发
- **Few-Shot CoT**：提供带推理过程的示例来引导 LLM

---

## 前置准备

```bash
# 安装依赖
uv add langchain langchain-openai langchain-core python-dotenv
```

---

## 完整代码

```python
"""
思维链增强 Agent 实战
演示：使用 Chain-of-Thought 技术提升 Agent 推理质量

运行方式：python examples/cot_agent_demo.py
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor

# ===== 0. 加载环境变量 =====
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ===== 1. 定义工具 =====

@tool
def search_product(query: str) -> str:
    """搜索产品信息数据库。
    输入：产品名称或关键词。
    返回：产品的价格、评分、库存等信息。"""

    products = {
        "macbook pro": "MacBook Pro 14寸：价格 14999 元，评分 4.8/5，库存 50 台",
        "iphone": "iPhone 16 Pro：价格 8999 元，评分 4.7/5，库存 200 台",
        "ipad": "iPad Air：价格 4799 元，评分 4.6/5，库存 150 台",
        "airpods": "AirPods Pro 2：价格 1899 元，评分 4.5/5，库存 300 台",
    }
    for key, value in products.items():
        if key in query.lower():
            return value
    return f"未找到 '{query}' 的产品信息"


@tool
def calculate_discount(expression: str) -> str:
    """计算折扣价格。
    输入：数学表达式，如 '14999 * 0.85' 表示 85 折。
    返回：计算结果。"""

    try:
        result = eval(expression, {"__builtins__": {}})
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{e}"


@tool
def check_user_budget(user_id: str) -> str:
    """查询用户预算信息。
    输入：用户ID。
    返回：用户的预算范围和偏好。"""

    users = {
        "user_001": "预算：10000-15000 元，偏好：高性能笔记本，用途：编程开发",
        "user_002": "预算：5000-8000 元，偏好：性价比手机，用途：日常使用",
        "user_003": "预算：3000-5000 元，偏好：平板电脑，用途：学习娱乐",
    }
    return users.get(user_id, f"未找到用户 {user_id} 的信息")


tools = [search_product, calculate_discount, check_user_budget]


# ===== 2. 普通 ReAct Prompt（无 CoT） =====
print("=" * 60)
print("对比实验：普通 ReAct vs CoT 增强 ReAct")
print("=" * 60)

normal_react_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

normal_prompt = PromptTemplate.from_template(normal_react_template)


# ===== 3. Zero-Shot CoT 增强 Prompt =====
# 核心改动：在 Thought 指令中加入 "think step by step"

zero_shot_cot_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: Let's think step by step. First, analyze what information is needed. Then, decide which tool to use and why. Consider if multiple steps are required.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: Let me summarize all the information I've gathered and provide a comprehensive answer.
Final Answer: the final answer to the original input question

Important: In each Thought step, explicitly reason about:
1. What do I know so far?
2. What do I still need to find out?
3. Which tool is best suited for the next step?
4. How does this step connect to the overall question?

Begin!

Question: {input}
Thought: Let's think step by step.{agent_scratchpad}"""

zero_shot_cot_prompt = PromptTemplate.from_template(zero_shot_cot_template)


# ===== 4. Few-Shot CoT 增强 Prompt =====
# 核心改动：提供带详细推理过程的示例

few_shot_cot_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think step by step about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Here is an example of good reasoning:

Question: 用户 user_002 想买手机，有什么推荐？如果打 9 折呢？
Thought: Let's think step by step. The user wants a phone recommendation for user_002. I need to: (1) check user_002's budget and preferences, (2) search for matching products, (3) calculate the discounted price. Let me start with the user's budget.
Action: check_user_budget
Action Input: user_002
Observation: 预算：5000-8000 元，偏好：性价比手机，用途：日常使用
Thought: Now I know user_002's budget is 5000-8000 yuan and they prefer cost-effective phones. Let me search for iPhone since it's a popular choice in this range.
Action: search_product
Action Input: iphone
Observation: iPhone 16 Pro：价格 8999 元，评分 4.7/5，库存 200 台
Thought: iPhone 16 Pro costs 8999 yuan, which is slightly above the budget. Let me calculate the 90% discount price to see if it fits.
Action: calculate_discount
Action Input: 8999 * 0.9
Observation: 计算结果：8099.1
Thought: With a 10% discount, the iPhone 16 Pro would be 8099.1 yuan, which is within the 5000-8000 budget range. I now have all the information to give a comprehensive recommendation.
Final Answer: 根据 user_002 的预算（5000-8000元）和偏好（性价比手机），推荐 iPhone 16 Pro（原价 8999 元，评分 4.7/5）。打 9 折后价格为 8099.1 元，刚好在预算范围内。库存充足（200台）。

Now answer the following question with the same level of detailed reasoning:

Question: {input}
Thought: Let's think step by step.{agent_scratchpad}"""

few_shot_cot_prompt = PromptTemplate.from_template(few_shot_cot_template)


# ===== 5. 创建三个 Agent 进行对比 =====

# Agent 1：普通 ReAct
normal_agent = create_react_agent(llm=llm, tools=tools, prompt=normal_prompt)
normal_executor = AgentExecutor(
    agent=normal_agent, tools=tools,
    verbose=True, max_iterations=6, handle_parsing_errors=True,
    return_intermediate_steps=True,
)

# Agent 2：Zero-Shot CoT
zs_cot_agent = create_react_agent(llm=llm, tools=tools, prompt=zero_shot_cot_prompt)
zs_cot_executor = AgentExecutor(
    agent=zs_cot_agent, tools=tools,
    verbose=True, max_iterations=6, handle_parsing_errors=True,
    return_intermediate_steps=True,
)

# Agent 3：Few-Shot CoT
fs_cot_agent = create_react_agent(llm=llm, tools=tools, prompt=few_shot_cot_prompt)
fs_cot_executor = AgentExecutor(
    agent=fs_cot_agent, tools=tools,
    verbose=True, max_iterations=6, handle_parsing_errors=True,
    return_intermediate_steps=True,
)


# ===== 6. 对比测试 =====
test_question = "用户 user_001 想买笔记本电脑，预算内有什么推荐？如果打 85 折呢？"

print("\n" + "=" * 60)
print(f"测试问题：{test_question}")
print("=" * 60)

# --- 测试1：普通 ReAct ---
print("\n" + "-" * 40)
print("Agent 1：普通 ReAct（无 CoT）")
print("-" * 40)
result_normal = normal_executor.invoke({"input": test_question})
normal_steps = len(result_normal["intermediate_steps"])

# --- 测试2：Zero-Shot CoT ---
print("\n" + "-" * 40)
print("Agent 2：Zero-Shot CoT")
print("-" * 40)
result_zs = zs_cot_executor.invoke({"input": test_question})
zs_steps = len(result_zs["intermediate_steps"])

# --- 测试3：Few-Shot CoT ---
print("\n" + "-" * 40)
print("Agent 3：Few-Shot CoT")
print("-" * 40)
result_fs = fs_cot_executor.invoke({"input": test_question})
fs_steps = len(result_fs["intermediate_steps"])


# ===== 7. 对比分析 =====
print("\n" + "=" * 60)
print("对比分析")
print("=" * 60)

print(f"""
┌──────────────────┬──────────┬──────────────┬──────────────┐
│   维度            │ 普通ReAct │ Zero-Shot CoT│ Few-Shot CoT │
├──────────────────┼──────────┼──────────────┼──────────────┤
│ 推理步骤数        │    {normal_steps}       │      {zs_steps}         │      {fs_steps}         │
│ 答案完整性        │ 见下方    │ 见下方        │ 见下方        │
└──────────────────┴──────────┴──────────────┴──────────────┘
""")

print("普通 ReAct 答案：")
print(f"  {result_normal['output'][:200]}...")
print(f"\nZero-Shot CoT 答案：")
print(f"  {result_zs['output'][:200]}...")
print(f"\nFew-Shot CoT 答案：")
print(f"  {result_fs['output'][:200]}...")


# ===== 8. CoT 在 RAG Agent 中的应用 =====
print("\n" + "=" * 60)
print("CoT 在 RAG Agent 中的应用")
print("=" * 60)

# RAG 场景下的 CoT Prompt 设计
rag_cot_template = """You are a helpful RAG assistant. Answer questions based on retrieved documents.

{tools}

Use the following format:

Question: the input question you must answer
Thought: Let's think step by step about this question:
  1. What is the user really asking? (Intent analysis)
  2. What keywords should I search for? (Query planning)
  3. Is the retrieved information sufficient? (Relevance check)
  4. Do I need to search again with different terms? (Query refinement)
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat as needed)
Thought: Let me synthesize the information and check for completeness.
Final Answer: the comprehensive answer based on retrieved information

Begin!

Question: {input}
Thought: Let's think step by step.{agent_scratchpad}"""

print("RAG CoT Prompt 的关键设计：")
print("  1. Intent Analysis - 理解用户真正想问什么")
print("  2. Query Planning - 规划搜索关键词")
print("  3. Relevance Check - 检查检索结果是否相关")
print("  4. Query Refinement - 必要时改写查询重新搜索")
print("  5. Synthesis - 综合多次检索结果给出完整答案")
```

---

## 预期输出

```
============================================================
对比实验：普通 ReAct vs CoT 增强 ReAct
============================================================

测试问题：用户 user_001 想买笔记本电脑，预算内有什么推荐？如果打 85 折呢？

----------------------------------------
Agent 1：普通 ReAct（无 CoT）
----------------------------------------

> Entering new AgentExecutor chain...
Thought: I need to check the user's budget first
Action: check_user_budget
Action Input: user_001
Observation: 预算：10000-15000 元，偏好：高性能笔记本，用途：编程开发
Thought: Now let me search for a laptop
Action: search_product
Action Input: macbook pro
Observation: MacBook Pro 14寸：价格 14999 元，评分 4.8/5，库存 50 台
Thought: Let me calculate the discount
Action: calculate_discount
Action Input: 14999 * 0.85
Observation: 计算结果：12749.15
Thought: I now know the final answer
Final Answer: 推荐 MacBook Pro 14寸，原价 14999 元，85折后 12749.15 元

> Finished chain.

----------------------------------------
Agent 2：Zero-Shot CoT
----------------------------------------

> Entering new AgentExecutor chain...
Thought: Let's think step by step. The user wants a laptop recommendation.
I need to: (1) check user_001's budget and preferences,
(2) find matching products, (3) calculate the discounted price,
(4) verify the discounted price is within budget.
Let me start by checking the user's information.
Action: check_user_budget
Action Input: user_001
Observation: 预算：10000-15000 元，偏好：高性能笔记本，用途：编程开发
...（后续步骤类似，但 Thought 更详细）

> Finished chain.
```

---

## 关键知识点解析

### 1. Zero-Shot CoT 的魔力

只需在 Prompt 中加入一句话，就能显著提升推理质量：

```python
# 普通 Prompt
"Thought: you should always think about what to do"

# Zero-Shot CoT（加一句话）
"Thought: Let's think step by step. First, analyze what information is needed."

# 效果差异：
# 普通：Thought 通常只有一句话，直接跳到 Action
# CoT：Thought 会展开多步分析，考虑更全面
```

### 2. Few-Shot CoT 的示例设计原则

```python
# 好的 Few-Shot 示例应该展示：
# 1. 分步推理过程（不是直接给答案）
# 2. 工具选择的理由（为什么用这个工具）
# 3. 信息整合（如何把多次查询的结果组合起来）
# 4. 边界检查（价格是否在预算内等）

# ❌ 不好的示例（太简单，没有推理过程）
"""
Question: 查一下 iPhone 价格
Thought: 搜索 iPhone
Action: search_product
Action Input: iphone
"""

# ✅ 好的示例（展示完整推理链）
"""
Question: 用户 user_002 想买手机，有什么推荐？
Thought: Let's think step by step. I need to first check user_002's
budget and preferences, then search for matching products, and finally
verify the price is within budget.
Action: check_user_budget
Action Input: user_002
"""
```

### 3. CoT 的适用场景

```
CoT 特别有效的场景：
  ✅ 多步推理（需要多次工具调用）
  ✅ 需要比较和判断（如价格是否在预算内）
  ✅ 需要综合多个信息源
  ✅ 复杂的 RAG 查询（需要查询改写）

CoT 效果不明显的场景：
  ❌ 简单的单步查询（"今天天气如何"）
  ❌ 直接的事实检索（"Python 是什么"）
  ❌ 简单计算（"1+1等于几"）
```

---

## 小结

思维链增强 Agent 的核心要点：

1. **Zero-Shot CoT**：加一句 "Let's think step by step" 就能提升推理质量，零成本
2. **Few-Shot CoT**：提供带推理过程的示例，效果更好但 Token 成本更高
3. **Prompt 设计**：在 Thought 指令中明确要求分析"已知/未知/下一步"
4. **RAG 应用**：CoT 特别适合需要多次检索和信息综合的 RAG 场景

---

## 下一步

- 想自定义输出解析？→ 阅读 `07_实战代码_场景4_自定义输出解析器.md`
- 想回顾推理模式？→ 阅读 `03_核心概念_1_ReAct推理模式.md`
