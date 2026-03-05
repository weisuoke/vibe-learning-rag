# Agent推理与决策 - 实战代码：ReAct Agent 实战

> 从零构建一个完整的 ReAct Agent，亲眼看到 Thought→Action→Observation 循环如何运转

---

## 场景说明

**目标：** 构建一个能够搜索信息和做数学计算的 ReAct Agent，展示完整的推理过程。

**你将学到：**
1. 如何定义自定义工具
2. 如何用 `create_react_agent` 构建 Agent
3. 如何用 `AgentExecutor` 运行 Agent
4. 如何通过 `verbose=True` 观察每一步推理
5. 如何控制最大迭代次数和错误处理

**使用的库：**
- `langchain` - Agent 框架
- `langchain-openai` - OpenAI 模型接入
- `langchain-core` - 核心组件
- `python-dotenv` - 环境变量管理

---

## 前置准备

```bash
# 安装依赖
uv add langchain langchain-openai langchain-core python-dotenv

# 确保 .env 文件中有 API Key
# OPENAI_API_KEY=your_key_here
# OPENAI_BASE_URL=https://your-proxy.com/v1  (可选)
```

---

## 完整代码

```python
"""
ReAct Agent 实战
演示：构建一个完整的 ReAct Agent，展示 Thought→Action→Observation 循环

运行方式：python examples/react_agent_demo.py
"""

import os
import math
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor

# ===== 0. 加载环境变量 =====
load_dotenv()

# ===== 1. 初始化 LLM =====
print("=" * 60)
print("ReAct Agent 实战演示")
print("=" * 60)

llm = ChatOpenAI(
    model="gpt-4o-mini",  # 推荐使用 gpt-4o-mini，性价比高
    temperature=0,          # 设为 0 让输出更确定，方便调试
)


# ===== 2. 定义自定义工具 =====
# 工具是 Agent 的"手和脚"，让它能与外部世界交互

@tool
def search_knowledge(query: str) -> str:
    """搜索知识库获取信息。输入应该是一个搜索查询字符串。
    适用于查询事实性信息，如人物、事件、概念等。"""

    # 模拟知识库搜索（实际项目中会调用真实的搜索 API）
    knowledge_base = {
        "python": "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。"
                  "最新稳定版本是 Python 3.13。广泛用于 Web 开发、数据科学、AI 等领域。",
        "langchain": "LangChain 是一个用于构建 LLM 应用的开源框架，"
                     "由 Harrison Chase 于 2022 年创建。支持 Agent、RAG、Chain 等模式。"
                     "最新版本支持 LCEL 和 LangGraph。",
        "react": "ReAct 是 Reasoning + Acting 的缩写，由 Yao et al. 在 2022 年提出。"
                 "核心思想是让 LLM 交替进行推理和行动，通过 Thought→Action→Observation 循环解决问题。",
        "rag": "RAG（Retrieval-Augmented Generation）是检索增强生成技术。"
               "通过先检索相关文档，再让 LLM 基于检索结果生成回答，减少幻觉。",
    }

    # 简单的关键词匹配搜索
    query_lower = query.lower()
    results = []
    for key, value in knowledge_base.items():
        if key in query_lower or query_lower in key:
            results.append(value)

    if results:
        return "\n".join(results)
    return f"未找到与 '{query}' 相关的信息。请尝试其他关键词。"


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入应该是一个有效的数学表达式字符串。
    支持基本运算（+、-、*、/）和常用函数（sqrt、pow、sin、cos 等）。
    示例输入：'2 + 3 * 4' 或 'sqrt(144)' 或 'pow(2, 10)'"""

    try:
        # 安全的数学计算（只允许数学相关的函数）
        allowed_names = {
            "sqrt": math.sqrt,
            "pow": pow,
            "abs": abs,
            "round": round,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "pi": math.pi,
            "e": math.e,
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}。请检查表达式是否正确。"


# 把工具放到列表中
tools = [search_knowledge, calculator]

print("\n已注册的工具：")
for t in tools:
    print(f"  - {t.name}: {t.description[:50]}...")


# ===== 3. 构建 ReAct Prompt =====
# 这是 ReAct Agent 的核心——告诉 LLM 按什么格式推理

react_prompt_template = """Answer the following questions as best you can. You have access to the following tools:

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

prompt = PromptTemplate.from_template(react_prompt_template)

print("\nPrompt 模板中的变量：", prompt.input_variables)


# ===== 4. 创建 ReAct Agent =====
# create_react_agent 会自动：
#   1. 绑定 stop sequence ["\nObservation"] 到 LLM
#   2. 设置 ReActSingleInputOutputParser 作为输出解析器
#   3. 构建 LCEL 链：prompt | llm_with_stop | output_parser

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

print("\nAgent 创建成功！")
print(f"Agent 类型：{type(agent).__name__}")


# ===== 5. 创建 AgentExecutor =====
# AgentExecutor 负责运行 Agent 的循环：
#   调用 Agent → 执行工具 → 把结果反馈给 Agent → 重复

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,           # 开启详细输出，看到每一步推理
    max_iterations=5,       # 最多循环 5 次，防止无限循环
    handle_parsing_errors=True,  # 解析失败时自动重试
    return_intermediate_steps=True,  # 返回中间步骤
)


# ===== 6. 运行示例 =====

# --- 示例1：简单知识查询 ---
print("\n" + "=" * 60)
print("示例1：简单知识查询")
print("=" * 60)

result1 = agent_executor.invoke({
    "input": "什么是 LangChain？"
})

print(f"\n最终答案：{result1['output']}")
print(f"推理步骤数：{len(result1['intermediate_steps'])}")


# --- 示例2：需要计算的问题 ---
print("\n" + "=" * 60)
print("示例2：数学计算")
print("=" * 60)

result2 = agent_executor.invoke({
    "input": "计算 2 的 10 次方，然后告诉我结果的平方根是多少"
})

print(f"\n最终答案：{result2['output']}")
print(f"推理步骤数：{len(result2['intermediate_steps'])}")


# --- 示例3：多步推理（搜索 + 计算） ---
print("\n" + "=" * 60)
print("示例3：多步推理（搜索 + 计算）")
print("=" * 60)

result3 = agent_executor.invoke({
    "input": "Python 是哪一年创建的？从那年到 2025 年过了多少年？"
})

print(f"\n最终答案：{result3['output']}")
print(f"推理步骤数：{len(result3['intermediate_steps'])}")


# ===== 7. 分析推理过程 =====
print("\n" + "=" * 60)
print("推理过程分析（以示例3为例）")
print("=" * 60)

for i, (action, observation) in enumerate(result3["intermediate_steps"]):
    print(f"\n--- 第 {i + 1} 步 ---")
    print(f"  工具：{action.tool}")
    print(f"  输入：{action.tool_input}")
    print(f"  结果：{observation[:100]}...")  # 只显示前100字符
    if hasattr(action, "log"):
        print(f"  完整日志：\n{action.log}")
```

---

## 预期输出

```
============================================================
ReAct Agent 实战演示
============================================================

已注册的工具：
  - search_knowledge: 搜索知识库获取信息。输入应该是一个搜索查询字符串。...
  - calculator: 计算数学表达式。输入应该是一个有效的数学表达式字符串。...

Prompt 模板中的变量：['agent_scratchpad', 'input', 'tool_names', 'tools']

Agent 创建成功！
Agent 类型：RunnableSequence

============================================================
示例1：简单知识查询
============================================================

> Entering new AgentExecutor chain...
Thought: 我需要搜索关于 LangChain 的信息
Action: search_knowledge
Action Input: langchain
Observation: LangChain 是一个用于构建 LLM 应用的开源框架...
Thought: I now know the final answer
Final Answer: LangChain 是一个用于构建 LLM 应用的开源框架，由 Harrison Chase 于 2022 年创建...

> Finished chain.

最终答案：LangChain 是一个用于构建 LLM 应用的开源框架...
推理步骤数：1

============================================================
示例2：数学计算
============================================================

> Entering new AgentExecutor chain...
Thought: 我需要先计算 2 的 10 次方，然后计算结果的平方根
Action: calculator
Action Input: pow(2, 10)
Observation: 计算结果：pow(2, 10) = 1024
Thought: 现在我需要计算 1024 的平方根
Action: calculator
Action Input: sqrt(1024)
Observation: 计算结果：sqrt(1024) = 32.0
Thought: I now know the final answer
Final Answer: 2 的 10 次方是 1024，1024 的平方根是 32

> Finished chain.

最终答案：2 的 10 次方是 1024，1024 的平方根是 32
推理步骤数：2
```

---

## 关键知识点解析

### 1. create_react_agent 内部做了什么？

```python
# create_react_agent 的核心逻辑（简化版）：
def create_react_agent(llm, tools, prompt):
    # 1. 绑定 stop sequence，防止 LLM 自己编造 Observation
    llm_with_stop = llm.bind(stop=["\nObservation"])

    # 2. 构建 LCEL 链
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"])
        )
        | prompt  # 填充 prompt 模板
        | llm_with_stop  # 调用 LLM（到 Observation 就停）
        | ReActSingleInputOutputParser()  # 解析输出
    )
    return agent
```

### 2. AgentExecutor 的循环机制

```
AgentExecutor 运行流程：
┌─────────────────────────────────────────┐
│  1. 调用 agent.invoke(input)            │
│     ↓                                   │
│  2. 得到 AgentAction 或 AgentFinish     │
│     ↓                                   │
│  3. 如果是 AgentAction：                │
│     a. 执行对应的工具                    │
│     b. 把结果加入 intermediate_steps     │
│     c. 回到步骤 1                       │
│     ↓                                   │
│  4. 如果是 AgentFinish：                │
│     返回最终答案                         │
│     ↓                                   │
│  5. 如果超过 max_iterations：           │
│     强制停止，返回错误信息               │
└─────────────────────────────────────────┘
```

### 3. handle_parsing_errors 的作用

```python
# 当 LLM 输出不符合预期格式时：

# handle_parsing_errors=False（默认）
# → 直接抛出 OutputParserException，程序崩溃

# handle_parsing_errors=True
# → 把错误信息发回给 LLM，让它重新生成
# → 相当于告诉 LLM："你的格式不对，请重新按格式输出"

# handle_parsing_errors="自定义错误提示"
# → 用自定义提示替代默认错误信息
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors="格式错误，请严格按照以下格式输出：\n"
                          "Thought: ...\nAction: ...\nAction Input: ...",
)
```

### 4. verbose 输出解读

```
> Entering new AgentExecutor chain...    ← 开始执行

Thought: 我需要搜索...                   ← LLM 的推理（Thought）
Action: search_knowledge                 ← LLM 选择的工具
Action Input: langchain                  ← 工具的输入参数
Observation: LangChain 是...             ← 工具的返回结果（真实的！）

Thought: I now know the final answer     ← LLM 决定结束
Final Answer: ...                        ← 最终答案

> Finished chain.                        ← 执行完成
```

---

## 常见问题与解决方案

### Q1：Agent 陷入无限循环怎么办？

```python
# 设置 max_iterations 限制最大循环次数
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,  # 最多 5 轮
    max_execution_time=30,  # 最多运行 30 秒
)
```

### Q2：Agent 选错了工具怎么办？

```python
# 改善工具描述，让 LLM 更容易选对
@tool
def search_knowledge(query: str) -> str:
    """搜索知识库获取事实性信息。
    适用场景：查询人物、事件、概念、定义等。
    不适用：数学计算、代码执行。
    输入示例：'什么是 Python' 或 'LangChain 的创建者是谁'"""
    ...
```

### Q3：如何查看完整的 Prompt？

```python
# 在调用前打印完整 prompt
from langchain.agents.format_scratchpad import format_log_to_str

full_prompt = prompt.format(
    tools="search_knowledge: 搜索知识库\ncalculator: 数学计算",
    tool_names="search_knowledge, calculator",
    input="什么是 LangChain？",
    agent_scratchpad="",
)
print(full_prompt)
```

---

## 小结

这个示例展示了 ReAct Agent 的完整工作流程：

1. **定义工具** → Agent 的能力边界
2. **构建 Prompt** → Agent 的推理格式
3. **create_react_agent** → 组装 LCEL 链
4. **AgentExecutor** → 运行推理循环
5. **verbose=True** → 观察推理过程

核心要点：ReAct Agent 的质量取决于**工具描述的清晰度**和 **Prompt 的格式约束**，而不仅仅是 LLM 的能力。

---

## 下一步

- 想了解零样本推理？→ 阅读 `07_实战代码_场景2_MRKL零样本推理.md`
- 想用思维链增强？→ 阅读 `07_实战代码_场景3_思维链增强Agent.md`
