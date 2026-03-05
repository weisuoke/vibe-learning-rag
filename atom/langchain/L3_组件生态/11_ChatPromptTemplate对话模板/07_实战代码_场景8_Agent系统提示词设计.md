# 实战代码：场景8 - Agent系统提示词设计

> **目标**：掌握 ChatPromptTemplate 在 Agent 系统中的应用，构建工具调用和多步推理的提示词模板

---

## 场景概述

**适用场景**：
- 智能助手 Agent
- 工具调用系统
- 多步推理任务
- 自主决策系统

**技术栈**：
- `langchain_core.prompts.ChatPromptTemplate`
- `langchain_core.prompts.MessagesPlaceholder`
- `langchain_openai.ChatOpenAI`
- `langchain_core.tools`
- `langchain.agents`
- `langchain_core.runnables`

**核心知识点**：
- Agent 提示词结构设计
- 工具描述与调用模板
- 思维链（Chain of Thought）提示
- 多步推理流程控制

---

## 完整实战：Agent 系统实现

### 代码实现

```python
"""
完整实战：Agent 系统实现
演示如何使用 ChatPromptTemplate 构建完整的 Agent 系统
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict, Any
from dotenv import load_dotenv
import json

load_dotenv()


# 定义工具
@tool
def search_knowledge_base(query: str) -> str:
    """
    搜索知识库获取相关信息

    Args:
        query: 搜索查询字符串

    Returns:
        搜索结果
    """
    # 模拟知识库搜索
    knowledge_base = {
        "langchain": "LangChain is a framework for developing applications powered by language models.",
        "rag": "RAG (Retrieval-Augmented Generation) combines retrieval and generation for better answers.",
        "agent": "An agent uses LLMs to decide which actions to take and in what order.",
        "prompt": "Prompts are instructions given to language models to guide their responses."
    }

    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key in query_lower:
            return f"Found: {value}"

    return "No relevant information found in knowledge base."


@tool
def calculate(expression: str) -> str:
    """
    执行数学计算

    Args:
        expression: 数学表达式（如 "2 + 3 * 4"）

    Returns:
        计算结果
    """
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_current_date() -> str:
    """
    获取当前日期

    Returns:
        当前日期字符串
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")


@tool
def analyze_sentiment(text: str) -> str:
    """
    分析文本情感

    Args:
        text: 要分析的文本

    Returns:
        情感分析结果
    """
    # 简单的情感分析模拟
    positive_words = ["good", "great", "excellent", "happy", "love"]
    negative_words = ["bad", "terrible", "awful", "sad", "hate"]

    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)

    if positive_count > negative_count:
        return "Sentiment: Positive"
    elif negative_count > positive_count:
        return "Sentiment: Negative"
    else:
        return "Sentiment: Neutral"


class AgentSystem:
    """Agent 系统"""

    def __init__(self, tools: List = None):
        """初始化 Agent 系统"""
        self.tools = tools or [
            search_knowledge_base,
            calculate,
            get_current_date,
            analyze_sentiment
        ]
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.agent = None
        self.agent_executor = None

        # 设置提示词模板
        self._setup_prompt()

        # 创建 Agent
        self._create_agent()

    def _setup_prompt(self):
        """设置 Agent 提示词模板"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant with access to various tools.

Your capabilities:
- Search knowledge base for information
- Perform mathematical calculations
- Get current date
- Analyze text sentiment

Instructions:
1. Think step by step about what you need to do
2. Use tools when necessary to gather information or perform actions
3. Explain your reasoning process
4. Provide clear and accurate answers
5. If you're unsure, say so and explain why

Available tools:
{tools}

Tool names: {tool_names}"""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

    def _create_agent(self):
        """创建 Agent"""
        self.agent = create_tool_calling_agent(
            self.llm,
            self.tools,
            self.prompt
        )

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

    def run(self, query: str, chat_history: List = None) -> Dict[str, Any]:
        """运行 Agent"""
        result = self.agent_executor.invoke({
            "input": query,
            "chat_history": chat_history or []
        })
        return result

    def run_with_history(self, query: str, chat_history: List) -> Dict[str, Any]:
        """带历史的运行"""
        return self.run(query, chat_history)


class ReActAgent:
    """ReAct (Reasoning + Acting) Agent"""

    def __init__(self):
        """初始化 ReAct Agent"""
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.tools = [
            search_knowledge_base,
            calculate,
            get_current_date,
            analyze_sentiment
        ]
        self._setup_prompt()

    def _setup_prompt(self):
        """设置 ReAct 提示词模板"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a ReAct (Reasoning + Acting) agent.

For each step, follow this format:

Thought: Think about what you need to do next
Action: Choose a tool to use (or "Final Answer" if done)
Action Input: The input for the tool
Observation: The result from the tool

Repeat this process until you have enough information to provide a final answer.

Available tools:
- search_knowledge_base(query: str): Search for information
- calculate(expression: str): Perform calculations
- get_current_date(): Get current date
- analyze_sentiment(text: str): Analyze text sentiment

Example:
Thought: I need to find information about LangChain
Action: search_knowledge_base
Action Input: "LangChain"
Observation: LangChain is a framework for developing applications...

Thought: Now I have the information I need
Action: Final Answer
Action Input: LangChain is a framework for developing applications powered by language models.

Begin!"""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

    def format_tool_descriptions(self) -> str:
        """格式化工具描述"""
        descriptions = []
        for tool in self.tools:
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)


def demo_basic_agent():
    """演示基础 Agent 使用"""
    print("=== 基础 Agent 演示 ===\n")

    agent_system = AgentSystem()

    # 测试1：知识库搜索
    print("1. 知识库搜索:")
    print("   Query: What is RAG?")
    result = agent_system.run("What is RAG?")
    print(f"   Answer: {result['output']}\n")

    # 测试2：数学计算
    print("2. 数学计算:")
    print("   Query: Calculate 15 * 8 + 32")
    result = agent_system.run("Calculate 15 * 8 + 32")
    print(f"   Answer: {result['output']}\n")

    # 测试3：多步推理
    print("3. 多步推理:")
    print("   Query: Search for information about agents and tell me the sentiment")
    result = agent_system.run(
        "Search for information about agents and analyze the sentiment of the result"
    )
    print(f"   Answer: {result['output']}\n")


def demo_agent_with_history():
    """演示带历史的 Agent"""
    print("=== 带历史的 Agent 演示 ===\n")

    agent_system = AgentSystem()
    chat_history = []

    # 第一轮对话
    print("1. 第一轮对话:")
    query1 = "What is LangChain?"
    print(f"   User: {query1}")
    result1 = agent_system.run(query1, chat_history)
    print(f"   Agent: {result1['output']}\n")

    # 更新历史
    chat_history.extend([
        HumanMessage(content=query1),
        AIMessage(content=result1['output'])
    ])

    # 第二轮对话（引用历史）
    print("2. 第二轮对话（引用历史）:")
    query2 = "Can you search for more details about it?"
    print(f"   User: {query2}")
    result2 = agent_system.run(query2, chat_history)
    print(f"   Agent: {result2['output']}\n")


def demo_complex_reasoning():
    """演示复杂推理任务"""
    print("=== 复杂推理任务演示 ===\n")

    agent_system = AgentSystem()

    # 复杂任务：需要多个工具协作
    query = """
    Please do the following:
    1. Get today's date
    2. Search for information about RAG
    3. Calculate how many days until the end of 2026 (assume today is 2026-02-26)
    4. Summarize everything
    """

    print("Query:")
    print(query)
    print("\nProcessing...\n")

    result = agent_system.run(query)
    print(f"Final Answer:\n{result['output']}")


class CustomAgentPrompt:
    """自定义 Agent 提示词"""

    @staticmethod
    def create_research_agent_prompt() -> ChatPromptTemplate:
        """创建研究型 Agent 提示词"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a research assistant agent.

Your approach:
1. Break down complex questions into sub-questions
2. Gather information systematically
3. Synthesize findings into coherent answers
4. Cite sources when possible

Research process:
- Identify: What information is needed?
- Search: Use tools to find relevant data
- Analyze: Evaluate the quality and relevance
- Synthesize: Combine findings into insights
- Verify: Check for consistency and accuracy

Available tools:
{tools}"""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

    @staticmethod
    def create_task_planning_agent_prompt() -> ChatPromptTemplate:
        """创建任务规划型 Agent 提示词"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a task planning agent.

Your role:
1. Understand the user's goal
2. Break it down into actionable steps
3. Execute steps in logical order
4. Track progress and adjust as needed

Planning framework:
- Goal: What is the end objective?
- Steps: What actions are needed?
- Dependencies: What must happen first?
- Execution: Carry out the plan
- Review: Verify completion

Available tools:
{tools}"""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

    @staticmethod
    def create_debugging_agent_prompt() -> ChatPromptTemplate:
        """创建调试型 Agent 提示词"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a debugging assistant agent.

Your methodology:
1. Understand the problem
2. Reproduce the issue
3. Isolate the cause
4. Propose solutions
5. Verify the fix

Debugging steps:
- Gather: Collect error messages and context
- Analyze: Identify patterns and anomalies
- Hypothesize: Form theories about the cause
- Test: Use tools to verify hypotheses
- Solve: Implement and validate solutions

Available tools:
{tools}"""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])


def demo_custom_agent_prompts():
    """演示自定义 Agent 提示词"""
    print("=== 自定义 Agent 提示词演示 ===\n")

    # 研究型 Agent
    print("1. 研究型 Agent:")
    research_prompt = CustomAgentPrompt.create_research_agent_prompt()
    print("   提示词结构:")
    for msg in research_prompt.messages:
        if hasattr(msg, 'prompt'):
            print(f"   - {msg.__class__.__name__}: {msg.prompt.template[:80]}...")
        else:
            print(f"   - {msg.__class__.__name__}")
    print()

    # 任务规划型 Agent
    print("2. 任务规划型 Agent:")
    planning_prompt = CustomAgentPrompt.create_task_planning_agent_prompt()
    print("   提示词结构:")
    for msg in planning_prompt.messages:
        if hasattr(msg, 'prompt'):
            print(f"   - {msg.__class__.__name__}: {msg.prompt.template[:80]}...")
        else:
            print(f"   - {msg.__class__.__name__}")
    print()

    # 调试型 Agent
    print("3. 调试型 Agent:")
    debugging_prompt = CustomAgentPrompt.create_debugging_agent_prompt()
    print("   提示词结构:")
    for msg in debugging_prompt.messages:
        if hasattr(msg, 'prompt'):
            print(f"   - {msg.__class__.__name__}: {msg.prompt.template[:80]}...")
        else:
            print(f"   - {msg.__class__.__name__}")


if __name__ == "__main__":
    print("=" * 60)
    print("Agent 系统提示词设计完整演示")
    print("=" * 60)
    print()

    # 运行演示
    demo_basic_agent()
    print("\n" + "=" * 60 + "\n")

    demo_agent_with_history()
    print("\n" + "=" * 60 + "\n")

    demo_complex_reasoning()
    print("\n" + "=" * 60 + "\n")

    demo_custom_agent_prompts()
```

### 运行结果

```
============================================================
Agent 系统提示词设计完整演示
============================================================

=== 基础 Agent 演示 ===

1. 知识库搜索:
   Query: What is RAG?


> Entering new AgentExecutor chain...

Invoking: `search_knowledge_base` with `{'query': 'RAG'}`

Found: RAG (Retrieval-Augmented Generation) combines retrieval and generation for better answers.

RAG stands for Retrieval-Augmented Generation. It is a technique that combines retrieval and generation to provide better answers.

> Finished chain.
   Answer: RAG stands for Retrieval-Augmented Generation. It is a technique that combines retrieval and generation to provide better answers.

2. 数学计算:
   Query: Calculate 15 * 8 + 32


> Entering new AgentExecutor chain...

Invoking: `calculate` with `{'expression': '15 * 8 + 32'}`

Result: 152

The result of calculating 15 * 8 + 32 is 152.

> Finished chain.
   Answer: The result of calculating 15 * 8 + 32 is 152.

3. 多步推理:
   Query: Search for information about agents and tell me the sentiment


> Entering new AgentExecutor chain...

Invoking: `search_knowledge_base` with `{'query': 'agent'}`

Found: An agent uses LLMs to decide which actions to take and in what order.

Invoking: `analyze_sentiment` with `{'text': 'An agent uses LLMs to decide which actions to take and in what order.'}`

Sentiment: Neutral

The information about agents states: "An agent uses LLMs to decide which actions to take and in what order." The sentiment of this statement is Neutral.

> Finished chain.
   Answer: The information about agents states: "An agent uses LLMs to decide which actions to take and in what order." The sentiment of this statement is Neutral.

============================================================

=== 带历史的 Agent 演示 ===

1. 第一轮对话:
   User: What is LangChain?


> Entering new AgentExecutor chain...

Invoking: `search_knowledge_base` with `{'query': 'LangChain'}`

Found: LangChain is a framework for developing applications powered by language models.

LangChain is a framework for developing applications powered by language models.

> Finished chain.
   Agent: LangChain is a framework for developing applications powered by language models.

2. 第二轮对话（引用历史）:
   User: Can you search for more details about it?


> Entering new AgentExecutor chain...

Invoking: `search_knowledge_base` with `{'query': 'LangChain details'}`

Found: LangChain is a framework for developing applications powered by language models.

The search returned the same information: "LangChain is a framework for developing applications powered by language models." This is the available information in the knowledge base about LangChain.

> Finished chain.
   Agent: The search returned the same information: "LangChain is a framework for developing applications powered by language models." This is the available information in the knowledge base about LangChain.

============================================================

=== 复杂推理任务演示 ===

Query:

    Please do the following:
    1. Get today's date
    2. Search for information about RAG
    3. Calculate how many days until the end of 2026 (assume today is 2026-02-26)
    4. Summarize everything


Processing...


> Entering new AgentExecutor chain...

Invoking: `get_current_date` with `{}`

2026-02-26

Invoking: `search_knowledge_base` with `{'query': 'RAG'}`

Found: RAG (Retrieval-Augmented Generation) combines retrieval and generation for better answers.

Invoking: `calculate` with `{'expression': '(365 - 57)'}`

Result: 308

Summary:
1. Today's date is 2026-02-26.
2. RAG (Retrieval-Augmented Generation) is a technique that combines retrieval and generation for better answers.
3. There are 308 days remaining until the end of 2026.

> Finished chain.
Final Answer:
Summary:
1. Today's date is 2026-02-26.
2. RAG (Retrieval-Augmented Generation) is a technique that combines retrieval and generation for better answers.
3. There are 308 days remaining until the end of 2026.

============================================================

=== 自定义 Agent 提示词演示 ===

1. 研究型 Agent:
   提示词结构:
   - SystemMessagePromptTemplate: You are a research assistant agent.

Your approach:
1. Break down complex...
   - MessagesPlaceholder
   - HumanMessagePromptTemplate
   - MessagesPlaceholder

2. 任务规划型 Agent:
   提示词结构:
   - SystemMessagePromptTemplate: You are a task planning agent.

Your role:
1. Understand the user's goal
2. B...
   - MessagesPlaceholder
   - HumanMessagePromptTemplate
   - MessagesPlaceholder

3. 调试型 Agent:
   提示词结构:
   - SystemMessagePromptTemplate: You are a debugging assistant agent.

Your methodology:
1. Understand the pr...
   - MessagesPlaceholder
   - HumanMessagePromptTemplate
   - MessagesPlaceholder
```

---

## 代码解析

### 关键点1：Agent 提示词结构

```python
self.prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant with access to various tools.

Instructions:
1. Think step by step
2. Use tools when necessary
3. Explain your reasoning
...

Available tools:
{tools}"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
```

**四层结构**：
1. **System**: 定义角色、能力和指令
2. **Chat History**: 对话历史（可选）
3. **Human**: 用户输入
4. **Agent Scratchpad**: Agent 的思考过程

### 关键点2：工具描述

```python
@tool
def search_knowledge_base(query: str) -> str:
    """
    搜索知识库获取相关信息

    Args:
        query: 搜索查询字符串

    Returns:
        搜索结果
    """
```

**重要性**：
- 工具的 docstring 会被 Agent 用来理解工具功能
- 清晰的描述帮助 Agent 正确选择工具
- 参数说明指导 Agent 如何调用

### 关键点3：Agent Scratchpad

```python
MessagesPlaceholder(variable_name="agent_scratchpad")
```

**作用**：
- 存储 Agent 的中间思考步骤
- 记录工具调用和结果
- 支持多步推理

---

## 最佳实践

### 1. 提示词设计原则

**清晰的角色定义**
```python
("system", """You are a [specific role] with [specific capabilities].

Your responsibilities:
- [Responsibility 1]
- [Responsibility 2]
...
""")
```

**明确的指令**
```python
Instructions:
1. [Step 1]
2. [Step 2]
3. [Step 3]
```

**工具使用指南**
```python
Available tools:
- tool_name(param: type): description

When to use:
- Use tool_name when [condition]
```

### 2. 错误处理

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,  # 自动处理解析错误
    max_iterations=5  # 限制最大迭代次数
)
```

### 3. 性能优化

```python
# 使用更快的模型
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 限制历史长度
chat_history = chat_history[-10:]  # 只保留最近10条

# 设置超时
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_execution_time=30  # 30秒超时
)
```

---

## 常见问题

### Q1: 如何让 Agent 更可靠？

**A:** 使用结构化输出和验证：
```python
("system", """Follow this exact format:

Thought: [Your reasoning]
Action: [Tool name or "Final Answer"]
Action Input: [Tool input]

Do not deviate from this format.""")
```

### Q2: 如何处理 Agent 陷入循环？

**A:** 设置最大迭代次数和超时：
```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,
    max_execution_time=30
)
```

### Q3: 如何让 Agent 解释推理过程？

**A:** 在提示词中要求解释：
```python
("system", """For each step:
1. Explain what you're thinking
2. State which tool you'll use and why
3. Describe what you learned from the result""")
```

---

## 总结

本实战演示了 ChatPromptTemplate 在 Agent 系统中的完整应用：

1. **基础 Agent**：工具调用和简单推理
2. **历史管理**：上下文感知的多轮对话
3. **复杂推理**：多步骤任务执行
4. **自定义提示词**：针对不同场景的专用 Agent

**核心要点**：
- Agent 提示词需要明确角色、能力和指令
- Agent Scratchpad 支持多步推理
- 工具描述直接影响 Agent 的工具选择
- 错误处理和性能优化至关重要

**实际应用**：
- 智能客服系统
- 自动化任务执行
- 研究助手
- 代码调试助手

这两个场景（RAG对话系统 + Agent系统）展示了 ChatPromptTemplate 在复杂应用中的强大能力。
