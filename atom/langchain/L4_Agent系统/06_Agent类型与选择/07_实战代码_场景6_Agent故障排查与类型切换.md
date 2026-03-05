# 实战代码 - 场景6：Agent故障排查与类型切换

> 诊断 Agent 常见问题并动态切换 Agent 类型的完整指南

---

## 场景说明

在实际开发中，Agent 可能会遇到各种问题：
- ❌ Agent 不调用工具，直接返回答案
- ❌ Agent 调用错误的工具
- ❌ Agent 陷入无限循环
- ❌ 工具调用失败或解析错误

本文档提供系统化的故障排查方法和解决方案，包括如何动态切换 Agent 类型来解决问题。

---

## 问题1：Agent 不调用工具

### 问题描述

**现象**：
```python
# 用户问题明确需要工具
result = agent.invoke({"input": "北京今天天气如何？"})
# Agent 输出: "抱歉，我无法查询实时天气信息"
# ❌ 没有调用 get_weather 工具
```

### 诊断方法

#### 方法1：启用 verbose 模式

```python
"""
诊断工具：启用 verbose 查看 Agent 推理过程
"""

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """查询城市天气"""
    return f"{city}今天晴天，15-25°C"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_openai_functions_agent(llm, [get_weather], prompt)

# 启用 verbose 模式
executor = AgentExecutor(
    agent=agent,
    tools=[get_weather],
    verbose=True,  # 🔍 关键：显示推理过程
    return_intermediate_steps=True
)

result = executor.invoke({"input": "北京今天天气如何？"})

# 查看中间步骤
print("\n=== 中间步骤 ===")
for step in result.get("intermediate_steps", []):
    action, observation = step
    print(f"工具: {action.tool}")
    print(f"输入: {action.tool_input}")
    print(f"输出: {observation}\n")
```

#### 方法2：检查工具描述

```python
"""
诊断工具：检查工具描述是否清晰
"""

# ❌ 错误示例：描述不清晰
@tool
def get_weather(city: str) -> str:
    """天气"""  # 太简短，模型不知道何时使用
    return f"{city}今天晴天"

# ✅ 正确示例：描述详细
@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气信息。

    使用场景：
    - 用户询问某个城市的天气
    - 用户询问温度、降雨等天气相关信息

    Args:
        city: 城市名称（如"北京"、"上海"）

    Returns:
        天气信息字符串
    """
    return f"{city}今天晴天，15-25°C"
```

### 解决方案

#### 方案1：优化 System Prompt

```python
"""
解决方案1：在 System Prompt 中明确要求使用工具
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ❌ 错误 Prompt：没有强调工具使用
bad_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手。"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# ✅ 正确 Prompt：明确要求使用工具
good_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个智能助手，拥有以下工具：

工具使用规则：
1. 当用户询问需要实时数据的问题时，必须使用工具
2. 不要猜测或编造信息，使用工具获取准确数据
3. 如果不确定，优先使用工具而不是直接回答

可用工具：
- get_weather: 查询城市天气
- search_web: 搜索网络信息
- calculate: 执行数学计算

记住：优先使用工具，而不是依赖你的知识！
"""),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, tools, good_prompt)
```

#### 方案2：降低模型 Temperature

```python
"""
解决方案2：降低 temperature 提高确定性
"""

# ❌ 错误：temperature 太高，模型行为不稳定
llm_bad = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ✅ 正确：temperature=0，确保稳定的工具调用
llm_good = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_openai_functions_agent(llm_good, tools, prompt)
```

#### 方案3：切换到 ReAct Agent（显式推理）

```python
"""
解决方案3：切换到 ReAct Agent，强制显式推理
"""

from langchain.agents import create_react_agent
from langchain_core.prompts import PromptTemplate

# ReAct Prompt 模板
react_prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

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

IMPORTANT: You MUST use tools to get accurate information. Do not guess!

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

# 创建 ReAct Agent
agent = create_react_agent(llm, tools, react_prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ReAct 会显式输出 Thought，更容易发现问题
result = executor.invoke({"input": "北京今天天气如何？"})
```

---

## 问题2：Agent 调用错误的工具

### 问题描述

**现象**：
```python
# 用户问题：计算数学表达式
result = agent.invoke({"input": "23 + 45 等于多少？"})
# ❌ Agent 调用了 search_web 而不是 calculator
```

### 诊断方法

```python
"""
诊断工具：分析工具选择逻辑
"""

from langchain.callbacks import StdOutCallbackHandler

class ToolSelectionCallback(StdOutCallbackHandler):
    """自定义回调：追踪工具选择"""

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "Unknown")
        print(f"\n🔧 选择工具: {tool_name}")
        print(f"📥 输入: {input_str}")

    def on_tool_end(self, output, **kwargs):
        print(f"📤 输出: {output[:100]}...")  # 只显示前100字符

# 使用回调
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[ToolSelectionCallback()],
    verbose=True
)

result = executor.invoke({"input": "23 + 45 等于多少？"})
```

### 解决方案

#### 方案1：优化工具名称和描述

```python
"""
解决方案1：使用清晰的工具名称和详细描述
"""

from langchain.tools import tool

# ❌ 错误示例：名称和描述模糊
@tool
def tool1(query: str) -> str:
    """搜索"""  # 太模糊
    return "结果"

@tool
def tool2(expr: str) -> str:
    """计算"""  # 太模糊
    return "结果"

# ✅ 正确示例：名称和描述清晰
@tool
def search_web(query: str) -> str:
    """在互联网上搜索信息。

    适用场景：
    - 查询最新新闻、事件
    - 搜索网页内容
    - 查找资料和文档

    不适用场景：
    - 数学计算（使用 calculator）
    - 本地数据查询（使用 database_query）

    Args:
        query: 搜索关键词
    """
    return f"搜索结果: {query}"

@tool
def calculator(expression: str) -> str:
    """执行数学计算。

    适用场景：
    - 加减乘除运算
    - 数学表达式求值
    - 数值计算

    不适用场景：
    - 搜索信息（使用 search_web）
    - 文本处理（使用其他工具）

    Args:
        expression: 数学表达式（如 "23 + 45" 或 "10 * 5"）

    Examples:
        - "2 + 2" → 4
        - "10 * 5" → 50
    """
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"
```

#### 方案2：使用 Pydantic Schema 增强类型提示

```python
"""
解决方案2：使用 Pydantic 定义严格的参数 Schema
"""

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

class CalculatorInput(BaseModel):
    """计算器输入参数"""
    expression: str = Field(
        description="数学表达式，如 '23 + 45' 或 '10 * 5'。只能包含数字和运算符（+, -, *, /）"
    )

class SearchInput(BaseModel):
    """搜索输入参数"""
    query: str = Field(
        description="搜索关键词或问题，用于在互联网上查找信息"
    )

def calculate_func(expression: str) -> str:
    """执行计算"""
    return f"结果: {eval(expression)}"

def search_func(query: str) -> str:
    """执行搜索"""
    return f"搜索: {query}"

# 创建工具（带 Schema）
calculator_tool = StructuredTool.from_function(
    func=calculate_func,
    name="calculator",
    description="执行数学计算。输入必须是数学表达式。",
    args_schema=CalculatorInput
)

search_tool = StructuredTool.from_function(
    func=search_func,
    name="search_web",
    description="在互联网上搜索信息。输入必须是搜索关键词。",
    args_schema=SearchInput
)

tools = [calculator_tool, search_tool]
```

#### 方案3：切换到 Structured Chat Agent

```python
"""
解决方案3：使用 Structured Chat Agent 处理复杂工具选择
"""

from langchain.agents import create_structured_chat_agent

# Structured Chat 对工具选择更精确
agent = create_structured_chat_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = executor.invoke({"input": "23 + 45 等于多少？"})
```

---

## 问题3：Agent 陷入无限循环

### 问题描述

**现象**：
```python
# Agent 一直重复相同的操作
result = agent.invoke({"input": "查询数据"})
# ❌ 输出：
# Thought: 我需要查询数据
# Action: query_db
# Observation: 查询失败
# Thought: 我需要查询数据  # 重复！
# Action: query_db
# ...（无限循环）
```

### 诊断方法

```python
"""
诊断工具：监控迭代次数和重复行为
"""

class LoopDetectionCallback(StdOutCallbackHandler):
    """检测循环的回调"""

    def __init__(self):
        super().__init__()
        self.action_history = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name")
        self.action_history.append((tool_name, input_str))

        # 检测重复
        if len(self.action_history) >= 3:
            last_three = self.action_history[-3:]
            if last_three[0] == last_three[1] == last_three[2]:
                print(f"\n⚠️ 警告：检测到循环！工具 '{tool_name}' 连续调用3次")
                print(f"输入: {input_str}")

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[LoopDetectionCallback()],
    verbose=True
)
```

### 解决方案

#### 方案1：设置最大迭代次数

```python
"""
解决方案1：限制最大迭代次数
"""

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,  # 🔑 最多5次迭代
    max_execution_time=30,  # 🔑 最多30秒
    early_stopping_method="generate",  # 🔑 超时后生成答案
    verbose=True
)

try:
    result = executor.invoke({"input": "查询数据"})
except Exception as e:
    print(f"Agent 执行失败: {e}")
    # 降级处理
```

#### 方案2：改进工具错误处理

```python
"""
解决方案2：工具返回明确的错误信息，避免重试
"""

@tool
def query_database(query: str) -> str:
    """查询数据库"""
    try:
        # 模拟查询
        if not query:
            # ❌ 错误：返回模糊错误
            # return "查询失败"

            # ✅ 正确：返回明确错误和建议
            return """查询失败：查询参数为空。

建议：
1. 请提供具体的查询条件
2. 示例：query_database("SELECT * FROM users WHERE age > 18")

如果不确定如何查询，请直接告诉用户你需要更多信息。
"""

        # 执行查询
        result = f"查询结果: {query}"
        return result

    except Exception as e:
        # 返回详细错误信息
        return f"""查询失败：{str(e)}

这是一个无法恢复的错误。请：
1. 检查查询语法
2. 或者告诉用户查询失败的原因

不要重试相同的查询！
"""
```

#### 方案3：切换到 ReAct Agent 并优化 Prompt

```python
"""
解决方案3：使用 ReAct Agent + 防循环 Prompt
"""

from langchain.agents import create_react_agent

react_prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

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

IMPORTANT RULES:
1. If a tool fails, DO NOT retry the same action immediately
2. If you get an error, analyze the error message and try a different approach
3. If you cannot solve the problem after 2 attempts, explain the issue to the user
4. Maximum 3 tool calls per question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, react_prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=3,  # 严格限制
    verbose=True
)
```

---

## 动态切换 Agent 类型

### 场景：根据问题复杂度自动切换

```python
"""
完整示例：智能 Agent 切换器
根据问题复杂度和工具类型自动选择最佳 Agent
"""

from typing import List, Any
from langchain.tools import BaseTool
from langchain.agents import (
    create_openai_functions_agent,
    create_react_agent,
    create_structured_chat_agent,
    AgentExecutor
)

class SmartAgentSelector:
    """智能 Agent 选择器"""

    def __init__(self, llm, tools: List[BaseTool]):
        self.llm = llm
        self.tools = tools
        self.execution_history = []

    def analyze_query_complexity(self, query: str) -> str:
        """分析查询复杂度"""
        # 简单启发式规则
        if len(query) < 20:
            return "simple"
        elif "步骤" in query or "首先" in query or "然后" in query:
            return "complex"
        else:
            return "medium"

    def select_agent_type(self, query: str) -> str:
        """选择 Agent 类型"""
        complexity = self.analyze_query_complexity(query)

        # 检查历史失败记录
        if self.execution_history:
            last_result = self.execution_history[-1]
            if not last_result["success"]:
                # 上次失败，切换类型
                last_type = last_result["agent_type"]
                if last_type == "openai_functions":
                    return "react"  # 切换到 ReAct 获得更好的可见性
                elif last_type == "react":
                    return "structured_chat"  # 切换到 Structured Chat

        # 根据复杂度选择
        if complexity == "simple":
            return "openai_functions"
        elif complexity == "complex":
            return "react"  # 复杂任务用 ReAct 便于调试
        else:
            return "openai_functions"

    def create_agent(self, agent_type: str):
        """创建指定类型的 Agent"""
        print(f"\n🤖 创建 Agent 类型: {agent_type}")

        if agent_type == "openai_functions":
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个智能助手。优先使用工具获取准确信息。"),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ])
            agent = create_openai_functions_agent(self.llm, self.tools, prompt)

        elif agent_type == "react":
            prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

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
Thought:{agent_scratchpad}
""")
            agent = create_react_agent(self.llm, self.tools, prompt)

        elif agent_type == "structured_chat":
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个智能助手。"),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ])
            agent = create_structured_chat_agent(self.llm, self.tools, prompt)

        else:
            raise ValueError(f"未知 Agent 类型: {agent_type}")

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )

    def invoke(self, query: str) -> dict:
        """执行查询（自动选择 Agent 类型）"""
        # 选择 Agent 类型
        agent_type = self.select_agent_type(query)

        # 创建 Agent
        executor = self.create_agent(agent_type)

        # 执行
        try:
            result = executor.invoke({"input": query})
            success = True
            error = None
        except Exception as e:
            result = {"output": f"执行失败: {str(e)}"}
            success = False
            error = str(e)

        # 记录历史
        self.execution_history.append({
            "query": query,
            "agent_type": agent_type,
            "success": success,
            "error": error
        })

        return result


# ===== 使用示例 =====

from langchain_openai import ChatOpenAI
from langchain.tools import tool

# 定义工具
@tool
def search_web(query: str) -> str:
    """搜索网络信息"""
    return f"搜索结果: {query}"

@tool
def calculator(expression: str) -> str:
    """执行数学计算"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

# 创建智能选择器
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
selector = SmartAgentSelector(llm, [search_web, calculator])

# 测试1: 简单查询 → OpenAI Functions
print("=== 测试1：简单查询 ===")
result1 = selector.invoke("23 + 45 等于多少？")
print(f"结果: {result1['output']}\n")

# 测试2: 复杂查询 → ReAct
print("=== 测试2：复杂查询 ===")
result2 = selector.invoke("首先搜索 LangChain 的信息，然后计算 100 * 50")
print(f"结果: {result2['output']}\n")

# 测试3: 如果失败，自动切换类型
print("=== 测试3：失败后自动切换 ===")
result3 = selector.invoke("复杂的多步骤任务")
print(f"结果: {result3['output']}\n")
```

---

## 使用 LangSmith 追踪 Agent 行为

### 配置 LangSmith

```python
"""
使用 LangSmith 追踪和调试 Agent
"""

import os
from dotenv import load_dotenv

load_dotenv()

# 配置 LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "agent-debugging"

# 现在所有 Agent 执行都会被追踪
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = executor.invoke({"input": "测试查询"})

# 在 LangSmith UI 中查看：
# - 每一步的输入输出
# - Token 使用情况
# - 执行时间
# - 错误堆栈
```

### LangSmith 调试技巧

```python
"""
LangSmith 高级调试技巧
"""

from langsmith import Client

client = Client()

# 1. 查询最近的运行记录
runs = client.list_runs(
    project_name="agent-debugging",
    execution_order=1,
    limit=10
)

for run in runs:
    print(f"Run ID: {run.id}")
    print(f"Status: {run.status}")
    print(f"Duration: {run.end_time - run.start_time}")
    print(f"Error: {run.error}")
    print("---")

# 2. 分析失败的运行
failed_runs = client.list_runs(
    project_name="agent-debugging",
    filter='eq(status, "error")',
    limit=5
)

for run in failed_runs:
    print(f"失败原因: {run.error}")
    print(f"输入: {run.inputs}")
    print(f"输出: {run.outputs}")
    print("---")
```

---

## 故障排查检查清单

### 问题：Agent 不调用工具

- [ ] 检查工具描述是否清晰详细
- [ ] 检查 System Prompt 是否强调工具使用
- [ ] 检查 temperature 是否设置为 0
- [ ] 启用 verbose 模式查看推理过程
- [ ] 考虑切换到 ReAct Agent

### 问题：Agent 调用错误工具

- [ ] 检查工具名称是否有歧义
- [ ] 检查工具描述是否包含适用场景
- [ ] 使用 Pydantic Schema 增强类型提示
- [ ] 添加工具选择回调监控
- [ ] 考虑切换到 Structured Chat Agent

### 问题：Agent 陷入无限循环

- [ ] 设置 max_iterations 限制
- [ ] 设置 max_execution_time 超时
- [ ] 改进工具错误信息
- [ ] 在 Prompt 中添加防循环规则
- [ ] 使用循环检测回调

### 问题：工具调用失败

- [ ] 检查工具函数是否有异常处理
- [ ] 检查工具输入参数是否正确
- [ ] 启用 handle_parsing_errors
- [ ] 查看 LangSmith 追踪日志
- [ ] 考虑降级处理

---

## 总结

### 核心要点

1. **诊断优先**：使用 verbose、回调、LangSmith 定位问题
2. **渐进式解决**：从简单方案（优化 Prompt）到复杂方案（切换 Agent）
3. **防御性编程**：设置超时、限制迭代、处理错误
4. **动态切换**：根据问题复杂度和历史表现自动选择 Agent 类型
5. **持续监控**：使用 LangSmith 追踪生产环境问题

### 最佳实践

- ✅ 开发阶段：使用 ReAct Agent + verbose 模式
- ✅ 测试阶段：使用 LangSmith 追踪所有执行
- ✅ 生产阶段：使用 OpenAI Functions + 错误处理
- ✅ 故障排查：启用详细日志 + 动态切换

### 下一步

- 学习 `08_面试必问.md` 掌握 Agent 类型选择的核心知识
- 学习 `07_实战代码_场景7_多Agent类型对比测试.md` 进行性能对比
- 实践：在自己的项目中实现智能 Agent 切换器

---

**记住**：故障排查是一个迭代过程。从简单的诊断开始，逐步深入，最终找到根本原因。不要害怕切换 Agent 类型，灵活性是解决问题的关键。
