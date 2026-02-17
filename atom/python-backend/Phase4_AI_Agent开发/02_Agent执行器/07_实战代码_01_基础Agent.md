# Agent执行器 - 实战代码1：基础Agent

> 创建最简单的 Agent，理解核心流程

---

## 学习目标

通过本文档，你将学会：
- 创建最简单的 Agent
- 定义基础工具（计算器、搜索）
- 运行 Agent 并观察输出
- 理解 Agent 的执行流程

---

## 示例1：最简单的计算器 Agent

### 完整代码

```python
"""
最简单的 Agent 示例：计算器
演示 Agent 的基本创建和使用
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.prompts import PromptTemplate

# ===== 1. 定义工具 =====
@tool
def calculator(expression: str) -> str:
    """
    计算数学表达式

    输入：数学表达式（如 '2+3*4'、'(10+5)/3'）
    输出：计算结果
    """
    try:
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

# ===== 2. 创建 LLM =====
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,  # 降低随机性
)

# ===== 3. 定义 Prompt =====
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
Thought: {agent_scratchpad}
""")

# ===== 4. 创建 Agent =====
tools = [calculator]
agent = create_react_agent(llm, tools, prompt)

# ===== 5. 创建 AgentExecutor =====
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 打印执行过程
    max_iterations=5,
)

# ===== 6. 运行 Agent =====
if __name__ == "__main__":
    # 测试问题
    questions = [
        "计算 (25 + 15) * 2",
        "100 除以 4 是多少？",
        "2的10次方是多少？",
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"问题：{question}")
        print('='*60)

        result = executor.invoke({"input": question})
        print(f"\n最终答案：{result['output']}")
```

### 运行输出

```
============================================================
问题：计算 (25 + 15) * 2
============================================================

> Entering new AgentExecutor chain...
Thought: 我需要计算这个数学表达式
Action: calculator
Action Input: (25 + 15) * 2
Observation: 计算结果：80
Thought: 我现在知道最终答案了
Final Answer: 80

> Finished chain.

最终答案：80
```

---

## 示例2：智能客服 Agent

### 完整代码

```python
"""
智能客服 Agent
演示多个工具的使用
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.prompts import PromptTemplate

# ===== 1. 定义工具 =====

@tool
def query_order(order_id: str) -> str:
    """
    查询订单信息

    输入：订单ID（如 '12345'）
    输出：订单状态和预计到达时间
    """
    # 模拟数据库查询
    orders = {
        "12345": "订单已发货，预计明天到达",
        "67890": "订单已签收",
        "11111": "订单处理中",
    }
    return orders.get(order_id, f"订单 {order_id} 不存在")

@tool
def query_inventory(product_id: str) -> str:
    """
    查询产品库存

    输入：产品ID（如 'A001'）
    输出：当前库存数量
    """
    # 模拟数据库查询
    inventory = {
        "A001": "库存100件",
        "A002": "库存50件",
        "A003": "库存0件（缺货）",
    }
    return inventory.get(product_id, f"产品 {product_id} 不存在")

@tool
def query_logistics(order_id: str) -> str:
    """
    查询物流信息

    输入：订单ID（如 '12345'）
    输出：物流公司和运单号
    """
    # 模拟物流查询
    logistics = {
        "12345": "顺丰快递，运单号：SF1234567890",
        "67890": "已签收，无需查询物流",
    }
    return logistics.get(order_id, f"订单 {order_id} 暂无物流信息")

# ===== 2. 创建 LLM =====
llm = ChatOpenAI(model="gpt-4", temperature=0)

# ===== 3. 定义 Prompt =====
prompt = PromptTemplate.from_template("""
你是一个智能客服助手，可以帮助用户查询订单、库存和物流信息。

你有以下工具可用：
{tools}

使用以下格式回答：

Question: 用户的问题
Thought: 你应该思考下一步做什么
Action: 要执行的操作，应该是 [{tool_names}] 之一
Action Input: 操作的输入
Observation: 操作的结果
... (这个 Thought/Action/Action Input/Observation 可以重复 N 次)
Thought: 我现在知道最终答案了
Final Answer: 对原始问题的最终答案

开始！

Question: {input}
Thought: {agent_scratchpad}
""")

# ===== 4. 创建 Agent =====
tools = [query_order, query_inventory, query_logistics]
agent = create_react_agent(llm, tools, prompt)

# ===== 5. 创建 AgentExecutor =====
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
)

# ===== 6. 运行 Agent =====
if __name__ == "__main__":
    questions = [
        "订单12345什么时候到？",
        "产品A001还有多少库存？",
        "订单12345的物流信息是什么？",
        "订单67890的状态和物流信息",
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"用户问题：{question}")
        print('='*60)

        result = executor.invoke({"input": question})
        print(f"\nAgent 回答：{result['output']}")
```

### 运行输出

```
============================================================
用户问题：订单12345什么时候到？
============================================================

> Entering new AgentExecutor chain...
Thought: 用户询问订单12345的到达时间，我需要查询订单信息
Action: query_order
Action Input: 12345
Observation: 订单已发货，预计明天到达
Thought: 我已经获取到订单信息，可以回答用户了
Final Answer: 您的订单12345已发货，预计明天到达

> Finished chain.

Agent 回答：您的订单12345已发货，预计明天到达

============================================================
用户问题：订单67890的状态和物流信息
============================================================

> Entering new AgentExecutor chain...
Thought: 用户询问订单状态和物流信息，我需要先查询订单状态
Action: query_order
Action Input: 67890
Observation: 订单已签收
Thought: 订单已签收，我还需要查询物流信息
Action: query_logistics
Action Input: 67890
Observation: 已签收，无需查询物流
Thought: 我现在知道最终答案了
Final Answer: 您的订单67890已签收

> Finished chain.

Agent 回答：您的订单67890已签收
```

---

## 示例3：搜索助手 Agent

### 完整代码

```python
"""
搜索助手 Agent
演示搜索工具的使用
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.prompts import PromptTemplate

# ===== 1. 定义工具 =====

@tool
def search_web(query: str) -> str:
    """
    搜索网络信息

    输入：搜索关键词
    输出：搜索结果摘要
    """
    # 模拟搜索结果
    search_results = {
        "python": "Python 是一种高级编程语言，由 Guido van Rossum 于1991年创建",
        "fastapi": "FastAPI 是一个现代、快速的 Web 框架，用于构建 API",
        "langchain": "LangChain 是一个用于开发 LLM 应用的框架",
    }

    # 简单的关键词匹配
    for key, value in search_results.items():
        if key.lower() in query.lower():
            return value

    return f"未找到关于 '{query}' 的相关信息"

@tool
def search_docs(query: str) -> str:
    """
    搜索文档

    输入：搜索关键词
    输出：文档内容摘要
    """
    # 模拟文档搜索
    docs = {
        "agent": "Agent 是一个能够自主决策并调用工具的 AI 系统",
        "react": "ReAct 是 Reasoning + Acting 的缩写，是一种 Agent 模式",
        "tool": "Tool 是 Agent 可以调用的外部函数",
    }

    for key, value in docs.items():
        if key.lower() in query.lower():
            return value

    return f"未找到关于 '{query}' 的文档"

# ===== 2. 创建 LLM =====
llm = ChatOpenAI(model="gpt-4", temperature=0)

# ===== 3. 定义 Prompt =====
prompt = PromptTemplate.from_template("""
你是一个搜索助手，可以帮助用户搜索网络信息和文档。

你有以下工具可用：
{tools}

使用以下格式回答：

Question: {input}
Thought: {agent_scratchpad}
""")

# ===== 4. 创建 Agent =====
tools = [search_web, search_docs]
agent = create_react_agent(llm, tools, prompt)

# ===== 5. 创建 AgentExecutor =====
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
)

# ===== 6. 运行 Agent =====
if __name__ == "__main__":
    questions = [
        "Python 是什么？",
        "什么是 Agent？",
        "FastAPI 的特点是什么？",
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"问题：{question}")
        print('='*60)

        result = executor.invoke({"input": question})
        print(f"\n答案：{result['output']}")
```

---

## 核心要点总结

### 1. 创建 Agent 的5个步骤

```python
# 步骤1：定义工具
@tool
def tool_name(param: str) -> str:
    """工具描述"""
    return "结果"

# 步骤2：创建 LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 步骤3：定义 Prompt
prompt = PromptTemplate.from_template("...")

# 步骤4：创建 Agent
agent = create_react_agent(llm, tools, prompt)

# 步骤5：创建 AgentExecutor
executor = AgentExecutor(agent=agent, tools=tools)
```

---

### 2. 关键参数

```python
executor = AgentExecutor(
    agent=agent,              # Agent 实例
    tools=tools,              # 工具列表
    verbose=True,             # 打印执行过程（调试必备）
    max_iterations=10,        # 最大迭代次数
    early_stopping_method="generate",  # 早停策略
    handle_parsing_errors=True,        # 处理解析错误
)
```

---

### 3. 运行 Agent

```python
# 同步调用
result = executor.invoke({"input": "问题"})
print(result["output"])

# 异步调用
result = await executor.ainvoke({"input": "问题"})
print(result["output"])
```

---

## 常见问题

### Q1: Agent 一直不返回 Final Answer 怎么办？

**A**: 设置 `max_iterations` 限制迭代次数。

```python
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,  # 最多10次迭代
)
```

---

### Q2: 如何查看 Agent 的执行过程？

**A**: 设置 `verbose=True`。

```python
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 打印执行过程
)
```

---

### Q3: Agent 选择了错误的工具怎么办？

**A**: 优化工具描述。

```python
@tool
def query_order(order_id: str) -> str:
    """
    查询订单信息（详细描述）

    输入：订单ID
    输出：订单状态
    适用场景：用户询问订单状态、物流信息时使用
    """
    return "结果"
```

---

### Q4: 如何处理 Agent 执行失败？

**A**: 使用 try-except 捕获异常。

```python
try:
    result = executor.invoke({"input": "问题"})
    print(result["output"])
except Exception as e:
    print(f"Agent 执行失败：{str(e)}")
```

---

## 实践练习

### 练习1：创建天气查询 Agent

**要求**：
- 定义 `query_weather` 工具
- 支持查询不同城市的天气
- 返回温度、天气状况

**提示**：
```python
@tool
def query_weather(city: str) -> str:
    """查询天气信息"""
    # 实现查询逻辑
    return f"{city} 的天气：晴天，温度25°C"
```

---

### 练习2：创建数据分析 Agent

**要求**：
- 定义 `query_data` 工具（查询数据）
- 定义 `calculate_stats` 工具（计算统计）
- 支持查询和分析数据

---

### 练习3：创建文件管理 Agent

**要求**：
- 定义 `list_files` 工具（列出文件）
- 定义 `read_file` 工具（读取文件）
- 定义 `search_file` 工具（搜索文件）

---

## 学习检查清单

完成本文档学习后，你应该能够：

- [ ] 创建最简单的 Agent
- [ ] 定义基础工具（使用 @tool 装饰器）
- [ ] 配置 AgentExecutor 的参数
- [ ] 运行 Agent 并观察输出
- [ ] 理解 Agent 的执行流程（Thought → Action → Observation）
- [ ] 处理常见问题（工具选择错误、执行失败等）
- [ ] 完成3个实践练习

---

## 下一步学习

- **手写 ReAct 实现**：`07_实战代码_02_ReAct实现.md`
- **自定义工具**：`07_实战代码_03_自定义工具.md`
- **执行控制**：`07_实战代码_04_执行控制.md`

---

**版本：** v1.0
**最后更新：** 2026-02-12
**维护者：** Claude Code
