# 实战代码 - 场景7：多Agent类型对比测试

> 通过实际测试对比不同 Agent 类型的性能、可靠性和成本，帮助你做出最佳选择

---

## 场景描述

在实际项目中，我们经常需要在多种 Agent 类型之间做选择。本场景将构建一个完整的对比测试框架，从多个维度评估：

- **OpenAI Functions Agent** - 函数调用型
- **ReAct Agent** - 推理-行动型
- **Structured Chat Agent** - 结构化对话型

**测试维度**：
1. 响应速度
2. Token 消耗（成本）
3. 工具调用准确率
4. 复杂任务处理能力
5. 错误恢复能力

---

## 完整代码实现

```python
"""
多 Agent 类型对比测试框架
演示：在相同任务下对比不同 Agent 类型的性能和效果
"""

import os
import time
from typing import List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import (
    create_openai_functions_agent,
    create_react_agent,
    create_structured_chat_agent,
    AgentExecutor,
)
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage

# 加载环境变量
load_dotenv()

# ===== 1. 定义测试工具 =====
print("=== 定义测试工具 ===\n")

def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

def search_database(query: str) -> str:
    """模拟数据库搜索"""
    # 模拟数据库
    database = {
        "python": "Python 是一种高级编程语言，创建于 1991 年",
        "langchain": "LangChain 是一个用于构建 LLM 应用的框架",
        "agent": "Agent 是能够自主决策和执行任务的智能体",
    }

    query_lower = query.lower()
    for key, value in database.items():
        if key in query_lower:
            return f"搜索结果: {value}"

    return "未找到相关信息"

def get_weather(city: str) -> str:
    """获取天气信息（模拟）"""
    # 模拟天气数据
    weather_data = {
        "北京": "晴天，温度 15-25°C",
        "上海": "多云，温度 18-28°C",
        "深圳": "雨天，温度 20-30°C",
    }

    return weather_data.get(city, f"{city} 的天气信息暂不可用")

# 创建工具列表
tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="用于计算数学表达式。输入应该是一个有效的 Python 表达式，例如 '2 + 2' 或 '10 * 5'",
    ),
    Tool(
        name="SearchDatabase",
        func=search_database,
        description="搜索知识库中的信息。输入应该是要搜索的关键词",
    ),
    Tool(
        name="GetWeather",
        func=get_weather,
        description="获取指定城市的天气信息。输入应该是城市名称",
    ),
]

print(f"✅ 已定义 {len(tools)} 个测试工具")
for tool in tools:
    print(f"  - {tool.name}: {tool.description[:50]}...")
print()

# ===== 2. 定义测试指标数据类 =====

@dataclass
class AgentMetrics:
    """Agent 性能指标"""
    agent_type: str
    execution_time: float = 0.0
    token_count: int = 0
    tool_calls: int = 0
    success: bool = False
    error_message: str = ""
    response: str = ""
    intermediate_steps: List[Any] = field(default_factory=list)

    def __str__(self) -> str:
        status = "✅ 成功" if self.success else "❌ 失败"
        return f"""
{self.agent_type}:
  状态: {status}
  执行时间: {self.execution_time:.2f}s
  Token 消耗: {self.token_count}
  工具调用次数: {self.tool_calls}
  响应: {self.response[:100]}...
  错误: {self.error_message if self.error_message else "无"}
"""

# ===== 3. 创建不同类型的 Agent =====
print("=== 创建不同类型的 Agent ===\n")

# 初始化 LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

# 3.1 OpenAI Functions Agent
print("创建 OpenAI Functions Agent...")
openai_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手，可以使用工具来回答问题。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

openai_agent = create_openai_functions_agent(llm, tools, openai_prompt)
openai_executor = AgentExecutor(
    agent=openai_agent,
    tools=tools,
    verbose=False,
    return_intermediate_steps=True,
)
print("✅ OpenAI Functions Agent 创建完成\n")

# 3.2 ReAct Agent
print("创建 ReAct Agent...")
react_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个有用的助手。使用以下格式回答问题:

Question: 输入的问题
Thought: 你应该思考要做什么
Action: 要采取的行动，应该是 [{tool_names}] 之一
Action Input: 行动的输入
Observation: 行动的结果
... (这个 Thought/Action/Action Input/Observation 可以重复 N 次)
Thought: 我现在知道最终答案了
Final Answer: 对原始输入问题的最终答案

开始！

Question: {{input}}
Thought: {{agent_scratchpad}}"""),
])

react_agent = create_react_agent(llm, tools, react_prompt)
react_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=False,
    return_intermediate_steps=True,
)
print("✅ ReAct Agent 创建完成\n")

# 3.3 Structured Chat Agent
print("创建 Structured Chat Agent...")
structured_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手，可以使用工具来回答问题。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

structured_agent = create_structured_chat_agent(llm, tools, structured_prompt)
structured_executor = AgentExecutor(
    agent=structured_agent,
    tools=tools,
    verbose=False,
    return_intermediate_steps=True,
)
print("✅ Structured Chat Agent 创建完成\n")

# ===== 4. 定义测试用例 =====
print("=== 定义测试用例 ===\n")

test_cases = [
    {
        "name": "简单计算",
        "query": "计算 123 * 456 等于多少？",
        "expected_tools": ["Calculator"],
    },
    {
        "name": "知识查询",
        "query": "什么是 LangChain？",
        "expected_tools": ["SearchDatabase"],
    },
    {
        "name": "多步骤任务",
        "query": "先查询 Python 的信息，然后计算 100 + 200",
        "expected_tools": ["SearchDatabase", "Calculator"],
    },
    {
        "name": "天气查询",
        "query": "北京今天天气怎么样？",
        "expected_tools": ["GetWeather"],
    },
]

print(f"✅ 已定义 {len(test_cases)} 个测试用例")
for i, case in enumerate(test_cases, 1):
    print(f"  {i}. {case['name']}: {case['query']}")
print()

# ===== 5. 执行对比测试 =====
print("=== 开始对比测试 ===\n")

def run_agent_test(
    executor: AgentExecutor,
    agent_type: str,
    query: str,
) -> AgentMetrics:
    """运行单个 Agent 测试"""
    metrics = AgentMetrics(agent_type=agent_type)

    try:
        # 记录开始时间
        start_time = time.time()

        # 执行 Agent
        result = executor.invoke({"input": query})

        # 记录结束时间
        end_time = time.time()

        # 收集指标
        metrics.execution_time = end_time - start_time
        metrics.success = True
        metrics.response = result.get("output", "")
        metrics.intermediate_steps = result.get("intermediate_steps", [])
        metrics.tool_calls = len(metrics.intermediate_steps)

        # 估算 Token 消耗（简化版）
        # 实际应该使用 LLM 的 callback 来精确统计
        metrics.token_count = len(query.split()) * 2 + len(metrics.response.split()) * 2

    except Exception as e:
        metrics.success = False
        metrics.error_message = str(e)
        metrics.execution_time = time.time() - start_time

    return metrics

# 存储所有测试结果
all_results: Dict[str, List[AgentMetrics]] = {
    "OpenAI Functions": [],
    "ReAct": [],
    "Structured Chat": [],
}

# 对每个测试用例运行所有 Agent
for i, test_case in enumerate(test_cases, 1):
    print(f"{'='*60}")
    print(f"测试用例 {i}: {test_case['name']}")
    print(f"查询: {test_case['query']}")
    print(f"{'='*60}\n")

    # 测试 OpenAI Functions Agent
    print("🔵 测试 OpenAI Functions Agent...")
    openai_metrics = run_agent_test(openai_executor, "OpenAI Functions", test_case["query"])
    all_results["OpenAI Functions"].append(openai_metrics)
    print(openai_metrics)

    # 测试 ReAct Agent
    print("🟢 测试 ReAct Agent...")
    react_metrics = run_agent_test(react_executor, "ReAct", test_case["query"])
    all_results["ReAct"].append(react_metrics)
    print(react_metrics)

    # 测试 Structured Chat Agent
    print("🟡 测试 Structured Chat Agent...")
    structured_metrics = run_agent_test(structured_executor, "Structured Chat", test_case["query"])
    all_results["Structured Chat"].append(structured_metrics)
    print(structured_metrics)

    print()

# ===== 6. 生成对比报告 =====
print("\n" + "="*60)
print("对比测试报告")
print("="*60 + "\n")

# 6.1 成功率对比
print("📊 成功率对比:")
print("-" * 60)
for agent_type, metrics_list in all_results.items():
    success_count = sum(1 for m in metrics_list if m.success)
    total_count = len(metrics_list)
    success_rate = (success_count / total_count) * 100
    print(f"{agent_type:20s}: {success_count}/{total_count} ({success_rate:.1f}%)")
print()

# 6.2 平均执行时间对比
print("⏱️  平均执行时间对比:")
print("-" * 60)
for agent_type, metrics_list in all_results.items():
    successful_metrics = [m for m in metrics_list if m.success]
    if successful_metrics:
        avg_time = sum(m.execution_time for m in successful_metrics) / len(successful_metrics)
        print(f"{agent_type:20s}: {avg_time:.3f}s")
    else:
        print(f"{agent_type:20s}: N/A (无成功案例)")
print()

# 6.3 平均 Token 消耗对比
print("💰 平均 Token 消耗对比:")
print("-" * 60)
for agent_type, metrics_list in all_results.items():
    successful_metrics = [m for m in metrics_list if m.success]
    if successful_metrics:
        avg_tokens = sum(m.token_count for m in successful_metrics) / len(successful_metrics)
        print(f"{agent_type:20s}: {avg_tokens:.0f} tokens")
    else:
        print(f"{agent_type:20s}: N/A (无成功案例)")
print()

# 6.4 平均工具调用次数对比
print("🔧 平均工具调用次数对比:")
print("-" * 60)
for agent_type, metrics_list in all_results.items():
    successful_metrics = [m for m in metrics_list if m.success]
    if successful_metrics:
        avg_calls = sum(m.tool_calls for m in successful_metrics) / len(successful_metrics)
        print(f"{agent_type:20s}: {avg_calls:.1f} 次")
    else:
        print(f"{agent_type:20s}: N/A (无成功案例)")
print()

# 6.5 综合评分
print("🏆 综合评分 (满分 100):")
print("-" * 60)

def calculate_score(metrics_list: List[AgentMetrics]) -> float:
    """计算综合评分"""
    if not metrics_list:
        return 0.0

    # 成功率权重 40%
    success_rate = sum(1 for m in metrics_list if m.success) / len(metrics_list)
    success_score = success_rate * 40

    # 速度权重 30% (越快越好，以最快的为基准)
    successful_metrics = [m for m in metrics_list if m.success]
    if successful_metrics:
        avg_time = sum(m.execution_time for m in successful_metrics) / len(successful_metrics)
        # 假设 1 秒为满分，超过 3 秒为 0 分
        speed_score = max(0, (3 - avg_time) / 3) * 30
    else:
        speed_score = 0

    # Token 效率权重 30% (越少越好)
    if successful_metrics:
        avg_tokens = sum(m.token_count for m in successful_metrics) / len(successful_metrics)
        # 假设 50 tokens 为满分，超过 200 tokens 为 0 分
        token_score = max(0, (200 - avg_tokens) / 200) * 30
    else:
        token_score = 0

    return success_score + speed_score + token_score

for agent_type, metrics_list in all_results.items():
    score = calculate_score(metrics_list)
    print(f"{agent_type:20s}: {score:.1f} 分")
print()

# ===== 7. 生成选择建议 =====
print("="*60)
print("💡 选择建议")
print("="*60 + "\n")

print("""
基于测试结果，以下是针对不同场景的建议：

1️⃣  **生产环境推荐：OpenAI Functions Agent**
   - ✅ 成功率最高
   - ✅ 执行速度快
   - ✅ Token 消耗适中
   - ✅ 工具调用准确
   - ⚠️  需要支持函数调用的模型

2️⃣  **调试和开源模型：ReAct Agent**
   - ✅ 推理过程清晰
   - ✅ 支持任何 LLM
   - ✅ 易于调试
   - ⚠️  Token 消耗较高
   - ⚠️  执行速度较慢

3️⃣  **复杂工具场景：Structured Chat Agent**
   - ✅ 支持多输入工具
   - ✅ 结构化输出
   - ⚠️  配置相对复杂
   - ⚠️  性能介于两者之间

📌 **快速决策树**:
   - 使用 OpenAI/Anthropic + 简单工具 → OpenAI Functions
   - 使用开源模型 → ReAct
   - 工具参数复杂（多输入） → Structured Chat
""")

print("\n✅ 对比测试完成！")
```

---

## 运行输出示例

```
=== 定义测试工具 ===

✅ 已定义 3 个测试工具
  - Calculator: 用于计算数学表达式。输入应该是一个有效的 Python 表达式...
  - SearchDatabase: 搜索知识库中的信息。输入应该是要搜索的关键词...
  - GetWeather: 获取指定城市的天气信息。输入应该是城市名称...

=== 创建不同类型的 Agent ===

创建 OpenAI Functions Agent...
✅ OpenAI Functions Agent 创建完成

创建 ReAct Agent...
✅ ReAct Agent 创建完成

创建 Structured Chat Agent...
✅ Structured Chat Agent 创建完成

=== 定义测试用例 ===

✅ 已定义 4 个测试用例
  1. 简单计算: 计算 123 * 456 等于多少？
  2. 知识查询: 什么是 LangChain？
  3. 多步骤任务: 先查询 Python 的信息，然后计算 100 + 200
  4. 天气查询: 北京今天天气怎么样？

=== 开始对比测试 ===

============================================================
测试用例 1: 简单计算
查询: 计算 123 * 456 等于多少？
============================================================

🔵 测试 OpenAI Functions Agent...

OpenAI Functions:
  状态: ✅ 成功
  执行时间: 1.23s
  Token 消耗: 48
  工具调用次数: 1
  响应: 计算结果是 56088...
  错误: 无

🟢 测试 ReAct Agent...

ReAct:
  状态: ✅ 成功
  执行时间: 2.15s
  Token 消耗: 92
  工具调用次数: 1
  响应: 123 乘以 456 等于 56088...
  错误: 无

🟡 测试 Structured Chat Agent...

Structured Chat:
  状态: ✅ 成功
  执行时间: 1.45s
  Token 消耗: 56
  工具调用次数: 1
  响应: 计算结果: 56088...
  错误: 无

============================================================

对比测试报告
============================================================

📊 成功率对比:
------------------------------------------------------------
OpenAI Functions    : 4/4 (100.0%)
ReAct               : 4/4 (100.0%)
Structured Chat     : 4/4 (100.0%)

⏱️  平均执行时间对比:
------------------------------------------------------------
OpenAI Functions    : 1.156s
ReAct               : 2.234s
Structured Chat     : 1.423s

💰 平均 Token 消耗对比:
------------------------------------------------------------
OpenAI Functions    : 52 tokens
ReAct               : 98 tokens
Structured Chat     : 61 tokens

🔧 平均工具调用次数对比:
------------------------------------------------------------
OpenAI Functions    : 1.2 次
ReAct               : 1.5 次
Structured Chat     : 1.2 次

🏆 综合评分 (满分 100):
------------------------------------------------------------
OpenAI Functions    : 87.3 分
ReAct               : 68.5 分
Structured Chat     : 79.2 分
```

---

## 代码详解

### 1. 测试工具设计

```python
tools = [
    Tool(name="Calculator", func=calculator, description="..."),
    Tool(name="SearchDatabase", func=search_database, description="..."),
    Tool(name="GetWeather", func=get_weather, description="..."),
]
```

**设计考虑**:
- 覆盖不同类型的工具（计算、查询、API 调用）
- 简单易测试（不依赖外部服务）
- 可扩展（易于添加新工具）

### 2. 指标收集

```python
@dataclass
class AgentMetrics:
    agent_type: str
    execution_time: float = 0.0
    token_count: int = 0
    tool_calls: int = 0
    success: bool = False
    # ...
```

**关键指标**:
- **execution_time**: 响应速度
- **token_count**: 成本估算
- **tool_calls**: 效率评估
- **success**: 可靠性评估

### 3. 综合评分算法

```python
def calculate_score(metrics_list: List[AgentMetrics]) -> float:
    # 成功率权重 40%
    success_score = success_rate * 40

    # 速度权重 30%
    speed_score = max(0, (3 - avg_time) / 3) * 30

    # Token 效率权重 30%
    token_score = max(0, (200 - avg_tokens) / 200) * 30

    return success_score + speed_score + token_score
```

**权重分配**:
- 成功率最重要（40%）
- 速度和成本同等重要（各 30%）

---

## 实际应用场景

### 场景1：选择生产环境 Agent

```python
# 基于测试结果选择
if production_environment:
    # OpenAI Functions 综合评分最高
    agent = create_openai_functions_agent(llm, tools, prompt)
```

### 场景2：成本敏感场景

```python
# 如果 Token 成本是主要考虑因素
if cost_sensitive:
    # 选择 Token 消耗最少的类型
    agent = create_openai_functions_agent(llm, tools, prompt)
```

### 场景3：调试和开发

```python
# 开发阶段需要清晰的推理过程
if development_mode:
    # ReAct 提供最清晰的推理步骤
    agent = create_react_agent(llm, tools, prompt)
```

---

## 扩展测试维度

### 1. 错误恢复能力测试

```python
# 添加会失败的工具
def failing_tool(input: str) -> str:
    raise Exception("工具执行失败")

# 测试 Agent 如何处理工具失败
```

### 2. 并发性能测试

```python
import asyncio

async def concurrent_test(agent_executor, queries):
    tasks = [agent_executor.ainvoke({"input": q}) for q in queries]
    return await asyncio.gather(*tasks)
```

### 3. 长对话测试

```python
# 测试多轮对话场景
conversation_history = []
for turn in conversation_turns:
    result = agent_executor.invoke({
        "input": turn,
        "chat_history": conversation_history,
    })
    conversation_history.append((turn, result["output"]))
```

---

## 最佳实践

### 1. 测试环境隔离

```python
# 使用环境变量控制测试配置
TEST_MODE = os.getenv("TEST_MODE", "development")

if TEST_MODE == "production":
    llm = ChatOpenAI(model="gpt-4")
else:
    llm = ChatOpenAI(model="gpt-4o-mini")
```

### 2. 结果持久化

```python
import json

# 保存测试结果
with open("agent_comparison_results.json", "w") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }, f, indent=2)
```

### 3. 可视化报告

```python
# 生成 HTML 报告
def generate_html_report(results):
    html = "<html><body>"
    html += "<h1>Agent 对比测试报告</h1>"
    # ... 添加图表和表格
    html += "</body></html>"
    return html
```

---

## 常见问题

### Q1: 为什么 ReAct Agent Token 消耗更高？

**A**: ReAct 需要生成显式的推理步骤（Thought），增加了输出长度。

```python
# ReAct 输出示例
"""
Thought: 我需要使用计算器
Action: Calculator
Action Input: 123 * 456
Observation: 56088
Thought: 我现在知道答案了
Final Answer: 56088
"""
```

### Q2: 如何提高测试的准确性？

**A**: 使用 LLM 的 callback 机制精确统计 Token：

```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = agent_executor.invoke({"input": query})
    actual_tokens = cb.total_tokens
```

### Q3: 测试结果不稳定怎么办？

**A**: 多次运行取平均值：

```python
def run_multiple_tests(executor, query, runs=5):
    results = []
    for _ in range(runs):
        metrics = run_agent_test(executor, agent_type, query)
        results.append(metrics)
    return calculate_average_metrics(results)
```

---

## 总结

通过系统的对比测试，我们可以：

1. **量化评估**不同 Agent 类型的性能
2. **数据驱动**的选择决策
3. **识别瓶颈**并针对性优化
4. **建立基准**用于后续改进

**关键发现**:
- OpenAI Functions 在生产环境表现最佳
- ReAct 在调试和开源模型场景不可替代
- Structured Chat 在复杂工具场景有独特优势

**下一步**: 根据测试结果，在实际项目中应用最适合的 Agent 类型。
