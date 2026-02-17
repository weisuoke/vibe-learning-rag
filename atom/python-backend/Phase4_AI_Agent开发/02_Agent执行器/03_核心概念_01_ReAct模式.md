# Agent执行器 - 核心概念1：ReAct模式

> 深入理解 ReAct 模式的推理-行动循环

---

## 什么是 ReAct 模式？

**ReAct = Reasoning（推理）+ Acting（行动）**

ReAct 是一种让 LLM 能够"边思考边行动"的模式，通过 **Thought（思考）→ Action（行动）→ Observation（观察）** 的循环，让 AI Agent 能够自主决策并完成复杂任务。

**论文来源**：ReAct: Synergizing Reasoning and Acting in Language Models (2022)

---

## ReAct 模式的核心流程

### 完整的 ReAct 循环

```
用户问题："我的订单12345什么时候到？"
   ↓
┌─────────────────────────────────────────┐
│  ReAct 循环                              │
│                                         │
│  1. Thought（思考）                      │
│     LLM: "我需要查询订单信息"            │
│                                         │
│  2. Action（行动）                       │
│     选择工具: query_order               │
│     参数: "12345"                       │
│                                         │
│  3. Observation（观察）                  │
│     工具返回: "订单已发货，预计明天到达"  │
│                                         │
│  4. 判断是否完成                         │
│     LLM: "已获取信息，可以回答用户"      │
│                                         │
│  5. Final Answer                        │
│     "您的订单已发货，预计明天到达"        │
└─────────────────────────────────────────┘
```

---

## 三个核心步骤详解

### 步骤1：Thought（思考）

**定义**：LLM 推理"下一步应该做什么"。

**LLM 的思考过程**：
```
当前信息：
- 用户问题："我的订单12345什么时候到？"
- 可用工具：query_order, query_inventory, calculate

思考：
- 用户问的是订单状态
- 需要查询订单信息
- 应该使用 query_order 工具
- 参数是订单ID "12345"
```

**Thought 的输出格式**：
```
Thought: 我需要查询订单信息来回答用户的问题
```

---

### 步骤2：Action（行动）

**定义**：根据 Thought 选择并调用工具。

**Action 包含两部分**：
1. **Action**：工具名称
2. **Action Input**：工具参数

**Action 的输出格式**：
```
Action: query_order
Action Input: 12345
```

**系统的处理**：
```python
# 1. 解析 Action
action_name = "query_order"
action_input = "12345"

# 2. 找到对应的工具
tool = find_tool(tools, action_name)

# 3. 执行工具
observation = tool.run(action_input)
```

---

### 步骤3：Observation（观察）

**定义**：观察工具执行的结果。

**Observation 的输出格式**：
```
Observation: 订单12345已发货，预计明天到达
```

**系统的处理**：
```python
# 1. 获取工具执行结果
observation = "订单12345已发货，预计明天到达"

# 2. 将结果添加到上下文
context.append(f"Observation: {observation}")

# 3. 继续下一轮 Thought
```

---

## 手写 ReAct 循环实现

### 最简实现（不使用 LangChain）

```python
"""
手写 ReAct 循环
演示 ReAct 模式的核心逻辑
"""

from openai import OpenAI
import re

# 初始化 LLM
client = OpenAI()

# 定义工具
def query_order(order_id: str) -> str:
    """查询订单信息"""
    # 模拟数据库查询
    orders = {
        "12345": "订单已发货，预计明天到达",
        "67890": "订单已签收",
    }
    return orders.get(order_id, "订单不存在")

def query_inventory(product_id: str) -> str:
    """查询库存信息"""
    inventory = {
        "A001": "库存100件",
        "A002": "库存50件",
    }
    return inventory.get(product_id, "产品不存在")

# 工具映射
tools = {
    "query_order": query_order,
    "query_inventory": query_inventory,
}

# 工具描述（给 LLM 看）
tools_description = """
可用工具：
1. query_order: 查询订单信息，输入订单ID，返回订单状态
2. query_inventory: 查询库存信息，输入产品ID，返回库存数量
"""

def parse_action(text: str) -> tuple:
    """解析 LLM 输出的 Action"""
    # 匹配 "Action: tool_name" 和 "Action Input: input"
    action_match = re.search(r"Action:\s*(\w+)", text)
    input_match = re.search(r"Action Input:\s*(.+)", text)

    if action_match and input_match:
        action = action_match.group(1)
        action_input = input_match.group(1).strip()
        return action, action_input
    return None, None

def react_loop(question: str, max_iterations: int = 10):
    """ReAct 循环的核心实现"""

    # 初始化上下文
    context = [f"Question: {question}"]

    # 循环执行
    for i in range(max_iterations):
        print(f"\n{'='*50}")
        print(f"迭代 {i+1}")
        print('='*50)

        # 构建 Prompt
        prompt = f"""
你是一个智能助手，可以使用工具来回答问题。

{tools_description}

请按照以下格式思考和行动：

Thought: 你应该思考下一步做什么
Action: 要执行的操作（工具名称）
Action Input: 操作的输入（工具参数）
Observation: 操作的结果（系统会填充）
... (这个过程可以重复多次)
Thought: 我现在知道最终答案了
Final Answer: 对原始问题的最终答案

当前对话历史：
{chr(10).join(context)}

请继续思考：
"""

        # 1. Thought: LLM 思考
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        thought = response.choices[0].message.content
        print(f"\nLLM 输出:\n{thought}")

        # 2. 判断是否完成
        if "Final Answer:" in thought:
            final_answer = thought.split("Final Answer:")[1].strip()
            print(f"\n{'='*50}")
            print(f"最终答案: {final_answer}")
            print('='*50)
            return final_answer

        # 3. Action: 解析并执行工具
        action_name, action_input = parse_action(thought)

        if action_name is None:
            print("无法解析 Action，停止执行")
            return "执行失败：无法解析 Action"

        print(f"\nAction: {action_name}")
        print(f"Action Input: {action_input}")

        # 4. 执行工具
        if action_name not in tools:
            observation = f"错误：工具 {action_name} 不存在"
        else:
            tool = tools[action_name]
            observation = tool(action_input)

        print(f"Observation: {observation}")

        # 5. 更新上下文
        context.append(f"Thought: {thought}")
        context.append(f"Observation: {observation}")

    return "达到最大迭代次数"

# 测试
if __name__ == "__main__":
    question = "订单12345什么时候到？"
    answer = react_loop(question)
    print(f"\n最终结果: {answer}")
```

**运行输出示例**：

```
==================================================
迭代 1
==================================================

LLM 输出:
Thought: 用户询问订单12345的到达时间，我需要查询订单信息
Action: query_order
Action Input: 12345

Action: query_order
Action Input: 12345
Observation: 订单已发货，预计明天到达

==================================================
迭代 2
==================================================

LLM 输出:
Thought: 我已经获取到订单信息，可以回答用户了
Final Answer: 您的订单12345已发货，预计明天到达

==================================================
最终答案: 您的订单12345已发货，预计明天到达
==================================================
```

---

## 使用 LangChain 实现 ReAct

### 标准实现

```python
"""
使用 LangChain 实现 ReAct Agent
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.prompts import PromptTemplate

# 1. 定义工具（使用 @tool 装饰器）
@tool
def query_order(order_id: str) -> str:
    """查询订单信息，输入订单ID，返回订单状态"""
    orders = {
        "12345": "订单已发货，预计明天到达",
        "67890": "订单已签收",
    }
    return orders.get(order_id, "订单不存在")

@tool
def query_inventory(product_id: str) -> str:
    """查询库存信息，输入产品ID，返回库存数量"""
    inventory = {
        "A001": "库存100件",
        "A002": "库存50件",
    }
    return inventory.get(product_id, "产品不存在")

# 2. 创建 LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 3. 定义 ReAct Prompt
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

# 4. 创建 ReAct Agent
tools = [query_order, query_inventory]
agent = create_react_agent(llm, tools, prompt)

# 5. 创建 AgentExecutor
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
)

# 6. 运行 Agent
result = executor.invoke({"input": "订单12345什么时候到？"})
print(f"\n最终答案: {result['output']}")
```

---

## ReAct 模式的关键要素

### 1. Prompt 设计

**好的 ReAct Prompt 应该包含**：

```python
prompt = """
1. 角色定义：你是一个智能助手

2. 工具列表：
{tools}

3. 格式说明：
Question: 用户问题
Thought: 你的思考
Action: 工具名称
Action Input: 工具参数
Observation: 工具结果
... (可以重复)
Thought: 我知道答案了
Final Answer: 最终答案

4. 示例（可选）：
Question: 订单12345什么时候到？
Thought: 需要查询订单信息
Action: query_order
Action Input: 12345
Observation: 订单已发货
Thought: 已获取信息
Final Answer: 您的订单已发货

5. 开始：
Question: {input}
Thought: {agent_scratchpad}
"""
```

---

### 2. 工具描述

**工具描述的重要性**：
- LLM 根据描述选择工具
- 描述越清晰，选择越准确

**好的工具描述**：

```python
@tool
def query_order(order_id: str) -> str:
    """
    查询订单信息

    功能：根据订单ID查询订单状态和物流信息
    输入：订单ID（字符串，如 '12345'）
    输出：订单状态和预计到达时间

    适用场景：
    - 用户询问"订单什么时候到"
    - 用户询问"订单状态"
    - 用户提供订单ID

    示例：
    输入：'12345'
    输出：'订单已发货，预计明天到达'
    """
    # 实现...
```

---

### 3. 停止条件

**三种停止条件**：

```python
# 1. LLM 返回 Final Answer
if "Final Answer:" in thought:
    return extract_final_answer(thought)

# 2. 达到最大迭代次数
if iterations >= max_iterations:
    return "达到最大迭代次数"

# 3. 遇到错误
try:
    observation = execute_tool(action)
except Exception as e:
    return f"执行失败：{str(e)}"
```

---

## ReAct 模式的优势

### 1. 可解释性

**传统 LLM**：
```
输入：订单12345什么时候到？
输出：您的订单已发货，预计明天到达
```
问题：不知道 LLM 是怎么得出这个答案的（可能是编造的）

**ReAct 模式**：
```
输入：订单12345什么时候到？

Thought: 需要查询订单信息
Action: query_order("12345")
Observation: 订单已发货，预计明天到达
Thought: 已获取信息
Final Answer: 您的订单已发货，预计明天到达
```
优势：可以看到完整的推理和执行过程

---

### 2. 准确性

**传统 LLM**：
- 可能编造信息（幻觉）
- 无法访问实时数据

**ReAct 模式**：
- 调用真实的工具获取数据
- 基于实际数据生成答案
- 减少幻觉

---

### 3. 灵活性

**传统方式（硬编码）**：
```python
def handle_question(question):
    if "订单" in question:
        return query_order()
    elif "库存" in question:
        return query_inventory()
    else:
        return "无法处理"
```
问题：无法处理复杂、多步骤的任务

**ReAct 模式**：
```python
agent = create_react_agent(llm, tools, prompt)
result = agent.run(question)  # 自动决策
```
优势：可以处理复杂、多步骤的任务

---

## ReAct 模式的局限性

### 1. 成本高

**问题**：多次调用 LLM，成本是普通调用的数倍。

**示例**：
```
简单问题："今天天气怎么样？"

普通 LLM：1次调用
ReAct Agent：3-5次调用（Thought → Action → Observation → Thought → Final Answer）

成本：ReAct 是普通调用的 3-5 倍
```

**解决方案**：
- 简单任务用 Chain
- 设置合理的 max_iterations
- 使用更便宜的模型（GPT-3.5）

---

### 2. 响应慢

**问题**：多次迭代导致响应时间长。

**示例**：
```
普通 LLM：1-2秒
ReAct Agent：5-15秒（取决于迭代次数）
```

**解决方案**：
- 使用流式输出，实时显示执行过程
- 异步处理，不阻塞主线程
- 对于长任务，使用后台任务队列

---

### 3. 不确定性

**问题**：LLM 的输出有随机性，同样的输入可能产生不同的执行路径。

**示例**：
```
问题："上个月销售额最高的产品是什么？"

执行路径1（最优）：
1. query_sales_summary("last_month")
2. Final Answer

执行路径2（次优）：
1. query_all_products()
2. query_product_sales(product_1)
3. query_product_sales(product_2)
...
100. Final Answer
```

**解决方案**：
- 设置 temperature=0 降低随机性
- 优化 Prompt 和工具描述
- 提供高层工具而不是底层工具

---

## ReAct vs 其他模式

### ReAct vs Plan-and-Execute

| 维度 | ReAct | Plan-and-Execute |
|------|-------|------------------|
| **决策方式** | 边思考边行动 | 先规划再执行 |
| **灵活性** | 高（可以根据结果调整） | 低（计划固定） |
| **效率** | 低（可能走弯路） | 高（计划最优） |
| **适用场景** | 不确定性高的任务 | 可以预先规划的任务 |

**示例**：

```python
# ReAct: 边思考边行动
1. Thought: 需要查询数据
2. Action: query_data()
3. Observation: 获取到数据
4. Thought: 需要计算
5. Action: calculate()
6. Final Answer

# Plan-and-Execute: 先规划再执行
1. Plan: [query_data, calculate, generate_answer]
2. Execute: query_data()
3. Execute: calculate()
4. Execute: generate_answer()
5. Final Answer
```

---

## 实际应用案例

### 案例1：智能客服

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# 定义工具
@tool
def query_order(order_id: str) -> str:
    """查询订单信息"""
    # 实际查询数据库
    return "订单已发货"

@tool
def query_logistics(order_id: str) -> str:
    """查询物流信息"""
    # 实际查询物流API
    return "预计明天到达"

@tool
def cancel_order(order_id: str) -> str:
    """取消订单"""
    # 实际取消订单
    return "订单已取消"

# 创建 Agent
llm = ChatOpenAI(model="gpt-4", temperature=0)
tools = [query_order, query_logistics, cancel_order]
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# 测试不同的用户问题
questions = [
    "订单12345什么时候到？",
    "我想取消订单67890",
    "订单12345的物流信息是什么？",
]

for q in questions:
    print(f"\n用户问题：{q}")
    result = executor.invoke({"input": q})
    print(f"Agent 回答：{result['output']}")
```

**执行过程**：

```
用户问题：订单12345什么时候到？

Thought: 需要查询订单信息和物流信息
Action: query_order
Action Input: 12345
Observation: 订单已发货

Thought: 还需要查询物流信息
Action: query_logistics
Action Input: 12345
Observation: 预计明天到达

Thought: 已获取所有信息
Final Answer: 您的订单已发货，预计明天到达
```

---

### 案例2：数据分析助手

```python
@tool
def query_sales_data(month: str) -> str:
    """查询销售数据"""
    return "产品A: 50000, 产品B: 30000, 产品C: 20000"

@tool
def calculate_growth(current: float, previous: float) -> str:
    """计算增长率"""
    growth = (current - previous) / previous * 100
    return f"增长率：{growth:.2f}%"

@tool
def generate_chart(data: str) -> str:
    """生成图表"""
    return "图表已生成：sales_chart.png"

# 创建 Agent
tools = [query_sales_data, calculate_growth, generate_chart]
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# 测试
question = "上个月销售额最高的产品是什么？生成图表"
result = executor.invoke({"input": question})
```

---

## 学习检查清单

完成本文档学习后，你应该能够：

- [ ] 理解 ReAct 模式的核心流程（Thought → Action → Observation）
- [ ] 手写一个简单的 ReAct 循环
- [ ] 使用 LangChain 创建 ReAct Agent
- [ ] 设计好的 ReAct Prompt
- [ ] 编写清晰的工具描述
- [ ] 理解 ReAct 的优势和局限性
- [ ] 知道什么时候用 ReAct，什么时候用其他模式

---

**版本：** v1.0
**最后更新：** 2026-02-12
**维护者：** Claude Code
