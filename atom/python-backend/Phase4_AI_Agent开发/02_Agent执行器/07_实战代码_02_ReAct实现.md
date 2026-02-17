# Agent执行器 - 实战代码2：ReAct实现

> 手写 ReAct 循环，深入理解核心原理

---

## 学习目标

通过本文档，你将学会：
- 手写一个完整的 ReAct 循环（不使用 LangChain）
- 理解 ReAct 的核心逻辑
- 对比手写实现与 LangChain 实现
- 掌握 ReAct 的关键细节

---

## 手写 ReAct 循环

### 完整实现

```python
"""
手写 ReAct 循环
演示 ReAct 模式的核心逻辑
"""

from openai import OpenAI
import re
from typing import Dict, Callable, Tuple, Optional

# ===== 1. 初始化 LLM =====
client = OpenAI()

# ===== 2. 定义工具 =====
def query_order(order_id: str) -> str:
    """查询订单信息"""
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

def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

# ===== 3. 工具映射 =====
tools: Dict[str, Callable] = {
    "query_order": query_order,
    "query_inventory": query_inventory,
    "calculator": calculator,
}

# ===== 4. 工具描述（给 LLM 看）=====
tools_description = """
可用工具：
1. query_order: 查询订单信息，输入订单ID，返回订单状态
2. query_inventory: 查询库存信息，输入产品ID，返回库存数量
3. calculator: 计算数学表达式，输入表达式，返回计算结果
"""

# ===== 5. 解析 Action =====
def parse_action(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    解析 LLM 输出的 Action

    输入：LLM 的输出文本
    输出：(action_name, action_input) 或 (None, None)
    """
    # 匹配 "Action: tool_name"
    action_match = re.search(r"Action:\s*(\w+)", text)
    # 匹配 "Action Input: input"
    input_match = re.search(r"Action Input:\s*(.+)", text)

    if action_match and input_match:
        action = action_match.group(1)
        action_input = input_match.group(1).strip()
        return action, action_input

    return None, None

# ===== 6. ReAct 循环核心实现 =====
def react_loop(
    question: str,
    max_iterations: int = 10,
    verbose: bool = True
) -> str:
    """
    ReAct 循环的核心实现

    参数：
    - question: 用户问题
    - max_iterations: 最大迭代次数
    - verbose: 是否打印执行过程

    返回：
    - 最终答案
    """

    # 初始化上下文
    context = [f"Question: {question}"]

    # 循环执行
    for i in range(max_iterations):
        if verbose:
            print(f"\n{'='*60}")
            print(f"迭代 {i+1}/{max_iterations}")
            print('='*60)

        # ===== 步骤1: Thought - LLM 思考 =====
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

        # 调用 LLM
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        thought = response.choices[0].message.content

        if verbose:
            print(f"\nLLM 输出:\n{thought}")

        # ===== 步骤2: 判断是否完成 =====
        if "Final Answer:" in thought:
            final_answer = thought.split("Final Answer:")[1].strip()
            if verbose:
                print(f"\n{'='*60}")
                print(f"✓ 任务完成")
                print(f"最终答案: {final_answer}")
                print('='*60)
            return final_answer

        # ===== 步骤3: Action - 解析并执行工具 =====
        action_name, action_input = parse_action(thought)

        if action_name is None:
            error_msg = "无法解析 Action，停止执行"
            if verbose:
                print(f"\n✗ {error_msg}")
            return f"执行失败：{error_msg}"

        if verbose:
            print(f"\nAction: {action_name}")
            print(f"Action Input: {action_input}")

        # ===== 步骤4: 执行工具 =====
        if action_name not in tools:
            observation = f"错误：工具 {action_name} 不存在。可用工具：{list(tools.keys())}"
        else:
            try:
                tool = tools[action_name]
                observation = tool(action_input)
            except Exception as e:
                observation = f"工具执行失败：{str(e)}"

        if verbose:
            print(f"Observation: {observation}")

        # ===== 步骤5: 更新上下文 =====
        context.append(f"Thought: {thought}")
        context.append(f"Observation: {observation}")

    # 达到最大迭代次数
    error_msg = f"达到最大迭代次数（{max_iterations}），任务未完成"
    if verbose:
        print(f"\n{'='*60}")
        print(f"✗ {error_msg}")
        print('='*60)
    return error_msg

# ===== 7. 测试 =====
if __name__ == "__main__":
    questions = [
        "订单12345什么时候到？",
        "产品A001还有多少库存？",
        "计算 (25 + 15) * 2",
    ]

    for question in questions:
        print(f"\n\n{'#'*60}")
        print(f"# 用户问题：{question}")
        print('#'*60)

        answer = react_loop(question, max_iterations=10, verbose=True)

        print(f"\n\n最终结果: {answer}")
```

### 运行输出

```
############################################################
# 用户问题：订单12345什么时候到？
############################################################

============================================================
迭代 1/10
============================================================

LLM 输出:
Thought: 用户询问订单12345的到达时间，我需要查询订单信息
Action: query_order
Action Input: 12345

Action: query_order
Action Input: 12345
Observation: 订单已发货，预计明天到达

============================================================
迭代 2/10
============================================================

LLM 输出:
Thought: 我已经获取到订单信息，可以回答用户了
Final Answer: 您的订单12345已发货，预计明天到达

============================================================
✓ 任务完成
最终答案: 您的订单12345已发货，预计明天到达
============================================================


最终结果: 您的订单12345已发货，预计明天到达
```

---

## 核心逻辑分析

### 1. 上下文管理

```python
# 初始化上下文
context = [f"Question: {question}"]

# 每次迭代后更新上下文
context.append(f"Thought: {thought}")
context.append(f"Observation: {observation}")

# 构建 Prompt 时使用上下文
prompt = f"""
当前对话历史：
{chr(10).join(context)}
"""
```

**关键点**：
- 上下文记录了所有的 Thought 和 Observation
- LLM 根据上下文决定下一步
- 上下文越长，LLM 的推理越准确

---

### 2. Action 解析

```python
def parse_action(text: str) -> Tuple[Optional[str], Optional[str]]:
    """解析 LLM 输出的 Action"""
    # 匹配 "Action: tool_name"
    action_match = re.search(r"Action:\s*(\w+)", text)
    # 匹配 "Action Input: input"
    input_match = re.search(r"Action Input:\s*(.+)", text)

    if action_match and input_match:
        action = action_match.group(1)
        action_input = input_match.group(1).strip()
        return action, action_input

    return None, None
```

**关键点**：
- 使用正则表达式解析 LLM 输出
- 提取工具名称和参数
- 如果解析失败，返回 None

---

### 3. 工具执行

```python
if action_name not in tools:
    observation = f"错误：工具 {action_name} 不存在"
else:
    try:
        tool = tools[action_name]
        observation = tool(action_input)
    except Exception as e:
        observation = f"工具执行失败：{str(e)}"
```

**关键点**：
- 检查工具是否存在
- 捕获工具执行异常
- 返回错误信息而不是抛出异常

---

### 4. 停止条件

```python
# 停止条件1：LLM 返回 Final Answer
if "Final Answer:" in thought:
    final_answer = thought.split("Final Answer:")[1].strip()
    return final_answer

# 停止条件2：达到最大迭代次数
for i in range(max_iterations):
    # ...

# 循环结束后
return f"达到最大迭代次数（{max_iterations}），任务未完成"
```

---

## 与 LangChain 实现对比

### 手写实现

```python
def react_loop(question: str, max_iterations: int = 10):
    context = [f"Question: {question}"]

    for i in range(max_iterations):
        # 1. LLM 思考
        prompt = build_prompt(context)
        thought = llm.invoke(prompt)

        # 2. 判断是否完成
        if "Final Answer:" in thought:
            return extract_final_answer(thought)

        # 3. 解析并执行工具
        action, action_input = parse_action(thought)
        observation = execute_tool(action, action_input)

        # 4. 更新上下文
        context.append(f"Thought: {thought}")
        context.append(f"Observation: {observation}")

    return "达到最大迭代次数"
```

---

### LangChain 实现

```python
from langchain.agents import AgentExecutor, create_react_agent

# 创建 Agent
agent = create_react_agent(llm, tools, prompt)

# 创建 AgentExecutor
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,
)

# 运行
result = executor.invoke({"input": question})
```

---

### 对比表

| 维度 | 手写实现 | LangChain 实现 |
|------|---------|---------------|
| **代码量** | 100+ 行 | 10 行 |
| **灵活性** | 高（完全控制） | 中（配置参数） |
| **易用性** | 低（需要理解细节） | 高（开箱即用） |
| **调试** | 容易（可以打印每一步） | 中（需要 verbose=True） |
| **错误处理** | 需要自己实现 | 内置错误处理 |
| **扩展性** | 高（可以自定义任何逻辑） | 中（受限于框架） |

---

## 进阶实现：添加更多功能

### 1. 添加中间步骤记录

```python
def react_loop_with_steps(question: str, max_iterations: int = 10):
    """ReAct 循环 + 中间步骤记录"""
    context = [f"Question: {question}"]
    intermediate_steps = []  # 记录中间步骤

    for i in range(max_iterations):
        # LLM 思考
        thought = llm.invoke(build_prompt(context))

        # 判断是否完成
        if "Final Answer:" in thought:
            return {
                "output": extract_final_answer(thought),
                "intermediate_steps": intermediate_steps,
            }

        # 解析并执行工具
        action, action_input = parse_action(thought)
        observation = execute_tool(action, action_input)

        # 记录中间步骤
        intermediate_steps.append({
            "action": action,
            "action_input": action_input,
            "observation": observation,
        })

        # 更新上下文
        context.append(f"Thought: {thought}")
        context.append(f"Observation: {observation}")

    return {
        "output": "达到最大迭代次数",
        "intermediate_steps": intermediate_steps,
    }
```

---

### 2. 添加早停策略

```python
def react_loop_with_early_stopping(
    question: str,
    max_iterations: int = 10,
    early_stopping_method: str = "generate"
):
    """ReAct 循环 + 早停策略"""
    context = [f"Question: {question}"]

    for i in range(max_iterations):
        thought = llm.invoke(build_prompt(context))

        if "Final Answer:" in thought:
            return extract_final_answer(thought)

        action, action_input = parse_action(thought)
        observation = execute_tool(action, action_input)

        context.append(f"Thought: {thought}")
        context.append(f"Observation: {observation}")

    # 达到最大迭代次数
    if early_stopping_method == "force":
        return "达到最大迭代次数，任务未完成"
    elif early_stopping_method == "generate":
        # 让 LLM 基于当前信息生成答案
        prompt = f"""
基于以下信息，生成一个答案：
{chr(10).join(context)}

Final Answer:
"""
        response = llm.invoke(prompt)
        return response
```

---

### 3. 添加错误处理

```python
def react_loop_with_error_handling(question: str, max_iterations: int = 10):
    """ReAct 循环 + 错误处理"""
    context = [f"Question: {question}"]

    for i in range(max_iterations):
        try:
            # LLM 思考
            thought = llm.invoke(build_prompt(context))

            # 判断是否完成
            if "Final Answer:" in thought:
                return extract_final_answer(thought)

            # 解析 Action
            action, action_input = parse_action(thought)
            if action is None:
                # 解析失败，返回错误信息给 LLM
                observation = "解析错误：无法识别 Action。请使用正确的格式：Action: tool_name"
                context.append(f"Observation: {observation}")
                continue

            # 执行工具
            if action not in tools:
                observation = f"错误：工具 {action} 不存在。可用工具：{list(tools.keys())}"
            else:
                try:
                    tool = tools[action]
                    observation = tool(action_input)
                except Exception as e:
                    observation = f"工具执行失败：{str(e)}"

            # 更新上下文
            context.append(f"Thought: {thought}")
            context.append(f"Observation: {observation}")

        except Exception as e:
            return f"执行失败：{str(e)}"

    return "达到最大迭代次数"
```

---

## 实际应用案例

### 案例：数据分析 Agent

```python
"""
数据分析 Agent
演示复杂的多步骤任务
"""

from openai import OpenAI
import re

client = OpenAI()

# 定义工具
def query_sales_data(month: str) -> str:
    """查询销售数据"""
    data = {
        "last_month": "产品A: 50000, 产品B: 30000, 产品C: 20000",
        "this_month": "产品A: 60000, 产品B: 35000, 产品C: 25000",
    }
    return data.get(month, "无数据")

def calculate_growth(current: str, previous: str) -> str:
    """计算增长率"""
    try:
        current_val = float(current)
        previous_val = float(previous)
        growth = (current_val - previous_val) / previous_val * 100
        return f"增长率：{growth:.2f}%"
    except Exception as e:
        return f"计算错误：{str(e)}"

def find_max(data: str) -> str:
    """找出最大值"""
    try:
        # 解析数据："产品A: 50000, 产品B: 30000"
        items = data.split(", ")
        max_product = None
        max_value = 0

        for item in items:
            product, value = item.split(": ")
            value = float(value)
            if value > max_value:
                max_value = value
                max_product = product

        return f"{max_product}: {max_value}"
    except Exception as e:
        return f"解析错误：{str(e)}"

# 工具映射
tools = {
    "query_sales_data": query_sales_data,
    "calculate_growth": calculate_growth,
    "find_max": find_max,
}

tools_description = """
可用工具：
1. query_sales_data: 查询销售数据，输入月份（'last_month' 或 'this_month'），返回销售数据
2. calculate_growth: 计算增长率，输入当前值和之前值，返回增长率
3. find_max: 找出最大值，输入数据字符串，返回最大值的产品和数值
"""

def parse_action(text: str):
    action_match = re.search(r"Action:\s*(\w+)", text)
    input_match = re.search(r"Action Input:\s*(.+)", text)
    if action_match and input_match:
        return action_match.group(1), input_match.group(1).strip()
    return None, None

def react_loop(question: str, max_iterations: int = 10):
    context = [f"Question: {question}"]

    for i in range(max_iterations):
        print(f"\n{'='*60}")
        print(f"迭代 {i+1}")
        print('='*60)

        prompt = f"""
你是一个数据分析助手。

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

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        thought = response.choices[0].message.content
        print(f"\nLLM 输出:\n{thought}")

        if "Final Answer:" in thought:
            final_answer = thought.split("Final Answer:")[1].strip()
            print(f"\n{'='*60}")
            print(f"最终答案: {final_answer}")
            print('='*60)
            return final_answer

        action, action_input = parse_action(thought)
        if action is None:
            return "解析失败"

        print(f"\nAction: {action}")
        print(f"Action Input: {action_input}")

        if action not in tools:
            observation = f"错误：工具 {action} 不存在"
        else:
            tool = tools[action]
            observation = tool(action_input)

        print(f"Observation: {observation}")

        context.append(f"Thought: {thought}")
        context.append(f"Observation: {observation}")

    return "达到最大迭代次数"

# 测试
if __name__ == "__main__":
    question = "上个月销售额最高的产品是什么？"
    answer = react_loop(question)
    print(f"\n\n最终结果: {answer}")
```

---

## 学习检查清单

完成本文档学习后，你应该能够：

- [ ] 手写一个完整的 ReAct 循环
- [ ] 理解上下文管理的重要性
- [ ] 实现 Action 解析逻辑
- [ ] 处理工具执行错误
- [ ] 实现停止条件判断
- [ ] 对比手写实现与 LangChain 实现
- [ ] 添加中间步骤记录
- [ ] 实现早停策略
- [ ] 实现完善的错误处理

---

## 下一步学习

- **自定义工具**：`07_实战代码_03_自定义工具.md`
- **执行控制**：`07_实战代码_04_执行控制.md`
- **FastAPI集成**：`07_实战代码_05_FastAPI集成.md`

---

**版本：** v1.0
**最后更新：** 2026-02-12
**维护者：** Claude Code
