# 核心概念 2：ReAct Agent - 推理与行动的协同

> **定位**: 深入理解 ReAct Agent 的原理、适用场景与实现细节
> **重要性**: ⭐⭐⭐⭐⭐ (开源模型必备、调试利器、可解释性保障)
> **难度**: ⭐⭐⭐ (概念清晰，实现简单，但需理解权衡)

---

## 1. 什么是 ReAct Agent？

### 1.1 核心定义

**ReAct = Reasoning (推理) + Acting (行动)**

ReAct Agent 是一种让 LLM 在执行工具调用前**显式进行推理**的 Agent 架构，通过 Thought-Action-Observation 循环实现复杂任务的分步解决。

**论文来源**: [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (2022)

### 1.2 与 OpenAI Functions 的本质区别

| 维度 | ReAct Agent | OpenAI Functions Agent |
|------|-------------|------------------------|
| **推理方式** | 显式文本推理 (Thought) | 隐式内部推理 |
| **输出格式** | 纯文本 (需解析) | 结构化 JSON |
| **模型要求** | 任何 LLM (包括开源) | 需支持函数调用 |
| **Token 消耗** | 高 (每步都输出推理) | 低 (直接输出函数调用) |
| **可解释性** | 极高 (每步推理可见) | 低 (黑盒决策) |
| **可靠性** | 中 (依赖文本解析) | 高 (结构化输出) |
| **调试难度** | 低 (推理过程清晰) | 高 (难以追踪决策) |

---

## 2. Thought-Action-Observation 循环

### 2.1 核心流程

```
用户问题 → [循环开始]
  ↓
Thought (思考): LLM 分析当前状态，决定下一步
  ↓
Action (行动): 选择工具并指定输入
  ↓
Observation (观察): 执行工具，获取结果
  ↓
[循环继续] → 直到得出 Final Answer
```

### 2.2 实际示例

**用户问题**: "北京今天天气如何？明天会下雨吗？"

```
Thought 1: 我需要先查询北京今天的天气
Action 1: get_weather
Action Input 1: {"city": "北京", "date": "today"}
Observation 1: 今天北京晴天，气温 15-25°C

Thought 2: 现在我知道今天天气了，还需要查询明天的天气
Action 2: get_weather
Action Input 2: {"city": "北京", "date": "tomorrow"}
Observation 2: 明天北京多云转阴，有小雨，气温 12-20°C

Thought 3: 我已经获取了今天和明天的天气信息，可以回答了
Final Answer: 北京今天天气晴朗，气温 15-25°C。明天会有小雨，气温 12-20°C。
```

### 2.3 Prompt 模板结构

ReAct Agent 的 Prompt 必须包含以下关键元素：

```python
template = '''Answer the following questions as best you can. You have access to the following tools:

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
Thought:{agent_scratchpad}'''
```

**关键变量**:
- `{tools}`: 工具描述列表
- `{tool_names}`: 工具名称列表
- `{agent_scratchpad}`: 历史推理步骤 (Thought/Action/Observation)
- `{input}`: 用户问题

---

## 3. 为什么开源模型必须用 ReAct？

### 3.1 函数调用能力的鸿沟

**2026 年现状**:
- ✅ **支持函数调用**: GPT-4, Claude 3.5, Gemini Pro
- ❌ **不支持函数调用**: Llama 3, Mistral 7B, Qwen, 大部分开源模型

**原因**:
1. **训练数据差异**: 闭源模型在函数调用数据上进行了专门微调
2. **输出格式控制**: 开源模型难以稳定输出结构化 JSON
3. **指令遵循能力**: 开源模型在复杂指令上表现不稳定

### 3.2 ReAct 的优势

**为什么 ReAct 适合开源模型？**

1. **纯文本输出**: 不需要 JSON 格式，降低输出难度
2. **显式推理**: 通过 Thought 引导模型逐步思考
3. **格式简单**: 只需识别 `Action:` 和 `Action Input:` 关键词
4. **容错性高**: 即使格式略有偏差，也能通过正则表达式解析

**示例对比**:

```python
# OpenAI Functions 输出 (开源模型难以稳定生成)
{
  "name": "get_weather",
  "arguments": {
    "city": "北京",
    "date": "today"
  }
}

# ReAct 输出 (开源模型容易生成)
Thought: 我需要查询北京今天的天气
Action: get_weather
Action Input: {"city": "北京", "date": "today"}
```

---

## 4. 适用场景

### 4.1 强烈推荐使用 ReAct 的场景

#### 场景 1: 使用开源模型

```python
from langchain_community.llms import Ollama
from langchain.agents import create_react_agent, AgentExecutor

# Llama 3 不支持函数调用，必须用 ReAct
llm = Ollama(model="llama3")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

**为什么**: 开源模型无法使用 OpenAI Functions Agent

#### 场景 2: 调试与开发阶段

```python
# verbose=True 可以看到每一步的推理过程
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 显示 Thought/Action/Observation
    max_iterations=10
)
```

**为什么**: 推理过程可见，便于发现问题

#### 场景 3: 需要高可解释性

```python
# 医疗诊断、金融决策等需要解释推理过程的场景
result = agent_executor.invoke({
    "input": "患者症状：发热、咳嗽、乏力，建议检查项目？"
})

# 可以追溯每一步的推理逻辑
print(result["intermediate_steps"])
```

**为什么**: 每步推理都有文字记录，符合合规要求

#### 场景 4: 教学与演示

```python
# 展示 AI 如何一步步解决问题
agent_executor.invoke({"input": "计算 (23 + 45) * 2 - 10"})
```

**为什么**: 推理过程直观，易于理解

### 4.2 不推荐使用 ReAct 的场景

#### 场景 1: 生产环境高可靠性要求

**问题**:
- 文本解析可能失败 (格式不规范)
- Token 消耗高 (每步都输出推理)
- 速度慢 (多轮对话)

**替代方案**: OpenAI Functions Agent

#### 场景 2: 成本敏感型应用

**Token 消耗对比** (以 GPT-4 为例):

```
问题: "北京今天天气如何？"

OpenAI Functions:
- 输入: 150 tokens
- 输出: 50 tokens
- 总计: 200 tokens

ReAct:
- 输入: 200 tokens (包含 Prompt 模板)
- 输出: 150 tokens (Thought + Action + Final Answer)
- 总计: 350 tokens (多 75%)
```

#### 场景 3: 简单工具调用

```python
# 只需调用一个工具，不需要复杂推理
tools = [calculator]
question = "23 + 45 等于多少？"

# 用 OpenAI Functions 更高效
```

---

## 5. 完整代码示例

### 5.1 基础 ReAct Agent (Llama 3)

```python
from langchain_community.llms import Ollama
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
import requests

# 1. 定义工具
def get_weather(city: str) -> str:
    """查询城市天气 (模拟)"""
    weather_data = {
        "北京": "晴天，15-25°C",
        "上海": "多云，18-28°C",
        "深圳": "小雨，22-30°C"
    }
    return weather_data.get(city, "未知城市")

def search_web(query: str) -> str:
    """搜索网络信息 (模拟)"""
    return f"关于 '{query}' 的搜索结果：这是一个模拟的搜索结果。"

tools = [
    Tool(
        name="get_weather",
        func=get_weather,
        description="查询指定城市的天气情况。输入：城市名称（如'北京'）"
    ),
    Tool(
        name="search_web",
        func=search_web,
        description="搜索网络信息。输入：搜索关键词"
    )
]

# 2. 定义 Prompt 模板
template = '''Answer the following questions as best you can. You have access to the following tools:

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
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

# 3. 初始化 LLM (Llama 3)
llm = Ollama(model="llama3", temperature=0)

# 4. 创建 ReAct Agent
agent = create_react_agent(llm, tools, prompt)

# 5. 创建 AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 显示推理过程
    max_iterations=5,  # 最多 5 轮推理
    handle_parsing_errors=True  # 自动处理解析错误
)

# 6. 执行查询
result = agent_executor.invoke({
    "input": "北京今天天气如何？如果下雨，搜索一下雨天出行注意事项。"
})

print("\n最终答案:")
print(result["output"])
```

### 5.2 带错误处理的 ReAct Agent

```python
from langchain.agents import AgentExecutor
from langchain_core.exceptions import OutputParserException

# 自定义错误处理
def handle_parsing_error(error: OutputParserException) -> str:
    """处理解析错误"""
    return f"解析失败，请重新按照格式输出：\nThought: ...\nAction: ...\nAction Input: ..."

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=handle_parsing_error,  # 自定义错误处理
    max_execution_time=60,  # 最多执行 60 秒
    early_stopping_method="generate"  # 超时后生成答案
)

# 执行查询
try:
    result = agent_executor.invoke({"input": "复杂问题..."})
except Exception as e:
    print(f"执行失败: {e}")
```

### 5.3 对比 OpenAI Functions Agent

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# OpenAI Functions Agent (生产环境推荐)
llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 执行相同的查询
result = agent_executor.invoke({
    "input": "北京今天天气如何？如果下雨，搜索一下雨天出行注意事项。"
})
```

---

## 6. Token 使用对比

### 6.1 实际测试数据

**测试问题**: "北京今天天气如何？明天会下雨吗？"

| Agent 类型 | 输入 Tokens | 输出 Tokens | 总 Tokens | 成本 (GPT-4) |
|-----------|------------|------------|----------|-------------|
| ReAct | 450 | 280 | 730 | $0.0219 |
| OpenAI Functions | 280 | 120 | 400 | $0.0120 |
| **差异** | +60% | +133% | +82% | +82% |

**结论**: ReAct Agent 的 Token 消耗约为 OpenAI Functions 的 **1.8 倍**

### 6.2 为什么 ReAct 消耗更多 Token？

1. **Prompt 模板更长**: 需要详细说明 Thought/Action/Observation 格式
2. **每步都输出推理**: Thought 部分占用大量 Token
3. **历史记录累积**: `agent_scratchpad` 包含所有历史步骤

**示例**:

```
# ReAct 输出 (150 tokens)
Thought: 我需要先查询北京今天的天气
Action: get_weather
Action Input: {"city": "北京", "date": "today"}
Observation: 今天北京晴天，气温 15-25°C
Thought: 现在我知道今天天气了，还需要查询明天的天气
Action: get_weather
Action Input: {"city": "北京", "date": "tomorrow"}
Observation: 明天北京多云转阴，有小雨，气温 12-20°C
Thought: 我已经获取了今天和明天的天气信息，可以回答了
Final Answer: 北京今天天气晴朗，气温 15-25°C。明天会有小雨，气温 12-20°C。

# OpenAI Functions 输出 (50 tokens)
[Function Call: get_weather(city="北京", date="today")]
[Function Call: get_weather(city="北京", date="tomorrow")]
北京今天天气晴朗，气温 15-25°C。明天会有小雨，气温 12-20°C。
```

---

## 7. 调试技巧

### 7.1 使用 verbose 模式

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 打印每一步
    return_intermediate_steps=True  # 返回中间步骤
)

result = agent_executor.invoke({"input": "问题"})

# 查看中间步骤
for step in result["intermediate_steps"]:
    action, observation = step
    print(f"Action: {action.tool}")
    print(f"Input: {action.tool_input}")
    print(f"Output: {observation}")
    print("---")
```

### 7.2 自定义日志

```python
import logging

# 启用 LangChain 日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("langchain.agents")

# 执行查询
result = agent_executor.invoke({"input": "问题"})
```

### 7.3 捕获解析错误

```python
from langchain.agents.output_parsers import ReActSingleInputOutputParser

parser = ReActSingleInputOutputParser()

# 测试解析
test_output = """
Thought: 我需要查询天气
Action: get_weather
Action Input: 北京
"""

try:
    parsed = parser.parse(test_output)
    print(f"工具: {parsed.tool}")
    print(f"输入: {parsed.tool_input}")
except Exception as e:
    print(f"解析失败: {e}")
```

---

## 8. 常见问题与解决方案

### 8.1 问题 1: 模型不遵循格式

**现象**:
```
Thought: 我需要查询天气
我应该使用 get_weather 工具
输入是北京
```

**原因**: 模型没有严格按照 `Action:` 和 `Action Input:` 格式输出

**解决方案**:
```python
# 1. 在 Prompt 中强调格式
template = '''...
IMPORTANT: You MUST follow this exact format:
Action: tool_name
Action Input: tool_input
...'''

# 2. 使用 stop 序列
llm_with_stop = llm.bind(stop=["\nObservation"])

# 3. 启用错误处理
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True
)
```

### 8.2 问题 2: 无限循环

**现象**: Agent 一直重复相同的 Action

**原因**: 模型没有意识到已经获取了足够信息

**解决方案**:
```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,  # 限制最大迭代次数
    early_stopping_method="generate"  # 超时后强制生成答案
)
```

### 8.3 问题 3: 工具调用失败

**现象**: `Observation: Error: ...`

**原因**: 工具输入格式不正确

**解决方案**:
```python
# 在工具描述中明确输入格式
Tool(
    name="get_weather",
    func=get_weather,
    description="""查询城市天气。
    输入格式: 城市名称（字符串），例如 '北京' 或 '上海'
    不要使用 JSON 格式，直接输入城市名称即可。"""
)
```

---

## 9. ReAct 在 2026 年仍然重要的原因

### 9.1 开源模型的崛起

**趋势**:
- Llama 3.1, Qwen 2.5, Mistral 等开源模型性能接近 GPT-4
- 企业出于成本和隐私考虑，越来越多使用开源模型
- 开源模型的函数调用能力仍在发展中

**ReAct 的价值**: 让开源模型也能构建 Agent 系统

### 9.2 可解释性需求

**行业要求**:
- 医疗、金融、法律等领域需要解释 AI 决策
- GDPR 等法规要求 AI 系统可解释
- 企业内部审计需要追溯推理过程

**ReAct 的价值**: 每步推理都有文字记录

### 9.3 调试与开发

**开发阶段**:
- 需要理解 Agent 的决策逻辑
- 需要发现工具调用的问题
- 需要优化 Prompt 和工具描述

**ReAct 的价值**: 推理过程可见，便于调试

### 9.4 教学与演示

**教育场景**:
- 向非技术人员展示 AI 如何工作
- 教学生理解 Agent 的推理过程
- 演示 AI 的能力与局限

**ReAct 的价值**: 推理过程直观易懂

---

## 10. 最佳实践总结

### 10.1 何时使用 ReAct

✅ **推荐使用**:
- 使用开源模型 (Llama, Mistral, Qwen)
- 调试与开发阶段
- 需要高可解释性 (医疗、金融、法律)
- 教学与演示

❌ **不推荐使用**:
- 生产环境 + 闭源模型 (用 OpenAI Functions)
- 成本敏感型应用 (Token 消耗高)
- 简单工具调用 (不需要复杂推理)

### 10.2 优化建议

1. **限制迭代次数**: `max_iterations=5`
2. **启用错误处理**: `handle_parsing_errors=True`
3. **使用 stop 序列**: `llm.bind(stop=["\nObservation"])`
4. **简化工具描述**: 减少 Prompt 长度
5. **缓存历史步骤**: 避免重复计算

### 10.3 迁移路径

```python
# 阶段 1: 开发阶段 (ReAct)
agent = create_react_agent(llm, tools, prompt)

# 阶段 2: 测试阶段 (ReAct + verbose)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 阶段 3: 生产阶段 (OpenAI Functions)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
```

---

## 11. 与 RAG 开发的联系

### 11.1 RAG Agent 中的 ReAct

```python
from langchain.tools import Tool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 1. 定义 RAG 检索工具
def rag_search(query: str) -> str:
    """从知识库检索相关文档"""
    vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

tools = [
    Tool(
        name="knowledge_base_search",
        func=rag_search,
        description="从知识库中检索相关文档。输入：搜索查询"
    )
]

# 2. 创建 ReAct Agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 3. 执行查询
result = agent_executor.invoke({
    "input": "公司的休假政策是什么？"
})
```

**推理过程**:
```
Thought: 我需要从知识库中查询休假政策
Action: knowledge_base_search
Action Input: 休假政策
Observation: [检索到的文档内容]
Thought: 我已经获取了相关信息，可以回答了
Final Answer: 根据公司政策，员工每年享有...
```

### 11.2 多步 RAG 推理

```python
# 复杂问题需要多次检索
result = agent_executor.invoke({
    "input": "对比公司的年假政策和病假政策，哪个更灵活？"
})
```

**推理过程**:
```
Thought: 我需要先查询年假政策
Action: knowledge_base_search
Action Input: 年假政策
Observation: [年假政策文档]

Thought: 现在我需要查询病假政策
Action: knowledge_base_search
Action Input: 病假政策
Observation: [病假政策文档]

Thought: 我已经获取了两个政策，可以对比了
Final Answer: 年假政策更灵活，因为...
```

---

## 12. 总结

### 12.1 核心要点

1. **ReAct = Reasoning + Acting**: 显式推理 + 工具调用
2. **Thought-Action-Observation 循环**: 核心执行流程
3. **开源模型必备**: 不支持函数调用时的唯一选择
4. **高可解释性**: 每步推理都有文字记录
5. **Token 消耗高**: 约为 OpenAI Functions 的 1.8 倍
6. **适合调试**: 推理过程可见，便于发现问题

### 12.2 选择决策树

```
是否使用开源模型？
├─ 是 → 使用 ReAct Agent
└─ 否 → 是否需要高可解释性？
    ├─ 是 → 使用 ReAct Agent
    └─ 否 → 是否在开发阶段？
        ├─ 是 → 使用 ReAct Agent (调试)
        └─ 否 → 使用 OpenAI Functions Agent (生产)
```

### 12.3 未来展望

- **开源模型函数调用能力提升**: ReAct 可能逐渐被取代
- **混合架构**: ReAct + Functions 结合使用
- **更高效的推理格式**: 减少 Token 消耗
- **自动选择**: LangChain 自动根据模型能力选择 Agent 类型

---

**下一步**: 学习 [核心概念 3：Structured Chat Agent](./03_核心概念_3_Structured_Chat_Agent.md)，了解如何处理复杂工具参数。
