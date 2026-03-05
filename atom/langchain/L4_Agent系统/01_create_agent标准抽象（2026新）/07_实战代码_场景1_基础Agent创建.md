# 实战代码 场景1：基础 Agent 创建与工具调用

> 从零开始创建一个具有工具调用能力的 Agent，演示 create_agent 的基础用法——三行代码到多工具组合，一步步来

---

## 场景概述

`create_agent` 是 LangChain 1.0 的核心 API，取代了旧版的 `AgentExecutor`、`create_react_agent`、`create_openai_functions_agent` 等多种函数。本文从最简单的"无工具纯对话"开始，逐步演示：

- 最简 Agent（3 行代码）
- 定义工具并绑定到 Agent
- 多工具组合与自动选择
- 流式输出（打字机效果）
- 多轮对话（Checkpointer）
- 异步调用（FastAPI 集成）

每个示例都是完整可运行的 Python 代码。

---

## 环境准备

```python
# 安装依赖（只需执行一次）
# pip install langchain langchain-openai langgraph python-dotenv

# 加载环境变量（需要 .env 文件中配置 OPENAI_API_KEY）
from dotenv import load_dotenv
load_dotenv()
```

---

## 示例1：最简 Agent（无工具，3 行代码）

**解决的问题：** 快速验证 create_agent 能跑通，理解最基本的输入输出格式。

```python
"""
最简 Agent 示例
演示：create_agent 的最小用法——只有模型，没有工具
"""
from langchain.agents import create_agent

# 创建最简 Agent：字符串格式模型标识符 + 空工具列表
agent = create_agent(
    "openai:gpt-4o",
    tools=[],
    system_prompt="你是一个友好的中文助手。回答简洁明了。",
)

# 输入格式：{"messages": [{"role": "user", "content": "..."}]}
# 注意：不是旧版的 {"input": "..."}
result = agent.invoke({
    "messages": [{"role": "user", "content": "用一句话解释什么是 RAG"}]
})

# 输出：result["messages"][-1] 是 AIMessage 对象
ai_reply = result["messages"][-1]
print(f"AI 回复: {ai_reply.content}")
print(f"消息类型: {type(ai_reply).__name__}")  # AIMessage
```

### 预期输出

```
AI 回复: RAG（检索增强生成）是让大语言模型先从外部知识库检索相关信息，再基于检索结果生成回答的技术。
消息类型: AIMessage
```

> **关键记忆点：** `create_agent()` 返回的是 `CompiledStateGraph`（LangGraph 编译图），
> 拥有 `.invoke()` / `.stream()` / `.ainvoke()` / `.astream()` 四个执行方法。
> 模型参数支持字符串（`"openai:gpt-4o"`）和实例（`ChatOpenAI(model="gpt-4o")`）两种格式。

---

## 示例2：单工具 Agent（函数即工具）

**解决的问题：** 理解如何定义工具、工具三要素（类型注解 + docstring + 返回值），以及 Agent 如何自动决定是否调用工具。

```python
"""
单工具 Agent 示例
演示：普通 Python 函数作为工具，Agent 自动决定何时调用
"""
import json
from langchain.agents import create_agent


# ===== 定义工具（普通 Python 函数，不需要装饰器） =====
# LLM 通过阅读 docstring 来决定"什么时候该调用这个工具"

def get_weather(city: str) -> str:
    """获取指定城市的实时天气信息。

    Args:
        city: 城市名称，如"北京"、"上海"、"深圳"。
    """
    weather_data = {
        "北京": {"temp": 15, "condition": "晴", "humidity": 45, "wind": "北风3级"},
        "上海": {"temp": 20, "condition": "多云", "humidity": 65, "wind": "东风2级"},
        "深圳": {"temp": 25, "condition": "阵雨", "humidity": 80, "wind": "南风4级"},
    }
    data = weather_data.get(city)
    if data:
        return json.dumps(data, ensure_ascii=False)
    return json.dumps({"error": f"暂无 {city} 的天气数据"}, ensure_ascii=False)


# ===== 创建带工具的 Agent =====
agent = create_agent(
    "openai:gpt-4o",
    tools=[get_weather],
    system_prompt="你是一个天气助手。查询天气时使用工具获取数据，然后用自然语言回答。",
)

# 需要工具的问题 → Agent 自动调用 get_weather
result = agent.invoke({"messages": [{"role": "user", "content": "北京今天天气怎么样？"}]})
print("需要工具:", result["messages"][-1].content)

# 不需要工具的问题 → Agent 直接回答
result = agent.invoke({"messages": [{"role": "user", "content": "你好，你能做什么？"}]})
print("不需要工具:", result["messages"][-1].content)
```

### 工具定义三要素

| 要素 | 作用 | 缺少会怎样 |
|------|------|-----------|
| **类型注解** `city: str` | 告诉 LLM 参数类型和名称 | LLM 不知道传什么参数 |
| **docstring** | 告诉 LLM 工具用途和使用场景 | LLM 不知道何时该调用 |
| **返回值** `-> str` | 工具执行结果反馈给 LLM | LLM 拿不到工具结果 |

> **docstring 是工具选择的关键。** LLM 靠它决定"什么时候该调用这个工具"，写得越清晰，选择越准确。

---

## 示例3：多工具 Agent（自动选择 + 组合调用）

**解决的问题：** Agent 面对多个工具时如何自动选择正确的工具，以及如何在一次对话中组合调用多个工具。

```python
"""
多工具 Agent 示例
演示：多工具注册 + LLM 自动选择 + 组合调用
"""
import json
from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """获取指定城市的实时天气信息。

    Args:
        city: 城市名称，如"北京"、"上海"。
    """
    data = {"北京": {"temp": 15, "condition": "晴"}, "上海": {"temp": 20, "condition": "多云"},
            "深圳": {"temp": 25, "condition": "阵雨"}}
    return json.dumps(data.get(city, {"temp": 22, "condition": "未知"}), ensure_ascii=False)


def calculate(expression: str) -> str:
    """计算数学表达式，支持加减乘除和常用数学函数。

    Args:
        expression: 合法的数学表达式，如 '(15 + 20) / 2' 或 'abs(-5)'。
    """
    try:
        allowed = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"计算错误: {e}"


def search_knowledge(query: str) -> str:
    """搜索技术知识库，获取编程和技术相关信息。

    Args:
        query: 搜索关键词，如 'RAG 架构' 或 'Python 异步编程'。
    """
    knowledge = {
        "RAG": "RAG 通过检索外部知识增强 LLM 生成能力，核心流程：文档分块→向量化→检索→注入上下文→生成回答。",
        "Embedding": "Embedding 将文本映射到高维向量空间，语义相近的文本距离更近，是 RAG 检索的基础。",
        "LangChain": "LangChain 是 LLM 应用开发框架，1.0 版本引入 create_agent 和 Middleware 作为核心抽象。",
    }
    for key, value in knowledge.items():
        if key.lower() in query.lower():
            return value
    return f"未找到关于 '{query}' 的信息"


# ===== 创建多工具 Agent =====
agent = create_agent(
    "openai:gpt-4o",
    tools=[get_weather, calculate, search_knowledge],
    system_prompt="你是一个全能助手，可以查天气、做计算、搜知识。根据用户问题自动选择工具。用中文回答。",
)

# 测试1：天气查询 → 自动选择 get_weather
result = agent.invoke({"messages": [{"role": "user", "content": "深圳天气如何？"}]})
print("【天气】", result["messages"][-1].content)

# 测试2：数学计算 → 自动选择 calculate
result = agent.invoke({"messages": [{"role": "user", "content": "计算 (100 * 1.15 - 50) / 2 的结果"}]})
print("【计算】", result["messages"][-1].content)

# 测试3：知识搜索 → 自动选择 search_knowledge
result = agent.invoke({"messages": [{"role": "user", "content": "什么是 RAG？"}]})
print("【知识】", result["messages"][-1].content)

# 测试4：组合调用 → 先查天气 x2，再做计算
result = agent.invoke({"messages": [{"role": "user", "content": "北京和上海的温度差是多少度？"}]})
print("【组合】", result["messages"][-1].content)
```

### 预期输出

```
【天气】 深圳今天有阵雨，气温25°C。建议出门带伞。
【计算】 (100 × 1.15 - 50) ÷ 2 = 32.5
【知识】 RAG 通过检索外部知识来增强大语言模型生成能力，核心流程是：文档分块→向量化→检索→注入上下文→生成回答。
【组合】 北京气温15°C，上海气温20°C，温度差为5°C。上海比北京暖和一些。
```

### 组合调用流程

```
用户: "北京和上海的温度差是多少度？"
  → Agent 推理: 需要查两个城市天气，再做减法
  → 调用 get_weather("北京") → {"temp": 15}
  → 调用 get_weather("上海") → {"temp": 20}
  → 调用 calculate("20 - 15") → "5"
  → Agent 生成: "温度差为5°C"
```

---

## 示例4：流式输出（打字机效果）

**解决的问题：** `invoke()` 要等 Agent 全部执行完才返回，用户体验差。`stream()` 可以实时展示执行过程和逐 token 输出。

```python
"""
流式输出示例
演示：stream() 方法的两种常用 stream_mode
"""
import json
from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """获取指定城市的实时天气信息。

    Args:
        city: 城市名称。
    """
    data = {"北京": {"temp": 15, "condition": "晴"}, "上海": {"temp": 20, "condition": "多云"}}
    return json.dumps(data.get(city, {"temp": 22, "condition": "未知"}), ensure_ascii=False)


agent = create_agent("openai:gpt-4o", tools=[get_weather], system_prompt="你是天气助手，用中文简洁回答。")


# ===== 模式1：stream_mode="updates"（最常用） =====
# 每个节点执行完毕后输出一次，能看到 Agent 的完整执行过程
print("=== stream_mode='updates' ===")
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "北京天气如何？"}]},
    stream_mode="updates",
):
    for node_name, node_output in chunk.items():
        last_msg = node_output["messages"][-1]
        if node_name == "agent":
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                print(f"  [{node_name}] 调用工具: {[tc['name'] for tc in last_msg.tool_calls]}")
            else:
                print(f"  [{node_name}] 回答: {last_msg.content[:80]}")
        elif node_name == "tools":
            print(f"  [{node_name}] 返回: {last_msg.content[:80]}")


# ===== 模式2：stream_mode="messages"（逐 token 流式） =====
# 适合聊天界面的打字机效果
print("\n=== stream_mode='messages' ===")
for message, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "上海天气如何？"}]},
    stream_mode="messages",
):
    if hasattr(message, "content") and message.content:
        if metadata.get("langgraph_node") == "agent":
            print(message.content, end="", flush=True)
print()
```

### 预期输出

```
=== stream_mode='updates' ===
  [agent] 调用工具: ['get_weather']
  [tools] 返回: {"temp": 15, "condition": "晴"}
  [agent] 回答: 北京今天天气晴，气温15°C。

=== stream_mode='messages' ===
上海今天多云，气温20°C，湿度65%。
```

### stream_mode 选择指南

| stream_mode | 返回内容 | 适合场景 |
|-------------|----------|----------|
| `"updates"` | 每个节点的增量更新 | 展示 Agent 执行过程（调试、日志） |
| `"messages"` | token 级流式消息 | 聊天界面逐字显示（打字机效果） |
| `"values"` | 每步的完整状态快照 | 调试、状态追踪 |

---

## 示例5：多轮对话（Checkpointer 记忆）

**解决的问题：** 默认情况下 Agent 无状态，每次调用都是全新对话。使用 `checkpointer` 让 Agent 记住之前的对话内容。

```python
"""
多轮对话示例
演示：checkpointer 持久化对话上下文，thread_id 隔离不同会话
"""
import json
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


def get_weather(city: str) -> str:
    """获取指定城市的实时天气信息。

    Args:
        city: 城市名称。
    """
    data = {"北京": {"temp": 15, "condition": "晴"}, "上海": {"temp": 20, "condition": "多云"}}
    return json.dumps(data.get(city, {"temp": 22, "condition": "未知"}), ensure_ascii=False)


def calculate(expression: str) -> str:
    """计算数学表达式。

    Args:
        expression: 合法的数学表达式。
    """
    try:
        return str(eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round}))
    except Exception as e:
        return f"计算错误: {e}"


# ===== 创建带记忆的 Agent =====
agent = create_agent(
    "openai:gpt-4o",
    tools=[get_weather, calculate],
    system_prompt="你是一个智能助手。记住用户告诉你的信息，在后续对话中使用。用中文回答。",
    checkpointer=InMemorySaver(),  # 关键：启用对话记忆
)

# ===== 多轮对话（同一个 thread_id 共享历史） =====
config = {"configurable": {"thread_id": "conversation-001"}}

# 第一轮：用户告诉 Agent 自己住在哪
result = agent.invoke({"messages": [{"role": "user", "content": "我住在北京"}]}, config)
print(f"第一轮: {result['messages'][-1].content}")

# 第二轮：Agent 记住了用户住在北京，自动查北京天气
result = agent.invoke({"messages": [{"role": "user", "content": "我所在城市的天气怎么样？"}]}, config)
print(f"第二轮: {result['messages'][-1].content}")

# 第三轮：基于前面的天气数据做计算
result = agent.invoke({"messages": [{"role": "user", "content": "把温度转换成华氏度（F = C * 9/5 + 32）"}]}, config)
print(f"第三轮: {result['messages'][-1].content}")

# ===== 不同 thread_id = 全新对话（互相隔离） =====
config_new = {"configurable": {"thread_id": "conversation-002"}}
result = agent.invoke({"messages": [{"role": "user", "content": "我所在城市的天气怎么样？"}]}, config_new)
print(f"新线程: {result['messages'][-1].content}")  # 不知道用户住在哪
```

### 预期输出

```
第一轮: 好的，记住了！你住在北京。有什么我可以帮你的吗？
第二轮: 北京今天天气晴，气温15°C。天气不错，适合外出。
第三轮: 15°C 转换为华氏度：15 × 9/5 + 32 = 59°F。
新线程: 请告诉我你所在的城市，我来帮你查询天气。
```

### 记忆机制图解

```
thread_id: "conversation-001"（共享历史）
  第一轮: User("我住在北京") → AI("好的，记住了")
  第二轮: User("我所在城市天气") → 工具(北京) → AI(15°C)  ← 记住了北京
  第三轮: User("转华氏度") → 工具(15*9/5+32) → AI(59°F)  ← 记住了15°C

thread_id: "conversation-002"（全新对话）
  第一轮: User("我所在城市天气") → AI("请告诉我城市")      ← 不知道
```

---

## 示例6：异步调用（适配 Web 应用）

**解决的问题：** Web 应用（FastAPI、Django）中需要异步调用 Agent，避免阻塞事件循环。

```python
"""
异步调用示例
演示：ainvoke() 和 astream() 的用法
核心：Web 应用中必须用异步版本，否则会阻塞整个服务
"""
import asyncio
import json
from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """获取指定城市的实时天气信息。

    Args:
        city: 城市名称。
    """
    data = {"北京": {"temp": 15, "condition": "晴"}, "上海": {"temp": 20, "condition": "多云"}}
    return json.dumps(data.get(city, {"temp": 22, "condition": "未知"}), ensure_ascii=False)


agent = create_agent("openai:gpt-4o", tools=[get_weather], system_prompt="你是天气助手，用中文简洁回答。")


async def main():
    # ===== 异步调用（对应同步的 invoke） =====
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "北京天气如何？"}]
    })
    print("异步结果:", result["messages"][-1].content)

    # ===== 异步流式（对应同步的 stream） =====
    print("\n异步流式:")
    async for message, metadata in agent.astream(
        {"messages": [{"role": "user", "content": "上海天气如何？"}]},
        stream_mode="messages",
    ):
        if hasattr(message, "content") and message.content:
            if metadata.get("langgraph_node") == "agent":
                print(message.content, end="", flush=True)
    print()


asyncio.run(main())
```

> **FastAPI 集成提示：** 在 FastAPI 路由中直接用 `await agent.ainvoke(...)` 和 `agent.astream(...)`，
> 配合 `StreamingResponse` 实现 SSE 流式响应。完整示例见 [07_实战代码_场景3_生产级Agent配置](./07_实战代码_场景3_生产级Agent配置.md)。

---

## 关键要点总结

### 速查卡片

```python
# 创建 Agent
from langchain.agents import create_agent
agent = create_agent("openai:gpt-4o", tools=[my_func], system_prompt="...")

# 输入输出
result = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
reply = result["messages"][-1].content

# 工具定义：函数 + 类型注解 + docstring
def my_tool(param: str) -> str:
    """工具描述（LLM 靠这个决定何时调用）。"""
    return "结果"

# 流式输出
for chunk in agent.stream(input, stream_mode="updates"):  # 或 "messages"
    ...

# 多轮对话
from langgraph.checkpoint.memory import InMemorySaver
agent = create_agent(..., checkpointer=InMemorySaver())
agent.invoke(input, config={"configurable": {"thread_id": "xxx"}})

# 异步调用
await agent.ainvoke(input)
async for msg, meta in agent.astream(input, stream_mode="messages"):
    ...
```

### 四种执行方式

| 方法 | 同步/异步 | 返回方式 | 适用场景 |
|------|----------|----------|----------|
| `invoke()` | 同步 | 一次性返回 | 脚本、测试 |
| `stream()` | 同步 | 流式迭代 | CLI 应用 |
| `ainvoke()` | 异步 | 一次性返回 | Web 后端 |
| `astream()` | 异步 | 异步流式迭代 | Web 流式响应 |

### 从旧版迁移

```python
# ❌ 旧版（已弃用）
from langchain.agents import AgentExecutor, create_react_agent
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
result = executor.invoke({"input": "你好"})

# ✅ 新版（LangChain 1.0）
from langchain.agents import create_agent
agent = create_agent("openai:gpt-4o", tools=tools, system_prompt="...")
result = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
```

---

**下一步：** 阅读 [07_实战代码_场景3_生产级Agent配置](./07_实战代码_场景3_生产级Agent配置.md) 了解 Middleware、模型回退、人工审批等生产级配置

---

[来源: reference/context7_langchain_01.md]
[来源: reference/source_create_agent_01.md]
[来源: reference/fetch_1dot0_blog_02.md]
