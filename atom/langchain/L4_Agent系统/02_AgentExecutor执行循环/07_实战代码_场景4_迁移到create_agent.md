# 实战代码 场景4：迁移到 create_agent

> 从 AgentExecutor 迁移到 LangChain 1.0 的 create_agent API——同样的循环思想，更现代的实现方式

---

## 场景概述

LangChain 1.0（2025年10月发布）引入了 `create_agent()` 作为创建 Agent 的标准方式，基于 LangGraph 状态图取代了旧版 `AgentExecutor`。本文通过 Before/After 对比，演示：

- AgentExecutor 代码如何迁移到 create_agent
- 哪些概念不变，哪些 API 变了
- AgentExecutor 的参数在 create_agent 中的等价物（中间件）
- create_agent 的流式输出
- 多轮对话（Checkpointer）

核心结论：**执行循环的思想完全一致（Think→Act→Observe），变的只是 API 表面。**

---

## 环境准备

```python
# 安装依赖（只需执行一次）
# pip install langchain langchain-openai langgraph python-dotenv

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()
```

---

## 示例1：最基础的迁移——Before vs After

**解决的问题：** 看清 AgentExecutor 和 create_agent 在最简场景下的差异。

### Before：AgentExecutor

```python
"""
AgentExecutor 版本（旧版）
"""
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate


@tool
def get_weather(city: str) -> str:
    """获取城市天气信息"""
    weather_data = {"北京": "晴天 25°C", "上海": "多云 22°C"}
    return weather_data.get(city, f"未找到 {city} 的天气数据")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"计算错误: {e}"


# --- AgentExecutor 特有的部分 ---

# 1. 需要手动创建 Prompt（必须包含 agent_scratchpad）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的中文助手。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),  # 必须有这个占位符
])

# 2. 需要手动创建 LLM 实例
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. 需要先创建 agent，再包装成 executor
agent = create_tool_calling_agent(llm, [get_weather, calculate], prompt)
executor = AgentExecutor(
    agent=agent,
    tools=[get_weather, calculate],
    max_iterations=10,
    handle_parsing_errors=True,
    verbose=True,
)

# 4. 输入格式：{"input": "..."}
result = executor.invoke({"input": "北京天气怎么样？"})
print(result["output"])  # 输出格式：result["output"]
```

### After：create_agent

```python
"""
create_agent 版本（新版，LangChain 1.0+）
"""
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """获取城市天气信息"""
    weather_data = {"北京": "晴天 25°C", "上海": "多云 22°C"}
    return weather_data.get(city, f"未找到 {city} 的天气数据")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"计算错误: {e}"


# --- create_agent 的简洁写法 ---

# 1. 不需要手动创建 Prompt（system_prompt 参数直接传）
# 2. 不需要手动创建 LLM 实例（字符串格式自动创建）
# 3. 一步到位，不需要分开创建 agent 和 executor
agent = create_agent(
    "openai:gpt-4o-mini",  # 字符串格式，自动创建 LLM
    tools=[get_weather, calculate],
    system_prompt="你是一个有用的中文助手。",
)

# 4. 输入格式变了：{"messages": [...]}
result = agent.invoke({
    "messages": [{"role": "user", "content": "北京天气怎么样？"}]
})
# 输出格式变了：result["messages"][-1].content
print(result["messages"][-1].content)
```

### 变化对照表

| 方面 | AgentExecutor | create_agent |
|------|--------------|-------------|
| 导入 | `AgentExecutor, create_tool_calling_agent` | `create_agent` |
| 模型 | `ChatOpenAI(model="gpt-4o-mini")` | `"openai:gpt-4o-mini"` |
| Prompt | 手动创建 `ChatPromptTemplate`（必须含 `agent_scratchpad`） | `system_prompt="..."` 参数 |
| 创建步骤 | 先 `create_tool_calling_agent` 再 `AgentExecutor` | 一步 `create_agent` |
| 输入格式 | `{"input": "问题"}` | `{"messages": [{"role": "user", "content": "问题"}]}` |
| 输出格式 | `result["output"]` | `result["messages"][-1].content` |
| 返回类型 | `dict` | `dict`（包含完整消息列表） |
| 底层实现 | while 循环 | LangGraph StateGraph |

---

## 示例2：参数迁移——中间件替代 AgentExecutor 参数

**解决的问题：** AgentExecutor 的 `max_iterations`、`handle_parsing_errors` 等参数在 create_agent 中怎么实现？答案是中间件（middleware）。

### Before：AgentExecutor 参数

```python
"""
AgentExecutor 的参数配置
"""
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate


@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"关于 '{query}' 的搜索结果..."


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个搜索助手。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_tool_calling_agent(llm, [search], prompt)

# AgentExecutor 的参数配置
executor = AgentExecutor(
    agent=agent,
    tools=[search],
    max_iterations=10,              # 最大迭代次数
    max_execution_time=30.0,        # 最大执行时间
    handle_parsing_errors=True,     # 自动处理解析错误
    return_intermediate_steps=True, # 返回中间步骤
    verbose=True,                   # 打印执行过程
)

result = executor.invoke({"input": "什么是 RAG？"})
```

### After：create_agent 中间件

```python
"""
create_agent 的中间件配置
对应 AgentExecutor 的各个参数
"""
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ToolCallLimitMiddleware,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.tools import tool


@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"关于 '{query}' 的搜索结果..."


# create_agent 使用中间件替代 AgentExecutor 的参数
agent = create_agent(
    "openai:gpt-4o-mini",
    tools=[search],
    system_prompt="你是一个搜索助手。",
    middleware=[
        # 替代 max_iterations：限制工具调用次数
        ToolCallLimitMiddleware(
            max_tool_calls=10,          # 最多调用 10 次工具
            on_exceed="end",            # 超限时结束（类似 early_stopping_method="force"）
        ),
        # 替代 handle_parsing_errors：LLM 调用失败时自动重试
        ModelRetryMiddleware(
            max_retries=3,              # 最多重试 3 次
            delay=1.0,                  # 重试间隔 1 秒
        ),
        # 工具执行失败时自动重试（AgentExecutor 没有的新能力）
        ToolRetryMiddleware(
            max_retries=2,              # 工具失败最多重试 2 次
        ),
    ],
    debug=True,  # 替代 verbose=True
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "什么是 RAG？"}]
})
print(result["messages"][-1].content)
```

### 参数迁移速查表

```
┌─────────────────────────────────────────────────────────────────┐
│              AgentExecutor → create_agent 参数迁移               │
├──────────────────────────┬──────────────────────────────────────┤
│  AgentExecutor 参数       │  create_agent 等价物                 │
├──────────────────────────┼──────────────────────────────────────┤
│  max_iterations=10       │  ToolCallLimitMiddleware(            │
│                          │    max_tool_calls=10)                │
├──────────────────────────┼──────────────────────────────────────┤
│  handle_parsing_errors   │  ModelRetryMiddleware(               │
│    =True                 │    max_retries=3)                    │
├──────────────────────────┼──────────────────────────────────────┤
│  early_stopping_method   │  ToolCallLimitMiddleware(            │
│    ="force"              │    on_exceed="end")                  │
│    ="generate"           │    on_exceed="continue"              │
├──────────────────────────┼──────────────────────────────────────┤
│  verbose=True            │  debug=True                          │
├──────────────────────────┼──────────────────────────────────────┤
│  return_intermediate     │  默认返回完整 messages 列表           │
│    _steps=True           │  （天然包含所有中间步骤）              │
├──────────────────────────┼──────────────────────────────────────┤
│  无                      │  ToolRetryMiddleware (新增能力)       │
│  无                      │  checkpointer (持久化状态)            │
│  无                      │  interrupt_before/after (人工审批)    │
└──────────────────────────┴──────────────────────────────────────┘
```

---

## 示例3：流式输出迁移

**解决的问题：** AgentExecutor 的 `stream()` 和 create_agent 的 `stream()` 输出格式不同，如何适配？

### Before：AgentExecutor 流式输出

```python
"""
AgentExecutor 的流式输出
"""
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate


@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"关于 '{query}' 的搜索结果：RAG 是检索增强生成技术..."


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个搜索助手。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_tool_calling_agent(llm, [search], prompt)
executor = AgentExecutor(agent=agent, tools=[search])

# AgentExecutor 的 stream：按事件类型分块
for chunk in executor.stream({"input": "什么是 RAG？"}):
    if "actions" in chunk:
        for action in chunk["actions"]:
            print(f"[调用工具] {action.tool}({action.tool_input})")
    elif "steps" in chunk:
        for step in chunk["steps"]:
            print(f"[工具结果] {step.observation[:80]}...")
    elif "output" in chunk:
        print(f"[最终答案] {chunk['output']}")
```

### After：create_agent 流式输出

```python
"""
create_agent 的流式输出
基于 LangGraph 的事件流，信息更丰富
"""
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import AIMessageChunk


@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"关于 '{query}' 的搜索结果：RAG 是检索增强生成技术..."


agent = create_agent(
    "openai:gpt-4o-mini",
    tools=[search],
    system_prompt="你是一个搜索助手。",
)

# 方式1：stream() — 按节点输出完整状态更新
print("=== 方式1：stream() ===")
for chunk in agent.stream({
    "messages": [{"role": "user", "content": "什么是 RAG？"}]
}):
    # chunk 是 {node_name: state_update} 格式
    for node_name, state_update in chunk.items():
        if node_name == "agent":
            # Agent 节点输出：LLM 的决策
            msg = state_update["messages"][-1]
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"[Agent → 调用工具] {tc['name']}({tc['args']})")
            else:
                print(f"[Agent → 最终回答] {msg.content[:100]}...")
        elif node_name == "tools":
            # Tools 节点输出：工具执行结果
            msg = state_update["messages"][-1]
            print(f"[Tools → 结果] {msg.content[:80]}...")


# 方式2：stream() + stream_mode="messages" — 逐 token 输出
print("\n=== 方式2：逐 token 流式 ===")
for msg, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "什么是 RAG？"}]},
    stream_mode="messages",
):
    # 只打印 AI 的文本输出（跳过工具调用消息）
    if isinstance(msg, AIMessageChunk) and msg.content:
        print(msg.content, end="", flush=True)
print()  # 换行
```

### 流式输出对比

| 方面 | AgentExecutor.stream() | create_agent.stream() |
|------|----------------------|---------------------|
| 粒度 | 按事件类型（actions/steps/output） | 按节点（agent/tools） |
| Token 级流式 | 需要 `astream_events` | `stream_mode="messages"` |
| 信息量 | 较少 | 完整状态更新 |
| 格式 | `{"actions": [...]}` | `{"agent": {"messages": [...]}}` |

---

## 示例4：多轮对话迁移

**解决的问题：** AgentExecutor 需要手动管理对话历史，create_agent 通过 Checkpointer 自动管理。

### Before：AgentExecutor 手动管理历史

```python
"""
AgentExecutor 的多轮对话（需要手动管理）
"""
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"搜索结果：{query} 相关信息..."


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 需要在 Prompt 中手动加入 chat_history 占位符
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个搜索助手。"),
    MessagesPlaceholder(variable_name="chat_history"),  # 手动管理
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, [search], prompt)
executor = AgentExecutor(agent=agent, tools=[search])

# 需要手动维护 chat_history
from langchain_core.messages import HumanMessage, AIMessage

chat_history = []

# 第一轮
result1 = executor.invoke({
    "input": "什么是 RAG？",
    "chat_history": chat_history,
})
print(f"回答1: {result1['output']}")
# 手动追加历史
chat_history.append(HumanMessage(content="什么是 RAG？"))
chat_history.append(AIMessage(content=result1["output"]))

# 第二轮
result2 = executor.invoke({
    "input": "它有什么优势？",  # "它"指代 RAG
    "chat_history": chat_history,
})
print(f"回答2: {result2['output']}")
```

### After：create_agent 自动管理历史

```python
"""
create_agent 的多轮对话（Checkpointer 自动管理）
"""
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver


@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"搜索结果：{query} 相关信息..."


# 使用 checkpointer 自动管理对话历史
agent = create_agent(
    "openai:gpt-4o-mini",
    tools=[search],
    system_prompt="你是一个搜索助手。",
    checkpointer=MemorySaver(),  # 内存中保存对话状态
)

# 通过 thread_id 标识对话
config = {"configurable": {"thread_id": "user-123"}}

# 第一轮：不需要手动管理历史
result1 = agent.invoke(
    {"messages": [{"role": "user", "content": "什么是 RAG？"}]},
    config=config,
)
print(f"回答1: {result1['messages'][-1].content}")

# 第二轮：自动记住上下文，"它"能正确指代 RAG
result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "它有什么优势？"}]},
    config=config,
)
print(f"回答2: {result2['messages'][-1].content}")

# 不同 thread_id = 不同对话（互不干扰）
config2 = {"configurable": {"thread_id": "user-456"}}
result3 = agent.invoke(
    {"messages": [{"role": "user", "content": "你好"}]},
    config=config2,
)
print(f"新对话: {result3['messages'][-1].content}")
```

### 多轮对话对比

| 方面 | AgentExecutor | create_agent |
|------|--------------|-------------|
| 历史管理 | 手动维护 `chat_history` 列表 | `checkpointer` 自动管理 |
| Prompt 配置 | 需要 `MessagesPlaceholder("chat_history")` | 不需要，自动处理 |
| 多用户隔离 | 需要自己实现 | `thread_id` 天然隔离 |
| 持久化 | 无内置支持 | 支持 SQLite、PostgreSQL 等 |

---

## 迁移决策树

```
你的项目应该迁移吗？

  ├─ 新项目？
  │   └─ 直接用 create_agent，不要用 AgentExecutor
  │
  ├─ 已有项目，AgentExecutor 运行良好？
  │   └─ 不急着迁移，AgentExecutor 仍然可用
  │      （但不会再有新功能）
  │
  ├─ 需要多轮对话 / 状态持久化？
  │   └─ 迁移到 create_agent（Checkpointer 比手动管理好太多）
  │
  ├─ 需要人工审批（Human-in-the-loop）？
  │   └─ 必须迁移（AgentExecutor 不支持 interrupt）
  │
  └─ 需要自定义循环逻辑？
      └─ 迁移到 LangGraph 自定义图（最灵活）
```

---

## 不变的核心

迁移改变的是 API 表面，不变的是执行循环的核心思想：

```
AgentExecutor:
  while _should_continue():
    decision = agent.plan(input, intermediate_steps)
    if isinstance(decision, AgentFinish): return
    observation = tool.run(decision.tool_input)
    intermediate_steps.append(AgentStep(decision, observation))

create_agent (LangGraph):
  while True:
    ai_message = model.invoke(messages)        # agent 节点
    if not ai_message.tool_calls: return       # 条件边判断
    tool_messages = execute_tools(tool_calls)  # tools 节点
    messages.extend(tool_messages)             # 状态更新
```

两者的本质完全一样：**LLM 思考 → 执行工具 → 反馈结果 → 重复，直到 LLM 认为任务完成。**

---

## 速查卡片

```
┌──────────────────────────────────────────────────────────────┐
│         AgentExecutor → create_agent 迁移速查                 │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  创建:                                                       │
│    旧: agent = create_tool_calling_agent(llm, tools, prompt) │
│        executor = AgentExecutor(agent=agent, tools=tools)    │
│    新: agent = create_agent("openai:gpt-4o-mini",            │
│            tools=tools, system_prompt="...")                  │
│                                                              │
│  调用:                                                       │
│    旧: result = executor.invoke({"input": "问题"})           │
│        answer = result["output"]                             │
│    新: result = agent.invoke(                                │
│            {"messages": [{"role":"user","content":"问题"}]})  │
│        answer = result["messages"][-1].content               │
│                                                              │
│  参数迁移:                                                    │
│    max_iterations    → ToolCallLimitMiddleware                │
│    handle_errors     → ModelRetryMiddleware                   │
│    verbose           → debug=True                            │
│    chat_history      → checkpointer=MemorySaver()            │
│                                                              │
│  新增能力:                                                    │
│    interrupt_before/after  → 人工审批                         │
│    checkpointer            → 状态持久化                       │
│    ToolRetryMiddleware     → 工具级重试                       │
│    stream_mode="messages"  → Token 级流式                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

**上一步：** 阅读 [07_实战代码_场景3_自定义执行循环.md](./07_实战代码_场景3_自定义执行循环.md)，手写 ReAct 循环理解底层原理
**下一步：** 阅读 [08_面试必问.md](./08_面试必问.md)，准备 AgentExecutor 相关面试题
