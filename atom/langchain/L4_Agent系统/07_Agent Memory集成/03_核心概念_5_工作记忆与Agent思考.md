# 核心概念 5：工作记忆与 Agent 思考过程

> Scratchpad / Intermediate Steps / 上下文工程 — Agent 的「内心独白」

---

## 一句话定义

**工作记忆是 Agent 在单次推理过程中用来存放中间步骤、工具调用结果和思考过程的临时空间，是 Agent 能多步推理的关键。**

---

## 什么是工作记忆？

### 人类类比

```
你在做一道数学题：

  计算 (3 + 4) × 5 - 2

  你的大脑工作记忆：
    第1步: 3 + 4 = 7        ← 临时存放中间结果
    第2步: 7 × 5 = 35       ← 临时存放中间结果
    第3步: 35 - 2 = 33      ← 最终答案

  如果没有工作记忆？
    你算完 3+4=7 后就忘了，无法继续 ×5
```

### Agent 的工作记忆

```
用户问: "北京今天天气怎么样？如果下雨，推荐个室内活动"

Agent 的工作记忆：
  第1步: 调用天气 Tool → "北京今天小雨，15°C"  ← 工具调用结果
  第2步: 判断：是下雨 → 需要推荐室内活动        ← 思考过程
  第3步: 调用推荐 Tool → "博物馆、看电影..."    ← 工具调用结果
  第4步: 整合所有信息 → 给出最终回答             ← 综合推理

如果没有工作记忆？
  第3步时 Agent 已经忘了天气结果，无法做出正确推荐
```

---

## Agent Scratchpad：工作记忆的核心

### 什么是 Scratchpad？

Scratchpad（便签/草稿纸）是 Agent 的**工作记忆空间**，存放：

1. **已执行的 Tool 调用**（Action）
2. **Tool 返回的结果**（Observation）
3. **Agent 的思考过程**（Thought）

```
Agent Scratchpad 示例：

Thought: 用户问了天气和活动推荐，我需要先查天气
Action: weather_tool(city="北京")
Observation: 北京今天小雨，15°C，东风2级

Thought: 下雨了，用户需要室内活动推荐
Action: activity_tool(type="indoor", city="北京")
Observation: 推荐：国家博物馆、IMAX电影院、密室逃脱

Thought: 我已经有了天气和活动信息，可以回答了
Final Answer: 北京今天小雨，15°C。推荐室内活动：...
```

### 在 Prompt 中的位置

```python
# Agent 的 Prompt 结构
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手。"),

    # 对话历史（短期记忆）
    MessagesPlaceholder("chat_history"),

    # 用户当前输入
    ("human", "{input}"),

    # 工作记忆（Agent Scratchpad）
    MessagesPlaceholder("agent_scratchpad"),  # ← 工具调用和结果在这里
])
```

---

## 经典方案：Intermediate Steps

### AgentExecutor 的工作记忆

在经典 AgentExecutor 中，工作记忆通过 `intermediate_steps` 管理：

```python
from langchain.agents import AgentExecutor

# AgentExecutor 自动管理 intermediate_steps
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,                   # 打印中间步骤
    max_iterations=10,              # 最大推理步数
    return_intermediate_steps=True, # 返回中间步骤
)

result = executor.invoke({"input": "北京天气如何？如果下雨推荐室内活动"})

# 查看中间步骤
for step in result["intermediate_steps"]:
    action, observation = step
    print(f"工具: {action.tool}")
    print(f"输入: {action.tool_input}")
    print(f"结果: {observation}")
    print("---")
```

### Intermediate Steps 的数据结构

```python
# intermediate_steps 是一个列表
# 每个元素是 (AgentAction, str) 的元组

intermediate_steps = [
    (
        AgentAction(
            tool="weather_tool",           # 工具名
            tool_input={"city": "北京"},    # 工具输入
            log="Thought: 需要查天气\n"     # Agent 的思考
        ),
        "北京今天小雨，15°C"                # 工具返回值（Observation）
    ),
    (
        AgentAction(
            tool="activity_tool",
            tool_input={"type": "indoor"},
            log="Thought: 下雨了，推荐室内活动\n"
        ),
        "国家博物馆、IMAX电影院"
    ),
]
```

[来源: sourcecode/langchain/libs/langchain/langchain_classic/agents/agent.py]

---

## 现代方案：消息列表即工作记忆

### LangGraph 的方式

在 LangGraph / `create_agent` 中，**没有单独的 scratchpad**。工具调用和结果直接作为消息存储：

```python
# LangGraph 中的工作记忆 = 消息列表中的 Tool 相关消息

state["messages"] = [
    SystemMessage("你是一个助手"),
    HumanMessage("北京天气如何？"),

    # 工作记忆 ↓
    AIMessage(
        content="",
        tool_calls=[{
            "name": "weather_tool",
            "args": {"city": "北京"},
            "id": "call_1"
        }]
    ),
    ToolMessage(
        content="北京今天小雨，15°C",
        tool_call_id="call_1"
    ),

    AIMessage(
        content="",
        tool_calls=[{
            "name": "activity_tool",
            "args": {"type": "indoor"},
            "id": "call_2"
        }]
    ),
    ToolMessage(
        content="国家博物馆、IMAX电影院",
        tool_call_id="call_2"
    ),
    # 工作记忆 ↑

    AIMessage("北京今天小雨。推荐：国家博物馆...")  # 最终回答
]
```

### 优势

```
经典方案（Scratchpad + Intermediate Steps）:
  - 对话消息和工具调用分开管理
  - 需要 format_agent_scratchpad() 转换
  - 两套数据结构，容易混乱

现代方案（统一消息列表）:
  - 所有内容都是消息
  - 工具调用 = AIMessage with tool_calls
  - 工具结果 = ToolMessage
  - 一套数据结构，简单统一
```

---

## 工作记忆的管理挑战

### 问题：工作记忆膨胀

```
复杂任务中，Agent 可能调用很多工具：

  步骤1: 搜索文档 → 返回 2000 tokens 的结果
  步骤2: 分析数据 → 返回 1500 tokens 的表格
  步骤3: 查询数据库 → 返回 3000 tokens 的记录
  步骤4: 计算统计 → 返回 500 tokens 的结果
  ...

  工作记忆已经占了 7000+ tokens！
  再加上对话历史和 system prompt，很容易超限
```

### 解决方案 1：裁剪工具输出

```python
from langchain.agents.middleware import after_model
from langchain.messages import RemoveMessage

@after_model
def trim_tool_outputs(state, runtime):
    """裁剪过长的工具输出"""
    messages = state["messages"]
    to_update = []

    for msg in messages:
        if msg.type == "tool" and len(msg.content) > 1000:
            # 截断过长的工具输出
            truncated = msg.content[:800] + "\n... [已截断，共 " + \
                       f"{len(msg.content)} 字符]"
            # 注意：ToolMessage 不能直接修改
            # 实际操作中可以在工具调用时就限制输出
            pass

    return None
```

### 解决方案 2：工具端限制输出

```python
from langchain_core.tools import tool

@tool
def search_documents(query: str) -> str:
    """搜索文档并返回相关内容"""
    results = vector_store.similarity_search(query, k=3)

    # 在工具端就限制输出长度
    output = []
    total_chars = 0
    for doc in results:
        if total_chars + len(doc.page_content) > 2000:
            output.append(doc.page_content[:2000 - total_chars] + "...")
            break
        output.append(doc.page_content)
        total_chars += len(doc.page_content)

    return "\n---\n".join(output)
```

### 解决方案 3：选择性保留中间步骤

```python
@after_model
def clean_intermediate_steps(state, runtime):
    """只保留最近 N 步的工具调用和结果"""
    messages = state["messages"]
    max_tool_pairs = 5  # 最多保留 5 次工具调用

    # 找出所有 tool_call 和 tool_result 对
    tool_pairs = []
    for i, msg in enumerate(messages):
        if msg.type == "ai" and hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_pairs.append(i)

    # 如果工具调用次数超过限制，删除最旧的
    if len(tool_pairs) > max_tool_pairs:
        to_remove = []
        for idx in tool_pairs[:-max_tool_pairs]:
            to_remove.append(RemoveMessage(id=messages[idx].id))
            # 同时删除对应的 ToolMessage
            if idx + 1 < len(messages) and messages[idx + 1].type == "tool":
                to_remove.append(RemoveMessage(id=messages[idx + 1].id))

        return {"messages": to_remove} if to_remove else None
    return None
```

---

## 上下文工程与工作记忆

### 什么是上下文工程？

2025年，Andrej Karpathy 提出了**上下文工程（Context Engineering）**的概念：

```
上下文工程 = 精心管理 LLM 上下文窗口中的每一个 Token

包括：
  1. 哪些信息放进上下文？（选择）
  2. 放在什么位置？（排列）
  3. 占多少空间？（分配）
  4. 什么时候清理？（回收）
```

[来源: atom/langchain/L4_Agent系统/07_Agent Memory集成/reference/search_agent_memory_01.md]

### 上下文窗口的空间分配

```
┌─────────────────────────────────────────────────┐
│  LLM Context Window (例: 128K tokens)            │
│                                                 │
│  ┌─────────────────────────────────────┐        │
│  │ System Prompt                       │ ~500   │
│  │ (角色定义、行为指令)                   │ tokens │
│  └─────────────────────────────────────┘        │
│  ┌─────────────────────────────────────┐        │
│  │ 长期记忆（从 Store 检索）             │ ~1000  │
│  │ (用户偏好、历史事实)                   │ tokens │
│  └─────────────────────────────────────┘        │
│  ┌─────────────────────────────────────┐        │
│  │ 短期记忆（对话历史）                   │ ~4000  │
│  │ (最近的对话消息)                      │ tokens │
│  └─────────────────────────────────────┘        │
│  ┌─────────────────────────────────────┐        │
│  │ 工作记忆（工具调用+结果）              │ ~6000  │
│  │ (当前任务的中间步骤)                   │ tokens │
│  └─────────────────────────────────────┘        │
│  ┌─────────────────────────────────────┐        │
│  │ RAG 检索结果                         │ ~3000  │
│  │ (从知识库检索的文档片段)                │ tokens │
│  └─────────────────────────────────────┘        │
│  ┌─────────────────────────────────────┐        │
│  │ 当前用户输入                         │ ~500   │
│  └─────────────────────────────────────┘        │
│                                                 │
│  已使用: ~15000 tokens                           │
│  剩余用于生成: ~113000 tokens                     │
└─────────────────────────────────────────────────┘
```

### 上下文工程实践

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    after_model,
    before_model,
)
from langchain.messages import RemoveMessage

# 1. 模型调用前：精心准备上下文
@before_model
def prepare_context(state, runtime):
    """在模型调用前优化上下文"""
    messages = state["messages"]

    # 确保 system prompt 始终在最前面
    # 长期记忆放在 system prompt 之后
    # 旧对话 → 新对话 → 工具结果 → 当前问题

    return None  # 不需要修改，只是日志/监控

# 2. 模型调用后：清理工作记忆
@after_model
def cleanup_working_memory(state, runtime):
    """清理不再需要的工作记忆"""
    messages = state["messages"]
    to_remove = []

    for msg in messages[:-10]:  # 只处理旧消息
        # 清理已完成任务的 ToolMessage（内容已被总结到回答中）
        if msg.type == "tool" and len(msg.content) > 2000:
            to_remove.append(RemoveMessage(id=msg.id))

    return {"messages": to_remove} if to_remove else None

# 3. 组合使用
agent = create_agent(
    model="gpt-4.1",
    tools=tools,
    middleware=[
        prepare_context,
        SummarizationMiddleware(
            model="gpt-4.1-mini",
            trigger=("tokens", 8000),
            keep=("messages", 30),
        ),
        cleanup_working_memory,
    ],
    checkpointer=checkpointer,
)
```

---

## 工作记忆 vs 短期记忆 vs 长期记忆

| 维度 | 工作记忆 | 短期记忆 | 长期记忆 |
|------|----------|----------|----------|
| **存什么** | 工具调用+结果 | 对话历史 | 用户事实 |
| **生命周期** | 单次推理 | 单次对话 | 跨对话 |
| **管理方式** | 自动（Agent循环） | Checkpointer | Store |
| **大小** | 通常最大 | 中等 | 可大可小 |
| **可控性** | 通过 middleware | 通过 middleware | 显式读写 |
| **人类类比** | 做数学题的草稿纸 | 今天的记忆 | 一辈子的知识 |

---

## AgentTokenBufferMemory（经典的 Agent 专用工作记忆）

```python
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory
)
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1")

# 专为 Agent 设计的 Token 限制记忆
memory = AgentTokenBufferMemory(
    llm=llm,
    max_token_limit=12000,               # 总 Token 限制
    memory_key="history",
    intermediate_steps_key="intermediate_steps",  # 包含工具调用
)

# 自动管理对话消息 + 工具调用的 Token 预算
# 当超过限制时，优先删除最旧的工具调用结果
```

[来源: sourcecode/langchain/libs/langchain/langchain_classic/agents/openai_functions_agent/agent_token_buffer_memory.py]

---

## 与 RAG 开发的关系

| RAG 场景 | 工作记忆的作用 |
|----------|---------------|
| **多步检索** | 记住第一次检索结果，用于优化第二次查询 |
| **文档对比** | 同时持有多个文档片段进行比较 |
| **链式推理** | 搜索→分析→再搜索，中间结果需要保留 |
| **Agent RAG** | 工具调用（检索）的结果是工作记忆的核心内容 |

---

## 本节小结

> **工作记忆是 Agent 多步推理的「草稿纸」。**
> - 经典方案用 `intermediate_steps` + `agent_scratchpad`
> - 现代方案统一到消息列表（AIMessage + ToolMessage）
> - **核心挑战**：工具输出可能很大，需要主动管理
> - **上下文工程**：精心分配 Context Window 中每种记忆的空间
> - 与短期/长期记忆配合，形成完整的 Agent 记忆体系

---

**上一篇：** [03_核心概念_4_长期记忆与跨会话.md](./03_核心概念_4_长期记忆与跨会话.md)
**下一篇：** [03_核心概念_6_记忆持久化后端.md](./03_核心概念_6_记忆持久化后端.md)
