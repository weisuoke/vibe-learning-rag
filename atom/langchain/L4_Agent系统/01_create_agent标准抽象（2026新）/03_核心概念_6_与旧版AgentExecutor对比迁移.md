# 核心概念 6：与旧版 AgentExecutor 对比与迁移

> LangChain 1.0 用一个 `create_agent()` 取代了旧版 `AgentExecutor` + 六种 `create_xxx_agent` 函数，从"选择困难症"走向"一个入口搞定一切"。

---

## 为什么要换？旧版到底有什么问题

在写迁移代码之前，先搞清楚旧版的痛点。不理解痛点，就不理解新版的设计决策。

### 旧版架构：两步创建 + 六种选择

```python
# ❌ 旧版方式（已弃用，移至 langchain-classic）
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# 第一步：选一种 Agent 类型，创建 Agent 对象
llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_messages([...])
agent = create_react_agent(llm, tools, prompt)  # 还是 create_openai_functions_agent？

# 第二步：把 Agent 塞进 AgentExecutor
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=15,
    max_execution_time=120,
    handle_parsing_errors=True,
    early_stopping_method="force",
    return_intermediate_steps=False,
    trim_intermediate_steps=-1,
)
result = executor.invoke({"input": "今天北京天气怎么样？"})
print(result["output"])
```

旧版有六种 Agent 创建函数，每种绑定不同的模型能力和输出格式：

| 函数名 | 绑定能力 | 输出格式 |
|--------|----------|----------|
| `create_react_agent()` | 文本推理 | ReAct 文本 |
| `create_openai_functions_agent()` | OpenAI Functions | JSON |
| `create_tool_calling_agent()` | 通用工具调用 | ToolCall |
| `create_structured_chat_agent()` | 结构化聊天 | JSON |
| `create_json_chat_agent()` | JSON 输出 | JSON |
| `create_xml_agent()` | XML 输出 | XML |

### 五大痛点

**痛点 1：选择困难症**

初学者面对 6 个函数，第一反应是"我该用哪个？"。答案取决于你用的模型支不支持 function calling、你想要什么输出格式——这些本不该是用户操心的事。

**痛点 2：参数爆炸**

`AgentExecutor` 有 7+ 个配置参数（`max_iterations`、`max_execution_time`、`early_stopping_method`、`handle_parsing_errors`、`return_intermediate_steps`、`trim_intermediate_steps` 等），参数之间还有隐含依赖。

**痛点 3：自定义困难**

想在 Agent 循环中加一个"每次调用模型前打日志"的逻辑？你得继承 `AgentExecutor`，重写 `_call` 方法，理解内部的 `intermediate_steps` 流转。

**痛点 4：模型耦合**

`create_openai_functions_agent` 只能用 OpenAI 模型。换成 Claude？得换函数。换成本地模型？再换一个。Agent 类型和模型能力绑死了。

**痛点 5：无内置持久化**

想让 Agent 记住上一轮对话？需要额外配置 `Memory` 组件，而且 Memory 的实现和 Agent 循环是割裂的。

---

## 新版架构：一个函数，一步到位

### create_agent（LangChain 1.0+）

```python
# ✅ 新版方式
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4o",                    # 字符串标识符，不用实例化
    tools=[get_weather, search_web],           # 直接传函数也行
    system_prompt="你是一个天气助手。",          # 直接传字符串
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "今天北京天气怎么样？"}]
})
print(result["messages"][-1].content)
```

就这么简单。没有选择困难，没有两步创建，没有参数爆炸。

---

## 全维度对比表

这张表是迁移的核心参考，建议收藏。

| 维度 | 旧版 AgentExecutor | 新版 create_agent |
|------|-------------------|-------------------|
| **创建方式** | 两步：`create_xxx_agent()` + `AgentExecutor()` | 一步：`create_agent()` |
| **Agent 类型** | 6+ 种函数，按模型能力区分 | 1 个统一函数，自动适配模型 |
| **模型指定** | 必须实例化 `ChatOpenAI()` 等对象 | 支持字符串 `"openai:gpt-4o"` |
| **Prompt** | 必须构造 `ChatPromptTemplate` | `system_prompt` 传字符串即可 |
| **自定义循环** | 继承重写或参数堆叠 | Middleware 组合 |
| **状态管理** | 外部 Memory 组件 | 内置 `AgentState`（基于 messages） |
| **持久化** | 需要额外配置 | `checkpointer` 参数一行搞定 |
| **底层引擎** | Chain 模式（线性执行） | LangGraph StateGraph（状态图） |
| **流式输出** | 有限支持 | 完整 `stream()` / `astream()` |
| **结构化输出** | 需要额外 OutputParser | `response_format` 参数内置 |
| **输入格式** | `{"input": "问题"}` | `{"messages": [{"role": "user", "content": "问题"}]}` |
| **输出格式** | `result["output"]`（字符串） | `result["messages"][-1].content` |
| **错误处理** | `handle_parsing_errors` 参数 | `ToolRetryMiddleware` |
| **最大迭代** | `max_iterations` 参数 | `ModelCallLimitMiddleware` |
| **执行超时** | `max_execution_time` 参数 | Middleware 自定义 |
| **人工审核** | 需要自定义实现 | `HumanInTheLoopMiddleware` |
| **返回值类型** | `CompiledStateGraph` 的前身 Chain | `CompiledStateGraph` |

---

## 逐场景迁移指南

### 场景 1：从 create_react_agent + AgentExecutor 迁移

这是最常见的旧版用法。

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ❌ 旧版：create_react_agent + AgentExecutor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain import hub

llm = ChatOpenAI(model="gpt-4", temperature=0)
prompt = hub.pull("hwchase17/react")  # 从 Hub 拉 ReAct prompt
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = executor.invoke({"input": "搜索最新的 AI 新闻"})
print(result["output"])
```

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ✅ 新版：create_agent（一步到位）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4o",
    tools=tools,
    system_prompt="你是一个有用的助手，可以搜索信息来回答问题。",
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "搜索最新的 AI 新闻"}]
})
print(result["messages"][-1].content)
```

**变化要点：**
- 不再需要从 Hub 拉 prompt 模板，`system_prompt` 直接传字符串
- 不再需要两步创建，一个函数搞定
- 输入从 `{"input": ...}` 变成 `{"messages": [...]}`
- 输出从 `result["output"]` 变成 `result["messages"][-1].content`

---

### 场景 2：从 create_openai_functions_agent 迁移

旧版中最常用的"OpenAI 专属"Agent 类型。

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ❌ 旧版：create_openai_functions_agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(model="gpt-4", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手。"),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),  # 必须有这个占位符
])
agent = create_openai_functions_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

result = executor.invoke({"input": "帮我查一下苹果的股价"})
```

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ✅ 新版：create_agent（自动处理 function calling）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4o",  # 自动检测模型是否支持 tool calling
    tools=tools,
    system_prompt="你是一个有用的助手。",
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "帮我查一下苹果的股价"}]
})
```

**变化要点：**
- 不再需要 `MessagesPlaceholder(variable_name="agent_scratchpad")` —— 新版内部自动管理
- 不再需要区分"这个模型支不支持 function calling"—— `create_agent` 自动适配
- 换模型只需改字符串：`"openai:gpt-4o"` → `"anthropic:claude-sonnet-4-20250514"`

---

### 场景 3：从 create_tool_calling_agent 迁移

这是旧版中最"通用"的 Agent 类型，也是最接近新版的。

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ❌ 旧版：create_tool_calling_agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个数据分析助手。"),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ✅ 新版：完全等价，更简洁
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4o",
    tools=tools,
    system_prompt="你是一个数据分析助手。",
    debug=True,  # 替代 verbose=True
)
```

**变化要点：**
- `verbose=True` → `debug=True`
- `create_tool_calling_agent` 的功能被 `create_agent` 完全吸收

---

## 参数映射：旧参数怎么迁移到新版

### max_iterations → ModelCallLimitMiddleware

```python
# ❌ 旧版
executor = AgentExecutor(agent=agent, tools=tools, max_iterations=10)

# ✅ 新版
from langchain.agents.middleware import ModelCallLimitMiddleware

agent = create_agent(
    model="openai:gpt-4o",
    tools=tools,
    middleware=[ModelCallLimitMiddleware(max_calls=10)],
)
```

**为什么用 Middleware 替代参数？** 因为"限制调用次数"是一种行为策略，不是 Agent 的核心配置。Middleware 模式让你可以自由组合多种策略，而不是把所有策略都塞进一个函数签名里。

### handle_parsing_errors → ToolRetryMiddleware

```python
# ❌ 旧版
executor = AgentExecutor(
    agent=agent, tools=tools,
    handle_parsing_errors=True  # 解析失败时把错误发回给 LLM
)

# ✅ 新版
from langchain.agents.middleware import ToolRetryMiddleware

agent = create_agent(
    model="openai:gpt-4o",
    tools=tools,
    middleware=[ToolRetryMiddleware(max_retries=3)],
)
```

新版更强大：不仅处理解析错误，还能处理工具执行错误，支持自定义重试策略。

### Memory → checkpointer

```python
# ❌ 旧版：外部 Memory 组件
from langchain_classic.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

# ✅ 新版：内置 checkpointer
from langgraph.checkpoint.memory import MemorySaver
agent = create_agent(
    model="openai:gpt-4o", tools=tools,
    checkpointer=MemorySaver(),  # 一行搞定
)
# 通过 thread_id 区分会话
config = {"configurable": {"thread_id": "user-123"}}
agent.invoke({"messages": [{"role": "user", "content": "我叫小明"}]}, config=config)
agent.invoke({"messages": [{"role": "user", "content": "我叫什么？"}]}, config=config)
```

旧版 Memory 是"追加到 prompt"模式，和 Agent 循环割裂；新版 checkpointer 是"持久化整个状态图"模式，天然支持多轮对话和断点恢复。

### verbose → debug

```python
# ❌ 旧版
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ✅ 新版
agent = create_agent(model="openai:gpt-4o", tools=tools, debug=True)
```

### early_stopping_method → 无直接对应

旧版的 `early_stopping_method="generate"` 会在达到最大迭代后让 LLM 生成一个最终回答。新版中，`ModelCallLimitMiddleware` 达到限制后直接停止，如果需要类似行为，可以自定义 Middleware。

---

## 输入/输出格式迁移

这是迁移中最容易忽略的部分，也是运行时报错最多的地方。

### 输入格式变化

```python
# ❌ 旧版输入
old_input = {"input": "今天天气怎么样？"}

# ✅ 新版输入
new_input = {
    "messages": [
        {"role": "user", "content": "今天天气怎么样？"}
    ]
}
```

### 输出格式变化

```python
# ❌ 旧版输出
result = executor.invoke({"input": "问题"})
answer = result["output"]  # 字符串

# ✅ 新版输出
result = agent.invoke({"messages": [...]})
answer = result["messages"][-1].content  # 最后一条消息的内容
```

新版输出是完整的消息列表，包含了 Agent 的全部思考过程（工具调用、中间结果等），比旧版的单一字符串更丰富。

---

## 从 langgraph.prebuilt.create_react_agent 迁移

如果你之前用的是 LangGraph 的 `create_react_agent`（不是 LangChain 旧版的），迁移更简单：

```python
# ❌ LangGraph 旧版（已弃用）
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=tools,
    state_modifier="你是一个助手。",
)

# ✅ LangChain 1.0 新版
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4o",
    tools=tools,
    system_prompt="你是一个助手。",
)
```

**变化要点：**
- `state_modifier` → `system_prompt`
- `model` 参数支持字符串标识符
- 功能完全兼容，API 更统一

---

## 架构层面的本质变化

理解表面的 API 变化不够，还要理解底层发生了什么。

### 旧版：Chain 线性模型

```
输入 → Prompt → LLM → OutputParser → 结束？
                                       ↓ 否
                                  执行工具 → 追加到 intermediate_steps → 回到 Prompt
```

旧版的 `AgentExecutor` 本质是一个 while 循环。状态通过 `intermediate_steps`（格式 `list[tuple[AgentAction, str]]`）传递。

### 新版：StateGraph 状态图模型

```
         ┌──────────────┐
         │  model node  │ ← Middleware 钩子
         └──────┬───────┘
                │ AIMessage
                ▼
         ┌──────────────┐
         │  判断节点     │ → 无 tool_calls → 结束
         └──────┬───────┘
                │ 有 tool_calls
                ▼
         ┌──────────────┐
         │  tool node   │ ← Middleware 钩子
         └──────┬───────┘
                │ ToolMessage → 回到 model node
```

新版基于 LangGraph 的 `StateGraph`，状态是 `AgentState`（核心字段 `messages: list[AnyMessage]`）。

| 维度 | 旧版 Chain | 新版 StateGraph |
|------|-----------|----------------|
| 状态表示 | `intermediate_steps: list[tuple]` | `messages: list[AnyMessage]` |
| 执行模型 | while 循环 | 图遍历 |
| 扩展方式 | 继承重写 | Middleware / 添加节点 |
| 持久化 | 外部 Memory | 内置 Checkpointer |

---

## 渐进式迁移策略

不需要一次性迁移所有代码。推荐分三步走：

**第一步：安装兼容包**

```bash
uv pip install --upgrade langchain      # 升级到 1.0
uv pip install langchain-classic        # 旧代码继续跑
```

旧代码只需改 import 路径：`from langchain.agents` → `from langchain_classic.agents`

**第二步：新功能用新 API**

所有新写的 Agent 代码统一用 `create_agent`。新旧代码可以在同一个项目中共存。

**第三步：逐步替换旧代码**

按优先级：先替换无自定义逻辑的简单 Agent → 有 Memory 的 Agent（改用 checkpointer）→ 有复杂自定义逻辑的 Agent（改用 Middleware）。

---

## 迁移检查清单

- [ ] 安装 `langchain` 1.0+ 和 `langchain-classic`
- [ ] 替换 `create_xxx_agent()` + `AgentExecutor()` → `create_agent()`
- [ ] 替换 `ChatPromptTemplate` → `system_prompt` 字符串
- [ ] 替换 `max_iterations` → `ModelCallLimitMiddleware`
- [ ] 替换 `handle_parsing_errors` → `ToolRetryMiddleware`
- [ ] 替换 `Memory` → `checkpointer`
- [ ] 更新输入格式：`{"input": ...}` → `{"messages": [...]}`
- [ ] 更新输出格式：`result["output"]` → `result["messages"][-1].content`
- [ ] 运行测试，确认功能正常
- [ ] 全部迁移完成后移除 `langchain-classic`

---

## 常见迁移问题 FAQ

### Q1: 旧代码还能用吗？

可以。安装 `langchain-classic` 后，把 import 路径从 `langchain.agents` 改为 `langchain_classic.agents`，功能完全一样。但 `langchain-classic` 不会有新功能，只做安全修复。

### Q2: 必须一次性迁移吗？

不需要。新旧代码可以在同一个项目中共存。`langchain` 1.0 和 `langchain-classic` 可以同时安装。

### Q3: langgraph.prebuilt.create_react_agent 还能用吗？

已弃用。功能已合并到 `langchain.agents.create_agent`。如果你之前用的是 LangGraph 的 `create_react_agent`，迁移成本很低（见上方迁移示例）。

### Q4: 迁移后性能会变吗？

新版基于 LangGraph StateGraph，在复杂场景下性能更好（状态管理更高效）。简单场景下性能基本一致。流式输出的延迟会明显降低。

### Q5: 自定义的 OutputParser 怎么迁移？

简单场景用 `response_format` 参数定义结构化输出；复杂场景写自定义 Middleware，在 `after_model` 钩子中处理。

---

## 一句话总结

旧版是"选一种 Agent 类型 → 配一堆参数 → 塞进 Executor"的三步走，新版是"一个 `create_agent()` + Middleware 组合"的一步到位——API 更少，能力更强，扩展更灵活。

---

> 来源：
> - [LangChain 1.0 发布公告](https://blog.langchain.com/langchain-langgraph-1dot0)
> - [Agent Middleware 博客](https://blog.langchain.com/agent-middleware)
> - 源码分析：`langchain/agents/factory.py`、`langchain_classic/agents/agent.py`
