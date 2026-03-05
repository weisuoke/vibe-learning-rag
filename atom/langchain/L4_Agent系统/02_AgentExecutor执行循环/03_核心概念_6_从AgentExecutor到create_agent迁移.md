# 核心概念 6：从 AgentExecutor 到 create_agent 迁移

> AgentExecutor 是 LangChain 经典版的 Agent 执行引擎，create_agent 是 v1 的现代替代。核心循环思想不变，但底层架构从 while 循环升级为 LangGraph 状态图。

---

## 为什么要迁移？

AgentExecutor 位于 `langchain_classic` 包中，属于遗留代码。LangChain v1 用 `create_agent` 工厂函数替代了它，基于 LangGraph StateGraph 重新实现了 Agent 执行循环。

迁移的核心动机：

| 痛点 | AgentExecutor 的问题 | create_agent 的解决方案 |
|------|---------------------|----------------------|
| 扩展性差 | 需要子类化才能修改行为 | 中间件组合，即插即用 |
| 状态管理原始 | `intermediate_steps` 列表，无持久化 | `AgentState` TypedDict + Checkpointer |
| 错误处理粗糙 | 单一 `handle_parsing_errors` 参数 | 独立的 Retry 中间件，支持指数退避 |
| 循环控制简单 | `max_iterations` 全局计数 | `ToolCallLimitMiddleware` 支持按工具限制 |
| 无法暂停恢复 | 执行过程不可中断 | `interrupt_before` / `interrupt_after` |
| 调试困难 | 依赖 verbose 和回调 | LangGraph Studio 可视化 + Time Travel |

一句话：**AgentExecutor 是"能用"，create_agent 是"好用"。**

---

## 架构对比：while 循环 vs StateGraph

### AgentExecutor：命令式 while 循环

```python
# AgentExecutor 的核心（agent.py:1570，简化）
def _call(self, inputs):
    intermediate_steps = []
    iterations = 0

    while self._should_continue(iterations, time_elapsed):
        # 1. 调用 LLM
        output = self.agent.plan(intermediate_steps, **inputs)

        # 2. 检查是否结束
        if isinstance(output, AgentFinish):
            return self._return(output, intermediate_steps)

        # 3. 执行工具
        for action in output:
            observation = self.tools[action.tool].run(action.tool_input)
            intermediate_steps.append((action, observation))

        iterations += 1

    # 超出限制
    return self._force_stop(intermediate_steps, **inputs)
```

特点：所有逻辑在一个方法里，用 `while` + `if/else` 控制流程。

### create_agent：声明式 StateGraph

```python
# create_agent 的核心（factory.py:658，简化）
def create_agent(model, tools, middleware, ...):
    graph = StateGraph(state_schema=AgentState)

    # 声明节点
    graph.add_node("model", model_node)    # 调用 LLM
    graph.add_node("tools", tool_node)     # 执行工具

    # 声明边（流转规则）
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route_after_model)
    graph.add_edge("tools", "model")

    return graph.compile(checkpointer=checkpointer)

# 路由函数
def route_after_model(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"    # 有工具调用 → 执行工具
    return END            # 无工具调用 → 结束
```

特点：节点和边分离，流程声明式定义，状态自动流转。

### 可视化对比

```
AgentExecutor（命令式）:
┌──────────────────────────────┐
│  while _should_continue():   │
│    output = agent.plan()     │
│    if AgentFinish: return    │
│    for action in output:     │
│      observation = tool.run()│
│    iterations += 1           │
│  force_stop()                │
└──────────────────────────────┘

create_agent（声明式）:
┌─────────┐    tool_calls?    ┌─────────┐
│  model  │ ───── yes ──────→ │  tools  │
│  node   │ ←────────────────  │  node   │
│         │ ───── no ────→ END│         │
└─────────┘                   └─────────┘
```

---

## 详细对比表

### 参数映射

| 功能 | AgentExecutor 参数 | create_agent 等价方案 |
|------|-------------------|---------------------|
| LLM | `agent`（需要预构建） | `model`（字符串或对象） |
| 工具 | `tools` | `tools` |
| 系统提示 | 在 agent 构建时设置 | `system_prompt` |
| 最大迭代 | `max_iterations=15` | `ToolCallLimitMiddleware` |
| 最大时间 | `max_execution_time` | 自定义中间件 |
| 解析错误 | `handle_parsing_errors` | `ModelRetryMiddleware` |
| 步骤裁剪 | `trim_intermediate_steps` | 消息管理（`RemoveMessage`） |
| 返回步骤 | `return_intermediate_steps` | 默认返回完整 `messages` |
| 提前停止 | `early_stopping_method` | `ToolCallLimitMiddleware(exit_behavior=...)` |
| 状态持久化 | 无 | `checkpointer` |
| 执行中断 | 无 | `interrupt_before` / `interrupt_after` |
| 结构化输出 | 无 | `response_format` |

### 核心数据类型映射

| AgentExecutor | create_agent | 说明 |
|--------------|-------------|------|
| `AgentAction` | `AIMessage.tool_calls` | 工具调用请求 |
| `AgentFinish` | `AIMessage`（无 tool_calls） | 最终回复 |
| `AgentStep` | `ToolMessage` | 工具执行结果 |
| `intermediate_steps: list[tuple]` | `AgentState["messages"]: list[AnyMessage]` | 执行历史 |

---

## 循环控制对比

### AgentExecutor：max_iterations + max_execution_time

```python
executor = AgentExecutor(
    agent=my_agent,
    tools=my_tools,
    max_iterations=10,           # 最多 10 次迭代
    max_execution_time=30.0,     # 最多 30 秒
    early_stopping_method="force",  # 超限时强制停止
)
```

源码逻辑（agent.py:1235）：

```python
def _should_continue(self, iterations, time_elapsed):
    if self.max_iterations is not None and iterations >= self.max_iterations:
        return False
    return self.max_execution_time is None or time_elapsed < self.max_execution_time
```

局限：只能全局计数，无法区分不同工具的调用次数。

### create_agent：ToolCallLimitMiddleware

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware

agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool, calculator_tool],
    middleware=[
        ToolCallLimitMiddleware(
            max_tool_calls=10,                    # 全局最多 10 次工具调用
            per_tool_limits={
                "search": 5,                      # search 最多 5 次
                "calculator": 3,                  # calculator 最多 3 次
            },
            exit_behavior="end",                  # 超限时结束（不报错）
        ),
    ],
)
```

优势：
- 支持按工具名称设置独立限制
- 支持 thread 级别和 run 级别的计数
- `exit_behavior` 提供三种策略：`"continue"`（跳过超限工具）、`"error"`（抛异常）、`"end"`（结束循环）

---

## 错误处理对比

### AgentExecutor：handle_parsing_errors

```python
# 方式一：布尔值
executor = AgentExecutor(
    agent=my_agent,
    tools=my_tools,
    handle_parsing_errors=True,  # 将解析错误作为 observation 反馈给 LLM
)

# 方式二：自定义字符串
executor = AgentExecutor(
    agent=my_agent,
    tools=my_tools,
    handle_parsing_errors="输出格式错误，请按照要求的格式重新回答。",
)

# 方式三：自定义函数
executor = AgentExecutor(
    agent=my_agent,
    tools=my_tools,
    handle_parsing_errors=lambda e: f"解析失败: {str(e)[:200]}",
)
```

源码逻辑（agent.py:1301，简化）：

```python
try:
    output = self.agent.plan(intermediate_steps, **inputs)
except OutputParserException as e:
    if isinstance(self.handle_parsing_errors, bool):
        if not self.handle_parsing_errors:
            raise e
        observation = str(e)  # 错误信息作为 observation
    elif isinstance(self.handle_parsing_errors, str):
        observation = self.handle_parsing_errors
    elif callable(self.handle_parsing_errors):
        observation = self.handle_parsing_errors(e)

    # 创建一个特殊的 AgentAction，工具名为 "_Exception"
    output = AgentAction("_Exception", observation, text)
    observation = ExceptionTool().run(output.tool_input)
    yield AgentStep(action=output, observation=observation)
```

局限：只处理解析错误，不处理工具执行错误和 LLM 调用错误。

### create_agent：独立的 Retry 中间件

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware

agent = create_agent(
    model="openai:gpt-4o",
    tools=my_tools,
    middleware=[
        # LLM 调用重试
        ModelRetryMiddleware(
            max_retries=3,
            retry_on=(TimeoutError, RateLimitError),  # 指定重试的异常类型
            retry_delay_seconds=1.0,                   # 初始延迟
            backoff_factor=2.0,                        # 指数退避因子
            on_failure="continue",                     # 失败后注入错误消息继续
        ),
        # 工具执行重试
        ToolRetryMiddleware(
            max_retries=2,
            retry_on=lambda e: "rate limit" in str(e).lower(),  # 自定义判断
            on_failure="error",                        # 最终失败时抛异常
        ),
    ],
)
```

优势：
- LLM 错误和工具错误分开处理
- 支持指数退避 + jitter
- 支持自定义重试条件（异常类型或判断函数）
- `on_failure` 提供灵活的失败策略

---

## 状态管理对比

### AgentExecutor：intermediate_steps 列表

```python
# 状态就是一个简单的列表
intermediate_steps: list[tuple[AgentAction, str]] = []

# 每步追加
intermediate_steps.append((action, observation))

# 裁剪
intermediate_steps = intermediate_steps[-5:]  # 保留最后 5 步

# 返回
if self.return_intermediate_steps:
    final_output["intermediate_steps"] = intermediate_steps
```

局限：
- 无类型约束
- 无持久化
- 裁剪逻辑与业务逻辑耦合
- 无法跨请求保持状态

### create_agent：AgentState + Checkpointer

```python
from typing import TypedDict, Annotated, Required, NotRequired
from langgraph.graph import add_messages

class AgentState(TypedDict):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[...]
    structured_response: NotRequired[...]

# 持久化
from langgraph.checkpoint.sqlite import SqliteSaver

agent = create_agent(
    model="openai:gpt-4o",
    tools=my_tools,
    checkpointer=SqliteSaver.from_conn_string("agent.db"),
)

# 第一次对话
config = {"configurable": {"thread_id": "user-123"}}
agent.invoke({"messages": [{"role": "user", "content": "我叫小明"}]}, config=config)

# 第二次对话（状态自动恢复）
result = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么？"}]},
    config=config,
)
# Agent 能记住"小明"
```

优势：
- TypedDict 提供类型安全
- `add_messages` reducer 自动处理消息合并
- Checkpointer 支持跨请求状态持久化
- 支持 Time Travel（回溯到任意历史状态）

---

## 迁移代码示例

### 场景 1：基础 Agent

**迁移前（AgentExecutor）：**

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 定义工具
@tool
def search(query: str) -> str:
    """搜索互联网"""
    return f"搜索结果: {query} 的相关信息..."

# 构建 Agent（需要手动组装 prompt）
llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手。"),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),  # 必须有这个占位符
])
agent = create_openai_tools_agent(llm, [search], prompt)

# 构建 Executor
executor = AgentExecutor(
    agent=agent,
    tools=[search],
    max_iterations=10,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
    verbose=True,
)

# 执行
result = executor.invoke({"input": "Python 3.13 有什么新特性？"})
print(result["output"])
```

**迁移后（create_agent）：**

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.tools import tool

# 工具定义不变
@tool
def search(query: str) -> str:
    """搜索互联网"""
    return f"搜索结果: {query} 的相关信息..."

# 一行创建 Agent（不需要手动组装 prompt）
agent = create_agent(
    model="openai:gpt-4o",
    tools=[search],
    system_prompt="你是一个有用的助手。",
    middleware=[
        ToolCallLimitMiddleware(max_tool_calls=10),
    ],
)

# 执行（输入格式不同：messages 而非 input）
result = agent.invoke({
    "messages": [{"role": "user", "content": "Python 3.13 有什么新特性？"}]
})
# 完整消息历史直接在 result["messages"] 中
for msg in result["messages"]:
    print(f"[{msg.type}] {msg.content[:100]}")
```

### 场景 2：带错误处理和循环控制

**迁移前：**

```python
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=15,
    max_execution_time=60.0,
    handle_parsing_errors="格式错误，请重试。",
    early_stopping_method="generate",
    trim_intermediate_steps=5,
)
```

**迁移后：**

```python
from langchain.agents.middleware import (
    ToolCallLimitMiddleware,
    ModelRetryMiddleware,
)

agent = create_agent(
    model="openai:gpt-4o",
    tools=tools,
    middleware=[
        ToolCallLimitMiddleware(
            max_tool_calls=15,
            exit_behavior="end",       # 对应 early_stopping_method="generate"
        ),
        ModelRetryMiddleware(
            max_retries=3,
            on_failure="continue",     # 对应 handle_parsing_errors=True
        ),
    ],
)
# 注意：max_execution_time 需要自定义中间件实现
# 注意：trim_intermediate_steps 由 messages 管理替代（RemoveMessage）
```

### 场景 3：带状态持久化的对话 Agent

**迁移前（AgentExecutor 无内置持久化）：**

```python
# 需要手动管理对话历史
chat_history = []

def chat(user_input: str) -> str:
    result = executor.invoke({
        "input": user_input,
        "chat_history": chat_history,
    })
    # 手动追加历史
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=result["output"]))
    return result["output"]
```

**迁移后（create_agent 内置持久化）：**

```python
from langgraph.checkpoint.memory import MemorySaver

agent = create_agent(
    model="openai:gpt-4o",
    tools=tools,
    checkpointer=MemorySaver(),  # 内存持久化（生产用 SqliteSaver/PostgresSaver）
)

config = {"configurable": {"thread_id": "user-123"}}

# 自动管理对话历史，无需手动追加
agent.invoke({"messages": [{"role": "user", "content": "你好"}]}, config=config)
agent.invoke({"messages": [{"role": "user", "content": "刚才说了什么？"}]}, config=config)
```

---

## 什么没有变？

尽管架构大幅升级，核心循环思想完全一致：

```
两者的核心循环都是：

1. 调用 LLM，获取决策
2. 检查决策：有工具调用？
   ├─ 是 → 执行工具，获取结果，回到步骤 1
   └─ 否 → 返回最终答案

这就是 ReAct 模式：Thought → Action → Observation → Thought → ...
```

不变的还有：
- **工具定义方式**：`@tool` 装饰器、BaseTool 子类、函数——两者通用
- **LLM 调用接口**：底层都是 `ChatModel.invoke(messages)`
- **回调系统**：`callbacks` 参数两者都支持
- **流式输出**：`.stream()` 和 `.astream()` 两者都有

---

## 迁移时机建议

### 现在就该迁移的情况

1. **新项目**：没有历史包袱，直接用 create_agent
2. **需要状态持久化**：AgentExecutor 没有内置方案
3. **需要细粒度控制**：按工具限制调用次数、复杂重试策略
4. **需要人机协作**：`interrupt_before` / `interrupt_after` 只有 create_agent 支持
5. **需要结构化输出**：`response_format` 只有 create_agent 支持

### 可以暂缓的情况

1. **已有稳定运行的 AgentExecutor**：如果没有上述需求，不必急于迁移
2. **简单的单轮 Agent**：AgentExecutor 足够用
3. **团队还不熟悉 LangGraph**：先学习 LangGraph 基础再迁移

### 迁移检查清单

```
□ 确认 langchain v1 已安装（pip install langchain>=1.0）
□ 将 agent 构建逻辑替换为 create_agent
□ 将 max_iterations 替换为 ToolCallLimitMiddleware
□ 将 handle_parsing_errors 替换为 ModelRetryMiddleware
□ 将 input/output 格式从 dict 改为 messages
□ 如需持久化，配置 checkpointer
□ 更新回调处理器（如有自定义）
□ 更新测试用例
□ 验证工具兼容性（@tool 装饰器通用，无需修改）
```

---

## 与生产 AI Agent 开发的关联

迁移到 create_agent 不只是"换个 API"，而是为生产级 Agent 打下基础：

| 生产需求 | AgentExecutor | create_agent |
|----------|--------------|-------------|
| 可观测性 | verbose + 回调 | LangGraph Studio + LangSmith |
| 容错 | 简单重试 | 指数退避 + 按异常类型重试 |
| 成本控制 | max_iterations | 按工具精细限制 |
| 多轮对话 | 手动管理历史 | Checkpointer 自动持久化 |
| 人工审核 | 无 | interrupt_before/after |
| 状态回溯 | 无 | Time Travel |
| 水平扩展 | 困难 | StateGraph 天然支持 |

在 RAG 场景中，create_agent 的优势更加明显：
- **检索工具限制**：用 `per_tool_limits` 限制向量检索次数，避免无意义的重复检索
- **对话式 RAG**：Checkpointer 自动保存对话上下文，无需手动管理
- **人工审核**：在检索结果返回后暂停，让人工确认是否继续生成

---

## 小结

从 AgentExecutor 到 create_agent 的迁移，本质是从"命令式编程"到"声明式编程"的转变：

- **循环控制**：从 `while` + `if/else` 到 StateGraph 节点和边
- **扩展方式**：从子类化到中间件组合
- **状态管理**：从裸列表到 TypedDict + Reducer + Checkpointer
- **错误处理**：从单一参数到独立中间件

核心 ReAct 循环思想不变，变的是实现方式和工程能力。

一句话：**AgentExecutor 教你理解 Agent 循环的本质，create_agent 给你生产级的工程能力。**

---

**上一篇**: [03_核心概念_5_中间步骤管理.md](./03_核心概念_5_中间步骤管理.md)
**下一篇**: [04_最小可用.md](./04_最小可用.md)

---

> [来源: sourcecode/langchain/libs/langchain/langchain_classic/agents/agent.py]
> [来源: sourcecode/langchain/libs/langchain_v1/langchain/agents/factory.py]
> [来源: sourcecode/langchain/libs/langchain_v1/langchain/agents/middleware/tool_call_limit.py]
> [来源: sourcecode/langchain/libs/langchain_v1/langchain/agents/middleware/_retry.py]
