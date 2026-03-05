# 核心概念 1: create_agent() 工厂函数

> LangChain 1.0 唯一的 Agent 创建入口，一个函数取代了旧版十几个 API

---

## 什么是 create_agent？

**`create_agent()` 是 LangChain 1.0 中创建 Agent 的唯一工厂函数。你给它模型和工具，它还你一个完整的 Agent —— 内部自动构建 LangGraph 状态图，处理模型调用、工具执行、状态流转的全部细节。**

**前端类比：** 就像 Vue 3 的 `createApp()` —— 接收配置，内部组装路由、状态管理、插件，最后返回一个可挂载的应用实例。

---

## 函数签名全景

```python
# 来源: langchain/agents/factory.py 第 658 行
def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | dict[str, Any] | None = None,
    state_schema: type[AgentState[ResponseT]] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache[Any] | None = None,
) -> CompiledStateGraph
```

**三个关键观察：**

1. 只有 `model` 是必填参数，其余全部可选
2. `*` 之后的参数都是 keyword-only，必须用关键字传递
3. 返回值是 `CompiledStateGraph`，不是某个 Agent 类 —— **Agent 就是一张图**

---

## 参数分组详解

13 个参数按用途分为三组：

| 分组 | 参数 | 适用阶段 |
|------|------|----------|
| 核心参数 | model / tools / system_prompt | 必须理解 |
| 扩展参数 | middleware / response_format / state_schema / context_schema | 进阶使用 |
| 生产参数 | checkpointer / store / interrupt_* / debug / name / cache | 部署使用 |

---

### 第一组：核心参数

#### 1. model —— 模型标识（唯一必填）

**类型：** `str | BaseChatModel`

支持两种传入方式：

```python
from langchain.agents import create_agent

# 方式一：字符串格式（推荐）
agent = create_agent(model="openai:gpt-4o")
agent = create_agent(model="anthropic:claude-sonnet-4-20250514")
agent = create_agent(model="gpt-4o")  # 省略 provider 默认 OpenAI
```

内部调用 `init_chat_model(model)` 自动实例化对应 ChatModel，不需要手动导入各家 SDK。

```python
# 方式二：BaseChatModel 实例（需要精细控制时）
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=4096)
agent = create_agent(model=llm)
```

**什么时候用哪种？**

| 场景 | 推荐方式 | 原因 |
|------|----------|------|
| 快速原型 | 字符串 | 一行搞定 |
| 需要调 temperature | 实例 | 字符串不支持额外参数 |
| 多 Agent 共享模型 | 实例 | 避免重复创建 |
| CI/CD 环境切换 | 字符串 | 通过环境变量控制 provider |

---

#### 2. tools —— 工具列表

**类型：** `Sequence[BaseTool | Callable | dict] | None`

Agent 能调用的工具集合，支持三种格式混用：

```python
from langchain_core.tools import tool

# 格式一：@tool 装饰器函数（最常用）
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    return f"{city}: 晴天, 25°C"

# 格式二：BaseTool 子类实例
from langchain_community.tools import TavilySearchResults
search_tool = TavilySearchResults(max_results=3)

# 格式三：dict 格式（provider 内置工具）
code_interpreter = {"type": "code_interpreter"}

# 混用
agent = create_agent(model="openai:gpt-4o", tools=[get_weather, search_tool, code_interpreter])
```

**tools 为 None 时：** Agent 只有 model 节点，没有工具循环，退化为带状态管理的 ChatModel。

**工具来源合并：** 最终工具列表 = 用户 `tools` + 所有 middleware 的 `tools` 属性。

---

#### 3. system_prompt —— 系统提示词

**类型：** `str | SystemMessage | None`

定义 Agent 的角色和行为规则：

```python
# 方式一：纯字符串（最常用）
agent = create_agent(
    model="openai:gpt-4o",
    system_prompt="你是一个专业的技术文档助手，回答要简洁准确。",
)

# 方式二：SystemMessage 对象（需要多模态内容时）
from langchain_core.messages import SystemMessage
system_msg = SystemMessage(content=[
    {"type": "text", "text": "你是一个图片分析助手。"},
])
agent = create_agent(model="openai:gpt-4o", system_prompt=system_msg)
```

内部统一转换为 `SystemMessage`，放在消息列表最前面。

---

### 第二组：扩展参数

#### 4. middleware —— 中间件列表

**类型：** `Sequence[AgentMiddleware]`

这是 LangChain 1.0 最核心的创新，取代了旧版的参数爆炸：

```python
from langchain.agents.middleware import (
    SummarizationMiddleware, ModelRetryMiddleware, HumanInTheLoopMiddleware,
)

agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool],
    middleware=[
        SummarizationMiddleware(max_messages=20),
        ModelRetryMiddleware(max_retries=3),
        HumanInTheLoopMiddleware(tools=[search_tool]),
    ],
)
```

**执行顺序遵循洋葱模型**（和 Express.js / Koa 一样）：

```
请求进入 →
  Middleware A: before_model →
    Middleware B: before_model →
      [实际模型调用]
    Middleware B: after_model →
  Middleware A: after_model →
响应返回
```

和 Express.js / Koa 的中间件模型一样。详见 [03_核心概念_3](./03_核心概念_3_Middleware基类与钩子系统.md)。

---

#### 5. response_format —— 结构化输出

**类型：** `ResponseFormat | type | dict | None`

让 Agent 返回结构化数据：

```python
from pydantic import BaseModel, Field

class AnalysisResult(BaseModel):
    summary: str = Field(description="摘要")
    key_points: list[str] = Field(description="关键要点")

agent = create_agent(model="openai:gpt-4o", response_format=AnalysisResult)
result = agent.invoke({"messages": [{"role": "user", "content": "分析这篇文章..."}]})
# result["structured_response"] 是 AnalysisResult 实例
```

**内部自动选择最优策略：**

| 策略 | 触发条件 | 原理 |
|------|----------|------|
| `AutoStrategy` | 默认 | 自动检测模型能力，选最优策略 |
| `ToolStrategy` | 模型不支持原生结构化输出 | 将输出格式包装为 tool_call |
| `ProviderStrategy` | 模型原生支持（如 GPT-4o） | 使用 provider 的 response_format 参数 |

详见 [03_核心概念_6](./03_核心概念_6_结构化输出ResponseFormat.md)。

---

#### 6. state_schema —— 自定义状态

**类型：** `type[AgentState] | None`

扩展 Agent 内部状态，添加自定义字段：

```python
from langchain.agents import create_agent, AgentState

class MyAgentState(AgentState):
    user_preferences: dict = {}
    interaction_count: int = 0

agent = create_agent(model="openai:gpt-4o", state_schema=MyAgentState)
```

默认 AgentState 只有三个字段：

```python
class AgentState(TypedDict, Generic[ResponseT]):
    messages: Required[Annotated[list[AnyMessage], add_messages]]  # 消息历史
    jump_to: NotRequired[...]    # 内部跳转控制（私有）
    structured_response: NotRequired[...]  # 结构化输出结果
```

middleware 也可以通过 `state_schema` 类属性扩展状态。最终 state_schema = 用户传入 + 所有 middleware 的合并。

---

#### 7. context_schema —— 运行时上下文类型

**类型：** `type[ContextT] | None`

定义调用时传入的临时上下文数据类型：

```python
from typing import TypedDict

class UserContext(TypedDict):
    user_id: str
    user_role: str

agent = create_agent(model="openai:gpt-4o", context_schema=UserContext)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "帮我查询订单"}]},
    context={"user_id": "u_123", "user_role": "vip"},
)
```

context 不存入状态、不持久化，是每次调用的临时数据。middleware 通过 `request.runtime.context` 读取。典型用途：多租户 tenant_id、用户角色、认证信息。

---

### 第三组：生产参数

#### 8. checkpointer —— 状态持久化

**类型：** `Checkpointer | None`

让 Agent 对话状态跨请求保存和恢复：

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="openai:gpt-4o",
    checkpointer=InMemorySaver(),  # 生产环境换 PostgresSaver
)

config = {"configurable": {"thread_id": "conversation_001"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "你好"}]},
    config=config,
)
# 下次用同一个 thread_id，Agent 记住之前的对话
```

---

#### 9. store —— 跨线程数据共享

**类型：** `BaseStore | None`

与 checkpointer 不同，store 用于跨对话线程共享数据：

| 特性 | checkpointer | store |
|------|-------------|-------|
| 作用域 | 单个 thread_id | 跨所有 thread |
| 典型用途 | 对话历史恢复 | 用户画像、全局配置 |
| 类比 | sessionStorage | localStorage |

---

#### 10. interrupt_before / interrupt_after —— 人机协作断点

**类型：** `list[str] | None`

在指定节点前/后暂停执行，等待人工确认（需配合 checkpointer）：

```python
agent = create_agent(
    model="openai:gpt-4o",
    tools=[send_email],
    checkpointer=InMemorySaver(),
    interrupt_before=["tools"],  # 在执行工具前暂停
)

config = {"configurable": {"thread_id": "t_001"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "发邮件给张三"}]},
    config=config,
)
# Agent 决定调用 send_email，但在执行前暂停
# 人工审核后继续：
result = agent.invoke(None, config=config)
```

可用节点名：`"agent"`（模型推理）和 `"tools"`（工具执行）。

---

#### 11. debug / name / cache —— 辅助参数

```python
researcher = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool],
    name="researcher",       # 多 Agent 系统中的节点标识
    debug=True,              # 输出每个节点的输入输出
    cache=InMemoryCache(),   # 相同输入直接返回缓存（省钱利器）
)
```

- **debug**：开启详细执行日志，开发时打开，生产关闭
- **name**：多 Agent 系统中标识子图，也用于 LangSmith 追踪
- **cache**：缓存模型调用结果，开发调试时避免重复调 API

---

## 内部执行流程

`create_agent()` 内部按顺序执行 10 步：

```
① init_chat_model(model)        ← 字符串 → BaseChatModel 实例
② 转换 system_prompt             ← str → SystemMessage 对象
③ 处理 response_format           ← 选择 Auto/Tool/Provider 策略
④ 收集 middleware 的 tools        ← 遍历所有 middleware.tools
⑤ 创建 ToolNode                  ← 合并用户工具 + middleware 工具
⑥ 验证 middleware                 ← 检查重复类型、冲突配置
⑦ 分类 middleware hooks           ← before/after vs wrap 钩子分组
⑧ 组合 wrap 处理器链              ← _chain_model_call_handlers()
⑨ 合并 state_schema              ← 用户 schema + middleware schema
⑩ 构建 StateGraph → compile()    ← model 节点 + tool 节点 + 边 → 返回
```

**关键细节：**

- 步骤 ① 中，`init_chat_model()` 根据 provider 前缀动态导入对应的 `langchain-openai`、`langchain-anthropic` 等包
- 步骤 ⑤ 中，工具被分为 `built_in_tools`（dict 格式）和 `regular_tools`（BaseTool/Callable），分别处理
- 步骤 ⑥ 中，同一类型的两个 middleware 实例会触发验证错误
- 步骤 ⑩ 中，图只有两个核心节点：`model` 和 `tools`，通过条件边连接

---

## 返回值：CompiledStateGraph

`create_agent()` 返回 LangGraph 的 `CompiledStateGraph`，支持全部执行方式：

```python
agent = create_agent(model="openai:gpt-4o", tools=[search_tool])
input_msg = {"messages": [{"role": "user", "content": "你好"}]}

# ① invoke —— 同步执行
result = agent.invoke(input_msg)

# ② ainvoke —— 异步执行
result = await agent.ainvoke(input_msg)

# ③ stream —— 流式输出
for chunk in agent.stream(input_msg, stream_mode="updates"):
    print(chunk)

# ④ astream —— 异步流式（打字机效果）
async for chunk in agent.astream(input_msg, stream_mode="messages"):
    print(chunk)

# ⑤ batch —— 批量调用
results = agent.batch([input_msg, input_msg])
```

**stream_mode 选项：**

| 模式 | 返回内容 | 适用场景 |
|------|----------|----------|
| `"values"` | 每步的完整状态 | 调试，查看全貌 |
| `"updates"` | 每步的状态增量 | 前端展示进度条 |
| `"messages"` | LLM token 级别流式 | 打字机效果 |

---

## 完整示例

```python
"""create_agent 完整示例：工具 + 中间件 + 持久化"""
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ModelRetryMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"

agent = create_agent(
    model="openai:gpt-4o",
    tools=[calculate],
    system_prompt="你是一个数学助手，请用中文回答。",
    middleware=[
        ModelRetryMiddleware(max_retries=2),
        SummarizationMiddleware(max_messages=50),
    ],
    checkpointer=InMemorySaver(),
    name="math_assistant",
)

# 多轮对话（Agent 通过 checkpointer 记住上下文）
config = {"configurable": {"thread_id": "demo_001"}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "123 * 456 等于多少？"}]},
    config=config,
)
print(result["messages"][-1].content)  # → 56088

result = agent.invoke(
    {"messages": [{"role": "user", "content": "再加上 789 呢？"}]},
    config=config,
)
print(result["messages"][-1].content)  # → 56877
```

---

## 与旧版 API 的对比

| 维度 | 旧版 (0.x) | 新版 (1.0) |
|------|-----------|------------|
| 创建函数 | `initialize_agent()` + 多个 `create_xxx_agent()` | 唯一的 `create_agent()` |
| Agent 类型 | `AgentType.ZERO_SHOT_REACT` 等枚举 | 通过 middleware 组合 |
| 执行器 | `AgentExecutor` | `CompiledStateGraph` |
| 定制方式 | 继承 Agent 类、重写方法 | middleware 钩子 |
| 状态管理 | 手动管理 memory | checkpointer 自动持久化 |
| 类型安全 | 弱 | 强（Generic + TypedDict） |

**迁移对照：**

```python
# ❌ 旧版
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
result = executor.invoke({"input": "你好"})

# ✅ 新版
agent = create_agent(model="openai:gpt-4o", tools=tools, system_prompt="...")
result = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
```

---

## 常见问题

**Q1: model 字符串格式支持哪些 provider？**

格式为 `"provider:model_name"`。常用：`openai`（langchain-openai）、`anthropic`（langchain-anthropic）、`google_genai`（langchain-google-genai）。省略 provider 前缀时默认 OpenAI。

**Q2: tools 为空列表和 None 有区别吗？**

没有。两种情况下 Agent 都只有 model 节点，不创建工具循环，模型直接生成回复。

**Q3: 如何在运行时动态切换 system_prompt？**

用 middleware 的 `wrap_model_call` 钩子：

```python
def wrap_model_call(self, request, handler):
    if request.runtime.context.get("language") == "en":
        new_request = request.override(
            system_message=SystemMessage(content="You are a helpful assistant.")
        )
        return handler(new_request)
    return handler(request)
```

**Q4: create_agent 和直接用 LangGraph 有什么区别？**

`create_agent` 是 LangGraph 之上的高层抽象，帮你处理 model/tool 节点定义、条件边连接、middleware 链组装等样板代码。符合标准 model → tools → response 循环时用它更简洁；需要自定义图结构（多分支、并行节点）时直接用 `StateGraph`。

---

## 设计哲学

`create_agent()` 的设计体现了三个核心理念：

1. **约定优于配置** —— 只有 model 必填，其余全有合理默认值。零配置就能跑，需要时再逐步添加参数。

2. **组合优于继承** —— 不需要继承 Agent 类来定制行为，通过 middleware 组合实现任意功能。19 个内置 middleware 覆盖了 90% 的生产需求。

3. **图即 Agent** —— 返回 `CompiledStateGraph` 而非 Agent 对象，意味着 Agent 本质上就是一个状态机。这让多 Agent 协作变成了图的嵌套，概念统一且强大。

---

[来源: `sourcecode/langchain/libs/langchain_v1/langchain/agents/factory.py` 第 658 行]
