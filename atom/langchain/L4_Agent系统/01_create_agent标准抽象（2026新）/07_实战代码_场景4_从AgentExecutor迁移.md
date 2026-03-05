# 实战代码 场景4：从 AgentExecutor 迁移到 create_agent

> 手把手演示四种典型旧版 Agent 代码如何迁移到新版 create_agent——旧版新版左右对照，一看就懂

---

## 场景概述

LangChain 1.0 用 `create_agent()` 统一取代了旧版的 `AgentExecutor` + 六种 `create_xxx_agent` 函数。本文只给代码，理论见 [03_核心概念_6](./03_核心概念_6_与旧版AgentExecutor对比迁移.md)。

---

## 环境准备

```python
# pip install langchain langchain-openai langchain-classic langgraph python-dotenv
from dotenv import load_dotenv
load_dotenv()
```

---

## 案例 1：基础 ReAct Agent 迁移

**迁移难度：** 低

### 旧版代码（create_react_agent + AgentExecutor）

```python
"""旧版：两步创建、需要 Hub prompt、需要 @tool 装饰器"""
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.tools import tool

llm = ChatOpenAI(model="gpt-4", temperature=0)

@tool
def search(query: str) -> str:
    """搜索网络信息"""
    return f"搜索结果: 关于 '{query}' 的最新信息..."

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算错误: {e}"

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, [search, calculate], prompt)
executor = AgentExecutor(
    agent=agent, tools=[search, calculate],
    max_iterations=10, verbose=True, handle_parsing_errors=True,
)

result = executor.invoke({"input": "Python 3.13 有什么新特性？"})
print(result["output"])
```

### 新版代码（create_agent）

```python
"""新版：一步到位，不需要 Hub prompt、不需要 @tool 装饰器"""
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolRetryMiddleware


def search(query: str) -> str:
    """搜索网络信息。

    Args:
        query: 搜索关键词。
    """
    return f"搜索结果: 关于 '{query}' 的最新信息..."


def calculate(expression: str) -> str:
    """计算数学表达式。

    Args:
        expression: 合法的数学表达式，如 '2 + 3 * 4'。
    """
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算错误: {e}"


agent = create_agent(
    model="openai:gpt-4o",                # 字符串标识符，不用实例化
    tools=[search, calculate],             # 普通函数，自动转换
    system_prompt="你是一个有用的助手，可以搜索信息和做数学计算。用中文回答。",
    middleware=[
        ModelCallLimitMiddleware(max_calls=10),  # 替代 max_iterations
        ToolRetryMiddleware(max_retries=3),      # 替代 handle_parsing_errors
    ],
    debug=True,                            # 替代 verbose
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Python 3.13 有什么新特性？"}]
})
print(result["messages"][-1].content)
```

### 逐行对照

| 旧版 | 新版 | 说明 |
|------|------|------|
| `ChatOpenAI(model="gpt-4")` | `model="openai:gpt-4o"` | 字符串标识符 |
| `@tool` 装饰器 | 普通函数 | 自动根据类型注解 + docstring 转换 |
| `hub.pull("hwchase17/react")` | `system_prompt="..."` | 不再需要外部 prompt |
| `create_react_agent()` + `AgentExecutor()` | `create_agent()` | 一步到位 |
| `max_iterations=10` | `ModelCallLimitMiddleware(max_calls=10)` | Middleware 模式 |
| `handle_parsing_errors=True` | `ToolRetryMiddleware(max_retries=3)` | 更强大 |
| `{"input": "问题"}` | `{"messages": [{"role": "user", "content": "问题"}]}` | 消息格式 |
| `result["output"]` | `result["messages"][-1].content` | 输出方式 |

---

## 案例 2：带 Memory 的对话 Agent 迁移

**迁移难度：** 中

### 旧版代码（ConversationBufferMemory）

```python
"""旧版：Memory 和 Agent 循环割裂，需要手动配置 memory_key"""
from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

llm = ChatOpenAI(model="gpt-4")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有记忆的助手。"),
    MessagesPlaceholder(variable_name="chat_history"),       # Memory 注入点
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),   # Agent 内部状态
])

@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"搜索结果: {query}"

agent = create_openai_functions_agent(llm, [search], prompt)
executor = AgentExecutor(agent=agent, tools=[search], memory=memory)

result1 = executor.invoke({"input": "我叫小明，今年25岁"})
result2 = executor.invoke({"input": "我叫什么名字？今年多大？"})
# 问题：不同用户的对话无法隔离，需要手动管理多个 Memory 实例
```

### 新版代码（checkpointer）

```python
"""新版：checkpointer 一行搞定，thread_id 天然隔离不同会话"""
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


def search(query: str) -> str:
    """搜索信息。

    Args:
        query: 搜索关键词。
    """
    return f"搜索结果: {query}"


agent = create_agent(
    model="openai:gpt-4o",
    tools=[search],
    system_prompt="你是一个有记忆的助手。记住用户告诉你的所有信息。",
    checkpointer=InMemorySaver(),  # 一行替代整个 Memory 配置
)

# 用户 A 的对话
config_a = {"configurable": {"thread_id": "user-xiaoming"}}
result1 = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫小明，今年25岁"}]}, config_a,
)
result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么名字？"}]}, config_a,
)
print(result2["messages"][-1].content)  # "你叫小明，今年25岁。"

# 用户 B 的对话（完全隔离）
config_b = {"configurable": {"thread_id": "user-xiaohong"}}
result3 = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么名字？"}]}, config_b,
)
print(result3["messages"][-1].content)  # "你还没有告诉我你的名字"
```

### 核心变化

| 旧版 | 新版 | 为什么更好 |
|------|------|-----------|
| `ConversationBufferMemory(memory_key=...)` | `checkpointer=InMemorySaver()` | 一行代码 |
| `MessagesPlaceholder("chat_history")` | 不需要 | 消息历史自动管理 |
| `MessagesPlaceholder("agent_scratchpad")` | 不需要 | Agent 内部状态自动管理 |
| 手动管理多个 Memory 实例 | `thread_id` 自动隔离 | 天然支持多用户 |

---

## 案例 3：自定义 Agent 逻辑迁移（继承重写 → Middleware）

**迁移难度：** 高

### 旧版代码（继承 AgentExecutor）

```python
"""旧版：继承 AgentExecutor 重写 _call 实现后处理（敏感词过滤 + 免责声明）"""
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

SENSITIVE_WORDS = ["暴力", "赌博", "违法"]
DISCLAIMER = "\n\n---\n⚠️ 免责声明：以上信息仅供参考，不构成专业建议。"


class PostProcessingExecutor(AgentExecutor):
    """继承 AgentExecutor，重写 _call 添加后处理"""
    def _call(self, inputs, run_manager=None):
        result = super()._call(inputs, run_manager=run_manager)
        output = result["output"]
        for word in SENSITIVE_WORDS:
            output = output.replace(word, "***")
        result["output"] = output + DISCLAIMER
        return result


llm = ChatOpenAI(model="gpt-4")
@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"搜索结果: {query}"

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个金融助手。"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
agent = create_tool_calling_agent(llm, [search], prompt)
executor = PostProcessingExecutor(agent=agent, tools=[search])
result = executor.invoke({"input": "比特币最新价格是多少？"})
```

### 新版代码（after_model Middleware）

```python
"""新版：Middleware 实现后处理，不需要继承，组合式扩展"""
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

SENSITIVE_WORDS = ["暴力", "赌博", "违法"]
DISCLAIMER = "\n\n---\n⚠️ 免责声明：以上信息仅供参考，不构成专业建议。"


def search(query: str) -> str:
    """搜索信息。

    Args:
        query: 搜索关键词。
    """
    return f"搜索结果: {query}"


class PostProcessingMiddleware(AgentMiddleware):
    """Agent 输出后处理：敏感词过滤 + 免责声明"""

    def after_model(self, state, runtime):
        last_msg = state["messages"][-1]
        # 只处理最终回答（没有 tool_calls 的 AIMessage）
        if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
            content = last_msg.content
            for word in SENSITIVE_WORDS:
                content = content.replace(word, "***")
            content += DISCLAIMER
            return {"messages": [AIMessage(content=content, id=last_msg.id)]}
        return None


agent = create_agent(
    model="openai:gpt-4o",
    tools=[search],
    system_prompt="你是一个金融助手。用中文回答。",
    middleware=[PostProcessingMiddleware()],
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "比特币最新价格是多少？"}]
})
print(result["messages"][-1].content)
```

### 对比

| 维度 | 旧版（继承 AgentExecutor） | 新版（Middleware） |
|------|--------------------------|-------------------|
| 实现方式 | 继承类，重写 `_call` | 实现 `after_model` 钩子 |
| 侵入性 | 高（修改执行流程） | 低（插件式组合） |
| 可组合性 | 差（只能继承一次） | 好（多个 Middleware 叠加） |
| 可测试性 | 差（依赖 AgentExecutor） | 好（Middleware 可独立测试） |

---

## 案例 4：从 langgraph.prebuilt.create_react_agent 迁移

**迁移难度：** 极低（改三行）

```python
# ❌ 旧版
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[get_weather, calculate],
    state_modifier="你是一个智能助手，用中文回答。",
)

# ✅ 新版
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4o",                          # 支持字符串
    tools=[get_weather, calculate],
    system_prompt="你是一个智能助手，用中文回答。",    # state_modifier → system_prompt
)
```

三处变化：`import` 路径、`state_modifier` → `system_prompt`、模型可用字符串。

---

## 参数映射速查表

| 旧版参数 | 新版等价 | 备注 |
|----------|----------|------|
| `max_iterations=10` | `ModelCallLimitMiddleware(max_calls=10)` | Middleware 模式 |
| `max_execution_time=120` | 自定义 Middleware（见下方） | 暂无内置 |
| `verbose=True` | `debug=True` | 调试输出 |
| `handle_parsing_errors=True` | `ToolRetryMiddleware(max_retries=3)` | 更强大 |
| `early_stopping_method="force"` | `ModelCallLimitMiddleware` 默认行为 | 达到限制直接停止 |
| `memory=ConversationBufferMemory(...)` | `checkpointer=InMemorySaver()` | 一行替代 |
| `return_intermediate_steps=True` | `stream_mode="updates"` | 流式获取每步结果 |
| `state_modifier="..."` | `system_prompt="..."` | langgraph prebuilt 专用 |

### 自定义超时 Middleware

```python
"""替代旧版 max_execution_time"""
import time
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage


class TimeoutMiddleware(AgentMiddleware):
    def __init__(self, max_seconds: float = 120.0):
        self._max_seconds = max_seconds
        self._start_time: float | None = None

    def before_model(self, state, runtime):
        now = time.time()
        if self._start_time is None:
            self._start_time = now
        if now - self._start_time > self._max_seconds:
            return {"messages": [AIMessage(content="处理超时，请简化问题后重试。")]}
        return None
```

---

## 常见迁移问题 FAQ

### Q1: agent_scratchpad 去哪了？

新版中 `AgentState.messages` 统一管理所有消息（用户输入、AI 回复、工具调用、工具结果），不再需要单独的 `agent_scratchpad` 占位符。

```python
# ❌ 旧版
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手"), ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),  # 必须有
])

# ✅ 新版：不需要，messages 列表自动管理
agent = create_agent(model="openai:gpt-4o", tools=tools, system_prompt="你是助手")
```

### Q2: 如何保持旧版输入/输出格式？

写一个薄适配层作为过渡：

```python
"""适配层：兼容旧版 {"input": "..."} / result["output"] 格式"""
from langchain.agents import create_agent


def create_compatible_agent(**kwargs):
    agent = create_agent(**kwargs)

    class CompatibleAgent:
        def invoke(self, inputs: dict) -> dict:
            if "input" in inputs and "messages" not in inputs:
                new_input = {"messages": [{"role": "user", "content": inputs["input"]}]}
            else:
                new_input = inputs
            result = agent.invoke(new_input)
            return {"output": result["messages"][-1].content, "messages": result["messages"]}

    return CompatibleAgent()


# 使用：和旧版一样
agent = create_compatible_agent(model="openai:gpt-4o", tools=[], system_prompt="你是助手")
result = agent.invoke({"input": "你好"})
print(result["output"])
```

### Q3: 自定义 OutputParser 怎么迁移？

| 旧版用途 | 新版替代 |
|----------|----------|
| 结构化输出（JSON/Pydantic） | `response_format=MyModel` 参数 |
| 输出后处理（过滤/格式化） | `after_model` Middleware |
| 输出验证 | `wrap_model_call` Middleware |

```python
"""用 response_format 替代 OutputParser"""
from pydantic import BaseModel
from langchain.agents import create_agent


class WeatherReport(BaseModel):
    city: str
    temperature: float
    condition: str
    suggestion: str


agent = create_agent(
    model="openai:gpt-4o", tools=[],
    system_prompt="根据用户问题生成天气报告。",
    response_format=WeatherReport,  # 直接传 Pydantic 模型
)

result = agent.invoke({"messages": [{"role": "user", "content": "北京天气？"}]})
report = result["structured_response"]
print(f"{report.city}: {report.temperature}°C, {report.condition}")
```

---

## 迁移检查清单

- [ ] 安装新版包：`pip install langchain langchain-classic langgraph`
- [ ] 旧代码加兼容：`from langchain.agents` → `from langchain_classic.agents`
- [ ] 替换 Agent 创建：`create_xxx_agent()` + `AgentExecutor()` → `create_agent()`
- [ ] 替换 Prompt：`ChatPromptTemplate` → `system_prompt` 字符串
- [ ] 替换 Memory：`ConversationBufferMemory` → `checkpointer=InMemorySaver()`
- [ ] 替换行为参数：`max_iterations` → `ModelCallLimitMiddleware`
- [ ] 替换错误处理：`handle_parsing_errors` → `ToolRetryMiddleware`
- [ ] 替换自定义逻辑：继承重写 → Middleware
- [ ] 更新输入格式：`{"input": "..."}` → `{"messages": [...]}`
- [ ] 更新输出格式：`result["output"]` → `result["messages"][-1].content`
- [ ] 移除 `@tool` 装饰器：普通函数 + 类型注解 + docstring 即可
- [ ] 运行测试确认功能正常
- [ ] 全部迁移完成后卸载 `langchain-classic`

---

**下一步：** 回顾 [03_核心概念_6_与旧版AgentExecutor对比迁移](./03_核心概念_6_与旧版AgentExecutor对比迁移.md) 了解架构层面的本质变化

---

[来源: reference/search_create_agent_01.md]
[来源: reference/fetch_1dot0_blog_02.md]
[来源: reference/source_create_agent_01.md]