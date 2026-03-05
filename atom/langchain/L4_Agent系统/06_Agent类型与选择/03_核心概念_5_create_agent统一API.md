# 核心概念 5: create_agent() 统一 API

> **2026 年推荐**: 使用 `create_agent()` 作为创建 Agent 的首选方式

---

## 什么是 create_agent()

`create_agent()` 是 LangChain 在 2026 年推出的**统一 Agent 创建接口**，它封装了所有 Agent 类型的创建逻辑，自动根据模型能力选择最佳的 Agent 实现。

### 核心特点

1. **自动类型选择**: 根据模型能力自动选择 OpenAI Functions / ReAct / Tool Calling
2. **统一接口**: 一个函数搞定所有 Agent 类型
3. **模型无关**: 支持 OpenAI、Anthropic、开源模型
4. **简化开发**: 无需手动判断使用哪种 Agent 类型
5. **向后兼容**: 底层仍使用 `create_openai_functions_agent()` 等函数

### 基本用法

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# 只需 3 个参数
agent = create_agent(
    model=ChatOpenAI(model="gpt-4"),
    tools=[search_tool, calculator_tool],
    system_prompt="You are a helpful assistant."
)
```

**对比传统方式**:
```python
# 传统方式：需要手动选择类型
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
```

---

## 为什么需要统一 API

### 问题 1: Agent 类型碎片化

在 `create_agent()` 出现之前，LangChain 有多个 Agent 创建函数：

```python
# 函数调用型
create_openai_functions_agent(llm, tools, prompt)

# 推理-行动型
create_react_agent(llm, tools, prompt)

# 结构化对话型
create_structured_chat_agent(llm, tools, prompt)

# 工具调用型
create_tool_calling_agent(llm, tools, prompt)

# 旧版初始化（已弃用）
initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS)
```

**开发者困惑**:
- 我应该用哪个函数？
- 我的模型支持哪种 Agent 类型？
- 如何在不同类型之间切换？

### 问题 2: 模型能力判断复杂

不同模型支持不同的 Agent 类型：

| 模型 | 支持函数调用 | 推荐 Agent 类型 |
|------|-------------|----------------|
| GPT-4 | ✅ | OpenAI Functions / Tool Calling |
| Claude 3 | ✅ | OpenAI Functions / Tool Calling |
| Llama 3 | ❌ | ReAct |
| Mistral | ⚠️ 部分支持 | ReAct (安全选择) |

**开发者需要手动判断**:
```python
# 手动判断逻辑
if model.supports_function_calling():
    agent = create_openai_functions_agent(...)
else:
    agent = create_react_agent(...)
```

### 问题 3: 迁移成本高

当需要切换 Agent 类型时，需要修改多处代码：

```python
# 从 OpenAI Functions 切换到 ReAct
# Before
agent = create_openai_functions_agent(llm, tools, prompt)

# After
agent = create_react_agent(llm, tools, prompt)
# 还需要修改 prompt 格式！
```

### create_agent() 的解决方案

```python
# 统一接口，自动选择
agent = create_agent(model, tools, system_prompt)

# 切换模型？无需修改代码
model = ChatOpenAI(model="gpt-4")  # 自动使用 OpenAI Functions
model = ChatAnthropic(model="claude-3")  # 自动使用 Tool Calling
model = ChatOllama(model="llama3")  # 自动使用 ReAct
```

---

## 工作原理

### 自动类型选择逻辑

`create_agent()` 内部实现了智能选择逻辑：

```python
def create_agent(model, tools, system_prompt=None):
    """
    自动选择最佳 Agent 类型
    """
    # 1. 检查模型是否支持工具调用（Tool Calling）
    if hasattr(model, 'bind_tools'):
        # 优先使用 Tool Calling（2026 推荐）
        return create_tool_calling_agent(model, tools, system_prompt)

    # 2. 检查模型是否支持函数调用（Function Calling）
    elif hasattr(model, 'bind_functions'):
        # 使用 OpenAI Functions
        return create_openai_functions_agent(model, tools, system_prompt)

    # 3. 回退到 ReAct（兼容所有模型）
    else:
        return create_react_agent(model, tools, system_prompt)
```

### 模型能力检测

LangChain 通过以下方式检测模型能力：

1. **检查方法存在性**: `hasattr(model, 'bind_tools')`
2. **检查模型元数据**: `model.model_name` 匹配已知模型列表
3. **尝试调用**: 发送测试请求，检查响应格式

### Prompt 自动适配

不同 Agent 类型需要不同的 prompt 格式：

```python
# OpenAI Functions: 需要 agent_scratchpad
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# ReAct: 需要 tools 和 tool_names
prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:
Question: the input question
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
...
""")
```

**create_agent() 自动处理**:
```python
# 你只需提供简单的 system_prompt
agent = create_agent(
    model,
    tools,
    system_prompt="You are a helpful assistant."
)
# 内部自动生成适配的 prompt
```

---

## 与 create_*_agent() 的关系

### 层次关系

```
create_agent()  (统一接口)
    ├─ create_tool_calling_agent()  (2026 推荐)
    ├─ create_openai_functions_agent()  (稳定)
    ├─ create_react_agent()  (兼容性)
    └─ create_structured_chat_agent()  (复杂工具)
```

### 何时直接使用 create_*_agent()

虽然 `create_agent()` 是推荐方式，但以下场景应直接使用具体函数：

#### 1. 需要精细控制 Prompt

```python
# 自定义 ReAct prompt 格式
from langchain.agents import create_react_agent
from langchain_core.prompts import PromptTemplate

custom_prompt = PromptTemplate.from_template("""
你是一个专业的数据分析师。

可用工具:
{tools}

格式要求:
问题: 用户的问题
思考: 分析问题，制定计划
行动: 选择工具 [{tool_names}]
观察: 工具返回结果
... (重复思考-行动-观察)
最终答案: 给出结论

问题: {input}
{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, custom_prompt)
```

#### 2. 强制使用特定类型

```python
# 即使模型支持函数调用，也强制使用 ReAct（用于调试）
from langchain.agents import create_react_agent

agent = create_react_agent(
    ChatOpenAI(model="gpt-4"),  # 支持函数调用
    tools,
    prompt  # 但我想看到显式的推理过程
)
```

#### 3. 使用 Structured Chat 处理复杂工具

```python
# 工具有多个输入参数
from langchain.agents import create_structured_chat_agent

complex_tool = StructuredTool.from_function(
    func=lambda query, filters, limit: search(query, filters, limit),
    name="advanced_search",
    description="Search with filters",
    args_schema=SearchInput  # 复杂的 Pydantic schema
)

agent = create_structured_chat_agent(llm, [complex_tool], prompt)
```

### 性能对比

| 方式 | 开发速度 | 灵活性 | 性能 | 推荐场景 |
|------|---------|--------|------|---------|
| `create_agent()` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 快速开发、原型验证 |
| `create_tool_calling_agent()` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 生产环境、最新模型 |
| `create_openai_functions_agent()` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 稳定性优先 |
| `create_react_agent()` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 开源模型、调试 |
| `create_structured_chat_agent()` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 复杂工具参数 |

---

## 完整代码示例

### 场景 1: 基础使用（推荐）

```python
from langchain.agents import create_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# 1. 定义工具
@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

@tool
def calculator(expression: str) -> float:
    """Calculate mathematical expressions."""
    return eval(expression)

# 2. 创建 Agent（自动选择类型）
model = ChatOpenAI(model="gpt-4", temperature=0)
tools = [search, calculator]

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="You are a helpful research assistant."
)

# 3. 创建 Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

# 4. 运行
result = agent_executor.invoke({
    "input": "What is the population of Tokyo multiplied by 2?"
})
print(result["output"])
```

### 场景 2: 多模型切换

```python
from langchain.agents import create_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama

# 定义工具（复用）
tools = [search, calculator]
system_prompt = "You are a helpful assistant."

# 方式 1: OpenAI GPT-4（自动使用 Tool Calling）
model_openai = ChatOpenAI(model="gpt-4")
agent_openai = create_agent(model_openai, tools, system_prompt)

# 方式 2: Anthropic Claude（自动使用 Tool Calling）
model_claude = ChatAnthropic(model="claude-3-opus-20240229")
agent_claude = create_agent(model_claude, tools, system_prompt)

# 方式 3: 开源 Llama（自动使用 ReAct）
model_llama = ChatOllama(model="llama3")
agent_llama = create_agent(model_llama, tools, system_prompt)

# 无需修改其他代码，直接使用
executor = AgentExecutor(agent=agent_openai, tools=tools)
# 或
executor = AgentExecutor(agent=agent_claude, tools=tools)
# 或
executor = AgentExecutor(agent=agent_llama, tools=tools)
```

### 场景 3: 自定义 System Prompt

```python
from langchain.agents import create_agent, AgentExecutor

# 详细的 system prompt
system_prompt = """
You are a professional data analyst with expertise in:
- Statistical analysis
- Data visualization
- Business intelligence

Guidelines:
1. Always verify data sources before analysis
2. Provide confidence intervals for estimates
3. Explain your reasoning step by step
4. Use tools when necessary

Available tools:
- search: Find information online
- calculator: Perform calculations
- python_repl: Execute Python code
"""

agent = create_agent(
    model=ChatOpenAI(model="gpt-4"),
    tools=[search, calculator, python_repl],
    system_prompt=system_prompt
)

executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

### 场景 4: 带记忆的 Agent

```python
from langchain.agents import create_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder

# 创建记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 创建 Agent（自动适配记忆）
agent = create_agent(
    model=ChatOpenAI(model="gpt-4"),
    tools=[search, calculator],
    system_prompt="You are a helpful assistant with memory."
)

# 创建 Executor（注入记忆）
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# 多轮对话
executor.invoke({"input": "My name is Alice"})
executor.invoke({"input": "What's my name?"})  # 记住之前的对话
```

### 场景 5: 错误处理与重试

```python
from langchain.agents import create_agent, AgentExecutor
from langchain_core.callbacks import StdOutCallbackHandler

# 创建 Agent
agent = create_agent(
    model=ChatOpenAI(model="gpt-4", temperature=0),
    tools=[search, calculator],
    system_prompt="You are a helpful assistant."
)

# 配置 Executor（错误处理）
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,  # 最多迭代 10 次
    max_execution_time=60,  # 最多执行 60 秒
    handle_parsing_errors=True,  # 自动处理解析错误
    callbacks=[StdOutCallbackHandler()]  # 输出日志
)

# 运行（自动重试）
try:
    result = executor.invoke({"input": "Complex query..."})
except Exception as e:
    print(f"Agent failed: {e}")
```

---

## 迁移路径

### 从 initialize_agent() 迁移

#### Before (旧版 API - 已弃用)

```python
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
tools = [search, calculator]

# 旧版方式
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,  # 手动指定类型
    verbose=True,
    max_iterations=5
)

result = agent_executor.run("What is 2+2?")
```

#### After (新版 API - 推荐)

```python
from langchain.agents import create_agent, AgentExecutor
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4")
tools = [search, calculator]

# 新版方式
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="You are a helpful assistant."  # 可选
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

result = agent_executor.invoke({"input": "What is 2+2?"})
print(result["output"])
```

**关键变化**:
1. `initialize_agent()` → `create_agent()` + `AgentExecutor()`
2. `agent=AgentType.OPENAI_FUNCTIONS` → 自动选择
3. `.run()` → `.invoke()`
4. 返回字符串 → 返回字典 `{"output": "..."}`

### 从 create_openai_functions_agent() 迁移

#### Before (手动选择类型)

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
```

#### After (自动选择类型)

```python
from langchain.agents import create_agent, AgentExecutor

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant."
)
executor = AgentExecutor(agent=agent, tools=tools)
```

**优势**:
- 代码减少 50%
- 无需手动构建 prompt
- 自动适配不同模型

### 迁移检查清单

- [ ] 替换 `initialize_agent()` 为 `create_agent()`
- [ ] 移除 `AgentType` 枚举
- [ ] 将 `llm` 参数改为 `model`
- [ ] 将 `agent` 参数改为 `system_prompt`
- [ ] 更新调用方式: `.run()` → `.invoke()`
- [ ] 更新返回值处理: 字符串 → 字典
- [ ] 测试所有 Agent 功能
- [ ] 更新错误处理逻辑

---

## 何时不用 create_agent()

### 场景 1: 需要自定义 Prompt 格式

```python
# 不推荐：create_agent() 无法自定义 prompt 结构
agent = create_agent(model, tools, system_prompt)

# 推荐：直接使用 create_react_agent()
from langchain.agents import create_react_agent
from langchain_core.prompts import PromptTemplate

custom_prompt = PromptTemplate.from_template("""
自定义格式...
{tools}
{tool_names}
{input}
{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, custom_prompt)
```

### 场景 2: 需要强制使用特定 Agent 类型

```python
# 不推荐：create_agent() 自动选择，无法强制
agent = create_agent(model, tools, system_prompt)

# 推荐：明确指定类型
from langchain.agents import create_react_agent

# 即使模型支持函数调用，也强制使用 ReAct（用于调试）
agent = create_react_agent(
    ChatOpenAI(model="gpt-4"),
    tools,
    prompt
)
```

### 场景 3: 需要使用 Structured Chat Agent

```python
# create_agent() 不会自动选择 Structured Chat
# 需要手动使用
from langchain.agents import create_structured_chat_agent

agent = create_structured_chat_agent(llm, tools, prompt)
```

### 场景 4: 需要精细控制 Agent 行为

```python
# 不推荐：create_agent() 封装了太多细节
agent = create_agent(model, tools, system_prompt)

# 推荐：手动构建 Agent
from langchain.agents import create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "Custom system message with {variable}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
```

---

## 最佳实践

### 1. 优先使用 create_agent()

```python
# ✅ 推荐：快速开发
agent = create_agent(model, tools, system_prompt)

# ❌ 不推荐：手动选择（除非有特殊需求）
agent = create_openai_functions_agent(llm, tools, prompt)
```

### 2. 使用环境变量管理 API Key

```python
import os
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量读取
)
```

### 3. 启用 Verbose 模式调试

```python
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 输出详细日志
    return_intermediate_steps=True  # 返回中间步骤
)

result = executor.invoke({"input": "..."})
print(result["intermediate_steps"])  # 查看 Agent 推理过程
```

### 4. 设置合理的限制

```python
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,  # 防止无限循环
    max_execution_time=60,  # 防止超时
    handle_parsing_errors=True  # 自动处理错误
)
```

### 5. 使用 LangSmith 监控

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_langsmith_key"

# 自动上传到 LangSmith
executor.invoke({"input": "..."})
```

### 6. 编写单元测试

```python
import pytest
from langchain.agents import create_agent, AgentExecutor

def test_agent_basic():
    agent = create_agent(model, tools, system_prompt)
    executor = AgentExecutor(agent=agent, tools=tools)

    result = executor.invoke({"input": "What is 2+2?"})
    assert "4" in result["output"]

def test_agent_tool_usage():
    agent = create_agent(model, tools, system_prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        return_intermediate_steps=True
    )

    result = executor.invoke({"input": "Search for Python"})
    assert len(result["intermediate_steps"]) > 0
    assert result["intermediate_steps"][0][0].tool == "search"
```

### 7. 处理多语言

```python
# 中文 Agent
agent_zh = create_agent(
    model=ChatOpenAI(model="gpt-4"),
    tools=tools,
    system_prompt="你是一个有帮助的助手。"
)

# 英文 Agent
agent_en = create_agent(
    model=ChatOpenAI(model="gpt-4"),
    tools=tools,
    system_prompt="You are a helpful assistant."
)
```

### 8. 使用类型提示

```python
from typing import List
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel

def create_my_agent(
    model: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: str
) -> AgentExecutor:
    agent = create_agent(model, tools, system_prompt)
    return AgentExecutor(agent=agent, tools=tools)
```

---

## 常见问题

### Q1: create_agent() 和 initialize_agent() 有什么区别？

**A**:
- `initialize_agent()` 是旧版 API（已弃用），需要手动指定 `AgentType`
- `create_agent()` 是新版 API（2026 推荐），自动选择最佳类型
- `create_agent()` 返回 Agent 对象，需要配合 `AgentExecutor` 使用
- `initialize_agent()` 直接返回 `AgentExecutor`

### Q2: create_agent() 如何选择 Agent 类型？

**A**: 按以下优先级：
1. 检查模型是否支持 Tool Calling → 使用 `create_tool_calling_agent()`
2. 检查模型是否支持 Function Calling → 使用 `create_openai_functions_agent()`
3. 回退到 ReAct → 使用 `create_react_agent()`

### Q3: 可以强制 create_agent() 使用特定类型吗？

**A**: 不可以。如果需要强制使用特定类型，应直接调用对应的 `create_*_agent()` 函数。

### Q4: create_agent() 支持自定义 Prompt 吗？

**A**: 仅支持简单的 `system_prompt` 字符串。如需复杂 prompt 结构，应使用 `create_*_agent()`。

### Q5: create_agent() 性能如何？

**A**: 与直接使用 `create_*_agent()` 性能相同，因为它只是一个封装层。

---

## 总结

### 核心要点

1. **create_agent() 是 2026 年推荐的统一 API**
2. **自动根据模型能力选择最佳 Agent 类型**
3. **简化开发流程，减少 50% 代码量**
4. **支持所有主流模型（OpenAI、Anthropic、开源）**
5. **需要精细控制时，仍可使用 create_*_agent()**

### 使用建议

| 场景 | 推荐方式 |
|------|---------|
| 快速开发、原型验证 | `create_agent()` |
| 生产环境、标准工具 | `create_agent()` |
| 自定义 Prompt 格式 | `create_*_agent()` |
| 强制使用特定类型 | `create_*_agent()` |
| 复杂工具参数 | `create_structured_chat_agent()` |
| 调试、可解释性 | `create_react_agent()` |

### 下一步

- 学习 **Agent 选择决策树**（下一个核心概念）
- 实践 **多种场景下的 Agent 创建**（实战代码）
- 掌握 **Agent 故障排查与类型切换**（实战代码）

---

**参考资料**:
- [来源: reference/context7_langchain_agent_types_01.md | LangChain 官方文档]
- [来源: reference/source_agent_init_02.md | LangChain 源码]
