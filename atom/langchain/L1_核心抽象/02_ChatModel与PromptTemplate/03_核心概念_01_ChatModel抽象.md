# 核心概念 01: ChatModel 抽象

> **本文目标**: 深入理解 ChatModel 的设计理念、架构实现和 2025-2026 最新特性

---

## 概述

**ChatModel** 是 LangChain 中最核心的抽象之一，它为不同的大语言模型提供了统一的接口。理解 ChatModel 的设计理念和实现细节，是掌握 LangChain 的关键。

**核心定位**: ChatModel 是消息驱动的模型抽象，专为对话场景设计。

---

## 1. ChatModel 的本质

### 1.1 函数式定义

**ChatModel 本质上是一个函数**:

```python
ChatModel: List[Message] → AIMessage
```

用 Python 类型注解表示:

```python
from typing import List
from langchain_core.messages import BaseMessage, AIMessage

class ChatModel:
    def __call__(self, messages: List[BaseMessage]) -> AIMessage:
        """接收消息列表，返回 AI 消息"""
        pass
```

**这个定义揭示了三个关键点**:

1. **输入是列表**: 不是单个消息，而是消息序列
2. **输入是消息对象**: 不是字符串，而是结构化对象
3. **输出是消息对象**: 不是字符串，而是 AIMessage

### 1.2 与 LLM 的本质区别

```python
# LLM 的函数签名
LLM: str → str

# ChatModel 的函数签名
ChatModel: List[Message] → AIMessage
```

**对比表**:

| 维度 | LLM | ChatModel |
|------|-----|-----------|
| **输入类型** | 字符串 | 消息列表 |
| **输出类型** | 字符串 | AIMessage 对象 |
| **角色感知** | 无（需手动格式化） | 有（原生支持） |
| **对话历史** | 手动拼接 | 自然管理 |
| **工具调用** | 不支持 | 原生支持（2025+） |
| **适用场景** | 文本生成、补全 | 对话、Agent |
| **底层模型** | 通常是 Completion API | 通常是 Chat API |

**代码对比**:

```python
# LLM - 字符串拼接（容易出错）
from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct")
prompt = """你是Python专家。

用户: 什么是装饰器？
助手: 装饰器是一种设计模式...
用户: 能举个例子吗？
助手:"""

response = llm.invoke(prompt)
# 问题：
# 1. 手动管理格式（容易出错）
# 2. 没有角色语义
# 3. 难以扩展（添加工具调用等）

# ChatModel - 结构化消息（清晰可靠）
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chatmodel = ChatOpenAI(model="gpt-4o-mini")
messages = [
    SystemMessage(content="你是Python专家"),
    HumanMessage(content="什么是装饰器？"),
    AIMessage(content="装饰器是一种设计模式..."),
    HumanMessage(content="能举个例子吗？")
]

response = chatmodel.invoke(messages)
# 优势：
# 1. 结构化（不易出错）
# 2. 角色明确
# 3. 易于扩展
```

---

## 2. Runnable 接口实现

### 2.1 Runnable 协议

**ChatModel 实现了 Runnable 协议**，这是 LangChain 的核心设计模式。

```python
from langchain_core.runnables import Runnable
from typing import List, Iterator, AsyncIterator

class ChatModel(Runnable[List[BaseMessage], AIMessage]):
    """ChatModel 是 Runnable 的实现"""

    def invoke(
        self,
        input: List[BaseMessage],
        config: Optional[RunnableConfig] = None
    ) -> AIMessage:
        """同步调用"""
        pass

    async def ainvoke(
        self,
        input: List[BaseMessage],
        config: Optional[RunnableConfig] = None
    ) -> AIMessage:
        """异步调用"""
        pass

    def stream(
        self,
        input: List[BaseMessage],
        config: Optional[RunnableConfig] = None
    ) -> Iterator[AIMessage]:
        """流式调用"""
        pass

    async def astream(
        self,
        input: List[BaseMessage],
        config: Optional[RunnableConfig] = None
    ) -> AsyncIterator[AIMessage]:
        """异步流式调用"""
        pass

    def batch(
        self,
        inputs: List[List[BaseMessage]],
        config: Optional[RunnableConfig] = None
    ) -> List[AIMessage]:
        """批量调用"""
        pass

    async def abatch(
        self,
        inputs: List[List[BaseMessage]],
        config: Optional[RunnableConfig] = None
    ) -> List[AIMessage]:
        """异步批量调用"""
        pass
```

### 2.2 Runnable 的价值

**为什么 ChatModel 要实现 Runnable？**

1. **统一接口**: 所有 Runnable 都有相同的方法签名
2. **可组合性**: 可以用 `|` 操作符组合
3. **配置传递**: 统一的配置机制
4. **可观测性**: 统一的回调和追踪

**实际应用**:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 所有组件都是 Runnable
template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}")
])
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# 可以用 | 组合（因为都实现了 Runnable）
chain = template | model | parser

# 统一的调用方式
result = chain.invoke({"role": "助手", "question": "你好"})

# 统一的流式调用
for chunk in chain.stream({"role": "助手", "question": "你好"}):
    print(chunk, end="")

# 统一的批量调用
results = chain.batch([
    {"role": "助手", "question": "问题1"},
    {"role": "助手", "question": "问题2"}
])
```

### 2.3 配置传递机制

**RunnableConfig** 是 Runnable 协议的配置对象:

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    # 回调配置
    callbacks=[...],
    # 标签（用于追踪）
    tags=["production", "chatbot"],
    # 元数据
    metadata={"user_id": "123", "session_id": "abc"},
    # 运行名称
    run_name="customer_support_chat",
    # 最大并发数
    max_concurrency=10,
    # 递归限制
    recursion_limit=25
)

# 配置会自动传递给链中的所有组件
response = chain.invoke(
    {"role": "助手", "question": "你好"},
    config=config
)
```

**2025-2026 新增配置**:

```python
# LangSmith 追踪（2025+）
config = RunnableConfig(
    callbacks=[LangChainTracer()],
    metadata={
        "environment": "production",
        "version": "v2.0",
        "cost_center": "customer_support"
    }
)

# MCP 协议配置（2026+）
config = RunnableConfig(
    configurable={
        "mcp_server": "openai",
        "fallback_servers": ["anthropic", "google"]
    }
)
```

---

## 3. 消息驱动架构

### 3.1 为什么是消息列表？

**问题**: 为什么 ChatModel 接收消息列表，而不是单个消息？

**答案**: 因为对话需要上下文。

```python
# ❌ 错误设计：单个消息
class BadChatModel:
    def invoke(self, message: str) -> str:
        """只接收单个消息 - 无法处理上下文"""
        pass

# 使用时需要手动拼接
history = "用户: 我叫张三\n助手: 你好张三\n用户: 我喜欢Python\n助手: Python很棒\n"
response = bad_model.invoke(history + "用户: 我叫什么名字？")
# 问题：
# 1. 手动管理格式
# 2. 容易出错
# 3. 没有角色语义

# ✅ 正确设计：消息列表
class GoodChatModel:
    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """接收消息列表 - 自然处理上下文"""
        pass

# 使用时结构化管理
messages = [
    HumanMessage(content="我叫张三"),
    AIMessage(content="你好张三"),
    HumanMessage(content="我喜欢Python"),
    AIMessage(content="Python很棒"),
    HumanMessage(content="我叫什么名字？")
]
response = good_model.invoke(messages)
# 优势：
# 1. 结构化
# 2. 角色明确
# 3. 易于管理
```

### 3.2 消息类型系统

**LangChain 的消息类型层次**:

```python
from langchain_core.messages import (
    BaseMessage,      # 基类
    SystemMessage,    # 系统消息
    HumanMessage,     # 用户消息
    AIMessage,        # AI 消息
    ToolMessage,      # 工具消息（2025+）
    FunctionMessage,  # 函数消息（已弃用，用 ToolMessage）
)

# 类型层次
BaseMessage
├── SystemMessage    # 定义行为规则
├── HumanMessage     # 用户输入
├── AIMessage        # AI 回复
│   └── tool_calls   # 工具调用（2025+）
└── ToolMessage      # 工具结果（2025+）
```

**每种消息的作用**:

```python
# SystemMessage - 定义 AI 的角色和行为
SystemMessage(content="你是专业的Python开发者，代码要简洁、有类型注解")

# HumanMessage - 用户的输入
HumanMessage(content="写一个计算斐波那契数列的函数")

# AIMessage - AI 的回复
AIMessage(content="好的，这是一个递归实现：\n```python\ndef fib(n: int) -> int:...")

# AIMessage with tool_calls - AI 决定调用工具（2025+）
AIMessage(
    content="我需要搜索相关信息",
    tool_calls=[
        {
            "id": "call_123",
            "name": "web_search",
            "args": {"query": "Python装饰器"}
        }
    ]
)

# ToolMessage - 工具执行结果（2025+）
ToolMessage(
    content="搜索结果: 装饰器是...",
    tool_call_id="call_123"
)
```

### 3.3 消息的内部结构

**BaseMessage 的核心属性**:

```python
from langchain_core.messages import BaseMessage

class BaseMessage:
    content: str | List[dict]  # 内容（文本或多模态）
    type: str                   # 消息类型
    name: Optional[str]         # 发送者名称
    additional_kwargs: dict     # 额外参数
    response_metadata: dict     # 响应元数据（2025+）
    id: Optional[str]           # 消息 ID（2025+）
```

**实际示例**:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-4o-mini")
response = model.invoke([HumanMessage(content="你好")])

# 查看响应的完整结构
print(f"内容: {response.content}")
print(f"类型: {response.type}")  # 'ai'
print(f"ID: {response.id}")      # 'msg_...'
print(f"元数据: {response.response_metadata}")
# {
#   'token_usage': {'prompt_tokens': 10, 'completion_tokens': 5, ...},
#   'model_name': 'gpt-4o-mini',
#   'finish_reason': 'stop'
# }
```

---

## 4. 2025-2026 新特性

### 4.1 LangChain 1.0 稳定版（2025年10月）

**重大变更**:

1. **中间件机制**: 类似 Express.js 的中间件
2. **MCP 协议集成**: 统一的模型调用协议
3. **改进的类型系统**: 更好的类型推断
4. **性能优化**: 批处理和流式的性能提升

**中间件示例**:

```python
from langchain_core.runnables import RunnableMiddleware

class LoggingMiddleware(RunnableMiddleware):
    """记录所有调用的中间件"""

    def invoke(self, input, config):
        print(f"输入: {input}")
        result = super().invoke(input, config)
        print(f"输出: {result}")
        return result

# 使用中间件
model_with_logging = LoggingMiddleware(model)
response = model_with_logging.invoke(messages)
```

### 4.2 MCP（Model Context Protocol）集成

**MCP 是什么？**

MCP 是 2026 年推出的统一模型调用协议，类似于 GraphQL 之于 REST API。

**优势**:

1. **统一接口**: 一套代码适配所有模型
2. **自动降级**: 主模型失败时自动切换
3. **负载均衡**: 自动分配请求到不同模型
4. **成本优化**: 根据任务复杂度选择模型

**使用示例**:

```python
from langchain_openai import ChatOpenAI

# 配置 MCP
model = ChatOpenAI(
    model="gpt-4o-mini",
    mcp_config={
        "primary": "openai",
        "fallback": ["anthropic", "google"],
        "load_balancing": "round_robin",
        "cost_optimization": True
    }
)

# 自动降级
response = model.invoke(messages)
# 如果 OpenAI 失败，自动切换到 Anthropic
```

### 4.3 结构化输出原生支持

**2025 年之前**: 需要手动解析

```python
# 旧方式（2024）
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()
chain = model | parser
result = chain.invoke(messages)
# 问题：可能解析失败
```

**2025 年之后**: 原生支持

```python
# 新方式（2025+）
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    skills: List[str]

# 直接指定输出结构
structured_model = model.with_structured_output(Person)
result = structured_model.invoke([
    HumanMessage(content="张三，30岁，擅长Python和JavaScript")
])

# result 是 Person 对象，类型安全
print(result.name)    # "张三"
print(result.age)     # 30
print(result.skills)  # ["Python", "JavaScript"]
```

### 4.4 多模态支持增强

**2025-2026 的多模态能力**:

```python
from langchain_core.messages import HumanMessage

# 文本 + 图像
message = HumanMessage(content=[
    {"type": "text", "text": "这张图片是什么？"},
    {"type": "image_url", "image_url": {"url": "https://..."}}
])

# 文本 + 视频（2026+）
message = HumanMessage(content=[
    {"type": "text", "text": "总结这个视频"},
    {"type": "video_url", "video_url": {"url": "https://..."}}
])

# 文本 + 音频（2026+）
message = HumanMessage(content=[
    {"type": "text", "text": "转录这段音频"},
    {"type": "audio_url", "audio_url": {"url": "https://..."}}
])

response = model.invoke([message])
```

---

## 5. 实际应用模式

### 5.1 基础对话模式

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

model = ChatOpenAI(model="gpt-4o-mini")

def chat(question: str) -> str:
    """简单的对话函数"""
    messages = [
        SystemMessage(content="你是友好的助手"),
        HumanMessage(content=question)
    ]
    response = model.invoke(messages)
    return response.content

# 使用
answer = chat("Python 和 JavaScript 有什么区别？")
print(answer)
```

### 5.2 有记忆的对话模式

```python
from typing import List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

class ConversationManager:
    """对话管理器"""

    def __init__(self, model: ChatOpenAI, system_message: str):
        self.model = model
        self.system_message = system_message
        self.history: List[BaseMessage] = []

    def chat(self, user_input: str) -> str:
        """发送消息并更新历史"""
        # 构建消息列表
        messages = [
            SystemMessage(content=self.system_message),
            *self.history,
            HumanMessage(content=user_input)
        ]

        # 调用模型
        response = self.model.invoke(messages)

        # 更新历史
        self.history.append(HumanMessage(content=user_input))
        self.history.append(response)

        return response.content

    def clear_history(self):
        """清空历史"""
        self.history = []

# 使用
manager = ConversationManager(
    model=ChatOpenAI(model="gpt-4o-mini"),
    system_message="你是Python专家"
)

print(manager.chat("我叫张三"))
# 输出: 你好张三！

print(manager.chat("我叫什么名字？"))
# 输出: 你叫张三
```

### 5.3 流式对话模式

```python
def stream_chat(question: str):
    """流式对话"""
    messages = [
        SystemMessage(content="你是助手"),
        HumanMessage(content=question)
    ]

    print("AI: ", end="", flush=True)
    for chunk in model.stream(messages):
        print(chunk.content, end="", flush=True)
    print()  # 换行

# 使用
stream_chat("讲一个关于Python的笑话")
# 输出: AI: 为什么Python程序员喜欢大自然？因为他们喜欢蟒蛇（Python）！
```

### 5.4 批量处理模式

```python
def batch_chat(questions: List[str]) -> List[str]:
    """批量处理问题"""
    messages_list = [
        [
            SystemMessage(content="你是助手"),
            HumanMessage(content=q)
        ]
        for q in questions
    ]

    responses = model.batch(messages_list)
    return [r.content for r in responses]

# 使用
questions = [
    "什么是Python？",
    "什么是JavaScript？",
    "什么是Rust？"
]
answers = batch_chat(questions)
for q, a in zip(questions, answers):
    print(f"Q: {q}\nA: {a}\n")
```

---

## 6. 性能优化策略

### 6.1 选择合适的调用方式

```python
# 场景1: 简单问答 - 用 invoke
response = model.invoke(messages)

# 场景2: 聊天界面 - 用 stream
for chunk in model.stream(messages):
    send_to_ui(chunk.content)

# 场景3: 批量评估 - 用 batch
responses = model.batch(messages_list)

# 场景4: 异步场景 - 用 ainvoke
response = await model.ainvoke(messages)
```

### 6.2 批处理降低成本（2025+）

```python
# OpenAI 批处理 API（成本降低 50%）
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o-mini",
    batch_mode=True  # 启用批处理模式
)

# 批量处理（自动使用批处理 API）
responses = model.batch(messages_list)
# 成本降低 50%，但延迟增加（适合离线任务）
```

### 6.3 缓存策略

```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

# 启用缓存
set_llm_cache(InMemoryCache())

# 相同的输入会从缓存返回
response1 = model.invoke(messages)  # 调用 API
response2 = model.invoke(messages)  # 从缓存返回（免费）
```

### 6.4 Token 计数和成本追踪

```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    response = model.invoke(messages)
    print(f"Tokens: {cb.total_tokens}")
    print(f"Cost: ${cb.total_cost}")
```

---

## 检查清单

完成本节学习后，你应该能够：

- [ ] 解释 ChatModel 的函数式定义
- [ ] 说明 ChatModel 和 LLM 的本质区别
- [ ] 理解 Runnable 协议的价值
- [ ] 使用 invoke/stream/batch 方法
- [ ] 理解消息驱动架构的优势
- [ ] 了解 2025-2026 的新特性（MCP、结构化输出）
- [ ] 实现基础的对话管理器
- [ ] 选择合适的性能优化策略

---

**下一步**: 阅读 `03_核心概念_02_消息类型系统.md` 深入理解消息类型的设计和使用
