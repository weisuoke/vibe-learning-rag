# 核心概念 06: Runnable 方法

> **本文目标**: 深入掌握 Runnable 协议的所有调用方法和使用场景

---

## 概述

**Runnable 协议**是 LangChain 的核心设计模式，定义了统一的调用接口。ChatModel 和 PromptTemplate 都实现了 Runnable 协议，因此它们都支持相同的调用方法。

**核心定位**: Runnable 方法是与 LangChain 组件交互的标准接口。

---

## 1. Runnable 协议概览

### 1.1 核心方法

**Runnable 协议定义了 6 个核心方法**:

```python
from langchain_core.runnables import Runnable

class Runnable:
    # 同步方法
    def invoke(self, input, config=None):
        """同步调用，返回单个结果"""
        pass

    def batch(self, inputs, config=None):
        """批量调用，返回多个结果"""
        pass

    def stream(self, input, config=None):
        """流式调用，返回迭代器"""
        pass

    # 异步方法
    async def ainvoke(self, input, config=None):
        """异步调用，返回单个结果"""
        pass

    async def abatch(self, inputs, config=None):
        """异步批量调用，返回多个结果"""
        pass

    async def astream(self, input, config=None):
        """异步流式调用，返回异步迭代器"""
        pass
```

### 1.2 方法分类

**按执行方式分类**:

| 方法 | 执行方式 | 返回类型 | 适用场景 |
|------|----------|----------|----------|
| **invoke** | 同步单次 | 单个结果 | 简单问答、测试 |
| **batch** | 同步批量 | 结果列表 | 批量处理、评估 |
| **stream** | 同步流式 | 迭代器 | 聊天界面、实时输出 |
| **ainvoke** | 异步单次 | 单个结果 | 异步应用、并发 |
| **abatch** | 异步批量 | 结果列表 | 异步批量处理 |
| **astream** | 异步流式 | 异步迭代器 | 异步流式输出 |

---

## 2. invoke - 同步调用

### 2.1 基础用法

**最简单的调用方式**:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-4o-mini")

# 调用模型
response = model.invoke([HumanMessage(content="你好")])

print(response.content)  # "你好！有什么可以帮你的吗？"
print(type(response))    # <class 'langchain_core.messages.ai.AIMessage'>
```

### 2.2 与模板组合

**模板 + 模型的组合**:

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}")
])

# 组合链
chain = template | model

# 调用链
response = chain.invoke({
    "role": "Python专家",
    "question": "什么是装饰器？"
})

print(response.content)
```

### 2.3 配置参数

**使用 RunnableConfig**:

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    tags=["production", "chatbot"],
    metadata={"user_id": "123", "session_id": "abc"},
    run_name="customer_support"
)

response = model.invoke(
    [HumanMessage(content="你好")],
    config=config
)
```

### 2.4 使用场景

**适用场景**:

1. **简单问答**: 单次问答，不需要流式输出
2. **测试脚本**: 快速测试模型或链的功能
3. **批处理**: 在循环中调用（但 batch 更高效）
4. **同步应用**: 不需要异步的简单应用

**示例：简单问答机器人**:

```python
def ask(question: str) -> str:
    """简单的问答函数"""
    template = ChatPromptTemplate.from_messages([
        ("system", "你是友好的助手"),
        ("human", "{question}")
    ])

    chain = template | model
    response = chain.invoke({"question": question})
    return response.content

# 使用
answer = ask("Python 是什么？")
print(answer)
```

---

## 3. stream - 流式调用

### 3.1 基础用法

**逐块接收响应**:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-4o-mini")

# 流式调用
for chunk in model.stream([HumanMessage(content="讲个笑话")]):
    print(chunk.content, end="", flush=True)

print()  # 换行
```

### 3.2 与模板组合

**流式输出完整链**:

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}")
])

chain = template | model

# 流式调用链
for chunk in chain.stream({
    "role": "故事讲述者",
    "question": "讲一个关于Python的故事"
}):
    print(chunk.content, end="", flush=True)
```

### 3.3 处理流式数据

**收集和处理流式数据**:

```python
def stream_with_processing(question: str):
    """流式输出并处理"""
    template = ChatPromptTemplate.from_messages([
        ("system", "你是助手"),
        ("human", "{question}")
    ])

    chain = template | model
    full_response = ""

    for chunk in chain.stream({"question": question}):
        content = chunk.content
        full_response += content

        # 实时处理
        print(content, end="", flush=True)

        # 检测不当内容
        if "不当词汇" in full_response:
            print("\n[检测到不当内容，停止输出]")
            break

    return full_response

# 使用
response = stream_with_processing("讲个笑话")
```

### 3.4 使用场景

**适用场景**:

1. **聊天界面**: 降低用户感知延迟
2. **长文本生成**: 让用户看到进度
3. **实时处理**: 边接收边处理内容
4. **早期检测**: 及时发现问题并中断

**示例：聊天界面**:

```python
def chat_with_stream(question: str):
    """流式聊天"""
    template = ChatPromptTemplate.from_messages([
        ("system", "你是友好的助手"),
        ("human", "{question}")
    ])

    chain = template | model

    print("AI: ", end="", flush=True)
    for chunk in chain.stream({"question": question}):
        print(chunk.content, end="", flush=True)
    print()  # 换行

# 使用
chat_with_stream("Python 有什么特点？")
# 输出: AI: Python 是一种...（逐字输出）
```

### 3.5 性能特性

**stream vs invoke 的性能对比**:

```python
import time

# invoke - 等待完整响应
start = time.time()
response = model.invoke([HumanMessage(content="讲个长故事")])
print(f"首字时间: {time.time() - start:.2f}s")  # 5.0s
print(f"总时间: {time.time() - start:.2f}s")    # 5.0s

# stream - 逐字接收
start = time.time()
first_chunk = True
for chunk in model.stream([HumanMessage(content="讲个长故事")]):
    if first_chunk:
        print(f"首字时间: {time.time() - start:.2f}s")  # 0.3s
        first_chunk = False
print(f"总时间: {time.time() - start:.2f}s")  # 5.0s
```

**关键发现**:
- **总时间相同**: stream 不会更快完成
- **首字时间更短**: stream 更快开始输出
- **感知延迟更低**: 用户体验更好

---

## 4. batch - 批量调用

### 4.1 基础用法

**批量处理多个输入**:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-4o-mini")

# 批量调用
inputs = [
    [HumanMessage(content="什么是Python？")],
    [HumanMessage(content="什么是JavaScript？")],
    [HumanMessage(content="什么是Rust？")]
]

responses = model.batch(inputs)

for i, response in enumerate(responses):
    print(f"问题{i+1}: {response.content}\n")
```

### 4.2 与模板组合

**批量处理模板输入**:

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}")
])

chain = template | model

# 批量输入
inputs = [
    {"role": "Python专家", "question": "什么是装饰器？"},
    {"role": "JavaScript专家", "question": "什么是闭包？"},
    {"role": "Rust专家", "question": "什么是所有权？"}
]

responses = chain.batch(inputs)

for i, response in enumerate(responses):
    print(f"回答{i+1}: {response.content}\n")
```

### 4.3 配置批量大小

**控制并发数**:

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    max_concurrency=5  # 最多同时处理5个请求
)

responses = chain.batch(inputs, config=config)
```

### 4.4 使用场景

**适用场景**:

1. **批量评估**: 评估模型在多个测试用例上的表现
2. **数据标注**: 批量标注数据
3. **离线分析**: 批量生成报告或分析
4. **成本优化**: 使用批处理 API 降低成本（2025+）

**示例：批量评估**:

```python
def evaluate_model(test_cases: List[dict]) -> List[dict]:
    """批量评估模型"""
    template = ChatPromptTemplate.from_messages([
        ("system", "你是助手"),
        ("human", "{question}")
    ])

    chain = template | model

    # 批量调用
    responses = chain.batch(test_cases)

    # 评估结果
    results = []
    for test_case, response in zip(test_cases, responses):
        results.append({
            "question": test_case["question"],
            "expected": test_case.get("expected"),
            "actual": response.content,
            "passed": response.content == test_case.get("expected")
        })

    return results

# 使用
test_cases = [
    {"question": "1+1=?", "expected": "2"},
    {"question": "2+2=?", "expected": "4"}
]
results = evaluate_model(test_cases)
```

### 4.5 性能优化

**batch vs 循环 invoke 的性能对比**:

```python
import time

inputs = [{"question": f"问题{i}"} for i in range(10)]

# 方式1: 循环 invoke（慢）
start = time.time()
responses1 = []
for input in inputs:
    response = chain.invoke(input)
    responses1.append(response)
print(f"循环 invoke: {time.time() - start:.2f}s")  # 50s

# 方式2: batch（快）
start = time.time()
responses2 = chain.batch(inputs)
print(f"batch: {time.time() - start:.2f}s")  # 10s
```

**关键发现**:
- **batch 更快**: 并发处理，节省时间
- **成本更低**: 2025+ 支持批处理 API，成本降低 50%

---

## 5. ainvoke - 异步调用

### 5.1 基础用法

**异步单次调用**:

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-4o-mini")

async def async_ask(question: str) -> str:
    """异步问答"""
    response = await model.ainvoke([HumanMessage(content=question)])
    return response.content

# 使用
answer = asyncio.run(async_ask("你好"))
print(answer)
```

### 5.2 并发调用

**同时处理多个请求**:

```python
import asyncio

async def concurrent_asks(questions: List[str]) -> List[str]:
    """并发处理多个问题"""
    tasks = [async_ask(q) for q in questions]
    responses = await asyncio.gather(*tasks)
    return responses

# 使用
questions = ["什么是Python？", "什么是JavaScript？", "什么是Rust？"]
answers = asyncio.run(concurrent_asks(questions))

for q, a in zip(questions, answers):
    print(f"Q: {q}\nA: {a}\n")
```

### 5.3 与模板组合

**异步链调用**:

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}")
])

chain = template | model

async def async_chain_call(role: str, question: str) -> str:
    """异步链调用"""
    response = await chain.ainvoke({
        "role": role,
        "question": question
    })
    return response.content

# 使用
answer = asyncio.run(async_chain_call("助手", "你好"))
```

### 5.4 使用场景

**适用场景**:

1. **Web 应用**: FastAPI、Sanic 等异步框架
2. **并发处理**: 同时处理多个用户请求
3. **I/O 密集**: 大量网络请求的场景
4. **实时应用**: WebSocket、SSE 等实时通信

**示例：FastAPI 集成**:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask_endpoint(q: Question):
    """异步问答端点"""
    template = ChatPromptTemplate.from_messages([
        ("system", "你是助手"),
        ("human", "{question}")
    ])

    chain = template | model
    response = await chain.ainvoke({"question": q.question})

    return {"answer": response.content}

# 使用
# POST /ask {"question": "你好"}
# 返回: {"answer": "你好！有什么可以帮你的吗？"}
```

---

## 6. astream - 异步流式调用

### 6.1 基础用法

**异步流式输出**:

```python
import asyncio

async def async_stream(question: str):
    """异步流式输出"""
    async for chunk in model.astream([HumanMessage(content=question)]):
        print(chunk.content, end="", flush=True)
    print()

# 使用
asyncio.run(async_stream("讲个笑话"))
```

### 6.2 与模板组合

**异步流式链**:

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}")
])

chain = template | model

async def async_stream_chain(role: str, question: str):
    """异步流式链调用"""
    async for chunk in chain.astream({
        "role": role,
        "question": question
    }):
        print(chunk.content, end="", flush=True)
    print()

# 使用
asyncio.run(async_stream_chain("故事讲述者", "讲个故事"))
```

### 6.3 使用场景

**适用场景**:

1. **WebSocket**: 实时推送流式响应
2. **SSE**: Server-Sent Events 流式输出
3. **异步聊天**: 异步聊天应用
4. **并发流式**: 同时处理多个流式请求

**示例：WebSocket 集成**:

```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket 聊天端点"""
    await websocket.accept()

    while True:
        # 接收用户消息
        question = await websocket.receive_text()

        # 流式响应
        template = ChatPromptTemplate.from_messages([
            ("system", "你是助手"),
            ("human", "{question}")
        ])

        chain = template | model

        async for chunk in chain.astream({"question": question}):
            await websocket.send_text(chunk.content)

        # 发送结束标记
        await websocket.send_text("[END]")
```

---

## 7. abatch - 异步批量调用

### 7.1 基础用法

**异步批量处理**:

```python
import asyncio

async def async_batch(questions: List[str]) -> List[str]:
    """异步批量处理"""
    inputs = [[HumanMessage(content=q)] for q in questions]
    responses = await model.abatch(inputs)
    return [r.content for r in responses]

# 使用
questions = ["问题1", "问题2", "问题3"]
answers = asyncio.run(async_batch(questions))
```

### 7.2 与模板组合

**异步批量链调用**:

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}")
])

chain = template | model

async def async_batch_chain(inputs: List[dict]) -> List[str]:
    """异步批量链调用"""
    responses = await chain.abatch(inputs)
    return [r.content for r in responses]

# 使用
inputs = [
    {"role": "助手", "question": "问题1"},
    {"role": "助手", "question": "问题2"}
]
answers = asyncio.run(async_batch_chain(inputs))
```

### 7.3 使用场景

**适用场景**:

1. **异步批量评估**: 异步环境下的批量评估
2. **并发批处理**: 同时处理多个批次
3. **异步数据标注**: 异步批量标注
4. **高并发场景**: Web 应用的批量处理

---

## 8. 方法选择指南

### 8.1 决策树

```
需要异步？
├─ 是 → 需要流式？
│   ├─ 是 → astream
│   └─ 否 → 需要批量？
│       ├─ 是 → abatch
│       └─ 否 → ainvoke
└─ 否 → 需要流式？
    ├─ 是 → stream
    └─ 否 → 需要批量？
        ├─ 是 → batch
        └─ 否 → invoke
```

### 8.2 场景对照表

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 简单问答 | invoke | 代码简单，等待时间可接受 |
| 聊天界面 | stream | 降低感知延迟，用户体验好 |
| 批量评估 | batch | 并发处理，节省时间 |
| Web API | ainvoke | 不阻塞事件循环 |
| WebSocket | astream | 实时推送，异步处理 |
| 批量标注 | batch | 成本优化（2025+ 批处理 API） |
| 测试脚本 | invoke | 简单直接 |
| 高并发 | ainvoke/abatch | 充分利用异步优势 |

### 8.3 性能对比

**不同方法的性能特性**:

| 方法 | 首字时间 | 总时间 | 并发能力 | 成本 |
|------|----------|--------|----------|------|
| **invoke** | 慢 | 正常 | 无 | 正常 |
| **stream** | 快 | 正常 | 无 | 正常 |
| **batch** | 慢 | 快 | 高 | 低（2025+） |
| **ainvoke** | 慢 | 正常 | 高 | 正常 |
| **astream** | 快 | 正常 | 高 | 正常 |
| **abatch** | 慢 | 快 | 高 | 低（2025+） |

---

## 9. 高级用法

### 9.1 配置传递

**使用 RunnableConfig**:

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    # 标签
    tags=["production", "chatbot"],
    # 元数据
    metadata={"user_id": "123", "session_id": "abc"},
    # 运行名称
    run_name="customer_support",
    # 最大并发数
    max_concurrency=10,
    # 回调
    callbacks=[...]
)

# 所有方法都支持 config
response = model.invoke(messages, config=config)
responses = model.batch(messages_list, config=config)
for chunk in model.stream(messages, config=config):
    pass
```

### 9.2 错误处理

**处理调用错误**:

```python
from langchain_core.exceptions import OutputParserException

try:
    response = chain.invoke({"question": "你好"})
except OutputParserException as e:
    print(f"解析错误: {e}")
except Exception as e:
    print(f"调用错误: {e}")
```

### 9.3 超时控制

**设置超时**:

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    timeout=30  # 30秒超时
)

try:
    response = model.invoke(messages, config=config)
except TimeoutError:
    print("调用超时")
```

### 9.4 重试机制

**自动重试**:

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    max_retries=3,  # 最多重试3次
    retry_delay=1.0  # 重试延迟1秒
)

response = model.invoke(messages, config=config)
```

---

## 10. 2025-2026 新特性

### 10.1 批处理 API 集成

**降低成本 50%**:

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o-mini",
    batch_mode=True  # 启用批处理模式（2025+）
)

# 批量调用自动使用批处理 API
responses = model.batch(inputs)
# 成本降低 50%，但延迟增加
```

### 10.2 流式工具调用

**流式接收工具调用**:

```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

model_with_tools = model.bind_tools([calculator])

# 流式接收工具调用
for chunk in model_with_tools.stream([HumanMessage("2+2=?")]):
    if chunk.tool_calls:
        print(f"工具调用: {chunk.tool_calls}")
    else:
        print(chunk.content, end="")
```

### 10.3 结构化流式输出

**流式接收结构化数据**:

```python
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str
    confidence: float

structured_model = model.with_structured_output(Answer)

# 流式接收结构化输出（2025+）
for chunk in structured_model.stream([HumanMessage("Python是什么？")]):
    print(chunk)  # 逐步构建 Answer 对象
```

---

## 检查清单

完成本节学习后，你应该能够：

- [ ] 理解 Runnable 协议的 6 个核心方法
- [ ] 使用 invoke 进行同步调用
- [ ] 使用 stream 实现流式输出
- [ ] 使用 batch 进行批量处理
- [ ] 使用 ainvoke 进行异步调用
- [ ] 使用 astream 实现异步流式输出
- [ ] 使用 abatch 进行异步批量处理
- [ ] 根据场景选择合适的方法
- [ ] 使用 RunnableConfig 配置调用
- [ ] 应用 2025-2026 新特性

---

**下一步**: 阅读 `07_实战代码_01_基础ChatModel调用.md` 开始动手实践
