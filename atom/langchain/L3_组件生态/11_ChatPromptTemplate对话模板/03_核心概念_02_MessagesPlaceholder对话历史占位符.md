# 核心概念 02：MessagesPlaceholder 对话历史占位符

## 一句话定义

**MessagesPlaceholder 是 ChatPromptTemplate 中用于动态插入消息列表的占位符，专门用于管理对话历史和动态消息流。**

---

## 核心概念详解

### 1. 什么是 MessagesPlaceholder？

MessagesPlaceholder 是一个特殊的消息模板组件，它不直接包含消息内容，而是作为一个"占位符"，在运行时动态接收并插入一组消息。

**核心特点**：
- 动态插入消息列表
- 支持可选参数（optional）
- 可限制消息数量（n_messages）
- 自动转换消息格式

**源码定义**（来自 langchain_core/prompts/chat.py:52-123）：

```python
class MessagesPlaceholder(BaseMessagePromptTemplate):
    """Prompt template that assumes variable is already list of messages.

    A placeholder which can be used to pass in a list of messages.
    """

    variable_name: str
    """Name of variable to use as messages."""

    optional: bool = False
    """Whether `format_messages` must be provided."""

    n_messages: PositiveInt | None = None
    """Maximum number of messages to include."""
```

---

### 2. 基础用法

#### 2.1 最简单的用法

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 创建包含 MessagesPlaceholder 的模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),  # 对话历史占位符
    ("human", "{question}")
])

# 使用时传入历史消息
result = prompt.invoke({
    "history": [
        ("human", "What's 5 + 2?"),
        ("ai", "5 + 2 is 7")
    ],
    "question": "Now multiply that by 4"
})

# 输出：
# ChatPromptValue(messages=[
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content="What's 5 + 2?"),
#     AIMessage(content="5 + 2 is 7"),
#     HumanMessage(content="Now multiply that by 4"),
# ])
```

**关键点**：
- `MessagesPlaceholder("history")` 创建一个名为 "history" 的占位符
- 调用 `invoke()` 时，通过 `history` 参数传入消息列表
- 消息列表会自动转换为 BaseMessage 对象

#### 2.2 使用元组简写

```python
# 方式 1：使用 MessagesPlaceholder 类
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("conversation"),
    ("human", "{input}")
])

# 方式 2：使用元组简写（等价）
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("placeholder", "{conversation}"),  # 简写形式
    ("human", "{input}")
])
```

**注意**：元组简写形式会自动创建一个 `optional=True` 的 MessagesPlaceholder。

---

### 3. optional 参数：处理可选历史

#### 3.1 必需 vs 可选

```python
# 必需历史（optional=False，默认）
prompt_required = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),  # 必须提供
    ("human", "{question}")
])

# 调用时必须提供 history，否则报错
try:
    prompt_required.invoke({"question": "Hello"})
except KeyError as e:
    print(f"错误：{e}")  # KeyError: 'history'

# 可选历史（optional=True）
prompt_optional = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history", optional=True),  # 可选
    ("human", "{question}")
])

# 调用时可以不提供 history
result = prompt_optional.invoke({"question": "Hello"})
# 输出：
# ChatPromptValue(messages=[
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content="Hello"),
# ])
```

#### 3.2 实际应用场景

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 创建支持可选历史的聊天机器人
llm = ChatOpenAI(model="gpt-4")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}")
])

chain = prompt | llm

# 首次对话（无历史）
response1 = chain.invoke({
    "input": "What's the capital of France?"
})
print(response1.content)  # "The capital of France is Paris."

# 后续对话（有历史）
response2 = chain.invoke({
    "chat_history": [
        ("human", "What's the capital of France?"),
        ("ai", "The capital of France is Paris.")
    ],
    "input": "What's its population?"
})
print(response2.content)  # "Paris has a population of approximately 2.2 million..."
```

---

### 4. n_messages 参数：限制消息数量

#### 4.1 基础用法

```python
from langchain_core.prompts import MessagesPlaceholder

# 只保留最后 2 条消息
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history", n_messages=2),
    ("human", "{question}")
])

# 传入 4 条历史消息
result = prompt.invoke({
    "history": [
        ("human", "Message 1"),
        ("ai", "Response 1"),
        ("human", "Message 2"),
        ("ai", "Response 2"),
    ],
    "question": "New question"
})

# 输出：只保留最后 2 条
# ChatPromptValue(messages=[
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content="Message 2"),      # 倒数第 2 条
#     AIMessage(content="Response 2"),        # 倒数第 1 条
#     HumanMessage(content="New question"),
# ])
```

**源码实现**（chat.py:187-189）：

```python
def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
    value = convert_to_messages(value)
    if self.n_messages:
        value = value[-self.n_messages:]  # 只保留最后 n 条
    return value
```

#### 4.2 为什么需要限制消息数量？

**问题**：对话历史越来越长，导致：
- Token 消耗增加
- 响应速度变慢
- 超出模型上下文窗口限制

**解决方案**：使用 `n_messages` 限制历史长度

```python
# 场景：长对话管理
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history", n_messages=10),  # 只保留最近 10 条
    ("human", "{input}")
])

# 即使传入 100 条历史消息，也只会使用最后 10 条
chain = prompt | llm
response = chain.invoke({
    "history": very_long_history,  # 100 条消息
    "input": "Current question"
})
```

#### 4.3 动态调整策略

```python
def create_prompt_with_dynamic_history(max_messages: int = 10):
    """根据需求动态创建提示模板"""
    return ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("history", n_messages=max_messages),
        ("human", "{input}")
    ])

# 短对话场景：保留更多历史
short_conv_prompt = create_prompt_with_dynamic_history(max_messages=20)

# 长对话场景：只保留最近的
long_conv_prompt = create_prompt_with_dynamic_history(max_messages=5)
```

---

### 5. 与对话历史管理集成

#### 5.1 使用 ConversationBufferMemory

```python
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# 创建内存管理器
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# 创建链
llm = ChatOpenAI(model="gpt-4")
chain = prompt | llm

# 对话循环
def chat(user_input: str):
    # 获取历史消息
    history = memory.load_memory_variables({})["chat_history"]

    # 调用链
    response = chain.invoke({
        "chat_history": history,
        "input": user_input
    })

    # 保存到内存
    memory.save_context(
        {"input": user_input},
        {"output": response.content}
    )

    return response.content

# 使用
print(chat("What's 5 + 2?"))        # "5 + 2 is 7"
print(chat("Multiply that by 4"))   # "7 * 4 is 28"
```

#### 5.2 使用 ConversationBufferWindowMemory（滑动窗口）

```python
from langchain.memory import ConversationBufferWindowMemory

# 只保留最近 5 轮对话
memory = ConversationBufferWindowMemory(
    k=5,  # 保留最近 5 轮
    memory_key="chat_history",
    return_messages=True
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

chain = prompt | llm

# 对话管理器
class ChatManager:
    def __init__(self, chain, memory):
        self.chain = chain
        self.memory = memory

    def chat(self, user_input: str) -> str:
        history = self.memory.load_memory_variables({})["chat_history"]
        response = self.chain.invoke({
            "chat_history": history,
            "input": user_input
        })
        self.memory.save_context(
            {"input": user_input},
            {"output": response.content}
        )
        return response.content

# 使用
manager = ChatManager(chain, memory)
print(manager.chat("Hello"))
print(manager.chat("What's your name?"))
```

---

### 6. 在 RAG 系统中的应用

#### 6.1 基础 RAG 对话系统

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

# 1. 创建向量存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(
    texts=[
        "LangChain is a framework for developing applications powered by LLMs.",
        "RAG stands for Retrieval-Augmented Generation.",
        "MessagesPlaceholder is used for managing conversation history."
    ],
    embedding=embeddings
)
retriever = vectorstore.as_retriever()

# 2. 创建提示模板（包含历史）
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the following context to answer questions:\n\n{context}"),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{question}")
])

# 3. 创建 RAG 链
llm = ChatOpenAI(model="gpt-4")

rag_chain = (
    {
        "context": lambda x: "\n".join([doc.page_content for doc in retriever.get_relevant_documents(x["question"])]),
        "chat_history": lambda x: x.get("chat_history", []),
        "question": lambda x: x["question"]
    }
    | prompt
    | llm
)

# 4. 使用
response1 = rag_chain.invoke({
    "question": "What is LangChain?"
})
print(response1.content)

response2 = rag_chain.invoke({
    "chat_history": [
        ("human", "What is LangChain?"),
        ("ai", response1.content)
    ],
    "question": "What about RAG?"
})
print(response2.content)
```

#### 6.2 带历史的 RAG 问答系统

```python
from langchain.memory import ConversationBufferMemory

class RAGChatbot:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # 创建提示模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the context to answer:\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}")
        ])

        # 创建链
        self.chain = (
            {
                "context": lambda x: self._get_context(x["question"]),
                "chat_history": lambda x: self.memory.load_memory_variables({})["chat_history"],
                "question": lambda x: x["question"]
            }
            | self.prompt
            | self.llm
        )

    def _get_context(self, question: str) -> str:
        docs = self.retriever.get_relevant_documents(question)
        return "\n".join([doc.page_content for doc in docs])

    def chat(self, question: str) -> str:
        response = self.chain.invoke({"question": question})

        # 保存到内存
        self.memory.save_context(
            {"input": question},
            {"output": response.content}
        )

        return response.content

# 使用
chatbot = RAGChatbot(retriever, llm)
print(chatbot.chat("What is LangChain?"))
print(chatbot.chat("Can you explain more about it?"))  # 带历史上下文
```

---

### 7. 高级用法

#### 7.1 多个 MessagesPlaceholder

```python
# 场景：区分系统消息和用户历史
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("system_messages", optional=True),  # 系统级消息
    MessagesPlaceholder("user_history", optional=True),     # 用户历史
    ("human", "{input}")
])

result = prompt.invoke({
    "system_messages": [
        ("system", "Additional context: User is a beginner.")
    ],
    "user_history": [
        ("human", "Previous question"),
        ("ai", "Previous answer")
    ],
    "input": "Current question"
})
```

#### 7.2 条件性历史注入

```python
def create_prompt_with_conditional_history(include_history: bool):
    messages = [("system", "You are a helpful assistant.")]

    if include_history:
        messages.append(MessagesPlaceholder("history"))

    messages.append(("human", "{input}"))

    return ChatPromptTemplate.from_messages(messages)

# 根据场景选择
simple_prompt = create_prompt_with_conditional_history(include_history=False)
contextual_prompt = create_prompt_with_conditional_history(include_history=True)
```

---

### 8. 最佳实践

#### 8.1 命名约定

```python
# 推荐：使用清晰的变量名
MessagesPlaceholder("chat_history")      # ✅ 清晰
MessagesPlaceholder("conversation")      # ✅ 清晰
MessagesPlaceholder("history")           # ✅ 简洁

# 避免：模糊的名称
MessagesPlaceholder("msgs")              # ❌ 不清晰
MessagesPlaceholder("h")                 # ❌ 太简短
```

#### 8.2 性能优化

```python
# 1. 使用 n_messages 限制历史长度
MessagesPlaceholder("history", n_messages=10)

# 2. 使用 optional=True 避免不必要的错误
MessagesPlaceholder("history", optional=True)

# 3. 结合使用
MessagesPlaceholder("history", optional=True, n_messages=10)
```

#### 8.3 错误处理

```python
def safe_invoke(prompt, **kwargs):
    """安全调用提示模板"""
    try:
        return prompt.invoke(kwargs)
    except KeyError as e:
        print(f"缺少必需的变量：{e}")
        # 提供默认值
        if "history" not in kwargs:
            kwargs["history"] = []
        return prompt.invoke(kwargs)
```

---

### 9. 常见问题

#### Q1: MessagesPlaceholder vs 普通变量？

```python
# MessagesPlaceholder：用于消息列表
MessagesPlaceholder("history")  # 接收 [("human", "..."), ("ai", "...")]

# 普通变量：用于字符串
("human", "{question}")  # 接收 "What is...?"
```

#### Q2: 如何处理空历史？

```python
# 方式 1：使用 optional=True
MessagesPlaceholder("history", optional=True)

# 方式 2：提供空列表
prompt.invoke({"history": [], "question": "..."})
```

#### Q3: 消息格式转换？

```python
# 支持多种格式
history = [
    ("human", "Hello"),                          # 元组格式
    HumanMessage(content="Hello"),               # 消息对象
    {"role": "human", "content": "Hello"}        # 字典格式
]

# 都会自动转换为 BaseMessage 对象
```

---

### 10. 在实际应用中的使用

#### 场景 1：客服聊天机器人

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a customer service assistant."),
    MessagesPlaceholder("chat_history", n_messages=20),  # 保留最近 20 条
    ("human", "{customer_query}")
])
```

#### 场景 2：代码助手

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a coding assistant."),
    MessagesPlaceholder("conversation", optional=True),
    ("human", "Code: {code}\nQuestion: {question}")
])
```

#### 场景 3：文档问答

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on: {context}"),
    MessagesPlaceholder("history", n_messages=5),
    ("human", "{query}")
])
```

---

## 总结

MessagesPlaceholder 是 LangChain 中管理对话历史的核心组件：

1. **动态插入**：运行时注入消息列表
2. **灵活配置**：支持 optional 和 n_messages 参数
3. **自动转换**：支持多种消息格式
4. **RAG 集成**：与检索系统无缝配合
5. **性能优化**：通过限制消息数量控制 token 消耗

**核心价值**：让对话系统能够"记住"历史，实现真正的上下文感知对话。
