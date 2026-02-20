# 核心概念 4：Runnable 泛型设计

## 一句话定义

**Runnable[Input, Output] 是 LangChain 的核心抽象，通过泛型类型参数定义组件的输入输出类型，让 LCEL 管道在编译时就能验证类型兼容性，是构建类型安全 AI Agent 的基础。**

---

## Runnable 的设计哲学

### 1. 为什么需要 Runnable？

在 LangChain 早期版本中，组件之间的连接是松散的：

```python
# 早期版本（无类型安全）
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# 创建组件
prompt = PromptTemplate.from_template("Tell me about {topic}")
llm = OpenAI()
chain = LLMChain(llm=llm, prompt=prompt)

# 运行
result = chain.run(topic="Python")  # ⚠️ 运行时才知道是否正确
```

**问题**：
- 类型不明确：不知道输入输出是什么类型
- 运行时错误：类型不匹配要到运行时才发现
- 难以组合：不同组件的接口不统一
- 缺乏可观测性：难以追踪数据流

**Runnable 的解决方案**：

```python
# 现代版本（类型安全）
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 每个组件都是 Runnable[Input, Output]
prompt: Runnable[dict, str] = ChatPromptTemplate.from_template(
    "Tell me about {topic}"
)
model: Runnable[str, str] = ChatOpenAI(model="gpt-4o-mini")

# 组合：类型检查器验证兼容性
chain: Runnable[dict, str] = prompt | model  # ✅ 类型正确

# 运行
result: str = chain.invoke({"topic": "Python"})
```

---

## Runnable 的类型定义

### 1. 核心接口（简化版）

```python
from typing import TypeVar, Generic, Any, Iterator

Input = TypeVar('Input', contravariant=True)
Output = TypeVar('Output', covariant=True)

class Runnable(Generic[Input, Output]):
    """LangChain Runnable 核心接口"""

    def invoke(self, input: Input, config: Any = None) -> Output:
        """同步调用"""
        ...

    async def ainvoke(self, input: Input, config: Any = None) -> Output:
        """异步调用"""
        ...

    def batch(
        self,
        inputs: list[Input],
        config: Any = None
    ) -> list[Output]:
        """批量调用"""
        ...

    async def abatch(
        self,
        inputs: list[Input],
        config: Any = None
    ) -> list[Output]:
        """异步批量调用"""
        ...

    def stream(
        self,
        input: Input,
        config: Any = None
    ) -> Iterator[Output]:
        """流式调用"""
        ...

    async def astream(
        self,
        input: Input,
        config: Any = None
    ):
        """异步流式调用"""
        ...

    def __or__(self, other: Runnable[Output, NewOutput]) -> Runnable[Input, NewOutput]:
        """管道操作符：self | other"""
        ...
```

**关键设计**：
- `Input` 是逆变的（contravariant）：可以接受更宽泛的输入
- `Output` 是协变的（covariant）：可以返回更具体的输出
- 统一接口：所有组件都实现相同的方法
- 可组合：通过 `|` 操作符连接

### 2. 为什么 Input 逆变，Output 协变？

```python
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# 假设有两个 Runnable
class MessageProcessor(Runnable[BaseMessage, str]):
    """处理任何 BaseMessage"""
    def invoke(self, input: BaseMessage) -> str:
        return input.content

class HumanMessageGenerator(Runnable[dict, HumanMessage]):
    """生成 HumanMessage"""
    def invoke(self, input: dict) -> HumanMessage:
        return HumanMessage(content=input["text"])

# 逆变：MessageProcessor 可以接受 HumanMessage（BaseMessage 的子类）
# 协变：HumanMessageGenerator 返回 HumanMessage（可以当作 BaseMessage 使用）

# 组合
chain: Runnable[dict, str] = HumanMessageGenerator() | MessageProcessor()
# ✅ 类型正确：dict -> HumanMessage -> str
```

---

## Runnable 的类型推断

### 1. 自动类型推断

```python
from langchain_core.runnables import RunnableLambda

# 定义类型明确的函数
def to_upper(text: str) -> str:
    return text.upper()

def count_words(text: str) -> int:
    return len(text.split())

# RunnableLambda 自动推断类型
upper_runnable = RunnableLambda(to_upper)
# 类型推断为：Runnable[str, str]

counter_runnable = RunnableLambda(count_words)
# 类型推断为：Runnable[str, int]

# 组合时自动验证类型
chain = upper_runnable | counter_runnable
# 类型推断为：Runnable[str, int]
```

### 2. with_types 显式指定类型

```python
from langchain_core.runnables import RunnableLambda

# 动态函数（类型不明确）
def process(x):
    return x.upper()

# 创建 Runnable（类型推断为 Any）
runnable = RunnableLambda(process)

# 使用 with_types 显式指定类型
typed_runnable: Runnable[str, str] = runnable.with_types(
    input_type=str,
    output_type=str
)

# 现在有类型检查
result: str = typed_runnable.invoke("hello")  # ✅
```

### 3. input_schema 和 output_schema

```python
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

class UserInput(BaseModel):
    name: str
    age: int

class UserOutput(BaseModel):
    greeting: str

def greet_user(input: UserInput) -> UserOutput:
    return UserOutput(greeting=f"Hello, {input.name}!")

# 创建 Runnable
runnable = RunnableLambda(greet_user)

# 自动生成 schema
print(runnable.input_schema)
# <class 'UserInput'>

print(runnable.output_schema)
# <class 'UserOutput'>

# 使用
result = runnable.invoke(UserInput(name="Alice", age=30))
print(result.greeting)  # "Hello, Alice!"
```

---

## LCEL 管道的类型组合

### 1. 线性组合

```python
from langchain_core.runnables import RunnableLambda

# 定义类型明确的转换
def format_prompt(topic: str) -> str:
    return f"Tell me about {topic}"

def extract_length(text: str) -> int:
    return len(text)

def format_result(length: int) -> str:
    return f"Length: {length}"

# 构建管道
formatter: Runnable[str, str] = RunnableLambda(format_prompt)
extractor: Runnable[str, int] = RunnableLambda(extract_length)
result_formatter: Runnable[int, str] = RunnableLambda(format_result)

# 组合：类型自动推断
chain: Runnable[str, str] = formatter | extractor | result_formatter
# str -> str -> int -> str

# 类型检查器验证整个链
result: str = chain.invoke("Python")  # ✅
result: int = chain.invoke("Python")  # ❌ 类型错误
```

### 2. 并行组合（RunnableParallel）

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda

# 定义多个转换
def get_length(text: str) -> int:
    return len(text)

def get_words(text: str) -> int:
    return len(text.split())

def get_upper(text: str) -> str:
    return text.upper()

# 并行执行
parallel: Runnable[str, dict] = RunnableParallel(
    length=RunnableLambda(get_length),
    words=RunnableLambda(get_words),
    upper=RunnableLambda(get_upper)
)

# 类型推断为：Runnable[str, dict[str, int | str]]
result = parallel.invoke("hello world")
# {"length": 11, "words": 2, "upper": "HELLO WORLD"}
```

### 3. 条件组合（RunnableBranch）

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

# 定义条件和处理器
def is_long(text: str) -> bool:
    return len(text) > 10

def summarize(text: str) -> str:
    return text[:10] + "..."

def pass_through(text: str) -> str:
    return text

# 条件分支
branch: Runnable[str, str] = RunnableBranch(
    (is_long, RunnableLambda(summarize)),
    RunnableLambda(pass_through)
)

# 类型推断为：Runnable[str, str]
result = branch.invoke("This is a very long text")  # "This is a ..."
result = branch.invoke("Short")  # "Short"
```

---

## 自定义 Runnable

### 1. 继承 Runnable 基类

```python
from typing import Any
from langchain_core.runnables import Runnable

class UppercaseRunnable(Runnable[str, str]):
    """自定义 Runnable：转换为大写"""

    def invoke(self, input: str, config: Any = None) -> str:
        return input.upper()

# 使用
runnable: Runnable[str, str] = UppercaseRunnable()
result: str = runnable.invoke("hello")  # "HELLO"
```

### 2. 泛型自定义 Runnable

```python
from typing import TypeVar, Generic, Callable, Any
from langchain_core.runnables import Runnable

Input = TypeVar('Input')
Output = TypeVar('Output')

class TransformRunnable(Runnable[Input, Output], Generic[Input, Output]):
    """泛型转换 Runnable"""

    def __init__(self, transform_fn: Callable[[Input], Output]):
        self.transform_fn = transform_fn

    def invoke(self, input: Input, config: Any = None) -> Output:
        return self.transform_fn(input)

# 使用
def length(text: str) -> int:
    return len(text)

# 类型推断为：TransformRunnable[str, int]
length_runnable: Runnable[str, int] = TransformRunnable(length)
result: int = length_runnable.invoke("hello")  # 5
```

### 3. 实现完整的 Runnable 接口

```python
from typing import Any, Iterator
from langchain_core.runnables import Runnable

class CounterRunnable(Runnable[str, int]):
    """完整实现 Runnable 接口"""

    def invoke(self, input: str, config: Any = None) -> int:
        return len(input)

    async def ainvoke(self, input: str, config: Any = None) -> int:
        return len(input)

    def batch(self, inputs: list[str], config: Any = None) -> list[int]:
        return [len(text) for text in inputs]

    async def abatch(self, inputs: list[str], config: Any = None) -> list[int]:
        return [len(text) for text in inputs]

    def stream(self, input: str, config: Any = None) -> Iterator[int]:
        # 逐字符流式返回长度
        for i in range(1, len(input) + 1):
            yield i

    async def astream(self, input: str, config: Any = None):
        for i in range(1, len(input) + 1):
            yield i

# 使用
counter = CounterRunnable()

# 同步调用
result = counter.invoke("hello")  # 5

# 批量调用
results = counter.batch(["hello", "world"])  # [5, 5]

# 流式调用
for length in counter.stream("hello"):
    print(length)  # 1, 2, 3, 4, 5
```

---

## LangChain 内置 Runnable 的类型

### 1. ChatModel

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# ChatModel 的类型签名
model: Runnable[
    str | list[BaseMessage] | PromptValue,
    AIMessage
] = ChatOpenAI(model="gpt-4o-mini")

# 使用
result: AIMessage = model.invoke("Hello")
result: AIMessage = model.invoke([HumanMessage(content="Hello")])
```

### 2. PromptTemplate

```python
from langchain_core.prompts import ChatPromptTemplate, PromptValue

# PromptTemplate 的类型签名
prompt: Runnable[dict, PromptValue] = ChatPromptTemplate.from_template(
    "Tell me about {topic}"
)

# 使用
result: PromptValue = prompt.invoke({"topic": "Python"})
```

### 3. OutputParser

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import BaseMessage

# StrOutputParser 的类型签名
str_parser: Runnable[str | BaseMessage, str] = StrOutputParser()

# JsonOutputParser 的类型签名
json_parser: Runnable[str | BaseMessage, dict] = JsonOutputParser()

# 使用
result: str = str_parser.invoke("Hello")
result: dict = json_parser.invoke('{"name": "Alice"}')
```

### 4. RunnablePassthrough

```python
from langchain_core.runnables import RunnablePassthrough

# RunnablePassthrough 的类型签名
passthrough: Runnable[T, T] = RunnablePassthrough()

# 使用
result: str = passthrough.invoke("hello")  # "hello"
result: int = passthrough.invoke(42)  # 42
```

---

## 类型安全的 LCEL 管道

### 1. 完整的类型检查链

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable

# 定义每个组件的类型
prompt: Runnable[dict, PromptValue] = ChatPromptTemplate.from_template(
    "Tell me a joke about {topic}"
)

model: Runnable[PromptValue, AIMessage] = ChatOpenAI(model="gpt-4o-mini")

parser: Runnable[AIMessage, str] = StrOutputParser()

# 组合：类型自动推断
chain: Runnable[dict, str] = prompt | model | parser
# dict -> PromptValue -> AIMessage -> str

# 类型检查器验证
result: str = chain.invoke({"topic": "programming"})  # ✅
result: int = chain.invoke({"topic": "programming"})  # ❌ 类型错误
```

### 2. 类型不匹配的错误

```python
from langchain_core.runnables import RunnableLambda

def to_int(text: str) -> int:
    return len(text)

def to_upper(num: int) -> str:
    return str(num).upper()

def double(text: str) -> str:
    return text * 2

# ✅ 类型兼容
chain1 = RunnableLambda(to_int) | RunnableLambda(to_upper)
# str -> int -> str

# ❌ 类型不兼容
chain2 = RunnableLambda(to_int) | RunnableLambda(double)
# str -> int -> str（期望 str，但得到 int）
# 类型检查器报错：Argument of type "int" cannot be assigned to parameter of type "str"
```

---

## 实战示例：类型安全的 RAG 管道

```python
"""
类型安全的 RAG 管道
演示：Runnable[Input, Output] 在实际项目中的应用
"""

from typing import Any
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ===== 1. 定义类型 =====
from pydantic import BaseModel

class Query(BaseModel):
    """查询输入"""
    question: str
    top_k: int = 3

class RetrievalResult(BaseModel):
    """检索结果"""
    question: str
    documents: list[str]

class RAGOutput(BaseModel):
    """RAG 输出"""
    answer: str
    sources: list[str]

# ===== 2. 构建组件 =====

# 2.1 向量存储（模拟）
def create_vectorstore() -> Chroma:
    """创建向量存储"""
    docs = [
        Document(page_content="Python is a programming language"),
        Document(page_content="LangChain is a framework for LLM applications"),
        Document(page_content="Type hints improve code quality"),
    ]
    return Chroma.from_documents(docs, OpenAIEmbeddings())

vectorstore = create_vectorstore()

# 2.2 检索器：Query -> RetrievalResult
def retrieve_documents(query: Query) -> RetrievalResult:
    """检索相关文档"""
    docs = vectorstore.similarity_search(query.question, k=query.top_k)
    return RetrievalResult(
        question=query.question,
        documents=[doc.page_content for doc in docs]
    )

retriever: Runnable[Query, RetrievalResult] = RunnableLambda(retrieve_documents)

# 2.3 格式化器：RetrievalResult -> dict
def format_context(result: RetrievalResult) -> dict:
    """格式化上下文"""
    context = "\n".join(result.documents)
    return {
        "question": result.question,
        "context": context
    }

formatter: Runnable[RetrievalResult, dict] = RunnableLambda(format_context)

# 2.4 Prompt：dict -> PromptValue
prompt: Runnable[dict, Any] = ChatPromptTemplate.from_template(
    """Answer the question based on the context:

Context: {context}

Question: {question}

Answer:"""
)

# 2.5 LLM：PromptValue -> AIMessage
model: Runnable[Any, Any] = ChatOpenAI(model="gpt-4o-mini")

# 2.6 Parser：AIMessage -> str
parser: Runnable[Any, str] = StrOutputParser()

# ===== 3. 组合管道 =====

# 完整的 RAG 管道：Query -> str
rag_chain: Runnable[Query, str] = (
    retriever
    | formatter
    | prompt
    | model
    | parser
)

# 类型推断：Query -> RetrievalResult -> dict -> PromptValue -> AIMessage -> str

# ===== 4. 使用管道 =====

print("=== 类型安全的 RAG 管道 ===\n")

# 创建查询
query = Query(question="What is Python?", top_k=2)

# 执行管道（类型安全）
answer: str = rag_chain.invoke(query)
print(f"Question: {query.question}")
print(f"Answer: {answer}\n")

# ===== 5. 添加并行处理 =====

# 并行获取答案和来源
def extract_sources(result: RetrievalResult) -> list[str]:
    """提取来源"""
    return result.documents

parallel_chain: Runnable[Query, dict] = RunnableParallel(
    answer=rag_chain,
    sources=retriever | RunnableLambda(extract_sources)
)

# 使用并行管道
result = parallel_chain.invoke(query)
print("=== 并行处理结果 ===")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")

# ===== 6. 类型检查验证 =====

# ✅ 类型正确
answer: str = rag_chain.invoke(Query(question="What is LangChain?"))

# ❌ 类型错误（编译时发现）
# answer: int = rag_chain.invoke(Query(question="What is LangChain?"))
# 类型检查器报错：Cannot assign str to int

# ❌ 输入类型错误
# answer = rag_chain.invoke("What is LangChain?")
# 类型检查器报错：Argument of type "str" cannot be assigned to parameter of type "Query"
```

**运行输出**：
```
=== 类型安全的 RAG 管道 ===

Question: What is Python?
Answer: Python is a programming language.

=== 并行处理结果 ===
Answer: Python is a programming language.
Sources: ['Python is a programming language', 'Type hints improve code quality']
```

---

## 2025-2026 最佳实践

### 1. 显式类型注解

```python
from langchain_core.runnables import Runnable

# ❌ 不推荐：类型不明确
chain = prompt | model | parser

# ✅ 推荐：显式类型注解
chain: Runnable[dict, str] = prompt | model | parser
```

### 2. 使用 Pydantic 模型

```python
from pydantic import BaseModel

# ✅ 推荐：使用 Pydantic 定义输入输出
class UserInput(BaseModel):
    name: str
    age: int

class UserOutput(BaseModel):
    greeting: str

def process(input: UserInput) -> UserOutput:
    return UserOutput(greeting=f"Hello, {input.name}!")

runnable: Runnable[UserInput, UserOutput] = RunnableLambda(process)
```

### 3. 组合前验证类型

```python
# ✅ 推荐：在组合前验证类型兼容性
component1: Runnable[str, int] = ...
component2: Runnable[int, str] = ...

# 类型检查器会验证 component1 的输出类型（int）
# 是否匹配 component2 的输入类型（int）
chain: Runnable[str, str] = component1 | component2  # ✅
```

### 4. 使用 with_types 处理动态代码

```python
# 动态代码（类型不明确）
def dynamic_process(x):
    return x.upper()

# ✅ 使用 with_types 显式指定类型
runnable = RunnableLambda(dynamic_process).with_types(
    input_type=str,
    output_type=str
)
```

---

## 常见陷阱

### 1. 忘记类型注解

```python
# ❌ 类型不明确
def process(x):
    return x.upper()

runnable = RunnableLambda(process)
# 类型推断为：Runnable[Any, Any]

# ✅ 添加类型注解
def process(x: str) -> str:
    return x.upper()

runnable = RunnableLambda(process)
# 类型推断为：Runnable[str, str]
```

### 2. 类型不匹配的组合

```python
# ❌ 类型不匹配
def to_int(text: str) -> int:
    return len(text)

def double_str(text: str) -> str:
    return text * 2

# 类型错误：int 不能赋值给 str
chain = RunnableLambda(to_int) | RunnableLambda(double_str)
```

### 3. 过度使用 Any

```python
# ❌ 失去类型安全
def process(x: Any) -> Any:
    return x.upper()

# ✅ 使用具体类型
def process(x: str) -> str:
    return x.upper()
```

---

## 学习检查清单

- [ ] 理解 Runnable[Input, Output] 的设计哲学
- [ ] 掌握 Runnable 的核心接口
- [ ] 理解 Input 逆变和 Output 协变
- [ ] 了解类型推断机制
- [ ] 掌握 with_types 的使用
- [ ] 能够创建自定义 Runnable
- [ ] 理解 LCEL 管道的类型组合
- [ ] 能够构建类型安全的 RAG 管道
- [ ] 遵循 2025-2026 最佳实践

---

## 下一步学习

- **核心概念 5**：类型推断机制 - 深入理解类型推断
- **核心概念 2**：TypeVar 与泛型 - 复习泛型知识
- **核心概念 3**：Protocol 协议 - 复习 Protocol

---

## 参考资源

1. [LangChain Runnables 官方文档](https://reference.langchain.com/python/langchain_core/runnables) - 2025-2026
2. [What Is a Runnable Interface in LangChain?](https://medium.com/@adatiyavinayshaileshbhai/what-is-a-runnable-interface-in-langchain-987991752afa) - Medium
3. [The Complete Guide to LangChain & LangGraph: 2025 Updates](https://ai.plainenglish.io/the-complete-guide-to-langchain-langgraph-2025-updates-and-production-ready-ai-frameworks-58bdb49a34b6)
4. [LangChain Runnable 源码](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/runnables/base.py)
5. [Building Production-Ready AI Pipelines with LangChain Runnables](https://medium.com/@sajo02/building-production-ready-ai-pipelines-with-langchain-runnables-a-complete-lcel-guide-2f9b27f6d557) - 2025
