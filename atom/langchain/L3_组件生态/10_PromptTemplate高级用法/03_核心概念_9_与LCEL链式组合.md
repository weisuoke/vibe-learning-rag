# 核心概念:与LCEL链式组合

> 本文档讲解 PromptTemplate 与 LCEL (LangChain Expression Language) 的链式组合,包括 `|` 操作符、与各组件的集成、表达式模式等核心特性

---

## 概述

**核心价值**:通过 LCEL 表达式将 PromptTemplate 与其他组件无缝组合,构建声明式、可组合、可观测的 AI 应用链路。

**适用场景**:
- RAG 系统的检索-生成链路
- Agent 系统的推理-执行链路
- 多步骤的复杂工作流
- 需要流式输出的实时应用

[来源: reference/context7_langchain_01.md + reference/context7_langchain_03.md]

---

## 1. LCEL 基础概念

### 1.1 什么是 LCEL?

**LCEL (LangChain Expression Language)** 是 LangChain 的声明式组合语言,使用 `|` 操作符将多个 Runnable 组件串联成链。

**核心特性**:
- **声明式**:用表达式描述数据流,而非命令式代码
- **可组合**:任何 Runnable 都可以与其他 Runnable 组合
- **可观测**:自动支持流式输出、批处理、异步执行
- **类型安全**:编译时检查输入输出类型

[来源: reference/context7_langchain_03.md]

### 1.2 Runnable 协议

所有可以用 LCEL 组合的组件都实现了 `Runnable` 协议:

```python
from langchain_core.runnables import Runnable

class Runnable:
    def invoke(self, input: Input) -> Output:
        """同步调用"""
        pass

    def ainvoke(self, input: Input) -> Output:
        """异步调用"""
        pass

    def stream(self, input: Input) -> Iterator[Output]:
        """流式输出"""
        pass

    def batch(self, inputs: List[Input]) -> List[Output]:
        """批量处理"""
        pass
```

**PromptTemplate 是 Runnable**:
- 输入:`Dict[str, Any]` (变量字典)
- 输出:`PromptValue` (格式化后的 Prompt)

---

## 2. 管道操作符 `|`

### 2.1 基础用法

**语法**:

```python
chain = component1 | component2 | component3
```

**工作原理**:
1. `component1` 的输出作为 `component2` 的输入
2. `component2` 的输出作为 `component3` 的输入
3. 最终返回 `component3` 的输出

**示例**:

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 创建组件
prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
llm = ChatOpenAI(model="gpt-4")

# 使用 | 操作符组合
chain = prompt | llm

# 调用
response = chain.invoke({"topic": "cats"})
print(response.content)
```

[来源: reference/context7_langchain_01.md:52-72]

### 2.2 数据流转

```
输入 {"topic": "cats"}
  ↓
PromptTemplate.invoke()
  ↓
PromptValue("Tell me a joke about cats")
  ↓
ChatOpenAI.invoke()
  ↓
AIMessage("Why did the cat...")
```

---

## 3. 与 ChatModel 集成

### 3.1 基础集成

```python
"""
PromptTemplate + ChatModel 基础集成
演示:最简单的链式组合
"""

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os

# 设置 API Key
os.environ["OPENAI_API_KEY"] = "your-api-key"

# ===== 1. 创建组件 =====
prompt = PromptTemplate.from_template(
    "How to say {input} in {output_language}:\n"
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ===== 2. 组合成链 =====
chain = prompt | llm

# ===== 3. 调用 =====
result = chain.invoke({
    "output_language": "German",
    "input": "I love programming."
})

print("=== 翻译结果 ===")
print(result.content)
```

**运行输出**:

```
=== 翻译结果 ===
Ich liebe Programmieren.
```

[来源: reference/context7_langchain_01.md:53-67]

### 3.2 流式输出

```python
"""
流式输出示例
演示:实时获取 LLM 生成的内容
"""

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

prompt = PromptTemplate.from_template(
    "Write a short story about {topic} in 3 sentences."
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

chain = prompt | llm

# 流式输出
print("=== 流式生成 ===")
for chunk in chain.stream({"topic": "a robot learning to paint"}):
    print(chunk.content, end="", flush=True)

print("\n")
```

**运行输出**:

```
=== 流式生成 ===
In a small workshop, a robot named Artie discovered an old paintbrush...
```

### 3.3 批量处理

```python
"""
批量处理示例
演示:一次处理多个输入
"""

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

prompt = PromptTemplate.from_template("Translate '{text}' to {language}")
llm = ChatOpenAI(model="gpt-4o-mini")

chain = prompt | llm

# 批量调用
inputs = [
    {"text": "Hello", "language": "Spanish"},
    {"text": "Goodbye", "language": "French"},
    {"text": "Thank you", "language": "Japanese"}
]

results = chain.batch(inputs)

print("=== 批量翻译结果 ===")
for i, result in enumerate(results):
    print(f"{i+1}. {result.content}")
```

---

## 4. 与 OutputParser 集成

### 4.1 结构化输出

```python
"""
PromptTemplate + ChatModel + OutputParser
演示:将 LLM 输出解析为结构化数据
"""

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# ===== 1. 字符串输出解析 =====
prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
llm = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

result = chain.invoke({"topic": "programming"})
print("=== 字符串输出 ===")
print(type(result))  # <class 'str'>
print(result)

# ===== 2. JSON 输出解析 =====
class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke")

json_prompt = PromptTemplate.from_template(
    """Tell me a joke about {topic}.

    Return the joke in JSON format with 'setup' and 'punchline' fields.
    """
)

json_parser = JsonOutputParser(pydantic_object=Joke)

json_chain = json_prompt | llm | json_parser

result = json_chain.invoke({"topic": "AI"})
print("\n=== JSON 输出 ===")
print(type(result))  # <class 'dict'>
print(result)
# {'setup': 'Why did the AI...', 'punchline': '...'}
```

### 4.2 自定义 OutputParser

```python
"""
自定义 OutputParser
演示:提取特定格式的数据
"""

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import BaseOutputParser
from typing import List

class CommaSeparatedListOutputParser(BaseOutputParser[List[str]]):
    """解析逗号分隔的列表"""

    def parse(self, text: str) -> List[str]:
        """解析输出文本"""
        return [item.strip() for item in text.split(",")]

# 使用自定义 Parser
prompt = PromptTemplate.from_template(
    "List 5 {category}. Return as comma-separated values."
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = CommaSeparatedListOutputParser()

chain = prompt | llm | parser

result = chain.invoke({"category": "programming languages"})
print("=== 列表输出 ===")
print(type(result))  # <class 'list'>
print(result)
# ['Python', 'JavaScript', 'Java', 'C++', 'Go']
```

---

## 5. 复杂链式组合

### 5.1 多步骤链

```python
"""
多步骤链式组合
演示:构建复杂的多步骤工作流
"""

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 步骤1:生成故事大纲
outline_prompt = PromptTemplate.from_template(
    "Create a 3-sentence outline for a story about {topic}"
)

# 步骤2:扩展成完整故事
story_prompt = PromptTemplate.from_template(
    "Expand this outline into a full story:\n\n{outline}"
)

# 步骤3:提取关键词
keywords_prompt = PromptTemplate.from_template(
    "Extract 5 keywords from this story:\n\n{story}\n\nKeywords:"
)

llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# 组合成多步骤链
chain = (
    outline_prompt
    | llm
    | parser
    | (lambda outline: {"outline": outline})
    | story_prompt
    | llm
    | parser
    | (lambda story: {"story": story})
    | keywords_prompt
    | llm
    | parser
)

result = chain.invoke({"topic": "a time-traveling detective"})
print("=== 提取的关键词 ===")
print(result)
```

### 5.2 并行执行

```python
"""
并行执行多个链
演示:使用 RunnableParallel 同时执行多个任务
"""

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# 定义多个并行任务
pros_chain = (
    PromptTemplate.from_template("List 3 pros of {topic}")
    | llm
    | parser
)

cons_chain = (
    PromptTemplate.from_template("List 3 cons of {topic}")
    | llm
    | parser
)

summary_chain = (
    PromptTemplate.from_template("Summarize {topic} in one sentence")
    | llm
    | parser
)

# 并行执行
parallel_chain = RunnableParallel(
    pros=pros_chain,
    cons=cons_chain,
    summary=summary_chain
)

result = parallel_chain.invoke({"topic": "remote work"})

print("=== 并行执行结果 ===")
print(f"Pros:\n{result['pros']}\n")
print(f"Cons:\n{result['cons']}\n")
print(f"Summary:\n{result['summary']}")
```

---

## 6. RAG 系统集成

### 6.1 基础 RAG 链

```python
"""
RAG 系统链式组合
演示:检索 + 生成的完整流程
"""

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ===== 1. 准备向量存储 =====
texts = [
    "LangChain is a framework for building LLM applications.",
    "LCEL is the expression language for composing chains.",
    "PromptTemplate helps format inputs for LLMs.",
    "Retrieval-Augmented Generation combines search and generation."
]

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(texts, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ===== 2. 构建 RAG 链 =====
rag_prompt = PromptTemplate.from_template(
    """Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:"""
)

llm = ChatOpenAI(model="gpt-4o-mini")

# 组合 RAG 链
rag_chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# ===== 3. 查询 =====
result = rag_chain.invoke("What is LCEL?")
print("=== RAG 回答 ===")
print(result)
```

### 6.2 带来源引用的 RAG

```python
"""
带来源引用的 RAG 系统
演示:返回答案和来源文档
"""

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 准备数据
texts = [
    "Python is a high-level programming language.",
    "JavaScript is used for web development.",
    "Rust is known for memory safety."
]

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(texts, embeddings)
retriever = vectorstore.as_retriever()

# RAG Prompt
rag_prompt = PromptTemplate.from_template(
    """Answer based on context:

{context}

Question: {question}
Answer:"""
)

llm = ChatOpenAI(model="gpt-4o-mini")

# 构建链(返回答案和来源)
rag_chain_with_source = RunnableParallel(
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
).assign(
    answer=(
        lambda x: rag_prompt.format(
            context="\n\n".join([d.page_content for d in x["context"]]),
            question=x["question"]
        )
    )
    | llm
    | StrOutputParser()
)

result = rag_chain_with_source.invoke("What is Python?")

print("=== 答案 ===")
print(result["answer"])
print("\n=== 来源文档 ===")
for i, doc in enumerate(result["context"]):
    print(f"{i+1}. {doc.page_content}")
```

---

## 7. 双重类比

### 7.1 前端开发类比

| LCEL 概念 | 前端对应概念 | 说明 |
|----------|------------|------|
| `\|` 操作符 | Promise 链 `.then()` | 串联异步操作 |
| Runnable | 中间件函数 | 统一的接口协议 |
| `chain.invoke()` | `fetch().then()` | 同步调用 |
| `chain.stream()` | Server-Sent Events | 流式数据传输 |
| `chain.batch()` | `Promise.all()` | 批量并行处理 |
| RunnableParallel | `Promise.all()` | 并行执行多个任务 |
| RunnablePassthrough | 透传中间件 | 不修改数据,直接传递 |

**前端类比示例**:

```javascript
// 前端:Promise 链
fetch('/api/data')
  .then(response => response.json())
  .then(data => processData(data))
  .then(result => displayResult(result));

// LangChain:LCEL 链
chain = prompt | llm | parser
result = chain.invoke(input)
```

### 7.2 日常生活类比

| LCEL 概念 | 日常生活类比 | 说明 |
|----------|------------|------|
| LCEL 链 | 工厂流水线 | 每个工序处理后传递给下一个 |
| `\|` 操作符 | 传送带 | 连接各个工序 |
| PromptTemplate | 订单模板 | 标准化的输入格式 |
| ChatModel | 专家顾问 | 处理核心任务 |
| OutputParser | 质检员 | 验证和格式化输出 |
| RunnableParallel | 多条生产线 | 同时进行多个任务 |
| 流式输出 | 实时播报 | 边生产边播报进度 |

---

## 8. 反直觉点

### 误区1:`|` 操作符会创建新对象,影响性能 ❌

**为什么错?**
- LCEL 的 `|` 操作符只是创建轻量级的组合对象
- 实际执行时才会调用各个组件
- 性能开销可以忽略不计

**正确理解**:

```python
# 创建链(几乎零开销)
chain = prompt | llm | parser

# 执行链(才会真正调用 LLM)
result = chain.invoke(input)
```

### 误区2:必须使用 LCEL,不能用传统方式 ❌

**为什么错?**
- LCEL 和传统方式可以混用
- 简单场景用传统方式更直观
- LCEL 的优势在于复杂组合和可观测性

**正确理解**:

```python
# 传统方式(简单场景)
prompt_text = prompt.format(topic="cats")
response = llm.invoke(prompt_text)
result = parser.parse(response.content)

# LCEL 方式(复杂场景)
chain = prompt | llm | parser
result = chain.invoke({"topic": "cats"})

# 两者可以混用
traditional_result = some_function()
lcel_result = chain.invoke(traditional_result)
```

### 误区3:LCEL 链不能动态修改 ❌

**为什么错?**
- 可以使用 `RunnableBranch` 实现条件路由
- 可以使用 `RunnableLambda` 包装任意函数
- 可以在运行时动态选择组件

**正确理解**:

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

# 条件路由
branch = RunnableBranch(
    (lambda x: len(x["text"]) < 100, short_chain),
    (lambda x: len(x["text"]) < 500, medium_chain),
    long_chain  # 默认
)

# 动态链
def choose_llm(input_dict):
    if input_dict.get("use_gpt4"):
        return ChatOpenAI(model="gpt-4")
    else:
        return ChatOpenAI(model="gpt-4o-mini")

dynamic_chain = prompt | RunnableLambda(choose_llm) | parser
```

---

## 9. 最佳实践

### 9.1 链式组合原则

1. **单一职责**:每个组件只做一件事
2. **类型匹配**:确保前一个组件的输出类型匹配后一个组件的输入类型
3. **错误处理**:在关键节点添加错误处理逻辑
4. **可测试性**:每个组件都应该可以独立测试

### 9.2 性能优化

1. **批量处理**:对于多个输入,使用 `batch()` 而非循环调用 `invoke()`
2. **并行执行**:使用 `RunnableParallel` 并行执行独立任务
3. **流式输出**:对于长文本生成,使用 `stream()` 提升用户体验
4. **缓存**:对于重复查询,使用缓存避免重复调用 LLM

### 9.3 可观测性

```python
"""
添加日志和监控
演示:在链中添加观测点
"""

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

def log_input(x):
    print(f"[INPUT] {x}")
    return x

def log_output(x):
    print(f"[OUTPUT] {x}")
    return x

prompt = PromptTemplate.from_template("Tell me about {topic}")
llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# 添加日志节点
chain = (
    RunnableLambda(log_input)
    | prompt
    | llm
    | parser
    | RunnableLambda(log_output)
)

result = chain.invoke({"topic": "LCEL"})
```

---

## 10. 与其他功能的集成

### 10.1 与 Memory 集成

```python
"""
带记忆的对话链
演示:在链中集成对话历史
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

# 创建带历史的 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

chain = prompt | llm | parser

# 模拟对话历史
history = [
    HumanMessage(content="My name is Alice"),
    AIMessage(content="Nice to meet you, Alice!")
]

result = chain.invoke({
    "history": history,
    "input": "What's my name?"
})

print(result)  # "Your name is Alice."
```

### 10.2 与 Tools 集成

```python
"""
带工具调用的链
演示:在链中集成外部工具
"""

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

def search_tool(query: str) -> str:
    """模拟搜索工具"""
    # 实际应用中这里会调用真实的搜索 API
    return f"Search results for '{query}': [Result 1, Result 2, Result 3]"

# 步骤1:生成搜索查询
query_prompt = PromptTemplate.from_template(
    "Generate a search query for: {question}"
)

# 步骤2:执行搜索
search_step = RunnableLambda(lambda x: search_tool(x))

# 步骤3:基于搜索结果回答
answer_prompt = PromptTemplate.from_template(
    """Based on the search results, answer the question.

Search Results:
{search_results}

Question: {question}

Answer:"""
)

llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# 组合成完整链
chain = (
    {"question": RunnablePassthrough()}
    | query_prompt
    | llm
    | parser
    | search_step
    | (lambda results: {"search_results": results, "question": "original question"})
    | answer_prompt
    | llm
    | parser
)
```

---

## 参考资料

1. **官方文档**:
   - [LangChain LCEL 文档](reference/context7_langchain_01.md)
   - [PromptTemplate 链式组合](reference/context7_langchain_03.md)

2. **社区资源**:
   - [LangChain Prompt Templates Guide](reference/search_composition_01.md)
   - [Dynamic Prompts with LangChain](https://www.newline.co/@zaoyang/dynamic-prompts-with-langchain-templates)

3. **源码分析**:
   - [PromptTemplate 核心实现](reference/source_prompttemplate_01.md)

---

**版本**:v1.0
**最后更新**:2026-02-26
**知识点**:PromptTemplate高级用法 - 与LCEL链式组合
