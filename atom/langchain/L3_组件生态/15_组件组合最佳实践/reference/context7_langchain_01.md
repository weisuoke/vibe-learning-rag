---
type: context7_documentation
library: langchain
version: latest (2026)
fetched_at: 2026-02-27
knowledge_point: 15_组件组合最佳实践
context7_query: "component composition best practices LCEL pipe operator RunnableSequence RunnableParallel"
---

# Context7 文档：LangChain 组件组合

## 文档来源
- 库名称：LangChain
- Context7 Library ID: /websites/langchain
- 官方文档链接：https://docs.langchain.com

## 关键信息提取

### 1. LCEL 管道组合模式

基本的 LCEL 链构建使用 `|` 操作符连接 PromptTemplate → Model → OutputParser：

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
    ("human", "{text}"),
])

chain = prompt | model | StrOutputParser()
result = chain.invoke({
    "input_language": "English",
    "output_language": "Spanish",
    "text": "Hello, how are you?"
})
```

### 2. RAG 链中的 RunnablePassthrough 模式

RAG 链中常用 RunnablePassthrough 传递 question 同时检索 context：

```python
from langchain_core.runnables import RunnablePassthrough

chain = (
    {
        "question": RunnablePassthrough(),
        "context": parent_retriever
        | (lambda docs: "\n\n".join(d.page_content for d in docs)),
    }
    | prompt
    | model
    | StrOutputParser()
)
```

### 3. RunnableGenerator 流式管道

使用 RunnableGenerator 构建流式处理管道，支持并发处理：

```python
from langchain_core.runnables import RunnableGenerator

pipeline = (
    RunnableGenerator(stt_stream)      # Audio → STT events
    | RunnableGenerator(agent_stream)  # STT events → Agent events
    | RunnableGenerator(tts_stream)    # Agent events → TTS audio
)
```

### 4. 重试策略

LangGraph 中的重试策略配置：
- maxAttempts: 最大重试次数
- initialInterval: 初始重试间隔
- 支持自定义异常过滤

### 5. 并发流式处理

RunnableGenerator 支持各阶段独立并发处理，实现 sub-700ms 延迟。
每个阶段在前一阶段产出数据后立即开始处理，无需等待完整完成。
