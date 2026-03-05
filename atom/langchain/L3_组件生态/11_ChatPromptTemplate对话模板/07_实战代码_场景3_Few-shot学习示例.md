# 实战代码：场景3 - Few-shot学习示例

> **目标**：掌握使用 AI 角色提供示例答案的 Few-shot 学习模式，实现格式化输出和任务模式学习

---

## 场景概述

**适用场景**：
- 翻译任务（提供示例翻译对）
- 分类任务（展示分类模式）
- 格式化输出（定义输出格式）
- 风格模仿（展示期望的写作风格）

**技术栈**：
- `langchain_core.prompts.ChatPromptTemplate`
- `langchain_openai.ChatOpenAI`
- `langchain_core.output_parsers.StrOutputParser`

**核心概念**：
- **Few-shot Learning**: 通过少量示例让 LLM 学习任务模式
- **AI 角色**: 使用 AI 消息展示期望的输出格式
- **示例对**: Human-AI 消息对构成学习示例

---

## 实战1：基础 Few-shot 翻译

### 代码实现

```python
"""
实战1：基础 Few-shot 翻译
演示如何使用 AI 角色提供翻译示例
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def demo_fewshot_translation():
    """演示 Few-shot 翻译"""
    print("=== 实战1：基础 Few-shot 翻译 ===\n")

    # 1. 创建 Few-shot 翻译模板
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a professional translator. Translate English to French."),
        # 示例 1
        ("human", "Hello"),
        ("ai", "Bonjour"),
        # 示例 2
        ("human", "Goodbye"),
        ("ai", "Au revoir"),
        # 示例 3
        ("human", "Thank you"),
        ("ai", "Merci"),
        # 用户输入
        ("human", "{input}")
    ])

    print("1. 创建 Few-shot 翻译模板（3个示例）")
    print(f"   示例数量: 3")
    print(f"   输入变量: {template.input_variables}")

    # 2. 配置模型
    model = ChatOpenAI(model="gpt-4", temperature=0)
    print("\n2. 配置模型: gpt-4 (temperature=0)")

    # 3. 创建链
    chain = template | model | StrOutputParser()
    print("3. 创建处理链: template | model | parser")

    # 4. 测试翻译
    test_inputs = [
        "Good morning",
        "How are you?",
        "See you later"
    ]

    print("\n4. 测试翻译:")
    for i, text in enumerate(test_inputs, 1):
        result = chain.invoke({"input": text})
        print(f"   [{i}] EN: {text}")
        print(f"       FR: {result}")

if __name__ == "__main__":
    demo_fewshot_translation()
```

### 运行结果

```
=== 实战1：基础 Few-shot 翻译 ===

1. 创建 Few-shot 翻译模板（3个示例）
   示例数量: 3
   输入变量: ['input']

2. 配置模型: gpt-4 (temperature=0)
3. 创建处理链: template | model | parser

4. 测试翻译:
   [1] EN: Good morning
       FR: Bonjour
   [2] EN: How are you?
       FR: Comment allez-vous?
   [3] EN: See you later
       FR: À plus tard
```

### 代码解析

**核心要点**：
1. **示例对结构**: `("human", "示例输入")` + `("ai", "期望输出")`
2. **示例数量**: 3-5个示例通常足够，过多会浪费 tokens
3. **temperature=0**: 确保输出稳定，遵循示例模式
4. **系统消息**: 定义任务类型（翻译）

---

## 实战2：Few-shot 分类任务

### 代码实现

```python
"""
实战2：Few-shot 分类任务
演示如何使用示例进行情感分类
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def demo_fewshot_classification():
    """演示 Few-shot 情感分类"""
    print("=== 实战2：Few-shot 分类任务 ===\n")

    # 1. 创建分类模板
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a sentiment classifier. Classify the sentiment as: Positive, Negative, or Neutral."),
        # 正面示例
        ("human", "I love this product! It's amazing!"),
        ("ai", "Positive"),
        # 负面示例
        ("human", "This is terrible. I hate it."),
        ("ai", "Negative"),
        # 中性示例
        ("human", "The product arrived on time."),
        ("ai", "Neutral"),
        # 混合示例
        ("human", "It's okay, but could be better."),
        ("ai", "Neutral"),
        # 用户输入
        ("human", "{text}")
    ])

    print("1. 创建情感分类模板（4个示例）")
    print("   类别: Positive, Negative, Neutral")

    # 2. 配置模型
    model = ChatOpenAI(model="gpt-4", temperature=0)
    chain = template | model | StrOutputParser()
    print("\n2. 配置模型和处理链")

    # 3. 测试分类
    test_texts = [
        "This is the best thing ever!",
        "I'm disappointed with the quality.",
        "The package was delivered.",
        "Not bad, but not great either.",
        "Absolutely fantastic experience!"
    ]

    print("\n3. 测试分类:")
    for i, text in enumerate(test_texts, 1):
        sentiment = chain.invoke({"text": text})
        print(f"   [{i}] Text: {text}")
        print(f"       Sentiment: {sentiment}")

    # 4. 批量处理
    print("\n4. 批量处理:")
    results = chain.batch([{"text": t} for t in test_texts])
    for text, sentiment in zip(test_texts, results):
        print(f"   {sentiment:8} | {text}")

if __name__ == "__main__":
    demo_fewshot_classification()
```

### 运行结果

```
=== 实战2：Few-shot 分类任务 ===

1. 创建情感分类模板（4个示例）
   类别: Positive, Negative, Neutral

2. 配置模型和处理链

3. 测试分类:
   [1] Text: This is the best thing ever!
       Sentiment: Positive
   [2] Text: I'm disappointed with the quality.
       Sentiment: Negative
   [3] Text: The package was delivered.
       Sentiment: Neutral
   [4] Text: Not bad, but not great either.
       Sentiment: Neutral
   [5] Text: Absolutely fantastic experience!
       Sentiment: Positive

4. 批量处理:
   Positive | This is the best thing ever!
   Negative | I'm disappointed with the quality.
   Neutral  | The package was delivered.
   Neutral  | Not bad, but not great either.
   Positive | Absolutely fantastic experience!
```

---

## 实战3：Few-shot 格式化输出

### 代码实现

```python
"""
实战3：Few-shot 格式化输出
演示如何使用示例定义输出格式
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import json

load_dotenv()

def demo_fewshot_formatting():
    """演示 Few-shot 格式化输出"""
    print("=== 实战3：Few-shot 格式化输出 ===\n")

    # 1. 创建格式化模板（JSON 输出）
    template = ChatPromptTemplate.from_messages([
        ("system", "Extract person information and output as JSON."),
        # 示例 1
        ("human", "John Smith is 30 years old and works as a software engineer."),
        ("ai", '{"name": "John Smith", "age": 30, "occupation": "software engineer"}'),
        # 示例 2
        ("human", "Mary Johnson, age 25, is a teacher."),
        ("ai", '{"name": "Mary Johnson", "age": 25, "occupation": "teacher"}'),
        # 示例 3
        ("human", "Bob Lee is a 35-year-old doctor."),
        ("ai", '{"name": "Bob Lee", "age": 35, "occupation": "doctor"}'),
        # 用户输入
        ("human", "{text}")
    ])

    print("1. 创建格式化模板（JSON 输出）")
    print("   输出格式: {name, age, occupation}")

    # 2. 配置模型
    model = ChatOpenAI(model="gpt-4", temperature=0)
    chain = template | model | StrOutputParser()
    print("\n2. 配置模型和处理链")

    # 3. 测试提取
    test_texts = [
        "Alice Wang is a 28-year-old data scientist.",
        "Tom Brown, 42, works as a manager.",
        "Sarah Davis is 31 and she's a designer."
    ]

    print("\n3. 测试信息提取:")
    for i, text in enumerate(test_texts, 1):
        result = chain.invoke({"text": text})
        print(f"\n   [{i}] Input: {text}")
        print(f"       Output: {result}")

        # 验证 JSON 格式
        try:
            data = json.loads(result)
            print(f"       Parsed: Name={data['name']}, Age={data['age']}, Job={data['occupation']}")
        except json.JSONDecodeError:
            print(f"       Warning: Invalid JSON format")

if __name__ == "__main__":
    demo_fewshot_formatting()
```

### 运行结果

```
=== 实战3：Few-shot 格式化输出 ===

1. 创建格式化模板（JSON 输出）
   输出格式: {name, age, occupation}

2. 配置模型和处理链

3. 测试信息提取:

   [1] Input: Alice Wang is a 28-year-old data scientist.
       Output: {"name": "Alice Wang", "age": 28, "occupation": "data scientist"}
       Parsed: Name=Alice Wang, Age=28, Job=data scientist

   [2] Input: Tom Brown, 42, works as a manager.
       Output: {"name": "Tom Brown", "age": 42, "occupation": "manager"}
       Parsed: Name=Tom Brown, Age=42, Job=manager

   [3] Input: Sarah Davis is 31 and she's a designer.
       Output: {"name": "Sarah Davis", "age": 31, "occupation": "designer"}
       Parsed: Name=Sarah Davis, Age=31, Job=designer
```

---

## 实战4：Few-shot 与 RAG 集成

### 代码实现

```python
"""
实战4：Few-shot 与 RAG 集成
演示如何在 RAG 系统中使用 Few-shot 学习
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def demo_fewshot_rag():
    """演示 Few-shot RAG 问答"""
    print("=== 实战4：Few-shot 与 RAG 集成 ===\n")

    # 1. 创建 RAG Few-shot 模板
    template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer questions based on the provided context.
If the answer is not in the context, say "I don't have enough information to answer that."
Always cite the relevant part of the context in your answer."""),

        # 示例 1：有答案的情况
        ("human", """Context: LangChain is a framework for building LLM applications. It provides tools for prompt management, chains, and agents.

Question: What is LangChain?"""),
        ("ai", "LangChain is a framework for building LLM applications. According to the context, it provides tools for prompt management, chains, and agents."),

        # 示例 2：无答案的情况
        ("human", """Context: Python is a high-level programming language known for its simplicity.

Question: What is the capital of France?"""),
        ("ai", "I don't have enough information to answer that. The provided context is about Python programming language and doesn't contain information about the capital of France."),

        # 用户输入
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])

    print("1. 创建 RAG Few-shot 模板")
    print("   示例类型: 有答案 + 无答案")

    # 2. 配置模型
    model = ChatOpenAI(model="gpt-4", temperature=0)
    chain = template | model | StrOutputParser()
    print("\n2. 配置模型和处理链")

    # 3. 测试 RAG 问答
    test_cases = [
        {
            "context": "Milvus is a vector database designed for AI applications. It supports similarity search and can handle billions of vectors.",
            "question": "What is Milvus?"
        },
        {
            "context": "Milvus is a vector database designed for AI applications. It supports similarity search and can handle billions of vectors.",
            "question": "How do I install Docker?"
        },
        {
            "context": "RAG (Retrieval-Augmented Generation) combines retrieval and generation. It first retrieves relevant documents, then generates answers based on them.",
            "question": "What does RAG stand for?"
        }
    ]

    print("\n3. 测试 RAG 问答:")
    for i, case in enumerate(test_cases, 1):
        print(f"\n   [{i}] Question: {case['question']}")
        print(f"       Context: {case['context'][:60]}...")

        answer = chain.invoke(case)
        print(f"       Answer: {answer}")

if __name__ == "__main__":
    demo_fewshot_rag()
```

### 运行结果

```
=== 实战4：Few-shot 与 RAG 集成 ===

1. 创建 RAG Few-shot 模板
   示例类型: 有答案 + 无答案

2. 配置模型和处理链

3. 测试 RAG 问答:

   [1] Question: What is Milvus?
       Context: Milvus is a vector database designed for AI applications...
       Answer: Milvus is a vector database designed for AI applications. According to the context, it supports similarity search and can handle billions of vectors.

   [2] Question: How do I install Docker?
       Context: Milvus is a vector database designed for AI applications...
       Answer: I don't have enough information to answer that. The provided context is about Milvus vector database and doesn't contain information about installing Docker.

   [3] Question: What does RAG stand for?
       Context: RAG (Retrieval-Augmented Generation) combines retrieval...
       Answer: RAG stands for Retrieval-Augmented Generation. According to the context, it combines retrieval and generation by first retrieving relevant documents, then generating answers based on them.
```

---

## 最佳实践总结

### 1. 示例数量选择

```python
# 推荐示例数量
- 简单任务（翻译、分类）: 3-5个示例
- 复杂任务（格式化、推理）: 5-8个示例
- 避免过多示例: 超过10个会浪费 tokens
```

### 2. 示例质量

```python
# 好的示例特征
✅ 覆盖不同情况（正面、负面、边界）
✅ 输出格式一致
✅ 示例简洁明了
✅ 代表性强

# 避免的问题
❌ 示例过于相似
❌ 输出格式不一致
❌ 示例过长
❌ 边界情况缺失
```

### 3. 模型配置

```python
# Few-shot 推荐配置
model = ChatOpenAI(
    model="gpt-4",           # 使用更强的模型
    temperature=0,           # 确保输出稳定
    max_tokens=None          # 根据任务调整
)
```

### 4. 与其他技术结合

```python
# Few-shot + Output Parser
chain = template | model | JsonOutputParser()

# Few-shot + RAG
rag_chain = retriever | format_docs | fewshot_template | model

# Few-shot + Memory
conversation = fewshot_template | model | memory
```

---

## 常见问题

### Q1: Few-shot 和 Zero-shot 如何选择？

**答案**：
- **Zero-shot**: 任务简单、模型能力强、不需要特定格式
- **Few-shot**: 需要特定格式、复杂任务、提高准确性

### Q2: 示例应该放在哪里？

**答案**：
```python
# 方式1: 放在消息列表中（推荐）
messages = [
    ("system", "任务说明"),
    ("human", "示例1输入"),
    ("ai", "示例1输出"),
    ("human", "{input}")
]

# 方式2: 放在系统消息中
system_msg = """任务说明

示例:
输入: 示例1输入
输出: 示例1输出
"""
```

### Q3: 如何处理示例不够的情况？

**答案**：
1. 使用更强的模型（如 GPT-4）
2. 在系统消息中详细说明任务
3. 使用 Chain-of-Thought 提示
4. 结合 Output Parser 验证输出

---

## 参考资料

- **LangChain 官方文档**: [Few-shot Examples](https://python.langchain.com/docs/modules/prompts/few_shot_examples)
- **Context7 文档**: [ChatPromptTemplate 消息类型](reference/context7_langchain_02.md)
- **源码参考**: `langchain_core/prompts/chat.py`

---

**版本**: v1.0
**最后更新**: 2026-02-26
**适用 LangChain 版本**: 0.2.x - 0.3.x
**维护者**: Claude Code
