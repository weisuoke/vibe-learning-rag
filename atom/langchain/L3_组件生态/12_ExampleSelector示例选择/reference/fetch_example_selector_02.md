---
type: fetched_content
source: https://www.pinecone.io/learn/series/langchain/langchain-prompt-templates
title: Prompt Engineering and LLMs with Langchain | Pinecone
fetched_at: 2026-02-25
status: success
author: Pinecone
knowledge_point: ExampleSelector示例选择
fetch_tool: Grok-mcp web_fetch
---

# Prompt Engineering and LLMs with Langchain

[系列导航：LangChain 系列文章]

我们一直依赖不同的模型来完成机器学习中的不同任务。随着多模态和大语言模型（LLMs）的出现，这种情况发生了改变。

过去，我们需要为分类、命名实体识别（NER）、问答（QA）等任务训练不同的模型。

在引入 transformers 和 *迁移学习* 之前，不同的任务和用例需要训练不同的模型。

借助 transformers 和迁移学习思想，我们只需在网络末端添加几个小层（即 *head*）并进行少量微调，就能让语言模型适配不同任务。

如今，连这种方法都显得过时了。为什么要修改最后的模型层并进行完整的微调，当你可以通过提示（prompt）直接让模型完成分类或问答呢？

许多任务现在只需使用同一个大语言模型（LLM），通过改变提示中的指令就能实现。

**大语言模型（LLMs）** 可以完成上述所有任务甚至更多。这些模型基于一个简单概念进行训练：输入一段文本，模型输出一段文本。这里唯一的变量就是输入文本——也就是 **提示（prompt）**。

**开始免费使用 Pinecone**
Pinecone 是开发者最喜欢的[向量数据库](/learn/vector-database/)，速度快、易用，可支持任意规模。

在 LLM 时代，**提示就是王道**。糟糕的提示产生糟糕的输出，而优秀的提示拥有超乎想象的力量。构建好的提示是使用 LLM 构建应用的关键技能。

[LangChain](/learn/series/langchain/langchain-intro/) 库充分认识到提示的力量，并为此构建了一整套对象。本文将全面介绍 `PromptTemplates` 及其有效实现方式。

## Prompt Engineering

在深入了解 LangChain 的 `PromptTemplate` 之前，我们需要更好地理解提示以及 **提示工程（Prompt Engineering）** 这门学科。

一个典型的提示通常由多个部分组成：

（此处原文有提示结构示意图，描述如下：一个典型的提示结构图，包含以下组件从上到下排列）

- **Instructions**（指令）
- **External information / Context(s)**（外部信息 / 上下文）
- **User input / Query**（用户输入 / 查询）
- **Output indicator**（输出指示符）

并非所有提示都包含这些组件，但一个好的提示通常会使用其中两个或更多。让我们更精确地定义它们：

**Instructions**（指令）：告诉模型要做什么、如何使用提供的外部信息、如何处理查询，以及如何构建输出。

**External information** 或 **Context(s)**（外部信息/上下文）：作为模型的额外知识来源。可以手动插入提示中，通过向量数据库检索（RAG）、API 调用、计算等方式获取。

**User input** 或 **Query**（用户输入/查询）：通常（但不总是）由人类用户（提示者）输入系统的查询。

**Output indicator**（输出指示符）：标记待生成文本的 *开始*。例如生成 Python 代码时，可以用 `import` 作为开头指示，因为大多数 Python 脚本都以 `import` 开始。

这些组件通常按以下顺序出现在提示中：指令 → 外部信息（若有）→ 用户输入 → 输出指示符。

下面看看如何使用 LangChain 将这样的提示输入给 OpenAI 模型：

```python
prompt = """Answer the question based on the context below. If the
question cannot be answered using the information provided answer
with "I don't know".

Context: Large Language Models (LLMs) are the latest models used in NLP.
Their superior performance over smaller models has made them incredibly
useful for developers building NLP enabled applications. These models
can be accessed via Hugging Face's `transformers` library, via OpenAI
using the `openai` library, and via Cohere using the `cohere` library.

Question: Which libraries and model providers offer LLMs?

Answer: """
```

```python
from langchain.llms import OpenAI

# 初始化模型
openai = OpenAI(
    model_name="text-davinci-003",
    openai_api_key="YOUR_API_KEY"
)
```

```python
print(openai(prompt))
```

输出示例：
```
Hugging Face's `transformers` library, OpenAI using the `openai` library, and Cohere using the `cohere` library.
```

实际上，我们不太可能对上下文和用户问题进行硬编码。通常会通过 *模板* 来动态传入它们——这正是 LangChain 的 `PromptTemplate` 的用武之地。

## Prompt Templates

LangChain 中的提示模板类旨在让构建带有动态输入的提示变得更简单。其中最基础的是 `PromptTemplate`。我们通过给之前的提示增加一个动态输入（用户 `query`）来测试它。

```python
from langchain import PromptTemplate

template = """Answer the question based on the context below. If the
question cannot be answered using the information provided answer
with "I don't know".

Context: Large Language Models (LLMs) are the latest models used in NLP.
Their superior performance over smaller models has made them incredibly
useful for developers building NLP enabled applications. These models
can be accessed via Hugging Face's `transformers` library, via OpenAI
using the `openai` library, and via Cohere using the `cohere` library.

Question: {query}

Answer: """

prompt_template = PromptTemplate(
    input_variables=["query"],
    template=template
)
```

使用 `format` 方法查看传入 `query` 后的效果：

```python
print(
    prompt_template.format(
        query="Which libraries and model providers offer LLMs?"
    )
)
```

输出：
```
Answer the question based on the context below. If the
question cannot be answered using the information provided answer
with "I don't know".

Context: Large Language Models (LLMs) are the latest models used in NLP.
Their superior performance over smaller models has made them incredibly
useful for developers building NLP enabled applications. These models
can be accessed via Hugging Face's `transformers` library, via OpenAI
using the `openai` library, and via Cohere using the `cohere` library.

Question: Which libraries and model providers offer LLMs?

Answer:
```

当然，我们可以直接将结果传入 LLM：

```python
print(openai(
    prompt_template.format(
        query="Which libraries and model providers offer LLMs?"
    )
))
```

输出：
```
Hugging Face's `transformers` library, OpenAI using the `openai` library, and Cohere using the `cohere` library.
```

这只是一个简单实现，完全可以用 f-string 替代。但使用 LangChain 的 `PromptTemplate` 对象，我们可以让过程更规范化、支持多个参数，并采用面向对象的方式构建提示。

这些是显著的优势，但 LangChain 为提示提供的功能远不止这些。

### Few Shot Prompt Templates

LLM 的成功来源于其巨大规模以及在训练过程中将"知识"存储在模型参数中的能力。但向 LLM 传递知识还有其他方式。两种主要方法是：

- **Parametric knowledge** —— 训练时学到的、存储在模型权重（参数）中的知识。
- **Source knowledge** —— 在推理时通过输入提示提供的知识。

LangChain 的 `FewShotPromptTemplate` 专门用于处理 **source knowledge**。其核心思想是通过在提示中给出少量示例（few-shot learning）来"训练"模型。

当模型需要帮助理解我们的要求时，few-shot learning 非常有效。请看下面的例子：

```python
prompt = """The following is a conversation with an AI assistant.
The assistant is typically sarcastic and witty, producing creative
and funny responses to the users questions. Here are some examples:

User: What is the meaning of life?
AI: """
```

```python
openai.temperature = 1.0  # 提高输出的创造性/随机性

print(openai(prompt))
```

输出示例（可能偏严肃）：
```
Life is like a box of chocolates, you never know what you're gonna get!
```

这里我们希望得到幽默的回答，但即使 temperature=1.0，模型仍可能给出较为正经的回答。

我们可以提供几个符合期望风格的示例来引导模型：

```python
prompt = """The following are exerpts from conversations with an AI
assistant. The assistant is typically sarcastic and witty, producing
creative  and funny responses to the users questions. Here are some
examples:

User: How are you?
AI: I can't complain but sometimes I still do.

User: What time is it?
AI: It's time to get a watch.

User: What is the meaning of life?
AI: """

print(openai(prompt))
```

输出示例：
```
42, of course!
```

通过示例强化指令，我们更容易获得幽默的回答。接下来使用 LangChain 的 `FewShotPromptTemplate` 规范化这个过程：

```python
from langchain import FewShotPromptTemplate

# 创建示例
examples = [
    {
        "query": "How are you?",
        "answer": "I can't complain but sometimes I still do."
    }, {
        "query": "What time is it?",
        "answer": "It's time to get a watch."
    }
]

# 创建示例模板
example_template = """
User: {query}
AI: {answer}
"""

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# 分离前缀（指令）和后缀（用户输入 + 输出指示）
prefix = """The following are exerpts from conversations with an AI
assistant. The assistant is typically sarcastic and witty, producing
creative  and funny responses to the users questions. Here are some
examples:
"""

suffix = """
User: {query}
AI: """

# 创建 FewShotPromptTemplate
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)
```

测试：

```python
query = "What is the meaning of life?"

print(few_shot_prompt_template.format(query=query))
```

输出：
```
The following are exerpts from conversations with an AI
assistant. The assistant is typically sarcastic and witty, producing
creative  and funny responses to the users questions. Here are some
examples:

User: How are you?
AI: I can't complain but sometimes I still do.

User: What time is it?
AI: It's time to get a watch.

User: What is the meaning of life?
AI:
```

这个过程看似复杂，但相比纯 f-string，它更规范、与 LangChain 其他功能（如 chains）集成更好，并提供了一些高级特性，例如根据查询长度动态调整示例数量。

（文章后续部分通常还会介绍 LengthBasedExampleSelector、更多示例选择器、其他模板类型如 ChatPromptTemplate 等，以及资源链接）

## Resources

（此处通常包含进一步阅读链接、LangChain 文档、Pinecone 相关资源等）

---

（注：以上为页面主要内容的结构化 Markdown 还原，代码块、格式、段落均尽量保持与原网页一致。部分代码输出为示例，实际运行可能因模型版本或随机性略有差异。页面可能包含少量图片或交互元素，此处以文字描述为主。）
