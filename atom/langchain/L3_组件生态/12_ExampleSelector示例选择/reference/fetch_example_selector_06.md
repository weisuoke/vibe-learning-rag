---
type: fetched_content
source: https://pub.aimind.so/langchain-in-chains-6-example-selectors-310f47b4cdf3
title: LangChain in Chains #6: Example Selectors
fetched_at: 2026-02-25
status: success
author: Okan Yenigün
knowledge_point: ExampleSelector示例选择
fetch_tool: Grok-mcp web_fetch
---

# LangChain in Chains #6: Example Selectors

**Author:** Okan Yenigün
**Publication:** AI Mind
**Date:** Jan 22, 2024
**Reading Time:** 12 min read

Feeding examples to a language model helps tailor its responses to be more accurate, relevant, and appropriate for the specific application, audience, and context. They provide context to the model and help the model to understand the domain.

## Understanding the Role of Example Selectors

An *example selector* in LangChain helps us choose which examples to include in our prompts.

Few-shot learning templates are a way to guide models to perform specific tasks with a limited amount of examples or data. A few-shot learning template typically includes a small set of examples that clearly demonstrate the task we want the model to perform.

In practice, we combine few-shot learners with example selection.

### Basic Example Without Selector

```python
# define your openai key
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"

# asking questions directly
from langchain.llms import OpenAI

prompt = """You are an alien from Mars:

Question: What are human homes like?
Response:
"""

llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

print(llm(prompt))
```

The response is quite objective and has a definitive, scientific linguistic tone.

### Adding Fixed Examples

```python
prompt = """You are an alien from Mars:
Here are some examples:

Question: What is human cuisine like?
Response: Their cuisine is a simplistic combination of various organic matter,
often heated in rudimentary ways. It's unrefined and unstructured,
especially compared to our molecular gastronomy.

Question: What is human entertainment?
Response: Crude moving images and loud sounds.

Question: What are human homes like?
Response: """

print(llm(prompt))
```

As you can observe, its responses become increasingly rude. The model has enhanced its grasp of the role (presumably that of a snobbish alien), resulting in more tailored, role-played responses.

### Using FewShotPromptTemplate

```python
from langchain.prompts import PromptTemplate
from langchain import FewShotPromptTemplate

# examples
examples = [
    {
        "query": "What is human cuisine like?",
        "answer": "Their cuisine is a simplistic combination of various organic matter, often heated in rudimentary ways. It's unrefined and unstructured, especially compared to our molecular gastronomy."
    }, {
        "query": "What is human entertainment?",
        "answer": "Crude moving images and loud sounds."
    },
]

# example prompt template
example_template = """
Question: {query}
Response: {answer}
"""

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# few shot prompt template
prefix = """You are an alien from Mars:
Here are some examples:
"""

suffix = """
Question: {userInput}
Response: """

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["userInput"],
    example_separator="\n\n"
)

query = "What are human homes like?"
print(llm(few_shot_prompt_template.format(userInput=query)))
```

## Example Selectors

### 1. LengthBasedExampleSelector

`LengthBasedExampleSelector` chooses examples based on their length, ensuring they fit within a specified constraint for your prompt. This is useful when dealing with character or token limitations.

```python
from langchain.prompts.example_selector import LengthBasedExampleSelector

# More examples with different lengths
examples = [
    {
        "query": "What is human cuisine like?",
        "answer": "Their cuisine is a simplistic combination of various organic matter, often heated in rudimentary ways. It's unrefined and unstructured, especially compared to our molecular gastronomy."
    }, {
        "query": "What is human entertainment?",
        "answer": "Crude moving images and loud sounds."
    }, {
        "query": "What do humans use for transportation?",
        "answer": "Humans rely on archaic and inefficient rolling contraptions they proudly call 'cars.' These are remarkably primitive compared to our teleportation beams and anti-gravity vessels."
    }, {
        "query": "How do humans communicate with each other?",
        "answer": "They use a very basic form of communication involving the modulation of sound waves, referred to as 'speech.' Astonishingly primitive compared to our telepathic links."
    }, {
        "query": "How do humans maintain health?",
        "answer": "Consuming organic compounds and performing physical movements."
    }, {
        "query": "What is human education?",
        "answer": "They engage in a very basic form of knowledge transfer in places called 'schools.' It's a slow and inefficient process compared to our instant knowledge assimilation."
    }, {
        "query": "How do humans manage their societies?",
        "answer": "Through chaotic and inefficient systems."
    }, {
        "query": "What is human art?",
        "answer": "Their art is a primitive expression through physical mediums like paint and stone, lacking the sophistication of our holographic emotion sculptures."
    },
]

# LengthBasedExampleSelector
MAX_LENGTH = 100

example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=MAX_LENGTH
)

new_prompt_template = FewShotPromptTemplate(
    example_selector=example_selector,  # use example_selector instead of examples
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["userInput"],
    example_separator="\n"
)
```

### 2. SemanticSimilarityExampleSelector

Selects examples based on semantic similarity to the input using embeddings and vector similarity.

```python
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    Chroma,
    k=2
)

similar_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["userInput"],
    example_separator="\n\n"
)
```

### 3. MaxMarginalRelevanceExampleSelector

Balances relevance and diversity using MMR algorithm.

### 4. NGramOverlapExampleSelector

Selects examples based on n-gram overlap with the input.

### 5. Custom Example Selector

You can create custom selectors by implementing the `BaseExampleSelector` interface.

## Key Takeaways

1. **Example selectors** help choose the most relevant examples dynamically
2. **LengthBasedExampleSelector** manages token limits
3. **SemanticSimilarityExampleSelector** finds semantically similar examples
4. **MMR** balances relevance and diversity
5. **Custom selectors** allow for domain-specific logic

---

**注**：完整文章包含更多选择器类型的详细实现和 Chat Models 相关内容。此处为精简版，聚焦核心概念和代码示例。
