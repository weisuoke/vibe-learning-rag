---
type: fetched_content
source: https://github.com/gkamradt/langchain-tutorials/blob/main/LangChain%20Cookbook%20Part%201%20-%20Fundamentals.ipynb
title: LangChain Cookbook Part 1 - Fundamentals
fetched_at: 2026-02-25
status: success
author: gkamradt
knowledge_point: ExampleSelector示例选择
fetch_tool: Grok-mcp web_fetch
---

# LangChain Cookbook Part 1 - Fundamentals

## Example Selectors 部分（精简版）

### 核心概念

Example Selectors 是一种动态选择示例的方式，允许根据输入动态放置上下文示例到 prompt 中。

### 代码示例

```python
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(model_name="text-davinci-003", openai_api_key=openai_api_key)

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Example Input: {input}\\nExample Output: {output}",
)

# Examples of locations that nouns are found
examples = [
    {"input": "pirate", "output": "ship"},
    {"input": "pilot", "output": "plane"},
    {"input": "driver", "output": "car"},
    {"input": "tree", "output": "ground"},
    {"input": "bird", "output": "nest"},
]

# SemanticSimilarityExampleSelector will select examples that are similar to your input by semantic meaning
example_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    examples,

    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIEmbeddings(openai_api_key=openai_api_key),

    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma,

    # This is the number of examples to produce.
    k=2
)

similar_prompt = FewShotPromptTemplate(
    # The object that will help select examples
    example_selector=example_selector,

    # Your prompt
    example_prompt=example_prompt,

    # Customizations that will be added to the top and bottom of your prompt
    prefix="Give the location an item is usually found in",
    suffix="Input: {noun}\\nOutput:",

    # What inputs your prompt will receive
    input_variables=["noun"],
)

# Select a noun!
my_noun = "plant"
print(similar_prompt.format(noun=my_noun))

# Output will show the most semantically similar examples
llm(similar_prompt.format(noun=my_noun))
```

### 关键特性

1. **语义相似度选择**：使用 `SemanticSimilarityExampleSelector` 根据语义相似度选择示例
2. **向量存储集成**：使用 Chroma 向量存储来存储和检索示例
3. **动态示例数量**：通过 `k` 参数控制返回的示例数量
4. **灵活的模板系统**：使用 `FewShotPromptTemplate` 组合示例选择器和提示模板

### 实际应用

- 根据用户输入动态选择最相关的示例
- 提高 Few-shot learning 的效果
- 减少不相关示例的干扰
- 优化 Prompt 长度和质量

---

**完整 Notebook 链接**：https://github.com/gkamradt/langchain-tutorials/blob/main/LangChain%20Cookbook%20Part%201%20-%20Fundamentals.ipynb

**注**：本文档为精简版，完整内容包含更多 LangChain 基础概念（Schema、Models、Prompts、Chains、Memory、Indexes、Agents 等）。
