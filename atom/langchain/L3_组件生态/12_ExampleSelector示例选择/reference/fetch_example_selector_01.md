---
type: fetched_content
source: https://medium.com/donato-story/exploring-few-shot-prompts-with-langchain-852f27ea4e1d
title: Exploring Few-Shot Prompts with LangChain
fetched_at: 2026-02-25
status: success
author: Donato_TH
knowledge_point: ExampleSelector示例选择
fetch_tool: Grok-mcp web_fetch
---

# Exploring Few-Shot Prompts with LangChain

**Data Mastery Series — Episode 28: LangChain Website (Part 3)**

Hey everyone! Welcome back to the Data Mastery Series! We're diving deeper into **LangChain** and exploring more about prompts.

If you're new here, feel free to catch up on our previous episodes:

- **Part 1:** [LangChain Model I/O Basics](https://medium.com/@designbynattapong/langchain-model-i-o-basics-a5e9d352f561)
- **Part 2:** [Unpacking Prompt Templates with LangChain](https://medium.com/donato-story/unpacking-prompt-templates-with-langchain-a16eb840b7bb)

> **Note**: As we dive into LangChain, I'll be sharing insights and key notes from my own study of the LangChain documentation. Let's jump in and explore some fascinating features in today's episode!
> Source: https://python.langchain.com/v0.1/docs/modules/model_io/prompts/

Today, we're continuing our discussion on [Prompt](https://python.langchain.com/v0.1/docs/modules/model_io/prompts/quick_start/), focusing on **Few-Shot Prompts**, **Partial Prompts**, and **Composition**. "Few-shot learning" allows us to give our language models examples to learn from, improving their performance.

## 1. Few-Shot Examples for Chat Models: Learning from Examples

There are two main approaches to using examples in few-shot prompting:

### A. Fixed Examples

With **Fixed Examples**, you give the model a set of examples that it will always follow. This approach is simple and effective when you want consistent responses.

```python
# Example - Fixed Few-shot Examples
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

chat = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
chain = final_prompt | chat
response = chain.invoke({"input": "What's the square of a triangle?"})

print(response)

# Output
'''
content='A triangle does not have a square. The square of a number is the result of multiplying the number by itself.' additional_kwargs={} response_metadata={'token_usage': {'completion_tokens': 23, 'prompt_tokens': 52, 'total_tokens': 75, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-8e084845-a7fe-4340-8225-efa68fbc4f4f-0'
'''
```

In this example, we've given the model some basic math problems, like `"2+2" → "4"` and `"2+3" → "5"`. When the user asks a question, the model combines these examples with its knowledge, generating a relevant response.

### B. Dynamic Examples

Sometimes, we want the model to pick examples that match the input's context. **Dynamic Examples** allow the AI to find the most relevant examples for the current query, adjusting its response accordingly.

```python
# Example - Dynamic Few-shot Examples
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
    {"input": "2+4", "output": "6"},
    {"input": "What did the cow say to the moon?", "output": "nothing at all"},
    {"input": "Write me a poem about the moon", "output": "One for the moon, and one for me, who are we to talk about the moon?"},
]

to_vectorize = [" ".join(example.values()) for example in examples]
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    input_variables=["input"],
    example_selector=example_selector,
    example_prompt=ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    ),
)

print(few_shot_prompt.format(input="What's 3+3?"))

# Output
'''
Human: 2+3
AI: 5
Human: 2+2
AI: 4
'''
```

In this example, we use a **Semantic Similarity Selector** to pull in the most relevant examples based on the query. When the user asks "What's 3+3?", the prompt selects math-related examples…

```python
# Example - Dynamic Few-shot Examples
print(few_shot_prompt.format(input="horse on the moon"))

# Output
'''
Human: What did the cow say to the moon?
AI: nothing at all
Human: Write me a poem about the moon
AI: One for the moon, and one for me, who are we to talk about the moon?
'''
```

This time, when the input is about "the moon," the model picks examples related to the moon, showing how it can adapt to different contexts…

## 2. Few-shot prompt templates

There are two main approaches: using an example set (**Fixed Set**) or an example selector.

### A. Using an example (Fixed Set)

```python
# Example - Using an example
examples = [
    {
        "question": "Who lived longer, Muhammad Ali or Alan Turing?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali
""",
    },
    # ... (其他3个例子，内容较长，已在原文中完整展示)
    # 此处省略重复内容以保持简洁，但实际包含全部4个例子
]

example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Question: {question}\n{answer}"
)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

print(prompt.format(input="Who was the father of Mary Ball Washington?"))

# Output（会输出全部例子 + 最后的问题）
```

### B. Using an example selector

```python
# Example - Using an example selector
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(api_key=OPENAI_API_KEY),
    Chroma,
    k=1,
)

question = "Who was the father of Mary Ball Washington?"
selected_examples = example_selector.select_examples({"question": question})

for example in selected_examples:
    print(f"question: {example['question']}")
    print("\n")
    print(f"answer: {example['answer']}")
```

## 3. Partial Prompt Templates

**Partial Prompts** allow you to pre-fill parts of a prompt and leave placeholders to fill in later…

### A. Partial with strings

```python
# Example - Partial with strings
prompt = PromptTemplate.from_template("{foo}{bar}")
partial_prompt = prompt.partial(foo="foo")

print(partial_prompt.format(bar="baz"))
# foobaz

# 或者定义时直接指定
prompt = PromptTemplate(
    template="{foo}{bar}",
    input_variables=["bar"],
    partial_variables={"foo": "foo"}
)
print(prompt.format(bar="baz"))
# foobaz
```

### B. Partial with Functions

```python
# Example - Partial with Functions

def _get_datetime():
    now = datetime.now()
    return now.strftime("%m/%d/%Y, %H:%M:%S")

prompt = PromptTemplate(
    template="Tell me the current time: {time}. And say {content}.",
    input_variables=["content"],
    partial_variables={"time": _get_datetime},
)
```

---

**作者：** Donato_TH
**系列：** Data Mastery Series
**参考官方文档：** https://python.langchain.com/v0.1/docs/modules/model_io/prompts/
