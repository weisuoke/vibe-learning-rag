# 实战代码 - 场景6：RAG系统中的Few-shot优化

## 场景描述

在 RAG 系统中，我们不仅需要检索相关文档，还需要通过 Few-shot 示例来优化 LLM 的输出质量。本场景展示如何将 ExampleSelector 集成到 RAG 系统中，动态选择最相关的示例来改善生成效果。

### 业务背景

构建一个技术文档问答系统，需要：
1. **检索相关文档**：从知识库中检索相关内容
2. **选择相关示例**：根据问题选择相似的 QA 示例
3. **优化 Prompt**：将检索结果和示例注入 Prompt
4. **生成高质量答案**：LLM 基于上下文和示例生成答案

### 技术挑战

1. **双重检索**：既要检索文档，又要选择示例
2. **Prompt 长度控制**：避免超过 context window
3. **示例相关性**：确保示例与问题高度相关
4. **性能优化**：减少延迟和成本

[来源: reference/context7_langchain_01.md | LangChain 官方文档]
[来源: reference/fetch_example_selector_01.md | Medium 教程]

---

## 核心技术点

### 1. RAG + Few-shot 架构

```
用户问题
    ↓
    ├─→ 文档检索（RAG）→ 相关文档
    └─→ 示例选择（ExampleSelector）→ 相关示例
    ↓
Prompt 构建（文档 + 示例）
    ↓
LLM 生成答案
```

### 2. 关键组件

- **VectorStore**：存储文档和示例的 embeddings
- **Retriever**：检索相关文档
- **ExampleSelector**：选择相关示例
- **FewShotPromptTemplate**：格式化示例
- **LLM**：生成最终答案

[来源: reference/context7_langchain_01.md]

---

## 完整可运行代码

```python
"""
场景6：RAG系统中的Few-shot优化
演示：将 ExampleSelector 集成到 RAG 系统中优化输出质量
"""

import os
from dotenv import load_dotenv
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.chains import RetrievalQA

# 加载环境变量
load_dotenv()

# ===== 1. 准备知识库文档 =====
print("=== 1. 准备知识库文档 ===\n")

# 技术文档知识库
documents = [
    Document(
        page_content="LangChain is a framework for developing applications powered by large language models. It provides tools for prompt management, chains, agents, and memory.",
        metadata={"source": "langchain_intro.md", "category": "framework"}
    ),
    Document(
        page_content="RAG (Retrieval-Augmented Generation) combines information retrieval with text generation. It retrieves relevant documents and uses them as context for the LLM.",
        metadata={"source": "rag_guide.md", "category": "technique"}
    ),
    Document(
        page_content="Vector databases store high-dimensional vectors (embeddings) and enable efficient similarity search. Examples include Chroma, FAISS, Pinecone, and Milvus.",
        metadata={"source": "vector_db.md", "category": "database"}
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors that capture semantic meaning. OpenAI provides text-embedding-3-small and text-embedding-3-large models.",
        metadata={"source": "embeddings.md", "category": "technique"}
    ),
    Document(
        page_content="Few-shot learning provides the LLM with examples to learn from. This helps the model understand the expected format and style of responses.",
        metadata={"source": "few_shot.md", "category": "technique"}
    ),
    Document(
        page_content="Prompt engineering is the practice of designing prompts to get better responses from LLMs. Techniques include few-shot learning, chain-of-thought, and role prompting.",
        metadata={"source": "prompt_eng.md", "category": "technique"}
    ),
    Document(
        page_content="LangChain chains are sequences of calls to LLMs or other utilities. They allow you to combine multiple components to create complex workflows.",
        metadata={"source": "chains.md", "category": "framework"}
    ),
    Document(
        page_content="Memory in LangChain allows agents to remember past interactions. Types include ConversationBufferMemory, ConversationSummaryMemory, and VectorStoreMemory.",
        metadata={"source": "memory.md", "category": "framework"}
    ),
]

print(f"知识库包含 {len(documents)} 个文档\n")

# ===== 2. 准备 QA 示例库 =====
print("=== 2. 准备 QA 示例库 ===\n")

qa_examples = [
    {
        "question": "What is LangChain?",
        "answer": "LangChain is a framework for developing applications powered by large language models (LLMs). It provides modular components for prompt management, chains, agents, and memory, making it easier to build complex AI applications."
    },
    {
        "question": "How does RAG work?",
        "answer": "RAG (Retrieval-Augmented Generation) works by first retrieving relevant documents from a knowledge base, then using those documents as context for the LLM to generate more accurate and informed answers. This reduces hallucinations and allows the model to access up-to-date information."
    },
    {
        "question": "What are embeddings?",
        "answer": "Embeddings are numerical vector representations of text that capture semantic meaning. They allow us to find similar content using mathematical operations like cosine similarity. OpenAI provides models like text-embedding-3-small for generating embeddings."
    },
    {
        "question": "Why use vector databases?",
        "answer": "Vector databases are specialized for storing and efficiently searching high-dimensional vectors (embeddings). They enable fast similarity search, which is essential for RAG systems. Popular options include Chroma, FAISS, Pinecone, and Milvus."
    },
    {
        "question": "What is few-shot learning?",
        "answer": "Few-shot learning is a technique where you provide the LLM with a few examples of the task you want it to perform. This helps the model understand the expected format, style, and quality of responses without requiring fine-tuning."
    },
    {
        "question": "How to optimize prompts?",
        "answer": "Prompt optimization involves techniques like few-shot learning (providing examples), chain-of-thought (showing reasoning steps), role prompting (defining the AI's role), and iterative refinement based on output quality. The goal is to get more accurate and consistent responses."
    },
]

print(f"示例库包含 {len(qa_examples)} 个 QA 对\n")

# ===== 3. 创建文档检索器（RAG） =====
print("=== 3. 创建文档检索器（RAG） ===\n")

# 初始化 Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 创建文档向量存储
doc_vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="tech_docs"
)

# 创建检索器
retriever = doc_vectorstore.as_retriever(
    search_kwargs={"k": 3}  # 检索 3 个最相关的文档
)

print("✓ 文档检索器创建成功")
print(f"  - 文档数量: {len(documents)}")
print(f"  - 检索数量: k=3\n")

# ===== 4. 创建示例选择器（Few-shot） =====
print("=== 4. 创建示例选择器（Few-shot） ===\n")

# 创建示例选择器
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=qa_examples,
    embeddings=embeddings,
    vectorstore_cls=Chroma,
    k=2  # 选择 2 个最相关的示例
)

print("✓ 示例选择器创建成功")
print(f"  - 示例数量: {len(qa_examples)}")
print(f"  - 选择数量: k=2\n")

# ===== 5. 定义示例模板 =====
print("=== 5. 定义示例模板 ===\n")

example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Q: {question}\nA: {answer}"
)

# 创建 Few-shot Prompt
few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="You are a helpful technical assistant. Answer questions based on the provided context and examples.\n\nExamples:",
    suffix="\nContext: {context}\n\nQ: {question}\nA:",
    input_variables=["context", "question"]
)

print("✓ Few-shot Prompt 模板创建成功\n")

# ===== 6. 创建 RAG + Few-shot 链 =====
print("=== 6. 创建 RAG + Few-shot 链 ===\n")

# 初始化 LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

# 手动构建 RAG + Few-shot 链
def rag_with_fewshot(question: str) -> dict:
    """RAG + Few-shot 问答"""
    # 1. 检索相关文档
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 2. 选择相关示例
    selected_examples = example_selector.select_examples({"question": question})
    
    # 3. 格式化 Prompt
    prompt = few_shot_prompt.format(context=context, question=question)
    
    # 4. 调用 LLM
    response = llm.invoke(prompt)
    
    return {
        "question": question,
        "answer": response.content,
        "retrieved_docs": docs,
        "selected_examples": selected_examples,
        "prompt": prompt
    }

print("✓ RAG + Few-shot 链创建成功\n")

# ===== 7. 测试不同类型的问题 =====
print("=== 7. 测试不同类型的问题 ===\n")

test_questions = [
    "What is the difference between RAG and fine-tuning?",
    "How do I choose a vector database?",
    "What are the benefits of using LangChain?",
]

for i, question in enumerate(test_questions, 1):
    print(f"--- 测试 {i}: {question} ---\n")
    
    result = rag_with_fewshot(question)
    
    print(f"检索到的文档（{len(result['retrieved_docs'])} 个）：")
    for j, doc in enumerate(result['retrieved_docs'], 1):
        print(f"  {j}. {doc.page_content[:80]}...")
    print()
    
    print(f"选中的示例（{len(result['selected_examples'])} 个）：")
    for j, ex in enumerate(result['selected_examples'], 1):
        print(f"  {j}. {ex['question']}")
    print()
    
    print("LLM 回答：")
    print(result['answer'])
    print("\n" + "=" * 80 + "\n")

# ===== 8. 性能对比：纯 RAG vs RAG + Few-shot =====
print("=== 8. 性能对比：纯 RAG vs RAG + Few-shot ===\n")

test_question = "How can I improve my RAG system's accuracy?"

# 纯 RAG（无示例）
print("--- 纯 RAG（无示例）---\n")

pure_rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Answer the question based on the context.\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
)

docs = retriever.invoke(test_question)
context = "\n\n".join([doc.page_content for doc in docs])
pure_rag_response = llm.invoke(pure_rag_prompt.format(context=context, question=test_question))

print("纯 RAG 回答：")
print(pure_rag_response.content)
print()

# RAG + Few-shot
print("--- RAG + Few-shot ---\n")

fewshot_result = rag_with_fewshot(test_question)

print("RAG + Few-shot 回答：")
print(fewshot_result['answer'])
print()

print("对比分析：")
print("  纯 RAG：直接基于检索文档回答，可能缺乏结构和深度")
print("  RAG + Few-shot：参考示例的格式和风格，回答更结构化和详细")
print()

# ===== 9. Prompt 长度分析 =====
print("=== 9. Prompt 长度分析 ===\n")

pure_rag_prompt_text = pure_rag_prompt.format(context=context, question=test_question)
fewshot_prompt_text = fewshot_result['prompt']

print(f"纯 RAG Prompt 长度: {len(pure_rag_prompt_text)} 字符")
print(f"RAG + Few-shot Prompt 长度: {len(fewshot_prompt_text)} 字符")
print(f"增加: {len(fewshot_prompt_text) - len(pure_rag_prompt_text)} 字符 ({(len(fewshot_prompt_text) / len(pure_rag_prompt_text) - 1) * 100:.1f}%)")
print()

# ===== 10. 实际应用建议 =====
print("=== 10. 实际应用建议 ===\n")

print("✓ 最佳实践：")
print("  1. 示例数量：k=2-3 个示例，平衡效果和成本")
print("  2. 文档数量：k=3-5 个文档，提供足够上下文")
print("  3. Prompt 优化：监控总长度，避免超过 context window")
print("  4. 示例质量：确保示例高质量、多样化")
print("  5. 缓存策略：缓存文档和示例的 embeddings")
print()

print("✓ 性能优化：")
print("  1. 并行检索：同时检索文档和选择示例")
print("  2. 批量处理：批量生成 embeddings")
print("  3. 异步操作：使用 async/await 提高并发")
print("  4. 结果缓存：缓存常见问题的答案")
print()

print("✓ 适用场景：")
print("  - 技术文档问答系统")
print("  - 客服机器人")
print("  - 教育辅导系统")
print("  - 代码助手")
print()

print("=" * 80)
print("实战完成！")
print("=" * 80)
```

---

## 代码解析

### 1. 双重检索架构

```python
# 文档检索（RAG）
docs = retriever.invoke(question)
context = "\n\n".join([doc.page_content for doc in docs])

# 示例选择（Few-shot）
selected_examples = example_selector.select_examples({"question": question})
```

**关键点：**
- 文档检索提供领域知识
- 示例选择提供格式和风格参考
- 两者结合提升输出质量

### 2. Prompt 构建

```python
few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="You are a helpful technical assistant...",
    suffix="\nContext: {context}\n\nQ: {question}\nA:",
    input_variables=["context", "question"]
)
```

**Prompt 结构：**
```
prefix（系统提示）
+ 动态选择的示例1
+ 动态选择的示例2
+ 检索到的文档上下文
+ 用户问题
```

### 3. 性能对比

| 指标 | 纯 RAG | RAG + Few-shot | 差异 |
|------|--------|----------------|------|
| Prompt 长度 | ~500 字符 | ~800 字符 | +60% |
| 回答质量 | 基础 | 结构化 | 更好 |
| 响应时间 | ~1.2s | ~1.5s | +25% |
| Token 成本 | 基准 | +30% | 可接受 |

[来源: reference/search_example_selector_02.md]

---

## 运行结果示例

### 测试1：RAG vs Fine-tuning

```
检索到的文档（3 个）：
  1. RAG (Retrieval-Augmented Generation) combines information retrieval with text...
  2. Few-shot learning provides the LLM with examples to learn from...
  3. LangChain is a framework for developing applications powered by large language...

选中的示例（2 个）：
  1. How does RAG work?
  2. What is few-shot learning?

LLM 回答：
RAG and fine-tuning are two different approaches to improving LLM performance:

**RAG (Retrieval-Augmented Generation):**
- Retrieves relevant documents at query time
- No model retraining required
- Can access up-to-date information
- Lower cost and faster to implement
- Best for: Knowledge-intensive tasks, frequently updated information

**Fine-tuning:**
- Retrains the model on domain-specific data
- Requires significant compute resources
- Model knowledge is frozen at training time
- Higher cost but potentially better performance
- Best for: Specialized tasks, consistent style/tone

In practice, RAG is often preferred for its flexibility and lower cost, while fine-tuning is used when you need the model to deeply internalize domain knowledge.
```

---

## 实际应用建议

### 1. 示例库管理

```python
# 从文件加载示例
import json

with open("qa_examples.json", "r") as f:
    qa_examples = json.load(f)

# 动态更新示例库
def add_qa_example(question: str, answer: str):
    new_example = {"question": question, "answer": answer}
    example_selector.add_example(new_example)
    
    # 持久化
    qa_examples.append(new_example)
    with open("qa_examples.json", "w") as f:
        json.dump(qa_examples, f, indent=2)
```

### 2. 并行检索优化

```python
import asyncio

async def parallel_rag_fewshot(question: str):
    """并行检索文档和选择示例"""
    # 并行执行
    docs_task = asyncio.create_task(retriever.ainvoke(question))
    examples_task = asyncio.create_task(
        example_selector.aselect_examples({"question": question})
    )
    
    # 等待结果
    docs, examples = await asyncio.gather(docs_task, examples_task)
    
    # 构建 Prompt 和生成答案
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = few_shot_prompt.format(context=context, question=question)
    response = await llm.ainvoke(prompt)
    
    return response.content
```

### 3. Prompt 长度控制

```python
def adaptive_rag_fewshot(question: str, max_length: int = 4000):
    """自适应控制 Prompt 长度"""
    # 检索文档
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 选择示例
    k = 2
    selected_examples = example_selector.select_examples({"question": question})
    
    # 估算长度
    prompt = few_shot_prompt.format(context=context, question=question)
    
    # 如果超长，减少示例数量
    while len(prompt) > max_length and k > 0:
        k -= 1
        example_selector.k = k
        selected_examples = example_selector.select_examples({"question": question})
        prompt = few_shot_prompt.format(context=context, question=question)
    
    # 生成答案
    response = llm.invoke(prompt)
    return response.content
```

### 4. 评估和监控

```python
from langsmith import traceable

@traceable(run_type="chain", name="RAG_FewShot_QA")
def traced_rag_fewshot(question: str):
    """带追踪的 RAG + Few-shot"""
    result = rag_with_fewshot(question)
    
    # 记录关键指标
    return {
        "answer": result['answer'],
        "num_docs": len(result['retrieved_docs']),
        "num_examples": len(result['selected_examples']),
        "prompt_length": len(result['prompt'])
    }
```

---

## 常见问题

### Q1: 如何平衡文档和示例的数量？

**建议：**
- 文档：k=3-5（提供足够上下文）
- 示例：k=2-3（提供格式参考）
- 总 Prompt 长度：< 4000 tokens

### Q2: 示例和文档的相似度计算是否会冲突？

**不会冲突：**
- 文档和示例存储在不同的 VectorStore
- 使用相同的 Embeddings 模型确保一致性
- 分别进行相似度搜索

### Q3: 如何评估 Few-shot 的效果？

**评估方法：**
1. A/B 测试：对比纯 RAG vs RAG + Few-shot
2. 人工评估：评估回答质量、结构、风格
3. 自动评估：使用 LLM 评估回答质量
4. LangSmith 追踪：监控 Prompt 长度和响应时间

---

## 总结

**核心优势：**
1. 提高回答质量：示例提供格式和风格参考
2. 结构化输出：回答更有条理
3. 减少幻觉：文档提供事实依据
4. 灵活可控：动态选择示例和文档

**最佳实践：**
- 文档 k=3-5，示例 k=2-3
- 监控 Prompt 长度，避免超限
- 缓存 embeddings 提高性能
- 使用 LangSmith 追踪和优化

**适用场景：**
- 技术文档问答
- 客服机器人
- 教育辅导
- 代码助手

[来源: 综合多个参考资料]

---

**参考资料：**
- [来源: reference/context7_langchain_01.md | LangChain 官方文档]
- [来源: reference/fetch_example_selector_01.md | Medium 教程]
- [来源: reference/search_example_selector_02.md | 2025-2026 最佳实践]

---

**版本：** v1.0
**最后更新：** 2026-02-26
**维护者：** Claude Code
