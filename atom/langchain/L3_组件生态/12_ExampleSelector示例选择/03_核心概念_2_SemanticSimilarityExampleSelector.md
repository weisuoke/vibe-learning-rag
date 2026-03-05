# 核心概念 2：SemanticSimilarityExampleSelector

## 概念定义

**SemanticSimilarityExampleSelector 是基于语义相似度动态选择 Few-shot 示例的选择器，通过计算输入与示例库之间的向量相似度，自动选择最相关的示例注入到 Prompt 中。**

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/semantic_similarity.py]

这是 LangChain 中最常用的示例选择器，特别适合需要根据用户输入动态调整示例的场景。

---

## 为什么需要语义相似度选择？

### 问题场景

在 Few-shot learning 中，我们面临一个核心挑战：

**如何从大量示例中选择最相关的几个？**

```python
# 假设你有 100 个示例
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "What is Python?", "output": "A programming language"},
    {"input": "How to cook pasta?", "output": "Boil water, add pasta..."},
    # ... 97 more examples
]

# 用户问题
user_query = "What is JavaScript?"

# 问题：应该选择哪些示例？
# ❌ 选择前 3 个？可能不相关
# ❌ 随机选择？效果不稳定
# ✅ 选择语义最相似的？最佳方案！
```

### 解决方案

SemanticSimilarityExampleSelector 通过以下步骤解决这个问题：

1. **将所有示例转换为向量**（embeddings）
2. **将用户输入转换为向量**
3. **计算余弦相似度**
4. **返回 top-k 最相似的示例**

[来源: reference/context7_langchain_01.md | LangChain 官方文档]

---

## 核心架构

### 类定义

```python
class SemanticSimilarityExampleSelector(_VectorStoreExampleSelector):
    """基于语义相似度选择示例

    Attributes:
        vectorstore: VectorStore - 向量存储后端
        k: int - 选择的示例数量（默认 4）
        example_keys: list[str] | None - 过滤示例的键
        input_keys: list[str] | None - 过滤输入的键
        vectorstore_kwargs: dict[str, Any] | None - 向量存储额外参数
    """
```

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/semantic_similarity.py]

### 关键特性

1. **VectorStore 集成**：支持多种向量数据库（Chroma、FAISS、Pinecone 等）
2. **灵活的键过滤**：可以只基于部分字段进行检索
3. **异步支持**：提供 `aselect_examples()` 异步方法
4. **类方法快速创建**：`from_examples()` 简化初始化流程

---

## VectorStore 集成

### 支持的向量存储

SemanticSimilarityExampleSelector 依赖 VectorStore 抽象，支持多种后端：

| 向量存储 | 特点 | 适用场景 |
|---------|------|---------|
| **Chroma** | 易用、持久化 | 开发和小规模生产 |
| **FAISS** | 内存中、速度快 | 高性能场景 |
| **Pinecone** | 云端、可扩展 | 大规模生产环境 |
| **Milvus** | 分布式、企业级 | 大规模企业应用 |

[来源: reference/search_example_selector_02.md]

### 向量存储工作流程

```
示例文本 → Embeddings → VectorStore
                              ↓
用户输入 → Embeddings → 相似度搜索 → Top-K 示例
```

---

## from_examples() 类方法

### 方法签名

```python
@classmethod
def from_examples(
    cls,
    examples: list[dict],              # 示例列表
    embeddings: Embeddings,            # Embedding 模型
    vectorstore_cls: type[VectorStore], # 向量存储类
    k: int = 4,                        # 返回示例数量
    input_keys: list[str] | None = None, # 输入键过滤
    **vectorstore_cls_kwargs: Any,     # 向量存储额外参数
) -> SemanticSimilarityExampleSelector:
    """从示例列表快速创建选择器"""
```

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/semantic_similarity.py]

### 使用示例

```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 定义示例
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
    {"input": "What is Python?", "output": "A programming language"},
    {"input": "What is JavaScript?", "output": "A scripting language"},
]

# 创建选择器
selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    vectorstore_cls=Chroma,
    k=2  # 返回 2 个最相似的示例
)

# 选择示例
selected = selector.select_examples({"input": "What is Java?"})
print(selected)
# 输出：
# [
#     {"input": "What is JavaScript?", "output": "A scripting language"},
#     {"input": "What is Python?", "output": "A programming language"}
# ]
```

[来源: reference/fetch_example_selector_01.md | https://medium.com/donato-story/exploring-few-shot-prompts-with-langchain-852f27ea4e1d]

### 内部实现

`from_examples()` 方法执行以下步骤：

```python
# 1. 将示例转换为文本
string_examples = [cls._example_to_text(eg, input_keys) for eg in examples]

# 2. 创建向量存储
vectorstore = vectorstore_cls.from_texts(
    string_examples,
    embeddings,
    metadatas=examples,  # 将原始示例存储为元数据
    **vectorstore_cls_kwargs
)

# 3. 返回选择器实例
return cls(vectorstore=vectorstore, k=k, input_keys=input_keys)
```

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/semantic_similarity.py]

---

## select_examples() 方法

### 核心逻辑

```python
def select_examples(self, input_variables: dict[str, str]) -> list[dict]:
    """根据输入选择示例

    Args:
        input_variables: 输入变量字典

    Returns:
        最相似的 k 个示例
    """
    # 1. 将输入转换为文本
    query_text = self._example_to_text(input_variables, self.input_keys)

    # 2. 使用向量存储进行相似度搜索
    example_docs = self.vectorstore.similarity_search(
        query_text,
        k=self.k,
        **self.vectorstore_kwargs or {}
    )

    # 3. 从文档中提取示例
    return self._documents_to_examples(example_docs)
```

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/semantic_similarity.py]

### 文本转换逻辑

```python
@staticmethod
def _example_to_text(
    example: dict[str, str],
    input_keys: list[str] | None
) -> str:
    """将示例转换为文本用于向量化

    如果指定了 input_keys，只使用这些键的值
    否则使用所有键的值
    """
    if input_keys:
        return " ".join(sorted_values({key: example[key] for key in input_keys}))
    return " ".join(sorted_values(example))
```

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/semantic_similarity.py]

---

## k 参数调优

### k 值的影响

| k 值 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **k=1-2** | Token 成本低、Prompt 简洁 | 可能缺少足够上下文 | 简单任务、Token 限制严格 |
| **k=3-5** | 平衡相关性和成本 | 适中 | **大多数场景（推荐）** |
| **k=6-10** | 提供丰富上下文 | Token 成本高、可能引入噪音 | 复杂任务、示例质量高 |

[来源: reference/search_example_selector_02.md]

### 2025-2026 最佳实践

根据社区实践和企业应用经验：

**推荐配置：k=3-5**

```python
# ✅ 推荐配置
selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    vectorstore_cls=Chroma,
    k=3  # 3-5 个示例效果最好
)
```

[来源: reference/fetch_example_selector_05.md | https://www.swarnendu.de/blog/langchain-best-practices]

### 动态 k 值调整

在某些场景中，可以根据输入长度动态调整 k 值：

```python
def dynamic_k_selector(input_text: str, base_k: int = 3) -> int:
    """根据输入长度动态调整 k 值"""
    input_length = len(input_text.split())

    if input_length < 10:
        return base_k + 2  # 短输入需要更多示例
    elif input_length > 50:
        return base_k - 1  # 长输入减少示例避免超限
    else:
        return base_k

# 使用
k = dynamic_k_selector(user_input)
selector.k = k
selected = selector.select_examples({"input": user_input})
```

---

## 与 FewShotPromptTemplate 集成

### 完整示例

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

# 1. 定义示例
examples = [
    {"query": "How are you?", "answer": "I can't complain but sometimes I still do."},
    {"query": "What time is it?", "answer": "It's time to get a watch."},
    {"query": "What's the meaning of life?", "answer": "42, of course!"},
]

# 2. 创建示例选择器
selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    vectorstore_cls=Chroma,
    k=2
)

# 3. 定义示例模板
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template="User: {query}\nAI: {answer}"
)

# 4. 创建 FewShotPromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    example_selector=selector,  # 使用选择器而非固定示例
    example_prompt=example_prompt,
    prefix="You are a witty AI assistant:",
    suffix="User: {query}\nAI:",
    input_variables=["query"]
)

# 5. 与 LLM 集成
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
chain = few_shot_prompt | llm

# 6. 执行
response = chain.invoke({"query": "What's the weather like?"})
print(response.content)
```

[来源: reference/fetch_example_selector_01.md | https://medium.com/donato-story/exploring-few-shot-prompts-with-langchain-852f27ea4e1d]

---

## 实际应用场景

### 场景 1：问答系统

```python
# 问答示例库
qa_examples = [
    {"question": "What is LangChain?", "answer": "LangChain is a framework for building LLM applications."},
    {"question": "What is RAG?", "answer": "RAG stands for Retrieval-Augmented Generation."},
    {"question": "What is a vector database?", "answer": "A vector database stores and retrieves embeddings."},
    # ... 更多示例
]

# 创建选择器
qa_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=qa_examples,
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    vectorstore_cls=Chroma,
    k=3
)

# 用户提问
user_question = "What is a retrieval system?"

# 选择相关示例
selected_examples = qa_selector.select_examples({"question": user_question})
# 返回与 "retrieval" 相关的示例（如 RAG、vector database）
```

[来源: reference/context7_langchain_01.md | LangChain 官方文档]

### 场景 2：分类任务

```python
# GitHub issue 分类示例
classification_examples = [
    {"topic": "The login button doesn't work", "output": "bug"},
    {"topic": "Add dark mode support", "output": "new_feature"},
    {"topic": "Improve API documentation", "output": "documentation"},
    {"topic": "Integrate with Slack", "output": "integration"},
    # ... 更多示例
]

# 创建选择器
classifier_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=classification_examples,
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    vectorstore_cls=Chroma,
    k=5
)

# 新 issue
new_issue = "The app crashes when I click submit"

# 选择相似示例
selected = classifier_selector.select_examples({"topic": new_issue})
# 返回与 "bug" 相关的示例
```

[来源: reference/context7_langchain_01.md | LangChain 官方文档]

### 场景 3：代码生成

```python
# 代码示例库
code_examples = [
    {
        "description": "Read a file",
        "code": "with open('file.txt', 'r') as f:\n    content = f.read()"
    },
    {
        "description": "Write to a file",
        "code": "with open('file.txt', 'w') as f:\n    f.write('Hello')"
    },
    {
        "description": "Make HTTP request",
        "code": "import requests\nresponse = requests.get('https://api.example.com')"
    },
    # ... 更多示例
]

# 创建选择器
code_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=code_examples,
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    vectorstore_cls=Chroma,
    k=2
)

# 用户需求
user_request = "How to download a file from URL?"

# 选择相关代码示例
selected = code_selector.select_examples({"description": user_request})
# 返回与 HTTP 请求相关的示例
```

---

## 高级特性

### 1. input_keys 过滤

只基于部分输入字段进行检索：

```python
examples = [
    {"user": "Alice", "query": "What is Python?", "context": "programming"},
    {"user": "Bob", "query": "What is Java?", "context": "programming"},
]

# 只基于 query 字段进行相似度匹配，忽略 user 和 context
selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=OpenAIEmbeddings(),
    vectorstore_cls=Chroma,
    k=2,
    input_keys=["query"]  # 只使用 query 字段
)
```

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/semantic_similarity.py]

### 2. example_keys 过滤

只返回示例的部分字段：

```python
selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,
    example_keys=["query", "answer"]  # 只返回这两个字段
)

# 即使示例包含更多字段，也只返回指定的字段
```

### 3. 异步支持

```python
# 异步选择示例
selected = await selector.aselect_examples({"input": user_query})
```

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/semantic_similarity.py]

---

## 性能优化

### 1. 缓存 Embeddings

```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

# 启用缓存避免重复计算 embeddings
set_llm_cache(InMemoryCache())
```

[来源: reference/search_example_selector_02.md]

### 2. 选择合适的 Embedding 模型

| 模型 | 维度 | 速度 | 成本 | 适用场景 |
|------|------|------|------|---------|
| **text-embedding-3-small** | 1536 | 快 | 低 | **推荐（大多数场景）** |
| text-embedding-3-large | 3072 | 中 | 中 | 高精度需求 |
| text-embedding-ada-002 | 1536 | 快 | 低 | 旧版本兼容 |

[来源: reference/search_example_selector_02.md]

### 3. 向量存储选择

```python
# 开发环境：使用 Chroma（易用）
from langchain_chroma import Chroma
vectorstore_cls = Chroma

# 生产环境：使用 FAISS（高性能）
from langchain_community.vectorstores import FAISS
vectorstore_cls = FAISS

# 大规模场景：使用 Pinecone（云端）
from langchain_pinecone import Pinecone
vectorstore_cls = Pinecone
```

---

## 常见误区

### 误区 1：k 值越大越好 ❌

**错误观点：** "k=10 比 k=3 效果更好"

**为什么错？**
- 过多示例会引入噪音
- 增加 Token 成本
- 可能超过 context window

**正确理解：**
k=3-5 通常效果最好，平衡了相关性和成本。

[来源: reference/search_example_selector_02.md]

### 误区 2：不考虑 Token 限制 ❌

**错误观点：** "只关注相关性，不考虑长度"

**为什么错？**
- 可能超过模型的 context window
- 导致 Prompt 被截断

**正确理解：**
结合 LengthBasedExampleSelector 或使用混合策略。

[来源: reference/fetch_example_selector_10.md | https://github.com/whitesmith/langchain-semantic-length-example-selector]

### 误区 3：不缓存 Embeddings ❌

**错误观点：** "每次都重新计算 embeddings"

**为什么错？**
- 浪费时间和成本
- 降低系统性能

**正确理解：**
使用缓存机制或持久化向量存储。

[来源: reference/search_example_selector_02.md]

---

## 与其他选择器对比

| 选择器 | 选择依据 | 优点 | 缺点 | 适用场景 |
|--------|---------|------|------|---------|
| **SemanticSimilarity** | 语义相似度 | 相关性高、智能 | 需要 embeddings、成本较高 | **大多数场景（推荐）** |
| LengthBased | 长度限制 | 简单、Token 可控 | 不考虑相关性 | Token 限制严格 |
| MMR | 相关性 + 多样性 | 平衡多样性 | 计算复杂 | 需要多样化示例 |
| NGramOverlap | N-gram 重叠 | 轻量级、无需 embeddings | 精度较低 | 简单文本匹配 |

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/]

---

## 总结

SemanticSimilarityExampleSelector 是 LangChain 中最强大和最常用的示例选择器：

**核心优势：**
1. 基于语义相似度，选择最相关的示例
2. 支持多种向量存储后端
3. 提供 `from_examples()` 快速创建
4. 支持异步操作
5. 灵活的键过滤机制

**最佳实践：**
- 使用 k=3-5 个示例
- 选择 text-embedding-3-small 模型
- 缓存 embeddings 提高性能
- 结合长度限制避免超限

**适用场景：**
- 问答系统
- 分类任务
- 代码生成
- 任何需要动态选择示例的场景

[来源: 综合多个参考资料]

---

**参考资料：**
- [来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/semantic_similarity.py]
- [来源: reference/context7_langchain_01.md | LangChain 官方文档]
- [来源: reference/search_example_selector_02.md]
- [来源: reference/fetch_example_selector_01.md | https://medium.com/donato-story/exploring-few-shot-prompts-with-langchain-852f27ea4e1d]
- [来源: reference/fetch_example_selector_05.md | https://www.swarnendu.de/blog/langchain-best-practices]
