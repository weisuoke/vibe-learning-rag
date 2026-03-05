---
type: source_code_analysis
source: sourcecode/langchain
analyzed_files:
  - libs/core/langchain_core/example_selectors/base.py
  - libs/core/langchain_core/example_selectors/semantic_similarity.py
  - libs/core/langchain_core/example_selectors/length_based.py
  - libs/langchain/langchain_classic/prompts/example_selector/ngram_overlap.py
analyzed_at: 2026-02-26
knowledge_point: ExampleSelector示例选择
---

# 源码分析：ExampleSelector 核心实现

## 分析的文件

### 1. 基础抽象类
- `libs/core/langchain_core/example_selectors/base.py` - BaseExampleSelector 接口定义

### 2. 核心实现
- `libs/core/langchain_core/example_selectors/semantic_similarity.py` - 语义相似度选择器
- `libs/core/langchain_core/example_selectors/length_based.py` - 基于长度的选择器

### 3. 扩展实现
- `libs/langchain/langchain_classic/prompts/example_selector/ngram_overlap.py` - N-gram 重叠选择器（重定向）

---

## 关键发现

### 1. BaseExampleSelector 抽象接口

**核心方法：**
```python
class BaseExampleSelector(ABC):
    @abstractmethod
    def add_example(self, example: dict[str, str]) -> Any:
        """添加新示例到存储"""

    async def aadd_example(self, example: dict[str, str]) -> Any:
        """异步添加示例"""
        return await run_in_executor(None, self.add_example, example)

    @abstractmethod
    def select_examples(self, input_variables: dict[str, str]) -> list[dict]:
        """根据输入选择示例"""

    async def aselect_examples(self, input_variables: dict[str, str]) -> list[dict]:
        """异步选择示例"""
        return await run_in_executor(None, self.select_examples, input_variables)
```

**设计特点：**
- 抽象基类定义统一接口
- 同步和异步方法都支持
- 简洁的 API 设计（只有两个核心方法）

---

### 2. SemanticSimilarityExampleSelector（语义相似度选择器）

**核心架构：**
```python
class SemanticSimilarityExampleSelector(_VectorStoreExampleSelector):
    """基于语义相似度选择示例"""

    vectorstore: VectorStore  # 向量存储后端
    k: int = 4  # 选择的示例数量
    example_keys: list[str] | None = None  # 过滤示例的键
    input_keys: list[str] | None = None  # 过滤输入的键
    vectorstore_kwargs: dict[str, Any] | None = None  # 向量存储额外参数
```

**关键方法：**

1. **select_examples()** - 选择示例
```python
def select_examples(self, input_variables: dict[str, str]) -> list[dict]:
    # 将输入转换为文本
    query_text = self._example_to_text(input_variables, self.input_keys)

    # 使用向量存储进行相似度搜索
    example_docs = self.vectorstore.similarity_search(
        query_text,
        k=self.k,
        **vectorstore_kwargs
    )

    # 从文档中提取示例
    return self._documents_to_examples(example_docs)
```

2. **from_examples()** - 类方法快速创建
```python
@classmethod
def from_examples(
    cls,
    examples: list[dict],
    embeddings: Embeddings,
    vectorstore_cls: type[VectorStore],
    k: int = 4,
    input_keys: list[str] | None = None,
    **vectorstore_cls_kwargs: Any,
) -> SemanticSimilarityExampleSelector:
    # 将示例转换为文本
    string_examples = [cls._example_to_text(eg, input_keys) for eg in examples]

    # 创建向量存储
    vectorstore = vectorstore_cls.from_texts(
        string_examples, embeddings, metadatas=examples, **vectorstore_cls_kwargs
    )

    return cls(vectorstore=vectorstore, k=k, input_keys=input_keys)
```

**设计亮点：**
- 依赖 VectorStore 抽象，支持多种向量数据库
- 支持 `input_keys` 过滤输入变量（只基于部分输入进行检索）
- 支持 `example_keys` 过滤示例字段（只返回部分字段）
- 同步和异步方法都支持
- `from_examples()` 类方法简化创建流程

---

### 3. MaxMarginalRelevanceExampleSelector（MMR 选择器）

**核心特性：**
```python
class MaxMarginalRelevanceExampleSelector(_VectorStoreExampleSelector):
    """基于最大边际相关性选择示例

    论文：https://arxiv.org/pdf/2211.13892.pdf
    """

    fetch_k: int = 20  # 先获取 20 个候选
    k: int = 4  # 最终返回 4 个
```

**选择策略：**
```python
def select_examples(self, input_variables: dict[str, str]) -> list[dict]:
    # 使用 MMR 算法选择示例
    example_docs = self.vectorstore.max_marginal_relevance_search(
        self._example_to_text(input_variables, self.input_keys),
        k=self.k,
        fetch_k=self.fetch_k,  # 先获取更多候选
    )
    return self._documents_to_examples(example_docs)
```

**MMR 优势：**
- 平衡相关性和多样性
- 避免选择过于相似的示例
- 提高 Few-shot learning 效果

---

### 4. LengthBasedExampleSelector（基于长度的选择器）

**核心架构：**
```python
class LengthBasedExampleSelector(BaseExampleSelector, BaseModel):
    """基于长度限制选择示例"""

    examples: list[dict]  # 示例列表
    example_prompt: PromptTemplate  # 示例格式化模板
    get_text_length: Callable[[str], int] = _get_length_based  # 长度计算函数
    max_length: int = 2048  # 最大长度限制
    example_text_lengths: list[int] = Field(default_factory=list)  # 每个示例的长度
```

**选择逻辑：**
```python
def select_examples(self, input_variables: dict[str, str]) -> list[dict]:
    # 计算输入的长度
    inputs = " ".join(input_variables.values())
    remaining_length = self.max_length - self.get_text_length(inputs)

    # 按顺序选择示例，直到达到长度限制
    i = 0
    examples = []
    while remaining_length > 0 and i < len(self.examples):
        new_length = remaining_length - self.example_text_lengths[i]
        if new_length < 0:
            break
        examples.append(self.examples[i])
        remaining_length = new_length
        i += 1

    return examples
```

**设计特点：**
- 简单直接的长度控制
- 自定义长度计算函数（默认按单词数）
- 按顺序选择（不考虑相关性）
- 适合 Token 限制场景

**默认长度计算：**
```python
def _get_length_based(text: str) -> int:
    return len(re.split(r"\n| ", text))  # 按空格和换行符分割
```

---

### 5. NGramOverlapExampleSelector（N-gram 重叠选择器）

**位置：** `langchain_community.example_selectors.ngram_overlap`

**特点：**
- 基于 N-gram 重叠度选择示例
- 适合文本相似度场景
- 不依赖向量化（更轻量）

---

## 核心设计模式总结

### 1. 抽象工厂模式
- `BaseExampleSelector` 定义接口
- 多种具体实现（Semantic、Length、MMR、NGram）

### 2. 策略模式
- 不同的选择策略可以互换
- 统一的 `select_examples()` 接口

### 3. 依赖注入
- `SemanticSimilarityExampleSelector` 依赖 `VectorStore` 抽象
- `LengthBasedExampleSelector` 依赖 `PromptTemplate`

### 4. 异步支持
- 所有选择器都支持异步操作
- 使用 `run_in_executor` 包装同步方法

---

## 与 Few-shot Learning 的关系

### 1. 动态示例选择
```python
# 静态 Few-shot（固定示例）
prompt = FewShotPromptTemplate(
    examples=[ex1, ex2, ex3],  # 固定示例
    ...
)

# 动态 Few-shot（ExampleSelector）
selector = SemanticSimilarityExampleSelector.from_examples(
    examples=[ex1, ex2, ex3, ex4, ex5],
    embeddings=embeddings,
    vectorstore_cls=Chroma,
    k=3
)
prompt = FewShotPromptTemplate(
    example_selector=selector,  # 动态选择
    ...
)
```

### 2. 优化 Prompt 效果
- 选择最相关的示例
- 控制 Prompt 长度
- 提高 LLM 理解能力

---

## 实际应用场景

### 1. RAG 系统中的 Few-shot
```python
# 根据用户查询动态选择示例
selector = SemanticSimilarityExampleSelector.from_examples(
    examples=qa_examples,
    embeddings=OpenAIEmbeddings(),
    vectorstore_cls=Chroma,
    k=3
)

# 在 RAG 链中使用
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_with_selector
    | llm
    | parser
)
```

### 2. 对话系统中的示例管理
```python
# 基于对话历史选择相关示例
selector = MaxMarginalRelevanceExampleSelector.from_examples(
    examples=conversation_examples,
    embeddings=embeddings,
    vectorstore_cls=FAISS,
    k=5,
    fetch_k=20  # 保证多样性
)
```

### 3. Token 限制场景
```python
# 控制 Prompt 长度
selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=2048  # 限制总长度
)
```

---

## 关键技术点

### 1. 示例文本化
```python
@staticmethod
def _example_to_text(example: dict[str, str], input_keys: list[str] | None) -> str:
    if input_keys:
        return " ".join(sorted_values({key: example[key] for key in input_keys}))
    return " ".join(sorted_values(example))
```

### 2. 文档转示例
```python
def _documents_to_examples(self, documents: list[Document]) -> list[dict]:
    # 从元数据中提取示例
    examples = [dict(e.metadata) for e in documents]

    # 如果指定了 example_keys，只返回这些键
    if self.example_keys:
        examples = [{k: eg[k] for k in self.example_keys} for eg in examples]

    return examples
```

### 3. 异步支持
```python
async def aselect_examples(self, input_variables: dict[str, str]) -> list[dict]:
    # 使用异步向量搜索
    example_docs = await self.vectorstore.asimilarity_search(
        self._example_to_text(input_variables, self.input_keys),
        k=self.k,
        **vectorstore_kwargs,
    )
    return self._documents_to_examples(example_docs)
```

---

## 性能优化考虑

### 1. 向量存储选择
- **FAISS**：内存中，速度快
- **Chroma**：持久化，易用
- **Pinecone**：云端，可扩展

### 2. 缓存策略
- 缓存 Embedding 结果
- 缓存相似度搜索结果

### 3. 批量操作
- 批量添加示例
- 批量选择示例

---

## 扩展性设计

### 1. 自定义选择器
```python
class CustomExampleSelector(BaseExampleSelector):
    def add_example(self, example: dict[str, str]) -> Any:
        # 自定义添加逻辑
        pass

    def select_examples(self, input_variables: dict[str, str]) -> list[dict]:
        # 自定义选择逻辑
        pass
```

### 2. 组合选择器
```python
# 先用语义相似度筛选，再用长度限制
semantic_selector = SemanticSimilarityExampleSelector(...)
length_selector = LengthBasedExampleSelector(...)
```

---

## 总结

### 核心优势
1. **统一接口**：BaseExampleSelector 定义清晰的抽象
2. **多种策略**：语义、长度、MMR、N-gram 等
3. **异步支持**：所有操作都支持异步
4. **易于扩展**：可以自定义选择器
5. **与 LangChain 生态集成**：无缝集成 VectorStore、PromptTemplate

### 适用场景
- Few-shot learning
- Prompt 优化
- RAG 系统
- 对话系统
- Token 限制场景

### 设计哲学
- 简单而强大的抽象
- 依赖注入和策略模式
- 异步优先
- 可组合性
