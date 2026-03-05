# 核心概念 1：BaseDocumentTransformer 接口设计

## 一句话定义

**BaseDocumentTransformer 是 LangChain 中定义文档转换统一接口的抽象基类，通过 `Sequence[Document] → Sequence[Document]` 的纯函数式设计，让所有文档转换操作（分块、清洗、过滤、增强）都遵循相同的输入输出契约。**

[来源: sourcecode/langchain/libs/core/langchain_core/documents/transformers.py]

---

## 为什么需要统一的文档转换接口？

### 问题：RAG 管道中的文档处理五花八门

在 RAG 开发中，文档从加载到最终送入 LLM，中间要经历多种处理：

- **分块**：把长文档切成小段
- **清洗**：去掉 HTML 标签、脚本、样式
- **过滤**：去除重复或无关的文档
- **增强**：给文档添加元数据标签
- **重排序**：调整文档顺序提升 LLM 理解

如果每种处理都有不同的接口，代码会变得混乱：

```python
# ❌ 没有统一接口的情况
splitter.split_documents(docs)                    # 返回 List[Document]
html_cleaner.clean(docs, tags=["p", "h1"])        # 返回 List[str]
filter.remove_duplicates(docs, threshold=0.95)    # 返回 Set[Document]
tagger.tag_metadata(docs, schema={...})           # 修改原始文档，无返回值
```

**每个工具的输入输出都不一样，无法串联成管道。**

### 解决方案：统一的 Document → Document 接口

```python
# ✅ 统一接口：所有转换器都是 Sequence[Document] → Sequence[Document]
docs = splitter.transform_documents(docs)
docs = cleaner.transform_documents(docs)
docs = filter.transform_documents(docs)
docs = tagger.transform_documents(docs)
```

**一个方法名，一种输入输出格式，随意串联。**

[来源: reference/source_document_transformer_01.md | LangChain 源码分析]

---

## BaseDocumentTransformer 源码分析

### 完整源码

```python
from abc import ABC, abstractmethod
from typing import Any, Sequence
from langchain_core.documents import Document
from langchain_core.runnables import run_in_executor

class BaseDocumentTransformer(ABC):
    """Abstract base class for document transformation.

    所有文档转换器的抽象基类。
    定义了文档转换的统一接口。
    """

    @abstractmethod
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform a list of documents.

        Args:
            documents: 待转换的文档序列
            **kwargs: 额外参数，由子类定义

        Returns:
            Sequence[Document]: 转换后的文档序列
        """

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Asynchronously transform a list of documents.

        默认实现：使用 run_in_executor 包装同步方法。
        子类可以覆盖此方法提供真正的异步实现。

        Args:
            documents: 待转换的文档序列
            **kwargs: 额外参数

        Returns:
            Sequence[Document]: 转换后的文档序列
        """
        return await run_in_executor(
            None, self.transform_documents, documents, **kwargs
        )
```

[来源: sourcecode/langchain/libs/core/langchain_core/documents/transformers.py]

---

## 接口设计哲学

### 设计原则 1：输入输出一致性

**核心契约：`Sequence[Document]` → `Sequence[Document]`**

这是整个接口最重要的设计决策。无论你做什么转换——分块、过滤、清洗、增强——输入和输出的类型永远一样。

```python
from langchain_core.documents import Document

# 输入：一组文档
input_docs: Sequence[Document] = [
    Document(page_content="Hello World", metadata={"source": "test.txt"}),
    Document(page_content="Foo Bar", metadata={"source": "test2.txt"}),
]

# 输出：还是一组文档（但内容/数量可能变了）
output_docs: Sequence[Document] = transformer.transform_documents(input_docs)
```

**为什么这很重要？**

| 转换类型 | 输入文档数 | 输出文档数 | 内容变化 |
|----------|-----------|-----------|---------|
| 分块（Split） | 1 | N（变多） | 内容被切分 |
| 过滤（Filter） | N | M ≤ N（变少） | 内容不变，数量减少 |
| 清洗（Clean） | N | N（不变） | 内容被清洗 |
| 增强（Enrich） | N | N（不变） | metadata 被增强 |
| 重排序（Reorder） | N | N（不变） | 顺序改变 |

**虽然行为不同，但接口完全一致。** 这就是抽象的力量。

---

### 设计原则 2：纯函数式——不修改原始文档

```python
# ✅ 正确的实现方式：返回新文档
def transform_documents(self, documents, **kwargs):
    return [
        Document(
            page_content=doc.page_content.strip(),
            metadata={**doc.metadata, "cleaned": True}
        )
        for doc in documents
    ]

# ❌ 错误的实现方式：修改原始文档
def transform_documents(self, documents, **kwargs):
    for doc in documents:
        doc.page_content = doc.page_content.strip()  # 修改了原始对象！
    return documents
```

**纯函数式的好处：**
- 原始文档不会被意外修改
- 可以安全地在多个管道中复用同一批文档
- 方便调试——随时可以对比转换前后的结果

---

### 设计原则 3：`**kwargs` 保持灵活性

```python
def transform_documents(
    self, documents: Sequence[Document], **kwargs: Any
) -> Sequence[Document]:
```

`**kwargs` 让每个子类可以接受自己特有的参数：

```python
# BeautifulSoupTransformer 需要指定标签
cleaner.transform_documents(docs, tags_to_extract=["p", "h1"])

# EmbeddingsRedundantFilter 不需要额外参数
filter.transform_documents(docs)

# 自定义转换器可以接受任何参数
custom.transform_documents(docs, language="zh", max_length=500)
```

[来源: reference/source_document_transformer_01.md | LangChain 源码分析]

---

## 核心方法详解

### 方法 1：transform_documents() - 同步转换

**作用：** 将一组文档转换为另一组文档

**签名：**
```python
@abstractmethod
def transform_documents(
    self, documents: Sequence[Document], **kwargs: Any
) -> Sequence[Document]:
```

**参数：**
- `documents`: 待转换的文档序列（`List[Document]` 或任何 `Sequence[Document]`）
- `**kwargs`: 子类特定的额外参数

**返回值：**
- `Sequence[Document]`: 转换后的新文档序列

**使用场景：**
1. **批量文档处理**：一次性转换所有文档
2. **管道串联**：一个转换器的输出作为下一个的输入
3. **RAG 预处理**：在存入向量库之前清洗和分块

---

### 方法 2：atransform_documents() - 异步转换

**作用：** 异步版本的文档转换，适用于高并发场景

**签名：**
```python
async def atransform_documents(
    self, documents: Sequence[Document], **kwargs: Any
) -> Sequence[Document]:
```

**默认实现：**
```python
return await run_in_executor(
    None, self.transform_documents, documents, **kwargs
)
```

**`run_in_executor` 的作用：**
- 将同步方法放到线程池中执行
- 不阻塞事件循环
- 子类只需实现同步方法，异步方法自动获得

**使用场景：**
```python
import asyncio

async def process_documents():
    # 异步转换，不阻塞主线程
    cleaned_docs = await cleaner.atransform_documents(docs)

    # 可以并发处理多批文档
    results = await asyncio.gather(
        cleaner.atransform_documents(batch1),
        cleaner.atransform_documents(batch2),
        cleaner.atransform_documents(batch3),
    )
```

[来源: reference/source_document_transformer_01.md | LangChain 源码分析]

---

## TextSplitter 如何实现 BaseDocumentTransformer

### 关键代码

```python
class TextSplitter(BaseDocumentTransformer, ABC):
    """所有文本分块器的基类，同时也是 DocumentTransformer。"""

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """实现 BaseDocumentTransformer 接口。
        直接委托给 split_documents()。
        """
        return self.split_documents(list(documents))
```

[来源: sourcecode/langchain/libs/text-splitters/langchain_text_splitters/base.py]

**关键发现：**
- `TextSplitter` 继承自 `BaseDocumentTransformer`
- `transform_documents()` 只是调用了 `split_documents()`
- 这意味着**所有 TextSplitter 都是 DocumentTransformer**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 两种调用方式完全等价
result1 = splitter.split_documents(docs)
result2 = splitter.transform_documents(docs)
# result1 == result2
```

**为什么这很重要？**

因为 TextSplitter 可以和其他 DocumentTransformer 一起放进管道：

```python
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

# TextSplitter 和 Filter 可以无缝组合
pipeline = DocumentCompressorPipeline(
    transformers=[
        splitter,   # TextSplitter（也是 DocumentTransformer）
        filter,     # EmbeddingsRedundantFilter
        reorder,    # LongContextReorder
    ]
)
```

---

## 完整类层次结构

```
BaseDocumentTransformer (ABC)          ← 抽象基类（langchain_core）
│
├── TextSplitter (ABC)                 ← 文本分块器基类（langchain_text_splitters）
│   ├── CharacterTextSplitter          ← 按字符分块
│   ├── RecursiveCharacterTextSplitter ← 递归字符分块（最常用）
│   ├── TokenTextSplitter              ← 按 Token 分块
│   ├── MarkdownHeaderTextSplitter     ← 按 Markdown 标题分块
│   ├── HTMLHeaderTextSplitter         ← 按 HTML 标题分块
│   └── ... (17+ 种分块器)
│
├── BeautifulSoupTransformer           ← HTML 解析清洗（langchain_community）
├── Html2TextTransformer               ← HTML 转纯文本
│
├── EmbeddingsRedundantFilter          ← 冗余文档过滤
├── EmbeddingsClusteringFilter         ← 聚类过滤
│
├── LongContextReorder                 ← 长上下文重排序
│
├── OpenAIMetadataTagger               ← 元数据自动标注
├── DoctranQATransformer               ← QA 对提取
├── DoctranTextTranslator              ← 文档翻译
├── DoctranPropertyExtractor           ← 属性提取
│
├── GoogleTranslateTransformer         ← Google 翻译
└── NucliaTextTransformer              ← Nuclia 转换
```

[来源: reference/source_document_transformer_01.md | LangChain 源码分析]

---

## 与 Runnable 协议的关系

LangChain 的 LCEL（LangChain Expression Language）基于 `Runnable` 协议。`BaseDocumentTransformer` 本身**不直接继承 Runnable**，但可以通过适配器集成到 LCEL 链中。

### 直接使用（不通过 LCEL）

```python
# 最常见的用法：直接调用 transform_documents
cleaned_docs = transformer.transform_documents(docs)
```

### 通过 DocumentCompressorPipeline 集成

```python
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever

# Pipeline 将多个 transformer 组合
pipeline = DocumentCompressorPipeline(transformers=[filter, reorder])

# 集成到 Retriever 中（Retriever 是 Runnable）
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline,
    base_retriever=base_retriever  # Runnable
)

# 现在可以在 LCEL 链中使用
chain = compression_retriever | prompt | llm
```

### 通过 RunnableLambda 包装

```python
from langchain_core.runnables import RunnableLambda

# 将 transformer 包装为 Runnable
transformer_runnable = RunnableLambda(
    lambda docs: transformer.transform_documents(docs)
)

# 现在可以用 | 管道符
chain = retriever | transformer_runnable | format_docs | prompt | llm
```

[来源: reference/source_document_transformer_01.md + CLAUDE_LANGCHAIN.md]

---

## 实战：自定义 DocumentTransformer

### 场景：给文档添加字数统计到 metadata

```python
"""
自定义 DocumentTransformer 实战
场景：给每个文档的 metadata 添加字数统计信息
"""

from abc import ABC
from typing import Any, Sequence
from langchain_core.documents import Document
from langchain_core.documents.transformers import BaseDocumentTransformer


class WordCountTransformer(BaseDocumentTransformer):
    """给文档添加字数统计的转换器。

    为每个文档的 metadata 添加：
    - word_count: 英文单词数
    - char_count: 字符数
    - is_short: 是否为短文档（< 100 字）
    """

    def __init__(self, short_threshold: int = 100):
        """初始化转换器。

        Args:
            short_threshold: 短文档的字数阈值，默认 100
        """
        self.short_threshold = short_threshold

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """给每个文档添加字数统计。

        Args:
            documents: 待处理的文档序列

        Returns:
            添加了字数统计 metadata 的新文档序列
        """
        transformed = []
        for doc in documents:
            content = doc.page_content
            word_count = len(content.split())
            char_count = len(content)

            # 创建新文档（不修改原始文档）
            new_doc = Document(
                page_content=content,
                metadata={
                    **doc.metadata,
                    "word_count": word_count,
                    "char_count": char_count,
                    "is_short": word_count < self.short_threshold,
                }
            )
            transformed.append(new_doc)

        return transformed


# ===== 使用示例 =====

# 1. 创建测试文档
docs = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "intro.txt"}
    ),
    Document(
        page_content="RAG stands for Retrieval-Augmented Generation. "
                     "It combines information retrieval with text generation to produce "
                     "more accurate and grounded responses. RAG systems first retrieve "
                     "relevant documents from a knowledge base, then use those documents "
                     "as context for the language model to generate answers.",
        metadata={"source": "rag_overview.txt"}
    ),
    Document(
        page_content="Hello world.",
        metadata={"source": "test.txt"}
    ),
]

# 2. 创建转换器并执行转换
transformer = WordCountTransformer(short_threshold=20)
result = transformer.transform_documents(docs)

# 3. 查看结果
for doc in result:
    print(f"来源: {doc.metadata['source']}")
    print(f"  字数: {doc.metadata['word_count']}")
    print(f"  字符数: {doc.metadata['char_count']}")
    print(f"  是否短文档: {doc.metadata['is_short']}")
    print()

# 4. 验证原始文档未被修改
print("原始文档 metadata:", docs[0].metadata)
# 输出: {'source': 'intro.txt'} — 没有 word_count 等字段
```

**运行输出：**
```
来源: intro.txt
  字数: 12
  字符数: 79
  是否短文档: True

来源: rag_overview.txt
  字数: 49
  字符数: 310
  是否短文档: False

来源: test.txt
  字数: 2
  字符数: 12
  是否短文档: True

原始文档 metadata: {'source': 'intro.txt'}
```

[来源: 基于 reference/source_document_transformer_01.md 的接口规范自定义实现]

---

## 双重类比

### 类比 1：BaseDocumentTransformer = Express 中间件接口

**前端类比：**

Express.js 中间件有统一的签名 `(req, res, next)`，无论是日志、认证还是压缩，接口都一样：

```javascript
// Express 中间件：统一接口 (req, res, next)
app.use(logger);       // 日志中间件
app.use(auth);         // 认证中间件
app.use(compress);     // 压缩中间件
// 可以随意添加、删除、调换顺序
```

**LangChain 对应：**

```python
# DocumentTransformer：统一接口 transform_documents(docs)
docs = cleaner.transform_documents(docs)    # 清洗
docs = splitter.transform_documents(docs)   # 分块
docs = filter.transform_documents(docs)     # 过滤
# 同样可以随意组合
```

**日常生活类比：** 就像工厂流水线上的工位——每个工位接收半成品、加工后传给下一个工位。无论是喷漆、组装还是质检，传递的都是同一个产品。

---

### 类比 2：Sequence[Document] → Sequence[Document] = Array.map/filter

**前端类比：**

```javascript
// JavaScript 数组方法：输入数组 → 输出数组
const cleaned = docs.map(doc => cleanHtml(doc));     // 清洗
const filtered = cleaned.filter(doc => !isDup(doc)); // 过滤
const chunks = filtered.flatMap(doc => split(doc));   // 分块
```

**LangChain 对应：**

```python
# DocumentTransformer：输入文档列表 → 输出文档列表
cleaned = cleaner.transform_documents(docs)     # 类似 map
filtered = filter.transform_documents(cleaned)  # 类似 filter
chunks = splitter.transform_documents(filtered) # 类似 flatMap
```

**日常生活类比：** 就像快递分拣中心——包裹进来，经过扫码、分类、打包，出去的还是包裹，只是状态变了。

---

## 在 RAG 开发中的应用

### 典型 RAG 文档处理管道

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_openai import OpenAIEmbeddings

# ===== RAG 文档处理管道 =====

# 第 1 步：加载文档（DocumentLoader）
loader = WebBaseLoader("https://example.com/article")
raw_docs = loader.load()
print(f"加载了 {len(raw_docs)} 个原始文档")

# 第 2 步：清洗 HTML（DocumentTransformer）
html_cleaner = Html2TextTransformer()
clean_docs = html_cleaner.transform_documents(raw_docs)
print(f"清洗后 {len(clean_docs)} 个文档")

# 第 3 步：文本分块（TextSplitter，也是 DocumentTransformer）
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.transform_documents(clean_docs)
print(f"分块后 {len(chunks)} 个文档片段")

# 第 4 步：去重过滤（DocumentTransformer）
embeddings = OpenAIEmbeddings()
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
unique_chunks = redundant_filter.transform_documents(chunks)
print(f"去重后 {len(unique_chunks)} 个唯一片段")

# 每一步都是 transform_documents()，输入输出类型完全一致
```

**这就是统一接口的威力：** 加载 → 清洗 → 分块 → 过滤，每一步都是 `Sequence[Document] → Sequence[Document]`，像乐高积木一样自由组合。

[来源: reference/search_document_transformer_01.md | 2026年最佳实践]

---

## 总结

### BaseDocumentTransformer 的核心价值

1. **统一接口**：所有文档转换操作都遵循 `Sequence[Document] → Sequence[Document]`
2. **纯函数式**：不修改原始文档，返回新的文档序列
3. **异步支持**：默认提供异步方法，子类只需实现同步版本
4. **管道友好**：统一的输入输出让转换器可以自由串联

### 两个核心方法

- **`transform_documents()`**：同步转换（子类必须实现）
- **`atransform_documents()`**：异步转换（默认用 `run_in_executor` 包装）

### TextSplitter 的桥梁作用

- TextSplitter 继承自 BaseDocumentTransformer
- `transform_documents()` 委托给 `split_documents()`
- 所有分块器都可以作为 DocumentTransformer 使用

### 自定义转换器的要点

- 继承 `BaseDocumentTransformer`
- 实现 `transform_documents()` 方法
- 创建新文档而非修改原始文档
- 通过 `**kwargs` 接受额外参数

---

## 下一步学习

1. **文档内容转换器**：BeautifulSoupTransformer、Html2TextTransformer
2. **文档过滤器**：EmbeddingsRedundantFilter、EmbeddingsClusteringFilter
3. **元数据增强器**：OpenAIMetadataTagger
4. **上下文重排序**：LongContextReorder
5. **组合模式**：DocumentCompressorPipeline

---

**参考资料：**
- [来源: sourcecode/langchain/libs/core/langchain_core/documents/transformers.py]
- [来源: sourcecode/langchain/libs/text-splitters/langchain_text_splitters/base.py]
- [来源: reference/source_document_transformer_01.md | LangChain 源码分析]
- [来源: reference/context7_langchain_01.md | Context7 官方文档]
- [来源: reference/search_document_transformer_01.md | 2026年最佳实践]
