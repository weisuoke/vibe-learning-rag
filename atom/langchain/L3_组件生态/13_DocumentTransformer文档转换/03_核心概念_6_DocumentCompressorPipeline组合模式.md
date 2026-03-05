# 核心概念 6：DocumentCompressorPipeline 组合模式

## 概念定义

**DocumentCompressorPipeline 是将多个文档转换器串联成管道的组合模式，每个转换器的输出作为下一个的输入，实现"加载 → 清洗 → 过滤 → 重排序"的完整文档处理流水线。**

[来源: Context7 + sourcecode/langchain]

这是 LangChain 文档转换体系中最重要的设计模式——不是单个转换器有多强，而是多个转换器组合起来能做什么。

---

## 为什么需要管道模式？

### 问题场景

```python
# 场景：RAG 系统中的文档处理需求
# 1. 检索到 20 个文档
# 2. 其中有冗余文档（内容重复）
# 3. 需要重排序优化 LLM 注意力
# 4. 最终送入 LLM 生成回答

# ❌ 手动串联：代码冗长，难以维护
docs = retriever.invoke(query)
docs = redundant_filter.transform_documents(docs)
docs = reordering.transform_documents(docs)
# 如果要加新步骤？改代码...
# 如果要调整顺序？改代码...
# 如果要复用这个流程？复制粘贴...

# ✅ 管道模式：声明式组合，灵活可复用
pipeline = DocumentCompressorPipeline(
    transformers=[redundant_filter, reordering]
)
docs = pipeline.transform_documents(docs)
```

### Unix 管道类比

```bash
# Unix 管道：每个命令的输出是下一个命令的输入
cat file.txt | grep "error" | sort | uniq | head -10

# DocumentCompressorPipeline 做的是同样的事：
# documents | filter_redundant | reorder | compress
```

---

## DocumentCompressorPipeline 核心架构

### 类定义

```python
# 来自 langchain.retrievers.document_compressors
class DocumentCompressorPipeline(BaseDocumentCompressor):
    """将多个转换器/压缩器串联成管道

    Attributes:
        transformers: List[Union[BaseDocumentTransformer, BaseDocumentCompressor]]
            转换器列表，按顺序执行
    """

    transformers: List[Union[BaseDocumentTransformer, BaseDocumentCompressor]]

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """依次执行每个转换器"""
        for transformer in self.transformers:
            if isinstance(transformer, BaseDocumentCompressor):
                documents = transformer.compress_documents(
                    documents, query, callbacks=callbacks
                )
            elif isinstance(transformer, BaseDocumentTransformer):
                documents = transformer.transform_documents(documents)
            else:
                raise ValueError(f"Unknown transformer type: {type(transformer)}")
        return documents
```

[来源: sourcecode/langchain]

### 关键设计

```
1. 统一接口：接受 BaseDocumentTransformer 和 BaseDocumentCompressor
2. 顺序执行：按 transformers 列表的顺序依次执行
3. 链式传递：每个转换器的输出是下一个的输入
4. 类型自适应：自动判断调用 transform_documents 还是 compress_documents
```

### 两种基类的区别

```python
# BaseDocumentTransformer：不需要 query
class BaseDocumentTransformer(ABC):
    def transform_documents(self, documents, **kwargs) -> Sequence[Document]:
        ...

# BaseDocumentCompressor：需要 query
class BaseDocumentCompressor(ABC):
    def compress_documents(self, documents, query, callbacks=None) -> Sequence[Document]:
        ...

# Pipeline 会自动处理这个区别：
# - 如果是 Transformer → 调用 transform_documents(docs)
# - 如果是 Compressor → 调用 compress_documents(docs, query)
```

---

## 与 ContextualCompressionRetriever 集成

### 核心架构

```
用户查询
    │
    ▼
ContextualCompressionRetriever
    │
    ├── base_retriever（基础检索器）
    │   └── 向量检索 → 返回 Top-K 文档
    │
    └── base_compressor（文档压缩器）
        └── DocumentCompressorPipeline
            ├── 转换器 1: 过滤冗余
            ├── 转换器 2: 重排序
            └── 转换器 3: ...
    │
    ▼
处理后的文档 → 送入 LLM
```

### 基础集成代码

```python
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# ===== 1. 创建基础检索器 =====
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# ===== 2. 创建管道组件 =====
# 组件 1: 冗余文档过滤
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

# 组件 2: 上下文重排序
reordering = LongContextReorder()

# ===== 3. 组装管道 =====
pipeline = DocumentCompressorPipeline(
    transformers=[redundant_filter, reordering]
)

# ===== 4. 创建压缩检索器 =====
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline,
    base_retriever=base_retriever,
)

# ===== 5. 使用 =====
results = compression_retriever.invoke("Python 异步编程怎么用？")
print(f"检索到 {len(results)} 个文档（已去重 + 重排序）")
```

[来源: Context7]

---

## 常见管道模式

### 模式 1：Clean → Split → Filter（清洗 → 分块 → 过滤）

```python
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import EmbeddingsRedundantFilter

# 场景：处理 HTML 网页文档
# 1. 清洗 HTML 标签
# 2. 分块（因为网页内容通常很长）
# 3. 过滤冗余分块

html_cleaner = Html2TextTransformer()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

pipeline = DocumentCompressorPipeline(
    transformers=[html_cleaner, text_splitter, redundant_filter]
)

# 流程：
# HTML 文档 → 纯文本 → 分块 → 去重
```

### 模式 2：Filter → Reorder（过滤 → 重排序）

```python
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)

# 场景：检索后优化
# 1. 过滤冗余文档
# 2. 重排序优化 LLM 注意力

redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
reordering = LongContextReorder()

pipeline = DocumentCompressorPipeline(
    transformers=[redundant_filter, reordering]
)

# 这是最常用的组合模式
# 来自 Context7 官方文档的推荐用法
```

[来源: Context7]

### 模式 3：Cluster → Reorder（聚类 → 重排序）

```python
from langchain_community.document_transformers import (
    EmbeddingsClusteringFilter,
    LongContextReorder,
)

# 场景：大量文档的多样性选择
# 1. 聚类选择代表性文档（保证多样性）
# 2. 重排序优化位置

clustering_filter = EmbeddingsClusteringFilter(
    embeddings=embeddings,
    num_clusters=5,      # 分成 5 个语义簇
    num_closest=1,       # 每个簇选 1 个代表
    sorted=True,         # 按原始检索分数排序
)
reordering = LongContextReorder()

pipeline = DocumentCompressorPipeline(
    transformers=[clustering_filter, reordering]
)

# 流程：
# 20 个文档 → 聚类选出 5 个代表 → 重排序
# 效果：既保证多样性，又优化 LLM 注意力
```

### 模式 4：完整 RAG 管道（Clean → Tag → Filter → Reorder）

```python
from langchain_community.document_transformers import (
    Html2TextTransformer,
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain_community.document_transformers.openai_functions import (
    create_metadata_tagger,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 完整的文档处理管道
# 适用于从网页抓取到 RAG 检索的全流程

# 步骤 1: HTML 清洗
html_cleaner = Html2TextTransformer()

# 步骤 2: 文本分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

# 步骤 3: 冗余过滤
embeddings = OpenAIEmbeddings()
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

# 步骤 4: 上下文重排序
reordering = LongContextReorder()

# 组装完整管道
full_pipeline = DocumentCompressorPipeline(
    transformers=[
        html_cleaner,       # 1. 清洗 HTML
        text_splitter,      # 2. 分块
        redundant_filter,   # 3. 去重
        reordering,         # 4. 重排序
    ]
)

# 使用
processed_docs = full_pipeline.transform_documents(raw_html_docs)
```

---

## 实战：构建完整的文档处理管道

```python
"""
DocumentCompressorPipeline 实战示例
演示：构建一个完整的 RAG 文档处理管道
"""

from langchain_core.documents import Document
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ===== 1. 准备测试数据 =====
documents = [
    Document(page_content="Python 的 asyncio 模块提供了异步编程的基础设施。"),
    Document(page_content="asyncio 是 Python 标准库中的异步 IO 框架。"),  # 与第1个冗余
    Document(page_content="async/await 语法让异步代码看起来像同步代码。"),
    Document(page_content="事件循环是 asyncio 的核心，负责调度协程。"),
    Document(page_content="协程通过 async def 定义，通过 await 调用。"),
    Document(page_content="asyncio.gather 可以并发运行多个协程。"),
    Document(page_content="Python 的 GIL 限制了多线程的并行能力。"),
    Document(page_content="异步编程适合 IO 密集型任务，如网络请求。"),
    Document(page_content="CPU 密集型任务应该使用多进程而非异步。"),
    Document(page_content="Python 异步编程框架提供了并发执行的能力。"),  # 与第1个冗余
]

# ===== 2. 创建向量库和检索器 =====
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# ===== 3. 构建处理管道 =====
# 步骤 1: 过滤冗余文档
redundant_filter = EmbeddingsRedundantFilter(
    embeddings=embeddings,
    similarity_threshold=0.95,  # 相似度 > 0.95 视为冗余
)

# 步骤 2: 上下文重排序
reordering = LongContextReorder()

# 组装管道
pipeline = DocumentCompressorPipeline(
    transformers=[redundant_filter, reordering]
)

# ===== 4. 创建压缩检索器 =====
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline,
    base_retriever=base_retriever,
)

# ===== 5. 构建 RAG 链 =====
prompt = ChatPromptTemplate.from_template(
    "根据以下上下文回答问题。\n\n上下文：{context}\n\n问题：{question}"
)
llm = ChatOpenAI(model="gpt-4o-mini")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ===== 6. 测试 =====
query = "Python 异步编程的核心概念是什么？"

# 直接检索（未处理）
raw_results = base_retriever.invoke(query)
print(f"原始检索: {len(raw_results)} 个文档")

# 管道处理后
processed_results = compression_retriever.invoke(query)
print(f"管道处理后: {len(processed_results)} 个文档（已去重 + 重排序）")

# RAG 回答
answer = rag_chain.invoke(query)
print(f"\n回答: {answer}")

# 预期输出：
# 原始检索: 10 个文档
# 管道处理后: 7-8 个文档（已去重 + 重排序）
# 回答: Python 异步编程的核心概念包括...
```

---

## 管道顺序最佳实践

### 顺序原则

```
推荐的管道顺序：

1. 内容转换（Clean）
   - Html2TextTransformer
   - BeautifulSoupTransformer
   → 先把内容变干净

2. 文本分块（Split）
   - RecursiveCharacterTextSplitter
   - TokenTextSplitter
   → 把长文档切成合适大小

3. 元数据增强（Tag）
   - OpenAIMetadataTagger
   - DoctranPropertyExtractor
   → 给文档打标签（可选）

4. 过滤（Filter）
   - EmbeddingsRedundantFilter
   - EmbeddingsClusteringFilter
   → 去除冗余，保留精华

5. 重排序（Reorder）
   - LongContextReorder
   → 最后一步，优化 LLM 注意力
```

### 为什么这个顺序？

```python
# 原则：从粗到细，从内容到位置

# ❌ 错误顺序：先重排序，再过滤
pipeline = DocumentCompressorPipeline(
    transformers=[reordering, redundant_filter]  # 错！
)
# 问题：重排序后再过滤，过滤可能破坏精心安排的顺序

# ❌ 错误顺序：先过滤，再清洗
pipeline = DocumentCompressorPipeline(
    transformers=[redundant_filter, html_cleaner]  # 错！
)
# 问题：HTML 标签会干扰冗余检测的准确性

# ✅ 正确顺序：清洗 → 过滤 → 重排序
pipeline = DocumentCompressorPipeline(
    transformers=[html_cleaner, redundant_filter, reordering]  # 对！
)
```

---

## 性能考量

### 管道中的瓶颈

```python
# 各组件的典型耗时：

# 1. Html2TextTransformer: ~1ms/doc（纯文本处理）
# 2. TextSplitter: ~1ms/doc（纯文本处理）
# 3. EmbeddingsRedundantFilter: ~100ms/batch（需要 Embedding API）← 瓶颈
# 4. EmbeddingsClusteringFilter: ~200ms/batch（需要 Embedding + 聚类）← 瓶颈
# 5. OpenAIMetadataTagger: ~500ms/doc（需要 LLM API）← 最大瓶颈
# 6. LongContextReorder: ~0.1ms（纯排列）

# 优化策略：
# 1. 把耗时操作放在后面（先用便宜的操作减少文档数量）
# 2. 元数据标注在入库时离线完成，不放在检索管道中
# 3. 使用缓存避免重复计算
```

### 文档数量变化

```python
# 管道中文档数量的变化：

# 输入: 20 个文档
#   ↓ Html2TextTransformer
# 20 个文档（数量不变，内容变干净）
#   ↓ RecursiveCharacterTextSplitter
# 60 个分块（数量增加！长文档被切分）
#   ↓ EmbeddingsRedundantFilter
# 45 个分块（数量减少，冗余被过滤）
#   ↓ LongContextReorder
# 45 个分块（数量不变，顺序改变）

# 注意：TextSplitter 会增加文档数量
# 所以过滤器要放在 Splitter 后面
```

---

## 自定义管道组件

### 创建自定义转换器

```python
from langchain_core.documents import Document
from langchain_core.documents.transformers import BaseDocumentTransformer
from typing import Sequence, Any

class ContentLengthFilter(BaseDocumentTransformer):
    """过滤掉内容过短的文档"""

    min_length: int = 50

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        return [
            doc for doc in documents
            if len(doc.page_content) >= self.min_length
        ]

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        return self.transform_documents(documents)

# 在管道中使用自定义组件
length_filter = ContentLengthFilter(min_length=100)

pipeline = DocumentCompressorPipeline(
    transformers=[
        length_filter,       # 自定义：过滤短文档
        redundant_filter,    # 内置：过滤冗余
        reordering,          # 内置：重排序
    ]
)
```

### 创建自定义压缩器

```python
from langchain.retrievers.document_compressors import BaseDocumentCompressor
from langchain_core.callbacks import Callbacks
from typing import Optional, Sequence

class TopNCompressor(BaseDocumentCompressor):
    """只保留前 N 个文档"""

    top_n: int = 5

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        return list(documents[:self.top_n])

# 在管道中使用
top_n = TopNCompressor(top_n=5)

pipeline = DocumentCompressorPipeline(
    transformers=[
        redundant_filter,    # 去重
        top_n,               # 只保留前 5 个
        reordering,          # 重排序
    ]
)
```

---

## 双重类比

### 前端类比：中间件管道

```javascript
// Express.js 中间件管道
app.use(cors());           // 1. CORS 处理
app.use(bodyParser());     // 2. 解析请求体
app.use(auth());           // 3. 认证
app.use(rateLimit());      // 4. 限流
app.use(router);           // 5. 路由处理

// DocumentCompressorPipeline 做的是同样的事：
// pipeline.use(htmlCleaner)      // 1. 清洗内容
// pipeline.use(textSplitter)     // 2. 分块
// pipeline.use(redundantFilter)  // 3. 去重
// pipeline.use(reordering)       // 4. 重排序

// 共同点：
// - 每个中间件/转换器只做一件事
// - 按顺序执行
// - 前一个的输出是后一个的输入
// - 可以灵活组合
```

### 日常生活类比：工厂流水线

```
场景：汽车工厂的生产流水线

原材料 → [冲压] → [焊接] → [涂装] → [总装] → 成品汽车

每个工位：
- 只做一件事（单一职责）
- 接收上一工位的半成品
- 输出给下一工位
- 可以替换或调整顺序

DocumentCompressorPipeline = 文档处理流水线

原始文档 → [清洗] → [分块] → [过滤] → [重排序] → 处理后的文档

同样的原则：
- 每个转换器只做一件事
- 链式传递
- 灵活组合
```

---

## 常见误区

### 误区 1：管道中的转换器越多越好 ❌

**为什么错？**
- 每个转换器都有计算成本
- 过多的处理步骤可能过度过滤，丢失有用信息
- 管道越长，调试越困难

**正确做法：**
```python
# ✅ 精简管道：只包含必要的步骤
pipeline = DocumentCompressorPipeline(
    transformers=[redundant_filter, reordering]
)

# ❌ 过度设计：不必要的步骤
pipeline = DocumentCompressorPipeline(
    transformers=[
        cleaner1, cleaner2, cleaner3,  # 三个清洗器？
        filter1, filter2, filter3,      # 三个过滤器？
        reordering,
    ]
)
```

### 误区 2：管道顺序无所谓 ❌

**为什么错？**
- 顺序直接影响结果
- 错误的顺序可能导致信息丢失或处理无效

**正确理解：**
```python
# 顺序 A: 先过滤再重排序 ✅
pipeline_a = DocumentCompressorPipeline(
    transformers=[redundant_filter, reordering]
)
# 结果：去重后的文档被合理重排序

# 顺序 B: 先重排序再过滤 ❌
pipeline_b = DocumentCompressorPipeline(
    transformers=[reordering, redundant_filter]
)
# 结果：精心安排的顺序被过滤打乱
```

### 误区 3：DocumentCompressorPipeline 只能用于检索 ❌

**为什么错？**
- Pipeline 可以独立使用，不一定要配合 ContextualCompressionRetriever
- 任何需要批量处理文档的场景都可以用

**正确理解：**
```python
# 场景 1: 配合检索器（最常见）
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline,
    base_retriever=base_retriever,
)

# 场景 2: 独立使用（文档预处理）
processed_docs = pipeline.transform_documents(raw_docs)

# 场景 3: 入库前处理
processed_docs = pipeline.transform_documents(loaded_docs)
vectorstore.add_documents(processed_docs)
```

---

## 总结

DocumentCompressorPipeline 是 LangChain 文档转换体系的"粘合剂"：

**核心价值：**
- 将多个转换器组合成可复用的处理流水线
- 声明式配置，灵活调整
- 与 ContextualCompressionRetriever 无缝集成

**常用组合：**
1. Filter → Reorder（最常用）
2. Clean → Split → Filter
3. Clean → Filter → Reorder（推荐的完整管道）

**顺序原则：** 清洗 → 分块 → 标注 → 过滤 → 重排序

**记住：** 管道的力量不在于单个组件，而在于组合。选择合适的组件、按正确的顺序排列，就能构建出高效的文档处理流水线。

[来源: Context7 + sourcecode/langchain]
