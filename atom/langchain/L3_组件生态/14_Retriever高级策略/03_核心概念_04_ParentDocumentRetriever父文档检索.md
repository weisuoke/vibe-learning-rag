# Retriever高级策略 - 核心概念：ParentDocumentRetriever 父文档检索

> **小到大检索模式**：用小块精准匹配，返回大块完整上下文，解决 Embedding 精度与上下文完整性的根本矛盾

---

## 一句话定义

ParentDocumentRetriever 将文档拆成小块做 Embedding 检索，命中后返回对应的父文档（大块），让 LLM 拿到完整上下文。它继承自 MultiVectorRetriever，核心是"两层存储 + ID 关联"的架构。

---

## 小到大检索的核心矛盾

传统检索面临一个根本性的两难：

**小块 Embedding**：向量精确聚焦、检索命中率高，但送给 LLM 的上下文太碎片化，代词指代不明，回答质量差。

**大块 Embedding**：上下文完整、LLM 能理解全貌，但 Embedding 被无关内容稀释，检索精度下降。

```
查询: "Python 的创建者是谁？"

小块: "由 Guido van Rossum 于 1991 年创建" → 精准命中，但缺少上下文
大块: "Python 是一种解释型语言...（500字）" → 只有一句相关，Embedding 被稀释
```

**ParentDocumentRetriever 的解决方案 — 检索用小块，返回用大块：**

```
查询 → Embedding → 在小块向量中搜索 → 命中小块
  → 提取 parent_id → 从 DocStore 取父文档 → 返回完整大块给 LLM
```

类比：图书馆的索引卡片系统 — 通过关键词卡片（小块）快速定位，但借走的是整本书（父文档）。

---

## 两层存储架构

核心设计：将"搜索索引"和"文档存储"分离到两个独立的存储层。

**VectorStore（子块嵌入层）**：存储小块的 Embedding 向量，执行语义搜索。每个子块 metadata 中包含 `doc_id` 指向父文档。

**DocStore（父文档存储层）**：存储完整父文档，通过 ID 快速查找，纯 KV 存储，不做向量化。

```python
# VectorStore 中的子块
{"page_content": "Python 由 Guido 创建", "metadata": {"doc_id": "parent_abc123"}}

# DocStore 中的父文档
{"parent_abc123": Document(page_content="Python 是一种解释型语言...（完整内容）")}
```

| 对比维度 | 单一存储 | 两层存储 |
|----------|----------|----------|
| 检索精度 | 大块稀释 Embedding | 小块精准匹配 |
| 上下文质量 | 碎片化或冗余 | 完整父文档 |
| 存储效率 | 大块向量化成本高 | 只对小块向量化 |
| 灵活性 | 块大小固定 | 检索粒度和返回粒度独立 |

---

## MultiVectorRetriever 源码解析

ParentDocumentRetriever 继承自 MultiVectorRetriever，先理解基类。

### 核心属性

```python
class MultiVectorRetriever(BaseRetriever):
    """小到大模式：搜索子块，返回父文档。"""

    vectorstore: VectorStore          # 存储子块嵌入
    docstore: BaseStore[str, Document]  # 存储父文档
    id_key: str = "doc_id"            # 子块 metadata 中链接父文档 ID 的键名
    search_type: SearchType = SearchType.similarity  # similarity / mmr / threshold
    search_kwargs: dict = Field(default_factory=dict)  # 额外搜索参数
```

### 核心检索流程

```python
def _get_relevant_documents(
    self, query: str, *, run_manager: CallbackManagerForRetrieverRun
) -> list[Document]:
    # 第一步：根据 search_type 在 vectorstore 中搜索子块
    if self.search_type == SearchType.similarity:
        sub_docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
    elif self.search_type == SearchType.mmr:
        sub_docs = self.vectorstore.max_marginal_relevance_search(
            query, **self.search_kwargs
        )
    elif self.search_type == SearchType.similarity_score_threshold:
        sub_docs_and_similarities = (
            self.vectorstore.similarity_search_with_relevance_scores(
                query, **self.search_kwargs
            )
        )
        sub_docs = [sub_doc for sub_doc, _ in sub_docs_and_similarities]

    # 第二步：收集唯一的父文档 ID（保持首次出现顺序）
    ids = []
    for d in sub_docs:
        if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
            ids.append(d.metadata[self.id_key])

    # 第三步：从 docstore 批量获取父文档，过滤 None
    docs = self.docstore.mget(ids)
    return [d for d in docs if d is not None]
```

**关键设计决策：**
- `ids` 用列表而非集合，保持首次出现顺序（相关性排序）
- `mget` 批量获取，避免 N+1 查询问题
- 过滤 None 处理数据不一致的情况

---

## ParentDocumentRetriever 源码解析

继承 MultiVectorRetriever，添加自动分割和存储逻辑。

### 额外属性

```python
class ParentDocumentRetriever(MultiVectorRetriever):
    child_splitter: TextSplitter              # 必需，将文档分割成小块用于 Embedding
    parent_splitter: TextSplitter | None = None  # 可选，创建中间"父块"
    child_metadata_fields: Sequence[str] | None = None  # 子块元数据白名单
```

### add_documents 流程

```python
def add_documents(self, documents, ids=None, add_to_docstore=True):
    # 第一步：如果有 parent_splitter，先切成父块
    if self.parent_splitter is not None:
        documents = self.parent_splitter.split_documents(documents)

    doc_ids = ids or [str(uuid.uuid4()) for _ in documents]

    # 第二步：用 child_splitter 将父文档分割成子块
    docs, full_docs = [], []
    for i, doc in enumerate(documents):
        _id = doc_ids[i]
        sub_docs = self.child_splitter.split_documents([doc])
        for _doc in sub_docs:
            _doc.metadata[self.id_key] = _id
            # 元数据白名单过滤：防止大 metadata 膨胀向量索引
            if self.child_metadata_fields is not None:
                _doc.metadata = {
                    k: _doc.metadata[k] for k in self.child_metadata_fields
                    if k in _doc.metadata
                }
                _doc.metadata[self.id_key] = _id
        docs.extend(sub_docs)
        full_docs.append((_id, doc))

    # 第三步：子块存入 VectorStore，父文档存入 DocStore
    self.vectorstore.add_documents(docs)
    if add_to_docstore:
        self.docstore.mset(full_docs)
```

### child_metadata_fields 的作用

```python
# 原始文档 metadata 可能很大
doc = Document(page_content="...", metadata={
    "source": "report.pdf", "page": 5, "author": "张三",
    "tags": ["AI", "RAG"], "summary": "这是一份很长的摘要..."  # 大字段
})

# 不过滤：所有 metadata 都存入向量索引，浪费空间
# 设置白名单后：子块只保留必要字段
child_metadata_fields = ["source", "page"]
# 子块 metadata = {"source": "report.pdf", "page": 5, "doc_id": "xxx"}
```

---

## 两层 vs 三层层次结构

### 两层结构（无 parent_splitter）

```
原始文档 → child_splitter → 子块（带 doc_id）→ VectorStore
原始文档 → DocStore（作为父文档）
命中子块 → doc_id → DocStore 取原始文档
```

适用场景：原始文档不太长（< 5000 字），直接作为父文档返回。

### 三层结构（有 parent_splitter）

```
原始文档 → parent_splitter → 父块（2000字）→ DocStore
父块 → child_splitter → 子块（300字）→ VectorStore
命中子块 → doc_id → DocStore 取父块（2000字，而非整个原始文档）
```

适用场景：原始文档太长，返回整个文档会超出 LLM 上下文窗口。

### 选择建议

| 场景 | 推荐结构 | parent_splitter | child_splitter |
|------|----------|-----------------|----------------|
| 短文档（< 3000 字） | 两层 | 不设置 | 300-500 字 |
| 中等文档（3000-10000 字） | 三层 | 1500-2000 字 | 300-500 字 |
| 长文档（> 10000 字） | 三层 | 2000-4000 字 | 200-400 字 |

---

## 使用方式（完整代码）

### 两层结构 — 基础用法

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.storage import InMemoryStore
from langchain_core.documents import Document

# 1. 准备两层存储
vectorstore = Chroma(collection_name="parent_doc_demo", embedding_function=OpenAIEmbeddings())
docstore = InMemoryStore()

# 2. 创建 Retriever（不设 parent_splitter → 两层结构）
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50),
)

# 3. 添加文档
docs = [
    Document(
        page_content="Python 是一种解释型、面向对象的高级编程语言。"
                     "由 Guido van Rossum 于 1991 年创建。"
                     "Python 的设计哲学强调代码可读性和简洁性。"
                     "它支持多种编程范式，包括面向对象、函数式和过程式编程。",
        metadata={"source": "python_intro.pdf", "topic": "programming"}
    ),
]
retriever.add_documents(docs)

# 4. 检索 — 搜索小块，返回完整父文档
results = retriever.invoke("Python 的创建者是谁？")
for doc in results:
    print(f"来源: {doc.metadata.get('source')}, 长度: {len(doc.page_content)} 字符")
```

### 三层结构 — 长文档场景

```python
retriever_3tier = ParentDocumentRetriever(
    vectorstore=Chroma(collection_name="3tier", embedding_function=OpenAIEmbeddings()),
    docstore=InMemoryStore(),
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100),
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50),
    child_metadata_fields=["source", "topic"],  # 只保留必要元数据，节省索引空间
)

retriever_3tier.add_documents([long_doc])
# parent_splitter 先切成 ~1000 字父块 → child_splitter 再切成 ~300 字子块
# 检索命中子块后，返回的是 1000 字的父块（而非整个原始文档）
```

### 直接使用 MultiVectorRetriever

当子块不是文本分割产生的（比如 LLM 生成的摘要、问题），直接用基类：

```python
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever, SearchType
import uuid

retriever = MultiVectorRetriever(
    vectorstore=vectorstore, docstore=InMemoryStore(),
    id_key="doc_id", search_type=SearchType.mmr, search_kwargs={"k": 3},
)

# 手动管理子块和父文档的关联
parent_id = str(uuid.uuid4())
parent_doc = Document(page_content="完整的 Python 教程...（2000字）")

# 子块可以是摘要、问题、关键词等任何形式
child_docs = [
    Document(page_content="Python 基础语法概述", metadata={"doc_id": parent_id}),
    Document(page_content="Python 的创建者和历史", metadata={"doc_id": parent_id}),
]

retriever.vectorstore.add_documents(child_docs)
retriever.docstore.mset([(parent_id, parent_doc)])
results = retriever.invoke("Python 有哪些应用？")  # 返回完整 parent_doc
```

### 集成到 LCEL 管道

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template(
    "根据以下上下文回答问题。\n\n上下文：\n{context}\n\n问题：{question}\n\n回答："
)

rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
     "question": RunnablePassthrough()}
    | prompt | ChatOpenAI(model="gpt-4o-mini")
)
answer = rag_chain.invoke("Python 的创建者是谁？")
```

---

## 生产最佳实践

### DocStore 选型

```python
# 开发/测试：InMemoryStore（重启丢失）
from langchain.storage import InMemoryStore

# 生产：LocalFileStore（文件系统持久化）
from langchain.storage import LocalFileStore
store = LocalFileStore("./parent_docs/")

# 生产（高性能）：RedisStore
from langchain_community.storage import RedisStore
store = RedisStore(redis_url="redis://localhost:6379")
```

### 常见陷阱

**陷阱一：两层存储不同步。** DocStore 清空了但 VectorStore 还有子块指向不存在的父文档。始终通过 `add_documents()` 操作，手动操作时注意一致性。

**陷阱二：parent_splitter + 手动 ids。** parent_splitter 会改变文档数量，导致 ids 长度不匹配。使用 parent_splitter 时让系统自动生成 UUID。

**陷阱三：子块太细导致结果单一。** 同一父文档的多个子块命中，去重后只返回一个父文档。适当增大 `search_kwargs={"k": 10}` 增加多样性。

---

## 核心设计模式总结

| 模式 | 体现 |
|------|------|
| 两层存储 | VectorStore（嵌入索引）+ DocStore（文档存储）分离 |
| 策略模式 | SearchType 枚举切换搜索算法 |
| 模板方法 | add_documents 封装分割 + 存储逻辑 |
| 元数据过滤 | `child_metadata_fields` 白名单防止索引膨胀 |
| ID 关联 | `id_key` 建立子块到父文档的映射 |

---

**版本**：v1.0
**最后更新**：2026-02-27
**数据来源**：LangChain 源码分析（`multi_vector.py`, `parent_document_retriever.py`）+ Context7 官方文档
