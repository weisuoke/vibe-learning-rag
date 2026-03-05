# Retriever高级策略 - 核心概念：SelfQueryRetriever 自查询

> **LLM 驱动的结构化查询**：让 LLM 自动从自然语言中提取语义查询和元数据过滤条件，实现"既搜内容又筛属性"

---

## 一句话定义

SelfQueryRetriever 用 LLM 将用户的自然语言查询解析为两部分 — 语义搜索文本 + 结构化元数据过滤器，然后在向量存储上执行精确的混合查询。通过访问者模式（Visitor Pattern）支持 20+ 种向量存储的原生过滤语法。

---

## 为什么需要自查询？

### 用户查询中隐含的元数据过滤条件

用户的自然语言查询往往同时包含两种信息：

```
用户查询: "推荐一部 2020 年之后的科幻电影，评分 8 分以上"

隐含的语义搜索: "科幻电影推荐"
隐含的元数据过滤: year > 2020, genre == "科幻", rating >= 8.0
```

传统向量检索把整个查询当语义搜索，"2020 年之后"和"8 分以上"是精确的数值过滤，Embedding 无法处理。结果可能返回 2019 年的电影或评分 7.5 的电影。

### SelfQueryRetriever 的转换流程

```
输入: "推荐一部 2020 年之后的科幻电影，评分 8 分以上"
  │
  ▼  LLM 解析（query_constructor）
StructuredQuery:
  query: "科幻电影推荐"                    ← 语义部分
  filter: AND(year > 2020, genre == "科幻", rating >= 8.0)  ← 结构化部分
  │
  ▼  Visitor 翻译为向量存储原生语法
vectorstore.similarity_search(
  query="科幻电影推荐",
  filter={"$and": [{"year": {"$gt": 2020}}, {"genre": "科幻"}, {"rating": {"$gte": 8.0}}]}
)
```

更多例子：

| 用户查询 | 语义部分 | 过滤部分 |
|----------|----------|----------|
| "关于机器学习的英文论文" | "机器学习" | language == "en" |
| "张三写的 Python 教程" | "Python 教程" | author == "张三" |
| "价格低于 100 元的编程书" | "编程书" | price < 100 |

---

## 访问者模式与查询翻译

### StructuredQuery 结构

LLM 解析后生成的中间表示是一棵表达式树：

```python
class StructuredQuery:
    query: str                        # 语义搜索文本
    filter: FilterDirective | None    # 过滤条件树

class Comparison:  # 叶节点：单个比较
    comparator: Comparator  # eq, ne, gt, gte, lt, lte, contain, like, in, nin
    attribute: str          # 元数据字段名
    value: Any              # 比较值

class Operation:   # 内部节点：逻辑运算
    operator: Operator      # and, or, not
    arguments: list[FilterDirective]
```

### Visitor 翻译器

不同向量存储有不同的过滤语法，Visitor 将统一的 StructuredQuery 翻译为各存储的原生格式：

```python
class Visitor:
    def visit_comparison(self, comparison: Comparison) -> Any: ...
    def visit_operation(self, operation: Operation) -> Any: ...
    def visit_structured_query(self, structured_query: StructuredQuery) -> tuple: ...
```

**同一过滤条件在不同存储中的翻译结果：**

```python
# 条件: year > 2020 AND genre == "科幻"

# Chroma:  {"$and": [{"year": {"$gt": 2020}}, {"genre": {"$eq": "科幻"}}]}
# Milvus:  'year > 2020 and genre == "科幻"'
# Elasticsearch: {"bool": {"must": [{"range": {"year": {"gt": 2020}}}, {"term": {"genre": "科幻"}}]}}
# Qdrant:  Filter(must=[FieldCondition(key="year", range=Range(gt=2020)), ...])
```

### 支持的向量存储（20+）

Chroma, Pinecone, Milvus, Qdrant, PGVector, Elasticsearch, Weaviate, Neo4j, AstraDB, MongoDB Atlas, Redis, Supabase 等。你写一次 `AttributeInfo` 定义，SelfQueryRetriever 自动适配任何存储的过滤语法。

---

## SelfQueryRetriever 源码解析

### 核心属性

```python
class SelfQueryRetriever(BaseRetriever):
    vectorstore: VectorStore                          # 要查询的向量存储
    query_constructor: Runnable[dict, StructuredQuery]  # LLM 链：自然语言 → StructuredQuery
    search_type: str = "similarity"                   # 搜索方法名
    structured_query_translator: Visitor              # StructuredQuery → 原生过滤语法
    use_original_query: bool = False                  # 是否忽略 LLM 重写的查询文本
```

### 核心检索流程

```python
def _get_relevant_documents(self, query, *, run_manager):
    # 第一步：LLM 解析自然语言 → StructuredQuery
    structured_query = self.query_constructor.invoke(
        {"query": query}, config={"callbacks": run_manager.get_child()}
    )

    # 第二步：Visitor 翻译为向量存储原生格式
    new_query, search_kwargs = self._prepare_query(query, structured_query)

    # 第三步：在向量存储上执行搜索
    docs = self._get_docs_with_query(new_query, search_kwargs)
    return docs

def _prepare_query(self, query, structured_query):
    new_query, new_kwargs = self.structured_query_translator.visit_structured_query(
        structured_query
    )
    if self.use_original_query:
        new_query = query  # 忽略 LLM 重写，使用用户原始查询
    return new_query, {**self.search_kwargs, **new_kwargs}
```

### from_llm 工厂方法

```python
@classmethod
def from_llm(cls, llm, vectorstore, document_content_description,
             metadata_field_info, structured_query_translator=None, **kwargs):
    # 自动检测向量存储对应的 Visitor
    if structured_query_translator is None:
        structured_query_translator = _get_builtin_translator(vectorstore)

    # 构建 query_constructor（LLM 链 = Prompt + OutputParser）
    query_constructor = load_query_constructor_runnable(
        llm, document_content_description, metadata_field_info,
        allowed_comparators=structured_query_translator.allowed_comparators,
        allowed_operators=structured_query_translator.allowed_operators,
    )

    return cls(vectorstore=vectorstore, query_constructor=query_constructor,
               structured_query_translator=structured_query_translator, **kwargs)
```

---

## 使用方式（完整代码）

```python
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document

# 1. 准备带元数据的文档
docs = [
    Document(page_content="一部关于人工智能觉醒的科幻电影。",
             metadata={"title": "机械姬", "year": 2014, "genre": "科幻", "rating": 7.7}),
    Document(page_content="在虚拟现实世界中，人类与机器的界限变得模糊。",
             metadata={"title": "黑客帝国", "year": 1999, "genre": "科幻", "rating": 8.7}),
    Document(page_content="穿越时空拯救地球的史诗级科幻巨作。",
             metadata={"title": "星际穿越", "year": 2014, "genre": "科幻", "rating": 9.4}),
    Document(page_content="一部温馨的家庭喜剧，讲述了三代人的故事。",
             metadata={"title": "你好李焕英", "year": 2021, "genre": "喜剧", "rating": 7.8}),
]

vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings(), collection_name="movies")

# 2. 定义元数据字段信息（告诉 LLM 有哪些字段可以过滤）
metadata_field_info = [
    AttributeInfo(name="title", description="电影的标题", type="string"),
    AttributeInfo(name="year", description="电影的上映年份", type="integer"),
    AttributeInfo(name="genre", description="电影类型，如 '科幻'、'喜剧'", type="string"),
    AttributeInfo(name="rating", description="电影评分，范围 0-10", type="float"),
]

# 3. 创建 SelfQueryRetriever
retriever = SelfQueryRetriever.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    vectorstore=vectorstore,
    document_content_description="电影的简介和描述",
    metadata_field_info=metadata_field_info,
)

# 4. 自然语言查询 — LLM 自动提取过滤条件
results = retriever.invoke("推荐评分 8 分以上的科幻电影")
for doc in results:
    print(f"{doc.metadata['title']} ({doc.metadata['year']}) - 评分: {doc.metadata['rating']}")
# 黑客帝国 (1999) - 评分: 8.7
# 星际穿越 (2014) - 评分: 9.4
```

---

## AttributeInfo 定义

```python
class AttributeInfo(BaseModel):
    name: str           # 元数据字段名（必须与文档 metadata 的 key 一致）
    description: str    # 字段描述（LLM 用这个理解字段含义）
    type: str           # 字段类型
```

**支持的类型：**

| type 值 | 支持的比较操作 | 示例字段 |
|---------|---------------|---------|
| `"string"` | eq, ne, contain, like, in, nin | 标题、作者 |
| `"integer"` | eq, ne, gt, gte, lt, lte | 年份、页数 |
| `"float"` | eq, ne, gt, gte, lt, lte | 评分、价格 |
| `"boolean"` | eq, ne | 是否公开 |
| `"date"` | eq, ne, gt, gte, lt, lte | 创建日期 |
| `"list[string]"` | contain, in | 标签列表 |

**定义技巧：** description 要清晰具体，帮助 LLM 准确理解。比如 `"电影上映年份，'最近'通常指 2023 年之后"` 比 `"年份"` 好得多。

### use_original_query 参数

```python
# 默认行为：LLM 可能重写查询文本
# 用户输入: "推荐 2020 年后的科幻片"
# LLM 重写: query="科幻电影推荐"  ← LLM 可能改变语义

# 设置 use_original_query=True：保留用户原始查询
retriever = SelfQueryRetriever.from_llm(
    ...,
    use_original_query=True,  # 语义搜索部分使用用户原始输入
)
# 实际搜索: query="推荐 2020 年后的科幻片"  ← 保持原样
# 过滤条件: year > 2020, genre == "科幻"    ← 仍然由 LLM 提取
```

---

## 集成到 LCEL 管道

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def format_docs(docs):
    return "\n\n".join(
        f"[{doc.metadata.get('title', '未知')}] {doc.page_content}"
        for doc in docs
    )

prompt = ChatPromptTemplate.from_template("""
你是一个电影推荐助手。根据以下电影信息回答用户的问题。

电影信息：
{context}

用户问题：{question}

回答：
""")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | ChatOpenAI(model="gpt-4o-mini")
)

answer = rag_chain.invoke("有没有评分 9 分以上的科幻电影？")
print(answer.content)
```

---

## 与其他 Retriever 的对比

| 维度 | SelfQueryRetriever | EnsembleRetriever | ParentDocumentRetriever |
|------|-------------------|-------------------|------------------------|
| 核心能力 | 语义搜索 + 元数据过滤 | 多路检索融合 | 小块检索、大块返回 |
| 是否需要 LLM | 是（每次查询） | 否 | 否 |
| 延迟 | 较高（LLM 调用） | 较低 | 较低 |
| 适用场景 | 结构化数据（有丰富元数据） | 非结构化文本 | 长文档上下文 |
| 精确过滤 | 支持（数值比较、范围） | 不支持 | 不支持 |

---

## 局限性与注意事项

**1. 每次查询都调用 LLM。** 额外 0.5-2 秒延迟和 Token 成本。不适合高频低延迟场景。对简单查询可以直接用普通 Retriever，只在包含明显过滤条件时才用 SelfQueryRetriever。

**2. LLM 解析不稳定。** "最近的科幻电影"中"最近"可能被解析为 year > 2023 或 year > 2020，不同 LLM 理解不同。通过 AttributeInfo 的 description 给出明确指引来缓解。

**3. 元数据字段必须预定义。** 只能过滤 `metadata_field_info` 中定义的字段。建议只暴露用户可能查询的字段，太多字段会让 LLM 困惑。

**4. 向量存储必须支持过滤。** 不是所有向量存储都有对应的 Visitor 翻译器。主流存储（Chroma、Pinecone、Milvus、Qdrant 等）都已支持。

---

## 核心设计模式总结

| 模式 | 体现 |
|------|------|
| 访问者模式 | Visitor 遍历 StructuredQuery，生成各存储的原生过滤语法 |
| 工厂方法 | `from_llm` 封装 LLM + Visitor + VectorStore 的复杂连接 |
| 策略模式 | `search_type` 在运行时选择搜索方法 |
| 职责链 | query_constructor → StructuredQuery → Visitor → vectorstore |

---

**版本**：v1.0
**最后更新**：2026-02-27
**数据来源**：LangChain 源码分析（`self_query/base.py`）+ Context7 官方文档
