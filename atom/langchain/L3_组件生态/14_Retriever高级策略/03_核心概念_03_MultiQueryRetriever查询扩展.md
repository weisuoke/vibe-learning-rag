# Retriever高级策略 - 核心概念:MultiQueryRetriever 查询扩展

> **LLM 驱动的查询扩展**：将单个用户查询扩展为多个语义等价变体，并行检索后取并集，提高召回率

---

## 一句话定义

MultiQueryRetriever 使用 LLM 将一个查询改写为多个不同措辞的变体，分别检索后合并去重，解决"用户不会提问"的问题。

---

## 为什么需要查询扩展？

### 用户查询的天然缺陷

向量检索对查询措辞高度敏感。同一个意图，换一种说法，检索结果可能完全不同：

```
用户原始查询: "Python 怎么读文件"

可能遗漏的相关文档:
- "使用 open() 函数打开文件"        ← 措辞不同，相似度低
- "文件 I/O 操作指南"              ← 术语不同，相似度低
- "读取 CSV/JSON 数据"             ← 具体格式，相似度低
```

**核心矛盾**：用户只会用一种方式提问，但相关文档可能用多种方式描述同一件事。

### 具体场景

| 场景 | 用户查询 | 遗漏的表述 |
|------|----------|------------|
| 技术问答 | "怎么部署 Docker" | "容器化部署"、"Docker Compose 配置" |
| 法律咨询 | "租房合同纠纷" | "房屋租赁争议"、"承租人权益" |
| 医疗问答 | "头疼怎么办" | "偏头痛治疗"、"头痛缓解方法" |

查询扩展的本质：**让 LLM 帮用户把问题问全**。

---

## 查询扩展的 IR 理论基础

### 经典信息检索中的查询扩展

查询扩展（Query Expansion）是信息检索领域的经典技术，核心思想是通过增加查询词来提高召回率：

```
传统方法:
├── 同义词扩展: "car" → "car", "automobile", "vehicle"
├── 词干化: "running" → "run"
├── 伪相关反馈: 用首轮结果中的高频词扩展查询
└── 知识图谱: 利用实体关系扩展
```

### LLM 驱动 vs 传统方法

| 维度 | 传统查询扩展 | LLM 查询扩展 |
|------|-------------|-------------|
| 扩展方式 | 词级别（同义词、词干） | 句子级别（语义改写） |
| 语义理解 | 浅层（词汇匹配） | 深层（理解意图） |
| 领域适应 | 需要领域词典 | 零样本适应 |
| 扩展质量 | 可能引入噪声 | 语义保真度高 |
| 成本 | 低（规则匹配） | 高（LLM 调用） |

**为什么 LLM 更适合现代 RAG**：向量检索本身就是语义级别的，查询扩展也应该在语义级别进行。LLM 能理解查询意图，生成语义等价但措辞不同的变体，与 Embedding 模型的工作方式天然匹配。

---

## MultiQueryRetriever 源码解析

### 类结构和属性

```python
class MultiQueryRetriever(BaseRetriever):
    """使用 LLM 生成多个查询变体的检索器。"""

    retriever: BaseRetriever          # 底层检索器（必需）
    llm_chain: Runnable               # 生成查询变体的链（必需）
    verbose: bool = True              # 是否记录生成的查询
    include_original: bool = False    # 是否包含原始查询
    parser_key: Optional[str] = None  # 向后兼容字段
```

关键设计：MultiQueryRetriever 本身是一个 BaseRetriever，可以无缝替换任何使用 Retriever 的地方。

### LineListOutputParser

```python
class LineListOutputParser(BaseOutputParser[list[str]]):
    """按换行符分割 LLM 输出。"""

    def parse(self, text: str) -> list[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # 过滤空行
```

简单但关键——LLM 返回的多个查询变体用换行符分隔，这个解析器负责拆分。

### 默认提示词

```python
DEFAULT_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is
to generate 3 different versions of the given user question to retrieve
relevant documents from a vector database. By generating multiple
perspectives on the user question, your goal is to help the user
overcome some of the limitations of distance-based similarity search.
Provide these alternative questions separated by newlines.
Original question: {question}""",
)
```

**设计要点**：
- 固定生成 3 个变体（不多不少，平衡成本和效果）
- 明确告知 LLM 目的是克服距离搜索的局限性
- 要求用换行符分隔（配合 LineListOutputParser）

### generate_queries 方法

```python
def generate_queries(
    self, question: str, run_manager: CallbackManagerForRetrieverRun
) -> list[str]:
    """从原始问题生成查询变体。"""
    # 调用 LLM 链生成变体
    response = self.llm_chain.invoke(
        {"question": question},
        config={"callbacks": run_manager.get_child()},
    )

    # 处理不同返回类型
    if isinstance(response, LLMResult):
        lines = response.generations[0][0].text
    else:
        lines = response

    # 解析为列表
    if isinstance(lines, list):
        queries = lines
    else:
        queries = self.output_parser.parse(lines)

    # 可选：追加原始查询
    if self.include_original:
        queries.append(question)

    if self.verbose:
        logger.info("Generated queries: %s", queries)

    return queries
```

### retrieve_documents 方法

同步版本逐个检索，异步版本并行检索：

```python
# 同步版本 - 逐个检索
def retrieve_documents(
    self, queries: list[str], run_manager
) -> list[Document]:
    documents = []
    for query in queries:
        docs = self.retriever.invoke(
            query, config={"callbacks": run_manager.get_child()}
        )
        documents.extend(docs)
    return documents

# 异步版本 - 并行检索（性能关键）
async def aretrieve_documents(
    self, queries: list[str], run_manager
) -> list[Document]:
    tasks = [
        self.retriever.ainvoke(
            query, config={"callbacks": run_manager.get_child()}
        )
        for query in queries
    ]
    results = await asyncio.gather(*tasks)
    return [doc for docs in results for doc in docs]
```

**性能差异**：假设底层检索器每次耗时 100ms，3 个查询变体：
- 同步：300ms（串行）
- 异步：~100ms（并行）

### unique_union 去重

```python
def unique_union(self, documents: list[Document]) -> list[Document]:
    """对检索结果去重，保持顺序。"""
    return _unique_documents(documents)

# 内部实现：基于 page_content 去重
def _unique_documents(documents: list[Document]) -> list[Document]:
    seen = set()
    unique = []
    for doc in documents:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique.append(doc)
    return unique
```

去重基于 `page_content` 文本内容，不是基于文档 ID。这意味着内容完全相同的文档会被合并，但内容略有不同的文档会保留。

### 完整流程图

```
用户查询: "Python 怎么读文件"
        │
        ▼
┌─────────────────────┐
│   generate_queries   │  ← LLM 生成 3 个变体
│   (llm_chain.invoke) │
└─────────┬───────────┘
          │
          ▼
  ┌───────────────────────────────────┐
  │ 变体1: "Python 文件读取方法有哪些"    │
  │ 变体2: "如何用 Python 打开和读取文件" │
  │ 变体3: "Python open read 文件操作"  │
  │ [原始]: "Python 怎么读文件"         │  ← include_original=True 时
  └───────────┬───────────────────────┘
              │
              ▼
  ┌─────────────────────┐
  │  retrieve_documents  │  ← 每个变体分别检索
  │  (并行 or 串行)       │
  └───────────┬─────────┘
              │
              ▼
  ┌─────────────────────┐
  │    unique_union      │  ← 合并去重
  └───────────┬─────────┘
              │
              ▼
      最终文档列表（召回率↑）
```

---

## 使用方式

### 基础用法：from_llm 工厂方法

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

# 1. 准备底层检索器
vectorstore = InMemoryVectorStore(embedding=OpenAIEmbeddings())
vectorstore.add_texts([
    "Python 使用 open() 函数打开文件，支持多种模式",
    "read() 方法读取文件全部内容，readline() 逐行读取",
    "with 语句可以自动管理文件资源的打开和关闭",
    "pandas 的 read_csv() 可以高效读取 CSV 文件",
    "json 模块提供 load() 和 dumps() 处理 JSON 数据",
    "pathlib 是 Python 3 推荐的文件路径处理方式",
])
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 2. 创建 MultiQueryRetriever
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
)

# 3. 检索
docs = multi_retriever.invoke("Python 怎么读文件")
print(f"检索到 {len(docs)} 篇文档（去重后）")
for doc in docs:
    print(f"  - {doc.page_content[:50]}...")
```

`from_llm` 内部做了什么：

```python
@classmethod
def from_llm(
    cls,
    retriever: BaseRetriever,
    llm: BaseLanguageModel,
    prompt: BasePromptTemplate = DEFAULT_QUERY_PROMPT,
    parser_key: Optional[str] = None,
    include_original: bool = False,
) -> "MultiQueryRetriever":
    # 构建 LCEL 链: prompt | llm | parser
    output_parser = LineListOutputParser()
    llm_chain = prompt | llm | output_parser

    return cls(
        retriever=retriever,
        llm_chain=llm_chain,
        include_original=include_original,
    )
```

### 自定义提示词

默认提示词生成 3 个英文变体。对于中文场景或特定领域，自定义提示词效果更好：

```python
from langchain_core.prompts import PromptTemplate

# 中文优化提示词
chinese_prompt = PromptTemplate(
    input_variables=["question"],
    template="""你是一个查询改写助手。请将用户的问题改写为 3 个不同的版本，
用于从向量数据库中检索相关文档。每个版本应该：
- 保持原始问题的核心意图
- 使用不同的措辞和角度
- 覆盖可能的同义表达

每行一个改写结果，不要编号，不要额外解释。

原始问题: {question}""",
)

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    prompt=chinese_prompt,
)
```

### include_original=True 的场景

```python
# 包含原始查询 — 确保原始查询的结果不会丢失
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    include_original=True,  # 原始查询也参与检索
)
```

**何时开启**：
- 原始查询本身质量较高时（用户是领域专家）
- 担心 LLM 改写偏离原意时
- 需要确保原始查询的精确匹配结果不丢失时

**何时关闭**（默认）：
- 用户查询质量参差不齐
- 希望完全依赖 LLM 的改写能力
- 减少一次检索调用的开销

### 集成到 LCEL 管道

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# MultiQueryRetriever 本身就是 BaseRetriever
# 可以直接用在任何需要 retriever 的地方

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template(
    "根据以下上下文回答问题：\n\n{context}\n\n问题：{question}"
)

# 完整 RAG 管道
rag_chain = (
    {"context": multi_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("Python 怎么读文件")
print(answer)
```

---

## 进阶：自定义查询生成

### 控制生成查询数量

默认生成 3 个变体。如果需要更多或更少：

```python
# 生成 5 个变体（更高召回率，更高成本）
more_queries_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Generate 5 different versions of the given question
to retrieve relevant documents from a vector database.
Provide these alternative questions separated by newlines.
Original question: {question}""",
)

# 生成 2 个变体（更低成本，适合简单场景）
fewer_queries_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Generate 2 different versions of the given question
to retrieve relevant documents from a vector database.
Provide these alternative questions separated by newlines.
Original question: {question}""",
)
```

### 针对特定领域优化

```python
# 医疗领域
medical_prompt = PromptTemplate(
    input_variables=["question"],
    template="""你是医疗信息检索助手。请将患者的问题改写为 3 个版本，
分别从以下角度：
1. 使用医学术语的专业表述
2. 使用通俗易懂的日常表述
3. 从症状/治疗/预防等不同维度表述

每行一个，不要编号。
原始问题: {question}""",
)

# 法律领域
legal_prompt = PromptTemplate(
    input_variables=["question"],
    template="""你是法律信息检索助手。请将用户的法律问题改写为 3 个版本，
分别从以下角度：
1. 使用法律条文中的正式术语
2. 使用当事人视角的日常表述
3. 从相关法律关系的角度表述

每行一个，不要编号。
原始问题: {question}""",
)
```

### 完全自定义查询生成逻辑

如果需要超越简单的提示词定制，可以直接构建 llm_chain：

```python
from langchain_core.runnables import RunnableLambda

def custom_query_generator(input_dict: dict) -> list[str]:
    """自定义查询生成：结合规则和 LLM。"""
    question = input_dict["question"]
    queries = []

    # 规则1：添加同义词变体
    synonyms = {"读文件": "文件读取", "部署": "上线发布"}
    for old, new in synonyms.items():
        if old in question:
            queries.append(question.replace(old, new))

    # 规则2：LLM 生成补充变体
    # （这里简化，实际可以调用 LLM）
    queries.append(f"{question} 最佳实践")
    queries.append(f"{question} 常见方法")

    return queries

multi_retriever = MultiQueryRetriever(
    retriever=base_retriever,
    llm_chain=RunnableLambda(custom_query_generator),
)
```

---

## 局限性与注意事项

### 成本与延迟

每次检索都会触发一次 LLM 调用：

```
普通检索:  用户查询 → 向量检索 → 结果
           延迟: ~100ms, 成本: $0

MultiQuery: 用户查询 → LLM生成变体 → N次向量检索 → 合并去重 → 结果
            延迟: ~500-1000ms, 成本: ~$0.001/次 (gpt-4o-mini)
```

### 查询质量依赖 LLM

LLM 可能生成偏离原意的变体：

```
原始查询: "Apple 最新产品"
LLM 变体: "苹果的营养价值"  ← 歧义导致偏离
```

**缓解方法**：使用更好的提示词，明确领域上下文。

### 何时使用 vs 何时不使用

**适合使用**：
- 用户查询质量不稳定（C 端产品）
- 知识库文档表述多样
- 召回率比精确率更重要
- 可以接受额外的 LLM 延迟

**不适合使用**：
- 对延迟要求极高（< 200ms）
- 查询已经很精确（内部工具、结构化查询）
- 预算有限，无法承担额外 LLM 成本
- 底层检索器本身召回率已经足够

### 与其他策略的组合

MultiQueryRetriever 可以与其他高级检索策略叠加：

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

# 第一层：MultiQuery 提高召回率
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever, llm=llm
)

# 第二层：Rerank 提高精确率
compressor = FlashrankRerank(top_n=5)
final_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=multi_retriever,  # 套在 MultiQuery 外面
)

# 效果：先扩大搜索范围，再精选最相关的
docs = final_retriever.invoke("Python 怎么读文件")
```

这是生产环境中常见的组合模式：**MultiQuery（召回率）+ Rerank（精确率）= 最优检索效果**。

---

**版本**: v1.0
**最后更新**: 2026-02-27
**数据来源**: LangChain 源码分析 (`multi_query.py`) + Context7 官方文档
