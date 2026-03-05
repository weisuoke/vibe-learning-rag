# Retriever高级策略 - 实战代码 场景3：MultiQuery 查询扩展

> **用 LLM 把一个问题改写成多个变体，分别检索后合并去重，显著提高召回率**

---

## 场景描述

用户的查询往往只表达了一种措辞，而相关文档可能用完全不同的词汇描述同一件事。MultiQueryRetriever 利用 LLM 自动生成多个查询变体，对每个变体分别检索，最后合并去重，解决"一次查询只能命中一种表述"的问题。

**典型场景：**
- 用户问"Python 怎么处理并发"，但文档里写的是"多线程"、"asyncio"、"协程"
- 用户问"如何提升 RAG 效果"，但文档里分别讲了"检索优化"、"Prompt 调优"、"重排序"

---

## 环境准备

```bash
uv add langchain langchain-openai langchain-chroma chromadb python-dotenv
```

```bash
# .env 文件
OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_BASE_URL=https://your-proxy.com/v1
```

---

## 完整代码

### Step 1: 准备向量存储和基础检索器

```python
"""
MultiQueryRetriever 实战
演示：LLM 生成多个查询变体 → 分别检索 → 合并去重
"""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# ── 1. 准备测试文档 ──
documents = [
    Document(page_content="Python 的 asyncio 模块提供了事件循环和协程支持，是异步编程的核心。",
             metadata={"topic": "async"}),
    Document(page_content="多线程适合 I/O 密集型任务，但受 GIL 限制无法真正并行计算。",
             metadata={"topic": "threading"}),
    Document(page_content="multiprocessing 模块通过子进程绕过 GIL，实现 CPU 密集型任务的并行。",
             metadata={"topic": "multiprocessing"}),
    Document(page_content="concurrent.futures 提供了 ThreadPoolExecutor 和 ProcessPoolExecutor 的统一接口。",
             metadata={"topic": "concurrent"}),
    Document(page_content="RAG 系统通过检索外部知识来增强大模型的生成能力，减少幻觉。",
             metadata={"topic": "rag"}),
    Document(page_content="向量数据库使用 Embedding 将文本转为高维向量，支持语义相似度搜索。",
             metadata={"topic": "vector_db"}),
    Document(page_content="Prompt Engineering 通过精心设计提示词来引导 LLM 输出更准确的结果。",
             metadata={"topic": "prompt"}),
    Document(page_content="文本分块策略直接影响检索质量，常见方法有固定大小、递归字符和语义分块。",
             metadata={"topic": "chunking"}),
]

# ── 2. 创建向量存储 ──
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings(),
    collection_name="multi_query_demo",
)

# ── 3. 基础检索器（对照组）──
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 测试基础检索
print("=" * 60)
print("【基础检索器】单一查询")
print("=" * 60)
query = "Python 怎么处理并发？"
base_docs = base_retriever.invoke(query)
print(f"查询: {query}")
print(f"返回 {len(base_docs)} 个文档:")
for i, doc in enumerate(base_docs, 1):
    print(f"  {i}. {doc.page_content[:50]}...")
```

### Step 2: 创建 MultiQueryRetriever

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

# ── 使用工厂方法创建 ──
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    # include_original=True,  # 是否保留原始查询（默认 False）
)

print("\n" + "=" * 60)
print("【MultiQueryRetriever】多查询扩展")
print("=" * 60)
docs = multi_query_retriever.invoke(query)
print(f"查询: {query}")
print(f"返回 {len(docs)} 个文档（去重后）:")
for i, doc in enumerate(docs, 1):
    print(f"  {i}. {doc.page_content[:50]}...")
```

### Step 3: 测试查询扩展效果

```python
import logging

# ── 开启日志查看生成的查询变体 ──
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.DEBUG)

print("\n" + "=" * 60)
print("【查看生成的查询变体】")
print("=" * 60)

# 重新创建（开启 include_original 对比）
multi_query_retriever_v2 = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    include_original=True,  # 包含原始查询
)

query = "如何提升 RAG 系统的检索效果？"
docs = multi_query_retriever_v2.invoke(query)
print(f"\n查询: {query}")
print(f"返回 {len(docs)} 个文档（去重后）:")
for i, doc in enumerate(docs, 1):
    print(f"  {i}. [{doc.metadata.get('topic', '?')}] {doc.page_content[:45]}...")

# ── 对比：基础检索 vs MultiQuery ──
print("\n" + "-" * 40)
print("对比：基础检索器")
base_docs = base_retriever.invoke(query)
print(f"基础检索返回 {len(base_docs)} 个文档:")
for i, doc in enumerate(base_docs, 1):
    print(f"  {i}. [{doc.metadata.get('topic', '?')}] {doc.page_content[:45]}...")

# 关闭 debug 日志
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.WARNING)
```

### Step 4: 自定义提示词

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser

# ── 自定义输出解析器 ──
class LineListOutputParser(BaseOutputParser[list[str]]):
    """按换行符分割 LLM 输出为查询列表。"""

    def parse(self, text: str) -> list[str]:
        lines = text.strip().split("\n")
        return [line.strip() for line in lines if line.strip()]

# ── 自定义提示词：生成 5 个变体，面向 RAG 场景 ──
CUSTOM_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个查询改写专家。给定用户的原始问题，生成 {num_queries} 个不同角度的查询变体。"
     "每个变体应该用不同的措辞、关注不同的方面，以便检索到更多相关文档。"
     "每行输出一个查询，不要编号，不要解释。"),
    ("human", "原始问题：{question}"),
])

# ── 构建自定义链 ──
output_parser = LineListOutputParser()
llm_chain = CUSTOM_PROMPT | llm | output_parser

# ── 创建使用自定义链的 MultiQueryRetriever ──
custom_retriever = MultiQueryRetriever(
    retriever=base_retriever,
    llm_chain=llm_chain,
    include_original=True,
)

print("\n" + "=" * 60)
print("【自定义提示词】生成 5 个查询变体")
print("=" * 60)

# 先单独测试链，看看生成了什么
query = "向量数据库怎么选型？"
variants = llm_chain.invoke({"question": query, "num_queries": 5})
print(f"原始查询: {query}")
print(f"生成的变体:")
for i, v in enumerate(variants, 1):
    print(f"  {i}. {v}")

# 用自定义 retriever 检索
docs = custom_retriever.invoke(query)
print(f"\n检索到 {len(docs)} 个文档（去重后）:")
for i, doc in enumerate(docs, 1):
    print(f"  {i}. [{doc.metadata.get('topic', '?')}] {doc.page_content[:45]}...")
```

### Step 5: 集成到 LCEL RAG 管道

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── RAG 提示词 ──
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "根据以下上下文回答问题。如果上下文不足以回答，请说明。"),
    ("human", "上下文：\n{context}\n\n问题：{question}"),
])

# ── 格式化文档 ──
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{doc.metadata.get('topic', '未知')}] {doc.page_content}"
        for doc in docs
    )

# ── 构建 LCEL RAG 管道 ──
rag_chain = (
    {
        "context": multi_query_retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

print("\n" + "=" * 60)
print("【LCEL RAG 管道】MultiQuery + 生成")
print("=" * 60)

question = "Python 有哪些并发编程方案？各自适合什么场景？"
print(f"问题: {question}\n")
answer = rag_chain.invoke(question)
print(f"回答:\n{answer}")

# ── 清理 ──
vectorstore.delete_collection()
```

---

## 运行输出示例

```
============================================================
【基础检索器】单一查询
============================================================
查询: Python 怎么处理并发？
返回 3 个文档:
  1. concurrent.futures 提供了 ThreadPoolExecutor 和 Proc...
  2. 多线程适合 I/O 密集型任务，但受 GIL 限制无法真正并行计算。...
  3. Python 的 asyncio 模块提供了事件循环和协程支持，是异步编程的核心。...

============================================================
【MultiQueryRetriever】多查询扩展
============================================================
查询: Python 怎么处理并发？
返回 4 个文档（去重后）:
  1. concurrent.futures 提供了 ThreadPoolExecutor 和 Proc...
  2. 多线程适合 I/O 密集型任务，但受 GIL 限制无法真正并行计算。...
  3. Python 的 asyncio 模块提供了事件循环和协程支持，是异步编程的核心。...
  4. multiprocessing 模块通过子进程绕过 GIL，实现 CPU 密集型...

============================================================
【查看生成的查询变体】
============================================================
INFO:langchain.retrievers.multi_query:Generated queries:
  ['Python并发编程有哪些方式', 'Python异步和多线程的区别', 'Python如何实现并行处理']

查询: 如何提升 RAG 系统的检索效果？
返回 5 个文档（去重后）:          ← 比基础检索多了 2 个
  1. [rag] RAG 系统通过检索外部知识来增强大模型的生成能力...
  2. [vector_db] 向量数据库使用 Embedding 将文本转为高维向量...
  3. [chunking] 文本分块策略直接影响检索质量，常见方法有固定大小...
  4. [prompt] Prompt Engineering 通过精心设计提示词来引导...
  5. [async] Python 的 asyncio 模块提供了事件循环和协程支持...

============================================================
【自定义提示词】生成 5 个查询变体
============================================================
原始查询: 向量数据库怎么选型？
生成的变体:
  1. 主流向量数据库有哪些，各自的优缺点是什么
  2. 如何根据业务场景选择合适的向量存储方案
  3. Chroma Milvus Faiss 性能对比
  ...
检索到 4 个文档（去重后）

============================================================
【LCEL RAG 管道】MultiQuery + 生成
============================================================
问题: Python 有哪些并发编程方案？各自适合什么场景？

回答:
Python 提供了多种并发编程方案：
1. asyncio（协程）：适合高并发 I/O 密集型任务
2. 多线程（threading）：适合 I/O 密集型，但受 GIL 限制
3. 多进程（multiprocessing）：绕过 GIL，适合 CPU 密集型
4. concurrent.futures：线程池和进程池的统一高级接口
```

---

## 关键参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `retriever` | `BaseRetriever` | (必需) | 底层基础检索器 |
| `llm` | `BaseLanguageModel` | (必需，from_llm) | 用于生成查询变体的 LLM |
| `include_original` | `bool` | `False` | 是否在变体中包含原始查询 |
| `llm_chain` | `Runnable` | (自动创建) | 自定义查询生成链 |

**默认行为：**
- 生成 3 个查询变体（由默认提示词控制）
- 不包含原始查询（`include_original=False`）
- 对所有变体的检索结果做唯一并集去重

**底层检索器的 `search_kwargs` 仍然生效：**
```python
# 每个变体检索 5 个文档，最终去重后可能返回 5~15 个
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

---

## 何时使用 MultiQueryRetriever

**适合的场景：**
- 用户查询措辞单一，可能遗漏相关文档
- 知识库中同一概念有多种表述方式
- 需要提高召回率，宁可多检索也不要漏掉

**不适合的场景：**
- 对延迟敏感（每次检索都要调用 LLM 生成变体）
- LLM 调用成本受限（每次查询额外消耗 token）
- 查询已经足够明确，不需要扩展

**与其他策略的组合：**
```
MultiQuery（提高召回） → Rerank（提高精度） → LLM 生成
```

MultiQueryRetriever 负责"捞得更多"，Rerank 负责"选得更准"，两者互补。

---

**版本**：v1.0
**最后更新**：2026-02-27
**数据来源**：LangChain 源码分析 + 官方文档
