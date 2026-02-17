# 核心概念 7: Query Decomposition

## 一句话定义

**将复杂查询拆分为多个简单子查询,分别检索后综合答案,是RAG系统处理复杂问题的核心优化技术。**

**RAG应用:** Query Decomposition是RAG系统的必备技术,通过将"比较Python和JavaScript的异步编程差异"拆分为"Python异步编程"和"JavaScript异步编程"两个子查询,显著提升检索准确率和答案完整性。

---

## 为什么重要?

### 问题场景

```python
# 场景:复杂的比较类查询
from openai import OpenAI
import chromadb

client = OpenAI()
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("docs")

# 添加文档
docs = [
    "Python使用async/await语法实现异步编程,基于事件循环",
    "JavaScript使用Promise和async/await实现异步,基于事件循环",
    "Python的类型系统支持静态类型检查(通过type hints)",
    "JavaScript是动态类型语言,TypeScript提供静态类型"
]
collection.add(documents=docs, ids=[f"doc{i}" for i in range(len(docs))])

# ❌ 直接查询:检索效果差
complex_query = "比较Python和JavaScript在异步编程和类型系统方面的差异"

results = collection.query(query_texts=[complex_query], n_results=3)
retrieved_docs = results['documents'][0]

print("检索到的文档:")
for doc in retrieved_docs:
    print(f"- {doc}")

# 可能输出:
# - Python使用async/await语法实现异步编程
# - JavaScript使用Promise和async/await实现异步
# - Python的类型系统支持静态类型检查
# 问题:可能遗漏JavaScript的类型系统信息
```

### 解决方案

```python
# ✅ Query Decomposition:拆分查询
def decompose_query(complex_query: str) -> List[str]:
    """拆分复杂查询"""
    prompt = f"""
将以下复杂查询拆分为多个简单子查询:

查询:{complex_query}

拆分原则:
1. 每个子查询只关注一个主题
2. 子查询应该独立且完整
3. 覆盖原查询的所有方面

子查询列表:
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # 解析子查询
    content = response.choices[0].message.content
    sub_queries = [line.strip('- ').strip() 
                   for line in content.split('\n') 
                   if line.strip().startswith('-')]
    return sub_queries

# 拆分查询
sub_queries = decompose_query(complex_query)
print("子查询:")
for sq in sub_queries:
    print(f"- {sq}")

# 输出:
# - Python的异步编程机制
# - JavaScript的异步编程机制
# - Python的类型系统
# - JavaScript的类型系统

# 分别检索
all_docs = []
for sub_query in sub_queries:
    results = collection.query(query_texts=[sub_query], n_results=2)
    all_docs.extend(results['documents'][0])

# 去重
unique_docs = list(set(all_docs))

print("\n检索到的文档(去重后):")
for doc in unique_docs:
    print(f"- {doc}")

# 综合答案
answer_prompt = f"""
基于以下文档回答问题:

文档:
{chr(10).join(f"{i+1}. {doc}" for i, doc in enumerate(unique_docs))}

问题:{complex_query}

请综合所有信息给出完整答案:
"""

answer = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": answer_prompt}]
)

print("\n最终答案:")
print(answer.choices[0].message.content)
```

**性能提升:**

| 指标 | 直接查询 | Query Decomposition | 提升 |
|------|---------|-------------------|------|
| 信息覆盖率 | 65% | 92% | +42% |
| 答案完整性 | 70% | 95% | +36% |
| 检索准确率 | 72% | 89% | +24% |

**来源:** [NVIDIA RAG Blueprint - Query Decomposition](https://docs.nvidia.com/rag/2.3.0/query_decomposition.html)

---

## 核心原理

### 原理1:复杂度降维

**定义:** 将高维复杂查询降维为多个低维简单查询。

**数学模型:**

```
复杂查询: Q_complex = {topic1, topic2, ..., topicN}
拆分后: Q1 = {topic1}, Q2 = {topic2}, ..., QN = {topicN}

检索效果:
P(相关文档|Q_complex) < P(相关文档|Q1) ∩ P(相关文档|Q2) ∩ ...
```

**示例:**

```python
# 复杂查询(3个维度)
Q_complex = "比较Python、JavaScript、Go在性能、语法、生态方面的差异"
# 维度: 3种语言 × 3个方面 = 9个子主题

# 拆分策略1:按语言拆分
sub_queries_1 = [
    "Python的性能、语法、生态",
    "JavaScript的性能、语法、生态",
    "Go的性能、语法、生态"
]

# 拆分策略2:按方面拆分
sub_queries_2 = [
    "Python、JavaScript、Go的性能比较",
    "Python、JavaScript、Go的语法比较",
    "Python、JavaScript、Go的生态比较"
]

# 拆分策略3:完全拆分(最细粒度)
sub_queries_3 = [
    "Python的性能", "JavaScript的性能", "Go的性能",
    "Python的语法", "JavaScript的语法", "Go的语法",
    "Python的生态", "JavaScript的生态", "Go的生态"
]
```

**来源:** [Haystack Advanced RAG - Query Decomposition](https://haystack.deepset.ai/cookbook/query_decomposition)

---

### 原理2:并行检索

**定义:** 子查询可以并行检索,提升效率。

**并行vs串行:**

```python
# 串行检索(慢)
def serial_retrieval(sub_queries):
    results = []
    for query in sub_queries:
        docs = retriever.search(query)
        results.append(docs)
    return results
# 时间: O(n * t) where n=子查询数, t=单次检索时间

# 并行检索(快)
import concurrent.futures

def parallel_retrieval(sub_queries):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(retriever.search, q) for q in sub_queries]
        results = [f.result() for f in futures]
    return results
# 时间: O(t) 理论上与单次检索相同
```

**性能对比:**

```python
# 实验:5个子查询
sub_queries = ["query1", "query2", "query3", "query4", "query5"]

# 串行: 5 * 200ms = 1000ms
# 并行: max(200ms) = 200ms
# 提速: 5倍
```

---

### 原理3:答案综合

**定义:** 将多个子答案综合为完整答案。

**综合策略:**

```python
# 策略1:简单拼接
def simple_concat(sub_answers):
    return "\n\n".join(sub_answers)

# 策略2:LLM综合
def llm_synthesis(sub_answers, original_query):
    prompt = f"""
原始问题:{original_query}

子答案:
{chr(10).join(f"{i+1}. {ans}" for i, ans in enumerate(sub_answers))}

请综合以上信息,给出完整、连贯的答案:
"""
    return llm.generate(prompt)

# 策略3:结构化综合
def structured_synthesis(sub_answers, structure):
    """按预定义结构组织答案"""
    result = {}
    for key, sub_answer in zip(structure.keys(), sub_answers):
        result[key] = sub_answer
    return result
```

---

## 手写实现

### 从零实现 Query Decomposition

```python
"""
Query Decomposition Implementation
功能:复杂查询拆分与综合
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from openai import OpenAI
import concurrent.futures

@dataclass
class SubQuery:
    """子查询"""
    query: str
    retrieved_docs: List[str] = None
    answer: Optional[str] = None

class QueryDecomposer:
    """查询分解器"""
    
    def __init__(self, client: OpenAI, retriever):
        self.client = client
        self.retriever = retriever
    
    def decompose(
        self,
        complex_query: str,
        strategy: str = "auto",
        model: str = "gpt-4o-mini"
    ) -> List[SubQuery]:
        """
        拆分复杂查询
        
        Args:
            complex_query: 复杂查询
            strategy: 拆分策略("auto", "by_topic", "by_aspect")
            model: 使用的模型
        """
        if strategy == "auto":
            prompt = f"""
将以下复杂查询拆分为多个简单子查询:

查询:{complex_query}

拆分原则:
1. 每个子查询只关注一个主题
2. 子查询应该独立且完整
3. 覆盖原查询的所有方面
4. 每行一个子查询,以"-"开头

子查询列表:
"""
        elif strategy == "by_topic":
            prompt = f"""
将查询按主题拆分:

查询:{complex_query}

识别所有主题,为每个主题生成一个子查询。
每行一个子查询,以"-"开头:
"""
        elif strategy == "by_aspect":
            prompt = f"""
将查询按方面拆分:

查询:{complex_query}

识别所有方面(如:性能、语法、生态等),为每个方面生成一个子查询。
每行一个子查询,以"-"开头:
"""
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # 解析子查询
        content = response.choices[0].message.content
        sub_query_texts = [
            line.strip('- ').strip() 
            for line in content.split('\n') 
            if line.strip() and line.strip().startswith('-')
        ]
        
        return [SubQuery(query=q) for q in sub_query_texts]
    
    def retrieve_parallel(
        self,
        sub_queries: List[SubQuery],
        top_k: int = 3
    ) -> List[SubQuery]:
        """并行检索所有子查询"""
        def retrieve_one(sub_query: SubQuery):
            docs = self.retriever.search(sub_query.query, top_k=top_k)
            sub_query.retrieved_docs = docs
            return sub_query
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(retrieve_one, sq) for sq in sub_queries]
            results = [f.result() for f in futures]
        
        return results
    
    def generate_sub_answers(
        self,
        sub_queries: List[SubQuery],
        model: str = "gpt-4o-mini"
    ) -> List[SubQuery]:
        """为每个子查询生成答案"""
        for sub_query in sub_queries:
            if not sub_query.retrieved_docs:
                continue
            
            prompt = f"""
文档:
{chr(10).join(f"{i+1}. {doc}" for i, doc in enumerate(sub_query.retrieved_docs))}

问题:{sub_query.query}

答案:
"""
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            sub_query.answer = response.choices[0].message.content
        
        return sub_queries
    
    def synthesize(
        self,
        original_query: str,
        sub_queries: List[SubQuery],
        model: str = "gpt-4o-mini"
    ) -> str:
        """综合子答案"""
        # 收集所有文档(去重)
        all_docs = []
        for sq in sub_queries:
            if sq.retrieved_docs:
                all_docs.extend(sq.retrieved_docs)
        unique_docs = list(set(all_docs))
        
        # 收集所有子答案
        sub_answers = [sq.answer for sq in sub_queries if sq.answer]
        
        prompt = f"""
原始问题:{original_query}

子问题和答案:
{chr(10).join(f"{i+1}. 问题:{sq.query}\n   答案:{sq.answer}" 
             for i, sq in enumerate(sub_queries) if sq.answer)}

参考文档:
{chr(10).join(f"{i+1}. {doc}" for i, doc in enumerate(unique_docs))}

请综合以上信息,给出完整、连贯、结构化的答案:
"""
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    def process(
        self,
        complex_query: str,
        strategy: str = "auto",
        top_k: int = 3,
        model: str = "gpt-4o-mini"
    ) -> Dict:
        """完整的Query Decomposition流程"""
        # 1. 拆分查询
        sub_queries = self.decompose(complex_query, strategy, model)
        
        # 2. 并行检索
        sub_queries = self.retrieve_parallel(sub_queries, top_k)
        
        # 3. 生成子答案
        sub_queries = self.generate_sub_answers(sub_queries, model)
        
        # 4. 综合答案
        final_answer = self.synthesize(complex_query, sub_queries, model)
        
        return {
            "original_query": complex_query,
            "sub_queries": [
                {
                    "query": sq.query,
                    "docs": sq.retrieved_docs,
                    "answer": sq.answer
                }
                for sq in sub_queries
            ],
            "final_answer": final_answer
        }


# 使用示例
if __name__ == "__main__":
    from dotenv import load_dotenv
    import chromadb
    
    load_dotenv()
    
    client = OpenAI()
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("docs")
    
    # 添加文档
    docs = [
        "Python使用async/await实现异步编程,基于asyncio事件循环",
        "JavaScript使用Promise和async/await实现异步,基于事件循环",
        "Python支持静态类型检查(type hints),但运行时是动态类型",
        "JavaScript是动态类型,TypeScript提供静态类型系统",
        "Python性能较慢,但有NumPy等优化库",
        "JavaScript在V8引擎上性能优秀,Node.js适合I/O密集型任务"
    ]
    collection.add(documents=docs, ids=[f"doc{i}" for i in range(len(docs))])
    
    # 创建简单的检索器
    class SimpleRetriever:
        def search(self, query, top_k=3):
            results = collection.query(query_texts=[query], n_results=top_k)
            return results['documents'][0]
    
    retriever = SimpleRetriever()
    decomposer = QueryDecomposer(client, retriever)
    
    # 测试
    complex_query = "比较Python和JavaScript在异步编程和类型系统方面的差异"
    
    result = decomposer.process(complex_query, strategy="auto", top_k=2)
    
    print(f"原始查询:{result['original_query']}\n")
    print("子查询:")
    for i, sq in enumerate(result['sub_queries'], 1):
        print(f"\n{i}. {sq['query']}")
        print(f"   检索文档:{sq['docs']}")
        print(f"   子答案:{sq['answer'][:100]}...")
    
    print(f"\n最终答案:\n{result['final_answer']}")
```

---

## RAG 应用场景

### 场景1:比较类查询

```python
# 问题:"比较A和B的差异"
query = "比较Python和JavaScript的异步编程差异"

# 拆分
sub_queries = [
    "Python的异步编程机制",
    "JavaScript的异步编程机制"
]

# 分别检索后综合
```

### 场景2:多维度查询

```python
# 问题:"从多个维度分析X"
query = "从性能、语法、生态三个维度分析Python"

# 拆分
sub_queries = [
    "Python的性能特点",
    "Python的语法特点",
    "Python的生态系统"
]
```

### 场景3:时间序列查询

```python
# 问题:"X的发展历程"
query = "Python从1991年到现在的发展历程"

# 拆分
sub_queries = [
    "Python 1991-2000年的发展",
    "Python 2000-2010年的发展",
    "Python 2010-2020年的发展",
    "Python 2020年至今的发展"
]
```

---

## 最佳实践

### 1. 拆分粒度

```python
# ✅ 好:适中的粒度
query = "比较Python和JavaScript"
sub_queries = [
    "Python的特点",
    "JavaScript的特点"
]

# ❌ 坏:过细
sub_queries = [
    "Python的语法",
    "Python的性能",
    "Python的生态",
    "JavaScript的语法",
    "JavaScript的性能",
    "JavaScript的生态"
]
# 问题:子查询太多,增加成本
```

### 2. 去重策略

```python
# 检索结果去重
def deduplicate_docs(docs: List[str]) -> List[str]:
    seen = set()
    unique = []
    for doc in docs:
        if doc not in seen:
            seen.add(doc)
            unique.append(doc)
    return unique
```

### 3. 并行控制

```python
# 限制并发数
max_workers = min(len(sub_queries), 5)  # 最多5个并发
```

---

## 参考资源

- [NVIDIA RAG Blueprint](https://docs.nvidia.com/rag/2.3.0/query_decomposition.html)
- [Haystack Query Decomposition](https://haystack.deepset.ai/cookbook/query_decomposition)
- [Stack AI RAG Guide](https://www.stack-ai.com/blog/prompt-engineering-for-rag-pipelines-the-complete-guide-to-prompt-engineering-for-retrieval-augmented-generation)
