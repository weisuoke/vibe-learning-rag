# 核心概念7：GraphRAG架构

> 结合图检索和生成的RAG架构 - 2026年AI Agent的核心技术

---

## 一句话定义

**GraphRAG是结合知识图谱检索和LLM生成的RAG架构，通过实体关系提取、社区检测、混合检索实现准确率提升3.4倍和搜索精度99%的突破。**

---

## 核心原理

### GraphRAG vs 传统RAG

```python
# 传统RAG
文档 → 分块 → 向量化 → 向量检索 → 生成答案

# GraphRAG
文档 → 分块 → 实体关系提取 → 知识图谱构建 → 社区检测 →
社区摘要 → 混合检索（局部+全局+向量） → 生成答案
```

**核心差异：**
- 传统RAG：基于文本相似度
- GraphRAG：基于实体关系和语义结构

---

## 完整架构

### Microsoft GraphRAG架构（2026标准）

```python
class GraphRAG:
    """完整的GraphRAG系统（基于Microsoft实现）"""

    def __init__(self, llm, embedding_model):
        self.llm = llm
        self.embedding_model = embedding_model

        # 核心组件
        self.knowledge_graph = nx.DiGraph()
        self.vector_store = {}  # 文本块向量存储
        self.communities = {}   # 社区划分
        self.community_summaries = {}  # 社区摘要

    # ===== 阶段1：索引构建 =====

    def index_documents(self, documents):
        """索引文档（完整流程）"""

        for doc in documents:
            # 1.1 文本分块
            chunks = self.chunk_document(doc)

            # 1.2 向量化存储
            for chunk in chunks:
                embedding = self.embedding_model.embed(chunk)
                self.vector_store[chunk] = embedding

            # 1.3 提取实体关系
            for chunk in chunks:
                triples = self.extract_entities_relations(chunk)

                # 1.4 构建知识图谱
                for s, p, o in triples:
                    self.knowledge_graph.add_edge(
                        s, o,
                        relation=p,
                        source=chunk
                    )

        # 1.5 社区检测
        self.detect_communities()

        # 1.6 生成社区摘要
        self.generate_community_summaries()

    def chunk_document(self, doc, chunk_size=500):
        """文本分块"""
        words = doc.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)
        return chunks

    def extract_entities_relations(self, text):
        """提取实体关系（使用LLM）"""
        prompt = f"""
从以下文本中提取实体和关系：
{text}

返回JSON格式：
{{"triples": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}
"""
        response = self.llm.generate(prompt)
        # 解析JSON并返回三元组
        return self._parse_triples(response)

    def detect_communities(self):
        """社区检测（Louvain算法）"""
        import community as community_louvain

        # 转换为无向图
        undirected = self.knowledge_graph.to_undirected()

        # 社区检测
        partition = community_louvain.best_partition(undirected)

        # 存储社区
        for node, comm_id in partition.items():
            if comm_id not in self.communities:
                self.communities[comm_id] = []
            self.communities[comm_id].append(node)

    def generate_community_summaries(self):
        """生成社区摘要"""
        for comm_id, nodes in self.communities.items():
            # 获取社区内的所有关系
            edges = []
            for u in nodes:
                for v in self.knowledge_graph.neighbors(u):
                    if v in nodes:
                        relation = self.knowledge_graph[u][v]['relation']
                        edges.append((u, relation, v))

            # 使用LLM生成摘要
            prompt = f"""
总结以下实体和关系的主题：
实体：{nodes}
关系：{edges}

生成一个简洁的主题摘要（2-3句话）。
"""
            summary = self.llm.generate(prompt)
            self.community_summaries[comm_id] = summary

    # ===== 阶段2：查询处理 =====

    def query(self, question, mode='hybrid'):
        """完整查询流程"""

        # 2.1 检索上下文
        context = self.retrieve(question, mode)

        # 2.2 生成答案
        answer = self.generate(question, context)

        return answer

    def retrieve(self, query, mode='hybrid'):
        """混合检索策略"""

        if mode == 'local':
            return self._local_retrieval(query)
        elif mode == 'global':
            return self._global_retrieval(query)
        elif mode == 'hybrid':
            local = self._local_retrieval(query)
            global_ctx = self._global_retrieval(query)
            vector = self._vector_retrieval(query)
            return self._merge_contexts(local, global_ctx, vector)

    def _local_retrieval(self, query):
        """局部检索：基于实体的邻居"""
        # 提取查询中的实体
        entities = self._extract_entities(query)

        context = []
        for entity in entities:
            if entity in self.knowledge_graph:
                # 获取1-2跳邻居
                neighbors = list(self.knowledge_graph.neighbors(entity))
                for neighbor in neighbors:
                    relation = self.knowledge_graph[entity][neighbor]['relation']
                    context.append(f"{entity} {relation} {neighbor}")

        return context

    def _global_retrieval(self, query):
        """全局检索：基于社区摘要"""
        query_embedding = self.embedding_model.embed(query)

        # 计算查询与社区摘要的相似度
        similarities = []
        for comm_id, summary in self.community_summaries.items():
            summary_embedding = self.embedding_model.embed(summary)
            similarity = self._cosine_similarity(query_embedding, summary_embedding)
            similarities.append((comm_id, similarity))

        # 返回最相关的社区摘要
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_communities = similarities[:3]

        context = [
            self.community_summaries[comm_id]
            for comm_id, _ in top_communities
        ]

        return context

    def _vector_retrieval(self, query, top_k=5):
        """向量检索：基于文本相似度"""
        query_embedding = self.embedding_model.embed(query)

        # 计算与所有文本块的相似度
        similarities = []
        for chunk, embedding in self.vector_store.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append((chunk, similarity))

        # 返回最相似的文本块
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in similarities[:top_k]]

    def generate(self, query, context):
        """生成答案"""
        prompt = f"""
基于以下知识回答问题：

知识：
{chr(10).join(str(c) for c in context)}

问题：{query}

要求：
1. 基于提供的知识回答
2. 如果知识不足，明确说明
3. 引用具体的实体和关系

答案：
"""
        return self.llm.generate(prompt)

    # ===== 辅助方法 =====

    def _cosine_similarity(self, vec1, vec2):
        """余弦相似度"""
        import numpy as np
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _merge_contexts(self, local, global_ctx, vector):
        """合并上下文"""
        # 去重并排序
        all_context = list(set(local + global_ctx + vector))
        return all_context[:10]  # 限制上下文长度
```

---

## 核心组件详解

### 组件1：实体关系提取

```python
def extract_with_schema(text, schema):
    """带Schema的实体关系提取"""

    prompt = f"""
从文本中提取实体和关系。

Schema定义：
实体类型：{schema['entity_types']}
关系类型：{schema['relation_types']}

文本：{text}

只提取符合Schema的三元组。
"""
    # 使用LLM提取
    return llm.generate(prompt)

# Schema示例
schema = {
    "entity_types": ["Person", "Organization", "Location"],
    "relation_types": ["worksFor", "locatedIn", "foundedBy"]
}
```

### 组件2：社区检测

```python
def detect_communities_leiden(graph):
    """使用Leiden算法（比Louvain更好）"""
    import leidenalg as la
    import igraph as ig

    # 转换为igraph格式
    ig_graph = ig.Graph.from_networkx(graph)

    # Leiden算法
    partition = la.find_partition(
        ig_graph,
        la.ModularityVertexPartition
    )

    return partition
```

### 组件3：社区摘要

```python
def generate_hierarchical_summary(community, graph, llm):
    """层次化社区摘要"""

    # 1. 实体级摘要
    entity_summaries = []
    for entity in community:
        neighbors = list(graph.neighbors(entity))
        summary = f"{entity}: 连接到 {', '.join(neighbors[:5])}"
        entity_summaries.append(summary)

    # 2. 社区级摘要
    prompt = f"""
总结以下实体群的主题：
{chr(10).join(entity_summaries)}

生成一个简洁的主题摘要。
"""
    community_summary = llm.generate(prompt)

    return community_summary
```

---

## 性能提升数据

### 2026年最新数据

| 指标 | 传统RAG | GraphRAG | 提升 | 来源 |
|------|---------|----------|------|------|
| **搜索精度** | 70% | 99% | +41% | Squirro 2026 |
| **准确率** | 30% | 102% | 3.4倍 | Medium 2026 |
| **幻觉率** | 高 | 显著降低 | - | IntelligentCIO 2026 |
| **可解释性** | 低 | 高 | - | Microsoft Research |

### 实际案例

```python
# 复杂问答示例
question = "张三的老板的出生地在哪里？"

# 传统RAG
# 问题：无法理解"老板"需要多跳推理
# 结果：可能返回不相关的文档

# GraphRAG
# 推理链：张三 → 工作于 → 阿里巴巴 → 创始人 → 马云 → 出生于 → 杭州
# 结果：准确回答"杭州"
```

---

## 在AI Agent中的应用

### 应用1：复杂问答系统

```python
class ComplexQASystem:
    """基于GraphRAG的复杂问答"""

    def __init__(self, graphrag):
        self.graphrag = graphrag

    def answer_multi_hop(self, question):
        """多跳问答"""
        # 使用局部检索（精确的实体关系）
        answer = self.graphrag.query(question, mode='local')
        return answer

    def answer_open_ended(self, question):
        """开放式问答"""
        # 使用全局检索（宏观主题理解）
        answer = self.graphrag.query(question, mode='global')
        return answer
```

### 应用2：知识发现

```python
def discover_hidden_connections(graphrag, entity1, entity2):
    """发现隐藏的连接"""

    # 在知识图谱中查找路径
    paths = nx.all_simple_paths(
        graphrag.knowledge_graph,
        entity1,
        entity2,
        cutoff=3  # 最多3跳
    )

    # 返回所有路径
    return list(paths)

# 使用
paths = discover_hidden_connections(graphrag, "张三", "杭州")
# [['张三', '阿里巴巴', '杭州'], ['张三', '杭州']]
```

### 应用3：推理链可视化

```python
def visualize_reasoning_chain(graphrag, question, answer):
    """可视化推理链"""

    # 提取推理路径
    entities = graphrag._extract_entities(question)
    answer_entities = graphrag._extract_entities(answer)

    # 找到连接路径
    paths = []
    for start in entities:
        for end in answer_entities:
            if nx.has_path(graphrag.knowledge_graph, start, end):
                path = nx.shortest_path(graphrag.knowledge_graph, start, end)
                paths.append(path)

    # 可视化
    import matplotlib.pyplot as plt
    G = nx.DiGraph()
    for path in paths:
        for i in range(len(path) - 1):
            G.add_edge(path[i], path[i+1])

    nx.draw(G, with_labels=True)
    plt.show()
```

---

## 实现框架对比

### Microsoft GraphRAG

```python
# 官方实现
from graphrag import GraphRAG

graphrag = GraphRAG(
    llm_model="gpt-4",
    embedding_model="text-embedding-3-small"
)

# 索引
graphrag.index(documents)

# 查询
answer = graphrag.query("问题", mode="global")
```

### Neo4j GraphRAG

```python
# Neo4j实现
from neo4j_graphrag import Neo4jGraphRAG

graphrag = Neo4jGraphRAG(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)

# 构建图谱
graphrag.build_from_documents(documents)

# 查询
answer = graphrag.query("问题")
```

### LangChain + NetworkX

```python
# 自定义实现
from langchain.chains import GraphRAGChain
import networkx as nx

chain = GraphRAGChain(
    llm=llm,
    graph=nx.DiGraph(),
    retriever=custom_retriever
)

answer = chain.run("问题")
```

---

## 优化技巧

### 技巧1：增量索引

```python
def incremental_index(graphrag, new_documents):
    """增量索引（不重建整个图谱）"""

    for doc in new_documents:
        # 提取新三元组
        triples = graphrag.extract_entities_relations(doc)

        # 添加到现有图谱
        for s, p, o in triples:
            graphrag.knowledge_graph.add_edge(s, o, relation=p)

        # 只重新计算受影响的社区
        affected_nodes = set([s for s, _, _ in triples] + [o for _, _, o in triples])
        graphrag.recompute_communities(affected_nodes)
```

### 技巧2：缓存社区摘要

```python
class CachedGraphRAG(GraphRAG):
    """带缓存的GraphRAG"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.summary_cache = {}

    def generate_community_summaries(self):
        """生成社区摘要（带缓存）"""
        for comm_id, nodes in self.communities.items():
            # 计算社区哈希
            comm_hash = hash(tuple(sorted(nodes)))

            # 检查缓存
            if comm_hash in self.summary_cache:
                self.community_summaries[comm_id] = self.summary_cache[comm_hash]
            else:
                # 生成新摘要
                summary = self._generate_summary(nodes)
                self.community_summaries[comm_id] = summary
                self.summary_cache[comm_hash] = summary
```

### 技巧3：混合检索权重调优

```python
def weighted_hybrid_retrieval(graphrag, query, weights):
    """加权混合检索"""

    # 三种检索
    local = graphrag._local_retrieval(query)
    global_ctx = graphrag._global_retrieval(query)
    vector = graphrag._vector_retrieval(query)

    # 加权合并
    scored_contexts = []

    for ctx in local:
        scored_contexts.append((ctx, weights['local']))

    for ctx in global_ctx:
        scored_contexts.append((ctx, weights['global']))

    for ctx in vector:
        scored_contexts.append((ctx, weights['vector']))

    # 排序并返回
    scored_contexts.sort(key=lambda x: x[1], reverse=True)
    return [ctx for ctx, _ in scored_contexts[:10]]

# 使用
weights = {'local': 0.5, 'global': 0.3, 'vector': 0.2}
context = weighted_hybrid_retrieval(graphrag, query, weights)
```

---

## 实战练习

### 练习1：实现简化版GraphRAG

```python
class SimpleGraphRAG:
    """简化版GraphRAG（核心功能）"""

    def __init__(self, llm):
        self.llm = llm
        self.kg = nx.DiGraph()

    def index(self, documents):
        # TODO: 实现索引
        pass

    def query(self, question):
        # TODO: 实现查询
        pass
```

### 练习2：评估GraphRAG性能

```python
def evaluate_graphrag(graphrag, test_cases):
    """评估GraphRAG性能"""

    correct = 0
    for question, expected_answer in test_cases:
        answer = graphrag.query(question)
        if answer == expected_answer:
            correct += 1

    accuracy = correct / len(test_cases)
    return accuracy
```

---

## 总结

**GraphRAG核心特点：**

1. **结构化知识**：实体关系图谱
2. **多层次理解**：局部细节 + 全局主题
3. **混合检索**：向量 + 图 + 社区
4. **显著提升**：准确率3.4倍，精度99%

**在AI Agent中的应用：**
- 复杂问答和多跳推理
- 知识发现和关系挖掘
- 推理链可视化
- 减少幻觉

**关键洞察：**
- GraphRAG是2026年RAG的标准
- 社区检测是核心创新
- 混合检索策略是关键
- 实体关系提取决定质量

---

**下一步：** 学习 `03_核心概念_08_图检索与推理.md`
