# 实战代码 - GraphRAG

演示如何使用知识图谱增强RAG系统，支持实体关系提取和图谱检索。

---

## 代码说明

**演示场景：** 构建一个基于知识图谱的RAG系统

**核心功能：**
1. 实体和关系提取
2. 知识图谱构建
3. 图谱检索（全局和局部）
4. 社区检测和摘要

**技术栈：**
- OpenAI API（实体提取和生成）
- NetworkX（图谱构建）
- ChromaDB（向量存储）

---

## 完整代码

```python
"""
GraphRAG实现
演示：知识图谱增强的RAG系统
"""

import os
from typing import List, Dict, Tuple
import json
from dotenv import load_dotenv

from openai import OpenAI
import networkx as nx
import chromadb

load_dotenv()

# ===== 1. 准备测试数据 =====
print("=== 准备测试数据 ===")

documents = [
    "Python是由Guido van Rossum创建的编程语言。",
    "FastAPI是一个基于Python的Web框架。",
    "FastAPI使用Starlette作为Web服务器。",
    "FastAPI使用Pydantic进行数据验证。",
    "LangChain是一个用于构建LLM应用的框架。",
    "LangChain支持多种LLM提供商，包括OpenAI。",
    "RAG系统结合了检索和生成技术。",
    "向量数据库用于存储文档的向量表示。",
    "ChromaDB是一个开源的向量数据库。",
    "Milvus是另一个流行的向量数据库。"
]

print(f"✓ 准备了 {len(documents)} 个测试文档")

# ===== 2. 初始化组件 =====
print("\n=== 初始化组件 ===")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 知识图谱
graph = nx.DiGraph()

# 向量存储（用于实体描述）
chroma_client = chromadb.Client()
entity_collection = chroma_client.create_collection("entities")

print("✓ 组件初始化完成")

# ===== 3. 实体和关系提取 =====
print("\n=== 实体和关系提取 ===")

def extract_entities_and_relations(text: str) -> Dict:
    """使用LLM提取实体和关系"""
    prompt = f"""从以下文本中提取实体和关系。

文本：{text}

以JSON格式返回：
{{
  "entities": [
    {{"name": "实体名", "type": "类型"}},
    ...
  ],
  "relations": [
    {{"source": "实体1", "relation": "关系类型", "target": "实体2"}},
    ...
  ]
}}

只返回JSON，不要其他内容。"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return result
    except json.JSONDecodeError:
        return {"entities": [], "relations": []}

# 提取所有文档的实体和关系
all_entities = []
all_relations = []

for doc in documents:
    extracted = extract_entities_and_relations(doc)
    all_entities.extend(extracted.get("entities", []))
    all_relations.extend(extracted.get("relations", []))

print(f"✓ 提取了 {len(all_entities)} 个实体")
print(f"✓ 提取了 {len(all_relations)} 个关系")

# 显示示例
print("\n实体示例:")
for entity in all_entities[:3]:
    print(f"  - {entity['name']} ({entity['type']})")

print("\n关系示例:")
for relation in all_relations[:3]:
    print(f"  - {relation['source']} --[{relation['relation']}]--> {relation['target']}")

# ===== 4. 构建知识图谱 =====
print("\n=== 构建知识图谱 ===")

# 添加实体节点
for entity in all_entities:
    if not graph.has_node(entity["name"]):
        graph.add_node(entity["name"], type=entity["type"])

# 添加关系边
for relation in all_relations:
    graph.add_edge(
        relation["source"],
        relation["target"],
        relation=relation["relation"]
    )

print(f"✓ 图谱包含 {graph.number_of_nodes()} 个节点")
print(f"✓ 图谱包含 {graph.number_of_edges()} 条边")

# ===== 5. 社区检测 =====
print("\n=== 社区检测 ===")

# 转换为无向图进行社区检测
undirected_graph = graph.to_undirected()

# 使用Louvain算法进行社区检测
try:
    import community as community_louvain
    communities = community_louvain.best_partition(undirected_graph)
except ImportError:
    # 如果没有安装python-louvain，使用简单的连通分量
    communities = {}
    for i, component in enumerate(nx.connected_components(undirected_graph)):
        for node in component:
            communities[node] = i

# 统计社区
community_groups = {}
for node, comm_id in communities.items():
    if comm_id not in community_groups:
        community_groups[comm_id] = []
    community_groups[comm_id].append(node)

print(f"✓ 检测到 {len(community_groups)} 个社区")

for comm_id, nodes in list(community_groups.items())[:3]:
    print(f"\n社区 {comm_id}:")
    print(f"  成员: {', '.join(nodes[:5])}")

# ===== 6. 生成社区摘要 =====
print("\n=== 生成社区摘要 ===")

def generate_community_summary(nodes: List[str], graph: nx.DiGraph) -> str:
    """生成社区摘要"""
    # 获取社区内的关系
    relations = []
    for source in nodes:
        for target in nodes:
            if graph.has_edge(source, target):
                edge_data = graph.get_edge_data(source, target)
                relations.append(f"{source} --[{edge_data['relation']}]--> {target}")

    # 使用LLM生成摘要
    context = f"实体: {', '.join(nodes)}\n关系:\n" + "\n".join(relations)

    prompt = f"""为以下知识图谱社区生成简洁摘要。

{context}

摘要（1-2句话）："""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=100
    )

    return response.choices[0].message.content

# 为每个社区生成摘要
community_summaries = {}
for comm_id, nodes in list(community_groups.items())[:3]:
    summary = generate_community_summary(nodes, graph)
    community_summaries[comm_id] = summary
    print(f"\n社区 {comm_id} 摘要:")
    print(f"  {summary}")

# ===== 7. 图谱检索 =====
print("\n=== 图谱检索 ===")

class GraphRAG:
    """基于知识图谱的RAG系统"""

    def __init__(self, graph, client, community_summaries):
        self.graph = graph
        self.client = client
        self.community_summaries = community_summaries

    def local_search(self, entity: str, hops: int = 2) -> Dict:
        """局部搜索：从实体出发的邻居检索"""
        if entity not in self.graph:
            return {"entities": [], "relations": []}

        # 获取N跳邻居
        neighbors = set([entity])
        current_level = set([entity])

        for _ in range(hops):
            next_level = set()
            for node in current_level:
                next_level.update(self.graph.successors(node))
                next_level.update(self.graph.predecessors(node))
            neighbors.update(next_level)
            current_level = next_level

        # 获取子图
        subgraph = self.graph.subgraph(neighbors)

        # 提取关系
        relations = []
        for source, target, data in subgraph.edges(data=True):
            relations.append({
                "source": source,
                "relation": data["relation"],
                "target": target
            })

        return {
            "entities": list(neighbors),
            "relations": relations
        }

    def global_search(self, query: str) -> str:
        """全局搜索：使用社区摘要"""
        # 构建全局上下文
        context = "知识图谱社区摘要:\n\n"
        for comm_id, summary in self.community_summaries.items():
            context += f"社区 {comm_id}: {summary}\n"

        # 使用LLM回答
        prompt = f"""{context}

基于以上知识图谱摘要回答问题。

问题：{query}

答案："""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )

        return response.choices[0].message.content

    def hybrid_search(self, query: str) -> str:
        """混合搜索：结合局部和全局"""
        # 1. 识别查询中的实体
        entities = self._extract_entities_from_query(query)

        # 2. 局部搜索
        local_context = []
        for entity in entities:
            result = self.local_search(entity, hops=1)
            if result["relations"]:
                local_context.extend([
                    f"{r['source']} --[{r['relation']}]--> {r['target']}"
                    for r in result["relations"]
                ])

        # 3. 全局搜索
        global_answer = self.global_search(query)

        # 4. 结合生成最终答案
        context = "局部知识:\n" + "\n".join(local_context[:5])
        context += f"\n\n全局理解:\n{global_answer}"

        prompt = f"""基于以下知识回答问题。

{context}

问题：{query}

答案："""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )

        return response.choices[0].message.content

    def _extract_entities_from_query(self, query: str) -> List[str]:
        """从查询中提取实体"""
        # 简单实现：检查图谱中的实体是否在查询中
        entities = []
        for node in self.graph.nodes():
            if node.lower() in query.lower():
                entities.append(node)
        return entities

# 创建GraphRAG系统
graph_rag = GraphRAG(graph, client, community_summaries)

# ===== 8. 测试不同检索模式 =====
print("\n=== 测试不同检索模式 ===")

# 测试局部搜索
print("\n【局部搜索】")
entity = "FastAPI"
local_result = graph_rag.local_search(entity, hops=1)
print(f"实体: {entity}")
print(f"邻居数量: {len(local_result['entities'])}")
print(f"关系:")
for rel in local_result["relations"][:3]:
    print(f"  - {rel['source']} --[{rel['relation']}]--> {rel['target']}")

# 测试全局搜索
print("\n【全局搜索】")
query = "Python生态系统中有哪些重要的框架？"
global_answer = graph_rag.global_search(query)
print(f"查询: {query}")
print(f"答案: {global_answer}")

# 测试混合搜索
print("\n【混合搜索】")
query = "FastAPI基于什么技术构建？"
hybrid_answer = graph_rag.hybrid_search(query)
print(f"查询: {query}")
print(f"答案: {hybrid_answer}")

# ===== 9. 对比传统RAG vs GraphRAG =====
print("\n=== 对比传统RAG vs GraphRAG ===")

# 传统RAG（向量检索）
def traditional_rag(query: str) -> str:
    """传统向量检索RAG"""
    # 简单实现：直接在文档中搜索
    relevant_docs = [doc for doc in documents if any(word in doc for word in query.split())][:3]
    context = "\n".join(relevant_docs)

    prompt = f"基于以下上下文回答问题。\n\n上下文：\n{context}\n\n问题：{query}\n\n答案："
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content

test_query = "FastAPI和Python有什么关系？"

print(f"\n查询: {test_query}\n")

print("【传统RAG】")
trad_answer = traditional_rag(test_query)
print(f"答案: {trad_answer}\n")

print("【GraphRAG】")
graph_answer = graph_rag.hybrid_search(test_query)
print(f"答案: {graph_answer}")

print("\n=== GraphRAG演示完成 ===")
```

---

## 运行输出示例

```
=== 准备测试数据 ===
✓ 准备了 10 个测试文档

=== 初始化组件 ===
✓ 组件初始化完成

=== 实体和关系提取 ===
✓ 提取了 15 个实体
✓ 提取了 12 个关系

实体示例:
  - Python (编程语言)
  - Guido van Rossum (人物)
  - FastAPI (框架)

关系示例:
  - Guido van Rossum --[创建]--> Python
  - FastAPI --[基于]--> Python
  - FastAPI --[使用]--> Starlette

=== 构建知识图谱 ===
✓ 图谱包含 15 个节点
✓ 图谱包含 12 条边

=== 社区检测 ===
✓ 检测到 3 个社区

社区 0:
  成员: Python, Guido van Rossum, FastAPI, Starlette, Pydantic

社区 1:
  成员: LangChain, OpenAI

社区 2:
  成员: RAG, 向量数据库, ChromaDB, Milvus

=== 生成社区摘要 ===

社区 0 摘要:
  这个社区围绕Python编程语言及其生态系统，包括FastAPI框架及其依赖的Starlette和Pydantic。

社区 1 摘要:
  这个社区关注LLM应用开发框架LangChain及其支持的OpenAI提供商。

社区 2 摘要:
  这个社区涉及RAG系统和向量数据库技术，包括ChromaDB和Milvus。

=== 图谱检索 ===

=== 测试不同检索模式 ===

【局部搜索】
实体: FastAPI
邻居数量: 4
关系:
  - FastAPI --[基于]--> Python
  - FastAPI --[使用]--> Starlette
  - FastAPI --[使用]--> Pydantic

【全局搜索】
查询: Python生态系统中有哪些重要的框架？
答案: Python生态系统中的重要框架包括FastAPI（Web框架）和LangChain（LLM应用开发框架）。

【混合搜索】
查询: FastAPI基于什么技术构建？
答案: FastAPI基于Python编程语言构建，并使用Starlette作为Web服务器和Pydantic进行数据验证。

=== 对比传统RAG vs GraphRAG ===

查询: FastAPI和Python有什么关系？

【传统RAG】
答案: FastAPI是一个基于Python的Web框架。

【GraphRAG】
答案: FastAPI是基于Python编程语言构建的Web框架，它使用Starlette和Pydantic等技术。这种关系体现了Python生态系统中框架之间的依赖关系。

=== GraphRAG演示完成 ===
```

---

## 关键要点

**1. GraphRAG的核心优势**
- 显式建模实体关系
- 支持全局理解（社区摘要）
- 支持多跳推理（邻居检索）
- 结构化知识表示

**2. 与传统RAG的对比**

| 维度 | 传统RAG | GraphRAG |
|------|---------|----------|
| 检索方式 | 向量相似度 | 图谱遍历 + 向量 |
| 关系理解 | 隐式 | 显式 |
| 全局理解 | 弱 | 强（社区摘要） |
| 多跳推理 | 困难 | 容易 |
| 构建成本 | 低 | 高 |

**3. 适用场景**
- 需要理解实体间关系
- 需要全局分析和总结
- 需要多跳推理
- 知识密集型领域

**4. 实现要点**
- 实体和关系提取（使用LLM）
- 社区检测（Louvain算法）
- 社区摘要生成
- 局部和全局检索结合

---

## 扩展建议

**1. 使用专业图数据库**
```python
from neo4j import GraphDatabase

# Neo4j连接
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

# 存储实体和关系
with driver.session() as session:
    session.run(
        "CREATE (a:Entity {name: $name, type: $type})",
        name="Python", type="Language"
    )
```

**2. 改进实体提取**
- 使用NER模型（spaCy、Stanza）
- 实体链接和消歧
- 关系分类模型

**3. 增强图谱检索**
- 路径查询（最短路径、所有路径）
- 子图匹配
- 图神经网络嵌入

---

## 总结

GraphRAG通过显式建模实体关系，提供了比传统RAG更强的结构化知识理解能力。特别适合需要理解复杂关系和进行全局分析的场景。

**完成所有16个文件！** 🎉

现在你已经掌握了RAG架构的完整知识体系，从基础概念到高级实现。
