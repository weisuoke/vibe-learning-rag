# GraphRAG知识图谱检索 - 核心概念4: Local vs Global搜索

> 双模式检索策略,GraphRAG的核心优势

---

## 概念定义

**Local Search**和**Global Search**是GraphRAG的两种互补检索模式:
- **Local Search**: 基于实体上下文的细节检索,适合特定查询
- **Global Search**: 基于社区摘要的整体理解,适合全局查询

**一句话**: Local找细节,Global看全局,就像放大镜和望远镜。

---

## 为什么需要双模式搜索?

### 问题场景

```python
# 场景1: 特定查询
query1 = "Alice的职位是什么?"
# 需要: 找到Alice相关的具体信息
# 适合: Local Search

# 场景2: 全局查询  
query2 = "公司的组织结构是什么?"
# 需要: 理解整体架构
# 适合: Global Search
```

**单一模式的局限**:
```python
# 只用Local Search
# ❌ 无法回答全局问题
# ❌ 看不到整体模式

# 只用Global Search
# ❌ 缺少具体细节
# ❌ 无法精确定位
```

---

## Local Search详解

### 1. 核心原理

**Local Search**通过构建实体的局部上下文来回答查询:

```
查询 → 识别实体 → 提取实体上下文 → 生成答案
```

**实体上下文包含**:
1. 实体的直接关系
2. 相关实体的信息
3. 实体所在社区的摘要

### 2. 实现流程

```python
import networkx as nx
from openai import OpenAI
import os
from typing import List, Dict

class LocalSearch:
    """Local Search实现"""
    
    def __init__(self, G: nx.Graph, client: OpenAI):
        self.G = G
        self.client = client
    
    def extract_entities(self, query: str) -> List[str]:
        """从查询中提取实体"""
        prompt = f"""
从以下查询中提取实体名称:

查询: {query}

返回JSON格式: {{"entities": ["实体1", "实体2", ...]}}

只返回JSON。
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        return result["entities"]
    
    def build_entity_context(self, entity: str, max_hops: int = 2) -> Dict:
        """构建实体的局部上下文"""
        context = {
            "entity": entity,
            "direct_relations": [],
            "neighbors": [],
            "subgraph": []
        }
        
        if entity not in self.G:
            return context
        
        # 1. 直接关系
        for neighbor in self.G.neighbors(entity):
            edge_data = self.G.get_edge_data(entity, neighbor)
            relation = edge_data.get('relation', 'connected_to')
            context["direct_relations"].append({
                "target": neighbor,
                "relation": relation
            })
            context["neighbors"].append(neighbor)
        
        # 2. 多跳子图
        if max_hops > 1:
            # 使用BFS提取子图
            visited = {entity}
            queue = [(entity, 0)]
            
            while queue:
                node, depth = queue.pop(0)
                if depth >= max_hops:
                    continue
                
                for neighbor in self.G.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
                        
                        edge_data = self.G.get_edge_data(node, neighbor)
                        relation = edge_data.get('relation', 'connected_to')
                        context["subgraph"].append({
                            "from": node,
                            "relation": relation,
                            "to": neighbor,
                            "depth": depth + 1
                        })
        
        return context
    
    def search(self, query: str) -> str:
        """Local Search主流程"""
        # 1. 提取查询中的实体
        entities = self.extract_entities(query)
        
        if not entities:
            return "未找到相关实体"
        
        # 2. 为每个实体构建上下文
        all_contexts = []
        for entity in entities:
            context = self.build_entity_context(entity)
            all_contexts.append(context)
        
        # 3. 构建prompt
        context_text = self._format_contexts(all_contexts)
        
        prompt = f"""
基于以下实体上下文回答问题:

{context_text}

问题: {query}

请提供详细的回答。
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        return response.choices[0].message.content
    
    def _format_contexts(self, contexts: List[Dict]) -> str:
        """格式化上下文"""
        formatted = []
        for ctx in contexts:
            entity = ctx["entity"]
            formatted.append(f"\n实体: {entity}")
            
            if ctx["direct_relations"]:
                formatted.append("直接关系:")
                for rel in ctx["direct_relations"]:
                    formatted.append(f"  - {entity} --[{rel['relation']}]--> {rel['target']}")
            
            if ctx["subgraph"]:
                formatted.append("扩展关系:")
                for rel in ctx["subgraph"][:10]:  # 限制数量
                    formatted.append(f"  - {rel['from']} --[{rel['relation']}]--> {rel['to']}")
        
        return "\n".join(formatted)

# 使用示例
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 构建示例图
G = nx.Graph()
edges = [
    ("Alice", "Marketing", {"relation": "manages"}),
    ("Bob", "Marketing", {"relation": "works_in"}),
    ("Marketing", "Charlie", {"relation": "reports_to"}),
    ("Charlie", "CEO", {"relation": "is"})
]
G.add_edges_from([(e[0], e[1], e[2]) for e in edges])

local_search = LocalSearch(G, client)

query = "Alice的职位是什么?"
answer = local_search.search(query)
print(f"问题: {query}")
print(f"回答: {answer}")
```

---

## Global Search详解

### 1. 核心原理

**Global Search**基于社区检测和摘要生成:

```
查询 → 社区检测 → 生成社区摘要 → Map-Reduce → 答案
```

**关键步骤**:
1. 将图划分为社区
2. 为每个社区生成摘要
3. 使用Map-Reduce聚合信息

### 2. 实现流程

```python
import community as community_louvain

class GlobalSearch:
    """Global Search实现"""
    
    def __init__(self, G: nx.Graph, client: OpenAI):
        self.G = G
        self.client = client
        self.communities = None
        self.community_summaries = {}
    
    def build_index(self):
        """构建全局索引"""
        # 1. 社区检测
        partition = community_louvain.best_partition(self.G)
        
        self.communities = {}
        for node, comm_id in partition.items():
            if comm_id not in self.communities:
                self.communities[comm_id] = []
            self.communities[comm_id].append(node)
        
        # 2. 生成社区摘要
        for comm_id, members in self.communities.items():
            summary = self._generate_community_summary(members)
            self.community_summaries[comm_id] = summary
    
    def _generate_community_summary(self, members: List[str]) -> str:
        """生成社区摘要"""
        # 提取社区内的关系
        relations = []
        for node1 in members:
            for node2 in members:
                if self.G.has_edge(node1, node2):
                    edge_data = self.G.get_edge_data(node1, node2)
                    relation = edge_data.get('relation', 'connected_to')
                    relations.append(f"{node1} --[{relation}]--> {node2}")
        
        prompt = f"""
分析以下社区并生成摘要:

成员: {', '.join(members)}

关系:
{chr(10).join(relations)}

生成一个简洁的摘要(2-3句话),描述这个社区的特征和作用。

只返回摘要文本。
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        return response.choices[0].message.content
    
    def search(self, query: str) -> str:
        """Global Search主流程"""
        if not self.communities:
            self.build_index()
        
        # Map阶段: 每个社区独立回答
        community_answers = []
        for comm_id, summary in self.community_summaries.items():
            answer = self._query_community(query, summary)
            community_answers.append({
                "community_id": comm_id,
                "answer": answer
            })
        
        # Reduce阶段: 聚合所有答案
        final_answer = self._reduce_answers(query, community_answers)
        
        return final_answer
    
    def _query_community(self, query: str, summary: str) -> str:
        """查询单个社区"""
        prompt = f"""
基于以下社区摘要回答问题:

社区摘要: {summary}

问题: {query}

如果这个社区与问题相关,请提供答案。如果不相关,返回"不相关"。

只返回答案文本。
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        return response.choices[0].message.content
    
    def _reduce_answers(self, query: str, answers: List[Dict]) -> str:
        """聚合答案"""
        # 过滤不相关的答案
        relevant_answers = [
            a for a in answers 
            if "不相关" not in a["answer"].lower()
        ]
        
        if not relevant_answers:
            return "未找到相关信息"
        
        # 聚合
        answers_text = "\n\n".join([
            f"社区{a['community_id']}: {a['answer']}"
            for a in relevant_answers
        ])
        
        prompt = f"""
基于以下来自不同社区的答案,生成一个综合回答:

{answers_text}

问题: {query}

请生成一个完整、连贯的回答。
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        return response.choices[0].message.content

# 使用示例
global_search = GlobalSearch(G, client)
global_search.build_index()

query = "公司的组织结构是什么?"
answer = global_search.search(query)
print(f"\n问题: {query}")
print(f"回答: {answer}")
```

---

## DRIFT Search: 动态融合

### 1. 核心思想

**DRIFT (Dynamic Reasoning and Inference with Flexible Traversal)** 动态融合Local和Global搜索:

```python
class DRIFTSearch:
    """DRIFT Search: 动态融合Local和Global"""
    
    def __init__(self, G: nx.Graph, client: OpenAI):
        self.local_search = LocalSearch(G, client)
        self.global_search = GlobalSearch(G, client)
        self.client = client
    
    def classify_query(self, query: str) -> str:
        """分类查询类型"""
        prompt = f"""
分析以下查询的类型:

查询: {query}

类型:
- "local": 查询特定实体的细节信息
- "global": 查询整体结构或模式
- "hybrid": 需要同时使用局部和全局信息

返回JSON格式: {{"type": "local|global|hybrid", "reason": "原因"}}

只返回JSON。
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        return result["type"]
    
    def search(self, query: str) -> str:
        """DRIFT Search主流程"""
        # 1. 分类查询
        query_type = self.classify_query(query)
        
        print(f"查询类型: {query_type}")
        
        # 2. 根据类型选择策略
        if query_type == "local":
            return self.local_search.search(query)
        elif query_type == "global":
            return self.global_search.search(query)
        else:  # hybrid
            # 同时使用Local和Global
            local_answer = self.local_search.search(query)
            global_answer = self.global_search.search(query)
            
            # 融合答案
            return self._merge_answers(query, local_answer, global_answer)
    
    def _merge_answers(self, query: str, local: str, global_: str) -> str:
        """融合Local和Global答案"""
        prompt = f"""
融合以下两个答案:

Local Search答案(细节):
{local}

Global Search答案(全局):
{global_}

问题: {query}

请生成一个融合了细节和全局视角的完整回答。
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        return response.choices[0].message.content

# 使用示例
drift_search = DRIFTSearch(G, client)

queries = [
    "Alice的职位是什么?",  # Local
    "公司的组织结构是什么?",  # Global
    "Alice在公司中的角色和影响力如何?"  # Hybrid
]

for query in queries:
    print(f"\n{'='*50}")
    print(f"问题: {query}")
    answer = drift_search.search(query)
    print(f"回答: {answer}")
```

---

## Local vs Global对比

| 维度 | Local Search | Global Search | DRIFT Search |
|------|-------------|--------------|--------------|
| **适用场景** | 特定实体查询 | 整体结构查询 | 复杂混合查询 |
| **检索范围** | 实体邻域(1-2跳) | 全图社区 | 动态选择 |
| **准确性** | 细节准确 | 整体准确 | 最高 |
| **成本** | 低 | 中 | 中-高 |
| **响应速度** | 快 | 中 | 中 |
| **示例查询** | "Alice是谁?" | "有哪些团队?" | "Alice的影响力?" |

---

## 性能优化

### 1. 缓存策略

```python
class CachedSearch:
    """带缓存的搜索"""
    
    def __init__(self):
        self.cache = {}
    
    def search(self, query: str, search_fn) -> str:
        """带缓存的搜索"""
        import hashlib
        key = hashlib.md5(query.encode()).hexdigest()
        
        if key in self.cache:
            return self.cache[key]
        
        result = search_fn(query)
        self.cache[key] = result
        return result
```

### 2. 并行查询

```python
from concurrent.futures import ThreadPoolExecutor

def parallel_community_query(communities, query, client):
    """并行查询多个社区"""
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(query_community, query, summary)
            for summary in communities.values()
        ]
        results = [f.result() for f in futures]
    return results
```

---

## 在RAG开发中的应用

### 应用场景1: 企业知识库

```python
# Local: "张三的联系方式是什么?"
# Global: "公司有哪些部门?"
# DRIFT: "张三在公司中负责哪些项目?"
```

### 应用场景2: 学术文献

```python
# Local: "这篇论文的作者是谁?"
# Global: "这个领域的研究热点是什么?"
# DRIFT: "这篇论文在领域中的影响力如何?"
```

---

## 检查清单

掌握Local vs Global搜索后,你应该能够:

- [ ] 理解Local和Global的区别
- [ ] 实现Local Search
- [ ] 实现Global Search
- [ ] 实现DRIFT Search
- [ ] 根据查询类型选择策略
- [ ] 优化搜索性能

---

**版本**: v1.0 (基于2025-2026生产级实践)
**最后更新**: 2026-02-17
**维护者**: Claude Code
