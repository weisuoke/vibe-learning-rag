# GraphRAG知识图谱检索 - 实战代码场景3: Local & Global 搜索

> 实现双模式检索,理解局部与全局的差异

---

## 场景描述

基于构建好的知识图谱,实现 Local Search 和 Global Search 两种检索模式,对比效果。

**学习目标**:
- 掌握 Local Search 实现
- 掌握 Global Search 实现
- 理解社区检测的作用

---

## 完整代码

```python
"""
场景 3: Local & Global 搜索实现
演示: Local Search (实体上下文) vs Global Search (社区摘要)
"""

import os
from openai import OpenAI
import networkx as nx
from typing import List, Dict
import community as community_louvain  # pip install python-louvain

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== 1. Local Search =====
def local_search(G: nx.DiGraph, query: str) -> str:
    """Local Search: 从实体出发,获取局部上下文"""
    print(f"\n=== Local Search ===")
    print(f"查询: {query}")

    # 步骤 1: 识别查询中的实体
    entities_in_query = [node for node in G.nodes() if node.lower() in query.lower()]

    if not entities_in_query:
        return "未找到相关实体"

    print(f"识别到实体: {entities_in_query}")

    # 步骤 2: 构建局部上下文
    context_parts = []

    for entity in entities_in_query:
        # 实体信息
        entity_type = G.nodes[entity].get("type", "Unknown")
        context_parts.append(f"{entity} 是一个 {entity_type}")

        # 出边关系
        for source, target, data in G.out_edges(entity, data=True):
            relation = data.get("relation", "相关")
            context_parts.append(f"{source} {relation} {target}")

        # 入边关系
        for source, target, data in G.in_edges(entity, data=True):
            relation = data.get("relation", "相关")
            context_parts.append(f"{source} {relation} {target}")

    context = "\n".join(context_parts)
    print(f"\n局部上下文:\n{context}")

    # 步骤 3: LLM 生成答案
    prompt = f"""
基于以下上下文回答问题:

上下文:
{context}

问题: {query}

请简洁回答。
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content
    print(f"\n答案: {answer}")
    return answer

# ===== 2. Global Search =====
def global_search(G: nx.DiGraph, query: str) -> str:
    """Global Search: 基于社区摘要,提供全局视角"""
    print(f"\n=== Global Search ===")
    print(f"查询: {query}")

    # 步骤 1: 社区检测
    # 转换为无向图用于社区检测
    G_undirected = G.to_undirected()
    communities = community_louvain.best_partition(G_undirected)

    print(f"检测到 {len(set(communities.values()))} 个社区")

    # 步骤 2: 生成社区摘要
    community_summaries = []

    for community_id in set(communities.values()):
        # 获取社区成员
        members = [node for node, comm in communities.items() if comm == community_id]

        # 获取社区内的关系
        community_edges = []
        for source, target, data in G.edges(data=True):
            if source in members and target in members:
                relation = data.get("relation", "相关")
                community_edges.append(f"{source} {relation} {target}")

        # 生成社区摘要
        community_context = "\n".join(community_edges)

        summary_prompt = f"""
总结以下社区的主要内容 (一句话):

成员: {', '.join(members)}
关系:
{community_context}

总结:
"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": summary_prompt}]
        )

        summary = response.choices[0].message.content.strip()
        community_summaries.append(f"社区 {community_id + 1}: {summary}")

        print(f"社区 {community_id + 1}: {members}")
        print(f"摘要: {summary}")

    # 步骤 3: 汇总摘要
    global_context = "\n".join(community_summaries)

    # 步骤 4: LLM 生成答案
    prompt = f"""
基于以下全局摘要回答问题:

全局摘要:
{global_context}

问题: {query}

请简洁回答。
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content
    print(f"\n答案: {answer}")
    return answer

# ===== 3. DRIFT Search (动态融合) =====
def drift_search(G: nx.DiGraph, query: str) -> str:
    """DRIFT Search: 根据查询复杂度选择策略"""
    print(f"\n=== DRIFT Search ===")
    print(f"查询: {query}")

    # 分析查询复杂度
    entities_mentioned = sum(1 for node in G.nodes() if node.lower() in query.lower())
    has_global_keywords = any(kw in query.lower() for kw in ["整体", "所有", "主要", "总体"])

    print(f"提到实体数: {entities_mentioned}")
    print(f"包含全局关键词: {has_global_keywords}")

    # 决策
    if entities_mentioned > 0 and not has_global_keywords:
        print("策略: Local Search")
        return local_search(G, query)
    elif has_global_keywords:
        print("策略: Global Search")
        return global_search(G, query)
    else:
        print("策略: 混合 (Local + Global)")
        local_answer = local_search(G, query)
        global_answer = global_search(G, query)
        return f"局部视角: {local_answer}\n全局视角: {global_answer}"

# ===== 4. 使用示例 =====
if __name__ == "__main__":
    # 构建示例图
    G = nx.DiGraph()

    # 添加节点
    G.add_node("Alice", type="Person")
    G.add_node("Bob", type="Person")
    G.add_node("Charlie", type="Person")
    G.add_node("Google", type="Organization")
    G.add_node("Microsoft", type="Organization")
    G.add_node("搜索引擎", type="Product")
    G.add_node("斯坦福大学", type="Organization")

    # 添加边
    G.add_edge("Alice", "Google", relation="工作于")
    G.add_edge("Alice", "搜索引擎", relation="负责")
    G.add_edge("Alice", "斯坦福大学", relation="毕业于")
    G.add_edge("Bob", "Google", relation="管理")
    G.add_edge("Charlie", "Microsoft", relation="工作于")
    G.add_edge("Charlie", "Alice", relation="认识")

    # 测试 Local Search
    local_search(G, "Alice 在哪里工作?")

    # 测试 Global Search
    global_search(G, "公司的主要人员有哪些?")

    # 测试 DRIFT Search
    drift_search(G, "Alice 的工作信息?")
    drift_search(G, "整体情况如何?")
```

---

## 运行输出

```
=== Local Search ===
查询: Alice 在哪里工作?
识别到实体: ['Alice']

局部上下文:
Alice 是一个 Person
Alice 工作于 Google
Alice 负责 搜索引擎
Alice 毕业于 斯坦福大学
Charlie 认识 Alice

答案: Alice 在 Google 工作。

=== Global Search ===
查询: 公司的主要人员有哪些?
检测到 2 个社区
社区 1: ['Alice', 'Bob', 'Google', '搜索引擎', '斯坦福大学']
摘要: Google 的主要人员包括 Alice (负责搜索引擎) 和 Bob (管理)
社区 2: ['Charlie', 'Microsoft']
摘要: Microsoft 的主要人员是 Charlie

答案: 主要人员包括:
1. Google: Alice (负责搜索引擎), Bob (管理)
2. Microsoft: Charlie

=== DRIFT Search ===
查询: Alice 的工作信息?
提到实体数: 1
包含全局关键词: False
策略: Local Search
[Local Search 输出...]

=== DRIFT Search ===
查询: 整体情况如何?
提到实体数: 0
包含全局关键词: True
策略: Global Search
[Global Search 输出...]
```

---

## 关键要点

1. **Local Search 适合特定实体查询,提供精确信息**
2. **Global Search 适合全局理解查询,提供宏观视角**
3. **社区检测是 Global Search 的核心依赖**
4. **DRIFT Search 根据查询复杂度自动选择策略**
5. **生产环境推荐使用 DRIFT Search**

---

**版本**: v1.0 (基于 2025-2026 生产级实践)
**最后更新**: 2026-02-17
**维护者**: Claude Code
