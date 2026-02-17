# 核心概念 06：Graphiti时序知识图谱

## 什么是时序知识图谱？

**时序知识图谱**（Temporal Knowledge Graph）是在传统知识图谱基础上增加时间维度，能够表示知识的演化和历史状态。

**核心特点**：
- 时间戳：每个实体和关系都有时间信息
- 历史追溯：可以查询任意时间点的知识状态
- 演化分析：可以分析知识的变化趋势
- 版本管理：维护知识的多个版本

**应用场景**：
- AI Agent长期记忆
- 对话历史管理
- 知识演化追踪
- 时序推理

---

## Graphiti简介

**Graphiti**是由Zep开发的时序知识图谱框架，专为AI Agent设计。

**2025-2026核心特性**：
- Episode-based记忆系统
- 实时增量更新
- 时序推理能力
- 与Neo4j/FalkorDB集成
- 混合搜索（向量+图+时序）

**架构**：
```
用户输入 → Episode创建 → 实体提取 → 图更新 → 时序索引
                ↓
         历史Episode维护
                ↓
         时序查询 + 推理
```

---

## Episode-based记忆

### 什么是Episode？

**Episode**是一段时间内的知识快照，包含：
- 时间戳
- 实体和关系
- 上下文信息
- 元数据

**示例**：
```python
episode_1 = {
    "timestamp": "2020-01-01",
    "content": "张三加入阿里巴巴，担任工程师",
    "entities": [
        {"name": "张三", "type": "Person"},
        {"name": "阿里巴巴", "type": "Company"}
    ],
    "relations": [
        {"subject": "张三", "predicate": "WORKS_AT", "object": "阿里巴巴"}
    ]
}

episode_2 = {
    "timestamp": "2024-01-01",
    "content": "张三离开阿里巴巴，加入腾讯",
    "entities": [
        {"name": "张三", "type": "Person"},
        {"name": "腾讯", "type": "Company"}
    ],
    "relations": [
        {"subject": "张三", "predicate": "WORKS_AT", "object": "腾讯"}
    ]
}
```

### Episode的优势

1. **完整上下文**：保留原始对话或事件
2. **时序关系**：维护事件的先后顺序
3. **可追溯性**：可以回溯到任意时间点
4. **增量更新**：新Episode不影响旧Episode

---

## Graphiti基础使用

### 安装

```bash
pip install graphiti-core
```

### 初始化

```python
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

# 初始化Graphiti（使用Neo4j）
graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)
```

### 添加Episode

```python
import asyncio
from datetime import datetime

async def add_episode_example():
    # 添加Episode
    episode = await graphiti.add_episode(
        name="张三入职",
        episode_body="张三在2020年1月1日加入阿里巴巴，担任高级工程师",
        source=EpisodeType.message,
        source_description="HR系统记录",
        reference_time=datetime(2020, 1, 1)
    )

    print(f"Episode ID: {episode.uuid}")
    print(f"创建时间: {episode.created_at}")

# 运行
asyncio.run(add_episode_example())
```

### 查询知识

```python
async def search_example():
    # 搜索相关知识
    results = await graphiti.search(
        query="张三在哪工作？",
        num_results=5
    )

    for result in results:
        print(f"相关度: {result.score}")
        print(f"内容: {result.content}")
        print(f"时间: {result.created_at}")
        print("---")

asyncio.run(search_example())
```

---

## 时序查询

### 查询特定时间点的知识

```python
async def temporal_query():
    # 查询2022年的知识状态
    results = await graphiti.search(
        query="张三在哪工作？",
        reference_time=datetime(2022, 1, 1),
        num_results=5
    )

    # 结果会返回2022年之前的最新信息
    for result in results:
        print(f"时间: {result.created_at}")
        print(f"内容: {result.content}")

asyncio.run(temporal_query())
```

### 查询知识演化

```python
async def evolution_query():
    # 查询张三的职业变化历史
    episodes = await graphiti.get_episodes(
        entity_name="张三",
        relation_type="WORKS_AT"
    )

    # 按时间排序
    episodes.sort(key=lambda e: e.created_at)

    print("张三的职业轨迹：")
    for episode in episodes:
        print(f"{episode.created_at}: {episode.episode_body}")

asyncio.run(evolution_query())
```

---

## 混合搜索

### 向量 + 图 + 时序

```python
async def hybrid_search():
    # 混合搜索
    results = await graphiti.search(
        query="张三的工作经历",
        num_results=10,
        # 向量搜索权重
        vector_weight=0.4,
        # 图遍历权重
        graph_weight=0.3,
        # 时序相关性权重
        temporal_weight=0.3
    )

    for result in results:
        print(f"综合得分: {result.score}")
        print(f"内容: {result.content}")
        print(f"时间: {result.created_at}")
        print("---")

asyncio.run(hybrid_search())
```

---

## 在AI Agent中的应用

### 1. 对话记忆系统

```python
class ConversationMemory:
    """对话记忆系统"""

    def __init__(self, graphiti: Graphiti):
        self.graphiti = graphiti

    async def add_message(self, user_id: str, message: str, timestamp: datetime):
        """添加对话消息"""
        await self.graphiti.add_episode(
            name=f"user_{user_id}_message",
            episode_body=message,
            source=EpisodeType.message,
            source_description=f"User {user_id} message",
            reference_time=timestamp
        )

    async def get_context(self, user_id: str, query: str, num_messages: int = 5):
        """获取对话上下文"""
        results = await self.graphiti.search(
            query=query,
            num_results=num_messages,
            # 只搜索该用户的消息
            filter={"user_id": user_id}
        )

        return [r.content for r in results]

# 使用
memory = ConversationMemory(graphiti)

# 添加对话
await memory.add_message(
    user_id="user_001",
    message="我想了解Python异步编程",
    timestamp=datetime.now()
)

# 获取上下文
context = await memory.get_context(
    user_id="user_001",
    query="异步编程"
)
print(context)
```

### 2. 知识演化追踪

```python
class KnowledgeTracker:
    """知识演化追踪器"""

    def __init__(self, graphiti: Graphiti):
        self.graphiti = graphiti

    async def track_entity_changes(self, entity_name: str):
        """追踪实体的变化历史"""
        episodes = await self.graphiti.get_episodes(
            entity_name=entity_name
        )

        # 按时间排序
        episodes.sort(key=lambda e: e.created_at)

        changes = []
        for i, episode in enumerate(episodes):
            if i == 0:
                changes.append({
                    "time": episode.created_at,
                    "type": "created",
                    "content": episode.episode_body
                })
            else:
                changes.append({
                    "time": episode.created_at,
                    "type": "updated",
                    "content": episode.episode_body,
                    "previous": episodes[i-1].episode_body
                })

        return changes

# 使用
tracker = KnowledgeTracker(graphiti)
changes = await tracker.track_entity_changes("张三")

for change in changes:
    print(f"{change['time']}: {change['type']}")
    print(f"  {change['content']}")
```

### 3. 时序推理

```python
class TemporalReasoner:
    """时序推理器"""

    def __init__(self, graphiti: Graphiti):
        self.graphiti = graphiti

    async def reason_at_time(self, query: str, target_time: datetime):
        """在特定时间点进行推理"""
        # 获取目标时间之前的所有相关知识
        results = await self.graphiti.search(
            query=query,
            reference_time=target_time,
            num_results=10
        )

        # 构建时间点的知识图谱快照
        snapshot = self._build_snapshot(results)

        # 在快照上进行推理
        answer = self._reason_on_snapshot(snapshot, query)

        return answer

    def _build_snapshot(self, results):
        """构建知识图谱快照"""
        entities = set()
        relations = []

        for result in results:
            # 提取实体和关系
            # （简化版，实际需要解析result.content）
            entities.update(result.entities)
            relations.extend(result.relations)

        return {"entities": entities, "relations": relations}

    def _reason_on_snapshot(self, snapshot, query):
        """在快照上推理"""
        # 使用图遍历或LLM进行推理
        # （简化版）
        return f"基于{len(snapshot['relations'])}个关系的推理结果"

# 使用
reasoner = TemporalReasoner(graphiti)
answer = await reasoner.reason_at_time(
    query="张三在2022年的职位是什么？",
    target_time=datetime(2022, 6, 1)
)
print(answer)
```

---

## 与LangGraph集成

### 构建Agent记忆系统

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class AgentState(TypedDict):
    """Agent状态"""
    messages: List[str]
    context: List[str]
    current_query: str

class GraphitiAgent:
    """集成Graphiti的Agent"""

    def __init__(self, graphiti: Graphiti):
        self.graphiti = graphiti
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建状态图"""
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("retrieve_memory", self.retrieve_memory)
        workflow.add_node("process_query", self.process_query)
        workflow.add_node("update_memory", self.update_memory)

        # 添加边
        workflow.set_entry_point("retrieve_memory")
        workflow.add_edge("retrieve_memory", "process_query")
        workflow.add_edge("process_query", "update_memory")
        workflow.add_edge("update_memory", END)

        return workflow.compile()

    async def retrieve_memory(self, state: AgentState):
        """检索记忆"""
        context = await self.graphiti.search(
            query=state["current_query"],
            num_results=5
        )

        state["context"] = [r.content for r in context]
        return state

    async def process_query(self, state: AgentState):
        """处理查询"""
        # 使用LLM处理查询（简化版）
        response = f"基于{len(state['context'])}条记忆的回答"
        state["messages"].append(response)
        return state

    async def update_memory(self, state: AgentState):
        """更新记忆"""
        await self.graphiti.add_episode(
            name="agent_interaction",
            episode_body=state["messages"][-1],
            source=EpisodeType.message,
            reference_time=datetime.now()
        )
        return state

    async def run(self, query: str):
        """运行Agent"""
        initial_state = {
            "messages": [],
            "context": [],
            "current_query": query
        }

        result = await self.graph.ainvoke(initial_state)
        return result["messages"][-1]

# 使用
agent = GraphitiAgent(graphiti)
response = await agent.run("张三的工作经历是什么？")
print(response)
```

---

## 性能优化

### 1. 批量添加Episode

```python
async def batch_add_episodes(episodes: List[dict]):
    """批量添加Episode"""
    tasks = []
    for episode in episodes:
        task = graphiti.add_episode(
            name=episode["name"],
            episode_body=episode["body"],
            source=EpisodeType.message,
            reference_time=episode["time"]
        )
        tasks.append(task)

    # 并发执行
    results = await asyncio.gather(*tasks)
    return results
```

### 2. 缓存查询结果

```python
from functools import lru_cache
import hashlib

class CachedGraphiti:
    """带缓存的Graphiti"""

    def __init__(self, graphiti: Graphiti):
        self.graphiti = graphiti
        self.cache = {}

    async def search(self, query: str, **kwargs):
        """带缓存的搜索"""
        # 生成缓存键
        cache_key = self._generate_cache_key(query, kwargs)

        # 检查缓存
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 执行搜索
        results = await self.graphiti.search(query, **kwargs)

        # 缓存结果
        self.cache[cache_key] = results

        return results

    def _generate_cache_key(self, query: str, kwargs: dict):
        """生成缓存键"""
        key_str = f"{query}_{str(kwargs)}"
        return hashlib.md5(key_str.encode()).hexdigest()
```

---

## 最佳实践

### 1. Episode粒度

```python
# ❌ 太细：每个句子一个Episode
await graphiti.add_episode(
    name="sentence_1",
    episode_body="张三说：你好",
    ...
)

# ✅ 适中：一轮对话一个Episode
await graphiti.add_episode(
    name="conversation_round_1",
    episode_body="用户：你好\nAgent：你好，有什么可以帮助你的？",
    ...
)

# ❌ 太粗：整个会话一个Episode
await graphiti.add_episode(
    name="full_session",
    episode_body="整个会话的所有内容...",
    ...
)
```

### 2. 时间戳管理

```python
# ✅ 使用准确的时间戳
await graphiti.add_episode(
    reference_time=datetime.now()  # 当前时间
)

# ✅ 使用事件发生的时间
await graphiti.add_episode(
    reference_time=event_timestamp  # 事件时间
)

# ❌ 不要使用固定时间
await graphiti.add_episode(
    reference_time=datetime(2020, 1, 1)  # 错误
)
```

### 3. 定期清理

```python
async def cleanup_old_episodes(days: int = 90):
    """清理旧Episode"""
    cutoff_time = datetime.now() - timedelta(days=days)

    # 删除旧Episode（保留重要的）
    await graphiti.delete_episodes(
        before=cutoff_time,
        keep_important=True
    )
```

---

## 总结

### Graphiti的核心价值

1. **时序感知**：维护知识的时间维度
2. **Episode-based**：完整保留上下文
3. **AI Agent优化**：专为Agent设计
4. **混合搜索**：向量+图+时序

### 适用场景

**适合Graphiti如果**：
- AI Agent长期记忆
- 对话历史管理
- 知识演化追踪
- 需要时序推理

**不适合Graphiti如果**：
- 静态知识图谱
- 不需要时间维度
- 简单的键值存储

---

**引用来源**：
- [Graphiti GitHub](https://github.com/getzep/graphiti)
- [arXiv 2501.13956](https://arxiv.org/abs/2501.13956) - Zep论文
- [Neo4j Graphiti博客](https://neo4j.com/blog/graphiti-knowledge-graph-memory/)
- [FalkorDB Graphiti集成](https://www.falkordb.com/blog/graphiti/)

---

**版本**：v1.0
**最后更新**：2026-02-14
**维护者**：Claude Code
