# 实战代码 场景3：Graphiti时序图谱

## 场景描述

使用Graphiti构建AI Agent的时序记忆系统，支持Episode管理和时序查询。

**目标**：
- 安装和配置Graphiti
- 添加Episode
- 时序查询
- 混合搜索
- 知识演化追踪

**技术栈**：
- Python 3.13+
- Graphiti
- Neo4j
- OpenAI API

---

## 完整代码

```python
"""
Graphiti时序知识图谱

功能：
1. Episode管理
2. 时序查询
3. 混合搜索
4. 知识演化追踪
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType


class TemporalKnowledgeGraph:
    """时序知识图谱"""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """初始化Graphiti"""
        self.graphiti = Graphiti(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password
        )
        print("✓ Graphiti初始化成功")

    async def add_episode(
        self,
        name: str,
        content: str,
        timestamp: datetime = None
    ) -> str:
        """添加Episode"""
        if timestamp is None:
            timestamp = datetime.now()

        episode = await self.graphiti.add_episode(
            name=name,
            episode_body=content,
            source=EpisodeType.message,
            source_description="User interaction",
            reference_time=timestamp
        )

        print(f"✓ Episode已添加: {episode.uuid[:8]}...")
        return episode.uuid

    async def search(
        self,
        query: str,
        num_results: int = 5,
        reference_time: datetime = None
    ) -> List[Dict]:
        """搜索知识"""
        results = await self.graphiti.search(
            query=query,
            num_results=num_results,
            reference_time=reference_time
        )

        return [
            {
                "content": r.content,
                "score": r.score,
                "created_at": r.created_at
            }
            for r in results
        ]

    async def get_entity_history(self, entity_name: str) -> List[Dict]:
        """获取实体历史"""
        episodes = await self.graphiti.get_episodes(
            entity_name=entity_name
        )

        # 按时间排序
        episodes.sort(key=lambda e: e.created_at)

        return [
            {
                "time": e.created_at,
                "content": e.episode_body,
                "uuid": e.uuid
            }
            for e in episodes
        ]

    async def temporal_query(
        self,
        query: str,
        target_time: datetime
    ) -> List[Dict]:
        """时序查询：查询特定时间点的知识"""
        results = await self.graphiti.search(
            query=query,
            reference_time=target_time,
            num_results=10
        )

        return [
            {
                "content": r.content,
                "score": r.score,
                "created_at": r.created_at,
                "time_diff": (target_time - r.created_at).days
            }
            for r in results
        ]


class ConversationMemory:
    """对话记忆系统"""

    def __init__(self, kg: TemporalKnowledgeGraph):
        self.kg = kg
        self.conversation_history = []

    async def add_message(
        self,
        user_id: str,
        message: str,
        role: str = "user"
    ):
        """添加对话消息"""
        # 记录到历史
        self.conversation_history.append({
            "user_id": user_id,
            "role": role,
            "message": message,
            "timestamp": datetime.now()
        })

        # 添加到Graphiti
        await self.kg.add_episode(
            name=f"{user_id}_{role}_message",
            content=f"[{role}] {message}",
            timestamp=datetime.now()
        )

    async def get_context(
        self,
        user_id: str,
        query: str,
        num_messages: int = 5
    ) -> List[str]:
        """获取对话上下文"""
        results = await self.kg.search(
            query=query,
            num_results=num_messages
        )

        return [r["content"] for r in results]

    async def get_conversation_summary(self, user_id: str) -> str:
        """获取对话摘要"""
        # 获取最近的对话
        recent_messages = [
            msg for msg in self.conversation_history[-10:]
            if msg["user_id"] == user_id
        ]

        if not recent_messages:
            return "无对话历史"

        summary = f"最近{len(recent_messages)}条对话：\n"
        for msg in recent_messages:
            summary += f"- [{msg['role']}] {msg['message']}\n"

        return summary


class KnowledgeEvolutionTracker:
    """知识演化追踪器"""

    def __init__(self, kg: TemporalKnowledgeGraph):
        self.kg = kg

    async def track_entity_changes(self, entity_name: str) -> List[Dict]:
        """追踪实体变化"""
        history = await self.kg.get_entity_history(entity_name)

        changes = []
        for i, episode in enumerate(history):
            if i == 0:
                changes.append({
                    "time": episode["time"],
                    "type": "created",
                    "content": episode["content"]
                })
            else:
                changes.append({
                    "time": episode["time"],
                    "type": "updated",
                    "content": episode["content"],
                    "previous": history[i-1]["content"]
                })

        return changes

    async def compare_knowledge_at_times(
        self,
        query: str,
        time1: datetime,
        time2: datetime
    ) -> Dict:
        """比较两个时间点的知识"""
        results1 = await self.kg.temporal_query(query, time1)
        results2 = await self.kg.temporal_query(query, time2)

        return {
            "time1": time1,
            "time2": time2,
            "results_at_time1": results1,
            "results_at_time2": results2,
            "changes": self._analyze_changes(results1, results2)
        }

    def _analyze_changes(
        self,
        results1: List[Dict],
        results2: List[Dict]
    ) -> Dict:
        """分析变化"""
        content1 = {r["content"] for r in results1}
        content2 = {r["content"] for r in results2}

        return {
            "added": list(content2 - content1),
            "removed": list(content1 - content2),
            "unchanged": list(content1 & content2)
        }


async def main():
    """主函数"""
    print("=== Graphiti时序知识图谱 ===\n")

    # 1. 初始化
    print("1. 初始化Graphiti")
    print("-" * 50)
    kg = TemporalKnowledgeGraph(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password"
    )

    # 2. 添加历史Episode
    print("\n2. 添加历史Episode")
    print("-" * 50)

    # 2020年的Episode
    await kg.add_episode(
        name="zhang_san_join_alibaba",
        content="张三加入阿里巴巴，担任初级工程师",
        timestamp=datetime(2020, 1, 1)
    )

    # 2022年的Episode
    await kg.add_episode(
        name="zhang_san_promotion",
        content="张三晋升为高级工程师，负责推荐系统开发",
        timestamp=datetime(2022, 6, 1)
    )

    # 2024年的Episode
    await kg.add_episode(
        name="zhang_san_leave_alibaba",
        content="张三离开阿里巴巴，加入腾讯担任技术专家",
        timestamp=datetime(2024, 1, 1)
    )

    # 3. 当前搜索
    print("\n3. 当前搜索")
    print("-" * 50)
    results = await kg.search("张三在哪工作？", num_results=3)
    print(f"找到 {len(results)} 条结果：")
    for i, result in enumerate(results, 1):
        print(f"\n结果{i}:")
        print(f"  内容: {result['content']}")
        print(f"  分数: {result['score']:.2f}")
        print(f"  时间: {result['created_at']}")

    # 4. 时序查询
    print("\n4. 时序查询")
    print("-" * 50)

    # 查询2021年的知识
    print("\n查询2021年的知识:")
    results_2021 = await kg.temporal_query(
        "张三的职位是什么？",
        datetime(2021, 1, 1)
    )
    for result in results_2021[:2]:
        print(f"  - {result['content']} (时间差: {result['time_diff']}天)")

    # 查询2023年的知识
    print("\n查询2023年的知识:")
    results_2023 = await kg.temporal_query(
        "张三的职位是什么？",
        datetime(2023, 1, 1)
    )
    for result in results_2023[:2]:
        print(f"  - {result['content']} (时间差: {result['time_diff']}天)")

    # 5. 对话记忆系统
    print("\n5. 对话记忆系统")
    print("-" * 50)
    memory = ConversationMemory(kg)

    # 模拟对话
    await memory.add_message("user_001", "我想了解Python异步编程", "user")
    await memory.add_message("user_001", "asyncio是Python的异步编程库", "assistant")
    await memory.add_message("user_001", "能给我一个例子吗？", "user")

    # 获取上下文
    context = await memory.get_context("user_001", "异步编程", num_messages=3)
    print(f"\n对话上下文 ({len(context)}条):")
    for i, msg in enumerate(context, 1):
        print(f"  {i}. {msg}")

    # 6. 知识演化追踪
    print("\n6. 知识演化追踪")
    print("-" * 50)
    tracker = KnowledgeEvolutionTracker(kg)

    changes = await tracker.track_entity_changes("张三")
    print(f"\n张三的职业轨迹 ({len(changes)}个变化):")
    for change in changes:
        print(f"\n{change['time'].strftime('%Y-%m-%d')} - {change['type']}:")
        print(f"  {change['content']}")

    # 7. 比较不同时间点的知识
    print("\n7. 比较不同时间点的知识")
    print("-" * 50)
    comparison = await tracker.compare_knowledge_at_times(
        "张三的工作",
        datetime(2021, 1, 1),
        datetime(2024, 6, 1)
    )

    print(f"\n2021年 vs 2024年:")
    print(f"  新增知识: {len(comparison['changes']['added'])}条")
    for item in comparison['changes']['added'][:2]:
        print(f"    + {item}")

    print(f"  移除知识: {len(comparison['changes']['removed'])}条")
    for item in comparison['changes']['removed'][:2]:
        print(f"    - {item}")

    print("\n✓ 完成")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 运行示例

```bash
# 1. 安装Graphiti
pip install graphiti-core

# 2. 启动Neo4j
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# 3. 设置环境变量
export OPENAI_API_KEY="your-api-key"

# 4. 运行
python examples/kg/03_graphiti_temporal.py
```

**预期输出**：
```
=== Graphiti时序知识图谱 ===

1. 初始化Graphiti
--------------------------------------------------
✓ Graphiti初始化成功

2. 添加历史Episode
--------------------------------------------------
✓ Episode已添加: a1b2c3d4...
✓ Episode已添加: e5f6g7h8...
✓ Episode已添加: i9j0k1l2...

3. 当前搜索
--------------------------------------------------
找到 3 条结果：

结果1:
  内容: 张三离开阿里巴巴，加入腾讯担任技术专家
  分数: 0.95
  时间: 2024-01-01 00:00:00

结果2:
  内容: 张三晋升为高级工程师，负责推荐系统开发
  分数: 0.88
  时间: 2022-06-01 00:00:00

结果3:
  内容: 张三加入阿里巴巴，担任初级工程师
  分数: 0.82
  时间: 2020-01-01 00:00:00

4. 时序查询
--------------------------------------------------

查询2021年的知识:
  - 张三加入阿里巴巴，担任初级工程师 (时间差: 365天)

查询2023年的知识:
  - 张三晋升为高级工程师，负责推荐系统开发 (时间差: 214天)

5. 对话记忆系统
--------------------------------------------------

对话上下文 (3条):
  1. [user] 我想了解Python异步编程
  2. [assistant] asyncio是Python的异步编程库
  3. [user] 能给我一个例子吗？

6. 知识演化追踪
--------------------------------------------------

张三的职业轨迹 (3个变化):

2020-01-01 - created:
  张三加入阿里巴巴，担任初级工程师

2022-06-01 - updated:
  张三晋升为高级工程师，负责推荐系统开发

2024-01-01 - updated:
  张三离开阿里巴巴，加入腾讯担任技术专家

7. 比较不同时间点的知识
--------------------------------------------------

2021年 vs 2024年:
  新增知识: 2条
    + 张三离开阿里巴巴，加入腾讯担任技术专家
    + 张三晋升为高级工程师，负责推荐系统开发
  移除知识: 0条

✓ 完成
```

---

## 扩展功能

### 1. AI Agent集成

```python
class GraphitiAgent:
    """集成Graphiti的Agent"""

    def __init__(self, kg: TemporalKnowledgeGraph):
        self.kg = kg

    async def process_query(self, query: str) -> str:
        """处理查询"""
        # 1. 检索相关记忆
        context = await self.kg.search(query, num_results=5)

        # 2. 构建prompt
        context_text = "\n".join([r["content"] for r in context])
        prompt = f"基于以下上下文回答问题：\n{context_text}\n\n问题：{query}"

        # 3. 调用LLM
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content

        # 4. 保存交互
        await self.kg.add_episode(
            name="agent_interaction",
            content=f"Q: {query}\nA: {answer}"
        )

        return answer
```

### 2. 批量导入

```python
async def batch_import_episodes(
    kg: TemporalKnowledgeGraph,
    episodes: List[Dict]
):
    """批量导入Episode"""
    tasks = []
    for episode in episodes:
        task = kg.add_episode(
            name=episode["name"],
            content=episode["content"],
            timestamp=episode["timestamp"]
        )
        tasks.append(task)

    await asyncio.gather(*tasks)
    print(f"✓ 批量导入完成: {len(episodes)}个Episode")
```

### 3. 定期清理

```python
async def cleanup_old_episodes(
    kg: TemporalKnowledgeGraph,
    days: int = 90
):
    """清理旧Episode"""
    cutoff_time = datetime.now() - timedelta(days=days)

    # 删除旧Episode
    await kg.graphiti.delete_episodes(
        before=cutoff_time,
        keep_important=True
    )

    print(f"✓ 清理完成: 删除{days}天前的Episode")
```

---

## 总结

本示例实现了完整的Graphiti时序知识图谱系统，包括：
- Episode管理
- 时序查询
- 对话记忆
- 知识演化追踪
- 时间点比较

**适用场景**：
- AI Agent长期记忆
- 对话历史管理
- 知识演化分析
- 时序推理

**下一步**：
- 集成LangGraph
- 实现多代理协作
- 添加向量检索

---

**版本**：v1.0
**最后更新**：2026-02-14
**维护者**：Claude Code
