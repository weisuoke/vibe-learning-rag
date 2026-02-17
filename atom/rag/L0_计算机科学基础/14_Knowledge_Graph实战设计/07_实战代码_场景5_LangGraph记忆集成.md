# 实战代码 场景5：LangGraph记忆集成

## 场景描述

使用LangGraph构建多代理协作系统，集成Neo4j作为共享知识图谱。

**目标**：
- LangGraph状态图
- Neo4j共享知识
- 多代理协作
- 跨会话记忆

**技术栈**：
- Python 3.13+
- LangGraph
- Neo4j
- OpenAI API

---

## 完整代码

```python
"""
LangGraph记忆集成

功能：
1. LangGraph状态图
2. Neo4j共享知识
3. 多代理协作
4. 跨会话记忆
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, List, Dict
from neo4j import GraphDatabase
from openai import OpenAI

client = OpenAI()


class AgentState(TypedDict):
    """Agent状态"""
    messages: List[str]
    context: List[str]
    entities: List[str]
    agent_id: str


class Neo4jMemoryStore:
    """Neo4j记忆存储"""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def save_knowledge(self, agent_id: str, knowledge: str):
        """保存知识"""
        with self.driver.session() as session:
            session.run("""
                MERGE (a:Agent {id: $agent_id})
                CREATE (k:Knowledge {
                    content: $knowledge,
                    created: timestamp()
                })
                CREATE (a)-[:KNOWS]->(k)
            """, agent_id=agent_id, knowledge=knowledge)

    def get_knowledge(self, agent_id: str, limit: int = 5) -> List[str]:
        """获取知识"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Agent {id: $agent_id})-[:KNOWS]->(k:Knowledge)
                RETURN k.content AS content
                ORDER BY k.created DESC
                LIMIT $limit
            """, agent_id=agent_id, limit=limit)

            return [record["content"] for record in result]

    def get_shared_knowledge(self, limit: int = 10) -> List[str]:
        """获取共享知识"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (k:Knowledge)
                RETURN k.content AS content
                ORDER BY k.created DESC
                LIMIT $limit
            """, limit=limit)

            return [record["content"] for record in result]

    def close(self):
        """关闭连接"""
        self.driver.close()


class MultiAgentSystem:
    """多代理系统"""

    def __init__(self, memory_store: Neo4jMemoryStore):
        self.memory_store = memory_store
        self.checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

    def create_agent(self, agent_id: str, role: str):
        """创建Agent"""
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("read_memory", lambda s: self._read_memory(s, agent_id))
        workflow.add_node("process", lambda s: self._process(s, role))
        workflow.add_node("write_memory", lambda s: self._write_memory(s, agent_id))

        # 添加边
        workflow.set_entry_point("read_memory")
        workflow.add_edge("read_memory", "process")
        workflow.add_edge("process", "write_memory")
        workflow.add_edge("write_memory", END)

        # 编译
        app = workflow.compile(checkpointer=self.checkpointer)

        return app

    def _read_memory(self, state: AgentState, agent_id: str):
        """读取记忆"""
        # 读取Agent自己的知识
        agent_knowledge = self.memory_store.get_knowledge(agent_id, limit=3)

        # 读取共享知识
        shared_knowledge = self.memory_store.get_shared_knowledge(limit=5)

        state["context"] = agent_knowledge + shared_knowledge
        state["agent_id"] = agent_id

        return state

    def _process(self, state: AgentState, role: str):
        """处理任务"""
        query = state["messages"][-1] if state["messages"] else ""
        context = state["context"]

        # 构建prompt
        context_text = "\n".join(context) if context else "无上下文"
        prompt = f"""
你是一个{role}。

上下文：
{context_text}

任务：{query}

请完成任务。
"""

        # 调用LLM
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content
        state["messages"].append(f"[{role}] {answer}")

        return state

    def _write_memory(self, state: AgentState, agent_id: str):
        """写入记忆"""
        if state["messages"]:
            last_message = state["messages"][-1]
            self.memory_store.save_knowledge(agent_id, last_message)

        return state


class CollaborativeRAG:
    """协作式RAG系统"""

    def __init__(self, memory_store: Neo4jMemoryStore):
        self.system = MultiAgentSystem(memory_store)

        # 创建多个Agent
        self.researcher = self.system.create_agent("researcher", "研究员")
        self.writer = self.system.create_agent("writer", "写作者")
        self.reviewer = self.system.create_agent("reviewer", "审核者")

    def process_query(self, query: str, thread_id: str = "default") -> Dict:
        """处理查询"""
        config = {"configurable": {"thread_id": thread_id}}

        # 1. 研究员收集信息
        print("1. 研究员收集信息...")
        researcher_state = {
            "messages": [f"研究主题：{query}"],
            "context": [],
            "entities": [],
            "agent_id": "researcher"
        }
        researcher_result = self.researcher.invoke(researcher_state, config=config)

        # 2. 写作者撰写内容
        print("2. 写作者撰写内容...")
        writer_state = {
            "messages": [f"基于研究结果撰写：{query}"],
            "context": [],
            "entities": [],
            "agent_id": "writer"
        }
        writer_result = self.writer.invoke(writer_state, config=config)

        # 3. 审核者审核
        print("3. 审核者审核...")
        reviewer_state = {
            "messages": [f"审核内容：{writer_result['messages'][-1]}"],
            "context": [],
            "entities": [],
            "agent_id": "reviewer"
        }
        reviewer_result = self.reviewer.invoke(reviewer_state, config=config)

        return {
            "research": researcher_result["messages"][-1],
            "draft": writer_result["messages"][-1],
            "review": reviewer_result["messages"][-1]
        }


def main():
    """主函数"""
    print("=== LangGraph记忆集成 ===\n")

    # 1. 初始化记忆存储
    print("1. 初始化记忆存储")
    print("-" * 50)
    memory_store = Neo4jMemoryStore(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )
    print("✓ 记忆存储初始化成功")

    # 2. 创建多代理系统
    print("\n2. 创建多代理系统")
    print("-" * 50)
    system = MultiAgentSystem(memory_store)

    # 创建Agent
    researcher = system.create_agent("agent_researcher", "研究员")
    writer = system.create_agent("agent_writer", "写作者")

    print("✓ 创建了2个Agent")

    # 3. 测试单个Agent
    print("\n3. 测试单个Agent")
    print("-" * 50)

    config = {"configurable": {"thread_id": "session_001"}}

    # 研究员任务
    print("\n研究员任务：")
    researcher_state = {
        "messages": ["研究知识图谱的应用场景"],
        "context": [],
        "entities": [],
        "agent_id": "agent_researcher"
    }
    result = researcher.invoke(researcher_state, config=config)
    print(f"结果：{result['messages'][-1][:100]}...")

    # 4. 测试Agent协作
    print("\n4. 测试Agent协作")
    print("-" * 50)

    # 写作者可以访问研究员的知识
    print("\n写作者任务（基于研究员的知识）：")
    writer_state = {
        "messages": ["撰写知识图谱应用的文章"],
        "context": [],
        "entities": [],
        "agent_id": "agent_writer"
    }
    result = writer.invoke(writer_state, config=config)
    print(f"结果：{result['messages'][-1][:100]}...")

    # 5. 查看共享知识
    print("\n5. 查看共享知识")
    print("-" * 50)
    shared_knowledge = memory_store.get_shared_knowledge(limit=5)
    print(f"共享知识库中有 {len(shared_knowledge)} 条知识：")
    for i, knowledge in enumerate(shared_knowledge, 1):
        print(f"  {i}. {knowledge[:80]}...")

    # 6. 协作式RAG
    print("\n6. 协作式RAG")
    print("-" * 50)
    rag = CollaborativeRAG(memory_store)

    query = "知识图谱在AI Agent中的应用"
    result = rag.process_query(query, thread_id="session_002")

    print(f"\n研究结果：")
    print(f"  {result['research'][:100]}...")

    print(f"\n撰写草稿：")
    print(f"  {result['draft'][:100]}...")

    print(f"\n审核意见：")
    print(f"  {result['review'][:100]}...")

    # 7. 跨会话记忆
    print("\n7. 跨会话记忆")
    print("-" * 50)

    # 新会话，但可以访问之前的知识
    config_new = {"configurable": {"thread_id": "session_003"}}

    researcher_state = {
        "messages": ["总结之前的研究成果"],
        "context": [],
        "entities": [],
        "agent_id": "agent_researcher"
    }
    result = researcher.invoke(researcher_state, config=config_new)
    print(f"新会话中访问历史知识：")
    print(f"  {result['messages'][-1][:100]}...")

    # 8. 清理
    memory_store.close()
    print("\n✓ 完成")


if __name__ == "__main__":
    main()
```

---

## 运行示例

```bash
# 1. 安装依赖
pip install langgraph

# 2. 启动Neo4j
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# 3. 设置环境变量
export OPENAI_API_KEY="your-api-key"

# 4. 运行
python examples/kg/05_langgraph_memory.py
```

**预期输出**：
```
=== LangGraph记忆集成 ===

1. 初始化记忆存储
--------------------------------------------------
✓ 记忆存储初始化成功

2. 创建多代理系统
--------------------------------------------------
✓ 创建了2个Agent

3. 测试单个Agent
--------------------------------------------------

研究员任务：
结果：[研究员] 知识图谱在多个领域有广泛应用，包括：1. 企业知识管理 2. 智能问答系统 3. 推荐系统 4. 金融风控...

4. 测试Agent协作
--------------------------------------------------

写作者任务（基于研究员的知识）：
结果：[写作者] 基于研究成果，知识图谱作为一种结构化知识表示方法，在现代AI系统中扮演着重要角色...

5. 查看共享知识
--------------------------------------------------
共享知识库中有 2 条知识：
  1. [写作者] 基于研究成果，知识图谱作为一种结构化知识表示方法，在现代AI系统中扮演着重要角色...
  2. [研究员] 知识图谱在多个领域有广泛应用，包括：1. 企业知识管理 2. 智能问答系统...

6. 协作式RAG
--------------------------------------------------
1. 研究员收集信息...
2. 写作者撰写内容...
3. 审核者审核...

研究结果：
  [研究员] AI Agent需要知识图谱来维护结构化记忆，支持复杂推理和决策...

撰写草稿：
  [写作者] 知识图谱为AI Agent提供了强大的记忆和推理能力，使Agent能够理解复杂的实体关系...

审核意见：
  [审核者] 内容结构清晰，论述充分，建议补充具体案例...

7. 跨会话记忆
--------------------------------------------------
新会话中访问历史知识：
  [研究员] 根据之前的研究，我们已经探讨了知识图谱在企业知识管理、智能问答等领域的应用...

✓ 完成
```

---

## 扩展功能

### 1. 添加Agent通信

```python
class AgentCommunication:
    """Agent通信"""

    def __init__(self, memory_store: Neo4jMemoryStore):
        self.memory_store = memory_store

    def send_message(
        self,
        from_agent: str,
        to_agent: str,
        message: str
    ):
        """发送消息"""
        with self.memory_store.driver.session() as session:
            session.run("""
                MATCH (from:Agent {id: $from_agent})
                MATCH (to:Agent {id: $to_agent})
                CREATE (m:Message {
                    content: $message,
                    created: timestamp()
                })
                CREATE (from)-[:SENT]->(m)-[:TO]->(to)
            """, from_agent=from_agent, to_agent=to_agent, message=message)

    def get_messages(self, agent_id: str) -> List[Dict]:
        """获取消息"""
        with self.memory_store.driver.session() as session:
            result = session.run("""
                MATCH (from:Agent)-[:SENT]->(m:Message)-[:TO]->(to:Agent {id: $agent_id})
                RETURN from.id AS from_agent, m.content AS content
                ORDER BY m.created DESC
            """, agent_id=agent_id)

            return [
                {"from": record["from_agent"], "content": record["content"]}
                for record in result
            ]
```

### 2. 添加任务分配

```python
class TaskCoordinator:
    """任务协调器"""

    def __init__(self, agents: Dict[str, any]):
        self.agents = agents

    def distribute_task(self, task: str) -> Dict:
        """分配任务"""
        # 分析任务
        subtasks = self._analyze_task(task)

        # 分配给不同Agent
        results = {}
        for subtask in subtasks:
            agent_id = self._select_agent(subtask)
            result = self.agents[agent_id].invoke({
                "messages": [subtask],
                "context": [],
                "entities": [],
                "agent_id": agent_id
            })
            results[agent_id] = result

        return results

    def _analyze_task(self, task: str) -> List[str]:
        """分析任务"""
        # 使用LLM分解任务
        prompt = f"将以下任务分解为子任务：{task}"
        # ... LLM调用
        return ["子任务1", "子任务2"]

    def _select_agent(self, subtask: str) -> str:
        """选择Agent"""
        # 根据任务类型选择Agent
        if "研究" in subtask:
            return "researcher"
        elif "写作" in subtask:
            return "writer"
        else:
            return "reviewer"
```

### 3. 添加状态持久化

```python
def save_agent_state(agent_id: str, state: Dict):
    """保存Agent状态"""
    import json

    with memory_store.driver.session() as session:
        session.run("""
            MERGE (a:Agent {id: $agent_id})
            SET a.state = $state,
                a.updated = timestamp()
        """, agent_id=agent_id, state=json.dumps(state))

def load_agent_state(agent_id: str) -> Dict:
    """加载Agent状态"""
    import json

    with memory_store.driver.session() as session:
        result = session.run("""
            MATCH (a:Agent {id: $agent_id})
            RETURN a.state AS state
        """, agent_id=agent_id)

        record = result.single()
        if record and record["state"]:
            return json.loads(record["state"])
        return {}
```

---

## 总结

本示例实现了完整的LangGraph记忆集成系统，包括：
- LangGraph状态图
- Neo4j共享知识存储
- 多代理协作
- 跨会话记忆
- 检查点机制

**核心优势**：
- 状态管理清晰
- 知识共享高效
- 支持多代理协作
- 跨会话记忆保留

**适用场景**：
- 多代理协作系统
- 复杂任务分解
- 知识积累和共享
- 长期记忆管理

**下一步**：
- 添加Agent通信机制
- 实现任务协调器
- 添加冲突解决
- 优化性能

---

**版本**：v1.0
**最后更新**：2026-02-14
**维护者**：Claude Code
