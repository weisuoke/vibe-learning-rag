# 核心概念 08：LangGraph集成

## LangGraph简介

**LangGraph**是LangChain生态中的状态图框架，用于构建复杂的AI Agent应用。

**核心特点**：
- 状态图模型：节点+边+状态
- 循环支持：支持Agent循环决策
- 检查点机制：支持状态持久化
- 人机协作：支持人工介入
- 记忆管理：支持短期和长期记忆

**2025-2026新特性**：
- 图存储集成：支持Neo4j等图数据库
- 时序记忆：与Graphiti集成
- 多代理协作：支持多Agent通信
- 流式输出：支持实时响应

---

## LangGraph记忆系统

### 三层记忆架构

```
┌─────────────────────────────────┐
│  短期记忆（Thread State）        │
│  - 当前会话状态                  │
│  - 临时变量                      │
│  - 消息历史                      │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│  工作记忆（Checkpoints）         │
│  - 会话快照                      │
│  - 可恢复状态                    │
│  - 分支管理                      │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│  长期记忆（Graph Store）         │
│  - 跨会话知识                    │
│  - 实体关系                      │
│  - 时序演化                      │
└─────────────────────────────────┘
```

---

## 基础使用

### 创建状态图

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class AgentState(TypedDict):
    """Agent状态"""
    messages: List[str]
    context: List[str]
    entities: List[str]

# 创建状态图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("extract_entities", extract_entities_node)
workflow.add_node("retrieve_knowledge", retrieve_knowledge_node)
workflow.add_node("generate_response", generate_response_node)

# 添加边
workflow.set_entry_point("extract_entities")
workflow.add_edge("extract_entities", "retrieve_knowledge")
workflow.add_edge("retrieve_knowledge", "generate_response")
workflow.add_edge("generate_response", END)

# 编译
app = workflow.compile()
```

### 节点函数

```python
def extract_entities_node(state: AgentState):
    """提取实体节点"""
    last_message = state["messages"][-1]
    entities = extract_entities(last_message)
    state["entities"] = entities
    return state

def retrieve_knowledge_node(state: AgentState):
    """检索知识节点"""
    entities = state["entities"]
    context = []
    for entity in entities:
        # 从图数据库检索
        knowledge = graph_db.query(entity)
        context.extend(knowledge)
    state["context"] = context
    return state

def generate_response_node(state: AgentState):
    """生成响应节点"""
    query = state["messages"][-1]
    context = state["context"]
    response = llm.generate(query, context)
    state["messages"].append(response)
    return state
```

---

## 图存储集成

### 使用Neo4j作为长期记忆

```python
from langgraph.checkpoint.memory import MemorySaver
from neo4j import GraphDatabase

class Neo4jMemory:
    """Neo4j长期记忆"""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def save_entity(self, entity: str, entity_type: str, metadata: dict):
        """保存实体"""
        with self.driver.session() as session:
            session.run("""
                MERGE (e:Entity {name: $entity})
                SET e.type = $entity_type,
                    e.metadata = $metadata,
                    e.updated = timestamp()
            """, entity=entity, entity_type=entity_type, metadata=metadata)

    def save_relation(self, subject: str, predicate: str, obj: str):
        """保存关系"""
        with self.driver.session() as session:
            session.run("""
                MATCH (s:Entity {name: $subject})
                MATCH (o:Entity {name: $object})
                MERGE (s)-[r:RELATION {type: $predicate}]->(o)
                SET r.updated = timestamp()
            """, subject=subject, predicate=predicate, object=obj)

    def query_entity(self, entity: str) -> List[dict]:
        """查询实体相关知识"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity {name: $entity})-[r]-(related)
                RETURN related.name AS name, type(r) AS relation
                LIMIT 10
            """, entity=entity)
            return [dict(record) for record in result]

# 集成到LangGraph
neo4j_memory = Neo4jMemory("bolt://localhost:7687", "neo4j", "password")

def retrieve_knowledge_node(state: AgentState):
    """从Neo4j检索知识"""
    entities = state["entities"]
    context = []

    for entity in entities:
        knowledge = neo4j_memory.query_entity(entity)
        for item in knowledge:
            context.append(f"{entity} {item['relation']} {item['name']}")

    state["context"] = context
    return state
```

---

## 与Graphiti集成

### 时序记忆Agent

```python
from graphiti_core import Graphiti
from datetime import datetime

class GraphitiLangGraphAgent:
    """集成Graphiti的LangGraph Agent"""

    def __init__(self, graphiti: Graphiti):
        self.graphiti = graphiti
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建状态图"""
        workflow = StateGraph(AgentState)

        workflow.add_node("retrieve_memory", self.retrieve_memory)
        workflow.add_node("process", self.process)
        workflow.add_node("update_memory", self.update_memory)

        workflow.set_entry_point("retrieve_memory")
        workflow.add_edge("retrieve_memory", "process")
        workflow.add_edge("process", "update_memory")
        workflow.add_edge("update_memory", END)

        return workflow.compile()

    async def retrieve_memory(self, state: AgentState):
        """检索时序记忆"""
        query = state["messages"][-1]

        # 从Graphiti检索
        results = await self.graphiti.search(
            query=query,
            num_results=5
        )

        state["context"] = [r.content for r in results]
        return state

    async def process(self, state: AgentState):
        """处理查询"""
        query = state["messages"][-1]
        context = state["context"]

        # 使用LLM生成响应
        response = generate_response(query, context)
        state["messages"].append(response)
        return state

    async def update_memory(self, state: AgentState):
        """更新时序记忆"""
        # 添加新Episode
        await self.graphiti.add_episode(
            name="agent_interaction",
            episode_body=state["messages"][-1],
            reference_time=datetime.now()
        )
        return state

    async def run(self, query: str):
        """运行Agent"""
        initial_state = {
            "messages": [query],
            "context": [],
            "entities": []
        }

        result = await self.graph.ainvoke(initial_state)
        return result["messages"][-1]
```

---

## 多代理协作

### 共享知识图谱

```python
class MultiAgentSystem:
    """多代理系统"""

    def __init__(self, graph_db):
        self.graph_db = graph_db
        self.agents = {}

    def create_agent(self, agent_id: str, role: str):
        """创建Agent"""
        agent = self._build_agent(agent_id, role)
        self.agents[agent_id] = agent
        return agent

    def _build_agent(self, agent_id: str, role: str):
        """构建Agent状态图"""
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("read_shared_knowledge",
                         lambda s: self.read_shared_knowledge(s, agent_id))
        workflow.add_node("process",
                         lambda s: self.process_task(s, role))
        workflow.add_node("write_shared_knowledge",
                         lambda s: self.write_shared_knowledge(s, agent_id))

        # 添加边
        workflow.set_entry_point("read_shared_knowledge")
        workflow.add_edge("read_shared_knowledge", "process")
        workflow.add_edge("process", "write_shared_knowledge")
        workflow.add_edge("write_shared_knowledge", END)

        return workflow.compile()

    def read_shared_knowledge(self, state: AgentState, agent_id: str):
        """读取共享知识"""
        # 从图数据库读取
        with self.graph_db.session() as session:
            result = session.run("""
                MATCH (a:Agent {id: $agent_id})-[:KNOWS]->(k:Knowledge)
                RETURN k.content AS content
            """, agent_id=agent_id)

            state["context"] = [record["content"] for record in result]

        return state

    def write_shared_knowledge(self, state: AgentState, agent_id: str):
        """写入共享知识"""
        # 提取新知识
        new_knowledge = state["messages"][-1]

        # 写入图数据库
        with self.graph_db.session() as session:
            session.run("""
                MATCH (a:Agent {id: $agent_id})
                CREATE (k:Knowledge {content: $content, created: timestamp()})
                CREATE (a)-[:KNOWS]->(k)
            """, agent_id=agent_id, content=new_knowledge)

        return state

    def process_task(self, state: AgentState, role: str):
        """处理任务"""
        # 根据角色处理
        response = f"[{role}] 处理任务: {state['messages'][-1]}"
        state["messages"].append(response)
        return state

# 使用
system = MultiAgentSystem(driver)

# 创建多个Agent
researcher = system.create_agent("agent_1", "研究员")
writer = system.create_agent("agent_2", "写作者")

# Agent协作
researcher_result = researcher.invoke({"messages": ["研究知识图谱"], "context": [], "entities": []})
writer_result = writer.invoke({"messages": ["写一篇文章"], "context": [], "entities": []})
```

---

## 检查点与状态恢复

### 使用检查点

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 创建检查点存储
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# 编译时指定检查点
app = workflow.compile(checkpointer=checkpointer)

# 运行时指定thread_id
config = {"configurable": {"thread_id": "user_123"}}
result = app.invoke(initial_state, config=config)

# 恢复会话
# 下次使用相同thread_id会自动恢复状态
result2 = app.invoke(new_state, config=config)
```

### 状态分支

```python
# 获取状态历史
history = app.get_state_history(config)

for state in history:
    print(f"Checkpoint: {state.config['configurable']['checkpoint_id']}")
    print(f"State: {state.values}")

# 从特定检查点恢复
checkpoint_config = {
    "configurable": {
        "thread_id": "user_123",
        "checkpoint_id": "checkpoint_abc"
    }
}
result = app.invoke(new_state, config=checkpoint_config)
```

---

## 实战示例

### 知识图谱问答Agent

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class QAState(TypedDict):
    """问答状态"""
    question: str
    entities: List[str]
    graph_context: List[str]
    vector_context: List[str]
    answer: str

class KnowledgeGraphQA:
    """知识图谱问答系统"""

    def __init__(self, graph_db, vector_db, llm):
        self.graph_db = graph_db
        self.vector_db = vector_db
        self.llm = llm
        self.app = self._build_app()

    def _build_app(self):
        """构建应用"""
        workflow = StateGraph(QAState)

        # 添加节点
        workflow.add_node("extract_entities", self.extract_entities)
        workflow.add_node("graph_retrieval", self.graph_retrieval)
        workflow.add_node("vector_retrieval", self.vector_retrieval)
        workflow.add_node("generate_answer", self.generate_answer)

        # 添加边
        workflow.set_entry_point("extract_entities")
        workflow.add_edge("extract_entities", "graph_retrieval")
        workflow.add_edge("extract_entities", "vector_retrieval")
        workflow.add_edge("graph_retrieval", "generate_answer")
        workflow.add_edge("vector_retrieval", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    def extract_entities(self, state: QAState):
        """提取实体"""
        entities = extract_entities_llm(state["question"])
        state["entities"] = entities
        return state

    def graph_retrieval(self, state: QAState):
        """图检索"""
        context = []
        for entity in state["entities"]:
            paths = self.graph_db.find_paths(entity, max_hops=2)
            context.extend(paths)
        state["graph_context"] = context
        return state

    def vector_retrieval(self, state: QAState):
        """向量检索"""
        results = self.vector_db.search(state["question"], top_k=5)
        state["vector_context"] = results
        return state

    def generate_answer(self, state: QAState):
        """生成答案"""
        # 合并上下文
        all_context = state["graph_context"] + state["vector_context"]

        # 生成答案
        answer = self.llm.generate(
            state["question"],
            context=all_context
        )

        state["answer"] = answer
        return state

    def query(self, question: str):
        """查询"""
        initial_state = {
            "question": question,
            "entities": [],
            "graph_context": [],
            "vector_context": [],
            "answer": ""
        }

        result = self.app.invoke(initial_state)
        return result["answer"]

# 使用
qa_system = KnowledgeGraphQA(graph_db, vector_db, llm)
answer = qa_system.query("张三在哪个城市工作？")
print(answer)
```

---

## 最佳实践

### 1. 状态设计

```python
# ✅ 好的状态设计
class AgentState(TypedDict):
    messages: List[str]  # 消息历史
    context: List[str]   # 检索上下文
    entities: List[str]  # 提取的实体
    metadata: dict       # 元数据

# ❌ 不好的状态设计
class AgentState(TypedDict):
    data: dict  # 太宽泛
```

### 2. 节点粒度

```python
# ✅ 适当的粒度
workflow.add_node("extract_entities", extract_entities)
workflow.add_node("retrieve_knowledge", retrieve_knowledge)
workflow.add_node("generate_response", generate_response)

# ❌ 粒度太细
workflow.add_node("extract_person", extract_person)
workflow.add_node("extract_company", extract_company)
workflow.add_node("extract_location", extract_location)
```

### 3. 错误处理

```python
def safe_node(func):
    """节点错误处理装饰器"""
    def wrapper(state):
        try:
            return func(state)
        except Exception as e:
            state["error"] = str(e)
            return state
    return wrapper

@safe_node
def risky_node(state):
    # 可能出错的操作
    return state
```

---

## 总结

### LangGraph的核心价值

1. **状态管理**：清晰的状态流转
2. **检查点机制**：支持状态恢复
3. **图存储集成**：支持长期记忆
4. **多代理协作**：支持Agent通信

### 适用场景

**适合LangGraph如果**：
- 复杂的Agent流程
- 需要状态管理
- 需要人机协作
- 多代理系统

**不适合LangGraph如果**：
- 简单的LLM调用
- 无状态应用
- 不需要循环决策

---

**引用来源**：
- [LangGraph官方文档](https://langchain-ai.github.io/langgraph/)
- [LangGraph记忆系统](https://langchain-ai.github.io/langgraph/concepts/memory/)
- [MongoDB LangGraph集成](https://www.mongodb.com/docs/atlas/atlas-vector-search/ai-integrations/langgraph/)
- [Mem0图记忆](https://docs.mem0.ai/features/graph-memory)

---

**版本**：v1.0
**最后更新**：2026-02-14
**维护者**：Claude Code
