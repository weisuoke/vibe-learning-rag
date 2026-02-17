# 实战代码：LangGraph记忆管理

> **使用 LangGraph 实现跨会话记忆系统**

---

## 学习目标

- 掌握 LangGraph 的 Checkpointer 机制
- 理解 PostgreSQL + pgvector 集成
- 实现跨会话的对话记忆
- 构建生产级的 AI Agent 记忆系统

---

## 基础实现

### PostgreSQL Checkpointer

```python
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from typing import TypedDict, Annotated
import operator

# 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_id: str

# 初始化 Checkpointer
def create_checkpointer(db_url: str) -> PostgresSaver:
    """
    创建 PostgreSQL Checkpointer

    Args:
        db_url: 数据库连接 URL

    Returns:
        PostgresSaver 实例
    """
    conn = Connection.connect(db_url)
    return PostgresSaver(conn)

# 使用示例
if __name__ == "__main__":
    checkpointer = create_checkpointer(
        "postgresql://user:password@localhost/agent_memory"
    )
```

---

## 完整的 Agent 实现

### 带记忆的对话 Agent

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_openai import ChatOpenAI
from psycopg import Connection
from typing import TypedDict, Annotated
import operator

class ConversationState(TypedDict):
    """对话状态"""
    messages: Annotated[list, operator.add]
    user_id: str

def create_conversation_agent(db_url: str):
    """
    创建带记忆的对话 Agent

    Args:
        db_url: 数据库连接 URL

    Returns:
        编译后的 Agent
    """
    # 初始化 LLM
    llm = ChatOpenAI(model="gpt-4")

    # 初始化 Checkpointer
    conn = Connection.connect(db_url)
    checkpointer = PostgresSaver(conn)

    # 定义节点
    def chat_node(state: ConversationState):
        """对话节点"""
        messages = state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    # 构建图
    workflow = StateGraph(ConversationState)
    workflow.add_node("chat", chat_node)
    workflow.set_entry_point("chat")
    workflow.add_edge("chat", END)

    # 编译（带 Checkpointer）
    app = workflow.compile(checkpointer=checkpointer)

    return app

# 使用示例
if __name__ == "__main__":
    import asyncio

    async def main():
        app = create_conversation_agent(
            "postgresql://localhost/agent_memory"
        )

        # 第一次会话
        config = {"configurable": {"thread_id": "user_123"}}

        result1 = await app.ainvoke(
            {
                "messages": [{"role": "user", "content": "我喜欢 Python"}],
                "user_id": "user_123"
            },
            config=config
        )
        print(result1["messages"][-1].content)

        # 第二次会话（自动加载历史）
        result2 = await app.ainvoke(
            {
                "messages": [{"role": "user", "content": "你还记得我喜欢什么吗？"}],
                "user_id": "user_123"
            },
            config=config
        )
        print(result2["messages"][-1].content)
        # 输出: "是的，你喜欢 Python"

    asyncio.run(main())
```

---

## 数据库表结构

### PostgreSQL Schema

```sql
-- LangGraph checkpoints 表
CREATE TABLE checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_id)
);

-- 索引
CREATE INDEX idx_checkpoints_thread ON checkpoints(thread_id);
CREATE INDEX idx_checkpoints_created ON checkpoints(created_at);

-- 用户偏好表
CREATE TABLE user_preferences (
    user_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value JSONB NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (user_id, key)
);

-- 向量检索表（pgvector）
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE conversation_embeddings (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    message TEXT NOT NULL,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX ON conversation_embeddings
USING ivfflat (embedding vector_cosine_ops);
```

---

## 跨会话记忆管理

### 完整实现

```python
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg.extras import Json
from typing import Optional, List, Dict
import asyncio

class CrossSessionMemory:
    """跨会话记忆管理"""

    def __init__(self, db_url: str):
        """
        初始化

        Args:
            db_url: 数据库连接 URL
        """
        self.conn = Connection.connect(db_url)
        self.checkpointer = PostgresSaver(self.conn)

    async def save_conversation(
        self,
        user_id: str,
        messages: List[Dict]
    ) -> None:
        """
        保存对话历史

        Args:
            user_id: 用户 ID
            messages: 消息列表
        """
        await self.checkpointer.aput(
            config={"configurable": {"thread_id": user_id}},
            checkpoint={"messages": messages}
        )
        print(f"✅ 保存对话: {user_id} ({len(messages)} 条消息)")

    async def load_conversation(
        self,
        user_id: str
    ) -> List[Dict]:
        """
        加载对话历史

        Args:
            user_id: 用户 ID

        Returns:
            消息列表
        """
        checkpoint = await self.checkpointer.aget(
            config={"configurable": {"thread_id": user_id}}
        )

        if checkpoint:
            messages = checkpoint.get("messages", [])
            print(f"✅ 加载对话: {user_id} ({len(messages)} 条消息)")
            return messages

        return []

    def save_preference(
        self,
        user_id: str,
        key: str,
        value: any
    ) -> None:
        """
        保存用户偏好

        Args:
            user_id: 用户 ID
            key: 偏好键
            value: 偏好值
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO user_preferences (user_id, key, value)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id, key)
            DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
        """, (user_id, key, Json(value)))
        self.conn.commit()
        print(f"✅ 保存偏好: {user_id}:{key}")

    def load_preference(
        self,
        user_id: str,
        key: str
    ) -> Optional[any]:
        """
        加载用户偏好

        Args:
            user_id: 用户 ID
            key: 偏好键

        Returns:
            偏好值
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT value FROM user_preferences
            WHERE user_id = %s AND key = %s
        """, (user_id, key))
        result = cursor.fetchone()
        return result[0] if result else None

    def get_all_preferences(
        self,
        user_id: str
    ) -> Dict:
        """
        获取用户所有偏好

        Args:
            user_id: 用户 ID

        Returns:
            偏好字典
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT key, value FROM user_preferences
            WHERE user_id = %s
        """, (user_id,))

        return {row[0]: row[1] for row in cursor.fetchall()}

# 使用示例
async def main():
    memory = CrossSessionMemory("postgresql://localhost/agent_memory")

    # 第一次会话
    await memory.save_conversation("user_123", [
        {"role": "user", "content": "我喜欢 Python"},
        {"role": "assistant", "content": "好的，我记住了"}
    ])
    memory.save_preference("user_123", "language", "zh-CN")
    memory.save_preference("user_123", "theme", "dark")

    # 第二次会话（新会话）
    history = await memory.load_conversation("user_123")
    lang = memory.load_preference("user_123", "language")
    all_prefs = memory.get_all_preferences("user_123")

    print(f"历史对话: {history}")
    print(f"用户语言: {lang}")
    print(f"所有偏好: {all_prefs}")

asyncio.run(main())
```

---

## 向量检索增强

### 语义搜索历史消息

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

class VectorEnhancedMemory(CrossSessionMemory):
    """向量检索增强的记忆系统"""

    def _get_embedding(self, text: str) -> List[float]:
        """获取文本向量"""
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def add_message_with_embedding(
        self,
        user_id: str,
        role: str,
        content: str
    ) -> None:
        """
        添加消息（带向量）

        Args:
            user_id: 用户 ID
            role: 角色
            content: 内容
        """
        embedding = self._get_embedding(content)

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO conversation_embeddings (user_id, message, embedding)
            VALUES (%s, %s, %s)
        """, (user_id, f"{role}: {content}", embedding))
        self.conn.commit()

    def search_similar_messages(
        self,
        user_id: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        搜索相似的历史消息

        Args:
            user_id: 用户 ID
            query: 查询文本
            top_k: 返回数量

        Returns:
            相似消息列表
        """
        query_embedding = self._get_embedding(query)

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT message, embedding <=> %s::vector AS distance
            FROM conversation_embeddings
            WHERE user_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, user_id, query_embedding, top_k))

        return [
            {
                "message": row[0],
                "similarity": 1 - row[1]  # 距离转相似度
            }
            for row in cursor.fetchall()
        ]

# 使用示例
async def main():
    memory = VectorEnhancedMemory("postgresql://localhost/agent_memory")

    # 添加历史消息
    memory.add_message_with_embedding("user_123", "user", "我喜欢 Python")
    memory.add_message_with_embedding("user_123", "assistant", "好的，我记住了")
    memory.add_message_with_embedding("user_123", "user", "推荐一些 Python 书籍")

    # 搜索相关历史
    similar = memory.search_similar_messages("user_123", "Python 学习资源")
    for msg in similar:
        print(f"{msg['message']} (相似度: {msg['similarity']:.3f})")

asyncio.run(main())
```

---

## 完整的 AI Agent 示例

### 带记忆的 RAG Agent

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated, List
import operator

class RAGAgentState(TypedDict):
    """RAG Agent 状态"""
    messages: Annotated[List, operator.add]
    user_id: str
    context: str

def create_rag_agent(db_url: str, memory: VectorEnhancedMemory):
    """
    创建带记忆的 RAG Agent

    Args:
        db_url: 数据库连接 URL
        memory: 记忆系统

    Returns:
        编译后的 Agent
    """
    llm = ChatOpenAI(model="gpt-4")
    conn = Connection.connect(db_url)
    checkpointer = PostgresSaver(conn)

    def retrieve_context(state: RAGAgentState):
        """检索上下文"""
        user_id = state["user_id"]
        last_message = state["messages"][-1].content

        # 搜索相关历史
        similar = memory.search_similar_messages(user_id, last_message, top_k=3)
        context = "\n".join([msg["message"] for msg in similar])

        return {"context": context}

    def generate_response(state: RAGAgentState):
        """生成回复"""
        messages = state["messages"]
        context = state.get("context", "")

        # 构建带上下文的提示
        system_message = f"基于以下历史上下文回答问题:\n{context}"
        full_messages = [
            {"role": "system", "content": system_message}
        ] + messages

        response = llm.invoke(full_messages)

        # 保存到向量数据库
        user_id = state["user_id"]
        memory.add_message_with_embedding(
            user_id,
            "user",
            messages[-1].content
        )
        memory.add_message_with_embedding(
            user_id,
            "assistant",
            response.content
        )

        return {"messages": [response]}

    # 构建图
    workflow = StateGraph(RAGAgentState)
    workflow.add_node("retrieve", retrieve_context)
    workflow.add_node("generate", generate_response)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # 编译
    app = workflow.compile(checkpointer=checkpointer)

    return app

# 使用示例
async def main():
    memory = VectorEnhancedMemory("postgresql://localhost/agent_memory")
    app = create_rag_agent("postgresql://localhost/agent_memory", memory)

    config = {"configurable": {"thread_id": "user_123"}}

    # 第一次对话
    result1 = await app.ainvoke(
        {
            "messages": [HumanMessage(content="我在学习 RAG")],
            "user_id": "user_123"
        },
        config=config
    )
    print(result1["messages"][-1].content)

    # 第二次对话（有上下文）
    result2 = await app.ainvoke(
        {
            "messages": [HumanMessage(content="能举个例子吗？")],
            "user_id": "user_123"
        },
        config=config
    )
    print(result2["messages"][-1].content)

asyncio.run(main())
```

---

## 记忆管理工具

### 清理和维护

```python
class MemoryManager:
    """记忆管理工具"""

    def __init__(self, db_url: str):
        self.conn = Connection.connect(db_url)

    def cleanup_old_checkpoints(self, days: int = 30) -> int:
        """
        清理旧的 checkpoints

        Args:
            days: 保留天数

        Returns:
            删除的数量
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM checkpoints
            WHERE created_at < NOW() - INTERVAL '%s days'
        """, (days,))
        self.conn.commit()

        deleted = cursor.rowcount
        print(f"✅ 清理了 {deleted} 个旧 checkpoints")
        return deleted

    def get_user_stats(self, user_id: str) -> Dict:
        """
        获取用户统计

        Args:
            user_id: 用户 ID

        Returns:
            统计信息
        """
        cursor = self.conn.cursor()

        # 对话数量
        cursor.execute("""
            SELECT COUNT(*) FROM checkpoints
            WHERE thread_id = %s
        """, (user_id,))
        checkpoint_count = cursor.fetchone()[0]

        # 消息数量
        cursor.execute("""
            SELECT COUNT(*) FROM conversation_embeddings
            WHERE user_id = %s
        """, (user_id,))
        message_count = cursor.fetchone()[0]

        # 偏好数量
        cursor.execute("""
            SELECT COUNT(*) FROM user_preferences
            WHERE user_id = %s
        """, (user_id,))
        preference_count = cursor.fetchone()[0]

        return {
            "checkpoints": checkpoint_count,
            "messages": message_count,
            "preferences": preference_count
        }

    def export_user_data(self, user_id: str) -> Dict:
        """
        导出用户数据

        Args:
            user_id: 用户 ID

        Returns:
            用户数据
        """
        cursor = self.conn.cursor()

        # 导出 checkpoints
        cursor.execute("""
            SELECT checkpoint FROM checkpoints
            WHERE thread_id = %s
            ORDER BY created_at DESC
            LIMIT 1
        """, (user_id,))
        checkpoint = cursor.fetchone()

        # 导出偏好
        cursor.execute("""
            SELECT key, value FROM user_preferences
            WHERE user_id = %s
        """, (user_id,))
        preferences = {row[0]: row[1] for row in cursor.fetchall()}

        return {
            "checkpoint": checkpoint[0] if checkpoint else None,
            "preferences": preferences
        }

    def delete_user_data(self, user_id: str) -> None:
        """
        删除用户数据（GDPR 合规）

        Args:
            user_id: 用户 ID
        """
        cursor = self.conn.cursor()

        # 删除 checkpoints
        cursor.execute("""
            DELETE FROM checkpoints WHERE thread_id = %s
        """, (user_id,))

        # 删除消息
        cursor.execute("""
            DELETE FROM conversation_embeddings WHERE user_id = %s
        """, (user_id,))

        # 删除偏好
        cursor.execute("""
            DELETE FROM user_preferences WHERE user_id = %s
        """, (user_id,))

        self.conn.commit()
        print(f"✅ 删除了用户数据: {user_id}")

# 使用示例
manager = MemoryManager("postgresql://localhost/agent_memory")

# 清理旧数据
manager.cleanup_old_checkpoints(days=30)

# 获取统计
stats = manager.get_user_stats("user_123")
print(f"用户统计: {stats}")

# 导出数据
data = manager.export_user_data("user_123")
print(f"用户数据: {data}")
```

---

## 性能优化

### 连接池配置

```python
from psycopg_pool import ConnectionPool

class OptimizedMemory:
    """优化的记忆系统"""

    def __init__(self, db_url: str, pool_size: int = 10):
        """
        初始化

        Args:
            db_url: 数据库连接 URL
            pool_size: 连接池大小
        """
        self.pool = ConnectionPool(
            db_url,
            min_size=1,
            max_size=pool_size
        )

    async def save_conversation(self, user_id: str, messages: List[Dict]):
        """使用连接池保存对话"""
        async with self.pool.connection() as conn:
            checkpointer = PostgresSaver(conn)
            await checkpointer.aput(
                config={"configurable": {"thread_id": user_id}},
                checkpoint={"messages": messages}
            )

    def close(self):
        """关闭连接池"""
        self.pool.close()
```

---

## 完整测试套件

```python
import unittest
import asyncio

class TestCrossSessionMemory(unittest.TestCase):
    """跨会话记忆测试"""

    def setUp(self):
        """测试前准备"""
        self.memory = CrossSessionMemory("postgresql://localhost/test_db")

    async def test_save_and_load_conversation(self):
        """测试保存和加载对话"""
        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"}
        ]

        await self.memory.save_conversation("test_user", messages)
        loaded = await self.memory.load_conversation("test_user")

        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["content"], "你好")

    def test_save_and_load_preference(self):
        """测试保存和加载偏好"""
        self.memory.save_preference("test_user", "language", "zh-CN")
        lang = self.memory.load_preference("test_user", "language")

        self.assertEqual(lang, "zh-CN")

    def test_search_similar_messages(self):
        """测试语义搜索"""
        memory = VectorEnhancedMemory("postgresql://localhost/test_db")

        memory.add_message_with_embedding("test_user", "user", "我喜欢 Python")
        similar = memory.search_similar_messages("test_user", "Python 编程")

        self.assertGreater(len(similar), 0)
        self.assertGreater(similar[0]["similarity"], 0.8)

if __name__ == "__main__":
    unittest.main()
```

---

## 总结

### 关键要点

1. **Checkpointer 机制**：自动保存和恢复状态
2. **PostgreSQL 集成**：持久化存储
3. **pgvector 支持**：语义检索
4. **连接池优化**：提升性能
5. **GDPR 合规**：数据导出和删除

### 最佳实践

- 使用连接池管理数据库连接
- 定期清理旧的 checkpoints
- 为向量检索创建索引
- 实现数据导出和删除功能
- 监控数据库性能

---

## 参考资源

- [LangGraph Memory Documentation](https://docs.langchain.com/oss/python/langgraph/add-memory)
- [LangGraph Long Memory Example](https://github.com/FareedKhan-dev/langgraph-long-memory)
- [PostgreSQL pgvector](https://github.com/pgvector/pgvector)
- [AI Agents 2026 Architecture](https://andriifurmanets.com/blogs/ai-agents-2026-practical-architecture-tools-memory-evals-guardrails)
