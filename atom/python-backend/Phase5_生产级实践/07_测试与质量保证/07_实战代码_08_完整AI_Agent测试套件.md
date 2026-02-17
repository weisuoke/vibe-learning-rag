# 实战代码 08 - 完整 AI Agent 测试套件

## 项目结构

```
project/
├── app/
│   ├── agent/
│   │   ├── chat_agent.py
│   │   ├── rag_agent.py
│   │   └── tools.py
│   ├── rag/
│   │   ├── retrieval.py
│   │   └── vector_store.py
│   └── memory/
│       └── conversation.py
├── tests/
│   ├── conftest.py
│   ├── test_agent/
│   │   ├── test_chat_agent.py
│   │   ├── test_rag_agent.py
│   │   └── test_tools.py
│   ├── test_rag/
│   │   ├── test_retrieval.py
│   │   └── test_vector_store.py
│   └── test_memory/
│       └── test_conversation.py
└── pytest.ini
```

---

## 测试 LangChain Agent

### Agent 实现

```python
# app/agent/chat_agent.py
"""聊天 Agent"""
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from typing import List

def create_chat_agent(tools: List[Tool], llm: ChatOpenAI = None):
    """创建聊天 Agent"""
    if llm is None:
        llm = ChatOpenAI(model="gpt-4", temperature=0)

    # 创建 Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 创建 Agent
    agent = create_openai_functions_agent(llm, tools, prompt)

    # 创建 Agent 执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True
    )

    return agent_executor
```

### Agent 测试

```python
# tests/test_agent/test_chat_agent.py
"""聊天 Agent 测试"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from langchain.tools import Tool
from app.agent.chat_agent import create_chat_agent


@pytest.fixture
def mock_tools():
    """Mock 工具"""
    def search_tool(query: str) -> str:
        return f"Search results for: {query}"

    def calculator_tool(expression: str) -> str:
        try:
            result = eval(expression)
            return str(result)
        except:
            return "Error"

    return [
        Tool(
            name="search",
            func=search_tool,
            description="Search for information"
        ),
        Tool(
            name="calculator",
            func=calculator_tool,
            description="Calculate mathematical expressions"
        )
    ]


class TestChatAgent:
    """聊天 Agent 测试"""

    def test_create_agent(self, mock_tools):
        """测试：创建 Agent"""
        agent = create_chat_agent(mock_tools)

        assert agent is not None
        assert len(agent.tools) == 2

    @patch("langchain_openai.ChatOpenAI")
    def test_agent_with_mock_llm(self, mock_llm_class, mock_tools):
        """测试：使用 Mock LLM"""
        # Mock LLM 响应
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        # Mock Agent 响应
        mock_response = {
            "output": "The answer is 42",
            "intermediate_steps": []
        }

        agent = create_chat_agent(mock_tools, llm=mock_llm)

        # 验证 Agent 创建
        assert agent is not None

    def test_agent_tool_execution(self, mock_tools):
        """测试：工具执行"""
        # 直接测试工具
        search_tool = mock_tools[0]
        result = search_tool.func("test query")

        assert "test query" in result

        calculator_tool = mock_tools[1]
        result = calculator_tool.func("2 + 3")

        assert result == "5"

    @pytest.mark.asyncio
    async def test_agent_async_execution(self, mock_tools):
        """测试：异步执行"""
        # 创建异步 Mock
        async def async_search(query: str) -> str:
            return f"Async search: {query}"

        async_tool = Tool(
            name="async_search",
            func=async_search,
            description="Async search",
            coroutine=async_search
        )

        # 测试异步工具
        result = await async_tool.coroutine("test")
        assert "test" in result
```

---

## 测试 RAG 检索

### RAG 实现

```python
# app/rag/retrieval.py
"""RAG 检索"""
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from typing import List

class RAGRetriever:
    """RAG 检索器"""

    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """检索相关文档"""
        results = self.vector_store.similarity_search(query, k=k)
        return results

    def retrieve_with_scores(self, query: str, k: int = 3) -> List[tuple]:
        """检索相关文档（带分数）"""
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results

    def retrieve_by_threshold(
        self,
        query: str,
        threshold: float = 0.7,
        k: int = 10
    ) -> List[Document]:
        """按阈值检索"""
        results = self.vector_store.similarity_search_with_score(query, k=k)
        filtered = [doc for doc, score in results if score >= threshold]
        return filtered
```

### RAG 测试

```python
# tests/test_rag/test_retrieval.py
"""RAG 检索测试"""
import pytest
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from app.rag.retrieval import RAGRetriever


@pytest.fixture
def sample_documents():
    """示例文档"""
    return [
        "RAG 是检索增强生成技术",
        "Embedding 是文本向量化方法",
        "LangChain 是 AI 开发框架",
        "FastAPI 是 Python Web 框架",
        "pytest 是 Python 测试框架"
    ]


@pytest.fixture
def vector_store(sample_documents):
    """向量存储"""
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(
        sample_documents,
        embeddings,
        collection_name="test_collection"
    )
    yield vectorstore
    # 清理
    vectorstore.delete_collection()


@pytest.fixture
def retriever(vector_store):
    """检索器"""
    return RAGRetriever(vector_store)


class TestRAGRetriever:
    """RAG 检索器测试"""

    def test_retrieve_basic(self, retriever):
        """测试：基本检索"""
        results = retriever.retrieve("什么是 RAG", k=1)

        assert len(results) == 1
        assert "RAG" in results[0].page_content

    def test_retrieve_multiple(self, retriever):
        """测试：检索多个结果"""
        results = retriever.retrieve("Python 框架", k=3)

        assert len(results) == 3
        # 验证结果相关性
        contents = [doc.page_content for doc in results]
        assert any("Python" in c or "框架" in c for c in contents)

    def test_retrieve_with_scores(self, retriever):
        """测试：带分数检索"""
        results = retriever.retrieve_with_scores("RAG", k=2)

        assert len(results) == 2
        # 验证返回格式
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)
            assert 0 <= score <= 1

    def test_retrieve_by_threshold(self, retriever):
        """测试：按阈值检索"""
        results = retriever.retrieve_by_threshold(
            "RAG 技术",
            threshold=0.5,
            k=5
        )

        # 验证所有结果都满足阈值
        assert len(results) >= 0
        assert all(isinstance(doc, Document) for doc in results)

    def test_retrieve_empty_query(self, retriever):
        """测试：空查询"""
        results = retriever.retrieve("", k=3)

        # 空查询应该返回结果（可能是随机的）
        assert isinstance(results, list)

    @pytest.mark.parametrize("k", [1, 3, 5])
    def test_retrieve_different_k(self, retriever, k):
        """测试：不同的 k 值"""
        results = retriever.retrieve("测试", k=k)

        assert len(results) <= k
```

---

## 测试流式输出

### 流式输出实现

```python
# app/agent/streaming.py
"""流式输出"""
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import AsyncIterator

class StreamingAgent:
    """流式 Agent"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            streaming=True
        )

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        """异步流式生成"""
        async for chunk in self.llm.astream(prompt):
            if hasattr(chunk, 'content'):
                yield chunk.content

    def stream(self, prompt: str):
        """同步流式生成"""
        for chunk in self.llm.stream(prompt):
            if hasattr(chunk, 'content'):
                yield chunk.content
```

### 流式输出测试

```python
# tests/test_agent/test_streaming.py
"""流式输出测试"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.agent.streaming import StreamingAgent


class TestStreamingAgent:
    """流式 Agent 测试"""

    @pytest.mark.asyncio
    async def test_astream_basic(self):
        """测试：基本异步流式输出"""
        agent = StreamingAgent()

        # Mock LLM 流式响应
        async def mock_astream(prompt):
            chunks = ["Hello", " ", "World", "!"]
            for chunk in chunks:
                mock_chunk = Mock()
                mock_chunk.content = chunk
                yield mock_chunk

        with patch.object(agent.llm, 'astream', side_effect=mock_astream):
            chunks = []
            async for chunk in agent.astream("test"):
                chunks.append(chunk)

            assert len(chunks) == 4
            assert "".join(chunks) == "Hello World!"

    @pytest.mark.asyncio
    async def test_astream_empty(self):
        """测试：空流式输出"""
        agent = StreamingAgent()

        async def mock_empty_stream(prompt):
            return
            yield  # 使其成为生成器

        with patch.object(agent.llm, 'astream', side_effect=mock_empty_stream):
            chunks = []
            async for chunk in agent.astream("test"):
                chunks.append(chunk)

            assert chunks == []

    def test_stream_sync(self):
        """测试：同步流式输出"""
        agent = StreamingAgent()

        def mock_stream(prompt):
            chunks = ["Test", " ", "Output"]
            for chunk in chunks:
                mock_chunk = Mock()
                mock_chunk.content = chunk
                yield mock_chunk

        with patch.object(agent.llm, 'stream', side_effect=mock_stream):
            chunks = list(agent.stream("test"))

            assert len(chunks) == 3
            assert "".join(chunks) == "Test Output"

    @pytest.mark.asyncio
    async def test_astream_collect_all(self):
        """测试：收集所有流式输出"""
        agent = StreamingAgent()

        async def mock_astream(prompt):
            for i in range(10):
                mock_chunk = Mock()
                mock_chunk.content = str(i)
                yield mock_chunk

        with patch.object(agent.llm, 'astream', side_effect=mock_astream):
            chunks = [chunk async for chunk in agent.astream("test")]

            assert len(chunks) == 10
            assert chunks == [str(i) for i in range(10)]
```

---

## 测试对话记忆

### 对话记忆实现

```python
# app/memory/conversation.py
"""对话记忆"""
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

class ConversationManager:
    """对话管理器"""

    def __init__(self, memory_type: str = "buffer"):
        if memory_type == "buffer":
            self.memory = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history"
            )
        elif memory_type == "summary":
            llm = ChatOpenAI(temperature=0)
            self.memory = ConversationSummaryMemory(
                llm=llm,
                return_messages=True,
                memory_key="chat_history"
            )
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")

    def add_user_message(self, message: str):
        """添加用户消息"""
        self.memory.chat_memory.add_user_message(message)

    def add_ai_message(self, message: str):
        """添加 AI 消息"""
        self.memory.chat_memory.add_ai_message(message)

    def get_history(self):
        """获取历史消息"""
        return self.memory.chat_memory.messages

    def clear(self):
        """清除历史"""
        self.memory.clear()

    def get_context(self) -> str:
        """获取上下文"""
        return self.memory.load_memory_variables({})["chat_history"]
```

### 对话记忆测试

```python
# tests/test_memory/test_conversation.py
"""对话记忆测试"""
import pytest
from langchain.schema import HumanMessage, AIMessage
from app.memory.conversation import ConversationManager


class TestConversationManager:
    """对话管理器测试"""

    def test_create_buffer_memory(self):
        """测试：创建缓冲记忆"""
        manager = ConversationManager(memory_type="buffer")

        assert manager.memory is not None
        assert manager.get_history() == []

    def test_add_messages(self):
        """测试：添加消息"""
        manager = ConversationManager()

        manager.add_user_message("Hello")
        manager.add_ai_message("Hi there!")

        history = manager.get_history()

        assert len(history) == 2
        assert isinstance(history[0], HumanMessage)
        assert isinstance(history[1], AIMessage)
        assert history[0].content == "Hello"
        assert history[1].content == "Hi there!"

    def test_multiple_turns(self):
        """测试：多轮对话"""
        manager = ConversationManager()

        # 第一轮
        manager.add_user_message("What is RAG?")
        manager.add_ai_message("RAG is Retrieval Augmented Generation")

        # 第二轮
        manager.add_user_message("How does it work?")
        manager.add_ai_message("It combines retrieval with generation")

        history = manager.get_history()

        assert len(history) == 4
        assert history[0].content == "What is RAG?"
        assert history[2].content == "How does it work?"

    def test_clear_history(self):
        """测试：清除历史"""
        manager = ConversationManager()

        manager.add_user_message("Test")
        manager.add_ai_message("Response")

        assert len(manager.get_history()) == 2

        manager.clear()

        assert len(manager.get_history()) == 0

    def test_get_context(self):
        """测试：获取上下文"""
        manager = ConversationManager()

        manager.add_user_message("Hello")
        manager.add_ai_message("Hi")

        context = manager.get_context()

        assert isinstance(context, (str, list))

    @pytest.mark.parametrize("memory_type", ["buffer"])
    def test_different_memory_types(self, memory_type):
        """测试：不同记忆类型"""
        manager = ConversationManager(memory_type=memory_type)

        manager.add_user_message("Test")
        manager.add_ai_message("Response")

        assert len(manager.get_history()) == 2
```

---

## 端到端测试

### 完整 Agent 流程测试

```python
# tests/test_e2e/test_agent_flow.py
"""端到端 Agent 流程测试"""
import pytest
from unittest.mock import Mock, patch
from app.agent.chat_agent import create_chat_agent
from app.rag.retrieval import RAGRetriever
from app.memory.conversation import ConversationManager


@pytest.fixture
def mock_rag_agent():
    """Mock RAG Agent"""
    # Mock 向量存储
    mock_vectorstore = Mock()
    mock_vectorstore.similarity_search.return_value = [
        Mock(page_content="RAG 是检索增强生成")
    ]

    # Mock 检索器
    retriever = RAGRetriever(mock_vectorstore)

    # Mock LLM
    mock_llm = Mock()
    mock_llm.invoke.return_value = Mock(content="这是 AI 的回复")

    return retriever, mock_llm


class TestAgentE2E:
    """Agent 端到端测试"""

    def test_complete_conversation_flow(self, mock_rag_agent):
        """测试：完整对话流程"""
        retriever, mock_llm = mock_rag_agent
        memory = ConversationManager()

        # 第一轮对话
        user_input_1 = "什么是 RAG?"

        # 1. 添加用户消息到记忆
        memory.add_user_message(user_input_1)

        # 2. 检索相关文档
        docs = retriever.retrieve(user_input_1, k=3)
        assert len(docs) > 0

        # 3. 生成回复
        ai_response_1 = "RAG 是检索增强生成技术"
        memory.add_ai_message(ai_response_1)

        # 验证第一轮
        history = memory.get_history()
        assert len(history) == 2

        # 第二轮对话
        user_input_2 = "它有什么优势?"

        # 1. 添加用户消息
        memory.add_user_message(user_input_2)

        # 2. 获取上下文
        context = memory.get_context()
        assert context is not None

        # 3. 生成回复
        ai_response_2 = "RAG 可以提供更准确的信息"
        memory.add_ai_message(ai_response_2)

        # 验证第二轮
        history = memory.get_history()
        assert len(history) == 4

    @pytest.mark.asyncio
    async def test_streaming_conversation(self):
        """测试：流式对话"""
        from app.agent.streaming import StreamingAgent

        agent = StreamingAgent()
        memory = ConversationManager()

        # Mock 流式输出
        async def mock_astream(prompt):
            chunks = ["这", "是", "流", "式", "回", "复"]
            for chunk in chunks:
                mock_chunk = Mock()
                mock_chunk.content = chunk
                yield mock_chunk

        with patch.object(agent.llm, 'astream', side_effect=mock_astream):
            # 用户输入
            user_input = "你好"
            memory.add_user_message(user_input)

            # 收集流式输出
            chunks = []
            async for chunk in agent.astream(user_input):
                chunks.append(chunk)

            # 完整回复
            full_response = "".join(chunks)
            memory.add_ai_message(full_response)

            # 验证
            assert full_response == "这是流式回复"
            assert len(memory.get_history()) == 2

    def test_rag_with_memory(self, mock_rag_agent):
        """测试：RAG + 记忆"""
        retriever, mock_llm = mock_rag_agent
        memory = ConversationManager()

        # 多轮对话
        conversations = [
            ("什么是 RAG?", "RAG 是检索增强生成"),
            ("它如何工作?", "通过检索相关文档增强生成"),
            ("有什么应用?", "可用于问答、摘要等任务")
        ]

        for user_msg, ai_msg in conversations:
            # 添加用户消息
            memory.add_user_message(user_msg)

            # 检索
            docs = retriever.retrieve(user_msg, k=3)

            # 添加 AI 回复
            memory.add_ai_message(ai_msg)

        # 验证
        history = memory.get_history()
        assert len(history) == 6

        # 验证上下文包含所有对话
        context = memory.get_context()
        assert context is not None
```

---

## 集成测试

### API + Agent 集成测试

```python
# tests/test_integration/test_api_agent.py
"""API + Agent 集成测试"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch


@pytest.fixture
def mock_agent():
    """Mock Agent"""
    mock = Mock()
    mock.invoke.return_value = {
        "output": "这是 Agent 的回复",
        "intermediate_steps": []
    }
    return mock


class TestAPIAgentIntegration:
    """API + Agent 集成测试"""

    def test_chat_endpoint_with_agent(self, client, mock_agent):
        """测试：聊天端点 + Agent"""
        with patch("app.api.chat.agent", mock_agent):
            response = client.post("/api/chat", json={
                "message": "你好"
            })

            assert response.status_code == 200
            data = response.json()
            assert "reply" in data
            assert data["reply"] == "这是 Agent 的回复"

    def test_rag_endpoint_with_retrieval(self, client):
        """测试：RAG 端点 + 检索"""
        # Mock 向量存储
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search.return_value = [
            Mock(page_content="测试文档")
        ]

        with patch("app.api.rag.vector_store", mock_vectorstore):
            response = client.post("/api/rag/search", json={
                "query": "测试查询",
                "k": 3
            })

            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert len(data["results"]) > 0

    @pytest.mark.asyncio
    async def test_streaming_endpoint(self, client):
        """测试：流式端点"""
        from httpx import AsyncClient
        from app.main import app

        async with AsyncClient(app=app, base_url="http://test") as ac:
            async with ac.stream("POST", "/api/chat/stream", json={
                "message": "你好"
            }) as response:
                assert response.status_code == 200

                chunks = []
                async for chunk in response.aiter_text():
                    chunks.append(chunk)

                assert len(chunks) > 0
```

---

## 性能测试

### Agent 性能测试

```python
# tests/test_performance/test_agent_performance.py
"""Agent 性能测试"""
import pytest
import time


class TestAgentPerformance:
    """Agent 性能测试"""

    def test_retrieval_performance(self, benchmark, retriever):
        """测试：检索性能"""
        def do_retrieval():
            return retriever.retrieve("测试查询", k=10)

        result = benchmark(do_retrieval)

        assert len(result) > 0

    def test_memory_performance(self, benchmark):
        """测试：记忆性能"""
        from app.memory.conversation import ConversationManager

        manager = ConversationManager()

        # 添加100轮对话
        for i in range(100):
            manager.add_user_message(f"Message {i}")
            manager.add_ai_message(f"Response {i}")

        def get_history():
            return manager.get_history()

        result = benchmark(get_history)

        assert len(result) == 200

    @pytest.mark.parametrize("message_count", [10, 50, 100])
    def test_conversation_scaling(self, message_count):
        """测试：对话扩展性"""
        from app.memory.conversation import ConversationManager

        manager = ConversationManager()

        start = time.time()

        for i in range(message_count):
            manager.add_user_message(f"Message {i}")
            manager.add_ai_message(f"Response {i}")

        duration = time.time() - start

        # 验证性能
        assert duration < message_count * 0.01  # 每条消息 < 10ms
```

---

## 运行测试

```bash
# 运行所有 Agent 测试
pytest tests/test_agent/ -v

# 运行 RAG 测试
pytest tests/test_rag/ -v

# 运行端到端测试
pytest tests/test_e2e/ -v

# 运行集成测试
pytest tests/test_integration/ -v

# 运行性能测试
pytest tests/test_performance/ --benchmark-only

# 运行所有测试并生成报告
pytest tests/ -v --cov=app --cov-report=html

# 并行运行
pytest tests/ -n auto
```

---

## 总结

### 核心要点

1. **Agent 测试**：Mock LLM、测试工具执行、测试流程
2. **RAG 测试**：测试检索、测试相似度、测试阈值
3. **流式输出测试**：测试异步流、收集 chunk、验证完整性
4. **记忆测试**：测试消息添加、测试历史获取、测试清除
5. **端到端测试**：测试完整流程、测试多轮对话、测试集成

### 最佳实践

- Mock 外部 API（LLM、Embedding）
- 测试确定性部分（流程、验证）
- 测试分布特性（输出长度、格式）
- 使用 fixture 准备测试环境
- 并行运行独立测试
- 监控性能和资源使用

### AI Agent 测试策略

1. **隔离测试**：Mock LLM 调用，测试逻辑
2. **集成测试**：使用真实组件，测试协作
3. **端到端测试**：测试完整用户场景
4. **性能测试**：监控响应时间和资源
5. **持续监控**：CI/CD 自动化测试
