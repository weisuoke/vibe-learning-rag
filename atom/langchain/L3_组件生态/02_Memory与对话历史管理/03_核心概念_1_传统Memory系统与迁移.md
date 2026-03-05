# 核心概念：传统 Memory 系统与迁移

> **重要警告**：本文档讲解的传统 Memory 系统已被弃用，仅用于理解历史和迁移。新代码请使用 RunnableWithMessageHistory 或 LangGraph Checkpointer。

---

## 一、传统 Memory 系统概述

### 1.1 什么是传统 Memory 系统

传统 Memory 系统是 LangChain 早期设计的对话历史管理机制，通过 `BaseMemory` 和 `BaseChatMemory` 抽象类提供了多种 Memory 类型。

**核心设计**：
```python
BaseMemory (抽象基类)
    ↓
BaseChatMemory (聊天记忆基类)
    ↓
具体实现 (ConversationBufferMemory, ConversationSummaryMemory, etc.)
```

### 1.2 为什么被弃用

**弃用原因**：
1. **不支持原生工具调用**：
   - 架构设计在聊天模型有原生工具调用能力之前
   - 无法与现代 LLM 的工具调用功能兼容
   - 会静默失败，不报错但功能异常

2. **不支持 LCEL 管道集成**：
   - 无法在 LCEL 管道中使用
   - 缺乏 Runnable 接口
   - 无法与现代 LangChain 生态集成

3. **架构限制**：
   - 紧耦合到 Chain
   - 缺乏灵活性和可扩展性
   - 无法适应现代 LLM 应用需求

**弃用时间线**：
- 自 0.3.1 版本起标记为 `@deprecated`
- 将在 1.0.0 版本移除
- 官方迁移指南：https://python.langchain.com/docs/versions/migrating_memory/

---

## 二、BaseChatMemory 核心设计

### 2.1 核心属性

```python
class BaseChatMemory(BaseMemory, ABC):
    chat_memory: BaseChatMessageHistory = Field(
        default_factory=InMemoryChatMessageHistory,
    )
    output_key: str | None = None
    input_key: str | None = None
    return_messages: bool = False
```

**属性说明**：
- `chat_memory`：底层存储，默认使用 `InMemoryChatMessageHistory`
- `output_key`：从输出字典中提取哪个键作为 AI 响应
- `input_key`：从输入字典中提取哪个键作为用户输入
- `return_messages`：是否返回消息列表（True）还是字符串（False）

### 2.2 核心方法

**保存上下文**：
```python
def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
    """Save context from this conversation to buffer."""
    input_str, output_str = self._get_input_output(inputs, outputs)
    self.chat_memory.add_messages(
        [
            HumanMessage(content=input_str),
            AIMessage(content=output_str),
        ],
    )
```

**加载记忆变量**：
```python
def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
    """Return history buffer."""
    return {self.memory_key: self.buffer}
```

**清空记忆**：
```python
def clear(self) -> None:
    """Clear memory contents."""
    self.chat_memory.clear()
```

### 2.3 设计模式

**存储抽象**：
- 使用 `BaseChatMessageHistory` 作为底层存储抽象
- 支持多种存储后端（内存、Redis、PostgreSQL 等）
- 通过依赖注入实现可替换性

**键提取机制**：
- 自动从输入输出字典中提取相关键
- 支持手动指定 `input_key` 和 `output_key`
- 处理多键情况的警告和错误

---

## 三、12种 Memory 类型详解

### 3.1 ConversationBufferMemory

**核心特性**：
- 最简单的记忆实现
- 完整存储所有对话历史
- 不做任何压缩或处理

**实现原理**：
```python
class ConversationBufferMemory(BaseChatMemory):
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"

    @property
    def buffer(self) -> Any:
        return self.buffer_as_messages if self.return_messages else self.buffer_as_str
```

**使用场景**：
- 短对话（<10轮）
- 开发和测试
- 不关心上下文窗口限制

**优缺点**：
- ✅ 简单直接，易于理解
- ✅ 信息完整，无损失
- ❌ 占用空间大
- ❌ 容易超出上下文窗口

[来源: sourcecode/langchain/libs/langchain/langchain_classic/memory/buffer.py]

### 3.2 ConversationBufferWindowMemory

**核心特性**：
- 只保留最近 N 轮对话
- 使用滑动窗口策略
- 自动丢弃旧消息

**实现原理**：
```python
class ConversationBufferWindowMemory(BaseChatMemory):
    k: int = 5  # 保留最近 k 轮对话

    @property
    def buffer(self) -> Any:
        messages = self.chat_memory.messages
        # 只返回最近 k*2 条消息（k 轮对话 = k 个用户消息 + k 个 AI 消息）
        return messages[-self.k * 2:]
```

**使用场景**：
- 中等长度对话（10-50轮）
- 需要控制上下文窗口
- 不需要完整历史

**优缺点**：
- ✅ 简单高效
- ✅ 自动控制上下文窗口
- ❌ 丢失旧信息
- ❌ 无法访问早期对话

[来源: sourcecode/langchain/libs/langchain/langchain_classic/memory/buffer_window.py]

### 3.3 ConversationSummaryMemory

**核心特性**：
- 使用 LLM 持续总结对话历史
- 只保存摘要，不保存完整历史
- 每次对话后更新摘要

**实现原理**：
```python
class ConversationSummaryMemory(BaseChatMemory, SummarizerMixin):
    buffer: str = ""
    llm: BaseLanguageModel
    prompt: BasePromptTemplate = SUMMARY_PROMPT

    def save_context(self, inputs, outputs):
        super().save_context(inputs, outputs)
        # 使用 LLM 生成新摘要
        self.buffer = self.predict_new_summary(
            self.chat_memory.messages[-2:],
            self.buffer,
        )
```

**使用场景**：
- 长对话（>50轮）
- 需要压缩历史
- 可以牺牲细节

**优缺点**：
- ✅ 节省空间
- ✅ 保留语义信息
- ❌ 需要额外 LLM 调用
- ❌ 信息损失
- ❌ 成本较高

[来源: sourcecode/langchain/libs/langchain/langchain_classic/memory/summary.py]

### 3.4 ConversationSummaryBufferMemory

**核心特性**：
- 结合摘要和缓冲策略
- 保留最近的完整消息
- 总结旧消息

**实现原理**：
```python
class ConversationSummaryBufferMemory(BaseChatMemory, SummarizerMixin):
    max_token_limit: int = 2000
    moving_summary_buffer: str = ""

    def save_context(self, inputs, outputs):
        super().save_context(inputs, outputs)
        # 如果超过 token 限制，总结旧消息
        if self._get_num_tokens() > self.max_token_limit:
            self._prune_buffer()
```

**使用场景**：
- 长对话
- 需要平衡细节和容量
- 重要的最近对话需要完整保留

**优缺点**：
- ✅ 平衡细节和容量
- ✅ 保留最近完整信息
- ❌ 复杂度较高
- ❌ 需要 LLM 调用

[来源: sourcecode/langchain/libs/langchain/langchain_classic/memory/summary_buffer.py]

### 3.5 ConversationTokenBufferMemory

**核心特性**：
- 基于 Token 数量限制
- 自动移除旧消息以保持在限制内
- 更精确的上下文控制

**实现原理**：
```python
class ConversationTokenBufferMemory(BaseChatMemory):
    max_token_limit: int = 2000
    llm: BaseLanguageModel

    def save_context(self, inputs, outputs):
        super().save_context(inputs, outputs)
        # 计算 token 数量
        while self._get_num_tokens() > self.max_token_limit:
            # 移除最旧的消息
            self.chat_memory.messages.pop(0)
```

**使用场景**：
- 需要精确控制 Token 数量
- 不同模型有不同上下文窗口
- 成本敏感的应用

**优缺点**：
- ✅ 精确控制 Token
- ✅ 适应不同模型
- ❌ 需要 LLM 计算 Token
- ❌ 丢失旧信息

[来源: sourcecode/langchain/libs/langchain/langchain_classic/memory/token_buffer.py]

### 3.6 ConversationEntityMemory

**核心特性**：
- 跟踪对话中的实体（人名、地名、组织等）
- 为每个实体维护独立的记忆
- 支持实体级别的信息检索

**实现原理**：
```python
class ConversationEntityMemory(BaseChatMemory):
    entity_store: BaseEntityStore
    llm: BaseLanguageModel

    def save_context(self, inputs, outputs):
        # 提取实体
        entities = self._extract_entities(inputs, outputs)
        # 为每个实体更新记忆
        for entity in entities:
            self.entity_store.set(entity, info)
```

**使用场景**：
- 需要跟踪多个实体
- 客户服务（跟踪客户信息）
- 复杂对话场景

**优缺点**：
- ✅ 结构化信息
- ✅ 实体级别检索
- ❌ 复杂度高
- ❌ 需要实体提取

[来源: sourcecode/langchain/libs/langchain/langchain_classic/memory/entity.py]

### 3.7 ConversationKGMemory

**核心特性**：
- 使用知识图谱存储对话信息
- 提取实体和关系
- 支持图查询

**实现原理**：
```python
class ConversationKGMemory(BaseChatMemory):
    kg: KnowledgeGraph
    llm: BaseLanguageModel

    def save_context(self, inputs, outputs):
        # 提取三元组 (主体, 关系, 客体)
        triples = self._extract_triples(inputs, outputs)
        # 添加到知识图谱
        for triple in triples:
            self.kg.add_triple(triple)
```

**使用场景**：
- 复杂关系跟踪
- 知识密集型对话
- 需要推理的场景

**优缺点**：
- ✅ 结构化知识
- ✅ 支持推理
- ❌ 非常复杂
- ❌ 需要知识图谱

[来源: sourcecode/langchain/libs/langchain/langchain_classic/memory/kg.py]

### 3.8 VectorStoreRetrieverMemory

**核心特性**：
- 使用向量存储检索相关历史
- 基于语义相似度检索
- 适合长期记忆

**实现原理**：
```python
class VectorStoreRetrieverMemory(BaseMemory):
    retriever: VectorStoreRetriever

    def save_context(self, inputs, outputs):
        # 将对话存储到向量数据库
        text = f"{inputs} {outputs}"
        self.retriever.add_documents([Document(page_content=text)])

    def load_memory_variables(self, inputs):
        # 检索相关历史
        docs = self.retriever.get_relevant_documents(inputs)
        return {"history": docs}
```

**使用场景**：
- 长期记忆
- 大量历史数据
- 需要语义检索

**优缺点**：
- ✅ 长期记忆
- ✅ 语义检索
- ❌ 需要向量数据库
- ❌ 检索可能不准确

[来源: sourcecode/langchain/libs/langchain/langchain_classic/memory/vectorstore.py]

### 3.9 ConversationVectorStoreTokenBufferMemory

**核心特性**：
- 结合向量存储和 Token 缓冲
- 最近的消息保留在缓冲中
- 旧消息存储到向量数据库

**使用场景**：
- 长对话 + 长期记忆
- 需要快速访问最近消息
- 需要检索旧消息

**优缺点**：
- ✅ 结合短期和长期记忆
- ✅ 灵活的检索策略
- ❌ 非常复杂
- ❌ 需要向量数据库

[来源: sourcecode/langchain/libs/langchain/langchain_classic/memory/vectorstore_token_buffer_memory.py]

### 3.10 CombinedMemory

**核心特性**：
- 组合多种 Memory 类型
- 支持多个记忆源
- 统一的接口

**实现原理**：
```python
class CombinedMemory(BaseMemory):
    memories: list[BaseMemory]

    def load_memory_variables(self, inputs):
        # 从所有记忆源加载
        result = {}
        for memory in self.memories:
            result.update(memory.load_memory_variables(inputs))
        return result
```

**使用场景**：
- 需要多种记忆策略
- 复杂应用场景
- 实验不同组合

**优缺点**：
- ✅ 灵活组合
- ✅ 多种策略
- ❌ 复杂度高
- ❌ 可能冲突

[来源: sourcecode/langchain/libs/langchain/langchain_classic/memory/combined.py]

### 3.11 SimpleMemory

**核心特性**：
- 简单的键值对存储
- 不处理对话历史
- 用于存储静态信息

**使用场景**：
- 存储用户偏好
- 存储配置信息
- 不需要对话历史

**优缺点**：
- ✅ 非常简单
- ✅ 适合静态信息
- ❌ 不处理对话

[来源: sourcecode/langchain/libs/langchain/langchain_classic/memory/simple.py]

### 3.12 ReadOnlySharedMemory

**核心特性**：
- 只读的共享记忆
- 多个 Chain 共享同一记忆
- 防止意外修改

**使用场景**：
- 多个 Chain 共享信息
- 需要只读访问
- 防止意外修改

**优缺点**：
- ✅ 安全的共享
- ✅ 防止修改
- ❌ 只读限制

[来源: sourcecode/langchain/libs/langchain/langchain_classic/memory/readonly.py]

---

## 四、对话历史存储抽象

### 4.1 BaseChatMessageHistory

**核心接口**：
```python
class BaseChatMessageHistory:
    def add_message(self, message: BaseMessage) -> None: ...
    def get_messages(self) -> list[BaseMessage]: ...
    def clear(self) -> None: ...
```

### 4.2 内存存储

**InMemoryChatMessageHistory**：
- 默认的内存存储
- 简单快速
- 不持久化

### 4.3 持久化存储

**支持的存储后端**（来自 `langchain_community`）：
- **CassandraChatMessageHistory**：Cassandra 数据库
- **ElasticsearchChatMessageHistory**：Elasticsearch
- **FileChatMessageHistory**：文件系统
- **MongoDBChatMessageHistory**：MongoDB
- **PostgresChatMessageHistory**：PostgreSQL
- **RedisChatMessageHistory**：Redis
- **SQLChatMessageHistory**：SQL 数据库
- **DynamoDBChatMessageHistory**：AWS DynamoDB
- **CosmosDBChatMessageHistory**：Azure Cosmos DB

[来源: sourcecode/langchain/libs/langchain/langchain_classic/memory/__init__.py]

---

## 五、迁移路径

### 5.1 从 ConversationBufferMemory 迁移

**旧方式**（已弃用）：
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory)
```

**新方式**（推荐）：
```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversation = RunnableWithMessageHistory(
    chat_model,
    get_session_history,
)

config = {"configurable": {"session_id": "1"}}
conversation.invoke("message", config=config)
```

[来源: reference/context7_langchain_01.md | LangChain 官方文档]

### 5.2 从 ConversationSummaryMemory 迁移

**旧方式**（已弃用）：
```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)
chain = ConversationChain(llm=llm, memory=memory)
```

**新方式**（推荐）：
```python
from langgraph.graph import StateGraph
from langmem.short_term import SummarizationNode

# 使用 LangGraph + SummarizationNode
summarization_node = SummarizationNode(
    model=summarization_model,
    max_tokens_before_summary=256,
    max_summary_tokens=128,
)

builder = StateGraph(State)
builder.add_node("summarize", summarization_node)
graph = builder.compile(checkpointer=checkpointer)
```

[来源: reference/context7_langchain_01.md | LangChain 官方文档]

### 5.3 迁移检查清单

**迁移前**：
- [ ] 识别当前使用的 Memory 类型
- [ ] 理解当前的使用场景
- [ ] 评估迁移的复杂度

**迁移中**：
- [ ] 选择新的存储方案（内存或持久化）
- [ ] 实现 session 管理函数
- [ ] 使用 RunnableWithMessageHistory 包装模型
- [ ] 更新调用代码以传递 config

**迁移后**：
- [ ] 测试迁移后的功能
- [ ] 验证对话历史正确保存和加载
- [ ] 性能测试
- [ ] 移除旧代码

---

## 六、常见问题

### Q1: 为什么不能继续使用传统 Memory？

**A**: 传统 Memory 类不支持原生工具调用，会静默失败。虽然代码可以运行，但功能会异常，导致难以调试的问题。

### Q2: 迁移会破坏现有代码吗？

**A**: 是的，迁移需要修改代码。但官方提供了详细的迁移指南和示例，迁移过程相对简单。

### Q3: 新方案比旧方案复杂吗？

**A**: 新方案更灵活，但基础用法并不复杂。对于简单场景，代码量相当。对于复杂场景，新方案提供了更好的支持。

### Q4: 可以混用新旧方案吗？

**A**: 不推荐。新旧方案的接口不兼容，混用会导致混乱。建议完全迁移到新方案。

---

## 数据来源

- [来源: sourcecode/langchain/libs/langchain/langchain_classic/memory/]
- [来源: reference/context7_langchain_01.md | LangChain 官方文档]
- [来源: reference/search_memory_01.md]

---

**版本**：v1.0
**最后更新**：2026-02-24
**维护者**：Claude Code
