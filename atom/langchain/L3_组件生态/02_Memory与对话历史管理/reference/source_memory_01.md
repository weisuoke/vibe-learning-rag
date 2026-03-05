---
type: source_code_analysis
source: sourcecode/langchain/libs/langchain/langchain_classic/memory
analyzed_files:
  - chat_memory.py
  - __init__.py
  - buffer.py
  - summary.py
analyzed_at: 2026-02-24
knowledge_point: Memory与对话历史管理
---

# 源码分析：LangChain Memory 核心架构

## 分析的文件

- `sourcecode/langchain/libs/langchain/langchain_classic/memory/chat_memory.py` - BaseChatMemory 基类
- `sourcecode/langchain/libs/langchain/langchain_classic/memory/__init__.py` - Memory 模块导出
- `sourcecode/langchain/libs/langchain/langchain_classic/memory/buffer.py` - Buffer Memory 实现
- `sourcecode/langchain/libs/langchain/langchain_classic/memory/summary.py` - Summary Memory 实现

## 关键发现

### 1. **重要警告：Memory 类已被弃用**

**所有 Memory 类都被标记为 `@deprecated`**：
- 自 0.3.1 版本起弃用
- 将在 1.0.0 版本移除
- 官方迁移指南：https://python.langchain.com/docs/versions/migrating_memory/

**弃用原因**：
- 这些抽象在聊天模型有原生工具调用能力之前创建
- **不支持**原生工具调用能力
- 如果与有原生工具调用的聊天模型一起使用，会**静默失败**
- **不要在新代码中使用这些抽象**

### 2. **Memory 类型体系**

从 `__init__.py` 中识别的 Memory 类型：

**基础 Memory 类型**：
- `ConversationBufferMemory` - 简单存储完整对话历史
- `ConversationStringBufferMemory` - 字符串形式的缓冲记忆
- `ConversationBufferWindowMemory` - 窗口缓冲（只保留最近 N 轮对话）
- `SimpleMemory` - 简单键值对记忆
- `ReadOnlySharedMemory` - 只读共享记忆

**高级 Memory 类型**：
- `ConversationSummaryMemory` - 使用 LLM 持续总结对话历史
- `ConversationSummaryBufferMemory` - 摘要+缓冲混合策略
- `ConversationTokenBufferMemory` - 基于 Token 数量限制的缓冲
- `ConversationVectorStoreTokenBufferMemory` - 向量存储+Token 缓冲混合
- `ConversationEntityMemory` - 实体记忆（跟踪对话中的实体）
- `ConversationKGMemory` - 知识图谱记忆
- `VectorStoreRetrieverMemory` - 向量存储检索记忆
- `CombinedMemory` - 组合多种记忆类型

### 3. **BaseChatMemory 核心设计**

**核心属性**：
```python
class BaseChatMemory(BaseMemory, ABC):
    chat_memory: BaseChatMessageHistory = Field(
        default_factory=InMemoryChatMessageHistory,
    )
    output_key: str | None = None
    input_key: str | None = None
    return_messages: bool = False
```

**核心方法**：
- `save_context(inputs, outputs)` - 保存对话上下文
- `asave_context(inputs, outputs)` - 异步保存上下文
- `clear()` - 清空记忆
- `aclear()` - 异步清空记忆
- `_get_input_output(inputs, outputs)` - 提取输入输出键

**设计模式**：
- 使用 `BaseChatMessageHistory` 作为底层存储抽象
- 支持同步和异步操作
- 自动处理输入输出键的提取

### 4. **ConversationBufferMemory 实现**

**核心特性**：
```python
class ConversationBufferMemory(BaseChatMemory):
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"

    @property
    def buffer(self) -> Any:
        """String buffer of memory."""
        return self.buffer_as_messages if self.return_messages else self.buffer_as_str
```

**两种输出格式**：
1. **字符串格式** (`buffer_as_str`)：
   ```
   Human: 用户输入
   AI: AI 回复
   ```

2. **消息列表格式** (`buffer_as_messages`)：
   ```python
   [HumanMessage(content="..."), AIMessage(content="...")]
   ```

**使用场景**：
- 最简单的记忆实现
- 适合短对话场景
- 不适合长对话（会超出上下文窗口）

### 5. **ConversationSummaryMemory 实现**

**核心特性**：
```python
class ConversationSummaryMemory(BaseChatMemory, SummarizerMixin):
    buffer: str = ""
    memory_key: str = "history"
    llm: BaseLanguageModel
    prompt: BasePromptTemplate = SUMMARY_PROMPT
```

**工作原理**：
1. 每次对话后，使用 LLM 生成摘要
2. 摘要基于现有摘要和新对话内容
3. 只保存摘要，不保存完整历史

**关键方法**：
```python
def predict_new_summary(
    self,
    messages: list[BaseMessage],
    existing_summary: str,
) -> str:
    """基于消息和现有摘要预测新摘要"""
    new_lines = get_buffer_string(messages, ...)
    chain = LLMChain(llm=self.llm, prompt=self.prompt)
    return chain.predict(summary=existing_summary, new_lines=new_lines)
```

**使用场景**：
- 长对话场景
- 需要压缩对话历史
- 可以牺牲细节换取上下文窗口

### 6. **对话历史存储抽象**

**核心接口**：`BaseChatMessageHistory`（来自 `langchain_core.chat_history`）

**内存存储**：
- `InMemoryChatMessageHistory` - 默认的内存存储

**持久化存储**（来自 `langchain_community`）：
- `CassandraChatMessageHistory` - Cassandra 数据库
- `ElasticsearchChatMessageHistory` - Elasticsearch
- `FileChatMessageHistory` - 文件系统
- `MongoDBChatMessageHistory` - MongoDB
- `PostgresChatMessageHistory` - PostgreSQL
- `RedisChatMessageHistory` - Redis
- `SQLChatMessageHistory` - SQL 数据库
- `DynamoDBChatMessageHistory` - AWS DynamoDB
- `CosmosDBChatMessageHistory` - Azure Cosmos DB
- 等等

### 7. **上下文窗口管理策略**

从源码中识别的策略：

1. **窗口策略** (`ConversationBufferWindowMemory`)：
   - 只保留最近 N 轮对话
   - 简单高效

2. **Token 限制策略** (`ConversationTokenBufferMemory`)：
   - 基于 Token 数量限制
   - 更精确的上下文控制

3. **摘要策略** (`ConversationSummaryMemory`)：
   - 使用 LLM 压缩历史
   - 保留语义信息

4. **混合策略** (`ConversationSummaryBufferMemory`)：
   - 结合摘要和缓冲
   - 平衡细节和容量

5. **向量检索策略** (`VectorStoreRetrieverMemory`)：
   - 使用向量存储检索相关历史
   - 适合长期记忆

## 代码片段

### BaseChatMemory 核心实现

```python
@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class BaseChatMemory(BaseMemory, ABC):
    """Abstract base class for chat memory.

    **ATTENTION** This abstraction was created prior to when chat models had
        native tool calling capabilities.
        It does **NOT** support native tool calling capabilities for chat models and
        will fail SILENTLY if used with a chat model that has native tool calling.

    DO NOT USE THIS ABSTRACTION FOR NEW CODE.
    """

    chat_memory: BaseChatMessageHistory = Field(
        default_factory=InMemoryChatMessageHistory,
    )
    output_key: str | None = None
    input_key: str | None = None
    return_messages: bool = False

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

### ConversationBufferMemory 实现

```python
@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class ConversationBufferMemory(BaseChatMemory):
    """A basic memory implementation that simply stores the conversation history.

    This stores the entire conversation history in memory without any
    additional processing.

    Note that additional processing may be required in some situations when the
    conversation history is too large to fit in the context window of the model.
    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"

    @property
    def buffer(self) -> Any:
        """String buffer of memory."""
        return self.buffer_as_messages if self.return_messages else self.buffer_as_str

    @property
    def buffer_as_str(self) -> str:
        """Exposes the buffer as a string in case return_messages is True."""
        return self._buffer_as_str(self.chat_memory.messages)

    @property
    def buffer_as_messages(self) -> list[BaseMessage]:
        """Exposes the buffer as a list of messages in case return_messages is False."""
        return self.chat_memory.messages

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Return history buffer."""
        return {self.memory_key: self.buffer}
```

### ConversationSummaryMemory 实现

```python
@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class ConversationSummaryMemory(BaseChatMemory, SummarizerMixin):
    """Continually summarizes the conversation history.

    The summary is updated after each conversation turn.
    The implementations returns a summary of the conversation history which
    can be used to provide context to the model.
    """

    buffer: str = ""
    memory_key: str = "history"

    def predict_new_summary(
        self,
        messages: list[BaseMessage],
        existing_summary: str,
    ) -> str:
        """Predict a new summary based on the messages and existing summary."""
        new_lines = get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        return chain.predict(summary=existing_summary, new_lines=new_lines)

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        super().save_context(inputs, outputs)
        self.buffer = self.predict_new_summary(
            self.chat_memory.messages[-2:],
            self.buffer,
        )
```

## 架构设计要点

### 1. **分层设计**

```
BaseMemory (抽象基类)
    ↓
BaseChatMemory (聊天记忆基类)
    ↓
具体实现 (ConversationBufferMemory, ConversationSummaryMemory, etc.)
```

### 2. **存储抽象**

```
BaseChatMessageHistory (存储接口)
    ↓
InMemoryChatMessageHistory (内存实现)
持久化实现 (Redis, MongoDB, PostgreSQL, etc.)
```

### 3. **策略模式**

不同的 Memory 类型实现不同的上下文管理策略：
- Buffer: 完整存储
- Window: 窗口截断
- Summary: LLM 摘要
- Token: Token 限制
- Vector: 向量检索

## 迁移建议

**官方迁移指南**：https://python.langchain.com/docs/versions/migrating_memory/

**新代码建议**：
1. 使用 LangGraph 的状态管理
2. 直接使用 `BaseChatMessageHistory` 和消息列表
3. 手动实现上下文窗口管理逻辑
4. 避免使用 `BaseMemory` 和 `BaseChatMemory` 抽象

## 总结

LangChain 的 Memory 系统提供了丰富的对话历史管理策略，但由于架构限制（不支持工具调用），官方已经弃用这些抽象。新代码应该使用更现代的方法（如 LangGraph）来管理对话状态。

**核心价值**：
- 理解不同的上下文管理策略
- 学习如何设计记忆系统
- 了解迁移路径和替代方案

**学习重点**：
- 各种 Memory 类型的原理和适用场景
- 对话历史存储的抽象设计
- 上下文窗口管理的策略
- 如何迁移到新的架构
