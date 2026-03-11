---
type: source_code_analysis
source: sourcecode/langchain
analyzed_files:
  - libs/langchain/langchain_classic/base_memory.py
  - libs/langchain/langchain_classic/memory/chat_memory.py
  - libs/langchain/langchain_classic/memory/buffer.py
  - libs/langchain/langchain_classic/memory/buffer_window.py
  - libs/langchain/langchain_classic/memory/token_buffer.py
  - libs/langchain/langchain_classic/memory/summary.py
  - libs/langchain/langchain_classic/memory/summary_buffer.py
  - libs/langchain/langchain_classic/memory/entity.py
  - libs/langchain/langchain_classic/memory/vectorstore.py
  - libs/langchain/langchain_classic/memory/combined.py
  - libs/langchain/langchain_classic/memory/simple.py
  - libs/langchain/langchain_classic/memory/readonly.py
  - libs/langchain/langchain_classic/agents/conversational/base.py
  - libs/langchain/langchain_classic/agents/conversational_chat/base.py
  - libs/langchain/langchain_classic/agents/openai_functions_agent/agent_token_buffer_memory.py
analyzed_at: 2026-03-06
knowledge_point: 07_Agent Memory集成
---

# 源码分析：LangChain Agent Memory 架构

## 分析的文件

### 核心抽象层
- `langchain_classic/base_memory.py` - BaseMemory 抽象基类
- `langchain_classic/memory/chat_memory.py` - BaseChatMemory 聊天记忆基类

### Memory 实现层
- `langchain_classic/memory/buffer.py` - ConversationBufferMemory（全量缓冲）
- `langchain_classic/memory/buffer_window.py` - ConversationBufferWindowMemory（滑动窗口）
- `langchain_classic/memory/token_buffer.py` - ConversationTokenBufferMemory（Token限制）
- `langchain_classic/memory/summary.py` - ConversationSummaryMemory（摘要）
- `langchain_classic/memory/summary_buffer.py` - ConversationSummaryBufferMemory（混合）
- `langchain_classic/memory/entity.py` - ConversationEntityMemory（实体追踪）
- `langchain_classic/memory/vectorstore.py` - VectorStoreRetrieverMemory（语义检索）
- `langchain_classic/memory/combined.py` - CombinedMemory（组合记忆）
- `langchain_classic/memory/simple.py` - SimpleMemory（静态记忆）
- `langchain_classic/memory/readonly.py` - ReadOnlySharedMemory（只读包装）

### Agent 集成层
- `langchain_classic/agents/conversational/base.py` - ConversationalAgent
- `langchain_classic/agents/conversational_chat/base.py` - ConversationalChatAgent
- `langchain_classic/agents/openai_functions_agent/agent_token_buffer_memory.py` - AgentTokenBufferMemory

## 关键发现

### 1. BaseMemory 核心接口

```python
class BaseMemory(ABC):
    """所有 Memory 实现的抽象基类"""

    @property
    @abstractmethod
    def memory_variables(self) -> list[str]:
        """返回此 memory 注入 chain 输入的键名列表"""

    @abstractmethod
    def load_memory_variables(self, inputs: dict) -> dict:
        """加载记忆变量，返回要注入 prompt 的数据"""

    @abstractmethod
    def save_context(self, inputs: dict, outputs: dict) -> None:
        """保存对话上下文（输入和输出）"""

    def clear(self) -> None:
        """清除记忆内容"""

    # 异步版本
    async def aload_memory_variables(self, inputs: dict) -> dict: ...
    async def asave_context(self, inputs: dict, outputs: dict) -> None: ...
    async def aclear(self) -> None: ...
```

### 2. Memory 三阶段生命周期

1. **Load 阶段**（Agent 步骤前）：
   - 调用 `memory.load_memory_variables(inputs)`
   - 返回包含对话历史的字典
   - 历史被注入到 prompt 模板中

2. **Save 阶段**（Agent 步骤后）：
   - 调用 `memory.save_context(inputs, outputs)`
   - 存储本轮交互
   - 应用裁剪/摘要逻辑

3. **Clear 阶段**（可选）：
   - `memory.clear()` 移除所有存储数据
   - 用于开始新对话

### 3. Memory 返回格式

Memory 支持两种返回格式：
- **字符串格式**（默认）：`"Human: ...\nAI: ..."`
- **消息对象格式**：`[HumanMessage(...), AIMessage(...)]`
- 通过 `return_messages` 参数控制

### 4. Agent 集成模式

```python
# 模式1：通过 Prompt 注入记忆
agent_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(system_message),
        MessagesPlaceholder(variable_name="chat_history"),  # 记忆注入点
        HumanMessagePromptTemplate.from_template(final_prompt),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ],
    input_variables=["input", "chat_history", "agent_scratchpad"]
)

# 模式2：Memory 传递给 Agent
agent = ConversationalAgent.from_llm_and_tools(
    llm=llm, tools=tools, memory=memory
)
```

### 5. 关键参数说明

- `memory_key`：记忆变量名（默认 "history"），决定注入 prompt 的变量名
- `input_key`：输入字段名（用于追踪用户查询）
- `output_key`：输出字段名（用于追踪响应）
- `return_messages`：是否返回消息对象（vs 字符串）

### 6. 弃用状态

注意：`langchain_classic` 中的大多数 Memory 类在 v0.3.1 后被标记弃用，将在 v1.0.0 移除。
2025-2026 推荐使用 LangGraph 的 checkpointer + store 架构替代。

### 7. 10种 Memory 类型特点

| 类型 | 策略 | 参数 | 适用场景 |
|------|------|------|----------|
| ConversationBufferMemory | 全量存储 | - | 短对话 |
| ConversationBufferWindowMemory | 滑动窗口 | k（消息对数） | 长对话/限制上下文 |
| ConversationTokenBufferMemory | Token 限制 | max_token_limit | 精确控制 Token |
| ConversationSummaryMemory | LLM 摘要 | llm, prompt | 超长对话 |
| ConversationSummaryBufferMemory | 混合策略 | llm, max_token_limit | 平衡近期/压缩 |
| ConversationEntityMemory | 实体追踪 | llm, entity_store | 追踪关键实体 |
| VectorStoreRetrieverMemory | 语义检索 | retriever | 语义相关召回 |
| CombinedMemory | 组合多个 | memories | 多策略并用 |
| SimpleMemory | 静态键值 | memories | 注入固定上下文 |
| ReadOnlySharedMemory | 只读包装 | memory | 跨 Agent 共享 |
