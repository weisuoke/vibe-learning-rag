# 核心概念 1：Memory 架构演进

> 从 BaseMemory 到 Checkpointer + Store — LangChain 记忆系统的三代变迁

---

## 一句话定义

**Memory 架构演进是 LangChain 记忆系统从「类继承 + 手动管理」到「声明式 + 自动持久化」的范式转变，理解这条演进线是选择正确记忆方案的前提。**

---

## 为什么要先理解架构演进？

很多初学者打开 LangChain 文档，会发现两套完全不同的记忆 API：

```python
# 你会看到这种写法（经典）
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()

# 也会看到这种写法（现代）
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()
```

**到底用哪个？** 这就是为什么必须先理解架构演进。

---

## 第一代：BaseMemory 类继承体系（2023-2024）

### 核心设计

LangChain 最早的记忆系统基于**面向对象继承**：

```
BaseMemory（抽象基类）
├── BaseChatMemory（聊天记忆基类）
│   ├── ConversationBufferMemory      # 全量存储
│   ├── ConversationBufferWindowMemory # 滑动窗口
│   ├── ConversationTokenBufferMemory  # Token 限制
│   ├── ConversationSummaryMemory      # LLM 摘要
│   ├── ConversationSummaryBufferMemory# 摘要 + 缓冲
│   ├── ConversationEntityMemory       # 实体追踪
│   └── VectorStoreRetrieverMemory     # 语义检索
├── CombinedMemory                     # 组合多个 Memory
├── SimpleMemory                       # 静态键值
└── ReadOnlySharedMemory               # 只读包装
```

[来源: sourcecode/langchain/libs/langchain/langchain_classic/base_memory.py]

### BaseMemory 核心接口

```python
from abc import ABC, abstractmethod

class BaseMemory(ABC):
    """所有 Memory 实现的抽象基类"""

    @property
    @abstractmethod
    def memory_variables(self) -> list[str]:
        """返回此 memory 注入到 chain 输入的键名列表"""
        # 例如返回 ["chat_history"]

    @abstractmethod
    def load_memory_variables(self, inputs: dict) -> dict:
        """加载记忆变量 —— Agent 执行前调用"""
        # 返回 {"chat_history": "Human: hi\nAI: hello"}

    @abstractmethod
    def save_context(self, inputs: dict, outputs: dict) -> None:
        """保存对话上下文 —— Agent 执行后调用"""
        # 把本轮 input/output 存入记忆

    def clear(self) -> None:
        """清除所有记忆内容"""
```

[来源: sourcecode/langchain/libs/langchain/langchain_classic/base_memory.py]

### 三阶段生命周期

```
用户提问
    │
    ▼
┌─────────────────────────────────────┐
│  1. LOAD 阶段                        │
│  memory.load_memory_variables({})    │
│  → 返回 {"chat_history": [...]}      │
│  → 注入到 Prompt 模板中               │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  2. Agent 推理 + Tool 调用            │
│  LLM 接收带记忆的 Prompt              │
│  → 生成回答                          │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  3. SAVE 阶段                        │
│  memory.save_context(inputs, outputs)│
│  → 存储本轮交互                      │
│  → 应用裁剪/摘要逻辑                  │
└─────────────────────────────────────┘
```

### 使用示例

```python
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor

# 创建记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",   # 注入 prompt 的变量名
    return_messages=True,        # 返回消息对象 vs 字符串
    input_key="input",           # 输入字段名
    output_key="output",         # 输出字段名
)

# 传递给 AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,    # ← 记忆在这里传入
    verbose=True,
)

# AgentExecutor 自动处理 load → 推理 → save 流程
result = agent_executor.invoke({"input": "你好"})
```

### 第一代的优点与局限

| 维度 | 优点 | 局限 |
|------|------|------|
| **易用性** | API 简单直观 | 需要手动选择 Memory 类型 |
| **灵活性** | 10种策略可选 | 组合使用复杂 |
| **持久化** | - | 默认只在内存中，需额外配置 |
| **多用户** | - | 不原生支持多 thread |
| **可恢复** | - | 无法回溯到历史状态 |

---

## 第二代：LangGraph Checkpointer（2024-2025）

### 为什么需要新架构？

第一代的核心问题是 **「记忆」和「持久化」混在一起**。开发者既要选记忆策略，又要操心数据存在哪里。

LangGraph 的解决方案：**把「状态管理」交给框架自动处理**。

### Checkpointer 核心概念

```
Checkpointer = 自动的状态快照系统

每一步操作都自动保存 → 可以回溯到任意步骤
类似于：游戏自动存档、Git 的每次 commit
```

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState

# 1. 创建 checkpointer
checkpointer = InMemorySaver()

# 2. 构建图
builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.set_entry_point("chat")

# 3. 编译时传入 checkpointer
graph = builder.compile(checkpointer=checkpointer)

# 4. 使用 thread_id 标识对话
config = {"configurable": {"thread_id": "user-123"}}
graph.invoke({"messages": [{"role": "user", "content": "你好"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "我刚才说了什么？"}]}, config)
# ↑ 第二次调用能记住第一次的对话，因为 thread_id 相同
```

[来源: atom/langchain/L4_Agent系统/07_Agent Memory集成/reference/context7_langgraph_memory_01.md]

### create_agent 的简化写法

2026 年的 `create_agent` API 进一步简化了记忆集成：

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4.1",
    tools=[...],
    checkpointer=InMemorySaver(),  # ← 一行搞定记忆
)

config = {"configurable": {"thread_id": "user-123"}}
agent.invoke({"messages": "你好"}, config)
agent.invoke({"messages": "我刚才说了什么？"}, config)
```

[来源: atom/langchain/L4_Agent系统/07_Agent Memory集成/reference/context7_langchain_memory_01.md]

### 与第一代的对比

| 维度 | 第一代 BaseMemory | 第二代 Checkpointer |
|------|-------------------|---------------------|
| **记忆管理** | 手动 load/save | 自动快照 |
| **多用户** | 需额外处理 | thread_id 原生支持 |
| **持久化** | 需手动配置后端 | 换 Saver 即可 |
| **状态回溯** | 不支持 | 可回到任意检查点 |
| **记忆策略** | 10种类选择 | 通过 middleware 实现 |
| **代码量** | ~10 行配置 | ~3 行配置 |

---

## 第三代：Checkpointer + Store 双层架构（2025-2026）

### 双层记忆的必要性

Checkpointer 解决了**短期记忆**（单次对话内），但没有解决**长期记忆**（跨对话）：

```
用户和 Agent 进行了 3 次独立对话：

对话 1 (thread-1)：用户说 "我喜欢吃披萨"
对话 2 (thread-2)：用户说 "推荐个午餐"
    → 如果只有 Checkpointer：Agent 不知道用户喜欢披萨
    → 如果有 Store：Agent 能检索到用户的偏好
```

### Store 核心概念

```
Store = 跨线程的共享记忆库

Checkpointer：线程内的对话历史（短期记忆）
Store：线程间共享的知识（长期记忆）

组合使用 = 完整的记忆体系
```

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain.embeddings import init_embeddings

# 短期记忆
checkpointer = InMemorySaver()

# 长期记忆（带语义搜索）
embeddings = init_embeddings("openai:text-embedding-3-small")
store = InMemoryStore(
    index={"embed": embeddings, "dims": 1536}
)

# 同时编译
graph = builder.compile(
    checkpointer=checkpointer,  # 短期
    store=store,                # 长期
)
```

[来源: atom/langchain/L4_Agent系统/07_Agent Memory集成/reference/context7_langgraph_memory_01.md]

### 在节点中使用 Store

```python
from langgraph.graph import MessagesState
from langgraph.runtime import Runtime

async def chat(state: MessagesState, runtime: Runtime):
    """带长期记忆的聊天节点"""

    user_msg = state["messages"][-1].content

    # 从 Store 中语义搜索相关记忆
    items = await runtime.store.asearch(
        ("user_123", "memories"),
        query=user_msg,
        limit=3
    )
    memories = "\n".join(item.value["text"] for item in items)

    # 注入到 system prompt
    system = f"用户的相关记忆：\n{memories}\n\n请基于这些信息回答。"

    response = await llm.ainvoke([
        {"role": "system", "content": system},
        *state["messages"]
    ])

    # 保存新的记忆到 Store
    await runtime.store.aput(
        ("user_123", "memories"),
        str(uuid4()),
        {"text": f"用户问了：{user_msg}"}
    )

    return {"messages": [response]}
```

[来源: atom/langchain/L4_Agent系统/07_Agent Memory集成/reference/context7_langgraph_memory_01.md]

### Middleware 记忆控制

第三代架构引入 **middleware** 替代第一代的多种 Memory 类：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4.1",
    tools=[...],
    middleware=[
        # 替代 ConversationSummaryBufferMemory
        SummarizationMiddleware(
            model="gpt-4.1-mini",      # 摘要用的模型
            trigger=("tokens", 4000),   # 超过 4000 token 触发
            keep=("messages", 20)       # 保留最近 20 条
        )
    ],
    checkpointer=InMemorySaver(),
)
```

[来源: atom/langchain/L4_Agent系统/07_Agent Memory集成/reference/context7_langchain_memory_01.md]

---

## 三代架构对照表

| 维度 | 第一代 (2023-2024) | 第二代 (2024-2025) | 第三代 (2025-2026) |
|------|---------------------|---------------------|---------------------|
| **核心组件** | BaseMemory 子类 | Checkpointer | Checkpointer + Store |
| **短期记忆** | ConversationBuffer* | Checkpointer | Checkpointer |
| **长期记忆** | VectorStoreRetriever | ❌ 不支持 | Store + 语义搜索 |
| **记忆策略** | 10种 Memory 类 | 手动管理 | Middleware |
| **持久化** | 需额外配置 | 换 Saver | 换 Saver + Store |
| **多用户** | 需额外处理 | thread_id | thread_id + namespace |
| **状态回溯** | ❌ | ✅ | ✅ |
| **弃用状态** | v0.3.1 弃用 | 可用 | **推荐** |

---

## 迁移指南：从第一代到第三代

### 最常见的迁移场景

```python
# ===== 场景 1：ConversationBufferMemory → Checkpointer =====

# 旧写法
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history")
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

# 新写法
from langgraph.checkpoint.memory import InMemorySaver
agent = create_agent(
    model="gpt-4.1",
    tools=tools,
    checkpointer=InMemorySaver(),
)

# ===== 场景 2：ConversationSummaryMemory → SummarizationMiddleware =====

# 旧写法
from langchain.memory import ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=llm)

# 新写法
from langchain.agents.middleware import SummarizationMiddleware
agent = create_agent(
    model="gpt-4.1",
    tools=tools,
    middleware=[SummarizationMiddleware(model="gpt-4.1-mini")],
    checkpointer=InMemorySaver(),
)

# ===== 场景 3：VectorStoreRetrieverMemory → Store =====

# 旧写法
from langchain.memory import VectorStoreRetrieverMemory
memory = VectorStoreRetrieverMemory(retriever=retriever)

# 新写法
from langgraph.store.memory import InMemoryStore
store = InMemoryStore(index={"embed": embeddings, "dims": 1536})
graph = builder.compile(checkpointer=checkpointer, store=store)
```

---

## 如何选择？决策流程图

```
需要 Agent 记忆？
    │
    ├── 新项目 / 可重构 ──→ 第三代（Checkpointer + Store + Middleware）
    │
    ├── 已有项目 / 用了 AgentExecutor ──→ 第一代（经典 Memory 类）
    │                                      但计划迁移到第三代
    │
    └── 只需简单对话记忆 ──→ 第二代（Checkpointer 即可）

需要长期记忆？
    │
    ├── 是 ──→ 必须用第三代（Store）
    │
    └── 否 ──→ Checkpointer 足够
```

---

## 与 RAG 开发的关系

| RAG 场景 | 记忆需求 | 推荐方案 |
|----------|----------|----------|
| **文档问答** | 记住对话上下文 | Checkpointer |
| **智能客服** | 记住用户偏好 + 历史工单 | Checkpointer + Store |
| **知识库助手** | 长期学习用户习惯 | Store + 语义搜索 |
| **多轮检索** | 基于上文优化查询 | Checkpointer + Query改写 |

---

## 本节小结

> **Memory 架构演进的核心是「关注点分离」**：
> - 第一代把策略、存储、生命周期混在一起
> - 第二代把持久化交给 Checkpointer
> - 第三代进一步分离短期记忆（Checkpointer）和长期记忆（Store），并用 Middleware 替代策略类
>
> **新项目直接用第三代，老项目渐进迁移。**

---

**下一篇：** [03_核心概念_2_短期记忆与对话历史.md](./03_核心概念_2_短期记忆与对话历史.md)
