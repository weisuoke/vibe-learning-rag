# 实战代码 场景 2：智能记忆压缩 Agent

> SummarizationMiddleware + RemoveMessage + Token 限制 — 让长对话永不崩溃

---

## 场景描述

**问题：** 当对话轮次超过 50 轮、Token 超过上下文窗口，Agent 会报错或质量下降。

**目标：** 创建一个能自动管理记忆长度的 Agent，聊多久都不会崩。

**需求：**
- 对话超长时自动压缩旧消息
- 压缩后仍保留关键信息（不是直接删除）
- 支持多种压缩策略的切换
- 监控 Token 使用量

---

## 方案 1：SummarizationMiddleware（推荐）

```python
"""
场景 2.1：使用 SummarizationMiddleware 自动压缩
原理：Token 超过阈值时，用小模型把旧消息压缩为摘要
"""
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


@tool
def search_docs(query: str) -> str:
    """搜索文档库（模拟）"""
    docs = {
        "RAG": "RAG 是一种结合检索和生成的 AI 技术架构...(省略 500 字)...",
        "LangChain": "LangChain 是一个用于构建 LLM 应用的框架...(省略 300 字)...",
        "向量数据库": "向量数据库用于存储和检索高维向量...(省略 400 字)...",
    }
    for key, value in docs.items():
        if key in query:
            return value
    return f"未找到关于 {query} 的文档"


# ========== 创建带自动压缩的 Agent ==========
agent = create_agent(
    model="gpt-4.1",
    tools=[search_docs],
    system_prompt=(
        "你是一个技术学习助手。"
        "帮助用户学习 RAG 和 LangChain 相关知识。"
        "始终记住用户之前的问题和你的回答，保持对话连贯。"
    ),
    middleware=[
        SummarizationMiddleware(
            model="gpt-4.1-mini",          # 用便宜的模型做摘要
            trigger=("tokens", 4000),       # Token 超过 4000 触发
            keep=("messages", 20),          # 保留最近 20 条消息
            # 工作原理：
            # 1. 每次模型调用前检查 Token 总量
            # 2. 超过 4000 tokens → 触发摘要
            # 3. 用 gpt-4.1-mini 对旧消息生成摘要
            # 4. 替换为：[摘要消息] + 最近 20 条消息
        )
    ],
    checkpointer=InMemorySaver(),
)


def chat(msg: str, thread: str = "test") -> str:
    config = {"configurable": {"thread_id": thread}}
    resp = agent.invoke(
        {"messages": [{"role": "user", "content": msg}]},
        config,
    )
    return resp["messages"][-1].content


# ========== 测试长对话 ==========
if __name__ == "__main__":
    thread = "long-learning-session"

    # 模拟一个完整的学习会话
    questions = [
        "你好，我想系统学习 RAG 开发",
        "什么是 RAG？请详细解释",
        "RAG 的核心组件有哪些？",
        "什么是向量数据库？它在 RAG 中的作用？",
        "LangChain 是什么？怎么用它构建 RAG？",
        "如何评估 RAG 系统的效果？",
        "RAG 有哪些常见的优化策略？",
        "什么是 HyDE？怎么用于 RAG？",
        "什么是 ReRank？和初始检索有什么区别？",
        "如何处理多语言 RAG？",
        "RAG 在生产环境中需要注意什么？",
        "请总结一下我今天学了哪些内容",  # ← 测试摘要后是否还记得
    ]

    for i, q in enumerate(questions):
        print(f"\n{'='*50}")
        print(f"第 {i+1}/{len(questions)} 轮")
        print(f"{'='*50}")
        print(f"👤 {q}")
        answer = chat(q, thread)
        print(f"🤖 {answer[:200]}...")  # 截断显示

    # 最终测试：Agent 是否还记得早期内容
    print(f"\n{'='*50}")
    print("记忆测试")
    print(f"{'='*50}")
    print(f"👤 我在第一个问题里说了什么？")
    answer = chat("我在第一个问题里说了什么？", thread)
    print(f"🤖 {answer}")
    # 即使早期消息被压缩为摘要，Agent 仍能通过摘要回忆大致内容
```

### SummarizationMiddleware 工作原理

```
对话进行中...

消息列表（正常状态）：
  [user1, ai1, user2, ai2, ..., user15, ai15]
  Token 总量: 3500  ← 还没触发

继续对话...

消息列表（即将触发）：
  [user1, ai1, ..., user20, ai20]
  Token 总量: 4200  ← 超过 4000！触发压缩！

压缩过程：
  1. 取出旧消息: [user1, ai1, ..., user10, ai10]
  2. 调用 gpt-4.1-mini 生成摘要:
     "用户开始学习 RAG，了解了基本概念和核心组件..."
  3. 保留最近消息: [user11, ai11, ..., user20, ai20]

压缩后的消息列表：
  [SystemMessage("对话摘要: 用户开始学习RAG..."),
   user11, ai11, ..., user20, ai20]
  Token 总量: 2000  ← 回到合理范围 ✓
```

---

## 方案 2：RemoveMessage 手动删除策略

```python
"""
场景 2.2：使用 RemoveMessage 手动控制消息删除
适用：需要精确控制哪些消息保留、哪些删除
"""
from langchain.agents import create_agent
from langchain.agents.middleware import after_model
from langchain.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver


@after_model
def keep_recent_messages(state, runtime):
    """只保留最近 N 条消息，删除更早的"""
    messages = state["messages"]
    max_messages = 30  # 最多保留 30 条消息

    if len(messages) <= max_messages:
        return None  # 不需要清理

    # 找出需要删除的旧消息
    # 注意：保留 system message（通常是第一条）
    to_remove = []
    for msg in messages[1:-max_messages]:  # 跳过第1条(system)和最后N条
        to_remove.append(RemoveMessage(id=msg.id))

    if to_remove:
        print(f"  ⚠️ 清理了 {len(to_remove)} 条旧消息")
        return {"messages": to_remove}

    return None


@after_model
def remove_large_tool_outputs(state, runtime):
    """删除过大的工具输出（超过 1000 字符的）"""
    messages = state["messages"]
    to_remove = []

    # 只检查较旧的消息（保留最近 5 条）
    for msg in messages[:-5]:
        if msg.type == "tool" and len(msg.content) > 1000:
            to_remove.append(RemoveMessage(id=msg.id))
            print(f"  ⚠️ 删除了过大的工具输出 ({len(msg.content)} 字符)")

    return {"messages": to_remove} if to_remove else None


agent = create_agent(
    model="gpt-4.1-mini",
    tools=[],
    middleware=[
        keep_recent_messages,
        remove_large_tool_outputs,
    ],
    checkpointer=InMemorySaver(),
)
```

### RemoveMessage 工作原理

```python
# RemoveMessage 通过消息 ID 删除消息
# 每条消息都有唯一的 id 属性

# 示例消息列表：
messages = [
    SystemMessage(content="你是助手", id="sys-1"),
    HumanMessage(content="你好", id="human-1"),        # 旧
    AIMessage(content="你好！", id="ai-1"),              # 旧
    HumanMessage(content="今天天气", id="human-2"),      # 旧
    AIMessage(content="晴天", id="ai-2"),                # 旧
    HumanMessage(content="谢谢", id="human-3"),          # 新 - 保留
    AIMessage(content="不客气", id="ai-3"),              # 新 - 保留
]

# 删除旧消息：
to_remove = [
    RemoveMessage(id="human-1"),
    RemoveMessage(id="ai-1"),
    RemoveMessage(id="human-2"),
    RemoveMessage(id="ai-2"),
]

# 执行后：
# messages = [
#     SystemMessage("你是助手", id="sys-1"),   # 保留
#     HumanMessage("谢谢", id="human-3"),      # 保留
#     AIMessage("不客气", id="ai-3"),           # 保留
# ]
```

---

## 方案 3：混合策略（摘要 + 删除 + 监控）

```python
"""
场景 2.3：生产级记忆压缩方案
组合多种策略，适用于正式产品
"""
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    after_model,
    before_model,
)
from langchain.messages import RemoveMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


# ========== 工具（模拟 RAG 检索） ==========
@tool
def retrieve_documents(query: str) -> str:
    """从知识库检索文档（模拟，返回较长内容）"""
    # 模拟返回较长的检索结果
    return f"""
    检索结果 - 查询: {query}

    文档1: {query} 的基础概念
    内容: 这是一段关于 {query} 的详细介绍...
    (假设这里有 500 字的内容)

    文档2: {query} 的最佳实践
    内容: 在实际项目中使用 {query} 的建议...
    (假设这里有 300 字的内容)

    文档3: {query} 的常见问题
    内容: 使用 {query} 时经常遇到的问题和解决方案...
    (假设这里有 400 字的内容)
    """


# ========== 中间件 1：Token 监控 ==========
@before_model
def monitor_tokens(state, runtime):
    """监控当前上下文的 Token 使用情况"""
    messages = state["messages"]

    # 粗略估算 Token 数（中文约 2 字符 = 1 token）
    total_chars = sum(len(str(m.content)) for m in messages)
    estimated_tokens = total_chars // 2

    print(f"  📊 当前消息数: {len(messages)}, 预估 Token: {estimated_tokens}")

    if estimated_tokens > 3000:
        print(f"  ⚠️ Token 较高，SummarizationMiddleware 即将触发")

    return None  # 只监控，不修改


# ========== 中间件 2：清理工具输出 ==========
@after_model
def clean_tool_outputs(state, runtime):
    """清理已完成任务的大型工具输出"""
    messages = state["messages"]
    to_remove = []

    # 策略：对于 5 轮之前的工具输出，如果超过 500 字符就删除
    tool_messages = [(i, msg) for i, msg in enumerate(messages) if msg.type == "tool"]

    if len(tool_messages) > 3:
        # 只保留最近 3 个工具输出
        for i, msg in tool_messages[:-3]:
            if len(msg.content) > 500:
                to_remove.append(RemoveMessage(id=msg.id))

    return {"messages": to_remove} if to_remove else None


# ========== 组合所有策略 ==========
agent = create_agent(
    model="gpt-4.1",
    tools=[retrieve_documents],
    system_prompt=(
        "你是一个知识库问答助手。"
        "使用 retrieve_documents 工具搜索信息，然后基于搜索结果回答用户问题。"
    ),
    middleware=[
        # 顺序很重要：先监控 → 再摘要 → 最后清理
        monitor_tokens,
        SummarizationMiddleware(
            model="gpt-4.1-mini",
            trigger=("tokens", 4000),
            keep=("messages", 15),
        ),
        clean_tool_outputs,
    ],
    checkpointer=InMemorySaver(),
)


# ========== 测试 ==========
if __name__ == "__main__":
    thread = "rag-learning"
    config = {"configurable": {"thread_id": thread}}

    queries = [
        "什么是 RAG？",
        "Embedding 是什么？",
        "向量数据库怎么选？",
        "Chunking 策略有哪些？",
        "如何评估 RAG 效果？",
        "ReRank 是什么？",
        "总结我今天学的所有内容",
    ]

    for q in queries:
        print(f"\n{'─'*40}")
        print(f"👤 {q}")
        resp = agent.invoke(
            {"messages": [{"role": "user", "content": q}]},
            config,
        )
        print(f"🤖 {resp['messages'][-1].content[:150]}...")

    # 验证记忆
    print(f"\n{'='*50}")
    print("验证：压缩后仍记得早期内容")
    print(f"{'='*50}")
    resp = agent.invoke(
        {"messages": [{"role": "user", "content": "我第一个问题问了什么？"}]},
        config,
    )
    print(f"🤖 {resp['messages'][-1].content}")
```

---

## 三种压缩策略对比

```
策略 1: SummarizationMiddleware（自动摘要）
  优点：保留语义信息，自动触发
  缺点：摘要需要额外 LLM 调用（有成本）
  适合：长期对话、需要回忆旧内容

策略 2: RemoveMessage（直接删除）
  优点：零额外成本，立即生效
  缺点：被删除的信息完全丢失
  适合：工具输出清理、明确不需要的消息

策略 3: 混合策略（摘要 + 删除 + 监控）
  优点：精细控制，兼顾效果和成本
  缺点：配置复杂
  适合：生产环境

┌──────────────────────────────────────────────┐
│  推荐决策：                                    │
│                                              │
│  简单项目 → SummarizationMiddleware 即可       │
│  成本敏感 → RemoveMessage 手动删除             │
│  正式产品 → 混合策略                           │
└──────────────────────────────────────────────┘
```

---

## 经典 API 对照：ConversationSummaryBufferMemory

```python
"""
场景 2.4：经典 API 的摘要缓冲记忆（对照学习用）
"""
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-mini")

# 经典版的"摘要 + 缓冲"组合
memory = ConversationSummaryBufferMemory(
    llm=llm,                  # 用于生成摘要的模型
    max_token_limit=2000,     # Token 上限
    return_messages=True,
)

# 模拟对话
memory.save_context(
    {"input": "我叫张三"},
    {"output": "你好张三！"}
)
memory.save_context(
    {"input": "我在学 RAG"},
    {"output": "RAG 是很棒的技术！"}
)

# 查看当前记忆
print(memory.load_memory_variables({}))

# 对照关系：
# ConversationSummaryBufferMemory
#   ↓ 现代替代
# SummarizationMiddleware + checkpointer
```

---

## 与 RAG 开发的关系

| 压缩策略 | RAG 应用场景 |
|----------|-------------|
| SummarizationMiddleware | RAG 多轮问答中压缩旧的检索结果 |
| RemoveMessage 清理工具输出 | 清理旧的文档检索内容，释放 Token 空间 |
| Token 监控 | 确保 RAG 检索结果 + 对话历史不超限 |
| 混合策略 | 生产级 RAG Agent 的标准配置 |

**RAG 特有的挑战：**
```
RAG Agent 的 Token 构成：
  System Prompt:     ~500 tokens
  对话历史:           ~2000 tokens
  检索结果（每次）:    ~1500 tokens  ← 这个特别大！
  工具调用记录:        ~1000 tokens
  ─────────────────────────────────
  总计:               ~5000 tokens

  如果不清理旧的检索结果，几轮之后就会超限！
  → 必须用压缩策略管理
```

---

**上一篇：** [07_实战代码_场景1_基础对话记忆Agent.md](./07_实战代码_场景1_基础对话记忆Agent.md)
**下一篇：** [07_实战代码_场景3_跨会话长期记忆.md](./07_实战代码_场景3_跨会话长期记忆.md)
