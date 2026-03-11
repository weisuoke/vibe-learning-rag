# 实战代码 场景 4：多记忆组合 Agent

> 短期 + 长期 + 工作记忆 + 压缩 — 全副武装的 Agent 记忆系统

---

## 场景描述

**目标：** 构建一个同时具备多种记忆能力的 Agent，模拟完整的记忆体系。

**需求：**
- 短期记忆：记住当前对话（Checkpointer）
- 长期记忆：跨会话记住用户（Store）
- 工作记忆：多工具调用的中间结果
- 记忆压缩：对话过长时自动摘要
- 全部协同工作

---

## 完整实现：全功能记忆 Agent

```python
"""
场景 4：多记忆组合 Agent
集成：Checkpointer + Store + SummarizationMiddleware + 工具系统
"""
import os
import uuid
from datetime import datetime

from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    before_model,
    after_model,
)
from langchain.embeddings import init_embeddings
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore


# ============================================================
# 第一层：Store（长期记忆）
# ============================================================
embeddings = init_embeddings("openai:text-embedding-3-small")
store = InMemoryStore(index={"embed": embeddings, "dims": 1536})


# ============================================================
# 第二层：工具（产生工作记忆）
# ============================================================
@tool
def search_knowledge_base(query: str) -> str:
    """搜索知识库，获取技术文档信息。"""
    # 模拟知识库检索
    knowledge = {
        "RAG": (
            "RAG (Retrieval-Augmented Generation) 是一种将检索和生成结合的技术。"
            "核心流程：用户提问 → 检索相关文档 → 将文档作为上下文 → LLM 生成回答。"
            "优点：减少幻觉、可引用来源、知识可更新。"
        ),
        "向量数据库": (
            "常见向量数据库：ChromaDB（轻量）、Milvus（企业级）、"
            "Pinecone（云托管）、Weaviate（开源）。"
            "选型建议：个人项目用 ChromaDB，企业用 Milvus 或 Pinecone。"
        ),
        "Embedding": (
            "Embedding 模型将文本转换为高维向量。"
            "推荐：OpenAI text-embedding-3-small（性价比高）、"
            "BGE-M3（中文优化）、Cohere embed-v3（多语言）。"
        ),
        "Chunking": (
            "文本分块策略：固定大小（简单）、语义分块（效果好）、"
            "递归分块（推荐）。推荐 chunk_size=512, overlap=50。"
        ),
    }

    results = []
    for key, value in knowledge.items():
        if key.lower() in query.lower() or any(
            word in query for word in key.split()
        ):
            results.append(f"📄 {key}: {value}")

    if results:
        return "\n\n".join(results)
    return f"未找到关于「{query}」的文档。请尝试其他关键词。"


@tool
def get_learning_progress(user_id: str) -> str:
    """查看用户的学习进度。"""
    # 从 Store 获取学习记录
    try:
        records = store.search((user_id, "learning"), limit=10)
        if records:
            progress = "\n".join(
                f"  - {r.value['text']}" for r in records
            )
            return f"用户 {user_id} 的学习记录：\n{progress}"
    except Exception:
        pass
    return f"用户 {user_id} 暂无学习记录"


@tool
def save_learning_note(topic: str, note: str) -> str:
    """保存用户的学习笔记到长期记忆中。
    topic: 学习主题，如 "RAG基础"
    note: 笔记内容
    """
    key = f"note-{uuid.uuid4().hex[:8]}"
    store.put(
        ("current-user", "learning"), key,
        {
            "text": f"[{topic}] {note}",
            "timestamp": datetime.now().isoformat(),
        }
    )
    return f"已保存学习笔记：[{topic}] {note}"


# ============================================================
# 第三层：Middleware（记忆管理）
# ============================================================

# 中间件 1：注入长期记忆
@before_model
def inject_memories(state, runtime):
    """注入长期记忆到 Agent 上下文"""
    messages = state["messages"]
    user_messages = [m for m in messages if m.type == "human"]
    if not user_messages:
        return None

    query = user_messages[-1].content
    user_id = runtime.config.get("configurable", {}).get(
        "user_id", "current-user"
    )

    # 搜索相关的长期记忆
    try:
        memories = runtime.store.search(
            (user_id, "memories"), query=query, limit=3
        )
        learning = runtime.store.search(
            (user_id, "learning"), query=query, limit=2
        )
    except Exception:
        memories, learning = [], []

    parts = []
    if memories:
        mem_text = "\n".join(f"  - {m.value['text']}" for m in memories)
        parts.append(f"用户信息：\n{mem_text}")
    if learning:
        learn_text = "\n".join(f"  - {l.value['text']}" for l in learning)
        parts.append(f"学习记录：\n{learn_text}")

    if parts:
        content = "【长期记忆】\n" + "\n".join(parts)
        return {"messages": [SystemMessage(content=content)]}

    return None


# 中间件 2：自动提取用户信息
@after_model
def extract_user_info(state, runtime):
    """从对话中自动提取和保存用户信息"""
    messages = state["messages"]
    if len(messages) < 2:
        return None

    # 获取最后一条用户消息
    human_msgs = [m for m in messages if m.type == "human"]
    if not human_msgs:
        return None

    latest = human_msgs[-1].content

    # 简单的关键词规则提取（生产环境用 LLM）
    user_id = runtime.config.get("configurable", {}).get(
        "user_id", "current-user"
    )

    info_patterns = {
        "我叫": "name",
        "我是": "identity",
        "我在": "location_or_activity",
        "我喜欢": "preference",
        "我使用": "tools",
        "我的目标": "goal",
    }

    for pattern, category in info_patterns.items():
        if pattern in latest:
            # 提取包含关键词的句子
            sentences = latest.split("，")
            for sent in sentences:
                if pattern in sent:
                    key = f"auto-{uuid.uuid4().hex[:6]}"
                    try:
                        runtime.store.put(
                            (user_id, "memories"), key,
                            {"text": sent.strip(), "category": category}
                        )
                    except Exception:
                        pass

    return None


# 中间件 3：Token 监控
@before_model
def token_monitor(state, runtime):
    """监控 Token 使用量"""
    messages = state["messages"]
    msg_count = len(messages)
    tool_count = sum(1 for m in messages if m.type == "tool")
    total_chars = sum(len(str(m.content)) for m in messages)

    print(
        f"  📊 消息: {msg_count} | 工具结果: {tool_count} | "
        f"字符: {total_chars} | 预估Token: {total_chars // 2}"
    )
    return None


# ============================================================
# 第四层：组装 Agent
# ============================================================
agent = create_agent(
    model="gpt-4.1",
    tools=[search_knowledge_base, get_learning_progress, save_learning_note],
    system_prompt=(
        "你是一个智能 RAG 学习助手，拥有以下能力：\n"
        "1. 搜索技术知识库回答问题\n"
        "2. 记住用户的学习进度和偏好\n"
        "3. 帮用户做学习笔记\n"
        "4. 根据用户背景个性化推荐\n\n"
        "请根据用户的背景和学习进度，提供最合适的帮助。"
    ),
    middleware=[
        # 执行顺序：token_monitor → inject_memories → (模型调用) →
        #           SummarizationMiddleware → extract_user_info
        token_monitor,
        inject_memories,
        SummarizationMiddleware(
            model="gpt-4.1-mini",
            trigger=("tokens", 6000),
            keep=("messages", 25),
        ),
        extract_user_info,
    ],
    checkpointer=InMemorySaver(),
    store=store,
)


# ============================================================
# 测试：完整的学习会话
# ============================================================
if __name__ == "__main__":

    def chat(msg: str, thread: str, user: str = "current-user") -> str:
        config = {
            "configurable": {
                "thread_id": thread,
                "user_id": user,
            }
        }
        resp = agent.invoke(
            {"messages": [{"role": "user", "content": msg}]},
            config,
        )
        return resp["messages"][-1].content

    # ===== 会话 1：初次见面 + 学习 =====
    print("=" * 60)
    print("会话 1：初次学习")
    print("=" * 60)
    thread1 = "session-001"
    user = "student-alice"

    msgs = [
        "你好！我叫 Alice，我是后端开发者，我想学 RAG",
        "什么是 RAG？请帮我搜索相关知识",
        "帮我把 RAG 的核心概念记到笔记里",
        "向量数据库有哪些选择？",
        "帮我记一下：选择了 ChromaDB 作为向量数据库",
        "总结一下我今天学了什么",
    ]

    for msg in msgs:
        print(f"\n👤 {msg}")
        reply = chat(msg, thread1, user)
        print(f"🤖 {reply[:200]}...")

    # ===== 会话 2：新会话（测试长期记忆） =====
    print("\n" + "=" * 60)
    print("会话 2：新会话（应记得用户信息）")
    print("=" * 60)
    thread2 = "session-002"

    msgs2 = [
        "继续上次的学习，我还需要学什么？",
        "查看一下我的学习进度",
    ]

    for msg in msgs2:
        print(f"\n👤 {msg}")
        reply = chat(msg, thread2, user)
        print(f"🤖 {reply[:200]}...")

    # ===== 查看所有记忆 =====
    print("\n" + "=" * 60)
    print("📋 Alice 的所有长期记忆")
    print("=" * 60)

    all_mem = store.search((user, "memories"), limit=20)
    for m in all_mem:
        print(f"  [memories/{m.key}] {m.value['text']}")

    all_learn = store.search((user, "learning"), limit=20)
    for l in all_learn:
        print(f"  [learning/{l.key}] {l.value['text']}")
```

---

## 架构解析

```
多记忆组合 Agent 的完整架构：

┌─────────────────────────────────────────────────────────┐
│                      Agent 系统                          │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  before_model 阶段                                │  │
│  │                                                   │  │
│  │  1. token_monitor → 监控资源使用                   │  │
│  │  2. inject_memories → 从 Store 注入长期记忆        │  │
│  └───────────────────────┬───────────────────────────┘  │
│                          ↓                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │  LLM 推理                                         │  │
│  │                                                   │  │
│  │  输入：System Prompt + 长期记忆 + 对话历史 + 用户输入│  │
│  │  输出：回复 或 工具调用                             │  │
│  │                                                   │  │
│  │  如果需要工具 → 工具循环（工作记忆）                 │  │
│  │    search_knowledge_base → 结果存入消息列表         │  │
│  │    save_learning_note → 结果存入消息列表            │  │
│  │    → 再次 LLM 推理 → 直到不需要工具                │  │
│  └───────────────────────┬───────────────────────────┘  │
│                          ↓                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │  after_model 阶段                                 │  │
│  │                                                   │  │
│  │  3. SummarizationMiddleware → 自动压缩旧消息       │  │
│  │  4. extract_user_info → 提取用户信息存入 Store     │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌─────────────────────┐  ┌─────────────────────────┐  │
│  │  Checkpointer       │  │  Store                  │  │
│  │  (短期记忆)          │  │  (长期记忆)              │  │
│  │                     │  │                         │  │
│  │  thread-001:        │  │  (alice, memories):     │  │
│  │    [对话消息...]     │  │    name, job, prefs...  │  │
│  │  thread-002:        │  │  (alice, learning):     │  │
│  │    [对话消息...]     │  │    notes, progress...   │  │
│  └─────────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 经典 API 对照：CombinedMemory

```python
"""
经典 API 中的多记忆组合方式（对照学习用）
注意：经典 API 已标记为 legacy
"""
from langchain_classic.memory import (
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    CombinedMemory,
)
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-mini")

# 经典方式：用 CombinedMemory 组合多种 Memory
buffer_memory = ConversationBufferWindowMemory(
    memory_key="recent_chat",
    k=10,
    return_messages=True,
)

summary_memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="conversation_summary",
    return_messages=True,
)

combined = CombinedMemory(memories=[buffer_memory, summary_memory])

# 对照关系：
# CombinedMemory = 把多个 Memory 的输出合并到 prompt 中
#   ↓ 现代替代
# middleware 数组 = 多个中间件分别处理不同类型的记忆

# 经典 CombinedMemory 的限制：
#   - 只能组合 load_memory_variables 的输出
#   - 无法实现 Store 的跨会话语义搜索
#   - 无法自动提取和存储用户信息
#   - 所有记忆共享同一个生命周期

# 现代 middleware 的优势：
#   - 每个中间件独立控制
#   - 支持 before_model / after_model 不同阶段
#   - 可以访问 runtime.store 实现跨会话
#   - 灵活组合，互不干扰
```

---

## 各记忆层次的 Token 预算分配

```
Context Window 分配建议（以 gpt-4.1 128K 为例）：

  ┌─────────────────────────────────────────────────┐
  │  System Prompt             500 tokens  (固定)    │
  │  长期记忆（Store注入）       800 tokens  (动态)    │
  │  对话摘要                   500 tokens  (压缩后)   │
  │  最近对话                  3000 tokens  (保留)    │
  │  工具调用 + 结果           2000 tokens  (工作记忆) │
  │  当前用户输入               200 tokens  (当前)    │
  │  ─────────────────────────────────────────────── │
  │  合计约:                   7000 tokens            │
  │  剩余用于生成:            121000 tokens           │
  └─────────────────────────────────────────────────┘

  原则：
  - 固定部分（System Prompt）尽量精简
  - 动态部分（长期记忆、工具结果）按需加载
  - 压缩部分（摘要）定期触发
  - 总和控制在模型上下文的 10-20%
```

---

## 与 RAG 开发的关系

| 记忆层次 | RAG 中的应用 |
|---------|-------------|
| 短期记忆（Checkpointer） | 多轮检索对话的上下文 |
| 长期记忆（Store） | 用户偏好、搜索历史、常用知识领域 |
| 工作记忆（工具调用） | 检索结果、分析中间数据 |
| 记忆压缩（Summarization） | 压缩旧的检索结果和对话 |
| 组合使用 | 完整的 RAG Agent 记忆系统 |

---

**上一篇：** [07_实战代码_场景3_跨会话长期记忆.md](./07_实战代码_场景3_跨会话长期记忆.md)
**下一篇：** [07_实战代码_场景5_生产级持久化部署.md](./07_实战代码_场景5_生产级持久化部署.md)
