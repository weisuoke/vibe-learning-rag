# 实战代码 场景 1：基础对话记忆 Agent

> 从零搭建一个能记住对话的 Agent，理解 checkpointer + thread_id 的完整流程

---

## 场景描述

**目标：** 创建一个带对话记忆的 Agent，支持多轮对话、多用户隔离。

**需求：**
- Agent 能记住同一会话中的对话内容
- 不同用户的对话互不干扰
- 能查看对话历史
- 提供简单的工具调用能力

---

## 方案 1：最简单的记忆 Agent（现代 API）

```python
"""
场景 1.1：使用 create_agent + InMemorySaver 的最简记忆 Agent
适用：快速原型、开发测试
"""
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


# ========== 1. 定义工具 ==========
@tool
def get_current_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式。例如：calculate('2 + 3 * 4')"""
    try:
        # 安全的数学计算（仅允许基本运算）
        allowed_chars = set("0123456789+-*/().% ")
        if not all(c in allowed_chars for c in expression):
            return "错误：只支持基本数学运算"
        result = eval(expression)  # 生产环境应使用更安全的方式
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


# ========== 2. 创建带记忆的 Agent ==========
checkpointer = InMemorySaver()

agent = create_agent(
    model="gpt-4.1-mini",
    tools=[get_current_time, calculate],
    system_prompt="你是一个友好的助手。请记住用户告诉你的信息，并在后续对话中引用。",
    checkpointer=checkpointer,
)


# ========== 3. 对话函数 ==========
def chat(user_input: str, thread_id: str = "default") -> str:
    """发送消息并返回 Agent 回复"""
    config = {"configurable": {"thread_id": thread_id}}
    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
    )
    return response["messages"][-1].content


# ========== 4. 测试多轮对话 ==========
if __name__ == "__main__":
    print("=" * 50)
    print("测试 1：多轮对话记忆")
    print("=" * 50)

    # 用户 Alice 的对话
    thread = "alice-chat-001"
    print(f"\n[Alice] 我叫 Alice，我是一名数据工程师")
    print(f"[Agent] {chat('我叫 Alice，我是一名数据工程师', thread)}")

    print(f"\n[Alice] 我正在学习 RAG 开发")
    print(f"[Agent] {chat('我正在学习 RAG 开发', thread)}")

    print(f"\n[Alice] 你还记得我的名字和职业吗？")
    print(f"[Agent] {chat('你还记得我的名字和职业吗？', thread)}")
    # 期望：Agent 能回忆起 Alice 是数据工程师，在学 RAG

    print(f"\n[Alice] 帮我算一下 1024 * 768")
    print(f"[Agent] {chat('帮我算一下 1024 * 768', thread)}")

    print("\n" + "=" * 50)
    print("测试 2：用户隔离")
    print("=" * 50)

    # 用户 Bob 的对话（不同 thread）
    thread_bob = "bob-chat-001"
    print(f"\n[Bob] 你知道我的名字吗？")
    print(f"[Agent] {chat('你知道我的名字吗？', thread_bob)}")
    # 期望：Agent 不知道 Bob 的名字（不同 thread）

    print(f"\n[Bob] 我叫 Bob，我是前端开发")
    print(f"[Agent] {chat('我叫 Bob，我是前端开发', thread_bob)}")

    # 回到 Alice 的对话
    print(f"\n[Alice] 我在学什么？")
    print(f"[Agent] {chat('我在学什么？', thread)}")
    # 期望：Agent 记得 Alice 在学 RAG，不会混淆 Bob 的信息
```

### 预期输出

```
==================================================
测试 1：多轮对话记忆
==================================================

[Alice] 我叫 Alice，我是一名数据工程师
[Agent] 你好 Alice！很高兴认识你。数据工程师是一个很棒的职业！

[Alice] 我正在学习 RAG 开发
[Agent] 很好！RAG（检索增强生成）是当下非常热门的技术...

[Alice] 你还记得我的名字和职业吗？
[Agent] 当然记得！你叫 Alice，是一名数据工程师，正在学习 RAG 开发。

[Alice] 帮我算一下 1024 * 768
[Agent] 1024 * 768 = 786432

==================================================
测试 2：用户隔离
==================================================

[Bob] 你知道我的名字吗？
[Agent] 抱歉，我还不知道你的名字，你可以告诉我吗？

[Bob] 我叫 Bob，我是前端开发
[Agent] 你好 Bob！前端开发也很棒...

[Alice] 我在学什么？
[Agent] 你在学习 RAG 开发！   ← 没有混淆 Bob 的信息 ✓
```

---

## 方案 2：带对话历史查看的 Agent

```python
"""
场景 1.2：可以查看和管理对话历史的 Agent
适用：需要审查对话记录的场景
"""
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


@tool
def search_knowledge(query: str) -> str:
    """搜索知识库。在实际项目中会连接向量数据库。"""
    # 模拟知识库搜索
    knowledge = {
        "RAG": "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的AI技术",
        "Embedding": "Embedding 将文本转换为高维向量，用于语义相似度计算",
        "Chunking": "Chunking 是将长文档分割成小块的过程，便于检索",
    }
    for key, value in knowledge.items():
        if key.lower() in query.lower():
            return value
    return f"未找到关于'{query}'的信息"


# 创建 Agent
checkpointer = InMemorySaver()

agent = create_agent(
    model="gpt-4.1-mini",
    tools=[search_knowledge],
    system_prompt=(
        "你是一个 RAG 学习助手。帮助用户学习 RAG 开发知识。"
        "当用户问技术问题时，先搜索知识库再回答。"
        "记住用户的学习进度和偏好。"
    ),
    checkpointer=checkpointer,
)


def chat_with_history(user_input: str, thread_id: str) -> dict:
    """对话并返回完整信息（包括历史消息数量）"""
    config = {"configurable": {"thread_id": thread_id}}

    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
    )

    # 获取当前对话的所有消息
    all_messages = response["messages"]

    return {
        "reply": all_messages[-1].content,
        "total_messages": len(all_messages),
        "message_types": [msg.type for msg in all_messages],
    }


def view_conversation_history(thread_id: str) -> list:
    """查看某个 thread 的完整对话历史"""
    config = {"configurable": {"thread_id": thread_id}}

    # 从 checkpointer 获取最新状态
    state = agent.get_state(config)

    if not state or not state.values.get("messages"):
        return []

    history = []
    for msg in state.values["messages"]:
        history.append({
            "role": msg.type,        # "human", "ai", "tool", "system"
            "content": msg.content[:200],  # 截断长内容
            "id": msg.id,
        })

    return history


# ========== 测试 ==========
if __name__ == "__main__":
    thread = "learner-001"

    # 多轮学习对话
    conversations = [
        "你好，我想学习 RAG 开发",
        "什么是 RAG？",
        "什么是 Embedding？",
        "我之前问了你什么问题？帮我总结一下",
    ]

    for user_msg in conversations:
        print(f"\n👤 {user_msg}")
        result = chat_with_history(user_msg, thread)
        print(f"🤖 {result['reply']}")
        print(f"   [总消息数: {result['total_messages']}]")

    # 查看完整历史
    print("\n" + "=" * 50)
    print("📋 完整对话历史：")
    print("=" * 50)
    history = view_conversation_history(thread)
    for i, msg in enumerate(history):
        role_icon = {"human": "👤", "ai": "🤖", "tool": "🔧", "system": "⚙️"}
        icon = role_icon.get(msg["role"], "❓")
        print(f"  {i+1}. {icon} [{msg['role']}] {msg['content'][:80]}...")
```

---

## 方案 3：经典 Memory API（兼容旧代码）

```python
"""
场景 1.3：使用经典 ConversationBufferMemory
适用：维护旧项目、理解 Memory 演进
注意：经典 API 已标记为 legacy，新项目推荐用 create_agent
"""
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# 工具
@tool
def greet(name: str) -> str:
    """向用户打招呼"""
    return f"你好 {name}！欢迎使用 Agent 助手。"


# 经典方式：显式创建 Memory 对象
memory = ConversationBufferMemory(
    memory_key="chat_history",    # 在 prompt 中的变量名
    return_messages=True,         # 返回消息对象（非字符串）
)

# 创建 Prompt（需要包含 chat_history 占位符）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手。"),
    MessagesPlaceholder("chat_history"),    # ← Memory 注入位置
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# 创建 Agent
llm = ChatOpenAI(model="gpt-4.1-mini")
agent = create_openai_functions_agent(llm, [greet], prompt)

executor = AgentExecutor(
    agent=agent,
    tools=[greet],
    memory=memory,          # ← 经典方式传入 Memory
    verbose=True,           # 打印中间步骤
)


# 测试
if __name__ == "__main__":
    # 第一轮
    result1 = executor.invoke({"input": "我叫张三"})
    print(f"回复: {result1['output']}")

    # 第二轮（Memory 自动注入历史）
    result2 = executor.invoke({"input": "我叫什么？"})
    print(f"回复: {result2['output']}")
    # → "你叫张三"

    # 查看 Memory 内容
    print("\n=== Memory 内容 ===")
    print(memory.load_memory_variables({}))
```

---

## 新旧 API 对比

```
经典 API（2023-2024）                 现代 API（2025-2026）
─────────────────────                 ─────────────────────
ConversationBufferMemory              checkpointer（自动管理）
memory_key="chat_history"             无需手动指定 key
return_messages=True                  默认消息格式
MessagesPlaceholder("chat_history")   内置，无需模板
memory=memory 传给 AgentExecutor      checkpointer= 传给 create_agent
memory.clear() 清除                   新 thread_id 自动新对话
需要理解 Memory 类层次结构             只需理解 checkpointer + thread_id

选择建议：
  新项目 → 现代 API（create_agent + checkpointer）
  旧项目 → 按需迁移
  学习理解 → 两者都了解
```

---

## 关键代码解析

### 1. checkpointer 的工作流程

```
用户发送消息 → Agent 处理流程：

  1. agent.invoke({"messages": [用户消息]}, config)

  2. Agent 内部：
     a. 从 checkpointer 加载 thread_id 的历史消息
     b. 历史消息 + 新消息 → 组成完整上下文
     c. 调用 LLM 推理
     d. 如果需要工具 → 调用工具 → 再推理
     e. 得到最终回复

  3. 自动保存到 checkpointer：
     checkpointer.put(thread_id, 更新后的消息列表)

  4. 返回响应

  整个过程对开发者透明，只需：
  - 创建时指定 checkpointer
  - 调用时指定 thread_id
```

### 2. thread_id 的隔离机制

```python
# thread_id 就像数据库的主键
# 不同的 thread_id → 不同的记忆空间

# 内部数据结构（简化理解）：
checkpointer_data = {
    "alice-001": {
        "messages": [
            HumanMessage("我叫 Alice"),
            AIMessage("你好 Alice！"),
        ]
    },
    "bob-001": {
        "messages": [
            HumanMessage("我叫 Bob"),
            AIMessage("你好 Bob！"),
        ]
    },
}

# 调用 agent.invoke(..., {"configurable": {"thread_id": "alice-001"}})
# → 只加载 alice-001 的消息
# → 不会看到 bob-001 的内容
```

### 3. get_state 查看状态

```python
# 查看某个 thread 的当前状态
config = {"configurable": {"thread_id": "alice-001"}}
state = agent.get_state(config)

# state.values["messages"] → 所有消息列表
# state.metadata → 元数据
# state.config → 配置信息

# 这对调试很有用：
# - 检查 Agent 是否正确记住了信息
# - 查看消息数量是否合理
# - 确认工具调用记录
```

---

## 常见问题及解决

### 问题 1：记忆没有生效

```python
# ❌ 常见错误：每次调用都创建新的 agent
for msg in messages:
    agent = create_agent(...)  # 每次重新创建！
    agent.invoke(...)          # 记忆不会延续

# ✅ 正确：复用同一个 agent 实例
agent = create_agent(
    model="gpt-4.1-mini",
    tools=[],
    checkpointer=InMemorySaver(),
)
for msg in messages:
    agent.invoke({"messages": [{"role": "user", "content": msg}]}, config)
```

### 问题 2：不同用户看到了对方的对话

```python
# ❌ 错误：所有用户用同一个 thread_id
config = {"configurable": {"thread_id": "default"}}
# 所有人共享一个记忆空间！

# ✅ 正确：每个用户独立的 thread_id
config = {"configurable": {"thread_id": f"user-{user_id}-{session_id}"}}
```

### 问题 3：程序重启后记忆丢失

```python
# ❌ InMemorySaver 重启就没了
checkpointer = InMemorySaver()

# ✅ 用 SqliteSaver 或 PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver(db_path="./agent_memory.db")
```

---

## 与 RAG 开发的关系

| 本场景技术 | RAG 应用 |
|-----------|---------|
| checkpointer 对话记忆 | RAG Agent 多轮检索中保持查询上下文 |
| thread_id 隔离 | 多用户知识库问答互不干扰 |
| 工具调用 + 记忆 | Agent 记住已检索的文档，避免重复检索 |
| get_state 查看状态 | 调试 RAG 对话流程，检查检索历史 |

---

**上一篇：** [06_反直觉点.md](./06_反直觉点.md)
**下一篇：** [07_实战代码_场景2_智能记忆压缩Agent.md](./07_实战代码_场景2_智能记忆压缩Agent.md)
