# LangSmith 可观测性

> LangSmith 是 LangChain 生态的可观测性平台——设两个环境变量，你的 LangGraph 工作流就自动拥有了生产级追踪、监控和调试能力

---

## 一句话定义

**LangSmith 通过环境变量零侵入接入 LangGraph，自动记录每个节点的输入/输出、LLM 调用详情和性能指标，是从开发到生产的全链路可观测性方案。**

---

## 为什么需要 LangSmith？

前面我们学了 `stream_mode="debug"`、`get_state()`、`get_state_history()` 和自定义调试工具。这些都是本地调试利器，但它们有一个共同的局限：

- **临时性**：终端输出关掉就没了
- **个人化**：只有你自己能看到
- **开发环境限定**：生产环境不可能到处加 `print()`

**前端类比：** 本地调试就像在浏览器 Console 里 `console.log()`——开发时很方便，但你不可能让线上用户帮你看 Console。你需要的是 Sentry、DataDog 这样的监控平台。LangSmith 就是 LangGraph 的 Sentry。

**日常生活类比：** 本地调试像是自己在家量体温，LangSmith 像是去医院做全面体检——有完整的检查报告、历史记录，医生（团队成员）也能看到。

---

## 1. 启用 LangSmith 追踪

### 最简配置：两个环境变量

```bash
# 方式一：Shell 环境变量
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=<your-api-key>
```

```python
# 方式二：Python 代码中设置
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your-api-key"
```

**就这样。** 不需要修改任何业务代码，不需要额外的装饰器，不需要 import 任何东西。你的 LangGraph 图的每一次 `invoke()` 和 `stream()` 都会自动被追踪。

### 为什么这么简单？

LangGraph 底层基于 LangChain 的 Runnable 协议，该协议内置了 tracing callback。当检测到 `LANGSMITH_TRACING=true` 时，会自动注入追踪回调，记录每个组件的执行信息。

```
你的代码                    LangSmith 平台
┌──────────┐               ┌──────────────┐
│ graph    │  自动上报      │  Web UI      │
│ .invoke()│ ──────────→   │  追踪详情    │
│          │  (环境变量     │  性能指标    │
│          │   触发)        │  错误记录    │
└──────────┘               └──────────────┘
```

---

## 2. 追踪配置：tags 和 metadata

启用追踪后，你可以通过 `config` 参数给每次执行打标签、附加元数据：

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict

class MyState(TypedDict):
    messages: list[str]
    result: str

def process_node(state: MyState) -> dict:
    return {"result": f"处理了 {len(state['messages'])} 条消息"}

# 构建图
builder = StateGraph(MyState)
builder.add_node("process", process_node)
builder.add_edge(START, "process")
builder.add_edge("process", END)
graph = builder.compile(checkpointer=InMemorySaver())

# 带追踪配置的调用
response = graph.invoke(
    {"messages": ["你好", "帮我查一下天气"], "result": ""},
    config={
        "configurable": {"thread_id": "user-session-001"},
        # ↓ LangSmith 追踪配置
        "tags": ["production", "v2.1", "weather-bot"],
        "metadata": {
            "user_id": "user_123",
            "session_id": "session_456",
            "environment": "production",
            "feature_flag": "new_retriever"
        }
    }
)
```

### tags 的用途

**tags 是字符串列表**，用于分类和筛选追踪记录：

| 常见 tag | 用途 |
|----------|------|
| `"production"` / `"staging"` | 区分环境 |
| `"v1.0"` / `"v2.1"` | 标记版本 |
| `"weather-bot"` / `"qa-bot"` | 标记功能模块 |
| `"experiment-A"` | A/B 测试标记 |

在 LangSmith Web UI 中，你可以按 tag 筛选，比如只看 `production` + `v2.1` 的追踪记录。

### metadata 的用途

**metadata 是键值对字典**，用于关联业务上下文：

```python
"metadata": {
    "user_id": "user_123",       # 关联到具体用户
    "session_id": "session_456", # 关联到具体会话
    "environment": "production", # 运行环境
    "model_version": "gpt-4o",   # 使用的模型
    "retriever_type": "hybrid"   # 检索策略
}
```

**调试场景：** 用户反馈"昨天下午的回答不对"，你可以在 LangSmith 中按 `user_id` + 时间范围筛选，精确找到那次执行的完整追踪。

---

## 3. 自定义项目名称

默认情况下，追踪记录会归入 LangSmith 的默认项目。你可以用 `langsmith` 库指定项目名称：

```python
import langsmith as ls

# 方式一：上下文管理器（推荐）
with ls.tracing_context(project_name="weather-bot-debug", enabled=True):
    response = graph.invoke(
        {"messages": ["今天北京天气怎么样？"], "result": ""},
        config={"configurable": {"thread_id": "debug-001"}}
    )

# 方式二：环境变量
import os
os.environ["LANGSMITH_PROJECT"] = "weather-bot-debug"
```

**为什么要分项目？**

| 项目 | 用途 |
|------|------|
| `my-bot-production` | 生产环境追踪 |
| `my-bot-staging` | 预发布环境测试 |
| `my-bot-debug` | 本地调试专用 |
| `my-bot-experiment` | 实验性功能追踪 |

不同项目的追踪记录互不干扰，方便管理和分析。

---

## 4. LangSmith 追踪提供的信息

每次图执行，LangSmith 会自动记录以下信息：

### 4.1 完整执行路径

```
graph.invoke()
├── __start__          (0.1ms)
├── retrieve_docs      (120ms)  ← 检索节点
│   └── embeddings     (45ms)   ← Embedding 调用
├── generate_answer    (850ms)  ← 生成节点
│   └── ChatOpenAI     (800ms)  ← LLM 调用
└── __end__            (0.1ms)
```

你可以看到每个节点的执行顺序、耗时、嵌套关系。

### 4.2 每个节点的输入/输出

```
节点: retrieve_docs
├── 输入: {"messages": ["今天天气怎么样？"], "docs": []}
└── 输出: {"docs": ["北京今天晴，25°C...", "..."]}
```

### 4.3 LLM 调用详情

```
ChatOpenAI 调用:
├── Model: gpt-4o
├── Prompt tokens: 1,234
├── Completion tokens: 256
├── Total tokens: 1,490
├── Latency: 800ms
├── Prompt: [完整的 system + user 消息]
└── Completion: [完整的模型回复]
```

### 4.4 延迟和性能指标

```
总耗时: 970ms
├── 检索耗时: 120ms (12.4%)
├── LLM 耗时: 800ms (82.5%)  ← 瓶颈在这里
└── 其他: 50ms (5.1%)
```

### 4.5 错误和异常追踪

当节点抛出异常时，LangSmith 会记录完整的错误堆栈、发生错误时的状态，以及错误前最后一个成功节点的输出。

---

## 5. LangGraph Studio 集成

LangGraph Studio 是专为 AI Agent 开发设计的可视化 IDE，与 LangSmith 深度集成。

### 核心功能

| 功能 | 描述 | 最佳场景 |
|------|------|----------|
| Graph Mode | 图结构 + 执行路径可视化 | 复杂多步代理调试 |
| Chat Mode | 对话式交互界面 | 聊天代理快速测试 |
| Interrupt | 在节点前后暂停，编辑状态 | 微调代理行为 |
| Hot Reload | 代码修改后自动重载 | 快速迭代开发 |
| Time Travel | 回到任意历史状态重新执行 | 复现和修复问题 |

### Graph Mode 工作流

```
1. 启动 LangGraph Studio
   ↓
2. 加载你的图定义（自动检测 langgraph.json）
   ↓
3. 发送输入，观察执行路径
   ↓
4. 点击任意节点，查看输入/输出/状态
   ↓
5. 修改代码 → 自动热重载 → 重新测试
```

### 状态编辑与时间旅行

Studio 最强大的调试能力是**运行时状态编辑**：

- 在节点执行前暂停（通过 `interrupt_before`）
- 查看并修改当前状态
- 继续执行，观察修改后的行为
- 回到任意历史检查点，从那里分叉执行

这和我们之前学的 `get_state()` / `update_state()` 是同一套机制，Studio 只是提供了可视化界面。

---

## 6. 生产监控最佳实践

### 6.1 环境分层追踪

```python
import os

# 根据环境设置不同的项目名
env = os.getenv("APP_ENV", "development")
os.environ["LANGSMITH_PROJECT"] = f"my-agent-{env}"

# 开发环境：详细追踪
if env == "development":
    os.environ["LANGSMITH_TRACING"] = "true"

# 生产环境：采样追踪（降低开销）
elif env == "production":
    os.environ["LANGSMITH_TRACING"] = "true"
    # 通过代码控制采样率
```

### 6.2 结构化 metadata 规范

团队统一 metadata 格式，方便后续查询和分析：

```python
def build_trace_config(user_id: str, session_id: str, version: str) -> dict:
    """构建标准化的追踪配置"""
    return {
        "tags": [
            os.getenv("APP_ENV", "dev"),
            f"v{version}",
        ],
        "metadata": {
            "user_id": user_id,
            "session_id": session_id,
            "app_version": version,
            "environment": os.getenv("APP_ENV", "dev"),
            "region": os.getenv("AWS_REGION", "unknown"),
        }
    }

# 使用
config = {
    "configurable": {"thread_id": session_id},
    **build_trace_config("user_123", "sess_456", "2.1.0")
}
response = graph.invoke(input_data, config=config)
```

### 6.3 关键指标监控

在 LangSmith 中关注这些指标：

| 指标 | 正常范围 | 告警阈值 | 含义 |
|------|----------|----------|------|
| 端到端延迟 | < 3s | > 10s | 用户体验 |
| LLM 调用延迟 | < 2s | > 5s | 模型响应速度 |
| Token 使用量 | 视场景 | 突增 50% | 成本控制 |
| 错误率 | < 1% | > 5% | 系统稳定性 |
| 检索命中率 | > 80% | < 60% | RAG 质量 |

---

## 7. 与本地调试的对比

| 维度 | 本地调试工具 | LangSmith |
|------|-------------|-----------|
| 启用方式 | 修改代码（print/logging） | 设置环境变量 |
| 侵入性 | 需要在代码中添加调试语句 | 零侵入，环境变量即可 |
| 可视化 | 终端文本输出 | Web UI，图形化展示 |
| 数据持久化 | 临时，关掉终端就没了 | 永久存储，随时回看 |
| 协作能力 | 仅个人可见 | 团队共享，协作排查 |
| 适用环境 | 仅开发环境 | 开发 + 预发布 + 生产 |
| 性能开销 | 几乎为零 | 有网络上报开销 |
| 信息丰富度 | 取决于你打印了什么 | 自动记录全部信息 |
| 历史对比 | 需要手动保存日志 | 内置历史对比功能 |

### 什么时候用哪个？

```
开发阶段（快速迭代）
├── print() / logging     ← 最快，即时反馈
├── stream_mode="debug"   ← 看执行流程
└── get_state()           ← 检查具体状态

测试阶段（问题定位）
├── get_state_history()   ← 历史追溯
├── 自定义 StreamWriter   ← 结构化日志
└── LangSmith             ← 可视化分析

生产阶段（监控运维）
└── LangSmith             ← 唯一推荐方案
    ├── 追踪记录
    ├── 性能监控
    ├── 错误告警
    └── 团队协作
```

---

## 8. 完整示例：从开发到生产的调试配置

```python
"""
LangSmith 可观测性完整示例
演示：如何为 LangGraph 工作流配置全链路追踪
"""
import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# ===== 1. 环境配置 =====
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your-api-key"  # 替换为你的 key
os.environ["LANGSMITH_PROJECT"] = "langgraph-debug-demo"


# ===== 2. 定义状态和节点 =====
class ChatState(TypedDict):
    messages: list[str]
    context: str
    answer: str


def retrieve(state: ChatState) -> dict:
    """模拟检索节点"""
    query = state["messages"][-1] if state["messages"] else ""
    # 模拟检索结果
    context = f"检索到与 '{query}' 相关的 3 篇文档"
    print(f"[retrieve] 检索完成: {context}")
    return {"context": context}


def generate(state: ChatState) -> dict:
    """模拟生成节点"""
    answer = f"基于上下文 '{state['context']}' 生成的回答"
    print(f"[generate] 生成完成: {answer}")
    return {"answer": answer}


# ===== 3. 构建图 =====
builder = StateGraph(ChatState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile(checkpointer=InMemorySaver())


# ===== 4. 带追踪配置的调用 =====
config = {
    "configurable": {"thread_id": "demo-001"},
    "tags": ["demo", "v1.0"],
    "metadata": {
        "user_id": "demo_user",
        "session_id": "demo_session",
        "environment": "development"
    }
}

result = graph.invoke(
    {"messages": ["LangGraph 怎么调试？"], "context": "", "answer": ""},
    config=config
)

print(f"\n最终结果: {result['answer']}")

# ===== 5. 结合本地调试验证 =====
snapshot = graph.get_state(config)
print(f"\n状态检查:")
print(f"  values: {snapshot.values}")
print(f"  next: {snapshot.next}")  # 应该是 () 表示执行完毕
```

**运行输出示例：**
```
[retrieve] 检索完成: 检索到与 'LangGraph 怎么调试？' 相关的 3 篇文档
[generate] 生成完成: 基于上下文 '检索到与 'LangGraph 怎么调试？' 相关的 3 篇文档' 生成的回答

最终结果: 基于上下文 '检索到与 'LangGraph 怎么调试？' 相关的 3 篇文档' 生成的回答

状态检查:
  values: {'messages': ['LangGraph 怎么调试？'], 'context': '...', 'answer': '...'}
  next: ()
```

同时，在 LangSmith Web UI 中你会看到这次执行的完整追踪记录，包括每个节点的耗时、输入输出，以及你设置的 tags 和 metadata。

---

## 小结

| 要点 | 说明 |
|------|------|
| 启用方式 | 两个环境变量：`LANGSMITH_TRACING` + `LANGSMITH_API_KEY` |
| 侵入性 | 零侵入，不需要修改业务代码 |
| tags | 字符串列表，用于分类筛选（环境、版本、功能） |
| metadata | 键值对字典，关联业务上下文（用户、会话） |
| 项目管理 | `LANGSMITH_PROJECT` 或 `ls.tracing_context()` |
| Studio | 可视化 IDE，支持状态编辑和时间旅行 |
| 生产建议 | 统一 metadata 规范 + 关键指标监控 + 环境分层 |

---

> **参考来源：**
> - [来源: atom/langgraph/L2_状态管理/11_状态调试技巧/reference/context7_langgraph_01.md | LangGraph 官方文档]
> - [来源: atom/langgraph/L2_状态管理/11_状态调试技巧/reference/fetch_studio调试_01.md | LangGraph Studio 调试指南]
