# Agent最佳实践 - 核心概念 1：Middleware 架构模式

> Middleware 是 LangChain 1.0 的核心创新——用中间件实现关注点分离，让 Agent 的可靠性、安全性、上下文管理等能力像乐高积木一样按需组合。

---

## 什么是 Middleware？

**Middleware（中间件）= 在 Agent 执行流程中插入的可组合处理器。**

每个 Middleware 只负责一个关注点（重试、安全、摘要等），通过 `create_agent()` 的 `middleware=[]` 参数按需组合。Agent 的核心逻辑（推理 + 工具调用）不需要修改。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
)

# 像搭乐高一样组合 Middleware
agent = create_agent(
    "gpt-4.1",
    tools=[search_tool, calculator_tool],
    middleware=[
        ModelRetryMiddleware(max_retries=3),              # 可靠性
        PIIMiddleware(strategy="mask"),                    # 安全
        SummarizationMiddleware(trigger=("tokens", 4000)), # 上下文
    ],
)
```

---

## 四大钩子机制

Middleware 通过四个钩子点介入 Agent 的执行流程：

```
Agent 执行流程与 Middleware 钩子

┌──────────────────────────────────────────────────┐
│                Agent 循环                          │
│                                                    │
│  用户输入                                          │
│    ↓                                               │
│  ┌──── before_model() ────┐                       │
│  │  • PII 检测用户输入      │                       │
│  │  • 上下文编辑/摘要       │                       │
│  │  • Token 预算检查        │                       │
│  └────────┬───────────────┘                       │
│           ↓                                        │
│  ┌──── wrap_model_call() ──┐                      │
│  │  ┌─────────────────┐    │                       │
│  │  │   LLM 调用       │    │  • 重试逻辑           │
│  │  │  (推理决策)      │    │  • 降级逻辑           │
│  │  └─────────────────┘    │  • 调用计数           │
│  └────────┬───────────────┘                       │
│           ↓                                        │
│  ┌──── after_model() ─────┐                       │
│  │  • 输出过滤/脱敏        │                       │
│  │  • 结果日志记录         │                       │
│  └────────┬───────────────┘                       │
│           ↓                                        │
│  ┌──── wrap_tool_call() ──┐                       │
│  │  ┌─────────────────┐    │                       │
│  │  │   工具执行       │    │  • 工具重试           │
│  │  │  (搜索/计算等)   │    │  • 人工审核           │
│  │  └─────────────────┘    │  • 执行限制           │
│  └────────┬───────────────┘                       │
│           ↓                                        │
│    继续循环 或 返回结果                               │
└──────────────────────────────────────────────────┘
```

### 钩子详解

| 钩子 | 签名 | 时机 | 能做什么 |
|------|------|------|---------|
| `before_model` | `(request) → request` | LLM 调用**前** | 修改请求（过滤消息、编辑提示词、检查 PII） |
| `after_model` | `(request, response) → response` | LLM 调用**后** | 修改响应（过滤输出、记录日志、缓存结果） |
| `wrap_model_call` | `(request, call_next) → response` | **环绕** LLM 调用 | 控制整个调用过程（重试、降级、计数） |
| `wrap_tool_call` | `(tool_call, call_next) → result` | **环绕**工具执行 | 控制工具执行（重试、审核、限制） |

### 源码中的钩子定义

```python
# langchain_v1/agents/middleware/types.py — AgentMiddleware 基类

class AgentMiddleware:
    """Agent 中间件基类，定义四个可覆写的钩子。"""

    def before_model(self, request: ModelRequest) -> ModelRequest:
        """在 LLM 调用之前修改请求。"""
        return request

    def after_model(self, request: ModelRequest, response: ModelResponse) -> ModelResponse:
        """在 LLM 调用之后修改响应。"""
        return response

    def wrap_model_call(self, request: ModelRequest, call_next):
        """环绕整个 LLM 调用，可用于重试、降级等。"""
        return call_next(request)

    def wrap_tool_call(self, tool_call, call_next):
        """环绕工具执行，可用于审核、重试等。"""
        return call_next(tool_call)
```

**关键设计**：每个方法都有默认实现（透传），Middleware 只需要覆写自己关心的钩子。

---

## Middleware 组合模式

### 堆叠执行（洋葱模型）

多个 Middleware 按声明顺序**从外到内**堆叠，执行时形成洋葱模型：

```
middleware=[A, B, C] 的执行顺序：

请求 → A.before_model()
         → B.before_model()
              → C.before_model()
                   → [LLM 调用]
              → C.after_model()
         → B.after_model()
       → A.after_model()
→ 响应

对于 wrap_model_call：
A.wrap_model_call(request,
    B.wrap_model_call(request,
        C.wrap_model_call(request,
            actual_llm_call
        )
    )
)
```

**重要原则：Middleware 的声明顺序很重要！**

```python
# ✅ 推荐顺序：外层限制 → 中层重试 → 内层安全
middleware=[
    ModelCallLimitMiddleware(run_limit=25),    # 最外层：限制总调用次数
    ModelRetryMiddleware(max_retries=3),        # 中层：每次调用可重试
    PIIMiddleware(strategy="mask"),             # 内层：处理输入输出
    SummarizationMiddleware(trigger=("tokens", 4000)),  # 内层：上下文管理
]

# ❌ 错误顺序：重试在限制外面
middleware=[
    ModelRetryMiddleware(max_retries=3),        # 重试可能突破限制！
    ModelCallLimitMiddleware(run_limit=25),
]
```

### 组合逻辑（源码分析）

```python
# langchain_v1/agents/factory.py — create_agent 中 Middleware 的组合逻辑

def create_agent(model, tools, *, middleware=None, ...):
    """创建带 Middleware 的 Agent。"""

    # 1. 初始化 Middleware 链
    middleware_chain = middleware or []

    # 2. 构建 LangGraph StateGraph
    graph = StateGraph(AgentState)

    # 3. 在 LLM 调用节点中应用 Middleware 钩子
    def model_node(state):
        request = build_model_request(state)

        # 依次调用 before_model
        for mw in middleware_chain:
            request = mw.before_model(request)

        # 构建 wrap_model_call 调用链
        def actual_call(req):
            return model.invoke(req.messages)

        wrapped_call = actual_call
        for mw in reversed(middleware_chain):
            prev_call = wrapped_call
            wrapped_call = lambda req, c=mw, p=prev_call: c.wrap_model_call(req, p)

        response = wrapped_call(request)

        # 依次调用 after_model（逆序）
        for mw in reversed(middleware_chain):
            response = mw.after_model(request, response)

        return response

    # 4. 编译图
    return graph.compile(checkpointer=checkpointer)
```

---

## 14 个内置 Middleware

LangChain 1.0 提供了 14 个开箱即用的 Middleware，覆盖生产环境的主要需求：

### 可靠性类（3 个）

| Middleware | 用途 | 关键参数 |
|-----------|------|---------|
| `ModelRetryMiddleware` | LLM 调用重试 | `max_retries`, `retry_on`, `backoff_factor`, `jitter` |
| `ModelFallbackMiddleware` | 模型降级 | `fallbacks` (备用模型列表) |
| `ToolRetryMiddleware` | 工具调用重试 | `max_retries`, `tools` (指定哪些工具) |

### 限制保护类（2 个）

| Middleware | 用途 | 关键参数 |
|-----------|------|---------|
| `ModelCallLimitMiddleware` | 模型调用次数限制 | `thread_limit`, `run_limit`, `exit_behavior` |
| `ToolCallLimitMiddleware` | 工具调用次数限制 | `allowed`/`blocked` (按工具限制), `exit_behavior` |

### 安全合规类（3 个）

| Middleware | 用途 | 关键参数 |
|-----------|------|---------|
| `PIIMiddleware` | PII 检测与脱敏 | `strategy` (block/redact/mask/hash), `pii_types` |
| `HumanInTheLoopMiddleware` | 人工审核 | `interrupt_on` (按工具配置审核条件) |
| `ShellToolMiddleware` | Shell 命令安全执行 | `sandbox` (docker/codex/host) |

### 上下文管理类（2 个）

| Middleware | 用途 | 关键参数 |
|-----------|------|---------|
| `SummarizationMiddleware` | 对话摘要压缩 | `trigger` (fraction/tokens/messages), `keep` |
| `ContextEditingMiddleware` | 上下文编辑 | 自定义编辑逻辑 |

### 高级功能类（4 个）

| Middleware | 用途 | 关键参数 |
|-----------|------|---------|
| `TodoListMiddleware` | 任务规划 | 自动提供 `write_todos` 工具 |
| `LLMToolEmulator` | LLM 模拟工具 | 用 LLM 代替真实工具执行 |
| `LLMToolSelectorMiddleware` | 智能工具选择 | 根据上下文动态选择工具 |
| `FilesystemFileSearchMiddleware` | 文件搜索 | 在文件系统中搜索相关文件 |

---

## 自定义 Middleware 开发

### 方式 1：继承 AgentMiddleware 基类

```python
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse

class LoggingMiddleware(AgentMiddleware):
    """记录每次 LLM 调用的日志。"""

    def before_model(self, request: ModelRequest) -> ModelRequest:
        print(f"[LOG] 即将调用 LLM，消息数: {len(request.messages)}")
        return request

    def after_model(self, request: ModelRequest, response: ModelResponse) -> ModelResponse:
        print(f"[LOG] LLM 返回，工具调用数: {len(response.tool_calls)}")
        return response

# 使用
agent = create_agent(
    "gpt-4.1",
    tools=tools,
    middleware=[LoggingMiddleware()],
)
```

### 方式 2：使用钩子装饰器

```python
from langchain.agents.middleware import before_model, after_model, wrap_tool_call

@before_model
def check_input_length(request):
    """检查输入长度，防止过长输入。"""
    total_tokens = sum(len(m.content) for m in request.messages)
    if total_tokens > 10000:
        # 截断过长的消息
        request.messages = request.messages[-10:]
    return request

@wrap_tool_call
def log_tool_execution(tool_call, call_next):
    """记录工具执行时间。"""
    import time
    start = time.time()
    result = call_next(tool_call)
    elapsed = time.time() - start
    print(f"[TOOL] {tool_call.name} 执行耗时: {elapsed:.2f}s")
    return result

# 使用：装饰器函数也可以作为 middleware 传入
agent = create_agent(
    "gpt-4.1",
    tools=tools,
    middleware=[check_input_length, log_tool_execution],
)
```

### 方式 3：使用 hook_config 配置

```python
from langchain.agents.middleware import hook_config

# 通过配置快速创建简单的钩子
config = hook_config(
    before_model=lambda req: print(f"Input: {req.messages[-1].content[:50]}...") or req,
    after_model=lambda req, res: print(f"Output: {res}") or res,
)

agent = create_agent("gpt-4.1", tools=tools, middleware=[config])
```

---

## Middleware vs 回调系统

初学者容易混淆 Middleware 和 Callbacks（09 中学过的回调系统），它们有本质区别：

| 维度 | Middleware（中间件） | Callbacks（回调） |
|------|--------------------|--------------------|
| **能否修改数据** | ✅ 可以修改请求和响应 | ❌ 只能观察，不能修改 |
| **执行模型** | 洋葱模型（环绕执行） | 线性通知（事件触发） |
| **主要用途** | 逻辑控制（重试、安全、上下文管理） | 可观测性（日志、追踪、监控） |
| **影响执行** | ✅ 可以改变执行流程 | ❌ 不应影响执行流程 |
| **引入时间** | LangChain 1.0 (langchain_v1) | LangChain 0.x (langchain_core) |

```python
# Callback：只观察，不干预
class LogCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM started with {len(prompts)} prompts")
    # ⚠️ 无法修改 prompts！

# Middleware：可以修改数据和控制流程
class PIIMiddleware(AgentMiddleware):
    def before_model(self, request):
        # ✅ 可以修改请求！
        for msg in request.messages:
            msg.content = mask_pii(msg.content)
        return request
```

**一句话区分**：Callback 是**观察者**（看但不碰），Middleware 是**拦截者**（可以改、可以拦、可以重试）。

---

## 实际应用中的 Middleware 组合策略

### 场景 1：智能客服 Agent

```python
agent = create_agent(
    "gpt-4.1",
    tools=[search_kb, create_ticket, check_order],
    middleware=[
        ModelCallLimitMiddleware(run_limit=20),          # 防止对话过长
        ModelRetryMiddleware(max_retries=2),              # API 偶尔超时
        PIIMiddleware(strategy="mask"),                   # 客户信息脱敏
        SummarizationMiddleware(trigger=("tokens", 4000)), # 长对话压缩
    ],
)
```

### 场景 2：代码助手 Agent

```python
agent = create_agent(
    "gpt-4.1",
    tools=[read_file, write_file, run_shell],
    middleware=[
        ModelCallLimitMiddleware(run_limit=50),           # 复杂任务允许更多步
        ModelRetryMiddleware(max_retries=3),               # 重试
        ModelFallbackMiddleware(fallbacks=["gpt-4.1-mini"]), # 降级
        ShellToolMiddleware(sandbox="docker"),             # Shell 沙箱
        HumanInTheLoopMiddleware(
            interrupt_on={"write_file": {"always": True}}, # 写文件需审核
        ),
    ],
)
```

### 场景 3：金融合规 Agent

```python
agent = create_agent(
    "gpt-4.1",
    tools=[query_account, transfer_funds, generate_report],
    middleware=[
        ModelCallLimitMiddleware(run_limit=15),            # 严格限制
        ModelRetryMiddleware(max_retries=3),                # 重试
        ModelFallbackMiddleware(fallbacks=["claude-3.5-sonnet"]), # 跨供应商降级
        PIIMiddleware(
            strategy="mask",
            pii_types=["credit_card", "email"],
        ),                                                 # 金融数据脱敏
        HumanInTheLoopMiddleware(
            interrupt_on={"transfer_funds": {"always": True}}, # 转账必须审核
        ),
        SummarizationMiddleware(trigger=("tokens", 8000)), # 保留更多上下文
    ],
)
```

---

## 关键设计原则总结

1. **单一职责**：每个 Middleware 只负责一个关注点
2. **可组合性**：通过 `middleware=[]` 列表按需组合
3. **顺序敏感**：声明顺序决定执行顺序（外层先执行）
4. **默认透传**：未覆写的钩子自动透传，不影响其他 Middleware
5. **三种开发方式**：继承基类 / 装饰器 / hook_config 配置

---

**下一步**: 阅读 [03_核心概念_2_可靠性设计模式.md](./03_核心概念_2_可靠性设计模式.md)，深入了解重试、降级和限制三重保护机制

---

**数据来源**：
- [源码分析] langchain_v1/agents/factory.py — create_agent + Middleware 组合逻辑
- [源码分析] langchain_v1/agents/middleware/types.py — AgentMiddleware 基类 + 四大钩子
- [源码分析] langchain_v1/agents/middleware/__init__.py — 14 个内置 Middleware 导出
