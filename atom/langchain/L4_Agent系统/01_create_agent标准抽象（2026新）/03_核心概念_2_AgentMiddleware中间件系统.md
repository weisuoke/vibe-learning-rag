# 核心概念 2：AgentMiddleware 中间件系统

> LangChain 1.0 最大的架构创新——借鉴 Web 服务器中间件模式，用可组合的钩子取代参数爆炸，让 Agent 行为定制从"改代码"变成"插插件"

---

## 为什么需要 Middleware？

### 旧版的痛点：参数爆炸

LangChain 过去三年的 Agent 抽象有一个共同问题：**开发者在非简单场景下缺乏对上下文工程的控制力。**

每当社区提出新需求，团队就往 API 上加参数：

```
2023: create_react_agent(model, tools)
2024: create_react_agent(model, tools, prompt, memory, callbacks, ...)
2025: create_react_agent(model, tools, prompt, memory, callbacks,
                         pre_model_hook, post_model_hook,
                         dynamic_prompt, model_selector, ...)
```

问题越来越严重：
- **参数之间有隐式依赖**：`pre_model_hook` 和 `dynamic_prompt` 谁先执行？
- **无法组合**：想同时用重试 + 日志 + 缓存，三个功能的代码纠缠在一起
- **无法复用**：在项目 A 写的重试逻辑，搬到项目 B 要重写

### Middleware 的解法：横切关注点分离

LangChain 官方博客（2025年9月）明确指出：

> "核心 Agent 循环仍由 model node 和 tool node 组成，但所有定制化需求通过 Middleware 注入。"

这个思路直接借鉴了 Web 服务器中间件：

| Web 中间件 | Agent 中间件 |
|-----------|-------------|
| Express.js `app.use(cors())` | `create_agent(middleware=[PIIMiddleware()])` |
| Django `MIDDLEWARE = [...]` | `middleware=[RetryMW(), LogMW()]` |
| 请求 → 中间件链 → 路由处理 → 中间件链 → 响应 | 输入 → 中间件链 → 模型调用 → 中间件链 → 输出 |

核心循环不变，行为通过组合叠加。每个 Middleware 只做一件事，互不干扰，随意插拔。

---

## AgentMiddleware 基类

### 源码定义

```python
# 来源: langchain/agents/middleware/types.py

class AgentMiddleware(Generic[StateT, ContextT, ResponseT]):
    """Agent 中间件基类"""

    # ---- 类属性 ----
    state_schema: type[StateT]   # 声明需要的状态字段
    tools: Sequence[BaseTool]    # 注册额外工具

    # ---- 标识 ----
    @property
    def name(self) -> str:
        return self.__class__.__name__  # 默认用类名

    # ---- 生命周期钩子（4种）----
    def before_agent(self, state, runtime) -> dict | None: ...
    def before_model(self, state, runtime) -> dict | None: ...
    def after_model(self, state, runtime) -> dict | None: ...
    def after_agent(self, state, runtime) -> dict | None: ...

    # ---- 包装器钩子（2种）----
    def wrap_model_call(self, request, handler) -> ModelResponse | AIMessage | ExtendedModelResponse: ...
    def wrap_tool_call(self, request, execute) -> ToolMessage | Command: ...

    # ---- 异步版本（每个钩子都有）----
    async def abefore_agent / abefore_model / aafter_model / aafter_agent: ...
    async def awrap_model_call / awrap_tool_call: ...
```

三个泛型参数的含义：

| 参数 | 含义 | 什么时候需要指定 |
|------|------|-----------------|
| `StateT` | 自定义状态类型 | 需要扩展 AgentState 时 |
| `ContextT` | 运行时上下文类型 | 需要读取 `runtime.context` 时 |
| `ResponseT` | 结构化输出类型 | 配合 `response_format` 时 |

大多数场景不需要指定泛型参数，直接继承 `AgentMiddleware` 即可。

---

## 6 种钩子的设计逻辑

### 为什么是 6 种，不是更多或更少？

这 6 种钩子覆盖了 Agent 循环的**每一个可干预点**：

```
Agent 生命周期
│
├── before_agent ──────── Agent 启动前（初始化、验证）
│
│   ┌── Agent 循环 ─────────────────────────┐
│   │                                       │
│   ├── before_model ──── 模型调用前（注入上下文）
│   │                                       │
│   ├── wrap_model_call ── 包裹模型调用（重试、缓存）
│   │                                       │
│   ├── after_model ───── 模型调用后（检查、跳转）
│   │                                       │
│   ├── wrap_tool_call ─── 包裹工具调用（权限、沙箱）
│   │                                       │
│   └── 回到 before_model（如果有 tool_calls）
│
└── after_agent ──────── Agent 结束后（清理、日志）
```

分成两大类：

**生命周期钩子**（4 种）—— 在特定时机触发，接收 `(state, runtime)`，返回状态更新：
- `before_agent`：整个 Agent 执行前，只跑一次
- `before_model`：每次模型调用前，可能跑多次
- `after_model`：每次模型调用后，可能跑多次
- `after_agent`：整个 Agent 执行后，只跑一次

**包装器钩子**（2 种）—— 包裹执行过程，接收 `(request, handler)`，完全控制执行流：
- `wrap_model_call`：包裹模型调用
- `wrap_tool_call`：包裹工具调用

### 生命周期钩子 vs 包装器钩子

这是理解 Middleware 系统最关键的区分：

| 维度 | 生命周期钩子 | 包装器钩子 |
|------|-------------|-----------|
| 参数 | `(state, runtime)` | `(request, handler)` |
| 能力 | 读写状态、控制跳转 | 完全控制执行流程 |
| 能否跳过执行 | 只能通过 `jump_to` 间接跳过 | 不调用 `handler` 即可短路 |
| 能否修改请求 | 不能直接修改模型请求 | 通过 `request.override()` 修改 |
| 能否重试 | 不能 | 在循环中多次调用 `handler` |
| 复杂度 | 低 | 高 |
| 适用场景 | 日志、计数、状态更新 | 重试、缓存、降级、拦截 |

**经验法则：** 能用生命周期钩子解决的，不要用包装器钩子。包装器钩子更强大，但也更容易出错。

---

## 生命周期钩子详解

### 1. before_agent —— Agent 启动前

```python
class InitMiddleware(AgentMiddleware):
    def before_agent(self, state, runtime) -> dict | None:
        """只在 Agent 启动时执行一次"""
        # 典型用途：输入验证
        if not state["messages"]:
            raise ValueError("至少需要一条消息")

        # 典型用途：初始化自定义状态
        return {"session_start_time": time.time()}
```

**关键特征：** 整个 Agent 生命周期只执行一次。适合做初始化和验证。

### 2. before_model —— 每次模型调用前

```python
class RAGInjectionMiddleware(AgentMiddleware):
    def before_model(self, state, runtime) -> dict | None:
        """每次调模型前注入检索结果"""
        last_msg = state["messages"][-1]
        docs = retriever.invoke(last_msg.content)
        context = "\n".join(d.page_content for d in docs[:3])
        return {
            "messages": [SystemMessage(content=f"参考资料：\n{context}")]
        }
```

**关键特征：** Agent 循环内每轮都执行。这是最常用的钩子——动态 prompt、上下文注入、消息摘要都在这里做。

**跳转控制：** 返回 `{"jump_to": "end"}` 可以跳过模型调用直接结束。

### 3. after_model —— 每次模型调用后

```python
class SafetyCheckMiddleware(AgentMiddleware):
    def after_model(self, state, runtime) -> dict | None:
        """模型返回后检查输出安全性"""
        last_msg = state["messages"][-1]

        if self._contains_harmful_content(last_msg.content):
            # 替换为安全回复，跳转到结束
            return {
                "messages": [AIMessage(content="抱歉，我无法回答这个问题。")],
                "jump_to": "end"
            }
        return None
```

**关键特征：** 模型已经返回了 AIMessage，你可以检查、修改、甚至替换它。

### 4. after_agent —— Agent 结束后

```python
class CleanupMiddleware(AgentMiddleware):
    def after_agent(self, state, runtime) -> dict | None:
        """Agent 执行完毕，做清理和统计"""
        total_msgs = len(state["messages"])
        print(f"[完成] 共 {total_msgs} 条消息")
        # 关闭数据库连接、释放资源等
        return None
```

**关键特征：** 整个 Agent 生命周期只执行一次。适合做清理和统计。

---

## 包装器钩子详解

### 5. wrap_model_call —— 最强大的钩子

```python
def wrap_model_call(self, request: ModelRequest, handler) -> ModelCallResult:
```

`request` 是 `ModelRequest` 对象，包含模型调用的全部信息（model、messages、tools、state、runtime 等）。通过 `request.override(**kwargs)` 可以不可变地修改请求。

`handler` 是一个回调函数——调用它就是执行模型（或下一层 middleware 的 wrap）。

**三种典型用法：**

```python
# 用法 1：重试
class RetryMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        for attempt in range(3):
            try:
                return handler(request)
            except Exception:
                if attempt == 2: raise

# 用法 2：缓存（短路返回，不调用 handler）
class CacheMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        key = hash(str(request.messages))
        if key in self.cache:
            return self.cache[key]  # 直接返回，跳过模型调用
        result = handler(request)
        self.cache[key] = result
        return result

# 用法 3：修改请求（动态切换模型）
class ModelSwitchMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        if len(request.messages) > 50:
            return handler(request.override(model=cheap_model))
        return handler(request)
```

### 6. wrap_tool_call —— 工具调用的守门人

```python
class ToolAuditMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request, execute):
        tool_name = request.tool_call["name"]
        tool_args = request.tool_call["args"]

        # 记录审计日志
        print(f"[AUDIT] 调用工具: {tool_name}, 参数: {tool_args}")

        # 权限检查
        if tool_name in self.blocked_tools:
            return ToolMessage(
                content=f"工具 {tool_name} 被禁止",
                tool_call_id=request.tool_call["id"],
                status="error",
            )

        # 正常执行
        result = execute(request)
        print(f"[AUDIT] 工具返回: {result.content[:100]}")
        return result
```

---

## 执行顺序：洋葱模型

当传入多个 Middleware 时，执行顺序遵循洋葱模型——和 Express.js、Koa 完全一致。

### before/after 钩子的顺序

```python
agent = create_agent(
    model="openai:gpt-4o",
    middleware=[A, B, C],  # 三个 middleware
)
```

```
进入方向（before）：A → B → C
退出方向（after）：C → B → A

完整流程：
A.before_model → B.before_model → C.before_model
    → [模型调用]
C.after_model → B.after_model → A.after_model
```

### wrap 钩子的嵌套

wrap 钩子像俄罗斯套娃一样嵌套：

```
A.wrap_model_call(request,
    handler_A = B.wrap_model_call(request,
        handler_B = C.wrap_model_call(request,
            handler_C = actual_model_call
        )
    )
)
```

**A 是最外层**——最先看到请求，最后看到响应。如果 A 是重试中间件，它的重试会包裹 B 和 C 的全部逻辑。

**设计启示：** Middleware 的顺序很重要。重试放最外层，缓存放最内层，日志放中间。

---

## 两种定义方式

### 方式 1：装饰器（简单场景）

当你只需要一个钩子、不需要实例状态时，装饰器最简洁：

```python
from langchain.agents.middleware import (
    before_model, after_model, wrap_model_call,
    wrap_tool_call, dynamic_prompt
)

# 生命周期钩子
@before_model
def log_messages(state: AgentState, runtime: Runtime) -> dict | None:
    print(f"消息数: {len(state['messages'])}")
    return None

# 包装器钩子
@wrap_model_call
def retry_on_error(request: ModelRequest, handler):
    for attempt in range(3):
        try:
            return handler(request)
        except Exception:
            if attempt == 2: raise

# 动态 prompt（特殊装饰器，内部实现为 wrap_model_call）
@dynamic_prompt
def role_based_prompt(request: ModelRequest) -> str:
    role = request.runtime.context.get("role", "user")
    return f"你正在帮助一位 {role}。"

# 使用
agent = create_agent(
    model="openai:gpt-4o",
    middleware=[log_messages, retry_on_error, role_based_prompt],
)
```

装饰器内部用 `type()` 动态创建了一个 `AgentMiddleware` 子类并实例化。

### 方式 2：类（复杂场景）

当你需要多个钩子、实例状态、工具注册时，用类：

```python
class MonitoringMiddleware(AgentMiddleware):
    """生产级监控中间件：多钩子 + 实例状态"""

    def __init__(self, alert_threshold: int = 10):
        self.alert_threshold = alert_threshold
        self.call_count = 0

    def before_model(self, state, runtime):
        self.call_count += 1
        if self.call_count > self.alert_threshold:
            print(f"[ALERT] 模型调用次数超过 {self.alert_threshold}")
        return None

    def wrap_model_call(self, request, handler):
        import time
        start = time.time()
        response = handler(request)
        print(f"[PERF] 模型调用耗时: {time.time() - start:.2f}s")
        return response

# 使用（注意要实例化）
agent = create_agent(
    model="openai:gpt-4o",
    middleware=[MonitoringMiddleware(alert_threshold=5)],
)
```

### 选择指南

| 场景 | 推荐方式 | 原因 |
|------|----------|------|
| 简单日志 | `@after_model` | 一行搞定 |
| 动态 prompt | `@dynamic_prompt` | 专用装饰器 |
| 重试逻辑 | `@wrap_model_call` | 单钩子足够 |
| 监控 + 重试 + 日志 | 类 | 需要多个钩子 |
| 需要配置参数 | 类 | `__init__` 接收参数 |
| 需要注册工具 | 类 | `tools` 类属性 |
| 需要扩展状态 | 类 | `state_schema` 类属性 |

---

## 自定义状态扩展

Middleware 可以通过 `state_schema` 声明自己需要的状态字段：

```python
from langchain.agents.middleware.types import PrivateStateAttr

class MyState(AgentState):
    call_count: Annotated[int, PrivateStateAttr]  # 纯内部字段，外部不可见

class CountingMiddleware(AgentMiddleware[MyState, None, Any]):
    state_schema = MyState

    def before_model(self, state, runtime):
        count = state.get("call_count", 0) + 1
        print(f"第 {count} 次模型调用")
        return {"call_count": count}
```

当多个 Middleware 各自定义 `state_schema` 时，`create_agent` 内部的 `_resolve_schema()` 会自动合并所有字段。你不需要手动协调——每个 Middleware 只管声明自己需要的字段。

---

## 19 个内置 Middleware 速览

LangChain 1.0 内置了 19 个 Middleware，分为 6 大类。这里只列核心分类，详细用法见 [03_核心概念_5_内置Middleware全景.md](./03_核心概念_5_内置Middleware全景.md)。

| 分类 | 代表 Middleware | 核心钩子 |
|------|----------------|---------|
| 模型控制 | `ModelRetryMiddleware`, `ModelFallbackMiddleware`, `ModelCallLimitMiddleware` | `wrap_model_call` / `before_model` |
| 工具控制 | `ToolRetryMiddleware`, `ToolCallLimitMiddleware`, `LLMToolEmulator` | `wrap_tool_call` / `after_model` |
| 安全审核 | `HumanInTheLoopMiddleware`, `PIIMiddleware` | `after_model` / `before_model` |
| 上下文管理 | `SummarizationMiddleware`, `ContextEditingMiddleware` | `before_model` / `wrap_model_call` |
| 工具增强 | `ShellToolMiddleware`, `FilesystemFileSearchMiddleware`, `TodoListMiddleware` | `tools` 注册 |

---

## RAG 场景的 Middleware 实战

把 Middleware 系统应用到 RAG 开发中，展示组合威力：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRetryMiddleware, ModelCallLimitMiddleware,
    SummarizationMiddleware, dynamic_prompt, after_model,
)

@dynamic_prompt
def rag_context(request):
    docs = retriever.invoke(request.messages[-1].content)
    context = "\n---\n".join(d.page_content for d in docs[:5])
    return f"基于以下资料回答问题：\n{context}\n\n如果资料不足，请明确说明。"

@after_model
def log_quality(state, runtime):
    if "资料不足" in state["messages"][-1].content:
        print("[WARN] 检索结果可能不够相关")
    return None

rag_agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool],
    middleware=[
        rag_context,                                         # 检索注入
        log_quality,                                         # 质量监控
        ModelRetryMiddleware(max_retries=2),                  # 模型重试
        ModelCallLimitMiddleware(run_limit=10),                # 防无限循环
        SummarizationMiddleware(model="openai:gpt-4o-mini"),  # 长对话摘要
    ],
)
```

5 个 Middleware 各司其职，互不干扰。想加 PII 脱敏？往列表里插一个 `PIIMiddleware`。想去掉摘要？从列表里删掉。这就是组合的力量。

---

## Middleware 与旧版方案的对比

| 维度 | 旧版（参数堆叠） | 新版（Middleware） |
|------|-----------------|-------------------|
| 定制方式 | 传参数 / 继承重写 | 组合 Middleware |
| 可复用性 | 低 | 高（独立于 Agent） |
| 可组合性 | 差（参数间有隐式依赖） | 好（洋葱模型） |
| 社区生态 | 无法共享 | 可发布为 pip 包 |

---

## 最佳实践

1. **简单场景用装饰器，复杂场景用类** —— 不要为了一行日志写一个类
2. **每个 Middleware 只做一件事** —— 重试是重试，日志是日志，不要混在一起
3. **注意执行顺序** —— 重试放外层，缓存放内层，日志放中间
4. **优先用生命周期钩子** —— `wrap_model_call` 更强大但更容易出错
5. **善用内置 Middleware** —— 19 个内置覆盖了 90% 的生产需求，不要重复造轮子
6. **用 `PrivateStateAttr` 隐藏内部状态** —— Middleware 的计数器、缓存等不应暴露给外部

---

**上一篇**: [03_核心概念_1_create_agent工厂函数.md](./03_核心概念_1_create_agent工厂函数.md)
**下一篇**: [03_核心概念_3_Middleware基类与钩子系统.md](./03_核心概念_3_Middleware基类与钩子系统.md)

---

> [来源: sourcecode/langchain/libs/langchain_v1/langchain/agents/middleware/types.py]
> [来源: reference/context7_langchain_01.md]
> [来源: reference/fetch_middleware_blog_01.md]
