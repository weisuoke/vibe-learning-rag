# 核心概念 4: ModelRequest 与 ModelResponse 数据流

> 理解 Agent 循环中请求与响应的封装、流转与扩展机制

---

## 为什么需要专门的数据流类型？

在 `create_agent` 的 Agent 循环中，每一轮模型调用都涉及大量信息：用哪个模型、传什么消息、带哪些工具、系统提示是什么、当前状态如何……如果把这些散落在函数参数里，middleware 就无法统一拦截和修改。

LangChain 1.0 的做法是：**用 dataclass 把请求和响应各封装成一个对象**，让 middleware 可以像操作"信封"一样读取、修改、转发。

**前端类比：** 就像浏览器的 `Request` 和 `Response` 对象——你不会把 URL、headers、body 散着传，而是封装成一个对象，这样 fetch interceptor 才能统一处理。

---

## 完整数据流概览

```
用户输入 → AgentState.messages
  ↓
before_model(state) → 可更新状态或跳转
  ↓
构建 ModelRequest（model + messages + tools + system_message + ...）
  ↓
wrap_model_call(request, handler)  ← 洋葱模型
  │
  ├─ 外层 middleware 可修改 request（通过 override）
  ├─ handler(request) → 实际调用模型
  ├─ 返回 ModelResponse / AIMessage / ExtendedModelResponse
  │
  ↓
after_model(state) → 可更新状态或跳转
  ↓
检查 AIMessage.tool_calls
  ├─ 有 tool_calls → wrap_tool_call → ToolMessage → 回到 before_model
  └─ 无 tool_calls → 结束循环（或返回 structured_response）
```

这个流程中，`ModelRequest` 是"去程信封"，`ModelResponse` 是"回程信封"。

---

## 1. ModelRequest：请求封装

### 1.1 数据结构定义

```python
from dataclasses import dataclass, field, replace
from typing import Any, Generic

@dataclass(init=False)
class ModelRequest(Generic[ContextT]):
    """模型请求信息的完整容器"""

    model: BaseChatModel              # 使用哪个模型
    messages: list[AnyMessage]        # 对话消息（不含 system message）
    system_message: SystemMessage | None  # 系统提示（独立存放）
    tool_choice: Any | None           # 工具选择策略
    tools: list[BaseTool | dict]      # 可用工具列表
    response_format: ResponseFormat | None  # 结构化输出格式
    state: AgentState[Any]            # 当前 Agent 状态
    runtime: Runtime[ContextT]        # 运行时上下文
    model_settings: dict[str, Any] = field(default_factory=dict)  # 额外模型参数
```

### 1.2 每个字段的作用

| 字段 | 类型 | Middleware 使用场景 |
|------|------|---------------------|
| `model` | `BaseChatModel` | 动态切换模型（如回退到备用模型） |
| `messages` | `list[AnyMessage]` | 过滤/截断/注入消息（如摘要中间件） |
| `system_message` | `SystemMessage \| None` | 动态修改系统提示（如注入上下文） |
| `tool_choice` | `Any \| None` | 强制使用特定工具（如 `"required"`） |
| `tools` | `list[BaseTool \| dict]` | 过滤可用工具（如按权限控制） |
| `response_format` | `ResponseFormat \| None` | 切换结构化输出格式 |
| `state` | `AgentState` | 读取当前状态（少直接修改） |
| `runtime` | `Runtime[ContextT]` | 访问用户传入的 context |
| `model_settings` | `dict[str, Any]` | 设置 temperature、max_tokens 等 |

**关键区分：`messages` vs `state["messages"]`**

```python
def wrap_model_call(self, request, handler):
    # request.messages → 即将发给模型的消息（不含 system message）
    # request.state["messages"] → Agent 状态中的完整消息列表（含 system）
    # 两者不一定相同！middleware 可能已经修改了 request.messages
    pass
```

### 1.3 init=False 与向后兼容

`ModelRequest` 使用 `@dataclass(init=False)` 并自定义 `__init__`，处理 `system_prompt`（旧）→ `system_message`（新）的兼容：

```python
# 新版推荐：直接传 SystemMessage 对象
request = ModelRequest(
    model=my_model,
    messages=[HumanMessage(content="你好")],
    system_message=SystemMessage(content="你是助手"),
)

# 旧版兼容：system_prompt 字符串自动转为 SystemMessage
request = ModelRequest(model=my_model, messages=[], system_prompt="你是助手")

# 两者不能同时指定 → ValueError
```

### 1.4 override() 方法：不可变更新的核心

这是 `ModelRequest` 最重要的方法。middleware **不应该直接修改** request 的属性，而是通过 `override()` 创建新实例：

```python
def override(self, **overrides) -> ModelRequest[ContextT]:
    """返回一个新的 ModelRequest，指定字段被替换，原实例不变。"""
    return replace(self, **overrides)
```

底层使用 `dataclasses.replace()`，这是 Python 标准库的不可变更新方式。

**为什么要不可变？** 因为 middleware 是链式执行的。如果 middleware A 直接修改了 request，middleware B 看到的就是被改过的版本——这会导致难以调试的副作用。不可变模式保证每个 middleware 看到的是独立的请求快照。

**使用示例：**

```python
# 单字段覆盖
new_request = request.override(model=ChatOpenAI(model="gpt-4o"))
new_request = request.override(
    system_message=SystemMessage(content="新的系统提示")
)

# 过滤工具（只保留允许的工具）
allowed = {"search", "calculator"}
new_request = request.override(
    tools=[t for t in request.tools if t.name in allowed]
)

# 多字段同时覆盖
new_request = request.override(
    model=ChatOpenAI(model="gpt-4o"),
    system_message=SystemMessage(content="你是专家级助手"),
    model_settings={"temperature": 0.0},
)
```

**在 middleware 中的完整用法：**

```python
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage

class InjectContextMiddleware(AgentMiddleware):
    """在系统提示中注入额外上下文"""

    def wrap_model_call(self, request, handler):
        # 读取当前系统消息
        original = request.system_message

        # 构建新的系统消息
        extra = "当前时间: 2026-02-28\n用户偏好: 中文回复"
        new_content = f"{original.content}\n\n{extra}" if original else extra
        new_system = SystemMessage(content=new_content)

        # 用 override() 创建新请求（原 request 不变）
        modified = request.override(system_message=new_system)

        # 把新请求传给下一层
        return handler(modified)
```

### 1.5 __setattr__ 弃用警告

直接赋值会触发 `DeprecationWarning`，强制推动不可变模式：

```python
# 已弃用 - 会发出警告
request.system_message = SystemMessage(content="新提示")
# DeprecationWarning: Use request.override(system_message=...) instead.

# 推荐方式
new_request = request.override(system_message=SystemMessage(content="新提示"))
```

`system_prompt` 作为只读属性可正常读取（返回 `system_message.text`），但赋值已弃用。

---

## 2. ModelResponse：响应封装

### 2.1 数据结构定义

```python
@dataclass
class ModelResponse(Generic[ResponseT]):
    """模型执行的响应结果"""

    result: list[BaseMessage]
    """消息列表 - 通常是单个 AIMessage，
    但如果模型用工具实现结构化输出，可能还包含一个 ToolMessage"""

    structured_response: ResponseT | None = None
    """解析后的结构化输出（仅当指定了 response_format 时才有值）"""
```

### 2.2 典型内容

**普通对话：**

```python
ModelResponse(
    result=[AIMessage(content="你好！有什么可以帮你的？")],
    structured_response=None,
)
```

**带工具调用：**

```python
ModelResponse(
    result=[AIMessage(
        content="",
        tool_calls=[{"name": "search", "args": {"query": "天气"}, "id": "call_1"}],
    )],
)
```

**带结构化输出（使用 response_format 时）：**

```python
from pydantic import BaseModel

class WeatherInfo(BaseModel):
    city: str
    temperature: float

ModelResponse(
    result=[AIMessage(content="", tool_calls=[...]), ToolMessage(content="...")],
    structured_response=WeatherInfo(city="北京", temperature=22.5),
)
```

### 2.3 在 middleware 中读取和修改响应

```python
class SafetyCheckMiddleware(AgentMiddleware):
    """检查模型回复是否包含敏感内容"""

    def wrap_model_call(self, request, handler):
        response = handler(request)
        ai_message = response.result[0]

        # 读取内容
        if self._contains_sensitive(ai_message.content):
            # 替换为安全回复（创建新的 ModelResponse）
            return ModelResponse(
                result=[AIMessage(content="抱歉，我无法回答这个问题。")],
                structured_response=None,
            )

        return response
```

注意：`ModelResponse` 没有 `override()` 方法，修改时直接创建新实例即可。

---

## 3. ExtendedModelResponse：扩展响应

### 3.1 为什么需要扩展响应？

有时 middleware 不仅想返回模型的回复，还想**附带一个 Command 来更新 Agent 状态**。比如：

- 在状态中记录模型调用次数
- 向消息列表追加额外的系统消息
- 更新 middleware 自定义的状态字段

`ExtendedModelResponse` 就是为这个场景设计的。

### 3.2 数据结构定义

```python
from langgraph.types import Command

@dataclass
class ExtendedModelResponse(Generic[ResponseT]):
    """带 Command 的扩展模型响应"""

    model_response: ModelResponse[ResponseT]
    """底层的模型响应"""

    command: Command[Any] | None = None
    """可选的 Command，用于额外的状态更新"""
```

### 3.3 Command 的 reducer 语义

关键点：Command 中的状态更新通过 **reducer** 应用，而不是直接替换。对于 `messages` 字段（使用 `add_messages` reducer），Command 中的消息会**追加**到现有消息列表：

```python
class AuditTrailMiddleware(AgentMiddleware):
    """在每次模型调用后追加审计消息"""

    def wrap_model_call(self, request, handler):
        response = handler(request)

        # 创建审计消息（通过 Command 追加，不会替换模型回复）
        audit_msg = SystemMessage(
            content=f"[审计] 模型调用完成，消息数: {len(request.messages)}"
        )

        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={"messages": [audit_msg]}),
        )
```

**状态更新顺序：** 模型回复的 AIMessage 先追加到 `state["messages"]`，然后 Command 中的消息再追加。对于非 reducer 字段，Command 的值直接覆盖。

### 3.4 注意事项

Command 的 `goto` / `resume` / `graph` 字段暂不支持，使用会抛出 `NotImplementedError`。

---

## 4. ModelCallResult：类型别名

`wrap_model_call` 的返回类型是一个联合类型，提供了三种灵活度不同的返回方式：

```python
from typing import TypeAlias

ModelCallResult: TypeAlias = (
    "ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]"
)
```

### 4.1 三种返回方式

```python
# 方式1：返回 ModelResponse（标准）
def wrap_model_call(self, request, handler):
    return handler(request)

# 方式2：返回 AIMessage（简化，框架自动包装为 ModelResponse）
def wrap_model_call(self, request, handler):
    return AIMessage(content="这是缓存的回复")

# 方式3：返回 ExtendedModelResponse（需要额外状态更新）
def wrap_model_call(self, request, handler):
    response = handler(request)
    return ExtendedModelResponse(
        model_response=response,
        command=Command(update={"call_count": 1}),
    )
```

### 4.2 框架内部的处理逻辑

```
wrap_model_call 返回值
  ├─ AIMessage → 自动包装为 ModelResponse(result=[msg])
  ├─ ModelResponse → 直接使用
  └─ ExtendedModelResponse → 拆分为 ModelResponse + Command
       ├─ ModelResponse 正常处理
       └─ Command 通过 reducer 应用到状态
```

---

## 5. 实战：完整的 wrap_model_call 示例

### 场景：模型回退 + 响应增强

```python
from langchain.agents.middleware import (
    AgentMiddleware, ModelRequest, ModelResponse, ExtendedModelResponse,
)
from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command

class SmartFallbackMiddleware(AgentMiddleware):
    """智能回退：主模型失败时切换到备用模型，并记录回退信息"""

    def __init__(self, fallback_model: str = "gpt-4o-mini"):
        self.fallback_model = ChatOpenAI(model=fallback_model)

    def wrap_model_call(self, request, handler):
        try:
            # 尝试用主模型
            return handler(request)

        except Exception as e:
            # 主模型失败，切换到备用模型
            fallback_request = request.override(model=self.fallback_model)
            response = handler(fallback_request)

            # 通过 ExtendedModelResponse 记录回退事件
            return ExtendedModelResponse(
                model_response=response,
                command=Command(update={
                    "messages": [
                        SystemMessage(content=f"[系统] 已回退到备用模型: {e}")
                    ],
                }),
            )
```

### 场景：请求字段速查

```python
def wrap_model_call(self, request: ModelRequest, handler):
    request.model              # BaseChatModel 实例
    request.messages           # 对话消息列表（不含 system）
    request.system_message     # SystemMessage | None
    request.system_prompt      # str | None（便捷访问）
    request.tools              # 可用工具列表
    request.state              # AgentState（含完整 messages）
    request.runtime.context    # 用户传入的 context
    return handler(request)
```

---

## 6. 类型关系总览

```
ModelRequest ──override()──→ 新 ModelRequest
     │
     ▼ handler(request)
     │
ModelCallResult（三选一）
  ├─ ModelResponse { result, structured_response }
  ├─ AIMessage → 自动包装为 ModelResponse
  └─ ExtendedModelResponse { model_response, command }
```

---

## 7. 常见陷阱

### 陷阱 1：直接修改 request 而非用 override()

```python
# 触发 DeprecationWarning，且可能影响其他 middleware
request.tools = [new_tool]

# 不可变更新
new_request = request.override(tools=[new_tool])
return handler(new_request)
```

### 陷阱 2：混淆 system_prompt 和 system_message

```python
# system_prompt 是只读属性（赋值会触发弃用警告）
request.system_prompt = "新提示"

# 使用 system_message
new_request = request.override(
    system_message=SystemMessage(content="新提示")
)
```

### 陷阱 3：以为 ExtendedModelResponse 的 Command 会替换消息

```python
# Command 中的 messages 是追加，不是替换！
return ExtendedModelResponse(
    model_response=response,
    command=Command(update={"messages": [extra_msg]}),
)
# 结果：state["messages"] = [...原有消息, AI回复, extra_msg]
# 而不是：state["messages"] = [extra_msg]
```

### 陷阱 4：忘记同时实现同步和异步版本

如果只实现 `awrap_model_call`，用 `agent.invoke()`（同步）会抛出 `NotImplementedError`。解决：同时实现 `wrap_model_call`。

### 陷阱 5：在 wrap_model_call 中忘记调用 handler

如果不调用 `handler`，模型就不会被调用。这在缓存场景下是有意为之，但在其他场景下是 bug。

---

## 总结

| 类型 | 角色 | 关键字段 | 修改方式 |
|------|------|----------|----------|
| `ModelRequest` | 请求封装 | model, messages, system_message, tools, state, runtime | `override()` 不可变更新 |
| `ModelResponse` | 响应封装 | result (消息列表), structured_response | 创建新实例 |
| `ExtendedModelResponse` | 扩展响应 | model_response + command | 创建新实例 |
| `ModelCallResult` | 类型别名 | ModelResponse \| AIMessage \| ExtendedModelResponse | — |

核心原则：**ModelRequest 用 override() 做不可变更新，ModelResponse 通过创建新实例来修改，ExtendedModelResponse 用 Command + reducer 追加状态。**

---

[来源: `sourcecode/langchain/libs/langchain_v1/langchain/agents/middleware/types.py`]

**上一篇**: [03_核心概念_3_Middleware基类与钩子系统.md](./03_核心概念_3_Middleware基类与钩子系统.md)
**下一篇**: [03_核心概念_5_内置Middleware全景.md](./03_核心概念_5_内置Middleware全景.md)
