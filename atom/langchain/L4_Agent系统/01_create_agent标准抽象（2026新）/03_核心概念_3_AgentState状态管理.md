# 核心概念 3：AgentState 状态管理

> Agent 的一切行为都围绕状态流转。AgentState 定义了"Agent 知道什么"、"这些信息如何在节点间传递"、以及"谁能看到什么"。

---

## 为什么状态管理是 Agent 的核心？

Agent 在执行过程中会经历多个阶段：接收用户输入、调用模型、执行工具、再调用模型……每个阶段都需要读取上一步的结果，并把自己的产出传递给下一步。这些"在阶段之间流动的数据"就是状态。

没有状态管理，Agent 就像一个失忆的人——每一步都不知道前一步发生了什么。

LangChain 1.0 的 `create_agent` 用 `AgentState` 这个 TypedDict 来统一管理所有状态数据。它不是一个普通的字典，而是一个带类型约束、带注解系统、带 reducer 语义的结构化容器。

---

## AgentState 源码定义

```python
# 源码位置: langchain/agents/middleware/types.py

JumpTo = Literal["tools", "model", "end"]

class AgentState(TypedDict, Generic[ResponseT]):
    """Agent 的状态 schema。"""
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]
```

三个字段，各司其职：

| 字段 | 类型 | 必需？ | 作用 | 特殊注解 |
|------|------|--------|------|----------|
| `messages` | `list[AnyMessage]` | 是 | 对话消息列表（Agent 的"记忆"） | `add_messages` reducer |
| `jump_to` | `JumpTo \| None` | 否 | 控制执行流程跳转 | `EphemeralValue` + `PrivateStateAttr` |
| `structured_response` | `ResponseT` | 否 | 结构化输出结果 | `OmitFromInput` |

下面逐一深入。

---

## 字段一：messages —— Agent 的记忆

`messages` 是唯一的必需字段，存储所有对话消息（HumanMessage、AIMessage、ToolMessage、SystemMessage）。

关键在于 `add_messages` reducer。在 LangGraph 中，reducer 决定了"新值如何与旧值合并"：

**规则 1：追加，不是替换**

```python
def model_node(state):
    response = model.invoke(state["messages"])
    return {"messages": [response]}  # 追加到末尾，不是替换整个列表！

# 状态演变：
# 初始:  messages = [HumanMessage("你好")]
# 模型:  messages = [HumanMessage("你好"), AIMessage("你好！")]
# 用户:  messages = [HumanMessage("你好"), AIMessage("你好！"), HumanMessage("天气如何")]
```

**规则 2：相同 ID 的消息会被更新**

```python
# 如果新消息的 id 与已有消息相同，会替换而非追加
existing = AIMessage(content="旧回复", id="msg-001")
updated = AIMessage(content="新回复", id="msg-001")
# add_messages([existing], [updated]) → [AIMessage(content="新回复", id="msg-001")]
```

**规则 3：支持 RemoveMessage 删除消息**

```python
from langchain_core.messages import RemoveMessage

# 不能直接截断 messages（reducer 会追加而非替换），正确做法：
def trim_messages(state):
    old_messages = state["messages"][:-5]  # 只保留最后 5 条
    return {"messages": [RemoveMessage(id=m.id) for m in old_messages]}
```

为什么用 Reducer？这是 LangGraph 的核心设计：**状态更新是声明式的**。每个节点只声明"我产生了什么新数据"，Reducer 负责正确合并。这让多个节点可以并行更新同一个字段而不冲突。

---

## 字段二：jump_to —— 流程跳转控制

`jump_to` 是 Agent 执行流程的"方向盘"，让 Middleware 在运行时动态决定下一步走向：

| 值 | 含义 | 场景 |
|----|------|------|
| `"tools"` | 跳转到工具执行节点 | 强制执行工具 |
| `"model"` | 跳转回模型调用节点 | 让模型重新思考 |
| `"end"` | 直接结束 Agent 循环 | 提前终止（如安全检查不通过） |
| `None` | 使用默认路由 | 有 tool_calls → tools，否则 → end |

### 在 Middleware 中使用

```python
class SafetyGuardMiddleware(AgentMiddleware):
    def after_model(self, state, runtime) -> dict | None:
        last_msg = state["messages"][-1]
        if self._contains_sensitive_content(last_msg.content):
            return {"jump_to": "end"}      # 强制结束
        return None

class QualityCheckMiddleware(AgentMiddleware):
    def after_model(self, state, runtime) -> dict | None:
        last_msg = state["messages"][-1]
        if len(last_msg.content) < 10 and not last_msg.tool_calls:
            return {
                "messages": [SystemMessage(content="请提供更详细的回答")],
                "jump_to": "model",        # 重新生成
            }
        return None
```

### can_jump_to 权限控制

需要通过 `@hook_config(can_jump_to=[...])` 声明跳转权限：

```python
from langchain.agents.middleware import hook_config

class StrictMiddleware(AgentMiddleware):
    @hook_config(can_jump_to=["end"])  # 只允许跳转到 end
    def before_model(self, state, runtime) -> dict | None:
        if self._should_block(state):
            return {"jump_to": "end"}  # 允许
            # return {"jump_to": "model"}  # 会报错！未授权
        return None
```

### EphemeralValue：阅后即焚

`jump_to` 标注了 `EphemeralValue`：每个超级步开始时自动重置为 `None`，不会被 checkpointer 持久化，设置一次用完就清空。

```python
# ❌ 错误理解：jump_to 会保持上一轮的值
def my_hook(state, runtime):
    print(state.get("jump_to"))  # 永远是 None！上一轮的值已被清空
```

---

## 字段三：structured_response —— 结构化输出

当配置了 `response_format` 时，模型的结构化输出存储在此字段：

```python
from pydantic import BaseModel

class WeatherReport(BaseModel):
    city: str
    temperature: float
    summary: str

agent = create_agent(model="openai:gpt-4o", response_format=WeatherReport)
result = agent.invoke({"messages": [{"role": "user", "content": "北京天气"}]})

report = result["structured_response"]  # WeatherReport 实例
print(report.city)  # "北京"
```

标注了 `OmitFromInput`：不出现在输入 schema 中（你不需要传入它），只在输出中可读。

---

## OmitFromSchema 注解系统

AgentState 用一套注解控制字段的可见性，核心是 `OmitFromSchema` 类：

```python
@dataclass
class OmitFromSchema:
    input: bool = True    # 是否从输入 schema 中排除
    output: bool = True   # 是否从输出 schema 中排除

# 四个预定义实例
OmitFromInput = OmitFromSchema(input=True, output=False)     # 排除输入，保留输出
OmitFromOutput = OmitFromSchema(input=False, output=True)    # 保留输入，排除输出
PrivateStateAttr = OmitFromSchema(input=True, output=True)   # 完全私有
```

### 可见性矩阵

```
                    输入可见?    输出可见?    典型用途
─────────────────────────────────────────────────────────────
无注解               ✅          ✅         普通字段（如 messages）
OmitFromInput        ❌          ✅         只读输出（如 structured_response）
OmitFromOutput       ✅          ❌         只写输入（如配置参数）
PrivateStateAttr     ❌          ❌         纯内部字段（如 jump_to、计数器）
EphemeralValue       -           -          每轮重置（与上面正交，可组合使用）
```

`EphemeralValue` 控制**生命周期**（每轮重置），`OmitFromSchema` 控制**可见性**（输入/输出 schema），两者正交。`jump_to` 就同时标注了 `EphemeralValue` 和 `PrivateStateAttr`。

---

## 输入/输出 Schema 分离

create_agent 内部为输入和输出分别定义了精简的 schema：

```python
# 输入 Schema：只需要 messages，且接受 dict 格式（宽松）
class _InputAgentState(TypedDict):
    messages: Required[Annotated[list[AnyMessage | dict[str, Any]], add_messages]]

# 输出 Schema：messages + structured_response（不含私有字段）
class _OutputAgentState(TypedDict, Generic[ResponseT]):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    structured_response: NotRequired[ResponseT]

# 内部 Schema：完整的 AgentState（包含所有字段）
# → 就是上面定义的 AgentState
```

这种分离带来三个好处：

1. **输入简洁**：调用者只需传 `messages`，不需要关心内部状态字段
2. **输出清晰**：返回结果只包含有意义的数据，私有字段被过滤
3. **类型安全**：输入接受 `dict` 格式（便利），输出保证是 `AnyMessage` 类型（严格）

```python
# 输入：dict 格式自动转换
result = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})

# 输出：结构化的 _OutputAgentState
print(result["messages"])                    # list[AnyMessage]
print(result.get("structured_response"))     # ResponseT 或 None
# result["jump_to"]                          # KeyError！私有字段不在输出中
```

---

## 自定义状态扩展

### 方式一：通过 state_schema 参数

```python
class MyAgentState(AgentState):
    user_preferences: NotRequired[dict[str, str]]
    conversation_topic: NotRequired[str]

agent = create_agent(
    model="openai:gpt-4o",
    state_schema=MyAgentState,
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "推荐一本书"}],
    "user_preferences": {"genre": "科幻"},
})
```

### 方式二：通过 Middleware 的 state_schema（推荐）

```python
class TokenCountState(AgentState):
    token_count: NotRequired[Annotated[int, PrivateStateAttr]]

class TokenCounterMiddleware(AgentMiddleware[TokenCountState]):
    state_schema = TokenCountState

    def after_model(self, state, runtime) -> dict | None:
        current = state.get("token_count", 0)
        new_count = current + len(str(state["messages"][-1].content)) // 4
        return {"token_count": new_count}

agent = create_agent(
    model="openai:gpt-4o",
    middleware=[TokenCounterMiddleware()],
)
# token_count 自动合并到内部状态，因为 PrivateStateAttr 外部看不到
```

### 为什么推荐方式二？

| 维度 | 方式一（state_schema 参数） | 方式二（Middleware state_schema） |
|------|---------------------------|--------------------------------|
| 作用域 | 全局，所有 Middleware 可见 | 局部，与 Middleware 绑定 |
| 可组合性 | 需手动合并 | create_agent 自动合并 |
| 封装性 | 字段暴露给所有人 | 配合 PrivateStateAttr 完全私有 |
| 可复用性 | 与特定 Agent 绑定 | Middleware 可跨 Agent 复用 |

---

## 状态合并机制：_resolve_schema()

当多个 Middleware 各自定义了 `state_schema` 时，create_agent 通过 `_resolve_schema()` 自动合并：

```python
# 源码位置: langchain/agents/factory.py
def _resolve_schema(schemas: set[type], schema_name: str, omit_flag: str | None = None) -> type:
    all_annotations = {}
    for schema in schemas:
        hints = get_type_hints(schema, include_extras=True)
        for field_name, field_type in hints.items():
            should_omit = False
            if omit_flag:
                metadata = _extract_metadata(field_type)
                for meta in metadata:
                    if isinstance(meta, OmitFromSchema) and getattr(meta, omit_flag) is True:
                        should_omit = True
                        break
            if not should_omit:
                all_annotations[field_name] = field_type
    return TypedDict(schema_name, all_annotations)
```

合并过程示例：假设 Middleware A 扩展了 `call_count: int`，Middleware B 扩展了 `user_score: float`：

```
state_schemas = {AgentState, StateA, StateB}

→ resolved_state_schema（内部）: messages + jump_to + structured_response + call_count + user_score
→ input_schema:  只有 messages（其余都被 OmitFromInput/PrivateStateAttr 过滤）
→ output_schema: messages + structured_response（PrivateStateAttr 字段被过滤）

→ 传给 StateGraph(state_schema=..., input=..., output=...)
```

注意：如果多个 schema 定义了同名字段且类型不同，后者覆盖前者，应避免这种情况。

---

## 状态流转全景图

一次完整的 Agent 调用中，状态如何流转：

```
agent.invoke({"messages": [HumanMessage("...")]})
│
▼ InputSchema 验证（只检查 messages）
▼ 状态初始化: messages=[HumanMessage], jump_to=None, structured_response=未设置
│
▼ before_agent 钩子 → 可修改状态（只执行一次）
│
▼ ┌─── Agent 循环 ─────────────────────────────────┐
│ │                                                 │
│ ▼ before_model 钩子 → 可修改状态、可设置 jump_to   │
│ ▼ model_node: 调用 LLM → messages += [AIMessage]  │
│ ▼ after_model 钩子 → 可修改状态、可设置 jump_to    │
│ │                                                 │
│ ▼ 路由决策:                                        │
│ ├─ jump_to="tools" → tool_node → 回到 before_model │
│ ├─ jump_to="model" → 回到 before_model             │
│ ├─ jump_to="end"   → 退出循环                      │
│ └─ None（默认）→ 有 tool_calls? tools : end        │
│                                                    │
└────────────────────────────────────────────────────┘
│
▼ after_agent 钩子 → 最终状态修改（只执行一次）
▼ OutputSchema 过滤（移除 PrivateStateAttr 字段）
▼ 返回 result["messages"] + result.get("structured_response")
```

---

## 与 LangGraph State 的关系

AgentState 就是 LangGraph `StateGraph` 的 `state_schema`，不是独立的状态管理系统：

```python
# create_agent 内部（简化版）
def create_agent(model, tools, middleware, state_schema, ...):
    resolved = _resolve_schema(all_schemas, "StateSchema", None)
    graph = StateGraph(state_schema=resolved, input=input_schema, output=output_schema)
    graph.add_node("model", model_node)
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("model", route_after_model)
    return graph.compile(checkpointer=checkpointer)
```

这意味着所有 LangGraph 特性开箱即用：Reducer、EphemeralValue、Checkpointer 持久化、Time Travel 等。

```python
from langgraph.checkpoint.sqlite import SqliteSaver

agent = create_agent(
    model="openai:gpt-4o",
    checkpointer=SqliteSaver.from_conn_string("agent.db"),
)

config = {"configurable": {"thread_id": "user-123"}}
agent.invoke({"messages": [{"role": "user", "content": "我叫小明"}]}, config=config)
# 第二次对话，状态自动恢复
agent.invoke({"messages": [{"role": "user", "content": "我叫什么？"}]}, config=config)
# Agent 能记住"小明"，因为 messages 被持久化了
```

---

## 常见陷阱

### 陷阱 1：直接替换 messages 列表

```python
# ❌ add_messages reducer 会追加而非替换
def bad_trim(state):
    return {"messages": state["messages"][-3:]}  # 不会生效！

# ✅ 使用 RemoveMessage
def correct_trim(state):
    to_remove = state["messages"][:-3]
    return {"messages": [RemoveMessage(id=m.id) for m in to_remove]}
```

### 陷阱 2：忘记 Middleware 状态字段需要 NotRequired

```python
# ❌ Required 字段必须在输入中提供
class BadState(AgentState):
    counter: int  # 默认是 Required！

# ✅ Middleware 扩展字段应该是 NotRequired
class GoodState(AgentState):
    counter: NotRequired[Annotated[int, PrivateStateAttr]]
```

### 陷阱 3：多个 Middleware 定义同名字段

```python
# ⚠️ 两个 Middleware 都定义了 counter，类型不同 → 合并后类型不确定
# 应该使用带前缀的命名：token_counter_total、retry_counter_attempts
```

---

## 最佳实践

1. **优先用 Middleware 的 state_schema**：状态与 Middleware 绑定，封装性好，可复用
2. **内部状态用 PrivateStateAttr**：不暴露给外部调用者
3. **理解 add_messages 追加语义**：返回新消息让 reducer 追加，不要试图替换
4. **善用 jump_to 控制流程**：配合 `@hook_config(can_jump_to=[...])` 声明权限
5. **字段命名保持唯一**：不同 Middleware 用有区分度的前缀避免冲突

---

## 小结

AgentState 的设计哲学是"约定优于配置"：

- 三个核心字段（messages、jump_to、structured_response）覆盖 90% 场景
- 注解系统（OmitFromInput、PrivateStateAttr、EphemeralValue）让字段行为声明式可控
- `_resolve_schema()` 自动合并多个 Middleware 的状态，开发者无需手动协调
- 输入/输出 schema 分离，对外接口简洁，内部状态丰富
- 底层就是 LangGraph StateGraph，所有 LangGraph 特性开箱即用

一句话：**AgentState 是 Agent 的"记忆结构"，注解系统是"访问控制"，合并机制是"自动装配"。**

---

**上一篇**: [03_核心概念_2_AgentState状态管理.md](./03_核心概念_2_AgentState状态管理.md)
**下一篇**: [03_核心概念_4_ModelRequest与ModelResponse数据流.md](./03_核心概念_4_ModelRequest与ModelResponse数据流.md)

---

> [来源: sourcecode/langchain/libs/langchain_v1/langchain/agents/middleware/types.py]
> [来源: sourcecode/langchain/libs/langchain_v1/langchain/agents/factory.py]
