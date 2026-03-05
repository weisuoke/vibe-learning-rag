# Runtime Context 机制

> LangGraph 核心概念：理解运行时上下文的不可变配置传递

---

## 1. 【30字核心】

**Runtime Context 是 LangGraph 中传递不可变运行时配置的机制,通过 context_schema 定义类型,在节点中通过 Runtime[Context] 访问,实现配置与状态的分离。**

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理,从源头思考问题。

### Runtime Context 的第一性原理

#### 1. 最基础的定义

**Runtime Context = 一个在运行时不可修改的配置对象**

仅此而已！它就是一个普通的 Python 对象(通常是 dataclass 或 TypedDict),但它在运行时是不可变的。

#### 2. 为什么需要 Runtime Context？

**核心问题：如何在不污染状态的情况下,传递运行时配置？**

传统方式的问题：
```python
# ❌ 传统方式：配置和状态混在一起
class State(TypedDict):
    messages: list[str]
    user_id: str          # 这是配置,不是状态
    model_name: str       # 这是配置,不是状态
    temperature: float    # 这是配置,不是状态
```

Runtime Context 的解决方案：
```python
# ✅ Runtime Context 方式：配置和状态分离
class State(TypedDict):
    messages: list[str]   # 这是状态

@dataclass
class Context:
    user_id: str          # 这是配置
    model_name: str       # 这是配置
    temperature: float    # 这是配置
```

#### 3. Runtime Context 的三层价值

##### 价值1：配置与状态分离（Separation of Concerns）

**清晰区分可变状态与不可变配置**

State 是可变的,会在节点间传递和更新。Context 是不可变的,只在运行时传递。

**示例：**
```python
# State：可变,会被节点更新
state["messages"].append(new_message)

# Context：不可变,只读取
user_id = runtime.context.user_id  # 只读
```

##### 价值2：类型安全（Type Safety）

**通过 context_schema 定义类型,编译时检查**

使用 TypedDict 或 dataclass 定义 context_schema,IDE 可以提供类型提示和检查。

**示例：**
```python
@dataclass
class Context:
    user_id: str
    model_name: str

# IDE 会提示类型错误
def node(state: State, runtime: Runtime[Context]):
    user_id: int = runtime.context.user_id  # 类型错误！
```

##### 价值3：运行时动态配置（Runtime Configuration）

**无需重新编译图,即可改变行为**

通过传递不同的 context,可以在运行时改变图的行为,无需重新编译。

**示例：**
```python
# 使用 GPT-4
result1 = graph.invoke(
    {"messages": ["hello"]},
    context=Context(user_id="user_1", model_name="gpt-4")
)

# 使用 Claude
result2 = graph.invoke(
    {"messages": ["hello"]},
    context=Context(user_id="user_1", model_name="claude-3")
)
```

#### 4. 从第一性原理推导上下文工程

**推理链：**
```
1. 节点需要访问配置(如 user_id、model_name)
   ↓
2. 配置不应该放在 State 中(State 是可变的)
   ↓
3. 需要一个不可变的配置传递机制
   ↓
4. 使用 Runtime Context 传递配置
   ↓
5. 通过 context_schema 定义类型
   ↓
6. 节点通过 Runtime[Context] 访问
   ↓
7. 实现配置与状态的分离
```

#### 5. 一句话总结第一性原理

**Runtime Context 是不可变配置的载体,通过类型化定义实现配置与状态的分离,支持运行时动态配置。**

---

## 3. 【核心概念】

### 核心概念1：context_schema 定义

**context_schema 是定义 Runtime Context 类型的机制。**

```python
from dataclasses import dataclass
from typing_extensions import TypedDict
from langgraph.graph import StateGraph

# 方式1：使用 dataclass
@dataclass
class Context:
    user_id: str
    model_name: str = "gpt-3.5-turbo"  # 默认值

# 方式2：使用 TypedDict
class Context(TypedDict):
    user_id: str
    model_name: str

# 创建图时指定 context_schema
builder = StateGraph(State, context_schema=Context)
```

**context_schema 的作用：**

1. **类型定义**
   - 定义 context 的字段和类型
   - IDE 提供类型提示和检查
   - 运行时类型验证

2. **默认值支持**
   - dataclass 支持默认值
   - TypedDict 不支持默认值

3. **文档化**
   - 清晰表达图需要哪些配置
   - 便于团队协作

**示例：**
```python
@dataclass
class Context:
    """运行时上下文配置"""
    user_id: str                    # 用户 ID
    model_name: str = "gpt-4"       # LLM 模型名称
    temperature: float = 0.7        # 温度参数
    max_tokens: int = 1000          # 最大 token 数
```

**在状态化工作流中：**
- context_schema 在图创建时指定
- 所有节点共享同一个 context_schema
- 节点通过 Runtime[Context] 访问

---

### 核心概念2：Runtime[Context] 访问

**Runtime[Context] 是在节点中访问 Runtime Context 的接口。**

```python
from langgraph.runtime import Runtime

def my_node(state: State, runtime: Runtime[Context]):
    # 访问 context
    user_id = runtime.context.user_id
    model_name = runtime.context.model_name
    
    # 使用 context
    print(f"User: {user_id}, Model: {model_name}")
    
    return state
```

**Runtime 对象的结构：**
```python
class Runtime[Context]:
    context: Context              # 运行时上下文
    store: BaseStore             # 长期存储
    # 其他运行时信息
```

**访问方式：**

1. **直接访问字段**
   ```python
   user_id = runtime.context.user_id
   ```

2. **使用 get() 方法（TypedDict）**
   ```python
   user_id = runtime.context.get("user_id", "default")
   ```

3. **解构访问**
   ```python
   context = runtime.context
   user_id = context.user_id
   model_name = context.model_name
   ```

**类型安全：**
```python
# ✅ 正确：类型匹配
user_id: str = runtime.context.user_id

# ❌ 错误：类型不匹配
user_id: int = runtime.context.user_id  # IDE 会提示错误
```

---

### 核心概念3：context 参数传递

**context 参数在调用图时传递,支持字典和对象两种方式。**

```python
# 方式1：传递对象
context = Context(user_id="user_123", model_name="gpt-4")
result = graph.invoke(
    {"messages": ["hello"]},
    context=context
)

# 方式2：传递字典
result = graph.invoke(
    {"messages": ["hello"]},
    context={"user_id": "user_123", "model_name": "gpt-4"}
)
```

**传递时机：**
- 在 `graph.invoke()` 时传递
- 在 `graph.stream()` 时传递
- 在 `graph.ainvoke()` 时传递（异步）

**传递范围：**
- context 在整个图执行期间有效
- 所有节点都可以访问同一个 context
- context 在调用结束后销毁

**示例：**
```python
# 不同的调用可以传递不同的 context
for user_id in ["user_1", "user_2", "user_3"]:
    result = graph.invoke(
        {"messages": ["hello"]},
        context=Context(user_id=user_id, model_name="gpt-4")
    )
    print(f"User {user_id}: {result}")
```

---

### 核心概念4：与 State 的区别

**Runtime Context 和 State 是两个不同的概念,有明确的区分。**

| 特性 | State | Runtime Context |
|------|-------|-----------------|
| **可变性** | 可变（节点可以修改） | 不可变（只读） |
| **生命周期** | 持久化（可以保存到 Checkpoint） | 临时（调用结束后销毁） |
| **用途** | 存储工作流的状态数据 | 传递运行时配置 |
| **示例** | messages、search_results | user_id、model_name |
| **访问方式** | 直接访问 `state["key"]` | 通过 `runtime.context.key` |
| **类型定义** | StateGraph(State) | StateGraph(State, context_schema=Context) |

**何时使用 State：**
- 需要在节点间传递和更新的数据
- 需要持久化的数据（如对话历史）
- 工作流的中间结果

**何时使用 Runtime Context：**
- 运行时配置（如 user_id、model_name）
- 不需要修改的数据（如数据库连接）
- 需要在运行时动态改变的配置

**示例：**
```python
# State：可变,持久化
class State(TypedDict):
    messages: list[str]           # 对话历史
    search_results: list[dict]    # 搜索结果

# Context：不可变,临时
@dataclass
class Context:
    user_id: str                  # 用户 ID
    model_name: str               # 模型名称
    db_connection: Any            # 数据库连接
```

---

## 4. 【最小可用】

掌握以下内容,就能开始使用 Runtime Context：

### 4.1 定义 context_schema

```python
from dataclasses import dataclass
from langgraph.graph import StateGraph

@dataclass
class Context:
    user_id: str
    model_name: str = "gpt-3.5-turbo"

# 创建图时指定 context_schema
builder = StateGraph(State, context_schema=Context)
```

### 4.2 在节点中访问 context

```python
from langgraph.runtime import Runtime

def my_node(state: State, runtime: Runtime[Context]):
    # 访问 context
    user_id = runtime.context.user_id
    model_name = runtime.context.model_name
    
    print(f"User: {user_id}, Model: {model_name}")
    
    return state
```

### 4.3 在调用时传递 context

```python
# 传递 context
result = graph.invoke(
    {"messages": ["hello"]},
    context=Context(user_id="user_123", model_name="gpt-4")
)
```

### 4.4 使用 context 实现动态配置

```python
# 动态选择 LLM
def call_llm(state: State, runtime: Runtime[Context]):
    model_name = runtime.context.model_name
    
    if model_name == "gpt-4":
        llm = ChatOpenAI(model="gpt-4")
    elif model_name == "claude-3":
        llm = ChatAnthropic(model="claude-3-opus-20240229")
    else:
        llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

**这些知识足以：**
- 定义和使用 Runtime Context
- 实现配置与状态的分离
- 支持运行时动态配置

---

## 5. 【双重类比】

### 类比1：Runtime Context = 环境变量

**前端类比：** process.env（环境变量）

在 Node.js 中,环境变量用于传递配置,不会被修改。

```javascript
// Node.js
const userId = process.env.USER_ID;
const apiKey = process.env.API_KEY;
// 环境变量是只读的
```

**日常生活类比：** 身份证

身份证包含了你的基本信息(姓名、身份证号),这些信息是不可变的。

```
身份证（Runtime Context）
├── 姓名（user_id）
├── 身份证号（model_name）
└── 出生日期（temperature）
```

**Python 示例：**
```python
# Runtime Context 就像环境变量
@dataclass
class Context:
    user_id: str          # 像 USER_ID 环境变量
    api_key: str          # 像 API_KEY 环境变量
    model_name: str       # 像 MODEL_NAME 环境变量
```

---

### 类比2：context_schema = TypeScript 接口

**前端类比：** TypeScript 接口定义

在 TypeScript 中,接口定义了对象的类型。

```typescript
// TypeScript
interface Config {
    userId: string;
    modelName: string;
    temperature: number;
}

function processData(config: Config) {
    // config 的类型是明确的
}
```

**日常生活类比：** 合同条款

合同条款定义了双方的权利和义务,是明确的约定。

```
合同条款（context_schema）
├── 甲方（user_id: str）
├── 乙方（model_name: str）
└── 金额（temperature: float）
```

**Python 示例：**
```python
# context_schema 就像 TypeScript 接口
@dataclass
class Context:
    user_id: str          # 类型明确
    model_name: str       # 类型明确
    temperature: float    # 类型明确
```

---

### 类比3：Runtime[Context] = 依赖注入

**前端类比：** React useContext

在 React 中,useContext 用于访问上下文。

```javascript
// React
const config = useContext(ConfigContext);
const userId = config.userId;
```

**日常生活类比：** 工具箱

工具箱里装着各种工具,需要时随时取用。

```
工具箱（Runtime）
├── 扳手（context.user_id）
├── 螺丝刀（context.model_name）
└── 锤子（context.temperature）
```

**Python 示例：**
```python
# Runtime[Context] 就像依赖注入
def my_node(state: State, runtime: Runtime[Context]):
    # 从 runtime 中获取 context
    user_id = runtime.context.user_id
    model_name = runtime.context.model_name
```

---

### 类比总结表

| LangGraph 概念 | 前端类比 | 日常生活类比 |
|----------------|----------|--------------|
| Runtime Context | process.env | 身份证 |
| context_schema | TypeScript 接口 | 合同条款 |
| Runtime[Context] | useContext | 工具箱 |
| context 参数 | 环境变量传递 | 出示身份证 |
| 不可变性 | const 常量 | 身份证信息 |
| 类型安全 | TypeScript 类型检查 | 合同条款约束 |


---

## 6. 【反直觉点】

### 误区1：Runtime Context 可以修改 ❌

**为什么错？**
- Runtime Context 在运行时是**不可变的**（immutable）
- 节点不能修改 context，只能读取
- 修改 context 不会影响其他节点或后续调用

**为什么人们容易这样错？**
因为 context 是一个普通的 Python 对象（dataclass 或 TypedDict），看起来可以修改。但实际上，LangGraph 的设计理念是 context 应该是不可变的。

**正确理解：**
```python
# ❌ 错误做法：修改 context
def my_node(state: State, runtime: Runtime[Context]):
    runtime.context.user_id = "new_user"  # 不要这样做！
    return state

# ✅ 正确做法：只读取 context
def my_node(state: State, runtime: Runtime[Context]):
    user_id = runtime.context.user_id  # 只读取
    print(f"User ID: {user_id}")
    return state

# ✅ 如果需要修改，应该放在 State 中
def my_node(state: State, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    return {"current_user": user_id}  # 放在 State 中
```

---

### 误区2：Runtime Context 和 State 是一样的 ❌

**为什么错？**
- Runtime Context 是**不可变的配置**，State 是**可变的状态**
- Runtime Context 是**临时的**（调用结束后销毁），State 是**持久化的**（可以保存到 Checkpoint）
- Runtime Context 用于**配置**，State 用于**数据**

**为什么人们容易这样错？**
因为两者都可以在节点间传递信息，看起来功能相似。但实际上，它们的用途和生命周期完全不同。

**正确理解：**
```python
# Runtime Context：不可变配置
@dataclass
class Context:
    user_id: str          # 配置：用户 ID
    model_name: str       # 配置：模型名称

# State：可变状态
class State(TypedDict):
    messages: list[str]   # 状态：对话历史
    search_results: list  # 状态：搜索结果

# 节点中的使用
def my_node(state: State, runtime: Runtime[Context]):
    # 读取配置（不可变）
    user_id = runtime.context.user_id
    
    # 修改状态（可变）
    return {"messages": state["messages"] + ["new message"]}
```

---

### 误区3：所有配置都应该放在 Runtime Context 中 ❌

**为什么错？**
- 只有**运行时动态配置**才应该放在 Runtime Context 中
- **编译时配置**应该在创建图时指定
- **状态相关的数据**应该放在 State 中

**为什么人们容易这样错？**
因为 Runtime Context 看起来很方便，可以传递任何配置。但实际上，过度使用 Runtime Context 会导致代码难以理解和维护。

**正确理解：**
```python
# ✅ 适合放在 Runtime Context 中：
@dataclass
class Context:
    user_id: str          # 运行时动态：每次调用不同
    model_name: str       # 运行时动态：可以在运行时切换

# ✅ 适合放在编译时配置中：
builder = StateGraph(State, context_schema=Context)
builder.add_node("node1", node1)  # 图结构是编译时确定的

# ✅ 适合放在 State 中：
class State(TypedDict):
    messages: list[str]   # 状态数据：会在节点间传递和更新
    user_preferences: dict  # 状态数据：需要持久化

# ❌ 不适合放在 Runtime Context 中：
@dataclass
class Context:
    messages: list[str]   # 错误！这是状态，不是配置
    graph_structure: dict  # 错误！这是编译时配置
```

---

## 7. 【实战代码】

```python
"""
Runtime Context 机制实战示例
演示：如何使用 Runtime Context 实现动态 LLM 选择和用户级别配置
"""

import os
from dataclasses import dataclass
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import Runtime

# ===== 1. 定义 Runtime Context =====
@dataclass
class Context:
    """运行时上下文配置"""
    user_id: str                    # 用户 ID
    model_provider: str = "openai"  # LLM 提供商
    model_name: str = "gpt-3.5-turbo"  # 模型名称
    temperature: float = 0.7        # 温度参数
    max_tokens: int = 1000          # 最大 token 数

# ===== 2. 定义 State =====
class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_info: dict

# ===== 3. 定义节点（使用 Runtime Context） =====
def call_llm(state: State, runtime: Runtime[Context]) -> State:
    """根据 Runtime Context 动态选择 LLM"""
    
    # 3.1 从 context 获取配置
    user_id = runtime.context.user_id
    model_provider = runtime.context.model_provider
    model_name = runtime.context.model_name
    temperature = runtime.context.temperature
    max_tokens = runtime.context.max_tokens
    
    print(f"[LLM] User: {user_id}, Provider: {model_provider}, Model: {model_name}")
    
    # 3.2 根据 provider 选择 LLM
    if model_provider == "openai":
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif model_provider == "anthropic":
        llm = ChatAnthropic(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(f"Unknown provider: {model_provider}")
    
    # 3.3 调用 LLM
    response = llm.invoke(state["messages"])
    
    return {"messages": [response]}

def log_user_activity(state: State, runtime: Runtime[Context]) -> State:
    """记录用户活动（使用 context 中的 user_id）"""
    
    user_id = runtime.context.user_id
    message_count = len(state["messages"])
    
    print(f"[Log] User {user_id} has {message_count} messages")
    
    # 更新用户信息（放在 State 中）
    return {
        "user_info": {
            "user_id": user_id,
            "message_count": message_count,
            "last_model": runtime.context.model_name,
        }
    }

# ===== 4. 创建图 =====
builder = StateGraph(State, context_schema=Context)
builder.add_node("call_llm", call_llm)
builder.add_node("log_activity", log_user_activity)
builder.add_edge(START, "call_llm")
builder.add_edge("call_llm", "log_activity")
builder.add_edge("log_activity", END)

# 编译图
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# ===== 5. 运行图（传递不同的 context） =====
print("=== 场景1：使用 OpenAI GPT-4 ===")
result1 = graph.invoke(
    {"messages": [HumanMessage(content="Hello, how are you?")]},
    config={"configurable": {"thread_id": "thread_1"}},
    context=Context(
        user_id="alice",
        model_provider="openai",
        model_name="gpt-4",
        temperature=0.7,
    )
)
print(f"Result 1: {result1['user_info']}\n")

print("=== 场景2：使用 Anthropic Claude ===")
result2 = graph.invoke(
    {"messages": [HumanMessage(content="What's the weather like?")]},
    config={"configurable": {"thread_id": "thread_2"}},
    context=Context(
        user_id="bob",
        model_provider="anthropic",
        model_name="claude-3-opus-20240229",
        temperature=0.5,
    )
)
print(f"Result 2: {result2['user_info']}\n")

print("=== 场景3：使用默认配置 ===")
result3 = graph.invoke(
    {"messages": [HumanMessage(content="Tell me a joke")]},
    config={"configurable": {"thread_id": "thread_3"}},
    context=Context(user_id="charlie")  # 使用默认 model_provider 和 model_name
)
print(f"Result 3: {result3['user_info']}\n")

# ===== 6. 演示 Context 与 State 的区别 =====
print("=== 演示 Context 与 State 的区别 ===")

# Context：不可变，每次调用可以不同
for i in range(3):
    result = graph.invoke(
        {"messages": [HumanMessage(content=f"Message {i}")]},
        config={"configurable": {"thread_id": f"thread_{i}"}},
        context=Context(
            user_id=f"user_{i}",
            model_name="gpt-3.5-turbo" if i % 2 == 0 else "gpt-4",
        )
    )
    print(f"Call {i}: User={result['user_info']['user_id']}, Model={result['user_info']['last_model']}")

# State：可变，持久化
print("\n=== 演示 State 的持久化 ===")
thread_id = "persistent_thread"

# 第一次调用
result1 = graph.invoke(
    {"messages": [HumanMessage(content="First message")]},
    config={"configurable": {"thread_id": thread_id}},
    context=Context(user_id="persistent_user")
)
print(f"First call: {result1['user_info']['message_count']} messages")

# 第二次调用（State 会累积）
result2 = graph.invoke(
    {"messages": [HumanMessage(content="Second message")]},
    config={"configurable": {"thread_id": thread_id}},
    context=Context(user_id="persistent_user")
)
print(f"Second call: {result2['user_info']['message_count']} messages")
```

**运行输出示例：**
```
=== 场景1：使用 OpenAI GPT-4 ===
[LLM] User: alice, Provider: openai, Model: gpt-4
[Log] User alice has 2 messages
Result 1: {'user_id': 'alice', 'message_count': 2, 'last_model': 'gpt-4'}

=== 场景2：使用 Anthropic Claude ===
[LLM] User: bob, Provider: anthropic, Model: claude-3-opus-20240229
[Log] User bob has 2 messages
Result 2: {'user_id': 'bob', 'message_count': 2, 'last_model': 'claude-3-opus-20240229'}

=== 场景3：使用默认配置 ===
[LLM] User: charlie, Provider: openai, Model: gpt-3.5-turbo
[Log] User charlie has 2 messages
Result 3: {'user_id': 'charlie', 'message_count': 2, 'last_model': 'gpt-3.5-turbo'}

=== 演示 Context 与 State 的区别 ===
Call 0: User=user_0, Model=gpt-3.5-turbo
Call 1: User=user_1, Model=gpt-4
Call 2: User=user_2, Model=gpt-3.5-turbo

=== 演示 State 的持久化 ===
First call: 2 messages
Second call: 4 messages
```

---

## 8. 【面试必问】

### 问题："Runtime Context 和 State 有什么区别？"

**普通回答（❌ 不出彩）：**
"Runtime Context 是配置，State 是状态。"

**出彩回答（✅ 推荐）：**

> **Runtime Context 和 State 有三个核心区别：**
>
> 1. **可变性**：
>    - Runtime Context 是**不可变的**（immutable），节点只能读取，不能修改
>    - State 是**可变的**（mutable），节点可以返回部分更新，框架会自动合并
>
> 2. **生命周期**：
>    - Runtime Context 是**临时的**，只在单次调用期间有效，调用结束后销毁
>    - State 是**持久化的**，可以保存到 Checkpoint，支持断点续传和多轮对话
>
> 3. **用途**：
>    - Runtime Context 用于传递**运行时配置**（如 user_id、model_name），支持动态改变图的行为
>    - State 用于存储**工作流状态数据**（如 messages、search_results），在节点间传递和更新
>
> **设计理念**：这种分离体现了"配置与状态分离"的设计原则，类似于函数式编程中的"纯函数 + 不可变数据"。
>
> **在实际工作中的应用**：在构建多租户 RAG 系统时，我们使用 Runtime Context 传递 user_id 和 tenant_id（配置），使用 State 存储对话历史和检索结果（状态），实现了用户级别的隔离和状态持久化。

**为什么这个回答出彩？**
1. ✅ 多维度对比（可变性、生命周期、用途）
2. ✅ 设计理念（配置与状态分离）
3. ✅ 实际应用场景

---

## 9. 【化骨绵掌】

### 卡片1：Runtime Context 的本质

**一句话：** Runtime Context 是不可变的运行时配置对象。

**举例：**
```python
@dataclass
class Context:
    user_id: str
    model_name: str
```

**应用：** 传递运行时动态配置。

---

### 卡片2：context_schema 的作用

**一句话：** context_schema 定义 Runtime Context 的类型。

**举例：**
```python
builder = StateGraph(State, context_schema=Context)
```

**应用：** 在创建图时指定。

---

### 卡片3：Runtime[Context] 访问

**一句话：** Runtime[Context] 是在节点中访问 context 的接口。

**举例：**
```python
def node(state: State, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
```

**应用：** 在节点中读取配置。

---

### 卡片4：context 参数传递

**一句话：** context 参数在调用图时传递。

**举例：**
```python
result = graph.invoke(
    {"messages": ["hello"]},
    context=Context(user_id="user_123")
)
```

**应用：** 每次调用可以传递不同的 context。

---

### 卡片5：Context 的不可变性

**一句话：** Context 在运行时不可修改。

**举例：**
```python
# ❌ 错误
runtime.context.user_id = "new_user"

# ✅ 正确
user_id = runtime.context.user_id
```

**应用：** 只读取，不修改。

---

### 卡片6：Context vs State

**一句话：** Context 是不可变配置，State 是可变状态。

**举例：**
```python
# Context：配置
context = Context(user_id="user_123")

# State：状态
state = {"messages": ["hello"]}
```

**应用：** 配置用 Context，状态用 State。

---

### 卡片7：动态 LLM 选择

**一句话：** 使用 Runtime Context 在运行时动态选择 LLM。

**举例：**
```python
def call_llm(state: State, runtime: Runtime[Context]):
    model_name = runtime.context.model_name
    llm = get_llm(model_name)
    return llm.invoke(state["messages"])
```

**应用：** 无需重新编译图即可切换模型。

---

### 卡片8：用户级别配置

**一句话：** 使用 Runtime Context 实现用户级别的配置隔离。

**举例：**
```python
context = Context(user_id="user_123", preferences={...})
```

**应用：** 多租户系统中的用户隔离。

---

### 卡片9：Context 的生命周期

**一句话：** Context 在调用期间有效，调用结束后销毁。

**举例：**
```python
result = graph.invoke(input, context=context)
# context 在调用期间有效
# 调用结束后，context 销毁
```

**应用：** 不要依赖 context 的持久性。

---

### 卡片10：Context 的最佳实践

**一句话：** 只在 Context 中放运行时动态配置，不要放状态数据。

**举例：**
```python
# ✅ 正确
@dataclass
class Context:
    user_id: str
    model_name: str

# ❌ 错误
@dataclass
class Context:
    messages: list  # 这是状态，不是配置
```

**应用：** 保持 Context 的纯粹性。

---

## 10. 【一句话总结】

**Runtime Context 是 LangGraph 中传递不可变运行时配置的机制，通过 context_schema 定义类型，在节点中通过 Runtime[Context] 访问，实现配置与状态的分离，支持运行时动态配置和用户级别隔离。**

---

## 引用来源

### 官方文档
- [LangGraph 官方文档 - Runtime Context](https://docs.langchain.com/oss/python/langgraph/) - Context7, 2026
- [LangGraph 官方文档 - Context Schema](https://docs.langchain.com/oss/python/concepts/context) - Context7, 2026

### 技术博客
- [Context Engineering for Agents](https://blog.langchain.com/context-engineering-for-agents/) - LangChain Blog, 2025
- [Context Engineering with LangGraph: Why State Management Matters](https://www.linkedin.com/pulse/context-engineering-langgraph-why-state-management-matters-mainkar-2relf) - Sagar Mainkar, 2025
- [LangGraph State: The Engine Behind Smarter AI Workflows](https://www.cloudthat.com/resources/blog/langgraph-state-the-engine-behind-smarter-ai-workflows) - Abhishek Srivastava, 2025

### 社区讨论
- [Reddit - Context management using State](https://www.reddit.com/r/LangChain/comments/1kz912z/context_management_using_state) - 2025
- [Reddit - Managing shared state in LangGraph multi-agent system](https://www.reddit.com/r/LangGraph/comments/1n867pe/managing_shared_state_in_langgraph_multiagent) - 2025

---

**文档版本：** v1.0
**最后更新：** 2026-02-26
**维护者：** Claude Code
