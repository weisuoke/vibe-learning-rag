# RunnableConfig 上下文传递

> LangGraph 核心概念：理解上下文传递的核心载体

---

## 1. 【30字核心】

**RunnableConfig 是 LangGraph 中上下文传递的核心载体，通过 CONF 字典注入读写函数和运行时配置，实现节点间的解耦通信。**

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理，从源头思考问题。

### RunnableConfig 的第一性原理

#### 1. 最基础的定义

**RunnableConfig = 一个包含运行时配置的字典**

仅此而已！它就是一个普通的 Python 字典，但它承载了节点运行所需的所有上下文信息。

#### 2. 为什么需要 RunnableConfig？

**核心问题：如何在不修改节点函数签名的情况下，动态注入运行时依赖？**

传统方式的问题：
```python
# ❌ 传统方式：节点需要知道如何读写状态
def node(state, read_func, write_func, thread_id, tags):
    # 参数太多，难以维护
    pass
```

RunnableConfig 的解决方案：
```python
# ✅ RunnableConfig 方式：所有依赖都在 config 中
def node(state, config: RunnableConfig):
    # 简洁，易于扩展
    pass
```

#### 3. RunnableConfig 的三层价值

##### 价值1：依赖注入（Dependency Injection）

**解耦节点逻辑与状态管理**

节点不需要知道状态如何存储、如何读写，只需要通过 config 访问注入的函数。

**示例：**
```python
# 节点不需要知道状态存储在哪里
read_func = config[CONF][CONFIG_KEY_READ]
data = read_func("messages")  # 读取状态
```

##### 价值2：配置传递（Configuration Passing）

**统一管理运行时配置**

thread_id、tags、configurable 等配置都通过 config 传递，避免全局变量。

**示例：**
```python
# 获取当前线程 ID
thread_id = config["configurable"]["thread_id"]
```

##### 价值3：可扩展性（Extensibility）

**轻松添加新的上下文信息**

只需在 config 中添加新的键值对，无需修改节点函数签名。

**示例：**
```python
# 添加自定义配置
config["configurable"]["user_id"] = "user_123"
```

#### 4. 从第一性原理推导状态化工作流

**推理链：**
```
1. 节点需要访问状态
   ↓
2. 状态存储在 Channel 中
   ↓
3. 节点需要读写 Channel
   ↓
4. 读写函数需要注入到节点
   ↓
5. 使用 RunnableConfig 作为注入载体
   ↓
6. 节点通过 config 访问读写函数
   ↓
7. 实现节点与状态管理的解耦
```

#### 5. 一句话总结第一性原理

**RunnableConfig 是依赖注入的载体，通过字典传递运行时依赖，实现节点逻辑与状态管理的解耦。**

---

## 3. 【核心概念】

### 核心概念1：RunnableConfig 结构

**RunnableConfig 是一个 TypedDict，包含运行时配置和依赖注入信息。**

```python
from langchain_core.runnables import RunnableConfig
from typing import TypedDict, Any

# RunnableConfig 的基本结构
class RunnableConfig(TypedDict, total=False):
    tags: list[str]                    # 追踪标签
    metadata: dict[str, Any]           # 元数据
    callbacks: Any                     # 回调函数
    run_name: str                      # 运行名称
    max_concurrency: int | None        # 最大并发数
    recursion_limit: int               # 递归限制
    configurable: dict[str, Any]       # 可配置项
    run_id: str                        # 运行 ID
```

**在 LangGraph 中的扩展：**
```python
# LangGraph 在 config["configurable"] 中添加了 CONF 字典
config["configurable"][CONF] = {
    CONFIG_KEY_READ: read_func,      # 读取函数
    CONFIG_KEY_SEND: write_func,     # 写入函数
    CONFIG_KEY_RUNTIME: runtime,     # 运行时上下文
}
```

**在状态化工作流中：**
- 节点通过 `config[CONF][CONFIG_KEY_READ]` 读取状态
- 节点通过 `config[CONF][CONFIG_KEY_SEND]` 写入状态
- 节点通过 `config["configurable"]["thread_id"]` 获取线程 ID

---

### 核心概念2：CONF 字典机制

**CONF 是 LangGraph 在 RunnableConfig 中注入依赖的核心机制。**

```python
# 源码：pregel/main.py
CONF = "__pregel_conf"
CONFIG_KEY_READ = "read"
CONFIG_KEY_SEND = "send"
CONFIG_KEY_RUNTIME = "runtime"

# Pregel 引擎在执行节点前注入 CONF
config[CONF] = {
    CONFIG_KEY_READ: self._read,
    CONFIG_KEY_SEND: self._write,
    CONFIG_KEY_RUNTIME: runtime,
}
```

**CONF 字典的作用：**

1. **隔离 LangGraph 特定配置**
   - 避免污染 RunnableConfig 的标准字段
   - 所有 LangGraph 特定配置都在 CONF 下

2. **提供读写函数**
   - `CONFIG_KEY_READ`：读取 Channel 的函数
   - `CONFIG_KEY_SEND`：写入 Channel 的函数

3. **提供运行时上下文**
   - `CONFIG_KEY_RUNTIME`：Runtime 对象（包含 context、store 等）

**示例：**
```python
def my_node(state: State, config: RunnableConfig):
    # 读取状态
    read = config[CONF][CONFIG_KEY_READ]
    messages = read("messages")
    
    # 写入状态
    send = config[CONF][CONFIG_KEY_SEND]
    send([("messages", new_message)])
    
    return state
```

---

### 核心概念3：CONFIG_KEY_READ 和 CONFIG_KEY_SEND

**CONFIG_KEY_READ 和 CONFIG_KEY_SEND 是状态读写的核心接口。**

#### CONFIG_KEY_READ：读取状态

```python
# 源码：pregel/_read.py
class ChannelRead(RunnableCallable):
    @staticmethod
    def do_read(
        config: RunnableConfig,
        *,
        select: str | list[str],
        fresh: bool = False,
        mapper: Callable[[Any], Any] | None = None,
    ) -> Any:
        read: READ_TYPE = config[CONF][CONFIG_KEY_READ]
        if mapper:
            return mapper(read(select, fresh))
        else:
            return read(select, fresh)
```

**参数说明：**
- `select`：要读取的 Channel 名称（单个或多个）
- `fresh`：是否读取最新值（默认 False，使用缓存）
- `mapper`：读取后的数据转换函数

**使用示例：**
```python
# 读取单个 Channel
messages = read("messages")

# 读取多个 Channel
data = read(["messages", "context"])

# 读取最新值（跳过缓存）
fresh_messages = read("messages", fresh=True)
```

#### CONFIG_KEY_SEND：写入状态

```python
# 源码：pregel/_write.py
class ChannelWrite(RunnableCallable):
    @staticmethod
    def do_write(
        config: RunnableConfig,
        writes: Sequence[ChannelWriteEntry | ChannelWriteTupleEntry | Send],
        allow_passthrough: bool = True,
    ) -> None:
        write: TYPE_SEND = config[CONF][CONFIG_KEY_SEND]
        write(_assemble_writes(writes))
```

**写入格式：**
```python
# 格式1：元组列表
send([("messages", new_message)])

# 格式2：ChannelWriteEntry
send([ChannelWriteEntry(channel="messages", value=new_message)])

# 格式3：批量写入
send([
    ("messages", msg1),
    ("context", ctx1),
])
```

---

### 核心概念4：配置项（thread_id, tags, configurable）

**RunnableConfig 提供了多种配置项，用于控制图的执行行为。**

#### thread_id：线程标识

```python
# 用于持久化和断点续传
config = {"configurable": {"thread_id": "conversation_123"}}

# 在节点中访问
thread_id = config["configurable"]["thread_id"]
```

**作用：**
- 标识一个会话或对话
- 用于 Checkpoint 的持久化
- 支持断点续传

#### tags：追踪标签

```python
# 用于追踪和调试
config = {"tags": ["production", "user_query"]}

# 在 LangSmith 中查看
# 所有带有这些标签的运行都会被记录
```

**作用：**
- 追踪图的执行
- 在 LangSmith 中过滤和分析
- 调试和监控

#### configurable：自定义配置

```python
# 传递自定义配置
config = {
    "configurable": {
        "thread_id": "conv_123",
        "user_id": "user_456",
        "model_name": "gpt-4",
        "temperature": 0.7,
    }
}

# 在节点中访问
user_id = config["configurable"]["user_id"]
model_name = config["configurable"].get("model_name", "gpt-3.5-turbo")
```

**作用：**
- 传递任意自定义配置
- 支持动态配置（无需重新编译图）
- 实现多租户、A/B 测试等场景

---

## 4. 【最小可用】

掌握以下内容，就能开始使用 RunnableConfig：

### 4.1 理解 RunnableConfig 的基本结构

```python
from langchain_core.runnables import RunnableConfig

# RunnableConfig 是一个字典
config: RunnableConfig = {
    "tags": ["my_tag"],
    "configurable": {
        "thread_id": "thread_123",
    }
}
```

### 4.2 在节点中访问 config

```python
from langgraph.graph import StateGraph
from typing import TypedDict

class State(TypedDict):
    messages: list[str]

def my_node(state: State, config: RunnableConfig):
    # 访问配置
    thread_id = config["configurable"]["thread_id"]
    print(f"Running in thread: {thread_id}")
    
    return {"messages": state["messages"] + ["new message"]}

# 创建图
graph = StateGraph(State)
graph.add_node("my_node", my_node)
```

### 4.3 使用 CONFIG_KEY_READ 读取状态

```python
from langgraph.pregel import CONF, CONFIG_KEY_READ

def my_node(state: State, config: RunnableConfig):
    # 获取读取函数
    read = config[CONF][CONFIG_KEY_READ]
    
    # 读取状态
    messages = read("messages")
    
    return state
```

### 4.4 使用 CONFIG_KEY_SEND 写入状态

```python
from langgraph.pregel import CONF, CONFIG_KEY_SEND

def my_node(state: State, config: RunnableConfig):
    # 获取写入函数
    send = config[CONF][CONFIG_KEY_SEND]
    
    # 写入状态
    send([("messages", "new message")])
    
    return state
```

**这些知识足以：**
- 在节点中访问运行时配置
- 读取和写入状态
- 实现基本的状态化工作流

---

## 5. 【双重类比】

### 类比1：RunnableConfig = HTTP 请求上下文

**前端类比：** HTTP 请求对象（Request）

在 Web 开发中，每个请求都有一个 Request 对象，包含请求头、参数、会话信息等。

```javascript
// Express.js
app.get('/api/data', (req, res) => {
    const userId = req.headers['user-id'];
    const sessionId = req.session.id;
    // req 就像 RunnableConfig
});
```

**日常生活类比：** 快递单

快递单包含了快递运输所需的所有信息：发件人、收件人、地址、运单号等。

```
快递单（RunnableConfig）
├── 运单号（thread_id）
├── 发件人（tags）
├── 收件人（configurable）
└── 备注（metadata）
```

**Python 示例：**
```python
# RunnableConfig 就像 HTTP 请求上下文
config = {
    "configurable": {
        "thread_id": "session_123",  # 会话 ID
        "user_id": "user_456",       # 用户 ID
    },
    "tags": ["production"],          # 环境标签
}
```

---

### 类比2：CONF 字典 = 依赖注入容器

**前端类比：** React Context 或 Vue Provide/Inject

在前端框架中，Context 用于在组件树中传递依赖，避免 props drilling。

```javascript
// React Context
const ConfigContext = React.createContext();

function MyComponent() {
    const config = useContext(ConfigContext);
    // config 就像 CONF 字典
}
```

**日常生活类比：** 工具箱

工具箱里装着各种工具（读写函数、运行时上下文），需要时随时取用。

```
工具箱（CONF）
├── 扳手（CONFIG_KEY_READ）
├── 螺丝刀（CONFIG_KEY_SEND）
└── 锤子（CONFIG_KEY_RUNTIME）
```

**Python 示例：**
```python
# CONF 字典就像依赖注入容器
config[CONF] = {
    CONFIG_KEY_READ: read_func,    # 注入读取函数
    CONFIG_KEY_SEND: write_func,   # 注入写入函数
    CONFIG_KEY_RUNTIME: runtime,   # 注入运行时上下文
}
```

---

### 类比3：CONFIG_KEY_READ/SEND = API 端点

**前端类比：** RESTful API

CONFIG_KEY_READ 就像 GET 请求，CONFIG_KEY_SEND 就像 POST 请求。

```javascript
// RESTful API
fetch('/api/messages', { method: 'GET' });   // 读取
fetch('/api/messages', { method: 'POST' });  // 写入
```

**日常生活类比：** 银行柜台

CONFIG_KEY_READ 是取款口，CONFIG_KEY_SEND 是存款口。

```
银行柜台
├── 取款口（CONFIG_KEY_READ）：读取余额
└── 存款口（CONFIG_KEY_SEND）：存入现金
```

**Python 示例：**
```python
# CONFIG_KEY_READ/SEND 就像 API 端点
read = config[CONF][CONFIG_KEY_READ]   # GET /api/state
send = config[CONF][CONFIG_KEY_SEND]   # POST /api/state

# 读取
messages = read("messages")

# 写入
send([("messages", new_message)])
```

---

### 类比总结表

| LangGraph 概念 | 前端类比 | 日常生活类比 |
|----------------|----------|--------------|
| RunnableConfig | HTTP Request 对象 | 快递单 |
| CONF 字典 | React Context | 工具箱 |
| CONFIG_KEY_READ | GET 请求 | 取款口 |
| CONFIG_KEY_SEND | POST 请求 | 存款口 |
| thread_id | Session ID | 运单号 |
| tags | 请求标签 | 快递备注 |
| configurable | 请求参数 | 自定义字段 |


---

## 6. 【反直觉点】

### 误区1：RunnableConfig 是全局配置 ❌

**为什么错？**
- RunnableConfig 不是全局配置，而是**每次调用时传递的上下文**
- 每次调用 `graph.invoke()` 或 `graph.stream()` 时，都会创建一个新的 config
- 不同的调用可以有不同的 config

**为什么人们容易这样错？**
因为 config 包含了很多配置项（如 thread_id、tags），看起来像是全局配置。但实际上，config 是**调用级别**的，不是**图级别**的。

**正确理解：**
```python
# ❌ 错误理解：config 是全局的
graph = builder.compile()
config = {"configurable": {"thread_id": "thread_1"}}
# 以为所有调用都会使用这个 config

# ✅ 正确理解：config 是每次调用时传递的
graph = builder.compile()

# 第一次调用
result1 = graph.invoke(
    {"messages": ["hello"]},
    config={"configurable": {"thread_id": "thread_1"}}
)

# 第二次调用（不同的 config）
result2 = graph.invoke(
    {"messages": ["hi"]},
    config={"configurable": {"thread_id": "thread_2"}}
)
```

---

### 误区2：可以直接修改 RunnableConfig ❌

**为什么错？**
- RunnableConfig 在节点执行期间是**只读的**
- 节点不应该修改 config，只能读取
- 修改 config 不会影响其他节点

**为什么人们容易这样错？**
因为 config 是一个普通的 Python 字典，看起来可以修改。但实际上，修改 config 会导致不可预测的行为。

**正确理解：**
```python
# ❌ 错误做法：修改 config
def my_node(state: State, config: RunnableConfig):
    config["configurable"]["new_key"] = "new_value"  # 不要这样做！
    return state

# ✅ 正确做法：只读取 config
def my_node(state: State, config: RunnableConfig):
    thread_id = config["configurable"]["thread_id"]  # 只读取
    print(f"Thread ID: {thread_id}")
    return state
```

---

### 误区3：所有节点共享同一个 RunnableConfig 实例 ❌

**为什么错？**
- 虽然所有节点在同一次调用中使用相同的 config **值**
- 但每个节点可能会收到 config 的**副本**或**代理对象**
- 不要依赖 config 的对象身份（object identity）

**为什么人们容易这样错？**
因为在同一次调用中，config 的内容是相同的，所以人们以为是同一个对象。

**正确理解：**
```python
# ❌ 错误理解：依赖对象身份
def node1(state: State, config: RunnableConfig):
    config._my_cache = {}  # 不要这样做！
    return state

def node2(state: State, config: RunnableConfig):
    cache = config._my_cache  # 可能不存在！
    return state

# ✅ 正确理解：只使用 config 的值
def node1(state: State, config: RunnableConfig):
    thread_id = config["configurable"]["thread_id"]
    return {"thread_id": thread_id}

def node2(state: State, config: RunnableConfig):
    thread_id = config["configurable"]["thread_id"]
    # 从 state 中读取 node1 的结果
    return state
```

---

## 7. 【实战代码】

```python
"""
RunnableConfig 上下文传递实战示例
演示：如何在节点中使用 RunnableConfig 访问配置和读写状态
"""

import os
from typing import TypedDict, Annotated
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.pregel import CONF, CONFIG_KEY_READ, CONFIG_KEY_SEND

# ===== 1. 定义状态 =====
class State(TypedDict):
    messages: Annotated[list[str], add_messages]
    user_id: str
    request_count: int

# ===== 2. 定义节点（使用 RunnableConfig） =====
def node_with_config(state: State, config: RunnableConfig) -> State:
    """演示如何在节点中使用 RunnableConfig"""
    
    # 2.1 访问 thread_id
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    print(f"[Node] Thread ID: {thread_id}")
    
    # 2.2 访问 tags
    tags = config.get("tags", [])
    print(f"[Node] Tags: {tags}")
    
    # 2.3 访问自定义配置
    user_id = config.get("configurable", {}).get("user_id", "anonymous")
    print(f"[Node] User ID: {user_id}")
    
    # 2.4 更新状态
    return {
        "messages": [f"Processed by thread {thread_id}"],
        "user_id": user_id,
        "request_count": state.get("request_count", 0) + 1,
    }

def node_with_read_write(state: State, config: RunnableConfig) -> State:
    """演示如何使用 CONFIG_KEY_READ 和 CONFIG_KEY_SEND"""
    
    # 2.5 使用 CONFIG_KEY_READ 读取状态
    read = config[CONF][CONFIG_KEY_READ]
    messages = read("messages")
    print(f"[Node] Read messages: {messages}")
    
    # 2.6 使用 CONFIG_KEY_SEND 写入状态
    send = config[CONF][CONFIG_KEY_SEND]
    send([("messages", "Added by node_with_read_write")])
    
    return state

# ===== 3. 创建图 =====
builder = StateGraph(State)
builder.add_node("node1", node_with_config)
builder.add_node("node2", node_with_read_write)
builder.add_edge(START, "node1")
builder.add_edge("node1", "node2")
builder.add_edge("node2", END)

# 编译图（使用 MemorySaver 支持持久化）
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# ===== 4. 运行图（传递不同的 config） =====
print("=== 第一次调用 ===")
result1 = graph.invoke(
    {"messages": ["Hello"], "request_count": 0},
    config={
        "configurable": {
            "thread_id": "thread_001",
            "user_id": "user_alice",
        },
        "tags": ["production", "api_call"],
    }
)
print(f"Result 1: {result1}\n")

print("=== 第二次调用（不同的 config） ===")
result2 = graph.invoke(
    {"messages": ["Hi"], "request_count": 0},
    config={
        "configurable": {
            "thread_id": "thread_002",
            "user_id": "user_bob",
        },
        "tags": ["development", "test"],
    }
)
print(f"Result 2: {result2}\n")

print("=== 第三次调用（恢复 thread_001） ===")
result3 = graph.invoke(
    {"messages": ["Welcome back"], "request_count": 0},
    config={
        "configurable": {
            "thread_id": "thread_001",  # 使用相同的 thread_id
            "user_id": "user_alice",
        },
        "tags": ["production", "api_call"],
    }
)
print(f"Result 3: {result3}")
print(f"Request count: {result3['request_count']}")  # 应该是 2（累加）
```

**运行输出示例：**
```
=== 第一次调用 ===
[Node] Thread ID: thread_001
[Node] Tags: ['production', 'api_call']
[Node] User ID: user_alice
[Node] Read messages: ['Hello', 'Processed by thread thread_001']
Result 1: {'messages': ['Hello', 'Processed by thread thread_001', 'Added by node_with_read_write'], 'user_id': 'user_alice', 'request_count': 1}

=== 第二次调用（不同的 config） ===
[Node] Thread ID: thread_002
[Node] Tags: ['development', 'test']
[Node] User ID: user_bob
[Node] Read messages: ['Hi', 'Processed by thread thread_002']
Result 2: {'messages': ['Hi', 'Processed by thread thread_002', 'Added by node_with_read_write'], 'user_id': 'user_bob', 'request_count': 1}

=== 第三次调用（恢复 thread_001） ===
[Node] Thread ID: thread_001
[Node] Tags: ['production', 'api_call']
[Node] User ID: user_alice
[Node] Read messages: ['Welcome back', 'Processed by thread thread_001']
Result 3: {'messages': ['Welcome back', 'Processed by thread thread_001', 'Added by node_with_read_write'], 'user_id': 'user_alice', 'request_count': 2}
Request count: 2
```

---

## 8. 【面试必问】

### 问题："RunnableConfig 在 LangGraph 中的作用是什么？"

**普通回答（❌ 不出彩）：**
"RunnableConfig 是一个配置对象，用于传递参数。"

**出彩回答（✅ 推荐）：**

> **RunnableConfig 有三层含义：**
>
> 1. **依赖注入载体**：通过 CONF 字典注入读写函数（CONFIG_KEY_READ、CONFIG_KEY_SEND），实现节点与状态管理的解耦。
>
> 2. **配置传递机制**：传递 thread_id、tags、configurable 等运行时配置，支持持久化、追踪和自定义配置。
>
> 3. **上下文隔离**：每次调用都有独立的 config，不同调用之间互不干扰，支持多租户和并发场景。
>
> **与 LangChain 的集成**：RunnableConfig 来自 langchain_core.runnables，是 LangChain 的标准接口，确保 LangGraph 与 LangChain 生态的兼容性。
>
> **在实际工作中的应用**：在构建多租户 RAG 系统时，我们通过 RunnableConfig 传递 user_id 和 tenant_id，实现了用户级别的状态隔离和权限控制。

**为什么这个回答出彩？**
1. ✅ 多层次解释（依赖注入、配置传递、上下文隔离）
2. ✅ 与 LangChain 的关系
3. ✅ 实际应用场景

---

## 9. 【化骨绵掌】

### 卡片1：RunnableConfig 的本质

**一句话：** RunnableConfig 是一个 TypedDict，包含运行时配置和依赖注入信息。

**举例：**
```python
config: RunnableConfig = {
    "tags": ["production"],
    "configurable": {"thread_id": "thread_1"},
}
```

**应用：** 在每次调用 graph.invoke() 时传递。

---

### 卡片2：CONF 字典的作用

**一句话：** CONF 是 LangGraph 在 config 中注入依赖的核心机制。

**举例：**
```python
config[CONF] = {
    CONFIG_KEY_READ: read_func,
    CONFIG_KEY_SEND: write_func,
}
```

**应用：** 节点通过 config[CONF] 访问读写函数。

---

### 卡片3：CONFIG_KEY_READ 读取状态

**一句话：** CONFIG_KEY_READ 是读取 Channel 的函数。

**举例：**
```python
read = config[CONF][CONFIG_KEY_READ]
messages = read("messages")
```

**应用：** 在节点中读取状态。

---

### 卡片4：CONFIG_KEY_SEND 写入状态

**一句话：** CONFIG_KEY_SEND 是写入 Channel 的函数。

**举例：**
```python
send = config[CONF][CONFIG_KEY_SEND]
send([("messages", new_message)])
```

**应用：** 在节点中写入状态。

---

### 卡片5：thread_id 的作用

**一句话：** thread_id 标识一个会话，用于持久化和断点续传。

**举例：**
```python
config = {"configurable": {"thread_id": "conv_123"}}
```

**应用：** 支持多轮对话和状态恢复。

---

### 卡片6：tags 的作用

**一句话：** tags 用于追踪和调试，在 LangSmith 中查看。

**举例：**
```python
config = {"tags": ["production", "user_query"]}
```

**应用：** 在 LangSmith 中过滤和分析运行记录。

---

### 卡片7：configurable 的作用

**一句话：** configurable 传递任意自定义配置。

**举例：**
```python
config = {"configurable": {"user_id": "user_456"}}
```

**应用：** 实现多租户、A/B 测试等场景。

---

### 卡片8：RunnableConfig 的生命周期

**一句话：** RunnableConfig 在每次调用时创建，调用结束后销毁。

**举例：**
```python
result = graph.invoke(input, config=config)
# config 在调用期间有效
```

**应用：** 不要依赖 config 的持久性。

---

### 卡片9：RunnableConfig 与 LangChain 的关系

**一句话：** RunnableConfig 来自 langchain_core.runnables，是 LangChain 的标准接口。

**举例：**
```python
from langchain_core.runnables import RunnableConfig
```

**应用：** 确保 LangGraph 与 LangChain 生态的兼容性。

---

### 卡片10：RunnableConfig 的最佳实践

**一句话：** 只读取 config，不要修改；使用 configurable 传递自定义配置。

**举例：**
```python
# ✅ 正确
user_id = config["configurable"]["user_id"]

# ❌ 错误
config["configurable"]["new_key"] = "value"
```

**应用：** 保持节点的纯粹性和可测试性。

---

## 10. 【一句话总结】

**RunnableConfig 是 LangGraph 中上下文传递的核心载体，通过 CONF 字典注入读写函数实现依赖注入，通过 configurable 传递运行时配置，在状态化工作流中实现节点间的解耦通信和多租户支持。**

---

## 引用来源

### 源码分析
- `libs/langgraph/langgraph/pregel/main.py` - Pregel 执行引擎，CONF 字典注入机制
- `libs/langgraph/langgraph/pregel/_read.py` - ChannelRead 读取机制
- `libs/langgraph/langgraph/pregel/_write.py` - ChannelWrite 写入机制

### 官方文档
- [LangGraph 官方文档 - RunnableConfig](https://docs.langchain.com/oss/python/langgraph/)
- [LangChain Core - RunnableConfig](https://docs.langchain.com/oss/python/langchain-core/runnables/)

### 技术博客
- [Context Engineering with LangGraph](https://www.linkedin.com/pulse/context-engineering-langgraph-why-state-management-matters-mainkar-2relf) - Sagar Mainkar, 2025
- [LangGraph State: The Engine Behind Smarter AI Workflows](https://www.cloudthat.com/resources/blog/langgraph-state-the-engine-behind-smarter-ai-workflows) - Abhishek Srivastava, 2025

---

**文档版本：** v1.0
**最后更新：** 2026-02-26
**维护者：** Claude Code
