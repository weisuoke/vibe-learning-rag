# 核心概念 01：Stream 模式详解

## 概述

深入理解 LangChain 流式执行的三种核心模式：updates、messages、custom，掌握每种模式的原理、使用场景和最佳实践。

---

## 模式 1：`stream_mode="updates"` - Agent 进度流式

### 核心原理

**updates 模式追踪 Agent 执行图中每个节点的状态更新。**

```python
# 执行流程
Agent 进入节点 A
↓
执行节点 A 的逻辑
↓
更新状态（state update）
↓
触发流式回调：yield {node_name: state_update}
↓
用户接收到更新
↓
重复直到所有节点执行完毕
```

---

### 返回数据结构

```python
# 单个 chunk 的结构
{
    "node_name": {
        "messages": [AIMessage(...), ToolMessage(...), ...],
        "other_state_keys": ...
    }
}
```

**关键字段**：
- `node_name`: 执行的节点名称（如 "model", "tools", "custom_node"）
- `messages`: 该节点产生的消息列表
- 其他状态键：根据 Agent 的状态定义

---

### 使用示例

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}的天气是晴天"

agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_weather]
)

# 流式追踪 Agent 执行
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "北京天气怎么样？"}]},
    stream_mode="updates"
):
    for node_name, data in chunk.items():
        print(f"\n[节点: {node_name}]")
        if "messages" in data:
            last_msg = data["messages"][-1]
            print(f"消息类型: {type(last_msg).__name__}")
            if hasattr(last_msg, 'content'):
                print(f"内容: {last_msg.content[:100]}")
            if hasattr(last_msg, 'tool_calls'):
                print(f"工具调用: {last_msg.tool_calls}")
```

**输出**：
```
[节点: model]
消息类型: AIMessage
工具调用: [{'name': 'get_weather', 'args': {'city': '北京'}, 'id': 'call_123'}]

[节点: tools]
消息类型: ToolMessage
内容: 北京的天气是晴天

[节点: model]
消息类型: AIMessage
内容: 北京今天是晴天
```

---

### 适用场景

1. **多步推理监控**
```python
# 监控 Agent 的推理步骤
for chunk in agent.stream(input, stream_mode="updates"):
    for node, data in chunk.items():
        if node == "model":
            print("🤖 LLM 正在思考...")
        elif node == "tools":
            print("🔧 工具正在执行...")
```

2. **工作流调试**
```python
# 调试复杂的 Agent 工作流
execution_log = []
for chunk in agent.stream(input, stream_mode="updates"):
    execution_log.append(chunk)
    print(f"步骤 {len(execution_log)}: {list(chunk.keys())}")

# 分析执行路径
print(f"总共执行了 {len(execution_log)} 个步骤")
```

3. **进度追踪**
```python
# 显示执行进度
total_steps = 5  # 预估步骤数
current_step = 0

for chunk in agent.stream(input, stream_mode="updates"):
    current_step += 1
    progress = (current_step / total_steps) * 100
    print(f"进度: {progress:.0f}%")
```

---

### 高级特性：Subgraph 流式

**启用子图流式**：

```python
# 追踪嵌套 Agent 的执行
for namespace, mode, data in agent.stream(
    input,
    stream_mode="updates",
    subgraphs=True  # 启用子图流式
):
    level = len(namespace)
    indent = "  " * level
    print(f"{indent}[Level {level}] {list(data.keys())}")
```

**命名空间格式**：
```python
# 顶层 Agent
namespace = ()

# 一层嵌套
namespace = ('tools:call_abc123',)

# 两层嵌套
namespace = ('tools:call_abc123', 'tools:call_def456')
```

---

## 模式 2：`stream_mode="messages"` - LLM 令牌流式

### 核心原理

**messages 模式实时返回 LLM 生成的每个 token 及其元数据。**

```python
# 执行流程
LLM 开始生成
↓
生成第一个 token
↓
触发回调：on_llm_new_token(token)
↓
创建 AIMessageChunk(content=token)
↓
yield (chunk, metadata)
↓
重复直到生成完成
```

---

### 返回数据结构

```python
# 返回元组：(token, metadata)
(
    AIMessageChunk(content="Hello"),  # token
    {
        "langgraph_node": "model",
        "langgraph_step": 1,
        "lc_agent_name": "main_agent"  # 2026 新增
    }  # metadata
)
```

**Token 类型**：
- `AIMessageChunk`: LLM 生成的文本或工具调用
- `ToolMessage`: 工具执行结果
- 其他消息类型

---

### 使用示例

```python
from langchain.agents import create_agent

agent = create_agent(model="gpt-4o-mini", tools=[...])

# 流式输出 LLM tokens
for token, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "讲个笑话"}]},
    stream_mode="messages"
):
    # 过滤：只输出文本内容
    if hasattr(token, 'content') and token.content:
        print(token.content, end="", flush=True)

    # 过滤：只输出来自特定节点的 token
    if metadata.get('langgraph_node') == 'model':
        # 只处理模型节点的输出
        pass

    # 过滤：只输出特定 Agent 的 token（2026 新增）
    if metadata.get('lc_agent_name') == 'main_agent':
        # 只处理主 Agent 的输出
        pass
```

---

### Token 类型详解

#### 1. 文本 Token

```python
AIMessageChunk(
    content="Hello",  # 文本内容
    chunk_position="first"  # 位置标记：first, middle, last
)
```

#### 2. 工具调用 Token（部分 JSON）

```python
AIMessageChunk(
    content="",
    tool_call_chunks=[
        {
            'name': 'get_weather',
            'args': '{"ci',  # 部分 JSON
            'id': 'call_123',
            'index': 0,
            'type': 'tool_call_chunk'
        }
    ]
)
```

#### 3. 工具结果

```python
ToolMessage(
    content="北京的天气是晴天",
    name="get_weather",
    tool_call_id="call_123"
)
```

---

### 适用场景

1. **ChatGPT 式对话**
```python
# 实时显示 LLM 输出
async for token, metadata in agent.astream(input, stream_mode="messages"):
    if hasattr(token, 'content'):
        print(token.content, end="", flush=True)
print()  # 换行
```

2. **工具调用追踪**
```python
# 追踪工具调用的完整过程
tool_call_buffer = ""

for token, metadata in agent.stream(input, stream_mode="messages"):
    if hasattr(token, 'tool_call_chunks') and token.tool_call_chunks:
        # 累积工具调用的 JSON
        tool_call_buffer += token.tool_call_chunks[0]['args']
        print(f"工具调用进度: {tool_call_buffer}")
```

3. **多 Agent 场景**
```python
# 区分不同 Agent 的输出（2026 新增）
current_agent = None

for token, metadata in agent.stream(input, stream_mode="messages"):
    agent_name = metadata.get('lc_agent_name')
    if agent_name != current_agent:
        print(f"\n[{agent_name}]:")
        current_agent = agent_name

    if hasattr(token, 'content'):
        print(token.content, end="")
```

---

### 高级特性：消息聚合

**聚合 Token 为完整消息**：

```python
from langchain_core.messages import AIMessageChunk

full_message = None

for token, metadata in agent.stream(input, stream_mode="messages"):
    if isinstance(token, AIMessageChunk):
        if full_message is None:
            full_message = token
        else:
            full_message = full_message + token  # 累加

        # 检查是否是最后一个 chunk
        if token.chunk_position == "last":
            print(f"\n完整消息: {full_message}")
            if full_message.tool_calls:
                print(f"工具调用: {full_message.tool_calls}")
            full_message = None
```

---

## 模式 3：`stream_mode="custom"` - 自定义数据流式

### 核心原理

**custom 模式允许在工具函数中发送任意自定义数据。**

```python
# 执行流程
工具函数被调用
↓
调用 get_stream_writer()
↓
writer(custom_data)
↓
数据放入流式队列
↓
yield custom_data
↓
用户接收到自定义数据
```

---

### 实现机制

```python
# langgraph/config.py（简化）
from contextvars import ContextVar

_stream_writer: ContextVar[Optional[StreamWriter]] = ContextVar(
    "_stream_writer", default=None
)

def get_stream_writer(config: Optional[RunnableConfig] = None) -> StreamWriter:
    """从上下文获取 writer"""
    writer = _stream_writer.get()
    if writer is None:
        raise RuntimeError("No stream writer in context")
    return writer

class StreamWriter:
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    def __call__(self, data: Any) -> None:
        """写入数据到流"""
        self.queue.put_nowait(data)
```

---

### 使用示例

```python
from langchain.agents import create_agent
from langgraph.config import get_stream_writer

def process_data(items: list[str]) -> str:
    """处理数据并发送进度"""
    writer = get_stream_writer()

    writer(f"开始处理 {len(items)} 个项目")

    for i, item in enumerate(items):
        # 处理逻辑
        result = process_item(item)

        # 发送进度
        progress = (i + 1) / len(items) * 100
        writer(f"进度: {progress:.1f}% - 已处理 {item}")

    writer("处理完成")
    return "Done"

agent = create_agent(model="gpt-4o-mini", tools=[process_data])

# 流式接收自定义数据
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "处理数据"}]},
    stream_mode="custom"
):
    print(chunk)
```

**输出**：
```
开始处理 10 个项目
进度: 10.0% - 已处理 item1
进度: 20.0% - 已处理 item2
...
进度: 100.0% - 已处理 item10
处理完成
```

---

### 适用场景

1. **数据处理进度**
```python
def batch_process(files: list) -> str:
    writer = get_stream_writer()

    for i, file in enumerate(files):
        writer({"type": "progress", "current": i+1, "total": len(files)})
        process_file(file)

    return "Done"
```

2. **业务状态更新**
```python
def order_workflow(order_id: str) -> str:
    writer = get_stream_writer()

    writer({"status": "验证订单", "order_id": order_id})
    validate_order(order_id)

    writer({"status": "处理支付", "order_id": order_id})
    process_payment(order_id)

    writer({"status": "发货", "order_id": order_id})
    ship_order(order_id)

    return "订单完成"
```

3. **调试信息**
```python
def complex_calculation(data: dict) -> str:
    writer = get_stream_writer()

    writer(f"输入数据: {data}")

    intermediate = step1(data)
    writer(f"步骤 1 结果: {intermediate}")

    result = step2(intermediate)
    writer(f"步骤 2 结果: {result}")

    return result
```

---

### Python < 3.11 兼容性

```python
from langchain_core.runnables import RunnableConfig

# Python < 3.11 需要手动传递 config
def my_tool(query: str, config: RunnableConfig) -> str:
    writer = get_stream_writer(config)  # 传递 config
    writer("Processing...")
    return "Done"

# Python 3.11+ 自动传播上下文
def my_tool(query: str) -> str:
    writer = get_stream_writer()  # 无需传递 config
    writer("Processing...")
    return "Done"
```

---

## 多模式组合

### 组合使用

```python
# 同时使用多种模式
for mode, data in agent.stream(
    input,
    stream_mode=["updates", "messages", "custom"]
):
    if mode == "updates":
        # 处理 Agent 进度
        for node, state in data.items():
            print(f"\n[步骤] {node} 执行完成")

    elif mode == "messages":
        # 处理 LLM tokens
        token, metadata = data
        if hasattr(token, 'content') and token.content:
            print(token.content, end="", flush=True)

    elif mode == "custom":
        # 处理自定义数据
        print(f"\n[进度] {data}")
```

---

### 数据顺序问题

**重要**：多模式流式的数据顺序不保证，不同模式的数据可能交错返回。

```python
# 可能的输出顺序
[messages] token1
[messages] token2
[custom] "Processing..."
[updates] {'model': {...}}
[messages] token3
[custom] "Done"
[updates] {'tools': {...}}
```

**正确处理方式**：

```python
# 使用缓冲区收集数据
buffers = {
    "updates": [],
    "messages": [],
    "custom": []
}

for mode, data in agent.stream(input, stream_mode=["updates", "messages", "custom"]):
    buffers[mode].append(data)

# 处理完所有数据后再使用
print(f"总共 {len(buffers['updates'])} 个步骤")
print(f"总共 {len(buffers['messages'])} 个 token")
print(f"总共 {len(buffers['custom'])} 个自定义消息")
```

---

## 模式选择决策

### 决策树

```
需要什么信息？
├─ Agent 执行步骤
│   └─ stream_mode="updates"
│       ├─ 需要子图信息？
│       │   └─ subgraphs=True
│       └─ 只需要顶层？
│           └─ subgraphs=False（默认）
│
├─ LLM 实时输出
│   └─ stream_mode="messages"
│       ├─ 需要区分 Agent？
│       │   └─ 使用 metadata['lc_agent_name']
│       └─ 需要过滤节点？
│           └─ 使用 metadata['langgraph_node']
│
├─ 自定义进度信号
│   └─ stream_mode="custom"
│       └─ 在工具中使用 get_stream_writer()
│
└─ 需要多种信息
    └─ stream_mode=["updates", "messages", "custom"]
        └─ 使用缓冲区收集数据
```

---

### 场景对照表

| 场景 | 推荐模式 | 原因 |
|------|----------|------|
| ChatGPT 式对话 | `messages` | 需要实时显示 LLM 输出 |
| 多步 Agent 监控 | `updates` | 需要追踪每个步骤 |
| 数据处理进度 | `custom` | 需要自定义进度信号 |
| 工具调用追踪 | `messages` + `updates` | 需要 token 和步骤信息 |
| 嵌套 Agent | `updates` + `subgraphs=True` | 需要追踪子图 |
| 全面监控 | 三种模式组合 | 需要所有信息 |

---

## 性能考虑

### 模式开销对比

```python
# 单模式开销
updates: ~2%
messages: ~5%
custom: ~1%

# 多模式开销（非线性累加）
updates + messages: ~6%
updates + messages + custom: ~8%
```

### 优化建议

1. **只启用需要的模式**
```python
# ❌ 不推荐：启用所有模式
stream_mode=["updates", "messages", "custom"]

# ✅ 推荐：只启用需要的
stream_mode="messages"  # 只需要 LLM 输出
```

2. **合理使用 subgraphs**
```python
# ❌ 不推荐：总是启用
subgraphs=True

# ✅ 推荐：只在需要时启用
subgraphs=False  # 默认，性能更好
```

3. **过滤不需要的数据**
```python
# 只处理特定节点的数据
for token, metadata in agent.stream(input, stream_mode="messages"):
    if metadata.get('langgraph_node') == 'model':
        # 只处理模型节点
        print(token.content, end="")
```

---

## 总结

### 三种模式对比

| 特性 | updates | messages | custom |
|------|---------|----------|--------|
| **用途** | Agent 步骤追踪 | LLM 实时输出 | 自定义进度 |
| **返回数据** | `{node: state}` | `(token, metadata)` | 任意数据 |
| **开销** | ~2% | ~5% | ~1% |
| **适用场景** | 监控、调试 | 对话、长文本 | 业务进度 |
| **高级特性** | Subgraph 流式 | 消息聚合 | 上下文依赖 |

### 核心要点

1. **updates**: 追踪 Agent 执行图的每个节点
2. **messages**: 实时返回 LLM 生成的 token
3. **custom**: 在工具中发送自定义数据
4. **多模式**: 可以组合使用，但数据顺序不保证
5. **性能**: 只启用需要的模式，避免不必要的开销

---

## 参考资源

- **官方文档**: https://docs.langchain.com/oss/python/langchain/streaming/overview
- **源码位置**:
  - `langchain_core/tracers/event_stream.py` - 事件流实现
  - `langgraph/config.py` - get_stream_writer 实现
- **相关知识点**:
  - 02_第一性原理 - 深入理解设计原理
  - 04_最小可用 - 最小 API 集
  - 07_实战代码 - 完整代码示例

---

**版本**: LangChain 0.3.x (2025-2026)
**最后更新**: 2026-02-21
