# 核心概念2：invoke 方法与图执行

> **来源**：基于 LangGraph 源码分析和官方文档整理

---

## 一句话定义

**`invoke()` 方法是执行编译后图的核心接口，内部通过 `stream()` 方法实现，支持同步/异步、流式/批量等多种执行模式，并集成 Checkpoint 实现状态持久化和断点续传。**

---

## 详细解释

### 1. invoke 方法的作用

`invoke()` 是 CompiledStateGraph 的核心执行方法，它完成以下关键任务：

1. **执行图**：运行编译后的图，执行所有节点
2. **状态管理**：通过 Checkpoint 管理状态持久化
3. **中断处理**：处理执行过程中的中断
4. **结果返回**：返回最终的执行结果

### 2. 方法签名

**来源**：`libs/langgraph/langgraph/pregel/main.py:3024`

```python
def invoke(
    self,
    input: InputT | Command | None,
    config: RunnableConfig | None = None,
    *,
    context: ContextT | None = None,
    stream_mode: StreamMode = "values",
    print_mode: StreamMode | Sequence[StreamMode] = (),
    output_keys: str | Sequence[str] | None = None,
    interrupt_before: All | Sequence[str] | None = None,
    interrupt_after: All | Sequence[str] | None = None,
    durability: Durability | None = None,
    **kwargs: Any,
) -> dict[str, Any] | Any:
```

### 3. 核心参数详解

#### 3.1 input（输入数据）

**作用**：图的输入数据，可以是字典或任何其他类型。

**特殊值**：
- `None`：用于恢复中断的执行
- `Command` 对象：用于控制执行流程

**示例**：
```python
# 普通输入
result = graph.invoke({"message": "Hello", "count": 0}, config=config)

# 恢复中断（传递 None）
result = graph.invoke(None, config=config)

# 使用 Command 控制执行
from langgraph.types import Command
result = graph.invoke(Command(resume=True), config=config)
```

#### 3.2 config（运行配置）

**作用**：配置执行参数，最重要的是 `thread_id`。

**必需配置**：
- `thread_id`：用于状态管理和持久化

**可选配置**：
- `callbacks`：回调函数
- `tags`：标签
- `metadata`：元数据

**示例**：
```python
config = {
    "configurable": {
        "thread_id": "thread-123"
    },
    "callbacks": [my_callback],
    "tags": ["production"],
    "metadata": {"user_id": "user-456"}
}

result = graph.invoke(input_data, config=config)
```

#### 3.3 context（静态上下文）

**作用**：在整个执行过程中保持不变的上下文数据（v0.6.0 新增）。

**使用场景**：
- 传递全局配置
- 传递用户信息
- 传递环境变量

**示例**：
```python
context = {
    "user_id": "user-123",
    "environment": "production"
}

result = graph.invoke(input_data, config=config, context=context)
```

#### 3.4 stream_mode（流模式）

**作用**：控制输出的格式。

**可选值**：
- `"values"`：输出完整状态（默认）
- `"updates"`：仅输出状态更新
- `"debug"`：输出调试信息

**示例**：
```python
# 输出完整状态
result = graph.invoke(input_data, config=config, stream_mode="values")

# 仅输出更新
result = graph.invoke(input_data, config=config, stream_mode="updates")
```

#### 3.5 print_mode（打印模式）

**作用**：仅用于调试，打印到控制台，不影响实际输出。

**示例**：
```python
result = graph.invoke(
    input_data,
    config=config,
    print_mode="values"  # 打印完整状态到控制台
)
```

#### 3.6 output_keys（输出键）

**作用**：指定要从图中检索的输出键。

**示例**：
```python
# 只返回特定字段
result = graph.invoke(
    input_data,
    config=config,
    output_keys=["message", "count"]
)
```

#### 3.7 interrupt_before/after（动态中断点）

**作用**：在运行时指定中断点，覆盖编译时的配置。

**示例**：
```python
# 在运行时指定中断点
result = graph.invoke(
    input_data,
    config=config,
    interrupt_before=["approval_node"]
)
```

#### 3.8 durability（持久化模式）

**作用**：控制状态持久化的时机。

**可选值**：
- `"sync"`：同步持久化（下一步开始前完成）
- `"async"`：异步持久化（默认，下一步执行时并行持久化）
- `"exit"`：仅在图退出时持久化

**性能对比**：
- `"sync"`：高持久性，性能成本高
- `"async"`：平衡性能和持久性
- `"exit"`：最高性能，但中途失败会丢失状态

**示例**：
```python
# 同步持久化（高可靠性）
result = graph.invoke(
    input_data,
    config=config,
    durability="sync"
)

# 异步持久化（默认）
result = graph.invoke(
    input_data,
    config=config,
    durability="async"
)

# 退出时持久化（高性能）
result = graph.invoke(
    input_data,
    config=config,
    durability="exit"
)
```

---

## 执行流程详解

### 1. invoke 是 stream 的封装

**来源**：`libs/langgraph/langgraph/pregel/main.py:3065-3103`

```python
def invoke(self, input, config=None, **kwargs):
    # 1. 准备输出键
    output_keys = output_keys if output_keys is not None else self.output_channels

    # 2. 初始化变量
    latest: dict[str, Any] | Any = None
    chunks: list[dict[str, Any] | Any] = []
    interrupts: list[Interrupt] = []

    # 3. 通过 stream 方法执行图
    for chunk in self.stream(
        input,
        config,
        context=context,
        stream_mode=["updates", "values"] if stream_mode == "values" else stream_mode,
        print_mode=print_mode,
        output_keys=output_keys,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        durability=durability,
        **kwargs,
    ):
        # 4. 处理流式输出
        if stream_mode == "values":
            if len(chunk) == 2:
                mode, payload = cast(tuple[StreamMode, Any], chunk)
            else:
                _, mode, payload = cast(
                    tuple[tuple[str, ...], StreamMode, Any], chunk
                )

            # 5. 收集中断信息
            if (
                mode == "updates"
                and isinstance(payload, dict)
                and (ints := payload.get(INTERRUPT)) is not None
            ):
                interrupts.extend(ints)
            # 6. 保存最新值
            elif mode == "values":
                latest = payload
        else:
            chunks.append(chunk)

    # 7. 返回结果
    if stream_mode == "values":
        # 返回最新值（如果有中断，抛出异常）
        if interrupts:
            raise GraphInterrupt(interrupts)
        return latest
    else:
        # 返回所有块
        return chunks
```

**关键发现**：
1. `invoke` 内部调用 `stream` 方法
2. 通过流式执行获取结果
3. 最终返回最后一个值或所有块

### 2. 执行流程图

```
用户调用 invoke()
  ↓
准备配置和参数
  ↓
调用 stream() 方法
  ↓
PregelLoop 执行循环
  ↓
逐步执行节点
  ↓
更新状态
  ↓
保存 Checkpoint（根据 durability）
  ↓
检查中断点
  ↓
返回最终结果
```

### 3. 中断处理

**中断类型**：
1. **静态中断**（编译时配置）：`interrupt_before/after`
2. **动态中断**（运行时配置）：`interrupt_before/after` 参数
3. **代码中断**（使用 `interrupt()` 函数）

**中断恢复**：
```python
# 第一次执行（运行到中断点）
config = {"configurable": {"thread_id": "thread-1"}}
result = graph.invoke(input_data, config=config)

# 检查中断信息
if "__interrupt__" in result:
    print(result["__interrupt__"])

# 恢复执行
result = graph.invoke(None, config=config)
```

---

## 代码示例

### 示例1：基础执行

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    message: str
    count: int

def node_a(state: State):
    return {"count": state["count"] + 1}

def node_b(state: State):
    return {"message": f"Processed {state['count']} times"}

# 构建和编译图
builder = StateGraph(State)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 执行图
config = {"configurable": {"thread_id": "thread-1"}}
result = graph.invoke({"message": "", "count": 0}, config=config)

print(result)
# 输出: {"message": "Processed 1 times", "count": 1}
```

### 示例2：同步 vs 异步执行

```python
import asyncio

# 同步执行
result = graph.invoke(input_data, config=config)

# 异步执行
async def run_async():
    result = await graph.ainvoke(input_data, config=config)
    return result

result = asyncio.run(run_async())
```

### 示例3：流式执行

```python
# 同步流式执行
for chunk in graph.stream(input_data, config=config):
    print(chunk)

# 异步流式执行
async def run_stream():
    async for chunk in graph.astream(input_data, config=config):
        print(chunk)

asyncio.run(run_stream())
```

### 示例4：不同的 stream_mode

```python
# stream_mode="values" - 输出完整状态
for chunk in graph.stream(
    input_data,
    config=config,
    stream_mode="values"
):
    print("完整状态:", chunk)

# stream_mode="updates" - 仅输出更新
for chunk in graph.stream(
    input_data,
    config=config,
    stream_mode="updates"
):
    print("状态更新:", chunk)
```

### 示例5：不同的 durability 模式

```python
# 同步持久化（高可靠性）
result = graph.invoke(
    input_data,
    config=config,
    durability="sync"
)

# 异步持久化（默认，平衡性能）
result = graph.invoke(
    input_data,
    config=config,
    durability="async"
)

# 退出时持久化（高性能）
result = graph.invoke(
    input_data,
    config=config,
    durability="exit"
)
```

### 示例6：处理中断

```python
from langgraph.types import Command

# 编译图（配置中断点）
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["approval_node"]
)

# 第一次执行（运行到中断点）
config = {"configurable": {"thread_id": "thread-1"}}
result = graph.invoke(input_data, config=config)

# 检查中断信息
if "__interrupt__" in result:
    print("中断信息:", result["__interrupt__"])

# 恢复执行（传递 None）
result = graph.invoke(None, config=config)

# 或者使用 Command 恢复
result = graph.invoke(Command(resume=True), config=config)
```

### 示例7：动态中断点

```python
# 在运行时指定中断点（覆盖编译时配置）
result = graph.invoke(
    input_data,
    config=config,
    interrupt_before=["node_b"],  # 动态指定中断点
)

# 恢复执行
result = graph.invoke(None, config=config)
```

### 示例8：批量执行

```python
# 批量执行多个输入
inputs = [
    {"message": "", "count": 0},
    {"message": "", "count": 5},
    {"message": "", "count": 10},
]

configs = [
    {"configurable": {"thread_id": f"thread-{i}"}}
    for i in range(len(inputs))
]

results = graph.batch(inputs, configs)

for i, result in enumerate(results):
    print(f"结果 {i}:", result)
```

---

## 在实际应用中的使用

### 1. 简单执行：使用 invoke()

```python
# 适合：简单的一次性执行
result = graph.invoke(input_data, config=config)
print(result)
```

**优点**：
- 简单直接
- 获取最终结果

**缺点**：
- 无法观察中间过程
- 不适合长时间运行的任务

### 2. 实时监控：使用 stream()

```python
# 适合：需要观察执行过程
for chunk in graph.stream(input_data, config=config, stream_mode="values"):
    print("当前状态:", chunk)
```

**优点**：
- 实时观察执行过程
- 可以提前终止
- 适合调试

**缺点**：
- 需要处理流式输出

### 3. 高并发：使用 ainvoke() 和 astream()

```python
# 适合：高并发场景
async def process_multiple():
    tasks = [
        graph.ainvoke(input1, config=config1),
        graph.ainvoke(input2, config=config2),
        graph.ainvoke(input3, config=config3),
    ]
    results = await asyncio.gather(*tasks)
    return results

results = asyncio.run(process_multiple())
```

**优点**：
- 高并发性能
- 非阻塞执行

**缺点**：
- 需要异步编程知识

### 4. 高可靠性：使用 durability="sync"

```python
# 适合：关键业务流程
result = graph.invoke(
    input_data,
    config=config,
    durability="sync"  # 同步持久化
)
```

**优点**：
- 高可靠性
- 不会丢失状态

**缺点**：
- 性能成本高

### 5. 人机协作：处理中断

```python
# 适合：需要人工审批的流程
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["approval_node"]
)

# 用户提交请求
config = {"configurable": {"thread_id": f"request-{request_id}"}}
result = graph.invoke(request_data, config=config)

# 显示中断信息给用户
if "__interrupt__" in result:
    show_approval_ui(result["__interrupt__"])

# 用户批准后恢复
result = graph.invoke(Command(resume=True), config=config)
```

---

## 执行模式对比

### 1. invoke vs stream

| 特性 | invoke() | stream() |
|------|----------|----------|
| 返回值 | 最终结果 | 流式输出 |
| 实时性 | 等待完成 | 实时输出 |
| 调试 | 不便 | 方便 |
| 性能 | 相同 | 相同 |
| 使用场景 | 简单执行 | 监控调试 |

### 2. 同步 vs 异步

| 特性 | invoke/stream | ainvoke/astream |
|------|---------------|-----------------|
| 执行方式 | 同步阻塞 | 异步非阻塞 |
| 并发性 | 低 | 高 |
| 编程复杂度 | 低 | 高 |
| 使用场景 | 单任务 | 多任务并发 |

### 3. durability 模式对比

| 模式 | 持久化时机 | 可靠性 | 性能 | 使用场景 |
|------|-----------|--------|------|----------|
| sync | 每步之前 | 最高 | 最低 | 关键业务 |
| async | 并行执行 | 中等 | 中等 | 一般业务 |
| exit | 退出时 | 最低 | 最高 | 非关键业务 |

---

## 关键技术点总结

### 1. invoke 的本质

- `invoke` 是 `stream` 的便捷封装
- 内部通过流式执行获取结果
- 返回最后一个值或所有块

### 2. 配置的重要性

- `thread_id` 是必需的（使用 checkpointer 时）
- `durability` 控制持久化时机
- `stream_mode` 控制输出格式

### 3. 中断与恢复

- 支持静态中断（编译时）和动态中断（运行时）
- 使用相同的 `thread_id` 恢复执行
- 中断信息在 `__interrupt__` 字段中返回

### 4. 执行模式选择

- 简单执行：`invoke()`
- 实时监控：`stream()`
- 高并发：`ainvoke()` / `astream()`
- 批量处理：`batch()`

---

## 常见问题

### Q1：invoke 和 stream 有什么区别？

**答**：`invoke` 是 `stream` 的封装，返回最终结果；`stream` 返回流式输出，可以观察中间过程。

```python
# invoke - 返回最终结果
result = graph.invoke(input_data, config=config)

# stream - 返回流式输出
for chunk in graph.stream(input_data, config=config):
    print(chunk)
```

### Q2：什么时候使用 durability="sync"？

**答**：在关键业务流程中使用，确保每一步都持久化，但性能成本高。

```python
# 关键业务：同步持久化
result = graph.invoke(input_data, config=config, durability="sync")

# 一般业务：异步持久化（默认）
result = graph.invoke(input_data, config=config)
```

### Q3：如何处理中断？

**答**：检查 `__interrupt__` 字段，使用相同的 `thread_id` 恢复执行。

```python
# 执行到中断点
result = graph.invoke(input_data, config=config)

# 检查中断
if "__interrupt__" in result:
    print(result["__interrupt__"])

# 恢复执行
result = graph.invoke(None, config=config)
```

### Q4：可以不配置 thread_id 吗？

**答**：如果不使用 checkpointer，可以不配置；如果使用 checkpointer，必须配置。

```python
# 不使用 checkpointer - 不需要 thread_id
graph = builder.compile()
result = graph.invoke(input_data)

# 使用 checkpointer - 需要 thread_id
graph = builder.compile(checkpointer=checkpointer)
result = graph.invoke(input_data, config={"configurable": {"thread_id": "t1"}})
```

### Q5：如何实现高并发执行？

**答**：使用异步方法 `ainvoke()` 和 `asyncio.gather()`。

```python
async def process_multiple():
    tasks = [
        graph.ainvoke(input1, config=config1),
        graph.ainvoke(input2, config=config2),
        graph.ainvoke(input3, config=config3),
    ]
    results = await asyncio.gather(*tasks)
    return results

results = asyncio.run(process_multiple())
```

---

## 参考来源

1. **源码分析**：
   - `libs/langgraph/langgraph/pregel/main.py:3024` - invoke 方法实现
   - `libs/langgraph/langgraph/pregel/main.py:3065-3103` - 执行流程

2. **官方文档**：
   - https://docs.langchain.com/oss/python/langgraph/functional-api - 执行方法
   - https://docs.langchain.com/oss/python/langgraph/durable-execution - 持久化模式
   - https://docs.langchain.com/oss/python/langgraph/streaming - 流式执行

3. **Context7 文档**：
   - Execute LangGraph Workflow: Invoke, Stream, Ainvoke, Astream
   - Configure Durability Mode in LangGraph Stream
   - Stream Full Graph State Values in LangGraph Python
   - Stream Graph State Updates in LangGraph Python

---

## 下一步学习

- **核心概念3**：Pregel 算法与执行流程
- **实战代码**：流式执行与监控
- **实战代码**：持久化模式与性能优化
