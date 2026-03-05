---
type: source_code_analysis
source: sourcecode/langgraph/libs/langgraph/tests/
analyzed_files:
  - test_retry.py (前200行)
  - test_pregel_async.py (前300行)
analyzed_at: 2026-02-25
knowledge_point: 08_图基础最佳实践
---

# 源码分析：错误处理与性能优化最佳实践

## 分析的文件

- `sourcecode/langgraph/libs/langgraph/tests/test_retry.py` - 重试策略测试
- `sourcecode/langgraph/libs/langgraph/tests/test_pregel_async.py` - 异步执行和错误处理测试

## 关键发现

### 1. RetryPolicy 设计

**RetryPolicy 配置**:
```python
from langgraph.types import RetryPolicy

# 基础重试策略
retry_policy = RetryPolicy(
    max_attempts=3,              # 最大重试次数
    initial_interval=0.01,       # 初始重试间隔(秒)
    backoff_factor=2.0,          # 退避因子
    jitter=False,                # 是否添加随机抖动
    retry_on=ValueError,         # 重试的异常类型
)

# 添加节点时配置重试策略
graph = StateGraph(State)
graph.add_node("failing_node", failing_node, retry_policy=retry_policy)
```

**retry_on 参数的多种形式**:
```python
# 1. 单个异常类型
RetryPolicy(retry_on=ValueError)

# 2. 多个异常类型
RetryPolicy(retry_on=(ValueError, KeyError))

# 3. 自定义判断函数
def should_retry(exc: Exception) -> bool:
    return isinstance(exc, ValueError) and "retry" in str(exc)

RetryPolicy(retry_on=should_retry)

# 4. 默认重试策略(retry_on 为空)
RetryPolicy()  # 使用 default_retry_on 函数
```

**默认重试策略**:
```python
# 默认会重试:
- ConnectionError (连接错误)
- httpx.HTTPStatusError (5xx 状态码)
- requests.HTTPError (5xx 状态码或无响应)
- 其他自定义异常(默认)

# 默认不会重试:
- ValueError, TypeError, ArithmeticError (编程错误)
- ImportError, NameError, SyntaxError (语法错误)
- RuntimeError, ReferenceError (运行时错误)
- StopIteration, StopAsyncIteration (迭代器停止)
- OSError (操作系统错误)
- httpx.HTTPStatusError (4xx 状态码)
- requests.HTTPError (4xx 状态码)
```

**最佳实践**:
- ✅ 使用 RetryPolicy 处理临时性错误(网络错误、服务暂时不可用)
- ✅ 不要重试编程错误(ValueError, TypeError 等)
- ✅ 使用 backoff_factor 实现指数退避
- ✅ 在生产环境启用 jitter 避免雷鸣群效应
- ✅ 使用自定义 retry_on 函数实现复杂的重试逻辑
- ✅ 合理设置 max_attempts,避免无限重试

### 2. 错误处理模式

**Checkpoint 错误处理**:
```python
# 测试各种 Checkpointer 错误场景
class FaultyGetCheckpointer(InMemorySaver):
    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        raise ValueError("Faulty get_tuple")

class FaultyPutCheckpointer(InMemorySaver):
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        raise ValueError("Faulty put")

class FaultySerializer(JsonPlusSerializer):
    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        raise ValueError("Faulty serializer")
```

**Reducer 错误处理**:
```python
def faulty_reducer(a: Any, b: Any) -> Any:
    raise ValueError("Faulty reducer")

builder = StateGraph(Annotated[str, faulty_reducer])
builder.add_node("agent", logic)
builder.add_edge(START, "agent")
graph = builder.compile(checkpointer=InMemorySaver())

# 执行时会抛出 ValueError: Faulty reducer
```

**最佳实践**:
- ✅ 在 Checkpointer 实现中处理序列化错误
- ✅ 在 Reducer 函数中处理状态合并错误
- ✅ 使用 pytest.raises 测试错误场景
- ✅ 为不同的错误类型提供清晰的错误消息
- ✅ 在图编译时验证配置,而不是运行时

### 3. 异步执行最佳实践

**异步上下文管理器**:
```python
class MyContextManager:
    async def __aenter__(self):
        logs.append("Entering")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logs.append("Starting exit")
        try:
            # 清理工作
            await asyncio.sleep(2)
            logs.append("Cleanup completed")
        except asyncio.CancelledError:
            logs.append("Cleanup was cancelled!")
            raise
        logs.append("Exit finished")
```

**取消处理**:
```python
async def main():
    try:
        async with MyContextManager():
            logs.append("In context")
            await asyncio.sleep(1)
            logs.append("This won't print if cancelled")
    except asyncio.CancelledError:
        logs.append("Context was cancelled")
        raise

# 创建任务并取消
t = asyncio.create_task(main())
await asyncio.sleep(0.2)
t.cancel()
```

**Checkpoint 在取消后的处理**:
```python
class LongPutCheckpointer(InMemorySaver):
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        logs.append("checkpoint.aput.start")
        try:
            await asyncio.sleep(1)
            return await super().aput(config, checkpoint, metadata, new_versions)
        finally:
            logs.append("checkpoint.aput.end")
```

**最佳实践**:
- ✅ 使用 async with 确保资源正确清理
- ✅ 在 __aexit__ 中处理 CancelledError
- ✅ 使用 finally 块确保清理代码执行
- ✅ 在取消后仍然保存 Checkpoint
- ✅ 测试取消场景,确保状态一致性

### 4. 节点函数最佳实践

**节点函数签名**:
```python
# 1. 简单节点
def node(state: State) -> dict:
    return {"foo": "success"}

# 2. 可能失败的节点
def failing_node(state: State):
    nonlocal attempt_count
    attempt_count += 1
    if attempt_count < 3:
        raise ValueError("Intentional failure")
    return {"foo": "success"}

# 3. 异步节点
async def awhile(input: Any) -> None:
    logs.append("awhile.start")
    try:
        await asyncio.sleep(1)
    except asyncio.CancelledError:
        logs.append("awhile.cancelled")
        raise
```

**最佳实践**:
- ✅ 节点函数应该是纯函数或幂等的
- ✅ 使用 nonlocal 或 state 跟踪重试次数
- ✅ 在异步节点中正确处理 CancelledError
- ✅ 节点函数应该返回部分状态更新
- ✅ 避免在节点函数中修改全局状态

### 5. 图编译与执行最佳实践

**图构建模式**:
```python
# 1. 使用 Builder 模式
graph = (
    StateGraph(State)
    .add_node("failing_node", failing_node, retry_policy=retry_policy)
    .add_node("other_node", other_node)
    .add_edge(START, "failing_node")
    .add_edge("failing_node", "other_node")
    .compile()
)

# 2. 分步构建
builder = StateGraph(State)
builder.add_node("agent", logic)
builder.add_edge(START, "agent")
graph = builder.compile(checkpointer=InMemorySaver())
```

**执行模式**:
```python
# 1. 同步执行
result = graph.invoke({"foo": ""})

# 2. 异步执行
result = await graph.ainvoke("", {"configurable": {"thread_id": "thread-1"}})

# 3. 流式执行
async for chunk in graph.astream("", {"configurable": {"thread_id": "thread-2"}}):
    process(chunk)

# 4. 事件流
async for event in graph.astream_events(
    "", {"configurable": {"thread_id": "thread-3"}}, version="v2"
):
    process(event)
```

**最佳实践**:
- ✅ 使用链式调用构建图,提高可读性
- ✅ 在编译时配置 Checkpointer
- ✅ 使用 configurable 传递运行时配置
- ✅ 根据需求选择合适的执行模式
- ✅ 在流式执行中正确处理异常

### 6. 测试最佳实践

**测试重试行为**:
```python
def test_graph_with_single_retry_policy():
    attempt_count = 0

    def failing_node(state: State):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ValueError("Intentional failure")
        return {"foo": "success"}

    retry_policy = RetryPolicy(
        max_attempts=3,
        initial_interval=0.01,
        backoff_factor=2.0,
        jitter=False,
        retry_on=ValueError,
    )

    graph = (
        StateGraph(State)
        .add_node("failing_node", failing_node, retry_policy=retry_policy)
        .add_node("other_node", other_node)
        .add_edge(START, "failing_node")
        .add_edge("failing_node", "other_node")
        .compile()
    )

    with patch("time.sleep") as mock_sleep:
        result = graph.invoke({"foo": ""})

    # 验证重试行为
    assert attempt_count == 3
    assert result["foo"] == "other_node"

    # 验证退避间隔
    call_args_list = [args[0][0] for args in mock_sleep.call_args_list]
    assert call_args_list == [0.01, 0.02]  # 指数退避
```

**测试错误场景**:
```python
async def test_checkpoint_errors():
    class FaultyGetCheckpointer(InMemorySaver):
        async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
            raise ValueError("Faulty get_tuple")

    builder = StateGraph(Annotated[str, operator.add])
    builder.add_node("agent", logic)
    builder.add_edge(START, "agent")

    graph = builder.compile(checkpointer=FaultyGetCheckpointer())

    with pytest.raises(ValueError, match="Faulty get_tuple"):
        await graph.ainvoke("", {"configurable": {"thread_id": "thread-1"}})
```

**最佳实践**:
- ✅ 使用 nonlocal 跟踪节点执行次数
- ✅ 使用 patch 模拟 time.sleep 加速测试
- ✅ 验证重试次数和退避间隔
- ✅ 使用 pytest.raises 测试异常
- ✅ 测试各种错误场景(Checkpointer, Reducer, Serializer)
- ✅ 测试异步取消场景

## 代码片段

### 完整的重试策略示例

```python
from typing_extensions import TypedDict
from langgraph.graph import START, StateGraph
from langgraph.types import RetryPolicy
from unittest.mock import patch

class State(TypedDict):
    foo: str

attempt_count = 0

def failing_node(state: State):
    global attempt_count
    attempt_count += 1
    if attempt_count < 3:  # 前两次失败
        raise ValueError("Intentional failure")
    return {"foo": "success"}

def other_node(state: State):
    return {"foo": "other_node"}

# 创建重试策略
retry_policy = RetryPolicy(
    max_attempts=3,
    initial_interval=0.01,
    backoff_factor=2.0,
    jitter=False,
    retry_on=ValueError,
)

# 构建图
graph = (
    StateGraph(State)
    .add_node("failing_node", failing_node, retry_policy=retry_policy)
    .add_node("other_node", other_node)
    .add_edge(START, "failing_node")
    .add_edge("failing_node", "other_node")
    .compile()
)

# 执行
with patch("time.sleep") as mock_sleep:
    result = graph.invoke({"foo": ""})

print(f"Attempts: {attempt_count}")  # 3
print(f"Result: {result}")  # {'foo': 'other_node'}
print(f"Sleep intervals: {[args[0][0] for args in mock_sleep.call_args_list]}")  # [0.01, 0.02]
```

### 自定义重试逻辑

```python
def should_retry(exc: Exception) -> bool:
    """只重试包含 'retry' 的 ValueError"""
    return isinstance(exc, ValueError) and "retry" in str(exc)

retry_policy = RetryPolicy(
    max_attempts=5,
    initial_interval=0.1,
    backoff_factor=1.5,
    jitter=True,  # 生产环境启用
    retry_on=should_retry,
)
```

### 异步节点错误处理

```python
import asyncio

async def async_node(state: State):
    try:
        # 模拟异步操作
        await asyncio.sleep(1)
        return {"foo": "success"}
    except asyncio.CancelledError:
        # 清理资源
        print("Node cancelled, cleaning up...")
        raise
    except Exception as e:
        # 记录错误
        print(f"Node failed: {e}")
        raise
```

## 总结

从测试文件中提取的核心最佳实践:

1. **重试策略**: 使用 RetryPolicy 处理临时性错误,配置合理的重试次数和退避策略
2. **错误处理**: 在 Checkpointer, Reducer, Serializer 中正确处理错误
3. **异步执行**: 使用 async with 和 finally 确保资源清理
4. **节点设计**: 节点函数应该是纯函数或幂等的
5. **测试**: 使用 mock 和 pytest.raises 测试各种错误场景
6. **取消处理**: 在异步节点中正确处理 CancelledError
