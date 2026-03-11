---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/tests/test_pregel_async.py
  - libs/sdk-py/langgraph_sdk/_sync/client.py
  - libs/sdk-py/langgraph_sdk/_async/client.py
  - libs/sdk-py/langgraph_sdk/schema.py
  - libs/sdk-py/langgraph_sdk/errors.py
analyzed_at: 2026-03-07
knowledge_point: 09_超时控制
---

# 源码分析：流式背压、SDK 超时与运行状态

## 分析的文件
- `libs/langgraph/tests/test_pregel_async.py` - `astream()` 背压导致超时的测试
- `libs/sdk-py/langgraph_sdk/_sync/client.py` - 同步客户端默认 HTTP timeout
- `libs/sdk-py/langgraph_sdk/_async/client.py` - 异步客户端默认 HTTP timeout
- `libs/sdk-py/langgraph_sdk/schema.py` - `RunStatus` 中的 `timeout`
- `libs/sdk-py/langgraph_sdk/errors.py` - `APITimeoutError`

## 关键发现

### 1. 流式消费端“变慢”也可能触发 step timeout

异步测试 `test_step_timeout_on_stream_hang()` 构造了一个典型场景：

- 一个慢节点 `awhile()` 睡眠 1.5 秒；
- 一个快节点 `alittlewhile()` 0.6 秒返回；
- 图设置 `graph.step_timeout = 1`；
- 消费端在 `async for chunk in graph.astream(...)` 内再 `await asyncio.sleep(...)`。

最终断言：

```python
with pytest.raises(asyncio.TimeoutError):
    async for chunk in graph.astream(...):
        ...

assert inner_task_cancelled
```

说明：**超时不仅来自节点慢，也可能来自“流式消费端背压 + 慢任务未完成”的组合。**

### 2. 异步 timeout 会触发内部任务取消

测试里通过 `inner_task_cancelled` 标记确认：

```python
except asyncio.CancelledError:
    inner_task_cancelled = True
    raise
```

这验证了 LangGraph 异步执行在 timeout 时不会只“表面报错”，而是会真的取消未完成任务。

### 3. SDK 的 `timeout` 是 HTTP 层超时，不是图步级超时

异步客户端：

```python
timeout=(
    httpx.Timeout(timeout)
    if timeout is not None
    else httpx.Timeout(connect=5, read=300, write=300, pool=5)
)
```

同步客户端也是同样默认值。

因此：

- `graph.step_timeout` 约束的是 **图内部一个 step**；
- `get_client(timeout=...)` 约束的是 **客户端 HTTP 请求生命周期**。

二者是两层不同预算。

### 4. 平台 / Server 侧运行状态有显式 `timeout`

SDK schema：

```python
RunStatus = Literal[
    "pending", "running", "error", "success", "timeout", "interrupted"
]
```

并附带说明：

```python
- "timeout": The run exceeded its time limit.
```

这表明在平台化运行中，“timeout” 会被提升为 run-level 状态，不只是 Python 进程里抛个异常。

### 5. SDK 还定义了 `APITimeoutError`

```python
class APITimeoutError(APIConnectionError):
    def __init__(self, request: httpx.Request) -> None:
        super().__init__(message="Request timed out.", request=request)
```

它描述的是“客户端请求超时”，不是“工作流节点超时”。

## 代码片段

### 流式背压 timeout 测试

```python
graph = builder.compile()
graph.step_timeout = 1

with pytest.raises(asyncio.TimeoutError):
    async for chunk in graph.astream({"hello": "world"}, stream_mode="updates"):
        assert chunk == {"alittlewhile": {"hello": "1"}}
        await asyncio.sleep(stream_hang_s)

assert inner_task_cancelled
```

### SDK 默认 HTTP timeout

```python
httpx.Timeout(connect=5, read=300, write=300, pool=5)
```

### RunStatus 含 `timeout`

```python
RunStatus = Literal["pending", "running", "error", "success", "timeout", "interrupted"]
```

## 结论

“超时控制”在 LangGraph 生态至少分三层：

1. **图执行层**：`step_timeout`；
2. **客户端网络层**：`httpx.Timeout` / SDK `timeout`；
3. **平台运行层**：`RunStatus = timeout`。

如果把这三层混为一谈，就会在排障时产生大量误判。

