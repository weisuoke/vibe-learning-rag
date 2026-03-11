---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/langgraph/pregel/main.py
  - libs/langgraph/langgraph/pregel/_runner.py
analyzed_at: 2026-03-07
knowledge_point: 09_超时控制
---

# 源码分析：LangGraph 步级超时（`step_timeout`）

## 分析的文件
- `libs/langgraph/langgraph/pregel/main.py` - `Pregel.step_timeout` 定义与调用链
- `libs/langgraph/langgraph/pregel/_runner.py` - 超时等待、取消剩余任务、抛出超时异常

## 关键发现

### 1. LangGraph 原生暴露的是“步级超时”，不是“整图总时长超时”

在 `Pregel` 类中，`step_timeout` 被定义为：

```python
step_timeout: float | None = None
"""Maximum time to wait for a step to complete, in seconds."""
```

这说明它控制的是 **一次 Pregel superstep** 的最长等待时间，而不是整次 `invoke()` / `ainvoke()` 的总运行时间。

### 2. `step_timeout` 在每一轮 `runner.tick()` / `runner.atick()` 中生效

同步路径：

```python
for _ in runner.tick(
    [t for t in loop.tasks.values() if not t.writes],
    timeout=self.step_timeout,
    get_waiter=get_waiter,
    schedule_task=loop.accept_push,
):
    yield from _output(...)
```

异步路径：

```python
async for _ in runner.atick(
    [t for t in loop.tasks.values() if not t.writes],
    timeout=self.step_timeout,
    get_waiter=get_waiter,
    schedule_task=loop.aaccept_push,
):
    for o in _output(...):
        yield o
```

结论：**超时并不是节点自身感知的，而是由执行器在等待当前 step 内所有任务完成时统一判定。**

### 3. 超时实现依赖底层等待器，不是“额外的线程监控器”

同步执行器：

```python
done, inflight = concurrent.futures.wait(
    futures,
    return_when=concurrent.futures.FIRST_COMPLETED,
    timeout=(max(0, end_time - time.monotonic()) if end_time else None),
)
```

异步执行器：

```python
done, inflight = await asyncio.wait(
    futures,
    return_when=asyncio.FIRST_COMPLETED,
    timeout=(max(0, end_time - loop.time()) if end_time else None),
)
```

也就是说，LangGraph 的超时控制是 **把同一 step 里并发执行的一组 future 统一套上 deadline**。

### 4. 超时时会取消剩余未完成任务，并抛出统一异常

`_panic_or_proceed()` 的核心逻辑：

```python
if inflight:
    while inflight:
        inflight.pop().cancel()
    raise timeout_exc_cls("Timed out")
```

同步分支默认抛出 `TimeoutError("Timed out")`，异步分支显式传入 `asyncio.TimeoutError`。

### 5. `step_timeout` 的本质是“外层步预算”

从调用链看，LangGraph 不会自动把 `step_timeout` 传到：
- HTTP 客户端
- 数据库驱动
- 模型 SDK
- 自定义工具函数

因此它只能保证：

1. 当前 step 最多等多久；
2. 超时后取消还没结束的并发任务；
3. 让图执行尽快失败或转入上层处理。

但它**不能代替节点内部 I/O 超时**。

## 代码片段

### `Pregel` 属性定义

```python
step_timeout: float | None = None
"""Maximum time to wait for a step to complete, in seconds."""
```

### 同步执行路径

```python
for _ in runner.tick(
    [t for t in loop.tasks.values() if not t.writes],
    timeout=self.step_timeout,
    get_waiter=get_waiter,
    schedule_task=loop.accept_push,
):
    yield from _output(...)
```

### 异步执行路径

```python
async for _ in runner.atick(
    [t for t in loop.tasks.values() if not t.writes],
    timeout=self.step_timeout,
    get_waiter=get_waiter,
    schedule_task=loop.aaccept_push,
):
    ...
```

### 超时最终抛错位置

```python
if inflight:
    while inflight:
        inflight.pop().cancel()
    raise timeout_exc_cls("Timed out")
```

## 结论

LangGraph 在 1.0.9 源码中提供的原生超时控制核心是 `step_timeout`：

- 粒度：**step 级**；
- 作用点：**执行器等待并发任务完成的阶段**；
- 行为：**取消未完成任务 + 抛出超时异常**；
- 不负责：**节点内部网络/模型/数据库调用的细粒度超时**。

这也是后续所有“超时控制”最佳实践的出发点。

