---
type: source_code_analysis
source: sourcecode/langgraph/libs/langgraph/langgraph/pregel/_retry.py
analyzed_files:
  - _retry.py (lines 1-218)
analyzed_at: 2026-02-25
knowledge_point: 节点函数约定
---

# 源码分析：节点函数重试机制

## 分析的文件
- `sourcecode/langgraph/libs/langgraph/langgraph/pregel/_retry.py` - 重试策略实现

## 关键发现

### 1. RetryPolicy 结构

```python
# types.py (referenced in _retry.py)
class RetryPolicy:
    max_attempts: int
    initial_interval: float
    backoff_factor: float
    max_interval: float
    jitter: bool
    retry_on: type[Exception] | Sequence[type[Exception]] | Callable[[Exception], bool]
```

### 2. 同步重试实现

```python
# _retry.py:26-106
def run_with_retry(
    task: PregelExecutableTask,
    retry_policy: Sequence[RetryPolicy] | None,
    configurable: dict[str, Any] | None = None,
) -> None:
    """Run a task with retries."""
    retry_policy = task.retry_policy or retry_policy
    attempts = 0
    config = task.config
    if configurable is not None:
        config = patch_configurable(config, configurable)
    while True:
        try:
            # clear any writes from previous attempts
            task.writes.clear()
            # run the task
            return task.proc.invoke(task.input, config)
        except ParentCommand as exc:
            # handle parent command
            ns: str = config[CONF][CONFIG_KEY_CHECKPOINT_NS]
            cmd = exc.args[0]
            if cmd.graph in (ns, recast_checkpoint_ns(ns), task.name):
                for w in task.writers:
                    w.invoke(cmd, config)
                break
            elif cmd.graph == Command.PARENT:
                parts = ns.split(NS_SEP)
                if parts[-1].isdigit():
                    parts.pop()
                parent_ns = NS_SEP.join(parts[:-1])
                exc.args = (replace(cmd, graph=parent_ns),)
            raise
        except GraphBubbleUp:
            raise
        except Exception as exc:
            if SUPPORTS_EXC_NOTES:
                exc.add_note(f"During task with name '{task.name}' and id '{task.id}'")
            if not retry_policy:
                raise

            # Check which retry policy applies to this exception
            matching_policy = None
            for policy in retry_policy:
                if _should_retry_on(policy, exc):
                    matching_policy = policy
                    break

            if not matching_policy:
                raise

            # increment attempts
            attempts += 1
            # check if we should give up
            if attempts >= matching_policy.max_attempts:
                raise
            # sleep before retrying
            interval = matching_policy.initial_interval
            # Apply backoff factor based on attempt count
            interval = min(
                matching_policy.max_interval,
                interval * (matching_policy.backoff_factor ** (attempts - 1)),
            )

            # Apply jitter if configured
            sleep_time = (
                interval + random.uniform(0, 1) if matching_policy.jitter else interval
            )
            time.sleep(sleep_time)

            # log the retry
            logger.info(
                f"Retrying task {task.name} after {sleep_time:.2f} seconds (attempt {attempts}) after {exc.__class__.__name__} {exc}",
                exc_info=exc,
            )
            # signal subgraphs to resume (if available)
            config = patch_configurable(config, {CONFIG_KEY_RESUMING: True})
```

**关键点**：
1. 清除之前尝试的写入
2. 检查匹配的重试策略
3. 指数退避算法
4. 可选的抖动（jitter）
5. 日志记录

### 3. 异步重试实现

```python
# _retry.py:108-202
async def arun_with_retry(
    task: PregelExecutableTask,
    retry_policy: Sequence[RetryPolicy] | None,
    stream: bool = False,
    match_cached_writes: Callable[[], Awaitable[Sequence[PregelExecutableTask]]]
    | None = None,
    configurable: dict[str, Any] | None = None,
) -> None:
    """Run a task asynchronously with retries."""
    retry_policy = task.retry_policy or retry_policy
    attempts = 0
    config = task.config
    if configurable is not None:
        config = patch_configurable(config, configurable)
    if match_cached_writes is not None and task.cache_key is not None:
        for t in await match_cached_writes():
            if t is task:
                # if the task is already cached, return
                return
    while True:
        try:
            # clear any writes from previous attempts
            task.writes.clear()
            # run the task
            if stream:
                async for _ in task.proc.astream(task.input, config):
                    pass
                # if successful, end
                break
            else:
                return await task.proc.ainvoke(task.input, config)
        except ParentCommand as exc:
            # handle parent command
            ns: str = config[CONF][CONFIG_KEY_CHECKPOINT_NS]
            cmd = exc.args[0]
            if cmd.graph in (ns, recast_checkpoint_ns(ns), task.name):
                for w in task.writers:
                    w.invoke(cmd, config)
                break
            elif cmd.graph == Command.PARENT:
                parts = ns.split(NS_SEP)
                if parts[-1].isdigit():
                    parts.pop()
                parent_ns = NS_SEP.join(parts[:-1])
                exc.args = (replace(cmd, graph=parent_ns),)
            raise
        except GraphBubbleUp:
            raise
        except Exception as exc:
            if SUPPORTS_EXC_NOTES:
                exc.add_note(f"During task with name '{task.name}' and id '{task.id}'")
            if not retry_policy:
                raise

            # Check which retry policy applies to this exception
            matching_policy = None
            for policy in retry_policy:
                if _should_retry_on(policy, exc):
                    matching_policy = policy
                    break

            if not matching_policy:
                raise

            # increment attempts
            attempts += 1
            # check if we should give up
            if attempts >= matching_policy.max_attempts:
                raise
            # sleep before retrying
            interval = matching_policy.initial_interval
            # Apply backoff factor based on attempt count
            interval = min(
                matching_policy.max_interval,
                interval * (matching_policy.backoff_factor ** (attempts - 1)),
            )

            # Apply jitter if configured
            sleep_time = (
                interval + random.uniform(0, 1) if matching_policy.jitter else interval
            )
            await asyncio.sleep(sleep_time)

            # log the retry
            logger.info(
                f"Retrying task {task.name} after {sleep_time:.2f} seconds (attempt {attempts}) after {exc.__class__.__name__} {exc}",
                exc_info=exc,
            )
            # signal subgraphs to resume (if available)
            config = patch_configurable(config, {CONFIG_KEY_RESUMING: True})
```

**关键点**：
1. 支持流式和非流式执行
2. 缓存检查机制
3. 异步睡眠（`asyncio.sleep`）
4. 与同步版本逻辑一致

### 4. 重试条件判断

```python
# _retry.py:204-218
def _should_retry_on(retry_policy: RetryPolicy, exc: Exception) -> bool:
    """Check if the given exception should be retried based on the retry policy."""
    if isinstance(retry_policy.retry_on, Sequence):
        return isinstance(exc, tuple(retry_policy.retry_on))
    elif isinstance(retry_policy.retry_on, type) and issubclass(
        retry_policy.retry_on, Exception
    ):
        return isinstance(exc, retry_policy.retry_on)
    elif callable(retry_policy.retry_on):
        return retry_policy.retry_on(exc)  # type: ignore[call-arg]
    else:
        raise TypeError(
            "retry_on must be an Exception class, a list or tuple of Exception classes, or a callable"
        )
```

**支持的重试条件**：
1. 单个异常类型
2. 多个异常类型（Sequence）
3. 自定义判断函数（Callable）

## 代码片段

### RetryPolicy 配置示例

```python
from langgraph.types import RetryPolicy

# 基础重试策略
retry_policy = RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,
    backoff_factor=2.0,
    max_interval=10.0,
    jitter=True,
    retry_on=Exception
)

# 自定义重试条件
def retry_on_rate_limit(error: Exception) -> bool:
    return "rate limit" in str(error).lower()

retry_policy = RetryPolicy(
    max_attempts=5,
    initial_interval=0.5,
    backoff_factor=2.0,
    max_interval=10.0,
    jitter=True,
    retry_on=retry_on_rate_limit
)

# 多个异常类型
retry_policy = RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,
    backoff_factor=2.0,
    max_interval=10.0,
    jitter=False,
    retry_on=[ValueError, TypeError, ConnectionError]
)
```

### 节点添加时配置重试

```python
builder.add_node(
    "api_call",
    unreliable_api_call,
    retry_policy=RetryPolicy(
        initial_interval=0.5,
        backoff_factor=2.0,
        max_interval=10.0,
        max_attempts=5,
        jitter=True,
        retry_on=retry_on_rate_limit
    )
)
```
