---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/langgraph/pregel/_retry.py
  - libs/langgraph/langgraph/_internal/_retry.py
  - libs/langgraph/langgraph/types.py
analyzed_at: 2026-03-07
knowledge_point: 08_错误处理与重试
---

# 源码分析：LangGraph 重试机制

## 分析的文件
- `libs/langgraph/langgraph/pregel/_retry.py` - 重试执行核心逻辑
- `libs/langgraph/langgraph/_internal/_retry.py` - 默认重试判断函数
- `libs/langgraph/langgraph/types.py` - RetryPolicy 类型定义

## 关键发现

### RetryPolicy 定义（NamedTuple）

```python
class RetryPolicy(NamedTuple):
    initial_interval: float = 0.5      # 首次重试前等待秒数
    backoff_factor: float = 2.0        # 间隔倍增因子
    max_interval: float = 128.0        # 重试间隔上限（秒）
    max_attempts: int = 3              # 最大尝试次数（含首次）
    jitter: bool = True                # 是否添加随机抖动
    retry_on: (type[Exception]         # 触发重试的异常类型
        | Sequence[type[Exception]]    # 或异常类型列表
        | Callable[[Exception], bool]  # 或判断函数
    ) = default_retry_on               # 默认使用智能判断
```

### 核心函数：run_with_retry()

执行流程：
1. 获取任务的重试策略（任务级 > 全局级）
2. 进入 while True 循环
3. 清除上次尝试的写入（`task.writes.clear()`）
4. 执行任务
5. 异常处理：
   - `ParentCommand` → 命名空间判断 → 冒泡或处理
   - `GraphBubbleUp` → 直接抛出（不重试）
   - 其他异常 → 匹配重试策略 → 退避等待 → 重试

### 退避计算公式

```python
interval = min(
    max_interval,
    initial_interval * (backoff_factor ** (attempts - 1))
)
sleep_time = interval + random.uniform(0, 1) if jitter else interval
```

示例（默认参数）：
- 第1次重试：0.5s + jitter
- 第2次重试：1.0s + jitter
- 第3次重试：2.0s + jitter
- ...
- 上限：128.0s + jitter

### 异常匹配策略（_should_retry_on）

支持三种匹配方式：
1. **单一异常类型**：`isinstance(exc, retry_on)`
2. **异常类型序列**：`isinstance(exc, tuple(retry_on))`
3. **自定义函数**：`retry_on(exc)` 返回 bool

### 默认重试判断函数（default_retry_on）

```python
def default_retry_on(exc: Exception) -> bool:
    # ✅ 重试：网络连接错误
    if isinstance(exc, ConnectionError): return True
    # ✅ 重试：HTTP 5xx 服务端错误
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    if isinstance(exc, requests.HTTPError):
        return 500 <= exc.response.status_code < 600 if exc.response else True
    # ❌ 不重试：编程错误
    if isinstance(exc, (ValueError, TypeError, ArithmeticError,
                        ImportError, LookupError, NameError,
                        SyntaxError, RuntimeError, ReferenceError,
                        StopIteration, StopAsyncIteration, OSError)):
        return False
    # ✅ 重试：其他未知异常
    return True
```

### 关键设计决策

1. **任务级优先**：`task.retry_policy or retry_policy` 任务自带策略优先
2. **写入清除**：每次重试前清除上次写入，避免状态污染
3. **异常注释**：Python 3.11+ 在异常上添加任务名和 ID（`exc.add_note()`）
4. **子图恢复信号**：重试时设置 `CONFIG_KEY_RESUMING: True`
5. **多策略支持**：遍历策略列表，使用第一个匹配的策略
6. **异步支持**：`arun_with_retry()` 使用 `asyncio.sleep()` 替代 `time.sleep()`

## 重要的异常处理优先级

```
ParentCommand → 命名空间路由（不重试）
GraphBubbleUp → 直接抛出（不重试）
匹配策略的异常 → 重试（有次数限制）
不匹配策略的异常 → 直接抛出
```
