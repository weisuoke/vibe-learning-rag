---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/langgraph/graph/state.py
  - libs/langgraph/tests/test_retry.py
analyzed_at: 2026-03-07
knowledge_point: 08_错误处理与重试
---

# 源码分析：RetryPolicy 集成方式

## 分析的文件
- `libs/langgraph/langgraph/graph/state.py` - StateGraph 中的重试策略集成
- `libs/langgraph/tests/test_retry.py` - 重试测试用例

## 关键发现

### add_node() 的 retry_policy 参数

```python
def add_node(
    self,
    node: str,
    action: StateNode,
    *,
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    ...
) -> Self:
```

支持三种传入方式：
1. `None`（默认）- 不使用重试策略
2. 单个 `RetryPolicy` 对象
3. `Sequence[RetryPolicy]` - 多个策略，第一个匹配的生效

### 测试用例分析

#### 1. 基本重试测试
```python
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
    .add_edge(START, "failing_node")
    .compile()
)
```

#### 2. 多策略测试
```python
value_error_policy = RetryPolicy(
    max_attempts=2, initial_interval=0.01,
    jitter=False, retry_on=ValueError,
)
key_error_policy = RetryPolicy(
    max_attempts=3, initial_interval=0.02,
    jitter=False, retry_on=KeyError,
)
graph = (
    StateGraph(State)
    .add_node("failing_node", failing_node,
              retry_policy=(value_error_policy, key_error_policy))
    .add_edge(START, "failing_node")
    .compile()
)
```

#### 3. 超过最大重试次数测试
验证当 `max_attempts` 耗尽时，异常会正常抛出。

#### 4. Jitter 测试
验证 `jitter=True` 时会调用 `random.uniform(0, 1)` 添加随机延迟。

### 函数式 API 集成

```python
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy

retry_policy = RetryPolicy(retry_on=ValueError)

@task(retry_policy=retry_policy)
def get_info():
    # ...
    pass
```

### 关键发现

1. **`retry` 参数已弃用**：旧的 `retry` 参数发出弃用警告，建议使用 `retry_policy`
2. **链式调用**：`add_node()` 返回 `Self`，支持方法链式调用
3. **策略传递链**：`add_node → StateNodeSpec → compile → PregelNode → run_with_retry`
4. **退避间隔验证**：测试确认 `[0.01, 0.02]` 的退避序列（factor=2.0）
