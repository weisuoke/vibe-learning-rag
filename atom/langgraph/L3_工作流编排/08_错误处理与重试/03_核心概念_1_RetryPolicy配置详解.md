# RetryPolicy 配置详解

> **核心概念 1/6** | 预计阅读：8分钟
> **来源**：sourcecode/langgraph/libs/langgraph/langgraph/types.py | _retry.py

---

## 什么是 RetryPolicy？

**RetryPolicy 是 LangGraph 提供的声明式重试配置，用一个 NamedTuple 描述"失败后怎么重试"的全部策略。**

它就像一份"重试说明书"，告诉 LangGraph 运行时：
- 等多久再重试？
- 最多重试几次？
- 哪些错误值得重试？
- 每次等待时间怎么增长？

---

## RetryPolicy 的 6 个参数

### 源码定义

```python
# 来自 langgraph/types.py
class RetryPolicy(NamedTuple):
    """节点的重试策略配置。"""
    initial_interval: float = 0.5
    backoff_factor: float = 2.0
    max_interval: float = 128.0
    max_attempts: int = 3
    jitter: bool = True
    retry_on: (
        type[Exception]
        | Sequence[type[Exception]]
        | Callable[[Exception], bool]
    ) = default_retry_on
```

### 参数逐一详解

---

### 参数1：`initial_interval`（首次重试间隔）

**默认值：** `0.5`（秒）

**含义：** 第一次重试前等待的时间。

```python
# 首次重试等 0.5 秒
RetryPolicy(initial_interval=0.5)

# LLM API 调用建议更长，给服务器恢复时间
RetryPolicy(initial_interval=1.0)

# 测试环境用极短间隔加速测试
RetryPolicy(initial_interval=0.01)
```

**实际影响：**
```
失败 → 等 0.5 秒 → 第1次重试
失败 → 等 1.0 秒 → 第2次重试（0.5 × 2.0）
失败 → 等 2.0 秒 → 第3次重试（0.5 × 2.0²）
```

**在 LangGraph 工作流中的应用：**
- 调用 LLM API 时：`initial_interval=1.0`（API 恢复需要时间）
- 向量数据库查询：`initial_interval=0.5`（通常很快恢复）
- 外部搜索引擎：`initial_interval=2.0`（第三方服务可能较慢）

---

### 参数2：`backoff_factor`（退避倍增因子）

**默认值：** `2.0`

**含义：** 每次重试间隔是上次的多少倍。

```python
# 默认指数退避：0.5 → 1.0 → 2.0 → 4.0 → ...
RetryPolicy(backoff_factor=2.0)

# 更激进的退避：0.5 → 1.5 → 4.5 → 13.5 → ...
RetryPolicy(backoff_factor=3.0)

# 固定间隔（不退避）：0.5 → 0.5 → 0.5 → ...
RetryPolicy(backoff_factor=1.0)
```

**退避计算公式（源码）：**

```python
# 来自 pregel/_retry.py
interval = min(
    retry_policy.max_interval,
    retry_policy.initial_interval
    * retry_policy.backoff_factor ** (attempt - 1),
)
```

**可视化不同 backoff_factor 的效果：**

```
factor=2.0: |0.5s|--1.0s--|----2.0s----|--------4.0s--------|
factor=3.0: |0.5s|---1.5s---|--------4.5s--------|
factor=1.0: |0.5s|0.5s|0.5s|0.5s|  （固定间隔，不推荐）
```

**为什么默认是 2.0？**
- 太小（如 1.0）：间隔不增长，可能持续给服务器压力
- 太大（如 5.0）：间隔增长太快，用户等待时间过长
- 2.0 是工业界公认的平衡点

---

### 参数3：`max_interval`（最大重试间隔）

**默认值：** `128.0`（秒）

**含义：** 退避增长的天花板，防止等待时间无限增大。

```python
# 默认最多等 128 秒
RetryPolicy(max_interval=128.0)

# 生产环境可能设更低
RetryPolicy(max_interval=30.0)

# API 密集场景
RetryPolicy(max_interval=60.0)
```

**为什么需要上限？**

```
没有上限时（backoff_factor=2.0, initial_interval=0.5）：
第1次: 0.5s
第7次: 32s
第8次: 64s
第9次: 128s
第10次: 256s  ← 等4分钟！用户早就走了
第11次: 512s  ← 等8.5分钟！不可接受

有上限 max_interval=128 时：
第9次之后都是 128s ← 有上限保护
```

---

### 参数4：`max_attempts`（最大尝试次数）

**默认值：** `3`

**含义：** 包含首次执行在内的总尝试次数。**注意：不是重试次数！**

```python
# 默认：1次执行 + 2次重试 = 3次尝试
RetryPolicy(max_attempts=3)

# LLM API 调用建议更多次（速率限制可能需要更多重试）
RetryPolicy(max_attempts=5)

# 关键任务：尽可能多尝试
RetryPolicy(max_attempts=10)
```

**⚠️ 常见误区：max_attempts ≠ 重试次数**

```
max_attempts=3 意味着：
  尝试1（首次执行）→ 失败
  尝试2（第1次重试）→ 失败
  尝试3（第2次重试）→ 失败 → 抛出异常

实际重试了 2 次，不是 3 次！
```

**源码验证：**

```python
# 来自 pregel/_retry.py 的 run_with_retry()
while True:
    try:
        task.proc.invoke(task.input, config)
        break  # 成功则跳出
    except Exception as exc:
        if attempt >= retry_policy.max_attempts:
            raise  # 超过最大尝试次数，抛出异常
        # ... 退避等待后重试
```

---

### 参数5：`jitter`（随机抖动）

**默认值：** `True`

**含义：** 在退避间隔上添加 0-1 秒的随机时间。

```python
# 开启抖动（推荐，默认）
RetryPolicy(jitter=True)

# 关闭抖动（仅测试用，需要可预测的行为）
RetryPolicy(jitter=False)
```

**源码实现：**

```python
# 来自 pregel/_retry.py
if retry_policy.jitter:
    interval += random.uniform(0, 1)  # 添加 0~1 秒随机值
```

**为什么需要抖动？防止"雷群效应"！**

```
场景：100 个客户端同时遇到服务器故障

❌ 无抖动：
  时间 0.0s: 100个客户端同时请求 → 服务器崩溃
  时间 0.5s: 100个客户端同时重试 → 服务器再次崩溃
  时间 1.0s: 100个客户端同时重试 → 服务器继续崩溃
  → 永远无法恢复！

✅ 有抖动：
  时间 0.5-1.5s: 客户端分散重试
  时间 1.0-2.0s: 客户端分散重试
  → 服务器能逐个处理，逐步恢复
```

**日常生活类比：**
- ❌ 停电后所有人同时开空调 → 电网再次过载
- ✅ 错开时间逐个开 → 电网平稳恢复

---

### 参数6：`retry_on`（异常过滤器）

**默认值：** `default_retry_on`（智能判断函数）

**含义：** 决定哪些异常值得重试。支持三种形式。

```python
# 形式1：单一异常类型
RetryPolicy(retry_on=ConnectionError)

# 形式2：异常类型序列
RetryPolicy(retry_on=[ConnectionError, TimeoutError])

# 形式3：自定义判断函数
def should_retry(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in [429, 500, 502, 503]
    return False

RetryPolicy(retry_on=should_retry)

# 默认：使用内置智能判断
RetryPolicy()  # retry_on=default_retry_on
```

**详细内容见：** `03_核心概念_2_异常匹配策略.md`

---

## 完整配置示例

### 场景1：LLM API 调用

```python
from langgraph.types import RetryPolicy
from langgraph.graph import StateGraph, START, END

# LLM 调用：需要较长间隔和更多重试次数
llm_retry = RetryPolicy(
    initial_interval=1.0,    # API 恢复需要时间
    backoff_factor=2.0,      # 标准指数退避
    max_interval=60.0,       # 最多等 1 分钟
    max_attempts=5,          # 多试几次（429 错误常见）
    jitter=True,             # 防止多用户同时重试
)

builder = StateGraph(State)
builder.add_node("call_llm", call_llm, retry_policy=llm_retry)
```

### 场景2：向量数据库查询

```python
# 数据库查询：恢复通常很快，不需要太长间隔
db_retry = RetryPolicy(
    initial_interval=0.3,    # 数据库恢复快
    backoff_factor=2.0,
    max_interval=10.0,       # 数据库超过 10 秒不响应就有问题了
    max_attempts=3,
    jitter=True,
    retry_on=ConnectionError,  # 只重试连接错误
)

builder.add_node("search_vectors", search_vectors, retry_policy=db_retry)
```

### 场景3：使用默认配置

```python
# 默认配置适用于大多数场景
builder.add_node(
    "my_node",
    my_function,
    retry_policy=RetryPolicy(),  # 全部使用默认值
)
```

### 场景4：函数式 API

```python
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy

@task(retry_policy=RetryPolicy(max_attempts=5))
def fetch_data(query: str):
    """使用 @task 装饰器配置重试"""
    # ... 可能失败的操作
    return result
```

---

## 多策略配置

LangGraph 支持为同一个节点配置**多个重试策略**，按顺序匹配：

```python
from langgraph.types import RetryPolicy

# 策略1：针对 ValueError 重试 2 次
value_error_policy = RetryPolicy(
    max_attempts=2,
    initial_interval=0.01,
    jitter=False,
    retry_on=ValueError,
)

# 策略2：针对 KeyError 重试 3 次
key_error_policy = RetryPolicy(
    max_attempts=3,
    initial_interval=0.02,
    jitter=False,
    retry_on=KeyError,
)

# 传入策略序列 → 第一个匹配的生效
builder.add_node(
    "my_node",
    my_function,
    retry_policy=(value_error_policy, key_error_policy),
)
```

**匹配逻辑（源码）：**

```python
# 来自 pregel/_retry.py
for retry_policy in retry_policies:
    if _should_retry_on(retry_policy, exc):
        # 使用这个策略的退避参数
        break
else:
    raise exc  # 没有策略匹配，直接抛出
```

---

## 参数速查表

| 参数 | 默认值 | 含义 | 推荐范围 |
|------|--------|------|----------|
| `initial_interval` | `0.5` | 首次重试等待（秒） | 0.1 - 5.0 |
| `backoff_factor` | `2.0` | 间隔倍增因子 | 1.5 - 3.0 |
| `max_interval` | `128.0` | 最大间隔上限（秒） | 10.0 - 300.0 |
| `max_attempts` | `3` | 总尝试次数（含首次） | 2 - 10 |
| `jitter` | `True` | 是否随机抖动 | 生产 True，测试 False |
| `retry_on` | `default_retry_on` | 异常过滤器 | 自定义或默认 |

---

## 关键设计决策

### 为什么选择 NamedTuple？

```python
# NamedTuple 的优势：
# 1. 不可变 → 线程安全，多节点共享无风险
# 2. 有默认值 → RetryPolicy() 直接可用
# 3. 轻量级 → 没有额外开销
# 4. 可解构 → 方便传参和序列化

policy = RetryPolicy()
print(policy.max_attempts)  # 3
print(policy[3])            # 3（也支持索引访问）
```

### 为什么 max_attempts 包含首次执行？

```
设计哲学："你最多可以尝试 N 次"
而不是："你失败后最多可以再试 N 次"

这更符合用户的直觉预期：
max_attempts=1 → 只执行一次，不重试
max_attempts=3 → 最多执行 3 次
```

---

[来源: sourcecode/langgraph/libs/langgraph/langgraph/types.py | sourcecode/langgraph/libs/langgraph/langgraph/pregel/_retry.py | Context7 官方文档]
