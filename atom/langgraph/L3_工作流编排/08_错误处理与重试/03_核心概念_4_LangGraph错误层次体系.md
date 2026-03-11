# LangGraph 错误层次体系

> **核心概念 4/6** | 预计阅读：7分钟
> **来源**：sourcecode/langgraph/libs/langgraph/langgraph/errors.py

---

## 什么是错误层次体系？

**错误层次体系 = LangGraph 为不同故障场景设计的异常分类树。**

不同的异常类型有不同的处理策略：有些会被重试，有些直接跳过重试，有些还承担着流程控制的职责。

---

## 完整的异常继承树

```
Exception
├── RecursionError
│   └── GraphRecursionError          # 图执行步数超限
├── InvalidUpdateError               # 无效的 Channel 更新
├── GraphBubbleUp                    # 图中断基类（⚠️ 不会被重试！）
│   ├── GraphInterrupt               # 子图中断信号
│   │   └── NodeInterrupt (deprecated) # 节点中断（已弃用）
│   └── ParentCommand                # 命令冒泡到父图
├── EmptyInputError                  # 空输入错误
└── TaskNotFound                     # 分布式模式中任务未找到
```

---

## 逐一详解每个异常

### 1. GraphRecursionError

**继承自：** `RecursionError`

**含义：** 图执行步数超过了 `recursion_limit`。

```python
class GraphRecursionError(RecursionError):
    """Raised when the graph has exhausted the maximum number of steps."""
    pass
```

**触发场景：**

```python
# 一个无限循环的图
graph = builder.compile()
# 默认 recursion_limit=25
graph.invoke(input)
# 如果图执行超过 25 步 → GraphRecursionError
```

**与重试的关系：** 继承自 `RecursionError`（属于 `RuntimeError`），**默认不会被重试**（default_retry_on 排除了 RuntimeError）。

**设计意图：** 防止图陷入无限循环，这是一个安全阀。

---

### 2. InvalidUpdateError

**继承自：** `Exception`

**含义：** 节点返回了无效的状态更新。

```python
class InvalidUpdateError(Exception):
    """Raised when an invalid update is applied to the graph state."""
    pass
```

**触发场景：**

```python
# 节点返回了与 State schema 不匹配的数据
def bad_node(state):
    return {"nonexistent_field": "value"}  # State 中没有这个字段
```

**与重试的关系：** 这是编程错误，重试也不会成功，**不应重试**。

**对应的 ErrorCode：**
- `INVALID_CONCURRENT_GRAPH_UPDATE` — 并发更新冲突
- `INVALID_GRAPH_NODE_RETURN_VALUE` — 节点返回值类型错误

---

### 3. GraphBubbleUp（核心设计）

**继承自：** `Exception`

**含义：** 图中断基类，用于流程控制，**不是真正的"错误"**。

```python
class GraphBubbleUp(Exception):
    """Raised when a signal should bubble up through the graph
    without being caught by retry logic."""
    pass
```

**⚠️ 关键特性：GraphBubbleUp 及其子类永远不会被重试！**

```python
# 来自 pregel/_retry.py 的 run_with_retry()
try:
    task.proc.invoke(task.input, config)
except GraphBubbleUp:
    raise  # ← 直接抛出！不进入重试逻辑！
except Exception as exc:
    # ... 重试逻辑
```

**设计哲学：** GraphBubbleUp 不是"失败"，是"信号"。就像前端的 `throw redirect()` 不是错误而是路由跳转。

---

### 4. GraphInterrupt

**继承自：** `GraphBubbleUp`

**含义：** 子图发出的中断信号，携带中断数据。

```python
class GraphInterrupt(GraphBubbleUp):
    """Raised when a subgraph is interrupted."""
    def __init__(self, interrupts: Sequence[Interrupt] = ()) -> None:
        super().__init__(interrupts)
```

**触发场景：**

```python
from langgraph.types import interrupt

def human_review_node(state):
    # 暂停执行，等待人类审批
    decision = interrupt("请审核以下内容")
    return {"approved": decision}
```

**与重试的关系：** 继承自 GraphBubbleUp，**永远不会被重试**。中断是有意为之的暂停，不是失败。

---

### 5. NodeInterrupt（已弃用）

**继承自：** `GraphInterrupt`

**状态：** ⚠️ **已弃用**，推荐使用 `interrupt()` 函数替代。

```python
class NodeInterrupt(GraphInterrupt):
    """Deprecated: Use `interrupt()` function instead."""
    pass
```

---

### 6. ParentCommand

**继承自：** `GraphBubbleUp`

**含义：** 子图向父图发送的命令信号。

```python
class ParentCommand(GraphBubbleUp):
    """Raised when a command should bubble up to the parent graph."""
    def __init__(self, command: Command) -> None:
        super().__init__(command)
```

**特殊处理逻辑（源码）：**

```python
# 来自 pregel/_retry.py
except ParentCommand as exc:
    cmd = exc.args[0]
    if cmd.graph == Command.PARENT:
        raise  # 冒泡到父图
    else:
        # 在当前图处理
        ...
```

**与重试的关系：** 继承自 GraphBubbleUp，**永远不会被重试**。这是跨图通信机制。

---

### 7. EmptyInputError

**继承自：** `Exception`

**含义：** 恢复执行时输入为空。

```python
class EmptyInputError(Exception):
    """Raised when empty input is provided to the graph
    on a resumption when input is required."""
    pass
```

**触发场景：** 图从检查点恢复时，需要输入但用户没有提供。

---

### 8. TaskNotFound

**继承自：** `Exception`

**含义：** 在分布式模式中找不到指定的任务。

```python
class TaskNotFound(Exception):
    """Raised when a task is not found."""
    pass
```

---

## ErrorCode 枚举

LangGraph 还定义了错误代码枚举，用于标准化错误识别：

```python
class ErrorCode(Enum):
    GRAPH_RECURSION_LIMIT = "GRAPH_RECURSION_LIMIT"
    INVALID_CONCURRENT_GRAPH_UPDATE = "INVALID_CONCURRENT_GRAPH_UPDATE"
    INVALID_GRAPH_NODE_RETURN_VALUE = "INVALID_GRAPH_NODE_RETURN_VALUE"
    MULTIPLE_SUBGRAPHS = "MULTIPLE_SUBGRAPHS"
    INVALID_CHAT_HISTORY = "INVALID_CHAT_HISTORY"
```

每个错误码都有对应的在线文档链接，方便开发者排查问题。

---

## 异常与重试的关系总结

```
异常类型                 是否重试？    原因
──────────────────────────────────────────────────
GraphBubbleUp           ❌ 永不重试   不是错误，是流程控制信号
├── GraphInterrupt      ❌ 永不重试   有意为之的暂停
├── NodeInterrupt       ❌ 永不重试   已弃用
└── ParentCommand       ❌ 永不重试   跨图通信

GraphRecursionError     ❌ 默认不重试  RuntimeError 子类
InvalidUpdateError      ❌ 默认不重试  编程错误
EmptyInputError         ✅ 可能重试    取决于 retry_on 配置
TaskNotFound            ✅ 可能重试    取决于 retry_on 配置

ConnectionError         ✅ 默认重试    瞬态网络错误
HTTP 5xx               ✅ 默认重试    服务端临时故障
ValueError/TypeError    ❌ 默认不重试  编程错误
其他未知异常            ✅ 默认重试    保守策略
```

---

## 异常处理的优先级

在 `run_with_retry()` 中，异常按以下优先级处理：

```python
try:
    task.proc.invoke(task.input, config)
except ParentCommand as exc:
    # 优先级1：命名空间路由判断
    # 根据 command.graph 决定冒泡还是本地处理
    ...
except GraphBubbleUp:
    # 优先级2：直接抛出（不重试）
    raise
except Exception as exc:
    # 优先级3：进入重试判断
    if _should_retry_on(retry_policy, exc):
        # 退避等待 → 重试
        ...
    else:
        raise  # 不匹配 → 直接抛出
```

---

## 设计决策分析

### 为什么用异常做流程控制？

```
LangGraph 的设计选择：用异常（GraphBubbleUp）实现流程控制

优势：
1. 跨层传播 — 异常自然穿透调用栈，不需要层层传递返回值
2. 不可忽视 — 异常必须被处理，不会被意外忽略
3. 与重试机制兼容 — 通过异常类型判断是否重试
4. 支持嵌套子图 — 子图的异常自动冒泡到父图

前端类比：
throw redirect("/login")  ← Next.js 也用异常做路由跳转
throw notFound()           ← Next.js 的 404 处理
```

### 为什么 GraphBubbleUp 不被重试？

```
GraphBubbleUp 代表"有意的中断"，不是"意外的失败"。

类比：
- 红灯停车 ← 交通信号（不是故障，不需要重试）
- 引擎熄火 ← 真正的故障（需要重试/修复）

GraphInterrupt（红灯）vs ConnectionError（引擎熄火）
```

---

## 实际应用建议

### 1. 在错误处理节点中区分异常类型

```python
from langgraph.errors import GraphRecursionError, InvalidUpdateError

def error_handler(state):
    """根据错误类型决定处理策略。"""
    error = state.get("last_error")

    if isinstance(error, GraphRecursionError):
        return {"action": "simplify_task"}  # 简化任务
    elif isinstance(error, InvalidUpdateError):
        return {"action": "log_and_alert"}  # 记录并告警
    else:
        return {"action": "retry_or_fallback"}  # 重试或降级
```

### 2. 自定义 retry_on 排除特定 LangGraph 异常

```python
from langgraph.errors import GraphRecursionError, InvalidUpdateError
from langgraph.types import RetryPolicy

def custom_retry_on(exc: Exception) -> bool:
    """排除 LangGraph 框架级错误。"""
    if isinstance(exc, (GraphRecursionError, InvalidUpdateError)):
        return False  # 框架错误不重试
    if isinstance(exc, ConnectionError):
        return True   # 网络错误重试
    return False

policy = RetryPolicy(retry_on=custom_retry_on)
```

---

[来源: sourcecode/langgraph/libs/langgraph/langgraph/errors.py | pregel/_retry.py]
