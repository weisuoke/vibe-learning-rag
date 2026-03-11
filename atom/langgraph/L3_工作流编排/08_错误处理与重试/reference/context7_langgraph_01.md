---
type: context7_documentation
library: langgraph
version: latest
fetched_at: 2026-03-07
knowledge_point: 08_错误处理与重试
context7_query: error handling retry policy RetryPolicy fallback
---

# Context7 文档：LangGraph 错误处理与重试

## 文档来源
- 库名称：LangGraph
- 版本：latest
- Context7 Library ID: /websites/langchain_oss_python_langgraph

## 关键信息提取

### 1. 添加默认重试策略

来源: https://docs.langchain.com/oss/python/langgraph/use-graph-api

默认 RetryPolicy 会自动重试大多数异常，排除 ValueError、TypeError 等编程错误，
并对 HTTP 5xx 错误进行重试。

```python
from langgraph.types import RetryPolicy

builder.add_node(
    "node_name",
    node_function,
    retry_policy=RetryPolicy(),
)
```

### 2. 针对瞬态错误的重试策略

来源: https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph

设计用于自动重试因网络问题或速率限制等瞬态问题失败的操作。

```python
from langgraph.types import RetryPolicy

workflow.add_node(
    "search_documentation",
    search_documentation,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0)
)
```

### 3. 函数式 API 中的重试

来源: https://docs.langchain.com/oss/python/langgraph/use-functional-api

使用 `@task` 装饰器配合 `retry_policy` 参数。

```python
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy

retry_policy = RetryPolicy(retry_on=ValueError)

@task(retry_policy=retry_policy)
def get_info():
    global attempts
    attempts += 1
    if attempts < 2:
        raise ValueError('Failure')
    return "OK"
```

### 4. 自定义差异化重试策略

来源: https://docs.langchain.com/oss/python/langgraph/use-graph-api

不同节点可以有不同的重试策略：
- 数据库节点：重试 `sqlite3.OperationalError`
- 模型调用节点：设置 `max_attempts=5`

```python
import sqlite3
from langgraph.types import RetryPolicy

builder.add_node(
    "query_database", query_database,
    retry_policy=RetryPolicy(retry_on=sqlite3.OperationalError),
)
builder.add_node(
    "model", call_model,
    retry_policy=RetryPolicy(max_attempts=5),
)
```

### 5. 重试策略设计原则

- 默认 `retry_on` 函数对大多数异常进行重试
- 排除常见 Python 编程错误（不应重试的错误）
- 对 HTTP 请求，仅重试 5xx 状态码（服务端错误）
- 4xx 错误不重试（客户端错误，重试也不会成功）
