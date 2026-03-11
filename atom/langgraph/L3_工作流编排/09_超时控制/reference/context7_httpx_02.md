---
type: context7_documentation
library: httpx
version: latest
fetched_at: 2026-03-07
knowledge_point: 09_超时控制
context7_query: timeout configuration connect read write pool timeouts
---

# Context7 文档：HTTPX 超时配置

## 文档来源
- 库名称：HTTPX
- 版本：latest
- Context7 Library ID：`/encode/httpx`

## 关键信息提取

### 1. HTTPX 支持总 timeout 和细粒度 timeout

```python
import httpx

response = httpx.get("https://httpbin.org/delay/1", timeout=5.0)
```

也可以使用细粒度配置：

```python
timeout = httpx.Timeout(
    10.0,
    connect=5.0,
    read=30.0,
    write=10.0,
    pool=5.0,
)
```

### 2. 官方文档默认强调四类预算

- `connect`：建立连接预算
- `read`：读取响应预算
- `write`：发送请求预算
- `pool`：连接池获取预算

这和 LangGraph `step_timeout` 是完全不同的维度。

### 3. 可以对 client 设置默认 timeout，也可对单次请求覆写

```python
client = httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0))

r1 = client.get("https://httpbin.org/get")
r2 = client.get("https://httpbin.org/delay/5", timeout=60.0)
```

### 4. 对 LangGraph 节点设计的启示

如果节点内部包含 HTTP 调用，推荐预算层次是：

1. 节点内部 `httpx.Timeout`；
2. 必要时再用 `asyncio.wait_for()` 包一层；
3. 最外层用图的 `step_timeout` 做 orchestration 兜底。

## 结论

HTTPX 官方文档给出的超时模型非常适合 LangGraph 节点内部的 I/O deadline 设计。

在生产中，**不要只配 `step_timeout` 而不给 `httpx.Timeout`**，否则网络调用可能在节点内部拖很久，直到外层图调度才发现问题。

