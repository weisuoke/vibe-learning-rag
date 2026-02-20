# 核心概念 01：RunnableConfig 结构

> **深入理解 RunnableConfig 的 8 个核心字段及其配置合并机制**

---

## 概述

`RunnableConfig` 是 LangChain 的运行时配置字典，定义了链执行时的行为。它包含 8 个核心字段，每个字段都有特定的用途和类型要求。

**引用**：
> "RunnableConfig provides a standardized way to pass runtime configuration to any Runnable, enabling observability, dynamic behavior, and production-grade control."
> — [LangChain RunnableConfig API](https://reference.langchain.com/v0.3/python/core/runnables/langchain_core.runnables.config.RunnableConfig.html)

---

## 字段 1：callbacks

### 定义

```python
callbacks: Optional[List[BaseCallbackHandler]] = None
```

### 用途

**监控和日志系统**，在链执行的不同阶段触发回调函数。

### 使用场景

1. **成本追踪**：记录每次 LLM 调用的 token 消耗
2. **性能监控**：追踪延迟和吞吐量
3. **错误告警**：LLM 调用失败时发送通知
4. **日志记录**：结构化日志输出

### 代码示例

```python
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

class CostTracker(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        tokens = response.llm_output["token_usage"]["total_tokens"]
        print(f"消耗 {tokens} tokens")

class Logger(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"开始调用: {prompts}")

# 使用多个回调
config = RunnableConfig(
    callbacks=[CostTracker(), Logger()]
)

llm.invoke("你好", config=config)
```

### 注意事项

- **执行顺序**：按列表顺序执行
- **异常处理**：回调异常不会中断主流程
- **性能影响**：开销 < 0.5%，可忽略不计

---

## 字段 2：tags

### 定义

```python
tags: Optional[List[str]] = None
```

### 用途

**过滤和分类**，用于决定哪些回调应该执行，以及在 LangSmith 中过滤追踪。

### 使用场景

1. **环境区分**：`["production"]` vs `["development"]`
2. **优先级标记**：`["critical"]`, `["normal"]`, `["low"]`
3. **功能分类**：`["rag"]`, `["chat"]`, `["summarization"]`
4. **回调过滤**：根据 tags 决定回调是否执行

### 代码示例

```python
# 1. 基础使用
config = RunnableConfig(
    tags=["production", "critical", "rag"]
)

# 2. 回调根据 tags 过滤
class ProductionOnlyCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        tags = kwargs.get("tags", [])
        if "production" not in tags:
            return  # 非生产环境不执行

        # 生产环境的监控逻辑
        send_to_monitoring(prompts)

# 3. LangSmith 中按 tags 过滤
# 在 LangSmith 仪表盘中可以按 tags 筛选追踪
```

### 注意事项

- **简单字符串**：不要在 tags 中编码数据（如 `"user_id:123"`）
- **用于过滤**：tags 用于分类，metadata 用于数据
- **可组合**：可以同时使用多个 tags

---

## 字段 3：metadata

### 定义

```python
metadata: Optional[Dict[str, Any]] = None
```

### 用途

**追踪和归因**，存储可序列化的上下文信息，用于成本归因、问题定位、数据分析。

### 使用场景

1. **用户归因**：`{"user_id": "123", "session": "abc"}`
2. **请求追踪**：`{"request_id": "req_xyz", "trace_id": "trace_123"}`
3. **业务上下文**：`{"department": "sales", "region": "us-west"}`
4. **A/B 测试**：`{"experiment": "model_comparison", "variant": "A"}`

### 代码示例

```python
# 1. 基础使用
config = RunnableConfig(
    metadata={
        "user_id": "user_123",
        "session": "session_abc",
        "request_id": "req_xyz",
        "env": "production"
    }
)

# 2. 在回调中读取 metadata
class UserCostTracker(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        metadata = kwargs.get("metadata", {})
        user_id = metadata.get("user_id", "unknown")

        # 按用户归因成本
        cost = calculate_cost(response)
        record_user_cost(user_id, cost)

# 3. LangSmith 自动记录 metadata
# metadata 会自动显示在 LangSmith 追踪详情中
```

### 注意事项

- **必须可序列化**：只能使用 JSON 可序列化的类型
- **允许的类型**：`str`, `int`, `float`, `bool`, `None`, `list`, `dict`
- **禁止的类型**：对象实例、函数、datetime、自定义类

**错误示例**：

```python
# ❌ 错误：不可序列化
config = RunnableConfig(
    metadata={
        "user": User(id=123),  # 对象实例
        "timestamp": datetime.now(),  # datetime 对象
        "callback": lambda x: x  # 函数
    }
)

# ✅ 正确：可序列化
config = RunnableConfig(
    metadata={
        "user_id": 123,
        "user_name": "Alice",
        "timestamp": "2025-01-15T10:30:00Z"  # ISO 8601 字符串
    }
)
```

---

## 字段 4：run_name

### 定义

```python
run_name: Optional[str] = None
```

### 用途

**可读标识**，在追踪系统（如 LangSmith）中显示的运行名称，便于识别和搜索。

### 使用场景

1. **用户查询标识**：`"user_123_query_456"`
2. **功能标识**：`"rag_document_search"`
3. **批处理标识**：`"batch_process_2025_01_15"`

### 代码示例

```python
# 1. 基础使用
config = RunnableConfig(
    run_name="user_query_123"
)

# 2. 动态生成 run_name
import uuid

def create_config(user_id: str, query_type: str):
    run_id = str(uuid.uuid4())[:8]
    return RunnableConfig(
        run_name=f"{user_id}_{query_type}_{run_id}",
        metadata={"user_id": user_id, "query_type": query_type}
    )

config = create_config("user_123", "rag_search")
# run_name: "user_123_rag_search_a1b2c3d4"
```

### 注意事项

- **可读性优先**：使用有意义的名称，避免随机字符串
- **唯一性**：建议包含时间戳或 UUID 确保唯一
- **长度限制**：建议 < 100 字符

---

## 字段 5：run_id

### 定义

```python
run_id: Optional[UUID] = None
```

### 用途

**唯一标识符**，自动生成的 UUID，用于追踪系统中的唯一标识。

### 使用场景

1. **分布式追踪**：跨服务追踪同一个请求
2. **日志关联**：关联不同系统的日志
3. **调试定位**：根据 run_id 查找特定调用

### 代码示例

```python
import uuid
from langchain_core.runnables import RunnableConfig

# 1. 自动生成（推荐）
config = RunnableConfig()
# run_id 会自动生成

# 2. 手动指定（用于分布式追踪）
trace_id = uuid.uuid4()
config = RunnableConfig(
    run_id=trace_id,
    metadata={"trace_id": str(trace_id)}
)

# 3. 在回调中读取 run_id
class TraceLogger(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        run_id = kwargs.get("run_id")
        print(f"Run ID: {run_id}")
```

### 注意事项

- **自动生成**：通常不需要手动指定
- **分布式追踪**：跨服务时可以手动传递
- **不可变**：一旦生成不应修改

---

## 字段 6：max_concurrency

### 定义

```python
max_concurrency: Optional[int] = None
```

### 用途

**并发控制**，限制批量调用时的最大并发数，避免 API 限流和资源耗尽。

### 使用场景

1. **批量处理**：处理大量请求时限制并发
2. **API 限流保护**：避免触发 API 速率限制
3. **资源控制**：控制内存和网络连接数

### 代码示例

```python
from langchain_core.runnables import RunnableConfig

# 1. 基础使用
config = RunnableConfig(max_concurrency=10)

# 批量调用时自动限流
inputs = [{"input": f"问题{i}"} for i in range(100)]
results = chain.batch(inputs, config=config)
# 最多同时执行 10 个，其余排队等待

# 2. 根据 API 限制计算
# OpenAI Tier 1: 3,500 RPM (每分钟请求数)
# 假设每次调用 2 秒，则最大并发 = 3500 / 30 ≈ 116
config = RunnableConfig(max_concurrency=100)

# 3. 动态调整
class AdaptiveConcurrency:
    def __init__(self):
        self.concurrency = 50

    def get_config(self):
        return RunnableConfig(max_concurrency=self.concurrency)

    def on_rate_limit(self):
        self.concurrency = max(10, self.concurrency // 2)
        print(f"降低并发数到 {self.concurrency}")
```

### 注意事项

- **不是越大越好**：过高会触发 API 限流
- **根据 API 限制设置**：参考 API 文档的 RPM 限制
- **内存考虑**：每个并发请求占用内存

**API 限制参考**：

| API | Tier | RPM | 推荐并发 |
|-----|------|-----|----------|
| OpenAI GPT-4 | Tier 1 | 500 | 10-20 |
| OpenAI GPT-3.5 | Tier 1 | 3,500 | 50-100 |
| Anthropic Claude | Tier 1 | 1,000 | 20-30 |

---

## 字段 7：recursion_limit

### 定义

```python
recursion_limit: Optional[int] = 25
```

### 用途

**递归限制**，防止 Agent 陷入无限循环，保护成本不失控。

### 使用场景

1. **Agent 执行**：限制 Agent 的最大决策次数
2. **工具调用链**：限制工具调用的嵌套深度
3. **成本保护**：避免因无限循环导致成本失控

### 代码示例

```python
from langchain_core.runnables import RunnableConfig

# 1. 基础使用
config = RunnableConfig(recursion_limit=10)

# Agent 最多执行 10 次决策
agent.invoke({"input": "查询天气"}, config=config)

# 2. 真实案例：Agent 陷入循环
# Agent: 调用 search_tool("天气")
# Tool: 返回错误（API 限流）
# Agent: 重新调用 search_tool("天气")
# Tool: 返回错误（API 限流）
# ...  # 无限循环

# 使用 recursion_limit 强制终止
config = RunnableConfig(recursion_limit=5)
try:
    agent.invoke({"input": "查询天气"}, config=config)
except RecursionError:
    print("Agent 超过递归限制，已终止")

# 3. 监控递归深度
class RecursionMonitor(BaseCallbackHandler):
    def __init__(self):
        self.depth = 0

    def on_chain_start(self, serialized, inputs, **kwargs):
        self.depth += 1
        if self.depth > 20:
            alert("递归深度过高", depth=self.depth)

    def on_chain_end(self, outputs, **kwargs):
        self.depth -= 1
```

### 注意事项

- **默认值 25**：足够处理复杂任务
- **不是性能优化**：是成本保护机制
- **Agent 必备**：使用 Agent 时务必设置

---

## 字段 8：configurable

### 定义

```python
configurable: Optional[Dict[str, Any]] = None
```

### 用途

**运行时可配置字段**，存储通过 `configurable_fields()` 声明的动态参数。

### 使用场景

1. **动态模型切换**：运行时切换 GPT-3.5 / GPT-4
2. **参数调整**：动态调整 temperature、max_tokens
3. **A/B 测试**：对比不同配置的效果
4. **降级策略**：主模型失败时切换到备用模型

### 代码示例

```python
from langchain_openai import ChatOpenAI
from langchain_core.runnables import ConfigurableField

# 1. 声明可配置字段
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
).configurable_fields(
    model=ConfigurableField(id="model_name"),
    temperature=ConfigurableField(id="temp")
)

# 2. 运行时动态调整
# 使用 GPT-3.5（便宜）
config1 = RunnableConfig(
    configurable={"model_name": "gpt-3.5-turbo", "temp": 0.7}
)
llm.invoke("你好", config=config1)

# 使用 GPT-4（质量高）
config2 = RunnableConfig(
    configurable={"model_name": "gpt-4", "temp": 0.3}
)
llm.invoke("你好", config=config2)

# 3. A/B 测试
def ab_test(query: str):
    configs = [
        RunnableConfig(configurable={"temp": 0.3}, tags=["variant_A"]),
        RunnableConfig(configurable={"temp": 0.9}, tags=["variant_B"])
    ]

    results = []
    for config in configs:
        result = llm.invoke(query, config=config)
        results.append(result)

    return results
```

### 注意事项

- **必须先声明**：使用前必须通过 `configurable_fields()` 声明
- **类型安全**：LangChain 会验证参数类型
- **优先级最高**：覆盖构造时和 with_config 的配置

---

## 配置合并机制

### 优先级规则

```
invoke 时配置 > with_config 配置 > 构造时配置
```

### 合并示例

```python
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

# 1. 构造时配置（优先级 3）
llm = ChatOpenAI(
    callbacks=[callback1],
    tags=["default"],
    metadata={"env": "dev"}
)

# 2. with_config 配置（优先级 2）
llm_with_config = llm.with_config(
    callbacks=[callback2],
    tags=["production"],
    metadata={"env": "prod", "version": "1.0"}
)

# 3. invoke 时配置（优先级 1，最高）
llm_with_config.invoke(
    "你好",
    config=RunnableConfig(
        callbacks=[callback3],
        tags=["critical"],
        metadata={"user_id": "123"}
    )
)

# 最终配置：
# callbacks: [callback3]  # 完全覆盖
# tags: ["critical"]      # 完全覆盖
# metadata: {"user_id": "123"}  # 完全覆盖（不合并）
```

### 合并规则

| 字段 | 合并方式 |
|------|----------|
| `callbacks` | **完全覆盖**（不合并） |
| `tags` | **完全覆盖**（不合并） |
| `metadata` | **完全覆盖**（不合并） |
| `run_name` | **完全覆盖** |
| `max_concurrency` | **完全覆盖** |
| `recursion_limit` | **完全覆盖** |
| `configurable` | **完全覆盖** |

**注意**：所有字段都是完全覆盖，不会合并。

---

## 完整示例

```python
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler
import uuid

# 1. 定义回调
class ProductionMonitor(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        metadata = kwargs.get("metadata", {})
        user_id = metadata.get("user_id")
        tokens = response.llm_output["token_usage"]["total_tokens"]
        print(f"用户 {user_id} 消耗 {tokens} tokens")

# 2. 创建完整配置
def create_production_config(user_id: str, query_type: str):
    return RunnableConfig(
        # 监控
        callbacks=[ProductionMonitor()],

        # 分类
        tags=["production", "critical", query_type],

        # 追踪
        metadata={
            "user_id": user_id,
            "query_type": query_type,
            "env": "production",
            "timestamp": "2025-01-15T10:30:00Z"
        },

        # 标识
        run_name=f"{user_id}_{query_type}_{uuid.uuid4().hex[:8]}",

        # 控制
        max_concurrency=10,
        recursion_limit=25,

        # 动态配置
        configurable={"temp": 0.7, "model": "gpt-4"}
    )

# 3. 使用
llm = ChatOpenAI().configurable_fields(
    model=ConfigurableField(id="model"),
    temperature=ConfigurableField(id="temp")
)

config = create_production_config("user_123", "rag_search")
result = llm.invoke("什么是量子纠缠？", config=config)
```

---

## 总结

### 8 个字段速查

| 字段 | 类型 | 用途 | 默认值 |
|------|------|------|--------|
| `callbacks` | `List[BaseCallbackHandler]` | 监控日志 | `None` |
| `tags` | `List[str]` | 过滤分类 | `None` |
| `metadata` | `Dict[str, Any]` | 追踪归因 | `None` |
| `run_name` | `str` | 可读标识 | `None` |
| `run_id` | `UUID` | 唯一标识 | 自动生成 |
| `max_concurrency` | `int` | 并发控制 | `None` |
| `recursion_limit` | `int` | 递归限制 | `25` |
| `configurable` | `Dict[str, Any]` | 动态配置 | `None` |

### 关键要点

1. **配置优先级**：invoke 时 > with_config > 构造时
2. **合并方式**：完全覆盖，不合并
3. **序列化要求**：metadata 必须 JSON 可序列化
4. **性能影响**：callbacks 开销 < 0.5%
5. **安全机制**：recursion_limit 防止成本失控

---

## 参考资料

- [RunnableConfig API Reference](https://reference.langchain.com/v0.3/python/core/runnables/langchain_core.runnables.config.RunnableConfig.html)
- [LangChain Configuration Guide](https://python.langchain.com/docs/how_to/configure/)
- [LangSmith Observability Platform](https://www.langchain.com/langsmith/observability)
