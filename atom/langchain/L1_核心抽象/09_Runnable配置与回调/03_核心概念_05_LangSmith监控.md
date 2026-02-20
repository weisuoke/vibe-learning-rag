# 核心概念 05：LangSmith 监控

> **LangSmith 平台的环境配置、自动追踪、成本监控和 2025-2026 新特性**

---

## 概述

LangSmith 是 LangChain 官方的可观测性平台，提供零代码侵入的自动追踪、可视化调试、成本分析和性能监控。

**核心特点**：
- **零配置**：只需设置环境变量
- **自动追踪**：无需手动传递 trace_id
- **可视化强**：内置仪表盘和调试工具
- **成本追踪**：自动计算 token 消耗和费用

**引用**：
> "LangSmith provides automatic tracing for all LangChain applications with zero code changes. Simply set environment variables and start monitoring."
> — [LangSmith Observability Platform](https://www.langchain.com/langsmith/observability)

---

## 环境配置

### 基础配置

```bash
# 1. 启用追踪
export LANGCHAIN_TRACING_V2=true

# 2. 设置 API Key
export LANGCHAIN_API_KEY=your_api_key

# 3. 设置项目名称
export LANGCHAIN_PROJECT=my_project

# 4. 可选：自定义端点
export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### Python 代码中配置

```python
import os

# 方式 1：直接设置环境变量
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_api_key"
os.environ["LANGCHAIN_PROJECT"] = "my_project"

# 方式 2：使用 dotenv
from dotenv import load_dotenv
load_dotenv()  # 从 .env 文件加载
```

### .env 文件示例

```bash
# .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_xxxxxxxxxxxxx
LANGCHAIN_PROJECT=production
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

---

## 自动追踪

### 零代码追踪

```python
from langchain_openai import ChatOpenAI

# 设置环境变量后，代码无需改动
llm = ChatOpenAI()
result = llm.invoke("你好")  # ← 自动发送到 LangSmith

# 访问 https://smith.langchain.com 查看追踪
```

### 追踪内容

LangSmith 自动记录：
1. **输入输出**：完整的 prompt 和 response
2. **Token 消耗**：prompt_tokens、completion_tokens、total_tokens
3. **延迟**：每个步骤的耗时
4. **成本**：根据 token 数自动计算
5. **错误堆栈**：失败时的完整错误信息
6. **链路结构**：嵌套链的完整调用树

---

## 使用 Tags 和 Metadata

### 添加 Tags

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    tags=["production", "critical", "rag"]
)

llm.invoke("你好", config=config)

# 在 LangSmith 中可以按 tags 筛选追踪
```

### 添加 Metadata

```python
config = RunnableConfig(
    metadata={
        "user_id": "user_123",
        "session": "session_abc",
        "env": "production",
        "version": "1.0.0"
    }
)

llm.invoke("你好", config=config)

# metadata 会显示在追踪详情中
```

---

## 成本追踪

### 自动成本计算

LangSmith 自动根据 token 数计算成本：

```python
# 无需手动计算成本
llm = ChatOpenAI(model="gpt-4")
llm.invoke("解释量子纠缠")

# LangSmith 自动显示：
# - Token 消耗: 245 tokens
# - 成本: $0.0049
```

### 成本归因

```python
# 按用户归因成本
for user_id in ["user_001", "user_002", "user_003"]:
    config = RunnableConfig(
        metadata={"user_id": user_id}
    )
    llm.invoke("你好", config=config)

# 在 LangSmith 中按 user_id 筛选，查看每个用户的成本
```

---

## 2025-2026 新特性

### 1. 实时告警（2025）

**功能**：成本、延迟、错误率超过阈值时自动告警

```python
# 在 LangSmith 仪表盘配置告警规则
# - 成本超过 $10/小时
# - 延迟超过 5 秒
# - 错误率超过 5%

# 告警渠道：
# - Email
# - Slack
# - PagerDuty
# - Webhook
```

**引用**：
> "LangSmith 2025 introduces real-time alerting for cost, latency, and error rate thresholds."
> — [LangSmith 2025 Release Notes](https://www.langchain.com/langsmith/observability)

---

### 2. 多轮评估（2025）

**功能**：评估对话式应用的多轮交互质量

```python
# 在 LangSmith 中创建评估数据集
# 包含多轮对话的输入输出

# 自动评估指标：
# - 上下文一致性
# - 回答相关性
# - 幻觉检测
# - 用户满意度
```

---

### 3. OpenTelemetry 集成（2025）

**功能**：与 OpenTelemetry 标准集成，支持分布式追踪

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from langchain_openai import ChatOpenAI

# 配置 OpenTelemetry
trace.set_tracer_provider(TracerProvider())

# LangSmith 自动集成 OpenTelemetry
llm = ChatOpenAI()
llm.invoke("你好")

# 追踪数据同时发送到：
# - LangSmith
# - OpenTelemetry Collector
# - Jaeger / Zipkin
```

**引用**：
> "LangSmith now supports OpenTelemetry for seamless integration with existing observability stacks."
> — [OpenTelemetry Integration Guide](https://oneuptime.com/blog/post/2026-02-06-monitor-langchain-opentelemetry/view)

---

## LangSmith vs 自定义回调

### 对比表

| 维度 | LangSmith | 自定义回调 |
|------|-----------|-----------|
| **配置复杂度** | 零配置（环境变量） | 需要编写代码 |
| **可视化** | 内置仪表盘 | 需要自己实现 |
| **成本** | 付费服务 | 免费 |
| **数据控制** | 数据上传到 LangSmith | 数据完全自控 |
| **定制性** | 有限 | 完全定制 |
| **集成难度** | 简单 | 中等 |
| **实时告警** | 内置（2025） | 需要自己实现 |
| **多轮评估** | 内置（2025） | 需要自己实现 |

### 选择建议

**使用 LangSmith**：
- 小团队，快速原型
- 需要可视化调试
- 不想自己维护监控系统

**使用自定义回调**：
- 大团队，定制需求
- 数据隐私要求高
- 已有监控系统（Datadog、Prometheus）

**混合方案**：
- LangSmith 用于开发调试
- 自定义回调用于生产监控
- 两者可以共存

---

## 完整生产示例

### 场景：RAG 文档问答系统

```python
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv

# 1. 加载环境变量
load_dotenv()

# 2. 创建 RAG 链
llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)
retriever = vectorstore.as_retriever()

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# 3. 执行查询（自动追踪到 LangSmith）
def query_with_tracking(query: str, user_id: str):
    config = RunnableConfig(
        tags=["production", "rag", "document-qa"],
        metadata={
            "user_id": user_id,
            "query_type": "document_qa",
            "env": "production"
        },
        run_name=f"rag_query_{user_id}"
    )

    result = rag_chain.invoke(
        {"query": query},
        config=config
    )

    return result

# 4. 使用
result = query_with_tracking(
    query="什么是量子纠缠？",
    user_id="user_123"
)

# 5. 在 LangSmith 中查看：
# - 检索到的文档
# - LLM 生成的回答
# - Token 消耗和成本
# - 每个步骤的延迟
```

---

## LangSmith 仪表盘功能

### 1. 追踪列表

**功能**：查看所有追踪记录

**筛选条件**：
- Tags
- Metadata
- 时间范围
- 成本范围
- 延迟范围
- 错误状态

---

### 2. 追踪详情

**功能**：查看单个追踪的完整信息

**包含内容**：
- 输入输出
- Token 消耗
- 延迟分解
- 成本计算
- 错误堆栈
- 链路结构

---

### 3. 成本分析

**功能**：分析成本趋势和分布

**分析维度**：
- 按时间：每小时/每天/每周成本
- 按用户：每个用户的成本
- 按模型：每个模型的成本
- 按功能：每个功能的成本

---

### 4. 性能监控

**功能**：监控延迟和吞吐量

**监控指标**：
- P50、P95、P99 延迟
- 每秒请求数（RPS）
- 错误率
- Token 消耗速率

---

### 5. 评估数据集

**功能**：创建和管理评估数据集

**用途**：
- 回归测试
- A/B 测试
- 质量评估
- 模型对比

---

## 最佳实践

### 1. 项目命名规范

```bash
# 按环境区分
export LANGCHAIN_PROJECT=myapp-production
export LANGCHAIN_PROJECT=myapp-staging
export LANGCHAIN_PROJECT=myapp-development

# 按功能区分
export LANGCHAIN_PROJECT=myapp-rag
export LANGCHAIN_PROJECT=myapp-chat
export LANGCHAIN_PROJECT=myapp-summarization
```

### 2. Tags 使用规范

```python
# 环境标签
tags = ["production"]  # or "staging", "development"

# 功能标签
tags.append("rag")  # or "chat", "summarization"

# 优先级标签
tags.append("critical")  # or "normal", "low"

# 完整示例
config = RunnableConfig(
    tags=["production", "rag", "critical"]
)
```

### 3. Metadata 使用规范

```python
# 必需字段
metadata = {
    "user_id": "user_123",      # 用户标识
    "session": "session_abc",   # 会话标识
    "env": "production"         # 环境
}

# 可选字段
metadata.update({
    "version": "1.0.0",         # 应用版本
    "region": "us-west",        # 地区
    "department": "sales"       # 部门
})

config = RunnableConfig(metadata=metadata)
```

### 4. 成本控制

```python
# 在 LangSmith 中设置预算告警
# - 每小时成本超过 $10
# - 每天成本超过 $100
# - 每月成本超过 $1000

# 代码中添加成本追踪
class CostLimiter(BaseCallbackHandler):
    def __init__(self, daily_limit: float = 100.0):
        self.daily_limit = daily_limit
        self.daily_cost = 0.0

    def on_llm_end(self, response, **kwargs):
        usage = response.llm_output.get("token_usage", {})
        cost = (
            usage.get("prompt_tokens", 0) * 0.00003 +
            usage.get("completion_tokens", 0) * 0.00006
        )
        self.daily_cost += cost

        if self.daily_cost > self.daily_limit:
            raise Exception(f"超过每日成本限制: ${self.daily_limit}")

# 使用
limiter = CostLimiter(daily_limit=100.0)
config = RunnableConfig(callbacks=[limiter])
```

---

## 故障排查

### 问题 1：追踪未显示

**原因**：
- 环境变量未设置
- API Key 无效
- 网络连接问题

**解决**：
```python
import os

# 检查环境变量
print(os.getenv("LANGCHAIN_TRACING_V2"))  # 应该是 "true"
print(os.getenv("LANGCHAIN_API_KEY"))     # 应该有值
print(os.getenv("LANGCHAIN_PROJECT"))     # 应该有值

# 测试连接
from langchain_openai import ChatOpenAI
llm = ChatOpenAI()
llm.invoke("test")  # 检查是否出现在 LangSmith
```

---

### 问题 2：成本计算不准确

**原因**：
- 模型价格更新
- Token 计数方式变化

**解决**：
- 在 LangSmith 设置中更新模型价格
- 使用自定义回调验证成本

---

### 问题 3：追踪数据过多

**原因**：
- 开发环境也启用了追踪
- 测试代码产生大量追踪

**解决**：
```python
# 只在生产环境启用追踪
import os

if os.getenv("ENV") == "production":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
```

---

## 总结

### LangSmith 核心优势

1. **零配置**：只需设置环境变量
2. **自动追踪**：无需手动传递 trace_id
3. **可视化强**：内置仪表盘和调试工具
4. **成本追踪**：自动计算 token 消耗和费用
5. **实时告警**：2025 新特性
6. **OpenTelemetry**：2025 新特性

### 使用场景

| 场景 | 推荐方案 |
|------|---------|
| **快速原型** | LangSmith |
| **小团队** | LangSmith |
| **可视化调试** | LangSmith |
| **大团队** | 自定义回调 |
| **定制需求** | 自定义回调 |
| **数据隐私** | 自定义回调 |
| **混合方案** | LangSmith + 自定义回调 |

---

## 参考资料

- [LangSmith Observability Platform](https://www.langchain.com/langsmith/observability)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [OpenTelemetry Integration](https://oneuptime.com/blog/post/2026-02-06-monitor-langchain-opentelemetry/view)
- [LangSmith Pricing](https://www.langchain.com/pricing)
