# 实战代码 04：LangSmith 集成

> **LangSmith 平台的完整集成方案、环境配置和高级功能使用**

---

## 场景 1：基础 LangSmith 集成

```bash
# 1. 设置环境变量
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=lsv2_pt_xxxxxxxxxxxxx
export LANGCHAIN_PROJECT=my_project
```

```python
from langchain_openai import ChatOpenAI

# 2. 代码无需改动，自动追踪
llm = ChatOpenAI(model="gpt-3.5-turbo")
result = llm.invoke("你好")

# 3. 访问 https://smith.langchain.com 查看追踪
print(result.content)
```

**自动记录**：
- 输入输出
- Token 消耗
- 延迟
- 成本
- 错误堆栈

---

## 场景 2：使用 .env 文件

```bash
# .env 文件
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_xxxxxxxxxxxxx
LANGCHAIN_PROJECT=production
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

# OpenAI API Key
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
```

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# 自动启用 LangSmith 追踪
llm = ChatOpenAI()
result = llm.invoke("什么是机器学习？")
```

---

## 场景 3：动态启用/禁用追踪

```python
import os
from langchain_openai import ChatOpenAI

def query_with_tracing(query: str, enable_tracing: bool = True):
    """动态控制追踪"""
    # 临时设置环境变量
    original_value = os.environ.get("LANGCHAIN_TRACING_V2")

    if enable_tracing:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    try:
        llm = ChatOpenAI()
        result = llm.invoke(query)
        return result
    finally:
        # 恢复原始值
        if original_value is not None:
            os.environ["LANGCHAIN_TRACING_V2"] = original_value
        else:
            os.environ.pop("LANGCHAIN_TRACING_V2", None)

# 使用
result1 = query_with_tracing("你好", enable_tracing=True)   # 追踪
result2 = query_with_tracing("你好", enable_tracing=False)  # 不追踪
```

---

## 场景 4：添加 Tags 和 Metadata

```python
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

llm = ChatOpenAI()

# 添加 tags 和 metadata
config = RunnableConfig(
    tags=["production", "rag", "critical"],
    metadata={
        "user_id": "user_123",
        "session": "session_abc",
        "env": "production",
        "version": "1.0.0"
    }
)

result = llm.invoke("什么是量子纠缠？", config=config)

# 在 LangSmith 中可以：
# - 按 tags 筛选追踪
# - 查看 metadata 详情
# - 按 user_id 分组统计
```

---

## 场景 5：自定义 Run Name

```python
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
import uuid

llm = ChatOpenAI()

# 自定义 run_name
def create_config(user_id: str, query_type: str):
    run_id = str(uuid.uuid4())[:8]
    return RunnableConfig(
        run_name=f"{user_id}_{query_type}_{run_id}",
        tags=[query_type],
        metadata={"user_id": user_id}
    )

config = create_config("user_123", "rag_search")
result = llm.invoke("你好", config=config)

# 在 LangSmith 中可以按 run_name 搜索
```

---

## 场景 6：RAG 系统集成

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnableConfig

# 加载环境变量
load_dotenv()

# 创建 RAG 链
llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)
retriever = vectorstore.as_retriever()

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# 执行查询（自动追踪）
def rag_query(query: str, user_id: str):
    config = RunnableConfig(
        tags=["production", "rag", "document-qa"],
        metadata={
            "user_id": user_id,
            "query_type": "document_qa",
            "env": "production"
        },
        run_name=f"rag_{user_id}_{query[:20]}"
    )

    result = rag_chain.invoke(
        {"query": query},
        config=config
    )

    return result

# 使用
result = rag_query("什么是量子纠缠？", "user_123")

# 在 LangSmith 中可以看到：
# - 检索到的文档
# - LLM 生成的回答
# - 每个步骤的延迟
# - Token 消耗和成本
```

---

## 场景 7：多项目管理

```python
import os
from langchain_openai import ChatOpenAI

def query_with_project(query: str, project: str):
    """使用不同项目"""
    # 临时切换项目
    original_project = os.environ.get("LANGCHAIN_PROJECT")
    os.environ["LANGCHAIN_PROJECT"] = project

    try:
        llm = ChatOpenAI()
        result = llm.invoke(query)
        return result
    finally:
        if original_project:
            os.environ["LANGCHAIN_PROJECT"] = original_project

# 使用不同项目
query_with_project("你好", "project_a")
query_with_project("你好", "project_b")

# 在 LangSmith 中可以按项目查看追踪
```

---

## 场景 8：环境隔离

```python
import os
from langchain_openai import ChatOpenAI

# 根据环境启用追踪
ENV = os.getenv("ENV", "development")

if ENV == "production":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "myapp-production"
elif ENV == "staging":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "myapp-staging"
else:
    # 开发环境不追踪
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

llm = ChatOpenAI()
result = llm.invoke("你好")
```

---

## 场景 9：结合自定义回调

```python
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

class CustomLogger(BaseCallbackHandler):
    """自定义日志回调"""
    def on_llm_end(self, response, **kwargs):
        tokens = response.llm_output["token_usage"]["total_tokens"]
        print(f"本地日志: 消耗 {tokens} tokens")

# LangSmith + 自定义回调
llm = ChatOpenAI()

config = RunnableConfig(
    callbacks=[CustomLogger()],  # 自定义回调
    tags=["production"],
    metadata={"user_id": "user_123"}
)

result = llm.invoke("你好", config=config)

# 同时：
# - 本地日志输出
# - LangSmith 自动追踪
```

---

## 场景 10：批量追踪

```python
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

llm = ChatOpenAI()

# 批量调用
queries = ["问题1", "问题2", "问题3"]

configs = [
    RunnableConfig(
        tags=["batch"],
        metadata={"query_id": i, "user_id": "user_123"}
    )
    for i in range(len(queries))
]

results = llm.batch(queries, config=configs)

# 在 LangSmith 中可以：
# - 查看所有批量调用
# - 按 query_id 排序
# - 分析批量性能
```

---

## 场景 11：错误追踪

```python
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

llm = ChatOpenAI()

config = RunnableConfig(
    tags=["production", "error-test"],
    metadata={"user_id": "user_123"}
)

try:
    # 故意触发错误
    result = llm.invoke("", config=config)
except Exception as e:
    print(f"错误: {e}")

# 在 LangSmith 中可以看到：
# - 完整的错误堆栈
# - 错误发生的上下文
# - 错误率统计
```

---

## 场景 12：完整生产集成

```python
import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler
import uuid

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

class ProductionCallback(BaseCallbackHandler):
    """生产环境回调"""
    def on_llm_start(self, serialized, prompts, **kwargs):
        metadata = kwargs.get("metadata", {})
        logger.info(f"LLM 开始: user_id={metadata.get('user_id')}")

    def on_llm_end(self, response, **kwargs):
        tokens = response.llm_output["token_usage"]["total_tokens"]
        logger.info(f"LLM 完成: tokens={tokens}")

    def on_llm_error(self, error, **kwargs):
        logger.error(f"LLM 失败: {error}")

def create_production_config(user_id: str, query_type: str):
    """创建生产环境配置"""
    return RunnableConfig(
        # 自定义回调
        callbacks=[ProductionCallback()],

        # LangSmith 标签
        tags=["production", query_type, "v1.0"],

        # LangSmith 元数据
        metadata={
            "user_id": user_id,
            "query_type": query_type,
            "env": "production",
            "version": "1.0.0",
            "timestamp": "2025-01-15T10:30:00Z"
        },

        # LangSmith 运行名称
        run_name=f"{user_id}_{query_type}_{uuid.uuid4().hex[:8]}"
    )

def query_with_full_tracking(query: str, user_id: str, query_type: str = "general"):
    """带完整追踪的查询"""
    llm = ChatOpenAI(model="gpt-4")
    config = create_production_config(user_id, query_type)

    try:
        result = llm.invoke(query, config=config)
        return result.content
    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise

# 使用
result = query_with_full_tracking(
    query="什么是量子纠缠？",
    user_id="user_123",
    query_type="rag_search"
)

print(result)

# 同时获得：
# - 本地日志输出
# - LangSmith 完整追踪
# - 成本统计
# - 性能分析
```

---

## LangSmith 仪表盘使用

### 1. 追踪列表筛选

```python
# 在代码中添加标签和元数据
config = RunnableConfig(
    tags=["production", "rag", "critical"],
    metadata={
        "user_id": "user_123",
        "department": "sales",
        "region": "us-west"
    }
)

# 在 LangSmith 仪表盘中：
# - 按 tags 筛选: tags:production AND tags:rag
# - 按 metadata 筛选: metadata.user_id:user_123
# - 按时间筛选: 最近 1 小时/24 小时/7 天
# - 按成本筛选: cost > $0.01
# - 按延迟筛选: latency > 2s
```

### 2. 追踪详情查看

访问单个追踪可以看到：
- **输入输出**：完整的 prompt 和 response
- **Token 消耗**：prompt_tokens、completion_tokens、total_tokens
- **延迟分解**：每个步骤的耗时
- **成本计算**：根据 token 数自动计算
- **链路结构**：嵌套链的完整调用树
- **Metadata**：所有自定义元数据

### 3. 成本分析

```python
# 在代码中添加用户信息
config = RunnableConfig(
    metadata={"user_id": "user_123"}
)

# 在 LangSmith 仪表盘中：
# - 按用户查看成本: 筛选 metadata.user_id:user_123
# - 按时间查看成本: 选择时间范围
# - 导出成本报告: 下载 CSV
```

### 4. 性能监控

LangSmith 自动统计：
- **P50、P95、P99 延迟**
- **每秒请求数（RPS）**
- **错误率**
- **Token 消耗速率**

---

## 高级功能（2025-2026）

### 1. 实时告警

```python
# 在 LangSmith 仪表盘配置告警规则：
# - 成本超过 $10/小时 → 发送 Slack 通知
# - 延迟超过 5 秒 → 发送 Email
# - 错误率超过 5% → 发送 PagerDuty

# 代码无需改动，自动触发告警
```

### 2. 多轮评估

```python
# 在 LangSmith 中创建评估数据集
# 包含多轮对话的输入输出

# 自动评估指标：
# - 上下文一致性
# - 回答相关性
# - 幻觉检测
# - 用户满意度
```

### 3. OpenTelemetry 集成

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

# 配置 OpenTelemetry
trace.set_tracer_provider(TracerProvider())

# LangSmith 自动集成 OpenTelemetry
# 追踪数据同时发送到：
# - LangSmith
# - OpenTelemetry Collector
# - Jaeger / Zipkin
```

---

## 故障排查

### 问题 1：追踪未显示

```python
import os

# 检查环境变量
print("LANGCHAIN_TRACING_V2:", os.getenv("LANGCHAIN_TRACING_V2"))
print("LANGCHAIN_API_KEY:", os.getenv("LANGCHAIN_API_KEY")[:10] + "...")
print("LANGCHAIN_PROJECT:", os.getenv("LANGCHAIN_PROJECT"))

# 测试连接
from langchain_openai import ChatOpenAI
llm = ChatOpenAI()
llm.invoke("test")

# 访问 https://smith.langchain.com 检查是否出现
```

### 问题 2：API Key 无效

```bash
# 检查 API Key 格式
# 正确格式: lsv2_pt_xxxxxxxxxxxxx

# 重新生成 API Key
# 访问 https://smith.langchain.com/settings
```

### 问题 3：追踪数据过多

```python
# 只在生产环境启用追踪
import os

ENV = os.getenv("ENV", "development")

if ENV == "production":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
```

---

## 最佳实践

### 1. 项目命名规范

```bash
# 按环境区分
export LANGCHAIN_PROJECT=myapp-production
export LANGCHAIN_PROJECT=myapp-staging
export LANGCHAIN_PROJECT=myapp-development
```

### 2. Tags 使用规范

```python
# 环境 + 功能 + 优先级
tags = ["production", "rag", "critical"]
```

### 3. Metadata 使用规范

```python
# 必需字段
metadata = {
    "user_id": "user_123",
    "session": "session_abc",
    "env": "production"
}
```

### 4. 成本控制

```python
# 在 LangSmith 中设置预算告警
# - 每小时成本超过 $10
# - 每天成本超过 $100
# - 每月成本超过 $1000
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
| **大团队** | LangSmith + 自定义回调 |
| **定制需求** | 自定义回调 |
| **数据隐私** | 自定义回调 |

---

## 参考资料

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangSmith Observability Platform](https://www.langchain.com/langsmith/observability)
- [LangSmith Pricing](https://www.langchain.com/pricing)
