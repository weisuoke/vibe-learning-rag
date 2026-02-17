# LLMOps与生产运维

**知识点数**: 15个 | **学习时长**: 2周 | **优先级**: P0

---

## 概览

**为什么重要**:
> "Whoever solves consistent production reliability wins" — r/LLMDevs 2026

95%的企业AI试点无法产生可衡量的P&L影响，核心原因是缺乏生产运维能力。

**学习路径**:
```
L1_监控与可观测性（5个）→ L2_部署与发布（5个）→ L3_成本与性能优化（5个）
```

**前置知识**:
- Python基础、Docker基础
- 分布式系统概念
- 可观测性基础（Tracing、Metrics、Logging）

---

## L1_监控与可观测性（5个知识点）

### 01_分布式追踪系统

**核心概念**: 使用OpenTelemetry追踪Agent执行路径，定位性能瓶颈。

**关键技术**:
- **Trace**: 完整的请求链路
- **Span**: 单个操作单元
- **Context Propagation**: 跨服务传递追踪上下文

**实战代码**:
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

# 配置追踪
trace.set_tracer_provider(TracerProvider())
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

tracer = trace.get_tracer(__name__)

# 追踪Agent执行
with tracer.start_as_current_span("agent_execution") as span:
    span.set_attribute("agent.type", "research")
    # Agent逻辑
    result = agent.run(query)
    span.set_attribute("agent.result_length", len(result))
```

**推荐工具**: `opentelemetry-api`, `opentelemetry-sdk`, `jaeger-client`

**学习资源**:
- [OpenTelemetry Python文档](https://opentelemetry.io/docs/languages/python/)
- [Jaeger分布式追踪](https://www.jaegertracing.io/)

---

### 02_指标收集与告警

**核心概念**: 使用Prometheus收集关键指标，Grafana可视化，Alertmanager告警。

**关键指标**:
- **Token使用量**: 成本控制
- **响应时间**: 用户体验
- **错误率**: 系统健康度
- **并发数**: 容量规划

**实战代码**:
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# 定义指标
token_usage = Counter('agent_token_usage_total', 'Total tokens used', ['model'])
response_time = Histogram('agent_response_seconds', 'Response time')
active_requests = Gauge('agent_active_requests', 'Active requests')

# 使用指标
@response_time.time()
def run_agent(query: str):
    active_requests.inc()
    try:
        result = agent.run(query)
        token_usage.labels(model='gpt-4').inc(result.token_count)
        return result
    finally:
        active_requests.dec()

# 启动指标服务器
start_http_server(8000)
```

**推荐工具**: `prometheus-client`, `grafana`, `alertmanager`

---

### 03_日志聚合与分析

**核心概念**: 结构化日志 + ELK Stack，快速定位生产问题。

**实战代码**:
```python
import structlog

# 配置结构化日志
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()

# 记录Agent执行
logger.info(
    "agent_execution_started",
    agent_type="research",
    query_length=len(query),
    user_id=user_id
)
```

**推荐工具**: `structlog`, `elasticsearch`, `kibana`, `logstash`

---

### 04_性能监控面板

**核心概念**: Grafana实时监控面板，展示系统健康度。

**关键面板**:
- **实时指标**: Token/s、QPS、延迟P50/P95/P99
- **历史趋势**: 7天/30天趋势分析
- **异常检测**: 自动标记异常点
- **容量规划**: 资源使用预测

**Grafana配置示例**:
```yaml
# grafana-dashboard.json
{
  "dashboard": {
    "title": "AI Agent Monitoring",
    "panels": [
      {
        "title": "Token Usage",
        "targets": [
          {
            "expr": "rate(agent_token_usage_total[5m])"
          }
        ]
      }
    ]
  }
}
```

---

### 05_成本追踪系统

**核心概念**: 按用户/项目分摊Token成本，实时预算告警。

**实战代码**:
```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CostTracker:
    user_id: str
    project_id: str

    def track_usage(self, model: str, tokens: int):
        cost = self.calculate_cost(model, tokens)

        # 记录到数据库
        db.execute("""
            INSERT INTO token_usage (user_id, project_id, model, tokens, cost, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (self.user_id, self.project_id, model, tokens, cost, datetime.now()))

        # 检查预算
        if self.check_budget_exceeded():
            alert_manager.send_alert(f"Budget exceeded for {self.project_id}")

    def calculate_cost(self, model: str, tokens: int) -> float:
        pricing = {
            "gpt-4": 0.03 / 1000,  # $0.03 per 1K tokens
            "gpt-3.5-turbo": 0.002 / 1000
        }
        return tokens * pricing.get(model, 0)
```

---

## L2_部署与发布（5个知识点）

### 01_模型版本管理

**核心概念**: MLflow管理模型版本，支持快速回滚。

**实战代码**:
```python
import mlflow

# 注册模型
with mlflow.start_run():
    mlflow.log_param("model", "gpt-4")
    mlflow.log_param("temperature", 0.7)
    mlflow.log_metric("accuracy", 0.95)

    # 注册模型版本
    mlflow.register_model(
        model_uri="runs:/<run_id>/model",
        name="agent_model"
    )

# 加载特定版本
model = mlflow.pyfunc.load_model("models:/agent_model/1")
```

**推荐工具**: `mlflow`, `dvc`, `wandb`

---

### 02_A/B测试与灰度发布

**核心概念**: 流量分配策略，对比不同模型/Prompt效果。

**实战代码**:
```python
import random

class ABTestRouter:
    def __init__(self, variants: dict):
        self.variants = variants  # {"A": 0.5, "B": 0.5}

    def route(self, user_id: str) -> str:
        # 基于用户ID的稳定路由
        hash_value = hash(user_id) % 100
        cumulative = 0

        for variant, percentage in self.variants.items():
            cumulative += percentage * 100
            if hash_value < cumulative:
                return variant

        return list(self.variants.keys())[0]

# 使用
router = ABTestRouter({"gpt-4": 0.8, "gpt-3.5": 0.2})
model = router.route(user_id)
```

---

### 03_蓝绿部署策略

**核心概念**: 双环境管理，零停机部署。

**Kubernetes配置**:
```yaml
# blue-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent
      version: blue
  template:
    metadata:
      labels:
        app: agent
        version: blue
    spec:
      containers:
      - name: agent
        image: agent:v1.0
```

**流量切换**:
```bash
# 切换到绿色环境
kubectl patch service agent -p '{"spec":{"selector":{"version":"green"}}}'
```

---

### 04_金丝雀发布

**核心概念**: 渐进式发布，逐步验证新版本。

**Flagger配置**:
```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: agent
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent
  progressDeadlineSeconds: 60
  service:
    port: 8080
  analysis:
    interval: 1m
    threshold: 5
    maxWeight: 50
    stepWeight: 10
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99
```

---

### 05_回滚机制

**核心概念**: 自动回滚触发，快速恢复生产故障。

**实战代码**:
```python
class AutoRollback:
    def __init__(self, error_threshold: float = 0.05):
        self.error_threshold = error_threshold

    def monitor_deployment(self, deployment_id: str):
        error_rate = self.get_error_rate(deployment_id)

        if error_rate > self.error_threshold:
            logger.error(f"Error rate {error_rate} exceeds threshold")
            self.rollback(deployment_id)

    def rollback(self, deployment_id: str):
        # 回滚到上一个版本
        previous_version = self.get_previous_version(deployment_id)
        self.deploy(previous_version)

        # 发送告警
        alert_manager.send_alert(f"Auto-rollback triggered for {deployment_id}")
```

---

## L3_成本与性能优化（5个知识点）

### 01_Token成本优化

**核心概念**: Prompt压缩 + 缓存策略，降低API调用成本50%+。

**实战代码**:
```python
import tiktoken
from functools import lru_cache

class TokenOptimizer:
    def __init__(self):
        self.encoder = tiktoken.encoding_for_model("gpt-4")
        self.cache = {}

    def compress_prompt(self, prompt: str) -> str:
        # 移除冗余空白
        compressed = " ".join(prompt.split())

        # 使用更短的指令
        compressed = compressed.replace(
            "Please provide a detailed answer",
            "Answer:"
        )

        return compressed

    @lru_cache(maxsize=1000)
    def cached_completion(self, prompt: str):
        return llm.complete(prompt)
```

**优化策略**:
- Prompt压缩: 减少冗余词汇
- 语义缓存: 相似查询复用结果
- 模型选择: 简单任务用小模型
- 批处理: 合并多个请求

---

### 02_延迟优化策略

**核心概念**: 并发请求 + 流式响应，提升用户体验。

**实战代码**:
```python
import asyncio

async def parallel_agent_calls(queries: list[str]):
    tasks = [agent.arun(query) for query in queries]
    results = await asyncio.gather(*tasks)
    return results

# 流式响应
async def stream_response(query: str):
    async for chunk in agent.astream(query):
        yield chunk
```

---

### 03_模型漂移检测

**核心概念**: 监控输入分布和输出质量，及时发现性能退化。

**实战代码**:
```python
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

class DriftDetector:
    def __init__(self, reference_data):
        self.reference_data = reference_data

    def detect_drift(self, current_data):
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )

        if report.as_dict()["metrics"][0]["result"]["dataset_drift"]:
            alert_manager.send_alert("Model drift detected")
```

**推荐工具**: `evidently`, `alibi-detect`

---

### 04_一致性监控

**核心概念**: 检测Agent输出的逻辑一致性和事实准确性。

**实战代码**:
```python
class ConsistencyMonitor:
    def check_consistency(self, query: str, response: str) -> float:
        # 多次生成，检查一致性
        responses = [agent.run(query) for _ in range(3)]

        # 计算相似度
        similarities = []
        for i in range(len(responses)):
            for j in range(i+1, len(responses)):
                sim = self.calculate_similarity(responses[i], responses[j])
                similarities.append(sim)

        consistency_score = sum(similarities) / len(similarities)

        if consistency_score < 0.7:
            logger.warning(f"Low consistency: {consistency_score}")

        return consistency_score
```

---

### 05_轨迹验证

**核心概念**: 验证Agent执行路径，避免无限循环和错误决策。

**实战代码**:
```python
class TrajectoryValidator:
    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps

    def validate(self, trajectory: list[dict]) -> bool:
        # 检查步数
        if len(trajectory) > self.max_steps:
            logger.error("Trajectory too long")
            return False

        # 检查循环
        actions = [step["action"] for step in trajectory]
        if self.has_loop(actions):
            logger.error("Loop detected in trajectory")
            return False

        # 检查终止条件
        if not trajectory[-1].get("is_final"):
            logger.error("Trajectory did not reach final state")
            return False

        return True

    def has_loop(self, actions: list[str]) -> bool:
        seen = set()
        for action in actions:
            if action in seen:
                return True
            seen.add(action)
        return False
```

---

## 核心20%（6个知识点）

优先学习以下6个知识点，可产生80%效果：

1. ⭐⭐⭐ **分布式追踪系统** - 生产环境调试基础
2. ⭐⭐⭐ **指标收集与告警** - 系统健康度监控
3. ⭐⭐⭐ **模型版本管理** - 快速回滚能力
4. ⭐⭐⭐ **A/B测试与灰度发布** - 验证新版本
5. ⭐⭐⭐ **Token成本优化** - 降低运营成本
6. ⭐⭐⭐ **模型漂移检测** - 及时发现问题

---

## 实战项目

### 项目1: 构建监控系统

**目标**: 为现有Agent系统添加完整的监控能力。

**步骤**:
1. 集成OpenTelemetry追踪
2. 添加Prometheus指标
3. 配置Grafana面板
4. 设置告警规则

**验收标准**:
- [ ] 能够追踪完整的Agent执行路径
- [ ] 实时监控Token使用量和响应时间
- [ ] 错误率超过5%时自动告警

---

### 项目2: 实现灰度发布

**目标**: 实现模型版本的灰度发布和自动回滚。

**步骤**:
1. 使用MLflow管理模型版本
2. 实现A/B测试路由
3. 配置金丝雀发布
4. 实现自动回滚机制

**验收标准**:
- [ ] 能够同时运行多个模型版本
- [ ] 流量按比例分配
- [ ] 错误率超标时自动回滚

---

## 学习资源

### 必读文档
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)
- [Prometheus最佳实践](https://prometheus.io/docs/practices/)
- [MLflow文档](https://mlflow.org/docs/latest/index.html)

### 推荐工具
- **追踪**: OpenTelemetry, Jaeger, Zipkin
- **指标**: Prometheus, Grafana, Datadog
- **日志**: Structlog, ELK Stack
- **版本管理**: MLflow, DVC, Weights & Biases
- **部署**: Kubernetes, Helm, Flagger

### 社区资源
- Reddit: r/LLMDevs, r/mlops
- GitHub: Awesome-LLMOps
- X.com: 关注LLMOps专家

---

## 常见误区

1. ❌ "监控严重影响性能" → ✅ 异步追踪开销<1%
2. ❌ "只在出问题时看监控" → ✅ 主动监控预防问题
3. ❌ "成本优化=减少调用" → ✅ 缓存+压缩+模型选择
4. ❌ "A/B测试只看准确率" → ✅ 综合考虑成本、延迟、用户体验
5. ❌ "模型漂移很少发生" → ✅ 用户行为变化导致频繁漂移

---

## 下一步

完成LLMOps学习后，继续学习：
- [02_替代框架.md](./02_替代框架.md) - 对比多个框架
- [03_企业架构.md](./03_企业架构.md) - 企业级架构模式

---

**版本**: v1.0
**最后更新**: 2026-02-12
**学习时长**: 2周
**难度**: ⭐⭐⭐⭐
