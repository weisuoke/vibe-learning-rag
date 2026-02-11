# 核心概念1：Prometheus 指标采集

> 理解 Prometheus 如何采集 Milvus 的运行指标，构建可观测性的数据基础

---

## 什么是 Prometheus 指标采集？

**Prometheus 指标采集是通过定期拉取（Pull）Milvus 暴露的 Metrics 端点，收集系统运行数据（CPU、内存、QPS、延迟等），并存储为时间序列数据，用于监控、告警和分析。**

---

## 1. Prometheus 的工作原理

### 1.1 Pull 模型 vs Push 模型

**Prometheus 采用 Pull（拉取）模型：**

```
传统 Push 模型（如 StatsD）：
应用 → 主动推送指标 → 监控系统

Prometheus Pull 模型：
应用 ← 定期拉取指标 ← Prometheus
```

**Pull 模型的优势：**
- ✅ **服务发现简单**：Prometheus 主动发现目标，应用无需知道监控系统地址
- ✅ **故障隔离**：监控系统故障不影响应用运行
- ✅ **灵活采集**：可以随时调整采集频率和目标
- ✅ **健康检查**：拉取失败即表示目标不可用

**在 RAG 系统中的应用：**
- Milvus 暴露 Metrics 端点，Prometheus 定期拉取
- 即使 Prometheus 宕机，Milvus 仍正常运行
- 可以动态添加新的 Milvus 实例到监控

---

### 1.2 Metrics 端点格式

**Milvus 暴露的 Metrics 端点：**

```bash
# Milvus Standalone 默认端点
http://localhost:9091/metrics

# Milvus 分布式部署各组件端点
http://proxy:9091/metrics      # Proxy 组件
http://querynode:9091/metrics   # QueryNode 组件
http://datanode:9091/metrics    # DataNode 组件
http://indexnode:9091/metrics   # IndexNode 组件
```

**Metrics 数据格式（Prometheus 文本格式）：**

```
# HELP milvus_proxy_search_vectors_count Total number of vectors searched
# TYPE milvus_proxy_search_vectors_count counter
milvus_proxy_search_vectors_count{collection="my_collection"} 12345

# HELP milvus_proxy_search_latency_milliseconds Search latency in milliseconds
# TYPE milvus_proxy_search_latency_milliseconds histogram
milvus_proxy_search_latency_milliseconds_bucket{le="10"} 100
milvus_proxy_search_latency_milliseconds_bucket{le="50"} 450
milvus_proxy_search_latency_milliseconds_bucket{le="100"} 800
milvus_proxy_search_latency_milliseconds_bucket{le="+Inf"} 1000
milvus_proxy_search_latency_milliseconds_sum 45000
milvus_proxy_search_latency_milliseconds_count 1000
```

**格式说明：**
- `# HELP`：指标说明
- `# TYPE`：指标类型（counter、gauge、histogram、summary）
- 指标名称：`milvus_proxy_search_vectors_count`
- 标签（Labels）：`{collection="my_collection"}`
- 指标值：`12345`

---

## 2. Prometheus 的三种指标类型

### 2.1 Counter（计数器）

**定义：** 只增不减的累计值，用于统计事件发生次数

**特点：**
- ✅ 单调递增（重启后归零）
- ✅ 适合统计总量（请求数、错误数）
- ✅ 通常使用 `rate()` 函数计算速率

**Milvus 中的 Counter 指标：**

```python
# 示例：查询总次数
milvus_proxy_search_vectors_count

# 示例：插入总次数
milvus_proxy_insert_vectors_count

# 示例：错误总次数
milvus_proxy_search_failed_count
```

**PromQL 查询示例：**

```promql
# 计算每秒查询速率（QPS）
rate(milvus_proxy_search_vectors_count[5m])

# 计算过去 1 小时的总查询数
increase(milvus_proxy_search_vectors_count[1h])

# 计算错误率
rate(milvus_proxy_search_failed_count[5m])
/
rate(milvus_proxy_search_vectors_count[5m])
```

**在 RAG 系统中的应用：**
- 统计向量检索的总次数
- 计算 Embedding 生成的速率
- 追踪缓存命中次数

---

### 2.2 Gauge（仪表盘）

**定义：** 可增可减的瞬时值，用于表示当前状态

**特点：**
- ✅ 可以上升或下降
- ✅ 适合表示当前值（内存使用、连接数）
- ✅ 直接使用，无需计算速率

**Milvus 中的 Gauge 指标：**

```python
# 示例：当前内存使用量（字节）
process_resident_memory_bytes

# 示例：当前 CPU 使用率
process_cpu_seconds_total

# 示例：当前加载的 Collection 数量
milvus_proxy_collection_loaded_count

# 示例：当前活跃连接数
milvus_proxy_connection_count
```

**PromQL 查询示例：**

```promql
# 查询当前内存使用量（MB）
process_resident_memory_bytes / 1024 / 1024

# 查询内存使用率（假设总内存 16GB）
process_resident_memory_bytes / (16 * 1024 * 1024 * 1024) * 100

# 查询 CPU 使用率（过去 5 分钟平均）
rate(process_cpu_seconds_total[5m]) * 100
```

**在 RAG 系统中的应用：**
- 监控 Milvus 的内存使用情况
- 追踪当前加载的知识库数量
- 观察并发查询连接数

---

### 2.3 Histogram（直方图）

**定义：** 统计数据分布，将数据分桶（Bucket）统计

**特点：**
- ✅ 提供分位数（P50、P95、P99）
- ✅ 适合统计延迟、大小等分布
- ✅ 自动生成 `_bucket`、`_sum`、`_count` 三个指标

**Milvus 中的 Histogram 指标：**

```python
# 示例：查询延迟分布
milvus_proxy_search_latency_milliseconds

# 生成的指标：
# milvus_proxy_search_latency_milliseconds_bucket{le="10"}   # ≤10ms 的请求数
# milvus_proxy_search_latency_milliseconds_bucket{le="50"}   # ≤50ms 的请求数
# milvus_proxy_search_latency_milliseconds_bucket{le="100"}  # ≤100ms 的请求数
# milvus_proxy_search_latency_milliseconds_bucket{le="+Inf"} # 所有请求数
# milvus_proxy_search_latency_milliseconds_sum               # 总延迟
# milvus_proxy_search_latency_milliseconds_count             # 请求总数
```

**PromQL 查询示例：**

```promql
# 计算 P95 延迟（95% 的请求延迟低于此值）
histogram_quantile(0.95,
  rate(milvus_proxy_search_latency_milliseconds_bucket[5m])
)

# 计算 P99 延迟
histogram_quantile(0.99,
  rate(milvus_proxy_search_latency_milliseconds_bucket[5m])
)

# 计算平均延迟
rate(milvus_proxy_search_latency_milliseconds_sum[5m])
/
rate(milvus_proxy_search_latency_milliseconds_count[5m])
```

**在 RAG 系统中的应用：**
- 监控向量检索的延迟分布（P50、P95、P99）
- 追踪 Embedding 生成的耗时分布
- 分析不同 Collection 的性能差异

---

## 3. Milvus 的关键指标

### 3.1 性能指标

#### 查询性能

```promql
# QPS（每秒查询数）
rate(milvus_proxy_search_vectors_count[5m])

# 查询延迟 P95
histogram_quantile(0.95,
  rate(milvus_proxy_search_latency_milliseconds_bucket[5m])
)

# 查询成功率
(
  rate(milvus_proxy_search_vectors_count[5m])
  -
  rate(milvus_proxy_search_failed_count[5m])
)
/
rate(milvus_proxy_search_vectors_count[5m])
* 100
```

#### 插入性能

```promql
# 插入速率（每秒插入向量数）
rate(milvus_proxy_insert_vectors_count[5m])

# 插入延迟 P95
histogram_quantile(0.95,
  rate(milvus_proxy_insert_latency_milliseconds_bucket[5m])
)
```

---

### 3.2 资源指标

#### 内存使用

```promql
# 当前内存使用量（MB）
process_resident_memory_bytes / 1024 / 1024

# 内存使用率（假设总内存 16GB）
process_resident_memory_bytes / (16 * 1024 * 1024 * 1024) * 100
```

#### CPU 使用

```promql
# CPU 使用率（过去 5 分钟平均）
rate(process_cpu_seconds_total[5m]) * 100
```

#### 磁盘使用

```promql
# 磁盘使用量（GB）
milvus_datanode_storage_size_bytes / 1024 / 1024 / 1024

# 磁盘使用率
milvus_datanode_storage_size_bytes
/
milvus_datanode_storage_capacity_bytes
* 100
```

---

### 3.3 业务指标

#### Collection 状态

```promql
# 已加载的 Collection 数量
milvus_proxy_collection_loaded_count

# Collection 的向量数量
milvus_proxy_collection_entity_count{collection="my_collection"}
```

#### 连接状态

```promql
# 当前活跃连接数
milvus_proxy_connection_count

# 连接失败次数
rate(milvus_proxy_connection_failed_count[5m])
```

---

## 4. Prometheus 配置详解

### 4.1 基础配置

**prometheus.yml 配置文件：**

```yaml
# 全局配置
global:
  scrape_interval: 15s      # 默认采集间隔
  evaluation_interval: 15s  # 告警规则评估间隔
  scrape_timeout: 10s       # 采集超时时间

# 告警管理器配置
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

# 告警规则文件
rule_files:
  - "alerts/*.yml"

# 采集目标配置
scrape_configs:
  # Milvus Standalone 监控
  - job_name: 'milvus-standalone'
    static_configs:
      - targets: ['milvus-standalone:9091']
        labels:
          instance: 'milvus-standalone'
          env: 'production'

    # 采集间隔（覆盖全局配置）
    scrape_interval: 15s

    # 采集超时
    scrape_timeout: 10s

    # Metrics 路径
    metrics_path: '/metrics'

    # 协议
    scheme: 'http'

  # Milvus 分布式部署监控
  - job_name: 'milvus-distributed'
    static_configs:
      # Proxy 组件
      - targets: ['milvus-proxy-1:9091', 'milvus-proxy-2:9091']
        labels:
          component: 'proxy'

      # QueryNode 组件
      - targets: ['milvus-querynode-1:9091', 'milvus-querynode-2:9091']
        labels:
          component: 'querynode'

      # DataNode 组件
      - targets: ['milvus-datanode-1:9091', 'milvus-datanode-2:9091']
        labels:
          component: 'datanode'

      # IndexNode 组件
      - targets: ['milvus-indexnode-1:9091']
        labels:
          component: 'indexnode'

  # Prometheus 自身监控
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

---

### 4.2 服务发现配置

**使用 Docker Swarm 服务发现：**

```yaml
scrape_configs:
  - job_name: 'milvus-swarm'
    dockerswarm_sd_configs:
      - host: unix:///var/run/docker.sock
        role: tasks

    relabel_configs:
      # 只监控带有 milvus 标签的服务
      - source_labels: [__meta_dockerswarm_service_label_app]
        regex: milvus
        action: keep

      # 使用服务名作为 job 标签
      - source_labels: [__meta_dockerswarm_service_name]
        target_label: job
```

**使用 Kubernetes 服务发现：**

```yaml
scrape_configs:
  - job_name: 'milvus-k8s'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - milvus

    relabel_configs:
      # 只监控带有 app=milvus 标签的 Pod
      - source_labels: [__meta_kubernetes_pod_label_app]
        regex: milvus
        action: keep

      # 使用 Pod 名称作为 instance 标签
      - source_labels: [__meta_kubernetes_pod_name]
        target_label: instance

      # 使用组件名称作为 component 标签
      - source_labels: [__meta_kubernetes_pod_label_component]
        target_label: component
```

---

### 4.3 指标重标签（Relabeling）

**重标签的作用：**
- 过滤不需要的指标
- 修改标签名称
- 添加自定义标签

**示例配置：**

```yaml
scrape_configs:
  - job_name: 'milvus'
    static_configs:
      - targets: ['milvus:9091']

    # 采集前重标签（metric_relabel_configs）
    metric_relabel_configs:
      # 只保留 milvus_ 开头的指标
      - source_labels: [__name__]
        regex: 'milvus_.*'
        action: keep

      # 删除不需要的标签
      - regex: 'pod_template_hash'
        action: labeldrop

      # 重命名标签
      - source_labels: [collection]
        target_label: collection_name
```

---

## 5. 实战示例：采集 Milvus 指标

### 5.1 验证 Metrics 端点

```python
"""
验证 Milvus Metrics 端点是否可访问
"""

import requests

def check_milvus_metrics(host="localhost", port=9091):
    """检查 Milvus Metrics 端点"""
    url = f"http://{host}:{port}/metrics"

    try:
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            print(f"✅ Metrics 端点可访问: {url}")

            # 解析指标数量
            lines = response.text.split('\n')
            metrics = [line for line in lines if line and not line.startswith('#')]

            print(f"📊 指标数量: {len(metrics)}")

            # 显示前 10 个指标
            print("\n前 10 个指标:")
            for metric in metrics[:10]:
                print(f"  {metric}")

            return True
        else:
            print(f"❌ Metrics 端点返回错误: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ 无法连接到 Metrics 端点: {e}")
        return False

# 运行检查
if __name__ == "__main__":
    check_milvus_metrics()
```

---

### 5.2 解析 Prometheus 指标

```python
"""
解析 Prometheus 文本格式的指标
"""

import re
from typing import Dict, List, Tuple

def parse_prometheus_metrics(metrics_text: str) -> Dict[str, List[Tuple[Dict, float]]]:
    """
    解析 Prometheus 指标文本

    返回格式：
    {
        "metric_name": [
            ({"label1": "value1", "label2": "value2"}, 123.45),
            ...
        ]
    }
    """
    metrics = {}

    for line in metrics_text.split('\n'):
        # 跳过注释和空行
        if not line or line.startswith('#'):
            continue

        # 解析指标行：metric_name{labels} value
        match = re.match(r'([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]*)\}\s+([0-9.e+-]+)', line)

        if match:
            metric_name = match.group(1)
            labels_str = match.group(2)
            value = float(match.group(3))

            # 解析标签
            labels = {}
            for label_pair in labels_str.split(','):
                if '=' in label_pair:
                    key, val = label_pair.split('=', 1)
                    labels[key.strip()] = val.strip('"')

            # 添加到结果
            if metric_name not in metrics:
                metrics[metric_name] = []
            metrics[metric_name].append((labels, value))

        # 解析无标签的指标行：metric_name value
        else:
            match = re.match(r'([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([0-9.e+-]+)', line)
            if match:
                metric_name = match.group(1)
                value = float(match.group(2))

                if metric_name not in metrics:
                    metrics[metric_name] = []
                metrics[metric_name].append(({}, value))

    return metrics

# 示例使用
if __name__ == "__main__":
    import requests

    # 获取 Metrics
    response = requests.get("http://localhost:9091/metrics")
    metrics = parse_prometheus_metrics(response.text)

    # 显示解析结果
    print(f"解析到 {len(metrics)} 个指标")

    # 查找特定指标
    if "milvus_proxy_search_vectors_count" in metrics:
        print("\n查询向量数统计:")
        for labels, value in metrics["milvus_proxy_search_vectors_count"]:
            print(f"  {labels}: {value}")
```

---

## 6. 在 RAG 系统中的应用

### 6.1 监控向量检索性能

```python
"""
监控 RAG 系统中的向量检索性能
"""

import requests
from typing import Dict

def get_milvus_search_metrics() -> Dict:
    """获取 Milvus 查询性能指标"""
    response = requests.get("http://localhost:9091/metrics")
    metrics = parse_prometheus_metrics(response.text)

    result = {}

    # 查询总次数
    if "milvus_proxy_search_vectors_count" in metrics:
        total_searches = sum(value for _, value in metrics["milvus_proxy_search_vectors_count"])
        result["total_searches"] = total_searches

    # 查询失败次数
    if "milvus_proxy_search_failed_count" in metrics:
        failed_searches = sum(value for _, value in metrics["milvus_proxy_search_failed_count"])
        result["failed_searches"] = failed_searches
        result["success_rate"] = (total_searches - failed_searches) / total_searches * 100

    # 查询延迟（需要从 histogram 计算）
    if "milvus_proxy_search_latency_milliseconds_sum" in metrics:
        latency_sum = sum(value for _, value in metrics["milvus_proxy_search_latency_milliseconds_sum"])
        latency_count = sum(value for _, value in metrics["milvus_proxy_search_latency_milliseconds_count"])
        result["avg_latency_ms"] = latency_sum / latency_count if latency_count > 0 else 0

    return result

# 使用示例
if __name__ == "__main__":
    metrics = get_milvus_search_metrics()
    print("RAG 系统检索性能:")
    print(f"  总查询次数: {metrics.get('total_searches', 0)}")
    print(f"  失败次数: {metrics.get('failed_searches', 0)}")
    print(f"  成功率: {metrics.get('success_rate', 0):.2f}%")
    print(f"  平均延迟: {metrics.get('avg_latency_ms', 0):.2f} ms")
```

---

## 7. 最佳实践

### 7.1 采集频率选择

| 场景 | 推荐频率 | 原因 |
|------|---------|------|
| 生产环境 | 15-30 秒 | 平衡精度和开销 |
| 开发环境 | 30-60 秒 | 降低资源消耗 |
| 高负载系统 | 10-15 秒 | 快速发现问题 |
| 低负载系统 | 30-60 秒 | 节省存储空间 |

---

### 7.2 数据保留策略

```yaml
# Prometheus 配置
global:
  # 数据保留时间
  storage.tsdb.retention.time: 30d

  # 数据保留大小
  storage.tsdb.retention.size: 50GB
```

**推荐策略：**
- **短期数据**（1-7 天）：高精度，用于实时监控
- **中期数据**（7-30 天）：中等精度，用于趋势分析
- **长期数据**（30+ 天）：低精度，用于容量规划

---

### 7.3 标签设计原则

**好的标签设计：**
```promql
# ✅ 使用有意义的标签
milvus_proxy_search_vectors_count{
  collection="knowledge_base",
  env="production",
  region="us-west"
}
```

**避免的标签设计：**
```promql
# ❌ 标签值过多（高基数）
milvus_proxy_search_vectors_count{
  user_id="12345",  # 每个用户一个标签值
  request_id="abc"  # 每个请求一个标签值
}
```

**原则：**
- ✅ 标签值数量有限（< 100）
- ✅ 标签有业务含义
- ❌ 避免高基数标签（如 user_id、request_id）

---

## 小结

**Prometheus 指标采集的核心要点：**

1. **Pull 模型**：Prometheus 主动拉取，应用被动暴露
2. **三种指标类型**：Counter（累计）、Gauge（瞬时）、Histogram（分布）
3. **关键指标**：性能（QPS、延迟）、资源（CPU、内存）、业务（Collection 状态）
4. **配置要点**：采集频率、服务发现、重标签
5. **最佳实践**：合理的采集频率、数据保留策略、标签设计

**在 RAG 系统中：**
- 监控向量检索的性能和质量
- 追踪 Embedding 生成的耗时
- 观察缓存命中率和资源使用

---

**下一步：** [03_核心概念_02_健康检查机制](./03_核心概念_02_健康检查机制.md)
