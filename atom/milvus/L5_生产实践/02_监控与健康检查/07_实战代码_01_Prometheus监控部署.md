# 实战代码1：Prometheus 监控部署

> 完整的 Prometheus + Milvus 监控部署实战

---

## 场景说明

本示例演示如何使用 Docker Compose 部署完整的 Milvus 监控栈：
- Milvus Standalone（向量数据库）
- Prometheus（指标采集和存储）
- Grafana（可视化仪表盘）
- AlertManager（告警管理）

---

## 完整代码

### 1. 项目结构

```bash
milvus-monitoring/
├── docker-compose.yml          # Docker Compose 配置
├── prometheus/
│   ├── prometheus.yml          # Prometheus 配置
│   └── alerts.yml              # 告警规则
├── alertmanager/
│   └── alertmanager.yml        # AlertManager 配置
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/
│   │   │   └── prometheus.yml  # 数据源配置
│   │   └── dashboards/
│   │       ├── dashboard.yml   # 仪表盘配置
│   │       └── milvus.json     # Milvus 仪表盘
└── scripts/
    └── test_monitoring.py      # 监控测试脚本
```

---

### 2. Docker Compose 配置

**docker-compose.yml：**

```yaml
version: '3.8'

services:
  # ===== Milvus 服务 =====
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - etcd-data:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio-data:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  milvus:
    image: milvusdb/milvus:v2.3.3
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - milvus-data:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"  # Metrics 端点
    depends_on:
      - etcd
      - minio
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 90s

  # ===== 监控服务 =====
  prometheus:
    image: prom/prometheus:v2.45.0
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./prometheus/alerts.yml:/etc/prometheus/alerts.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    ports:
      - "9090:9090"
    depends_on:
      - milvus
    restart: unless-stopped

  alertmanager:
    image: prom/alertmanager:v0.26.0
    volumes:
      - ./alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager-data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
    ports:
      - "9093:9093"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:10.0.3
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning
      - grafana-data:/var/lib/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  etcd-data:
  minio-data:
  milvus-data:
  prometheus-data:
  alertmanager-data:
  grafana-data:
```

---

### 3. Prometheus 配置

**prometheus/prometheus.yml：**

```yaml
# Prometheus 全局配置
global:
  scrape_interval: 15s      # 默认采集间隔
  evaluation_interval: 15s  # 告警规则评估间隔
  scrape_timeout: 10s       # 采集超时时间

  # 外部标签（用于联邦和远程存储）
  external_labels:
    cluster: 'milvus-prod'
    environment: 'production'

# 告警管理器配置
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

# 告警规则文件
rule_files:
  - 'alerts.yml'

# 采集目标配置
scrape_configs:
  # Milvus 监控
  - job_name: 'milvus'
    static_configs:
      - targets: ['milvus:9091']
        labels:
          instance: 'milvus-standalone'
          component: 'milvus'

    # 指标重标签
    metric_relabel_configs:
      # 只保留 milvus_ 开头的指标
      - source_labels: [__name__]
        regex: 'milvus_.*'
        action: keep

      # 删除不需要的标签
      - regex: 'pod_template_hash'
        action: labeldrop

  # Prometheus 自身监控
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # AlertManager 监控
  - job_name: 'alertmanager'
    static_configs:
      - targets: ['alertmanager:9093']
```

---

### 4. 告警规则配置

**prometheus/alerts.yml：**

```yaml
groups:
  # Milvus 性能告警
  - name: milvus_performance
    interval: 30s
    rules:
      # 高查询延迟告警
      - alert: HighSearchLatency
        expr: |
          histogram_quantile(0.95,
            rate(milvus_proxy_search_latency_milliseconds_bucket[5m])
          ) > 500
        for: 5m
        labels:
          severity: warning
          component: milvus
        annotations:
          summary: "Milvus search latency is high"
          description: "P95 search latency is {{ $value }}ms (threshold: 500ms)"

      # 极高查询延迟告警
      - alert: CriticalSearchLatency
        expr: |
          histogram_quantile(0.95,
            rate(milvus_proxy_search_latency_milliseconds_bucket[5m])
          ) > 1000
        for: 2m
        labels:
          severity: critical
          component: milvus
        annotations:
          summary: "Milvus search latency is critically high"
          description: "P95 search latency is {{ $value }}ms (threshold: 1000ms)"

      # 高错误率告警
      - alert: HighErrorRate
        expr: |
          (
            rate(milvus_proxy_search_failed_count[5m])
            /
            rate(milvus_proxy_search_vectors_count[5m])
          ) * 100 > 5
        for: 5m
        labels:
          severity: warning
          component: milvus
        annotations:
          summary: "Milvus error rate is high"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"

  # Milvus 资源告警
  - name: milvus_resources
    interval: 30s
    rules:
      # 高内存使用告警
      - alert: HighMemoryUsage
        expr: |
          (
            process_resident_memory_bytes{job="milvus"}
            /
            (16 * 1024 * 1024 * 1024)
          ) * 100 > 80
        for: 5m
        labels:
          severity: warning
          component: milvus
        annotations:
          summary: "Milvus memory usage is high"
          description: "Memory usage is {{ $value | humanizePercentage }} (threshold: 80%)"

      # 极高内存使用告警
      - alert: CriticalMemoryUsage
        expr: |
          (
            process_resident_memory_bytes{job="milvus"}
            /
            (16 * 1024 * 1024 * 1024)
          ) * 100 > 90
        for: 2m
        labels:
          severity: critical
          component: milvus
        annotations:
          summary: "Milvus memory usage is critically high"
          description: "Memory usage is {{ $value | humanizePercentage }} (threshold: 90%)"

      # 高 CPU 使用告警
      - alert: HighCPUUsage
        expr: |
          rate(process_cpu_seconds_total{job="milvus"}[5m]) * 100 > 80
        for: 5m
        labels:
          severity: warning
          component: milvus
        annotations:
          summary: "Milvus CPU usage is high"
          description: "CPU usage is {{ $value | humanizePercentage }} (threshold: 80%)"

  # Milvus 可用性告警
  - name: milvus_availability
    interval: 30s
    rules:
      # Milvus 服务不可用
      - alert: MilvusDown
        expr: up{job="milvus"} == 0
        for: 1m
        labels:
          severity: critical
          component: milvus
        annotations:
          summary: "Milvus is down"
          description: "Milvus instance {{ $labels.instance }} is down"

      # Milvus QPS 异常低
      - alert: LowQPS
        expr: |
          rate(milvus_proxy_search_vectors_count[5m]) < 1
          and
          rate(milvus_proxy_search_vectors_count[5m] offset 1h) > 10
        for: 10m
        labels:
          severity: warning
          component: milvus
        annotations:
          summary: "Milvus QPS is abnormally low"
          description: "Current QPS is {{ $value }}, was {{ $value offset 1h }} 1 hour ago"
```

---

### 5. AlertManager 配置

**alertmanager/alertmanager.yml：**

```yaml
global:
  resolve_timeout: 5m

# 告警路由
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'

  routes:
    # Critical 告警立即发送
    - match:
        severity: critical
      receiver: 'critical'
      continue: true

    # Warning 告警延迟发送
    - match:
        severity: warning
      receiver: 'warning'

# 告警接收器
receivers:
  - name: 'default'
    webhook_configs:
      - url: 'http://localhost:5001/webhook'

  - name: 'critical'
    # 邮件通知
    email_configs:
      - to: 'ops@company.com'
        from: 'alertmanager@company.com'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'alertmanager@company.com'
        auth_password: 'your_password'
        headers:
          Subject: '🚨 Critical Alert: {{ .GroupLabels.alertname }}'

    # Slack 通知
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#alerts'
        title: '🚨 Critical Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ .Annotations.description }}\n{{ end }}'

  - name: 'warning'
    # 邮件通知
    email_configs:
      - to: 'ops@company.com'
        from: 'alertmanager@company.com'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'alertmanager@company.com'
        auth_password: 'your_password'
        headers:
          Subject: '⚠️ Warning Alert: {{ .GroupLabels.alertname }}'

# 告警抑制规则
inhibit_rules:
  # 如果 Milvus 服务不可用，抑制其他所有告警
  - source_match:
      alertname: 'MilvusDown'
    target_match_re:
      alertname: '.*'
    equal: ['instance']
```

---

### 6. Grafana 数据源配置

**grafana/provisioning/datasources/prometheus.yml：**

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
    jsonData:
      timeInterval: "15s"
      queryTimeout: "60s"
```

---

### 7. 监控测试脚本

**scripts/test_monitoring.py：**

```python
"""
Milvus 监控测试脚本
生成测试负载，验证监控系统是否正常工作
"""

import time
import random
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import requests

def setup_milvus():
    """初始化 Milvus 连接和 Collection"""
    print("=== 连接 Milvus ===")
    connections.connect(host="localhost", port="19530")

    # 创建 Collection
    collection_name = "test_monitoring"

    # 定义 Schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
    ]
    schema = CollectionSchema(fields=fields, description="Test collection for monitoring")

    # 创建 Collection（如果不存在）
    from pymilvus import utility
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        collection.drop()

    collection = Collection(collection_name, schema)

    # 创建索引
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)

    # 加载 Collection
    collection.load()

    print(f"✅ Collection '{collection_name}' 创建并加载成功")
    return collection

def insert_test_data(collection, num_vectors=1000):
    """插入测试数据"""
    print(f"\n=== 插入 {num_vectors} 个测试向量 ===")

    # 生成随机向量
    vectors = [[random.random() for _ in range(128)] for _ in range(num_vectors)]

    # 插入数据
    start_time = time.time()
    collection.insert([vectors])
    collection.flush()
    duration = time.time() - start_time

    print(f"✅ 插入完成，耗时: {duration:.2f} 秒")

def run_search_load(collection, duration_seconds=60, qps=10):
    """运行查询负载"""
    print(f"\n=== 运行查询负载 ===")
    print(f"持续时间: {duration_seconds} 秒")
    print(f"目标 QPS: {qps}")

    start_time = time.time()
    query_count = 0
    error_count = 0

    while time.time() - start_time < duration_seconds:
        try:
            # 生成随机查询向量
            query_vector = [[random.random() for _ in range(128)]]

            # 执行查询
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = collection.search(
                data=query_vector,
                anns_field="embedding",
                param=search_params,
                limit=10
            )

            query_count += 1

            # 控制 QPS
            time.sleep(1.0 / qps)

        except Exception as e:
            error_count += 1
            print(f"❌ 查询失败: {e}")

    actual_duration = time.time() - start_time
    actual_qps = query_count / actual_duration

    print(f"\n查询统计:")
    print(f"  总查询数: {query_count}")
    print(f"  失败数: {error_count}")
    print(f"  实际 QPS: {actual_qps:.2f}")
    print(f"  成功率: {(query_count - error_count) / query_count * 100:.2f}%")

def check_prometheus_metrics():
    """检查 Prometheus 是否采集到指标"""
    print("\n=== 检查 Prometheus 指标 ===")

    try:
        # 查询 Milvus QPS
        response = requests.get(
            "http://localhost:9090/api/v1/query",
            params={"query": "rate(milvus_proxy_search_vectors_count[1m])"}
        )

        if response.status_code == 200:
            data = response.json()
            if data['data']['result']:
                qps = float(data['data']['result'][0]['value'][1])
                print(f"✅ Prometheus 采集正常")
                print(f"   当前 QPS: {qps:.2f}")
            else:
                print("⚠️ Prometheus 未采集到 QPS 指标")
        else:
            print(f"❌ Prometheus 查询失败: {response.status_code}")

    except Exception as e:
        print(f"❌ 无法连接 Prometheus: {e}")

def check_grafana():
    """检查 Grafana 是否可访问"""
    print("\n=== 检查 Grafana ===")

    try:
        response = requests.get("http://localhost:3000/api/health")
        if response.status_code == 200:
            print("✅ Grafana 运行正常")
            print("   访问地址: http://localhost:3000")
            print("   用户名/密码: admin/admin")
        else:
            print(f"❌ Grafana 健康检查失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 无法连接 Grafana: {e}")

def main():
    """主函数"""
    print("=" * 60)
    print("Milvus 监控测试脚本")
    print("=" * 60)

    # 1. 设置 Milvus
    collection = setup_milvus()

    # 2. 插入测试数据
    insert_test_data(collection, num_vectors=10000)

    # 3. 运行查询负载（60 秒，10 QPS）
    run_search_load(collection, duration_seconds=60, qps=10)

    # 4. 检查 Prometheus
    check_prometheus_metrics()

    # 5. 检查 Grafana
    check_grafana()

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print("\n下一步:")
    print("1. 访问 Prometheus: http://localhost:9090")
    print("2. 访问 Grafana: http://localhost:3000")
    print("3. 查看 Milvus 监控仪表盘")

if __name__ == "__main__":
    main()
```

---

## 部署步骤

### 步骤1：准备配置文件

```bash
# 创建项目目录
mkdir -p milvus-monitoring/{prometheus,alertmanager,grafana/provisioning/{datasources,dashboards},scripts}
cd milvus-monitoring

# 创建配置文件（复制上面的内容）
# - docker-compose.yml
# - prometheus/prometheus.yml
# - prometheus/alerts.yml
# - alertmanager/alertmanager.yml
# - grafana/provisioning/datasources/prometheus.yml
# - scripts/test_monitoring.py
```

---

### 步骤2：启动服务

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f milvus
```

---

### 步骤3：验证部署

```bash
# 1. 验证 Milvus Metrics 端点
curl http://localhost:9091/metrics | head -20

# 2. 验证 Prometheus
open http://localhost:9090

# 3. 验证 Grafana
open http://localhost:3000
```

---

### 步骤4：运行测试脚本

```bash
# 安装依赖
pip install pymilvus requests

# 运行测试
python scripts/test_monitoring.py
```

---

## 预期输出

```
============================================================
Milvus 监控测试脚本
============================================================
=== 连接 Milvus ===
✅ Collection 'test_monitoring' 创建并加载成功

=== 插入 10000 个测试向量 ===
✅ 插入完成，耗时: 2.34 秒

=== 运行查询负载 ===
持续时间: 60 秒
目标 QPS: 10

查询统计:
  总查询数: 600
  失败数: 0
  实际 QPS: 10.02
  成功率: 100.00%

=== 检查 Prometheus 指标 ===
✅ Prometheus 采集正常
   当前 QPS: 10.15

=== 检查 Grafana ===
✅ Grafana 运行正常
   访问地址: http://localhost:3000
   用户名/密码: admin/admin

============================================================
测试完成！
============================================================

下一步:
1. 访问 Prometheus: http://localhost:9090
2. 访问 Grafana: http://localhost:3000
3. 查看 Milvus 监控仪表盘
```

---

## 在 RAG 系统中的应用

### 添加自定义 RAG 指标

```python
"""
为 RAG 系统添加自定义监控指标
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# 定义 RAG 指标
rag_search_total = Counter(
    'rag_search_total',
    'Total number of RAG searches',
    ['collection', 'status']
)

rag_search_duration = Histogram(
    'rag_search_duration_seconds',
    'RAG search duration in seconds',
    ['collection'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

rag_cache_hits = Counter(
    'rag_cache_hits_total',
    'Total number of cache hits',
    ['collection']
)

rag_embedding_duration = Histogram(
    'rag_embedding_duration_seconds',
    'Embedding generation duration in seconds',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
)

def rag_search_with_monitoring(collection_name, query_text, top_k=10):
    """带监控的 RAG 检索"""
    start_time = time.time()

    try:
        # 1. 生成 Embedding
        embedding_start = time.time()
        query_vector = generate_embedding(query_text)
        rag_embedding_duration.observe(time.time() - embedding_start)

        # 2. 向量检索
        results = milvus_search(collection_name, query_vector, top_k)

        # 3. 记录成功指标
        duration = time.time() - start_time
        rag_search_total.labels(collection=collection_name, status='success').inc()
        rag_search_duration.labels(collection=collection_name).observe(duration)

        return results

    except Exception as e:
        # 记录失败指标
        duration = time.time() - start_time
        rag_search_total.labels(collection=collection_name, status='failure').inc()
        rag_search_duration.labels(collection=collection_name).observe(duration)
        raise e

# 启动指标服务
if __name__ == "__main__":
    start_http_server(8000)
    print("RAG metrics server started on :8000")
```

**添加到 Prometheus 配置：**

```yaml
scrape_configs:
  - job_name: 'rag-app'
    static_configs:
      - targets: ['rag-app:8000']
```

---

## 小结

本实战示例展示了：

1. **完整的监控栈部署**：Prometheus + Grafana + AlertManager
2. **Milvus 指标采集**：自动采集所有 Milvus 指标
3. **告警规则配置**：性能、资源、可用性告警
4. **监控测试**：自动化测试脚本验证监控系统
5. **RAG 集成**：为 RAG 系统添加自定义指标

**关键要点：**
- 使用 Docker Compose 简化部署
- 配置合理的告警规则和阈值
- 测试监控系统是否正常工作
- 为 RAG 应用添加自定义指标

---

**下一步：** [07_实战代码_02_健康检查实现](./07_实战代码_02_健康检查实现.md)
