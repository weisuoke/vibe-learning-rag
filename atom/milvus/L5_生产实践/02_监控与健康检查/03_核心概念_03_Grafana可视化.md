# 核心概念3：Grafana 可视化

> 理解如何使用 Grafana 将 Prometheus 指标转换为直观的监控仪表盘

---

## 什么是 Grafana 可视化？

**Grafana 可视化是通过连接 Prometheus 数据源，使用 PromQL 查询指标数据，将数字转换为图表（折线图、柱状图、热力图），构建监控仪表盘，实现系统状态的可视化展示和实时监控。**

---

## 1. Grafana 的核心概念

### 1.1 数据源（Data Source）

**定义：** Grafana 从哪里获取数据

**Prometheus 数据源配置：**

```yaml
# Grafana 配置文件：provisioning/datasources/prometheus.yml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
```

**通过 UI 配置：**
1. 登录 Grafana（默认 http://localhost:3000）
2. Configuration → Data Sources → Add data source
3. 选择 Prometheus
4. 填写 URL：`http://prometheus:9090`
5. 点击 "Save & Test"

---

### 1.2 仪表盘（Dashboard）

**定义：** 多个可视化面板的集合

**仪表盘结构：**

```
Dashboard（仪表盘）
├── Row 1（行）
│   ├── Panel 1（面板）：QPS 折线图
│   └── Panel 2（面板）：延迟热力图
├── Row 2（行）
│   ├── Panel 3（面板）：CPU 使用率
│   └── Panel 4（面板）：内存使用率
└── Variables（变量）
    ├── $instance：选择实例
    └── $collection：选择 Collection
```

---

### 1.3 面板（Panel）

**定义：** 单个可视化图表

**面板类型：**

| 类型 | 用途 | 适用指标 |
|------|------|---------|
| **Time series**（时间序列） | 显示指标随时间变化 | QPS、延迟、CPU |
| **Gauge**（仪表盘） | 显示当前值和阈值 | 内存使用率、磁盘使用率 |
| **Stat**（统计值） | 显示单个数值 | 总查询数、错误率 |
| **Table**（表格） | 显示多维度数据 | Collection 列表、实例状态 |
| **Heatmap**（热力图） | 显示分布密度 | 延迟分布 |
| **Bar chart**（柱状图） | 对比不同维度 | 各 Collection 的 QPS |

---

### 1.4 查询（Query）

**定义：** 使用 PromQL 从 Prometheus 获取数据

**基础查询示例：**

```promql
# 查询 QPS
rate(milvus_proxy_search_vectors_count[5m])

# 查询 P95 延迟
histogram_quantile(0.95,
  rate(milvus_proxy_search_latency_milliseconds_bucket[5m])
)

# 查询内存使用率
process_resident_memory_bytes / (16 * 1024 * 1024 * 1024) * 100
```

---

## 2. 创建 Milvus 监控仪表盘

### 2.1 仪表盘设计原则

**分层设计：**

```
Level 1: 概览仪表盘（Overview）
- 整体健康状态
- 关键指标（QPS、延迟、错误率）
- 资源使用（CPU、内存、磁盘）

Level 2: 详细仪表盘（Details）
- 各组件详细指标
- Collection 级别指标
- 性能分析

Level 3: 诊断仪表盘（Troubleshooting）
- 错误日志
- 慢查询分析
- 资源瓶颈定位
```

---

### 2.2 概览仪表盘示例

**仪表盘 JSON 配置：**

```json
{
  "dashboard": {
    "title": "Milvus Overview",
    "tags": ["milvus", "overview"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "QPS (Queries Per Second)",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "targets": [
          {
            "expr": "rate(milvus_proxy_search_vectors_count{job=\"milvus\"}[5m])",
            "legendFormat": "{{instance}}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "reqps",
            "color": {"mode": "palette-classic"}
          }
        }
      },
      {
        "id": 2,
        "title": "Search Latency P95",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(milvus_proxy_search_latency_milliseconds_bucket{job=\"milvus\"}[5m]))",
            "legendFormat": "P95",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "ms",
            "color": {"mode": "thresholds"},
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": 0, "color": "green"},
                {"value": 100, "color": "yellow"},
                {"value": 500, "color": "red"}
              ]
            }
          }
        }
      }
    ]
  }
}
```

---

### 2.3 关键指标面板配置

#### 面板1：QPS（每秒查询数）

**PromQL 查询：**

```promql
# 总 QPS
sum(rate(milvus_proxy_search_vectors_count[5m]))

# 按 Collection 分组的 QPS
sum by (collection) (rate(milvus_proxy_search_vectors_count[5m]))

# 按实例分组的 QPS
sum by (instance) (rate(milvus_proxy_search_vectors_count[5m]))
```

**面板配置：**
- 类型：Time series
- 单位：reqps（requests per second）
- 图例：显示 Collection 或实例名称

---

#### 面板2：查询延迟

**PromQL 查询：**

```promql
# P50 延迟
histogram_quantile(0.50, rate(milvus_proxy_search_latency_milliseconds_bucket[5m]))

# P95 延迟
histogram_quantile(0.95, rate(milvus_proxy_search_latency_milliseconds_bucket[5m]))

# P99 延迟
histogram_quantile(0.99, rate(milvus_proxy_search_latency_milliseconds_bucket[5m]))

# 平均延迟
rate(milvus_proxy_search_latency_milliseconds_sum[5m])
/
rate(milvus_proxy_search_latency_milliseconds_count[5m])
```

**面板配置：**
- 类型：Time series
- 单位：ms（毫秒）
- 阈值：
  - 绿色：< 100ms
  - 黄色：100-500ms
  - 红色：> 500ms

---

#### 面板3：错误率

**PromQL 查询：**

```promql
# 错误率（百分比）
(
  rate(milvus_proxy_search_failed_count[5m])
  /
  rate(milvus_proxy_search_vectors_count[5m])
) * 100

# 成功率（百分比）
(
  (rate(milvus_proxy_search_vectors_count[5m]) - rate(milvus_proxy_search_failed_count[5m]))
  /
  rate(milvus_proxy_search_vectors_count[5m])
) * 100
```

**面板配置：**
- 类型：Gauge
- 单位：percent（百分比）
- 阈值：
  - 绿色：< 1%
  - 黄色：1-5%
  - 红色：> 5%

---

#### 面板4：资源使用

**PromQL 查询：**

```promql
# CPU 使用率
rate(process_cpu_seconds_total[5m]) * 100

# 内存使用量（GB）
process_resident_memory_bytes / (1024 * 1024 * 1024)

# 内存使用率（假设总内存 16GB）
process_resident_memory_bytes / (16 * 1024 * 1024 * 1024) * 100

# 磁盘使用率
milvus_datanode_storage_size_bytes / milvus_datanode_storage_capacity_bytes * 100
```

**面板配置：**
- 类型：Gauge 或 Time series
- 单位：percent（百分比）或 bytes
- 阈值：
  - 绿色：< 70%
  - 黄色：70-90%
  - 红色：> 90%

---

## 3. 高级可视化技巧

### 3.1 使用变量（Variables）

**定义：** 动态过滤和切换数据

**创建实例变量：**

```
Name: instance
Type: Query
Data source: Prometheus
Query: label_values(milvus_proxy_search_vectors_count, instance)
Refresh: On Dashboard Load
```

**在查询中使用变量：**

```promql
# 使用 $instance 变量
rate(milvus_proxy_search_vectors_count{instance="$instance"}[5m])

# 使用 $collection 变量
rate(milvus_proxy_search_vectors_count{collection="$collection"}[5m])

# 多选变量（使用正则）
rate(milvus_proxy_search_vectors_count{instance=~"$instance"}[5m])
```

---

### 3.2 告警可视化

**在面板上显示告警状态：**

```json
{
  "alert": {
    "name": "High Search Latency",
    "conditions": [
      {
        "evaluator": {
          "type": "gt",
          "params": [500]
        },
        "query": {
          "model": "A",
          "params": ["5m", "now"]
        }
      }
    ],
    "frequency": "1m",
    "handler": 1,
    "message": "Search latency P95 is above 500ms",
    "noDataState": "no_data",
    "executionErrorState": "alerting"
  }
}
```

---

### 3.3 模板化仪表盘

**使用变量创建通用仪表盘：**

```promql
# 通用查询模板
rate(milvus_${component}_${metric}_count{instance=~"$instance"}[5m])

# 示例：
# component=proxy, metric=search → milvus_proxy_search_count
# component=querynode, metric=query → milvus_querynode_query_count
```

---

## 4. 实战示例：创建 Milvus 仪表盘

### 4.1 使用 Python 自动创建仪表盘

```python
"""
使用 Grafana API 自动创建 Milvus 监控仪表盘
"""

import requests
import json

class GrafanaDashboardCreator:
    def __init__(self, grafana_url: str, api_key: str):
        self.grafana_url = grafana_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def create_milvus_dashboard(self):
        """创建 Milvus 监控仪表盘"""
        dashboard = {
            "dashboard": {
                "title": "Milvus Monitoring",
                "tags": ["milvus", "monitoring"],
                "timezone": "browser",
                "panels": [
                    self._create_qps_panel(),
                    self._create_latency_panel(),
                    self._create_error_rate_panel(),
                    self._create_memory_panel()
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "10s"
            },
            "overwrite": True
        }

        response = requests.post(
            f"{self.grafana_url}/api/dashboards/db",
            headers=self.headers,
            data=json.dumps(dashboard)
        )

        if response.status_code == 200:
            print("✅ Dashboard created successfully")
            print(f"URL: {response.json()['url']}")
        else:
            print(f"❌ Failed to create dashboard: {response.text}")

    def _create_qps_panel(self):
        """创建 QPS 面板"""
        return {
            "id": 1,
            "title": "QPS (Queries Per Second)",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            "targets": [
                {
                    "expr": "sum(rate(milvus_proxy_search_vectors_count[5m]))",
                    "legendFormat": "Total QPS",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "reqps",
                    "color": {"mode": "palette-classic"}
                }
            }
        }

    def _create_latency_panel(self):
        """创建延迟面板"""
        return {
            "id": 2,
            "title": "Search Latency",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
            "targets": [
                {
                    "expr": "histogram_quantile(0.50, rate(milvus_proxy_search_latency_milliseconds_bucket[5m]))",
                    "legendFormat": "P50",
                    "refId": "A"
                },
                {
                    "expr": "histogram_quantile(0.95, rate(milvus_proxy_search_latency_milliseconds_bucket[5m]))",
                    "legendFormat": "P95",
                    "refId": "B"
                },
                {
                    "expr": "histogram_quantile(0.99, rate(milvus_proxy_search_latency_milliseconds_bucket[5m]))",
                    "legendFormat": "P99",
                    "refId": "C"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "ms",
                    "color": {"mode": "palette-classic"}
                }
            }
        }

    def _create_error_rate_panel(self):
        """创建错误率面板"""
        return {
            "id": 3,
            "title": "Error Rate",
            "type": "gauge",
            "gridPos": {"h": 8, "w": 6, "x": 0, "y": 8},
            "targets": [
                {
                    "expr": "(rate(milvus_proxy_search_failed_count[5m]) / rate(milvus_proxy_search_vectors_count[5m])) * 100",
                    "legendFormat": "Error Rate",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "green"},
                            {"value": 1, "color": "yellow"},
                            {"value": 5, "color": "red"}
                        ]
                    }
                }
            }
        }

    def _create_memory_panel(self):
        """创建内存使用面板"""
        return {
            "id": 4,
            "title": "Memory Usage",
            "type": "gauge",
            "gridPos": {"h": 8, "w": 6, "x": 6, "y": 8},
            "targets": [
                {
                    "expr": "process_resident_memory_bytes / (16 * 1024 * 1024 * 1024) * 100",
                    "legendFormat": "Memory Usage",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "green"},
                            {"value": 70, "color": "yellow"},
                            {"value": 90, "color": "red"}
                        ]
                    }
                }
            }
        }

# 使用示例
if __name__ == "__main__":
    creator = GrafanaDashboardCreator(
        grafana_url="http://localhost:3000",
        api_key="your_api_key_here"
    )
    creator.create_milvus_dashboard()
```

---

## 5. 在 RAG 系统中的应用

### 5.1 RAG 系统监控仪表盘

**关键指标：**

```promql
# 1. 向量检索性能
rate(milvus_proxy_search_vectors_count{collection="knowledge_base"}[5m])

# 2. 检索延迟分布
histogram_quantile(0.95, rate(milvus_proxy_search_latency_milliseconds_bucket{collection="knowledge_base"}[5m]))

# 3. 缓存命中率（自定义指标）
rate(rag_cache_hits_total[5m]) / rate(rag_cache_requests_total[5m]) * 100

# 4. Embedding 生成耗时（自定义指标）
rate(rag_embedding_duration_seconds_sum[5m]) / rate(rag_embedding_duration_seconds_count[5m])

# 5. 端到端响应时间（自定义指标）
histogram_quantile(0.95, rate(rag_request_duration_seconds_bucket[5m]))
```

---

## 6. 最佳实践

### 6.1 仪表盘设计原则

**1. 信息层次清晰**
- 最重要的指标放在顶部
- 使用行（Row）分组相关指标
- 避免一个仪表盘放太多面板（< 20 个）

**2. 颜色使用一致**
- 绿色：正常
- 黄色：警告
- 红色：严重

**3. 单位标准化**
- 延迟：ms（毫秒）
- 速率：reqps（每秒请求数）
- 百分比：percent
- 内存：bytes 或 GB

**4. 时间范围合理**
- 实时监控：最近 1 小时
- 趋势分析：最近 24 小时
- 容量规划：最近 30 天

---

### 6.2 性能优化

**1. 减少查询复杂度**
```promql
# ❌ 避免：复杂的嵌套查询
sum(rate(metric1[5m])) / sum(rate(metric2[5m])) * sum(rate(metric3[5m]))

# ✅ 推荐：拆分成多个简单查询
rate(metric1[5m])
rate(metric2[5m])
```

**2. 使用合理的时间范围**
```promql
# ❌ 避免：时间范围过长
rate(metric[1h])  # 计算量大

# ✅ 推荐：使用 5 分钟
rate(metric[5m])  # 计算量小，响应快
```

**3. 限制数据点数量**
- 设置合理的 `Max data points`（默认 1000）
- 使用 `Min interval` 限制最小采样间隔

---

## 小结

**Grafana 可视化的核心要点：**

1. **数据源**：连接 Prometheus 获取指标数据
2. **仪表盘**：分层设计（概览、详细、诊断）
3. **面板类型**：Time series、Gauge、Stat、Table、Heatmap
4. **PromQL 查询**：从 Prometheus 提取和计算指标
5. **变量**：动态过滤和切换数据
6. **告警**：在面板上可视化告警状态

**在 RAG 系统中：**
- 监控向量检索的性能和质量
- 追踪 Embedding 生成的耗时
- 观察缓存命中率和端到端延迟
- 实现全链路可观测性

---

**下一步：** [04_最小可用](./04_最小可用.md)
