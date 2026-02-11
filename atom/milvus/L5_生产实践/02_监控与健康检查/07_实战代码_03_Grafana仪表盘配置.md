# 实战代码3：Grafana 仪表盘配置

> 使用 Python 自动创建 Milvus 监控仪表盘

---

## 场景说明

本示例演示如何使用 Grafana API 自动创建 Milvus 监控仪表盘，包括：
- QPS 监控面板
- 延迟监控面板
- 资源使用监控面板
- 错误率监控面板

---

## 完整代码

**create_grafana_dashboard.py：**

```python
"""
自动创建 Milvus Grafana 监控仪表盘
"""

import requests
import json
from typing import Dict, List

class GrafanaDashboardCreator:
    def __init__(self, grafana_url: str, api_key: str):
        """
        初始化 Grafana 仪表盘创建器

        Args:
            grafana_url: Grafana URL (如 http://localhost:3000)
            api_key: Grafana API Key
        """
        self.grafana_url = grafana_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def create_milvus_dashboard(self):
        """创建 Milvus 监控仪表盘"""
        dashboard = {
            "dashboard": {
                "title": "Milvus Monitoring",
                "tags": ["milvus", "monitoring", "production"],
                "timezone": "browser",
                "schemaVersion": 16,
                "version": 0,
                "refresh": "10s",

                # 仪表盘变量
                "templating": {
                    "list": [
                        {
                            "name": "instance",
                            "type": "query",
                            "datasource": "Prometheus",
                            "query": "label_values(milvus_proxy_search_vectors_count, instance)",
                            "refresh": 1,
                            "multi": False,
                            "includeAll": False
                        }
                    ]
                },

                # 面板配置
                "panels": [
                    self._create_qps_panel(1, 0, 0),
                    self._create_latency_panel(2, 12, 0),
                    self._create_error_rate_panel(3, 0, 8),
                    self._create_memory_panel(4, 6, 8),
                    self._create_cpu_panel(5, 12, 8),
                    self._create_disk_panel(6, 18, 8),
                ]
            },
            "overwrite": True
        }

        # 创建仪表盘
        response = requests.post(
            f"{self.grafana_url}/api/dashboards/db",
            headers=self.headers,
            data=json.dumps(dashboard)
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ 仪表盘创建成功")
            print(f"   URL: {self.grafana_url}{result['url']}")
            return result
        else:
            print(f"❌ 仪表盘创建失败: {response.status_code}")
            print(f"   错误: {response.text}")
            return None

    def _create_qps_panel(self, panel_id: int, x: int, y: int) -> Dict:
        """创建 QPS 面板"""
        return {
            "id": panel_id,
            "title": "QPS (Queries Per Second)",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 12, "x": x, "y": y},
            "targets": [
                {
                    "expr": "sum(rate(milvus_proxy_search_vectors_count{instance=\"$instance\"}[5m]))",
                    "legendFormat": "Total QPS",
                    "refId": "A"
                },
                {
                    "expr": "sum by (collection) (rate(milvus_proxy_search_vectors_count{instance=\"$instance\"}[5m]))",
                    "legendFormat": "{{collection}}",
                    "refId": "B"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "reqps",
                    "color": {"mode": "palette-classic"},
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "smooth",
                        "fillOpacity": 10
                    }
                }
            }
        }

    def _create_latency_panel(self, panel_id: int, x: int, y: int) -> Dict:
        """创建延迟面板"""
        return {
            "id": panel_id,
            "title": "Search Latency",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 12, "x": x, "y": y},
            "targets": [
                {
                    "expr": "histogram_quantile(0.50, rate(milvus_proxy_search_latency_milliseconds_bucket{instance=\"$instance\"}[5m]))",
                    "legendFormat": "P50",
                    "refId": "A"
                },
                {
                    "expr": "histogram_quantile(0.95, rate(milvus_proxy_search_latency_milliseconds_bucket{instance=\"$instance\"}[5m]))",
                    "legendFormat": "P95",
                    "refId": "B"
                },
                {
                    "expr": "histogram_quantile(0.99, rate(milvus_proxy_search_latency_milliseconds_bucket{instance=\"$instance\"}[5m]))",
                    "legendFormat": "P99",
                    "refId": "C"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "ms",
                    "color": {"mode": "palette-classic"},
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "smooth"
                    },
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

    def _create_error_rate_panel(self, panel_id: int, x: int, y: int) -> Dict:
        """创建错误率面板"""
        return {
            "id": panel_id,
            "title": "Error Rate",
            "type": "gauge",
            "gridPos": {"h": 8, "w": 6, "x": x, "y": y},
            "targets": [
                {
                    "expr": "(rate(milvus_proxy_search_failed_count{instance=\"$instance\"}[5m]) / rate(milvus_proxy_search_vectors_count{instance=\"$instance\"}[5m])) * 100",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "green"},
                            {"value": 1, "color": "yellow"},
                            {"value": 5, "color": "red"}
                        ]
                    }
                }
            },
            "options": {
                "showThresholdLabels": True,
                "showThresholdMarkers": True
            }
        }

    def _create_memory_panel(self, panel_id: int, x: int, y: int) -> Dict:
        """创建内存使用面板"""
        return {
            "id": panel_id,
            "title": "Memory Usage",
            "type": "gauge",
            "gridPos": {"h": 8, "w": 6, "x": x, "y": y},
            "targets": [
                {
                    "expr": "(process_resident_memory_bytes{job=\"milvus\", instance=\"$instance\"} / (16 * 1024 * 1024 * 1024)) * 100",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100,
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

    def _create_cpu_panel(self, panel_id: int, x: int, y: int) -> Dict:
        """创建 CPU 使用面板"""
        return {
            "id": panel_id,
            "title": "CPU Usage",
            "type": "gauge",
            "gridPos": {"h": 8, "w": 6, "x": x, "y": y},
            "targets": [
                {
                    "expr": "rate(process_cpu_seconds_total{job=\"milvus\", instance=\"$instance\"}[5m]) * 100",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100,
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

    def _create_disk_panel(self, panel_id: int, x: int, y: int) -> Dict:
        """创建磁盘使用面板"""
        return {
            "id": panel_id,
            "title": "Disk Usage",
            "type": "gauge",
            "gridPos": {"h": 8, "w": 6, "x": x, "y": y},
            "targets": [
                {
                    "expr": "(milvus_datanode_storage_size_bytes{instance=\"$instance\"} / milvus_datanode_storage_capacity_bytes{instance=\"$instance\"}) * 100",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "green"},
                            {"value": 80, "color": "yellow"},
                            {"value": 95, "color": "red"}
                        ]
                    }
                }
            }
        }

def create_api_key(grafana_url: str, admin_user: str, admin_password: str) -> str:
    """
    创建 Grafana API Key

    Args:
        grafana_url: Grafana URL
        admin_user: 管理员用户名
        admin_password: 管理员密码

    Returns:
        API Key
    """
    response = requests.post(
        f"{grafana_url}/api/auth/keys",
        auth=(admin_user, admin_password),
        json={
            "name": "milvus-monitoring",
            "role": "Admin"
        }
    )

    if response.status_code == 200:
        api_key = response.json()["key"]
        print(f"✅ API Key 创建成功")
        return api_key
    else:
        print(f"❌ API Key 创建失败: {response.status_code}")
        print(f"   错误: {response.text}")
        return None

def main():
    """主函数"""
    print("=" * 60)
    print("Grafana 仪表盘自动创建脚本")
    print("=" * 60)

    # 配置
    GRAFANA_URL = "http://localhost:3000"
    ADMIN_USER = "admin"
    ADMIN_PASSWORD = "admin"

    # 1. 创建 API Key
    print("\n=== 创建 API Key ===")
    api_key = create_api_key(GRAFANA_URL, ADMIN_USER, ADMIN_PASSWORD)

    if not api_key:
        print("❌ 无法创建 API Key，退出")
        return

    # 2. 创建仪表盘
    print("\n=== 创建 Milvus 监控仪表盘 ===")
    creator = GrafanaDashboardCreator(GRAFANA_URL, api_key)
    result = creator.create_milvus_dashboard()

    if result:
        print("\n" + "=" * 60)
        print("仪表盘创建成功！")
        print("=" * 60)
        print(f"\n访问地址: {GRAFANA_URL}{result['url']}")
        print(f"用户名/密码: {ADMIN_USER}/{ADMIN_PASSWORD}")
    else:
        print("\n❌ 仪表盘创建失败")

if __name__ == "__main__":
    main()
```

---

## 使用方法

### 步骤1：安装依赖

```bash
pip install requests
```

---

### 步骤2：运行脚本

```bash
python create_grafana_dashboard.py
```

---

### 步骤3：访问仪表盘

```bash
# 打开浏览器访问
open http://localhost:3000
```

---

## 预期输出

```
============================================================
Grafana 仪表盘自动创建脚本
============================================================

=== 创建 API Key ===
✅ API Key 创建成功

=== 创建 Milvus 监控仪表盘 ===
✅ 仪表盘创建成功
   URL: http://localhost:3000/d/milvus-monitoring

============================================================
仪表盘创建成功！
============================================================

访问地址: http://localhost:3000/d/milvus-monitoring
用户名/密码: admin/admin
```

---

## 仪表盘效果

创建的仪表盘包含以下面板：

1. **QPS 面板**（左上）
   - 显示总 QPS 和按 Collection 分组的 QPS
   - 折线图，实时更新

2. **延迟面板**（右上）
   - 显示 P50、P95、P99 延迟
   - 颜色阈值：绿色（<100ms）、黄色（100-500ms）、红色（>500ms）

3. **错误率面板**（左中）
   - 仪表盘显示当前错误率
   - 阈值：绿色（<1%）、黄色（1-5%）、红色（>5%）

4. **内存使用面板**（中中）
   - 仪表盘显示内存使用率
   - 阈值：绿色（<70%）、黄色（70-90%）、红色（>90%）

5. **CPU 使用面板**（右中）
   - 仪表盘显示 CPU 使用率
   - 阈值：绿色（<70%）、黄色（70-90%）、红色（>90%）

6. **磁盘使用面板**（右下）
   - 仪表盘显示磁盘使用率
   - 阈值：绿色（<80%）、黄色（80-95%）、红色（>95%）

---

## 在 RAG 系统中的应用

### 添加 RAG 自定义面板

```python
def _create_rag_cache_hit_rate_panel(self, panel_id: int, x: int, y: int) -> Dict:
    """创建 RAG 缓存命中率面板"""
    return {
        "id": panel_id,
        "title": "RAG Cache Hit Rate",
        "type": "gauge",
        "gridPos": {"h": 8, "w": 6, "x": x, "y": y},
        "targets": [
            {
                "expr": "(rate(rag_cache_hits_total[5m]) / rate(rag_cache_requests_total[5m])) * 100",
                "refId": "A"
            }
        ],
        "fieldConfig": {
            "defaults": {
                "unit": "percent",
                "min": 0,
                "max": 100,
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"value": 0, "color": "red"},
                        {"value": 50, "color": "yellow"},
                        {"value": 80, "color": "green"}
                    ]
                }
            }
        }
    }

def _create_rag_embedding_duration_panel(self, panel_id: int, x: int, y: int) -> Dict:
    """创建 RAG Embedding 生成耗时面板"""
    return {
        "id": panel_id,
        "title": "Embedding Generation Duration",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": x, "y": y},
        "targets": [
            {
                "expr": "histogram_quantile(0.95, rate(rag_embedding_duration_seconds_bucket[5m]))",
                "legendFormat": "P95",
                "refId": "A"
            }
        ],
        "fieldConfig": {
            "defaults": {
                "unit": "s",
                "color": {"mode": "palette-classic"}
            }
        }
    }
```

---

## 小结

本实战示例展示了：

1. **自动化创建仪表盘**：使用 Python 和 Grafana API
2. **完整的监控面板**：QPS、延迟、错误率、资源使用
3. **可视化配置**：面板类型、阈值、颜色
4. **RAG 集成**：添加自定义 RAG 指标面板

**关键要点：**
- 使用 Grafana API 自动化创建仪表盘
- 配置合理的阈值和颜色
- 为不同指标选择合适的面板类型
- 支持变量动态过滤

---

**下一步：** [08_面试必问](./08_面试必问.md)
