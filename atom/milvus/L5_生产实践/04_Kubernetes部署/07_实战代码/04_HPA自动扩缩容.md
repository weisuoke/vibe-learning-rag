# 实战代码 - 场景4：HPA自动扩缩容

## 场景描述

**目标：**配置HPA（Horizontal Pod Autoscaler）实现基于负载的自动扩缩容

**特点：**
- 根据CPU/内存自动调整副本数
- 配置扩缩容策略（快速扩容、缓慢缩容）
- 支持自定义指标扩缩容
- 模拟流量测试扩缩容效果

**适用场景：**
- 流量波动大的RAG系统
- 需要成本优化
- 需要自动应对突发流量

---

## 完整配置脚本

### 1. 基于CPU的HPA配置

```yaml
# hpa-cpu.yaml - 基于CPU的自动扩缩容

apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: milvus-querynode-hpa
  namespace: milvus-prod
spec:
  # 目标Deployment
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: milvus-querynode

  # 副本数范围
  minReplicas: 2   # 最少2个副本
  maxReplicas: 20  # 最多20个副本

  # 扩缩容指标
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70  # CPU超过70%扩容

  # 扩缩容行为
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60  # 扩容前等待60秒
      policies:
      - type: Percent
        value: 50  # 每次扩容50%
        periodSeconds: 60
      - type: Pods
        value: 2   # 或每次扩容2个Pod
        periodSeconds: 60
      selectPolicy: Max  # 选择扩容更多的策略

    scaleDown:
      stabilizationWindowSeconds: 300  # 缩容前等待5分钟
      policies:
      - type: Percent
        value: 10  # 每次缩容10%
        periodSeconds: 60
      - type: Pods
        value: 1   # 或每次缩容1个Pod
        periodSeconds: 60
      selectPolicy: Min  # 选择缩容更少的策略
```

### 2. 基于内存的HPA配置

```yaml
# hpa-memory.yaml - 基于内存的自动扩缩容

apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: milvus-querynode-memory-hpa
  namespace: milvus-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: milvus-querynode

  minReplicas: 2
  maxReplicas: 20

  metrics:
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80  # 内存超过80%扩容

  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Pods
        value: 3  # 内存压力大时快速扩容
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 600  # 内存缩容更保守
      policies:
      - type: Pods
        value: 1
        periodSeconds: 120
```

### 3. 多指标HPA配置

```yaml
# hpa-multi-metrics.yaml - 多指标自动扩缩容

apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: milvus-querynode-multi-hpa
  namespace: milvus-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: milvus-querynode

  minReplicas: 2
  maxReplicas: 20

  metrics:
  # CPU指标
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70

  # 内存指标
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80

  # 自定义指标（需要Prometheus Adapter）
  - type: Pods
    pods:
      metric:
        name: milvus_search_latency_p99
      target:
        type: AverageValue
        averageValue: "1000"  # P99延迟超过1秒扩容

  - type: Pods
    pods:
      metric:
        name: milvus_search_qps
      target:
        type: AverageValue
        averageValue: "100"  # 每个Pod QPS超过100扩容

  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### 4. 部署脚本

```bash
#!/bin/bash
# deploy-hpa.sh - 部署HPA

set -e

echo "=== 部署HPA自动扩缩容 ==="

# 检查metrics-server是否安装
echo "📦 检查metrics-server..."
if ! kubectl get deployment metrics-server -n kube-system &> /dev/null; then
    echo "❌ metrics-server未安装，正在安装..."
    kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

    # 等待metrics-server就绪
    kubectl wait --for=condition=available deployment/metrics-server \
      -n kube-system \
      --timeout=300s

    echo "✅ metrics-server安装完成"
else
    echo "✅ metrics-server已安装"
fi

# 部署HPA
echo "📦 部署HPA..."
kubectl apply -f hpa-cpu.yaml

echo "✅ HPA部署完成"

# 查看HPA状态
echo ""
echo "=== HPA状态 ==="
kubectl get hpa -n milvus-prod

# 查看详细信息
echo ""
echo "=== HPA详细信息 ==="
kubectl describe hpa milvus-querynode-hpa -n milvus-prod
```

---

## 压力测试脚本

### Python压测工具

```python
"""
load_test.py - Milvus压力测试工具

功能：
1. 生成持续的查询负载
2. 观察HPA扩缩容行为
3. 记录性能指标
"""

from pymilvus import connections, Collection
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import argparse
from datetime import datetime

class LoadTester:
    def __init__(self, host, port, collection_name, dim=128):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dim = dim
        self.collection = None
        self.running = False
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_latency": 0,
            "latencies": []
        }

    def connect(self):
        """连接到Milvus"""
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port
        )
        self.collection = Collection(self.collection_name)
        print(f"✅ 连接到Milvus: {self.host}:{self.port}")

    def search_once(self):
        """执行一次搜索"""
        try:
            query_vector = np.random.random(self.dim).tolist()
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

            start_time = time.time()
            results = self.collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=10
            )
            latency = (time.time() - start_time) * 1000

            self.stats["successful_queries"] += 1
            self.stats["total_latency"] += latency
            self.stats["latencies"].append(latency)

            return True, latency
        except Exception as e:
            self.stats["failed_queries"] += 1
            return False, 0

    def run_load_test(self, qps, duration, num_workers=10):
        """运行压力测试"""
        print(f"\n=== 开始压力测试 ===")
        print(f"目标QPS: {qps}")
        print(f"持续时间: {duration}秒")
        print(f"并发数: {num_workers}")

        self.running = True
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_latency": 0,
            "latencies": []
        }

        start_time = time.time()
        interval = 1.0 / qps  # 每次查询的间隔

        def worker():
            while self.running and (time.time() - start_time) < duration:
                self.search_once()
                self.stats["total_queries"] += 1
                time.sleep(interval * num_workers)  # 调整间隔以达到目标QPS

        # 启动工作线程
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker) for _ in range(num_workers)]

            # 定期打印统计信息
            last_print = time.time()
            while self.running and (time.time() - start_time) < duration:
                time.sleep(5)
                if time.time() - last_print >= 5:
                    self.print_stats()
                    last_print = time.time()

            self.running = False

        # 等待所有线程完成
        for future in futures:
            future.result()

        # 打印最终统计
        print("\n=== 压力测试完成 ===")
        self.print_final_stats()

    def print_stats(self):
        """打印当前统计信息"""
        if self.stats["successful_queries"] > 0:
            avg_latency = self.stats["total_latency"] / self.stats["successful_queries"]
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"查询数: {self.stats['total_queries']}, "
                  f"成功: {self.stats['successful_queries']}, "
                  f"失败: {self.stats['failed_queries']}, "
                  f"平均延迟: {avg_latency:.2f}ms")

    def print_final_stats(self):
        """打印最终统计信息"""
        if self.stats["successful_queries"] > 0:
            latencies = sorted(self.stats["latencies"])
            avg_latency = self.stats["total_latency"] / self.stats["successful_queries"]
            p50 = latencies[int(len(latencies) * 0.5)]
            p95 = latencies[int(len(latencies) * 0.95)]
            p99 = latencies[int(len(latencies) * 0.99)]

            print(f"总查询数: {self.stats['total_queries']}")
            print(f"成功查询: {self.stats['successful_queries']}")
            print(f"失败查询: {self.stats['failed_queries']}")
            print(f"成功率: {self.stats['successful_queries']/self.stats['total_queries']*100:.2f}%")
            print(f"平均延迟: {avg_latency:.2f}ms")
            print(f"P50延迟: {p50:.2f}ms")
            print(f"P95延迟: {p95:.2f}ms")
            print(f"P99延迟: {p99:.2f}ms")

def main():
    parser = argparse.ArgumentParser(description='Milvus压力测试工具')
    parser.add_argument('--host', default='localhost', help='Milvus主机地址')
    parser.add_argument('--port', default='19530', help='Milvus端口')
    parser.add_argument('--collection', required=True, help='Collection名称')
    parser.add_argument('--qps', type=int, default=100, help='目标QPS')
    parser.add_argument('--duration', type=int, default=300, help='持续时间（秒）')
    parser.add_argument('--workers', type=int, default=10, help='并发工作线程数')
    parser.add_argument('--dim', type=int, default=128, help='向量维度')

    args = parser.parse_args()

    # 创建测试器
    tester = LoadTester(
        host=args.host,
        port=args.port,
        collection_name=args.collection,
        dim=args.dim
    )

    # 连接并运行测试
    tester.connect()
    tester.run_load_test(
        qps=args.qps,
        duration=args.duration,
        num_workers=args.workers
    )

if __name__ == "__main__":
    main()
```

### 使用压测工具

```bash
# 安装依赖
pip install pymilvus numpy

# 低负载测试（不触发扩容）
python load_test.py \
  --host localhost \
  --port 19530 \
  --collection test_collection \
  --qps 50 \
  --duration 300 \
  --workers 5

# 高负载测试（触发扩容）
python load_test.py \
  --host localhost \
  --port 19530 \
  --collection test_collection \
  --qps 500 \
  --duration 600 \
  --workers 20

# 突发流量测试
python load_test.py \
  --host localhost \
  --port 19530 \
  --collection test_collection \
  --qps 1000 \
  --duration 180 \
  --workers 50
```

---

## 监控HPA行为

### 实时监控脚本

```bash
#!/bin/bash
# monitor-hpa.sh - 实时监控HPA行为

echo "=== 实时监控HPA行为 ==="

# 在后台持续监控
while true; do
    clear
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo ""

    # HPA状态
    echo "📊 HPA状态："
    kubectl get hpa milvus-querynode-hpa -n milvus-prod

    echo ""
    echo "📦 Pod状态："
    kubectl get pods -n milvus-prod -l app.kubernetes.io/component=querynode

    echo ""
    echo "📈 资源使用："
    kubectl top pods -n milvus-prod -l app.kubernetes.io/component=querynode

    echo ""
    echo "📝 最近事件："
    kubectl get events -n milvus-prod \
      --field-selector involvedObject.name=milvus-querynode-hpa \
      --sort-by='.lastTimestamp' \
      | tail -5

    sleep 10
done
```

### Prometheus查询

```promql
# CPU使用率
rate(container_cpu_usage_seconds_total{pod=~"milvus-querynode-.*"}[5m]) * 100

# 内存使用率
container_memory_working_set_bytes{pod=~"milvus-querynode-.*"} /
container_spec_memory_limit_bytes{pod=~"milvus-querynode-.*"} * 100

# Pod副本数
count(kube_pod_info{pod=~"milvus-querynode-.*"})

# 搜索QPS
rate(milvus_search_total[1m])

# 搜索延迟P99
histogram_quantile(0.99, rate(milvus_search_latency_bucket[5m]))
```

---

## 扩缩容场景测试

### 场景1：逐步增加负载

```bash
#!/bin/bash
# test-gradual-scale.sh - 逐步增加负载测试

echo "=== 场景1：逐步增加负载 ==="

# 阶段1：低负载（50 QPS）
echo "📊 阶段1：低负载（50 QPS）- 5分钟"
python load_test.py --qps 50 --duration 300 &
PID1=$!
sleep 300

# 阶段2：中负载（200 QPS）
echo "📊 阶段2：中负载（200 QPS）- 5分钟"
python load_test.py --qps 200 --duration 300 &
PID2=$!
sleep 300

# 阶段3：高负载（500 QPS）
echo "📊 阶段3：高负载（500 QPS）- 5分钟"
python load_test.py --qps 500 --duration 300 &
PID3=$!
sleep 300

# 阶段4：回到低负载（50 QPS）
echo "📊 阶段4：回到低负载（50 QPS）- 10分钟"
python load_test.py --qps 50 --duration 600 &
PID4=$!
sleep 600

echo "✅ 测试完成"
```

### 场景2：突发流量

```bash
#!/bin/bash
# test-burst-traffic.sh - 突发流量测试

echo "=== 场景2：突发流量 ===

"

# 正常负载
echo "📊 正常负载（100 QPS）- 5分钟"
python load_test.py --qps 100 --duration 300 &
sleep 300

# 突发流量
echo "📊 突发流量（1000 QPS）- 3分钟"
python load_test.py --qps 1000 --duration 180 --workers 50 &
sleep 180

# 恢复正常
echo "📊 恢复正常（100 QPS）- 10分钟"
python load_test.py --qps 100 --duration 600 &
sleep 600

echo "✅ 测试完成"
```

### 场景3：周期性波动

```bash
#!/bin/bash
# test-periodic-load.sh - 周期性负载测试

echo "=== 场景3：周期性负载 ==="

for i in {1..5}; do
    echo "📊 周期 $i - 高负载（500 QPS）- 3分钟"
    python load_test.py --qps 500 --duration 180 &
    sleep 180

    echo "📊 周期 $i - 低负载（50 QPS）- 3分钟"
    python load_test.py --qps 50 --duration 180 &
    sleep 180
done

echo "✅ 测试完成"
```

---

## 验证和分析

### 验证HPA工作

```bash
#!/bin/bash
# verify-hpa.sh - 验证HPA是否正常工作

echo "=== 验证HPA配置 ==="

# 1. 检查HPA是否存在
if kubectl get hpa milvus-querynode-hpa -n milvus-prod &> /dev/null; then
    echo "✅ HPA已创建"
else
    echo "❌ HPA不存在"
    exit 1
fi

# 2. 检查metrics-server
if kubectl get deployment metrics-server -n kube-system &> /dev/null; then
    echo "✅ metrics-server已安装"
else
    echo "❌ metrics-server未安装"
    exit 1
fi

# 3. 检查HPA指标
echo ""
echo "📊 HPA当前状态："
kubectl get hpa milvus-querynode-hpa -n milvus-prod

# 4. 检查Pod资源配置
echo ""
echo "📦 检查Pod资源配置："
kubectl get deployment milvus-querynode -n milvus-prod -o jsonpath='{.spec.template.spec.containers[0].resources}'

# 5. 检查当前副本数
echo ""
echo "📊 当前副本数："
kubectl get deployment milvus-querynode -n milvus-prod -o jsonpath='{.spec.replicas}'

echo ""
echo "✅ HPA验证完成"
```

### 分析扩缩容日志

```bash
#!/bin/bash
# analyze-scaling-events.sh - 分析扩缩容事件

echo "=== 分析扩缩容事件 ==="

# 获取HPA事件
echo "📝 HPA扩缩容事件："
kubectl get events -n milvus-prod \
  --field-selector involvedObject.name=milvus-querynode-hpa \
  --sort-by='.lastTimestamp'

echo ""
echo "📝 Deployment扩缩容事件："
kubectl get events -n milvus-prod \
  --field-selector involvedObject.name=milvus-querynode \
  --sort-by='.lastTimestamp'

# 统计扩缩容次数
echo ""
echo "📊 扩缩容统计："
SCALE_UP=$(kubectl get events -n milvus-prod \
  --field-selector involvedObject.name=milvus-querynode-hpa \
  | grep "Scaled up" | wc -l)
SCALE_DOWN=$(kubectl get events -n milvus-prod \
  --field-selector involvedObject.name=milvus-querynode-hpa \
  | grep "Scaled down" | wc -l)

echo "扩容次数: $SCALE_UP"
echo "缩容次数: $SCALE_DOWN"
```

---

## 优化建议

### 1. 扩缩容策略优化

```yaml
# 优化后的HPA配置
behavior:
  scaleUp:
    # 快速扩容
    stabilizationWindowSeconds: 30  # 缩短等待时间
    policies:
    - type: Percent
      value: 100  # 每次翻倍
      periodSeconds: 30
    selectPolicy: Max

  scaleDown:
    # 保守缩容
    stabilizationWindowSeconds: 600  # 延长等待时间
    policies:
    - type: Pods
      value: 1  # 每次只缩容1个
      periodSeconds: 120
    selectPolicy: Min
```

### 2. 资源配置优化

```yaml
# 确保Pod有合理的资源配置
resources:
  requests:
    cpu: 4
    memory: 16Gi
  limits:
    cpu: 8
    memory: 32Gi

# HPA基于requests计算使用率
# 如果requests太低，会导致频繁扩容
# 如果requests太高，会导致资源浪费
```

### 3. 监控告警配置

```yaml
# Prometheus告警规则
groups:
- name: hpa-alerts
  rules:
  # HPA达到最大副本数
  - alert: HPAMaxedOut
    expr: kube_hpa_status_current_replicas >= kube_hpa_spec_max_replicas
    for: 5m
    annotations:
      summary: "HPA已达到最大副本数"

  # HPA频繁扩缩容
  - alert: HPAFlapping
    expr: rate(kube_hpa_status_current_replicas[30m]) > 0.1
    for: 10m
    annotations:
      summary: "HPA频繁扩缩容"
```

---

## 总结

### HPA自动扩缩容的价值

| 维度 | 价值 |
|------|------|
| **成本优化** | 低峰期自动缩容，节省60-80%成本 |
| **性能保障** | 高峰期自动扩容，保证服务质量 |
| **自动化** | 无需人工干预，自动应对负载变化 |
| **弹性** | 快速响应突发流量 |

### 配置要点

1. **扩容快，缩容慢**：避免频繁波动
2. **合理的资源配置**：requests不能太低或太高
3. **多指标结合**：CPU + 内存 + 自定义指标
4. **监控告警**：及时发现HPA异常

### 适用场景

- ✅ 流量波动大的RAG系统
- ✅ 需要成本优化
- ✅ 需要自动应对突发流量
- ✅ 7x24小时运行的服务

### 下一步

完成HPA配置后，继续学习：
- **场景5：灰度发布** - 零停机升级
- **监控告警** - Prometheus + Grafana
- **成本优化** - 资源利用率分析
