# 实战代码 - 场景3：Operator自动化部署

## 场景描述

**目标：**使用Milvus Operator实现自动化运维，简化配置和管理

**特点：**
- 使用CRD声明式配置
- Operator自动配置最佳实践
- 自动故障恢复
- 简化的配置接口

**适用场景：**
- 大规模生产环境
- 需要自动化运维
- 多集群管理

---

## 完整部署脚本

### 1. 安装Operator

```bash
#!/bin/bash
# install-operator.sh - 安装Milvus Operator

set -e

echo "=== 安装Milvus Operator ==="

# 检查环境
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl未安装"
    exit 1
fi

if ! command -v helm &> /dev/null; then
    echo "❌ Helm未安装"
    exit 1
fi

echo "✅ 环境检查通过"

# 创建命名空间
kubectl create namespace milvus-operator --dry-run=client -o yaml | kubectl apply -f -

# 添加Operator仓库
echo "📦 添加Milvus Operator仓库..."
helm repo add milvus-operator https://zilliztech.github.io/milvus-operator/
helm repo update

# 安装Operator
echo "📦 安装Milvus Operator..."
helm install milvus-operator milvus-operator/milvus-operator \
  -n milvus-operator \
  --wait

echo "✅ Operator安装完成"

# 验证安装
echo "📦 验证Operator状态..."
kubectl get pods -n milvus-operator

# 检查CRD
echo "📦 检查CRD..."
kubectl get crd | grep milvus

echo "✅ Milvus Operator安装成功"
```

### 2. 基础Milvus集群配置

```yaml
# milvus-cluster-basic.yaml - 基础集群配置

apiVersion: milvus.io/v1beta1
kind: Milvus
metadata:
  name: milvus-cluster
  namespace: default
spec:
  # 集群模式
  mode: cluster

  # 镜像配置
  components:
    image: milvusdb/milvus:v2.3.0

  # Operator自动配置：
  # - Pod反亲和性
  # - 健康检查
  # - 资源限制
  # - 滚动更新策略
  # - 服务发现
```

### 3. 生产级集群配置

```yaml
# milvus-cluster-prod.yaml - 生产级配置

apiVersion: milvus.io/v1beta1
kind: Milvus
metadata:
  name: milvus-prod
  namespace: milvus-prod
  labels:
    app: milvus
    env: production
spec:
  # 集群模式
  mode: cluster

  # 镜像配置
  components:
    image: milvusdb/milvus:v2.3.0
    imagePullPolicy: IfNotPresent

    # Proxy配置
    proxy:
      replicas: 3
      resources:
        requests:
          cpu: "2"
          memory: 4Gi
        limits:
          cpu: "4"
          memory: 8Gi

    # QueryNode配置
    queryNode:
      replicas: 5
      resources:
        requests:
          cpu: "4"
          memory: 16Gi
        limits:
          cpu: "8"
          memory: 32Gi

    # DataNode配置
    dataNode:
      replicas: 3
      resources:
        requests:
          cpu: "2"
          memory: 8Gi
        limits:
          cpu: "4"
          memory: 16Gi

    # IndexNode配置
    indexNode:
      replicas: 2
      resources:
        requests:
          cpu: "4"
          memory: 8Gi
        limits:
          cpu: "8"
          memory: 16Gi

  # 依赖配置
  dependencies:
    # etcd配置
    etcd:
      inCluster:
        deletionPolicy: Retain  # 删除Milvus时保留etcd
        pvcDeletion: false
        values:
          replicaCount: 3
          persistence:
            enabled: true
            storageClass: fast-ssd
            size: 20Gi
          resources:
            requests:
              cpu: "1"
              memory: 2Gi
            limits:
              cpu: "2"
              memory: 4Gi

    # 存储配置（MinIO）
    storage:
      inCluster:
        deletionPolicy: Retain
        pvcDeletion: false
        values:
          mode: distributed
          replicas: 4
          persistence:
            enabled: true
            storageClass: standard
            size: 500Gi
          resources:
            requests:
              cpu: "1"
              memory: 4Gi
            limits:
              cpu: "2"
              memory: 8Gi

    # Pulsar配置
    pulsar:
      inCluster:
        values:
          components:
            broker: true
            bookkeeper: true
            zookeeper: true
          broker:
            replicaCount: 3
          bookkeeper:
            replicaCount: 3
          zookeeper:
            replicaCount: 3
          persistence:
            enabled: true
            storageClass: fast-ssd

  # Milvus配置
  config:
    log:
      level: info
      format: json
    dataCoord:
      segment:
        maxSize: "1024"
      enableCompaction: true
    queryNode:
      gracefulTime: "5000"
    common:
      retentionDuration: "432000"  # 5天
```

### 4. 使用外部依赖的配置

```yaml
# milvus-cluster-external.yaml - 外部依赖配置

apiVersion: milvus.io/v1beta1
kind: Milvus
metadata:
  name: milvus-external
  namespace: milvus-prod
spec:
  mode: cluster

  components:
    image: milvusdb/milvus:v2.3.0

  dependencies:
    # 外部etcd
    etcd:
      external: true
      endpoints:
        - etcd-0.etcd.default.svc.cluster.local:2379
        - etcd-1.etcd.default.svc.cluster.local:2379
        - etcd-2.etcd.default.svc.cluster.local:2379

    # 外部S3
    storage:
      external: true
      type: S3
      endpoint: s3.amazonaws.com:443
      secretRef: milvus-s3-secret

    # 外部Pulsar
    pulsar:
      external: true
      endpoint: pulsar://pulsar-broker.default.svc.cluster.local:6650

---
# S3 Secret
apiVersion: v1
kind: Secret
metadata:
  name: milvus-s3-secret
  namespace: milvus-prod
type: Opaque
stringData:
  accesskey: "your-access-key"
  secretkey: "your-secret-key"
  bucketname: "milvus-bucket"
```

### 5. 部署脚本

```bash
#!/bin/bash
# deploy-with-operator.sh - 使用Operator部署Milvus

set -e

echo "=== 使用Operator部署Milvus ==="

# 创建命名空间
kubectl create namespace milvus-prod --dry-run=client -o yaml | kubectl apply -f -

# 部署Milvus集群
echo "🚀 部署Milvus集群..."
kubectl apply -f milvus-cluster-prod.yaml

# 等待Milvus就绪
echo "⏳ 等待Milvus集群就绪..."
kubectl wait --for=condition=Ready milvus/milvus-prod \
  -n milvus-prod \
  --timeout=600s

echo "✅ Milvus集群部署完成"

# 查看集群状态
echo ""
echo "=== 集群状态 ==="
kubectl get milvus -n milvus-prod
kubectl get pods -n milvus-prod

# 查看Service
echo ""
echo "=== Service信息 ==="
kubectl get svc -n milvus-prod

echo ""
echo "✅ 部署完成"
```

---

## Operator运维操作

### 1. 扩缩容

```bash
#!/bin/bash
# scale-cluster.sh - 扩缩容集群

# 扩容QueryNode到10个副本
kubectl patch milvus milvus-prod -n milvus-prod --type='json' -p='[
  {"op": "replace", "path": "/spec/components/queryNode/replicas", "value": 10}
]'

echo "✅ 扩容命令已提交"
echo "⏳ Operator将自动创建新的Pod..."

# 查看扩容进度
kubectl get pods -n milvus-prod -l app.kubernetes.io/component=querynode -w
```

### 2. 升级版本

```bash
#!/bin/bash
# upgrade-version.sh - 升级Milvus版本

# 升级到v2.4.0
kubectl patch milvus milvus-prod -n milvus-prod --type='json' -p='[
  {"op": "replace", "path": "/spec/components/image", "value": "milvusdb/milvus:v2.4.0"}
]'

echo "✅ 升级命令已提交"
echo "⏳ Operator将执行滚动升级..."

# 查看升级进度
kubectl get milvus milvus-prod -n milvus-prod -w
```

### 3. 修改配置

```bash
#!/bin/bash
# update-config.sh - 修改配置

# 修改日志级别
kubectl patch milvus milvus-prod -n milvus-prod --type='json' -p='[
  {"op": "replace", "path": "/spec/config/log/level", "value": "debug"}
]'

echo "✅ 配置修改已提交"
echo "⏳ Operator将自动应用新配置..."

# 查看Pod重启情况
kubectl get pods -n milvus-prod -w
```

### 4. 查看集群状态

```bash
#!/bin/bash
# check-cluster.sh - 查看集群状态

echo "=== Milvus集群状态 ==="

# 查看Milvus CR
kubectl get milvus -n milvus-prod

# 查看详细信息
kubectl describe milvus milvus-prod -n milvus-prod

# 查看所有Pod
echo ""
echo "=== Pod状态 ==="
kubectl get pods -n milvus-prod -o wide

# 查看事件
echo ""
echo "=== 最近事件 ==="
kubectl get events -n milvus-prod --sort-by='.lastTimestamp' | tail -20
```

---

## Python验证脚本

```python
"""
verify_operator_deployment.py - 验证Operator部署

功能：
1. 连接到Operator部署的集群
2. 验证基本功能
3. 测试自动恢复能力
"""

from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import numpy as np
import time
import subprocess

# ===== 配置 =====
HOST = "milvus-prod-milvus.milvus-prod.svc.cluster.local"  # Operator创建的Service
PORT = "19530"
COLLECTION_NAME = "operator_test"
DIM = 128

# ===== 1. 连接集群 =====
print("=== 步骤1：连接到Operator部署的集群 ===")

# 使用端口转发
print("请先执行端口转发：")
print("kubectl port-forward svc/milvus-prod-milvus 19530:19530 -n milvus-prod")
print("")

try:
    connections.connect(
        alias="default",
        host="localhost",
        port="19530",
        timeout=10
    )
    print(f"✅ 成功连接到Milvus")
except Exception as e:
    print(f"❌ 连接失败: {e}")
    exit(1)

# ===== 2. 创建测试Collection =====
print("\n=== 步骤2：创建测试Collection ===")

if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
]
schema = CollectionSchema(fields=fields)
collection = Collection(name=COLLECTION_NAME, schema=schema)
print(f"✅ 创建Collection: {COLLECTION_NAME}")

# ===== 3. 创建索引并插入数据 =====
print("\n=== 步骤3：创建索引并插入数据 ===")

index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)

NUM_ENTITIES = 10000
ids = list(range(NUM_ENTITIES))
embeddings = np.random.random((NUM_ENTITIES, DIM)).tolist()
entities = [ids, embeddings]
collection.insert(entities)
collection.flush()
print(f"✅ 插入 {NUM_ENTITIES} 条数据")

# ===== 4. 加载并搜索 =====
print("\n=== 步骤4：加载并搜索 ===")

collection.load()
time.sleep(2)

query_vectors = np.random.random((5, DIM)).tolist()
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

start_time = time.time()
results = collection.search(
    data=query_vectors,
    anns_field="embedding",
    param=search_params,
    limit=10
)
search_time = (time.time() - start_time) * 1000

print(f"✅ 搜索完成，耗时: {search_time:.2f}ms")

# ===== 5. 测试Operator自动恢复 =====
print("\n=== 步骤5：测试Operator自动恢复 ===")
print("模拟Pod故障...")
print("请在另一个终端执行：")
print("kubectl delete pod -l app.kubernetes.io/component=querynode -n milvus-prod --force")
print("\n等待30秒观察Operator自动恢复...")
time.sleep(30)

# 尝试继续搜索
print("测试故障后搜索...")
try:
    results = collection.search(
        data=query_vectors[:1],
        anns_field="embedding",
        param=search_params,
        limit=10
    )
    print("✅ 故障后搜索成功，Operator自动恢复正常")
except Exception as e:
    print(f"❌ 故障后搜索失败: {e}")

# ===== 6. 清理 =====
print("\n=== 步骤6：清理 ===")

collection.release()
utility.drop_collection(COLLECTION_NAME)
connections.disconnect("default")

print("✅ 验证完成")
```

---

## Operator高级功能

### 1. 自动备份配置

```yaml
# milvus-with-backup.yaml
apiVersion: milvus.io/v1beta1
kind: Milvus
metadata:
  name: milvus-backup
  namespace: milvus-prod
spec:
  mode: cluster

  # 启用自动备份
  backup:
    enabled: true
    schedule: "0 2 * * *"  # 每天凌晨2点
    retention: 7  # 保留7天
    destination:
      type: S3
      s3:
        endpoint: s3.amazonaws.com
        bucket: milvus-backups
        secretRef: backup-s3-secret
```

### 2. 监控配置

```yaml
# milvus-with-monitoring.yaml
apiVersion: milvus.io/v1beta1
kind: Milvus
metadata:
  name: milvus-monitor
  namespace: milvus-prod
spec:
  mode: cluster

  # 启用监控
  monitoring:
    enabled: true
    prometheus:
      enabled: true
      serviceMonitor:
        enabled: true
        interval: 30s
    grafana:
      enabled: true
      dashboards:
        enabled: true
```

### 3. 自动扩缩容配置

```yaml
# milvus-with-hpa.yaml
apiVersion: milvus.io/v1beta1
kind: Milvus
metadata:
  name: milvus-hpa
  namespace: milvus-prod
spec:
  mode: cluster

  components:
    queryNode:
      replicas: 5
      # Operator自动创建HPA
      autoscaling:
        enabled: true
        minReplicas: 2
        maxReplicas: 20
        targetCPUUtilizationPercentage: 70
```

---

## 故障排查

### 查看Operator日志

```bash
# 查看Operator日志
kubectl logs -f deployment/milvus-operator -n milvus-operator

# 查看Milvus CR状态
kubectl describe milvus milvus-prod -n milvus-prod

# 查看事件
kubectl get events -n milvus-prod --sort-by='.lastTimestamp'
```

### 常见问题

**问题1：Milvus CR一直处于Pending状态**

```bash
# 检查Operator日志
kubectl logs -f deployment/milvus-operator -n milvus-operator

# 检查资源是否足够
kubectl describe nodes

# 检查依赖是否就绪
kubectl get pods -n milvus-prod
```

**问题2：Pod无法启动**

```bash
# 查看Pod详情
kubectl describe pod <pod-name> -n milvus-prod

# 查看Pod日志
kubectl logs <pod-name> -n milvus-prod

# 检查镜像是否可用
kubectl get pods -n milvus-prod -o jsonpath='{.items[*].spec.containers[*].image}'
```

---

## 总结

### Operator部署的优势

| 特性 | 价值 |
|------|------|
| **简化配置** | 只需高层意图，Operator自动配置细节 |
| **自动化运维** | 自动故障恢复、扩缩容、备份 |
| **最佳实践** | 内置Milvus最佳实践配置 |
| **声明式管理** | 修改CR即可，Operator自动调谐 |
| **持续监控** | Operator持续监控集群健康状态 |

### 适用场景

- ✅ 大规模生产环境（>10节点）
- ✅ 需要自动化运维
- ✅ 多集群管理
- ✅ 需要自动故障恢复
- ✅ 需要自动备份和监控

### 下一步

完成Operator部署后，继续学习：
- **场景4：自动扩缩容** - HPA配置和测试
- **场景5：灰度发布** - 零停机升级策略
- **监控告警** - Prometheus + Grafana集成
