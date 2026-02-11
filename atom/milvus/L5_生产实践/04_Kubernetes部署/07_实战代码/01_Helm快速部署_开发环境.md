# 实战代码 - 场景1：Helm快速部署（开发环境）

## 场景描述

**目标：**在开发环境快速部署Milvus集群，用于功能测试和开发调试

**特点：**
- 单机模式或最小集群配置
- 不启用持久化（快速重建）
- 使用内置依赖（etcd、MinIO、Pulsar）
- 资源配置较低

**适用场景：**
- 本地开发测试
- 功能验证
- 学习Kubernetes部署

---

## 完整部署脚本

### 1. 环境准备

```bash
#!/bin/bash
# deploy-dev.sh - 开发环境部署脚本

set -e  # 遇到错误立即退出

echo "=== Milvus开发环境部署脚本 ==="

# 检查kubectl
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl未安装，请先安装kubectl"
    exit 1
fi

# 检查Helm
if ! command -v helm &> /dev/null; then
    echo "❌ Helm未安装，请先安装Helm"
    exit 1
fi

# 检查Kubernetes集群连接
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ 无法连接到Kubernetes集群"
    exit 1
fi

echo "✅ 环境检查通过"

# 创建命名空间
echo "📦 创建命名空间..."
kubectl create namespace milvus-dev --dry-run=client -o yaml | kubectl apply -f -

# 添加Milvus Helm仓库
echo "📦 添加Milvus Helm仓库..."
helm repo add milvus https://zilliztech.github.io/milvus-helm/
helm repo update

echo "✅ 环境准备完成"
```

### 2. 配置文件

```yaml
# dev-values.yaml - 开发环境配置

# 集群模式（可选：standalone 或 cluster）
cluster:
  enabled: false  # 开发环境使用单机模式

# 镜像配置
image:
  all:
    repository: milvusdb/milvus
    tag: v2.3.0
    pullPolicy: IfNotPresent

# Proxy配置（单机模式下也需要）
proxy:
  replicas: 1
  resources:
    requests:
      cpu: 0.5
      memory: 1Gi
    limits:
      cpu: 1
      memory: 2Gi

# QueryNode配置
queryNode:
  replicas: 1
  resources:
    requests:
      cpu: 0.5
      memory: 2Gi
    limits:
      cpu: 1
      memory: 4Gi

# DataNode配置
dataNode:
  replicas: 1
  resources:
    requests:
      cpu: 0.5
      memory: 1Gi
    limits:
      cpu: 1
      memory: 2Gi

# IndexNode配置
indexNode:
  replicas: 1
  resources:
    requests:
      cpu: 0.5
      memory: 1Gi
    limits:
      cpu: 1
      memory: 2Gi

# 持久化配置（开发环境不启用）
persistence:
  enabled: false

# 内置etcd配置
etcd:
  enabled: true
  replicaCount: 1
  persistence:
    enabled: false  # 不持久化，快速重建
  resources:
    requests:
      cpu: 0.1
      memory: 128Mi
    limits:
      cpu: 0.5
      memory: 512Mi

# 内置MinIO配置
minio:
  enabled: true
  mode: standalone
  persistence:
    enabled: false  # 不持久化
  resources:
    requests:
      cpu: 0.1
      memory: 256Mi
    limits:
      cpu: 0.5
      memory: 1Gi

# 内置Pulsar配置
pulsar:
  enabled: true
  components:
    broker: true
    bookkeeper: false  # 开发环境不需要
    zookeeper: true
  broker:
    replicaCount: 1
    resources:
      requests:
        cpu: 0.1
        memory: 256Mi
      limits:
        cpu: 0.5
        memory: 1Gi
  zookeeper:
    replicaCount: 1
    persistence:
      enabled: false
    resources:
      requests:
        cpu: 0.1
        memory: 256Mi
      limits:
        cpu: 0.5
        memory: 512Mi

# Service配置
service:
  type: NodePort
  port: 19530
  nodePort: 30530  # 固定端口，方便访问

# 日志配置
log:
  level: info

# 配置
config:
  common:
    retentionDuration: "86400"  # 1天数据保留
  dataCoord:
    segment:
      maxSize: "512"  # 较小的segment，快速测试
  queryNode:
    gracefulTime: "1000"
```

### 3. 部署命令

```bash
#!/bin/bash
# 继续 deploy-dev.sh

echo "🚀 开始部署Milvus..."

# 部署Milvus
helm install milvus-dev milvus/milvus \
  -f dev-values.yaml \
  -n milvus-dev \
  --wait \
  --timeout 10m

echo "✅ Milvus部署完成"

# 等待所有Pod就绪
echo "⏳ 等待所有Pod就绪..."
kubectl wait --for=condition=ready pod \
  -l app.kubernetes.io/instance=milvus-dev \
  -n milvus-dev \
  --timeout=300s

echo "✅ 所有Pod已就绪"

# 显示部署信息
echo ""
echo "=== 部署信息 ==="
kubectl get pods -n milvus-dev
echo ""
kubectl get svc -n milvus-dev

# 获取访问地址
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
NODE_PORT=$(kubectl get svc milvus-dev -n milvus-dev -o jsonpath='{.spec.ports[0].nodePort}')

echo ""
echo "=== 访问信息 ==="
echo "Milvus地址: ${NODE_IP}:${NODE_PORT}"
echo ""
echo "使用以下命令连接："
echo "  from pymilvus import connections"
echo "  connections.connect(host='${NODE_IP}', port='${NODE_PORT}')"
echo ""
echo "或使用端口转发："
echo "  kubectl port-forward svc/milvus-dev 19530:19530 -n milvus-dev"
echo "  connections.connect(host='localhost', port='19530')"
```

---

## Python连接验证

### 验证脚本

```python
"""
verify_dev_deployment.py - 验证开发环境部署

功能：
1. 连接到Milvus集群
2. 创建测试Collection
3. 插入测试数据
4. 执行向量检索
5. 清理测试数据
"""

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
import numpy as np
import time

# ===== 1. 连接配置 =====
print("=== 步骤1：连接到Milvus ===")

# 方式1：使用NodePort（需要替换为实际的Node IP）
# HOST = "192.168.1.100"
# PORT = "30530"

# 方式2：使用端口转发（推荐）
# 先执行：kubectl port-forward svc/milvus-dev 19530:19530 -n milvus-dev
HOST = "localhost"
PORT = "19530"

try:
    connections.connect(
        alias="default",
        host=HOST,
        port=PORT,
        timeout=10
    )
    print(f"✅ 成功连接到Milvus: {HOST}:{PORT}")
except Exception as e:
    print(f"❌ 连接失败: {e}")
    exit(1)

# ===== 2. 创建测试Collection =====
print("\n=== 步骤2：创建测试Collection ===")

COLLECTION_NAME = "dev_test_collection"
DIM = 128

# 删除已存在的Collection
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)
    print(f"🗑️  删除已存在的Collection: {COLLECTION_NAME}")

# 定义Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512)
]
schema = CollectionSchema(fields=fields, description="开发测试Collection")

# 创建Collection
collection = Collection(name=COLLECTION_NAME, schema=schema)
print(f"✅ 创建Collection: {COLLECTION_NAME}")

# ===== 3. 创建索引 =====
print("\n=== 步骤3：创建索引 ===")

index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}

collection.create_index(
    field_name="embedding",
    index_params=index_params
)
print("✅ 创建索引完成")

# ===== 4. 插入测试数据 =====
print("\n=== 步骤4：插入测试数据 ===")

NUM_ENTITIES = 1000

# 生成随机数据
ids = list(range(NUM_ENTITIES))
embeddings = np.random.random((NUM_ENTITIES, DIM)).tolist()
texts = [f"测试文本_{i}" for i in range(NUM_ENTITIES)]

# 插入数据
entities = [ids, embeddings, texts]
insert_result = collection.insert(entities)
print(f"✅ 插入 {NUM_ENTITIES} 条数据")
print(f"   插入ID范围: {insert_result.primary_keys[0]} - {insert_result.primary_keys[-1]}")

# 刷新数据（确保数据持久化）
collection.flush()
print("✅ 数据刷新完成")

# ===== 5. 加载Collection =====
print("\n=== 步骤5：加载Collection到内存 ===")

collection.load()
print("✅ Collection加载完成")

# 等待加载完成
time.sleep(2)

# ===== 6. 执行向量检索 =====
print("\n=== 步骤6：执行向量检索 ===")

# 生成查询向量
query_vectors = np.random.random((5, DIM)).tolist()

# 搜索参数
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10}
}

# 执行搜索
start_time = time.time()
results = collection.search(
    data=query_vectors,
    anns_field="embedding",
    param=search_params,
    limit=5,
    output_fields=["text"]
)
search_time = (time.time() - start_time) * 1000

print(f"✅ 搜索完成，耗时: {search_time:.2f}ms")
print(f"   查询向量数: {len(query_vectors)}")
print(f"   每个查询返回: {len(results[0])} 个结果")

# 显示第一个查询的结果
print("\n第一个查询的Top-5结果：")
for i, hit in enumerate(results[0]):
    print(f"  {i+1}. ID={hit.id}, 距离={hit.distance:.4f}, 文本={hit.entity.get('text')}")

# ===== 7. 统计信息 =====
print("\n=== 步骤7：查看统计信息 ===")

stats = collection.num_entities
print(f"Collection总数据量: {stats}")

# ===== 8. 清理测试数据 =====
print("\n=== 步骤8：清理测试数据 ===")

# 释放Collection
collection.release()
print("✅ 释放Collection")

# 删除Collection
utility.drop_collection(COLLECTION_NAME)
print(f"✅ 删除Collection: {COLLECTION_NAME}")

# 断开连接
connections.disconnect("default")
print("✅ 断开连接")

print("\n=== 验证完成 ===")
print("✅ 开发环境部署验证成功！")
```

### 运行验证

```bash
# 1. 安装依赖
pip install pymilvus numpy

# 2. 启动端口转发（新终端）
kubectl port-forward svc/milvus-dev 19530:19530 -n milvus-dev

# 3. 运行验证脚本
python verify_dev_deployment.py
```

**预期输出：**

```
=== 步骤1：连接到Milvus ===
✅ 成功连接到Milvus: localhost:19530

=== 步骤2：创建测试Collection ===
✅ 创建Collection: dev_test_collection

=== 步骤3：创建索引 ===
✅ 创建索引完成

=== 步骤4：插入测试数据 ===
✅ 插入 1000 条数据
   插入ID范围: 0 - 999
✅ 数据刷新完成

=== 步骤5：加载Collection到内存 ===
✅ Collection加载完成

=== 步骤6：执行向量检索 ===
✅ 搜索完成，耗时: 45.23ms
   查询向量数: 5
   每个查询返回: 5 个结果

第一个查询的Top-5结果：
  1. ID=342, 距离=5.2341, 文本=测试文本_342
  2. ID=789, 距离=5.4567, 文本=测试文本_789
  3. ID=123, 距离=5.6789, 文本=测试文本_123
  4. ID=456, 距离=5.8901, 文本=测试文本_456
  5. ID=234, 距离=6.0123, 文本=测试文本_234

=== 步骤7：查看统计信息 ===
Collection总数据量: 1000

=== 步骤8：清理测试数据 ===
✅ 释放Collection
✅ 删除Collection: dev_test_collection
✅ 断开连接

=== 验证完成 ===
✅ 开发环境部署验证成功！
```

---

## 常用运维操作

### 查看集群状态

```bash
#!/bin/bash
# check-status.sh - 查看集群状态

echo "=== Milvus集群状态 ==="

# 查看所有Pod
echo "📦 Pod状态："
kubectl get pods -n milvus-dev -o wide

# 查看Service
echo ""
echo "🌐 Service状态："
kubectl get svc -n milvus-dev

# 查看资源使用
echo ""
echo "📊 资源使用："
kubectl top pods -n milvus-dev

# 查看Helm Release
echo ""
echo "📦 Helm Release："
helm list -n milvus-dev
```

### 查看日志

```bash
#!/bin/bash
# view-logs.sh - 查看日志

# 查看Proxy日志
echo "=== Proxy日志 ==="
kubectl logs -f deployment/milvus-dev-proxy -n milvus-dev --tail=50

# 查看QueryNode日志
# kubectl logs -f deployment/milvus-dev-querynode -n milvus-dev --tail=50

# 查看所有组件日志
# kubectl logs -f -l app.kubernetes.io/instance=milvus-dev -n milvus-dev --tail=50
```

### 重启组件

```bash
#!/bin/bash
# restart-component.sh - 重启组件

# 重启Proxy
kubectl rollout restart deployment/milvus-dev-proxy -n milvus-dev

# 重启QueryNode
kubectl rollout restart deployment/milvus-dev-querynode -n milvus-dev

# 查看重启状态
kubectl rollout status deployment/milvus-dev-proxy -n milvus-dev
```

### 清理环境

```bash
#!/bin/bash
# cleanup-dev.sh - 清理开发环境

echo "=== 清理Milvus开发环境 ==="

# 删除Helm Release
echo "🗑️  删除Helm Release..."
helm uninstall milvus-dev -n milvus-dev

# 删除PVC（如果有）
echo "🗑️  删除PVC..."
kubectl delete pvc -l app.kubernetes.io/instance=milvus-dev -n milvus-dev

# 删除命名空间
echo "🗑️  删除命名空间..."
kubectl delete namespace milvus-dev

echo "✅ 清理完成"
```

---

## 故障排查

### 问题1：Pod一直处于Pending状态

**排查步骤：**

```bash
# 查看Pod详情
kubectl describe pod <pod-name> -n milvus-dev

# 常见原因：
# 1. 资源不足 → 降低资源requests
# 2. 镜像拉取失败 → 检查网络和镜像地址
# 3. PVC无法绑定 → 检查StorageClass
```

**解决方案：**

```yaml
# 降低资源配置
resources:
  requests:
    cpu: 0.1  # 从0.5降到0.1
    memory: 512Mi  # 从1Gi降到512Mi
```

### 问题2：连接超时

**排查步骤：**

```bash
# 1. 检查Service
kubectl get svc milvus-dev -n milvus-dev

# 2. 检查Pod是否就绪
kubectl get pods -n milvus-dev

# 3. 测试端口转发
kubectl port-forward svc/milvus-dev 19530:19530 -n milvus-dev
```

### 问题3：内存不足OOM

**排查步骤：**

```bash
# 查看Pod事件
kubectl describe pod <pod-name> -n milvus-dev | grep -A 10 Events

# 查看资源使用
kubectl top pod <pod-name> -n milvus-dev
```

**解决方案：**

```yaml
# 增加内存限制
resources:
  limits:
    memory: 4Gi  # 从2Gi增加到4Gi
```

---

## 总结

### 开发环境部署特点

| 特性 | 配置 | 原因 |
|------|------|------|
| **模式** | 单机或最小集群 | 资源占用少 |
| **持久化** | 关闭 | 快速重建 |
| **依赖** | 内置 | 简化部署 |
| **资源** | 低配置 | 节省资源 |
| **Service** | NodePort | 方便访问 |

### 适用场景

- ✅ 本地开发测试
- ✅ 功能验证
- ✅ 学习Kubernetes部署
- ❌ 生产环境（需要高可用配置）
- ❌ 性能测试（资源配置太低）

### 下一步

完成开发环境部署后，可以继续学习：
- **场景2：生产部署** - 高可用配置
- **场景3：Operator部署** - 自动化运维
- **场景4：自动扩缩容** - 弹性伸缩
