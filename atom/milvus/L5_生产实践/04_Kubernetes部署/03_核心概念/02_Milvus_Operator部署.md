# 核心概念2：Milvus Operator 部署

## 什么是Operator？

**Operator = Kubernetes控制器 + 领域知识**

一句话定义：**Operator是将运维专家的知识编码成软件，实现应用的自动化运维。**

## Operator模式的核心思想

### 传统方式 vs Operator方式

```yaml
# 传统方式（Helm）：你需要知道所有细节
helm install milvus milvus/milvus \
  --set queryNode.replicas=3 \
  --set queryNode.resources.requests.cpu=4 \
  --set queryNode.resources.requests.memory=16Gi \
  --set queryNode.affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].labelSelector.matchExpressions[0].key=app \
  --set queryNode.affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].labelSelector.matchExpressions[0].operator=In \
  --set queryNode.affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].labelSelector.matchExpressions[0].values[0]=milvus-querynode \
  --set queryNode.affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].topologyKey=kubernetes.io/hostname
  # 还有几十个参数...

# Operator方式：只需要高层意图
kubectl apply -f - <<EOF
apiVersion: milvus.io/v1beta1
kind: Milvus
metadata:
  name: my-milvus
spec:
  mode: cluster
  # Operator自动配置所有最佳实践
EOF
```

**类比：**
- Helm = 通用工具（你告诉它每个螺丝怎么拧）
- Operator = 专业工程师（你告诉它要什么，它知道怎么做）

---

## Operator的工作原理

### 1. 自定义资源（CRD）

**CRD = 扩展Kubernetes API**

```yaml
# 定义新的资源类型
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: milvuses.milvus.io
spec:
  group: milvus.io
  names:
    kind: Milvus
    plural: milvuses
  scope: Namespaced
  versions:
  - name: v1beta1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              mode:
                type: string
                enum: [standalone, cluster]
              components:
                type: object
```

**安装CRD后，Kubernetes就认识Milvus资源了：**

```bash
# 查看Milvus资源
kubectl get milvus

# 描述Milvus资源
kubectl describe milvus my-milvus
```

### 2. 控制器（Controller）

**Controller = 持续运行的调谐循环**

```python
# Operator控制器的伪代码
while True:
    # 1. 观察当前状态
    current_state = get_current_milvus_state()

    # 2. 读取期望状态
    desired_state = get_milvus_cr_spec()

    # 3. 对比差异
    if current_state != desired_state:
        # 4. 执行调谐操作
        reconcile(current_state, desired_state)

    # 5. 等待一段时间
    sleep(10)
```

**示例：自动扩容**

```yaml
# 用户创建Milvus CR
apiVersion: milvus.io/v1beta1
kind: Milvus
metadata:
  name: my-milvus
spec:
  components:
    queryNode:
      replicas: 5  # 期望5个副本
```

```python
# Operator检测到变化
def reconcile():
    current_replicas = count_querynode_pods()  # 当前3个
    desired_replicas = milvus_cr.spec.components.queryNode.replicas  # 期望5个

    if current_replicas < desired_replicas:
        # 创建2个新Pod
        for i in range(desired_replicas - current_replicas):
            create_querynode_pod()
```

### 3. 领域知识封装

**Operator内置Milvus最佳实践：**

```python
# Operator自动配置反亲和性
def create_querynode_deployment():
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "spec": {
            "replicas": desired_replicas,
            "template": {
                "spec": {
                    # 自动配置：Pod分布在不同节点
                    "affinity": {
                        "podAntiAffinity": {
                            "requiredDuringSchedulingIgnoredDuringExecution": [{
                                "labelSelector": {
                                    "matchExpressions": [{
                                        "key": "app",
                                        "operator": "In",
                                        "values": ["milvus-querynode"]
                                    }]
                                },
                                "topologyKey": "kubernetes.io/hostname"
                            }]
                        }
                    },
                    # 自动配置：健康检查
                    "containers": [{
                        "livenessProbe": {
                            "httpGet": {"path": "/healthz", "port": 9091},
                            "initialDelaySeconds": 30,
                            "periodSeconds": 10
                        },
                        "readinessProbe": {
                            "httpGet": {"path": "/healthz", "port": 9091},
                            "initialDelaySeconds": 10,
                            "periodSeconds": 5
                        }
                    }]
                }
            }
        }
    }
    return deployment
```

---

## 安装Milvus Operator

### 方式1：使用Helm安装Operator

```bash
# 1. 添加Operator仓库
helm repo add milvus-operator https://zilliztech.github.io/milvus-operator/
helm repo update

# 2. 安装Operator
helm install milvus-operator milvus-operator/milvus-operator

# 3. 验证安装
kubectl get pods -n milvus-operator
# NAME                                READY   STATUS    RESTARTS   AGE
# milvus-operator-xxx                 1/1     Running   0          1m
```

### 方式2：使用kubectl安装

```bash
# 安装CRD和Operator
kubectl apply -f https://raw.githubusercontent.com/zilliztech/milvus-operator/main/deploy/manifests/deployment.yaml

# 验证CRD
kubectl get crd | grep milvus
# milvuses.milvus.io
```

---

## 使用Operator部署Milvus

### 基础部署

```yaml
# milvus-cluster.yaml
apiVersion: milvus.io/v1beta1
kind: Milvus
metadata:
  name: my-milvus
  namespace: default
spec:
  mode: cluster  # 集群模式
  # Operator自动配置所有组件
```

```bash
# 部署
kubectl apply -f milvus-cluster.yaml

# 查看状态
kubectl get milvus my-milvus
# NAME        MODE      STATUS    AGE
# my-milvus   cluster   Healthy   5m

# 查看详细信息
kubectl describe milvus my-milvus
```

### 自定义配置

```yaml
# milvus-custom.yaml
apiVersion: milvus.io/v1beta1
kind: Milvus
metadata:
  name: my-milvus
spec:
  mode: cluster

  # 依赖配置
  dependencies:
    etcd:
      inCluster:
        deletionPolicy: Retain  # 删除Milvus时保留etcd
        pvcDeletion: false
        values:
          replicaCount: 3
          resources:
            requests:
              memory: 2Gi
    storage:
      inCluster:
        deletionPolicy: Retain
        pvcDeletion: false
        values:
          mode: distributed  # MinIO分布式模式
          resources:
            requests:
              memory: 4Gi
    pulsar:
      inCluster:
        values:
          components:
            broker: true
          resources:
            requests:
              memory: 4Gi

  # 组件配置
  components:
    proxy:
      replicas: 3
      resources:
        requests:
          cpu: 2
          memory: 4Gi
        limits:
          cpu: 4
          memory: 8Gi

    queryNode:
      replicas: 5
      resources:
        requests:
          cpu: 4
          memory: 16Gi
        limits:
          cpu: 8
          memory: 32Gi

    dataNode:
      replicas: 3
      resources:
        requests:
          cpu: 2
          memory: 8Gi
        limits:
          cpu: 4
          memory: 16Gi

  # 配置文件
  config:
    log:
      level: info
    dataCoord:
      segment:
        maxSize: "512"
    queryNode:
      gracefulTime: "1000"
```

```bash
kubectl apply -f milvus-custom.yaml
```

### 使用外部依赖

```yaml
# milvus-external-deps.yaml
apiVersion: milvus.io/v1beta1
kind: Milvus
metadata:
  name: my-milvus
spec:
  mode: cluster

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
      secretRef: milvus-s3-secret  # 引用Secret

    # 外部Pulsar
    pulsar:
      external: true
      endpoint: pulsar://pulsar-broker.default.svc.cluster.local:6650
```

```yaml
# milvus-s3-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: milvus-s3-secret
type: Opaque
stringData:
  accesskey: "your-access-key"
  secretkey: "your-secret-key"
  bucketname: "milvus-bucket"
```

```bash
kubectl apply -f milvus-s3-secret.yaml
kubectl apply -f milvus-external-deps.yaml
```

---

## Operator的自动化能力

### 1. 自动配置最佳实践

```yaml
# 你只需要写
spec:
  mode: cluster
  components:
    queryNode:
      replicas: 3

# Operator自动配置
# - Pod反亲和性（不同节点）
# - 健康检查（Liveness/Readiness）
# - 资源限制（防止OOM）
# - 滚动更新策略
# - 服务发现（Service/Endpoints）
# - 配置管理（ConfigMap）
```

### 2. 自动故障恢复

```python
# Operator持续监控
def watch_milvus_health():
    while True:
        for component in ["proxy", "querynode", "datanode"]:
            if not is_healthy(component):
                # 自动重启异常组件
                restart_component(component)

                # 记录事件
                create_event(f"{component} restarted due to health check failure")

        sleep(10)
```

### 3. 自动升级

```yaml
# 修改镜像版本
spec:
  components:
    image: milvusdb/milvus:v2.4.0  # 从v2.3.0升级到v2.4.0
```

```python
# Operator执行滚动升级
def rolling_upgrade():
    for component in ["proxy", "querynode", "datanode"]:
        # 逐个更新Pod
        for pod in get_pods(component):
            # 1. 创建新版本Pod
            create_pod(component, new_version)

            # 2. 等待新Pod就绪
            wait_for_ready(new_pod)

            # 3. 删除旧版本Pod
            delete_pod(old_pod)

            # 4. 等待一段时间
            sleep(30)
```

### 4. 自动扩缩容

```yaml
# 修改副本数
spec:
  components:
    queryNode:
      replicas: 10  # 从5扩容到10
```

```python
# Operator自动扩容
def scale_querynode():
    current = count_querynode_pods()  # 5
    desired = 10

    # 创建5个新Pod
    for i in range(desired - current):
        create_querynode_pod()

    # 等待所有Pod就绪
    wait_for_all_ready()
```

---

## Operator vs Helm对比

| 维度 | Helm | Operator |
|------|------|----------|
| **部署复杂度** | 需要配置所有细节 | 只需高层意图 |
| **运维自动化** | 无（一次性操作） | 有（持续监控） |
| **故障恢复** | 手动 | 自动 |
| **升级策略** | 手动配置 | 自动滚动升级 |
| **领域知识** | 需要用户提供 | 内置最佳实践 |
| **学习成本** | 低 | 中 |
| **适用场景** | 简单部署 | 生产环境 |

---

## 在RAG系统中的应用

### 场景1：自动故障恢复

```yaml
# 部署Milvus
apiVersion: milvus.io/v1beta1
kind: Milvus
metadata:
  name: rag-milvus
spec:
  mode: cluster
  components:
    queryNode:
      replicas: 5
```

**故障场景：**
```
1. QueryNode-2 OOM被杀
2. Operator检测到Pod异常（10秒内）
3. Operator自动重启Pod
4. 新Pod启动并加载数据（30秒）
5. Service自动将流量路由到新Pod
6. 用户几乎无感知（总恢复时间<1分钟）
```

### 场景2：自动扩缩容

```yaml
# 高峰期扩容
kubectl patch milvus rag-milvus --type='json' -p='[
  {"op": "replace", "path": "/spec/components/queryNode/replicas", "value": 20}
]'

# Operator自动：
# 1. 创建15个新QueryNode Pod
# 2. 等待Pod就绪
# 3. 自动加入Service
# 4. 开始接收流量
```

### 场景3：零停机升级

```yaml
# 升级到新版本
kubectl patch milvus rag-milvus --type='json' -p='[
  {"op": "replace", "path": "/spec/components/image", "value": "milvusdb/milvus:v2.4.0"}
]'

# Operator自动滚动升级：
# 1. 逐个更新Pod
# 2. 确保始终有足够的副本在线
# 3. 新Pod就绪后才删除旧Pod
# 4. 整个过程服务不中断
```

---

## 最佳实践

### 1. 使用外部依赖（生产环境）

```yaml
# ✅ 推荐：外部依赖独立部署
dependencies:
  etcd:
    external: true
  storage:
    external: true
  pulsar:
    external: true

# ❌ 不推荐：内置依赖（仅开发环境）
dependencies:
  etcd:
    inCluster: true
```

**原因：**
- 外部依赖可以独立扩缩容
- 外部依赖可以独立备份恢复
- Milvus故障不影响依赖服务

### 2. 配置资源限制

```yaml
components:
  queryNode:
    resources:
      requests:
        cpu: 4
        memory: 16Gi
      limits:
        cpu: 8
        memory: 32Gi  # 防止OOM影响其他Pod
```

### 3. 启用监控

```yaml
spec:
  config:
    metrics:
      enabled: true
      port: 9091
```

### 4. 配置持久化策略

```yaml
dependencies:
  etcd:
    inCluster:
      deletionPolicy: Retain  # 删除Milvus时保留数据
      pvcDeletion: false
```

---

## 常见问题

### Q1: Operator和Helm能一起用吗？

**可以！推荐方式：**

```bash
# 1. 用Helm安装Operator
helm install milvus-operator milvus-operator/milvus-operator

# 2. 用Operator管理Milvus
kubectl apply -f milvus-cluster.yaml
```

### Q2: 如何查看Operator日志？

```bash
# 查看Operator Pod
kubectl get pods -n milvus-operator

# 查看日志
kubectl logs -f milvus-operator-xxx -n milvus-operator
```

### Q3: 如何删除Operator管理的Milvus？

```bash
# 删除Milvus CR
kubectl delete milvus my-milvus

# Operator自动清理所有相关资源
# 如果配置了deletionPolicy: Retain，数据会保留
```

---

## 总结

### Operator的核心价值

1. **简化配置**：只需高层意图，无需配置细节
2. **自动化运维**：持续监控，自动故障恢复
3. **领域知识**：内置Milvus最佳实践
4. **生产就绪**：适合大规模生产环境

### 何时使用Operator？

- ✅ 生产环境部署
- ✅ 大规模集群（>10节点）
- ✅ 需要自动化运维
- ✅ 需要自动故障恢复
- ✅ 多集群管理

### 下一步

学习完Operator后，继续学习：
- **集群配置与扩缩容**：优化生产环境性能
- **监控和告警**：保障服务稳定性
- **备份和恢复**：数据安全保障
