# 核心概念1：Helm Charts 部署

## 什么是Helm Charts？

**Helm Charts是Kubernetes的包管理器，类似npm之于Node.js、pip之于Python。**

一句话定义：**Helm Charts将多个Kubernetes资源打包成一个可复用、可配置的部署单元。**

## 为什么需要Helm Charts？

### 问题：原生Kubernetes部署的痛点

```bash
# 部署Milvus需要创建多个YAML文件
kubectl apply -f proxy-deployment.yaml
kubectl apply -f rootcoord-deployment.yaml
kubectl apply -f querynode-deployment.yaml
kubectl apply -f datanode-deployment.yaml
kubectl apply -f indexnode-deployment.yaml
kubectl apply -f etcd-statefulset.yaml
kubectl apply -f minio-deployment.yaml
kubectl apply -f pulsar-deployment.yaml
kubectl apply -f proxy-service.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
# 11个文件，容易遗漏或顺序错误
```

**痛点：**
1. **文件分散**：多个YAML文件难以管理
2. **配置重复**：相同的配置在多个文件中重复
3. **环境差异**：开发、测试、生产环境需要不同配置，需要维护多套文件
4. **版本管理**：难以追踪哪些文件属于哪个版本
5. **依赖管理**：组件之间的依赖关系需要手动维护

### 解决方案：Helm Charts

```bash
# 一条命令部署所有组件
helm install milvus milvus/milvus

# 自定义配置
helm install milvus milvus/milvus -f values.yaml

# 升级
helm upgrade milvus milvus/milvus --version 4.1.0

# 回滚
helm rollback milvus 1
```

---

## Helm的核心概念

### 1. Chart（图表）

**Chart = 应用的打包格式**

```
milvus-chart/
├── Chart.yaml          # Chart元数据（名称、版本、描述）
├── values.yaml         # 默认配置值
├── templates/          # Kubernetes资源模板
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   └── _helpers.tpl   # 模板辅助函数
├── charts/             # 依赖的子Chart
│   ├── etcd/
│   ├── minio/
│   └── pulsar/
└── README.md           # 使用说明
```

**类比：**
- Chart = npm包（package.json + 源代码）
- Chart.yaml = package.json（包的元数据）
- values.yaml = .env（配置参数）
- templates/ = src/（应用代码）

### 2. Release（发布）

**Release = Chart的一次部署实例**

```bash
# 创建Release
helm install my-milvus milvus/milvus
# Release名称：my-milvus
# Chart：milvus/milvus

# 同一个Chart可以部署多次
helm install milvus-dev milvus/milvus -f dev-values.yaml
helm install milvus-prod milvus/milvus -f prod-values.yaml
# 两个独立的Release，互不影响
```

**类比：**
- Chart = 软件安装包（.exe、.dmg）
- Release = 安装后的软件实例
- 同一个软件可以安装多次（不同目录、不同配置）

### 3. Repository（仓库）

**Repository = Chart的存储和分发中心**

```bash
# 添加官方仓库
helm repo add milvus https://zilliztech.github.io/milvus-helm/

# 搜索Chart
helm search repo milvus
# NAME                    CHART VERSION   APP VERSION     DESCRIPTION
# milvus/milvus           4.1.0           2.3.0           Milvus is an open-source vector database

# 更新仓库索引
helm repo update

# 查看所有仓库
helm repo list
```

**类比：**
- Repository = npm registry、PyPI
- helm repo add = npm config set registry
- helm search = npm search

---

## Helm模板引擎

### 模板语法

Helm使用Go模板语法，支持变量替换、条件判断、循环等。

```yaml
# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "milvus.fullname" . }}-proxy
  labels:
    {{- include "milvus.labels" . | nindent 4 }}
    component: proxy
spec:
  replicas: {{ .Values.proxy.replicas }}
  selector:
    matchLabels:
      {{- include "milvus.selectorLabels" . | nindent 6 }}
      component: proxy
  template:
    metadata:
      labels:
        {{- include "milvus.selectorLabels" . | nindent 8 }}
        component: proxy
    spec:
      containers:
      - name: proxy
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        resources:
          {{- toYaml .Values.proxy.resources | nindent 10 }}
        env:
        {{- range .Values.proxy.env }}
        - name: {{ .name }}
          value: {{ .value | quote }}
        {{- end }}
```

### 变量引用

```yaml
# values.yaml
proxy:
  replicas: 3
  resources:
    requests:
      cpu: 2
      memory: 4Gi
  env:
    - name: LOG_LEVEL
      value: info

image:
  repository: milvusdb/milvus
  tag: v2.3.0
```

**模板渲染结果：**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-milvus-proxy
  labels:
    app.kubernetes.io/name: milvus
    app.kubernetes.io/instance: my-milvus
    component: proxy
spec:
  replicas: 3
  selector:
    matchLabels:
      app.kubernetes.io/name: milvus
      app.kubernetes.io/instance: my-milvus
      component: proxy
  template:
    metadata:
      labels:
        app.kubernetes.io/name: milvus
        app.kubernetes.io/instance: my-milvus
        component: proxy
    spec:
      containers:
      - name: proxy
        image: "milvusdb/milvus:v2.3.0"
        resources:
          requests:
            cpu: 2
            memory: 4Gi
        env:
        - name: LOG_LEVEL
          value: "info"
```

### 条件判断

```yaml
# templates/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ include "milvus.fullname" . }}
spec:
  type: {{ .Values.service.type }}
  {{- if eq .Values.service.type "LoadBalancer" }}
  loadBalancerIP: {{ .Values.service.loadBalancerIP }}
  {{- end }}
  {{- if eq .Values.service.type "NodePort" }}
  ports:
  - port: {{ .Values.service.port }}
    nodePort: {{ .Values.service.nodePort }}
  {{- else }}
  ports:
  - port: {{ .Values.service.port }}
  {{- end }}
```

### 循环

```yaml
# templates/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "milvus.fullname" . }}-config
data:
  {{- range $key, $value := .Values.config }}
  {{ $key }}: {{ $value | quote }}
  {{- end }}
```

```yaml
# values.yaml
config:
  log.level: info
  dataCoord.segment.maxSize: "512"
  queryNode.gracefulTime: "1000"
```

**渲染结果：**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: my-milvus-config
data:
  log.level: "info"
  dataCoord.segment.maxSize: "512"
  queryNode.gracefulTime: "1000"
```

---

## Helm部署Milvus实战

### 基础部署

```bash
# 1. 添加Milvus仓库
helm repo add milvus https://zilliztech.github.io/milvus-helm/
helm repo update

# 2. 查看可用版本
helm search repo milvus/milvus --versions

# 3. 部署（使用默认配置）
helm install my-milvus milvus/milvus

# 4. 查看部署状态
helm status my-milvus

# 5. 查看所有资源
kubectl get all -l app.kubernetes.io/instance=my-milvus
```

### 自定义配置部署

```yaml
# custom-values.yaml

# 集群模式
cluster:
  enabled: true

# Proxy配置
proxy:
  replicas: 3
  resources:
    requests:
      cpu: 2
      memory: 4Gi
    limits:
      cpu: 4
      memory: 8Gi

# QueryNode配置
queryNode:
  replicas: 5
  resources:
    requests:
      cpu: 4
      memory: 16Gi
    limits:
      cpu: 8
      memory: 32Gi

# DataNode配置
dataNode:
  replicas: 3
  resources:
    requests:
      cpu: 2
      memory: 8Gi
    limits:
      cpu: 4
      memory: 16Gi

# 持久化存储
persistence:
  enabled: true
  storageClass: "fast-ssd"
  size: 100Gi

# 外部etcd（生产推荐）
externalEtcd:
  enabled: true
  endpoints:
    - etcd-0.etcd.default.svc.cluster.local:2379
    - etcd-1.etcd.default.svc.cluster.local:2379
    - etcd-2.etcd.default.svc.cluster.local:2379

# 外部S3（生产推荐）
externalS3:
  enabled: true
  host: "s3.amazonaws.com"
  port: 443
  useSSL: true
  bucketName: "milvus-bucket"
  accessKey: "your-access-key"
  secretKey: "your-secret-key"

# 外部Pulsar（生产推荐）
externalPulsar:
  enabled: true
  host: "pulsar-broker.default.svc.cluster.local"
  port: 6650

# Service配置
service:
  type: LoadBalancer
  port: 19530
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"

# 监控
metrics:
  enabled: true
  serviceMonitor:
    enabled: true
```

```bash
# 使用自定义配置部署
helm install my-milvus milvus/milvus -f custom-values.yaml

# 或使用命令行参数覆盖
helm install my-milvus milvus/milvus \
  --set proxy.replicas=5 \
  --set queryNode.replicas=10 \
  --set service.type=LoadBalancer
```

### 升级和回滚

```bash
# 查看当前版本
helm list

# 升级到新版本
helm upgrade my-milvus milvus/milvus --version 4.1.0

# 修改配置并升级
helm upgrade my-milvus milvus/milvus -f new-values.yaml

# 查看升级历史
helm history my-milvus
# REVISION  UPDATED                   STATUS      CHART           DESCRIPTION
# 1         Mon Jan 1 10:00:00 2024   superseded  milvus-4.0.0    Install complete
# 2         Mon Jan 2 11:00:00 2024   deployed    milvus-4.1.0    Upgrade complete

# 回滚到上一个版本
helm rollback my-milvus

# 回滚到指定版本
helm rollback my-milvus 1
```

### 调试和验证

```bash
# 渲染模板（不部署）
helm template my-milvus milvus/milvus -f values.yaml

# 验证Chart
helm lint milvus/

# 模拟安装（dry-run）
helm install my-milvus milvus/milvus --dry-run --debug

# 查看实际使用的values
helm get values my-milvus

# 查看所有values（包括默认值）
helm get values my-milvus --all
```

---

## 在RAG系统中的应用

### 场景1：多环境部署

**需求：**开发、测试、生产环境使用不同配置

```yaml
# dev-values.yaml（开发环境）
cluster:
  enabled: false  # 单机模式

proxy:
  replicas: 1

queryNode:
  replicas: 1
  resources:
    requests:
      cpu: 1
      memory: 2Gi

persistence:
  enabled: false  # 不持久化，快速重建

externalEtcd:
  enabled: false  # 使用内置etcd

externalS3:
  enabled: false  # 使用内置MinIO
```

```yaml
# prod-values.yaml（生产环境）
cluster:
  enabled: true  # 集群模式

proxy:
  replicas: 3

queryNode:
  replicas: 10
  resources:
    requests:
      cpu: 4
      memory: 16Gi

persistence:
  enabled: true
  storageClass: "fast-ssd"
  size: 500Gi

externalEtcd:
  enabled: true
  endpoints:
    - etcd-0.etcd:2379
    - etcd-1.etcd:2379
    - etcd-2.etcd:2379

externalS3:
  enabled: true
  host: "s3.amazonaws.com"
  bucketName: "prod-milvus-bucket"
```

```bash
# 部署开发环境
helm install milvus-dev milvus/milvus -f dev-values.yaml -n dev

# 部署生产环境
helm install milvus-prod milvus/milvus -f prod-values.yaml -n prod
```

### 场景2：灰度发布

**需求：**新版本先在部分流量上验证，再全量发布

```bash
# 第1步：部署稳定版本（v2.3.0）
helm install milvus-stable milvus/milvus \
  --set image.tag=v2.3.0 \
  --set proxy.replicas=8

# 第2步：部署灰度版本（v2.4.0）
helm install milvus-canary milvus/milvus \
  --set image.tag=v2.4.0 \
  --set proxy.replicas=2 \
  --set service.name=milvus-canary

# 第3步：配置流量分配（使用Ingress）
# 90%流量 → milvus-stable
# 10%流量 → milvus-canary

# 第4步：观察灰度版本指标
# - 错误率
# - 响应时间
# - 资源使用

# 第5步：灰度成功，全量发布
helm upgrade milvus-stable milvus/milvus --set image.tag=v2.4.0
helm uninstall milvus-canary
```

### 场景3：动态扩缩容

**需求：**根据业务负载动态调整资源

```bash
# 高峰期扩容
helm upgrade my-milvus milvus/milvus \
  --set queryNode.replicas=20 \
  --set proxy.replicas=10 \
  --reuse-values

# 低峰期缩容
helm upgrade my-milvus milvus/milvus \
  --set queryNode.replicas=5 \
  --set proxy.replicas=3 \
  --reuse-values
```

---

## Helm最佳实践

### 1. 使用values文件而非命令行参数

```bash
# ❌ 不推荐：命令行参数难以追踪
helm install milvus milvus/milvus \
  --set proxy.replicas=3 \
  --set queryNode.replicas=5 \
  --set service.type=LoadBalancer \
  --set persistence.enabled=true \
  --set persistence.size=100Gi

# ✅ 推荐：使用values文件
helm install milvus milvus/milvus -f prod-values.yaml
```

### 2. 版本控制values文件

```bash
# 将values文件纳入Git管理
git add values/
git commit -m "Update Milvus configuration"

# 目录结构
values/
├── dev-values.yaml
├── test-values.yaml
└── prod-values.yaml
```

### 3. 使用命名空间隔离环境

```bash
# 创建命名空间
kubectl create namespace dev
kubectl create namespace prod

# 部署到不同命名空间
helm install milvus-dev milvus/milvus -f dev-values.yaml -n dev
helm install milvus-prod milvus/milvus -f prod-values.yaml -n prod
```

### 4. 设置资源限制

```yaml
# 防止资源耗尽
queryNode:
  resources:
    requests:
      cpu: 4
      memory: 16Gi
    limits:
      cpu: 8
      memory: 32Gi  # 限制最大内存，防止OOM影响其他Pod
```

### 5. 启用持久化存储

```yaml
# 生产环境必须启用
persistence:
  enabled: true
  storageClass: "fast-ssd"
  size: 500Gi

# 使用外部存储
externalS3:
  enabled: true
  # 云存储天然高可用
```

### 6. 配置健康检查

```yaml
# Helm Chart通常已包含，确认启用
livenessProbe:
  enabled: true
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  enabled: true
  initialDelaySeconds: 10
  periodSeconds: 5
```

---

## 常见问题

### Q1: Helm安装失败，如何排查？

```bash
# 1. 查看详细日志
helm install my-milvus milvus/milvus --debug

# 2. 查看Pod状态
kubectl get pods -l app.kubernetes.io/instance=my-milvus

# 3. 查看Pod日志
kubectl logs <pod-name>

# 4. 查看Pod事件
kubectl describe pod <pod-name>

# 5. 检查资源是否创建
kubectl get all -l app.kubernetes.io/instance=my-milvus
```

### Q2: 如何查看Helm使用的默认值？

```bash
# 查看Chart的默认values.yaml
helm show values milvus/milvus > default-values.yaml

# 查看实际部署使用的values
helm get values my-milvus

# 查看所有values（包括默认值）
helm get values my-milvus --all
```

### Q3: 如何只更新部分配置？

```bash
# 使用--reuse-values保留现有配置
helm upgrade my-milvus milvus/milvus \
  --set queryNode.replicas=10 \
  --reuse-values

# 或使用--set覆盖特定值
helm upgrade my-milvus milvus/milvus \
  -f prod-values.yaml \
  --set image.tag=v2.4.0
```

### Q4: 如何完全删除Milvus？

```bash
# 1. 删除Helm Release
helm uninstall my-milvus

# 2. 删除PVC（可选，会删除数据）
kubectl delete pvc -l app.kubernetes.io/instance=my-milvus

# 3. 删除命名空间（如果独立命名空间）
kubectl delete namespace milvus
```

---

## 总结

### Helm Charts的核心价值

| 维度 | 价值 |
|------|------|
| **打包** | 将多个资源打包成一个Chart |
| **模板化** | 通过变量实现配置复用 |
| **版本管理** | 支持升级、回滚、历史记录 |
| **依赖管理** | 自动管理组件依赖关系 |
| **标准化** | 提供最佳实践配置模板 |

### 适用场景

- ✅ 快速部署和测试
- ✅ 多环境配置管理
- ✅ 版本升级和回滚
- ✅ 标准化部署流程
- ✅ 团队协作（共享配置）

### 下一步

学习完Helm Charts后，可以继续学习：
- **Milvus Operator**：更高级的自动化运维
- **集群配置与扩缩容**：生产环境优化
- **监控和告警**：保障服务稳定性
