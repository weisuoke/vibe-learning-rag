# 实战代码6：Kubernetes健康探测配置

> Kubernetes Liveness、Readiness、Startup Probe 的完整配置

---

## 概述

本文提供 Kubernetes 健康探测的完整配置示例，包括：
- Deployment YAML 配置
- Liveness Probe 配置
- Readiness Probe 配置
- Startup Probe 配置
- 探测参数调优
- 完整的 AI Agent API 部署配置

---

## 完整配置

### 1. 基础 Deployment 配置

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-agent-api
  namespace: default
  labels:
    app: ai-agent-api
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-agent-api
  template:
    metadata:
      labels:
        app: ai-agent-api
        version: v1.0.0
    spec:
      containers:
      - name: api
        image: ai-agent-api:1.0.0
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP

        # 环境变量
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: ai-agent-secrets
              key: database-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-agent-secrets
              key: openai-api-key
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"

        # 资源限制
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"

        # Liveness Probe（存活探测）
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
            scheme: HTTP
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 3

        # Readiness Probe（就绪探测）
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
            scheme: HTTP
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 2

        # Startup Probe（启动探测）
        startupProbe:
          httpGet:
            path: /startup
            port: 8000
            scheme: HTTP
          initialDelaySeconds: 0
          periodSeconds: 10
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 12  # 12 * 10s = 2 分钟启动时间
```

---

## 2. 探测参数详解

### Liveness Probe 参数

```yaml
livenessProbe:
  httpGet:
    path: /health              # 健康检查路径
    port: 8000                 # 端口
    scheme: HTTP               # 协议（HTTP 或 HTTPS）
  initialDelaySeconds: 30      # 容器启动后多久开始检查
  periodSeconds: 10            # 检查间隔
  timeoutSeconds: 5            # 单次检查超时时间
  successThreshold: 1          # 连续成功多少次标记为健康
  failureThreshold: 3          # 连续失败多少次重启 Pod
```

**参数说明：**

| 参数 | 说明 | 推荐值 | 原因 |
|------|------|--------|------|
| initialDelaySeconds | 启动后多久开始检查 | 30-60秒 | 给应用足够的启动时间 |
| periodSeconds | 检查间隔 | 10秒 | 不要太频繁，避免影响性能 |
| timeoutSeconds | 单次检查超时 | 5秒 | 足够长，避免误判 |
| failureThreshold | 连续失败多少次重启 | 3次 | 避免短暂抖动导致重启 |

**计算重启时间：**
```
重启时间 = failureThreshold × periodSeconds
         = 3 × 10s = 30秒
```

### Readiness Probe 参数

```yaml
readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 5       # 比 Liveness 更早开始
  periodSeconds: 5             # 比 Liveness 更频繁
  timeoutSeconds: 3            # 比 Liveness 更短
  successThreshold: 1          # 连续成功 1 次恢复流量
  failureThreshold: 2          # 连续失败 2 次停止流量
```

**参数说明：**

| 参数 | 说明 | 推荐值 | 原因 |
|------|------|--------|------|
| initialDelaySeconds | 启动后多久开始检查 | 5-10秒 | 比 Liveness 更早 |
| periodSeconds | 检查间隔 | 5秒 | 比 Liveness 更频繁 |
| timeoutSeconds | 单次检查超时 | 3秒 | 比 Liveness 更短 |
| failureThreshold | 连续失败多少次停止流量 | 2次 | 比 Liveness 更敏感 |

**计算停止流量时间：**
```
停止流量时间 = failureThreshold × periodSeconds
            = 2 × 5s = 10秒
```

### Startup Probe 参数

```yaml
startupProbe:
  httpGet:
    path: /startup
    port: 8000
  initialDelaySeconds: 0       # 立即开始检查
  periodSeconds: 10            # 检查间隔
  timeoutSeconds: 5            # 超时时间
  successThreshold: 1          # 成功 1 次即可
  failureThreshold: 12         # 给足够的启动时间
```

**参数说明：**

| 参数 | 说明 | 推荐值 | 原因 |
|------|------|--------|------|
| initialDelaySeconds | 启动后多久开始检查 | 0秒 | 立即开始检查 |
| periodSeconds | 检查间隔 | 10秒 | 不要太频繁 |
| failureThreshold | 连续失败多少次重启 | 12-30次 | 给足够的启动时间 |

**计算最大启动时间：**
```
最大启动时间 = failureThreshold × periodSeconds
            = 12 × 10s = 2分钟
```

---

## 3. 不同场景的配置

### 场景1：快速启动的应用

```yaml
# 不需要 Startup Probe
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10      # 启动快，10秒即可
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 2
```

### 场景2：慢启动的应用（加载 ML 模型）

```yaml
# 需要 Startup Probe
startupProbe:
  httpGet:
    path: /startup
    port: 8000
  initialDelaySeconds: 0
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 30         # 30 * 10s = 5 分钟启动时间

livenessProbe:
  httpGet:
    path: /health
    port: 8000
  # Startup Probe 成功后才开始执行
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  # Startup Probe 成功后才开始执行
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 2
```

### 场景3：高可用应用

```yaml
# 更敏感的探测配置
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 5             # 更频繁的检查
  timeoutSeconds: 3
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 3             # 更频繁的检查
  timeoutSeconds: 2
  failureThreshold: 2
```

---

## 4. 完整的 AI Agent API 部署

```yaml
# ai-agent-deployment.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ai-agent

---
apiVersion: v1
kind: Secret
metadata:
  name: ai-agent-secrets
  namespace: ai-agent
type: Opaque
stringData:
  database-url: "postgresql://user:password@postgres:5432/dbname"
  openai-api-key: "sk-..."
  redis-url: "redis://redis:6379/0"

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-agent-api
  namespace: ai-agent
  labels:
    app: ai-agent-api
    component: backend
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: ai-agent-api
  template:
    metadata:
      labels:
        app: ai-agent-api
        component: backend
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: api
        image: ai-agent-api:1.0.0
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP

        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: ai-agent-secrets
              key: database-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-agent-secrets
              key: openai-api-key
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: ai-agent-secrets
              key: redis-url
        - name: LOG_LEVEL
          value: "INFO"
        - name: ENVIRONMENT
          value: "production"

        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"

        # Startup Probe：给 2 分钟启动时间
        startupProbe:
          httpGet:
            path: /startup
            port: 8000
          initialDelaySeconds: 0
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 12

        # Liveness Probe：检测死锁
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        # Readiness Probe：检测依赖服务
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2

        # 优雅关闭
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 5"]

---
apiVersion: v1
kind: Service
metadata:
  name: ai-agent-api
  namespace: ai-agent
  labels:
    app: ai-agent-api
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: ai-agent-api

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-agent-api
  namespace: ai-agent
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: ai-agent-tls
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ai-agent-api
            port:
              number: 80
```

---

## 5. 部署和测试

### 部署应用

```bash
# 1. 创建命名空间
kubectl create namespace ai-agent

# 2. 创建 Secret
kubectl apply -f secrets.yaml

# 3. 部署应用
kubectl apply -f ai-agent-deployment.yaml

# 4. 查看部署状态
kubectl get pods -n ai-agent
kubectl get svc -n ai-agent
```

### 测试健康检查

```bash
# 1. 查看 Pod 状态
kubectl get pods -n ai-agent

# 输出示例：
# NAME                            READY   STATUS    RESTARTS   AGE
# ai-agent-api-7d9f8b5c6d-abc12   1/1     Running   0          2m
# ai-agent-api-7d9f8b5c6d-def34   1/1     Running   0          2m
# ai-agent-api-7d9f8b5c6d-ghi56   1/1     Running   0          2m

# 2. 查看 Pod 详情
kubectl describe pod ai-agent-api-7d9f8b5c6d-abc12 -n ai-agent

# 3. 查看健康检查日志
kubectl logs ai-agent-api-7d9f8b5c6d-abc12 -n ai-agent | grep health

# 4. 手动测试健康检查端点
kubectl port-forward -n ai-agent ai-agent-api-7d9f8b5c6d-abc12 8000:8000

# 在另一个终端测试
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

### 查看健康检查事件

```bash
# 查看 Pod 事件
kubectl get events -n ai-agent --field-selector involvedObject.name=ai-agent-api-7d9f8b5c6d-abc12

# 输出示例：
# LAST SEEN   TYPE      REASON      OBJECT                              MESSAGE
# 2m          Normal    Scheduled   pod/ai-agent-api-7d9f8b5c6d-abc12   Successfully assigned...
# 2m          Normal    Pulled      pod/ai-agent-api-7d9f8b5c6d-abc12   Container image pulled
# 2m          Normal    Created     pod/ai-agent-api-7d9f8b5c6d-abc12   Created container api
# 2m          Normal    Started     pod/ai-agent-api-7d9f8b5c6d-abc12   Started container api
```

---

## 6. 常见问题和调优

### 问题1：Pod 频繁重启

**原因：** Liveness Probe 配置太敏感

**解决方案：**

```yaml
livenessProbe:
  initialDelaySeconds: 60      # 增加启动延迟
  periodSeconds: 15            # 增加检查间隔
  timeoutSeconds: 10           # 增加超时时间
  failureThreshold: 5          # 增加失败阈值
```

### 问题2：启动时被误判为不健康

**原因：** 没有配置 Startup Probe

**解决方案：**

```yaml
startupProbe:
  httpGet:
    path: /startup
    port: 8000
  periodSeconds: 10
  failureThreshold: 30         # 给足够的启动时间
```

### 问题3：依赖服务短暂不可用导致流量中断

**原因：** Readiness Probe 太敏感

**解决方案：**

```yaml
readinessProbe:
  periodSeconds: 10            # 增加检查间隔
  failureThreshold: 3          # 增加失败阈值
```

或者在应用中使用缓存：

```python
# 健康检查缓存 30 秒
@app.get("/ready")
async def ready():
    # 使用缓存，避免短暂抖动
    status = await get_cached_health_status()
    return status
```

---

## 7. 监控健康检查

### Prometheus 指标

```yaml
# ServiceMonitor for Prometheus
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ai-agent-api
  namespace: ai-agent
spec:
  selector:
    matchLabels:
      app: ai-agent-api
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

### Grafana 仪表板

```json
{
  "dashboard": {
    "title": "AI Agent Health Check",
    "panels": [
      {
        "title": "Pod Restarts",
        "targets": [
          {
            "expr": "rate(kube_pod_container_status_restarts_total{namespace=\"ai-agent\"}[5m])"
          }
        ]
      },
      {
        "title": "Health Check Success Rate",
        "targets": [
          {
            "expr": "rate(health_check_total{status=\"success\"}[5m]) / rate(health_check_total[5m])"
          }
        ]
      }
    ]
  }
}
```

---

## 总结

Kubernetes 健康探测配置的关键：

1. **Liveness Probe**：检测死锁，失败重启 Pod
2. **Readiness Probe**：检测依赖服务，失败停止流量
3. **Startup Probe**：给慢启动应用足够时间
4. **参数调优**：平衡检测速度和稳定性
5. **监控集成**：使用 Prometheus 监控健康状态

在 AI Agent 后端中，合理的 Kubernetes 健康探测配置可以实现自动化的故障检测和恢复，提高系统可用性。
