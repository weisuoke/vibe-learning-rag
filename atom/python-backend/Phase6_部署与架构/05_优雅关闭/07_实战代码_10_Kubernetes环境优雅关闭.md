# 实战代码：Kubernetes环境优雅关闭

> Kubernetes 环境中的优雅关闭配置和实现

---

## 代码说明

本示例演示如何在 Kubernetes 环境中配置优雅关闭，包括：
- terminationGracePeriodSeconds 配置
- preStop 钩子实现
- 与健康检查的配合
- 零停机部署

---

## Kubernetes 配置

### 1. 完整的 Pod 配置

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  labels:
    app: myapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      # 优雅关闭超时时间（秒）
      terminationGracePeriodSeconds: 60

      containers:
      - name: app
        image: myapp:latest
        ports:
        - containerPort: 8000
          name: http

        # 环境变量
        env:
        - name: GRACEFUL_SHUTDOWN_TIMEOUT
          value: "50"  # 应用内部超时应小于 terminationGracePeriodSeconds

        # 资源限制
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"

        # 存活探测
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        # 就绪探测
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2

        # 生命周期钩子
        lifecycle:
          preStop:
            exec:
              command:
              - /bin/sh
              - -c
              - |
                # 给5秒时间让 readiness 探测失败
                # 这样负载均衡器会停止发送新请求
                sleep 5
```

### 2. Service 配置

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: myapp
spec:
  selector:
    app: myapp
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

---

## 应用代码

### 1. 健康检查端点

```python
"""
健康检查端点实现
"""

from fastapi import FastAPI, Response
from enum import Enum

app = FastAPI()

class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    SHUTTING_DOWN = "shutting_down"

# 全局状态
health_status = HealthStatus.HEALTHY
ready_status = True

@app.get("/health")
async def health_check():
    """
    存活探测端点

    用途：检查应用是否存活
    返回：只要进程在运行就返回 200
    """
    if health_status == HealthStatus.UNHEALTHY:
        return Response(
            content='{"status": "unhealthy"}',
            status_code=503,
            media_type="application/json"
        )

    return {"status": health_status.value}

@app.get("/ready")
async def readiness_check():
    """
    就绪探测端点

    用途：检查应用是否准备好接收流量
    返回：
    - 200: 准备好接收流量
    - 503: 未准备好（启动中或关闭中）
    """
    global ready_status

    if not ready_status:
        return Response(
            content='{"status": "not_ready", "reason": "shutting_down"}',
            status_code=503,
            media_type="application/json"
        )

    # 检查依赖服务
    if not await check_dependencies():
        return Response(
            content='{"status": "not_ready", "reason": "dependencies_unavailable"}',
            status_code=503,
            media_type="application/json"
        )

    return {"status": "ready"}

async def check_dependencies():
    """检查依赖服务"""
    # 检查数据库
    try:
        async with db_engine.connect() as conn:
            await conn.execute("SELECT 1")
    except:
        return False

    # 检查 Redis
    try:
        await redis_client.ping()
    except:
        return False

    return True
```

### 2. 优雅关闭实现

```python
"""
Kubernetes 环境的优雅关闭实现
"""

import signal
import asyncio
import sys
import os

# 从环境变量读取超时配置
GRACEFUL_SHUTDOWN_TIMEOUT = int(
    os.getenv("GRACEFUL_SHUTDOWN_TIMEOUT", "50")
)

shutdown_event = asyncio.Event()
accepting_requests = True
active_requests = 0

def signal_handler(signum, frame):
    """信号处理器"""
    print(f"收到信号 {signum}")
    shutdown_event.set()

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

async def graceful_shutdown():
    """优雅关闭"""
    global ready_status, accepting_requests

    print(f"开始优雅关闭（超时 {GRACEFUL_SHUTDOWN_TIMEOUT}秒）")

    # 1. 标记为未就绪（readiness 探测会失败）
    ready_status = False
    print("✓ 标记为未就绪")

    # 2. 等待5秒，让 K8s 从 Service 中移除 Pod
    await asyncio.sleep(5)
    print("✓ 已从 Service 中移除")

    # 3. 停止接收新请求
    accepting_requests = False
    print("✓ 停止接收新请求")

    # 4. 等待现有请求完成
    timeout = GRACEFUL_SHUTDOWN_TIMEOUT - 5  # 减去前面等待的5秒
    try:
        await asyncio.wait_for(
            wait_for_requests(),
            timeout=timeout
        )
        print("✓ 所有请求已完成")
    except asyncio.TimeoutError:
        print(f"⚠ 请求排空超时（{timeout}秒）")

    # 5. 清理资源
    await cleanup_resources()

    print("优雅关闭完成")
    sys.exit(0)

async def wait_for_requests():
    """等待所有请求完成"""
    while active_requests > 0:
        await asyncio.sleep(0.1)

async def cleanup_resources():
    """清理资源"""
    # 关闭数据库连接
    if db_engine:
        await db_engine.dispose()

    # 关闭 Redis 连接
    if redis_client:
        await redis_client.close()
```

---

## 优雅关闭流程

### Kubernetes 删除 Pod 的完整流程

```
1. kubectl delete pod myapp-xxx
   ↓
2. Pod 状态变为 Terminating
   ↓
3. 同时执行两个操作：
   a) 从 Service 的 Endpoints 中移除 Pod
   b) 执行 preStop 钩子（sleep 5）
   ↓
4. 向容器的 PID 1 发送 SIGTERM 信号
   ↓
5. 应用执行优雅关闭：
   - 标记 readiness 为 false
   - 停止接收新请求
   - 等待现有请求完成
   - 清理资源
   ↓
6. 等待 terminationGracePeriodSeconds（60秒）
   ↓
7. 如果容器还在运行，发送 SIGKILL 强制终止
```

### 时间线示例

```
T+0s:  kubectl delete pod
T+0s:  Pod 从 Service 移除 + preStop 钩子开始
T+5s:  preStop 钩子完成，发送 SIGTERM
T+5s:  应用标记 readiness=false
T+10s: 应用停止接收新请求
T+10s: 应用等待现有请求完成
T+15s: 所有请求完成
T+15s: 应用清理资源
T+16s: 应用退出（exit 0）
T+60s: 如果应用还未退出，K8s 发送 SIGKILL
```

---

## Dockerfile

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 确保 Python 进程是 PID 1
# 使用 exec 形式，不使用 shell
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 测试脚本

### 1. 部署应用

```bash
#!/bin/bash
# deploy.sh

echo "部署应用到 Kubernetes..."

# 应用配置
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# 等待 Pod 就绪
kubectl wait --for=condition=ready pod -l app=myapp --timeout=60s

echo "部署完成"
```

### 2. 测试优雅关闭

```bash
#!/bin/bash
# test_graceful_shutdown.sh

echo "=== 测试 Kubernetes 优雅关闭 ==="

# 1. 获取 Pod 名称
POD=$(kubectl get pods -l app=myapp -o jsonpath='{.items[0].metadata.name}')
echo "Pod: $POD"

# 2. 发送请求到 Pod
echo "发送慢速请求..."
kubectl exec -it $POD -- curl -s http://localhost:8000/slow &

# 3. 等待1秒
sleep 1

# 4. 删除 Pod
echo "删除 Pod..."
kubectl delete pod $POD &

# 5. 查看日志
echo "查看日志..."
kubectl logs -f $POD

echo "测试完成"
```

### 3. 监控优雅关闭

```bash
#!/bin/bash
# monitor_shutdown.sh

POD=$(kubectl get pods -l app=myapp -o jsonpath='{.items[0].metadata.name}')

# 监控 Pod 事件
kubectl get events --field-selector involvedObject.name=$POD -w &

# 监控 Pod 状态
watch -n 1 "kubectl get pod $POD"
```

---

## 最佳实践

### 1. 超时时间配置

```yaml
# 推荐配置
terminationGracePeriodSeconds: 60  # K8s 全局超时

env:
- name: GRACEFUL_SHUTDOWN_TIMEOUT
  value: "50"  # 应用超时应小于 K8s 超时
```

**原则：**
- 应用超时 < terminationGracePeriodSeconds
- 留出10秒缓冲时间
- 考虑最慢请求的处理时间

### 2. preStop 钩子

```yaml
lifecycle:
  preStop:
    exec:
      command: ["/bin/sh", "-c", "sleep 5"]
```

**作用：**
- 给 K8s 时间从 Service 中移除 Pod
- 避免新请求发送到正在关闭的 Pod
- 5秒通常足够

### 3. 健康检查配置

```yaml
readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  periodSeconds: 5  # 每5秒检查一次
  failureThreshold: 2  # 失败2次后标记为未就绪
```

**原则：**
- readiness 用于流量控制
- liveness 用于重启检测
- 关闭时只需要 readiness 失败

### 4. 滚动更新策略

```yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0  # 确保始终有可用的 Pod
      maxSurge: 1        # 一次最多创建1个新 Pod
```

---

## 常见问题

### Q1: 为什么需要 preStop 钩子？

**A:** 因为 K8s 同时执行两个操作：
1. 从 Service 移除 Pod
2. 发送 SIGTERM 信号

如果没有 preStop 钩子，应用可能在从 Service 移除前就开始关闭，导致新请求失败。

### Q2: terminationGracePeriodSeconds 设置多少合适？

**A:** 取决于应用的请求处理时间：
- 快速 API：30秒
- 普通应用：60秒
- LLM 流式响应：90-120秒

### Q3: 如何验证优雅关闭是否生效？

**A:**
```bash
# 1. 发送慢速请求
curl http://myapp/slow &

# 2. 立即删除 Pod
kubectl delete pod myapp-xxx

# 3. 检查请求是否正常完成
# 如果返回 200，说明优雅关闭生效
```

### Q4: 如何处理超长请求？

**A:**
- 增加 terminationGracePeriodSeconds
- 或者在应用中检测关闭信号并中断请求
- 或者使用任务队列处理长时间任务

---

## 监控和告警

### 1. Prometheus 指标

```python
from prometheus_client import Counter, Histogram

# 优雅关闭指标
graceful_shutdown_total = Counter(
    "graceful_shutdown_total",
    "Total number of graceful shutdowns"
)

graceful_shutdown_duration = Histogram(
    "graceful_shutdown_duration_seconds",
    "Graceful shutdown duration"
)

# 在优雅关闭时记录
graceful_shutdown_total.inc()
with graceful_shutdown_duration.time():
    await graceful_shutdown()
```

### 2. 日志记录

```python
import structlog

logger = structlog.get_logger()

async def graceful_shutdown():
    logger.info(
        "graceful_shutdown_started",
        timeout=GRACEFUL_SHUTDOWN_TIMEOUT
    )

    # ... 执行关闭 ...

    logger.info(
        "graceful_shutdown_completed",
        duration=elapsed_time
    )
```

---

## 总结

Kubernetes 环境的优雅关闭关键点：

1. **terminationGracePeriodSeconds**：设置合理的全局超时
2. **preStop 钩子**：给时间让 Pod 从 Service 移除
3. **readiness 探测**：关闭时标记为未就绪
4. **应用超时**：应小于 K8s 超时
5. **滚动更新**：配置 maxUnavailable=0 确保可用性

**完整流程：**
```
删除 Pod → preStop 钩子 → SIGTERM →
标记未就绪 → 停止接收 → 等待完成 →
清理资源 → 退出
```

**测试验证：**
- 发送慢速请求后立即删除 Pod
- 检查请求是否正常完成
- 查看日志确认优雅关闭流程
- 监控 Pod 事件和状态变化
