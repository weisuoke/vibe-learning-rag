# Phase 6: 安全与运维 - 知识点列表

> 目标：理解安全机制，掌握生产部署能力
> 学习时长：第 12-13 周
> 前置要求：Phase 1-5 完成

---

## 知识点列表

### 47. SecurityPolicy Trait 与安全模型
- deny-by-default 设计、安全策略 Trait、威胁模型、与 OpenClaw 对比
- 前端类比：CORS + CSP 安全策略
- ZeroClaw 场景：security/ — secure-by-default 的设计哲学

### 48. 配对认证机制
- 6 位配对码生成、Webhook token 发放、Channel 绑定、过期与撤销
- 前端类比：扫码登录 / 2FA 验证
- ZeroClaw 场景：identity.rs 中的配对流程和安全保证

### 49. 沙箱执行（Native/Docker）
- 文件系统隔离（workspace scoping）、命令白名单、Docker 容器限制
- 前端类比：iframe sandbox 属性
- ZeroClaw 场景：runtime/ — Native 和 Docker RuntimeAdapter

### 50. Secret Store 与密钥管理
- 加密存储、环境变量优先级、密钥轮换、敏感路径阻断
- 前端类比：.env 文件 + Vault 密钥管理
- ZeroClaw 场景：API Key、Webhook Secret 的安全存储

### 51. Gateway 架构与 Webhook
- HTTP 服务器设计、Webhook 路由、签名验证、localhost 绑定
- 前端类比：Express + nginx 反向代理
- ZeroClaw 场景：gateway/ — Webhook 接收和消息转发

### 52. Daemon 守护进程
- 后台运行、自动重启、PID 管理、资源监控
- 前端类比：pm2 进程管理
- ZeroClaw 场景：daemon/ — 24/7 自主运行模式

### 53. Cron 调度系统
- 定时任务配置、Cron 表达式、主动行为触发
- 前端类比：node-cron / setTimeout 定时任务
- ZeroClaw 场景：cron/ — 让 Agent 定时执行任务

### 54. 隧道与远程访问
- Cloudflare Tunnel、Tailscale、ngrok 集成、安全暴露服务
- 前端类比：ngrok 内网穿透开发调试
- ZeroClaw 场景：tunnel/ — 安全地将本地 Agent 暴露到公网
