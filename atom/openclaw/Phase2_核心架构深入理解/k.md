# Phase 2: 核心架构深入理解

**目标：** 深入理解 OpenClaw 的核心架构设计

**学习时长：** 第 3-5 周

**知识点数：** 10 个

---

## 知识点列表

### 09. Gateway 架构设计
- Gateway 作为控制平面
- 路由机制
- 消息分发
- 通道抽象层
- 源码位置：`src/gateway/`

### 10. Agent 系统集成（Pi-agent-core）
- Pi-agent-core 集成方式
- Agent 运行时
- Agent 模式（RPC, Interactive）
- 源码位置：`src/agents/`

### 11. 配置系统详解
- 配置文件结构（~/.openclaw/config/）
- 配置优先级
- 环境变量
- 配置 API
- 源码位置：`src/config/`

### 12. 会话管理机制
- Session 存储（~/.openclaw/sessions/）
- JSONL 格式
- Session 分支
- Session 持久化
- 源码位置：`src/infra/`

### 13. 守护进程管理
- launchd（macOS）
- systemd（Linux）
- 守护进程配置
- 自动启动
- 源码位置：`src/daemon/`

### 14. 日志系统
- tslog 集成
- 日志级别
- 日志轮转
- 日志查询
- macOS 统一日志（scripts/clawlog.sh）

### 15. 错误处理机制
- 错误类型
- 错误传播
- 错误恢复
- 用户友好的错误消息

### 16. 性能监控
- 性能指标
- 监控工具
- 性能优化策略

### 17. 协议设计（Protocol）
- Gateway Protocol
- 协议生成（scripts/protocol-gen.ts）
- Swift 协议生成
- 协议版本管理

### 18. 依赖注入与模块化
- createDefaultDeps 模式
- 模块化设计
- 依赖管理

---

**验证标准：**
- ✅ 理解 Gateway 架构
- ✅ 阅读核心源码
- ✅ 理解配置系统
- ✅ 理解会话管理
