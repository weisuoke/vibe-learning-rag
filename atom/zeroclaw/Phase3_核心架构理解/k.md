# Phase 3: 核心架构理解 - 知识点列表

> 目标：理解 ZeroClaw 的顶层设计哲学和 Agent 核心循环
> 学习时长：第 5-7 周
> 前置要求：Phase 1-2 完成

---

## 知识点列表

### 17. 源码目录结构总览
- src/ 下 30+ 模块分层、crates/、web/、python/、firmware/ 的定位
- 前端类比：monorepo 目录结构
- ZeroClaw 场景：快速定位功能代码所在模块

### 18. Trait 驱动架构设计
- Provider/Tool/Memory/Channel 四大核心 Trait、六边形架构、可插拔设计
- 前端类比：依赖注入 + 接口解耦（如 Repository Pattern）
- ZeroClaw 场景：理解为什么 ZeroClaw 能用 config.toml 切换任何组件

### 19. AgentBuilder 与构建器模式
- Builder pattern 在 Rust 中的实现、链式调用、依赖注入
- 前端类比：React Context Provider 配置
- ZeroClaw 场景：agent.rs 中如何组装 Provider + Tool + Memory

### 20. Agent 核心循环（ReAct）
- 系统提示 → LLM 调用 → 工具解析 → 执行 → 更新历史 → 迭代
- 前端类比：Redux dispatch → reducer → state update 循环
- ZeroClaw 场景：agent.rs 中的 run/chat 方法核心逻辑

### 21. 消息模型与数据流
- Message struct、Role enum（User/Assistant/System/Tool）、序列化
- 前端类比：Redux action/state 数据流
- ZeroClaw 场景：消息如何在 Channel → Agent → Provider 间流转

### 22. 配置系统详解（config.toml）
- 配置 Schema、TOML 解析、默认值、热重载、配置层级
- 前端类比：next.config.js / vite.config.ts
- ZeroClaw 场景：config/ 模块的完整解析流程

### 23. 模块注册与工厂模式
- Provider/Channel/Tool 的注册机制、动态实例化、配置驱动
- 前端类比：Plugin 注册表（如 Vite plugin 系统）
- ZeroClaw 场景：如何根据 config.toml 中的名称创建对应实例

### 24. 错误传播与 anyhow
- anyhow::Result、context()、bail!、错误链、与 thiserror 的区别
- 前端类比：Error Boundary 链式错误处理
- ZeroClaw 场景：源码中的错误处理模式和最佳实践

### 25. 日志与可观测性
- tracing crate、span/event、Observer trait、Prometheus/OTel 集成
- 前端类比：console.log + 性能监控（如 Datadog）
- ZeroClaw 场景：observability/ 模块的设计和使用

### 26. 入口分析：main.rs 解读
- clap CLI 解析 → 配置加载 → 子命令路由 → Agent/Daemon/Gateway 启动
- 前端类比：index.ts 入口文件 → 路由初始化 → 应用启动
- ZeroClaw 场景：从 main.rs 的 1977 行代码梳理完整启动流程
