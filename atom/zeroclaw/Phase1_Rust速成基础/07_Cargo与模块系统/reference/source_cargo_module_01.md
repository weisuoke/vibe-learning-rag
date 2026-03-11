---
type: source_code_analysis
source: sourcecode/zeroclaw
analyzed_files:
  - sourcecode/zeroclaw/Cargo.toml
  - sourcecode/zeroclaw/src/lib.rs
  - sourcecode/zeroclaw/src/main.rs
  - sourcecode/zeroclaw/crates/robot-kit/Cargo.toml
  - sourcecode/zeroclaw/crates/robot-kit/src/lib.rs
analyzed_at: 2026-03-10
knowledge_point: 07_Cargo与模块系统
---

# 源码分析：ZeroClaw 的 Cargo 与模块系统

## 分析的文件

- `sourcecode/zeroclaw/Cargo.toml` - 顶层 workspace 配置，包含主 crate 和 robot-kit 子 crate
- `sourcecode/zeroclaw/src/lib.rs` - 主 crate 的模块声明，30+ 顶层模块
- `sourcecode/zeroclaw/src/main.rs` - 二进制入口点
- `sourcecode/zeroclaw/crates/robot-kit/Cargo.toml` - 子 crate 配置
- `sourcecode/zeroclaw/crates/robot-kit/src/lib.rs` - 子 crate 模块声明

## 关键发现

### 1. Workspace 配置

**顶层 Cargo.toml**:
- Workspace members: `["."]`（主 crate）和 `["crates/robot-kit"]`（子 crate）
- Resolver: `2`（workspace resolver）
- Package: `zeroclaw` v0.1.7, Rust 1.87+
- License: MIT OR Apache-2.0
- 同时包含 `[[bin]]` 和 `[lib]` 配置（二进制 + 库双模式）

### 2. Feature Flags（15+ 可选特性）

**默认特性**: 空（最小构建）

**硬件与外设**:
- `hardware` - USB 设备枚举 (nusb + tokio-serial)
- `peripheral-rpi` - 树莓派 GPIO 支持 (rppal)
- `probe` - probe-rs 用于 STM32 内存读取

**通道**:
- `channel-matrix` - Matrix 协议支持 (matrix-sdk)
- `channel-lark` - 飞书/钉钉协议 (prost)

**内存**:
- `memory-postgres` - PostgreSQL 后端

**可观测性**:
- `observability-otel` - OpenTelemetry 追踪 + 指标导出

**浏览器与自动化**:
- `browser-native` / `fantoccini` - Rust 原生浏览器自动化
- `sandbox-landlock` / `landlock` - Linux Landlock 沙箱

**RAG 与数据**:
- `rag-pdf` - PDF 数据提取

**集成**:
- `whatsapp-web` - 原生 WhatsApp Web 客户端

### 3. 模块结构（30+ 顶层模块）

```rust
// src/lib.rs 中的模块声明
pub mod agent              // 代理循环、分类器、调度器
pub mod approval           // 审批工作流
pub mod auth               // 认证服务
pub mod channels           // 多通道消息（Telegram、Discord、Slack 等）
pub mod config             // 配置管理
pub mod cost               // 成本追踪
pub mod cron               // 定时任务
pub mod daemon             // 服务管理
pub mod doctor             // 健康检查
pub mod gateway            // HTTP 网关/API 服务器
pub mod hardware           // 硬件发现与控制
pub mod health             // 健康监控
pub mod heartbeat          // 心跳/保活
pub mod hooks              // 生命周期钩子
pub mod identity           // 用户/身份管理
pub mod integrations       // 第三方集成
pub mod memory             // 内存后端（markdown、SQLite、PostgreSQL）
pub mod migration          // 数据迁移工具
pub mod multimodal         // 视觉/图像处理
pub mod observability      // 追踪、指标、日志
pub mod onboard            // 引导工作流
pub mod peripherals        // 硬件外设管理
pub mod providers          // LLM 提供商后端（OpenAI、Anthropic、Ollama 等）
pub mod rag                // RAG 管道
pub mod runtime            // 运行时适配器
pub mod security           // 安全策略与沙箱
pub mod service            // 服务管理
pub mod skills             // 用户定义的技能系统
pub mod tools              // 代理可调用工具（40+ 工具）
pub mod tunnel             // 隧道/代理支持
pub mod util               // 工具集
```

### 4. 模块组织模式

**Tools 模块** (`src/tools/mod.rs`):
- 40+ 工具实现
- 工具注册函数：`default_tools()`, `all_tools()`, `all_tools_with_runtime()`
- Trait 架构：`Tool` trait with `execute()` 方法
- Feature 门控：`#[cfg(feature = "hardware")]` 用于硬件工具

**Channels 模块** (`src/channels/mod.rs`):
- 20+ 通道实现
- Trait 架构：`Channel` trait
- 每个发送者的对话历史管理
- 指数退避重连

**Providers 模块** (`src/providers/mod.rs`):
- 13+ LLM 提供商后端
- 工厂模式：`create_provider()` 函数
- `ReliableProvider` 包装器用于降级和重试

### 5. Re-export 模式

**lib.rs** 使用选择性 re-export：
```rust
pub use config::Config;
pub use agent::{Agent, AgentBuilder};
pub use tools::Tool;
pub use channels::Channel;
pub use providers::Provider;
```

**模块级 re-export** (如 `tools/mod.rs`):
- 单个工具类型：`pub use shell::ShellTool;`
- Trait 类型：`pub use traits::Tool;`
- 工厂函数：`pub fn default_tools()`, `pub fn all_tools()`

### 6. 构建配置

**Release profile** (体积优化):
- `opt-level = "z"` (体积优化)
- `lto = "fat"` (跨 crate LTO)
- `codegen-units = 1` (串行代码生成，适合低内存设备)
- `strip = true` (移除调试符号)
- `panic = "abort"` (减小二进制体积)

**release-fast profile** (性能机器):
- `codegen-units = 8` (并行代码生成)
- 继承 release 的其他设置

### 7. Robot-Kit 子 crate

**位置**: `crates/robot-kit/`

**特性**:
- `safety` (默认) - 安全监控
- `ros2` - ROS2 集成
- `gpio` - 直接 GPIO 控制
- `lidar` - 激光雷达支持
- `vision` - 摄像头 + 视觉模型

**关键模块**:
- `config` - 机器人配置
- `traits` - Tool trait 定义
- `drive`, `look`, `listen`, `speak`, `sense`, `emote` - 机器人能力
- `safety` - 独立安全监控（feature 门控）

### 8. Firmware Crates（不在 workspace 中）

- `firmware/zeroclaw-esp32/` - ESP32 JSON-over-serial 外设
- `firmware/zeroclaw-esp32-ui/` - ESP32 + Slint UI
- `firmware/zeroclaw-nucleo/` - STM32 Nucleo-F401RE 外设

这些 firmware crate 有各自的 `build.rs` 用于嵌入式编译配置。
