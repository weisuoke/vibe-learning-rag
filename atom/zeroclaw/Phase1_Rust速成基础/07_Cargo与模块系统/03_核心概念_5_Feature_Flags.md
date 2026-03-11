# Cargo 与模块系统 - 核心概念 5：Feature Flags

> **知识点**: Cargo 与模块系统
> **层级**: Phase1_Rust速成基础
> **维度**: 核心概念（Feature Flags 条件编译、可选依赖、可加性原则与 ZeroClaw 实战）
> **目标受众**: 有 TypeScript/前端经验但零 Rust 经验的开发者

---

## 一句话定义

> **Feature Flags 是 Cargo 的条件编译机制，在编译时选择启用哪些功能——像 webpack 的 DefinePlugin 但在编译阶段完成，未启用的代码根本不会出现在二进制文件中。**

---

## 一、什么是 Feature Flags？

### 1.1 编译时条件开关

Feature Flags 让你的 crate 拥有**可选的功能模块**。使用者在 `Cargo.toml` 或命令行中声明想要启用哪些 feature，Cargo 在编译时把不需要的代码**完全剔除** [^context7_cargo1]。

**核心区别——编译时 vs 运行时**：

```
TypeScript（运行时）                          Rust Feature Flags（编译时）
────────────────────────────────────────────────────────────────────────────
if (process.env.NODE_ENV === 'production')    #[cfg(feature = "logging")]
→ 代码打包进 bundle，运行时判断               → 代码根本不编译进二进制
未使用的代码仍在 bundle 中（需 tree-shake）    完全不存在——零开销
可以在部署后动态切换                           必须重新编译才能切换
```

### 1.2 TypeScript 类比

Feature Flags 最接近 webpack `DefinePlugin` + tree-shaking 的组合 [^search1]：

```javascript
// webpack.config.js —— TypeScript 的"Feature Flags"
new webpack.DefinePlugin({
  'process.env.ENABLE_ANALYTICS': JSON.stringify(false),
});
// 代码中——需要 terser 等工具才能真正删除 dead code
if (process.env.ENABLE_ANALYTICS) {
  import('./analytics').then(m => m.init());
}
```

```rust
// Rust 的方式——编译阶段完成，没有任何运行时开销
#[cfg(feature = "analytics")]
mod analytics;

#[cfg(feature = "analytics")]
pub fn init_analytics() {
    analytics::init();
}
// 如果 "analytics" feature 未启用，这些代码就像从未存在过
```

### 1.3 为什么 Rust 选择编译时而不是运行时？

**零开销抽象**（Rust 核心哲学）、**嵌入式友好**（ZeroClaw 跑在树莓派上，每字节珍贵）、**依赖裁剪**（未启用的 feature 对应的依赖不会编译和链接）、**编译时安全**（未启用的代码不会引入错误或隐患）。

---

## 二、Feature 定义语法

### 2.1 [features] 表

Feature 在 `Cargo.toml` 的 `[features]` 表中定义 [^context7_cargo1]：

```toml
[package]
name = "my-library"
version = "0.1.0"

[features]
# default：使用者没指定 feature 时自动启用
default = ["std", "logging"]

# 自定义 feature——空数组 = 纯开关
std = []
logging = []
json = []

# feature 可以启用其他 feature（传递启用）
full = ["logging", "json", "analytics"]

# feature 可以启用可选依赖
analytics = ["dep:reqwest", "json"]    # 启用可选依赖 reqwest + feature json

[dependencies]
reqwest = { version = "0.12", optional = true }   # optional = true 表示可选依赖
log = "0.4"   # 非可选——始终编译
```

### 2.2 可选依赖与 `dep:` 前缀

可选依赖会**自动创建同名 feature**——用 `dep:` 前缀可避免冲突 [^context7_cargo1]：

```toml
[dependencies]
serde = { version = "1.0", optional = true }

[features]
# ❌ 旧写法——serde 可选依赖自动变成名为 "serde" 的 feature（名称冲突风险）
# ✅ 新写法——用 dep: 前缀，自定义 feature 名
serialization = ["dep:serde"]    # feature 名和依赖名解耦
```

### 2.3 完整 Cargo.toml Feature 示例

```toml
[package]
name = "my-agent"
version = "0.1.0"
edition = "2024"

[features]
default = []                                  # 默认不启用任何可选功能
hardware = ["dep:nusb", "dep:tokio-serial"]  # 硬件支持 + 拉入两个可选依赖
web = ["dep:reqwest"]                         # HTTP 客户端
full = ["hardware", "web", "logging"]         # 全功能模式
logging = []                                  # 纯开关

[dependencies]
tokio = { version = "1.42", features = ["rt-multi-thread"] }  # 始终需要
serde = { version = "1.0", features = ["derive"] }            # 始终需要
nusb = { version = "0.1", optional = true }                   # 可选
tokio-serial = { version = "5.4", optional = true }           # 可选
reqwest = { version = "0.12", optional = true }               # 可选
```

---

## 三、在代码中使用 Feature

### 3.1 `#[cfg(feature = "name")]` 属性

最常用的方式——在函数、结构体、模块上方添加条件编译属性 [^search1]：

```rust
// 条件编译模块
#[cfg(feature = "hardware")]
pub mod hardware;          // 只在 hardware feature 启用时编译

// 条件编译函数
#[cfg(feature = "logging")]
pub fn init_logging() {
    env_logger::init();
}
// 没启用 logging？这个函数完全不存在——调用它会编译报错

// 条件编译 impl 块
pub struct Agent { name: String }

#[cfg(feature = "hardware")]
impl Agent {
    pub fn scan_usb_devices(&self) -> Vec<Device> {
        nusb::list_devices().collect()
    }
}

// 条件编译结构体字段
pub struct Config {
    pub name: String,
    #[cfg(feature = "metrics")]
    pub metrics_endpoint: String,   // 只在 metrics 启用时才有
}
```

### 3.2 `cfg!` 宏 vs `#[cfg()]` 属性

```rust
// cfg! 宏——编译时求值为 true/false，但不排除代码
if cfg!(feature = "logging") {
    println!("Processing data");   // 即使 logging 未启用，这行也会编译
}
```

```
#[cfg(feature = "x")]        cfg!(feature = "x")
──────────────────────────────────────────────────────────
代码在编译时被排除             代码始终编译，值在编译时确定
可用于模块/函数/结构体         只能用在表达式中（if 条件）
首选方式 ✅                   特殊场景使用
```

### 3.3 条件编译的组合

```rust
// AND——两个 feature 都启用时
#[cfg(all(feature = "hardware", feature = "logging"))]
fn log_hardware_event() { /* ... */ }

// OR——任一 feature 启用时
#[cfg(any(feature = "web", feature = "api"))]
fn init_http_client() { /* ... */ }

// NOT——feature 未启用时
#[cfg(not(feature = "std"))]
fn no_std_fallback() { /* ... */ }

// 组合——只在 Linux + hardware 时编译
#[cfg(all(target_os = "linux", feature = "hardware"))]
fn scan_gpio_pins() { /* 树莓派 GPIO——macOS 上不可能工作 */ }
```

---

## 四、Feature 的可加性（Additive）原则

### 4.1 为什么 Feature 必须是可加的？

这是 Feature Flags 最重要的约束 [^search1] [^context7_cargo1]：

> **Feature 只能增加功能，不能移除功能。启用 A 不能导致 B 的代码失效。**

**类比**：点外卖——加辣、加芝士、加大份互不干扰。"加辣"不能让"加芝士"消失。

### 4.2 Feature 统一（Unification）

Workspace 中多个 crate 依赖同一个库但启用不同 feature 时，Cargo **取并集** [^search1]：

```
crate-a 依赖 tokio，启用 ["macros"]
crate-b 依赖 tokio，启用 ["fs", "net"]

→ Cargo 编译时，tokio 启用 ["macros", "fs", "net"]（并集）
→ 整个 workspace 只编译一份 tokio
```

**这就是为什么 feature 必须可加**——如果 `no-std` feature 禁用标准库，而另一个 crate 需要标准库，统一后就会破坏。

### 4.3 非可加 Feature 的危险与解决

```rust
// ❌ 错误设计——互斥 feature，两个同时启用会编译错误
#[cfg(feature = "backend-sqlite")]
type Database = SqliteDb;
#[cfg(feature = "backend-postgres")]
type Database = PostgresDb;
```

```rust
// ✅ 方案 1：compile_error! 给出清晰错误
#[cfg(all(feature = "backend-sqlite", feature = "backend-postgres"))]
compile_error!("'backend-sqlite' and 'backend-postgres' are mutually exclusive.");

// ✅ 方案 2：trait object 运行时选择（两个 feature 可共存）
pub trait Database: Send + Sync {
    fn query(&self, sql: &str) -> Result<Vec<Row>>;
}
#[cfg(feature = "backend-sqlite")]
pub fn create_sqlite_db() -> Box<dyn Database> { /* ... */ }
#[cfg(feature = "backend-postgres")]
pub fn create_postgres_db() -> Box<dyn Database> { /* ... */ }
```

### 4.4 使用 `cargo tree -e features` 调试

```bash
# 查看某个依赖最终启用了哪些 feature
cargo tree -e features -i tokio

# 查看整个 workspace 的 feature 传播
cargo tree -e features --workspace

# 反向查找：某个 feature 被谁启用
cargo tree --invert tokio -e features
```

---

## 五、ZeroClaw 的 Feature Flags 实战

### 5.1 完整 Feature 列表

ZeroClaw 定义了 **15+ 可选特性**，默认全部关闭 [^source1]：

```toml
# zeroclaw/Cargo.toml
[features]
default = []   # ← 最小构建——不启用任何可选功能

# ── 硬件与外设 ──
hardware = ["dep:nusb", "dep:tokio-serial"]     # USB 设备枚举
peripheral-rpi = ["dep:rppal"]                   # 树莓派 GPIO
probe = ["dep:probe-rs"]                         # STM32 内存读取

# ── 通信通道 ──
channel-matrix = ["dep:matrix-sdk"]              # Matrix 协议
channel-lark = ["dep:prost"]                     # 飞书/钉钉

# ── 存储后端 ──
memory-postgres = ["dep:sqlx"]                   # PostgreSQL 后端

# ── 可观测性 ──
observability-otel = ["dep:opentelemetry"]       # OpenTelemetry 追踪+指标

# ── 浏览器与安全 ──
browser-native = ["dep:chromiumoxide"]           # Rust 原生浏览器自动化
sandbox-landlock = ["dep:landlock"]              # Linux Landlock 沙箱

# ── 数据处理 & 集成 ──
rag-pdf = ["dep:pdf-extract"]                    # PDF 数据提取
whatsapp-web = ["dep:whatsapp-rs"]               # WhatsApp Web 客户端
```

### 5.2 为什么默认 Feature 为空？

```
设计考量               解释
──────────────────────────────────────────────────────────────────────
嵌入式/低资源设备      ZeroClaw 跑在树莓派——默认只编译核心，省内存减体积
编译速度              matrix-sdk 等重量级依赖编译耗时长，不需要就不编
跨平台兼容            rppal 只在 Linux/RPi 可用，macOS 开发者不该被强制编译
按需组合              树莓派 → hardware + peripheral-rpi
                     服务器 → memory-postgres + observability-otel
                     本地开发 → 什么都不启用
```

### 5.3 Feature 门控代码示例

ZeroClaw 大量使用 `#[cfg(feature = "...")]` 门控模块和功能 [^source1]：

```rust
// src/lib.rs —— 条件编译整个模块
pub mod agent;       // 始终编译——核心模块
pub mod config;      // 始终编译
pub mod channels;    // 始终编译

#[cfg(feature = "hardware")]
pub mod hardware;    // 只在 hardware feature 启用时编译

// src/tools/mod.rs —— 条件注册工具
pub fn default_tools() -> Vec<Box<dyn Tool>> {
    let mut tools: Vec<Box<dyn Tool>> = vec![
        Box::new(ShellTool::new()),      // 始终可用
        Box::new(FileTool::new()),       // 始终可用
    ];

    #[cfg(feature = "hardware")]
    {
        tools.push(Box::new(UsbScanTool::new()));
        tools.push(Box::new(SerialTool::new()));
    }

    #[cfg(feature = "browser-native")]
    tools.push(Box::new(BrowserTool::new()));

    tools
}
```

### 5.4 Robot-Kit 的 Feature 设计对比

```toml
# crates/robot-kit/Cargo.toml
[features]
default = ["safety"]      # ← 注意：safety 默认启用——安全模块始终在线
safety = []               # 安全监控
ros2 = ["dep:r2r"]        # ROS2 集成
gpio = ["dep:rppal"]      # 直接 GPIO 控制
lidar = ["dep:rplidar"]   # 激光雷达
vision = ["dep:opencv"]   # 摄像头 + 视觉模型
```

**设计哲学对比**：主 crate `default = []`（最小构建），robot-kit `default = ["safety"]`（安全优先）。安全监控太重要，不该让用户忘记启用。

### 5.5 构建不同场景

```bash
cargo build                                              # 最小构建（开发调试）
cargo build --features "hardware,peripheral-rpi"         # 树莓派部署
cargo build --features "memory-postgres,observability-otel"  # 服务器部署
cargo build --all-features                               # CI 全量测试
```

---

## 六、Feature Flags 最佳实践

### 6.1 默认保持最小

```toml
# ✅ 好——默认不启用可选功能
default = []

# ⚠️ 谨慎——只把"几乎所有人都需要"的放进 default
default = ["std"]

# ❌ 不好——default 太重
default = ["hardware", "postgres", "opentelemetry", "browser"]
```

### 6.2 命名规范

```toml
# ✅ 好——分类前缀 + 具体名称，清晰可预测
channel-matrix = []
channel-lark = []
memory-postgres = []
observability-otel = []

# ❌ 不好——含糊、缩写、冗余前缀
matrix = []            # 不知道类别
pg = []                # 缩写不直观
enable-logs = []       # "enable-" 前缀多余——feature 本身就是启用
```

### 6.3 文档化每个 Feature

```rust
//! # Feature Flags
//!
//! | Feature              | Description                  | Dependencies       |
//! |----------------------|------------------------------|--------------------|
//! | `hardware`           | USB + serial device support  | nusb, tokio-serial |
//! | `memory-postgres`    | PostgreSQL storage backend   | sqlx               |
//! | `observability-otel` | OpenTelemetry tracing        | opentelemetry      |
```

### 6.4 使用 `cargo-hack` 测试组合

15 个 feature 有 2^15 = 32,768 种组合。`cargo-hack` 系统地测试 [^search1]：

```bash
cargo install cargo-hack

cargo hack check --each-feature                    # 逐个 feature 测试
cargo hack check --feature-powerset --depth 2      # 最多同时启用 2 个
cargo hack check --feature-powerset --skip "peripheral-rpi,probe"  # 跳过不兼容组合
```

### 6.5 配置 rust-analyzer

IDE 默认只分析 `default` feature 的代码 [^search1]：

```json
// .vscode/settings.json
{
  "rust-analyzer.cargo.features": ["hardware", "memory-postgres"]
  // 或启用全部："rust-analyzer.cargo.features": "all"
}
```

---

## 总结

```
Cargo Feature Flags 核心要点：

  是什么？
    编译时的功能开关——未启用的代码不编译、不链接、零开销
    ≈ webpack DefinePlugin + tree-shaking，但更彻底

  怎么定义？
    [features] 表               → 声明 feature 名称和关联
    optional = true             → 标记可选依赖
    dep: 前缀                   → 区分 feature 名和依赖名

  怎么用？
    #[cfg(feature = "x")]       → 条件编译（推荐，零开销）
    cfg!(feature = "x")         → 编译时布尔值（代码仍存在）
    all() / any() / not()       → 组合条件

  核心约束：
    ⚠️ Feature 必须是可加的      → 启用 A 不能破坏 B
    Feature 统一（取并集）       → workspace 中自动合并
    互斥需求 → compile_error!   → 给出清晰错误信息

  ZeroClaw 实践：
    default = []                → 最小构建（嵌入式友好）
    15+ 可选 feature            → 分类前缀：channel-*, memory-*, ...
    robot-kit: default=["safety"] → 安全模块默认启用

  最佳实践：
    默认保持最小                 → default = [] 或仅核心
    用 dep: 前缀管理可选依赖     → 避免名称冲突
    cargo-hack 测试组合          → cargo hack check --each-feature
    配置 rust-analyzer          → 完整 IDE 代码提示
    文档化每个 feature           → 注释 + lib.rs 表格
```

---

> **下一步**: 阅读 `03_核心概念_6_构建配置与Profile.md`，了解 Cargo 的构建优化配置——ZeroClaw 如何通过 `opt-level = "z"` 和 `lto = "fat"` 把二进制体积压缩到极致。

---

**参考来源**

[^source1]: ZeroClaw 源码分析 — `reference/source_cargo_module_01.md`
[^context7_cargo1]: Cargo 官方文档 — `reference/context7_cargo_01.md`
[^search1]: Rust Cargo 社区最佳实践 — `reference/search_cargo_module_01.md`

---

**文件信息**
- 知识点: Cargo 与模块系统
- 维度: 03_核心概念_5_Feature_Flags
- 版本: v1.0
- 日期: 2026-03-10
