# 核心概念 5：tracing 日志系统

> **知识点**: 常用库速查（serde/anyhow/clap）
> **层级**: Phase1_Rust速成基础
> **维度**: 核心概念（tracing 日志系统 — 结构化诊断、日志级别、Span、EnvFilter）
> **目标受众**: 有 TypeScript/前端经验但零 Rust 经验的开发者

---

## 一句话定义

> **tracing 是 Rust 的结构化诊断系统——`console.log` + `console.warn` + `console.error` + Performance API + pino 结构化日志 的编译时增强版，支持按模块过滤、零开销、自动附加上下文。**

---

## 1. tracing 是什么

### 1.1 前端类比

```typescript
// TypeScript 世界
console.log("Server started on port", 3000);    // 无级别、无过滤
console.warn("Config missing, using defaults");  // 浏览器黄色
console.error("Failed to connect:", err);        // 浏览器红色
// 生产环境？额外装 winston/pino
```

```rust
// Rust 世界：tracing 一站式搞定
use tracing::{info, warn, error, debug, trace};
info!("Server started on port {}", port);        // 结构化、可过滤
warn!("Config missing, using defaults");
error!("Failed to connect: {}", err);
debug!("Request payload: {:?}", payload);         // 开发时才看
trace!("Entering function load_config");          // 最细粒度
```

### 1.2 tracing vs log crate

```
特性              log crate            tracing
──────────────────────────────────────────────────────
基本日志           ✅                   ✅
结构化字段         ❌                   ✅ key=value
Span 时间段追踪    ❌                   ✅ 独有杀手锏
异步支持           一般                 原生支持
社区趋势           维护模式             主流首选
```

**结论**：新项目一律用 tracing。ZeroClaw 用的就是 tracing [^source1]。

### 1.3 两个核心原语

tracing 不只是"日志库"，而是**结构化诊断系统**，围绕两个原语：

- **Event**（事件）：某个时刻发生了什么 → 类似 `console.log`
- **Span**（时间段）：一段操作从开始到结束 → 类似 `performance.measure`

```
传统日志：  [INFO] Processing request     ← 只知道"发生了"
tracing：   [INFO] request{id=abc123}:     ← 知道"在哪个上下文中发生的"
              Processing request
```

---

## 2. 基本使用 — 五个日志宏

```rust
use tracing::{trace, debug, info, warn, error};

trace!("最细粒度，跟踪执行流程");       // 几乎不在生产用
debug!("开发调试信息");                 // 开发时打开
info!("重要业务事件");                  // 生产环境默认级别
warn!("可恢复的问题，需要注意");         // 黄色警告
error!("严重错误，需要立即处理");        // 红色错误
```

**TypeScript 对照**：

```
Rust tracing         TypeScript              何时用
──────────────────────────────────────────────────────────
trace!()             —（无对应）             跟踪每一步执行
debug!()             console.debug()         开发调试
info!()              console.log()           业务事件
warn!()              console.warn()          可恢复问题
error!()             console.error()         严重错误
```

### 格式化语法

```rust
let port = 8080;
let user = "admin";

// 基本插值（和 println! 一样）
info!("Server started on port {}", port);

// 命名参数（结构化字段）
info!(port = port, "Server started");
// 输出: ... INFO port=8080 Server started

// Debug 格式
let items = vec!["a", "b", "c"];
debug!("Items: {:?}", items);

// 多参数
info!(user = user, action = "login", "User event");
// 输出: ... INFO user=admin action=login User event
```

**关键区别**：被过滤掉的日志级别，**格式化代码根本不会执行**——零开销！TypeScript 的 `console.log` 即使你不看控制台，字符串拼接照样执行。

---

## 3. tracing-subscriber 配置

tracing 只定义了日志**接口**，`tracing-subscriber` 负责**输出**。类比：tracing = `emit`，subscriber = `listener`。

### 3.1 Cargo.toml 配置

```toml
[dependencies]
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

### 3.2 初始化

```rust
// 最简配置：一行搞定
tracing_subscriber::fmt::init();

// ZeroClaw 风格：环境变量控制
use tracing_subscriber::EnvFilter;
tracing_subscriber::fmt()
    .with_env_filter(EnvFilter::from_default_env()) // 读取 RUST_LOG
    .with_target(false)                              // 不显示模块路径
    .init();
```

### 3.3 RUST_LOG 环境变量控制

```bash
# 全局设置级别
RUST_LOG=debug cargo run                # 显示 debug 及以上
RUST_LOG=warn cargo run                 # 只显示 warn 和 error

# 按模块过滤（杀手锏！）
RUST_LOG=zeroclaw=debug cargo run       # zeroclaw 模块开 debug

# 多模块分别设置
RUST_LOG=zeroclaw=debug,reqwest=warn,hyper=error cargo run

# 子模块精细控制
RUST_LOG=info,zeroclaw::tools=debug cargo run
```

**TypeScript 对照**：

```typescript
DEBUG=myapp:* node server.js           // 类似 RUST_LOG=myapp=debug
// 但 Node.js 的 DEBUG 过滤远不如 RUST_LOG 灵活
```

---

## 4. 结构化日志 — tracing 的杀手锏

传统日志是纯文本拼接，结构化日志是带字段的记录，可以被 ELK/Loki/Datadog 索引：

```
传统: [INFO] User admin logged in from 192.168.1.1 after 3 attempts
结构: [INFO] user=admin ip=192.168.1.1 attempts=3 User logged in
```

### tracing 的结构化语法

```rust
// 字段在消息前
info!(user = "admin", action = "login", "User logged in");
// 输出: ... INFO user=admin action=login User logged in

// Display 格式（%）—— 用类型的 Display trait
let path = std::path::Path::new("/tmp/data.pdf");
info!(path = %path.display(), "Reading file");

// Debug 格式（?）—— 用类型的 Debug trait
let config = vec!["a", "b"];
debug!(config = ?config, "Loaded config");
// 输出: ... DEBUG config=["a", "b"] Loaded config
```

**前端类比**：像 pino 的 `logger.info({ user: "admin" }, "User logged in")`，但编译时检查。

---

## 5. Span — 时间段追踪（tracing 独有）

Event 记录"某个时刻"，Span 记录"一段时间"：

```
├── [SPAN] process_request (request_id=abc)
│     ├── [EVENT] Parsing input...
│     ├── [SPAN] database_query
│     │     └── [EVENT] Query executed (50ms)
│     └── [EVENT] Sending response
└── [SPAN END] process_request (120ms)
```

**前端类比**：`performance.mark()` + `performance.measure()`，但自动嵌套、自动上下文传播。

### 5.1 同步 Span

```rust
use tracing::{info, info_span};

fn process_order(order_id: u64) {
    let span = info_span!("process_order", order_id = order_id);
    let _guard = span.enter();  // _guard 析构时自动退出 span

    info!("Validating order");
    // 输出: ... INFO process_order{order_id=123}: Validating order
    //            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 自动附加上下文！
}
```

### 5.2 异步 Span（.instrument）

```rust
use tracing::{info, info_span, Instrument};

async fn handle_request(request_id: String) {
    let span = info_span!("handle_request", request_id = %request_id);
    async {
        info!("Processing...");  // 自动带上 request_id
        do_work().await;
    }
    .instrument(span)  // ← 把 span 绑定到 async 块
    .await;
}
```

为什么异步需要 `.instrument()`？因为 async 函数在 `.await` 时暂停，`_guard` 的 RAII 机制跨 `.await` 不可靠。

### 5.3 #[instrument] 属性宏（最推荐）

```rust
use tracing::instrument;

// 自动创建 span，参数自动成为字段
#[instrument]
fn process_payment(user_id: u64, amount: f64) {
    info!("Processing payment");
    // 输出: ... INFO process_payment{user_id=123 amount=99.9}: Processing
}

// 跳过敏感参数
#[instrument(skip(password))]
fn login(username: &str, password: &str) {
    info!("Login attempt");
    // password 不会出现在日志中
}

// 异步也行
#[instrument]
async fn fetch_data(url: &str) -> Result<String, anyhow::Error> {
    info!("Fetching...");
    Ok("data".to_string())
}
```

---

## 6. 在 ZeroClaw 中的使用

ZeroClaw 的 tracing 使用模式是学习的最佳范例 [^source1]：

### 初始化（main.rs）

```rust
tracing_subscriber::fmt()
    .with_env_filter(EnvFilter::from_default_env())
    .with_target(false)
    .init();
```

### 各模块的日志级别选择

```rust
// tools/ — debug! 记录操作详情
tracing::debug!("Reading PDF: {}", resolved_path.display());

// memory/ — warn! 记录降级行为
tracing::warn!("web_fetch: timeout_secs is 0, using safe default of 30s");

// providers/ — info! 记录 API 调用
tracing::info!("Searching web for: {}", query);
```

### 运行时控制

```bash
RUST_LOG=debug cargo run                          # 开发：全量日志
RUST_LOG=zeroclaw=debug,reqwest=warn cargo run    # 只看自己的 debug
RUST_LOG=info ./zeroclaw                          # 生产：只看 info
```

---

## 7. 速查表 — tracing vs console.log 对照

```
TypeScript                         Rust tracing
──────────────────────────────────────────────────────────────
console.debug("msg")               debug!("msg")
console.log("msg")                 info!("msg")
console.warn("msg")                warn!("msg")
console.error("msg")               error!("msg")
console.log("x=", x)               info!(x = x, "msg")
console.log(JSON.stringify(obj))   debug!(obj = ?obj, "msg")
DEBUG=app:* node .                 RUST_LOG=app=debug cargo run
performance.measure("op")          info_span!("op")
logger.info({k: v}, "msg")        info!(k = v, "msg")

关键差异：
  TS:   console.log 始终执行字符串拼接
  Rust: 被过滤的日志连格式化都不执行（零开销）

  TS:   无原生 Span 概念
  Rust: Span 是一等公民，支持嵌套和上下文传播
```

---

## 8. 最佳实践

### 日志级别选择指南

```
级别      用途                    示例                       生产可见？
──────────────────────────────────────────────────────────────────────
error!    需要立即处理的错误       数据库连接失败               ✅
warn!     可恢复但需注意           超时后使用默认值             ✅
info!     重要业务事件             API 调用、服务启动           ✅ (默认)
debug!    开发调试信息             请求详情、中间状态           ❌
trace!    极细粒度追踪             函数进入/退出               ❌
```

### 实用规则

```rust
// ✅ 结构化字段，方便查询
info!(user_id = 123, action = "purchase", "Order placed");

// ❌ 纯文本拼接，无法被索引
info!("User 123 purchased for 99.9");

// ✅ 用 debug! 而非 info! 记录大量调试信息
debug!(payload = ?request_body, "Incoming request");

// ✅ 用 #[instrument] 自动追踪，跳过敏感字段
#[instrument(skip(api_key))]
fn authenticate(user: &str, api_key: &str) { ... }
```

### 环境配置清单

```bash
RUST_LOG=debug                        # 开发环境
RUST_LOG=info,zeroclaw=debug          # 测试环境
RUST_LOG=info                         # 生产环境
```

---

## 常见陷阱

### 陷阱 1：忘记初始化 subscriber

```rust
fn main() {
    tracing::info!("This won't print!");  // ❌ 没有 subscriber → 静默消失！
    tracing_subscriber::fmt::init();      // ✅ 必须先初始化
    tracing::info!("Now this works!");
}
```

### 陷阱 2：异步中用 span.enter()

```rust
// ❌ _guard 跨 .await 行为不正确
async fn bad() {
    let _guard = info_span!("op").enter();
    do_work().await;  // 出问题！
}

// ✅ 用 .instrument() 或 #[instrument]
#[instrument]
async fn good() { do_work().await; }
```

### 陷阱 3：初始化两次 subscriber

```rust
tracing_subscriber::fmt::init();
tracing_subscriber::fmt::init();   // ❌ 第二次 panic!

let _ = tracing_subscriber::fmt::try_init(); // ✅ 用 try_init 避免
```

---

## 速查卡

```
Cargo.toml:
  tracing = "0.1"
  tracing-subscriber = { version = "0.3", features = ["env-filter"] }

初始化:
  tracing_subscriber::fmt::init()                    # 最简
  tracing_subscriber::fmt()                          # 带配置
      .with_env_filter(EnvFilter::from_default_env())
      .init()

五个宏:
  trace!  debug!  info!  warn!  error!

结构化字段:
  info!(key = value, "msg")             # 基本
  info!(key = %val, "msg")              # Display
  info!(key = ?val, "msg")              # Debug

Span:
  let _g = info_span!("name").enter();  # 同步
  async {}.instrument(span).await;      # 异步
  #[instrument]                         # 最简单

环境变量:
  RUST_LOG=debug cargo run              # 全局
  RUST_LOG=app=debug,lib=warn cargo run # 按模块
```

---

> **下一篇**: 阅读 `04_最小可用.md`，用最少代码跑通 serde + anyhow + clap + tracing 的基本用法。

---

**参考来源**

[^source1]: ZeroClaw 源码分析 — `reference/source_常用库_01.md`

---

**文件信息**
- 知识点: 常用库速查（serde/anyhow/clap）
- 维度: 03_核心概念_5_tracing日志系统
- 版本: v1.0
- 日期: 2026-03-11
