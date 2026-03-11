# Cargo 与模块系统 - 实战场景 3：Feature Flags 实战

> **知识点**: Cargo 与模块系统
> **层级**: Phase1_Rust速成基础
> **维度**: 实战代码（场景 3：为库 crate 实现可选功能——Feature Flags 条件编译）
> **目标受众**: 有 TypeScript/前端经验但零 Rust 经验的开发者
> **阅读时间**: 约 20 分钟
> **前提**: 已阅读 03_核心概念_5_Feature_Flags、07_实战代码_场景2

---

## 场景目标

**构建一个 `mini-provider` 库 crate**，用 Feature Flags 实现可选功能：

1. **基础 Provider trait**——始终可用，无 feature 依赖
2. **JSON 序列化**（`json` feature）——可选的 serde 支持
3. **Async 异步支持**（`async` feature）——可选的 tokio + async-trait
4. **日志记录**（`logging` feature）——可选的 tracing 日志
5. **全功能模式**（`full` feature）——一键启用全部

```
mini-provider/
├── Cargo.toml              ← feature flags + 可选依赖定义
└── src/
    ├── lib.rs              ← 模块声明 + 条件编译门控
    ├── provider.rs         ← 基础 Provider trait（始终编译）
    ├── message.rs          ← Message 结构体（条件 derive）
    ├── serialization.rs    ← JSON 序列化（json feature 门控）
    ├── async_provider.rs   ← 异步 Provider（async feature 门控）
    └── logger.rs           ← 日志工具（logging feature 门控）
```

---

## Step 1：Cargo.toml——Feature Flags 总开关

```toml
[package]
name = "mini-provider"
version = "0.1.0"
edition = "2024"

[features]
default = ["json"]                          # 大多数人需要 JSON
json = ["dep:serde", "dep:serde_json"]      # dep: 前缀避免名称冲突
async = ["dep:tokio", "dep:async-trait"]    # 异步支持
logging = ["dep:tracing"]                   # 结构化日志
full = ["json", "async", "logging"]         # 一键全部——方便 CI

[dependencies]
anyhow = "1.0"                                                          # 始终编译
serde = { version = "1.0", features = ["derive"], optional = true }     # 可选
serde_json = { version = "1.0", optional = true }                       # 可选
tokio = { version = "1", features = ["rt-multi-thread", "macros"], optional = true }
async-trait = { version = "0.1", optional = true }
tracing = { version = "0.1", optional = true }

[dev-dependencies]
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

**关键**：`optional = true` 的依赖只在对应 feature 启用时编译。未启用 → 不下载、不编译、不链接——比 npm 的 `optionalDependencies` 更彻底 [^1]。

---

## Step 2：基础 Provider trait（无 feature 依赖）

**src/provider.rs**——核心抽象，任何 feature 组合下都编译：

```rust
use anyhow::Result;

/// ≈ TypeScript: interface Provider { name(): string; chat(msg: string): string }
pub trait Provider {
    fn name(&self) -> &str;
    fn chat(&self, message: &str) -> Result<String>;
}

/// 简单的回声 Provider——用于测试
pub struct EchoProvider {
    pub model: String,
}

impl EchoProvider {
    pub fn new(model: &str) -> Self {
        EchoProvider { model: model.to_string() }
    }
}

impl Provider for EchoProvider {
    fn name(&self) -> &str { "echo" }

    fn chat(&self, message: &str) -> Result<String> {
        Ok(format!("[echo@{}] {}", self.model, message))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_echo_provider() {
        let p = EchoProvider::new("gpt-4");
        assert_eq!(p.name(), "echo");
        assert!(p.chat("hello").unwrap().contains("hello"));
    }
}
```

**注意**：这个文件没有任何 `#[cfg(feature)]`——它是核心模块，始终存在。

---

## Step 3：条件编译——JSON 序列化支持

**src/message.rs**——`#[cfg_attr]` 有条件地添加 derive：

```rust
/// #[cfg_attr(condition, attr)]：condition 为真时添加 attr，为假时忽略
#[derive(Debug, Clone)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
pub struct Message {
    pub role: String,
    pub content: String,
    pub timestamp: u64,
}

impl Message {
    pub fn user(content: &str) -> Self {
        Message { role: "user".into(), content: content.into(), timestamp: 0 }
    }
    pub fn assistant(content: &str) -> Self {
        Message { role: "assistant".into(), content: content.into(), timestamp: 0 }
    }
}
```

**src/serialization.rs**——整个文件被 lib.rs 中的 `#[cfg(feature = "json")]` 门控，文件内不需要再写 `#[cfg]`：

```rust
use anyhow::{Context, Result};
use crate::message::Message;

pub fn to_json(msg: &Message) -> Result<String> {
    serde_json::to_string_pretty(msg)
        .with_context(|| "Failed to serialize message")
}

pub fn from_json(json_str: &str) -> Result<Message> {
    serde_json::from_str(json_str)
        .with_context(|| format!("Failed to parse JSON: {}", json_str))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let msg = Message::user("hello JSON");
        let json = to_json(&msg).unwrap();
        let restored = from_json(&json).unwrap();
        assert_eq!(restored.content, "hello JSON");
    }
}
```

---

## Step 4：条件编译——Async 异步支持

**src/async_provider.rs**——整个模块被 `async` feature 门控：

```rust
//! 异步 Provider——仅在 async feature 启用时可用。
//! ≈ TypeScript: interface AsyncProvider { chat(msg: string): Promise<string> }

use anyhow::Result;
use async_trait::async_trait;

#[async_trait]
pub trait AsyncProvider: Send + Sync {
    fn name(&self) -> &str;
    async fn chat(&self, message: &str) -> Result<String>;
}

pub struct AsyncMockProvider {
    model: String,
    delay_ms: u64,
}

impl AsyncMockProvider {
    pub fn new(model: &str, delay_ms: u64) -> Self {
        AsyncMockProvider { model: model.to_string(), delay_ms }
    }
}

#[async_trait]
impl AsyncProvider for AsyncMockProvider {
    fn name(&self) -> &str { "async-mock" }

    async fn chat(&self, message: &str) -> Result<String> {
        tokio::time::sleep(tokio::time::Duration::from_millis(self.delay_ms)).await;
        Ok(format!("[async-mock@{}] {}", self.model, message))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_async_provider() {
        let p = AsyncMockProvider::new("gpt-4", 10);
        assert!(p.chat("async hello").await.unwrap().contains("async hello"));
    }
}
```

---

## Step 5：条件编译——日志支持

**src/logger.rs**——整个模块被 `logging` feature 门控：

```rust
use tracing::{info, warn};

pub fn log_chat(provider_name: &str, message: &str) {
    info!(provider = provider_name, len = message.len(), "Provider chat called");
}

pub fn log_warning(context: &str, detail: &str) {
    warn!(context = context, "Warning: {}", detail);
}
```

---

## Step 6：lib.rs——条件编译整个模块

**src/lib.rs**——模块声明的「总控室」：

```rust
//! # mini-provider
//!
//! | Feature   | Description            | Dependencies       |
//! |-----------|------------------------|--------------------|
//! | `json`    | JSON 序列化（默认启用） | serde, serde_json  |
//! | `async`   | 异步 Provider          | tokio, async-trait |
//! | `logging` | 结构化日志             | tracing            |
//! | `full`    | 全部功能               | 以上全部           |

// ── 始终编译 ──
pub mod provider;
pub mod message;

// ── 条件编译：feature 未启用时整个 .rs 文件「不存在」 ──
#[cfg(feature = "json")]
pub mod serialization;
#[cfg(feature = "async")]
pub mod async_provider;
#[cfg(feature = "logging")]
pub mod logger;

// ── Re-export ──
pub use provider::{Provider, EchoProvider};
pub use message::Message;
#[cfg(feature = "json")]
pub use serialization::{to_json, from_json};
#[cfg(feature = "async")]
pub use async_provider::{AsyncProvider, AsyncMockProvider};

/// 处理消息——演示「函数内部」条件编译
/// 函数本身始终存在，但内部行为根据 feature 变化。
pub fn process_message(provider: &dyn Provider, content: &str) -> String {
    #[cfg(feature = "logging")]
    logger::log_chat(provider.name(), content);

    let response = provider.chat(content).unwrap_or_else(|e| {
        #[cfg(feature = "logging")]
        logger::log_warning("process_message", &e.to_string());
        format!("Error: {}", e)
    });

    #[cfg(feature = "json")]
    {
        let msg = Message::assistant(&response);
        if let Ok(json) = to_json(&msg) {
            return json;
        }
    }

    response
}

/// 报告启用的 feature
pub fn enabled_features() -> Vec<&'static str> {
    let mut f = vec!["core"];
    #[cfg(feature = "json")]
    f.push("json");
    #[cfg(feature = "async")]
    f.push("async");
    #[cfg(feature = "logging")]
    f.push("logging");
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_message() {
        let p = EchoProvider::new("test");
        let result = process_message(&p, "hello");
        assert!(!result.is_empty());
    }

    #[test]
    fn test_enabled_features() {
        assert!(enabled_features().contains(&"core"));
    }
}
```

---

## Step 7：`cfg!` 宏 vs `#[cfg()]` 属性

两种条件编译方式——初学者最容易混淆：

```rust
// ── #[cfg()]：代码级排除——未启用时函数不存在 ──
#[cfg(feature = "json")]
fn json_only() { /* 可以引用 serde 类型 */ }

// ── cfg!()：值级判断——代码始终编译，值在编译时确定 ──
fn always_exists() {
    if cfg!(feature = "json") {
        println!("JSON enabled");
        // ⚠️ 不能调用 serde 函数——它可能没被编译！
    }
}
```

```
#[cfg(feature = "x")]                 cfg!(feature = "x")
──────────────────────────────────────────────────────────────
代码在编译时被排除               代码始终编译，值在编译时确定
可用于模块/函数/结构体           只能用在 if 表达式中
可以引用可选依赖的类型           不能引用可能不存在的类型
首选方式 ✅                     仅用于「两个分支都能编译」的场景
```

**正确用法——两个分支都不依赖可选依赖**：

```rust
pub fn version_info() -> String {
    let mut info = format!("mini-provider v{}", env!("CARGO_PKG_VERSION"));
    if cfg!(feature = "json") { info.push_str(" +json"); }
    if cfg!(feature = "async") { info.push_str(" +async"); }
    if cfg!(feature = "logging") { info.push_str(" +logging"); }
    info
}
```

---

## Step 8：构建和测试不同 feature 组合

```bash
cargo build                              # default（json）→ 编译 serde
cargo build --no-default-features        # 最小 → 只编译 anyhow
cargo build --features "async"           # json + async → serde + tokio
cargo build --no-default-features --features "async"  # 只要 async
cargo build --all-features               # 编译所有可选依赖

cargo test                               # 测试 default features
cargo test --all-features                # 测试全部（CI 推荐）

cargo check --no-default-features        # 最小能编译？
cargo check --all-features               # 全功能能编译？

# cargo install cargo-hack
cargo hack check --each-feature          # 逐个 feature 检查 [^3]
```

---

## Step 9：ZeroClaw 中的 Feature Flags 实例

### 9.1 Feature 定义对比

```
mini-provider（本教程）            ZeroClaw（真实项目）
──────────────────────────────────────────────────────────────────
default = ["json"]                default = []（最小构建——嵌入式友好）
json = ["dep:serde", ...]         serde 是非可选依赖（始终需要）
async = ["dep:tokio", ...]        tokio 是非可选依赖（始终需要）
logging = ["dep:tracing"]         observability-otel = ["dep:opentelemetry"]
full = [全部]                     无 full——按场景组合 [^2]
```

### 9.2 模块门控与函数内条件编译

```rust
// ═══ ZeroClaw src/lib.rs ═══
pub mod agent;       // 始终编译
pub mod config;

#[cfg(feature = "hardware")]
pub mod hardware;    // USB 枚举——需要 nusb + tokio-serial

// ═══ ZeroClaw src/tools/mod.rs ═══
pub fn default_tools() -> Vec<Box<dyn Tool>> {
    let mut tools = vec![Box::new(ShellTool::new())];

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

### 9.3 default 策略对比

```toml
# 主 crate：灵活性优先——默认不启用任何可选功能
[features]
default = []

# Robot-Kit 子 crate：安全优先——safety 默认启用
[features]
default = ["safety"]    # 忘记启用 → 机器人可能伤人
```

**设计原则**：`default` 放什么取决于「忘记启用的后果」[^2]。

---

## Step 10：预期输出

```bash
$ cargo build --no-default-features       # 只编译 anyhow
$ cargo build                             # + serde, serde_json
$ cargo build --all-features              # + tokio, async-trait, tracing

$ cargo test --all-features
test provider::tests::test_echo_provider ... ok
test serialization::tests::test_roundtrip ... ok
test async_provider::tests::test_async_provider ... ok
test tests::test_process_message ... ok
test result: ok. 7 passed; 0 failed
```

**`enabled_features()` 在不同构建下**：

```
--no-default-features  → ["core"]
default                → ["core", "json"]
--features "async"     → ["core", "json", "async"]
--all-features         → ["core", "json", "async", "logging"]
```

---

## 知识点速览

| 知识点 | 项目中的体现 |
|--------|-------------|
| `[features]` 表 | 定义 json、async、logging、full 四个 feature |
| `default = [...]` | 默认启用 json |
| `dep:` 前缀 | `json = ["dep:serde", "dep:serde_json"]` |
| `optional = true` | serde、tokio、tracing 都是可选依赖 |
| `#[cfg(feature)]` | 门控整个模块和代码块 |
| `#[cfg_attr]` | Message 的 Serialize/Deserialize 条件 derive |
| `cfg!()` 宏 | version_info() 中——代码始终编译 |
| `--no-default-features` | 最小构建测试 |
| `--all-features` | 全功能测试（CI 推荐） |

---

> **下一步**: 阅读 `08_面试必问.md`，了解 Cargo 与模块系统的高频面试题。

---

**引用来源**

[^1]: Cargo 官方文档：Feature Flags 定义语法、`dep:` 前缀、可选依赖机制（`reference/context7_cargo_01.md`）
[^2]: ZeroClaw 源码分析：15+ Feature Flags、`default = []`、模块门控、Robot-Kit 的 `default = ["safety"]`（`reference/source_cargo_module_01.md`）
[^3]: Rust Cargo 社区最佳实践：Feature 可加性原则、`cargo-hack` 测试（`reference/search_cargo_module_01.md`）

---

**文件信息**
- 知识点: Cargo 与模块系统
- 维度: 07_实战代码_场景3_Feature_Flags实战
- 版本: v1.0
- 日期: 2026-03-10
