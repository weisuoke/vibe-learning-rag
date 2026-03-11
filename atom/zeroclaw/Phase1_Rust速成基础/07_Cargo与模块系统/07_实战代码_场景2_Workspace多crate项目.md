# Cargo 与模块系统 - 实战场景 2：Workspace 多 crate 项目

> **知识点**: Cargo 与模块系统
> **层级**: Phase1_Rust速成基础
> **维度**: 实战代码（场景 2：构建一个简化版 ZeroClaw workspace）
> **目标受众**: 有 TypeScript/前端经验但零 Rust 经验的开发者
> **阅读时间**: 约 25 分钟
> **前提**: 已阅读 03_核心概念_4_Workspace、07_实战代码_场景1

---

## 场景目标

**构建一个 `mini-zeroclaw` workspace 项目**，模拟 ZeroClaw 的真实架构：

1. **虚拟 manifest** 管理多个 crate（workspace 根不含业务代码）
2. **core 库 crate** 提供配置管理和 Provider trait
3. **cli 二进制 crate** 引用 core 并实现命令行工具
4. **workspace.dependencies** 集中管理依赖版本

**最终目录结构**：

```
mini-zeroclaw/
├── Cargo.toml              ← 虚拟 workspace manifest（≈ pnpm-workspace.yaml）
├── Cargo.lock              ← 整个 workspace 共享（自动生成）
└── crates/
    ├── core/               ← 核心库 crate（≈ @zeroclaw/core 包）
    │   ├── Cargo.toml
    │   └── src/
    │       ├── lib.rs      ← 模块声明 + re-export
    │       ├── config.rs   ← Config 结构体
    │       └── provider.rs ← Provider trait
    └── cli/                ← CLI 二进制 crate（≈ @zeroclaw/cli 包）
        ├── Cargo.toml
        └── src/
            └── main.rs     ← 入口：使用 core 的公共 API
```

---

## Step 1：创建 workspace 根目录

```bash
mkdir mini-zeroclaw && cd mini-zeroclaw
mkdir -p crates
```

**Cargo.toml**（workspace 根——虚拟 manifest）：

```toml
# 虚拟 manifest——没有 [package]，只管理 workspace
# ≈ pnpm-workspace.yaml + 根 package.json 的结合

[workspace]
members = [
    "crates/core",      # 核心库
    "crates/cli",       # CLI 工具
]
resolver = "2"          # 推荐：更智能的 feature 解析

# ── 集中管理依赖版本 ──
[workspace.dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
mini-zeroclaw-core = { path = "crates/core" }  # 内部 crate 也在这里声明

# ── 共享元数据 ──
[workspace.package]
edition = "2024"
version = "0.1.0"
license = "MIT"
```

**与 pnpm workspace 对比**：`[workspace.dependencies]` ≈ pnpm 9+ 的 `catalog:` 协议；`[workspace.package]` ≈ `tsconfig.base.json` 的 `extends` 继承。`resolver = "2"` 让 dev-dependencies 的 features 不泄漏到正常构建中 [^1]。

---

## Step 2：创建 core 库 crate

```bash
cargo new --lib crates/core --name mini-zeroclaw-core
touch crates/core/src/config.rs crates/core/src/provider.rs
```

### 2.1 core/Cargo.toml

```toml
[package]
name = "mini-zeroclaw-core"
version.workspace = true        # ← 继承 0.1.0
edition.workspace = true        # ← 继承 2024
license.workspace = true        # ← 继承 MIT

[dependencies]
serde = { workspace = true }    # ← 继承版本 + features
serde_json = { workspace = true }
anyhow = { workspace = true }
```

每一行 `workspace = true` 都从根 `[workspace.dependencies]` 继承。就像 pnpm 子包写 `"lodash": "catalog:"`，版本由根目录统一定义 [^3]。

### 2.2 config.rs

**crates/core/src/config.rs**：

```rust
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Agent 配置——对应 ZeroClaw 的 config 模块
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub name: String,
    pub model: String,
    pub max_retries: u32,
    pub verbose: bool,
}

impl Config {
    pub fn from_json(json_str: &str) -> Result<Self> {
        serde_json::from_str(json_str)
            .with_context(|| "Failed to parse config JSON")
    }

    pub fn default_config() -> Self {
        Config {
            name: "mini-zeroclaw".to_string(),
            model: "gpt-4".to_string(),
            max_retries: 3,
            verbose: false,
        }
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .with_context(|| "Failed to serialize config")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_json() {
        let json = r#"{ "name": "test", "model": "gpt-4", "max_retries": 5, "verbose": true }"#;
        let config = Config::from_json(json).unwrap();
        assert_eq!(config.name, "test");
        assert_eq!(config.max_retries, 5);
    }

    #[test]
    fn test_roundtrip() {
        let config = Config::default_config();
        let json = config.to_json().unwrap();
        let restored = Config::from_json(&json).unwrap();
        assert_eq!(config.name, restored.name);
    }
}
```

### 2.3 provider.rs

**crates/core/src/provider.rs**：

```rust
use anyhow::Result;

/// LLM 提供商 trait——对应 ZeroClaw 的 13+ 提供商（OpenAI、Anthropic 等）
pub trait Provider {
    fn name(&self) -> &str;
    fn chat(&self, message: &str) -> Result<String>;
}

/// 模拟提供商
pub struct MockProvider {
    model: String,
}

impl MockProvider {
    pub fn new(model: &str) -> Self {
        MockProvider { model: model.to_string() }
    }
}

impl Provider for MockProvider {
    fn name(&self) -> &str { "mock-openai" }

    fn chat(&self, message: &str) -> Result<String> {
        Ok(format!("[{}@{}] Echo: {}", self.name(), self.model, message))
    }
}

/// 工厂函数——简化版 ZeroClaw 的 create_provider()
pub fn create_provider(name: &str, model: &str) -> Result<Box<dyn Provider>> {
    match name {
        "mock" | "mock-openai" => Ok(Box::new(MockProvider::new(model))),
        _ => anyhow::bail!("Unknown provider: {}", name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_provider() {
        let provider = MockProvider::new("gpt-4");
        assert_eq!(provider.name(), "mock-openai");
        let response = provider.chat("hello").unwrap();
        assert!(response.contains("Echo: hello"));
    }

    #[test]
    fn test_create_provider() {
        let ok = create_provider("mock", "gpt-4");
        assert!(ok.is_ok());
        let err = create_provider("nonexistent", "model");
        assert!(err.is_err());
    }

    #[test]
    fn test_trait_object() {
        // 验证 Provider 可以作为 trait object（动态分发）
        let providers: Vec<Box<dyn Provider>> = vec![
            Box::new(MockProvider::new("gpt-4")),
            Box::new(MockProvider::new("gpt-3.5")),
        ];
        for p in &providers {
            assert!(p.chat("test").unwrap().contains("Echo: test"));
        }
    }
}
```

### 2.4 lib.rs —— 模块声明与 re-export

**crates/core/src/lib.rs**：

```rust
//! mini-zeroclaw-core: 核心库
//! 提供配置管理和 LLM Provider 抽象。

pub mod config;
pub mod provider;

// Re-export 关键类型——用户写 `use mini_zeroclaw_core::Config` 即可
// 对应 ZeroClaw lib.rs 的 `pub use config::Config;` 模式
pub use config::Config;
pub use provider::{Provider, MockProvider, create_provider};
```

**TypeScript 类比**——这就是 barrel export：

```typescript
// index.ts
export { Config } from './config';
export { Provider, MockProvider, createProvider } from './provider';
```

---

## Step 3：创建 cli 二进制 crate

```bash
cargo new crates/cli --name mini-zeroclaw-cli
```

### 3.1 cli/Cargo.toml

```toml
[package]
name = "mini-zeroclaw-cli"
version.workspace = true
edition.workspace = true

[dependencies]
mini-zeroclaw-core = { workspace = true }  # 内部 crate 依赖
anyhow = { workspace = true }
serde_json = { workspace = true }
```

cli 写 `mini-zeroclaw-core = { workspace = true }`，Cargo 在根 `[workspace.dependencies]` 找到 `{ path = "crates/core" }`。相当于 pnpm 的 `"@zeroclaw/core": "workspace:*"` [^3]。

### 3.2 cli/src/main.rs

```rust
// 因为 lib.rs 做了 re-export，可以直接从根路径导入
// 注意：Cargo 名称中的连字符 - 在代码中变成下划线 _
use mini_zeroclaw_core::{Config, Provider, create_provider};

fn main() -> anyhow::Result<()> {
    println!("🤖 mini-zeroclaw CLI v{}", env!("CARGO_PKG_VERSION"));
    println!("─────────────────────────────────────────");

    // 加载配置
    let config = Config::default_config();
    println!("📋 Config: {} (model: {})", config.name, config.model);

    // 创建 Provider（工厂模式）
    let provider = create_provider("mock", &config.model)?;
    println!("🔌 Provider: {}", provider.name());

    // 模拟 agent 对话
    let questions = ["What is Rust?", "Explain ownership.", "How do lifetimes work?"];
    println!("\n💬 Running {} queries...\n", questions.len());

    for (i, q) in questions.iter().enumerate() {
        let response = provider.chat(q)?;
        println!("  [{}/{}] Q: {}", i + 1, questions.len(), q);
        println!("        A: {}", response);
    }

    // 序列化配置（展示跨 crate 的 serde 使用）
    println!("\n📄 Config JSON:\n{}", config.to_json()?);
    println!("\n✅ Done!");
    Ok(())
}
```

**关键语法**：
- `use mini_zeroclaw_core::{A, B, C}` ≈ `import { A, B, C } from '@zeroclaw/core'`
- `fn main() -> anyhow::Result<()>` —— main 返回 Result，出错自动打印并退出
- `env!("CARGO_PKG_VERSION")` —— 编译期从 Cargo.toml 读版本号

---

## Step 4：workspace.dependencies 继承机制

```
根 Cargo.toml 定义                    成员 Cargo.toml 继承
─────────────────────────────────────────────────────────────────
serde = { version = "1.0",      →     serde = { workspace = true }
          features = ["derive"] }      # 自动获得 version + features

mini-zeroclaw-core =            →     mini-zeroclaw-core = { workspace = true }
  { path = "crates/core" }           # 自动获得 path 依赖
```

**核心价值**：版本只在一个地方定义，所有 crate 保持一致。Rust 的 workspace.dependencies 比 pnpm catalog 更灵活——还能继承 `features`、`default-features`、`optional` [^1]。

---

## Step 5：跨 crate 引用链路

```
cli/main.rs                     ← use mini_zeroclaw_core::Config
    │                              （Cargo 名 - 变 _）
    ▼
core/lib.rs                     ← pub use config::Config  (re-export)
    │
    ▼
core/config.rs                  ← pub struct Config { ... }
```

**没有 re-export 时**必须写完整路径 `use mini_zeroclaw_core::config::Config`。ZeroClaw 的 `src/lib.rs` 也用同样的 re-export 模式 [^2]。

---

## Step 6：构建和测试

```bash
cargo build --workspace           # 构建所有           ≈ pnpm -r build
cargo test --workspace            # 测试所有           ≈ pnpm -r test
cargo run -p mini-zeroclaw-cli    # 运行 cli           ≈ pnpm --filter cli start
cargo test -p mini-zeroclaw-core  # 只测试 core        ≈ pnpm --filter core test
cargo check --workspace           # 快速类型检查       ≈ pnpm -r typecheck
cargo clippy --workspace          # lint 所有          ≈ pnpm -r lint
cargo fmt --all                   # 格式化所有         ≈ pnpm -r format
```

> **注意**：虚拟 manifest 不能直接 `cargo run`——必须用 `-p` 指定包名 [^3]。

### 预期输出

**`cargo run -p mini-zeroclaw-cli`**：

```
🤖 mini-zeroclaw CLI v0.1.0
─────────────────────────────────────────
📋 Config: mini-zeroclaw (model: gpt-4)
🔌 Provider: mock-openai

💬 Running 3 queries...

  [1/3] Q: What is Rust?
        A: [mock-openai@gpt-4] Echo: What is Rust?
  [2/3] Q: Explain ownership.
        A: [mock-openai@gpt-4] Echo: Explain ownership.
  [3/3] Q: How do lifetimes work?
        A: [mock-openai@gpt-4] Echo: How do lifetimes work?

📄 Config JSON:
{
  "name": "mini-zeroclaw",
  "model": "gpt-4",
  "max_retries": 3,
  "verbose": false
}

✅ Done!
```

**`cargo test --workspace`**：

```
test config::tests::test_from_json ... ok
test config::tests::test_roundtrip ... ok
test provider::tests::test_mock_provider ... ok
test provider::tests::test_create_provider ... ok
test provider::tests::test_trait_object ... ok

test result: ok. 5 passed; 0 failed
```

---

## Step 7：与 ZeroClaw 实际架构的对比

### 结构映射

```
mini-zeroclaw（本教程）              ZeroClaw（真实项目）
──────────────────────────────────────────────────────────────────
虚拟 manifest                        根包 workspace（根也是 crate）
crates/core（库 crate）              . (根 crate = zeroclaw 主包)
crates/cli（二进制 crate）           同一个根 crate 有 lib.rs + main.rs
2 个模块（config, provider）         30+ 个模块
workspace.dependencies               直接在根 [dependencies] 中管理
```

### 为什么 ZeroClaw 用「根包 workspace」？

ZeroClaw 的根 `Cargo.toml` 同时包含 `[package]` 和 `[workspace]` [^2]：

```toml
[package]
name = "zeroclaw"
version = "0.1.7"

[workspace]
members = [".", "crates/robot-kit"]
resolver = "2"
```

**原因**：只有一个主 binary + 一个子库时，根包 workspace 更简洁。3+ 个 crate 时推荐虚拟 manifest。

### Robot-Kit 独立为 crate 的原因

- **不同的 feature flags**——`safety`（默认）、`ros2`、`gpio`、`vision` 等
- **并行编译**——改动 robot-kit 不触发整个 zeroclaw 重编译
- **清晰边界**——有独立的公共 API 和测试

Firmware crates（ESP32、STM32）**不在 workspace 中**——不同编译目标（ARM/Xtensa vs x86）和不同运行时（`no_std` vs `std`）不应共享 [^2]。

---

## 知识点速览

| 知识点 | 项目中的体现 |
|--------|-------------|
| 虚拟 manifest | 根 Cargo.toml 只有 `[workspace]`，没有 `[package]` |
| workspace.dependencies | serde、anyhow 版本统一在根定义 |
| workspace.package | version、edition、license 统一继承 |
| `{ workspace = true }` | 成员 crate 的每个依赖都用此语法继承 |
| 内部路径依赖 | `mini-zeroclaw-core = { path = "crates/core" }` |
| `pub mod` + `pub use` | lib.rs 声明模块并 re-export 关键类型 |
| trait + 工厂模式 | `Provider` trait + `create_provider()` → `Box<dyn Provider>` |
| `-p` 指定包 | `cargo run -p mini-zeroclaw-cli` |
| 连字符转下划线 | `mini-zeroclaw-core` → `mini_zeroclaw_core` |

---

> **下一步**: 阅读 `07_实战代码_场景3_Feature_Flags条件编译.md`，学习如何用 Cargo feature flags 实现条件编译——ZeroClaw 的 15+ 可选特性就是这么做的。

---

**引用来源**

[^1]: Cargo 官方文档：workspace.dependencies 依赖继承与 resolver 配置（`reference/context7_cargo_01.md`）
[^2]: ZeroClaw 源码分析：workspace 配置、30+ 模块、re-export 模式、robot-kit 子 crate（`reference/source_cargo_module_01.md`）
[^3]: Rust Cargo 社区最佳实践：虚拟 manifest、依赖继承、TypeScript 类比表（`reference/search_cargo_module_01.md`）

---

**文件信息**
- 知识点: Cargo 与模块系统
- 维度: 07_实战代码_场景2_Workspace多crate项目
- 版本: v1.0
- 日期: 2026-03-10
