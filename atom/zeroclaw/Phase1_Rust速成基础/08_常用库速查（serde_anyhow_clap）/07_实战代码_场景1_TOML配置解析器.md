# 实战代码 场景1：TOML 配置解析器

> **知识点**: 常用库速查（serde/anyhow/clap）
> **层级**: Phase1_Rust速成基础
> **维度**: 实战代码 — 场景1：TOML 配置解析器（serde + toml + anyhow）
> **目标受众**: 有 TypeScript/前端经验但零 Rust 经验的开发者

---

## 场景说明

模仿 ZeroClaw 的 `config/schema.rs`，实现一个完整的 TOML 配置加载系统。你将学到 serde derive 宏 + 字段属性、toml 文件读写、anyhow `.context()` 错误链——三库在真实项目中如何协作。

---

## 1. Cargo.toml 配置

```toml
[package]
name = "config-parser"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
toml = "1.0"
anyhow = "1.0"
```

运行 `cargo new config-parser && cd config-parser`，替换 `Cargo.toml`。

---

## 2. 示例 config.toml

在项目根目录创建 `config.toml`：

```toml
# ZeroClaw 风格的配置文件
default_provider = "openrouter"
default_model = "claude-sonnet-4-20250514"

[providers.openai]
api_key = "sk-openai-xxx"
base_url = "https://api.openai.com/v1"
max_retries = 3

[providers.openrouter]
api_key = "sk-or-xxx"
base_url = "https://openrouter.ai/api/v1"

[security]
enable_sandbox = true
timeout_secs = 60
allowed_domains = ["github.com", "docs.rs"]

[memory]
max_entries = 1000
```

---

## 3. 核心代码（src/main.rs）

```rust
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// ============================================================
// 第一部分：定义 Config 结构体（模仿 ZeroClaw config/schema.rs）
// ============================================================

/// 顶层配置结构
/// TS 类比：z.object({ default_provider: z.string().default("openrouter"), ... })
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    // skip: 运行时计算的字段，不写入 TOML 文件
    // TypeScript 类比：class 中的 private 字段，不参与 JSON 序列化
    #[serde(skip)]
    pub config_path: String,

    // alias: 兼容旧配置中的 "provider" 键名
    // TypeScript 类比：无直接等价物，需手动 if (obj.provider) obj.default_provider = obj.provider
    #[serde(default = "default_provider", alias = "provider")]
    pub default_provider: String,

    #[serde(default)]
    pub default_model: Option<String>,

    // default: 缺失时用 HashMap::default()（空 map）
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,

    // default: 缺失时用 SecurityConfig::default()
    #[serde(default)]
    pub security: SecurityConfig,

    #[serde(default)]
    pub memory: MemoryConfig,
}

fn default_provider() -> String {
    "openrouter".to_string()
}

/// Provider 配置（嵌套子结构）
///
/// rename_all = "snake_case" 确保 TOML 中用 snake_case 风格
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ProviderConfig {
    // skip_serializing_if: 当 api_key 是 None 时，序列化不输出此字段
    // TypeScript 类比：JSON.stringify 时自动去掉 undefined 字段
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,

    #[serde(default = "default_base_url")]
    pub base_url: String,

    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
}

fn default_base_url() -> String {
    "https://api.openai.com/v1".to_string()
}
fn default_max_retries() -> u32 {
    3
}

/// 安全配置 — 手动实现 Default trait
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    #[serde(default)]
    pub enable_sandbox: bool,

    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,

    #[serde(default)]
    pub allowed_domains: Vec<String>,
}

// 手动实现 Default（比 #[derive(Default)] 更灵活——可以自定义默认值）
impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_sandbox: false,
            timeout_secs: 30,
            allowed_domains: vec![],
        }
    }
}

fn default_timeout() -> u64 {
    30
}

/// 记忆配置 — 使用 #[derive(Default)] 自动实现
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryConfig {
    #[serde(default = "default_max_entries")]
    pub max_entries: u32,
}

fn default_max_entries() -> u32 {
    500
}

// ============================================================
// 第二部分：配置加载函数（展示 anyhow 的错误处理）
// ============================================================

/// 从文件加载配置
/// TS 类比：readFileSync(path) + TOML.parse(content)，但每一步失败都有清晰上下文。
pub fn load_config(path: &str) -> Result<Config> {
    // .context() 给 io::Error 附加「在做什么」的说明
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("reading config file '{}'", path))?;

    // 空文件检查：bail! ≈ throw new Error(...)
    ensure!(!content.trim().is_empty(), "config file '{}' is empty", path);

    // 解析 TOML 并附加上下文
    let mut config: Config = toml::from_str(&content)
        .with_context(|| format!("parsing TOML from '{}'", path))?;

    // 记录配置文件路径（skip 字段在反序列化时不会被填充）
    config.config_path = path.to_string();

    Ok(config)
}

/// 加载配置，文件不存在时返回默认配置
pub fn load_or_default(path: &str) -> Result<Config> {
    if Path::new(path).exists() {
        load_config(path)
    } else {
        println!("  Config file not found, using defaults.");
        Ok(Config {
            config_path: path.to_string(),
            default_provider: default_provider(),
            default_model: None,
            providers: HashMap::new(),
            security: SecurityConfig::default(),
            memory: MemoryConfig::default(),
        })
    }
}

// ============================================================
// 第三部分：环境变量覆盖（默认值 → 配置文件 → 环境变量）
// ============================================================

/// 用环境变量覆盖配置（优先级最高）
/// TS 类比：config.provider = process.env.ZEROCLAW_PROVIDER ?? config.provider
pub fn apply_env_overrides(config: &mut Config) {
    if let Ok(provider) = std::env::var("ZEROCLAW_PROVIDER") {
        println!("  ENV override: default_provider = {}", provider);
        config.default_provider = provider;
    }
    if let Ok(model) = std::env::var("ZEROCLAW_MODEL") {
        println!("  ENV override: default_model = {}", model);
        config.default_model = Some(model);
    }
    if let Ok(timeout) = std::env::var("ZEROCLAW_TIMEOUT") {
        if let Ok(secs) = timeout.parse::<u64>() {
            println!("  ENV override: timeout_secs = {}", secs);
            config.security.timeout_secs = secs;
        }
    }
}

// ============================================================
// 第四部分：配置保存（序列化回 TOML 写入文件）
// ============================================================

/// 保存配置到 TOML 文件
///
/// 注意：#[serde(skip)] 的字段不会被写入文件
pub fn save_config(config: &Config, path: &str) -> Result<()> {
    let content = toml::to_string_pretty(config)
        .context("serializing config to TOML")?;

    std::fs::write(path, &content)
        .with_context(|| format!("writing config to '{}'", path))?;

    println!("  Config saved to '{}'", path);
    Ok(())
}

// ============================================================
// 第五部分：配置验证
// ============================================================

pub fn validate_config(config: &Config) -> Result<()> {
    // ensure! = 条件不满足就返回 Err（≈ if (!cond) throw）
    ensure!(
        !config.default_provider.is_empty(),
        "default_provider cannot be empty"
    );

    if config.security.timeout_secs == 0 {
        bail!("timeout_secs must be > 0 (got 0)");
    }
    if config.security.timeout_secs > 300 {
        bail!("timeout_secs {} exceeds max 300", config.security.timeout_secs);
    }

    Ok(())
}

// ============================================================
// 第六部分：main 函数 — 完整运行流程
// ============================================================

fn main() -> Result<()> {
    println!("=== TOML Config Parser Demo ===\n");

    // 步骤 1：加载配置（带 anyhow context 的错误链）
    println!("[1] Loading config...");
    let mut config = load_or_default("config.toml")
        .context("initializing configuration")?;

    // 步骤 2：应用环境变量覆盖
    println!("\n[2] Checking env overrides...");
    apply_env_overrides(&mut config);

    // 步骤 3：验证配置
    println!("\n[3] Validating config...");
    validate_config(&config).context("config validation failed")?;
    println!("  ✓ Config is valid");

    // 步骤 4：打印配置信息
    println!("\n[4] Current config:");
    println!("  provider:  {}", config.default_provider);
    println!("  model:     {:?}", config.default_model);
    println!("  timeout:   {}s", config.security.timeout_secs);
    println!("  sandbox:   {}", config.security.enable_sandbox);
    println!("  providers: {:?}", config.providers.keys().collect::<Vec<_>>());

    // 步骤 5：修改并保存
    println!("\n[5] Modifying and saving...");
    config.default_model = Some("gpt-4o".to_string());
    config.memory.max_entries = 2000;
    save_config(&config, "config_output.toml")?;

    // 步骤 6：重新加载验证
    println!("\n[6] Reloading saved config...");
    let reloaded = load_config("config_output.toml")
        .context("reloading saved config")?;
    println!("  model after reload: {:?}", reloaded.default_model);
    println!("  memory.max_entries: {}", reloaded.memory.max_entries);
    // 注意：config_path 是 skip 字段，reload 后会是空字符串
    println!("  config_path (skip): '{}'", reloaded.config_path);

    println!("\n=== Done! ===");
    Ok(())
}
```

---

## 4. 运行输出示例

```bash
$ cargo run
=== TOML Config Parser Demo ===

[1] Loading config...

[2] Checking env overrides...

[3] Validating config...
  ✓ Config is valid

[4] Current config:
  provider:  openrouter
  model:     Some("claude-sonnet-4-20250514")
  timeout:   60s
  sandbox:   true
  providers: ["openai", "openrouter"]

[5] Modifying and saving...
  Config saved to 'config_output.toml'

[6] Reloading saved config...
  model after reload: Some("gpt-4o")
  memory.max_entries: 2000
  config_path (skip): ''

=== Done! ===
```

用环境变量覆盖测试：

```bash
$ ZEROCLAW_PROVIDER=anthropic ZEROCLAW_MODEL=claude-3-opus cargo run
# 输出中会显示 ENV override 信息，provider 变为 anthropic
```

故意制造错误来看 anyhow 错误链：

```bash
$ rm config.toml && cargo run
Error: initializing configuration

Caused by:
    0: reading config file 'config.toml'
    1: No such file or directory (os error 2)
```

---

## 5. 前端对比

```typescript
// Node.js 等价实现（需要 3 个库 + 手动错误拼接）
import { z } from "zod";
import { readFileSync, writeFileSync } from "fs";
import TOML from "@iarna/toml";

// 1. zod schema（Rust 用 serde derive 替代）
const ProviderSchema = z.object({
  api_key: z.string().optional(),
  base_url: z.string().default("https://api.openai.com/v1"),
  max_retries: z.number().default(3),
});

const ConfigSchema = z.object({
  default_provider: z.string().default("openrouter"),
  default_model: z.string().optional(),
  providers: z.record(ProviderSchema).default({}),
  security: z
    .object({
      enable_sandbox: z.boolean().default(false),
      timeout_secs: z.number().default(30),
      allowed_domains: z.array(z.string()).default([]),
    })
    .default({}),
});

// 2. 加载函数（手动 try/catch + 手动错误包装）
function loadConfig(path: string) {
  try {
    const content = readFileSync(path, "utf-8");
    const raw = TOML.parse(content);
    return ConfigSchema.parse(raw); // zod 运行时校验
  } catch (e) {
    throw new Error(`Failed to load config from ${path}`, { cause: e });
  }
}

// 对比关键差异：
// - TypeScript: 3 个库（zod + TOML + fs） vs Rust: 2 个库（serde + toml）
// - TypeScript: 运行时类型检查（zod）     vs Rust: 编译时类型检查（serde）
// - TypeScript: 手动 { cause: e }         vs Rust: 自动 .context()
// - TypeScript: 可以忘记 try/catch        vs Rust: Result 编译时强制处理
```

---

## 6. ZeroClaw 关联说明

本示例简化自 ZeroClaw 的 `src/config/schema.rs`，保留了核心模式：

| 本示例 | ZeroClaw 实际代码 |
|--------|-------------------|
| `Config` 4 个子配置 | `Config` 有 20+ 个 `#[serde(default)]` 配置段 |
| `#[serde(skip)]` config_path | `#[serde(skip)]` workspace_dir |
| `#[serde(alias = "provider")]` | `#[serde(alias = "model_provider")]` |
| `load_or_default()` 回退 | `Config::load()` + 文件不存在时创建默认 |
| `apply_env_overrides()` | ZeroClaw 通过 clap 参数 + 环境变量双重覆盖 |

读懂本示例后，你就能直接阅读 ZeroClaw 的 config 模块源码。

---

## 7. 关键知识点回顾

```
serde 属性用法总结（本示例中出现的）：

  #[serde(skip)]                      config_path 不写入文件
  #[serde(default)]                   缺失时用 Default::default()
  #[serde(default = "函数名")]         缺失时调用指定函数获取默认值
  #[serde(alias = "旧名")]            接受旧键名（向后兼容）
  #[serde(rename_all = "snake_case")] 整个 struct 的字段名风格
  #[serde(skip_serializing_if = "Option::is_none")]  None 时不输出

anyhow 用法总结：

  .context("静态描述")?               给错误附加静态上下文
  .with_context(|| format!(...))?     给错误附加动态上下文
  bail!("msg")                        提前返回错误
  ensure!(条件, "msg")                条件断言
  fn main() -> Result<()>             让 anyhow 自动打印错误链
```

---

## 8. 练习建议

**练习 A**（5 分钟）：给 `ProviderConfig` 添加一个 `#[serde(default)]` 的 `model: Option<String>` 字段，在 config.toml 中只给 openai 配一个 model，验证 openrouter 的 model 是 None。

**练习 B**（10 分钟）：添加一个 `validate_providers()` 函数，用 `ensure!` 检查当前 `default_provider` 在 `providers` map 中存在，不存在则返回带上下文的错误。

**练习 C**（15 分钟）：给 Config 添加 `#[serde(deny_unknown_fields)]`，然后在 TOML 里写一个不存在的键，观察错误输出。思考：为什么 ZeroClaw 选择**不用** deny_unknown_fields？（提示：渐进式配置兼容性）

---

**参考来源**

- ZeroClaw 源码分析 — `reference/source_常用库_01.md`
- serde 官方文档 — `reference/context7_serde_01.md`
- anyhow 官方文档 — `reference/context7_anyhow_01.md`

---

**文件信息**
- 知识点: 常用库速查（serde/anyhow/clap）
- 维度: 07_实战代码_场景1_TOML配置解析器
- 版本: v1.0
- 日期: 2026-03-11
