# 核心概念 3：anyhow 错误处理

> **知识点**: 常用库速查（serde/anyhow/clap）
> **层级**: Phase1_Rust速成基础
> **维度**: 核心概念（anyhow 错误处理 — Result/Context/bail!/anyhow!、错误链、thiserror 配合）
> **目标受众**: 有 TypeScript/前端经验但零 Rust 经验的开发者

---

## 一句话定义

> **anyhow 是 Rust 的"万能错误容器"——它把所有错误类型统一收进一个 `anyhow::Error`，自动串联上下文信息形成"错误链"，让你写应用层代码时不再为「该用哪个错误类型」而纠结。相当于 `try/catch` + Sentry breadcrumbs 的编译时增强版。**

---

## 1. anyhow 是什么

### 1.1 解决的核心痛点

Rust 标准库的错误处理要求你**精确指定错误类型**：

```rust
fn load_config(path: &str) -> Result<Config, ???> {
    let content = std::fs::read_to_string(path)?;  // 返回 io::Error
    let config: Config = toml::from_str(&content)?;  // 返回 toml::de::Error
    // ❌ 两种不同的错误类型，Result 的 E 该写什么？
    Ok(config)
}
```

三个选择：
1. 写自定义 enum 包装所有错误 —— **太啰嗦**
2. 用 `Box<dyn std::error::Error>` —— **丢失上下文**
3. 用 `anyhow::Result` —— ✅ **简单且保留完整信息**

### 1.2 前端类比

```typescript
// TypeScript：catch 的 error 是 unknown，你不知道它是什么类型
async function loadConfig(path: string): Promise<Config> {
  try {
    const content = await fs.readFile(path, "utf-8");
    return JSON.parse(content);
  } catch (e: unknown) {
    throw new Error(`Failed to load config`, { cause: e }); // 手动包装
  }
}
// anyhow = 自动帮你做 { cause: e } 包装的增强版 try/catch
// 而且是编译时强制的——你不可能"忘了处理错误"
```

### 1.3 anyhow 的解法

```rust
use anyhow::Result;

fn load_config(path: &str) -> Result<Config> {
    let content = std::fs::read_to_string(path)?;  // io::Error → 自动转换
    let config: Config = toml::from_str(&content)?; // toml::Error → 自动转换
    Ok(config)
}
// anyhow::Result<T> = Result<T, anyhow::Error>
// anyhow::Error 能装任何实现了 std::error::Error 的类型
```

---

## 2. 核心 API

### 2.1 `anyhow::Result<T>` — 统一返回类型

```rust
use anyhow::Result;

// anyhow::Result<Config> = Result<Config, anyhow::Error>
fn read_config() -> Result<String> {
    let content = std::fs::read_to_string("config.toml")?;
    Ok(content)
}

// main 也可以返回 anyhow::Result，出错时自动打印错误链到 stderr
fn main() -> Result<()> {
    let config = read_config()?;
    println!("Config: {}", config);
    Ok(())
}
```

TypeScript 对比：anyhow 给你两全其美——像 TS 一样不需要精确指定错误类型（灵活），像 Rust 一样编译时强制处理错误（安全），还自动保留完整错误链（可调试）。

### 2.2 `.context()` 和 `.with_context()` — 添加错误上下文

这是 anyhow **最核心的功能**——给错误附加"面包屑"：

```rust
use anyhow::Context;

fn load_config(path: &str) -> anyhow::Result<Config> {
    // .context() — 静态字符串上下文
    let content = std::fs::read_to_string(path)
        .context("Failed to read config file")?;

    // .with_context() — 动态构建（惰性求值，只在出错时执行 format!）
    let config: Config = toml::from_str(&content)
        .with_context(|| format!("Failed to parse config from {}", path))?;

    Ok(config)
}
// 错误输出：
//   Error: Failed to read config file
//   Caused by: No such file or directory (os error 2)
```

**选择规则**：固定字符串用 `.context()`，包含变量用 `.with_context(|| format!(...))`。

### 2.3 `bail!` 宏 — 提前返回错误

```rust
use anyhow::bail;

fn validate_temperature(temp: f64) -> anyhow::Result<()> {
    if temp < 0.0 || temp > 2.0 {
        bail!("temperature must be between 0.0 and 2.0, got {}", temp);
        // 等价于 return Err(anyhow!("..."))，但更简洁
    }
    Ok(())
}
```

TypeScript 对比：`bail!` ≈ `throw new Error(msg)`，但 bail! 是返回值（编译器强制处理），throw 是异常（可以被忽略）。

### 2.4 `ensure!` 宏 — 条件断言

```rust
use anyhow::ensure;

fn process_request(tokens: u32, max: u32) -> anyhow::Result<()> {
    ensure!(tokens <= max, "token count {} exceeds limit {}", tokens, max);
    // 等价于 if !(tokens <= max) { bail!("..."); }
    Ok(())
}
```

### 2.5 `anyhow!` 宏 — 创建错误（不立即返回）

```rust
use anyhow::anyhow;

// bail! = 创建 + 立即 return Err
// anyhow! = 只创建，你决定怎么用
fn find_provider(name: &str) -> anyhow::Result<Provider> {
    let provider = providers.get(name)
        .ok_or_else(|| anyhow!("unknown provider: {}", name))?;
    Ok(provider.clone())
}
```

---

## 3. 错误链遍历

anyhow 的错误是一条**链**——每次 `.context()` 增加一环：

```rust
fn main() {
    if let Err(e) = load_config() {
        eprintln!("Error: {}", e);
        // Error: Failed to load application config

        for (i, cause) in e.chain().enumerate() {
            eprintln!("  [{}]: {}", i, cause);
        }
        // [0]: Failed to load application config
        // [1]: Failed to open file: config.toml
        // [2]: No such file or directory (os error 2)

        eprintln!("Root cause: {}", e.root_cause());
        // Root cause: No such file or directory (os error 2)
    }
}
```

### 下转型到具体错误类型

```rust
fn handle_error(err: &anyhow::Error) {
    if let Some(io_err) = err.downcast_ref::<std::io::Error>() {
        match io_err.kind() {
            std::io::ErrorKind::NotFound => {
                eprintln!("File not found, creating default...");
            }
            std::io::ErrorKind::PermissionDenied => {
                eprintln!("Permission denied");
            }
            _ => eprintln!("IO error: {}", io_err),
        }
    }
}
```

---

## 4. anyhow vs thiserror — 黄金搭档

```
            anyhow                          thiserror
───────────────────────────────────────────────────────────────
定位        万能错误容器（垃圾桶）            精确错误类型（分类标签）
用途        应用层代码（binary）              库层代码（library）
类比(前端)  catch (e: unknown) { ... }       自定义 Error class
适用文件    main.rs, CLI handlers            error.rs, 公开 API
错误创建    bail!("msg"), anyhow!("msg")     #[derive(thiserror::Error)]
```

### 配合使用示例

```rust
// ===== 库层：用 thiserror 定义精确错误类型 =====
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("config file not found: {path}")]
    NotFound { path: String },
    #[error("invalid config format")]
    InvalidFormat(#[from] toml::de::Error),
    #[error("missing required field: {0}")]
    MissingField(String),
}

// ===== 应用层：用 anyhow 统一处理 =====
use anyhow::{Context, Result};

fn main() -> Result<()> {
    let config = load_config("config.toml")  // ConfigError → 自动转为 anyhow::Error
        .context("failed to initialize application")?;
    run(config)?;
    Ok(())
}
```

> **一句话判断**：写给别人用的库？用 thiserror。写自己的应用？用 anyhow。两个同时依赖，不冲突。

---

## 5. ZeroClaw 中的真实使用模式

### 5.1 统一的 trait 返回类型 [^source1]

```rust
// ZeroClaw 四大核心 trait 全部返回 anyhow::Result
trait Tool:     async fn execute(&self, args: Value) -> anyhow::Result<ToolResult>;
trait Memory:   async fn store(&self, entry: MemoryEntry) -> anyhow::Result<()>;
trait Channel:  async fn receive(&mut self) -> anyhow::Result<ChannelMessage>;
trait Provider: async fn chat(&self, msgs: Vec<Message>) -> anyhow::Result<Response>;
// 因为 ZeroClaw 是应用（binary），不是库，用 anyhow 统一兜底最省心
```

### 5.2 配置加载中的 .context()

```rust
// 来自 ZeroClaw src/config/schema.rs 风格
use anyhow::{Context, Result};

pub async fn load(config_dir: Option<&str>) -> Result<Config> {
    let config_path = resolve_config_path(config_dir)?;
    let content = tokio::fs::read_to_string(&config_path)
        .await
        .with_context(|| format!("reading config from {}", config_path.display()))?;
    let config: Config = toml::from_str(&content)
        .context("parsing TOML config")?;
    Ok(config)
}
```

### 5.3 Cargo.toml 配置

```toml
[dependencies]
anyhow = "1.0"       # 应用层错误处理
thiserror = "2.0"    # 库层自定义错误类型
```

---

## 6. 完整可运行示例

```rust
use anyhow::{bail, ensure, Context, Result};
use std::collections::HashMap;

fn main() -> Result<()> {
    // 正常路径
    match load_config("app.toml") {
        Ok(config) => {
            validate_config(&config)?;
            println!("Config loaded: {:?}", config);
        }
        Err(e) => {
            // 演示错误链遍历
            eprintln!("Error: {}", e);
            for (i, cause) in e.chain().enumerate() {
                eprintln!("  [{}]: {}", i, cause);
            }
            eprintln!("Root cause: {}", e.root_cause());
        }
    }
    Ok(())
}

#[derive(Debug)]
struct AppConfig {
    provider: String,
    model: String,
    temperature: f64,
}

fn load_config(path: &str) -> Result<AppConfig> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("reading config file '{}'", path))?;

    let map = parse_kv(&content).context("parsing config key-value pairs")?;

    let provider = map.get("provider")
        .ok_or_else(|| anyhow::anyhow!("missing 'provider' field"))?
        .to_string();
    let model = map.get("model")
        .ok_or_else(|| anyhow::anyhow!("missing 'model' field"))?
        .to_string();
    let temperature: f64 = map.get("temperature")
        .unwrap_or(&"0.7")
        .parse()
        .context("invalid temperature value")?;

    Ok(AppConfig { provider, model, temperature })
}

fn parse_kv(content: &str) -> Result<HashMap<String, String>> {
    let mut map = HashMap::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let parts: Vec<&str> = line.splitn(2, '=').collect();
        ensure!(parts.len() == 2, "invalid line format: '{}'", line);
        map.insert(parts[0].trim().to_string(), parts[1].trim().to_string());
    }
    Ok(map)
}

fn validate_config(config: &AppConfig) -> Result<()> {
    ensure!(!config.provider.is_empty(), "provider cannot be empty");
    if config.temperature < 0.0 || config.temperature > 2.0 {
        bail!("temperature {} out of range [0.0, 2.0]", config.temperature);
    }
    Ok(())
}
```

---

## 7. 最佳实践清单

### ✅ 推荐做法

```
实践                              说明
─────────────────────────────────────────────────────────────────
函数返回 anyhow::Result<T>       应用层的默认选择
每个 ? 加 .context()              让错误信息可读：「在做什么 + 为什么失败」
动态信息用 .with_context(||...)   包含变量时用闭包，正常路径零开销
参数校验用 bail!/ensure!          比 panic! 优雅，比手动 Err 简洁
main() -> Result<()>              让 anyhow 自动格式化打印错误信息
```

### ❌ 常见反模式

```rust
// ❌ 用 unwrap() 代替 ?（panic 无错误信息）
let content = std::fs::read_to_string(path).unwrap();
// ✅ 用 ? + context
let content = std::fs::read_to_string(path).context("reading config")?;

// ❌ .context() 信息太模糊
.context("error")?;
// ✅ 说清楚「在做什么」
.context("reading config file")?;

// ❌ 库层用 anyhow（调用方无法匹配具体错误）
pub fn lib_function() -> anyhow::Result<()> { ... }
// ✅ 库层用 thiserror
pub fn lib_function() -> Result<(), MyError> { ... }
```

### `.context()` 消息写法规范

```rust
// 模式：动词短语（doing what），小写开头
.context("reading config file")?;
.context("connecting to database")?;
.with_context(|| format!("loading model '{}'", model_name))?;
```

---

## 速查卡

```
anyhow 核心概念：

  Cargo.toml：
    anyhow = "1.0"              # 应用层错误处理
    thiserror = "2.0"           # 库层自定义错误（可选搭配）

  核心类型：
    anyhow::Result<T>           = Result<T, anyhow::Error>

  四个宏（都支持 format! 风格参数）：
    bail!("msg")                提前返回错误（≈ throw new Error）
    ensure!(cond, "msg")        条件断言（≈ assert 但返回 Err）
    anyhow!("msg")              创建错误不返回（≈ new Error）

  两个上下文方法（需 use anyhow::Context）：
    .context("static msg")?           静态上下文
    .with_context(|| format!(...))?   动态上下文（惰性求值）

  错误链 API：
    e.chain()                   遍历完整错误链
    e.root_cause()              获取根本原因
    e.downcast_ref::<T>()       转回具体错误类型（引用）
    e.downcast::<T>()           转回具体错误类型（消耗所有权）

  使用场景：
    应用层（main.rs, 业务逻辑）  → anyhow
    库层（公开 API, error.rs）   → thiserror

  vs TypeScript：
    throw new Error(msg)        → bail!(msg)
    new Error(msg, {cause: e})  → .context(msg)
    catch (e: unknown)          → anyhow::Result<T>（编译时强制）
    error.cause 链              → e.chain()（自动管理）
```

---

> **下一篇**: 阅读 `03_核心概念_4_clap_CLI解析.md`，学习 clap 的 derive API 四件套——理解了 serde 和 anyhow 的 derive 模式后，clap 的 `#[derive(Parser)]` 会非常自然。

---

**参考来源**

[^source1]: ZeroClaw 源码分析 — `reference/source_常用库_01.md`
[^context7_anyhow1]: anyhow 官方文档 — `reference/context7_anyhow_01.md`
[^search1]: Reddit 社区最佳实践 — `reference/search_常用库_01.md`

---

**文件信息**
- 知识点: 常用库速查（serde/anyhow/clap）
- 维度: 03_核心概念_3_anyhow错误处理
- 版本: v1.0
- 日期: 2026-03-11
