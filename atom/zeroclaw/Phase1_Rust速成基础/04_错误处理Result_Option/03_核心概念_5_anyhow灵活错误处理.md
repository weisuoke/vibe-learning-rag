# 核心概念 5：anyhow 灵活错误处理

> **前置知识**：03_核心概念_1_Result枚举.md, 03_核心概念_3_问号运算符与错误传播.md, 03_核心概念_4_组合子方法.md
> **预计阅读**：20 分钟
> **难度**：入门

---

## 一句话定义

**anyhow 是 Rust 的"万能错误容器"，把所有不同类型的错误统一装进一个 `anyhow::Error` 里，让你不用为每个函数定义具体的错误类型——ZeroClaw 所有 trait 方法都返回 `anyhow::Result<T>`。**

[来源: reference/context7_anyhow_01.md - anyhow 官方文档]

---

## 为什么需要 anyhow？

### 问题：标准 Result 的错误类型必须统一

```rust
use std::fs;

// 这个函数会遇到两种不同的错误类型
fn load_config(path: &str) -> Result<serde_json::Value, ???> {
    let content = fs::read_to_string(path)?;       // io::Error
    let config = serde_json::from_str(&content)?;   // serde_json::Error
    Ok(config)
}
// ??? 填什么？io::Error 和 serde_json::Error 是两个不同的类型！
```

标准 Result 要求你指定一个具体的错误类型 `E`。当函数内部可能产生多种错误时，你有三个选择：

| 方案 | 代码 | 缺点 |
|------|------|------|
| 自定义错误枚举 | `enum MyError { Io(io::Error), Json(serde_json::Error) }` | 每个函数都要定义，样板代码多 |
| `Box<dyn Error>` | `Result<T, Box<dyn std::error::Error>>` | 写起来长，没有便捷方法 |
| **anyhow** | `anyhow::Result<T>` | 一行搞定，还有 context/bail/ensure |

**anyhow 就是方案 3——最省事、最实用的选择。**

### TypeScript 对照

```typescript
// TypeScript：所有错误都是 Error 类（或 unknown），天然"统一"
function loadConfig(path: string): Config {
  const content = fs.readFileSync(path, "utf-8"); // 可能抛 Error
  return JSON.parse(content);                      // 可能抛 SyntaxError
  // 不需要声明错误类型——catch(e) 的 e 是 unknown
}
```

Rust 的类型系统更严格，不允许"随便抛个东西"。anyhow 在保持类型安全的前提下，提供了接近 TypeScript 的便利性。

[来源: reference/search_error_handling_01.md - 库生态]

---

## anyhow::Result<T>

`anyhow::Result<T>` 是 `Result<T, anyhow::Error>` 的类型别名：

```rust
// anyhow 库内部定义（简化）
pub type Result<T> = std::result::Result<T, anyhow::Error>;
```

**`anyhow::Error` 能装什么？** 任何实现了 `std::error::Error + Send + Sync + 'static` 的类型。标准库和几乎所有第三方库的错误类型都满足这个条件。

```rust
use anyhow::Result;

// 之前的问题，现在一行解决
fn load_config(path: &str) -> Result<serde_json::Value> {
    let content = std::fs::read_to_string(path)?;   // io::Error → anyhow::Error（自动）
    let config = serde_json::from_str(&content)?;    // serde_json::Error → anyhow::Error（自动）
    Ok(config)
}
```

**ZeroClaw 的选择**：所有 trait 方法统一返回 `anyhow::Result<T>`。

```rust
// src/tools/traits.rs
async fn execute(&self, args: Value) -> anyhow::Result<ToolResult>;

// src/memory/traits.rs
async fn store(&self, entry: MemoryEntry) -> anyhow::Result<()>;
async fn recall(&self, query: &str, limit: usize) -> anyhow::Result<Vec<MemoryEntry>>;

// src/channels/traits.rs
async fn receive(&mut self) -> anyhow::Result<ChannelMessage>;
async fn send(&self, message: ChannelMessage) -> anyhow::Result<()>;
```

[来源: reference/source_error_handling_01.md - 统一错误类型：anyhow::Result<T>]

---

## 四大核心 API

### 1. anyhow!() 宏 — 创建临时错误

从字符串创建一个 `anyhow::Error`，支持 `format!` 风格的格式化。

```rust
use anyhow::{anyhow, Result};

fn validate_id(id: &str) -> Result<u64> {
    if id.is_empty() {
        return Err(anyhow!("user ID cannot be empty"));
    }
    id.parse::<u64>()
        .map_err(|e| anyhow!("invalid user ID '{}': {}", id, e))
}
```

**ZeroClaw 实战**：

```rust
// src/tools/web_fetch.rs — 参数缺失时创建错误
let url = args.get("url")
    .and_then(|v| v.as_str())
    .ok_or_else(|| anyhow!("Missing 'url' parameter"))?;
//                 ^^^^^^^^ 创建临时错误
```

```typescript
// TypeScript 等价
throw new Error(`Missing 'url' parameter`);
// 或
return { ok: false, error: new Error("Missing 'url' parameter") };
```

[来源: reference/context7_anyhow_01.md - anyhow! 宏]
[来源: reference/source_error_handling_01.md - tools/web_fetch.rs 分析]

### 2. bail!() 宏 — 提前返回错误

`bail!("msg")` 等价于 `return Err(anyhow!("msg"))`。用于验证失败时提前退出函数。

```rust
use anyhow::{bail, Result};

fn check_permissions(user: &str) -> Result<()> {
    if user != "admin" {
        bail!("insufficient permissions for user {}", user);
        // 等价于：return Err(anyhow!("insufficient permissions for user {}", user));
    }
    Ok(())
}
```

**ZeroClaw 实战**：URL 验证连续使用 bail!

```rust
// src/tools/web_fetch.rs
if url_str.is_empty() {
    anyhow::bail!("URL cannot be empty");
}
if !url_str.starts_with("http://") && !url_str.starts_with("https://") {
    anyhow::bail!("Only http:// and https:// URLs are allowed");
}
```

```typescript
// TypeScript 等价：提前 throw
if (url === "") throw new Error("URL cannot be empty");
if (!url.startsWith("http://") && !url.startsWith("https://")) {
  throw new Error("Only http:// and https:// URLs are allowed");
}
```

> **双重类比**
> - **前端类比**：`bail!` 就像 Express 中间件里的 `return res.status(400).json({ error: "..." })`——验证不通过就立刻返回，不再往下执行。
> - **日常生活类比**：`bail!` 就像安检——不合格直接拦住，不让你进入后续流程。

[来源: reference/context7_anyhow_01.md - bail! 宏]
[来源: reference/source_error_handling_01.md - tools/web_fetch.rs 分析]

### 3. ensure!() 宏 — 条件断言

`ensure!(condition, "msg")` 等价于 `if !condition { bail!("msg") }`。更简洁的条件检查。

```rust
use anyhow::{ensure, Result};

fn process_data(data: &[u8], depth: usize) -> Result<Vec<u8>> {
    ensure!(!data.is_empty(), "input data cannot be empty");
    ensure!(depth < 100, "recursion limit exceeded at depth {}", depth);
    Ok(data.to_vec())
}
```

**ZeroClaw 实战**：加密数据长度校验

```rust
// src/security/secrets.rs
anyhow::ensure!(blob.len() > NONCE_LEN, "Encrypted value too short");
// 等价于：
// if blob.len() <= NONCE_LEN {
//     anyhow::bail!("Encrypted value too short");
// }
```

```typescript
// TypeScript 等价：assert 函数
function ensure(condition: boolean, msg: string): asserts condition {
  if (!condition) throw new Error(msg);
}
ensure(blob.length > NONCE_LEN, "Encrypted value too short");
```

**bail! vs ensure! 选择**：
- 条件简单、一行能写完 → `ensure!`
- 条件复杂、需要 if/else 逻辑 → `if ... { bail!(...) }`

[来源: reference/context7_anyhow_01.md - ensure! 宏]
[来源: reference/source_error_handling_01.md - security/secrets.rs 分析]

### 4. .context() / .with_context() — 错误上下文注入

给错误附加一层可读的上下文描述，同时保留原始错误。**这是 anyhow 最强大的特性。**

```rust
use anyhow::{Context, Result};
use std::fs;

fn read_config(path: &str) -> Result<String> {
    fs::read_to_string(path)
        .context("Failed to read config file")?
    // 错误信息变成：
    // "Failed to read config file"
    //   Caused by: No such file or directory (os error 2)
}
```

**ZeroClaw 实战**：

```rust
// src/memory/sqlite.rs
Connection::open(&path_buf)
    .context("SQLite failed to open database")?;

// src/security/secrets.rs
fs::read_to_string(&self.key_path)
    .context("Failed to read secret key file")?;
```

**`.context()` vs `.with_context()`**：

```rust
// .context()：固定字符串（大多数场景够用）
file.read().context("read failed")?;

// .with_context()：需要格式化时用闭包（惰性求值）
file.read().with_context(|| format!("Failed to read {}", path.display()))?;
```

```typescript
// TypeScript 等价：Error cause（ES2022）
try {
  fs.readFileSync(path);
} catch (e) {
  throw new Error("Failed to read config file", { cause: e });
}
```

> **双重类比**
> - **前端类比**：`.context()` 就像给 HTTP 错误加一层业务描述——底层是 `500 Internal Server Error`，你包装成 `"Failed to load user profile"`，前端展示更友好，但开发者仍能看到原始错误。
> - **日常生活类比**：`.context()` 就像快递追踪——包裹丢了（原始错误），物流系统告诉你"在XX分拣中心丢失"（上下文），你知道去哪里找。

[来源: reference/context7_anyhow_01.md - Context trait]
[来源: reference/source_error_handling_01.md - memory/sqlite.rs 分析]

---

## 错误链

anyhow 的错误是链式的——每次 `.context()` 都会在链上加一层。你可以遍历整条链来调试。

```rust
fn find_root_cause(error: &anyhow::Error) {
    // chain() 返回错误链的迭代器，从最外层到最内层
    for (i, cause) in error.chain().enumerate() {
        println!("  {}: {}", i, cause);
    }
}

// 输出示例：
//   0: Configuration loading failed
//   1: Failed to read config from /etc/app/config.json
//   2: No such file or directory (os error 2)
```

### downcast_ref() — 类型检查

检查错误链中是否包含某种特定类型的错误（类似 TypeScript 的 `error instanceof TypeError`）：

```rust
use std::io;

fn is_not_found(error: &anyhow::Error) -> bool {
    for cause in error.chain() {
        if let Some(io_err) = cause.downcast_ref::<io::Error>() {
            return io_err.kind() == io::ErrorKind::NotFound;
        }
    }
    false
}
```

[来源: reference/context7_anyhow_01.md - 错误链遍历]

---

## ZeroClaw 中的 anyhow 六大模式

从 ZeroClaw 源码中提取的六种 anyhow 使用模式，覆盖了日常开发 95% 的场景：

### 模式 A：? 直接传播

最简单的模式——错误自动转换为 `anyhow::Error` 并传播。

```rust
// src/security/secrets.rs
let key_bytes = self.load_or_create_key()?;
```

**适用场景**：错误信息已经足够清晰，不需要额外上下文。

### 模式 B：.context() 添加上下文

给错误附加业务描述，保留原始错误。

```rust
// src/memory/sqlite.rs
Connection::open(&path_buf).context("SQLite failed to open database")?;
```

**适用场景**：底层错误信息太技术化（如 "os error 2"），需要加一层人类可读的描述。

### 模式 C：bail! 提前返回

主动判断条件，不满足就返回错误。

```rust
// src/tools/web_fetch.rs
if url_str.is_empty() {
    anyhow::bail!("URL cannot be empty");
}
```

**适用场景**：输入验证、前置条件检查。

### 模式 D：ensure! 条件断言

`bail!` 的简写形式。

```rust
// src/security/secrets.rs
anyhow::ensure!(blob.len() > NONCE_LEN, "Encrypted value too short");
```

**适用场景**：简单的布尔条件检查。

### 模式 E：map_err 错误转换

手动把非标准错误转换为 `anyhow::Error`。

```rust
// src/security/secrets.rs
cipher.encrypt(&nonce, plaintext.as_bytes())
    .map_err(|e| anyhow::anyhow!("Encryption failed: {e}"))?;
```

**适用场景**：第三方库的错误类型没有实现 `std::error::Error`，`?` 无法自动转换。

### 模式 F：Option → Result 转换

把 `None` 变成带错误信息的 `Err`。

```rust
// src/tools/shell.rs
let command = args.get("command")
    .and_then(|v| v.as_str())
    .ok_or_else(|| anyhow::anyhow!("Missing 'command' parameter"))?;
```

**适用场景**：从 JSON/HashMap 中提取必需参数。

### 六大模式速查

```
A: result?                              直接传播
B: result.context("描述")?              添加上下文后传播
C: bail!("原因")                        主动返回错误
D: ensure!(条件, "原因")                条件不满足则返回错误
E: result.map_err(|e| anyhow!("..."))?  手动转换错误类型
F: option.ok_or_else(|| anyhow!("..."))?  Option → Result
```

[来源: reference/source_error_handling_01.md - 错误处理模式分类]

---

## anyhow vs 标准 Result：何时用哪个？

| 场景 | 用 anyhow | 用标准 Result<T, E> | 理由 |
|------|:---------:|:-------------------:|------|
| 应用程序（二进制） | **推荐** | | 不需要让调用者区分错误类型 |
| 库（给别人用的 crate） | | **推荐** | 调用者需要 match 具体错误类型 |
| 原型/快速开发 | **推荐** | | 减少样板代码，快速迭代 |
| 需要按错误类型恢复 | | **推荐** | `match err { IoError => retry, ... }` |
| ZeroClaw trait 方法 | **推荐** | | 项目约定：统一 anyhow |
| 内部模块边界 | 都可以 | 都可以 | 看团队约定 |

**Rust 社区共识（2025-2026）**：
- **应用层**（main、handler、tool）→ anyhow
- **库层**（给别人用的 crate）→ thiserror 定义具体错误类型
- **混合模式**：内部用 thiserror 定义错误枚举，顶层用 anyhow 包装

ZeroClaw 是应用程序，所以全面使用 anyhow。

[来源: reference/search_error_handling_01.md - 2025-2026 最佳实践共识]

---

## TypeScript 对照

| anyhow 概念 | TypeScript 等价 | 说明 |
|-------------|----------------|------|
| `anyhow::Error` | `Error` 类 | 统一错误容器 |
| `anyhow::Result<T>` | `Promise<T>` 或 `Result<T, Error>` | 可能失败的返回值 |
| `anyhow!("msg")` | `new Error("msg")` | 创建错误 |
| `bail!("msg")` | `throw new Error("msg")` | 提前返回错误 |
| `ensure!(cond, "msg")` | `assert(cond, "msg")` | 条件断言 |
| `.context("msg")` | `new Error("msg", { cause: e })` | 错误链（ES2022 cause） |
| `error.chain()` | 遍历 `error.cause` 链 | 错误链遍历 |
| `error.downcast_ref::<T>()` | `error instanceof TypeError` | 类型检查 |

**核心区别**：TypeScript 的 Error 是运行时的、无类型约束的；anyhow::Error 是编译时强制处理的、类型安全的。anyhow 在保持 Rust 类型安全的前提下，提供了接近 TypeScript 的灵活性。

---

## 小结

### 速查卡

```
依赖：
  anyhow = "1.0"                    # Cargo.toml

导入：
  use anyhow::{Result, anyhow, bail, ensure, Context};

函数签名：
  fn do_something() -> Result<T>    # 等价于 Result<T, anyhow::Error>

四大 API：
  anyhow!("msg {}", x)             创建临时错误
  bail!("msg {}", x)               return Err(anyhow!("msg"))
  ensure!(cond, "msg")             if !cond { bail!("msg") }
  expr.context("msg")?             给错误附加上下文后传播

错误链：
  error.chain()                    遍历错误链
  error.downcast_ref::<T>()        检查是否包含特定错误类型
```

### ZeroClaw 关键认知

| 要点 | 说明 |
|------|------|
| 为什么用 anyhow | 多种错误类型统一处理，减少样板代码 |
| 返回类型 | 所有 trait 方法统一 `anyhow::Result<T>` |
| 最常用 API | `.context()` 和 `?` 的组合 |
| 参数提取 | `.ok_or_else(\|\| anyhow!("..."))` 把 Option 转 Result |
| 输入验证 | `bail!` / `ensure!` 提前返回 |
| 非标准错误 | `.map_err(\|e\| anyhow!("..."))` 手动转换 |
| 设计原则 | 零 panic + 上下文丰富 + 安全优先 |

**记住**：anyhow 让你专注于业务逻辑而不是错误类型定义。在 ZeroClaw 中，看到 `-> Result<T>` 就知道是 `anyhow::Result<T>`，看到 `.context("...")?` 就知道是"给错误加描述然后传播"。

[来源: reference/source_error_handling_01.md, reference/context7_anyhow_01.md, reference/search_error_handling_01.md]

---

*上一篇：[03_核心概念_4_组合子方法](./03_核心概念_4_组合子方法.md) -- map/and_then/unwrap_or/ok_or_else 链式操作*
*下一篇：[03_核心概念_6_thiserror自定义错误](./03_核心概念_6_thiserror自定义错误.md) -- 库层类型化错误定义*
