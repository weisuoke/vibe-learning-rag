# 核心概念 6：thiserror 自定义错误

> **前置知识**：03_核心概念_1_Result枚举.md, 03_核心概念_3_问号运算符与错误传播.md, 03_核心概念_5_anyhow灵活错误处理.md
> **预计阅读**：20 分钟
> **难度**：进阶

---

## 一句话定义

**thiserror 是一个 derive 宏库，让你用几行注解就能定义完整的自定义错误类型——自动实现 Display、Error、From trait，省去大量样板代码。**

如果说 anyhow 是"我不关心错误具体是什么，统一装箱就好"，那 thiserror 就是"我要精确定义每种错误，让调用者能 match 处理"。

[来源: reference/context7_thiserror_01.md - thiserror vs anyhow 使用场景]

---

## 1. 为什么需要自定义错误类型

### 1.1 anyhow 的局限

anyhow 做了**类型擦除**——所有错误都变成 `anyhow::Error`，调用者无法用 match 区分具体错误：

```rust
use anyhow::Result;

fn load_config(path: &str) -> Result<Config> {
    let content = std::fs::read_to_string(path)?;  // io::Error → anyhow::Error
    let config: Config = serde_json::from_str(&content)?;  // serde::Error → anyhow::Error
    Ok(config)
}

// 调用者想区分"文件不存在"和"JSON 格式错误"？
// 用 anyhow 很难做到——错误类型被擦除了
match load_config("app.json") {
    Ok(c) => use_config(c),
    Err(e) => {
        // e 是 anyhow::Error，你只能打印它，不能 match 具体变体
        // 虽然可以 downcast，但很笨重
        eprintln!("Failed: {e}");
    }
}
```

**应用代码**（如 ZeroClaw 的 main 函数）这样做没问题——打印错误就够了。

**库代码**不行——调用者需要根据不同错误做不同处理。

```typescript
// TypeScript 类比：想象所有错误都是 Error 基类
// 调用者无法区分 NetworkError 和 ValidationError
try { loadConfig("app.json"); }
catch (e) {
    // e 是 unknown，你只能 console.error(e)
    // 无法 if (e instanceof NetworkError) { retry(); }
}
```

### 1.2 库代码需要具体错误类型

```rust
// 理想的库 API：调用者可以 match 每种错误
enum ConfigError {
    FileNotFound(String),
    ParseFailed(String),
    InvalidValue { key: String, reason: String },
}

match load_config("app.json") {
    Ok(c) => use_config(c),
    Err(ConfigError::FileNotFound(path)) => create_default_config(&path),
    Err(ConfigError::ParseFailed(msg)) => eprintln!("Fix your JSON: {msg}"),
    Err(ConfigError::InvalidValue { key, reason }) => {
        eprintln!("Config key '{key}' is invalid: {reason}");
    }
}
```

问题是：手动实现这个错误类型**非常繁琐**。

[来源: reference/search_error_handling_01.md - 库暴露具体错误类型（thiserror）]

---

## 2. 手动实现 Error trait（繁琐版）

要让一个类型成为合法的 Rust 错误，需要实现三样东西：

```rust
use std::fmt;

// 第一步：定义错误枚举
#[derive(Debug)]
enum ConfigError {
    FileNotFound(String),
    ParseFailed(String),
}

// 第二步：实现 Display trait（错误消息）
impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::FileNotFound(path) => write!(f, "file not found: {}", path),
            ConfigError::ParseFailed(msg) => write!(f, "parse failed: {}", msg),
        }
    }
}

// 第三步：实现 Error trait
impl std::error::Error for ConfigError {}

// 第四步：实现 From trait（让 ? 自动转换）
impl From<std::io::Error> for ConfigError {
    fn from(e: std::io::Error) -> Self {
        ConfigError::FileNotFound(e.to_string())
    }
}

impl From<serde_json::Error> for ConfigError {
    fn from(e: serde_json::Error) -> Self {
        ConfigError::ParseFailed(e.to_string())
    }
}
```

**两个变体就写了 30 多行样板代码。** 如果有 5 个变体、每个都需要 From 转换？代码量爆炸。

```typescript
// TypeScript 对照：手动定义 Error 子类也很繁琐
class ConfigError extends Error {
    constructor(message: string) { super(message); this.name = "ConfigError"; }
}
class FileNotFoundError extends ConfigError {
    constructor(public path: string) { super(`file not found: ${path}`); }
}
class ParseFailedError extends ConfigError {
    constructor(public detail: string) { super(`parse failed: ${detail}`); }
}
// 每个子类都要写 constructor + super + name...
```

**这就是 thiserror 要解决的问题：用 derive 宏消灭样板代码。**

[来源: reference/context7_thiserror_01.md - 基本用法]

---

## 3. thiserror derive 宏

### 3.1 基本用法：#[derive(Error, Debug)]

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("file not found: {0}")]
    FileNotFound(String),

    #[error("parse failed: {0}")]
    ParseFailed(String),
}
```

**就这么多。** thiserror 自动生成了：
- `impl Display for ConfigError`（根据 `#[error("...")]` 注解）
- `impl std::error::Error for ConfigError`

对比手动实现省了 20 行代码，而且每新增一个变体只需要 2 行。

### 3.2 #[error("...")] 格式化错误消息

```rust
#[derive(Error, Debug)]
pub enum ToolError {
    // {0} 引用元组结构体的第一个字段
    #[error("tool not found: {0}")]
    NotFound(String),

    // {name}, {reason} 引用命名字段
    #[error("tool '{name}' failed: {reason}")]
    ExecutionFailed { name: String, reason: String },

    // 可以调用方法
    #[error("invalid argument count: expected {expected}, got {actual}")]
    InvalidArgs { expected: usize, actual: usize },
}
```

```typescript
// TypeScript 对照：模板字符串
class ToolNotFoundError extends Error {
    constructor(name: string) {
        super(`tool not found: ${name}`);  // 类似 #[error("tool not found: {0}")]
    }
}
```

### 3.3 #[from] 自动实现 From trait

这是 thiserror 最省力的特性——让 `?` 运算符自动转换错误类型：

```rust
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("I/O error")]
    Io(#[from] std::io::Error),          // 自动: impl From<io::Error> for ConfigError

    #[error("JSON parse error")]
    Json(#[from] serde_json::Error),     // 自动: impl From<serde_json::Error> for ConfigError

    #[error("invalid value: {0}")]
    InvalidValue(String),                 // 无 #[from]，需要手动构造
}

// 现在 ? 可以自动转换了
fn load_config(path: &str) -> Result<Config, ConfigError> {
    let content = std::fs::read_to_string(path)?;       // io::Error → ConfigError::Io
    let config: Config = serde_json::from_str(&content)?; // serde::Error → ConfigError::Json
    Ok(config)
}
```

**没有 `#[from]` 时**，你需要手动写 `impl From<io::Error> for ConfigError { ... }`。有了 `#[from]`，一个注解搞定。

### 3.4 #[source] 标记错误源

`#[source]` 告诉 thiserror "这个字段是底层错误原因"，让 `Error::source()` 方法能返回它：

```rust
#[derive(Error, Debug)]
#[error("database query failed: {msg}")]
pub struct DatabaseError {
    msg: String,
    #[source]                    // Error::source() 会返回这个字段
    source: anyhow::Error,
}
```

**`#[from]` vs `#[source]` 的区别**：
- `#[from]` = `#[source]` + 自动实现 `From` trait（两件事）
- `#[source]` = 只标记错误源，不实现 `From`（一件事）

用 `#[source]` 的场景：你想保留错误链，但不想让 `?` 自动转换（比如需要额外字段）。

[来源: reference/context7_thiserror_01.md - #[source] 属性]

### 3.5 #[error(transparent)] 透明委托

把 Display 和 source 完全委托给内部错误，自己不添加任何消息：

```rust
#[derive(Error, Debug)]
pub enum MyError {
    #[error("validation failed: {0}")]
    Validation(String),

    #[error(transparent)]                    // Display 和 source 都委托给内部的 anyhow::Error
    Other(#[from] anyhow::Error),
}
```

**实际用途**：创建不透明的公共 API 错误类型，内部实现可以自由变更：

```rust
// 公共 API：调用者只看到 PublicError
#[derive(Error, Debug)]
#[error(transparent)]
pub struct PublicError(#[from] ErrorRepr);

// 内部实现：可以随时增删变体，不破坏公共 API
#[derive(Error, Debug)]
enum ErrorRepr {
    #[error("non-critical error")]
    NonCritical,
    #[error("critical system failure")]
    Critical,
}
```

```typescript
// TypeScript 对照：类似于只暴露基类，隐藏子类
// 公共 API
export class PublicError extends Error { /* 不暴露内部细节 */ }
// 内部实现
class NonCriticalError extends PublicError { }
class CriticalError extends PublicError { }
```

[来源: reference/context7_thiserror_01.md - 不透明公共错误类型]

---

## 4. 实战示例：完整的自定义错误枚举

假设你在写一个工具执行模块（类似 ZeroClaw 的 Tool 系统），需要区分不同的失败原因：

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ToolError {
    #[error("tool '{name}' not found")]
    NotFound { name: String },

    #[error("missing required parameter: {0}")]
    MissingParam(String),

    #[error("execution timed out after {seconds}s")]
    Timeout { seconds: u64 },

    #[error("I/O error in tool execution")]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

// 使用示例
fn execute_tool(name: &str, args: &serde_json::Value) -> Result<String, ToolError> {
    let tool = find_tool(name)
        .ok_or_else(|| ToolError::NotFound { name: name.to_string() })?;

    let input = args.get("input")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ToolError::MissingParam("input".to_string()))?;

    let result = std::fs::read_to_string(input)?;  // io::Error → ToolError::Io（自动）
    Ok(result)
}

// 调用者可以精确 match
match execute_tool("read_file", &args) {
    Ok(output) => println!("{output}"),
    Err(ToolError::NotFound { name }) => eprintln!("No such tool: {name}"),
    Err(ToolError::MissingParam(p)) => eprintln!("Please provide: {p}"),
    Err(ToolError::Timeout { seconds }) => eprintln!("Tool took too long ({seconds}s)"),
    Err(e) => eprintln!("Unexpected error: {e}"),
}
```

[来源: reference/context7_thiserror_01.md - 基本用法]

---

## 5. thiserror vs anyhow 选择指南

### 5.1 一句话原则

| 场景 | 选择 | 原因 |
|------|------|------|
| **库代码**（给别人用的 crate） | thiserror | 调用者需要 match 具体错误变体 |
| **应用代码**（你自己的 main） | anyhow | 灵活，不需要定义枚举，打印就够了 |
| **混合使用** | 内部 thiserror，顶层 anyhow | 内部精确，边界灵活 |

### 5.2 混合使用模式

这是 Rust 社区 2025-2026 年的最佳实践共识：

```rust
// 库层：用 thiserror 定义精确错误
// my_lib/src/error.rs
#[derive(thiserror::Error, Debug)]
pub enum LibError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("connection failed")]
    Connection(#[from] std::io::Error),
}

// 应用层：用 anyhow 包装一切
// my_app/src/main.rs
use anyhow::{Context, Result};

fn main() -> Result<()> {
    let result = my_lib::do_something()
        .context("library operation failed")?;  // LibError → anyhow::Error（自动）
    Ok(())
}
```

**为什么能自动转换？** 因为 thiserror 生成的类型实现了 `std::error::Error`，而 anyhow 对所有 `std::error::Error` 实现了 `From`。所以 `?` 可以把任何 thiserror 错误自动转为 `anyhow::Error`。

> **双重类比**
> - **前端类比**：thiserror 像 TypeScript 的 discriminated union（`type Result = { type: "success", data: T } | { type: "error", code: number }`），调用者可以 switch(result.type)。anyhow 像 `catch (e: unknown)`——什么都能接，但不能精确区分。
> - **日常生活类比**：thiserror 是医院的分诊系统——发烧去内科、骨折去骨科、皮肤问题去皮肤科，每种病有明确的科室。anyhow 是急诊——不管什么问题先送进来再说。

[来源: reference/search_error_handling_01.md - 混合模式：内部用 thiserror，顶层转换为 anyhow]

---

## 6. ZeroClaw 的选择

### 6.1 为什么 ZeroClaw 主要用 anyhow 而非 thiserror

ZeroClaw 的 Cargo.toml 同时依赖了 anyhow 和 thiserror：

```toml
[dependencies]
anyhow = "1.0"
thiserror = "2.0"
```

但实际代码中，**所有 trait 方法统一返回 `anyhow::Result<T>`**：

```rust
// Tool trait — anyhow::Result
async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult>;

// Memory trait — anyhow::Result
async fn store(&self, entry: MemoryEntry) -> anyhow::Result<()>;
async fn recall(&self, query: &str, limit: usize) -> anyhow::Result<Vec<MemoryEntry>>;

// Channel trait — anyhow::Result
async fn send(&self, message: &str) -> anyhow::Result<()>;
```

**原因**：

1. **ZeroClaw 是应用层代码**，不是给别人用的库。错误最终都是打印给用户看的，不需要调用者 match 具体变体。

2. **Trait 统一性**。如果 Tool trait 返回 `Result<ToolResult, ToolError>`，Memory trait 返回 `Result<T, MemoryError>`，Channel trait 返回 `Result<T, ChannelError>`——每个模块一种错误类型，组合起来极其痛苦。`anyhow::Result<T>` 统一了所有 trait 的错误类型。

3. **灵活性**。不同的 Tool 实现可能遇到完全不同的错误（Shell 工具有超时错误，Web 工具有网络错误，文件工具有 IO 错误）。用 anyhow 可以 `?` 传播任何错误，不需要在枚举里穷举所有可能。

### 6.2 什么时候 ZeroClaw 会用 thiserror

如果 ZeroClaw 将来把某个模块拆成独立 crate（比如 `zeroclaw-tools`），那个 crate 就应该用 thiserror 定义 `ToolError`，让 ZeroClaw 主程序可以 match 处理。

```rust
// 假设 zeroclaw-tools 独立成 crate
// zeroclaw-tools/src/error.rs
#[derive(thiserror::Error, Debug)]
pub enum ToolError {
    #[error("tool '{0}' not found")]
    NotFound(String),
    #[error("execution timed out")]
    Timeout,
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

// zeroclaw 主程序
fn handle_tool_result(result: Result<ToolResult, ToolError>) {
    match result {
        Ok(r) => display_result(r),
        Err(ToolError::Timeout) => retry_with_longer_timeout(),
        Err(ToolError::NotFound(name)) => suggest_similar_tools(&name),
        Err(e) => eprintln!("Tool error: {e}"),
    }
}
```

[来源: reference/source_error_handling_01.md - 统一错误类型：anyhow::Result<T>]

---

## 7. TypeScript 对照总结

### 7.1 自定义 Error 子类 vs thiserror 枚举

```typescript
// TypeScript：自定义 Error 子类
class AppError extends Error {
    constructor(message: string) { super(message); this.name = "AppError"; }
}
class NotFoundError extends AppError {
    constructor(public resource: string) {
        super(`not found: ${resource}`);
    }
}
class ValidationError extends AppError {
    constructor(public field: string, public reason: string) {
        super(`validation failed on '${field}': ${reason}`);
    }
}

// 使用：instanceof 判断
try { doSomething(); }
catch (e) {
    if (e instanceof NotFoundError) { /* 处理未找到 */ }
    else if (e instanceof ValidationError) { /* 处理验证失败 */ }
    else { /* 兜底 */ }
}
```

```rust
// Rust：thiserror 枚举
#[derive(thiserror::Error, Debug)]
pub enum AppError {
    #[error("not found: {resource}")]
    NotFound { resource: String },
    #[error("validation failed on '{field}': {reason}")]
    Validation { field: String, reason: String },
}

// 使用：match 模式匹配
match do_something() {
    Ok(v) => use_value(v),
    Err(AppError::NotFound { resource }) => { /* 处理未找到 */ }
    Err(AppError::Validation { field, reason }) => { /* 处理验证失败 */ }
}
```

### 7.2 核心差异

| 维度 | TypeScript Error 子类 | Rust thiserror 枚举 |
|------|----------------------|-------------------|
| 定义方式 | class extends Error | #[derive(Error)] enum |
| 判断方式 | instanceof（运行时） | match（编译时穷尽检查） |
| 遗漏检查 | 无（忘了 catch 也能编译） | 编译器警告未处理的变体 |
| 错误消息 | 手动 super(msg) | #[error("...")] 自动生成 |
| 自动转换 | 无 | #[from] 自动实现 From |
| 错误链 | cause 属性（可选） | #[source] 强类型链 |

**一句话：thiserror 把 TypeScript 中需要手写的 Error 子类 + instanceof 判断，变成了 derive 宏 + 编译器穷尽检查。更安全，更简洁。**

[来源: reference/search_error_handling_01.md - TypeScript 对比]

---

## 8. 速查卡

```
thiserror 核心注解：
  #[derive(Error, Debug)]           启用 derive 宏
  #[error("msg {0} {field}")]       格式化错误消息（自动实现 Display）
  #[from]                           自动实现 From trait（让 ? 自动转换）
  #[source]                         标记错误源（Error::source() 返回它）
  #[error(transparent)]             Display 和 source 委托给内部错误

选择指南：
  库代码（给别人用）  → thiserror（调用者能 match）
  应用代码（自己用）  → anyhow（灵活，打印就够）
  混合使用            → 内部 thiserror，顶层 anyhow
  ZeroClaw            → anyhow 为主（应用层 + trait 统一性）
```

---

*上一篇：[03_核心概念_5_anyhow灵活错误处理](./03_核心概念_5_anyhow灵活错误处理.md) -- 应用层错误处理*
*下一篇：[04_最小可用](./04_最小可用.md) -- 20% 核心知识解决 80% 问题*

[来源: reference/context7_thiserror_01.md, reference/search_error_handling_01.md]
