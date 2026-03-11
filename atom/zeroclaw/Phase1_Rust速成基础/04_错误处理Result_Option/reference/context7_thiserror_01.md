---
type: context7_documentation
library: thiserror
version: "2.0"
fetched_at: 2026-03-10
knowledge_point: 04_错误处理Result_Option
context7_query: "thiserror derive Error custom error types from transparent"
---

# Context7 文档：thiserror

## 文档来源
- 库名称：thiserror
- 版本：2.0.18
- Context7 ID: /dtolnay/thiserror

## 关键信息提取

### 1. 基本用法 - derive Error
使用 #[derive(Error)] 自动实现 std::error::Error trait。

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DataError {
    #[error("validation failed: {0}")]
    Validation(String),
    #[error("not found: {id}")]
    NotFound { id: u64 },
    #[error("I/O error")]
    Io(#[from] std::io::Error),  // #[from] 自动实现 From trait
}
```

### 2. #[error(transparent)] - 透明错误委托
将 Display 和 source 方法直接委托给底层错误，不添加额外消息。

```rust
#[derive(Error, Debug)]
pub enum MyError {
    #[error("custom validation error")]
    Validation,
    #[error(transparent)]
    Other(#[from] anyhow::Error),  // Display 和 source 委托给 anyhow::Error
}
```

### 3. #[source] 属性 - 错误源追踪
标记底层错误原因，允许调用者通过 Error trait 检查错误源。

```rust
#[derive(Error, Debug)]
#[error("database query failed: {msg}")]
pub struct DatabaseError {
    msg: String,
    #[source]
    source: anyhow::Error,
}
```

### 4. 不透明公共错误类型 - API 稳定性
创建不透明的公共错误结构体，内部可自由变更而不破坏公共 API。

```rust
#[derive(Error, Debug)]
#[error(transparent)]
pub struct PublicError(#[from] ErrorRepr);

#[derive(Error, Debug)]
enum ErrorRepr {
    #[error("non-critical error")]
    NonCritical,
    #[error("critical system failure")]
    Critical,
}
```

### 5. thiserror vs anyhow 使用场景
- **thiserror**：用于库代码，定义具体的错误类型枚举，调用者可以 match 处理
- **anyhow**：用于应用代码，类型擦除的灵活错误处理，附加上下文信息
- **组合使用**：内部用 thiserror 定义错误，顶层用 anyhow 包装
