---
type: context7_documentation
library: anyhow
version: "1.0"
fetched_at: 2026-03-10
knowledge_point: 04_错误处理Result_Option
context7_query: "anyhow Result Context bail ensure map_err error handling Rust"
---

# Context7 文档：anyhow

## 文档来源
- 库名称：anyhow
- 版本：1.0.102
- Context7 ID: /dtolnay/anyhow

## 关键信息提取

### 1. bail! 宏 - 提前返回错误
等价于 `return Err(anyhow!(...))`, 支持字符串字面量、格式化字符串和自定义错误类型。

```rust
use anyhow::{bail, Result};

fn check_permissions(user: &str) -> Result<()> {
    if user != "admin" {
        bail!("insufficient permissions for user {}", user);
    }
    Ok(())
}
```

### 2. ensure! 宏 - 条件断言
条件为 false 时返回 Err，不会 panic。用于输入验证。

```rust
use anyhow::{ensure, Result};

fn process_data(data: &[u8], depth: usize) -> Result<Vec<u8>> {
    ensure!(depth < 100, "recursion limit exceeded at depth {}", depth);
    ensure!(!data.is_empty(), "input data cannot be empty");
    Ok(data.to_vec())
}
```

### 3. Context trait - 错误注释
使用 .context() 和 .with_context() 为错误附加可读的上下文信息，同时保留原始错误链。

```rust
use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

fn read_config(path: &Path) -> Result<String> {
    fs::read_to_string(path)
        .with_context(|| format!("Failed to read config from {}", path.display()))
}

fn load_config(path: &Path) -> Result<serde_json::Value> {
    let content = read_config(path).context("Configuration loading failed")?;
    serde_json::from_str(&content).context("Configuration parsing failed")
}
```

### 4. anyhow! 宏 - 创建临时错误
从字符串或 trait 实现构造临时错误，保留错误链。

```rust
use anyhow::{anyhow, Result};

fn validate_user_id(id: &str) -> Result<u64> {
    if id.is_empty() {
        return Err(anyhow!("user ID cannot be empty"));
    }
    id.parse::<u64>().map_err(|e| anyhow!("invalid user ID: {}", e))
}
```

### 5. 错误链遍历
使用 chain() 方法遍历错误链，访问根因和中间原因。

```rust
fn find_io_error_kind(error: &anyhow::Error) -> Option<io::ErrorKind> {
    for cause in error.chain() {
        if let Some(io_error) = cause.downcast_ref::<io::Error>() {
            return Some(io_error.kind());
        }
    }
    None
}
```
