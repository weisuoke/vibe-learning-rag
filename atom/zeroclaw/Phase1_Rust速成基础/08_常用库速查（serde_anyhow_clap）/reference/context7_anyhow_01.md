---
type: context7_documentation
library: anyhow
version: "1.0"
fetched_at: 2026-03-11
knowledge_point: 08_常用库速查（serde/anyhow/clap）
context7_query: Result Context bail error handling downcast
---

# Context7 文档：anyhow

## 文档来源
- 库名称：anyhow
- 版本：1.0
- 官方文档链接：https://github.com/dtolnay/anyhow

## 关键信息提取

### 1. 错误下转型 (Downcasting)
anyhow::Error 可以下转型到具体错误类型。

```rust
use anyhow::{Context, Result, anyhow};
use thiserror::Error;

#[derive(Error, Debug)]
#[error("database connection failed: {0}")]
struct DbError(String);

fn connect_db() -> Result<()> {
    Err(anyhow!(DbError("timeout after 30s".to_string())))
        .context("Failed to establish database connection")
}

fn handle_error(error: anyhow::Error) {
    if let Some(db_err) = error.downcast_ref::<DbError>() {
        eprintln!("Database error: {}", db_err);
        return;
    }
    // downcast by value consumes the error
    match error.downcast::<CacheError>() {
        Ok(cache_err) => eprintln!("Cache error: {}", cache_err),
        Err(e) => eprintln!("Unknown error: {}", e),
    }
}
```

### 2. 错误链遍历 (Error Chain)
使用 `chain()` 方法遍历错误链。

```rust
use anyhow::{Context, Result};

fn read_file(path: &str) -> Result<String> {
    let mut file = File::open(path)
        .with_context(|| format!("Failed to open file: {}", path))?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .context("Failed to read file contents")?;
    Ok(contents)
}

fn main() {
    match read_file("/nonexistent/file.txt") {
        Ok(contents) => println!("{}", contents),
        Err(e) => {
            eprintln!("Error: {}", e);
            for (i, cause) in e.chain().enumerate() {
                if i > 0 {
                    eprintln!("  Caused by [{}]: {}", i, cause);
                }
            }
        }
    }
}
```

### 3. 核心模式总结
- `anyhow::Result<T>` 作为所有可失败函数的返回类型
- `Context` trait 的 `.context()` 和 `.with_context()` 添加语义信息
- `main() -> Result<()>` 自动打印详细错误信息
- 应用层用 anyhow，库层用 thiserror
- 支持通过 downcast 恢复具体错误类型
