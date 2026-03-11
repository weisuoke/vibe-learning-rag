---
type: source_code_analysis
source: sourcecode/zeroclaw
analyzed_files:
  - src/tools/traits.rs
  - src/tools/shell.rs
  - src/tools/web_fetch.rs
  - src/memory/traits.rs
  - src/memory/sqlite.rs
  - src/security/secrets.rs
  - src/gateway/api.rs
  - src/channels/traits.rs
  - src/config/schema.rs
  - src/lib.rs
  - Cargo.toml
analyzed_at: 2026-03-10
knowledge_point: 04_错误处理Result_Option
---

# 源码分析：ZeroClaw 错误处理模式

## 分析的文件

- `Cargo.toml` - 错误处理依赖：anyhow = "1.0", thiserror = "2.0"
- `src/tools/traits.rs` - Tool trait 定义，execute 返回 anyhow::Result<ToolResult>
- `src/tools/shell.rs` - Shell 工具，Option 组合子 + 模式匹配
- `src/tools/web_fetch.rs` - Web 抓取工具，bail! + validate + match Result
- `src/memory/traits.rs` - Memory trait，所有方法返回 anyhow::Result<T>
- `src/memory/sqlite.rs` - SQLite 实现，.context() 错误上下文
- `src/security/secrets.rs` - 加密解密，map_err + ensure! + context
- `src/gateway/api.rs` - HTTP API，自定义 Result 类型用于 HTTP 响应
- `src/channels/traits.rs` - Channel trait，默认实现返回 Ok(())
- `src/config/schema.rs` - 配置 schema，大量 Option<T> 字段
- `src/lib.rs` - CLI 入口，parse_temperature 返回 Result

## 关键发现

### 1. 统一错误类型：anyhow::Result<T>
所有 trait 方法统一使用 `anyhow::Result<T>` 作为返回类型，不使用自定义错误枚举。

### 2. 错误处理模式分类

#### A. ? 运算符传播（最常用）
```rust
// src/security/secrets.rs
let key_bytes = self.load_or_create_key()?;
```

#### B. .context() 添加上下文
```rust
// src/memory/sqlite.rs
Connection::open(&path_buf).context("SQLite failed to open database")?
```

#### C. bail! 提前返回
```rust
// src/tools/web_fetch.rs
anyhow::bail!("URL cannot be empty");
anyhow::bail!("Only http:// and https:// URLs are allowed");
```

#### D. ensure! 条件断言
```rust
// src/security/secrets.rs
anyhow::ensure!(blob.len() > NONCE_LEN, "Encrypted value too short");
```

#### E. map_err 错误转换
```rust
// src/security/secrets.rs
cipher.encrypt(&nonce, plaintext.as_bytes())
    .map_err(|e| anyhow::anyhow!("Encryption failed: {e}"))?;
```

#### F. Option 组合子链
```rust
// src/tools/shell.rs
let command = args.get("command")
    .and_then(|v| v.as_str())
    .ok_or_else(|| anyhow::anyhow!("Missing 'command' parameter"))?;
```

#### G. match 模式匹配 Result
```rust
// src/tools/web_fetch.rs
let url = match self.validate_url(url) {
    Ok(v) => v,
    Err(e) => return Ok(ToolResult { success: false, error: Some(e.to_string()), .. })
};
```

#### H. 嵌套 match（超时 + 执行结果）
```rust
// src/tools/shell.rs
match tokio::time::timeout(Duration::from_secs(SHELL_TIMEOUT_SECS), cmd.output()).await {
    Ok(Ok(output)) => { /* 成功 */ },
    Ok(Err(e)) => { /* 执行失败 */ },
    Err(_) => { /* 超时 */ },
}
```

### 3. Option<T> 使用场景
- 配置字段：`api_key: Option<String>`, `api_url: Option<String>`
- 内存条目：`session_id: Option<String>`, `score: Option<f64>`
- 安全默认值：`.unwrap_or(false)`, `.unwrap_or_default()`, `.unwrap_or("")`

### 4. 设计原则
- **零 panic**：生产代码不使用 unwrap()，仅测试中使用
- **安全优先**：错误信息不泄露敏感数据
- **上下文丰富**：每个错误都附带描述性上下文
- **Trait 一致性**：所有 trait 方法统一返回 anyhow::Result<T>
- **工具层包装**：Tool 执行错误包装在 ToolResult 中而非 panic
