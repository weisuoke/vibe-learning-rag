---
type: context7_documentation
library: toml (Rust TOML serialization)
version: latest
fetched_at: 2026-03-11
knowledge_point: 10_Onboarding_Wizard与首次配置
context7_query: TOML serialize deserialize config file to_string_pretty
---

# Context7 文档：TOML (Rust TOML Serialization)

## 文档来源
- 库名称：toml
- 版本：latest
- Context7 ID: /websites/rs_toml

## 关键信息提取

### 1. 序列化（Struct → TOML String）

```rust
use serde::Serialize;

#[derive(Serialize)]
struct Config {
    ip: String,
    port: Option<u16>,
    keys: Keys,
}

#[derive(Serialize)]
struct Keys {
    github: String,
    travis: Option<String>,
}

let config = Config {
    ip: "127.0.0.1".to_string(),
    port: None,
    keys: Keys {
        github: "xxxxxxxxxxxxxxxxx".to_string(),
        travis: Some("yyyyyyyyyyyyyyyyy".to_string()),
    },
};

let toml = toml::to_string(&config).unwrap();
```

### 2. 反序列化（TOML String → Struct）

```rust
use serde::Deserialize;

#[derive(Deserialize)]
struct Config {
    ip: String,
    port: Option<u16>,
    keys: Keys,
}

let config: Config = toml::from_str(r#"
    ip = '127.0.0.1'
    [keys]
    github = 'xxxxxxxxxxxxxxxxx'
    travis = 'yyyyyyyyyyyyyyyyy'
"#).unwrap();
```

### 3. Pretty Print（格式化输出）

```rust
// 使用 to_string_pretty 获取更可读的 TOML 输出
let pretty_toml = toml::to_string_pretty(&config).unwrap();
```

ZeroClaw 使用 `toml::to_string_pretty()` 序列化 Config 结构体到 config.toml 文件。

### 4. 与 ZeroClaw Config 的关联

ZeroClaw 的 config 系统使用 serde + toml 实现：
- `Config` struct 使用 `#[derive(Serialize, Deserialize)]`
- 保存时调用 `toml::to_string_pretty(&self)`
- 加载时调用 `toml::from_str(&content)`
- 支持 Option 字段（如 `api_key: Option<String>`）
- 嵌套结构映射到 TOML sections
