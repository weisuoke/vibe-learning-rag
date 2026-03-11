---
type: context7_documentation
library: serde
version: "1.0"
fetched_at: 2026-03-11
knowledge_point: 08_常用库速查（serde/anyhow/clap）
context7_query: derive macros serialize deserialize attributes rename default skip
---

# Context7 文档：serde

## 文档来源
- 库名称：serde
- 版本：1.0
- 官方文档链接：https://serde.rs

## 关键信息提取

### 1. 字段属性 - skip
`#[serde(skip)]`, `#[serde(skip_serializing)]`, `#[serde(skip_deserializing)]` 控制字段是否参与序列化/反序列化。

```rust
#[derive(Serialize, Deserialize)]
struct UserProfile {
    username: String,
    #[serde(skip)]
    internal_id: u64,
    #[serde(skip_serializing)]
    password_hash: String,
    #[serde(skip_deserializing)]
    created_at: u64,
}
```

### 2. 字段重命名 - rename
使用 `#[serde(rename)]` 指定序列化/反序列化时的不同字段名。

```rust
#[derive(Serialize, Deserialize)]
struct MyStruct {
    #[serde(rename = "rustName")]
    rust_name: String,
    #[serde(rename(serialize = "serName", deserialize = "deName"))]
    field_name: String,
}
```

### 3. 默认值 - default
`#[serde(default)]` 在反序列化时为缺失字段提供默认值。

```rust
fn default_value() -> String {
    String::from("default string")
}

#[derive(Deserialize)]
struct Config {
    #[serde(default)]
    option_field: Option<String>,
    #[serde(default = "default_value")]
    custom_default: String,
}
```

### 4. 容器属性
```rust
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]  // 容器属性
struct S {
    #[serde(default)]  // 字段属性
    f: i32,
}

#[derive(Serialize, Deserialize)]
#[serde(rename = "e")]  // 容器属性
enum E {
    #[serde(rename = "a")]  // 变体属性
    A(String),
}
```

### 5. 三层属性体系
- **容器属性** (Container): 应用于 struct/enum 整体，如 `deny_unknown_fields`, `rename_all`
- **变体属性** (Variant): 应用于 enum 变体，如 `rename`, `skip`
- **字段属性** (Field): 应用于 struct 字段，如 `default`, `skip`, `rename`, `alias`
