# 核心概念 2：serde 高级属性与实战技巧

> **知识点**: 常用库速查（serde/anyhow/clap）
> **层级**: Phase1_Rust速成基础
> **维度**: 核心概念（serde 高级属性 — rename/alias/default/skip/flatten、serde_json 联用）
> **目标受众**: 有 TypeScript/前端经验但零 Rust 经验的开发者
> **前置阅读**: `03_核心概念_1_serde序列化框架.md`（三层属性体系、derive 基础）

---

## 一句话定位

> **serde 的高级属性让你用声明式注解精确控制 JSON/TOML 字段映射、默认值、条件跳过和结构扁平化——相当于把 zod 的 `.transform()` / `.default()` / `.optional()` 全部变成了编译时零开销的一行属性。**

---

## 1. rename 系列 — 字段/容器重命名

### 1.1 字段级重命名

当 Rust 字段名与 JSON/TOML 键名不一致时，用 `rename` 单独映射：

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct ApiUser {
    #[serde(rename = "userName")]       // Rust 字段 → JSON 键 "userName"
    user_name: String,

    #[serde(rename = "createdAt")]
    created_at: String,

    // 更精细：序列化和反序列化使用不同名
    #[serde(rename(serialize = "ID", deserialize = "id"))]
    user_id: u64,
}
```

**TS 对比**：TypeScript 没有原生字段重命名，你需要 `const toRust = (u) => ({ user_name: u.userName })` 手动映射。Rust 一个属性自动搞定。

### 1.2 容器级批量重命名

给整个 struct 设置命名风格转换规则：

```rust
#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]              // 所有字段 → camelCase
struct FrontendResponse {
    user_name: String,         // → "userName"
    created_at: String,        // → "createdAt"
    total_count: u32,          // → "totalCount"
}
```

**支持的命名风格**：

| 属性值 | `user_name` → | 使用场景 |
|--------|---------------|----------|
| `"camelCase"` | `"userName"` | JS/前端 API |
| `"PascalCase"` | `"UserName"` | C# / .NET API |
| `"snake_case"` | `"user_name"` | Rust / Python |
| `"SCREAMING_SNAKE_CASE"` | `"USER_NAME"` | 环境变量 / 常量 |
| `"kebab-case"` | `"user-name"` | HTTP 头 / CSS |

**前端类比**：像 CSS 的 `text-transform: uppercase` 一键转换大小写，`rename_all` 一键转换所有字段的命名风格。

### 1.3 字段属性覆盖容器属性

```rust
#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]          // 全部 camelCase
struct Config {
    user_name: String,                       // → "userName" ✅ 走容器规则
    #[serde(rename = "ID")]                  // 字段属性覆盖容器属性
    user_id: u32,                            // → "ID"（不是 "userId"）
}
```

**优先级**：字段属性 > 变体属性 > 容器属性，跟 CSS 的特异性（specificity）一个道理。

---

## 2. alias — 向后兼容的别名

`alias` 只在**反序列化**时生效：接受旧字段名，序列化时仍用原名。

```rust
#[derive(Serialize, Deserialize, Debug)]
struct AppConfig {
    #[serde(alias = "model_provider")]   // 旧名 "model_provider" 也能识别
    pub default_provider: Option<String>,

    #[serde(alias = "model")]            // 旧名 "model" 也能识别
    pub default_model: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 旧版配置 → 能解析
    let old: AppConfig = toml::from_str(r#"model_provider = "openai"
model = "gpt-4""#)?;

    // 新版配置 → 同样能解析
    let new: AppConfig = toml::from_str(r#"default_provider = "anthropic"
default_model = "claude-3-sonnet""#)?;

    println!("old: {:?}\nnew: {:?}", old, new);
    Ok(())
}
```

**ZeroClaw 实例**：ZeroClaw 的 `Config` 使用 `alias` 让新旧版本的 `config.toml` 都能正确解析——用户升级后不需要改配置文件 [^source1]。

**TS 对比**：TS 需要 `raw.default_provider ?? raw.model_provider` 手动兼容，Rust 一行 `alias` 搞定。

---

## 3. default — 缺失字段的默认值

### 3.1 使用 Default trait

```rust
#[derive(Serialize, Deserialize, Debug)]
struct ServerConfig {
    #[serde(default)]     // 缺失 → bool::default() = false
    pub enable_ssl: bool,
    #[serde(default)]     // 缺失 → u16::default() = 0
    pub port: u16,
    #[serde(default)]     // 缺失 → Vec::default() = vec![]
    pub allowed_origins: Vec<String>,
    #[serde(default)]     // 缺失 → Option::default() = None
    pub api_key: Option<String>,
}
```

### 3.2 自定义默认值函数

当 `Default` trait 的值不满足需求时，指定自定义函数：

```rust
fn default_port() -> u16 { 8080 }
fn default_timeout() -> u64 { 30 }

#[derive(Deserialize, Debug)]
struct ServerConfig {
    #[serde(default = "default_port")]      // 缺失 → 8080
    pub port: u16,
    #[serde(default = "default_timeout")]   // 缺失 → 30
    pub timeout_secs: u64,
}
```

### 3.3 容器级 default

给整个 struct 设置——所有字段缺失时都用 Default trait：

```rust
#[derive(Deserialize, Debug, Default)]        // 需要 derive Default
#[serde(default)]                              // 容器级：所有字段都有默认值
struct SecurityConfig {
    pub enable_sandbox: bool,                  // 缺失 → false
    pub timeout_secs: u64,                     // 缺失 → 0
    pub allowed_commands: Vec<String>,         // 缺失 → vec![]
}
```

**ZeroClaw 实例**：Config 的 20+ 子配置段全部使用 `#[serde(default)]`。用户的 `config.toml` 只需写关心的配置项，其余用默认值。新增配置项也不会破坏旧配置文件 [^source1]。

**TS 对比**：zod 用 `.default(8080)` 在运行时设默认值；Rust serde 在编译时生成默认值代码，零运行时开销。

---

## 4. skip 系列 — 控制序列化方向

### 4.1 四种 skip 模式

```rust
#[derive(Serialize, Deserialize, Debug, Default)]
struct UserSession {
    pub username: String,
    #[serde(skip)]                          // 完全跳过（反序列化时用 Default）
    pub internal_cache: Vec<String>,
    #[serde(skip_serializing)]              // 只反序列化，不序列化
    pub password_hash: String,
    #[serde(skip_deserializing)]            // 只序列化，不反序列化
    pub server_timestamp: u64,
    #[serde(skip_serializing_if = "Option::is_none")]  // 条件跳过
    pub bio: Option<String>,                // None 时 JSON 中不输出此字段
}
```

### 4.2 skip_serializing_if 常用条件

```rust
#[derive(Serialize, Debug)]
struct CleanJson {
    #[serde(skip_serializing_if = "Option::is_none")]  // 最常用
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]    // 空数组不输出
    pub tags: Vec<String>,
    #[serde(skip_serializing_if = "String::is_empty")] // 空字符串不输出
    pub description: String,
    #[serde(skip_serializing_if = "is_default_temp")]   // 自定义函数
    pub temperature: f64,
}
fn is_default_temp(t: &f64) -> bool { *t == 0.7 }
```

**ZeroClaw 实例**：`Config.workspace_dir` 用 `#[serde(skip)]`——工作目录是运行时计算的，不应存储到配置文件 [^source1]。

**TS 对比**：TS 需要 `Object.fromEntries(Object.entries(obj).filter(...))` 手动过滤，Rust 声明式搞定。

---

## 5. flatten — 结构扁平化

### 5.1 展开嵌套结构

`flatten` 把嵌套 struct 的字段"展开"到外层——就像 JavaScript 的 `...spread`：

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct Pagination { page: u32, per_page: u32 }

#[derive(Serialize, Deserialize, Debug)]
struct UserListRequest {
    query: String,
    #[serde(flatten)]            // 展开 Pagination 的字段到这一层
    pagination: Pagination,
}
// 序列化结果：{"query":"rust","page":1,"per_page":20}  ← 平级，不是嵌套
// 前端类比：const req = { query: "rust", ...pagination };
```

### 5.2 收集未知字段

`flatten` + `HashMap` 把未定义的字段全部收集起来——就像 JS 的 `...rest` 解构：

```rust
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use serde_json::Value;

#[derive(Serialize, Deserialize, Debug)]
struct ApiResponse {
    status: String,
    message: String,
    #[serde(flatten)]
    extra: HashMap<String, Value>,   // 剩余字段全收集
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let json = r#"{"status":"ok","message":"success","request_id":"abc","ts":123}"#;
    let resp: ApiResponse = serde_json::from_str(json)?;
    println!("extra: {:?}", resp.extra);
    // {"request_id": String("abc"), "ts": Number(123)}
    Ok(())
}
// 前端类比：const { status, message, ...extra } = response;
```

---

## 6. 与 serde_json 联用

### 6.1 json! 宏 — 快速构建 JSON

像写 JavaScript 对象字面量一样构建 JSON：

```rust
use serde_json::json;

fn main() {
    let model = "claude-3-sonnet";
    let temp = 0.8;

    let request = json!({
        "model": model,            // 变量直接引用
        "temperature": temp,       // 自动类型转换
        "stream": true,
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"}
        ]
    });
    println!("{}", serde_json::to_string_pretty(&request).unwrap());
}
```

**ZeroClaw 实例**：`tools/` 模块大量使用 `json!` 构建工具定义和调用参数 [^source1]。

### 6.2 serde_json::Value — 动态 JSON

JSON 结构不确定时，用 `Value` 类型（类似 TS 的 `Record<string, any>`）：

```rust
use serde_json::Value;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data: Value = serde_json::from_str(r#"{"name":"Alice","age":30}"#)?;
    println!("{}", data["name"]);                              // "Alice"
    println!("{}", data["missing"]);                           // null（不 panic）
    let name: &str = data["name"].as_str().unwrap_or("?");     // 安全转换
    let age: u64 = data["age"].as_u64().unwrap_or(0);
    println!("name={}, age={}", name, age);
    Ok(())
}
```

### 6.3 基本序列化/反序列化 API

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct Message { role: String, content: String }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let msg = Message { role: "user".into(), content: "Hi".into() };

    // struct → JSON
    let json = serde_json::to_string(&msg)?;           // 紧凑
    let pretty = serde_json::to_string_pretty(&msg)?;  // 美化

    // JSON → struct
    let parsed: Message = serde_json::from_str(&json)?;

    // Value ↔ struct
    let val = serde_json::to_value(&msg)?;              // struct → Value
    let msg2: Message = serde_json::from_value(val)?;   // Value → struct
    println!("{:?}", msg2);
    Ok(())
}
```

---

## 7. 实战速查表

### 7.1 字段属性速查

| 属性 | 效果 | 典型场景 |
|------|------|----------|
| `rename = "x"` | 重命名单个字段 | API 字段名不同 |
| `alias = "x"` | 反序列化时接受别名 | 向后兼容旧配置 |
| `default` | 缺失时用 Default trait | 可选配置项 |
| `default = "fn"` | 缺失时调用自定义函数 | 非零默认值 |
| `skip` | 完全不参与序列化 | 运行时计算字段 |
| `skip_serializing` | 只反序列化 | 密码等敏感字段 |
| `skip_serializing_if` | 条件跳过序列化 | 输出干净 JSON |
| `flatten` | 展开嵌套 / 收集未知字段 | 结构复用 |

### 7.2 容器属性速查

| 属性 | 效果 | 典型场景 |
|------|------|----------|
| `rename_all = "camelCase"` | 全部字段批量重命名 | 对接前端 API |
| `deny_unknown_fields` | 拒绝未知字段 | 严格配置校验 |
| `default` | 所有字段用 Default | 宽松解析 |
| `tag = "type"` | enum 内部标签 | API 多态响应 |
| `untagged` | enum 无标签匹配 | 灵活 API 响应 |

### 7.3 serde_json API 速查

| Rust API | 作用 | TS 对应 |
|----------|------|---------|
| `serde_json::to_string(&v)?` | struct → JSON | `JSON.stringify(obj)` |
| `serde_json::to_string_pretty(&v)?` | struct → 美化 JSON | `JSON.stringify(obj, null, 2)` |
| `serde_json::from_str::<T>(s)?` | JSON → struct | `JSON.parse(s) as T` |
| `serde_json::from_value::<T>(v)?` | Value → struct | 类型断言 |
| `serde_json::to_value(&v)?` | struct → Value | — |
| `serde_json::json!({...})` | 构建 JSON 值 | 对象字面量 `{...}` |

---

## 8. ZeroClaw 中的真实使用总结

### 8.1 配置系统（config/schema.rs）[^source1]

```
属性                    使用位置                     目的
──────────────────────────────────────────────────────────────────
#[serde(skip)]          Config.workspace_dir         运行时计算，不存储到文件
#[serde(alias)]         Config.default_provider      兼容旧版 "model_provider" 键名
#[serde(alias)]         Config.default_model         兼容旧版 "model" 键名
#[serde(default)]       20+ 子配置段                  用户只需写关心的配置项
Default trait           SecurityConfig 等             为子结构提供完整默认值
```

### 8.2 工具参数构建（tools/ 模块）

```
技术                    使用位置                     目的
──────────────────────────────────────────────────────────────────
json! 宏                工具定义构建                  灵活构建动态 JSON 结构
serde_json::Value       工具参数接收                  处理不确定结构的参数
from_str / to_string    API 请求/响应                 JSON ↔ struct 转换
```

### 8.3 设计原则

1. **渐进式配置**：`#[serde(default)]` 让用户只写关心的配置项
2. **向后兼容**：`#[serde(alias)]` 让旧配置文件无缝升级
3. **干净输出**：`#[serde(skip_serializing_if)]` 避免输出无意义的 null 字段
4. **关注点分离**：`#[serde(skip)]` 分离序列化数据与运行时状态
5. **灵活构建**：`json!` 宏在结构不确定时替代 struct 定义

---

## 速查卡

```
serde 高级属性速记：

  重命名：
    字段级:  #[serde(rename = "apiName")]         单个字段
    容器级:  #[serde(rename_all = "camelCase")]    批量转换
    优先级:  字段 > 变体 > 容器

  别名（只影响反序列化）：
    #[serde(alias = "old_name")]                  向后兼容

  默认值（只影响反序列化）：
    #[serde(default)]                             用 Default trait
    #[serde(default = "custom_fn")]               自定义函数

  跳过：
    #[serde(skip)]                                完全跳过
    #[serde(skip_serializing_if = "Option::is_none")]  条件跳过

  扁平化：
    #[serde(flatten)]                             展开嵌套 = JS ...spread
    #[serde(flatten)] + HashMap                   收集未知字段 = JS ...rest

  serde_json 三件套：
    json!({...})                    像 JS 字面量构建 JSON
    serde_json::from_str::<T>(s)?   JSON → struct
    serde_json::to_string(&v)?      struct → JSON
```

---

> **下一篇**: 阅读 `03_核心概念_3_anyhow错误处理.md`，学习 Rust 生态最流行的错误处理库——`anyhow::Result`、`.context()` 错误链、`bail!` / `ensure!` 宏。

---

**参考来源**

[^source1]: ZeroClaw 源码分析 — `reference/source_常用库_01.md`
[^context7_serde1]: serde 官方文档 — `reference/context7_serde_01.md`

---

**文件信息**
- 知识点: 常用库速查（serde/anyhow/clap）
- 维度: 03_核心概念_2_serde高级属性
- 版本: v1.0
- 日期: 2026-03-11
