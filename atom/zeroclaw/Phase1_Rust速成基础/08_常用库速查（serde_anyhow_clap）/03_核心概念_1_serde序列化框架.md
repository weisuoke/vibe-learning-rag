# 核心概念 1：serde 序列化框架

> **知识点**: 常用库速查（serde/anyhow/clap）
> **层级**: Phase1_Rust速成基础
> **维度**: 核心概念（serde 序列化框架 — derive 宏、三层属性体系、常用格式生态）
> **目标受众**: 有 TypeScript/前端经验但零 Rust 经验的开发者

---

## 一句话定义

> **serde（Serialize + Deserialize）是 Rust 的通用序列化框架——一行 `#[derive]` 让你的 struct 自动获得 JSON/TOML/YAML/Binary 等所有格式的读写能力，相当于 `JSON.parse` + `JSON.stringify` + `zod` 类型验证的编译时超级增强版。**

---

## 1. serde 是什么

### 1.1 名字的由来

**ser**ialize + **de**serialize = **serde**。它做两件事：

```
Serialize:   Rust 数据结构 → 目标格式（JSON/TOML/YAML/Binary...）
Deserialize: 目标格式 → Rust 数据结构
```

### 1.2 前端类比：JSON.parse + JSON.stringify + zod

```typescript
// TypeScript 世界：三个独立的工具
const json = JSON.stringify(user);       // 序列化（无类型检查）
const parsed = JSON.parse(json);         // 反序列化（返回 any！）
const validated = userSchema.parse(data); // zod 运行时验证

// 问题：
// 1. JSON.parse 返回 any，类型不安全
// 2. zod 验证在运行时，有性能开销
// 3. 三个工具各自独立，容易不一致
```

```rust
// Rust 世界：serde 一站式搞定
#[derive(Serialize, Deserialize)]
struct User {
    name: String,
    age: u32,
}

let json = serde_json::to_string(&user)?;          // 序列化（类型安全）
let parsed: User = serde_json::from_str(&json)?;   // 反序列化（编译时类型检查）
// 不需要 zod！编译器保证类型正确，反序列化失败返回 Err
```

**关键区别**：TypeScript 的类型在运行时消失（type erasure），而 Rust 的 serde 在**编译时**生成序列化代码——零运行时反射开销，类型永远安全 [^context7_serde1]。

### 1.3 serde 的设计哲学：数据模型解耦

serde 最精妙的设计是**将数据结构与数据格式解耦**：

```
你的 Rust struct ←→ serde 数据模型 ←→ 具体格式
                     （中间层）

   User                                JSON   (serde_json)
   Config          ←→  通用表示  ←→    TOML   (toml)
   ApiResponse                         YAML   (serde_yaml)
                                       Binary (bincode)
```

这意味着你只需要 derive 一次 `Serialize`/`Deserialize`，就自动获得**所有格式**的支持。不像 TypeScript 需要为每种格式写不同的解析逻辑。

---

## 2. derive 宏 — 核心魔法

### 2.1 基本使用

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct User {
    name: String,
    age: u32,
    email: Option<String>,  // Option<T> 自动处理 null/缺失
}
```

**TypeScript 对比**：

```typescript
// TypeScript 的 interface 只是类型标注，编译后消失
interface User {
    name: string;
    age: number;
    email?: string;  // 可选字段
}

// 你仍然需要手动写解析逻辑
const user: User = JSON.parse(jsonString); // ← 这里没有任何验证！
// 如果 jsonString 里 age 是 "25"（字符串），TypeScript 不会报错
// 但 Rust 的 serde 会在反序列化时返回 Err
```

```rust
// Rust 的 derive 自动生成完整的序列化/反序列化代码
// 编译器保证：如果 JSON 中 age 是字符串 "25"，反序列化会失败并返回错误
let user: User = serde_json::from_str(json_str)?;
// age 保证是 u32，email 保证是 Option<String>
```

### 2.2 derive 的前提条件

derive `Serialize`/`Deserialize` 时，**所有字段的类型都必须实现对应 trait**：

```rust
#[derive(Serialize, Deserialize)]
struct Agent {
    name: String,           // ✅ String 实现了 Serialize + Deserialize
    age: u32,               // ✅ u32 实现了
    tools: Vec<String>,     // ✅ Vec<T> 在 T 实现时也实现
    metadata: HashMap<String, serde_json::Value>,  // ✅ 都实现了
    // raw_ptr: *const u8,  // ❌ 原始指针没有实现 → 编译错误
}
```

**常见已支持的类型**：所有基本类型（`i32`, `f64`, `bool`, `String`）、`Vec<T>`, `HashMap<K,V>`, `Option<T>`, `Box<T>`, 元组等——只要内部类型支持，容器就自动支持。

### 2.3 Cargo.toml 配置

```toml
[dependencies]
# serde 核心 + derive 宏（必选）
serde = { version = "1.0", features = ["derive"] }

# 格式支持（按需选一个或多个）
serde_json = "1.0"   # JSON 格式
toml = "1.0"         # TOML 格式（配置文件常用）
```

**ZeroClaw 的实际配置** [^source1]：

```toml
serde = { version = "1.0", default-features = false, features = ["derive"] }
serde_json = { version = "1.0", default-features = false, features = ["std"] }
toml = "1.0"
```

注意 ZeroClaw 使用了 `default-features = false` 来减小编译体积——这是生产项目的常见做法。

---

## 3. 三层属性体系 — serde 最强大的设计

serde 通过 `#[serde(...)]` 属性来精细控制序列化/反序列化行为。属性分为三层，作用于不同级别 [^context7_serde1]：

### 3.1 概览

```
层级          作用对象               类比
──────────────────────────────────────────────────────────
容器属性      struct / enum 整体     CSS 的全局样式
变体属性      enum 的各个变体        CSS 的类选择器
字段属性      struct 的各个字段      CSS 的 ID 选择器
```

### 3.2 容器属性（Container Attributes）

**应用于 struct 或 enum 整体**，写在 `#[derive]` 下方：

```rust
#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]       // 所有字段名：snake_case → camelCase
#[serde(deny_unknown_fields)]           // 反序列化时拒绝未知字段
struct ApiRequest {
    user_name: String,      // JSON 中是 "userName"
    request_id: String,     // JSON 中是 "requestId"
    max_tokens: u32,        // JSON 中是 "maxTokens"
}
```

**TypeScript 类比**：

```typescript
// TypeScript 中你需要手动映射字段名
interface ApiRequest {
    userName: string;    // 前端用 camelCase
}
// 发送到后端时：{ user_name: req.userName }  ← 手动转换
// Rust serde：自动转换，一个属性搞定！
```

**常用容器属性速查**：

```
属性                              效果                           使用场景
──────────────────────────────────────────────────────────────────────────
#[serde(rename_all = "camelCase")]  全部字段转 camelCase          对接 JS/前端 API
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]  全部转大写蛇形    环境变量风格
#[serde(deny_unknown_fields)]       拒绝 JSON 中的未知字段       严格解析配置文件
#[serde(default)]                   所有字段使用 Default 默认值   宽松解析
#[serde(tag = "type")]              enum 使用内部标签             API 多态响应
#[serde(untagged)]                  enum 无标签（按结构匹配）     灵活 API 响应
```

### 3.3 变体属性（Variant Attributes）

**应用于 enum 的各个变体**：

```rust
#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]                  // 容器属性：内部标签
enum Message {
    #[serde(rename = "text")]           // 变体属性：重命名
    TextMessage { content: String },

    #[serde(rename = "image")]
    ImageMessage { url: String },

    #[serde(skip)]                      // 变体属性：跳过此变体
    Internal { data: Vec<u8> },
}

// 序列化结果：{"type": "text", "content": "hello"}
// 而不是：   {"type": "TextMessage", "content": "hello"}
```

### 3.4 字段属性（Field Attributes）

**应用于 struct 的各个字段**，这是最常用的属性：

```rust
#[derive(Serialize, Deserialize, Debug)]
struct Config {
    // rename — 重命名单个字段
    #[serde(rename = "api_key")]
    key: String,

    // alias — 反序列化时接受别名（兼容旧配置）
    #[serde(alias = "model_provider")]
    provider: String,

    // default — 缺失时使用默认值
    #[serde(default)]
    temperature: f64,                    // 缺失时为 0.0

    // default 指定自定义函数
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,

    // skip — 完全不参与序列化/反序列化
    #[serde(skip)]
    internal_cache: Vec<String>,

    // skip_serializing_if — 条件跳过
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,         // None 时 JSON 中不输出此字段
}

fn default_max_tokens() -> u32 {
    4096
}
```

**TypeScript 类比**：

```typescript
// TypeScript：需要 zod 或 class-transformer 来实现类似功能
const ConfigSchema = z.object({
    api_key: z.string(),                           // rename: 无直接等价物
    provider: z.string(),                          // alias: 无直接等价物
    temperature: z.number().default(0.0),           // default
    max_tokens: z.number().default(4096),            // default 自定义
    // skip: 无等价物（TypeScript 没有跳过序列化的概念）
    description: z.string().optional(),              // skip_serializing_if
});

// Rust 一行属性就搞定了 zod 需要额外库+运行时的所有功能
```

### 3.5 三层属性的优先级

```
字段属性 > 变体属性 > 容器属性

示例：
#[serde(rename_all = "camelCase")]   // 容器：全部 camelCase
struct Config {
    user_name: String,                // → "userName"（容器规则）

    #[serde(rename = "ID")]           // 字段属性覆盖容器属性
    user_id: u32,                     // → "ID"（不是 "userId"）
}
```

---

## 4. 常用格式生态

serde 的数据模型解耦设计催生了丰富的格式生态：

### 4.1 格式对照表

```
格式      Crate         用途              前端类比                  特点
───────────────────────────────────────────────────────────────────────────
JSON      serde_json    API 交互          JSON.parse/stringify     最通用
TOML      toml          配置文件          dotenv + yaml-loader     Rust 社区首选配置格式
YAML      serde_yaml    配置文件          js-yaml                  层级复杂配置
Binary    bincode       高性能传输        protobuf                 最快，但不可读
MessagePack rmp-serde   紧凑传输          msgpack                  比 JSON 小，比 bincode 灵活
```

### 4.2 ZeroClaw 中的格式选择 [^source1]

```
用途                   格式      原因
───────────────────────────────────────────────────────────────
配置文件（config.toml） TOML     Rust 社区标准，人类可读，注释支持好
API 请求/响应           JSON     HTTP API 通用标准
工具参数构建            JSON     serde_json::json! 宏方便快速构建
内部日志/调试           Debug    直接 {:?} 打印
```

---

## 5. 基本用法代码示例

### 5.1 JSON 序列化与反序列化

```rust
use serde::{Serialize, Deserialize};

// 定义数据结构
#[derive(Serialize, Deserialize, Debug)]
struct ChatMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    temperature: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // === 序列化：Rust → JSON ===
    let request = ChatRequest {
        model: "gpt-4".to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a helpful assistant.".to_string(),
                name: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hello!".to_string(),
                name: Some("Alice".to_string()),
            },
        ],
        temperature: 0.7,
    };

    // to_string — 紧凑 JSON（一行）
    let json = serde_json::to_string(&request)?;
    println!("紧凑 JSON:\n{}\n", json);
    // {"model":"gpt-4","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Hello!","name":"Alice"}],"temperature":0.7}

    // to_string_pretty — 美化 JSON（多行缩进）
    let pretty = serde_json::to_string_pretty(&request)?;
    println!("美化 JSON:\n{}\n", pretty);

    // === 反序列化：JSON → Rust ===
    let json_str = r#"{
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Hi there!"}
        ]
    }"#;
    // 注意：temperature 缺失，但有 #[serde(default)] 所以不报错
    let parsed: ChatRequest = serde_json::from_str(json_str)?;
    println!("反序列化结果: {:#?}", parsed);
    println!("temperature 默认值: {}", parsed.temperature); // 0.0

    Ok(())
}
```

### 5.2 TOML 配置文件读取

```rust
use serde::Deserialize;
use std::collections::HashMap;

// 定义配置结构——模仿 ZeroClaw 的 Config
#[derive(Deserialize, Debug)]
struct AppConfig {
    #[serde(default = "default_provider")]
    default_provider: String,

    #[serde(alias = "model")]          // 兼容旧配置中的 "model" 键名
    default_model: Option<String>,

    #[serde(default)]
    providers: HashMap<String, ProviderConfig>,

    #[serde(default)]
    security: SecurityConfig,
}

#[derive(Deserialize, Debug)]
struct ProviderConfig {
    api_key: Option<String>,
    #[serde(default = "default_base_url")]
    base_url: String,
}

#[derive(Deserialize, Debug, Default)]
struct SecurityConfig {
    #[serde(default)]
    enable_sandbox: bool,
    #[serde(default = "default_timeout")]
    timeout_secs: u64,
}

fn default_provider() -> String { "openai".to_string() }
fn default_base_url() -> String { "https://api.openai.com/v1".to_string() }
fn default_timeout() -> u64 { 30 }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 模拟 TOML 配置文件内容
    let toml_content = r#"
        default_provider = "anthropic"
        model = "claude-3-sonnet"

        [providers.openai]
        api_key = "sk-xxx"
        base_url = "https://api.openai.com/v1"

        [providers.anthropic]
        api_key = "sk-ant-xxx"
        base_url = "https://api.anthropic.com"

        [security]
        enable_sandbox = true
    "#;

    // 反序列化 TOML → Rust struct
    let config: AppConfig = toml::from_str(toml_content)?;
    println!("{:#?}", config);

    // 验证 alias 生效："model" 被映射到 default_model
    println!("default_model: {:?}", config.default_model);
    // Some("claude-3-sonnet")

    // 验证 default 生效：timeout_secs 缺失，使用默认值 30
    println!("timeout: {}s", config.security.timeout_secs);
    // 30

    Ok(())
}
```

### 5.3 serde_json::json! 宏 — 快速构建 JSON

`json!` 宏让你用类似 JavaScript 对象字面量的语法直接构建 JSON 值：

```rust
use serde_json::json;

fn main() {
    // json! 宏：像写 JavaScript 一样写 JSON
    let tool_call = json!({
        "type": "function",
        "function": {
            "name": "web_search",
            "arguments": {
                "query": "Rust serde tutorial",
                "max_results": 5
            }
        }
    });

    println!("{}", tool_call);
    // {"function":{"arguments":{"max_results":5,"query":"Rust serde tutorial"},"name":"web_search"},"type":"function"}

    // 可以在 json! 中使用变量
    let query = "Rust programming";
    let count = 10;
    let request = json!({
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a search assistant."},
            {"role": "user", "content": format!("Search for: {}", query)}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {
                        "query": query,        // 直接引用变量
                        "count": count         // 自动转换类型
                    }
                }
            }
        ]
    });

    println!("{}", serde_json::to_string_pretty(&request).unwrap());
}
```

**TypeScript 类比**：

```typescript
// TypeScript 中直接写对象字面量
const toolCall = {
    type: "function",
    function: {
        name: "web_search",
        arguments: { query: "Rust serde tutorial", max_results: 5 }
    }
};

// Rust 的 json! 宏几乎等价于 TypeScript 对象字面量
// 区别：json! 返回 serde_json::Value（类似 TypeScript 的 any object）
```

### 5.4 serde_json::Value — 动态 JSON

当 JSON 结构不确定或太灵活时，用 `serde_json::Value` 处理：

```rust
use serde_json::Value;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let json_str = r#"{"name": "Alice", "age": 30, "scores": [90, 85, 95]}"#;

    // 解析为动态 Value（类似 TypeScript 的 any）
    let v: Value = serde_json::from_str(json_str)?;

    // 用 [] 索引访问（返回 &Value，不存在返回 Value::Null）
    println!("name: {}", v["name"]);         // "Alice"
    println!("age: {}", v["age"]);           // 30
    println!("first score: {}", v["scores"][0]); // 90
    println!("missing: {}", v["missing"]);   // null（不会 panic）

    // 类型转换
    let name: &str = v["name"].as_str().unwrap_or("unknown");
    let age: u64 = v["age"].as_u64().unwrap_or(0);
    println!("name={}, age={}", name, age);

    Ok(())
}
```

**TypeScript 类比**：`serde_json::Value` ≈ TypeScript 中 `Record<string, any>` 的感觉，但更安全——你必须显式 `.as_str()` / `.as_u64()` 转换类型。

---

## 6. Enum 序列化的四种策略

这是前端开发者最容易困惑的点——Rust 的 enum 比 TypeScript 的联合类型强大得多，serde 提供了四种序列化策略：

### 6.1 外部标签（默认）

```rust
#[derive(Serialize, Deserialize, Debug)]
enum ApiResponse {
    Success { data: String },
    Error { code: u32, message: String },
}
// → {"Success": {"data": "ok"}}
// → {"Error": {"code": 404, "message": "not found"}}
```

### 6.2 内部标签（最常用）

```rust
#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]                    // ← 指定标签字段名
enum ApiResponse {
    #[serde(rename = "success")]
    Success { data: String },
    #[serde(rename = "error")]
    Error { code: u32, message: String },
}
// → {"type": "success", "data": "ok"}
// → {"type": "error", "code": 404, "message": "not found"}
```

**TypeScript 类比**：这就是 TypeScript 的**可辨识联合**（Discriminated Union）！

```typescript
// TypeScript 的可辨识联合
type ApiResponse =
    | { type: "success"; data: string }
    | { type: "error"; code: number; message: string };

// Rust 的 #[serde(tag = "type")] 完美对应
```

### 6.3 邻接标签

```rust
#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", content = "data")]  // ← 标签 + 内容分离
enum Event {
    Click { x: u32, y: u32 },
    KeyPress { key: String },
}
// → {"type": "Click", "data": {"x": 100, "y": 200}}
```

### 6.4 无标签（Untagged）

```rust
#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]                        // ← 按数据结构自动匹配
enum FlexibleValue {
    Text(String),
    Number(f64),
    Items(Vec<String>),
}
// "hello" → Text("hello")
// 42.0    → Number(42.0)
// ["a"]   → Items(vec!["a"])
```

**⚠️ 注意**：无标签模式按定义顺序**依次尝试**匹配，第一个成功的变体就是结果。如果多个变体都能匹配，用的是第一个。

### 6.5 四种策略选择指南

```
策略           语法                          适用场景                 性能
──────────────────────────────────────────────────────────────────────────
外部标签       （默认）                       Rust-to-Rust 内部通信    最快
内部标签       #[serde(tag = "type")]         REST API 响应           快
邻接标签       #[serde(tag/content)]          标签+数据需要分离        快
无标签         #[serde(untagged)]             灵活 API/多格式兼容      最慢（逐个尝试）
```

**推荐**：对接外部 API 时首选**内部标签**，它最接近 TypeScript 的可辨识联合。

---

## 7. 在 ZeroClaw 中的实际使用

### 7.1 config/schema.rs — 配置文件解析

ZeroClaw 的配置系统是 serde 属性的重度用户 [^source1]：

```rust
// ZeroClaw 风格的配置结构（简化版）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(skip)]                       // 运行时计算，不存储到文件
    pub workspace_dir: PathBuf,

    #[serde(alias = "model_provider")]   // 兼容旧版配置键名
    pub default_provider: Option<String>,

    #[serde(alias = "model")]            // "model" 也能被识别为 default_model
    pub default_model: Option<String>,

    #[serde(default)]                    // 缺失时使用 HashMap::default()（空 map）
    pub model_providers: HashMap<String, ModelProviderConfig>,

    #[serde(default)]                    // 缺失时使用 ObservabilityConfig::default()
    pub observability: ObservabilityConfig,

    #[serde(default)]
    pub security: SecurityConfig,
    // ... 20+ 配置段，全部使用 #[serde(default)]
}
```

**为什么大量使用 `#[serde(default)]`？** 因为配置文件应该**渐进式**——用户只需要写自己关心的配置项，其余使用合理的默认值。新增配置项不会破坏旧配置文件。

### 7.2 tools/ 模块 — serde_json::json! 构建工具参数

ZeroClaw 在调用 LLM 工具时大量使用 `json!` 宏构建参数 [^source1]：

```rust
use serde_json::json;

// 构建工具定义发送给 LLM
let tool_definition = json!({
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "max_results": {
                    "type": "integer",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
});
```

### 7.3 providers/ — API 响应反序列化

几乎每个 API 响应结构都 derive `Deserialize` [^source1]：

```rust
#[derive(Deserialize, Debug)]
struct CompletionResponse {
    id: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Deserialize, Debug)]
struct Choice {
    index: u32,
    message: ResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

// 从 HTTP 响应中反序列化
// let response: CompletionResponse = serde_json::from_str(&body)?;
```

---

## 8. 常见陷阱与注意事项

### 陷阱 1：忘记启用 derive feature

```toml
# ❌ 错误：缺少 features = ["derive"]
serde = "1.0"

# ✅ 正确
serde = { version = "1.0", features = ["derive"] }
```

没有 `derive` feature，`#[derive(Serialize, Deserialize)]` 会编译失败。

### 陷阱 2：serde_json::Value 的类型陷阱

```rust
let v: serde_json::Value = serde_json::from_str(r#"{"age": 25}"#)?;

// ❌ 常见错误：直接当具体类型用
// let age: u32 = v["age"];  // 编译错误！Value 不能直接转 u32

// ✅ 正确：用 as_u64() 等方法转换
let age: u64 = v["age"].as_u64().unwrap_or(0);
```

### 陷阱 3：rename_all 对 enum 变体名的影响

```rust
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Status {
    InProgress,   // → "in_progress"（不是 "InProgress"）
    Done,         // → "done"
}
// 如果 API 返回 "InProgress"，反序列化会失败！
// 要确认 API 实际使用的命名风格
```

---

## 速查卡

```
serde 核心概念：

  Cargo.toml 配置：
    serde = { version = "1.0", features = ["derive"] }   # 必选
    serde_json = "1.0"                                    # JSON 格式
    toml = "1.0"                                          # TOML 格式

  derive 两件套：
    #[derive(Serialize)]      → Rust struct → JSON/TOML/...
    #[derive(Deserialize)]    → JSON/TOML/... → Rust struct

  三层属性体系：
    容器:  #[serde(rename_all = "camelCase")]     全局重命名
           #[serde(deny_unknown_fields)]          拒绝未知字段
           #[serde(tag = "type")]                 enum 内部标签
    变体:  #[serde(rename = "name")]              变体重命名
    字段:  #[serde(default)]                      缺失用默认值
           #[serde(skip)]                         跳过序列化/反序列化
           #[serde(alias = "old_name")]           接受别名
           #[serde(rename = "api_name")]          重命名
           #[serde(skip_serializing_if = "...")]  条件跳过

  常用 API：
    serde_json::to_string(&data)?       序列化（紧凑）
    serde_json::to_string_pretty(&data)? 序列化（美化）
    serde_json::from_str::<T>(json)?    反序列化
    serde_json::json!({...})            快速构建 JSON
    toml::from_str::<T>(toml_str)?      TOML 反序列化

  Enum 四种策略：
    默认                → {"Variant": {...}}      Rust 内部用
    #[serde(tag)]       → {"type": "variant",...}  API 常用 ✅
    #[serde(tag,content)]→ {"type":"v","data":{}} 分离式
    #[serde(untagged)]  → 按结构匹配               灵活但慢

  vs TypeScript：
    JSON.parse        → serde_json::from_str    （但有编译时类型安全）
    JSON.stringify     → serde_json::to_string   （但保证类型正确）
    zod validation     → derive 自动完成          （零运行时开销）
    interface          → #[derive(Deserialize)]   （生成真正的解析代码）
```

---

> **下一篇**: 阅读 `03_核心概念_2_serde高级属性.md`，深入学习 flatten、自定义序列化函数、枚举策略的高级用法——这些是处理复杂 API 响应时的必备技能。

---

**参考来源**

[^source1]: ZeroClaw 源码分析 — `reference/source_常用库_01.md`
[^context7_serde1]: serde 官方文档 — `reference/context7_serde_01.md`

---

**文件信息**
- 知识点: 常用库速查（serde/anyhow/clap）
- 维度: 03_核心概念_1_serde序列化框架
- 版本: v1.0
- 日期: 2026-03-11
