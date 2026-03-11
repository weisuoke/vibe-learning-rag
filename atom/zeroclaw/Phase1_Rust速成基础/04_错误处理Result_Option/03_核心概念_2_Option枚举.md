# 核心概念 2：Option<T> 枚举

> **前置知识**：03_核心概念_1_Result枚举.md
> **预计阅读**：15 分钟
> **难度**：入门

---

## 一句话定义

**Option<T> 是 Rust 表达"值可能存在也可能不存在"的枚举类型，Some(T) 包含值，None 表示空。**

Rust 没有 null。所有"可能为空"的场景，都必须用 Option 显式声明。编译器会强制你处理"值不存在"的情况，从根源上消灭空指针崩溃。

[来源: Comprehensive Rust (Google) - Result Option error handling]

---

## Option 的定义

```rust
// 标准库中的定义，非常简单
enum Option<T> {
    Some(T),  // 有值，包裹着一个 T 类型的值
    None,     // 无值，什么都没有
}
```

就两个变体，没有更多了。`Some(T)` 装着值，`None` 表示空。

**为什么是泛型 `<T>`？** 因为"可能为空"这件事跟具体类型无关——`Option<String>` 是可能为空的字符串，`Option<f64>` 是可能为空的浮点数，`Option<Vec<u8>>` 是可能为空的字节数组。

[来源: Comprehensive Rust (Google)]

---

## TypeScript 对照

### null/undefined vs Option

```typescript
// TypeScript：值可能为空的三种表达方式
let name: string | null = null;
let age: number | undefined = undefined;
let score: string | null | undefined = null;

// 问题1：null 和 undefined 是两个不同的"空"，容易混淆
// 问题2：忘记检查 null 就访问属性 → 运行时崩溃
console.log(name.length);  // 编译通过（strict 模式下不通过），运行时炸
```

```rust
// Rust：只有一种"空"——None
let name: Option<String> = None;
let age: Option<u32> = Some(25);

// 你不能直接使用 Option<String> 当 String 用
// println!("{}", name.len());  // 编译错误！Option<String> 没有 len() 方法
// 必须先"解包"才能使用里面的值
```

### optional chaining ?. vs Option 方法

```typescript
// TypeScript 的 optional chaining
const result = obj?.nested?.value ?? "default";
```

```rust
// Rust 的 Option 链式调用（功能类似，但更显式）
let result = obj
    .nested()           // 返回 Option<Nested>
    .and_then(|n| n.value())  // 返回 Option<String>
    .unwrap_or("default".to_string());
```

**核心区别**：TypeScript 的 `?.` 是语法糖，遇到 null/undefined 就短路返回 undefined。Rust 的 Option 方法是真正的类型系统约束——编译器知道每一步的类型，不可能"忘记处理"。

[来源: search_error_handling_01.md - TypeScript 对比]

---

## 创建 Option

```rust
// 方式1：直接构造
let some_number: Option<i32> = Some(42);
let no_number: Option<i32> = None;

// 方式2：类型推断（通常不需要写类型注解）
let name = Some("Alice");   // Option<&str>
let empty: Option<String> = None;  // None 需要注解，编译器无法推断 T

// 方式3：标准库方法返回 Option
let text = "hello world";
let found = text.find("world");     // Option<usize> → Some(6)
let not_found = text.find("rust");  // Option<usize> → None

let numbers = vec![1, 2, 3];
let first = numbers.first();        // Option<&i32> → Some(&1)
let empty_vec: Vec<i32> = vec![];
let nothing = empty_vec.first();    // Option<&i32> → None

// 方式4：HashMap 查找
use std::collections::HashMap;
let mut map = HashMap::new();
map.insert("key", "value");
let got = map.get("key");           // Option<&&str> → Some(&"value")
let miss = map.get("nope");         // Option<&&str> → None
```

[来源: Comprehensive Rust (Google)]

---

## 处理 Option

### 方式1：match 模式匹配（最基础）

```rust
fn greet(name: Option<&str>) {
    match name {
        Some(n) => println!("Hello, {}!", n),
        None => println!("Hello, stranger!"),
    }
}

greet(Some("Alice"));  // Hello, Alice!
greet(None);           // Hello, stranger!
```

**优点**：穷尽检查，编译器确保你处理了 Some 和 None 两种情况。
**缺点**：简单场景下代码略显冗长。

### 方式2：if let Some(value)（只关心有值的情况）

```rust
let config_value: Option<String> = Some("production".to_string());

// 只在有值时执行
if let Some(env) = config_value {
    println!("Environment: {}", env);
}
// 没有 else 分支也行——不关心 None 就不写

// 带 else 分支
if let Some(env) = config_value {
    println!("Environment: {}", env);
} else {
    println!("Using default environment");
}
```

**适用场景**：你只关心"有值"的情况，None 时什么都不做或只做简单处理。

### 方式3：unwrap / expect（危险！仅用于开发调试）

```rust
let value: Option<i32> = Some(42);
let x = value.unwrap();    // 42 —— 有值时正常解包

let empty: Option<i32> = None;
// let y = empty.unwrap();  // panic! 程序崩溃！
// let z = empty.expect("值不应该为空");  // panic! 带自定义消息
```

**ZeroClaw 原则：生产代码零 panic，绝不在生产代码中对 Option 使用 unwrap()。**

> **双重类比**
> - **前端类比**：`unwrap()` 就像 TypeScript 的非空断言 `value!.property`——你告诉编译器"我保证不为空"，但如果你错了，运行时就炸了。
> - **日常生活类比**：`unwrap()` 就像不看路就过马路——大部分时候没事，但出事就是大事。

[来源: search_error_handling_01.md - 常见初学者陷阱]

### 方式4：unwrap_or / unwrap_or_default（安全默认值）

```rust
// unwrap_or：提供一个固定默认值
let port: Option<u16> = None;
let actual_port = port.unwrap_or(8080);  // 8080

// unwrap_or_default：使用类型的 Default 实现
let name: Option<String> = None;
let actual_name = name.unwrap_or_default();  // "" (空字符串)

let count: Option<i32> = None;
let actual_count = count.unwrap_or_default();  // 0

let flag: Option<bool> = None;
let actual_flag = flag.unwrap_or_default();  // false

// unwrap_or_else：用闭包延迟计算默认值（默认值计算开销大时使用）
let config: Option<String> = None;
let actual = config.unwrap_or_else(|| {
    // 只在 None 时才执行这个闭包
    load_default_config()
});
```

**这是 ZeroClaw 中最常见的 Option 处理方式之一。**

[来源: source_error_handling_01.md - Option<T> 使用场景]

### 方式5：map / and_then（函数式变换）

```rust
// map：对 Some 里的值做变换，None 保持 None
let number: Option<i32> = Some(5);
let doubled = number.map(|n| n * 2);  // Some(10)

let empty: Option<i32> = None;
let doubled = empty.map(|n| n * 2);   // None（不会执行闭包）

// and_then：变换函数本身也返回 Option（避免 Option<Option<T>> 嵌套）
let text: Option<&str> = Some("42");
let parsed = text.and_then(|t| t.parse::<i32>().ok());  // Some(42)

let bad: Option<&str> = Some("abc");
let parsed = bad.and_then(|t| t.parse::<i32>().ok());   // None

// 链式调用（ZeroClaw 风格）
let result = args.get("timeout")           // Option<&Value>
    .and_then(|v| v.as_u64())             // Option<u64>
    .map(|secs| Duration::from_secs(secs)) // Option<Duration>
    .unwrap_or(Duration::from_secs(30));   // Duration
```

> **双重类比**
> - **前端类比**：`map` 就像数组的 `.map()`——对里面的值做变换。`and_then` 就像 `.flatMap()`——变换后展平一层。
> - **日常生活类比**：`map` 是"如果有包裹，就拆开加工再装回去"。`and_then` 是"如果有包裹，拆开看看里面还有没有东西"。

[来源: Comprehensive Rust (Google) - 组合子]

---

## Option → Result 转换

这是 ZeroClaw 中最关键的模式之一：把"值可能不存在"转换为"带错误信息的失败"。

### ok_or()：None → 固定错误

```rust
let name: Option<&str> = None;
let result: Result<&str, &str> = name.ok_or("name is required");
// Err("name is required")

let name: Option<&str> = Some("Alice");
let result: Result<&str, &str> = name.ok_or("name is required");
// Ok("Alice")
```

### ok_or_else()：None → 延迟构造错误（推荐）

```rust
use anyhow::anyhow;

let url: Option<&str> = None;
let result = url.ok_or_else(|| anyhow!("Missing 'url' parameter"));
// Err(anyhow::Error: Missing 'url' parameter)
```

**为什么推荐 `ok_or_else` 而不是 `ok_or`？** 因为 `ok_or` 会在调用时就构造错误对象（即使是 Some 也会构造），而 `ok_or_else` 只在 None 时才构造——性能更好，尤其是错误构造开销大时。

[来源: Comprehensive Rust (Google) - Option → Result 转换]

---

## ZeroClaw 中的 Option 使用

### 配置字段（config/schema.rs）

ZeroClaw 的配置结构体大量使用 Option 表示可选字段：

```rust
// 来自 ZeroClaw config/schema.rs
pub struct LlmConfig {
    pub api_key: Option<String>,          // API 密钥，可能从环境变量获取
    pub api_url: Option<String>,          // 自定义 API 端点，不填则用默认
    pub default_provider: Option<String>, // 默认 LLM 提供商
}
```

**为什么用 Option 而不是空字符串？**
- `Option<String>` 语义清晰：`None` = 未配置，`Some("")` = 配置了但值为空（这是两种不同的状态）
- 序列化时 `None` 字段可以省略，配置文件更干净
- 编译器强制你处理"未配置"的情况

```rust
// 使用配置时的典型模式
let api_url = config.api_url
    .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
```

[来源: source_error_handling_01.md - config/schema.rs 分析]

### 内存条目（memory/traits.rs）

```rust
// 来自 ZeroClaw memory/traits.rs
pub struct MemoryEntry {
    pub content: String,                // 必填：内容
    pub session_id: Option<String>,     // 可选：会话 ID
    pub score: Option<f64>,             // 可选：相关性分数（检索时才有）
}
```

`score` 字段是一个经典案例：存储时没有分数（None），检索时才计算出分数（Some(0.95)）。同一个结构体在不同阶段有不同的"完整度"。

[来源: source_error_handling_01.md - memory/traits.rs 分析]

### 工具参数提取（tools/shell.rs）

```rust
// 来自 ZeroClaw tools/shell.rs
// 从 JSON 参数中提取布尔值，缺失时默认 false
let approved = args.get("approved")     // Option<&Value>
    .and_then(|v| v.as_bool())          // Option<bool>
    .unwrap_or(false);                  // bool
```

**拆解这个链式调用**：
1. `args.get("approved")` → HashMap 查找，返回 `Option<&Value>`
2. `.and_then(|v| v.as_bool())` → 如果有值，尝试转为 bool；如果 JSON 值不是布尔类型，返回 None
3. `.unwrap_or(false)` → 如果前面任何一步是 None，就用 false 作为默认值

这三步中任何一步"失败"（返回 None），最终都会安全地得到 `false`。没有 panic，没有崩溃。

[来源: source_error_handling_01.md - tools/shell.rs 分析]

### HTTP 头提取（tools/web_fetch.rs）

```rust
// 来自 ZeroClaw tools/web_fetch.rs
let content_type = response.headers()
    .get(reqwest::header::CONTENT_TYPE)  // Option<&HeaderValue>
    .and_then(|v| v.to_str().ok())       // Option<&str>
    .unwrap_or("")                        // &str
    .to_lowercase();
```

**拆解**：
1. `.get(CONTENT_TYPE)` → HTTP 响应可能没有 Content-Type 头
2. `.and_then(|v| v.to_str().ok())` → 头的值可能不是合法 UTF-8（`.ok()` 把 Result 转成 Option）
3. `.unwrap_or("")` → 都没有就当空字符串处理

> **注意 `.ok()` 的用法**：`Result<T, E>.ok()` 把 `Ok(v)` 变成 `Some(v)`，把 `Err(_)` 变成 `None`。这是 Result → Option 的转换，跟前面讲的 `ok_or()` 方向相反。

[来源: source_error_handling_01.md - tools/web_fetch.rs 分析]

---

## Option vs Result 选择指南

| 场景 | 用 Option | 用 Result | 理由 |
|------|:---------:|:---------:|------|
| 配置字段可选 | ✅ | | 缺失是正常情况，不是错误 |
| HashMap 查找 | ✅ | | key 不存在是预期行为 |
| 数组取第一个元素 | ✅ | | 空数组不是错误 |
| 文件读取 | | ✅ | 失败需要知道原因（权限？不存在？） |
| 网络请求 | | ✅ | 失败需要错误信息 |
| 参数验证 | | ✅ | 缺失参数需要告诉调用者哪个参数缺了 |
| 字符串解析为数字 | | ✅ | 失败需要知道为什么解析不了 |

**简单判断法**：
- 如果"没有值"是正常的、预期的 → **Option**
- 如果"没有值"是异常的、需要解释原因的 → **Result**

> **双重类比**
> - **前端类比**：Option 像 `Array.find()` 返回 `T | undefined`——找不到很正常。Result 像 `fetch()` 返回 `Promise<Response>`——失败了你需要知道是 404 还是 500。
> - **日常生活类比**：Option 是"冰箱里有没有牛奶"——没有就没有，不需要解释。Result 是"快递到没到"——没到你得知道是在路上还是丢了。

---

## 常见错误与修复

### 错误1：直接使用 Option 的值

```rust
let name: Option<String> = Some("Alice".to_string());
// println!("{}", name.len());  // 编译错误：Option<String> 没有 len()

// 修复：先解包
if let Some(n) = name {
    println!("{}", n.len());
}
```

### 错误2：在需要 Result 的地方返回 Option

```rust
// 错误：函数签名要求 Result，但 HashMap::get 返回 Option
fn get_config(key: &str) -> anyhow::Result<String> {
    let map = load_config();
    // map.get(key)  // 这返回 Option，不是 Result！

    // 修复：用 ok_or_else 转换
    map.get(key)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Config key '{}' not found", key))
}
```

### 错误3：不必要的 unwrap 嵌套

```rust
// 不好：多层 unwrap，任何一层 None 都会 panic
let value = args.get("key").unwrap().as_str().unwrap();

// 好：用 and_then 链式处理
let value = args.get("key")
    .and_then(|v| v.as_str())
    .ok_or_else(|| anyhow::anyhow!("Missing 'key'"))?;
```

---

## 小结

| 要点 | 说明 |
|------|------|
| Option 是什么 | `Some(T)` 有值，`None` 无值，Rust 的 null 替代品 |
| 为什么没有 null | 编译器强制处理空值，消灭空指针崩溃 |
| 最安全的处理方式 | `match`、`if let`、`unwrap_or`、`map`/`and_then` |
| 最危险的处理方式 | `unwrap()` / `expect()` —— 生产代码禁用 |
| Option → Result | `ok_or()` / `ok_or_else()` —— ZeroClaw 高频模式 |
| Result → Option | `.ok()` / `.err()` —— 丢弃错误信息 |
| ZeroClaw 核心模式 | `args.get("x").and_then(\|v\| v.as_str()).unwrap_or("")` |

**记住**：在 Rust 中，`Option` 不是"可能出错"，而是"可能没有"。需要错误信息时，用 `ok_or_else()` 转成 `Result`。

[来源: source_error_handling_01.md, context7_rust_01.md, search_error_handling_01.md]
