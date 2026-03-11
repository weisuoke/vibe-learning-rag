# 核心概念 1：Result<T, E> 枚举

> Result<T, E> 是 Rust 表达"操作可能成功也可能失败"的枚举类型，Ok(T) 包含成功值，Err(E) 包含错误信息。
> 本文详解 Result 的定义、创建方式、四种处理方式、常见错误类型，并用 ZeroClaw 源码中的 Tool/Memory/Shell 模块作为实战案例。

---

## 一句话定义

**Result<T, E> 是 Rust 表达"操作可能成功也可能失败"的枚举类型，Ok(T) 包含成功值，Err(E) 包含错误信息。**

TypeScript 开发者可以先这样理解：Result 就是把 try/catch 变成了一个**返回值**——函数不再"抛异常"，而是"返回成功或失败"。

---

## 1. Result 的定义

Result 是 Rust 标准库中的枚举，只有两个变体：

```rust
// 标准库定义（简化）
enum Result<T, E> {
    Ok(T),    // 成功，包含值 T
    Err(E),   // 失败，包含错误 E
}
```

- `T` = 成功时的值类型（比如 `String`、`Config`、`Vec<u8>`）
- `E` = 失败时的错误类型（比如 `io::Error`、`serde_json::Error`、`anyhow::Error`）

```rust
// 具体例子
let success: Result<i32, String> = Ok(42);         // 成功，值是 42
let failure: Result<i32, String> = Err("oops".into()); // 失败，原因是 "oops"
```

**关键认知：Result 是一个普通的枚举值，不是异常，不是特殊控制流。** 它就是一个数据，可以存在变量里、传来传去、放进 Vec 里。

[来源: reference/context7_rust_01.md - Result<T, E> 基础]

---

## 2. TypeScript 对照理解

### 2.1 思维模型转换

```typescript
// TypeScript：异常模型
// 函数签名看不出可能失败
function parseAge(input: string): number {
  const age = parseInt(input);
  if (isNaN(age)) {
    throw new Error("Invalid age");  // 抛异常——调用者可能不知道
  }
  return age;
}

// 调用者可能忘记 try/catch
const age = parseAge("abc");  // 运行时崩溃！
```

```rust
// Rust：Result 模型
// 函数签名明确告诉你：这个操作可能失败
fn parse_age(input: &str) -> Result<u32, String> {
    input.parse::<u32>()
        .map_err(|_| format!("Invalid age: '{}'", input))
}

// 调用者必须处理 Result
let age = parse_age("abc");  // 不会崩溃，返回 Err
```

### 2.2 核心差异

| 维度 | TypeScript throw/catch | Rust Result<T, E> |
|------|----------------------|-------------------|
| 错误在哪 | 隐藏在函数体内（throw） | 写在函数签名上（-> Result） |
| 调用者义务 | 可选（可以不 try/catch） | 强制（不处理编译器警告） |
| 错误类型 | unknown（catch 的 e） | 具体类型 E（编译期已知） |
| 传播方式 | 自动冒泡（沿调用栈向上） | 显式传播（? 运算符） |
| 性能 | 异常有运行时开销 | 零成本（枚举就是普通数据） |

**一句话：TypeScript 的错误是"飞出来的"（throw），Rust 的错误是"返回来的"（Result）。**

### 2.3 如果 TypeScript 有 Result

社区已经有人在 TypeScript 中模拟 Rust 的 Result 模式（ts-result、oxide.ts 等库）：

```typescript
type Result<T, E> = { ok: true; value: T } | { ok: false; error: E };

function parseAge(input: string): Result<number, string> {
  const age = parseInt(input);
  if (isNaN(age)) return { ok: false, error: `Invalid age: '${input}'` };
  return { ok: true, value: age };
}
```

这说明 Result 模式的价值是跨语言的——Rust 只是把它内置到了语言核心。

[来源: reference/search_error_handling_01.md - TS 开发者使用 ts-result、oxide.ts 等库模拟 Rust 的 Result 模式]

---

## 3. 创建 Result

### 3.1 直接创建

```rust
let ok_val: Result<i32, String> = Ok(42);           // 成功，值是 42
let ok_unit: Result<(), String> = Ok(());            // 成功但无返回值（常见于写操作）
let err_val: Result<i32, String> = Err("oops".into()); // 失败
```

### 3.2 从标准库函数获得

大多数时候你不需要手动创建 Result——标准库函数会返回它：

```rust
let content = std::fs::read_to_string("config.json");  // Result<String, io::Error>
let number = "42".parse::<i32>();                       // Result<i32, ParseIntError>
let value = serde_json::from_str::<serde_json::Value>(r#"{"k":"v"}"#); // Result<Value, Error>
```

### 3.3 在函数中返回 Result

```rust
fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 { Err("division by zero".into()) } else { Ok(a / b) }
}
```

**ZeroClaw 中的 Result 返回：**

```rust
// src/lib.rs（简化）
fn parse_temperature(s: &str) -> Result<f64, String> {
    let temp: f64 = s.parse().map_err(|_| format!("Invalid temperature: {}", s))?;
    if !(0.0..=2.0).contains(&temp) {
        return Err(format!("Temperature must be 0.0-2.0, got {}", temp));
    }
    Ok(temp)
}
```

[来源: reference/source_error_handling_01.md - parse_temperature 返回 Result]

---

## 4. 处理 Result

拿到一个 Result 后，有四种主要的处理方式，从最安全到最危险排列：

### 方式 1：match 模式匹配（最基础、最安全）

```rust
match std::fs::read_to_string("config.json") {
    Ok(content) => println!("配置内容: {}", content),  // content 是 String
    Err(e) => eprintln!("读取失败: {}", e),            // e 是 io::Error
}
```

```typescript
// TypeScript 等价物（如果用 Result 模式）
const result = readConfig("config.json");
if (result.ok) { console.log(result.value); }
else { console.error(result.error); }
```

**match 的优势：编译器强制你处理 Ok 和 Err 两种情况，不会遗漏。**

### 方式 2：if let（只关心一种情况）

```rust
if let Ok(content) = std::fs::read_to_string("config.json") {
    println!("配置内容: {}", content);
}
// 失败了？静默忽略（适合非关键操作）
```

**适用场景：日志记录、非关键操作、只需要处理一种情况时。**

### 方式 3：unwrap / expect（快速但危险）

```rust
// unwrap：成功就取值，失败就 panic（程序崩溃）
let content = std::fs::read_to_string("config.json").unwrap();

// expect：同 unwrap，但可以自定义 panic 消息
let content = std::fs::read_to_string("config.json")
    .expect("config.json must exist");
```

**ZeroClaw 的原则：生产代码不使用 unwrap()，仅测试中使用。**

[来源: reference/source_error_handling_01.md - 零 panic：生产代码不使用 unwrap()]

### 方式 4：unwrap_or / unwrap_or_else（安全默认值）

```rust
// unwrap_or：失败时返回默认值
let config = std::fs::read_to_string("custom.json")
    .unwrap_or_else(|_| "{}".to_string());  // 读取失败？用空 JSON

// unwrap_or_default：失败时用类型的 Default 值
let count: i32 = "abc".parse().unwrap_or_default();  // 解析失败？用 0
```

```typescript
// TypeScript 等价物：?? 空值合并运算符
const config = readConfigSync("custom.json") ?? "{}";
```

**ZeroClaw 中的安全默认值：**

```rust
// src/config/schema.rs（简化）
let streaming = config.streaming.unwrap_or(false);     // None → false
```

[来源: reference/source_error_handling_01.md - 安全默认值 .unwrap_or(false)]

#### 四种方式的类比

日常生活类比——收到一个快递包裹（Result）：

| 方式 | 类比 | 安全性 |
|------|------|--------|
| match | 打开检查，对的就用，错的就退货 | 最稳妥 |
| if let | 是你要的就拿走，不是就不管 | 适合非关键 |
| unwrap | 闭眼拆，不对就摔手机（panic） | 危险 |
| unwrap_or | 不对就用备用品 | 有 Plan B |

前端开发类比：

```typescript
// match     ≈ if/else 完整处理
// if let    ≈ if (result.ok) { ... }  只处理成功
// unwrap    ≈ result.value!           非空断言（危险）
// unwrap_or ≈ result.value ?? default  空值合并
```

---

## 5. ZeroClaw 中的 Result 使用

### 5.1 Tool trait 的 execute 方法

ZeroClaw 所有工具的核心方法都返回 `anyhow::Result<ToolResult>`：

```rust
// src/tools/traits.rs
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;

    // 核心：返回 anyhow::Result<ToolResult>
    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult>;
}
```

**为什么返回 `anyhow::Result<ToolResult>` 而不是直接返回 `ToolResult`？**

因为工具执行可能失败——网络超时、参数非法、权限不足。用 Result 让调用者知道"这个操作可能不成功"。

[来源: reference/source_error_handling_01.md - Tool trait 定义]

### 5.2 Memory trait 的方法签名

Memory trait 的每个方法都返回 Result，因为存储操作天然可能失败：

```rust
// src/memory/traits.rs
#[async_trait]
pub trait Memory: Send + Sync {
    async fn store(&self, entry: MemoryEntry) -> anyhow::Result<()>;
    //                                          ^^^^^^^^^^^^^^^^
    //                                          成功无返回值，失败有错误

    async fn recall(&self, query: &str, limit: usize) -> anyhow::Result<Vec<MemoryEntry>>;
    //                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                                                    成功返回记忆列表，失败有错误

    async fn forget(&self, key: &str) -> anyhow::Result<bool>;
    //                                   ^^^^^^^^^^^^^^^^^^
    //                                   成功返回是否删除了，失败有错误
}
```

**注意 `anyhow::Result<()>`：** `()` 是 Rust 的"空值"类型，表示"成功但没有返回值"。等价于 TypeScript 中 `Promise<void>` 的概念。

[来源: reference/source_error_handling_01.md - Memory trait 方法签名]

### 5.3 Shell 工具的嵌套 match

Shell 工具展示了 Result 最强大的模式——嵌套 match 处理多层可能的失败：

```rust
// src/tools/shell.rs（简化）
match tokio::time::timeout(Duration::from_secs(30), cmd.output()).await {
    Ok(Ok(output)) => { /* 未超时 + 执行成功 → 处理输出 */ },
    Ok(Err(e))     => { /* 未超时 + 执行失败 → 报告错误 */ },
    Err(_)         => { /* 超时 → 报告超时 */ },
}
```

类型解读：

```
timeout(...).await 返回 Result<Result<Output, io::Error>, Elapsed>
                          ↑外层                              ↑超时
                               ↑内层成功    ↑内层失败
```

三种情况，三种不同的处理策略，编译器保证你不会遗漏任何一种。TypeScript 中需要嵌套 try/catch + 手动区分错误类型才能做到同样的事。

[来源: reference/source_error_handling_01.md - 嵌套 match（超时 + 执行结果）]

### 5.4 Web Fetch 工具的 validate + match

验证失败时不传播错误，而是包装成业务结果：

```rust
// src/tools/web_fetch.rs（简化）
let validated_url = match self.validate_url(url) {
    Ok(v) => v,
    Err(e) => return Ok(ToolResult {
        success: false, error: Some(e.to_string()), ..Default::default()
    }),
};
```

**设计思路：** 参数缺失用 `?` 传播（程序错误），URL 验证失败用 match 包装（用户输入错误）。两种错误，两种处理策略。

[来源: reference/source_error_handling_01.md - match 模式匹配 Result]

---

## 6. 常见错误类型

Rust 生态中常见的错误类型：

| 错误类型 | 来源 | 场景 |
|----------|------|------|
| `std::io::Error` | 标准库 | 文件读写、网络 I/O |
| `serde_json::Error` | serde_json | JSON 解析/序列化 |
| `reqwest::Error` | reqwest | HTTP 请求 |
| `std::num::ParseIntError` | 标准库 | 字符串转数字 |
| `anyhow::Error` | anyhow | 通用错误容器（类型擦除） |
| 自定义枚举 | thiserror | 库定义的具体错误类型 |

**ZeroClaw 的选择：统一使用 `anyhow::Error`。**

```rust
// 所有这些不同的错误类型，都可以用 ? 自动转换为 anyhow::Error
fn load_and_parse(path: &str) -> anyhow::Result<Config> {
    let content = std::fs::read_to_string(path)?;  // io::Error → anyhow::Error
    let config: Config = serde_json::from_str(&content)?;  // serde_json::Error → anyhow::Error
    Ok(config)
}
```

**为什么能自动转换？** 因为 `anyhow::Error` 实现了 `From<E>` trait，对任何实现了 `std::error::Error` 的类型 E，`?` 运算符会自动调用 `.into()` 进行转换。这就是 Trait 系统的威力。

[来源: reference/context7_rust_01.md - 自定义错误类型 + From trait]
[来源: reference/context7_anyhow_01.md - anyhow 错误处理]

---

## 7. 组合子方法与 Option 转换（预览）

> 详细内容见 [03_核心概念_4_组合子方法](./03_核心概念_4_组合子方法.md) 和 [03_核心概念_2_Option枚举](./03_核心概念_2_Option枚举.md)

Result 提供一组函数式风格的组合子方法，避免到处写 match：

```rust
let r: Result<i32, String> = Ok(5);

r.map(|x| x * 2)                    // Ok(10) — 转换成功值
r.map_err(|e| format!("err: {e}"))   // 转换错误值
r.and_then(|x| if x > 0 { Ok(x) } else { Err("neg".into()) })  // 链式操作
r.unwrap_or(0)                       // 失败用默认值
```

Option 和 Result 经常互相转换：

```rust
// Option → Result
let opt: Option<i32> = Some(42);
let res = opt.ok_or("missing")?;     // Some → Ok, None → Err

// ZeroClaw 最常见模式：从 JSON 提取参数
// src/tools/shell.rs
let command = args.get("command")       // Option<&Value>
    .and_then(|v| v.as_str())           // Option<&str>
    .ok_or_else(|| anyhow::anyhow!("Missing 'command' parameter"))?;
```

[来源: reference/context7_rust_01.md - 组合子]
[来源: reference/source_error_handling_01.md - Option 组合子链]

---

## 9. 小结

### 速查卡

```
创建 Result：
  Ok(value)                    成功
  Err(error)                   失败

处理 Result（安全 → 危险）：
  match r { Ok(v) => ..., Err(e) => ... }   完整处理（推荐）
  if let Ok(v) = r { ... }                  只处理成功
  r.unwrap_or(default)                      失败用默认值（安全）
  r.unwrap_or_else(|e| ...)                 失败用闭包计算默认值
  r.unwrap() / r.expect("msg")             失败就 panic（仅测试用）

组合子方法：
  .map(|v| ...)              转换成功值
  .map_err(|e| ...)          转换错误值
  .and_then(|v| ...)         链式操作
  .or_else(|e| ...)          失败恢复

Option ↔ Result 转换：
  option.ok_or(err)          Option → Result
  option.ok_or_else(|| err)  Option → Result（惰性）
  result.ok()                Result → Option（丢弃错误）
```

### TypeScript -> Rust 对照

| TypeScript | Rust | 备注 |
|-----------|------|------|
| `throw new Error("msg")` | `Err("msg".into())` 或 `bail!("msg")` | 返回错误而非抛出 |
| `try { ... } catch (e) { ... }` | `match result { Ok(v) => ..., Err(e) => ... }` | 模式匹配 |
| `value!` (非空断言) | `.unwrap()` | 都危险，都不推荐 |
| `value ?? default` | `.unwrap_or(default)` | 安全默认值 |
| `Promise.then()` | `.map()` / `.and_then()` | 链式转换 |
| `catch` 的 `e` 是 `unknown` | `Err(e)` 的 `e` 是具体类型 | Rust 类型更安全 |
| 无等价物 | 编译器强制处理 Result | Rust 独有优势 |

### ZeroClaw 中的 Result 模式总结

| 模式 | 代码 | 使用场景 |
|------|------|----------|
| ? 传播 | `file.read()?` | 错误直接向上传播 |
| match 分支 | `match validate() { Ok(v) => ..., Err(e) => ... }` | 需要不同处理策略 |
| 嵌套 match | `match timeout(cmd).await { Ok(Ok(..)) => ... }` | 多层可能失败 |
| Option → Result | `.ok_or_else(\|\| anyhow!("msg"))?` | 从 JSON 提取参数 |
| 安全默认值 | `.unwrap_or(false)` | 配置字段缺失 |

---

*上一篇：[02_第一性原理](./02_第一性原理.md) -- 错误处理的根本推理*
*下一篇：[03_核心概念_2_Option枚举](./03_核心概念_2_Option枚举.md) -- Option 的完整语法与 ZeroClaw 配置实战*
