# 核心概念 5：derive 宏与属性

> **一句话定位：** `#[derive]` 让编译器自动生成 trait 实现代码，一行注解省掉几十行 boilerplate——Debug 看调试、Clone 能复制、PartialEq 能比较、Serde 能序列化。

---

## 1. 什么是 derive 宏

### 1.1 基本概念

`#[derive(...)]` 是一个**属性宏**，告诉编译器："帮我自动生成这些 trait 的实现代码"。

```rust
#[derive(Debug, Clone, PartialEq)]  // 编译器自动生成三个 impl 块
struct AgentConfig {
    name: String,
    model: String,
    temperature: f64,
}
```

等价于你手写了：

```rust
impl std::fmt::Debug for AgentConfig { /* 几十行代码 */ }
impl Clone for AgentConfig { /* 几十行代码 */ }
impl PartialEq for AgentConfig { /* 几十行代码 */ }
```

### 1.2 TypeScript 类比

```typescript
// TypeScript: 装饰器（运行时）
@Serializable
@Injectable
class AgentConfig { ... }

// Rust: derive（编译时，零运行时开销）
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AgentConfig { ... }
```

**关键区别**：TypeScript 装饰器是运行时反射，Rust derive 是编译时代码生成——零开销。

### 1.3 前提条件

derive 一个 trait 时，**所有字段都必须也实现了该 trait**：

```rust
#[derive(Clone)]
struct Agent {
    name: String,      // String 实现了 Clone ✓
    config: AgentConfig, // AgentConfig 也必须实现 Clone ✓
    // raw: RawPointer, // 如果 RawPointer 没实现 Clone → 编译错误 ✗
}
```

---

## 2. Debug — 调试输出

### 2.1 基本用法

```rust
#[derive(Debug)]
struct ChatMessage { role: String, content: String }

let msg = ChatMessage { role: "user".into(), content: "Hello".into() };
println!("{:?}", msg);   // ChatMessage { role: "user", content: "Hello" }
println!("{:#?}", msg);  // 美化输出（多行缩进）
```

### 2.2 为什么几乎所有类型都应该 derive Debug

- **调试必备**：`println!("{:?}", value)` 是 Rust 最常用的调试手段
- **错误信息**：`unwrap()` panic 时会打印 Debug 输出
- **日志系统**：ZeroClaw 的所有类型都 derive Debug，方便 tracing 日志

```rust
// 没有 Debug 的类型无法用 {:?} 打印
// println!("{:?}", no_debug_value);  // 编译错误！
```

**经验法则**：除非有特殊原因，**每个 struct 和 enum 都应该 derive Debug**。

---

## 3. Clone 与 Copy — 复制语义

### 3.1 Clone：显式深拷贝

```rust
#[derive(Debug, Clone)]
struct AgentConfig { name: String, model: String, temperature: f64 }

let c1 = AgentConfig { name: "a".into(), model: "gpt-4".into(), temperature: 0.7 };
let c2 = c1.clone();  // 显式调用，深拷贝所有字段（包括堆上的 String）
println!("{:?}", c1);  // c1 仍然可用
println!("{:?}", c2);  // c2 是独立副本
```

### 3.2 Copy：隐式按位拷贝

```rust
#[derive(Debug, Clone, Copy)]
enum AutonomyLevel { ReadOnly, Supervised, Full }

let a = AutonomyLevel::Full;
let b = a;       // 隐式拷贝，不需要 .clone()
println!("{:?}", a);  // a 仍然可用！（如果没有 Copy，a 会被 move）
```

### 3.3 哪些类型能 Copy

| 能 Copy | 不能 Copy |
|---------|-----------|
| `i32`, `f64`, `bool`, `char` | `String`（堆数据） |
| `(i32, f64)` 元组（全 Copy） | `Vec<T>`（堆数据） |
| `[i32; 3]` 数组（元素 Copy） | `Box<T>`（堆数据） |
| 纯单元变体 enum | 含 String/Vec 字段的 struct |

**规则**：Copy 要求所有字段都是 Copy。String、Vec、Box 不是 Copy → 包含它们的类型也不能 Copy。

### 3.4 ZeroClaw 应用

```rust
// AutonomyLevel 可以 Copy（纯单元变体，栈上数据）
#[derive(Debug, Clone, Copy, PartialEq)]
enum AutonomyLevel { ReadOnly, Supervised, Full }

// Agent 不能 Copy（包含 String、Vec、Box）
#[derive(Debug, Clone)]
struct Agent { name: String, tools: Vec<String> }
```

---

## 4. PartialEq 与 Eq — 相等比较

### 4.1 PartialEq：启用 == 和 !=

```rust
#[derive(Debug, PartialEq)]
enum MemoryCategory { Conversation, Knowledge, Task, Custom(String) }

let a = MemoryCategory::Task;
let b = MemoryCategory::Task;
println!("{}", a == b);  // true
println!("{}", a != MemoryCategory::Knowledge);  // true
```

### 4.2 Eq：完全相等（marker trait）

```rust
#[derive(Debug, PartialEq, Eq, Hash)]
enum MemoryCategory { Conversation, Knowledge, Task, Custom(String) }
```

- `PartialEq`：允许"部分相等"（`f64` 的 NaN != NaN）
- `Eq`：保证"完全相等"（自反性：a == a 永远为 true）
- **用于 HashMap key 必须同时有 Eq + Hash**

**经验法则**：如果类型不含 `f64`/`f32`，derive PartialEq 时顺便 derive Eq。

---

## 5. Hash — 哈希计算

```rust
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum MemoryCategory { Conversation, Knowledge, Task }

let mut counts: HashMap<MemoryCategory, u32> = HashMap::new();
counts.insert(MemoryCategory::Conversation, 42);
counts.insert(MemoryCategory::Task, 7);
println!("{:?}", counts);
```

**规则**：Hash 必须和 Eq 一致——`a == b` 则 `hash(a) == hash(b)`。

---

## 6. Default — 默认值

### 6.1 自动 Default

```rust
#[derive(Debug, Default)]
struct AgentConfig {
    name: String,        // 默认 ""
    temperature: f64,    // 默认 0.0
    max_tokens: u32,     // 默认 0
    verbose: bool,       // 默认 false
}

let config = AgentConfig::default();
// AgentConfig { name: "", temperature: 0.0, max_tokens: 0, verbose: false }
```

### 6.2 自定义 Default

```rust
#[derive(Debug)]
struct AgentConfig { name: String, model: String, temperature: f64, max_tokens: u32 }

impl Default for AgentConfig {
    fn default() -> Self {
        AgentConfig {
            name: "assistant".into(),
            model: "gpt-4".into(),
            temperature: 0.7,
            max_tokens: 4096,
        }
    }
}

// 用 Default + struct update 语法覆盖部分字段
let config = AgentConfig { name: "creative".into(), temperature: 1.5, ..Default::default() };
```

---

## 7. PartialOrd 与 Ord — 排序

```rust
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Priority { Low, Medium, High, Critical }

let mut tasks = vec![Priority::High, Priority::Low, Priority::Critical];
tasks.sort();
println!("{:?}", tasks);  // [Low, Medium, High, Critical]（按定义顺序）
```

- `PartialOrd`：启用 `<`, `>`, `<=`, `>=`
- `Ord`：完全排序，用于 `BTreeMap` key 和 `.sort()`

---

## 8. Serde — 序列化与反序列化（重点！）

### 8.1 基础用法

```toml
# Cargo.toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

```rust
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
struct AgentConfig {
    name: String,
    model: String,
    temperature: f64,
}

let config = AgentConfig { name: "bot".into(), model: "gpt-4".into(), temperature: 0.7 };
let json = serde_json::to_string(&config).unwrap();
// {"name":"bot","model":"gpt-4","temperature":0.7}

let parsed: AgentConfig = serde_json::from_str(&json).unwrap();
```

**TypeScript 对比**：`JSON.stringify` / `JSON.parse` 但有**编译时类型安全**。

### 8.2 常用容器属性

```rust
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]       // 字段名 snake_case → camelCase
#[serde(deny_unknown_fields)]            // 拒绝未知字段
struct ApiResponse {
    user_name: String,     // JSON: "userName"
    total_count: u32,      // JSON: "totalCount"
}
```

### 8.3 常用字段属性

```rust
#[derive(Serialize, Deserialize)]
struct AgentConfig {
    name: String,

    #[serde(rename = "llm_model")]           // 重命名
    model: String,

    #[serde(default)]                        // 缺失时用 Default
    temperature: f64,

    #[serde(skip)]                           // 完全跳过
    internal_state: String,

    #[serde(skip_serializing_if = "Option::is_none")]  // None 时不输出
    description: Option<String>,
}
```

### 8.4 Enum 序列化策略

```rust
// 1. 外部标签（默认）: {"Chat": {"role":"user","content":"hi"}}
#[derive(Serialize, Deserialize)]
enum Message { Chat { role: String, content: String } }

// 2. 内部标签: {"type":"Chat","role":"user","content":"hi"}
#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
enum Message2 { Chat { role: String, content: String } }

// 3. 邻接标签: {"type":"Chat","data":{"role":"user","content":"hi"}}
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
enum Message3 { Chat { role: String, content: String } }

// 4. 无标签: {"role":"user","content":"hi"}（按数据结构匹配）
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum Message4 { Chat { role: String, content: String } }
```

**ZeroClaw 应用**：`config/schema.rs` 中 70+ struct 全部使用 `#[derive(Serialize, Deserialize)]` 解析 TOML/YAML 配置文件。

---

## 9. 组合使用与最佳实践

### 常用 derive 组合

```rust
// 基础三件套（几乎所有类型）
#[derive(Debug, Clone, PartialEq)]

// HashMap key / HashSet 元素
#[derive(Debug, Clone, PartialEq, Eq, Hash)]

// 小型值类型（纯栈数据）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]

// 需要默认值
#[derive(Debug, Clone, Default)]

// API / 配置类型
#[derive(Debug, Clone, Serialize, Deserialize)]

// 公有库 API（尽可能多）
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
```

### 什么时候不该 derive

- 含 `f64` 字段 → 不要 derive `Eq`/`Hash`（NaN 问题）
- 需要自定义相等逻辑 → 手写 `impl PartialEq`
- 含不可 Clone 字段（如 `MutexGuard`）→ 不能 derive Clone

---

## 10. 完整可运行代码示例

```rust
use std::collections::HashMap;

// --- Debug + Clone + PartialEq ---
#[derive(Debug, Clone, PartialEq)]
struct AgentConfig {
    name: String,
    model: String,
    temperature: f64,
    max_tokens: u32,
}

impl Default for AgentConfig {
    fn default() -> Self {
        AgentConfig {
            name: "assistant".into(), model: "gpt-4".into(),
            temperature: 0.7, max_tokens: 4096,
        }
    }
}

// --- Copy + Eq + Hash ---
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum AutonomyLevel { ReadOnly, Supervised, Full }

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum MemoryCategory { Conversation, Knowledge, Task }

fn main() {
    // Debug
    let config = AgentConfig::default();
    println!("{:?}", config);
    println!("{:#?}", config);

    // Clone + PartialEq
    let c2 = AgentConfig { name: "creative".into(), temperature: 1.5, ..config.clone() };
    println!("same? {}", config == c2);  // false

    let c3 = config.clone();
    println!("same? {}", config == c3);  // true

    // Copy
    let level = AutonomyLevel::Full;
    let level2 = level;  // Copy, not move
    println!("{:?} == {:?}? {}", level, level2, level == level2);

    // Hash — 用 enum 做 HashMap key
    let mut counts: HashMap<MemoryCategory, u32> = HashMap::new();
    counts.insert(MemoryCategory::Conversation, 42);
    counts.insert(MemoryCategory::Task, 7);
    println!("conversation count: {}", counts[&MemoryCategory::Conversation]);

    // Default + struct update
    let custom = AgentConfig { name: "bot".into(), ..Default::default() };
    println!("custom: {:?}", custom);
}
```

**编译运行：** `rustc main.rs && ./main`

---

## 速查卡

```
常用 derive trait：
  Debug        {:?} 调试输出          几乎所有类型都要
  Clone        .clone() 深拷贝        需要复制时
  Copy         隐式拷贝（栈上）        小型值类型（i32, enum 无数据变体）
  PartialEq    == != 比较             需要比较时
  Eq           完全相等（marker）      HashMap key 需要
  Hash         哈希计算               HashMap key / HashSet 需要
  Default      ::default() 默认值     需要默认构造时
  Ord          排序                   BTreeMap key / .sort() 需要

Serde：
  Serialize, Deserialize             JSON/TOML/YAML 序列化
  #[serde(rename_all = "camelCase")] 全局重命名
  #[serde(tag = "type")]            enum 内部标签
  #[serde(skip)]                    跳过字段
  #[serde(default)]                 缺失用默认值

组合速查：
  基础三件套:  Debug, Clone, PartialEq
  HashMap key: + Eq, Hash
  小型值类型:  + Copy
  序列化:      + Serialize, Deserialize
```

---

*上一篇：[03_核心概念_4_模式匹配match](./03_核心概念_4_模式匹配match.md)*
*下一篇：[03_核心概念_6_组合与嵌套](./03_核心概念_6_组合与嵌套.md)*
