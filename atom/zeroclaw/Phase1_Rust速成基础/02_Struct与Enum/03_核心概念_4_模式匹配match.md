# 核心概念 4：模式匹配 match

> **一句话定位：** match 是 Rust 最强大的控制流工具——它解构数据、绑定变量、强制穷尽，远超 TypeScript 的 switch。

---

## 1. match 表达式基础

### 1.1 match 是表达式，不是语句

```rust
// Rust: match 返回值，可以直接赋给变量
let label = match level {
    AutonomyLevel::ReadOnly => "safe",
    AutonomyLevel::Supervised => "moderate",
    AutonomyLevel::Full => "powerful",
};
```

```typescript
// TypeScript: switch 是语句，不能直接赋值
let label: string;
switch (level) {
    case "readonly": label = "safe"; break;
    case "supervised": label = "moderate"; break;
    case "full": label = "powerful"; break;
}
```

### 1.2 基本语法

```rust
enum AutonomyLevel { ReadOnly, Supervised, Full }

fn describe(level: &AutonomyLevel) -> &str {
    match level {
        AutonomyLevel::ReadOnly => "AI can only read",
        AutonomyLevel::Supervised => "AI acts with approval",
        AutonomyLevel::Full => "AI acts autonomously",
    }
}
```

**关键规则：**
- 每个分支（arm）用 `=>` 分隔模式和表达式
- 分支之间用 `,` 分隔
- 所有分支必须返回**相同类型**
- 没有 fall-through（不需要 break）

---

## 2. 穷尽性检查（Exhaustiveness）

### 2.1 编译器强制处理所有变体

```rust
fn describe(level: &AutonomyLevel) -> &str {
    match level {
        AutonomyLevel::ReadOnly => "safe",
        AutonomyLevel::Supervised => "moderate",
        // 忘了 Full？编译错误！
    }
}
// error[E0004]: non-exhaustive patterns: `AutonomyLevel::Full` not covered
```

### 2.2 新增变体时的威力

```rust
enum AutonomyLevel {
    ReadOnly, Supervised, Full,
    Custom(String),  // 新增！
}
// 项目中所有 match AutonomyLevel 的地方立即编译失败
// 编译器精确告诉你哪些地方需要更新
```

**TypeScript 对比**：旧代码中的 switch 不会报错，默默返回 `undefined`。

### 2.3 `_` 通配符（兜底）

```rust
match level {
    AutonomyLevel::Full => "dangerous",
    _ => "acceptable",  // 匹配所有其他情况
}
```

**注意**：`_` 会吞掉未来新增的变体，慎用！优先写全所有分支。

### 2.4 #[non_exhaustive]（库 API）

```rust
#[non_exhaustive]  // 告诉外部代码：未来可能新增变体
pub enum AutonomyLevel {
    ReadOnly, Supervised, Full,
}
// 外部 crate 的 match 必须加 _ => 兜底
```

---

## 3. 解构与变量绑定

### 3.1 解构 Enum 变体

```rust
enum ConversationMessage {
    Chat(ChatMessage),                          // 元组变体
    AssistantToolCalls { calls: Vec<String> },   // 结构体变体
    ToolResults(Vec<String>),                    // 元组变体
}

fn process(msg: &ConversationMessage) {
    match msg {
        // 元组变体：绑定内部数据到变量
        ConversationMessage::Chat(chat) => {
            println!("Chat: {}", chat.content);
        }
        // 结构体变体：解构命名字段
        ConversationMessage::AssistantToolCalls { calls } => {
            println!("Calling {} tools", calls.len());
        }
        // 元组变体 + 绑定
        ConversationMessage::ToolResults(results) => {
            println!("Got {} results", results.len());
        }
    }
}
```

### 3.2 解构嵌套 Struct

```rust
struct ChatMessage { role: String, content: String }

match msg {
    ConversationMessage::Chat(ChatMessage { role, content }) => {
        println!("[{}] {}", role, content);
    }
    _ => {}
}
```

### 3.3 忽略字段

```rust
// .. 忽略剩余字段
ConversationMessage::Chat(ChatMessage { content, .. }) => {
    println!("{}", content);
}

// _ 忽略特定值
ConversationMessage::ToolResults(_) => {
    println!("got some results");
}
```

---

## 4. 守卫条件（Match Guards）

在模式后加 `if` 条件：

```rust
fn check_tool(msg: &ConversationMessage) {
    match msg {
        ConversationMessage::AssistantToolCalls { calls } if calls.is_empty() => {
            println!("No tools called");
        }
        ConversationMessage::AssistantToolCalls { calls } if calls.len() > 5 => {
            println!("Too many tool calls: {}", calls.len());
        }
        ConversationMessage::AssistantToolCalls { calls } => {
            println!("Calling {} tools", calls.len());
        }
        _ => {}
    }
}
```

```rust
// 数值守卫
match temperature {
    t if t < 0.0 => println!("invalid"),
    t if t > 2.0 => println!("too creative"),
    t => println!("temperature {} is fine", t),
}
```

**TypeScript 对比**：switch 没有守卫，需要在 case 内部写 if-else。

---

## 5. @ 绑定、Or 模式、范围模式

### 5.1 @ 绑定：同时测试和捕获

```rust
match tokens_used {
    n @ 0 => println!("no tokens"),
    n @ 1..=100 => println!("light usage: {}", n),
    n @ 101..=1000 => println!("moderate: {}", n),
    n => println!("heavy: {}", n),
}
```

### 5.2 Or 模式：多值匹配

```rust
match status_code {
    200 | 201 | 204 => println!("success"),
    400 | 422 => println!("client error"),
    500 | 502 | 503 => println!("server error"),
    code => println!("other: {}", code),
}
```

### 5.3 范围模式

```rust
match score {
    0..=59 => "fail",
    60..=79 => "pass",
    80..=99 => "good",
    100 => "perfect",
    _ => "invalid",
}
```

---

## 6. if let 与 while let

### 6.1 if let：只关心一种模式

```rust
// 完整 match（啰嗦）
match msg {
    ConversationMessage::Chat(chat) => println!("{}", chat.content),
    _ => {}  // 不关心但必须写
}

// if let（简洁）
if let ConversationMessage::Chat(chat) = msg {
    println!("{}", chat.content);
}
```

### 6.2 if let + else

```rust
if let Some(value) = config.get("temperature") {
    println!("temperature = {}", value);
} else {
    println!("using default temperature");
}
```

### 6.3 while let：循环解构

```rust
let mut stack = vec![1, 2, 3];
while let Some(top) = stack.pop() {
    println!("popped: {}", top);
}
// 输出: 3, 2, 1
```

---

## 7. let-else（Rust 1.65+）

模式不匹配时提前返回，避免嵌套：

```rust
fn process_chat(msg: &ConversationMessage) -> String {
    // 不匹配就 return，匹配就绑定 chat
    let ConversationMessage::Chat(chat) = msg else {
        return "not a chat message".to_string();
    };
    format!("Processing: {}", chat.content)
}
```

**对比 if let：**

```rust
// if let 版本（多一层嵌套）
fn process_chat(msg: &ConversationMessage) -> String {
    if let ConversationMessage::Chat(chat) = msg {
        format!("Processing: {}", chat.content)
    } else {
        "not a chat message".to_string()
    }
}
```

let-else 的优势：**绑定的变量在后续代码中直接可用**，不增加缩进层级。

---

## 8. let-chains（Rust 2024 Edition）

用 `&&` 链接多个 `let` 模式，替代嵌套 if let：

```rust
// 旧写法：嵌套地狱
if let Some(msg) = messages.first() {
    if let ConversationMessage::Chat(chat) = msg {
        if chat.role == "user" {
            println!("First user message: {}", chat.content);
        }
    }
}

// let-chains（2024 Edition）：扁平化
if let Some(msg) = messages.first()
    && let ConversationMessage::Chat(chat) = msg
    && chat.role == "user"
{
    println!("First user message: {}", chat.content);
}
```

---

## 9. matches! 宏

快速布尔判断，不需要完整 match：

```rust
let is_full = matches!(level, AutonomyLevel::Full);

let is_error = matches!(event, ObserverEvent::Error { .. });

let is_success = matches!(status, 200 | 201 | 204);

// 带守卫
let is_heavy = matches!(tokens, n if n > 1000);
```

**TypeScript 对比**：`level === "full"` 但没有编译器保证拼写正确。

---

## 10. 完整可运行代码示例

```rust
#[derive(Debug)]
struct ChatMessage { role: String, content: String }

#[derive(Debug)]
enum ConversationMessage {
    Chat(ChatMessage),
    ToolCalls(Vec<String>),
    ToolResults(Vec<String>),
}

#[derive(Debug)]
enum AutonomyLevel { ReadOnly, Supervised, Full }

impl ConversationMessage {
    fn describe(&self) -> String {
        match self {
            ConversationMessage::Chat(chat) => format!("[{}] {}", chat.role, chat.content),
            ConversationMessage::ToolCalls(calls) => format!("Calling: {:?}", calls),
            ConversationMessage::ToolResults(results) => format!("Results: {:?}", results),
        }
    }
}

fn check_autonomy(level: &AutonomyLevel) -> &str {
    match level {
        AutonomyLevel::ReadOnly => "safe",
        AutonomyLevel::Supervised => "moderate",
        AutonomyLevel::Full => "powerful",
    }
}

fn main() {
    // 1. 基本 match + 解构
    let msgs: Vec<ConversationMessage> = vec![
        ConversationMessage::Chat(ChatMessage {
            role: "user".into(), content: "Hello".into(),
        }),
        ConversationMessage::ToolCalls(vec!["search".into(), "write".into()]),
        ConversationMessage::ToolResults(vec!["found 3 results".into()]),
    ];
    for msg in &msgs {
        println!("{}", msg.describe());
    }

    // 2. if let
    if let ConversationMessage::Chat(chat) = &msgs[0] {
        println!("First message role: {}", chat.role);
    }

    // 3. let-else
    let ConversationMessage::ToolCalls(calls) = &msgs[1] else {
        panic!("expected tool calls");
    };
    println!("Tool calls: {:?}", calls);

    // 4. 守卫
    for msg in &msgs {
        match msg {
            ConversationMessage::ToolCalls(c) if c.len() > 3 => println!("too many calls"),
            ConversationMessage::ToolCalls(c) => println!("{} calls", c.len()),
            _ => {}
        }
    }

    // 5. @ 绑定 + 范围
    let tokens = 150u32;
    match tokens {
        n @ 0..=100 => println!("light: {}", n),
        n @ 101..=500 => println!("moderate: {}", n),
        n => println!("heavy: {}", n),
    }

    // 6. matches! 宏
    let level = AutonomyLevel::Full;
    println!("is full? {}", matches!(level, AutonomyLevel::Full));
    println!("autonomy: {}", check_autonomy(&level));

    // 7. Or 模式
    let code = 200u16;
    let status = match code {
        200 | 201 | 204 => "success",
        400 | 422 => "client error",
        500 | 502 | 503 => "server error",
        _ => "unknown",
    };
    println!("HTTP {}: {}", code, status);
}
```

**编译运行：** `rustc main.rs && ./main`

---

## 速查卡

```
match 基础：
  match value { Pattern => expr, }     返回值，强制穷尽
  _ =>                                 通配符兜底

解构：
  Enum::Variant(x)                     元组变体绑定
  Enum::Variant { field, .. }          结构体变体 + 忽略其余
  Enum::Variant(_)                     忽略内部数据

守卫：
  pat if condition => expr             附加条件

@ 绑定：
  n @ 1..=100 => ...                   同时测试范围和捕获值

Or / 范围：
  1 | 2 | 3 => ...                     多值匹配
  1..=100 => ...                       范围匹配

简化匹配：
  if let Pat = expr { }               只关心一种模式
  while let Pat = expr { }            循环解构
  let Pat = expr else { return; };    不匹配就提前返回（1.65+）
  if let P = a && let Q = b { }       链式匹配（2024 Edition）

快速判断：
  matches!(value, Pattern)             返回 bool
  matches!(v, Pat if guard)            带守卫
```

---

*上一篇：[03_核心概念_3_Enum与变体类型](./03_核心概念_3_Enum与变体类型.md)*
*下一篇：[03_核心概念_5_derive宏与属性](./03_核心概念_5_derive宏与属性.md)*
