# 核心概念 3：Enum 与变体类型

> **一句话定位：** Rust 的 enum 不是 C/Java 那种简单的数字常量，而是能携带不同数据的"标签联合体"，配合 match 实现编译期穷尽检查，是 Rust 类型安全的核心武器。

---

## 1. Enum 概念：为什么 Rust enum 比 C/Java enum 强大 100 倍

### 1.1 其他语言的 enum 有多弱

```java
// Java enum -- 本质就是数字常量套了个名字
enum Color { RED, GREEN, BLUE }
// RED = 0, GREEN = 1, BLUE = 2
// 不能携带额外数据
```

```typescript
// TypeScript enum -- 同样是数字/字符串映射
enum Color { Red = "RED", Green = "GREEN", Blue = "BLUE" }
// 不能给某个变体附带不同类型的数据
```

### 1.2 Rust enum = 可携带数据的标签联合

Rust 的 enum 每个变体（variant）可以携带不同类型、不同数量的数据：

```rust
// 一个 enum，三种完全不同的数据结构
enum ConversationMessage {
    // 变体 1：包装一个 ChatMessage 结构体
    Chat(ChatMessage),

    // 变体 2：携带三个命名字段
    AssistantToolCalls {
        text: Option<String>,
        tool_calls: Vec<ToolCall>,
        reasoning_content: Option<String>,
    },

    // 变体 3：包装一个 Vec
    ToolResults(Vec<ToolResultMessage>),
}
```

> **这在 TypeScript 中需要用 discriminated union 来模拟：**

```typescript
// TypeScript 需要手动维护 type 字段来区分
type ConversationMessage =
    | { type: "chat"; message: ChatMessage }
    | { type: "toolCalls"; text?: string; toolCalls: ToolCall[]; reasoningContent?: string }
    | { type: "toolResults"; results: ToolResultMessage[] };
```

### 1.3 关键区别

| 特性               | C/Java enum    | TypeScript union | Rust enum       |
|--------------------|----------------|------------------|-----------------|
| 携带数据           | 不能            | 能（手动 tag）    | 能（编译器管理） |
| 编译期穷尽检查     | 不能            | 有限              | 完整保证        |
| 每个变体数据不同   | 不能            | 能                | 能              |
| 忘记处理某个变体   | 运行时 bug      | 部分检查          | 编译不过        |
| 可以有方法         | Java 可以       | 不能              | 可以            |

> **日常生活类比：** C/Java 的 enum 就像一排编了号的空储物柜（1号、2号、3号）。Rust 的 enum 就像一排不同大小的快递包裹 -- 1号是信封（放一张纸），2号是纸箱（放三件东西），3号是麻袋（放一堆东西）。每个"柜子"能装完全不同的内容。

> **前端开发类比：** TypeScript 的 `discriminated union` 需要你手动加 `type` 字段做区分，稍不注意就忘了处理某个 case。Rust 的 enum 让编译器帮你检查 -- 你忘处理任何一个变体，代码直接编译不过。

---

## 2. 三种变体类型详解

### 2.1 Unit Variant（单元变体）-- 无数据

最简单的变体，不携带任何数据，类似 C 的 enum：

```rust
// ZeroClaw 的 AutonomyLevel -- Agent 的自主等级
#[derive(Debug, Clone, PartialEq)]
enum AutonomyLevel {
    ReadOnly,       // 只读模式：只能观察，不能执行
    #[default]
    Supervised,     // 监督模式：执行前需要人类批准（默认值）
    Full,           // 完全自主：自行决策并执行
}
```

```typescript
// TypeScript 等价
type AutonomyLevel = "readonly" | "supervised" | "full";
// 但 TypeScript 版本没有默认值的概念
```

**使用场景：** 表示有限的、不需要附加数据的状态或模式。

```rust
fn check_permission(level: &AutonomyLevel) -> bool {
    match level {
        AutonomyLevel::ReadOnly => {
            println!("Read-only mode: cannot execute commands");
            false
        }
        AutonomyLevel::Supervised => {
            println!("Supervised mode: requesting approval...");
            true // 简化：假设获得批准
        }
        AutonomyLevel::Full => {
            println!("Full autonomy: executing directly");
            true
        }
    }
}
```

### 2.2 Tuple Variant（元组变体）-- 匿名数据

变体携带未命名的数据，用位置区分：

```rust
// ZeroClaw 的 ConversationMessage 部分变体
enum MessageType {
    // 元组变体：包装一个 ChatMessage
    Chat(ChatMessage),

    // 元组变体：包装一个 Vec
    ToolResults(Vec<ToolResult>),

    // 元组变体：可以有多个字段
    Error(u32, String),  // (错误码, 错误消息)
}

struct ChatMessage {
    role: String,
    content: String,
}

struct ToolResult {
    tool_name: String,
    output: String,
}
```

**使用场景：** 变体只需要包装一个或少量值，字段含义从上下文就能看出来。

```rust
fn describe_message(msg: &MessageType) -> String {
    match msg {
        MessageType::Chat(chat) => {
            format!("[{}]: {}", chat.role, chat.content)
        }
        MessageType::ToolResults(results) => {
            format!("Tool results: {} items", results.len())
        }
        MessageType::Error(code, message) => {
            format!("Error {}: {}", code, message)
        }
    }
}
```

### 2.3 Struct Variant（结构体变体）-- 命名数据

变体携带命名字段，就像内嵌了一个匿名 struct：

```rust
// ZeroClaw 的 ConversationMessage 的 AssistantToolCalls 变体
enum ConversationMessage {
    Chat(ChatMessage),

    // 结构体变体：每个字段有名字
    AssistantToolCalls {
        text: Option<String>,
        tool_calls: Vec<ToolCall>,
        reasoning_content: Option<String>,
    },

    ToolResults(Vec<ToolResultMessage>),
}

#[derive(Debug, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Clone)]
struct ToolCall {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Debug, Clone)]
struct ToolResultMessage {
    tool_call_id: String,
    output: String,
}
```

**使用场景：** 变体携带多个字段，需要用名字区分以提高可读性。

```rust
fn process_message(msg: &ConversationMessage) {
    match msg {
        ConversationMessage::Chat(chat) => {
            println!("[{}]: {}", chat.role, chat.content);
        }

        // 结构体变体的解构 -- 字段用名字匹配
        ConversationMessage::AssistantToolCalls {
            text,
            tool_calls,
            reasoning_content,
        } => {
            if let Some(t) = text {
                println!("Assistant: {}", t);
            }
            println!("Tool calls: {}", tool_calls.len());
            if let Some(reasoning) = reasoning_content {
                println!("Reasoning: {}", reasoning);
            }
        }

        ConversationMessage::ToolResults(results) => {
            for result in results {
                println!("Tool {}: {}", result.tool_call_id, result.output);
            }
        }
    }
}
```

### 2.4 三种变体对比

| 变体类型   | 语法                           | 数据   | 何时使用              |
|-----------|--------------------------------|--------|----------------------|
| Unit      | `Variant`                      | 无     | 纯状态/标签            |
| Tuple     | `Variant(T1, T2)`              | 位置   | 包装 1-2 个值          |
| Struct    | `Variant { a: T1, b: T2 }`    | 命名   | 多字段且需要可读性      |

---

## 3. ZeroClaw 核心 Enum 实例

### 3.1 AutonomyLevel -- 安全等级（纯 Unit 变体）

```rust
#[derive(Debug, Clone, Copy, PartialEq, Default)]
enum AutonomyLevel {
    ReadOnly,
    #[default]        // Supervised 是默认值
    Supervised,
    Full,
}

impl AutonomyLevel {
    fn can_execute(&self) -> bool {
        match self {
            AutonomyLevel::ReadOnly => false,
            AutonomyLevel::Supervised => true,
            AutonomyLevel::Full => true,
        }
    }

    fn requires_approval(&self) -> bool {
        matches!(self, AutonomyLevel::Supervised)
    }
}

fn main() {
    let level = AutonomyLevel::default();
    println!("{:?}", level);  // Supervised
    println!("Can execute: {}", level.can_execute());       // true
    println!("Needs approval: {}", level.requires_approval()); // true
}
```

> **`#[default]` 属性：** 配合 `derive(Default)` 使用，指定某个变体为默认值。这是 Rust 的声明式风格 -- 不用写代码，用属性标注即可。

### 3.2 MemoryCategory -- 混合变体（Unit + Data）

```rust
use std::fmt;

#[derive(Debug, Clone)]
enum MemoryCategory {
    Core,                // Unit: 核心记忆
    Daily,               // Unit: 日常记忆
    Conversation,        // Unit: 对话记忆
    Custom(String),      // Tuple: 自定义类别（携带名称）
}

// 自定义 Display -- 控制输出格式
impl fmt::Display for MemoryCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryCategory::Core => write!(f, "core"),
            MemoryCategory::Daily => write!(f, "daily"),
            MemoryCategory::Conversation => write!(f, "conversation"),
            MemoryCategory::Custom(name) => write!(f, "custom:{}", name),
        }
    }
}

fn main() {
    let categories = vec![
        MemoryCategory::Core,
        MemoryCategory::Daily,
        MemoryCategory::Custom("security_audit".to_string()),
    ];

    for cat in &categories {
        println!("Memory: {}", cat);
    }
    // 输出:
    // Memory: core
    // Memory: daily
    // Memory: custom:security_audit
}
```

> **设计要点：** `Custom(String)` 变体让 enum 有了"开放扩展点"。前三个是预定义的类别，`Custom` 允许用户定义自己的类别，无需修改 enum 定义。

### 3.3 ObserverEvent -- 事件系统（11 个变体展示表达力）

```rust
#[derive(Debug, Clone)]
enum ObserverEvent {
    // Unit 变体 -- 简单事件信号
    AgentStarted,
    AgentStopped,
    ToolExecutionStarted,

    // Tuple 变体 -- 包装单个数据
    Error(String),

    // Struct 变体 -- 复杂事件数据
    MessageReceived {
        channel: String,
        sender: String,
        content: String,
    },

    ToolExecutionCompleted {
        tool_name: String,
        duration_ms: u64,
        success: bool,
    },

    MemoryStored {
        category: String,
        key: String,
        size_bytes: usize,
    },

    SecurityAlert {
        level: String,
        description: String,
        source: String,
    },

    TokenUsage {
        prompt_tokens: u32,
        completion_tokens: u32,
        model: String,
    },

    StateChanged {
        from: String,
        to: String,
    },

    Custom {
        event_type: String,
        payload: String,
    },
}

impl ObserverEvent {
    fn severity(&self) -> &str {
        match self {
            ObserverEvent::Error(_) => "error",
            ObserverEvent::SecurityAlert { .. } => "critical",
            ObserverEvent::AgentStopped => "warning",
            _ => "info",  // 其余都是 info 级别
        }
    }

    fn event_name(&self) -> &str {
        match self {
            ObserverEvent::AgentStarted => "agent.started",
            ObserverEvent::AgentStopped => "agent.stopped",
            ObserverEvent::ToolExecutionStarted => "tool.started",
            ObserverEvent::ToolExecutionCompleted { .. } => "tool.completed",
            ObserverEvent::MessageReceived { .. } => "message.received",
            ObserverEvent::MemoryStored { .. } => "memory.stored",
            ObserverEvent::SecurityAlert { .. } => "security.alert",
            ObserverEvent::TokenUsage { .. } => "token.usage",
            ObserverEvent::StateChanged { .. } => "state.changed",
            ObserverEvent::Error(_) => "error",
            ObserverEvent::Custom { event_type, .. } => "custom",
        }
    }
}
```

> **这展示了 enum 的强大表达力：** 一个 enum 就能表达整个事件系统，每种事件携带完全不同的数据。如果用 TypeScript，你需要定义 11 个 interface + 一个 union type + 手动维护 type 字段。

### 3.4 更多简单 Enum 示例

```rust
// 命令风险等级
#[derive(Debug, Clone, Copy, PartialEq)]
enum CommandRiskLevel {
    Safe,       // 安全（ls, cat）
    Moderate,   // 中等（mkdir, cp）
    High,       // 高风险（rm, chmod）
    Critical,   // 极高（rm -rf, dd）
}

// 工具操作类型
#[derive(Debug, Clone)]
enum ToolOperation {
    Read,
    Write,
    Execute,
    Delete,
}

// 引号解析状态（状态机模式）
#[derive(Debug, Clone, Copy, PartialEq)]
enum QuoteState {
    Normal,
    InSingleQuote,
    InDoubleQuote,
    Escaped,
}
```

---

## 4. Enum 上的方法

Enum 和 struct 一样可以有 `impl` 块。上面的 `ObserverEvent` 已经展示了这一点。再看一个更完整的例子：

```rust
#[derive(Debug)]
enum Shape {
    Circle(f64),
    Rectangle(f64, f64),
    Triangle { base: f64, height: f64 },
}

impl Shape {
    // 关联函数（构造器）
    fn unit_circle() -> Self {
        Shape::Circle(1.0)
    }

    fn square(side: f64) -> Self {
        Shape::Rectangle(side, side)
    }

    // &self 方法
    fn area(&self) -> f64 {
        match self {
            Shape::Circle(r) => std::f64::consts::PI * r * r,
            Shape::Rectangle(w, h) => w * h,
            Shape::Triangle { base, height } => 0.5 * base * height,
        }
    }

    fn perimeter(&self) -> f64 {
        match self {
            Shape::Circle(r) => 2.0 * std::f64::consts::PI * r,
            Shape::Rectangle(w, h) => 2.0 * (w + h),
            Shape::Triangle { base, height } => {
                let hypotenuse = (base * base + height * height).sqrt();
                base + height + hypotenuse
            }
        }
    }

    // 判断方法
    fn is_circle(&self) -> bool {
        matches!(self, Shape::Circle(_))
    }
}

fn main() {
    let shapes = vec![
        Shape::unit_circle(),
        Shape::square(5.0),
        Shape::Triangle { base: 3.0, height: 4.0 },
    ];

    for shape in &shapes {
        println!("{:?}: area = {:.2}, perimeter = {:.2}",
            shape, shape.area(), shape.perimeter());
    }
}
```

> **`matches!` 宏：** 快速判断一个值是否匹配某个模式，返回 `bool`。比写完整的 `match` 简洁很多：

```rust
// 等价于：
fn is_circle(&self) -> bool {
    match self {
        Shape::Circle(_) => true,
        _ => false,
    }
}
```

---

## 5. Option\<T\> 和 Result\<T, E\>：标准库最重要的两个 Enum

### 5.1 Option\<T\> -- 干掉 null

Rust 没有 `null`、`nil`、`undefined`、`None`...等等，Rust 有 `None`，但它不是"空"，而是 `Option` enum 的一个变体：

```rust
// 标准库定义（你不需要自己写）
enum Option<T> {
    Some(T),   // 有值
    None,      // 无值
}
```

**为什么这比 null 安全？** 因为编译器会强制你处理 None 的情况：

```rust
fn find_agent(name: &str) -> Option<String> {
    match name {
        "zeroclaw" => Some("ZeroClaw AI Agent v2.0".to_string()),
        "scout" => Some("Scout Recon Agent v1.0".to_string()),
        _ => None,
    }
}

fn main() {
    let result = find_agent("zeroclaw");

    // 必须处理两种情况，否则编译不过
    match result {
        Some(agent) => println!("Found: {}", agent),
        None => println!("Agent not found"),
    }

    // if let -- 只关心 Some 的情况
    if let Some(agent) = find_agent("scout") {
        println!("Also found: {}", agent);
    }

    // unwrap_or -- 提供默认值
    let name = find_agent("unknown").unwrap_or("default agent".to_string());
    println!("Agent: {}", name);

    // map -- 变换 Some 内的值
    let upper = find_agent("zeroclaw").map(|s| s.to_uppercase());
    println!("Upper: {:?}", upper);  // Some("ZEROCLAW AI AGENT V2.0")
}
```

**TypeScript 对照：**

```typescript
// TypeScript
function findAgent(name: string): string | null {
    if (name === "zeroclaw") return "ZeroClaw AI Agent v2.0";
    return null;
}

const result = findAgent("zeroclaw");
// 危险：TypeScript 不会强制你检查 null
console.log(result.toUpperCase()); // 可能运行时报错！

// 安全写法（但编译器不强制）
if (result !== null) {
    console.log(result.toUpperCase());
}
```

```rust
// Rust
let result = find_agent("zeroclaw");
// 编译错误：不能直接对 Option<String> 调用 String 的方法
// println!("{}", result.to_uppercase()); // 编译不过！
// 必须先"解包"
```

### 5.2 Result\<T, E\> -- 优雅的错误处理

```rust
// 标准库定义
enum Result<T, E> {
    Ok(T),    // 成功，携带结果
    Err(E),   // 失败，携带错误信息
}
```

```rust
#[derive(Debug)]
enum AgentError {
    NotFound(String),
    PermissionDenied(String),
    ConfigInvalid { field: String, reason: String },
}

fn create_agent(name: &str, level: &str) -> Result<String, AgentError> {
    if name.is_empty() {
        return Err(AgentError::ConfigInvalid {
            field: "name".to_string(),
            reason: "cannot be empty".to_string(),
        });
    }

    if level == "full" {
        return Err(AgentError::PermissionDenied(
            "Full autonomy requires admin approval".to_string(),
        ));
    }

    Ok(format!("Agent '{}' created with level '{}'", name, level))
}

fn main() {
    // match 处理
    match create_agent("zeroclaw", "supervised") {
        Ok(msg) => println!("Success: {}", msg),
        Err(AgentError::NotFound(name)) => println!("Not found: {}", name),
        Err(AgentError::PermissionDenied(reason)) => println!("Denied: {}", reason),
        Err(AgentError::ConfigInvalid { field, reason }) => {
            println!("Config error in '{}': {}", field, reason);
        }
    }

    // ? 操作符 -- 错误自动传播（在返回 Result 的函数中使用）
    fn setup_agents() -> Result<(), AgentError> {
        let agent1 = create_agent("zeroclaw", "supervised")?; // ? = 出错就 return Err
        println!("{}", agent1);
        let agent2 = create_agent("scout", "readonly")?;
        println!("{}", agent2);
        Ok(())
    }

    if let Err(e) = setup_agents() {
        println!("Setup failed: {:?}", e);
    }
}
```

**TypeScript 对照：**

```typescript
// TypeScript 用 try-catch（运行时错误处理）
function createAgent(name: string, level: string): string {
    if (!name) throw new Error("name cannot be empty");
    if (level === "full") throw new Error("Permission denied");
    return `Agent '${name}' created`;
}

// 问题：调用方可能忘记 try-catch，运行时才崩溃
// Rust 的 Result 在编译期就强制你处理错误
```

### 5.3 Option 和 Result 速查

```rust
// ===== Option 常用方法 =====
let x: Option<i32> = Some(42);

x.is_some();              // true
x.is_none();              // false
x.unwrap();               // 42（None 时 panic！慎用）
x.unwrap_or(0);           // 42（None 时返回 0）
x.unwrap_or_default();    // 42（None 时返回类型默认值）
x.map(|v| v * 2);         // Some(84)
x.filter(|v| *v > 50);    // None（42 不大于 50）
x.and_then(|v| if v > 0 { Some(v) } else { None }); // Some(42)

// ===== Result 常用方法 =====
let r: Result<i32, String> = Ok(42);

r.is_ok();                // true
r.is_err();               // false
r.unwrap();               // 42（Err 时 panic！）
r.unwrap_or(0);           // 42
r.map(|v| v * 2);         // Ok(84)
r.map_err(|e| format!("Error: {}", e)); // 转换错误类型
r.ok();                   // Some(42) -- Result -> Option
```

---

## 6. 枚举的内存布局（简要）

### 6.1 基本布局：判别符 + 最大变体

每个 enum 值在内存中由两部分组成：

```
[判别符 | 数据区域]
 1-8字节   最大变体的大小
```

```rust
use std::mem::size_of;

enum Small {
    A,          // 0 字节数据
    B(u8),      // 1 字节数据
    C(u16),     // 2 字节数据（最大变体）
}
// 内存 = 判别符(1字节) + 对齐 + 最大变体(2字节) = 4 字节

fn main() {
    println!("Small: {} bytes", size_of::<Small>());  // 4
}
```

### 6.2 Niche 优化：Option<&T> 零开销

Rust 编译器非常聪明 -- 它知道引用永远不会是 null（地址 0），所以 `Option<&T>` 用地址 0 表示 None，不需要额外的判别符：

```rust
use std::mem::size_of;

fn main() {
    // 引用是 8 字节（64 位系统）
    println!("&i32: {} bytes", size_of::<&i32>());           // 8

    // Option<&i32> 也是 8 字节！零开销！
    println!("Option<&i32>: {} bytes", size_of::<Option<&i32>>()); // 8

    // 对比：Option<i32> 需要额外空间
    println!("i32: {} bytes", size_of::<i32>());             // 4
    println!("Option<i32>: {} bytes", size_of::<Option<i32>>()); // 8
}
```

> **日常生活类比：** Niche 优化就像快递柜的妙用 -- 如果快递柜编号永远不可能是 0，那"柜号 0"就可以用来表示"这个柜子是空的"。不需要额外贴一张"空/满"标签。

### 6.3 大变体用 Box 优化

如果一个变体特别大，会浪费其他小变体的内存：

```rust
use std::mem::size_of;

// 不好：BigData 占 1000 字节，但 Simple 只需要 0 字节
// 每个 BadEnum 值都占 1000+ 字节
enum BadEnum {
    Simple,
    BigData([u8; 1000]),
}

// 好：用 Box 把大数据放到堆上
// 每个 GoodEnum 值只占 8 字节（一个指针大小）
enum GoodEnum {
    Simple,
    BigData(Box<[u8; 1000]>),
}

fn main() {
    println!("BadEnum: {} bytes", size_of::<BadEnum>());   // 1004
    println!("GoodEnum: {} bytes", size_of::<GoodEnum>()); // 16
}
```

> **前端类比：** Box 就像 React 的懒加载 -- 不把所有组件代码打包进主 bundle，而是用一个小小的"指针"（动态 import）按需加载大组件。

---

## 7. TypeScript 对照代码

完整对比 TypeScript discriminated union 和 Rust enum 的使用：

```typescript
// ========== TypeScript ==========

// 1. 定义（手动维护 type 字段）
type AutonomyLevel = "readonly" | "supervised" | "full";

interface ChatMsg { role: string; content: string }
interface ToolCallData { text?: string; toolCalls: any[]; reasoning?: string }
interface ToolResultData { toolCallId: string; output: string }

type ConversationMessage =
    | { type: "chat"; data: ChatMsg }
    | { type: "toolCalls"; data: ToolCallData }
    | { type: "toolResults"; data: ToolResultData[] };

// 2. 使用
function processMessage(msg: ConversationMessage): string {
    switch (msg.type) {
        case "chat":
            return `[${msg.data.role}]: ${msg.data.content}`;
        case "toolCalls":
            return `Tool calls: ${msg.data.toolCalls.length}`;
        case "toolResults":
            return `Results: ${msg.data.length}`;
        // 如果忘了某个 case，TypeScript 可能不报错！
    }
}

// 3. 错误处理
function findAgent(name: string): string | null {
    if (name === "zeroclaw") return "found";
    return null;
}
// 调用方可以忘记检查 null
```

```rust
// ========== Rust ==========

// 1. 定义（编译器管理标签）
#[derive(Debug, Clone, Copy, PartialEq, Default)]
enum AutonomyLevel {
    ReadOnly,
    #[default]
    Supervised,
    Full,
}

#[derive(Debug, Clone)]
struct ChatMsg { role: String, content: String }

#[derive(Debug, Clone)]
struct ToolCall { id: String, name: String }

#[derive(Debug, Clone)]
struct ToolResultData { tool_call_id: String, output: String }

#[derive(Debug, Clone)]
enum ConversationMessage {
    Chat(ChatMsg),
    ToolCalls {
        text: Option<String>,
        tool_calls: Vec<ToolCall>,
        reasoning: Option<String>,
    },
    ToolResults(Vec<ToolResultData>),
}

// 2. 使用
fn process_message(msg: &ConversationMessage) -> String {
    match msg {
        ConversationMessage::Chat(chat) => {
            format!("[{}]: {}", chat.role, chat.content)
        }
        ConversationMessage::ToolCalls { text, tool_calls, .. } => {
            format!("Tool calls: {}", tool_calls.len())
        }
        ConversationMessage::ToolResults(results) => {
            format!("Results: {}", results.len())
        }
        // 如果忘了某个变体 -> 编译错误！
    }
}

// 3. 错误处理
fn find_agent(name: &str) -> Option<String> {
    match name {
        "zeroclaw" => Some("found".to_string()),
        _ => None,
    }
}
// 调用方必须处理 None 才能使用内部值
```

---

## 8. 完整可运行示例

```rust
use std::fmt;

// ===== 1. Unit 变体 =====
#[derive(Debug, Clone, Copy, PartialEq, Default)]
enum AutonomyLevel {
    ReadOnly,
    #[default]
    Supervised,
    Full,
}

impl fmt::Display for AutonomyLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AutonomyLevel::ReadOnly => write!(f, "read-only"),
            AutonomyLevel::Supervised => write!(f, "supervised"),
            AutonomyLevel::Full => write!(f, "full"),
        }
    }
}

impl AutonomyLevel {
    fn can_execute(&self) -> bool {
        !matches!(self, AutonomyLevel::ReadOnly)
    }
}

// ===== 2. 混合变体 =====
#[derive(Debug, Clone)]
enum MemoryCategory {
    Core,
    Daily,
    Conversation,
    Custom(String),
}

impl fmt::Display for MemoryCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryCategory::Core => write!(f, "core"),
            MemoryCategory::Daily => write!(f, "daily"),
            MemoryCategory::Conversation => write!(f, "conversation"),
            MemoryCategory::Custom(name) => write!(f, "custom:{}", name),
        }
    }
}

// ===== 3. 复杂变体 =====
#[derive(Debug, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

impl ChatMessage {
    fn system(content: impl Into<String>) -> Self {
        ChatMessage { role: "system".to_string(), content: content.into() }
    }
    fn user(content: impl Into<String>) -> Self {
        ChatMessage { role: "user".to_string(), content: content.into() }
    }
    fn assistant(content: impl Into<String>) -> Self {
        ChatMessage { role: "assistant".to_string(), content: content.into() }
    }
}

#[derive(Debug, Clone)]
struct ToolCall {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Debug, Clone)]
struct ToolResultMessage {
    tool_call_id: String,
    output: String,
}

#[derive(Debug, Clone)]
enum ConversationMessage {
    Chat(ChatMessage),
    AssistantToolCalls {
        text: Option<String>,
        tool_calls: Vec<ToolCall>,
        reasoning_content: Option<String>,
    },
    ToolResults(Vec<ToolResultMessage>),
}

impl ConversationMessage {
    fn summary(&self) -> String {
        match self {
            ConversationMessage::Chat(msg) => {
                format!("[{}] {}", msg.role, &msg.content[..msg.content.len().min(50)])
            }
            ConversationMessage::AssistantToolCalls { tool_calls, .. } => {
                let names: Vec<&str> = tool_calls.iter().map(|t| t.name.as_str()).collect();
                format!("[tool_calls] {}", names.join(", "))
            }
            ConversationMessage::ToolResults(results) => {
                format!("[tool_results] {} results", results.len())
            }
        }
    }

    fn is_from_user(&self) -> bool {
        matches!(self, ConversationMessage::Chat(msg) if msg.role == "user")
    }
}

// ===== 4. 简单状态 Enum =====
#[derive(Debug, Clone, Copy, PartialEq)]
enum CommandRiskLevel {
    Safe,
    Moderate,
    High,
    Critical,
}

impl CommandRiskLevel {
    fn from_command(cmd: &str) -> Self {
        match cmd {
            "ls" | "cat" | "echo" => CommandRiskLevel::Safe,
            "mkdir" | "cp" | "mv" => CommandRiskLevel::Moderate,
            "rm" | "chmod" | "chown" => CommandRiskLevel::High,
            _ if cmd.contains("rm -rf") => CommandRiskLevel::Critical,
            _ => CommandRiskLevel::Moderate,
        }
    }

    fn requires_approval(&self, autonomy: &AutonomyLevel) -> bool {
        match (self, autonomy) {
            (_, AutonomyLevel::ReadOnly) => true,         // 只读模式全部需要审批
            (CommandRiskLevel::Safe, _) => false,          // 安全命令不需要审批
            (_, AutonomyLevel::Full) => false,             // 完全自主不需要审批
            _ => true,                                     // 其余需要审批
        }
    }
}

// ===== main =====
fn main() {
    println!("===== 1. AutonomyLevel (Unit 变体) =====");
    let level = AutonomyLevel::default();
    println!("Default: {} (can execute: {})", level, level.can_execute());

    let all_levels = [AutonomyLevel::ReadOnly, AutonomyLevel::Supervised, AutonomyLevel::Full];
    for l in &all_levels {
        println!("  {} -> can_execute: {}", l, l.can_execute());
    }

    println!("\n===== 2. MemoryCategory (混合变体) =====");
    let categories = vec![
        MemoryCategory::Core,
        MemoryCategory::Daily,
        MemoryCategory::Conversation,
        MemoryCategory::Custom("security_audit".to_string()),
        MemoryCategory::Custom("project_alpha".to_string()),
    ];
    for cat in &categories {
        println!("  Memory category: {}", cat);
    }

    println!("\n===== 3. ConversationMessage (复杂变体) =====");
    let messages: Vec<ConversationMessage> = vec![
        ConversationMessage::Chat(ChatMessage::system("You are a security AI.")),
        ConversationMessage::Chat(ChatMessage::user("Scan port 443.")),
        ConversationMessage::AssistantToolCalls {
            text: Some("I'll scan port 443 for you.".to_string()),
            tool_calls: vec![
                ToolCall {
                    id: "call_1".to_string(),
                    name: "port_scan".to_string(),
                    arguments: r#"{"port": 443}"#.to_string(),
                },
            ],
            reasoning_content: Some("User wants a security scan.".to_string()),
        },
        ConversationMessage::ToolResults(vec![
            ToolResultMessage {
                tool_call_id: "call_1".to_string(),
                output: "Port 443: OPEN (HTTPS)".to_string(),
            },
        ]),
        ConversationMessage::Chat(
            ChatMessage::assistant("Port 443 is open and serving HTTPS.")
        ),
    ];

    for msg in &messages {
        println!("  {}", msg.summary());
    }

    // 统计用户消息
    let user_msg_count = messages.iter().filter(|m| m.is_from_user()).count();
    println!("  User messages: {}", user_msg_count);

    println!("\n===== 4. Option<T> (空值安全) =====");
    let names = vec!["zeroclaw", "scout", "phantom"];
    for name in &names {
        match find_agent_description(name) {
            Some(desc) => println!("  {} -> {}", name, desc),
            None => println!("  {} -> not found", name),
        }
    }

    println!("\n===== 5. Result<T, E> (错误处理) =====");
    let commands = vec!["ls", "rm", "rm -rf /"];
    for cmd in &commands {
        match execute_command(cmd, &AutonomyLevel::Supervised) {
            Ok(output) => println!("  {} -> OK: {}", cmd, output),
            Err(reason) => println!("  {} -> DENIED: {}", cmd, reason),
        }
    }

    println!("\n===== 6. Enum 内存大小 =====");
    println!("  AutonomyLevel: {} bytes", std::mem::size_of::<AutonomyLevel>());
    println!("  MemoryCategory: {} bytes", std::mem::size_of::<MemoryCategory>());
    println!("  ConversationMessage: {} bytes", std::mem::size_of::<ConversationMessage>());
    println!("  Option<&str>: {} bytes", std::mem::size_of::<Option<&str>>());
    println!("  &str: {} bytes", std::mem::size_of::<&str>());
    println!("  (Niche 优化: Option<&str> == &str 大小!)");
}

// ===== 辅助函数 =====
fn find_agent_description(name: &str) -> Option<String> {
    match name {
        "zeroclaw" => Some("ZeroClaw AI Security Agent v2.0".to_string()),
        "scout" => Some("Scout Reconnaissance Agent v1.0".to_string()),
        _ => None,
    }
}

fn execute_command(cmd: &str, autonomy: &AutonomyLevel) -> Result<String, String> {
    let risk = CommandRiskLevel::from_command(cmd);

    if risk.requires_approval(autonomy) {
        return Err(format!(
            "Command '{}' requires approval (risk: {:?}, autonomy: {})",
            cmd, risk, autonomy
        ));
    }

    Ok(format!("Executed '{}' successfully", cmd))
}
```

**运行输出：**

```
===== 1. AutonomyLevel (Unit 变体) =====
Default: supervised (can execute: true)
  read-only -> can_execute: false
  supervised -> can_execute: true
  full -> can_execute: true

===== 2. MemoryCategory (混合变体) =====
  Memory category: core
  Memory category: daily
  Memory category: conversation
  Memory category: custom:security_audit
  Memory category: custom:project_alpha

===== 3. ConversationMessage (复杂变体) =====
  [system] You are a security AI.
  [user] Scan port 443.
  [tool_calls] port_scan
  [tool_results] 1 results
  [assistant] Port 443 is open and serving HTTPS.
  User messages: 1

===== 4. Option<T> (空值安全) =====
  zeroclaw -> ZeroClaw AI Security Agent v2.0
  scout -> Scout Reconnaissance Agent v1.0
  phantom -> not found

===== 5. Result<T, E> (错误处理) =====
  ls -> OK: Executed 'ls' successfully
  rm -> DENIED: Command 'rm' requires approval (risk: High, autonomy: supervised)
  rm -rf / -> DENIED: Command 'rm -rf /' requires approval (risk: Critical, autonomy: supervised)

===== 6. Enum 内存大小 =====
  AutonomyLevel: 1 bytes
  MemoryCategory: 32 bytes
  ConversationMessage: 104 bytes
  Option<&str>: 16 bytes
  &str: 16 bytes
  (Niche 优化: Option<&str> == &str 大小!)
```

---

## 速记卡片

| 概念                | Rust                              | TypeScript                       |
|---------------------|-----------------------------------|----------------------------------|
| 简单枚举            | `enum Color { Red, Green }`       | `type Color = "red" \| "green"`  |
| 携带数据            | `enum Msg { Text(String) }`       | `{ type: "text"; data: string }` |
| 命名字段            | `Variant { x: i32 }`             | `{ type: "...", x: number }`     |
| 穷尽检查            | 编译器强制                         | 有限支持                          |
| 空值处理            | `Option<T>`                       | `T \| null \| undefined`         |
| 错误处理            | `Result<T, E>`                    | `try-catch`（运行时）             |
| 枚举方法            | `impl Enum { fn ... }`           | 不支持                            |
| 默认变体            | `#[default]`                      | 无内置支持                        |
| 内存优化            | Niche / Box                       | 引擎黑盒                          |

> **核心记忆点：** Rust 的 enum 是"类固醇版的 TypeScript union type"。它不仅能表达"A 或 B 或 C"，还能让每个选项携带不同的数据，而且编译器会帮你确保每一种情况都被处理。忘记处理某个变体？代码直接编译不过。这就是"让非法状态不可表示"的哲学。
