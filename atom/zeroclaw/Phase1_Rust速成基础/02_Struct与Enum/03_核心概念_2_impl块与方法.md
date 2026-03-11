# 核心概念 2：impl 块与方法

> **一句话定位：** Rust 用 `impl` 块把方法"贴"到类型上，取代了 class 语法，通过 `&self` / `&mut self` / `self` 三种形式精确控制数据的访问权限。

---

## 1. impl 块基础：给类型"贴"行为

### 1.1 基本语法

在 TypeScript 中，你把方法直接写在 class 里面。Rust 不同 -- 数据定义（struct/enum）和行为定义（impl 块）是分开的：

```rust
// -------- 数据定义 --------
struct Agent {
    name: String,
    model: String,
    temperature: f32,
}

// -------- 行为定义（impl 块）--------
impl Agent {
    // 关联函数（构造函数）
    fn new(name: &str, model: &str) -> Self {
        Agent {
            name: name.to_string(),
            model: model.to_string(),
            temperature: 0.7,
        }
    }

    // 方法（只读访问）
    fn describe(&self) -> String {
        format!("Agent '{}' using model '{}'", self.name, self.model)
    }
}
```

**TypeScript 对照：**

```typescript
// TypeScript 把数据和行为写在一起
class Agent {
    name: string;
    model: string;
    temperature: number;

    constructor(name: string, model: string) {
        this.name = name;
        this.model = model;
        this.temperature = 0.7;
    }

    describe(): string {
        return `Agent '${this.name}' using model '${this.model}'`;
    }
}
```

### 1.2 一个类型可以有多个 impl 块

这是 Rust 独有的能力 -- 你可以把同一个类型的方法分散到不同的 impl 块中。ZeroClaw 源码中大量使用这个特性来组织代码：

```rust
struct Agent {
    name: String,
    model: String,
    history: Vec<String>,
}

// impl 块 1：核心功能
impl Agent {
    fn new(name: &str) -> Self {
        Agent {
            name: name.to_string(),
            model: "gpt-4".to_string(),
            history: Vec::new(),
        }
    }
}

// impl 块 2：历史记录功能
impl Agent {
    fn add_history(&mut self, entry: String) {
        self.history.push(entry);
    }

    fn history_count(&self) -> usize {
        self.history.len()
    }
}

// impl 块 3：Trait 实现（后续章节详解）
impl std::fmt::Display for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Agent({})", self.name)
    }
}
```

> **日常生活类比：** 一个人（struct）可以有多张名片（impl 块）。一张写工作技能，一张写兴趣爱好，一张写联系方式。名片分开放，但都属于同一个人。

> **前端开发类比：** 想象 TypeScript 支持 partial class -- 你可以把 `Agent` 的方法分散到不同文件里。Rust 的多 impl 块就是天然的 partial class。

### 1.3 固有 impl vs Trait impl

Rust 有两种 impl 块，区别很重要：

```rust
// 固有 impl（Inherent impl）-- 给类型直接添加方法
impl Agent {
    fn name(&self) -> &str { &self.name }
}

// Trait impl -- 实现某个接口
impl Clone for Agent {
    fn clone(&self) -> Self {
        Agent {
            name: self.name.clone(),
            model: self.model.clone(),
            history: self.history.clone(),
        }
    }
}
```

| 区别       | 固有 impl            | Trait impl               |
|------------|----------------------|--------------------------|
| 语法       | `impl Type { ... }`  | `impl Trait for Type { ... }` |
| TS 类比    | 普通 class 方法       | 实现 interface            |
| 用途       | 类型专属功能           | 满足通用接口约定            |
| 调用方式   | `agent.name()`       | `agent.clone()`          |

---

## 2. 方法的三种 self 形式：Rust 的精髓

这是从 TypeScript 转 Rust 最关键的概念差异。TypeScript 的 `this` 永远是可变引用，Rust 的 `self` 有三种精确控制：

### 2.1 `&self` -- 只读访问（最常用）

```rust
struct ActionTracker {
    name: String,
    actions: Vec<String>,
}

impl ActionTracker {
    // &self = 只读借用，不能修改任何字段
    fn count(&self) -> usize {
        self.actions.len()
    }

    fn name(&self) -> &str {
        &self.name
    }
}
```

> **ZeroClaw 实例：** `ActionTracker` 的 `count(&self)` 方法只统计动作数量，不修改追踪器状态。`HardwareConfig` 的 `transport_mode(&self)` 方法只读取配置，不改变它。

### 2.2 `&mut self` -- 可变访问

```rust
impl ActionTracker {
    // &mut self = 可变借用，可以修改字段
    fn record(&mut self, action: String) {
        self.actions.push(action);
    }

    fn clear(&mut self) {
        self.actions.clear();
    }
}
```

> **ZeroClaw 实例：** `ActionTracker` 的 `record(&self)` 实际上内部使用了 `Mutex`（互斥锁），所以签名是 `&self` 但能安全修改内部状态。这是一种叫"内部可变性"的高级模式，我们后续会详细讲。

### 2.3 `self` -- 消耗所有权（Builder 模式核心）

```rust
struct SendMessage {
    content: String,
    subject: Option<String>,
    thread_id: Option<String>,
}

impl SendMessage {
    fn new(content: &str) -> Self {
        SendMessage {
            content: content.to_string(),
            subject: None,
            thread_id: None,
        }
    }

    // self（不是 &self）-- 消耗自身，返回新的 Self
    fn with_subject(mut self, subject: &str) -> Self {
        self.subject = Some(subject.to_string());
        self
    }

    // 同样消耗自身
    fn in_thread(mut self, thread_id: &str) -> Self {
        self.thread_id = Some(thread_id.to_string());
        self
    }
}
```

> **关键：** `self`（没有 `&`）意味着这个方法会**消耗**调用者。调用后原来的变量就不能用了。这听起来可怕，但正是 Builder 模式的基石。

### 2.4 三种形式对比总表

```rust
impl MyType {
    fn read(&self)         {}  // 只读借用 -- "我看看就好"
    fn modify(&mut self)   {}  // 可变借用 -- "我要改一下"
    fn consume(self)       {}  // 获取所有权 -- "给我，我拿走了"
}
```

| 形式          | 所有权   | 能否修改 | 调用后原变量 | 典型场景          |
|---------------|---------|---------|-------------|-------------------|
| `&self`       | 借用     | 不能    | 仍可用       | getter、计算、展示  |
| `&mut self`   | 可变借用 | 可以    | 仍可用       | setter、状态变更    |
| `self`        | 转移     | 可以    | 不可用       | builder、转换、销毁 |

**TypeScript 对照：**

```typescript
class MyType {
    // TypeScript 的 this 永远是隐式的可变引用
    // 没有办法在类型层面区分"只读"和"可变"
    read() { /* this 可读可写 */ }
    modify() { /* this 可读可写 */ }
    // TypeScript 没有"消耗 this"的概念
}
```

> **日常生活类比：**
> - `&self` = 图书馆借书看：你能翻阅，但不能在上面写字
> - `&mut self` = 在自己的笔记本上写：可以随便改
> - `self` = 把书送给别人：你就没有这本书了

---

## 3. 关联函数（构造函数模式）

### 3.1 什么是关联函数

没有 `self` 参数的函数叫**关联函数**（Associated Function）。它不操作实例，而是"属于"这个类型本身：

```rust
struct ChatMessage {
    role: String,
    content: String,
}

impl ChatMessage {
    // ---------- 关联函数（没有 self）----------
    // 等价于 TypeScript 的 static method

    fn system(content: &str) -> Self {
        ChatMessage {
            role: "system".to_string(),
            content: content.to_string(),
        }
    }

    fn user(content: &str) -> Self {
        ChatMessage {
            role: "user".to_string(),
            content: content.to_string(),
        }
    }

    fn assistant(content: &str) -> Self {
        ChatMessage {
            role: "assistant".to_string(),
            content: content.to_string(),
        }
    }

    // ---------- 普通方法（有 self）----------
    fn is_from_user(&self) -> bool {
        self.role == "user"
    }
}

fn main() {
    // 关联函数：用 :: 调用
    let msg = ChatMessage::system("You are a helpful assistant.");

    // 方法：用 . 调用
    println!("Is from user? {}", msg.is_from_user()); // false
}
```

### 3.2 双冒号 `::` vs 点号 `.`

这是很多初学者困惑的地方，规则其实很简单：

```rust
// :: 用于"类型级别"的东西
let msg = ChatMessage::system("hello");    // 关联函数
let max = i32::MAX;                        // 关联常量
let v = Vec::<i32>::new();                 // 带泛型的关联函数

// . 用于"实例级别"的东西
let len = msg.content.len();               // 方法调用
let role = msg.role.clone();               // 方法调用
```

**TypeScript 对照：**

```typescript
// TypeScript 也有类似区分
const msg = ChatMessage.system("hello"); // static method (≈ Rust ::)
const len = msg.content.length;          // instance property (≈ Rust .)
```

### 3.3 `new()` 约定

Rust 没有 `constructor` 关键字。社区约定用 `new()` 作为主要构造函数名：

```rust
impl Agent {
    // 标准构造函数
    pub fn new(name: &str) -> Self {
        Agent {
            name: name.to_string(),
            model: "default".to_string(),
            temperature: 0.7,
        }
    }

    // 附加构造函数 -- 带更多参数
    pub fn with_model(name: &str, model: &str) -> Self {
        Agent {
            name: name.to_string(),
            model: model.to_string(),
            temperature: 0.7,
        }
    }
}
```

---

## 4. 方法链（Fluent API）-- Builder 模式

### 4.1 核心原理：返回 Self

方法链的秘密很简单 -- 每个方法吃掉 `self`，做完事后把修改过的 `self` 还回去：

```rust
// ===== ZeroClaw 风格：AgentBuilder 模式 =====

struct AgentBuilder {
    name: Option<String>,
    model: Option<String>,
    temperature: f32,
    system_prompt: Option<String>,
    max_tokens: u32,
}

impl AgentBuilder {
    fn new() -> Self {
        AgentBuilder {
            name: None,
            model: None,
            temperature: 0.7,
            system_prompt: None,
            max_tokens: 4096,
        }
    }

    // 每个方法：mut self -> Self
    fn name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self  // 关键：返回自己
    }

    fn model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }

    fn temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    fn system_prompt(mut self, prompt: &str) -> Self {
        self.system_prompt = Some(prompt.to_string());
        self
    }

    fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = tokens;
        self
    }

    // 最终构建方法
    fn build(self) -> Result<Agent, String> {
        Ok(Agent {
            name: self.name.ok_or("name is required")?,
            model: self.model.ok_or("model is required")?,
            temperature: self.temperature,
        })
    }
}
```

### 4.2 使用 Builder

```rust
fn main() {
    // 链式调用 -- 像流水线一样
    let agent = AgentBuilder::new()
        .name("zeroclaw")
        .model("gpt-4")
        .temperature(0.3)
        .system_prompt("You are a security analyst.")
        .max_tokens(8192)
        .build()
        .expect("Failed to build agent");

    println!("Created: {}", agent.describe());
}
```

**TypeScript 对照：**

```typescript
// TypeScript Builder 几乎一样，只是每个方法返回 this
class AgentBuilder {
    private _name?: string;
    private _model?: string;

    name(name: string): this {
        this._name = name;
        return this; // 返回 this 实现链式调用
    }

    model(model: string): this {
        this._model = model;
        return this;
    }

    build(): Agent {
        if (!this._name || !this._model) throw new Error("missing fields");
        return new Agent(this._name, this._model);
    }
}

// 用法几乎相同
const agent = new AgentBuilder()
    .name("zeroclaw")
    .model("gpt-4")
    .build();
```

> **关键区别：** TypeScript 返回 `this`（可变引用），builder 可以复用。Rust 返回 `Self`（转移所有权），builder 被消耗，每条链只能用一次。这防止了"builder 被意外复用导致状态混乱"的 bug。

### 4.3 ZeroClaw SendMessage 链式调用

```rust
// 模拟 ZeroClaw 的 SendMessage 链式 API
struct SendMessage {
    content: String,
    subject: Option<String>,
    thread_id: Option<String>,
    metadata: Vec<(String, String)>,
}

impl SendMessage {
    fn new(content: impl Into<String>) -> Self {
        SendMessage {
            content: content.into(),
            subject: None,
            thread_id: None,
            metadata: Vec::new(),
        }
    }

    fn with_subject(mut self, subject: impl Into<String>) -> Self {
        self.subject = Some(subject.into());
        self
    }

    fn in_thread(mut self, thread_id: impl Into<String>) -> Self {
        self.thread_id = Some(thread_id.into());
        self
    }

    fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.push((key.to_string(), value.to_string()));
        self
    }
}

fn main() {
    let msg = SendMessage::new("Hello, team!")
        .with_subject("Sprint Planning")
        .in_thread("thread-42")
        .with_metadata("priority", "high");

    println!("Message: '{}', Subject: {:?}", msg.content, msg.subject);
}
```

---

## 5. 工厂方法模式

### 5.1 ChatMessage 的工厂方法

ZeroClaw 中 `ChatMessage` 不用 `new()`，而是提供多个语义化的工厂方法：

```rust
struct ChatMessage {
    role: String,
    content: String,
}

impl ChatMessage {
    // 工厂方法 -- 语义比 new() 更清晰
    fn system(content: impl Into<String>) -> Self {
        ChatMessage { role: "system".to_string(), content: content.into() }
    }

    fn user(content: impl Into<String>) -> Self {
        ChatMessage { role: "user".to_string(), content: content.into() }
    }

    fn assistant(content: impl Into<String>) -> Self {
        ChatMessage { role: "assistant".to_string(), content: content.into() }
    }

    fn tool(content: impl Into<String>) -> Self {
        ChatMessage { role: "tool".to_string(), content: content.into() }
    }
}

fn main() {
    // 直观、自文档化的 API
    let conversation = vec![
        ChatMessage::system("You are a helpful assistant."),
        ChatMessage::user("What is Rust?"),
        ChatMessage::assistant("Rust is a systems programming language."),
    ];

    for msg in &conversation {
        println!("[{}]: {}", msg.role, msg.content);
    }
}
```

### 5.2 `impl Into<String>` 的灵活性

注意上面参数类型不是 `&str` 也不是 `String`，而是 `impl Into<String>`。这让调用方可以传任何能转成 String 的东西：

```rust
// 以下调用全部合法：
ChatMessage::user("hello");                    // &str
ChatMessage::user(String::from("hello"));      // String
ChatMessage::user(format!("Hello, {}", name)); // String（format! 返回 String）
```

> **前端类比：** 类似于 TypeScript 函数接受 `string | { toString(): string }` -- 更灵活的参数接收方式。

---

## 6. 自定义 Trait 实现

有时候 `#[derive]` 自动生成的实现不够用，需要手写。ZeroClaw 有三个典型场景：

### 6.1 手写 Clone -- ActionTracker 的 Mutex 问题

```rust
use std::sync::Mutex;

struct ActionTracker {
    name: String,
    count: Mutex<u32>,  // Mutex 没有实现 Clone！
}

// 不能用 #[derive(Clone)]，因为 Mutex 不支持
// 必须手写
impl Clone for ActionTracker {
    fn clone(&self) -> Self {
        ActionTracker {
            name: self.name.clone(),
            // 手动处理 Mutex：锁定 -> 读取值 -> 创建新 Mutex
            count: Mutex::new(*self.count.lock().unwrap()),
        }
    }
}

fn main() {
    let tracker = ActionTracker {
        name: "security_scan".to_string(),
        count: Mutex::new(5),
    };

    let tracker2 = tracker.clone();
    println!("Cloned tracker: {}", tracker2.name);
}
```

### 6.2 手写 Default -- SecurityPolicy 的安全默认值

```rust
struct SecurityPolicy {
    max_commands_per_minute: u32,
    allowed_directories: Vec<String>,
    require_approval: bool,
    autonomy_level: String,
}

// 手写 Default 提供"安全第一"的默认配置
impl Default for SecurityPolicy {
    fn default() -> Self {
        SecurityPolicy {
            max_commands_per_minute: 10,         // 保守限制
            allowed_directories: vec![           // 只允许安全目录
                "/tmp".to_string(),
                "/home".to_string(),
            ],
            require_approval: true,              // 默认需要审批
            autonomy_level: "supervised".to_string(), // 默认监督模式
        }
    }
}

fn main() {
    // 用 Default 创建安全配置
    let policy = SecurityPolicy::default();
    println!("Max commands/min: {}", policy.max_commands_per_minute);
    println!("Require approval: {}", policy.require_approval);

    // 部分覆盖（struct update 语法）
    let relaxed = SecurityPolicy {
        require_approval: false,
        ..SecurityPolicy::default()  // 其余用默认值
    };
    println!("Relaxed approval: {}", relaxed.require_approval);
}
```

### 6.3 手写 Display -- 自定义输出格式

```rust
use std::fmt;

enum MemoryCategory {
    Core,
    Daily,
    Conversation,
    Custom(String),
}

// 手写 Display 控制打印格式
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
        MemoryCategory::Custom("security_logs".to_string()),
    ];

    for cat in &categories {
        println!("Category: {}", cat);  // 使用我们自定义的 Display
    }
    // 输出:
    // Category: core
    // Category: custom:security_logs
}
```

---

## 7. TypeScript vs Rust 完整对照

```typescript
// ========== TypeScript ==========
class Agent {
    // 数据和行为在一起
    private name: string;
    private model: string;

    // 构造函数
    constructor(name: string, model: string) {
        this.name = name;
        this.model = model;
    }

    // 静态方法
    static default(): Agent {
        return new Agent("default", "gpt-4");
    }

    // 实例方法（this 隐式可变）
    describe(): string {
        return `${this.name} (${this.model})`;
    }

    // 没有"只读this"的概念
    rename(newName: string): void {
        this.name = newName;  // 随时可变
    }
}
```

```rust
// ========== Rust ==========

// 数据定义
struct Agent {
    name: String,
    model: String,
}

// 行为定义（可以分多个 impl 块）
impl Agent {
    // 关联函数（≈ static）
    fn default() -> Self {
        Agent {
            name: "default".to_string(),
            model: "gpt-4".to_string(),
        }
    }

    // 只读方法 -- 编译器保证不会修改
    fn describe(&self) -> String {
        format!("{} ({})", self.name, self.model)
    }

    // 可变方法 -- 明确声明"我要改数据"
    fn rename(&mut self, new_name: &str) {
        self.name = new_name.to_string();
    }

    // 消耗方法 -- 明确声明"用完就没了"
    fn into_name(self) -> String {
        self.name  // self 被消耗，Agent 实例不再存在
    }
}
```

---

## 8. 完整可运行示例

将上面所有概念整合到一个完整程序中：

```rust
use std::fmt;

// ===== 1. 数据定义 =====
struct TaskAgent {
    name: String,
    model: String,
    temperature: f32,
    history: Vec<String>,
}

// ===== 2. 固有 impl -- 核心方法 =====
impl TaskAgent {
    // 关联函数：构造器
    fn new(name: impl Into<String>, model: impl Into<String>) -> Self {
        TaskAgent {
            name: name.into(),
            model: model.into(),
            temperature: 0.7,
            history: Vec::new(),
        }
    }

    // &self：只读
    fn name(&self) -> &str {
        &self.name
    }

    fn history_count(&self) -> usize {
        self.history.len()
    }

    // &mut self：可变
    fn record(&mut self, action: &str) {
        self.history.push(action.to_string());
    }

    fn set_temperature(&mut self, temp: f32) {
        self.temperature = temp.clamp(0.0, 2.0);
    }

    // self：消耗（提取数据）
    fn into_history(self) -> Vec<String> {
        self.history
    }
}

// ===== 3. Trait impl -- Display =====
impl fmt::Display for TaskAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TaskAgent('{}', model={}, temp={})",
            self.name, self.model, self.temperature)
    }
}

// ===== 4. Builder 模式 =====
struct TaskAgentBuilder {
    name: Option<String>,
    model: String,
    temperature: f32,
}

impl TaskAgentBuilder {
    fn new() -> Self {
        TaskAgentBuilder {
            name: None,
            model: "gpt-4".to_string(),
            temperature: 0.7,
        }
    }

    fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    fn temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    fn build(self) -> Result<TaskAgent, String> {
        let name = self.name.ok_or("Agent name is required")?;
        Ok(TaskAgent {
            name,
            model: self.model,
            temperature: self.temperature,
            history: Vec::new(),
        })
    }
}

// ===== 5. 工厂方法 =====
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

impl fmt::Display for ChatMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]: {}", self.role, self.content)
    }
}

// ===== main =====
fn main() {
    println!("=== 1. 基本构造与方法调用 ===");
    let mut agent = TaskAgent::new("zeroclaw", "gpt-4");
    println!("{}", agent);                    // Display trait
    println!("Name: {}", agent.name());       // &self 方法

    println!("\n=== 2. 可变方法 ===");
    agent.record("scanned /etc/passwd");      // &mut self
    agent.record("analyzed network traffic");
    agent.set_temperature(0.3);
    println!("{}", agent);
    println!("Actions recorded: {}", agent.history_count());

    println!("\n=== 3. Builder 模式 ===");
    let agent2 = TaskAgentBuilder::new()
        .name("scout")
        .model("claude-3")
        .temperature(0.5)
        .build()
        .expect("Failed to build");
    println!("{}", agent2);

    println!("\n=== 4. 工厂方法 ===");
    let conversation = vec![
        ChatMessage::system("You are a security analyst."),
        ChatMessage::user("Scan the system for vulnerabilities."),
        ChatMessage::assistant("Starting security scan..."),
    ];
    for msg in &conversation {
        println!("{}", msg);
    }

    println!("\n=== 5. 消耗方法 ===");
    let history = agent.into_history();       // self -- agent 被消耗
    println!("History: {:?}", history);
    // println!("{}", agent);  // 编译错误！agent 已被消耗
}
```

**运行输出：**

```
=== 1. 基本构造与方法调用 ===
TaskAgent('zeroclaw', model=gpt-4, temp=0.7)
Name: zeroclaw

=== 2. 可变方法 ===
TaskAgent('zeroclaw', model=gpt-4, temp=0.3)
Actions recorded: 2

=== 3. Builder 模式 ===
TaskAgent('scout', model=claude-3, temp=0.5)

=== 4. 工厂方法 ===
[system]: You are a security analyst.
[user]: Scan the system for vulnerabilities.
[assistant]: Starting security scan...

=== 5. 消耗方法 ===
History: ["scanned /etc/passwd", "analyzed network traffic"]
```

---

## 速记卡片

| 概念           | Rust                          | TypeScript                  |
|----------------|-------------------------------|-----------------------------|
| 定义方法        | `impl Type { fn ... }`        | `class { method() {} }`    |
| 构造函数        | `Type::new()`                 | `new Type()`               |
| 静态方法        | `Type::func()`（关联函数）      | `Type.func()`              |
| 只读方法        | `fn f(&self)`                 | 无法区分                    |
| 可变方法        | `fn f(&mut self)`             | `method()` (this 默认可变)  |
| 消耗方法        | `fn f(self)`                  | 无此概念                    |
| 方法链          | `builder.a(x).b(y)` (Self)    | `builder.a(x).b(y)` (this) |
| 多个方法块      | 多个 `impl` 块                | 不支持（单个 class）         |
| 实现接口        | `impl Trait for Type`         | `implements Interface`     |

> **核心记忆点：** Rust 的三种 self 就像三种门禁卡 -- `&self` 是访客卡（只能看），`&mut self` 是员工卡（能改），`self` 是一次性通行证（用了就作废）。TypeScript 只有万能卡，方便但不安全。
