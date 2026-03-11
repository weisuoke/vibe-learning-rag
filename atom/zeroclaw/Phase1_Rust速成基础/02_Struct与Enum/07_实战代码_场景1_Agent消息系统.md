# 实战代码 场景 1：Agent 消息系统

> **一句话定位：** 用 Struct + Enum + match 构建一个完整的 Agent 对话消息系统，模拟 ZeroClaw 的 ChatMessage + ConversationMessage 设计。

---

## 场景背景

### ZeroClaw 的消息系统

ZeroClaw 的对话系统核心是两层类型：
- `ChatMessage`（struct）：一条具体的聊天消息，包含角色和内容
- `ConversationMessage`（enum）：对话中的一个条目，可能是聊天、工具调用、或工具结果

```
用户输入 → ChatMessage → ConversationMessage::Chat
                       → ConversationMessage::ToolCalls
                       → ConversationMessage::ToolResults
```

### TypeScript 中你怎么做

```typescript
// TypeScript: discriminated union
type Message =
  | { type: "chat"; role: string; content: string }
  | { type: "tool_call"; calls: { name: string; input: string }[] }
  | { type: "tool_result"; results: { name: string; output: string }[] };

function render(msg: Message): string {
  switch (msg.type) {
    case "chat": return `[${msg.role}] ${msg.content}`;
    case "tool_call": return `Calling ${msg.calls.length} tools`;
    case "tool_result": return `Got ${msg.results.length} results`;
    // 忘了某个 type？TypeScript 不报错
  }
}
```

Rust 版本：编译器保证你处理了每种消息类型。

---

## 第一步：定义消息类型

```rust
#[derive(Debug, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Clone)]
struct ToolCall {
    name: String,
    input: String,
}

#[derive(Debug, Clone)]
struct ToolResult {
    name: String,
    output: String,
    success: bool,
}

#[derive(Debug, Clone)]
enum ConversationMessage {
    Chat(ChatMessage),
    ToolCalls(Vec<ToolCall>),
    ToolResults(Vec<ToolResult>),
}
```

---

## 第二步：为消息添加方法

```rust
impl ChatMessage {
    fn new(role: &str, content: &str) -> Self {
        ChatMessage { role: role.into(), content: content.into() }
    }

    fn is_user(&self) -> bool { self.role == "user" }
    fn is_assistant(&self) -> bool { self.role == "assistant" }
}

impl ConversationMessage {
    fn render(&self) -> String {
        match self {
            ConversationMessage::Chat(chat) => {
                format!("[{}] {}", chat.role, chat.content)
            }
            ConversationMessage::ToolCalls(calls) => {
                let names: Vec<&str> = calls.iter().map(|c| c.name.as_str()).collect();
                format!("🔧 Calling: {}", names.join(", "))
            }
            ConversationMessage::ToolResults(results) => {
                let summary: Vec<String> = results.iter()
                    .map(|r| format!("{}:{}", r.name, if r.success { "✓" } else { "✗" }))
                    .collect();
                format!("📋 Results: {}", summary.join(", "))
            }
        }
    }

    fn is_chat(&self) -> bool { matches!(self, ConversationMessage::Chat(_)) }
    fn is_error(&self) -> bool {
        matches!(self, ConversationMessage::ToolResults(results) if results.iter().any(|r| !r.success))
    }
}
```

---

## 第三步：对话历史管理

```rust
struct Conversation {
    messages: Vec<ConversationMessage>,
}

impl Conversation {
    fn new() -> Self { Conversation { messages: Vec::new() } }

    fn push(&mut self, msg: ConversationMessage) { self.messages.push(msg); }

    fn user_messages(&self) -> Vec<&ChatMessage> {
        self.messages.iter().filter_map(|m| {
            if let ConversationMessage::Chat(chat) = m {
                if chat.is_user() { return Some(chat); }
            }
            None
        }).collect()
    }

    fn has_errors(&self) -> bool {
        self.messages.iter().any(|m| m.is_error())
    }

    fn summary(&self) -> String {
        let chats = self.messages.iter().filter(|m| m.is_chat()).count();
        let tool_calls = self.messages.iter()
            .filter(|m| matches!(m, ConversationMessage::ToolCalls(_))).count();
        format!("{} messages ({} chats, {} tool calls)", self.messages.len(), chats, tool_calls)
    }
}
```

---

## 第四步：完整可运行代码

```rust
// --- 数据类型 ---
#[derive(Debug, Clone)]
struct ChatMessage { role: String, content: String }

#[derive(Debug, Clone)]
struct ToolCall { name: String, input: String }

#[derive(Debug, Clone)]
struct ToolResult { name: String, output: String, success: bool }

#[derive(Debug, Clone)]
enum ConversationMessage {
    Chat(ChatMessage),
    ToolCalls(Vec<ToolCall>),
    ToolResults(Vec<ToolResult>),
}

// --- ChatMessage impl ---
impl ChatMessage {
    fn new(role: &str, content: &str) -> Self {
        ChatMessage { role: role.into(), content: content.into() }
    }
    fn is_user(&self) -> bool { self.role == "user" }
}

// --- ConversationMessage impl ---
impl ConversationMessage {
    fn render(&self) -> String {
        match self {
            ConversationMessage::Chat(chat) =>
                format!("[{}] {}", chat.role, chat.content),
            ConversationMessage::ToolCalls(calls) => {
                let names: Vec<&str> = calls.iter().map(|c| c.name.as_str()).collect();
                format!("Calling: {}", names.join(", "))
            }
            ConversationMessage::ToolResults(results) => {
                let summary: Vec<String> = results.iter()
                    .map(|r| format!("{}:{}", r.name, if r.success { "ok" } else { "fail" }))
                    .collect();
                format!("Results: {}", summary.join(", "))
            }
        }
    }

    fn is_chat(&self) -> bool { matches!(self, ConversationMessage::Chat(_)) }
    fn is_error(&self) -> bool {
        matches!(self, ConversationMessage::ToolResults(r) if r.iter().any(|r| !r.success))
    }
}

// --- Conversation ---
struct Conversation { messages: Vec<ConversationMessage> }

impl Conversation {
    fn new() -> Self { Conversation { messages: Vec::new() } }
    fn push(&mut self, msg: ConversationMessage) { self.messages.push(msg); }

    fn user_messages(&self) -> Vec<&ChatMessage> {
        self.messages.iter().filter_map(|m| {
            if let ConversationMessage::Chat(c) = m { if c.is_user() { return Some(c); } }
            None
        }).collect()
    }

    fn summary(&self) -> String {
        let chats = self.messages.iter().filter(|m| m.is_chat()).count();
        let errors = self.messages.iter().filter(|m| m.is_error()).count();
        format!("total={}, chats={}, errors={}", self.messages.len(), chats, errors)
    }
}

// --- main ---
fn main() {
    let mut conv = Conversation::new();

    // 模拟一轮对话
    conv.push(ConversationMessage::Chat(
        ChatMessage::new("user", "Search for Rust tutorials")
    ));
    conv.push(ConversationMessage::Chat(
        ChatMessage::new("assistant", "I'll search for that.")
    ));
    conv.push(ConversationMessage::ToolCalls(vec![
        ToolCall { name: "search".into(), input: "Rust tutorials 2025".into() },
    ]));
    conv.push(ConversationMessage::ToolResults(vec![
        ToolResult { name: "search".into(), output: "Found 42 results".into(), success: true },
    ]));
    conv.push(ConversationMessage::Chat(
        ChatMessage::new("assistant", "I found 42 Rust tutorials.")
    ));

    // 渲染所有消息
    println!("=== Conversation ===");
    for msg in &conv.messages {
        println!("  {}", msg.render());
    }

    // 统计
    println!("\n=== Summary ===");
    println!("{}", conv.summary());

    // 提取用户消息
    let user_msgs = conv.user_messages();
    println!("\nUser messages ({}):", user_msgs.len());
    for m in &user_msgs {
        println!("  {}", m.content);
    }

    // 错误检查
    println!("\nHas errors: {}", conv.messages.iter().any(|m| m.is_error()));
}
```

**编译运行：** `rustc main.rs && ./main`

---

## 运行输出

```
=== Conversation ===
  [user] Search for Rust tutorials
  [assistant] I'll search for that.
  Calling: search
  Results: search:ok
  [assistant] I found 42 Rust tutorials.

=== Summary ===
total=5, chats=3, errors=0

User messages (1):
  Search for Rust tutorials

Has errors: false
```

---

## 学到了什么

| 知识点 | 在本场景中的体现 |
|--------|-----------------|
| Struct 定义 | ChatMessage、ToolCall、ToolResult 三个数据结构 |
| Enum 变体 | ConversationMessage 的三种变体携带不同数据 |
| match 解构 | render() 中解构每种变体并格式化 |
| matches! 宏 | is_chat()、is_error() 快速判断 |
| if let | user_messages() 中提取特定变体 |
| Vec\<T\> | 对话历史、工具调用列表 |
| 迭代器链 | filter_map、iter().any()、collect() |
| derive | Debug + Clone 自动生成 |

---

*上一篇：[06_反直觉点](./06_反直觉点.md)*
*下一篇：[07_实战代码_场景2_Builder模式](./07_实战代码_场景2_Builder模式.md)*
