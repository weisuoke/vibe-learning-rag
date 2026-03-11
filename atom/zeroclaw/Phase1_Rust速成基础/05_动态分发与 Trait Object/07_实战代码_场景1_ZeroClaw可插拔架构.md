# 实战代码 场景1：ZeroClaw 可插拔架构

> **目标**：用动态分发构建一个迷你版 ZeroClaw，展示 `Box<dyn Provider>` + `Vec<Box<dyn Tool>>` 的可插拔架构
> **运行**：`cargo run`（纯标准库，无外部依赖）

---

## 场景说明

模拟 ZeroClaw 的核心架构：
- Agent 持有 `Box<dyn Provider>` 和 `Vec<Box<dyn Tool>>`
- 通过 config 字符串在**运行时**选择具体实现
- 所有组件通过 **Trait 接口**交互，Agent 不知道具体类型

```
┌─────────────────────────────────────────────┐
│              Agent（运行时）                  │
│                                              │
│  provider: Box<dyn Provider>  ← 运行时选择   │
│  tools: Vec<Box<dyn Tool>>    ← 异构集合     │
└─────────────────────────────────────────────┘
```

---

## 完整代码

```rust
// ============================================================
// 迷你版 ZeroClaw —— 可插拔架构演示
// ============================================================

use std::collections::HashMap;

// ============================================================
// 第一部分：定义核心 Trait
// ============================================================

/// Provider Trait — 负责 LLM 调用
/// 所有方法以 &self 为参数 → 满足对象安全 → 可作为 dyn Trait
trait Provider: Send + Sync {
    fn name(&self) -> &str;
    fn chat(&self, message: &str) -> String;
}

/// Tool Trait — 负责工具执行
trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn execute(&self, args: &str) -> String;
}

// ============================================================
// 第二部分：实现多个 Provider
// ============================================================

struct EchoProvider;  // 零大小类型，仍可放进 Box<dyn Provider>

impl Provider for EchoProvider {
    fn name(&self) -> &str { "echo" }
    fn chat(&self, message: &str) -> String {
        format!("[Echo] You said: {}", message)
    }
}

struct MockOpenAI { model: String }

impl Provider for MockOpenAI {
    fn name(&self) -> &str { "openai" }
    fn chat(&self, message: &str) -> String {
        format!("[OpenAI/{}] Processing '{}' → AI response here", self.model, message)
    }
}

struct MockOllama { model: String }

impl Provider for MockOllama {
    fn name(&self) -> &str { "ollama" }
    fn chat(&self, message: &str) -> String {
        format!("[Ollama/{}] Local inference: '{}' → response", self.model, message)
    }
}

// ============================================================
// 第三部分：实现多个 Tool
// ============================================================

struct ShellTool;

impl Tool for ShellTool {
    fn name(&self) -> &str { "shell" }
    fn description(&self) -> &str { "Execute shell commands on the system" }
    fn execute(&self, args: &str) -> String {
        format!("[Shell] $ {} → (simulated output)", args)
    }
}

struct CalculatorTool;

impl Tool for CalculatorTool {
    fn name(&self) -> &str { "calculator" }
    fn description(&self) -> &str { "Perform basic arithmetic calculations" }
    fn execute(&self, args: &str) -> String {
        let parts: Vec<&str> = args.split_whitespace().collect();
        if parts.len() == 3 {
            let a: f64 = parts[0].parse().unwrap_or(0.0);
            let op = parts[1];
            let b: f64 = parts[2].parse().unwrap_or(0.0);
            let result = match op {
                "+" => a + b, "-" => a - b,
                "*" => a * b, "/" => if b != 0.0 { a / b } else { f64::NAN },
                _ => return format!("[Calculator] Unknown operator: {}", op),
            };
            format!("[Calculator] {} {} {} = {}", a, op, b, result)
        } else {
            format!("[Calculator] Invalid expression: {}", args)
        }
    }
}

struct GreetTool;

impl Tool for GreetTool {
    fn name(&self) -> &str { "greet" }
    fn description(&self) -> &str { "Generate a friendly greeting message" }
    fn execute(&self, args: &str) -> String {
        if args.is_empty() {
            "[Greet] Hello, World!".to_string()
        } else {
            format!("[Greet] Hello, {}! Welcome to ZeroClaw!", args)
        }
    }
}

// ============================================================
// 第四部分：工厂函数（config 驱动的运行时选择）
// ============================================================

/// 根据配置字符串创建 Provider — 动态分发的核心入口
/// 输入：字符串（来自 config.toml），输出：Box<dyn Provider>（类型擦除）
fn create_provider(name: &str) -> Result<Box<dyn Provider>, String> {
    match name {
        "echo" => Ok(Box::new(EchoProvider)),
        //  EchoProvider → Box::new → 移到堆上 → 类型擦除为 Box<dyn Provider>
        "openai" => Ok(Box::new(MockOpenAI { model: "gpt-4o".to_string() })),
        "ollama" => Ok(Box::new(MockOllama { model: "llama3".to_string() })),
        unknown => Err(format!("Unknown provider: '{}'", unknown)),
    }
}

/// 创建默认工具集 — 异构集合：不同类型统一为 Box<dyn Tool>
fn create_tools() -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(ShellTool),        // ShellTool      → Box<dyn Tool>
        Box::new(CalculatorTool),   // CalculatorTool  → Box<dyn Tool>
        Box::new(GreetTool),        // GreetTool       → Box<dyn Tool>
    ]
    // 三种不同类型，统一为 16 字节胖指针
}

// ============================================================
// 第五部分：Agent 结构体
// ============================================================

/// Agent — ZeroClaw 的简化版核心
/// Agent 不知道 provider 和 tools 的具体类型 → 完全通过 Trait 接口交互
struct Agent {
    provider: Box<dyn Provider>,   // 不知道是哪个 Provider
    tools: Vec<Box<dyn Tool>>,     // 不知道具体有哪些 Tool
}

impl Agent {
    fn new(provider: Box<dyn Provider>, tools: Vec<Box<dyn Tool>>) -> Self {
        Agent { provider, tools }
    }

    fn list_tools(&self) {
        println!("  Registered tools ({}):", self.tools.len());
        for tool in &self.tools {
            // tool: &Box<dyn Tool> → 每次 tool.name() 都是 vtable 查找
            println!("    - {} : {}", tool.name(), tool.description());
        }
    }

    /// 处理用户输入：调用 Provider + 匹配 Tool
    fn process(&self, input: &str) -> String {
        // 动态分发：self.provider.chat() 通过 vtable 找到具体实现
        let llm_response = self.provider.chat(input);

        // 匹配工具：输入以 /toolname 开头时执行对应工具
        for tool in &self.tools {
            if input.starts_with(&format!("/{}", tool.name())) {
                let args = input
                    .strip_prefix(&format!("/{} ", tool.name()))
                    .unwrap_or("");
                // 动态分发：tool.execute() 通过 vtable 调用正确的实现
                let tool_result = tool.execute(args);
                return format!("{}\n  Tool result: {}", llm_response, tool_result);
            }
        }
        llm_response
    }
}

// ============================================================
// 第六部分：main 函数 — 从"配置"切换不同 Provider
// ============================================================

fn main() {
    println!("===========================================");
    println!("  迷你 ZeroClaw — 可插拔架构演示");
    println!("===========================================\n");

    // 模拟从 config.toml 读取不同配置
    let configs = vec!["echo", "openai", "ollama"];

    for config in &configs {
        // 关键点 1：工厂函数返回 Box<dyn Provider>（运行时选择）
        let provider = create_provider(config).unwrap();
        // 关键点 2：异构集合 Vec<Box<dyn Tool>>
        let tools = create_tools();
        // 关键点 3：Agent 通过 Trait 接口交互
        let agent = Agent::new(provider, tools);

        println!("=== Agent with provider: '{}' ===", config);
        agent.list_tools();
        println!("  Chat: {}", agent.process("Hello!"));
        println!("  Tool: {}", agent.process("/calculator 42 + 58"));
        println!("  Tool: {}", agent.process("/greet Felix"));
        println!();
    }

    // 演示：运行时动态添加工具（无需修改 Agent 代码）
    println!("=== 动态添加工具演示 ===");
    let provider = create_provider("openai").unwrap();
    let mut tools = create_tools();
    struct TimeTool;
    impl Tool for TimeTool {
        fn name(&self) -> &str { "time" }
        fn description(&self) -> &str { "Show current time (simulated)" }
        fn execute(&self, _args: &str) -> String {
            "[Time] 2026-03-10 14:30:00 UTC".to_string()
        }
    }
    tools.push(Box::new(TimeTool));  // 运行时添加，Agent 零修改
    let agent = Agent::new(provider, tools);
    agent.list_tools();
    println!("  Tool: {}", agent.process("/time"));

    // 胖指针大小验证
    println!("\n=== 内存布局验证 ===");
    println!("  Box<dyn Provider>: {} bytes (胖指针)", std::mem::size_of::<Box<dyn Provider>>());
    println!("  Box<dyn Tool>:     {} bytes (胖指针)", std::mem::size_of::<Box<dyn Tool>>());
    println!("  Box<ShellTool>:    {} bytes (瘦指针)", std::mem::size_of::<Box<ShellTool>>());
}
```

---

## 运行输出示例

```
===========================================
  迷你 ZeroClaw — 可插拔架构演示
===========================================

=== Agent with provider: 'echo' ===
  Registered tools (3):
    - shell : Execute shell commands on the system
    - calculator : Perform basic arithmetic calculations
    - greet : Generate a friendly greeting message
  Chat: [Echo] You said: Hello!
  Tool: [Echo] You said: /calculator 42 + 58
  Tool result: [Calculator] 42 + 58 = 100
  Tool: [Echo] You said: /greet Felix
  Tool result: [Greet] Hello, Felix! Welcome to ZeroClaw!

=== Agent with provider: 'openai' ===
  Registered tools (3):
    ...（结构相同，Provider 输出不同）
  Chat: [OpenAI/gpt-4o] Processing 'Hello!' → AI response here

=== Agent with provider: 'ollama' ===
  ...

=== 动态添加工具演示 ===
  Registered tools (4):
    - shell : Execute shell commands on the system
    - calculator : Perform basic arithmetic calculations
    - greet : Generate a friendly greeting message
    - time : Show current time (simulated)
  Tool: [OpenAI/gpt-4o] Processing '/time' → AI response here
  Tool result: [Time] 2026-03-10 14:30:00 UTC

=== 内存布局验证 ===
  Box<dyn Provider>: 16 bytes (胖指针)
  Box<dyn Tool>:     16 bytes (胖指针)
  Box<ShellTool>:    8 bytes (瘦指针)
```

---

## 代码解析：7 个动态分发关键点

| # | 关键点 | 代码位置 | 说明 |
|---|--------|---------|------|
| 1 | **Trait 定义 = 行为契约** | `trait Provider/Tool` | `Send + Sync` + `&self` → 满足对象安全 |
| 2 | **工厂函数返回 Box\<dyn\>** | `create_provider()` | match 分支返回不同类型，统一为 Box<dyn Provider> |
| 3 | **异构集合** | `create_tools()` | 三种不同类型放进同一个 Vec<Box<dyn Tool>> |
| 4 | **Agent 不依赖具体类型** | `struct Agent` | 字段全是 dyn Trait，新增实现零修改 |
| 5 | **方法调用 = vtable 查找** | `self.provider.chat()` | 运行时从胖指针取 vtable_ptr，查表调用 |
| 6 | **运行时动态注册** | `tools.push(Box::new(TimeTool))` | 无需修改 Agent 代码 |
| 7 | **胖指针 16 字节** | `size_of::<Box<dyn Tool>>()` | data_ptr(8) + vtable_ptr(8) |

---

## 与 TypeScript 对照

```typescript
// TypeScript 版本 — 同样的可插拔架构
interface Provider {
  name(): string;
  chat(message: string): string;
}

interface Tool {
  name(): string;
  description(): string;
  execute(args: string): string;
}

class EchoProvider implements Provider {
  name() { return "echo"; }
  chat(message: string) { return `[Echo] ${message}`; }
}

class MockOpenAI implements Provider {
  constructor(private model = "gpt-4o") {}
  name() { return "openai"; }
  chat(message: string) { return `[OpenAI/${this.model}] ${message}`; }
}

// 工厂函数 — 不需要 Box！
function createProvider(name: string): Provider {
  switch (name) {
    case "echo": return new EchoProvider();
    case "openai": return new MockOpenAI();
    default: throw new Error(`Unknown: ${name}`);
  }
}

// Agent — 不需要 Box<dyn>，TypeScript 默认动态分发
class Agent {
  constructor(
    private provider: Provider,    // 对比 Rust: Box<dyn Provider>
    private tools: Tool[],         // 对比 Rust: Vec<Box<dyn Tool>>
  ) {}

  process(input: string): string {
    return this.provider.chat(input);  // JS 原型链查找 = 动态分发
  }
}

// 使用
for (const config of ["echo", "openai"]) {
  const agent = new Agent(createProvider(config), [new ShellTool()]);
  console.log(agent.process("Hello!"));
}
```

### 关键差异对照表

| 概念 | Rust | TypeScript | 原因 |
|------|------|------------|------|
| 行为契约 | `trait Provider` | `interface Provider` | 语法不同，语义相似 |
| 运行时多态 | `Box<dyn Provider>` | `Provider` | TS 默认动态分发 |
| 异构集合 | `Vec<Box<dyn Tool>>` | `Tool[]` | TS 数组天然支持 |
| 堆分配 | `Box::new(ShellTool)` | `new ShellTool()` | TS 自动堆分配 |
| 所有权 | 选 Box/Arc/& | 无需关心 | Rust 无 GC |

> **一句话对比**：TypeScript 中所有方法调用都是动态分发（免费的）。Rust 默认静态（零开销），需要时显式用 `dyn`。代价是多写 `Box<dyn Trait>`，收益是可以自由选择。

---

## 练习题

1. **添加新 Provider**：实现 `MockClaude`，在 `create_provider` 中添加 `"claude"` 分支
2. **添加新 Tool**：实现 `WordCountTool`，用 `args.split_whitespace().count()` 统计单词
3. **思考题**：为什么 `create_provider` 返回 `Result` 而不是直接 `panic!`？

---

*上一篇：[06_反直觉点](./06_反直觉点.md)*
*下一篇：[07_实战代码_场景2_异构集合与工厂模式](./07_实战代码_场景2_异构集合与工厂模式.md)*

---

**文件信息**
- 知识点: 动态分发与 Trait Object
- 维度: 07_实战代码_场景1
- 版本: v1.0
- 日期: 2026-03-10
