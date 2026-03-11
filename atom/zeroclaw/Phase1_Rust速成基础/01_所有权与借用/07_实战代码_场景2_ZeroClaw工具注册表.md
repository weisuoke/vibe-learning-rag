# 实战代码 场景 2 —— ZeroClaw 工具注册表

> **一句话定位：** 从 ZeroClaw 真实源码中提炼出工具注册表模式，用 `Box<dyn Trait>` 存储异构工具、用 `Arc<T>` 共享安全策略，理解 Rust 中 trait 对象的所有权流转。

---

## 场景说明

### 目标

模拟 ZeroClaw 的 Tool 注册表系统，深入理解以下所有权模式：

- **`Box<dyn Tool>`** —— 堆分配 + 拥有所有权的 trait 对象，用于存储大小不确定的异构工具
- **`Vec<Box<dyn Tool>>`** —— 注册表拥有所有工具的所有权
- **`Arc<SecurityPolicy>`** —— 引用计数共享，多个工具共享同一份安全策略
- **`&dyn Tool`** —— 从注册表借用工具引用，不转移所有权

### 背景

在 ZeroClaw 源码中，工具系统是核心架构之一：

```
// 来自 sourcecode/zeroclaw/src/tools/traits.rs
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult>;
}

// 来自 sourcecode/zeroclaw/src/tools/mod.rs
pub fn default_tools(security: Arc<SecurityPolicy>) -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(ShellTool::new(security.clone(), runtime)),
        Box::new(FileReadTool::new(security.clone())),
        Box::new(FileWriteTool::new(security.clone())),
        // ...
    ]
}
```

注意关键模式：`security.clone()` 在这里是 `Arc::clone()`，只增加引用计数，不复制数据。每个工具共享同一份安全策略。

[来源: sourcecode/zeroclaw/src/tools/mod.rs]
[来源: sourcecode/zeroclaw/src/tools/traits.rs]

---

## TypeScript 对比：先看你熟悉的版本

在进入 Rust 之前，先看 TypeScript 中同样逻辑会怎么写：

```typescript
// TypeScript 版本 —— 零所有权烦恼
interface Tool {
  name(): string;
  description(): string;
  execute(input: string): string;
}

class EchoTool implements Tool {
  name() { return "echo"; }
  description() { return "原样返回输入"; }
  execute(input: string) { return input; }
}

class UppercaseTool implements Tool {
  name() { return "uppercase"; }
  description() { return "转大写"; }
  execute(input: string) { return input.toUpperCase(); }
}

interface SecurityPolicy {
  allowedTools: string[];
}

class ToolRegistry {
  private tools: Tool[] = [];           // 引用数组，GC 管理
  private security: SecurityPolicy;     // 共享引用，GC 管理

  constructor(security: SecurityPolicy) {
    this.security = security;
  }

  register(tool: Tool) { this.tools.push(tool); }

  find(name: string): Tool | undefined {
    return this.tools.find(t => t.name() === name);
  }

  execute(name: string, input: string): string {
    const tool = this.find(name);
    if (!tool) throw new Error(`Tool '${name}' not found`);
    if (!this.security.allowedTools.includes(name)) {
      throw new Error(`Tool '${name}' not allowed`);
    }
    return tool.execute(input);
  }
}
```

**关键差异预告：**

| 问题 | TypeScript | Rust |
|------|-----------|------|
| 工具存哪里？ | `Tool[]` 引用数组 | `Vec<Box<dyn Tool>>` 堆上的拥有式容器 |
| 谁释放工具？ | GC 垃圾回收 | `Vec` 被 drop 时自动 drop 每个 `Box` |
| 安全策略共享 | 随便多处引用 | `Arc<SecurityPolicy>` 显式引用计数 |
| 类型检查 | 运行时 duck typing | 编译期 trait 约束 |
| 线程安全 | 无保证 | `Send + Sync` 编译期保证 |

---

## Step 1：定义 Tool Trait

```rust
use std::sync::{Arc, Mutex};

/// 工具 trait —— 所有工具必须实现的接口
///
/// 对比 ZeroClaw 源码中的 Tool trait:
///   - 真实版本是 async 的（async fn execute），这里简化为同步
///   - 真实版本使用 serde_json::Value 作为输入/输出
///   - 但所有权模式完全一致
trait Tool {
    /// 返回工具名称
    /// 注意返回 &str 而非 String —— 借用而非拥有
    fn name(&self) -> &str;

    /// 返回工具描述
    fn description(&self) -> &str;

    /// 执行工具
    /// input: &str —— 借用输入，不获取所有权
    /// 返回 String —— 将新创建的结果的所有权交给调用者
    fn execute(&self, input: &str) -> String;
}
```

**所有权分析：**

```
fn name(&self) -> &str
         ^^^^     ^^^^
         |        |
         借用 self  返回值的生命周期绑定到 self
                   （只要 self 活着，返回的 &str 就有效）

fn execute(&self, input: &str) -> String
           ^^^^   ^^^^^^^^^^      ^^^^^^
           |      |               |
           借用    借用输入         创建新 String，所有权给调用者
```

---

## Step 2：实现具体工具

### 2.1 EchoTool —— 最简单的工具

```rust
/// 回声工具：原样返回输入
struct EchoTool;

impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
        // 字符串字面量是 &'static str，生命周期无限
        // 可以安全地作为 &str 返回
    }

    fn description(&self) -> &str {
        "原样返回输入内容"
    }

    fn execute(&self, input: &str) -> String {
        // input.to_string() 从 &str 创建一个新的 String
        // 新 String 的所有权传给调用者
        input.to_string()
    }
}
```

### 2.2 UppercaseTool —— 转大写

```rust
/// 大写工具：将输入转为大写
struct UppercaseTool;

impl Tool for UppercaseTool {
    fn name(&self) -> &str {
        "uppercase"
    }

    fn description(&self) -> &str {
        "将输入转为大写字母"
    }

    fn execute(&self, input: &str) -> String {
        // to_uppercase() 创建新的 String，所有权给调用者
        input.to_uppercase()
    }
}
```

### 2.3 CounterTool —— 带共享状态的计数器

```rust
/// 计数器工具：每次执行时计数 +1
///
/// 关键点：使用 Arc<Mutex<u32>> 实现内部可变性
/// - Arc：允许多个引用共享同一个计数器
/// - Mutex：保证同一时刻只有一个线程能修改计数器
struct CounterTool {
    count: Arc<Mutex<u32>>,
}

impl CounterTool {
    fn new() -> Self {
        CounterTool {
            count: Arc::new(Mutex::new(0)),
        }
    }
}

impl Tool for CounterTool {
    fn name(&self) -> &str {
        "counter"
    }

    fn description(&self) -> &str {
        "每次执行计数+1，返回当前计数"
    }

    fn execute(&self, input: &str) -> String {
        // 注意：execute 接收 &self（不可变借用）
        // 但我们需要修改 count —— 这就是"内部可变性"的用武之地
        let mut count = self.count.lock().unwrap();
        *count += 1;
        format!("[第{}次调用] {}", count, input)
    }
}
```

**所有权要点：** `&self` 方法签名意味着 `execute` 不拥有工具、也不独占工具。通过 `Mutex` 实现"外部不可变、内部可变"，这和 ZeroClaw 中 `SecurityPolicy` 用 `AtomicU32` 计数的模式一致。

---

## Step 3：构建工具注册表

```rust
/// 安全策略 —— 控制哪些工具可以被执行
///
/// 在 ZeroClaw 中，SecurityPolicy 通过 Arc 在所有工具间共享
/// 参见 sourcecode/zeroclaw/src/security/policy.rs
struct SecurityPolicy {
    allowed_tools: Vec<String>,
}

impl SecurityPolicy {
    fn new(allowed: Vec<&str>) -> Self {
        SecurityPolicy {
            allowed_tools: allowed.into_iter().map(String::from).collect(),
        }
    }

    fn is_allowed(&self, tool_name: &str) -> bool {
        self.allowed_tools.iter().any(|name| name == tool_name)
    }
}

/// 工具注册表 —— 管理所有工具的生命周期
///
/// 直接对标 ZeroClaw 源码中的模式：
///   fn default_tools(security: Arc<SecurityPolicy>) -> Vec<Box<dyn Tool>>
///
/// 关键所有权设计：
///   - tools: Vec<Box<dyn Tool>> —— 注册表拥有所有工具
///   - security: Arc<SecurityPolicy> —— 引用计数共享安全策略
struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
    security: Arc<SecurityPolicy>,
}

impl ToolRegistry {
    /// 创建新的注册表
    ///
    /// security: Arc<SecurityPolicy>
    /// ^^^^^^^^  ^^^
    /// |         |
    /// 值传递    Arc 的 clone 只增加引用计数
    fn new(security: Arc<SecurityPolicy>) -> Self {
        ToolRegistry {
            tools: Vec::new(),
            security,
        }
    }

    /// 注册一个工具 —— 获取工具的所有权
    ///
    /// tool: Box<dyn Tool>
    /// ^^^^  ^^^^^^^^^^^^^^
    /// |     |
    /// 值传递 所有权从调用者转移到注册表
    ///
    /// 注册后，调用者不再拥有这个工具
    fn register(&mut self, tool: Box<dyn Tool>) {
        println!("  注册工具: {} - {}", tool.name(), tool.description());
        self.tools.push(tool);
        // tool 的所有权已经 move 到 self.tools 中
        // 此后无法再使用 tool 变量
    }

    /// 查找工具 —— 返回借用引用，不转移所有权
    ///
    /// 返回 Option<&dyn Tool>
    ///             ^^^^^^^^
    ///             |
    ///             借用！注册表仍然拥有工具
    ///             返回的引用生命周期绑定到 &self
    fn find(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.iter().find(|t| t.name() == name).map(|t| {
            // t 的类型是 &Box<dyn Tool>
            // &**t 解引用为 &dyn Tool
            // Rust 会自动做这个转换（deref coercion）
            t.as_ref()
        })
    }

    /// 执行工具 —— 查找 + 安全检查 + 执行
    fn execute(&self, name: &str, input: &str) -> Result<String, String> {
        // 1. 安全检查（借用 security）
        if !self.security.is_allowed(name) {
            return Err(format!("安全策略禁止执行工具 '{}'", name));
        }

        // 2. 查找工具（借用 tools）
        match self.find(name) {
            Some(tool) => {
                // tool 是 &dyn Tool —— 借用引用
                // 工具仍然属于注册表
                Ok(tool.execute(input))
            }
            None => Err(format!("工具 '{}' 未注册", name)),
        }
    }

    /// 列出所有已注册的工具名称
    fn list_tools(&self) -> Vec<&str> {
        // 返回 Vec<&str> —— 字符串数据属于各工具
        // 只要 &self 有效，这些 &str 就有效
        self.tools.iter().map(|t| t.name()).collect()
    }
}
```

**所有权状态图：**

```
创建工具并注册后的内存布局:

ToolRegistry (栈上)
├── tools: Vec<Box<dyn Tool>> (堆上)
│   ├── Box<EchoTool>       ──→ [EchoTool 数据] (堆上)
│   ├── Box<UppercaseTool>  ──→ [UppercaseTool 数据] (堆上)
│   └── Box<CounterTool>    ──→ [CounterTool 数据] (堆上)
│                                  └── Arc<Mutex<u32>> ──→ [计数值] (堆上)
└── security: Arc ─────────────→ [SecurityPolicy 数据] (堆上, 引用计数=?)
                                    └── allowed_tools: Vec<String>

当调用 find("echo") 返回 &dyn Tool 时:
  - 返回的是指向堆上 EchoTool 的借用指针
  - 不发生任何所有权转移
  - 只要 ToolRegistry 活着，这个引用就有效
```

---

## Step 4：安全策略共享（Arc 详解）

```rust
/// 演示 Arc 的引用计数行为
///
/// 在 ZeroClaw 中，SecurityPolicy 通过 Arc 在所有工具和注册表之间共享：
///   Arc::new(ShellTool::new(security.clone(), runtime)),
///   Arc::new(FileReadTool::new(security.clone())),
///   ↑ 每次 clone 只增加引用计数，不复制 SecurityPolicy 数据
fn demonstrate_arc_sharing() {
    println!("\n=== Arc 引用计数演示 ===\n");

    let policy = Arc::new(SecurityPolicy::new(vec!["echo", "uppercase", "counter"]));
    println!("创建 Arc<SecurityPolicy>");
    println!("  引用计数: {}", Arc::strong_count(&policy));   // 1

    // clone 只增加引用计数，不复制数据
    let policy_clone1 = Arc::clone(&policy);
    println!("Arc::clone() 之后");
    println!("  引用计数: {}", Arc::strong_count(&policy));   // 2
    println!("  两个 Arc 指向同一块内存: {}",
             Arc::ptr_eq(&policy, &policy_clone1));           // true

    let policy_clone2 = policy.clone();  // 等价于 Arc::clone(&policy)
    println!("再次 clone 之后");
    println!("  引用计数: {}", Arc::strong_count(&policy));   // 3

    // 当一个 clone 被 drop，引用计数减少
    drop(policy_clone1);
    println!("drop 一个 clone 之后");
    println!("  引用计数: {}", Arc::strong_count(&policy));   // 2

    drop(policy_clone2);
    println!("drop 另一个 clone 之后");
    println!("  引用计数: {}", Arc::strong_count(&policy));   // 1
    println!("  数据仍然存活 —— 至少还有一个 Arc 指向它");
}
```

**Arc 与 TypeScript 的引用对比：**

```
TypeScript:
  const policy = { allowed: ["echo"] };
  const ref1 = policy;    // 共享引用，GC 追踪
  const ref2 = policy;    // 共享引用，GC 追踪
  // 当所有引用都不可达时，GC 回收

Rust:
  let policy = Arc::new(SecurityPolicy { ... });
  let ref1 = Arc::clone(&policy);  // 引用计数 +1
  let ref2 = Arc::clone(&policy);  // 引用计数 +1
  drop(ref1);                      // 引用计数 -1
  drop(ref2);                      // 引用计数 -1
  // 当引用计数归零，立即释放（确定性释放，无 GC 暂停）
```

---

## Step 5：完整的 main 函数

将以上所有代码组合成可运行的完整程序。将下面的代码保存为 `main.rs` 并用 `rustc main.rs && ./main` 编译运行。

```rust
use std::sync::{Arc, Mutex};

// ===== Trait 定义 =====
trait Tool {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn execute(&self, input: &str) -> String;
}

// ===== 具体工具实现 =====
struct EchoTool;

impl Tool for EchoTool {
    fn name(&self) -> &str { "echo" }
    fn description(&self) -> &str { "原样返回输入内容" }
    fn execute(&self, input: &str) -> String {
        input.to_string()
    }
}

struct UppercaseTool;

impl Tool for UppercaseTool {
    fn name(&self) -> &str { "uppercase" }
    fn description(&self) -> &str { "将输入转为大写字母" }
    fn execute(&self, input: &str) -> String {
        input.to_uppercase()
    }
}

struct CounterTool {
    count: Arc<Mutex<u32>>,
}

impl CounterTool {
    fn new() -> Self {
        CounterTool {
            count: Arc::new(Mutex::new(0)),
        }
    }
}

impl Tool for CounterTool {
    fn name(&self) -> &str { "counter" }
    fn description(&self) -> &str { "每次执行计数+1，返回当前计数" }
    fn execute(&self, input: &str) -> String {
        let mut count = self.count.lock().unwrap();
        *count += 1;
        format!("[第{}次调用] {}", count, input)
    }
}

// ===== 安全策略 =====
struct SecurityPolicy {
    allowed_tools: Vec<String>,
}

impl SecurityPolicy {
    fn new(allowed: Vec<&str>) -> Self {
        SecurityPolicy {
            allowed_tools: allowed.into_iter().map(String::from).collect(),
        }
    }

    fn is_allowed(&self, tool_name: &str) -> bool {
        self.allowed_tools.iter().any(|name| name == tool_name)
    }
}

// ===== 工具注册表 =====
struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
    security: Arc<SecurityPolicy>,
}

impl ToolRegistry {
    fn new(security: Arc<SecurityPolicy>) -> Self {
        ToolRegistry {
            tools: Vec::new(),
            security,
        }
    }

    fn register(&mut self, tool: Box<dyn Tool>) {
        println!("  [注册] {} - {}", tool.name(), tool.description());
        self.tools.push(tool);
    }

    fn find(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.iter().find(|t| t.name() == name).map(|t| t.as_ref())
    }

    fn execute(&self, name: &str, input: &str) -> Result<String, String> {
        if !self.security.is_allowed(name) {
            return Err(format!("安全策略禁止执行工具 '{}'", name));
        }
        match self.find(name) {
            Some(tool) => Ok(tool.execute(input)),
            None => Err(format!("工具 '{}' 未注册", name)),
        }
    }

    fn list_tools(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name()).collect()
    }
}

// ===== Arc 引用计数演示 =====
fn demonstrate_arc_sharing() {
    println!("\n==================================================");
    println!("=== Step 4: Arc 引用计数演示 ===");
    println!("==================================================\n");

    let policy = Arc::new(SecurityPolicy::new(vec!["echo", "uppercase", "counter"]));
    println!("创建 Arc<SecurityPolicy>");
    println!("  强引用计数: {}", Arc::strong_count(&policy));

    let policy_clone1 = Arc::clone(&policy);
    println!("Arc::clone() #1 之后");
    println!("  强引用计数: {}", Arc::strong_count(&policy));
    println!("  指向同一内存: {}", Arc::ptr_eq(&policy, &policy_clone1));

    let policy_clone2 = policy.clone();
    println!("Arc::clone() #2 之后");
    println!("  强引用计数: {}", Arc::strong_count(&policy));

    drop(policy_clone1);
    println!("drop clone1 之后");
    println!("  强引用计数: {}", Arc::strong_count(&policy));

    drop(policy_clone2);
    println!("drop clone2 之后");
    println!("  强引用计数: {}", Arc::strong_count(&policy));
    println!("  数据仍然存活，因为原始 Arc 还在");
}

// ===== 主函数 =====
fn main() {
    // ========================================
    // Step 1 & 2: 创建安全策略和工具
    // ========================================
    println!("==================================================");
    println!("=== Step 1 & 2: 创建安全策略和工具 ===");
    println!("==================================================\n");

    // 创建安全策略（用 Arc 包装，支持多处共享）
    let security = Arc::new(SecurityPolicy::new(vec!["echo", "uppercase", "counter"]));
    println!("安全策略已创建");
    println!("  允许的工具: echo, uppercase, counter");
    println!("  Arc 引用计数: {}\n", Arc::strong_count(&security));

    // ========================================
    // Step 3: 构建注册表并注册工具
    // ========================================
    println!("==================================================");
    println!("=== Step 3: 构建注册表并注册工具 ===");
    println!("==================================================\n");

    // Arc::clone 只增加引用计数
    let mut registry = ToolRegistry::new(Arc::clone(&security));
    println!("注册表已创建");
    println!("  Arc 引用计数: {} (注册表持有一份)\n", Arc::strong_count(&security));

    // 创建工具并注册
    // 注意：Box::new() 把工具放到堆上
    // register() 接收 Box<dyn Tool>，获取其所有权
    println!("注册工具:");
    let echo = Box::new(EchoTool);               // 在堆上创建 EchoTool
    registry.register(echo);                       // 所有权转移到 registry
    // echo 在这里已经无效！不能再使用

    registry.register(Box::new(UppercaseTool));    // 更常见的写法：直接传入
    registry.register(Box::new(CounterTool::new()));

    println!("\n已注册工具列表: {:?}", registry.list_tools());

    // ========================================
    // Step 5a: 执行工具
    // ========================================
    println!("\n==================================================");
    println!("=== Step 5: 执行工具 ===");
    println!("==================================================\n");

    // 测试正常执行
    let test_cases = vec![
        ("echo", "Hello, ZeroClaw!"),
        ("uppercase", "hello world"),
        ("counter", "第一次"),
        ("counter", "第二次"),
        ("counter", "第三次"),
    ];

    for (tool_name, input) in &test_cases {
        match registry.execute(tool_name, input) {
            Ok(output) => println!("  {} ({:?}) => {:?}", tool_name, input, output),
            Err(e) => println!("  {} ({:?}) => 错误: {}", tool_name, input, e),
        }
    }

    // 测试安全策略拦截
    println!("\n--- 测试安全策略拦截 ---");
    match registry.execute("dangerous_tool", "rm -rf /") {
        Ok(output) => println!("  输出: {}", output),
        Err(e) => println!("  被拦截: {}", e),
    }

    // 测试未注册工具
    println!("\n--- 测试未注册工具 ---");
    // 先临时修改安全策略为允许 nonexistent
    let permissive_security = Arc::new(SecurityPolicy::new(
        vec!["echo", "uppercase", "counter", "nonexistent"],
    ));
    let registry2 = ToolRegistry::new(permissive_security);
    // registry2 没有注册任何工具
    match registry2.execute("nonexistent", "test") {
        Ok(output) => println!("  输出: {}", output),
        Err(e) => println!("  未找到: {}", e),
    }

    // ========================================
    // Step 4: Arc 引用计数详解
    // ========================================
    demonstrate_arc_sharing();

    // ========================================
    // 所有权流转总结
    // ========================================
    println!("\n==================================================");
    println!("=== 所有权流转总结 ===");
    println!("==================================================\n");

    println!("当 registry 被 drop 时，以下事件按顺序发生:");
    println!("  1. Vec<Box<dyn Tool>> 被 drop");
    println!("     -> 每个 Box<dyn Tool> 被 drop");
    println!("        -> 每个具体工具被 drop (EchoTool, UppercaseTool, CounterTool)");
    println!("           -> CounterTool 的 Arc<Mutex<u32>> 引用计数 -1，归零则释放");
    println!("  2. Arc<SecurityPolicy> 引用计数 -1");
    println!("     -> 如果引用计数归零，SecurityPolicy 被释放");
    println!("  3. 一切都是确定性的 —— 没有 GC，没有 finalizer，没有意外");

    println!("\n当前 security 的 Arc 引用计数: {}", Arc::strong_count(&security));
    println!("registry 持有另一份，所以计数 = 2");

    // 手动 drop registry 观察引用计数变化
    drop(registry);
    println!("drop(registry) 之后:");
    println!("  security 的 Arc 引用计数: {}", Arc::strong_count(&security));
    println!("  注册表中的所有 Box<dyn Tool> 已被释放");
    println!("  但 SecurityPolicy 数据仍然存活（因为 security 变量还持有一份 Arc）");
}
```

---

## 预期输出

```
==================================================
=== Step 1 & 2: 创建安全策略和工具 ===
==================================================

安全策略已创建
  允许的工具: echo, uppercase, counter
  Arc 引用计数: 1

==================================================
=== Step 3: 构建注册表并注册工具 ===
==================================================

注册表已创建
  Arc 引用计数: 2 (注册表持有一份)

注册工具:
  [注册] echo - 原样返回输入内容
  [注册] uppercase - 将输入转为大写字母
  [注册] counter - 每次执行计数+1，返回当前计数

已注册工具列表: ["echo", "uppercase", "counter"]

==================================================
=== Step 5: 执行工具 ===
==================================================

  echo ("Hello, ZeroClaw!") => "Hello, ZeroClaw!"
  uppercase ("hello world") => "HELLO WORLD"
  counter ("第一次") => "[第1次调用] 第一次"
  counter ("第二次") => "[第2次调用] 第二次"
  counter ("第三次") => "[第3次调用] 第三次"

--- 测试安全策略拦截 ---
  被拦截: 安全策略禁止执行工具 'dangerous_tool'

--- 测试未注册工具 ---
  未找到: 工具 'nonexistent' 未注册

==================================================
=== Step 4: Arc 引用计数演示 ===
==================================================

创建 Arc<SecurityPolicy>
  强引用计数: 1
Arc::clone() #1 之后
  强引用计数: 2
  指向同一内存: true
Arc::clone() #2 之后
  强引用计数: 3
drop clone1 之后
  强引用计数: 2
drop clone2 之后
  强引用计数: 1
  数据仍然存活，因为原始 Arc 还在

==================================================
=== 所有权流转总结 ===
==================================================

当 registry 被 drop 时，以下事件按顺序发生:
  1. Vec<Box<dyn Tool>> 被 drop
     -> 每个 Box<dyn Tool> 被 drop
        -> 每个具体工具被 drop (EchoTool, UppercaseTool, CounterTool)
           -> CounterTool 的 Arc<Mutex<u32>> 引用计数 -1，归零则释放
  2. Arc<SecurityPolicy> 引用计数 -1
     -> 如果引用计数归零，SecurityPolicy 被释放
  3. 一切都是确定性的 —— 没有 GC，没有 finalizer，没有意外

当前 security 的 Arc 引用计数: 2
registry 持有另一份，所以计数 = 2
drop(registry) 之后:
  security 的 Arc 引用计数: 1
  注册表中的所有 Box<dyn Tool> 已被释放
  但 SecurityPolicy 数据仍然存活（因为 security 变量还持有一份 Arc）
```

---

## 所有权分析：逐步状态图

### 阶段 1：创建

```
main() 栈帧
│
├── security: Arc ──→ SecurityPolicy { allowed: [...] }  引用计数=1
│                     (堆上)
```

### 阶段 2：创建注册表

```
main() 栈帧
│
├── security: Arc ──┐
│                   ├──→ SecurityPolicy { allowed: [...] }  引用计数=2
├── registry        │
│   ├── tools: []   │
│   └── security: Arc ┘
```

### 阶段 3：注册工具

```
main() 栈帧
│
├── security: Arc ──────────────────┐
│                                   ├──→ SecurityPolicy  引用计数=2
├── registry                        │
│   ├── tools: Vec                  │
│   │   ├── Box ──→ EchoTool        │
│   │   ├── Box ──→ UppercaseTool   │
│   │   └── Box ──→ CounterTool     │
│   │                └── Arc<Mutex> ──→ u32(0)
│   └── security: Arc ─────────────┘
│
├── echo: (已移动，不可使用)  ← 所有权已转移给 registry.tools
```

### 阶段 4：查找工具（借用）

```
registry.find("echo") 返回 Option<&dyn Tool>
                                    │
                                    │ 借用引用
                                    ↓
tools: Vec ──→ [Box ──→ EchoTool]   ← 数据不动，只是创建一个临时引用
               ↑
               所有权不变
```

### 阶段 5：drop(registry)

```
drop 顺序（Rust 按字段声明的逆序 drop）：
  1. security: Arc 引用计数 2→1（不释放 SecurityPolicy）
  2. tools: Vec 被 drop
     → Box<CounterTool> 被 drop → CounterTool 被 drop → Arc<Mutex<u32>> 计数→0 → 释放
     → Box<UppercaseTool> 被 drop → UppercaseTool 被 drop（无堆内存）
     → Box<EchoTool> 被 drop → EchoTool 被 drop（无堆内存）

最终状态：
  main() 栈帧
  │
  ├── security: Arc ──→ SecurityPolicy  引用计数=1 (仍然存活)
  ├── registry: (已 drop)
```

---

## 与 TypeScript 详细对比

| 方面 | TypeScript | Rust |
|------|-----------|------|
| **工具存储** | `tools: Tool[]` 存引用 | `tools: Vec<Box<dyn Tool>>` 拥有所有权 |
| **异构容器** | 天然支持（鸭子类型） | 需要 `dyn Trait` 进行类型擦除 |
| **工具注册** | `push(tool)` 共享引用 | `push(tool)` 转移所有权 |
| **工具查找** | 返回 `Tool \| undefined` | 返回 `Option<&dyn Tool>` 借用 |
| **安全策略共享** | 所有地方用同一引用，GC 管理 | `Arc::clone()` 显式计数 |
| **释放时机** | GC 决定，不确定 | drop 时立即释放，确定性 |
| **线程安全** | 无保证（单线程事件循环） | `Send + Sync` 编译期保证 |
| **内存泄漏** | 循环引用可能泄漏 | `Arc` 循环引用也可能泄漏，但 `Box` 不会 |
| **运行时开销** | GC 暂停 | 零 GC 开销，只有引用计数原子操作 |

### 关键收获

TypeScript 开发者转 Rust 时，关于工具注册表模式要记住三件事：

1. **`Box<dyn Tool>` 就是 Rust 版的 `interface` 多态** —— 把不同类型装进同一个容器的方式。TypeScript 自然支持，Rust 需要显式声明。

2. **`Arc::clone()` 就是 Rust 版的共享引用** —— 但比 GC 更精确。你能随时查看引用计数，释放时机完全可预测。

3. **`&dyn Tool` 就是临时借来看看** —— 不拿走、不拥有。编译器保证你借来的东西在你用的时候不会被释放。

[来源: sourcecode/zeroclaw/src/tools/mod.rs]
[来源: sourcecode/zeroclaw/src/tools/delegate.rs]
