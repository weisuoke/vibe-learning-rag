# 实战代码 场景 2：Builder 模式

> **一句话定位：** 用 Option<T> + 链式方法 + build() 验证，模拟 ZeroClaw 的 AgentBuilder，体验 Rust 中最常用的构造模式。

---

## 场景背景

### ZeroClaw 的 AgentBuilder

ZeroClaw 用 Builder 模式构造 Agent：字段多（10+）、部分可选、需要验证。直接 `Agent { ... }` 创建太痛苦，Builder 让你按需设置、最后一次性验证。

```
AgentBuilder::new()
    .with_name("assistant")
    .with_provider(provider)
    .with_security_policy(policy)
    .build()  // → Result<Agent, String>
```

### TypeScript 中你怎么做

```typescript
// TypeScript: 可选参数 + Partial<T>
interface AgentConfig {
  name: string;
  model: string;
  temperature?: number;
  maxActions?: number;
}

function createAgent(config: AgentConfig) {
  return { ...config, temperature: config.temperature ?? 0.7 };
}

// 问题 1：必填字段忘了传？运行时才发现
// 问题 2：字段多了之后参数顺序混乱
// 问题 3：没有统一的验证时机
```

Rust 的 Builder 模式：编译期区分"已设置"和"未设置"，build() 时统一验证。

---

## 第一步：定义目标 Struct 和 Builder

```rust
#[derive(Debug)]
struct AgentConfig {
    name: String,
    model: String,
    temperature: f64,
    max_actions: u32,
    description: Option<String>,
}

#[derive(Default)]
struct AgentConfigBuilder {
    name: Option<String>,        // 必填，但构建时先用 Option
    model: Option<String>,       // 必填
    temperature: Option<f64>,    // 可选，有默认值
    max_actions: Option<u32>,    // 可选，有默认值
    description: Option<String>, // 可选
}
```

> **关键设计：** Builder 的所有字段都是 `Option<T>`，即使目标 struct 中是必填的。这样 Builder 创建时不需要任何参数。

---

## 第二步：链式方法

```rust
impl AgentConfigBuilder {
    fn new() -> Self {
        Self::default()
    }

    fn name(mut self, name: &str) -> Self {
        self.name = Some(name.into());
        self
    }

    fn model(mut self, model: &str) -> Self {
        self.model = Some(model.into());
        self
    }

    fn temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp);
        self
    }

    fn max_actions(mut self, max: u32) -> Self {
        self.max_actions = Some(max);
        self
    }

    fn description(mut self, desc: &str) -> Self {
        self.description = Some(desc.into());
        self
    }
}
```

> **`mut self`（不是 `&mut self`）：** 每个方法消费 self 并返回修改后的 self。这保证链式调用的线性使用——你不能在 `.name()` 之后再用旧的 builder。

---

## 第三步：build() 验证

```rust
impl AgentConfigBuilder {
    fn build(self) -> Result<AgentConfig, String> {
        let name = self.name.ok_or("name is required")?;
        let model = self.model.ok_or("model is required")?;

        Ok(AgentConfig {
            name,
            model,
            temperature: self.temperature.unwrap_or(0.7),
            max_actions: self.max_actions.unwrap_or(10),
            description: self.description,
        })
    }
}
```

> **`.ok_or("...")?`：** 把 `Option<T>` 转成 `Result<T, E>`。如果是 `None`，立即返回错误。这是 Rust 中验证必填字段的惯用写法。

---

## 第四步：完整可运行代码

```rust
// --- 目标类型 ---
#[derive(Debug)]
struct AgentConfig {
    name: String,
    model: String,
    temperature: f64,
    max_actions: u32,
    description: Option<String>,
}

// --- Builder ---
#[derive(Default)]
struct AgentConfigBuilder {
    name: Option<String>,
    model: Option<String>,
    temperature: Option<f64>,
    max_actions: Option<u32>,
    description: Option<String>,
}

impl AgentConfigBuilder {
    fn new() -> Self { Self::default() }

    fn name(mut self, name: &str) -> Self { self.name = Some(name.into()); self }
    fn model(mut self, model: &str) -> Self { self.model = Some(model.into()); self }
    fn temperature(mut self, t: f64) -> Self { self.temperature = Some(t); self }
    fn max_actions(mut self, m: u32) -> Self { self.max_actions = Some(m); self }
    fn description(mut self, d: &str) -> Self { self.description = Some(d.into()); self }

    fn build(self) -> Result<AgentConfig, String> {
        let name = self.name.ok_or("name is required".to_string())?;
        let model = self.model.ok_or("model is required".to_string())?;
        Ok(AgentConfig {
            name,
            model,
            temperature: self.temperature.unwrap_or(0.7),
            max_actions: self.max_actions.unwrap_or(10),
            description: self.description,
        })
    }
}

// --- 给 AgentConfig 加便捷入口 ---
impl AgentConfig {
    fn builder() -> AgentConfigBuilder {
        AgentConfigBuilder::new()
    }
}

// --- main ---
fn main() {
    // 成功：必填字段都提供了
    let config = AgentConfig::builder()
        .name("assistant")
        .model("gpt-4")
        .temperature(0.9)
        .description("A helpful agent")
        .build();

    match &config {
        Ok(c) => println!("OK: {:#?}", c),
        Err(e) => println!("Error: {}", e),
    }

    // 成功：只填必填，其余用默认值
    let minimal = AgentConfig::builder()
        .name("bot")
        .model("gpt-3.5")
        .build();

    println!("\nMinimal: {:#?}", minimal);

    // 失败：缺少必填字段
    let bad = AgentConfig::builder()
        .name("oops")
        .build();

    println!("\nBad: {:?}", bad);
}
```

**编译运行：** `rustc main.rs && ./main`

---

## 运行输出

```
OK: AgentConfig {
    name: "assistant",
    model: "gpt-4",
    temperature: 0.9,
    max_actions: 10,
    description: Some(
        "A helpful agent",
    ),
}

Minimal: Ok(
    AgentConfig {
        name: "bot",
        model: "gpt-3.5",
        temperature: 0.7,
        max_actions: 10,
        description: None,
    },
)

Bad: Err("model is required")
```

---

## TypeScript vs Rust 对比

| 对比项 | TypeScript | Rust Builder |
|--------|-----------|-------------|
| 可选字段 | `?` 语法 | `Option<T>` |
| 默认值 | `?? 0.7` 散落各处 | `unwrap_or(0.7)` 集中在 build() |
| 必填验证 | 运行时检查或无 | `ok_or()?` 编译期强制处理 |
| 链式调用 | 返回 `this` | 返回 `self`（所有权转移） |
| 构建失败 | throw 或 undefined | `Result<T, E>` 强制处理 |

---

## 学到了什么

| 知识点 | 在本场景中的体现 |
|--------|-----------------|
| Option\<T\> 全字段 | Builder 所有字段都是 Option，创建时无需任何参数 |
| `mut self` 链式 | 每个方法消费并返回 self，保证线性使用 |
| `.ok_or()?` | Option → Result 转换，验证必填字段 |
| `.unwrap_or()` | 提供默认值，处理可选字段 |
| `#[derive(Default)]` | Builder::new() 零样板代码 |
| `Result<T, E>` | build() 返回值强制调用者处理成功/失败 |
| 便捷入口 | `AgentConfig::builder()` 隐藏 Builder 类型 |

---

*上一篇：[07_实战代码_场景1_Agent消息系统](./07_实战代码_场景1_Agent消息系统.md)*
*下一篇：[07_实战代码_场景3_事件调度系统](./07_实战代码_场景3_事件调度系统.md)*
