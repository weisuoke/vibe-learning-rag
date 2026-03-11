# 03_核心概念_1：Struct 定义与字段

> Struct 是 Rust 中组织数据的基础构建块，等价于 TypeScript 的 interface + class 的组合。
> 本文详解 struct 的三种形式、字段所有权、可见性封装、实例化语法和 Newtype 模式。

---

## 1. Struct 的三种形式

### 1.1 Named Struct（命名结构体）——最常用

所有字段都有名字，类似 TypeScript 的 interface：

```rust
struct Agent {                    // Rust: Named struct
    name: String,
    model: String,
    temperature: f64,
    max_tokens: u32,
}
```

```typescript
interface Agent {                 // TypeScript 等价物
  name: string;
  model: string;
  temperature: number;
  maxTokens: number;
}
```

**ZeroClaw 实战**：核心 `Agent` 结构体有 20+ 命名字段，`SecurityPolicy` 有 3 个字段，配置系统 `schema.rs` 中 70+ 个 struct 几乎全部是 Named struct。

### 1.2 Tuple Struct（元组结构体）——轻量包装

字段没有名字，通过位置索引访问：

```rust
struct Color(u8, u8, u8);          // RGB 颜色
struct Point2D(f64, f64);          // 二维坐标
struct UserId(u64);                // 单字段包装（Newtype 模式！）

let red = Color(255, 0, 0);
println!("R={}, G={}, B={}", red.0, red.1, red.2);  // 用 .0 .1 .2 访问
```

```typescript
type Color = [number, number, number];  // TypeScript 近似等价
type UserId = number;                   // 注意：这里没有类型安全！
```

**什么时候用？** 字段含义从类型名就能看出、单字段包装（Newtype）、字段少且语义明确。

### 1.3 Unit Struct（单元结构体）——零大小标记

没有任何字段，不占内存，用作类型标记：

```rust
struct Production;
struct Development;

struct Config<Env> {
    database_url: String,
    _env: std::marker::PhantomData<Env>,
}

// 编译期区分不同环境的配置——不同类型不能混用
fn create_prod_config() -> Config<Production> { /* ... */ }
fn create_dev_config() -> Config<Development> { /* ... */ }
```

### 三种形式对照总结

| 形式 | 语法 | 字段访问 | 典型用途 | TypeScript 类比 |
|------|------|---------|---------|----------------|
| Named | `struct S { x: T }` | `s.x` | 大多数数据结构 | `interface` |
| Tuple | `struct S(T1, T2)` | `s.0`, `s.1` | 轻量包装、Newtype | `type S = [T1, T2]` |
| Unit | `struct S;` | 无字段 | 类型标记 | 无直接等价 |

#### 双重类比

- **Named struct** = 表格（每栏都有表头）| 前端的 `interface`
- **Tuple struct** = 信封编号 (区号, 柜号, 层号) | 前端的 `tuple type`
- **Unit struct** = 一面旗帜，只是标记 | 无等价物

---

## 2. 字段类型与所有权

这是 TypeScript 开发者转向 Rust 时遇到的**第一个大坑**。

### 2.1 自有类型 vs 引用（String vs &str）

```rust
// 方案 A：自有类型——struct 拥有数据
struct AgentOwned {
    name: String,      // 拥有这块堆内存
    model: String,
}

// 方案 B：引用——struct 借用别人的数据
struct AgentBorrowed<'a> {
    name: &'a str,     // 借用，需要生命周期标注
    model: &'a str,
}
```

| 特性 | `String`（自有） | `&str`（引用） |
|------|------------------|----------------|
| 所有权 | struct 拥有数据 | struct 借用数据 |
| 生命周期 | 随 struct 销毁而释放 | 原始数据必须活得比 struct 久 |
| 灵活性 | 可随意存储、返回 | 受生命周期约束 |
| 复杂度 | 简单直接 | 需要 `<'a>` 标注 |

### 2.2 为什么 ZeroClaw 用 String 而不是 &str

ZeroClaw 的 `Agent` 字段全部使用自有类型（`String`、`Vec<T>`、`Box<dyn T>`）。原因：

1. **Agent 是长生命周期对象**——用 `&str` 需保证原始数据也活这么久，极难管理
2. **需要跨线程传递**——自有类型不依赖外部生命周期，可安全 `Arc<Mutex<Agent>>`
3. **Builder 模式需要移交所有权**——`String` 直接移动，`&str` 涉及复杂生命周期传递

**初学者经验法则：struct 字段优先用自有类型（String, Vec<T>, Box<dyn T>），等理解生命周期后再考虑引用。**

#### 双重类比

- `String` = 买了书放自己书架（你的，搬家跟着走）| React 组件的 **state**
- `&str` = 借了图书馆的书（到期必须还，图书馆关门就失效）| React 组件的 **props**

---

## 3. 字段可见性（pub / private）

### 3.1 默认私有

```rust
struct SecurityPolicy {
    autonomy_level: AutonomyLevel,   // 私有：模块外无法访问
    max_actions: usize,              // 私有
}
```

**Rust 的 private 是真的 private**——编译器强制执行，不像 TypeScript 的 `private` 可以用 `as any` 绕过。

### 3.2 pub 修饰符——选择性公开

```rust
// ZeroClaw 风格：SecurityPolicy 的字段是公开的
pub struct SecurityPolicy {
    pub autonomy_level: AutonomyLevel,   // 公开
    pub max_actions_per_turn: usize,     // 公开
    pub require_approval: bool,          // 公开
}
```

### 3.3 封装模式：私有字段 + 公有方法

```rust
pub struct ActionTracker {
    actions: Vec<String>,      // 私有：不允许外部直接操作
    max_actions: usize,        // 私有：创建后不可变
}

impl ActionTracker {
    pub fn new(max: usize) -> Self {
        ActionTracker { actions: Vec::new(), max_actions: max }
    }

    pub fn record_action(&mut self, action: String) -> Result<(), String> {
        if self.actions.len() >= self.max_actions {
            return Err("Action limit reached".to_string());
        }
        self.actions.push(action);
        Ok(())
    }

    pub fn action_count(&self) -> usize { self.actions.len() }
    pub fn can_act(&self) -> bool { self.actions.len() < self.max_actions }
}
```

**选择标准：** 字段可被随意修改而不破坏规则 -> 用 `pub`（如 SecurityPolicy）。修改字段需要额外逻辑 -> 私有字段 + 方法（如 ActionTracker）。

#### 双重类比

- **pub 字段** = 超市开放货架（自由拿取）| 组件的 public props
- **私有字段 + 方法** = 药房柜台（只能通过药剂师拿药）| 组件内部 state + setter

---

## 4. 实例化与字段简写

### 4.1 基本实例化

```rust
let p = Point { x: 1.0, y: 2.0 };  // 必须初始化所有字段（编译器强制，没有 undefined）
```

### 4.2 字段简写（Field Init Shorthand）

```rust
let x = 3.0;
let y = 4.0;
let p = Point { x, y };  // 等价于 Point { x: x, y: y }——和 TypeScript 完全相同！
```

### 4.3 Struct Update 语法（..展开）

```rust
#[derive(Debug, Clone)]
struct AgentConfig { name: String, model: String, temperature: f64, max_tokens: u32 }

let default_config = AgentConfig {
    name: String::from("default"), model: String::from("gpt-4"),
    temperature: 0.7, max_tokens: 4096,
};

let custom = AgentConfig {
    name: String::from("creative-agent"),
    temperature: 1.2,
    ..default_config  // 其余字段从 default_config "取来"
};
```

```typescript
const custom = { ...defaultConfig, name: "creative-agent", temperature: 1.2 };
```

### 4.4 移动语义陷阱（重要！）

**TypeScript 开发者最容易踩的坑之一：**

```rust
let config1 = AgentConfig { /* ... */ };
let config2 = AgentConfig { temperature: 1.0, ..config1 };

// 编译错误！config1 的 String 字段被 MOVE（移动）了！
// println!("{}", config1.name);  // ERROR: value used after move

// 但 Copy 类型字段仍然可以访问
println!("{}", config1.max_tokens);  // OK! u32 是 Copy 类型
```

`..config1` 不是"复制"——是**移动**（move）。TypeScript 的 `...` 是浅拷贝，两边都还能用。

**解决方案：**

```rust
// 方案 1：clone() 后再展开
let config2 = AgentConfig { temperature: 1.0, ..config1.clone() };

// 方案 2：为所有非 Copy 字段提供新值
let config2 = AgentConfig {
    name: String::from("new"),
    model: String::from("gpt-4o"),
    temperature: 1.0,
    ..config1  // 只有 max_tokens (u32, Copy) 从 config1 取
};
```

#### 双重类比

- TypeScript `...` = **复印机**（原件复印件都在）
- Rust `..` = **搬家公司**（旧家空了，想两边都有需先 clone）

---

## 5. Newtype 模式（重要！）

Rust 社区最推崇的设计模式之一：用 Tuple struct 包装现有类型，获得**零开销的类型安全**。

### 5.1 问题：类型太通用

```rust
// 危险：所有 ID 都是 u64，编译器无法区分
fn transfer(from_account: u64, to_account: u64, amount: u64) { /* ... */ }
transfer(amount, from_id, to_id);  // 参数顺序错了！编译器不报错！
```

### 5.2 解决：Newtype 包装

```rust
struct AccountId(u64);
struct Amount(u64);

fn transfer(from: AccountId, to: AccountId, amount: Amount) { /* ... */ }

transfer(from, to, amount);      // OK
// transfer(amount, from, to);   // 编译错误！Amount != AccountId
```

**零开销**：`size_of::<UserId>() == size_of::<u64>()` = 8 字节，编译器优化掉包装层。

### 5.3 ZeroClaw 应用场景

```rust
// 区分不同类型的字符串标识符
struct AgentId(String);
struct SessionId(String);
struct ToolCallId(String);

fn find_agent(id: AgentId) -> Option<Agent> { /* ... */ }
// find_agent(SessionId("abc".into()));  // 编译错误！

// 包装配置值，增加语义 + 验证
struct Temperature(f64);
impl Temperature {
    pub fn new(value: f64) -> Result<Self, String> {
        if value < 0.0 || value > 2.0 {
            return Err(format!("Temperature must be 0.0-2.0, got {}", value));
        }
        Ok(Temperature(value))
    }
    pub fn value(&self) -> f64 { self.0 }
}
```

### 5.4 Newtype 增强技巧

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct UserId(u64);

impl UserId {
    pub fn new(id: u64) -> Self { UserId(id) }
    pub fn inner(&self) -> u64 { self.0 }
}

impl std::fmt::Display for UserId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "User#{}", self.0)
    }
}
```

#### 前端开发类比

Newtype = TypeScript 的 branded types，但**编译器原生支持**：

```typescript
// TypeScript branded type（社区模式，需要 as 断言）
type UserId = number & { readonly __brand: unique symbol };
const userId = 42 as UserId;  // 需要 as 逃生舱
```

Rust 的 Newtype 不需要 `as`——类型安全是编译器直接保证的。

---

## 6. TypeScript -> Rust 对照总结

| TypeScript | Rust | 备注 |
|-----------|------|------|
| `interface Point { x: number }` | `struct Point { x: f64 }` | 定义数据结构 |
| `class Point { ... }` | `struct` + `impl` | Rust 分离数据和行为 |
| `public x` | `pub x: f64` | 公有字段 |
| `private x` | `x: f64`（默认） | Rust 的 private 是真 private |
| `readonly x` | `x: f64`（默认不可变） | Rust 默认不可变 |
| `{ x, y }` | `Point { x, y }` | 字段简写（相同！） |
| `{ ...old, x: 3.0 }` | `Point { x: 3.0, ..old }` | 注意移动语义！ |
| `type UserId = number` | `struct UserId(u64)` | Newtype 更安全 |
| `new Point(...)` | `Point::new(...)` | 关联函数用 `::` |
| `p.method()` | `p.method()` | 方法调用（相同！） |

---

## 7. 完整可运行代码示例

```rust
use std::fmt;

// --- Named Struct ---
#[derive(Debug, Clone)]
struct AgentConfig {
    pub name: String, pub model: String,
    pub temperature: f64, pub max_tokens: u32,
}
impl AgentConfig {
    fn new(name: &str, model: &str) -> Self {
        AgentConfig { name: name.into(), model: model.into(), temperature: 0.7, max_tokens: 4096 }
    }
}

// --- Tuple Struct (Newtype) ---
#[derive(Debug, Clone, PartialEq)]
struct AgentId(String);
#[derive(Debug, Clone, PartialEq)]
struct SessionId(String);

impl fmt::Display for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "Agent({})", self.0) }
}

// --- Unit Struct (Type State) ---
struct Validated;
struct Unvalidated;

struct UserInput<State> { content: String, _state: std::marker::PhantomData<State> }

impl UserInput<Unvalidated> {
    fn new(content: &str) -> Self {
        UserInput { content: content.into(), _state: std::marker::PhantomData }
    }
    fn validate(self) -> Result<UserInput<Validated>, String> {
        if self.content.is_empty() { return Err("Cannot be empty".into()); }
        Ok(UserInput { content: self.content, _state: std::marker::PhantomData })
    }
}
impl UserInput<Validated> {
    fn process(&self) -> String { format!("Processing: {}", self.content) }
}

// --- Encapsulation ---
struct ActionTracker { actions: Vec<String>, max: usize }
impl ActionTracker {
    fn new(max: usize) -> Self { ActionTracker { actions: Vec::new(), max } }
    fn record(&mut self, action: &str) -> Result<(), String> {
        if self.actions.len() >= self.max { return Err("Limit reached".into()); }
        self.actions.push(action.into()); Ok(())
    }
    fn count(&self) -> usize { self.actions.len() }
}

fn main() {
    // 1. Named struct + struct update
    let c1 = AgentConfig::new("assistant", "gpt-4");
    let c2 = AgentConfig { name: "creative".into(), temperature: 1.5, ..c1.clone() };
    println!("c1: {:?}", c1);
    println!("c2: {:?}", c2);

    // 2. Newtype 类型安全
    let aid = AgentId("agent-001".into());
    let sid = SessionId("session-abc".into());
    println!("{}", aid);
    // aid == sid;  // 编译错误！不同类型不能比较

    // 3. Unit struct + type state
    let input = UserInput::<Unvalidated>::new("Hello, Agent!");
    // input.process();  // 编译错误！Unvalidated 没有 process 方法
    if let Ok(valid) = input.validate() { println!("{}", valid.process()); }

    // 4. 封装：私有字段 + 公有方法
    let mut t = ActionTracker::new(2);
    t.record("search").unwrap();
    t.record("write").unwrap();
    match t.record("deploy") {
        Err(e) => println!("Expected: {}", e),  // "Limit reached"
        Ok(()) => {}
    }
    println!("Total actions: {}", t.count());
}
```

**编译运行：** `rustc main.rs && ./main`

**输出：**
```
c1: AgentConfig { name: "assistant", model: "gpt-4", temperature: 0.7, max_tokens: 4096 }
c2: AgentConfig { name: "creative", model: "gpt-4", temperature: 1.5, max_tokens: 4096 }
Agent(agent-001)
Processing: Hello, Agent!
Expected: Limit reached
Total actions: 2
```

---

## 速查卡

```
Struct 三种形式：
  Named:  struct S { field: Type }    -> s.field     -> 大多数场景
  Tuple:  struct S(Type1, Type2)      -> s.0, s.1    -> Newtype
  Unit:   struct S;                   -> 无字段      -> 类型标记

字段所有权：
  String（自有）: struct 拥有数据，安全简单    <- 初学者优先用这个
  &str（引用）:   struct 借用数据，需要生命周期

可见性：
  默认私有 -> 用 pub 公开
  封装 = 私有字段 + pub 方法

实例化：
  完整:  Point { x: 1.0, y: 2.0 }
  简写:  Point { x, y }           // 变量同名
  更新:  Point { x: 3.0, ..old }  // 注意：非 Copy 字段会被移动！

Newtype：
  struct UserId(u64);  // 零开销类型安全，编译器原生支持
```

---

*上一篇：[02_第一性原理](./02_第一性原理.md) -- Struct 与 Enum 的根本推理*
*下一篇：03_核心概念_2_impl块与方法 -- 方法定义、Builder 模式与方法链*
