---
type: context7_documentation
library: Rust
version: latest
fetched_at: 2026-03-09
knowledge_point: 02_Struct与Enum
context7_query: "struct enum pattern matching derive"
context7_sources:
  - /rust-lang/rust (Code Snippets: 5082, Reputation: High)
  - /google/comprehensive-rust (Code Snippets: 2111, Reputation: High)
---

# Context7 文档：Rust Struct 与 Enum

## 文档来源
- 库名称：Rust (`/rust-lang/rust`) + Comprehensive Rust (`/google/comprehensive-rust`)
- 版本：latest
- 获取时间：2026-03-09

---

## 关键信息提取

### 1. Struct 相关

#### 1.1 Struct 基本定义

Rust 中的 `struct` 用于定义自定义数据类型，包含命名字段。Struct 只定义数据字段，方法通过单独的 `impl` 块定义。

```rust
struct Foo {
    x: u32,
}
```

**关键点：**
- Struct 替代了传统 OOP 中的 class
- Struct 定义数据，`impl` 块定义行为
- 没有显式的构造函数语法，约定使用 `new()` 关联函数

#### 1.2 Struct 字段可见性

字段默认是私有的。如果需要外部访问，需要用 `pub` 修饰：

```rust
mod bar {
    pub struct Foo {
        pub a: isize,   // 公有字段，外部可访问
        b: isize,       // 私有字段，外部不可直接访问
    }

    impl Foo {
        pub fn new() -> Foo {  // 提供公有构造函数
            Foo { a: 0, b: 0 }
        }
    }
}

let f = bar::Foo::new(); // 通过构造函数创建实例
```

**关键点：**
- 字段默认私有，用 `pub` 声明公有
- 私有字段需要通过公有方法（如 `new()`）来初始化
- 这种模式保证了封装性

#### 1.3 impl 块与方法定义

方法通过 `impl` 块关联到类型。可以有多个 `impl` 块。

```rust
struct Foo {
    x: u32,
}

// Trait 实现
trait Bar {
    fn bar(&self) -> i32;
}

impl Bar for Foo {
    fn bar(&self) -> i32 {
        self.x
    }
}

// 固有方法实现
impl Foo {
    // 消耗 self 的方法（获取所有权）
    fn baz(mut self, f: Foo) -> i32 {
        f.baz(self)
    }

    // 可变引用方法
    fn qux(&mut self) {
        self.x = 0;
    }

    // 不可变引用方法
    fn quop(&self) -> i32 {
        self.x
    }
}
```

**方法的三种 self 形式：**
| 形式 | 含义 | 使用场景 |
|------|------|---------|
| `self` | 获取所有权 | 消耗实例，如 builder 模式 |
| `&self` | 不可变引用 | 只读访问 |
| `&mut self` | 可变引用 | 需要修改实例 |

#### 1.4 关联函数（构造函数模式）

没有 `self` 参数的函数是关联函数（类似于静态方法），常用作构造函数：

```rust
impl Foo {
    pub const bar: bool = true;

    /// 构造函数 - 关联函数（没有 self 参数）
    pub const fn new() -> Foo {
        Foo { bar: true }
    }
}

// 调用方式：使用 :: 而不是 .
let foo = Foo::new();
```

**关键点：**
- 关联函数用 `Type::function()` 调用（双冒号）
- 方法用 `instance.method()` 调用（点号）
- `new()` 是 Rust 社区约定的构造函数名称
- `const fn` 可以在编译时求值

---

### 2. Enum 相关

#### 2.1 Enum 基本定义与变体类型

Rust 的 enum 比 C/Java 的枚举强大得多，每个变体可以携带不同类型的数据：

```rust
#[derive(Debug)]
enum Message {
    Quit,                       // 无数据变体（Unit variant）
    Move { x: i32, y: i32 },   // 结构体变体（Struct variant）
    Write(String),              // 元组变体（Tuple variant）
    ChangeColor(u8, u8, u8),   // 多字段元组变体
}
```

**三种变体类型：**
| 变体类型 | 语法 | 示例 |
|---------|------|------|
| Unit variant | `Name` | `Quit` - 不携带数据 |
| Tuple variant | `Name(T1, T2)` | `Write(String)` - 携带匿名数据 |
| Struct variant | `Name { field: T }` | `Move { x: i32, y: i32 }` - 携带命名数据 |

#### 2.2 Enum 上的方法实现

Enum 也可以有 `impl` 块，定义方法：

```rust
impl Message {
    fn process(&self) {
        match self {
            Message::Quit => println!("Quitting"),
            Message::Move { x, y } => println!("Moving to ({x}, {y})"),
            Message::Write(text) => println!("Writing: {text}"),
            Message::ChangeColor(r, g, b) => println!("Color: RGB({r}, {g}, {b})"),
        }
    }
}

fn main() {
    let messages = [
        Message::Quit,
        Message::Move { x: 10, y: 20 },
        Message::Write(String::from("Hello")),
        Message::ChangeColor(255, 128, 0),
    ];

    for msg in &messages {
        msg.process();
    }
}
```

#### 2.3 Option<T> - 处理可能为空的值

`Option<T>` 是 Rust 标准库的核心 enum，替代 null：

```rust
// Option 定义（标准库内置）
// enum Option<T> {
//     Some(T),
//     None,
// }

fn find_first_even(numbers: &[i32]) -> Option<i32> {
    for &num in numbers {
        if num % 2 == 0 {
            return Some(num);
        }
    }
    None
}

fn main() {
    let numbers = vec![1, 3, 5, 7, 9];
    match find_first_even(&numbers) {
        Some(even) => println!("First even number: {}", even),
        None => println!("No even numbers found."),
    }
}
```

**Option 的常用方法：**

```rust
fn divide(numerator: f64, denominator: f64) -> Option<f64> {
    if denominator == 0.0 {
        None
    } else {
        Some(numerator / denominator)
    }
}

fn main() {
    // 1. Pattern matching（最安全）
    match divide(10.0, 2.0) {
        Some(result) => println!("Result: {result}"),
        None => println!("Cannot divide by zero"),
    }

    // 2. unwrap 方法（不安全，None 时 panic）
    let x = Some(5);
    println!("Unwrap: {}", x.unwrap());

    // 3. unwrap_or - 提供默认值
    println!("Unwrap or: {}", None::<i32>.unwrap_or(0));

    // 4. unwrap_or_else - 提供默认值计算函数
    println!("Unwrap or else: {}", None::<i32>.unwrap_or_else(|| 42));

    // 5. if let - 只关心 Some 的情况
    if let Some(value) = divide(10.0, 3.0) {
        println!("Got value: {value}");
    }

    // 6. map - 转换 Some 中的值
    let doubled = Some(5).map(|x| x * 2);
    println!("Doubled: {:?}", doubled);  // Some(10)

    // 7. filter - 按条件过滤
    let filtered = Some(10).filter(|x| *x > 5);
    println!("Filtered: {:?}", filtered);  // Some(10)
}
```

#### 2.4 Result<T, E> - 错误处理

`Result<T, E>` 是 Rust 的错误处理核心 enum：

```rust
// Result 定义（标准库内置）
// enum Result<T, E> {
//     Ok(T),
//     Err(E),
// }

// 自定义 Result 示例
enum Result {
    Ok(i32),
    Err(String),
}

fn divide_in_two(n: i32) -> Result {
    if n % 2 == 0 {
        Result::Ok(n / 2)
    } else {
        Result::Err(format!("cannot divide {n} into two equal parts"))
    }
}

fn main() {
    let n = 100;
    match divide_in_two(n) {
        Result::Ok(half) => println!("{n} divided in two is {half}"),
        Result::Err(msg) => println!("sorry, an error happened: {msg}"),
    }
}
```

**使用标准库 Result 和 ? 操作符：**

```rust
use std::fs::File;
use std::io::{self, Read};

fn read_file_contents(path: &str) -> Result<String, io::Error> {
    let mut file = File::open(path)?;  // ? 操作符：出错时提前返回 Err
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

fn main() {
    // Pattern matching
    match read_file_contents("example.txt") {
        Ok(contents) => println!("File contents: {contents}"),
        Err(e) => println!("Error reading file: {e}"),
    }

    // Combinators
    let result: Result<i32, &str> = Ok(5);
    let doubled = result.map(|x| x * 2);
    println!("Doubled: {:?}", doubled);

    // unwrap_or
    let value = Ok::<i32, &str>(42).unwrap_or(0);
    println!("Value: {value}");

    // Option -> Result 转换
    let opt = Some(5);
    let res: Result<i32, &str> = opt.ok_or("No value");
    println!("Converted: {:?}", res);
}
```

**? 操作符关键点：**
- `?` 只能在返回 `Result` 或 `Option` 的函数中使用
- 如果值是 `Ok(v)` / `Some(v)`，提取 `v` 继续执行
- 如果值是 `Err(e)` / `None`，立即从函数返回该错误
- 大幅减少 `match` 嵌套，使错误处理代码更简洁

---

### 3. Pattern Matching 相关

#### 3.1 match 表达式

`match` 是 Rust 中最强大的控制流结构，必须穷尽所有可能：

```rust
// 解构 enum 变体
match self {
    Message::Quit => println!("Quitting"),
    Message::Move { x, y } => println!("Moving to ({x}, {y})"),
    Message::Write(text) => println!("Writing: {text}"),
    Message::ChangeColor(r, g, b) => println!("Color: RGB({r}, {g}, {b})"),
}
```

**match 的关键规则：**
- **穷尽性（Exhaustiveness）**：必须覆盖所有可能的模式
- **模式绑定**：可以在模式中绑定变量（如 `x`, `y`, `text`）
- **`_` 通配符**：匹配任意值，用于兜底

#### 3.2 if let 简化匹配

当只关心一种模式时，使用 `if let` 代替完整的 `match`：

```rust
// 完整 match
match divide(10.0, 3.0) {
    Some(value) => println!("Got value: {value}"),
    None => {},  // 不关心 None，但必须写
}

// 等价的 if let（更简洁）
if let Some(value) = divide(10.0, 3.0) {
    println!("Got value: {value}");
}
```

#### 3.3 解构模式总结

| 模式 | 示例 | 用途 |
|------|------|------|
| 字面值 | `42`, `"hello"` | 匹配具体值 |
| 变量绑定 | `x`, `name` | 捕获值到变量 |
| 通配符 | `_` | 忽略值 |
| 元组解构 | `(x, y)` | 解构元组 |
| 结构体解构 | `Point { x, y }` | 解构结构体字段 |
| Enum 解构 | `Some(v)`, `Err(e)` | 解构 enum 变体 |
| 引用模式 | `&x`, `ref x` | 匹配引用 |

---

### 4. Derive Macros 相关

#### 4.1 #[derive] 属性基础

`#[derive]` 属性自动为类型实现常见 trait，避免手写 boilerplate 代码：

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
struct Player {
    name: String,
    score: u32,
    level: u8,
}
```

#### 4.2 常用 Derive Trait 详解

| Trait | 功能 | 使用示例 |
|-------|------|---------|
| `Debug` | 调试格式化输出 | `println!("{:?}", value)` |
| `Clone` | 深拷贝 | `let copy = value.clone()` |
| `Copy` | 隐式拷贝（栈上数据） | `let y = x;` (x 仍可用) |
| `PartialEq` | `==` 和 `!=` 比较 | `a == b` |
| `Eq` | 完全相等（PartialEq 的子集） | 用于 HashMap key |
| `PartialOrd` | `<`, `>`, `<=`, `>=` 比较 | `a > b` |
| `Ord` | 完全排序 | 用于 BTreeMap key |
| `Hash` | 哈希计算 | 用于 HashSet/HashMap |
| `Default` | 默认值 | `Player::default()` |

#### 4.3 Derive 使用示例

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
struct Player {
    name: String,
    score: u32,
    level: u8,
}

fn main() {
    // Default - 创建默认值实例
    let mut player1 = Player::default();
    player1.name = String::from("Alice");
    player1.score = 100;

    // Debug - 启用 {:?} 格式化
    println!("Player: {:?}", player1);

    // Clone - 深拷贝
    let mut player2 = player1.clone();
    player2.name = String::from("Bob");

    // PartialEq - 启用 == 比较
    let player3 = player1.clone();
    println!("player1 == player3: {}", player1 == player3);  // true
    println!("player1 == player2: {}", player1 == player2);  // false

    // Hash - 可用于 HashSet/HashMap
    use std::collections::HashSet;
    let mut players = HashSet::new();
    players.insert(player1);
    players.insert(player2);
    println!("Unique players: {}", players.len());  // 2
}
```

#### 4.4 Derive 用于 Enum

Enum 同样可以使用 `#[derive]`：

```rust
#[derive(Debug)]
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(u8, u8, u8),
}

// 现在可以用 {:?} 打印
println!("{:?}", Message::Quit);  // Message::Quit
println!("{:?}", Message::Move { x: 10, y: 20 });  // Move { x: 10, y: 20 }
```

#### 4.5 公有 Struct 的推荐 Derive 组合

针对公有 API，Rust 社区推荐尽可能多地 derive 常用 trait：

```rust
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub struct MyData {
    pub name: String,
    pub number: usize,
    pub data: [u8; 64],
}
```

**推荐原则：**
- 公有类型应尽可能实现 `Debug`、`Clone`、`PartialEq`
- 如果 `PartialEq` 可 derive，通常也应 derive `Eq`
- 用于集合键（HashMap/HashSet）的类型需要 `Hash` + `Eq`
- 需要排序的类型需要 `Ord` + `Eq`
- `Copy` 仅适用于小型栈上数据（所有字段都是 Copy 的类型）

#### 4.6 Copy vs Clone 关键区别

| 特性 | Copy | Clone |
|------|------|-------|
| 语义 | 隐式按位拷贝 | 显式深拷贝 |
| 调用方式 | 自动 (`let y = x;`) | 手动 (`.clone()`) |
| 适用类型 | 简单栈上数据 (i32, f64, bool) | 任意类型 |
| String 能否用 | 不能（堆上数据） | 可以 |
| 性能 | 极快（memcpy） | 可能慢（堆分配） |

---

## 速查表

### Struct 速查

```rust
// 定义
struct Point { x: f64, y: f64 }

// 实例化
let p = Point { x: 1.0, y: 2.0 };

// 字段访问
println!("{}", p.x);

// 方法
impl Point {
    fn new(x: f64, y: f64) -> Self { Point { x, y } }  // 关联函数
    fn distance(&self) -> f64 { (self.x.powi(2) + self.y.powi(2)).sqrt() }  // 方法
}

// 使用
let p = Point::new(3.0, 4.0);
println!("{}", p.distance());  // 5.0
```

### Enum 速查

```rust
// 定义
enum Shape {
    Circle(f64),           // 元组变体
    Rectangle(f64, f64),   // 多字段元组
    Triangle { base: f64, height: f64 },  // 结构体变体
}

// 使用 match
fn area(shape: &Shape) -> f64 {
    match shape {
        Shape::Circle(r) => std::f64::consts::PI * r * r,
        Shape::Rectangle(w, h) => w * h,
        Shape::Triangle { base, height } => 0.5 * base * height,
    }
}
```

### Derive 速查

```rust
// 最常用组合
#[derive(Debug, Clone, PartialEq)]           // 基础三件套
#[derive(Debug, Clone, PartialEq, Eq, Hash)] // 可用于 HashMap key
#[derive(Debug, Clone, Copy, PartialEq)]     // 小型值类型
#[derive(Debug, Default)]                     // 需要默认值
```
