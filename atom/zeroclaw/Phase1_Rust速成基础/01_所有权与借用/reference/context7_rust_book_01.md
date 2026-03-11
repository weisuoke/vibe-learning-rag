---
type: context7_documentation
library: The Rust Programming Language (Rust Book)
version: 2024 Edition
fetched_at: 2026-03-09
knowledge_point: 01_所有权与借用
context7_query: "ownership borrowing lifetimes"
---

# Context7 文档：The Rust Programming Language

## 文档来源
- 库名称：The Rust Programming Language (Rust Book)
- 版本：2024 Edition
- 官方文档链接：https://doc.rust-lang.org/stable/book/

## 关键信息提取

### 1. 所有权三规则

> * Each value in Rust has an **owner**.
> * There can only be **one owner** at a time.
> * When the owner goes out of scope, the value will be **dropped**.

### 2. 移动语义（Move Semantics）

当把 String 赋给另一个变量时，Rust 执行"移动"而非复制，原变量被无效化：

```rust
let s1 = String::from("hello");
let s2 = s1;
// s1 现在无效 —— 使用它会导致编译错误
```

**设计原因：** Rust 只复制栈上的指针、长度和容量，不深拷贝堆数据。原变量被无效化以防止"双重释放"错误。

### 3. 函数传参与所有权

传递堆分配值给函数会**移动**所有权：

```rust
fn main() {
    let s = String::from("hello");
    takes_ownership(s);     // s 的值移入函数
    // s 在这里不再有效

    let x = 5;
    makes_copy(x);          // i32 实现了 Copy，x 仍然有效
}

fn takes_ownership(some_string: String) {
    println!("{some_string}");
} // some_string 离开作用域，drop 被调用，内存被释放

fn makes_copy(some_integer: i32) {
    println!("{some_integer}");
} // 没有特殊操作
```

### 4. 返回所有权

```rust
fn gives_ownership() -> String {
    let some_string = String::from("yours");
    some_string  // 所有权转移给调用方
}

fn takes_and_gives_back(a_string: String) -> String {
    a_string  // 所有权返回给调用方
}
```

### 5. Copy Trait vs Move

实现了 Copy trait 的类型（栈数据）会被复制而非移动：
- 所有整数类型
- bool
- f32/f64
- char
- 只包含 Copy 类型的元组

### 6. clone() 深拷贝

```rust
let s1 = String::from("hello");
let s2 = s1.clone(); // 深拷贝堆数据
println!("s1 = {}, s2 = {}", s1, s2); // 两者都有效
```

### 7. 借用规则

编译器在编译时强制执行：
1. 任意时刻，要么有**一个**可变引用，要么有**任意数量**的不可变引用（但不能同时存在）
2. 引用必须始终**有效**（无悬垂引用）

### 8. 多个不可变引用 —— OK

```rust
let s = String::from("hello");
let r1 = &s; // OK
let r2 = &s; // OK —— 多个不可变引用没问题
```

### 9. 多个可变引用 —— 编译错误

```rust
let mut s = String::from("hello");
let r1 = &mut s;
let r2 = &mut s; // 错误：不能同时有两个可变借用
```

### 10. 混合可变和不可变引用 —— 编译错误

```rust
let mut s = String::from("hello");
let r1 = &s;     // OK
let r2 = &s;     // OK
let r3 = &mut s; // 错误：不可变引用存在时不能创建可变引用
```

### 11. 非词法生命周期（NLL）

引用的作用域从引入处延伸到**最后使用处**，而非到块末尾：

```rust
let mut s = String::from("hello");
let r1 = &s;
let r2 = &s;
println!("{} and {}", r1, r2);
// r1 和 r2 不再使用

let r3 = &mut s; // OK —— 之前的不可变借用已结束
println!("{}", r3);
```

### 12. 防止悬垂引用

```rust
fn dangle() -> &String {  // 错误：缺少生命周期
    let s = String::from("hello");
    &s  // s 在函数结束时被释放 —— 这将是悬垂引用
}

fn no_dangle() -> String {
    let s = String::from("hello");
    s  // 返回所有权，而非引用
}
```

### 13. 生命周期注解语法

```rust
&i32        // 一个引用
&'a i32     // 一个带显式生命周期的引用
&'a mut i32 // 一个带显式生命周期的可变引用
```

### 14. longest 函数

```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

含义：返回的引用至少与两个输入引用中**较短**的生命周期一样长。

### 15. 结构体中的生命周期

```rust
struct ImportantExcerpt<'a> {
    part: &'a str,
}
```

结构体不能比它引用的数据活得更久。

### 16. 生命周期省略规则

**规则 1：** 每个引用参数获得自己的生命周期参数。
**规则 2：** 如果只有一个输入生命周期参数，该生命周期赋给所有输出引用。
**规则 3：** 如果输入参数之一是 `&self` 或 `&mut self`，`self` 的生命周期赋给所有输出引用。

示例推导：
```rust
// 原始签名：
fn first_word(s: &str) -> &str {
// 规则 1 后：
fn first_word<'a>(s: &'a str) -> &str {
// 规则 2 后：
fn first_word<'a>(s: &'a str) -> &'a str {
```

### 17. 'static 生命周期

```rust
let s: &'static str = "I have a static lifetime.";
```

字符串字面量都有 `'static` 生命周期，因为数据嵌入在程序二进制中。

### 18. 泛型 + Trait 约束 + 生命周期组合

```rust
fn longest_with_an_announcement<'a, T>(
    x: &'a str,
    y: &'a str,
    ann: T,
) -> &'a str
where
    T: Display,
{
    println!("Announcement! {ann}");
    if x.len() > y.len() { x } else { y }
}
```
