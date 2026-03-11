---
type: context7_documentation
library: Rust Reference
version: latest (2026)
fetched_at: 2026-03-10
knowledge_point: 05_动态分发与 Trait Object
context7_query: trait objects dynamic dispatch dyn keyword Box dyn Arc dyn object safety vtable rules
---

# Context7 文档：Rust Reference - Trait Objects

## 文档来源
- 库名称：Rust Reference
- 版本：latest
- Context7 Library ID: /rust-lang/reference

## 关键信息提取

### 1. Dyn Compatible Trait Methods

可以通过 trait object 动态分发的方法，需要使用以下 receiver 类型：

```rust
trait TraitMethods {
    fn by_ref(self: &Self) {}
    fn by_ref_mut(self: &mut Self) {}
    fn by_box(self: Box<Self>) {}
    fn by_rc(self: Rc<Self>) {}
    fn by_arc(self: Arc<Self>) {}
    fn by_pin(self: Pin<&Self>) {}
    fn with_lifetime<'a>(self: &'a Self) {}
    fn nested_pin(self: Pin<Arc<Self>>) {}
}
```

### 2. Dyn Incompatible Methods（不能动态分发但 trait 仍然 object-safe）

以下方法不能通过 trait object 调用，但加上 `where Self: Sized` 后 trait 本身仍然是 object-safe 的：

```rust
trait NonDispatchable {
    fn foo() where Self: Sized {}           // 非方法
    fn returns(&self) -> Self where Self: Sized;  // 返回 Self
    fn param(&self, other: Self) where Self: Sized {}  // Self 作为参数
    fn typed<T>(&self, x: T) where Self: Sized {}  // 泛型参数
}
```

### 3. Dyn Incompatible Traits（完全不能作为 trait object）

以下特征使 trait 完全不能作为 `dyn Trait` 使用：

```rust
trait DynIncompatible {
    const CONST: i32 = 1;           // 关联常量
    fn foo() {}                      // 关联函数（无 self，无 Sized 约束）
    fn returns(&self) -> Self;       // 返回 Self（无 Sized 约束）
    fn typed<T>(&self, x: T) {}     // 泛型方法（无 Sized 约束）
    fn nested(self: Rc<Box<Self>>) {} // 嵌套 receiver
}
```

### 4. Trait Object 的本质

> The purpose of trait objects is to permit "late binding" of methods.
> Calling a method on a trait object results in virtual dispatch at runtime:
> that is, a function pointer is loaded from the trait object vtable and invoked indirectly.
> The actual implementation for each vtable entry can vary on an object-by-object basis.

**Trait Object 是胖指针**：data pointer + vtable pointer

```rust
trait Printable {
    fn stringify(&self) -> String;
}

impl Printable for i32 {
    fn stringify(&self) -> String { self.to_string() }
}

fn print(a: Box<dyn Printable>) {
    println!("{}", a.stringify());
}

fn main() {
    print(Box::new(10) as Box<dyn Printable>);
}
```
