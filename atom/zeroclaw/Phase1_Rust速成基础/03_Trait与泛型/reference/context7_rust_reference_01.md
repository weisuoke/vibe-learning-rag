---
type: context7_documentation
library: Rust Reference
version: latest
fetched_at: 2026-03-10
knowledge_point: 03_Trait与泛型
context7_query: traits generics where clause associated types impl trait
---

# Context7 文档：Rust Reference - Traits & Generics

## 文档来源
- 库名称：Rust Reference
- Context7 ID：/rust-lang/reference
- 官方文档链接：https://doc.rust-lang.org/reference/

## 关键信息提取

### 1. 泛型函数与 Trait 约束
```rust
fn foo<T>(x: T) where T: Debug { }
```
- `where` 子句指定泛型参数的 Trait 约束
- 等价于内联语法 `fn foo<T: Debug>(x: T)`

### 2. Trait 约束满足规则
```rust
trait Shape {
    fn draw(&self, surface: Surface);
    fn name() -> &'static str;
}

fn draw_twice<T: Shape>(surface: Surface, sh: T) {
    sh.draw(surface);  // T: Shape 所以可以调用
}

fn copy_and_draw_twice<T: Copy>(surface: Surface, sh: T) where T: Shape {
    let shape_copy = sh;        // T: Copy 所以不会移动
    draw_twice(surface, sh);    // T: Shape 所以可以调用泛型函数
}

struct Figure<S: Shape>(S, S);  // 泛型结构体约束
```

### 3. Where 子句高级用法
```rust
struct A<T>
where
    T: Iterator,            // 基本约束
    T::Item: Copy,          // 关联类型约束
    String: PartialEq<T>,   // 非参数类型约束
{
    f: T,
}
```
- 可以约束关联类型：`T::Item: Copy`
- 可以约束非泛型参数的类型：`String: PartialEq<T>`
- `for<'a>` 引入高阶生命周期

### 4. TypeParamBounds 语法
- 多个约束用 `+` 连接：`T: Debug + Clone + Send`
- 支持生命周期约束：`T: 'static`
- 支持 `?Sized` 放松约束
- 支持 `for<'a>` 高阶 Trait 约束（HRTB）
- Rust 2024: `use<'a, T>` 精确控制捕获
