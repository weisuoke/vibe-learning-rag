---
type: context7_documentation
library: Comprehensive Rust (Google)
version: 2026
fetched_at: 2026-03-10
knowledge_point: 07_Cargo与模块系统
context7_query: Rust module system mod use pub visibility crate path super self
---

# Context7 文档：Rust 模块系统 (Google Comprehensive Rust)

## 文档来源
- 库名称：Comprehensive Rust (Google)
- 版本：2026
- 官方文档链接：https://github.com/google/comprehensive-rust

## 关键信息提取

### 1. 模块可见性 (pub)

```rust
mod outer {
    fn private() {
        println!("outer::private");
    }

    pub fn public() {
        println!("outer::public");
    }

    mod inner {
        fn private() {
            println!("outer::inner::private");
        }

        pub fn public() {
            println!("outer::inner::public");
            super::private();  // 子模块可以访问父模块的私有成员
        }
    }
}

fn main() {
    outer::public();
    // outer::private();        // ❌ 编译错误：private 是私有的
    // outer::inner::public();  // ❌ 编译错误：inner 模块是私有的
}
```

**关键点**：
- Rust 中所有项目默认是 **私有的**
- 使用 `pub` 关键字使其对外可见
- 子模块可以通过 `super::` 访问父模块的私有成员
- 模块本身也需要 `pub` 才能从外部访问

### 2. 模块与文件系统层级

```rust
// src/main.rs
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
    }
}

use crate::front_of_house::hosting;

fn main() {
    hosting::add_to_waitlist();
}
```

**文件结构映射**：
- `mod foo;` 在 `main.rs`/`lib.rs` 中声明 → 对应 `src/foo.rs` 或 `src/foo/mod.rs`
- 嵌套模块 `mod bar;` 在 `foo/mod.rs` 中声明 → 对应 `src/foo/bar.rs`

### 3. 路径解析规则 (use, super, self)

路径按以下方式解析：

1. **相对路径**:
   - `foo` 或 `self::foo` 引用当前模块中的 `foo`
   - `super::foo` 引用父模块中的 `foo`

2. **绝对路径**:
   - `crate::foo` 引用当前 crate 根中的 `foo`
   - `bar::foo` 引用 `bar` crate 中的 `foo`

### 4. Re-export (pub use)

```rust
mod storage;

pub use storage::disk::DiskStorage;
pub use storage::network::NetworkStorage;
```

通过 `pub use` 将子模块的项目重新导出到父模块或顶层模块，
使特定功能以更便捷的路径对外部 crate 可用。

### 5. 可见性级别总结

| 可见性 | 语法 | 说明 |
|--------|------|------|
| 私有（默认）| 无修饰符 | 仅当前模块及子模块可访问 |
| 公开 | `pub` | 对所有外部模块可访问 |
| Crate 内公开 | `pub(crate)` | 仅当前 crate 内可访问 |
| 父模块公开 | `pub(super)` | 仅父模块可访问 |
| 指定路径公开 | `pub(in path)` | 仅指定路径内可访问 |
