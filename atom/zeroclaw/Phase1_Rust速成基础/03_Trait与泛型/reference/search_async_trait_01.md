---
type: search_result
search_query: Rust async_trait Send Sync trait object patterns 2025
search_engine: grok-mcp
searched_at: 2026-03-10
knowledge_point: 03_Trait与泛型
---

# 搜索结果：Rust async_trait Send Sync 模式

## 搜索摘要

2025-2026 年 Rust 异步 Trait 对象的主流模式仍然依赖 `async-trait` crate（v0.1.89, 2025年8月）。

## 关键信息提取

### 1. 标准线程安全模式
```rust
#[async_trait]
pub trait MyService: Send + Sync {
    async fn process(&self, input: String) -> Result<String, Error>;
}
pub type DynService = Arc<dyn MyService + Send + Sync>;
```

### 2. 非 Send 模式
```rust
#[async_trait(?Send)]
trait LocalService {
    async fn process(&self);
}
```

### 3. 原生 async fn in traits
- Rust 1.75+ 稳定了 async fn in traits
- 但 dyn Trait 的完整支持仍未稳定（截至 2025H1）
- `dynosaur` crate 作为替代方案出现

### 4. 关键要点
- `Send + Sync` 超级 Trait 确保实现者线程安全
- async 方法被脱糖为 `Pin<Box<dyn Future<Output = ...> + Send + 'async_trait>>`
- `&dyn MyService` 要 Send，需要 Trait 对象是 Sync
- `Arc<dyn Trait + Send + Sync>` 优于 `Box<...>` 用于共享所有权
