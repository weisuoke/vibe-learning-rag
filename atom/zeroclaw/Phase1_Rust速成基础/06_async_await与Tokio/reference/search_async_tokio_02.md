---
type: search_result
search_query: "Rust Future lazy vs JavaScript Promise eager execution state machine async comparison"
search_engine: grok-mcp
searched_at: 2026-03-10
knowledge_point: 06_async_await与Tokio
---

# 搜索结果：Rust Future vs JavaScript Promise 对比

## 搜索摘要

详细对比 Rust Future（惰性执行）与 JavaScript Promise（立即执行）的执行模型、状态机编译机制、取消行为等。

## 相关链接

- [Rust Book - Futures and Syntax](https://doc.rust-lang.org/book/ch17-01-futures-and-syntax.html) - "futures in Rust are lazy... invisible state machine"
- [The New Stack - How Rust Does Async Differently](https://thenewstack.io/how-rust-does-async-differently-and-why-it-matters/) - Pull vs push model
- [MDN Promise](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise) - "Executor executed immediately"
- [V8 Blog - fast-async](https://v8.dev/blog/fast-async) - await internals, microtasks
- [Eventhelix - Rust to Assembly Async](https://www.eventhelix.com/rust/rust-to-assembly-async-await) - 编译细节

## 关键信息提取

### 核心对比表

| 方面 | Rust `Future`（惰性） | JavaScript `Promise`（立即） |
|------|----------------------|---------------------------|
| 执行触发 | `poll()` 调用或 `.await` | `new Promise(executor)` 或 `async` 函数调用 |
| 工作何时开始 | 仅在首次 poll 时；可廉价构造不启动 | 立即且同步地 |
| 状态机驱动 | 编译器生成 `enum` + `Future::poll`（跳转表） | JS 引擎 (V8等) + 微任务队列 |
| 取消 | Drop Future（未启动则无工作） | 困难；需要 `AbortController` |
| 并发模型 | 协作式 polling（零开销，无栈） | 事件循环推送模型 |

### 术语定义

- **Future（Rust）**：表示可能稍后完成的值的 trait；在执行器反复调用其 `poll` 方法之前保持惰性（不发生计算）
- **Promise（JS）**：持有异步操作最终结果的对象；其内部状态从 pending 转变为 fulfilled/rejected
- **状态机**：编译器或运行时生成的结构（Rust 中为 enum，JS 中为内部标志），跟踪跨暂停点（如 `.await`）的执行进度

### Rust 状态机内部机制

Rust 编译器将每个 `async fn` 或 `async {}` 块重写为一个 `enum`，每个变体代表一个状态（如 `Start`、`AfterAwait1`、`Done`），存储局部变量和下一个恢复点。

```rust
// 简化的状态机心智模型
enum HelloFuture {
    Start,
    Done,
}
impl Future for HelloFuture {
    fn poll(...) -> Poll<()> {
        match self.state {
            Start => { println!("hello"); self.state = Done; Poll::Ready(()) }
            Done => Poll::Ready(()),
        }
    }
}
```

生成的 `poll` 方法使用汇编中的跳转表分派到当前状态，直到 `Poll::Ready` 或返回 `Poll::Pending`（注册 `Waker`）。

### 生活类比

- **Rust Future** 像一张详细的食谱卡：你可以传递它、堆叠很多张、或扔掉它（取消），而不用打开炉子——做菜（执行）只在厨师（执行器）主动按步骤操作时才开始
- **JavaScript Promise** 像在线点外卖：你提交订单（创建 Promise）的那一刻，厨房就开火做菜了，不管你准备好没有

### 实践影响

- **Rust**：适合服务器、嵌入式、高性能代码——drop Future 立即取消 I/O，组合数千个 Future 而不启动工作
- **JavaScript**：适合 Web 脚本——零样板执行器、自动微任务集成、可预测的"发射后不管"行为

### 重要细节

- 纯 `async fn` Future 在 Rust 中总是完全惰性的
- 但普通函数返回 `impl Future` 可以在返回 Future 之前执行急切的设置——这是允许的但被认为是不好的风格
- Rust 牺牲简单性换取显式控制和性能
- JavaScript 在其事件循环运行时内优先考虑开发者体验
