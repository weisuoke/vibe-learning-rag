# async/await 与 Tokio - 核心概念 1：async fn 与 Future 机制

> **知识点**: async/await 与 Tokio
> **层级**: Phase1_Rust速成基础
> **维度**: 核心概念 1 / 6
> **目标受众**: 有 TypeScript/前端经验但零 Rust 经验的开发者
> **阅读时间**: 约 25 分钟

---

## 概述

这是六大核心概念中最重要的一个。理解了 `async fn` 和 `Future`，后面的 Tokio 运行时、spawn、select!、mpsc、async_trait 都是在此基础上的应用。

**本节核心问题**：
1. `async fn` 到底做了什么？
2. `Future` trait 怎么工作？
3. `.await` 的真正含义是什么？
4. 编译器如何将 async fn 变成状态机？
5. 和 JavaScript 的 async/Promise 有什么本质区别？

---

## 1. async fn 的本质

### 1.1 async fn 是语法糖

`async fn` 本身不是什么特殊的运行时功能——它是**编译器的语法糖**。编译器将 async fn 转换为一个返回 `impl Future<Output = T>` 的普通函数。

```rust
// 你写的
async fn hello() -> String {
    "Hello, World!".to_string()
}

// 编译器看到的（等价形式）
fn hello() -> impl Future<Output = String> {
    async {
        "Hello, World!".to_string()
    }
}
```

**关键理解**：`async fn` 的返回类型不是 `String`——它返回的是一个"将来会产出 String 的东西"（即 `impl Future<Output = String>`）。

### 1.2 惰性执行：调用 ≠ 执行

**这是 Rust async 和 JavaScript async 最关键的区别。**

```rust
// Rust: 调用 async fn 只创建 Future，不执行任何代码
async fn do_work() -> i32 {
    println!("开始工作！");  // ← 这行代码在调用 do_work() 时不会执行！
    42
}

fn main() {
    let future = do_work();  // ← 什么都没打印！只是创建了一个 Future 实例
    // future 此刻是一个状态机对象，安静地躺在栈上
    // "开始工作！" 这行代码完全没有执行
}
```

对比 TypeScript：

```typescript
// JavaScript: 调用 async function 会立即执行到第一个 await
async function doWork(): Promise<number> {
    console.log("开始工作！");  // ← 调用 doWork() 时立刻执行！
    return 42;
}

const promise = doWork();  // 控制台已经打印了 "开始工作！"
// 即使你永远不 await 这个 promise，console.log 已经执行了
```

**用图来理解**：

```
JavaScript（立即执行 / Eager）：
  调用 doWork()
    │
    ├── console.log("开始工作！")  ← 立即执行
    ├── 遇到 await → 暂停，返回 Promise
    │
    └── （后续代码等事件循环调度）

Rust（惰性执行 / Lazy）：
  调用 do_work()
    │
    └── 返回 Future 对象（一个状态机 enum 实例）
        │
        └── 什么代码都没执行！println! 还没运行！
            │
            └── 直到有人 .await 或 poll() 这个 Future
                │
                └── 才开始执行 println!("开始工作！")
```

### 1.3 为什么 Rust 选择惰性？

三个理由：

**理由 1：不做不必要的工作**

```rust
async fn expensive_computation() -> i32 {
    // 假设这里有复杂的 I/O 操作
    heavy_network_call().await
}

// 条件分支中，只有满足条件才需要执行
if need_computation {
    let result = expensive_computation().await;
    // ← 只在需要时才真正执行
} else {
    // 如果是 JavaScript，Promise 已经开始执行了，即使走的是 else 分支
}
```

**理由 2：免费的取消**

```rust
let future = expensive_computation();
// 改主意了，不需要了
drop(future);  // ← 直接丢弃，没有任何 I/O 发生，没有资源泄漏
```

```typescript
const promise = expensiveComputation();
// 改主意了？HTTP 请求已经发出去了……
// 需要 AbortController 来取消，而且不保证能取消
```

**理由 3：组合灵活性**

```rust
let future_a = fetch_user("alice");
let future_b = fetch_user("bob");

// 你可以在执行前决定如何组合：
// 方式 1：并发执行
let (a, b) = tokio::join!(future_a, future_b);

// 方式 2：竞争执行（第一个完成就返回）
// let result = tokio::select! { ... };

// 方式 3：顺序执行
// let a = future_a.await;
// let b = future_b.await;
```

---

## 2. Future trait 详解

### 2.1 trait 定义

`Future` 是 Rust 标准库中定义的核心 trait：

```rust
// std::future::Future — 标准库定义
pub trait Future {
    /// Future 完成后产出的值的类型
    type Output;

    /// 尝试推进 Future 到完成状态
    /// 返回 Poll::Ready(output) 表示完成
    /// 返回 Poll::Pending 表示还没完成，需要稍后再来
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
}

/// poll 的返回值
pub enum Poll<T> {
    Ready(T),     // 完成了，结果是 T
    Pending,      // 还没完成，先去忙别的吧
}
```

**TypeScript 对照**：

```typescript
// 如果 JavaScript 的 Promise 用类似的 trait 风格来描述
interface Future<Output> {
    poll(context: WakerContext): "Ready" | "Pending";
    getOutput(): Output;  // 仅在 Ready 时调用
}
```

但 JavaScript 不需要这个接口——因为 Promise 是 push 模型（完成后主动通知你），不需要你主动 poll。

### 2.2 poll 方法：执行器如何驱动 Future

`poll` 是 Future 唯一的方法。执行器（Tokio）通过反复调用 `poll` 来驱动 Future 前进。

```
执行器的工作流程：

1. 拿到一个 Future
2. 调用 future.poll(cx)
3. 如果返回 Poll::Ready(value) → 完成！拿到结果
4. 如果返回 Poll::Pending → 记住这个 Future，等 Waker 通知后再 poll
```

**你不需要手写 poll 方法！** 当你写 `async fn` 时，编译器会自动帮你实现 `Future` trait 和 `poll` 方法。

你只需要知道这个心智模型：

```
async fn fetch_user(id: &str) -> User {
    let resp = http_get(url).await;       // poll 点 1
    let user = parse_json(resp).await;    // poll 点 2
    user                                   // 返回 Ready(user)
}

第一次 poll: 开始执行 → 发起 HTTP 请求 → Pending（等网络）
            （Tokio 去执行别的任务）
Waker 通知: 网络响应到了！
第二次 poll: 拿到 resp → 开始解析 JSON → Pending（等解析）
            （Tokio 去执行别的任务）
Waker 通知: 解析完了！
第三次 poll: 拿到 user → Ready(user) 🎉
```

### 2.3 Pin：防止 Future 在内存中移动

`poll` 的签名中有一个奇怪的参数：`self: Pin<&mut Self>`。

**为什么需要 Pin？**

当编译器生成状态机时，状态机可能包含**自引用**（一个字段引用另一个字段）：

```rust
async fn example() {
    let data = vec![1, 2, 3];
    let reference = &data;          // reference 指向 data

    some_async_op().await;          // .await 暂停点

    println!("{:?}", reference);    // 恢复后仍需使用 reference
}

// 编译器生成的状态机（简化）：
struct ExampleFuture {
    data: Vec<i32>,
    reference: *const Vec<i32>,     // 指向 self.data！这是自引用！
    state: u8,
}
```

如果这个状态机在内存中被移动（比如被 push 到 Vec 里，Vec 扩容时会搬移元素），`reference` 就会指向错误的内存地址！

`Pin` 的作用就是**承诺：这个 Future 一旦开始被 poll，就不会在内存中移动**。

**实际使用中你几乎不需要操心 Pin**：
- `async fn` 和 `.await` 会自动处理
- 只有手写 `Future` trait 实现时才需要关注
- `#[async_trait]` 使用 `Pin<Box<dyn Future>>` 自动处理

### 2.4 Waker：通知执行器"我准备好了"

`poll` 的第二个参数是 `cx: &mut Context<'_>`，其中包含一个 `Waker`。

**Waker 的作用**：当 Future 返回 `Pending` 时，它需要告诉执行器"什么时候再来 poll 我"。Waker 就是这个通知机制。

```
没有 Waker 的世界（忙轮询 / busy polling）：

执行器: "好了吗？" → Pending
执行器: "好了吗？" → Pending
执行器: "好了吗？" → Pending
执行器: "好了吗？" → Pending
...（CPU 空转，浪费 100% CPU）

有 Waker 的世界（事件通知）：

执行器: "好了吗？" → Pending（Future 注册 Waker）
执行器: 去做别的事情……
          ……
          ……
（IO 完成）→ Waker: "嘿！我好了！"
执行器: "好了吗？" → Ready(result) ✓
```

**TypeScript 对照**：

```
JavaScript:   Promise 完成 → 回调被推入微任务队列 → 事件循环执行
Rust:         IO 完成 → Waker.wake() → 执行器把 Future 放回就绪队列 → poll
```

核心思路是一样的：避免忙等待，通过通知机制实现高效调度。

---

## 3. 状态机编译

### 3.1 编译器的魔法

编译器将 `async fn` 中的每个 `.await` 作为**状态分界点**，生成一个 enum 状态机。

**规则**：`N` 个 `.await` 生成 `N + 1` 个状态。

```rust
// 原始代码：2 个 .await → 3 个状态
async fn process_message(msg: &str) -> Result<String> {
    // ─── 状态 0：初始 ───
    let prompt = format!("Process: {}", msg);

    // ─── .await 分界线 ① ───
    let llm_response = provider.chat(&prompt).await?;

    // ─── 状态 1：拿到 LLM 响应 ───
    let parsed = parse_response(&llm_response);

    // ─── .await 分界线 ② ───
    memory.store(&parsed).await?;

    // ─── 状态 2：存储完成 ───
    Ok(parsed)
}
```

### 3.2 生成的状态机（概念模型）

```rust
// 编译器生成的 enum（简化版 —— 实际生成的更复杂）
enum ProcessMessageFuture {
    /// 状态 0：准备调用 LLM
    State0 {
        msg: String,
        prompt: String,
    },
    /// 状态 1：等待 LLM 响应
    State1 {
        chat_future: ChatFuture,  // provider.chat() 返回的 Future
        // msg 和 prompt 已经不需要了，不保存（节省内存）
    },
    /// 状态 2：等待存储完成
    State2 {
        parsed: String,
        store_future: StoreFuture,  // memory.store() 返回的 Future
    },
    /// 完成
    Done,
}
```

### 3.3 生成的 poll 方法（概念模型）

```rust
impl Future for ProcessMessageFuture {
    type Output = Result<String>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Result<String>> {
        loop {
            match self.state {
                State0 { msg, prompt } => {
                    // 执行状态 0 的逻辑
                    let chat_future = provider.chat(&prompt);
                    self.state = State1 { chat_future };
                    // 不 return，继续循环尝试 poll State1
                }
                State1 { chat_future } => {
                    // 尝试推进 LLM 调用
                    match chat_future.poll(cx) {
                        Poll::Pending => return Poll::Pending,  // LLM 还没回复
                        Poll::Ready(Ok(llm_response)) => {
                            let parsed = parse_response(&llm_response);
                            let store_future = memory.store(&parsed);
                            self.state = State2 { parsed, store_future };
                            // 继续循环
                        }
                        Poll::Ready(Err(e)) => {
                            self.state = Done;
                            return Poll::Ready(Err(e));
                        }
                    }
                }
                State2 { parsed, store_future } => {
                    // 尝试推进存储操作
                    match store_future.poll(cx) {
                        Poll::Pending => return Poll::Pending,  // 存储还没完成
                        Poll::Ready(Ok(())) => {
                            self.state = Done;
                            return Poll::Ready(Ok(parsed));
                        }
                        Poll::Ready(Err(e)) => {
                            self.state = Done;
                            return Poll::Ready(Err(e));
                        }
                    }
                }
                Done => panic!("Future polled after completion"),
            }
        }
    }
}
```

### 3.4 可视化状态转换

```
                    poll()
           ┌────────────────────┐
           │                    │
           ▼                    │
    ┌────────────┐              │
    │  State0    │  创建 chat   │
    │  初始化     │  Future     │
    └─────┬──────┘              │
          │ 立即进入             │
          ▼                     │
    ┌────────────┐   Pending    │
    │  State1    │─────────────►│  等待 LLM 响应
    │  等 LLM    │              │  （Waker 注册在网络 IO 上）
    └─────┬──────┘              │
          │ Ready               │
          ▼                     │
    ┌────────────┐   Pending    │
    │  State2    │─────────────►│  等待存储完成
    │  等存储     │              │  （Waker 注册在数据库 IO 上）
    └─────┬──────┘              │
          │ Ready               │
          ▼                     │
    ┌────────────┐              │
    │   Done     │              │
    │  完成！     │              │
    └────────────┘

    每次 poll 返回 Pending → 执行器去做别的事
    Waker 通知 → 执行器回来再 poll → 状态机推进一步
```

---

## 4. .await 的含义

### 4.1 .await 不是"等待"

虽然叫 "await"（等待），但 `.await` 的真正含义是：

> **"让出控制权，在就绪时恢复执行"**

它**不会阻塞当前线程**。当你写 `response.await` 时：
1. 当前 Future 告诉执行器："我需要等这个子 Future"
2. 执行器去做别的任务
3. 子 Future 就绪后，执行器回来继续执行当前 Future

```rust
// .await 的心智模型
let response = http_get(url).await;
//                           ^^^^^
//  1. 创建 http_get 的 Future
//  2. poll 这个 Future
//  3. 如果 Pending → 让出控制权（执行器去做别的）
//  4. 就绪后恢复 → 拿到 response
//  注意：当前线程没有阻塞！它在做其他任务
```

### 4.2 .await vs JavaScript 的 await

```typescript
// JavaScript await: 暂停当前函数，但事件循环继续
// Promise 已经在执行了——await 只是等结果
const data = await fetchData();
//  fetchData() 在你写 await 之前可能就执行了
//  await 只是"等 Promise 兑现"
```

```rust
// Rust .await: 开始执行子 Future + 在未完成时让出
let data = fetch_data().await;
//  fetch_data() 只是创建了一个 Future（什么都没执行！）
//  .await 才开始执行 + 在 Pending 时让出
```

**核心区别**：

| 方面 | JavaScript `await` | Rust `.await` |
|------|-------------------|---------------|
| 执行时机 | Promise 已在执行中 | Future 此时才开始执行 |
| 本质 | 等 Promise 兑现 | 驱动 Future + 让出/恢复 |
| 线程行为 | 单线程事件循环 | 多线程调度器 |
| 取消 | await 不能取消 Promise | drop Future = 取消 |

### 4.3 链式 .await vs 并发执行

初学者常犯的错误——**把本可以并发的操作写成串行**：

```rust
// ❌ 串行执行——第二个请求等第一个完成才开始
async fn fetch_both_serial() -> (User, Orders) {
    let user = fetch_user("alice").await;     // 等 1 秒
    let orders = fetch_orders("alice").await; // 再等 1 秒
    (user, orders)                             // 总共 2 秒
}

// ✅ 并发执行——两个请求同时进行
async fn fetch_both_concurrent() -> (User, Orders) {
    let user_future = fetch_user("alice");     // 创建 Future（不执行）
    let orders_future = fetch_orders("alice"); // 创建 Future（不执行）

    // join! 同时驱动两个 Future
    let (user, orders) = tokio::join!(user_future, orders_future);
    (user, orders)                              // 总共 1 秒（并发）
}
```

**TypeScript 对照**：

```typescript
// ❌ 串行
const user = await fetchUser("alice");     // 等 1 秒
const orders = await fetchOrders("alice"); // 再等 1 秒

// ✅ 并发
const [user, orders] = await Promise.all([
    fetchUser("alice"),     // 两个同时开始
    fetchOrders("alice"),
]);
```

注意一个微妙的差异：

```typescript
// JavaScript: 即使不用 Promise.all，两个请求也已经开始了
const userPromise = fetchUser("alice");     // 已经开始请求了！
const ordersPromise = fetchOrders("alice"); // 已经开始请求了！
// Promise.all 只是同时等待两个已经在执行的 Promise

// 甚至这样也是"并发"的：
const p1 = fetchUser("alice");     // 已经开始
const p2 = fetchOrders("alice");   // 已经开始
const user = await p1;             // 等 p1
const orders = await p2;           // p2 可能已经完成了
```

```rust
// Rust: 如果不用 join!，就是真的串行
let future1 = fetch_user("alice");     // 什么都没执行！
let future2 = fetch_orders("alice");   // 什么都没执行！
let user = future1.await;             // 现在才开始执行第一个
let orders = future2.await;           // 第一个完成后才开始第二个

// 必须用 join! 才能并发
let (user, orders) = tokio::join!(future1, future2);
```

这就是惰性执行的实际影响——你必须**显式地**要求并发（`join!`），Rust 不会帮你偷偷并发。

### 4.4 .await 只能在 async 上下文中使用

```rust
// ❌ 编译错误：不能在非 async 函数中使用 .await
fn sync_function() {
    let data = fetch_data().await;  // 错误！
}

// ✅ 正确：在 async fn 中使用
async fn async_function() {
    let data = fetch_data().await;  // OK
}

// ✅ 在 async 块中使用
fn another_function() {
    let future = async {
        let data = fetch_data().await;  // OK，在 async 块内
        data
    };
    // 但 future 还需要被执行器（Tokio）驱动
}
```

---

## 5. ZeroClaw 中的实际使用

### 5.1 四大 Trait 的 async fn

ZeroClaw 的四大核心 Trait 全部使用 `#[async_trait]`，因为它们的方法涉及 I/O 操作：

```rust
// Provider Trait — LLM 调用（网络 I/O）
#[async_trait]
pub trait Provider: Send + Sync {
    /// 单轮对话
    async fn chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
        temperature: f64,
    ) -> Result<String>;

    /// 带工具调用的对话
    async fn chat_with_tools(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        model: &str,
        temperature: f64,
    ) -> Result<Message>;

    /// 流式对话（返回异步 Stream）
    async fn stream_chat_with_history(
        &self,
        messages: &[Message],
        model: &str,
        temperature: f64,
    ) -> Result<BoxStream<'static, StreamResult<StreamChunk>>>;
}
```

```rust
// Tool Trait — 工具执行（可能是 Shell 命令、文件 I/O）
#[async_trait]
pub trait Tool: Send + Sync {
    /// 执行工具
    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult>;
}
```

```rust
// Memory Trait — 数据库 I/O
#[async_trait]
pub trait Memory: Send + Sync {
    async fn store(&self, content: &str, metadata: Option<Value>) -> Result<String>;
    async fn recall(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>>;
    async fn get(&self, id: &str) -> Result<Option<MemoryEntry>>;
    async fn forget(&self, id: &str) -> Result<bool>;
}
```

```rust
// Channel Trait — 网络通信（WebSocket、HTTP 长连接）
#[async_trait]
pub trait Channel: Send + Sync {
    async fn send(&self, message: ChannelMessage) -> Result<()>;
    async fn listen(&self, tx: mpsc::Sender<ChannelMessage>) -> Result<()>;
}
```

### 5.2 异步调用链

一个完整的消息处理流程：

```rust
// 整个调用链都是 async 的
async fn handle_user_message(agent: &Agent, msg: ChannelMessage) -> Result<()> {
    // Step 1: 回忆相关记忆（数据库 I/O）
    let memories = agent.memory.recall(&msg.content, 5).await?;

    // Step 2: 构建 Prompt
    let prompt = build_prompt(&msg, &memories);

    // Step 3: 调用 LLM（网络 I/O，1-30 秒）
    let response = agent.provider.chat_with_tools(
        &prompt,
        &agent.tools,
        "claude-3-5-sonnet",
        0.7,
    ).await?;

    // Step 4: 如果 LLM 请求工具调用
    if let Some(tool_call) = response.tool_calls().first() {
        // 执行工具（可能是 Shell 命令、文件操作）
        let tool_result = agent.execute_tool(tool_call).await?;
        // 把结果反馈给 LLM
        let final_response = agent.provider.chat_with_tools(
            &[response, tool_result.to_message()],
            &agent.tools,
            "claude-3-5-sonnet",
            0.7,
        ).await?;
    }

    // Step 5: 存储对话记忆（数据库 I/O）
    agent.memory.store(&response.content, None).await?;

    // Step 6: 发送回复（网络 I/O）
    agent.channel.send(response.to_channel_message()).await?;

    Ok(())
}
```

这个函数有 **6 个 `.await`**，意味着编译器会生成一个 **7 状态** 的状态机。在每个 `.await` 点，当前线程都可以去处理其他任务——比如同时处理另一个 Channel 的消息。

### 5.3 统计概览

| 模式 | 使用次数 | 含义 |
|------|---------|------|
| `async fn` | 1939 | ZeroClaw 几乎所有函数都是异步的 |
| `#[async_trait]` | 124 | 四大 Trait + 扩展接口全部异步 |
| `.await` | ~3000+ | 每个 async fn 平均 1-2 个 .await |

---

## 6. 常见误区

### 误区 1："async fn 调用后代码就开始执行了" ❌

```rust
// 很多从 JavaScript 转来的开发者会犯这个错误
async fn send_email(to: &str) -> Result<()> {
    println!("Sending email to {}...", to);
    // ... 发送邮件
    Ok(())
}

// ❌ 错误理解：认为调用后邮件就开始发了
let future = send_email("alice@example.com");
// 实际上：什么都没发生！println! 都没执行！

// ✅ 必须 .await 或交给执行器
send_email("alice@example.com").await?;  // 现在才开始
```

### 误区 2：".await 会阻塞线程" ❌

```rust
// ❌ 错误理解：以为 .await 像 Thread::sleep 那样阻塞
let data = fetch_data(url).await;
// "当前线程是不是卡在这里了？"

// ✅ 正确理解：.await 让出控制权，线程去做别的
// 当前线程会：
// 1. 把这个 Future 标记为"等待 IO"
// 2. 从任务队列取出下一个就绪的任务来执行
// 3. IO 完成后，通过 Waker 通知，执行器回来继续执行这个 Future
```

**什么才是"阻塞线程"？**

```rust
// ⚠️ 这才是真正阻塞线程的操作
async fn bad_code() {
    std::thread::sleep(Duration::from_secs(5));  // 阻塞！线程真的卡住了
    // 应该用 tokio::time::sleep(Duration::from_secs(5)).await;
}

async fn also_bad() {
    let data = std::fs::read_to_string("big_file.txt")?;  // 阻塞！同步文件 IO
    // 应该用 tokio::fs::read_to_string("big_file.txt").await?;
}
```

### 误区 3："Future 和 Promise 是一样的" ❌

| 特性 | Rust Future | JavaScript Promise |
|------|------------|-------------------|
| 执行时机 | 惰性（调用不执行） | 立即（调用即执行） |
| 驱动模型 | Pull（执行器来 poll） | Push（事件循环推送结果） |
| 取消 | drop Future = 取消 | 需要 AbortController |
| 内存模型 | 编译时确定大小的 enum | 堆分配的对象 |
| 线程安全 | 编译器检查 Send/Sync | 单线程，不需要考虑 |
| 运行时 | 外部提供（Tokio） | 语言内置（V8 事件循环） |

**最简单的记忆方式**：

- **Promise = 外卖订单**：下单即开火，你只能等或取消（麻烦）
- **Future = 食谱卡片**：拿着卡片什么都不发生，交给厨师才开始做

### 误区 4：以为不 .await 也能执行

```rust
async fn important_side_effect() {
    save_to_database().await;  // 重要的副作用！
}

async fn main_logic() {
    // ❌ 忘记 .await —— 数据库保存根本没执行！
    important_side_effect();  // 编译器会给警告：unused Future

    // ✅ 必须 .await
    important_side_effect().await;
}
```

Rust 编译器会对"未使用的 Future"发出警告：

```
warning: unused implementer of `Future` that must be used
  --> src/main.rs:10:5
   |
10 |     important_side_effect();
   |     ^^^^^^^^^^^^^^^^^^^^^^^^
   |
   = note: futures do nothing unless you `.await` or poll them
```

### 误区 5：在 async 中使用阻塞操作

```rust
// ❌ 危险：在 async 上下文中使用阻塞操作
async fn bad_practice() {
    // 这会阻塞 Tokio 的工作线程，其他任务全部卡住
    std::thread::sleep(Duration::from_secs(10));
    let data = std::fs::read("huge_file.txt")?;
    let result = compute_heavy_math();  // CPU 密集计算
}

// ✅ 正确做法
async fn good_practice() {
    // 异步 sleep
    tokio::time::sleep(Duration::from_secs(10)).await;

    // 异步文件读取
    let data = tokio::fs::read("huge_file.txt").await?;

    // CPU 密集任务放到阻塞线程池
    let result = tokio::task::spawn_blocking(|| {
        compute_heavy_math()
    }).await?;
}
```

---

## 7. 小结：async fn 与 Future 的核心要点

```
┌──────────────────────────────────────────────────────────────┐
│                   async fn 与 Future 核心要点                  │
│                                                               │
│  1. async fn 是语法糖                                         │
│     → 编译器将其转为返回 impl Future<Output = T> 的函数         │
│                                                               │
│  2. Future 是惰性的                                           │
│     → 调用 async fn 不执行代码，只创建状态机实例                 │
│     → 与 JavaScript Promise 的立即执行完全相反                  │
│                                                               │
│  3. 编译器生成状态机                                           │
│     → N 个 .await = N+1 个状态                                │
│     → enum 变体存储每个状态需要的数据                           │
│     → 编译时确定大小，零开销                                   │
│                                                               │
│  4. .await = 让出 + 恢复                                      │
│     → 不是阻塞线程，是让执行器去做别的事                        │
│     → IO 就绪后通过 Waker 通知，继续执行                       │
│                                                               │
│  5. 并发需要显式声明                                           │
│     → join! 并发执行多个 Future                                │
│     → 串行 .await 就是串行执行（不像 JS 的 Promise 自动并发）    │
│                                                               │
│  6. ZeroClaw 的 1939 个 async fn                              │
│     → 四大 Trait 全部 async + Send + Sync                     │
│     → 每个 .await 点都是一次"让出 CPU 去做别的事"的机会         │
└──────────────────────────────────────────────────────────────┘
```

---

> **下一步**: 阅读 `03_核心概念_2_Tokio运行时与tokio_main.md`，了解 Tokio 如何驱动这些 Future 执行，以及 `#[tokio::main]` 做了什么。

---

**文件信息**
- 知识点: async/await 与 Tokio
- 维度: 核心概念 1 — async fn 与 Future 机制
- 版本: v1.0
- 日期: 2026-03-10
