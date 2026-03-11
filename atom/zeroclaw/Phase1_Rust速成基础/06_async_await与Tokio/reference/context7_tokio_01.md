---
type: context7_documentation
library: tokio
version: "1.42+"
fetched_at: 2026-03-10
knowledge_point: 06_async_await与Tokio
context7_query: "async await spawn Future trait basics + select mpsc channel oneshot cancellation timeout"
---

# Context7 文档：Tokio 异步运行时

## 文档来源
- 库名称：tokio
- 版本：1.42+
- 官方文档链接：https://docs.rs/tokio/latest/tokio/

## 关键信息提取

### 1. tokio::spawn - 生成异步任务

`tokio::spawn` 生成一个新的异步任务，返回 `JoinHandle`。任务会立即在后台开始运行。

**核心签名**：
```rust
pub fn spawn<Fut>(future: Fut) -> JoinHandle<Fut::Output>
where
    Fut: Future + Send + 'static,
    Fut::Output: Send + 'static,
```

**关键约束**：`Send + 'static` — 因为任务在线程池上运行，闭包不能引用栈数据。

**基本用法**：
```rust
let join = tokio::spawn(async {
    "hello world!"
});
let result = join.await?;
```

**并发处理模式**：
```rust
let mut tasks = Vec::new();
for op in ops {
    tasks.push(tokio::spawn(my_background_op(op)));
}
for task in tasks {
    outputs.push(task.await.unwrap());
}
```

### 2. tokio::select! - 并发选择

`select!` 等待多个并发分支，第一个完成时返回，自动取消其余分支。

**超时模式**：
```rust
tokio::select! {
    _ = &mut sleep => {
        println!("operation timed out");
        break;
    }
    _ = some_async_work() => {
        println!("operation completed");
    }
}
```

**与 oneshot 结合**：
```rust
let (send, mut recv) = oneshot::channel();
loop {
    tokio::select! {
        _ = interval.tick() => println!("Another 100ms"),
        msg = &mut recv => {
            println!("Got message: {}", msg.unwrap());
            break;
        }
    }
}
```

### 3. mpsc 通道 - 消息传递

**带超时发送**：
```rust
let (tx, mut rx) = mpsc::channel(1);
tokio::spawn(async move {
    for i in 0..10 {
        if let Err(e) = tx.send_timeout(i, Duration::from_millis(100)).await {
            return;
        }
    }
});
while let Some(i) = rx.recv().await {
    println!("got = {}", i);
}
```

### 4. LocalSet - 非 Send 任务

```rust
let nonsend_data = Rc::new("world");
let local = task::LocalSet::new();
local.spawn_local(async move {
    println!("hello {}", nonsend_data)
});
local.await;
```

---

# Context7 文档：async-trait

## 文档来源
- 库名称：async-trait
- 版本：latest
- 官方文档链接：https://github.com/dtolnay/async-trait

## 关键信息提取

### async-trait 宏原理

`#[async_trait]` 宏将 async trait 方法转换为返回 `Pin<Box<dyn Future<Output = ReturnType> + Send + 'async_trait>>` 的方法，从而支持动态分发。

**Rust 1.75** 稳定了 async fn in traits，但**不支持** `dyn Trait` 动态分发。`async-trait` 弥补了这个差距。

### 基本用法

```rust
use async_trait::async_trait;

#[async_trait]
trait Advertisement {
    async fn run(&self);
}

#[async_trait]
impl Advertisement for Modal {
    async fn run(&self) {
        self.render_fullscreen().await;
    }
}
```

### 动态分发 pipeline

```rust
async fn pipeline(processors: Vec<Box<dyn DataProcessor + Send + Sync>>, data: Vec<u8>) -> Vec<u8> {
    let mut result = data;
    for processor in processors.iter() {
        result = processor.process(result).await;
    }
    result
}
```

**关键点**：`async-trait` 将 async fn 转换为 Pin<Box<dyn Future>>，使其可以与 trait object 一起使用。
