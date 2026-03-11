# async/await 与 Tokio - 核心概念 5：mpsc 通道与消息传递

> **知识点**: async/await 与 Tokio
> **层级**: Phase1_Rust速成基础
> **维度**: 核心概念 5 / 6
> **目标受众**: 有 TypeScript/前端经验但零 Rust 经验的开发者
> **阅读时间**: 约 25 分钟

---

## 概述

上一节我们学了 `select!` ——同时等待多个 Future 并处理最先完成的。但 select! 解决的是"**一个任务内部如何等多个事件**"的问题。还有一个更基本的问题没解决：**多个独立任务之间怎么通信？**

ZeroClaw 的架构中，Telegram/Discord Channel 监听器收到用户消息后，需要把消息**传递给** Agent 循环处理。两个独立的 `tokio::spawn` 任务之间不能直接调用函数——它们需要一个**安全的通信管道**。

答案是：**通道（Channel）**，具体来说是 `tokio::sync::mpsc`。

**本节核心问题**：
1. 多个异步任务之间如何安全通信？
2. mpsc 通道是什么？为什么叫"多生产者、单消费者"？
3. 有界通道和无界通道有什么区别？什么时候用哪个？
4. oneshot 和 watch 通道适用什么场景？
5. ZeroClaw 的 Channel Trait 为什么选择 mpsc 而不是共享内存？

---

## 1. 为什么需要通道？

### 1.1 问题场景：两个独立任务要通信

```rust
#[tokio::main]
async fn main() {
    // 任务 A：监听用户消息
    tokio::spawn(async {
        loop {
            let msg = listen_for_message().await;
            // 怎么把 msg 传给任务 B？
            // 不能直接调用任务 B 的函数——它们是独立任务！
        }
    });

    // 任务 B：处理消息
    tokio::spawn(async {
        loop {
            // 怎么拿到任务 A 收到的消息？
            let msg = ???;
            process(msg).await;
        }
    });
}
```

### 1.2 两种思路：共享内存 vs 消息传递

**思路 1：共享内存（用锁保护）**

```rust
use std::sync::Arc;
use tokio::sync::Mutex;
use std::collections::VecDeque;

// ❌ 能跑，但不优雅
let queue = Arc::new(Mutex::new(VecDeque::<String>::new()));

// 任务 A：往队列里塞消息
let queue_a = queue.clone();
tokio::spawn(async move {
    loop {
        let msg = listen_for_message().await;
        queue_a.lock().await.push_back(msg);
    }
});

// 任务 B：从队列里取消息
let queue_b = queue.clone();
tokio::spawn(async move {
    loop {
        let msg = {
            let mut q = queue_b.lock().await;
            q.pop_front()
        };
        match msg {
            Some(msg) => process(msg).await,
            None => {
                // 没消息？等一下再看
                tokio::time::sleep(Duration::from_millis(100)).await;
                // ← 忙等待！浪费 CPU
            }
        }
    }
});
```

**问题**：
- 消费者需要**忙轮询**（不断检查有没有新消息）
- 锁竞争：生产者和消费者同时访问队列时互相阻塞
- 没有背压（back-pressure）：生产者可以无限往队列里塞

**思路 2：消息传递（通道）**

```rust
use tokio::sync::mpsc;

// ✅ 优雅！
let (tx, mut rx) = mpsc::channel::<String>(100);

// 任务 A：发送消息
tokio::spawn(async move {
    loop {
        let msg = listen_for_message().await;
        tx.send(msg).await.unwrap();  // 发到通道里
    }
});

// 任务 B：接收消息
tokio::spawn(async move {
    while let Some(msg) = rx.recv().await {
        // ← 没消息时自动等待（不浪费 CPU）
        // ← 有消息时立即唤醒
        process(msg).await;
    }
});
```

### 1.3 Rust 哲学

> **"不要通过共享内存来通信，而是通过通信来共享内存"**
> — 这句话源自 Go 语言社区，但 Rust 同样推崇

```
共享内存（Mutex<VecDeque>）：
  - 任务 A 和 B 共享同一块内存
  - 用锁来保证安全 → 锁竞争、死锁风险
  - 消费者需要忙等待

消息传递（mpsc channel）：
  - 任务 A 通过管道发送消息给任务 B
  - 不需要锁 → 无竞争、无死锁
  - 消费者自动等待 → 零 CPU 开销
  - 天然提供背压（通道满了，发送者自动等待）
```

### 1.4 前端类比

```typescript
// mpsc 通道 ≈ 以下前端模式的组合

// 1. EventEmitter / EventBus
const bus = new EventEmitter();
bus.on("message", (msg) => process(msg));    // 消费者
bus.emit("message", "hello");                 // 生产者

// 2. postMessage（Worker 通信）
// Worker ──postMessage──► 主线程
worker.postMessage({ type: "result", data });
self.onmessage = (e) => process(e.data);

// 3. Redux dispatch → reducer
store.dispatch({ type: "ADD_MESSAGE", payload: msg });  // 生产者
// reducer 自动接收并处理                                  // 消费者

// 4. RxJS Subject
const subject = new Subject<Message>();
subject.subscribe((msg) => process(msg));    // 消费者
subject.next(msg);                            // 生产者
```

**关键区别**：Rust 的 mpsc 通道在**编译时**保证类型安全和所有权转移，不会出现 JavaScript 中常见的"消息格式错误"运行时报错。

---

## 2. mpsc 通道基础

### 2.1 mpsc = Multi-Producer, Single-Consumer

```
mpsc 的含义：

Multi-Producer（多生产者）：
  多个任务可以同时向通道发送消息
  通过 Sender.clone() 实现

Single-Consumer（单消费者）：
  只有一个任务能从通道接收消息
  Receiver 不能 clone

  Producer A ──►┐
  Producer B ──►├──► [缓冲区] ──► Consumer
  Producer C ──►┘
```

### 2.2 创建通道

```rust
use tokio::sync::mpsc;

// 创建有界通道：缓冲区最多存 100 条消息
let (tx, mut rx) = mpsc::channel::<String>(100);
//   ^^       ^^                            ^^^
//   发送端   接收端                          缓冲区大小

// tx: Sender<String>    — 可以 clone，给多个生产者
// rx: Receiver<String>  — 不能 clone，只有一个消费者
```

### 2.3 发送消息

```rust
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let (tx, mut rx) = mpsc::channel::<String>(10);

    // ─── 发送消息 ───
    // send() 是异步的：如果缓冲区满了，会等待有空位
    tx.send("hello".to_string()).await.unwrap();
    tx.send("world".to_string()).await.unwrap();

    // try_send() 是同步的：缓冲区满了立即返回 Err
    match tx.try_send("sync message".to_string()) {
        Ok(()) => println!("发送成功"),
        Err(mpsc::error::TrySendError::Full(msg)) => {
            println!("缓冲区满了，消息被退回: {}", msg);
        }
        Err(mpsc::error::TrySendError::Closed(msg)) => {
            println!("接收端已关闭: {}", msg);
        }
    }
}
```

### 2.4 接收消息

```rust
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let (tx, mut rx) = mpsc::channel::<String>(10);

    // 生产者
    tokio::spawn(async move {
        for i in 0..5 {
            tx.send(format!("消息 {}", i)).await.unwrap();
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        // tx 在这里被 drop → 通道关闭
    });

    // ─── 消费者：while let 模式（最常用）───
    while let Some(msg) = rx.recv().await {
        println!("收到: {}", msg);
    }
    // recv() 返回 None → 通道关闭（所有 Sender 都被 drop 了）
    println!("通道关闭，所有消息已处理");
}
```

**`while let Some(msg) = rx.recv().await` 解析**：

```
1. rx.recv().await
   → 异步等待下一条消息
   → 有消息返回 Some(msg)
   → 所有 Sender 被 drop 后返回 None

2. while let Some(msg) = ...
   → Some(msg) → 匹配成功，循环继续
   → None → 匹配失败，循环退出

这是 Rust 中消费通道消息的标准写法。
```

### 2.5 多生产者：Sender.clone()

```rust
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let (tx, mut rx) = mpsc::channel::<String>(100);

    // ─── 多个生产者 ───
    for i in 0..3 {
        let tx_clone = tx.clone();  // 克隆 Sender
        tokio::spawn(async move {
            for j in 0..5 {
                tx_clone.send(format!("生产者{} 消息{}", i, j)).await.unwrap();
            }
            // tx_clone 在这里被 drop
        });
    }

    // 重要：原始 tx 也要 drop，否则通道永远不会关闭！
    drop(tx);

    // 消费者
    while let Some(msg) = rx.recv().await {
        println!("{}", msg);
    }
    // 所有 3 个 tx_clone 和原始 tx 都被 drop 后，recv 返回 None
    println!("所有生产者完成");
}
```

**常见陷阱：忘记 drop 原始 tx**

```rust
// ❌ 程序会永远卡住！
let (tx, mut rx) = mpsc::channel::<String>(10);

let tx_clone = tx.clone();
tokio::spawn(async move {
    tx_clone.send("hello".to_string()).await.unwrap();
    // tx_clone 被 drop
});

// tx 还活着！→ 通道没关闭 → recv() 永远不会返回 None
while let Some(msg) = rx.recv().await {
    println!("{}", msg);
}
// ← 永远到不了这里

// ✅ 修复：drop 原始 tx
drop(tx);
while let Some(msg) = rx.recv().await {
    println!("{}", msg);
}
```

### 2.6 有界 vs 无界通道

```rust
// ─── 有界通道（推荐） ───
let (tx, rx) = mpsc::channel::<String>(100);
// 缓冲区最多 100 条消息
// 缓冲区满了 → tx.send() 自动等待（背压机制）
// 防止生产者过快导致内存爆炸

// ─── 无界通道（谨慎使用） ───
let (tx, rx) = mpsc::unbounded_channel::<String>();
// 缓冲区无上限
// tx.send() 永远不会阻塞（立即返回，非 async）
// ⚠️ 如果消费者处理不过来，内存会持续增长直到 OOM
```

**什么时候用无界通道？**

```
有界通道（channel）：
  ✅ 绝大多数情况的首选
  ✅ 自动背压——生产者不会压垮消费者
  ✅ 内存使用可预测
  ❌ send() 是 async 的，需要 .await

无界通道（unbounded_channel）：
  ✅ send() 是同步的，可以在非 async 代码中使用
  ✅ 适合消息量小、处理速度快的场景
  ❌ 没有背压——消费者慢了内存会爆
  ❌ 不适合网络服务等高吞吐场景

经验法则：
  → 不确定用哪个？用有界通道
  → 缓冲区大小设多少？先设 32 或 100，后续根据需求调整
```

---

## 3. mpsc 实战模式

### 3.1 生产者-消费者模式

```rust
use tokio::sync::mpsc;

#[derive(Debug)]
enum Command {
    Process(String),
    SaveToDisk(String, Vec<u8>),
    Shutdown,
}

#[tokio::main]
async fn main() {
    let (tx, mut rx) = mpsc::channel::<Command>(32);

    // ─── 消费者任务 ───
    let consumer = tokio::spawn(async move {
        while let Some(cmd) = rx.recv().await {
            match cmd {
                Command::Process(data) => {
                    println!("处理数据: {}", data);
                }
                Command::SaveToDisk(path, data) => {
                    tokio::fs::write(&path, &data).await.unwrap();
                    println!("已保存到: {}", path);
                }
                Command::Shutdown => {
                    println!("收到关闭命令，退出");
                    break;
                }
            }
        }
    });

    // ─── 生产者 ───
    tx.send(Command::Process("hello".into())).await.unwrap();
    tx.send(Command::SaveToDisk("output.txt".into(), b"data".to_vec())).await.unwrap();
    tx.send(Command::Shutdown).await.unwrap();

    consumer.await.unwrap();
}
```

### 3.2 Actor 模式：用通道封装状态

```rust
use tokio::sync::mpsc;

/// Actor 接收的消息类型
enum CounterMsg {
    Increment,
    Decrement,
    GetValue(tokio::sync::oneshot::Sender<i64>),  // 请求-响应
}

/// Counter Actor
struct CounterActor {
    rx: mpsc::Receiver<CounterMsg>,
    value: i64,
}

impl CounterActor {
    fn new(rx: mpsc::Receiver<CounterMsg>) -> Self {
        Self { rx, value: 0 }
    }

    async fn run(mut self) {
        while let Some(msg) = self.rx.recv().await {
            match msg {
                CounterMsg::Increment => self.value += 1,
                CounterMsg::Decrement => self.value -= 1,
                CounterMsg::GetValue(reply) => {
                    // 通过 oneshot 通道回复当前值
                    let _ = reply.send(self.value);
                }
            }
        }
    }
}

/// Counter 的句柄（给外部使用）
#[derive(Clone)]
struct CounterHandle {
    tx: mpsc::Sender<CounterMsg>,
}

impl CounterHandle {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel(32);
        let actor = CounterActor::new(rx);
        tokio::spawn(actor.run());  // Actor 在后台运行
        Self { tx }
    }

    async fn increment(&self) {
        self.tx.send(CounterMsg::Increment).await.unwrap();
    }

    async fn get_value(&self) -> i64 {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.tx.send(CounterMsg::GetValue(reply_tx)).await.unwrap();
        reply_rx.await.unwrap()
    }
}

#[tokio::main]
async fn main() {
    let counter = CounterHandle::new();

    counter.increment().await;
    counter.increment().await;
    counter.increment().await;

    let value = counter.get_value().await;
    println!("计数器值: {}", value);  // 输出: 3
}
```

**为什么用 Actor 模式？**
- 状态（`value`）只在 Actor 内部——不需要 `Mutex`
- 多个任务通过 `CounterHandle`（它包含 `Sender`）并发操作
- 所有操作自动串行化——因为 Actor 是单消费者
- 不会有数据竞争——连锁都不需要

---

## 4. oneshot 通道

### 4.1 什么是 oneshot？

`oneshot` = **一次性通道**，只能发送一条消息。用于**请求-响应**模式。

```rust
use tokio::sync::oneshot;

#[tokio::main]
async fn main() {
    // 创建一次性通道
    let (tx, rx) = oneshot::channel::<String>();

    // 发送方：只能发一次
    tokio::spawn(async move {
        let result = expensive_computation().await;
        tx.send(result).unwrap();  // 发送后 tx 被消费（moved）
        // tx.send(another) → 编译错误！已经被 move 了
    });

    // 接收方：只能收一次
    let result = rx.await.unwrap();
    println!("结果: {}", result);
}
```

### 4.2 典型用法：RPC 风格请求-响应

```rust
use tokio::sync::{mpsc, oneshot};

// Actor 接收的消息：包含回复通道
enum DbCommand {
    Query {
        sql: String,
        reply: oneshot::Sender<Result<Vec<Row>, DbError>>,
    },
    Execute {
        sql: String,
        reply: oneshot::Sender<Result<u64, DbError>>,
    },
}

// 使用方
async fn query_database(db_tx: &mpsc::Sender<DbCommand>) -> Result<Vec<Row>> {
    // 创建一次性通道来接收回复
    let (reply_tx, reply_rx) = oneshot::channel();

    // 发送请求
    db_tx.send(DbCommand::Query {
        sql: "SELECT * FROM users".into(),
        reply: reply_tx,
    }).await?;

    // 等待回复
    reply_rx.await?
}
```

**TypeScript 对照**：

```typescript
// oneshot ≈ 一次性的 Promise resolve
function queryDatabase(sql: string): Promise<Row[]> {
    return new Promise((resolve, reject) => {
        // resolve ≈ oneshot::Sender
        // 只能调用一次
        dbWorker.postMessage({ sql, resolve });
    });
}
```

### 4.3 oneshot 的特点

| 特性 | 说明 |
|------|------|
| 发送次数 | 只能发送 **1 次**（Sender 被消费） |
| 接收次数 | 只能接收 **1 次** |
| 异步/同步 | `tx.send()` 是同步的（不需要 `.await`） |
| 适用场景 | 请求-响应、Future 结果传递 |

---

## 5. watch 通道

### 5.1 什么是 watch？

`watch` = **广播最新值**的通道。多个接收者都能看到**最新**的值（但可能跳过中间值）。

```rust
use tokio::sync::watch;

#[tokio::main]
async fn main() {
    // 创建 watch 通道，初始值为 "disconnected"
    let (tx, mut rx) = watch::channel("disconnected".to_string());

    // ─── 接收者 1 ───
    let mut rx1 = rx.clone();  // watch::Receiver 可以 clone！
    tokio::spawn(async move {
        while rx1.changed().await.is_ok() {
            let status = rx1.borrow();
            println!("接收者 1 看到状态: {}", *status);
        }
    });

    // ─── 接收者 2 ───
    let mut rx2 = rx.clone();
    tokio::spawn(async move {
        while rx2.changed().await.is_ok() {
            let status = rx2.borrow();
            println!("接收者 2 看到状态: {}", *status);
        }
    });

    // ─── 发送者更新状态 ───
    tx.send("connecting".into()).unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    tx.send("connected".into()).unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    tx.send("disconnected".into()).unwrap();
}
```

### 5.2 watch 的特点

| 特性 | 说明 |
|------|------|
| 发送者 | 只有 1 个（不能 clone） |
| 接收者 | 可以有多个（`Receiver` 可以 clone） |
| 保留值 | 只保留**最新值**，中间值可能被跳过 |
| 适用场景 | 配置变更通知、状态广播、优雅关闭信号 |

### 5.3 典型场景：配置热更新

```rust
use tokio::sync::watch;

#[derive(Clone, Debug)]
struct AppConfig {
    log_level: String,
    max_connections: usize,
}

async fn config_watcher(tx: watch::Sender<AppConfig>) {
    loop {
        // 定期检查配置文件
        tokio::time::sleep(Duration::from_secs(30)).await;
        if let Ok(new_config) = load_config_from_file().await {
            tx.send(new_config).unwrap();  // 广播新配置
        }
    }
}

async fn worker(mut config_rx: watch::Receiver<AppConfig>) {
    loop {
        // 使用当前配置工作
        let config = config_rx.borrow().clone();
        do_work_with_config(&config).await;

        // 检查是否有新配置
        if config_rx.has_changed().unwrap() {
            let new_config = config_rx.borrow_and_update().clone();
            println!("配置已更新: {:?}", new_config);
        }
    }
}
```

---

## 6. ZeroClaw 的通道架构

### 6.1 Channel Trait：mpsc 作为核心通信管道

ZeroClaw 的 `Channel` Trait 使用 `mpsc::Sender` 将消息从各个平台（Telegram、Discord、CLI）传递给 Agent 循环：

```rust
// ZeroClaw 源码（traits.rs 简化版）
use tokio::sync::mpsc;

/// Channel Trait — 所有通信平台都实现这个接口
#[async_trait]
pub trait Channel: Send + Sync {
    /// 平台名称
    fn name(&self) -> &str;

    /// 发送消息到平台
    async fn send(&self, message: &SendMessage) -> anyhow::Result<()>;

    /// 监听平台消息，通过 mpsc 通道传递给调用者
    async fn listen(
        &self,
        tx: tokio::sync::mpsc::Sender<ChannelMessage>,
    ) -> anyhow::Result<()>;
    //    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //    关键设计：调用者传入 Sender，Channel 往里发消息
}
```

### 6.2 为什么用 mpsc？

```
设计选择分析：

方案 A：回调函数
  channel.listen(|msg| process(msg));
  ❌ 闭包生命周期复杂
  ❌ 错误处理困难
  ❌ 与 async 生态不兼容

方案 B：共享队列 (Arc<Mutex<VecDeque>>)
  ❌ 消费者需要忙轮询
  ❌ 锁竞争
  ❌ 没有背压

方案 C：mpsc 通道 ✅
  ✅ 消费者自动等待，零 CPU 开销
  ✅ 多个 Channel 可以 clone Sender，共用一个 Receiver
  ✅ 有界通道提供背压
  ✅ 通道关闭自动通知（recv 返回 None）
  ✅ 与 async/await 完美配合
```

### 6.3 多 Channel 共用一个 mpsc

```rust
// ZeroClaw 的多 Channel 架构
async fn run_channels(config: Config) -> Result<()> {
    // 创建一个 mpsc 通道——所有 Channel 共用
    let (tx, mut rx) = mpsc::channel::<ChannelMessage>(100);

    // ─── 启动多个 Channel 监听器 ───
    // 每个 Channel 拿到 tx 的克隆

    if config.telegram_enabled {
        let tx_telegram = tx.clone();
        tokio::spawn(async move {
            let telegram = TelegramChannel::new(config.telegram_token);
            telegram.listen(tx_telegram).await.unwrap();
            // Telegram 收到消息 → tx_telegram.send(msg)
        });
    }

    if config.discord_enabled {
        let tx_discord = tx.clone();
        tokio::spawn(async move {
            let discord = DiscordChannel::new(config.discord_token);
            discord.listen(tx_discord).await.unwrap();
            // Discord 收到消息 → tx_discord.send(msg)
        });
    }

    if config.cli_enabled {
        let tx_cli = tx.clone();
        tokio::spawn(async move {
            let cli = CliChannel::new();
            cli.listen(tx_cli).await.unwrap();
            // CLI 收到输入 → tx_cli.send(msg)
        });
    }

    // 重要：drop 原始 tx，这样所有 Channel 关闭后通道才会关闭
    drop(tx);

    // ─── Agent 循环：统一处理所有来源的消息 ───
    while let Some(msg) = rx.recv().await {
        // msg 可能来自 Telegram、Discord 或 CLI
        // Agent 不需要关心消息来源！
        println!("[{}] 收到消息: {}", msg.source, msg.content);
        agent.process(msg).await?;
    }

    Ok(())
}
```

**架构可视化**：

```
┌──────────────┐
│  Telegram    │──► tx.clone() ──┐
│  Channel     │                  │
└──────────────┘                  │
                                  │     ┌───────────┐
┌──────────────┐                  ├────►│  mpsc     │
│  Discord     │──► tx.clone() ──┤     │  缓冲区   │──► rx ──► Agent Loop
│  Channel     │                  │     │  (100条)  │
└──────────────┘                  │     └───────────┘
                                  │
┌──────────────┐                  │
│  CLI         │──► tx.clone() ──┘
│  Channel     │
└──────────────┘

多个生产者（Multi-Producer）→ 一个消费者（Single-Consumer）
这就是 mpsc 的完美应用场景！
```

### 6.4 Discord 心跳的 mpsc 用法

Discord 的 WebSocket 连接需要定期发送心跳来保持连接。ZeroClaw 用 mpsc 通道来协调心跳定时器和 WebSocket 发送：

```rust
// ZeroClaw 源码（discord.rs 简化版）
async fn run_discord_connection(ws: WebSocket) -> Result<()> {
    let heartbeat_interval = 41250;  // Discord 指定的心跳间隔（毫秒）

    // 创建心跳通道
    let (hb_tx, mut hb_rx) = tokio::sync::mpsc::channel::<()>(1);

    // ─── 心跳定时器任务 ───
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(
            Duration::from_millis(heartbeat_interval)
        );
        loop {
            interval.tick().await;
            // 每 41.25 秒发送一个心跳信号
            if hb_tx.send(()).await.is_err() {
                break;  // 接收端关闭 → WebSocket 已断开
            }
        }
    });

    // ─── WebSocket 主循环 ───
    loop {
        tokio::select! {
            // 分支 1：收到心跳信号 → 发送心跳包
            Some(()) = hb_rx.recv() => {
                ws.send_heartbeat().await?;
            }
            // 分支 2：收到 WebSocket 消息 → 处理
            Some(msg) = ws.recv() => {
                handle_ws_message(msg).await?;
            }
            // 分支 3：WebSocket 关闭
            else => break,
        }
    }

    Ok(())
}
```

**设计精巧之处**：
- 心跳定时和 WebSocket 消息处理在**同一个 select! 循环**中
- 不需要额外的锁或共享状态
- 心跳定时器任务崩溃 → `hb_tx` drop → `hb_rx.recv()` 返回 `None` → 主循环自动感知

### 6.5 完整的消息流转图

```
用户在 Telegram 发消息 "你好"
          │
          ▼
┌─────────────────────┐
│ Telegram API        │  ← 外部服务
│ (WebSocket/Polling) │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ TelegramChannel     │
│ .listen(tx) {       │
│   loop {            │
│     let msg = ...   │
│     tx.send(msg)    │──────► mpsc channel ──────┐
│   }                 │                            │
│ }                   │                            │
└─────────────────────┘                            │
                                                    ▼
                                          ┌─────────────────┐
                                          │  Agent Loop     │
                                          │                 │
                                          │  rx.recv()      │
                                          │    │            │
                                          │    ▼            │
                                          │  provider       │
                                          │    .chat()      │──► LLM API
                                          │    │            │
                                          │    ▼            │
                                          │  tool           │
                                          │    .execute()   │──► 工具执行
                                          │    │            │
                                          │    ▼            │
                                          │  channel        │
                                          │    .send()      │──► 回复用户
                                          └─────────────────┘
```

---

## 7. 通道选择指南

### 7.1 四种通道对比

| 特性 | `mpsc` | `oneshot` | `watch` | `broadcast` |
|------|--------|-----------|---------|-------------|
| 发送者数量 | 多个（clone） | 1 个 | 1 个 | 多个（clone） |
| 接收者数量 | 1 个 | 1 个 | 多个（clone） | 多个（subscribe） |
| 消息保留 | 缓冲区内全部 | 1 条 | 只有最新值 | 缓冲区内全部 |
| 消息次数 | 多次 | 1 次 | 多次 | 多次 |
| 典型场景 | 任务通信 | 请求-响应 | 状态广播 | 事件广播 |
| TypeScript 类比 | EventEmitter | Promise | 状态管理 | Pub/Sub |

### 7.2 场景决策树

```
你需要在异步任务之间传递数据？
│
├─ 只需要传一次？
│   └─ → oneshot（请求-响应、Future 结果）
│
├─ 需要持续传递？
│   │
│   ├─ 多个发送者，一个接收者？
│   │   └─ → mpsc（最常用！任务通信、命令队列）
│   │
│   ├─ 一个发送者，多个接收者？
│   │   │
│   │   ├─ 接收者只关心最新值？
│   │   │   └─ → watch（配置更新、状态广播）
│   │   │
│   │   └─ 接收者需要所有消息？
│   │       └─ → broadcast（事件通知、日志广播）
│   │
│   └─ 多个发送者，多个接收者？
│       └─ → broadcast 或 多个 mpsc
│
└─ 不确定？
    └─ → 先用 mpsc，它覆盖 80% 的场景
```

### 7.3 代码示例速查

```rust
use tokio::sync::{mpsc, oneshot, watch, broadcast};

// ─── mpsc：多生产者单消费者 ───
let (tx, mut rx) = mpsc::channel::<String>(100);
tx.send("msg".into()).await?;
let msg = rx.recv().await;  // Some("msg") or None

// ─── oneshot：一次性 ───
let (tx, rx) = oneshot::channel::<i32>();
tx.send(42).unwrap();           // 同步发送，不需要 .await
let value = rx.await.unwrap();  // 42

// ─── watch：广播最新值 ───
let (tx, mut rx) = watch::channel("initial".to_string());
tx.send("updated".into())?;
rx.changed().await?;
let value = rx.borrow().clone();  // "updated"

// ─── broadcast：广播所有消息 ───
let (tx, mut rx1) = broadcast::channel::<String>(100);
let mut rx2 = tx.subscribe();  // 额外的接收者
tx.send("event".into())?;
let msg1 = rx1.recv().await?;  // "event"
let msg2 = rx2.recv().await?;  // "event"（两个都收到）
```

---

## 8. 常见错误

### 错误 1：忘记 drop 原始 Sender 导致死锁

```rust
// ❌ 程序永远卡住
let (tx, mut rx) = mpsc::channel::<String>(10);

let tx2 = tx.clone();
tokio::spawn(async move {
    tx2.send("hello".into()).await.unwrap();
    // tx2 drop
});

// tx 还活着！recv() 永远等待
while let Some(msg) = rx.recv().await {
    println!("{}", msg);
}

// ✅ 修复：显式 drop 原始 tx
drop(tx);  // ← 加这一行
while let Some(msg) = rx.recv().await {
    println!("{}", msg);
}
```

### 错误 2：缓冲区太小导致死锁

```rust
// ❌ 如果生产者和消费者在同一个任务中，缓冲区满了会死锁
let (tx, mut rx) = mpsc::channel::<i32>(2);

tx.send(1).await.unwrap();  // 缓冲区: [1]
tx.send(2).await.unwrap();  // 缓冲区: [1, 2]（满了）
tx.send(3).await.unwrap();  // ← 永远卡住！缓冲区满了，没人消费

// ✅ 修复：生产者和消费者在不同任务中
let (tx, mut rx) = mpsc::channel::<i32>(2);

tokio::spawn(async move {
    for i in 0..100 {
        tx.send(i).await.unwrap();  // 缓冲区满了会等待，但消费者在另一个任务中消费
    }
});

while let Some(val) = rx.recv().await {
    println!("{}", val);
}
```

### 错误 3：在非 async 上下文中使用 send()

```rust
// ❌ mpsc::Sender::send() 是 async 的，不能在同步代码中用
fn sync_function(tx: mpsc::Sender<String>) {
    tx.send("hello".into()).await;  // 编译错误！不在 async 上下文中
}

// ✅ 方案 1：用 try_send()（同步，但可能失败）
fn sync_function(tx: mpsc::Sender<String>) {
    tx.try_send("hello".into()).unwrap();
}

// ✅ 方案 2：用 unbounded_channel（send 是同步的）
let (tx, rx) = mpsc::unbounded_channel::<String>();
tx.send("hello".into()).unwrap();  // 同步发送，不需要 .await

// ✅ 方案 3：用 blocking_send()
fn sync_function(tx: mpsc::Sender<String>) {
    tx.blocking_send("hello".into()).unwrap();
    // 注意：不能在 async 上下文中用 blocking_send（会死锁）
}
```

### 错误 4：watch 通道的 borrow 死锁

```rust
// ❌ borrow() 持有读锁期间不能调用 send()
let (tx, rx) = watch::channel(0);
let value = rx.borrow();
tx.send(1).unwrap();  // 可能死锁！borrow 还没释放
println!("{}", *value);

// ✅ 修复：在 borrow 作用域结束后再操作
let value = { *rx.borrow() };  // 作用域内借用，立即释放
tx.send(1).unwrap();  // 安全
```

---

## 9. 小结：mpsc 通道与消息传递的核心要点

```
┌──────────────────────────────────────────────────────────────┐
│          mpsc 通道与消息传递核心要点                          │
│                                                               │
│  1. 通道是异步任务间通信的首选方式                            │
│     → 比共享内存（Mutex）更安全、更高效                       │
│     → 自动等待、自动通知、零 CPU 开销                         │
│                                                               │
│  2. mpsc 是最常用的通道类型                                   │
│     → Multi-Producer, Single-Consumer                        │
│     → Sender 可以 clone，Receiver 不能                       │
│     → 有界通道提供背压（推荐！）                              │
│                                                               │
│  3. 四种通道覆盖所有场景                                      │
│     → mpsc: 多生产者单消费者（80% 场景）                      │
│     → oneshot: 一次性请求-响应                                │
│     → watch: 广播最新值                                       │
│     → broadcast: 广播所有消息                                 │
│                                                               │
│  4. while let Some(msg) = rx.recv().await                    │
│     → 消费通道的标准写法                                      │
│     → 通道关闭自动退出循环（recv 返回 None）                  │
│                                                               │
│  5. ZeroClaw 的 mpsc 架构                                    │
│     → Channel Trait: listen(tx) 将消息发到共用通道            │
│     → 多个 Channel 共享一个 Sender                           │
│     → Agent 从一个 Receiver 统一处理所有消息                  │
│     → Discord 心跳：mpsc + select! 协调定时和消息处理        │
│                                                               │
│  6. 常见陷阱                                                  │
│     → 忘记 drop 原始 Sender → 通道永远不关闭                 │
│     → 同任务中生产+消费 → 缓冲区满死锁                       │
│     → 非 async 中用 send() → 用 try_send 或 unbounded       │
└──────────────────────────────────────────────────────────────┘
```

---

> **下一步**: 阅读 `03_核心概念_6_async_trait与异步Trait.md`，了解 `#[async_trait]` 宏如何让 Trait 支持 async fn，以及 ZeroClaw 四大 Trait 的 Send + Sync 约束。

---

**文件信息**
- 知识点: async/await 与 Tokio
- 维度: 核心概念 5 — mpsc 通道与消息传递
- 版本: v1.0
- 日期: 2026-03-10
