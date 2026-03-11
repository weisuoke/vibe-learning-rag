# async/await 与 Tokio - 实战代码 场景6：Actor 模式实战

> **知识点**: async/await 与 Tokio
> **层级**: Phase1_Rust速成基础
> **维度**: 实战代码 - 场景6
> **场景**: 用 mpsc 通道实现 Actor 模式
> **目标受众**: 有 TypeScript/前端经验但零 Rust 经验的开发者
> **阅读时间**: 约 25 分钟

---

## 概述

前面学了 `mpsc::channel`（场景1 中 Channel → Agent 的通信桥梁），现在把它用到极致——**Actor 模式**。

Actor 模式是 Rust 异步编程中最重要的架构模式之一。ZeroClaw 的每个 Channel 就是一个 Actor——独立运行、通过消息通信、不共享状态。本节从零构建一个完整的 Actor 系统，帮你掌握这个模式的每个细节。

---

## 1. 什么是 Actor 模式？

### 一句话定义

> 每个 Actor 是一个独立的异步任务，拥有私有状态，只通过消息与外界通信。

### 前端类比

```
Actor 模式                    ←→    前端对应物
──────────────────────────────────────────────────
Actor（独立任务 + 私有状态）   ←→    React 组件（独立 state + 生命周期）
Actor 之间发消息              ←→    Context/Redux dispatch
ActorHandle（外部接口）       ←→    组件暴露的 ref / props 回调
mpsc::Sender（发消息）        ←→    dispatchEvent / postMessage
mpsc::Receiver（收消息）      ←→    addEventListener / onMessage
oneshot::Sender（请求-响应）  ←→    fetch().then() / Promise
```

更贴切的类比：**Web Worker**
```
Web Worker                           Rust Actor
────────────────────────────────────────────────────
new Worker('worker.js')              tokio::spawn(actor.run())
worker.postMessage(data)             tx.send(message).await
worker.onmessage = (e) => {...}      while let Some(msg) = rx.recv().await
worker.terminate()                   drop(tx) → Actor 自然退出
不能直接访问 Worker 的内部变量        不能直接访问 Actor 的 state
```

### 日常生活类比

想象一个公司，每个部门就是一个 Actor：

```
┌──────────┐    邮件/电话     ┌──────────┐
│  销售部   │ ──────────────► │  仓库部   │
│ (私有数据: │                 │ (私有数据: │
│  客户列表) │                 │  库存表)  │
└──────────┘                  └──────────┘
     │                              │
     │         邮件/电话             │
     └──────────────────────────────┘

规则：
- 每个部门有自己的私有数据（状态）
- 部门之间只通过邮件/电话沟通（消息）
- 不能直接走进别的部门翻文件（不共享状态）
- 每个部门可以同时运行（并发）
```

---

## 2. Rust 中实现 Actor 的标准四件套

Actor 模式有一个固定的四步套路，每个 Actor 都遵循：

```
Actor 四件套：

1. Message（消息类型）     → enum：定义 Actor 能处理的所有操作
2. Actor（Actor 本体）     → struct：私有状态 + Receiver
3. Actor::run()（运行循环） → async fn：while let Some(msg) = rx.recv() { ... }
4. ActorHandle（外部接口）  → struct：只暴露 Sender，封装发消息的方法
```

### 为什么需要 ActorHandle？

```rust
// ❌ 不好：让调用者直接操作 tx 和消息类型
let (tx, rx) = mpsc::channel(32);
tx.send(ActorMessage::GetCount { reply: reply_tx }).await?;

// ✅ 好：封装成 Handle，提供友好 API
let handle = CounterHandle::new();
let count = handle.get_count().await?;  // 调用者不需要知道消息细节
```

**前端类比**：就像把 `useReducer` 的 dispatch 封装成自定义 hook 一样——`useCounter()` 返回 `{ count, increment, decrement }` 而不是暴露 `dispatch({ type: 'INCREMENT' })`。

---

## 3. 示例1：Counter Actor（最小示例）

先用最简单的计数器理解 Actor 四件套的每一步：

### Cargo.toml

```toml
[package]
name = "actor-patterns"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1", features = ["full"] }
anyhow = "1"
```

### 代码

```rust
use tokio::sync::{mpsc, oneshot};
use anyhow::Result;

// ═══════════════════════════════════════════════════════
// 第1步：定义消息类型
// ═══════════════════════════════════════════════════════

/// Actor 能处理的所有消息
/// 类比 Redux 的 Action 类型：
///   type Action = { type: 'INCREMENT' } | { type: 'GET_COUNT', reply: ... }
enum CounterMessage {
    /// 增加计数（不需要回复 → 单向消息）
    Increment,

    /// 获取当前计数（需要回复 → 请求-响应模式）
    GetCount {
        reply: oneshot::Sender<i64>,  // 回复通道
    },

    /// 重置计数
    Reset {
        reply: oneshot::Sender<i64>,  // 返回重置前的值
    },
}

// ═══════════════════════════════════════════════════════
// 第2步：定义 Actor 本体
// ═══════════════════════════════════════════════════════

/// Counter Actor：拥有私有状态和消息接收器
struct CounterActor {
    rx: mpsc::Receiver<CounterMessage>,  // 消息入口
    count: i64,                          // 私有状态——外部无法直接访问！
}

impl CounterActor {
    fn new(rx: mpsc::Receiver<CounterMessage>) -> Self {
        Self { rx, count: 0 }
    }

    // ═══════════════════════════════════════════════════
    // 第3步：Actor 运行循环
    // ═══════════════════════════════════════════════════

    /// Actor 的事件循环——不断接收和处理消息
    /// 当所有 Sender 被 drop 后，recv() 返回 None → 循环结束 → Actor 退出
    async fn run(mut self) {
        println!("  🎬 [CounterActor] 启动，初始值: {}", self.count);

        // while let Some = 核心循环
        // 类似 JS: for await (const msg of messageStream) { ... }
        while let Some(msg) = self.rx.recv().await {
            match msg {
                CounterMessage::Increment => {
                    self.count += 1;
                    println!("  📈 [CounterActor] +1 → 当前: {}", self.count);
                }
                CounterMessage::GetCount { reply } => {
                    // 通过 oneshot 回复当前值
                    // send 可能失败（调用者不再等待），用 let _ = 忽略错误
                    let _ = reply.send(self.count);
                    println!("  📊 [CounterActor] 查询 → 返回: {}", self.count);
                }
                CounterMessage::Reset { reply } => {
                    let old = self.count;
                    self.count = 0;
                    let _ = reply.send(old);
                    println!("  🔄 [CounterActor] 重置 {} → 0", old);
                }
            }
        }

        println!("  🛑 [CounterActor] 所有 Handle 已断开，退出");
    }
}

// ═══════════════════════════════════════════════════════
// 第4步：ActorHandle（外部接口）
// ═══════════════════════════════════════════════════════

/// CounterHandle：Actor 的"遥控器"
/// 封装了 mpsc::Sender，提供友好的 API
///
/// #[derive(Clone)] 很关键：
/// - mpsc::Sender 可以 Clone（多生产者！）
/// - 所以 Handle 也可以 Clone → 多个地方可以同时操作同一个 Actor
///
/// 前端类比：就像 useContext() 返回的 dispatch 函数，
/// 多个组件都能 dispatch，但 state 只在一个地方管理
#[derive(Clone)]
struct CounterHandle {
    tx: mpsc::Sender<CounterMessage>,
}

impl CounterHandle {
    /// 创建 Actor 和 Handle
    /// 这个函数做了三件事：
    ///   1. 创建 mpsc channel
    ///   2. spawn Actor 到后台运行
    ///   3. 返回 Handle 给调用者
    fn new() -> Self {
        let (tx, rx) = mpsc::channel(32);  // buffer 大小 32
        let actor = CounterActor::new(rx);

        // spawn Actor 到后台——从此 Actor 独立运行
        tokio::spawn(actor.run());

        Self { tx }
    }

    /// 增加计数（Fire-and-forget：发完不等回复）
    async fn increment(&self) -> Result<()> {
        self.tx.send(CounterMessage::Increment).await?;
        Ok(())
    }

    /// 获取计数（Request-Response：发消息 + 等回复）
    async fn get_count(&self) -> Result<i64> {
        // 1. 创建一次性回复通道
        let (reply_tx, reply_rx) = oneshot::channel();

        // 2. 发送带回复通道的消息
        self.tx.send(CounterMessage::GetCount { reply: reply_tx }).await?;

        // 3. 等待 Actor 的回复
        let count = reply_rx.await?;
        Ok(count)
    }

    /// 重置计数
    async fn reset(&self) -> Result<i64> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx.send(CounterMessage::Reset { reply: reply_tx }).await?;
        let old_count = reply_rx.await?;
        Ok(old_count)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("═══ Actor 模式：Counter Actor ═══\n");

    // 创建 Actor（spawn 到后台运行）
    let handle = CounterHandle::new();

    // ─── 基本操作 ───
    handle.increment().await?;
    handle.increment().await?;
    handle.increment().await?;

    let count = handle.get_count().await?;
    println!("  → 当前计数: {}\n", count);

    // ─── Clone Handle：多个"遥控器"操作同一个 Actor ───
    println!("--- 多 Handle 并发操作 ---\n");
    let handle2 = handle.clone();  // Clone！共享同一个 Actor

    // 两个 Handle 同时发消息（并发安全！）
    let (r1, r2) = tokio::join!(
        async {
            for _ in 0..5 {
                handle.increment().await.unwrap();
            }
            handle.get_count().await.unwrap()
        },
        async {
            for _ in 0..3 {
                handle2.increment().await.unwrap();
            }
            handle2.get_count().await.unwrap()
        },
    );
    println!("  → Handle1 看到: {}, Handle2 看到: {}", r1, r2);

    // ─── 重置 ───
    let old = handle.reset().await?;
    println!("  → 重置前的值: {}", old);

    let count = handle.get_count().await?;
    println!("  → 重置后的值: {}", count);

    // ─── Actor 自动退出 ───
    // 当所有 Handle（tx）被 drop 后，Actor 的 rx.recv() 返回 None
    // Actor 循环结束 → 任务完成
    drop(handle);
    drop(handle2);

    // 给 Actor 一点时间打印退出消息
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    println!("\n═══ Counter Actor 演示结束 ═══");
    Ok(())
}
```

### oneshot 通道详解

```
请求-响应模式（oneshot）：

调用者                          Actor
  │                               │
  │  1. 创建 oneshot channel      │
  │  let (reply_tx, reply_rx)     │
  │                               │
  │  2. 发送消息（带 reply_tx）    │
  │  ─── GetCount { reply } ────► │
  │                               │  3. 处理请求
  │                               │     计算 count
  │                               │
  │  4. 等待回复（reply_rx）       │  4. 发送回复
  │  ◄── reply.send(count) ────── │
  │                               │
  │  5. 得到结果                   │
  count = reply_rx.await?         │


前端类比：
  // 调用者
  const result = await fetch('/api/count');  // fetch 内部就是"发请求 + 等回复"

  // Rust 的 oneshot 就是手动实现了 fetch 的 request-response 模式
  // oneshot::channel() = 创建一个只能用一次的 Promise
```

---

## 4. 示例2：聊天室 Actor（完整实战）

用 Actor 模式实现一个多用户聊天室，展示更复杂的消息类型和多 Actor 协作。

```rust
use tokio::sync::{mpsc, oneshot};
use std::collections::HashMap;
use std::time::Duration;
use anyhow::Result;

// ═══════════════════════════════════════════════════════
// 聊天室消息类型
// ═══════════════════════════════════════════════════════

#[derive(Debug, Clone)]
struct ChatMessage {
    user: String,
    content: String,
    timestamp: u64,
}

enum RoomMessage {
    /// 用户加入聊天室
    Join {
        user: String,
        /// 给该用户推送消息的通道
        user_tx: mpsc::Sender<ChatMessage>,
        reply: oneshot::Sender<Result<Vec<ChatMessage>, String>>,
    },

    /// 用户离开聊天室
    Leave {
        user: String,
    },

    /// 发送聊天消息
    SendChat {
        user: String,
        content: String,
    },

    /// 获取在线用户列表
    ListUsers {
        reply: oneshot::Sender<Vec<String>>,
    },

    /// 获取消息历史
    GetHistory {
        limit: usize,
        reply: oneshot::Sender<Vec<ChatMessage>>,
    },
}

// ═══════════════════════════════════════════════════════
// ChatRoom Actor
// ═══════════════════════════════════════════════════════

struct ChatRoomActor {
    rx: mpsc::Receiver<RoomMessage>,
    /// 在线用户和他们的消息通道
    users: HashMap<String, mpsc::Sender<ChatMessage>>,
    /// 消息历史
    history: Vec<ChatMessage>,
    /// 时间戳计数器（简化版）
    time_counter: u64,
}

impl ChatRoomActor {
    fn new(rx: mpsc::Receiver<RoomMessage>) -> Self {
        Self {
            rx,
            users: HashMap::new(),
            history: Vec::new(),
            time_counter: 0,
        }
    }

    async fn run(mut self) {
        println!("  🏠 [ChatRoom] 聊天室已创建");

        while let Some(msg) = self.rx.recv().await {
            match msg {
                RoomMessage::Join { user, user_tx, reply } => {
                    if self.users.contains_key(&user) {
                        let _ = reply.send(Err(format!("用户 {} 已在聊天室中", user)));
                    } else {
                        println!("  ✅ [ChatRoom] {} 加入聊天室", user);
                        self.users.insert(user.clone(), user_tx);

                        // 广播加入通知
                        self.broadcast(&format!("📢 {} 加入了聊天室", user), "系统").await;

                        // 返回最近 10 条历史消息
                        let recent: Vec<ChatMessage> = self.history
                            .iter()
                            .rev()
                            .take(10)
                            .rev()
                            .cloned()
                            .collect();
                        let _ = reply.send(Ok(recent));
                    }
                }

                RoomMessage::Leave { user } => {
                    if self.users.remove(&user).is_some() {
                        println!("  👋 [ChatRoom] {} 离开聊天室", user);
                        self.broadcast(&format!("📢 {} 离开了聊天室", user), "系统").await;
                    }
                }

                RoomMessage::SendChat { user, content } => {
                    self.time_counter += 1;
                    let chat_msg = ChatMessage {
                        user: user.clone(),
                        content: content.clone(),
                        timestamp: self.time_counter,
                    };

                    // 保存到历史
                    self.history.push(chat_msg.clone());

                    // 广播给所有用户（包括发送者）
                    println!("  💬 [ChatRoom] {}: {}", user, content);
                    self.broadcast_msg(chat_msg).await;
                }

                RoomMessage::ListUsers { reply } => {
                    let users: Vec<String> = self.users.keys().cloned().collect();
                    let _ = reply.send(users);
                }

                RoomMessage::GetHistory { limit, reply } => {
                    let history: Vec<ChatMessage> = self.history
                        .iter()
                        .rev()
                        .take(limit)
                        .rev()
                        .cloned()
                        .collect();
                    let _ = reply.send(history);
                }
            }
        }

        println!("  🏠 [ChatRoom] 聊天室关闭");
    }

    /// 广播文本消息
    async fn broadcast(&self, content: &str, from: &str) {
        self.time_counter;
        let msg = ChatMessage {
            user: from.to_string(),
            content: content.to_string(),
            timestamp: self.time_counter,
        };
        self.broadcast_msg(msg).await;
    }

    /// 广播消息对象给所有在线用户
    async fn broadcast_msg(&self, msg: ChatMessage) {
        for (user, tx) in &self.users {
            // 发送失败说明用户已断开，忽略错误
            if tx.send(msg.clone()).await.is_err() {
                println!("  ⚠️ [ChatRoom] 用户 {} 的通道已断开", user);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════
// ChatRoom Handle
// ═══════════════════════════════════════════════════════

#[derive(Clone)]
struct ChatRoomHandle {
    tx: mpsc::Sender<RoomMessage>,
}

impl ChatRoomHandle {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel(64);
        let actor = ChatRoomActor::new(rx);
        tokio::spawn(actor.run());
        Self { tx }
    }

    /// 加入聊天室，返回（消息接收器, 历史消息）
    async fn join(&self, user: &str) -> Result<(mpsc::Receiver<ChatMessage>, Vec<ChatMessage>)> {
        let (user_tx, user_rx) = mpsc::channel(32);
        let (reply_tx, reply_rx) = oneshot::channel();

        self.tx.send(RoomMessage::Join {
            user: user.to_string(),
            user_tx,
            reply: reply_tx,
        }).await?;

        let history = reply_rx.await?
            .map_err(|e| anyhow::anyhow!(e))?;

        Ok((user_rx, history))
    }

    /// 离开聊天室
    async fn leave(&self, user: &str) -> Result<()> {
        self.tx.send(RoomMessage::Leave {
            user: user.to_string(),
        }).await?;
        Ok(())
    }

    /// 发送聊天消息
    async fn send_chat(&self, user: &str, content: &str) -> Result<()> {
        self.tx.send(RoomMessage::SendChat {
            user: user.to_string(),
            content: content.to_string(),
        }).await?;
        Ok(())
    }

    /// 获取在线用户列表
    async fn list_users(&self) -> Result<Vec<String>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx.send(RoomMessage::ListUsers { reply: reply_tx }).await?;
        Ok(reply_rx.await?)
    }

    /// 获取消息历史
    async fn get_history(&self, limit: usize) -> Result<Vec<ChatMessage>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx.send(RoomMessage::GetHistory { limit, reply: reply_tx }).await?;
        Ok(reply_rx.await?)
    }
}

// ═══════════════════════════════════════════════════════
// 模拟用户（也是一个 Actor！）
// ═══════════════════════════════════════════════════════

/// 模拟一个聊天用户：加入聊天室，收发消息
async fn simulate_user(
    name: &str,
    room: ChatRoomHandle,
    messages: Vec<String>,
) -> Result<()> {
    // 加入聊天室
    let (mut rx, history) = room.join(name).await?;
    println!("  👤 [{}] 加入成功，收到 {} 条历史消息", name, history.len());

    // 启动消息接收任务（后台监听）
    let user_name = name.to_string();
    let receiver = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            // 不打印自己发的消息（避免重复输出）
            if msg.user != user_name {
                println!("  👀 [{}] 收到 {} 的消息: {}", user_name, msg.user, msg.content);
            }
        }
    });

    // 发送消息
    for content in messages {
        tokio::time::sleep(Duration::from_millis(100)).await;
        room.send_chat(name, &content).await?;
    }

    // 等一会儿接收其他人的消息
    tokio::time::sleep(Duration::from_millis(300)).await;

    // 离开聊天室
    room.leave(name).await?;

    // 停止接收任务
    receiver.abort();
    let _ = receiver.await;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("═══ Actor 模式：聊天室 ═══\n");

    let room = ChatRoomHandle::new();

    // 多个用户并发聊天
    let (r1, r2, r3) = tokio::join!(
        simulate_user("Alice", room.clone(), vec![
            "大家好！".to_string(),
            "今天天气不错".to_string(),
        ]),
        simulate_user("Bob", room.clone(), vec![
            "嗨 Alice！".to_string(),
            "确实，适合写代码".to_string(),
        ]),
        async {
            // Charlie 晚一点加入
            tokio::time::sleep(Duration::from_millis(250)).await;
            simulate_user("Charlie", room.clone(), vec![
                "我来晚了！".to_string(),
            ]).await
        },
    );

    r1?; r2?; r3?;

    // 查看最终状态
    let users = room.list_users().await?;
    println!("\n  📋 最终在线用户: {:?}", users);

    let history = room.get_history(20).await?;
    println!("  📜 消息历史 ({} 条):", history.len());
    for msg in &history {
        println!("    [{}] {}: {}", msg.timestamp, msg.user, msg.content);
    }

    println!("\n═══ 聊天室演示结束 ═══");
    Ok(())
}
```

### 架构图

```
                    ┌───────────────────────────────────────────────┐
                    │              ChatRoom Actor                    │
                    │                                               │
                    │  私有状态：                                    │
                    │  ├── users: HashMap<String, Sender>            │
                    │  └── history: Vec<ChatMessage>                 │
                    │                                               │
                    │  循环：while let Some(msg) = rx.recv() {...}   │
                    └──────────────────┬────────────────────────────┘
                                       │
                              mpsc::Receiver<RoomMessage>
                                       │
                    ┌──────────────────┴────────────────────────────┐
                    │                                               │
         mpsc::Sender              mpsc::Sender              mpsc::Sender
              │                         │                         │
    ┌─────────┴────────┐    ┌──────────┴─────────┐    ┌─────────┴────────┐
    │  Alice Handle     │    │  Bob Handle         │    │  Charlie Handle   │
    │  (clone of room)  │    │  (clone of room)    │    │  (clone of room)  │
    └──────────────────┘    └────────────────────┘    └──────────────────┘

    每个用户还有自己的接收通道（user_tx → user_rx）：
    ChatRoom Actor 通过 user_tx 把消息推送给每个用户
```

---

## 5. ZeroClaw 中的 Actor 模式

### Channel 作为 Actor

ZeroClaw 的每个 Channel（Discord、Slack 等）本质上就是一个 Actor：

```rust
// ZeroClaw 源码模式（简化版）

// 1. Channel trait 定义了 Actor 的行为
#[async_trait]
trait Channel: Send + Sync {
    async fn listen(&self, tx: mpsc::Sender<ChannelMessage>) -> Result<()>;
    // listen() 就是 Actor 的 run() 循环
    // tx 就是发给 Agent Loop 的消息通道
}

// 2. 每个 Channel 被 spawn 成独立任务（= 独立 Actor）
fn spawn_supervised_listener(
    ch: Arc<dyn Channel>,
    tx: mpsc::Sender<ChannelMessage>,
    initial_backoff_secs: u64,
    max_backoff_secs: u64,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        // Actor 循环：监听 → 出错 → 退避 → 重试
        let mut backoff = initial_backoff_secs.max(1);
        loop {
            match ch.listen(tx.clone()).await {
                Ok(()) => { backoff = initial_backoff_secs.max(1); }
                Err(e) => { /* 记录错误 */ }
            }
            tokio::time::sleep(Duration::from_secs(backoff)).await;
            backoff = backoff.saturating_mul(2).min(max_backoff_secs);
        }
    })
}

// 3. Agent Loop 作为消费者
//    rx.recv() 接收来自所有 Channel 的消息
async fn agent_loop(mut rx: mpsc::Receiver<ChannelMessage>) {
    while let Some(msg) = rx.recv().await {
        // 处理消息...
    }
}
```

### ZeroClaw 的 Actor 架构图

```
┌────────────┐     ┌────────────┐     ┌────────────┐
│  Discord    │     │  Slack      │     │  CLI        │
│  Channel    │     │  Channel    │     │  Channel    │
│  (Actor 1)  │     │  (Actor 2)  │     │  (Actor 3)  │
└──────┬─────┘     └──────┬─────┘     └──────┬─────┘
       │                  │                   │
       │     mpsc::Sender (tx.clone())        │
       │                  │                   │
       └──────────────────┼───────────────────┘
                          │
                   mpsc::Receiver (rx)
                          │
                ┌─────────┴──────────┐
                │    Agent Loop       │
                │    (消费者 Actor)    │
                │                    │
                │    select! {       │
                │      msg = rx =>   │
                │      signal =>     │
                │    }               │
                └────────────────────┘
```

---

## 6. Actor vs 共享状态

### 什么时候用 Actor？什么时候用 Arc<Mutex>？

```
┌─────────────────┬──────────────────────┬──────────────────────┐
│ 方面             │ Actor (mpsc)          │ 共享状态 (Arc<Mutex>) │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 复杂度           │ 中等                  │ 简单                  │
│                 │ 需要定义消息类型       │ 直接 lock + 操作       │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 安全性           │ 高（无死锁风险）       │ 可能死锁               │
│                 │ 消息顺序处理           │ 锁竞争                │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 性能             │ 好                    │ 简单场景更快           │
│                 │ 无锁竞争              │ 高并发时锁成瓶颈       │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 请求-响应        │ ✅ oneshot 通道        │ ✅ 直接返回值          │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 异步操作         │ ✅ 天然支持            │ ⚠️ Mutex 不能跨 await │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 适用场景         │ 复杂状态管理           │ 简单计数器/缓存        │
│                 │ 需要异步处理           │ 读多写少               │
│                 │ 多组件交互             │ 状态简单               │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 前端类比         │ Redux / useReducer    │ React.useState         │
│                 │ 集中管理复杂状态       │ 简单本地状态           │
└─────────────────┴──────────────────────┴──────────────────────┘
```

### 代码对比

```rust
// ─── 共享状态方式（简单但有限制） ───

use std::sync::Arc;
use tokio::sync::Mutex;

let counter = Arc::new(Mutex::new(0i64));

// 多个任务共享
let c1 = counter.clone();
tokio::spawn(async move {
    let mut lock = c1.lock().await;  // 获取锁
    *lock += 1;                      // 修改
    // lock 在这里自动释放
});

// 问题：如果在持有锁的时候 .await，其他任务会被阻塞
// let mut lock = counter.lock().await;
// do_something_async().await;  // ⚠️ 这里还持有锁！其他任务无法访问
// *lock += 1;

// ─── Actor 方式（稍复杂但更安全） ───

// Actor 内部不需要锁——因为只有一个任务（Actor 的 run 循环）访问状态
// 外部通过 Handle 发消息——天然异步安全
let handle = CounterHandle::new();
handle.increment().await?;  // 不持有任何锁，不阻塞其他操作
let count = handle.get_count().await?;  // 通过 oneshot 获取结果
```

### 选择指南

```
问自己这 3 个问题：

1. 状态是否需要异步操作来修改？
   是 → Actor（Mutex 不能跨 .await 点持有）
   否 → 都行

2. 有多少个地方需要修改状态？
   多个 → Actor（避免锁竞争）
   1-2个 → Arc<Mutex> 就够了

3. 状态修改逻辑是否复杂？
   复杂（需要多步操作、条件判断）→ Actor
   简单（递增、设置值）→ Arc<Mutex>

ZeroClaw 的选择：
  Channel 监听 → Actor（复杂的异步操作）
  状态写入    → Actor（spawn_state_writer）
  配置共享    → Arc<Config>（只读，不需要锁）
```

---

## 7. TypeScript 对照：Actor 模式

```typescript
// ═══ TypeScript 实现 Actor 模式对照 ═══

// ─── 方式1：用 EventTarget 模拟 Actor ───

class CounterActor extends EventTarget {
  private count = 0;  // 私有状态

  constructor() {
    super();
    // "run 循环"——监听消息
    this.addEventListener('increment', () => {
      this.count++;
      console.log(`+1 → ${this.count}`);
    });

    this.addEventListener('get_count', (e: Event) => {
      const detail = (e as CustomEvent).detail;
      detail.resolve(this.count);  // "oneshot 回复"
    });
  }

  // Handle 方法
  async getCount(): Promise<number> {
    return new Promise((resolve) => {
      this.dispatchEvent(new CustomEvent('get_count', {
        detail: { resolve }
      }));
    });
  }

  increment() {
    this.dispatchEvent(new Event('increment'));
  }
}

// ─── 方式2：用 Web Worker 模拟 Actor（更贴切） ───

// worker.js (Actor 本体)
// let count = 0;
// self.onmessage = (e) => {
//   switch (e.data.type) {
//     case 'increment':
//       count++;
//       break;
//     case 'get_count':
//       self.postMessage({ type: 'count_result', value: count });
//       break;
//   }
// };

// main.js (Handle)
// const worker = new Worker('worker.js');
// worker.postMessage({ type: 'increment' });
// worker.postMessage({ type: 'get_count' });
// worker.onmessage = (e) => {
//   if (e.data.type === 'count_result') {
//     console.log('Count:', e.data.value);
//   }
// };

// ─── Rust vs TS Actor 对比 ───
//
// Rust Actor                        TypeScript Actor
// ─────────────────────────────────────────────────────
// enum Message { ... }              { type: string, payload: any }
// mpsc::channel(32)                 new MessageChannel() / postMessage
// tokio::spawn(actor.run())         new Worker('actor.js')
// handle.tx.send(msg).await         worker.postMessage(msg)
// oneshot::channel()                new Promise((resolve) => ...)
// drop(handle) → Actor 退出         worker.terminate()
//
// 核心区别：
// Rust 的 Actor 可以真正并行在不同 CPU 核心上运行
// JS 的 Web Worker 也能并行，但 EventTarget 方式仍然是单线程
```

---

## 8. 速查表

```
Actor 模式四件套：
  1. enum Message { ... }              ← 定义消息类型（类似 Redux Action）
  2. struct Actor { rx, state }        ← Actor 本体（私有状态 + 接收器）
  3. async fn run(mut self)            ← while let Some(msg) = rx.recv() { ... }
  4. struct Handle { tx }              ← 外部接口（Clone → 多个遥控器）

关键操作：
  Handle::new()                        ← 创建 channel + spawn Actor + 返回 Handle
  handle.clone()                       ← 复制遥控器（共享同一个 Actor）
  tx.send(msg).await                   ← 发消息给 Actor
  let (reply_tx, reply_rx) = oneshot   ← 请求-响应模式
  drop(所有 handle)                    ← Actor 自然退出

Actor vs Arc<Mutex>：
  需要异步操作？          → Actor
  多处修改 + 复杂逻辑？   → Actor
  简单读写 + 少量修改？   → Arc<Mutex>

ZeroClaw 中的 Actor：
  Channel.listen(tx)                   ← Channel 是 Actor
  spawn_supervised_listener(ch, tx)    ← spawn = 启动 Actor
  agent_loop(rx)                       ← Agent = 消费者
```

### 一句话总结

> **Actor = 独立任务 + 私有状态 + 消息通信**——用 `mpsc::channel` 传命令、`oneshot::channel` 传回复、`tokio::spawn` 启动运行。ZeroClaw 的每个 Channel 就是一个 Actor，通过 mpsc 把消息汇聚到 Agent Loop。当你发现需要"一个独立运行的有状态服务"时，就该用 Actor 模式——它比 `Arc<Mutex>` 更安全、比锁更适合异步场景。
