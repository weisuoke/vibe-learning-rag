# async/await 与 Tokio - 核心概念 2：Tokio 运行时与 #[tokio::main]

> **知识点**: async/await 与 Tokio
> **层级**: Phase1_Rust速成基础
> **维度**: 核心概念 2 / 6
> **目标受众**: 有 TypeScript/前端经验但零 Rust 经验的开发者
> **阅读时间**: 约 20 分钟

---

## 概述

上一节我们知道了：`async fn` 创建的 Future 是**惰性**的——调用它什么都不会发生，就像一张食谱卡片。那么问题来了：**谁来当厨师？谁来驱动这些 Future 执行？**

答案是：**运行时（Runtime）**。Rust 的 async 运行时不是语言内置的，你需要显式选择一个——绝大多数项目（包括 ZeroClaw）选择的是 **Tokio**。

**本节核心问题**：
1. 为什么 Rust 需要独立的运行时？
2. Tokio 的多线程和单线程运行时有什么区别？
3. `#[tokio::main]` 这个宏到底做了什么？
4. ZeroClaw 的 Tokio Feature Flags 为什么那么多？
5. 从 `main()` 启动到 Agent 运行的完整流程是什么？

---

## 1. 什么是运行时（Runtime）？

### 1.1 为什么 Rust async 需要独立运行时

在 JavaScript 的世界里，你**永远不需要关心运行时**——V8 引擎内置了事件循环（Event Loop），`Promise` 自动被调度，你只管写 `async/await` 就行。

```typescript
// JavaScript: 你不需要做任何"启动运行时"的操作
async function main() {
    const data = await fetchData();  // 事件循环自动调度
    console.log(data);
}

main();  // V8 引擎自动处理 Promise 调度
```

**Rust 刻意选择了不同的路**：语言本身只定义了 `Future` trait 和 `async/await` 语法，**不包含任何运行时**。

```rust
// Rust: 如果不引入运行时，你的 Future 无法执行！
async fn main_logic() -> String {
    "Hello".to_string()
}

fn main() {
    let future = main_logic();  // 创建了一个 Future
    // ??? 谁来 poll 它？谁来驱动它执行？
    // 没有运行时 = 没有厨师 = 食谱永远只是一张纸
}
```

**为什么这样设计？** 三个理由：

| 理由 | 说明 |
|------|------|
| **零开销原则** | 不需要 async 的程序不付出任何代价（嵌入式、操作系统内核） |
| **选择自由** | 不同场景用不同运行时：Tokio（通用）、async-std、smol（轻量）、Embassy（嵌入式） |
| **可替换性** | 不满意？换个运行时，业务代码不用改（只要都实现 `Future` trait） |

### 1.2 运行时到底做什么？

运行时本质上是一个**调度器**——它接收 Future，反复 poll 它们，在 IO 就绪时唤醒对应的 Future。

```
运行时的核心工作：

┌────────────────────────────────────────────────────────┐
│                    Tokio Runtime                        │
│                                                         │
│   1. 接收 Future（通过 block_on 或 spawn）                │
│   2. 调用 future.poll(cx) 驱动执行                       │
│   3. 如果 Pending → 注册 Waker，去做别的任务              │
│   4. IO 就绪 → Waker 触发 → 再次 poll                   │
│   5. Ready(value) → 完成！返回结果                       │
│                                                         │
│   同时管理：                                             │
│   ├── 任务队列（就绪 / 等待中的 Future）                  │
│   ├── IO 事件循环（epoll/kqueue/IOCP）                   │
│   ├── 定时器（sleep、timeout、interval）                  │
│   └── 线程池（工作窃取调度）                              │
└────────────────────────────────────────────────────────┘
```

### 1.3 类比理解

**前端类比——Node.js 事件循环 vs Tokio 调度器**：

```
Node.js:
  ┌──────────────────────────────────┐
  │  V8 引擎（内置，不可选择）         │
  │  ├── 微任务队列（Promise）        │
  │  ├── 宏任务队列（setTimeout）     │
  │  └── libuv（IO 事件循环）         │
  └──────────────────────────────────┘
  你的代码 → 直接运行，V8 自动管理一切
  单线程，一个事件循环

Tokio:
  ┌──────────────────────────────────┐
  │  Tokio Runtime（你主动创建）       │
  │  ├── 任务队列（Future 调度）       │
  │  ├── IO 驱动（epoll/kqueue）      │
  │  ├── 定时器驱动（时间轮）          │
  │  └── 工作线程池（可配置数量）       │
  └──────────────────────────────────┘
  你的代码 → 先创建 Runtime → 交给它运行
  多线程，多个工作线程可以并行
```

**日常生活类比——交通信号系统**：

- **Future** = 一辆等着通行的车
- **运行时** = 整个交通信号系统（红绿灯 + 调度中心）
- **poll** = 绿灯亮了，车可以走一段
- **Pending** = 红灯，车停在这里等
- **Waker** = 感应器检测到前方路口清了，通知这个路口变绿灯
- **多线程** = 多车道，多辆车同时通行
- **单线程** = 单车道，一次只过一辆（但切换很快）

没有交通信号系统，所有车都不知道什么时候能走——这就是"没有运行时就无法执行 Future"的含义。

---

## 2. Tokio 多线程 vs 单线程运行时

Tokio 提供两种运行时模式，适用于不同场景。

### 2.1 多线程运行时（rt-multi-thread）

**这是 ZeroClaw 使用的模式**，也是大多数服务器应用的默认选择。

```rust
// 使用多线程运行时
#[tokio::main]  // 默认就是多线程
async fn main() {
    // Tokio 创建多个工作线程（默认 = CPU 核心数）
    // 所有 spawn 的任务分布在这些线程上
}
```

**工作窃取调度器（Work-Stealing Scheduler）**：

```
多线程运行时的工作原理：

线程 1 的任务队列     线程 2 的任务队列     线程 3 的任务队列
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Task A       │    │ Task D       │    │ （空）       │
│ Task B       │    │ Task E       │    │             │
│ Task C       │    │              │    │             │
└─────────────┘    └─────────────┘    └─────────────┘

线程 3 空闲了！ → 从线程 1 或线程 2"偷"一个任务来执行
                   （这就是"工作窃取"）

结果：所有线程都保持忙碌，不会出现一个线程堆满任务、
     另一个线程闲着的情况。
```

**TypeScript 类比**：

```typescript
// JavaScript 只有一个线程
// 相当于 Tokio 的 current-thread 模式
// 如果一个任务计算很久，其他任务全部卡住

// 如果 Node.js 能用 Worker Threads 自动分配任务
// 那就类似 Tokio 的 rt-multi-thread
// （但 Node.js 的 Worker Threads 需要手动管理，不如 Tokio 自动）
```

### 2.2 单线程运行时（current-thread）

```rust
// 使用单线程运行时
#[tokio::main(flavor = "current_thread")]
async fn main() {
    // 只有一个线程，所有任务在这个线程上交替执行
    // 类似 Node.js 的事件循环
}
```

**什么时候用单线程？**

| 场景 | 推荐 | 原因 |
|------|------|------|
| 服务器、网络服务 | **多线程** ✅ | 充分利用多核 CPU |
| 简单 CLI 工具 | 单线程 | 开销小，不需要多线程 |
| WASM 环境 | 单线程 | WASM 不支持多线程 |
| 测试 | 单线程 | 确定性执行，便于调试 |
| 嵌入式 | 单线程 | 资源有限 |
| ZeroClaw | **多线程** ✅ | 同时处理 LLM 调用、工具执行、Channel 监听 |

### 2.3 两者的关键区别

```
                     多线程 (rt-multi-thread)      单线程 (current-thread)
线程数               CPU 核心数（默认）             1
spawn 的任务         可能在任意工作线程执行          只在当前线程执行
Send 约束            spawn 的 Future 必须 Send      spawn_local 不要求 Send
适用场景             服务器、高并发                  CLI、测试、WASM
Cargo feature        "rt-multi-thread"              "rt"
调度策略             工作窃取                        协作调度（FIFO）
```

```rust
// 多线程：spawn 的任务可能跑在任何线程上
// → 所以 Future 必须 Send（能安全跨线程传输）
tokio::spawn(async move {
    // 这段代码可能在线程 1 开始，在线程 3 继续
    // 所以捕获的所有变量必须是 Send 的
    process_data(data).await;
});

// 单线程：所有任务都在同一个线程
// → 可以用 spawn_local，不要求 Send
tokio::task::spawn_local(async {
    // 这段代码永远在当前线程执行
    // 可以使用 Rc 等非 Send 类型
    let rc_data = Rc::new(42);
    use_data(rc_data).await;
});
```

---

## 3. #[tokio::main] 宏

### 3.1 宏展开：它到底做了什么？

`#[tokio::main]` 是一个**属性宏**（attribute macro），它将你的 `async fn main()` 转换为一个普通的 `fn main()`，内部创建 Tokio 运行时并执行你的异步代码。

```rust
// ═══════════════════════════════════════════════════
// 你写的代码
// ═══════════════════════════════════════════════════
#[tokio::main]
async fn main() {
    println!("Hello from async world!");
    let data = fetch_data().await;
    println!("Got: {}", data);
}

// ═══════════════════════════════════════════════════
// 宏展开后的等价代码（编译器实际看到的）
// ═══════════════════════════════════════════════════
fn main() {
    // 1. 创建 Tokio 多线程运行时
    let rt = tokio::runtime::Runtime::new().unwrap();

    // 2. 在运行时中执行异步代码
    //    block_on：阻塞当前线程，直到传入的 Future 完成
    rt.block_on(async {
        println!("Hello from async world!");
        let data = fetch_data().await;
        println!("Got: {}", data);
    });
}
```

**关键点**：`block_on` 是同步世界（`fn main()`）和异步世界（`async { ... }`）之间的**桥梁**。它是唯一允许"阻塞等待 Future"的地方——因为 `main()` 函数必须是同步的。

### 3.2 不同配置的展开

```rust
// ─── 配置 1：默认多线程 ───
#[tokio::main]
async fn main() { /* ... */ }

// 展开为：
fn main() {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async { /* ... */ });
}

// ─── 配置 2：单线程 ───
#[tokio::main(flavor = "current_thread")]
async fn main() { /* ... */ }

// 展开为：
fn main() {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async { /* ... */ });
}

// ─── 配置 3：指定工作线程数 ───
#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() { /* ... */ }

// 展开为：
fn main() {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap()
        .block_on(async { /* ... */ });
}
```

### 3.3 手动创建运行时

有时你需要更多控制，可以手动创建运行时：

```rust
use tokio::runtime::Builder;

fn main() {
    // 完全手动控制
    let runtime = Builder::new_multi_thread()
        .worker_threads(8)            // 8 个工作线程
        .thread_name("zeroclaw")      // 线程名称（调试用）
        .enable_io()                  // 启用 IO 驱动
        .enable_time()                // 启用定时器
        .build()
        .expect("Failed to create Tokio runtime");

    runtime.block_on(async {
        println!("Running with custom runtime!");
        // 你的异步代码
    });
}
```

**什么时候需要手动创建？**
- 需要精确控制线程数
- 需要自定义线程名称（方便日志/调试）
- 在库代码中需要嵌入运行时（库不应该用 `#[tokio::main]`）
- 需要多个运行时实例

### 3.4 TypeScript 对照：没有"启动运行时"的概念

```typescript
// JavaScript: 你从不需要"创建事件循环"
// V8 引擎启动时自动创建，你不能选择、不能配置

// 最接近 #[tokio::main] 的概念是 Node.js 的入口文件
// Node.js 读取你的文件 → 自动启动事件循环 → 执行代码

// 如果 JavaScript 像 Rust 一样需要手动创建运行时：
// （虚构代码，JavaScript 不需要这样做）
import { EventLoop } from "v8-runtime";

const loop = new EventLoop({ threads: 4 });
loop.blockOn(async () => {
    const data = await fetchData();
    console.log(data);
});
```

### 3.5 #[tokio::test] 宏

测试也需要运行时！`#[tokio::test]` 为每个测试函数创建独立的运行时：

```rust
// 普通的同步测试
#[test]
fn test_sync() {
    assert_eq!(2 + 2, 4);
}

// 异步测试——需要 Tokio 运行时
#[tokio::test]
async fn test_async() {
    let result = fetch_data().await;
    assert_eq!(result, "expected");
}

// 等价于：
#[test]
fn test_async() {
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let result = fetch_data().await;
        assert_eq!(result, "expected");
    });
}

// 单线程测试运行时（更确定性）
#[tokio::test(flavor = "current_thread")]
async fn test_deterministic() {
    // ...
}
```

---

## 4. Tokio Feature Flags

### 4.1 ZeroClaw 的 Feature 配置

```toml
# ZeroClaw 的 Cargo.toml
tokio = { version = "1.42", default-features = false, features = [
    "rt-multi-thread",  # 多线程运行时
    "macros",           # #[tokio::main], #[tokio::test]
    "time",             # sleep, interval, timeout
    "net",              # TCP/UDP 异步网络
    "io-util",          # AsyncReadExt, AsyncWriteExt
    "sync",             # mpsc, oneshot, Mutex, RwLock
    "process",          # 异步子进程管理
    "io-std",           # 异步 stdin/stdout
    "fs",               # 异步文件系统操作
    "signal"            # Ctrl+C 信号处理
] }
```

### 4.2 逐个解释每个 Feature

**为什么不用 `default-features = true`？** 因为默认 feature 包含了一些可能不需要的东西。`default-features = false` + 精确选择 = **最小化编译产物 + 最快的编译速度**。

```
Feature 完整解析：

┌────────────────────────────────────────────────────────────────────┐
│ Feature              │ 作用                    │ ZeroClaw 使用场景    │
├────────────────────────────────────────────────────────────────────┤
│ rt-multi-thread      │ 多线程工作窃取运行时      │ Daemon 多组件并行     │
│                      │ （不选就没有运行时！）     │ Agent 并发处理消息    │
├────────────────────────────────────────────────────────────────────┤
│ macros               │ #[tokio::main]          │ main.rs 入口         │
│                      │ #[tokio::test]          │ 异步测试             │
├────────────────────────────────────────────────────────────────────┤
│ time                 │ tokio::time::sleep      │ 指数退避重试          │
│                      │ tokio::time::interval   │ 定时心跳检查          │
│                      │ tokio::time::timeout    │ LLM 调用超时控制      │
├────────────────────────────────────────────────────────────────────┤
│ net                  │ TCP/UDP 异步连接         │ WebSocket 通信        │
│                      │ TcpStream, TcpListener  │ HTTP API 服务         │
├────────────────────────────────────────────────────────────────────┤
│ io-util              │ AsyncReadExt            │ 读取网络流             │
│                      │ AsyncWriteExt           │ 写入网络流             │
│                      │ BufReader, BufWriter     │ 缓冲 IO              │
├────────────────────────────────────────────────────────────────────┤
│ sync                 │ mpsc::channel           │ Channel → Agent 通信  │
│                      │ oneshot::channel         │ 请求-响应模式         │
│                      │ Mutex, RwLock           │ 异步共享状态           │
│                      │ Notify, Semaphore       │ 协调同步原语           │
├────────────────────────────────────────────────────────────────────┤
│ process              │ tokio::process::Command  │ Shell 工具执行         │
│                      │ 异步子进程管理            │ 执行系统命令           │
├────────────────────────────────────────────────────────────────────┤
│ io-std               │ tokio::io::stdin()      │ CLI 模式终端输入       │
│                      │ tokio::io::stdout()     │ 流式输出到终端         │
├────────────────────────────────────────────────────────────────────┤
│ fs                   │ tokio::fs::read         │ 异步读取配置文件       │
│                      │ tokio::fs::write        │ 异步写入状态文件       │
│                      │ tokio::fs::create_dir   │ 创建工作目录           │
├────────────────────────────────────────────────────────────────────┤
│ signal               │ tokio::signal::ctrl_c   │ Daemon 优雅停止       │
│                      │ 捕获系统信号             │ Ctrl+C → 清理 → 退出  │
└────────────────────────────────────────────────────────────────────┘
```

### 4.3 TypeScript 对照：没有 Feature Flag 的概念

```typescript
// JavaScript: 所有功能都是内置的，你不需要选择
// Node.js 的 fs、net、process、signal 都是核心模块

// 如果 Node.js 像 Tokio 一样按需加载：
// （虚构代码）
import { createRuntime } from "node-runtime";

const runtime = createRuntime({
    features: [
        "multi-thread",      // Worker Threads
        "net",               // TCP/UDP
        "fs",                // 文件系统
        "process",           // 子进程
        "signal",            // 信号处理
    ]
});
// 不需要 net？就不加载网络模块 → 更小的二进制文件
```

### 4.4 Feature 组合模式

```toml
# 最小配置——只需要基本 async 能力
tokio = { version = "1", features = ["rt", "macros"] }

# 网络服务——加上网络和 IO
tokio = { version = "1", features = ["rt-multi-thread", "macros", "net", "io-util"] }

# 全功能——什么都要（但编译更慢）
tokio = { version = "1", features = ["full"] }

# ZeroClaw 的选择——精确选择需要的（推荐）
tokio = { version = "1", default-features = false, features = [
    "rt-multi-thread", "macros", "time", "net",
    "io-util", "sync", "process", "io-std", "fs", "signal"
] }
# 注意：ZeroClaw 选了 10 个 feature 中的 10 个
# 基本等同于 "full"，但 default-features = false 意味着
# 如果 Tokio 未来加了新 feature，不会自动引入
```

---

## 5. ZeroClaw 的入口分析

### 5.1 main.rs 的实际代码

```rust
// ZeroClaw 的 main.rs（简化版）
#[tokio::main]
async fn main() -> Result<()> {
    // Step 1: 初始化 TLS 加密提供者
    if let Err(e) = rustls::crypto::ring::default_provider().install_default() {
        eprintln!("Warning: Failed to install default crypto provider: {e:?}");
    }

    // Step 2: 解析命令行参数
    let cli = Cli::parse();

    // Step 3: 根据命令分发
    match cli.command {
        Commands::Run { .. } => {
            // 启动 Agent 交互式会话
            run_agent(config).await?;
        }
        Commands::Daemon { host, port } => {
            // 启动后台 Daemon（多组件并发）
            daemon::run(config, host, port).await?;
        }
        Commands::Chat { message } => {
            // 一次性对话
            one_shot_chat(config, message).await?;
        }
    }

    Ok(())
}
```

### 5.2 执行流程可视化

```
程序启动
│
├─ Rust 编译器看到 #[tokio::main]
│  └─ 宏展开为：
│     fn main() -> Result<()> {
│         tokio::runtime::Builder::new_multi_thread()
│             .enable_all()
│             .build()
│             .unwrap()
│             .block_on(async { ... })     ← 这里开始异步世界
│     }
│
├─ Tokio Runtime 创建
│  ├─ 创建 N 个工作线程（N = CPU 核心数）
│  ├─ 初始化 IO 驱动（epoll/kqueue）
│  ├─ 初始化定时器驱动
│  └─ 初始化任务调度器
│
├─ block_on(async { ... }) 开始执行
│  │
│  ├─ 初始化 TLS（同步操作，在 async 块内也能执行）
│  ├─ 解析 CLI 参数（同步操作）
│  │
│  └─ match 命令分发 → 例如 Daemon 模式：
│     │
│     └─ daemon::run(config, host, port).await
│        │
│        ├─ tokio::spawn(spawn_state_writer(...))      → 后台任务 1
│        ├─ tokio::spawn(spawn_component_supervisor("gateway", ...))  → 后台任务 2
│        ├─ tokio::spawn(spawn_component_supervisor("channels", ...)) → 后台任务 3
│        ├─ tokio::spawn(spawn_component_supervisor("heartbeat", ...))→ 后台任务 4
│        ├─ tokio::spawn(spawn_component_supervisor("scheduler", ...))→ 后台任务 5
│        │
│        ├─ println!("🧠 ZeroClaw daemon started")
│        │
│        ├─ tokio::signal::ctrl_c().await    ← 主任务在这里等待 Ctrl+C
│        │  │                                   同时后台任务 1-5 在工作线程上运行
│        │  │
│        │  └─ 用户按下 Ctrl+C → .await 完成
│        │
│        ├─ for handle in &handles { handle.abort(); }  ← 取消所有后台任务
│        └─ for handle in handles { let _ = handle.await; } ← 等待清理完成
│
└─ block_on 返回 → Runtime 被 drop → 所有资源释放 → 程序退出
```

### 5.3 Runtime 的生命周期

```rust
// Runtime 的生命周期 = 整个程序的运行时间

#[tokio::main]
async fn main() -> Result<()> {
    // ┌──────────────────────────────────────────────┐
    // │  Runtime 已经存在，正在驱动这个 async 块       │
    // │                                               │
    // │  所有 tokio::spawn 的任务都在这个 Runtime 上    │
    // │  所有 .await 都由这个 Runtime 调度             │
    // │                                               │
    // │  当 main 的 async 块完成（或 panic）→           │
    // │    block_on 返回 → Runtime drop               │
    // │    所有未完成的 spawn 任务被取消                │
    // │    所有资源被释放                              │
    // └──────────────────────────────────────────────┘

    daemon::run(config, host, port).await?;

    Ok(())
    // ← 这里 async 块结束，Runtime 被 drop
}
```

**关键理解**：Runtime 被 drop 时，**所有还在运行的 spawn 任务会被取消**。这就是为什么 ZeroClaw 的 Daemon 在 `Ctrl+C` 后要显式 `abort()` 并 `await` 每个 handle——确保优雅关闭。

---

## 6. 常见问题

### 问题 1：能在 async 函数里创建新的 Runtime 吗？

```rust
// ❌ 嵌套 Runtime 会 panic！
#[tokio::main]
async fn main() {
    // 已经在 Tokio Runtime 里了
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        // panic: Cannot start a runtime from within a runtime
        // 不能在运行时里面再创建运行时
    });
}

// ✅ 正确做法：直接 spawn 或 .await
#[tokio::main]
async fn main() {
    // 已经有 Runtime 了，直接用！
    let handle = tokio::spawn(async {
        do_something().await
    });
    handle.await.unwrap();
}
```

### 问题 2：库代码应该用 #[tokio::main] 吗？

```rust
// ❌ 库不应该决定运行时
// 在一个库的 lib.rs 中：
#[tokio::main]
async fn main() { }  // 库没有 main！

// ✅ 库应该暴露 async fn，让用户选择运行时
pub async fn process(data: &str) -> Result<String> {
    // 纯 async 逻辑，不涉及运行时创建
    let result = transform(data).await?;
    Ok(result)
}

// 用户在自己的 main.rs 中选择运行时：
#[tokio::main]
async fn main() {
    let result = my_lib::process("hello").await.unwrap();
}
```

### 问题 3：block_on vs .await 的区别

```rust
// block_on：同步 → 异步的桥梁（阻塞当前线程）
fn sync_function() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(async {
        fetch_data().await  // 当前线程被阻塞，直到完成
    });
}

// .await：异步 → 异步的连接（让出当前线程）
async fn async_function() {
    let result = fetch_data().await;  // 让出线程，去做别的
}
```

| 操作 | block_on | .await |
|------|----------|--------|
| 调用环境 | 同步函数中 | async 函数中 |
| 线程行为 | **阻塞**当前线程 | **让出**当前线程 |
| 用途 | 程序入口点（main） | 正常的异步编程 |
| 嵌套使用 | ❌ 不能在 Runtime 内使用 | ✅ 随便嵌套 |

---

## 7. 小结：Tokio 运行时与 #[tokio::main] 的核心要点

```
┌──────────────────────────────────────────────────────────────┐
│           Tokio 运行时与 #[tokio::main] 核心要点              │
│                                                               │
│  1. Rust 不内置运行时                                         │
│     → 与 JavaScript/Node.js 不同，需要显式选择               │
│     → Tokio 是最流行的选择                                    │
│                                                               │
│  2. 运行时 = 调度器 + IO 驱动 + 定时器                        │
│     → 负责 poll Future、管理 Waker、调度任务                  │
│     → 多线程模式使用工作窃取算法                               │
│                                                               │
│  3. #[tokio::main] 是语法糖                                   │
│     → 展开为：创建 Runtime + block_on(async { ... })          │
│     → 默认多线程，可配置单线程或线程数                         │
│                                                               │
│  4. Feature Flags 按需选择                                    │
│     → ZeroClaw 使用 10 个 feature 覆盖所有需求                │
│     → default-features = false 精确控制                       │
│                                                               │
│  5. Runtime 生命周期 = 程序生命周期                            │
│     → Runtime drop 时所有 spawn 任务被取消                    │
│     → 不能嵌套 Runtime（会 panic）                            │
│                                                               │
│  6. block_on 是同步-异步的唯一桥梁                             │
│     → 只在 main() 中使用                                      │
│     → 与 .await 的区别：阻塞线程 vs 让出线程                  │
└──────────────────────────────────────────────────────────────┘
```

---

> **下一步**: 阅读 `03_核心概念_3_tokio_spawn与任务生成.md`，了解如何用 `tokio::spawn` 生成后台任务，以及 `Send + 'static` 约束的含义。

---

**文件信息**
- 知识点: async/await 与 Tokio
- 维度: 核心概念 2 — Tokio 运行时与 #[tokio::main]
- 版本: v1.0
- 日期: 2026-03-10
