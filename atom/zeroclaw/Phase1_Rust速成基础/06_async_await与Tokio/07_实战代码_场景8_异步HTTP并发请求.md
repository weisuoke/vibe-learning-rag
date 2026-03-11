# async/await 与 Tokio - 实战代码 场景8：异步 HTTP 并发请求

> **知识点**: async/await 与 Tokio
> **层级**: Phase1_Rust速成基础
> **维度**: 实战代码 - 场景8
> **场景**: reqwest + spawn + join!/select! 并发 HTTP 调用
> **目标受众**: 有 TypeScript/前端经验但零 Rust 经验的开发者
> **阅读时间**: 约 25 分钟

---

## 概述

实际开发中，你经常需要并发调用多个 HTTP API——比如同时查询多个 LLM Provider（Anthropic、OpenAI、Gemini），或者批量抓取多个页面。这正是 ZeroClaw 的核心场景：通过 `reqwest` 调用各种 LLM API。

本节把前面学到的 `join!`、`spawn`、`select!` 全部用在真实的 HTTP 请求场景中，从简单到复杂，覆盖 5 个实战场景。

---

## 1. reqwest 异步基础

### Cargo.toml

```toml
[package]
name = "async-http-demo"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.12", features = ["json"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow = "1"
```

### 最简单的异步 HTTP 请求

```rust
use anyhow::Result;

/// 最基础的异步 HTTP GET 请求
/// 对比 TypeScript 的 fetch()
async fn fetch_url(url: &str) -> Result<String> {
    // reqwest::get(url) 返回一个 Future
    // .await 真正发起网络请求
    let response = reqwest::get(url).await?;

    // .text() 也是异步的——需要等待 body 读完
    let body = response.text().await?;

    Ok(body)
}

// TypeScript 对照：
// async function fetchUrl(url: string): Promise<string> {
//     const response = await fetch(url);
//     const body = await response.text();
//     return body;
// }
```

### reqwest::Client 复用连接

```rust
use reqwest::Client;
use std::time::Duration;

/// 生产环境应该复用 Client——它内部维护连接池
/// 类似 TypeScript 中复用 axios 实例
fn create_client() -> Client {
    Client::builder()
        .timeout(Duration::from_secs(10))    // 全局超时
        .connect_timeout(Duration::from_secs(5)) // 连接超时
        .pool_max_idle_per_host(10)          // 每个 host 最多 10 个空闲连接
        .build()
        .expect("Failed to create HTTP client")
}

// TypeScript 对照：
// const client = axios.create({
//     timeout: 10000,
//     // axios 自动管理连接池
// });

async fn fetch_with_client(client: &Client, url: &str) -> Result<String> {
    let response = client.get(url).send().await?;
    let body = response.text().await?;
    Ok(body)
}
```

### 关键理解

```
reqwest 的两种 API：

1. reqwest::get(url)           ← 每次创建新 Client（简单但低效）
2. client.get(url).send()      ← 复用 Client 连接池（生产推荐）

类比 TypeScript：
  1. fetch(url)                 ← 全局 fetch
  2. axiosInstance.get(url)     ← 复用 axios 实例

reqwest::Client 的连接池：
  · Client 内部维护一个 HTTP 连接池
  · 对同一个 host 的请求会复用 TCP 连接（HTTP/2 多路复用）
  · clone() 是廉价的（内部是 Arc）——可以安全传给多个 spawn 任务
  · 生产环境：创建一次，到处 clone 使用
```

---

## 2. 场景1：join! 并发请求多个 URL

同时请求 3 个 API，等全部返回。最常见的场景。

```rust
use std::time::Instant;
use reqwest::Client;
use anyhow::Result;

async fn fetch_api(client: &Client, name: &str, url: &str) -> Result<String> {
    let start = Instant::now();
    let response = client.get(url).send().await?;
    let status = response.status();
    let body = response.text().await?;
    let elapsed = start.elapsed();

    println!("  ✅ [{}] {} → {} ({:?}, {} bytes)",
        name, url, status, elapsed, body.len());

    Ok(body)
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("═══ 场景1：join! 并发请求多个 URL ═══\n");

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;

    // ─── 串行请求（对照） ───
    println!("--- 串行请求 ---");
    let start = Instant::now();
    let _ = fetch_api(&client, "API-1", "https://httpbin.org/delay/1").await?;
    let _ = fetch_api(&client, "API-2", "https://httpbin.org/delay/1").await?;
    let _ = fetch_api(&client, "API-3", "https://httpbin.org/delay/1").await?;
    println!("串行总耗时: {:?}\n", start.elapsed());  // ≈ 3 秒

    // ─── join! 并行请求 ───
    println!("--- join! 并行请求 ---");
    let start = Instant::now();

    // tokio::join! 同时推进三个 Future
    // 类似 Promise.all([fetch1, fetch2, fetch3])
    let (r1, r2, r3) = tokio::join!(
        fetch_api(&client, "API-1", "https://httpbin.org/delay/1"),
        fetch_api(&client, "API-2", "https://httpbin.org/delay/1"),
        fetch_api(&client, "API-3", "https://httpbin.org/delay/1"),
    );

    // 每个结果独立处理错误
    let results: Vec<String> = [r1, r2, r3]
        .into_iter()
        .filter_map(|r| r.ok())
        .collect();

    println!("并行总耗时: {:?}（≈ 1 秒，3 倍加速！）", start.elapsed());
    println!("成功 {}/3 个请求\n", results.len());

    Ok(())
}
```

### 时间线对比

```
串行：
  |---- API-1 (1s) ----|---- API-2 (1s) ----|---- API-3 (1s) ----|
  总耗时 ≈ 3 秒

join! 并行：
  |---- API-1 (1s) ----|
  |---- API-2 (1s) ----|
  |---- API-3 (1s) ----|
  总耗时 ≈ 1 秒 ← max(1s, 1s, 1s)
```

---

## 3. 场景2：spawn 动态数量的并发请求

URL 列表长度在运行时才确定——不能用 `join!`（它需要固定数量的参数），必须用 `spawn` + `Vec<JoinHandle>`。

```rust
use std::time::Instant;
use reqwest::Client;
use tokio::task::JoinHandle;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("═══ 场景2：spawn 动态数量的并发请求 ═══\n");

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;

    // 动态 URL 列表——可能来自数据库、配置文件、用户输入
    let urls = vec![
        "https://httpbin.org/get",
        "https://httpbin.org/ip",
        "https://httpbin.org/user-agent",
        "https://httpbin.org/headers",
        "https://httpbin.org/uuid",
    ];

    println!("并发请求 {} 个 URL...\n", urls.len());
    let start = Instant::now();

    // ─── 为每个 URL spawn 一个独立任务 ───
    // 类似 JS: Promise.all(urls.map(url => fetch(url)))
    let mut handles: Vec<JoinHandle<Result<(String, String)>>> = Vec::new();

    for url in &urls {
        // clone 是廉价的：
        //   client.clone() → 内部是 Arc，只复制指针
        //   url.to_string() → 需要 owned String，因为 spawn 要求 'static
        let client = client.clone();
        let url = url.to_string();

        let handle = tokio::spawn(async move {
            let response = client.get(&url).send().await?;
            let status = response.status().to_string();
            let body = response.text().await?;
            println!("  ✅ {} → {} ({} bytes)", url, status, body.len());
            Ok((url, body))
        });

        handles.push(handle);
    }

    // ─── 收集所有结果 ───
    let mut successes = 0;
    let mut failures = 0;

    for handle in handles {
        match handle.await {
            // 任务正常完成 + 函数返回 Ok
            Ok(Ok((_url, _body))) => {
                successes += 1;
            }
            // 任务正常完成 + 函数返回 Err（HTTP 错误等）
            Ok(Err(e)) => {
                println!("  ❌ 请求失败: {}", e);
                failures += 1;
            }
            // 任务本身崩溃（panic 或被 abort）
            Err(join_err) => {
                println!("  💥 任务异常: {}", join_err);
                failures += 1;
            }
        }
    }

    println!("\n总耗时: {:?}", start.elapsed());
    println!("成功: {}, 失败: {}", successes, failures);

    Ok(())
}
```

### 为什么 spawn 需要 clone？

```
spawn 的 Send + 'static 约束意味着：

  tokio::spawn(async move {
      client.get(&url).send().await    // client 和 url 必须被 move 进来
  });

  · client.clone()：Client 内部是 Arc<ClientInner>，clone 只复制 Arc 指针
    → 所有 spawn 的任务共享同一个连接池，非常高效
    → 类比：多个 React 组件共享同一个 axios 实例

  · url.to_string()：&str 是借用，不满足 'static
    → 必须创建 owned String，move 进闭包
    → 类比：JS 闭包捕获变量（但 Rust 更严格，需要明确 move）
```

---

## 4. 场景3：超时保护

给 HTTP 请求设置超时——两种方式对比。

```rust
use std::time::Duration;
use reqwest::Client;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("═══ 场景3：超时保护 ═══\n");

    // ─── 方式1：reqwest 内置 timeout（推荐） ───
    println!("--- 方式1：reqwest 内置 timeout ---\n");

    let client = Client::builder()
        .timeout(Duration::from_secs(3))  // 全局超时：3 秒
        .build()?;

    // 正常请求（1 秒延迟 < 3 秒超时）
    match client.get("https://httpbin.org/delay/1").send().await {
        Ok(resp) => println!("  ✅ 正常请求成功: {}", resp.status()),
        Err(e) => println!("  ❌ 请求失败: {}", e),
    }

    // 超时请求（5 秒延迟 > 3 秒超时）
    match client.get("https://httpbin.org/delay/5").send().await {
        Ok(resp) => println!("  ✅ 慢请求成功: {}", resp.status()),
        Err(e) if e.is_timeout() => {
            println!("  ⏰ 请求超时！（超过 3 秒）");
        }
        Err(e) => println!("  ❌ 其他错误: {}", e),
    }

    // 也可以单个请求设置超时（覆盖全局设置）：
    // client.get(url).timeout(Duration::from_secs(5)).send().await?;

    // ─── 方式2：tokio::time::timeout 包裹（更灵活） ───
    println!("\n--- 方式2：tokio::time::timeout ---\n");

    let client = Client::new();  // 不设全局超时

    // tokio::time::timeout 可以包裹任何 Future
    // 不仅限于 HTTP 请求——数据库查询、LLM 调用都能用
    match tokio::time::timeout(
        Duration::from_secs(3),
        client.get("https://httpbin.org/delay/5").send(),
    ).await {
        Ok(Ok(resp)) => println!("  ✅ 请求成功: {}", resp.status()),
        Ok(Err(e))   => println!("  ❌ HTTP 错误: {}", e),
        Err(_)       => println!("  ⏰ tokio 超时！（超过 3 秒）"),
    }

    // ─── 方式3：select! 手动超时（最灵活） ───
    println!("\n--- 方式3：select! 手动超时 ---\n");

    let client = Client::new();

    tokio::select! {
        result = client.get("https://httpbin.org/delay/5").send() => {
            match result {
                Ok(resp) => println!("  ✅ 请求成功: {}", resp.status()),
                Err(e) => println!("  ❌ HTTP 错误: {}", e),
            }
        }
        _ = tokio::time::sleep(Duration::from_secs(3)) => {
            // select! 会自动 drop HTTP 请求的 Future
            // → 底层 TCP 连接被关闭
            // → 比 JS 的 AbortController 更彻底
            println!("  ⏰ select! 超时！请求已被取消");
        }
    }

    Ok(())
}
```

### 三种超时方式对比

```
┌─────────────────────┬──────────────────────────┬──────────────────────────┐
│ 方式                 │ 优点                      │ 适用场景                  │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ reqwest timeout     │ 简单，HTTP 专用           │ 只需要 HTTP 超时          │
│ (client/请求级)      │ 自动处理连接级细节        │ 大多数场景                │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ tokio::time::timeout│ 通用，可包裹任何 Future   │ 数据库、LLM 调用等        │
│                     │ 三层 Result 需要处理      │ 需要统一的超时策略         │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ tokio::select!      │ 最灵活，可同时等多个条件  │ 超时 + 取消信号同时监听   │
│                     │ 自动取消未选中的分支      │ Agent 循环、优雅关闭       │
└─────────────────────┴──────────────────────────┴──────────────────────────┘

推荐选择：
  · 单纯 HTTP 超时 → reqwest 内置 timeout
  · 通用异步超时 → tokio::time::timeout
  · 复杂控制流（超时 + 取消 + 多条件） → select!
```

---

## 5. 场景4：并发请求 + 错误处理

部分请求失败不影响其他请求——类似 `Promise.allSettled()`。

```rust
use std::time::Duration;
use reqwest::Client;
use tokio::task::JoinHandle;
use anyhow::Result;

/// 单个请求的结果（成功 or 失败）
#[derive(Debug)]
enum FetchResult {
    Success { url: String, body: String, elapsed_ms: u64 },
    Failed { url: String, error: String },
}

async fn fetch_one(client: Client, url: String) -> FetchResult {
    let start = std::time::Instant::now();

    match client.get(&url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
    {
        Ok(resp) => {
            match resp.text().await {
                Ok(body) => FetchResult::Success {
                    url,
                    elapsed_ms: start.elapsed().as_millis() as u64,
                    body,
                },
                Err(e) => FetchResult::Failed {
                    url,
                    error: format!("读取 body 失败: {}", e),
                },
            }
        }
        Err(e) => {
            let error = if e.is_timeout() {
                "请求超时".to_string()
            } else if e.is_connect() {
                "连接失败".to_string()
            } else {
                format!("请求错误: {}", e)
            };
            FetchResult::Failed { url, error }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("═══ 场景4：并发请求 + 错误处理 ═══\n");

    let client = Client::new();

    // 混合 URL——有些会成功，有些会失败
    let urls = vec![
        "https://httpbin.org/get",                    // ✅ 正常
        "https://httpbin.org/status/404",             // ✅ 返回 404（但请求成功）
        "https://httpbin.org/delay/10",               // ❌ 会超时（> 5s timeout）
        "https://httpbin.org/uuid",                   // ✅ 正常
        "https://this-domain-does-not-exist.invalid", // ❌ DNS 解析失败
    ];

    println!("并发请求 {} 个 URL（部分会失败）...\n", urls.len());

    // ─── spawn 所有请求 ───
    let handles: Vec<JoinHandle<FetchResult>> = urls
        .into_iter()
        .map(|url| {
            let client = client.clone();
            tokio::spawn(fetch_one(client, url.to_string()))
        })
        .collect();

    // ─── 收集结果（类似 Promise.allSettled） ───
    let mut successes: Vec<FetchResult> = Vec::new();
    let mut failures: Vec<FetchResult> = Vec::new();

    for handle in handles {
        match handle.await {
            Ok(result) => {
                match &result {
                    FetchResult::Success { url, elapsed_ms, body } => {
                        println!("  ✅ {} → {} bytes ({} ms)", url, body.len(), elapsed_ms);
                        successes.push(result);
                    }
                    FetchResult::Failed { url, error } => {
                        println!("  ❌ {} → {}", url, error);
                        failures.push(result);
                    }
                }
            }
            Err(join_err) => {
                println!("  💥 任务 panic: {}", join_err);
            }
        }
    }

    println!("\n📊 结果汇总：");
    println!("  成功: {} / 失败: {}", successes.len(), failures.len());

    Ok(())
}
```

### 错误分类

```
reqwest 的错误类型判断：

e.is_timeout()   → 请求超时（对端太慢或网络卡）
e.is_connect()   → 连接失败（DNS 解析失败、端口不通）
e.is_request()   → 请求构造错误（URL 格式错误等）
e.is_body()      → 读取 body 失败（传输中断）
e.is_decode()    → JSON 反序列化失败
e.is_redirect()  → 重定向次数超限
e.is_status()    → HTTP 状态码错误（需要调用 error_for_status()）

注意：HTTP 4xx/5xx 状态码默认不算 Error！
  · response.status() == 404  → 请求"成功"了，只是服务器返回 404
  · 如果你想把 4xx/5xx 当作错误：response.error_for_status()?
```

---

## 6. 场景5：限制并发数（Semaphore）

如果有 100 个 URL 全部 spawn，可能导致：目标服务器被打崩、自己内存爆炸、达到操作系统文件描述符上限。需要限制并发数。

```rust
use std::sync::Arc;
use std::time::{Duration, Instant};
use reqwest::Client;
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("═══ 场景5：Semaphore 限制并发数 ═══\n");

    let client = Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;

    // Semaphore：信号量，控制同时运行的任务数
    // 类比：餐厅只有 3 张桌子，最多同时接待 3 桌客人
    let semaphore = Arc::new(Semaphore::new(3)); // 最多 3 个并发请求

    // 模拟 10 个 URL 需要请求
    let urls: Vec<String> = (1..=10)
        .map(|i| format!("https://httpbin.org/delay/1?id={}", i))
        .collect();

    println!("共 {} 个 URL，最多 {} 个并发\n", urls.len(), 3);
    let start = Instant::now();

    // ─── spawn 所有任务（但 Semaphore 控制同时运行的数量） ───
    let mut handles: Vec<JoinHandle<Result<String>>> = Vec::new();

    for (i, url) in urls.into_iter().enumerate() {
        let client = client.clone();
        let semaphore = semaphore.clone();

        let handle = tokio::spawn(async move {
            // 获取 permit：如果已经有 3 个任务在运行，这里会等待
            // 类比：等位——餐厅满了就在门口排队
            let _permit = semaphore.acquire().await
                .expect("Semaphore closed");

            // _permit 存在期间，占用一个"名额"
            // 当 _permit 被 drop（离开作用域），名额自动释放
            println!("  🚀 [{}] 开始请求 (t={:?})", i + 1, start.elapsed());

            let response = client.get(&url).send().await?;
            let body = response.text().await?;

            println!("  ✅ [{}] 完成 ({} bytes, t={:?})",
                i + 1, body.len(), start.elapsed());

            Ok(body)
            // ← _permit 在这里被 drop → 释放一个名额 → 等待中的任务可以开始
        });

        handles.push(handle);
    }

    // ─── 等待所有任务完成 ───
    let mut success_count = 0;
    for handle in handles {
        if let Ok(Ok(_)) = handle.await {
            success_count += 1;
        }
    }

    println!("\n总耗时: {:?}", start.elapsed());
    println!("成功: {}/10", success_count);
    println!("理论最短耗时: ceil(10/3) × 1s ≈ 4s\n");

    Ok(())
}
```

### Semaphore 原理图解

```
Semaphore::new(3) — 3 个 permit

时间线：

t=0s:   任务1 获取 permit ✅  |---- 请求 (1s) ----|
        任务2 获取 permit ✅  |---- 请求 (1s) ----|
        任务3 获取 permit ✅  |---- 请求 (1s) ----|
        任务4 等待...  ⏳
        任务5 等待...  ⏳
        ...

t=1s:   任务1 完成，释放 permit
        任务2 完成，释放 permit
        任务3 完成，释放 permit
        任务4 获取 permit ✅  |---- 请求 (1s) ----|
        任务5 获取 permit ✅  |---- 请求 (1s) ----|
        任务6 获取 permit ✅  |---- 请求 (1s) ----|
        任务7 等待...  ⏳
        ...

t=2s:   任务4-6 完成，释放
        任务7-9 获取 permit ✅
        任务10 等待...  ⏳

t=3s:   任务7-9 完成，释放
        任务10 获取 permit ✅

t=4s:   全部完成 ✅

总耗时 ≈ ceil(10/3) × 1s = 4s
```

### Semaphore vs 前端方案对比

```
Rust Semaphore:
  let sem = Arc::new(Semaphore::new(5));
  let _permit = sem.acquire().await;  // 等待获取
  // ... 工作 ...
  // _permit drop → 自动释放

TypeScript 手动实现（没有原生 Semaphore）：
  // 方案1：p-limit 库
  import pLimit from 'p-limit';
  const limit = pLimit(5);
  await Promise.all(urls.map(url => limit(() => fetch(url))));

  // 方案2：手动分批
  for (let i = 0; i < urls.length; i += 5) {
      const batch = urls.slice(i, i + 5);
      await Promise.all(batch.map(url => fetch(url)));
  }

Rust 的优势：
  · Semaphore 是原生异步原语，与 Tokio 深度集成
  · RAII：permit drop 自动释放，不会忘记释放
  · 精细控制：每个任务独立获取/释放，不需要分批
```

---

## 7. 完整综合示例：多 LLM Provider 并发调用

模拟 ZeroClaw 同时调用多个 LLM Provider 的场景——带超时、重试、并发限制、错误降级。

```rust
use std::sync::Arc;
use std::time::{Duration, Instant};
use reqwest::Client;
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use anyhow::{Result, bail};

// ═══════════════════════════════════════════════════════════
// 数据结构定义
// ═══════════════════════════════════════════════════════════

/// LLM Provider 配置
#[derive(Clone, Debug)]
struct ProviderConfig {
    name: String,
    base_url: String,
    timeout: Duration,
    max_retries: u32,
}

/// Provider 调用结果
#[derive(Debug)]
struct ProviderResult {
    provider: String,
    response: String,
    latency_ms: u64,
    retries_used: u32,
}

/// 调用失败信息
#[derive(Debug)]
struct ProviderError {
    provider: String,
    error: String,
    retries_used: u32,
}

// ═══════════════════════════════════════════════════════════
// 核心逻辑
// ═══════════════════════════════════════════════════════════

/// 带重试的单个 Provider 调用
async fn call_provider_with_retry(
    client: &Client,
    config: &ProviderConfig,
) -> std::result::Result<ProviderResult, ProviderError> {
    let mut last_error = String::new();

    for attempt in 0..=config.max_retries {
        if attempt > 0 {
            // 指数退避：100ms, 200ms, 400ms...
            let backoff = Duration::from_millis(100 * 2u64.pow(attempt - 1));
            println!("    🔄 [{}] 重试 #{} (等待 {:?})", config.name, attempt, backoff);
            tokio::time::sleep(backoff).await;
        }

        let start = Instant::now();

        // 带超时的 HTTP 请求
        let result = tokio::time::timeout(
            config.timeout,
            client.get(&config.base_url).send(),
        ).await;

        match result {
            // 超时
            Err(_) => {
                last_error = format!("超时 ({:?})", config.timeout);
                println!("    ⏰ [{}] 第 {} 次尝试超时", config.name, attempt + 1);
            }
            // HTTP 错误
            Ok(Err(e)) => {
                last_error = format!("HTTP 错误: {}", e);
                println!("    ❌ [{}] 第 {} 次尝试失败: {}", config.name, attempt + 1, e);
            }
            // HTTP 成功
            Ok(Ok(response)) => {
                let status = response.status();
                if status.is_server_error() {
                    // 5xx 错误可以重试
                    last_error = format!("服务器错误: {}", status);
                    println!("    ⚠️  [{}] 第 {} 次得到 {}，将重试",
                        config.name, attempt + 1, status);
                    continue;
                }

                // 读取 body
                match response.text().await {
                    Ok(body) => {
                        return Ok(ProviderResult {
                            provider: config.name.clone(),
                            response: body,
                            latency_ms: start.elapsed().as_millis() as u64,
                            retries_used: attempt,
                        });
                    }
                    Err(e) => {
                        last_error = format!("读取 body 失败: {}", e);
                    }
                }
            }
        }
    }

    Err(ProviderError {
        provider: config.name.clone(),
        error: last_error,
        retries_used: config.max_retries,
    })
}

/// 并发调用多个 Provider（带并发限制）
async fn call_multiple_providers(
    client: &Client,
    providers: Vec<ProviderConfig>,
    max_concurrent: usize,
    total_timeout: Duration,
) -> (Vec<ProviderResult>, Vec<ProviderError>) {
    let semaphore = Arc::new(Semaphore::new(max_concurrent));
    let client = client.clone();

    // 为每个 Provider spawn 一个任务
    let handles: Vec<JoinHandle<std::result::Result<ProviderResult, ProviderError>>> = providers
        .into_iter()
        .map(|config| {
            let client = client.clone();
            let semaphore = semaphore.clone();

            tokio::spawn(async move {
                let _permit = semaphore.acquire().await
                    .expect("Semaphore closed");

                println!("  🚀 [{}] 开始调用...", config.name);
                call_provider_with_retry(&client, &config).await
            })
        })
        .collect();

    // 等待所有结果（带总超时）
    let mut successes = Vec::new();
    let mut failures = Vec::new();

    // 总超时包裹所有请求
    let collect_all = async {
        for handle in handles {
            match handle.await {
                Ok(Ok(result)) => successes.push(result),
                Ok(Err(error)) => failures.push(error),
                Err(join_err) => {
                    failures.push(ProviderError {
                        provider: "unknown".to_string(),
                        error: format!("任务 panic: {}", join_err),
                        retries_used: 0,
                    });
                }
            }
        }
    };

    match tokio::time::timeout(total_timeout, collect_all).await {
        Ok(()) => {}
        Err(_) => {
            println!("\n  ⏰ 总超时 ({:?}) 到达，部分请求可能被取消", total_timeout);
        }
    }

    (successes, failures)
}

// ═══════════════════════════════════════════════════════════
// 主函数
// ═══════════════════════════════════════════════════════════

#[tokio::main]
async fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  多 LLM Provider 并发调用（模拟 ZeroClaw Provider 调用）");
    println!("═══════════════════════════════════════════════════════════\n");

    let client = Client::builder()
        .pool_max_idle_per_host(10)
        .build()?;

    // 模拟多个 LLM Provider 配置
    // 实际项目中这些 URL 会指向 Anthropic/OpenAI/Gemini API
    let providers = vec![
        ProviderConfig {
            name: "Anthropic".to_string(),
            base_url: "https://httpbin.org/delay/1".to_string(), // 模拟 1s 延迟
            timeout: Duration::from_secs(5),
            max_retries: 2,
        },
        ProviderConfig {
            name: "OpenAI".to_string(),
            base_url: "https://httpbin.org/get".to_string(),     // 快速返回
            timeout: Duration::from_secs(5),
            max_retries: 2,
        },
        ProviderConfig {
            name: "Gemini".to_string(),
            base_url: "https://httpbin.org/delay/2".to_string(), // 模拟 2s 延迟
            timeout: Duration::from_secs(5),
            max_retries: 2,
        },
        ProviderConfig {
            name: "Ollama-Local".to_string(),
            base_url: "https://httpbin.org/uuid".to_string(),    // 快速返回
            timeout: Duration::from_secs(3),
            max_retries: 1,
        },
        ProviderConfig {
            name: "SlowProvider".to_string(),
            base_url: "https://httpbin.org/delay/8".to_string(), // 模拟超时
            timeout: Duration::from_secs(3),
            max_retries: 1,
        },
    ];

    let start = Instant::now();

    let (successes, failures) = call_multiple_providers(
        &client,
        providers,
        3,                          // 最多 3 个并发
        Duration::from_secs(15),    // 总超时 15 秒
    ).await;

    // ─── 结果汇总 ───
    println!("\n╔═══════════════════════════════════════════╗");
    println!("║  📊 调用结果汇总                           ║");
    println!("╚═══════════════════════════════════════════╝\n");

    println!("成功 ({}):", successes.len());
    for result in &successes {
        println!("  ✅ {} → {} ms, {} bytes, {} 次重试",
            result.provider,
            result.latency_ms,
            result.response.len(),
            result.retries_used);
    }

    if !failures.is_empty() {
        println!("\n失败 ({}):", failures.len());
        for error in &failures {
            println!("  ❌ {} → {} ({} 次重试)",
                error.provider, error.error, error.retries_used);
        }
    }

    println!("\n总耗时: {:?}", start.elapsed());
    println!("成功率: {}/{}", successes.len(), successes.len() + failures.len());

    // ─── 选择最快的成功结果 ───
    if let Some(fastest) = successes.iter().min_by_key(|r| r.latency_ms) {
        println!("\n🏆 最快 Provider: {} ({} ms)", fastest.provider, fastest.latency_ms);
    }

    Ok(())
}
```

### 运行效果预览

```bash
$ cargo run
═══════════════════════════════════════════════════════════
  多 LLM Provider 并发调用（模拟 ZeroClaw Provider 调用）
═══════════════════════════════════════════════════════════

  🚀 [Anthropic] 开始调用...
  🚀 [OpenAI] 开始调用...
  🚀 [Gemini] 开始调用...
                                          ← Ollama 和 SlowProvider 等待 Semaphore
  ✅ [OpenAI] 完成
  🚀 [Ollama-Local] 开始调用...           ← OpenAI 完成，释放 permit
  ✅ [Anthropic] 完成
  🚀 [SlowProvider] 开始调用...           ← Anthropic 完成，释放 permit
  ✅ [Ollama-Local] 完成
  ✅ [Gemini] 完成
    ⏰ [SlowProvider] 第 1 次尝试超时
    🔄 [SlowProvider] 重试 #1 (等待 100ms)
    ⏰ [SlowProvider] 第 2 次尝试超时

╔═══════════════════════════════════════════╗
║  📊 调用结果汇总                           ║
╚═══════════════════════════════════════════╝

成功 (4):
  ✅ OpenAI → 150 ms, 1234 bytes, 0 次重试
  ✅ Anthropic → 1050 ms, 567 bytes, 0 次重试
  ✅ Gemini → 2100 ms, 890 bytes, 0 次重试
  ✅ Ollama-Local → 80 ms, 234 bytes, 0 次重试

失败 (1):
  ❌ SlowProvider → 超时 (3s) (1 次重试)

总耗时: 8.2s
成功率: 4/5

🏆 最快 Provider: Ollama-Local (80 ms)
```

### 架构图

```
                     main()
                       │
                       ▼
             call_multiple_providers()
                       │
                       ▼
              Semaphore::new(3)       ← 限制并发
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   spawn(Anthropic) spawn(OpenAI)  spawn(Gemini)
        │              │              │
        ▼              ▼              ▼
   call_provider    call_provider   call_provider
   _with_retry()    _with_retry()   _with_retry()
        │              │              │
   ┌────▼────┐    ┌────▼────┐   ┌────▼────┐
   │ attempt 0│    │ attempt 0│   │ attempt 0│
   │ timeout  │    │ ✅ 成功   │   │ timeout  │
   │ wrapper  │    └──────────┘   │ wrapper  │
   │          │                   │          │
   │ attempt 1│         等待 permit 释放...   │ attempt 1│
   │ ✅ 成功   │                   │ ✅ 成功   │
   └──────────┘                   └──────────┘

   Ollama-Local 和 SlowProvider 在 Semaphore 排队
   等前面的完成后获取 permit 开始执行
```

---

## 8. TypeScript 对照

```typescript
// ═══ TypeScript 等价实现 ═══

// ─── 场景1：Promise.all = join! ───
const [r1, r2, r3] = await Promise.all([
    fetch("https://api.example.com/users"),
    fetch("https://api.example.com/posts"),
    fetch("https://api.example.com/comments"),
]);

// ─── 场景2：Promise.all + map = spawn + Vec<JoinHandle> ───
const urls = ["url1", "url2", "url3", "url4", "url5"];
const results = await Promise.all(
    urls.map(url => fetch(url).then(r => r.text()))
);

// ─── 场景3：超时 = AbortController + setTimeout ───
const controller = new AbortController();
const timeout = setTimeout(() => controller.abort(), 3000);
try {
    const response = await fetch(url, { signal: controller.signal });
    clearTimeout(timeout);
} catch (e) {
    if (e.name === 'AbortError') console.log('超时！');
}
// Rust 对比：tokio::time::timeout(3s, reqwest::get(url)).await

// ─── 场景4：Promise.allSettled = spawn + 逐个处理 ───
const settled = await Promise.allSettled(
    urls.map(url => fetch(url))
);
const successes = settled.filter(r => r.status === 'fulfilled');
const failures = settled.filter(r => r.status === 'rejected');
// Rust 对比：handle.await 的 Ok/Err 分支

// ─── 场景5：并发限制 = p-limit ───
import pLimit from 'p-limit';
const limit = pLimit(3);  // 最多 3 个并发
const results2 = await Promise.all(
    urls.map(url => limit(() => fetch(url)))
);
// Rust 对比：Arc<Semaphore::new(3)>

// ─── 完整示例对照 ───
async function callProviderWithRetry(
    config: ProviderConfig
): Promise<ProviderResult> {
    for (let attempt = 0; attempt <= config.maxRetries; attempt++) {
        if (attempt > 0) {
            await new Promise(r =>
                setTimeout(r, 100 * Math.pow(2, attempt - 1))
            );
        }
        const controller = new AbortController();
        const timeout = setTimeout(
            () => controller.abort(),
            config.timeoutMs
        );
        try {
            const resp = await fetch(config.url, {
                signal: controller.signal
            });
            clearTimeout(timeout);
            if (resp.status >= 500) continue; // 5xx 重试
            const body = await resp.text();
            return { provider: config.name, response: body };
        } catch (e) {
            clearTimeout(timeout);
            if (attempt === config.maxRetries) throw e;
        }
    }
    throw new Error('All retries exhausted');
}
```

### 关键差异总结

```
┌──────────────────┬──────────────────────────┬──────────────────────────┐
│ 功能              │ Rust (reqwest + Tokio)    │ TypeScript (fetch)       │
├──────────────────┼──────────────────────────┼──────────────────────────┤
│ HTTP 客户端       │ reqwest::Client          │ fetch() / axios          │
│ 连接池            │ Client 内置（自动复用）   │ 浏览器/Node 自动管理     │
│ 并发全部等待      │ join! / spawn + collect   │ Promise.all()            │
│ 部分失败不影响    │ 逐个 match Ok/Err        │ Promise.allSettled()     │
│ 超时              │ timeout() / select!      │ AbortController          │
│ 并发限制          │ Semaphore（原生）         │ p-limit（第三方）        │
│ 重试              │ 手动循环 + 指数退避       │ 手动或 axios-retry      │
│ 取消              │ drop Future（自动释放）   │ AbortController（手动） │
│ 错误类型          │ 编译时 Result 强制处理    │ 运行时 try/catch        │
│ 线程安全          │ Send + 'static 编译检查  │ 单线程无需考虑           │
└──────────────────┴──────────────────────────┴──────────────────────────┘

最大优势：Rust 的 reqwest + Tokio 组合在并发 HTTP 场景下
  · 内存使用更低（没有 GC 开销）
  · 取消更彻底（drop Future → TCP 连接立即关闭）
  · 并发控制更精细（Semaphore 原生支持）
  · 错误处理更安全（编译器强制你处理每种错误情况）
```

---

## 9. 与 ZeroClaw 的关联

### ZeroClaw 中的 HTTP 请求

```
ZeroClaw 的 Provider 调用就是 HTTP 请求：

Provider Trait (async_trait):
  async fn chat(&self, messages: Vec<Message>) -> Result<Response>
  async fn chat_stream(&self, messages: Vec<Message>) -> Result<BoxStream<Response>>

底层实现：
  · AnthropicProvider → reqwest::Client → https://api.anthropic.com/v1/messages
  · OpenAIProvider    → reqwest::Client → https://api.openai.com/v1/chat/completions
  · OllamaProvider    → reqwest::Client → http://localhost:11434/api/chat

关键设计：
  · Client 在 Provider 创建时初始化，之后复用
  · 每个 Provider 调用通过 reqwest 发起 HTTP POST
  · 超时通过 reqwest timeout + tokio timeout 双重保护
  · 错误通过 Result + 重试 + 降级处理
```

### 本节技术在 ZeroClaw 中的映射

```
本节技术               →  ZeroClaw 实际应用
──────────────────────────────────────────────────────────────
reqwest::Client        →  Provider 内部的 HTTP 客户端
join! 并发请求          →  同时查询多个知识源（RAG 场景）
spawn + Semaphore      →  控制 Provider 并发调用数
timeout 超时保护        →  LLM API 调用超时处理
retry 重试机制          →  spawn_component_supervisor 指数退避
错误处理 + 降级        →  Provider 失败后切换备用 Provider
```

---

## 10. 速查表

```
reqwest 基础：
  reqwest::get(url).await?                   ← 简单 GET（每次新建 Client）
  Client::new()                              ← 创建可复用 Client
  client.get(url).send().await?              ← 复用连接池的 GET
  client.post(url).json(&body).send().await? ← POST JSON
  response.text().await?                     ← 读取 body 为 String
  response.json::<T>().await?                ← 反序列化为类型 T

并发请求：
  join!(a, b, c)                             ← 固定数量并发
  spawn + Vec<JoinHandle>                    ← 动态数量并发
  Semaphore::new(n)                          ← 限制最大并发数
  sem.acquire().await                        ← 获取 permit（会等待）

超时保护：
  Client::builder().timeout(dur)             ← reqwest 全局超时
  client.get(url).timeout(dur)               ← 单请求超时
  tokio::time::timeout(dur, future)          ← Tokio 通用超时
  tokio::select! { ... sleep(dur) => ... }   ← select! 手动超时

错误判断：
  e.is_timeout()                             ← 超时？
  e.is_connect()                             ← 连接失败？
  response.error_for_status()?               ← 4xx/5xx 转 Err
  response.status().is_server_error()        ← 5xx？可重试

Client clone：
  client.clone()                             ← 廉价（Arc 指针复制）
  所有 spawn 任务共享同一个连接池
```

### 一句话总结

> **reqwest + Tokio = Rust 的 fetch + Promise.all**——用 `reqwest::Client` 发起异步 HTTP 请求，用 `join!` 固定数量并发、`spawn` 动态数量并发、`Semaphore` 限制并发数、`timeout` 保护超时、重试循环处理错误。Client 的 `clone()` 是廉价的（内部 Arc），所有 spawn 的任务共享同一个连接池。ZeroClaw 的每一次 LLM API 调用，底层都是 reqwest 的异步 HTTP 请求。
