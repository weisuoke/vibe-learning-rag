# 实战代码 · 场景 3：源码构建与 Feature 定制

> **场景目标：** 从源码克隆、构建 ZeroClaw，使用 Feature Flags 定制功能组合，理解 Debug/Release/Release-Fast 三种 Profile 的差异，并完成一次交叉编译，最终得到可部署的定制化二进制。

---

## 场景概述

### 你将完成什么？

```
克隆源码 → 基础构建 → Feature 定制 → Profile 对比 → 交叉编译 → 开发工作流
 (1分钟)   (5-40分钟)  (5-40分钟)    (10分钟)      (10-20分钟)  (5分钟)
```

### 何时需要源码构建？（决策指南）

```
你的场景是什么？
│
├── 只想用 ZeroClaw？
│   └── cargo install zeroclaw-cli 或 brew install zeroclaw → 不需要源码构建
│
├── 需要启用特定 Feature（如 hardware / channel-matrix）？
│   └── 预编译版不含可选 Feature → ✅ 需要源码构建
│
├── 想魔改 ZeroClaw 源码 / 提 PR？
│   └── ✅ 需要源码构建
│
├── 想在 CI/CD 中自定义构建？
│   └── ✅ 需要源码构建
│
└── 想交叉编译到树莓派等 ARM 设备？
    └── 预编译版够用？→ 不需要
    └── 需要带自定义 Feature？→ ✅ 需要源码构建
```

> **前端类比：** 就像大多数人 `npm install next` 即可，但如果你想改 Next.js 内核或开启实验性功能，就需要 `git clone` + 本地构建。

### 前置条件

| 条件 | 说明 |
|------|------|
| Rust 工具链已安装 | `rustc --version` 输出 ≥ 1.87（完成场景 1 步骤 3） |
| Git 已安装 | `git --version` 可用 |
| 磁盘空间 ≥ 6 GB | 源码 + 编译缓存（`target/` 目录可能很大） |
| 内存 ≥ 4 GB | Debug 构建最低要求；Release 构建推荐 8 GB+ |

---

## 场景 3.1：基础源码构建

> **目标：** 克隆 ZeroClaw 源码，分别用 Debug 和 Release 模式构建，理解两者差异。

### 步骤 1：克隆源码

```bash
git clone https://github.com/zeroclaw-labs/zeroclaw.git
cd zeroclaw
```

**预期输出：**

```
Cloning into 'zeroclaw'...
remote: Enumerating objects: 3847, done.
remote: Counting objects: 100% (3847/3847), done.
Receiving objects: 100% (3847/3847), 2.1 MiB | 5.42 MiB/s, done.
Resolving deltas: 100% (2631/2631), done.
```

> 进入目录后，`rust-toolchain.toml` 会自动生效——rustup 检测到后会自动下载指定版本（1.92.0）的工具链。

```bash
# 验证 Rust 版本已自动切换
rustc --version
# 预期: rustc 1.92.0 (xxxxxxxx 2026-xx-xx)
```

### 步骤 2：Debug 构建（快速，用于开发）

```bash
cargo build
```

**预期输出：**

```
   Compiling autocfg v1.4.0
   Compiling proc-macro2 v1.0.92
   ...（约 167 个 crate）
   Compiling zeroclaw v0.1.7 (/path/to/zeroclaw)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 2m 15s
```

```bash
# 查看产物
ls -lh target/debug/zeroclaw
# -rwxr-xr-x 1 user user 32M  Mar 11 10:00 target/debug/zeroclaw
```

> **注意 Debug 产物大小：~32 MB！** 因为包含完整调试符号、未优化代码。这就像 `webpack mode: development` 产出未压缩的 JS bundle。

### 步骤 3：Release 构建（慢速，用于部署）

```bash
cargo build --release
```

**预期输出：**

```
   Compiling autocfg v1.4.0
   ...（约 167 个 crate）
   Compiling zeroclaw v0.1.7 (/path/to/zeroclaw)
    Finished `release` profile [optimized] target(s) in 15m 42s
```

```bash
# 查看产物
ls -lh target/release/zeroclaw
# -rwxr-xr-x 1 user user 8.8M  Mar 11 10:20 target/release/zeroclaw
```

### Debug vs Release 对比

| 维度 | Debug (`cargo build`) | Release (`cargo build --release`) |
|------|----------------------|----------------------------------|
| **产物大小** | ~32 MB | ~8.8 MB |
| **编译时间** | ~2 分钟 | ~15 分钟 |
| **运行速度** | 较慢（未优化） | 较快（深度优化） |
| **调试符号** | ✅ 包含 | ❌ 去除（`strip = true`） |
| **用途** | 开发、调试 | 测试、部署、分发 |
| **前端类比** | `npm run dev` | `npm run build` |

> **前端类比：** Debug vs Release = `webpack mode: development` vs `webpack mode: production`。
> - development：不压缩、保留 source map、HMR 快速刷新
> - production：代码压缩、tree-shaking、去除 console.log

[来源: sourcecode/zeroclaw/Cargo.toml — `[profile.release]` 配置段]

---

## 场景 3.2：Feature Flags 定制构建

> **目标：** 按需启用可选 Feature，构建定制化 ZeroClaw。

### 回顾：默认构建 = 最小构建

```bash
# 默认不启用任何可选 Feature
cargo build --release
# 产物只包含核心功能：CLI + Gateway + Agent + SQLite
```

> **ZeroClaw 哲学：** `default = []`（默认全空）。就像餐厅基础套餐只含主食，配菜自选。

### 逐个 Feature 构建示例

#### Feature 1：hardware（USB + 串口）

```bash
cargo build --release --features hardware
```

**预期输出尾部：**

```
   Compiling nusb v0.1.12
   Compiling tokio-serial v5.4.4
   Compiling zeroclaw v0.1.7
    Finished `release` profile [optimized] target(s) in 16m 05s
```

```bash
ls -lh target/release/zeroclaw
# ~9.3 MB  （+0.5 MB）
```

#### Feature 2：channel-matrix（Matrix 通道）

```bash
cargo build --release --features channel-matrix
```

**预期输出尾部：**

```
   Compiling matrix-sdk v0.8.0
   ...（matrix-sdk 有较多子依赖）
   Compiling zeroclaw v0.1.7
    Finished `release` profile [optimized] target(s) in 18m 30s
```

```bash
ls -lh target/release/zeroclaw
# ~10.8 MB  （+2.0 MB）
```

#### Feature 3：memory-postgres（PostgreSQL 后端）

```bash
cargo build --release --features memory-postgres
```

```bash
ls -lh target/release/zeroclaw
# ~9.1 MB  （+0.3 MB）
```

#### Feature 4：observability-otel（OpenTelemetry 可观测性）

```bash
cargo build --release --features observability-otel
```

```bash
ls -lh target/release/zeroclaw
# ~9.8 MB  （+1.0 MB）
```

### 组合多个 Feature

```bash
# 企业 Bot 场景：飞书 + PostgreSQL + 监控
cargo build --release --features "channel-lark,memory-postgres,observability-otel"
```

**预期输出尾部：**

```
   Compiling prost v0.13.0
   Compiling postgres v0.19.9
   Compiling opentelemetry v0.28.0
   Compiling zeroclaw v0.1.7
    Finished `release` profile [optimized] target(s) in 19m 10s
```

```bash
ls -lh target/release/zeroclaw
# ~10.5 MB
```

### 全功能构建（不推荐生产环境）

```bash
cargo build --release --all-features
```

```bash
ls -lh target/release/zeroclaw
# ~16-18 MB（所有 12 个 Feature 全部编入）
```

> ⚠️ **`--all-features` 仅用于测试目的**。在生产环境应该只启用你需要的 Feature，原因：
> - 增大攻击面（更多代码 = 更多潜在漏洞）
> - `probe` Feature 引入 ~50 个额外 crate，99% 用户不需要
> - 编译时间显著增加

### Feature 对构建的影响一览表

| Feature | 新增依赖 | 体积增量 | 编译时间增量 | 适用场景 |
|---------|---------|---------|------------|---------|
| 无（默认） | — | 基准 8.8 MB | 基准 ~15 min | 个人使用 |
| `hardware` | nusb, tokio-serial | +~0.5 MB | +~30s | USB/串口硬件 |
| `channel-matrix` | matrix-sdk | +~2.0 MB | +~3 min | Matrix 聊天 |
| `channel-lark` | prost | +~0.4 MB | +~1 min | 飞书 Bot |
| `memory-postgres` | postgres | +~0.3 MB | +~30s | 高并发存储 |
| `observability-otel` | opentelemetry | +~1.0 MB | +~2 min | 分布式追踪 |
| `peripheral-rpi` | rppal | +~0.2 MB | +~20s | 树莓派 GPIO |
| `probe` | probe-rs (~50 deps) | +~3.0 MB | +~5 min | STM32 调试 |
| `rag-pdf` | pdf-extract | +~0.8 MB | +~1 min | PDF 知识库 |
| `browser-native` | fantoccini | +~1.2 MB | +~2 min | 网页自动化 |
| `sandbox-landlock` | —（内核接口） | +~0.1 MB | +~10s | Linux 沙箱 |
| `sandbox-bubblewrap` | —（系统调用） | +~0.1 MB | +~10s | Linux 沙箱 |
| **全部（`--all-features`）** | **~120+ crates** | **~16-18 MB** | **~25 min** | ⚠️ 仅测试 |

> **前端类比：** 每个 Feature 就像 `npm install` 一个可选 adapter 包。
> - `hardware` ≈ `npm install serialport`（原生模块，需要编译）
> - `channel-matrix` ≈ `npm install matrix-js-sdk`（SDK 包，体积大）
> - `observability-otel` ≈ `npm install @sentry/nextjs`（监控集成）

### Feature 组合推荐表（按使用场景）

| 场景 | 推荐 Feature 组合 | 构建命令 | 预估体积 |
|------|-------------------|---------|---------|
| **个人开发者** | 无 | `cargo build --release` | ~8.8 MB |
| **企业飞书 Bot** | lark + postgres + otel | `--features "channel-lark,memory-postgres,observability-otel"` | ~10.5 MB |
| **IoT / 树莓派** | hardware + rpi | `--features "hardware,peripheral-rpi"` | ~9.5 MB |
| **知识库助手** | pdf + browser + postgres | `--features "rag-pdf,browser-native,memory-postgres"` | ~11.0 MB |
| **安全敏感部署** | landlock + bubblewrap + otel | `--features "sandbox-landlock,sandbox-bubblewrap,observability-otel"` | ~10.0 MB |
| **多平台 Bot** | matrix + lark + whatsapp | `--features "channel-matrix,channel-lark,whatsapp-web"` | ~12.0 MB |

[来源: sourcecode/zeroclaw/Cargo.toml — `[features]` 节完整定义]

---

## 场景 3.3：Profile 切换对比

> **目标：** 对比 `dev` / `release` / `release-fast` 三种 Profile 的构建行为。

### ZeroClaw 的三种 Profile

```toml
# Cargo.toml 中定义的 Profile（简化展示）

# Profile 1: dev（cargo build 默认）
[profile.dev]
opt-level = 0            # 不优化
# debug = true           # 包含调试符号（Cargo 默认）

# Profile 2: release（cargo build --release）
[profile.release]
opt-level = "z"          # 体积最小化
lto = "fat"              # 全量链接时优化
codegen-units = 1        # 单线程代码生成（最大优化空间）
strip = true             # 去除调试符号
panic = "abort"          # panic 直接终止

# Profile 3: release-fast（cargo build --profile release-fast）
[profile.release-fast]
inherits = "release"     # 继承 release 所有设置
codegen-units = 8        # 改为 8 线程并行编译（快但优化略差）
```

### 实战对比

```bash
# ===== 1. Dev 构建 =====
time cargo build
# 预期: 约 2m 15s

ls -lh target/debug/zeroclaw
# 预期: ~32 MB

# ===== 2. Release 构建（体积优化） =====
time cargo build --release
# 预期: 约 15m 42s

ls -lh target/release/zeroclaw
# 预期: ~8.8 MB

# ===== 3. Release-Fast 构建（编译速度优化） =====
time cargo build --profile release-fast
# 预期: 约 8m 30s

ls -lh target/release-fast/zeroclaw
# 预期: ~9.2 MB
```

> **注意：** `release-fast` 的产物在 `target/release-fast/` 目录，不是 `target/release/`。

### 三种 Profile 对比表

| 维度 | dev | release-fast | release |
|------|-----|-------------|---------|
| **编译时间** | ⚡ ~2 min | 🔶 ~8 min | 🐢 ~15 min |
| **二进制大小** | ~32 MB | ~9.2 MB | ~8.8 MB |
| **运行速度** | 最慢 | 接近最优 | 最优 |
| **调试支持** | ✅ 完整 | ❌ 无 | ❌ 无 |
| **LTO** | 无 | fat | fat |
| **codegen-units** | 256 | 8 | 1 |
| **适用场景** | 日常开发 | 本地测试 | 生产部署 / CI |
| **内存需求** | 4 GB | 16 GB+ | 8 GB |
| **前端类比** | `npm run dev` | `next dev --turbo` | `npm run build` |

> **前端类比：** 三种 Profile = webpack 的三种配置：
> - `dev` = `mode: development` + `devtool: eval-source-map`（快速编译，大文件）
> - `release-fast` = `mode: production` + `parallel: 8`（并行优化，略大）
> - `release` = `mode: production` + `minimize: true` + `concatenateModules: true`（极致优化）

### 什么时候用哪个？

```
你在做什么？
│
├── 写代码、改 bug、跑测试
│   └── cargo build（dev）→ 编译快，有调试信息
│
├── 本地验证性能 / 测试 Release 行为
│   └── cargo build --profile release-fast → 折中方案
│
├── CI/CD 构建 / 打包发布 / 部署生产
│   └── cargo build --release → 最优产物，值得等
│
└── 资源受限设备（树莓派等）
    └── cargo build --release → 最小体积优先
```

[来源: sourcecode/zeroclaw/Cargo.toml — `[profile.release]` 和 `[profile.release-fast]` 配置]

---

## 场景 3.4：交叉编译入门

> **目标：** 在你的开发机上编译出其他平台可运行的 ZeroClaw 二进制。

### 为什么需要交叉编译？

```
场景：你在 macOS M2 上开发，需要部署到 Linux x86_64 服务器

方案 A：在服务器上装 Rust + 编译（❌ 浪费服务器资源，编译可能很慢）
方案 B：在本地交叉编译，scp 过去（✅ 服务器不需要 Rust 环境）
```

> **前端类比：** 前端天然交叉编译——你在 macOS 上 `npm run build`，产出的 HTML/CSS/JS 在任何平台的浏览器里都能跑。Rust 需要显式指定目标平台，但思路一样：**在开发机上构建，在目标机上运行**。

### 方式 1：rustup + 手动交叉编译

```bash
# 1. 查看当前已安装的 target
rustup target list --installed
# 预期: aarch64-apple-darwin（你的本机平台）

# 2. 添加 Linux ARM64 目标
rustup target add aarch64-unknown-linux-gnu

# 3. 安装交叉编译器（Ubuntu/Debian）
sudo apt install -y gcc-aarch64-linux-gnu

# 4. 交叉编译
cargo build --release --target aarch64-unknown-linux-gnu

# 5. 查看产物
ls -lh target/aarch64-unknown-linux-gnu/release/zeroclaw
# ~8.8 MB — ARM64 Linux 可执行文件

# 6. 部署到目标设备
scp target/aarch64-unknown-linux-gnu/release/zeroclaw pi@raspberrypi:~/
ssh pi@raspberrypi ./zeroclaw --version
```

> ⚠️ **注意：** 手动交叉编译需要安装目标平台的 C 链接器和系统库，配置较繁琐。推荐使用 `cross` 工具。

### 方式 2：使用 cross 工具（推荐）

`cross` 自动在 Docker 容器中提供目标平台的完整工具链——你不需要手动安装交叉编译器。

```bash
# 1. 安装 cross
cargo install cross

# 2. 确保 Docker 运行中
docker info > /dev/null 2>&1 && echo "✅ Docker 运行中" || echo "❌ 请启动 Docker"

# 3. 交叉编译（自动拉取对应 Docker 镜像）
cross build --release --target aarch64-unknown-linux-gnu
```

**预期输出：**

```
info: Downloading component 'rust-std' for 'aarch64-unknown-linux-gnu'
   Compiling autocfg v1.4.0
   ...
   Compiling zeroclaw v0.1.7
    Finished `release` profile [optimized] target(s) in 18m 30s
```

```bash
# 产物在本地 target 目录
file target/aarch64-unknown-linux-gnu/release/zeroclaw
# ELF 64-bit LSB executable, ARM aarch64 — ✅ ARM64 Linux 二进制
```

> **前端类比：** `cross` 就像 `docker run -v $(pwd):/app node:20 npm run build`——在标准化容器中构建，免去环境配置的烦恼。

### 方式 3：带自定义 Feature 的交叉编译

```bash
# 交叉编译 + 启用 hardware Feature（为树莓派构建带硬件支持的版本）
cross build --release \
  --target aarch64-unknown-linux-gnu \
  --features "hardware,peripheral-rpi"
```

```bash
# 部署到树莓派
scp target/aarch64-unknown-linux-gnu/release/zeroclaw pi@raspberrypi:~/
ssh pi@raspberrypi './zeroclaw --version'
# zeroclaw 0.1.7 (built 2026-xx-xx, commit xxxxxx)
```

### 方式 4：Docker 多阶段构建

如果你已经使用 Docker 部署，可以直接在 Dockerfile 中完成：

```bash
# 使用 ZeroClaw 仓库自带的 Dockerfile
docker build --target release -t zeroclaw:custom .

# 带自定义 Feature 的 Docker 构建
docker build --target release \
  --build-arg FEATURES="channel-lark,memory-postgres" \
  -t zeroclaw:enterprise .
```

> **前端类比：** `docker build --target release` = 前端 Docker 多阶段构建中只取 `nginx:alpine` 阶段——编译环境（~2 GB）不进入最终镜像（~23 MB）。

### 常用 Target Triple 速查

| Target Triple | 设备 | 用 cross？ |
|--------------|------|-----------|
| `x86_64-unknown-linux-gnu` | Linux PC / 服务器 | 可选 |
| `aarch64-unknown-linux-gnu` | ARM64 Linux / 树莓派 4+ | 推荐 |
| `armv7-unknown-linux-gnueabihf` | 树莓派 3 (32位) | 推荐 |
| `aarch64-apple-darwin` | macOS M1/M2+ | 不需要（原生） |
| `x86_64-pc-windows-msvc` | Windows PC | 仅 Windows 上 |

[来源: sourcecode/zeroclaw/.github/workflows/release.yml — Release 构建矩阵]

---

## 场景 3.5：开发工作流

> **目标：** 掌握从源码参与 ZeroClaw 开发的完整流程。

### 代码质量检查

```bash
# ===== 格式检查（= prettier --check）=====
cargo fmt --check
# 通过 → 无输出
# 不通过 → 显示需要格式化的文件

# 自动格式化
cargo fmt

# ===== Lint 检查（= eslint .）=====
cargo clippy
# 通过 → 无输出
# 不通过 → 显示警告和建议
```

**预期输出（如果有问题）：**

```
warning: unused variable: `x`
 --> src/main.rs:42:9
  |
42 |     let x = 5;
  |         ^ help: if this is intentional, prefix it with an underscore: `_x`
```

> **前端类比：** `cargo fmt` = `prettier --write`，`cargo clippy` = `eslint --fix`。区别是 Rust 的 clippy 不仅检查风格，还检查潜在的逻辑 bug。

### 运行测试

```bash
# 运行所有测试
cargo test

# 运行特定测试
cargo test test_agent_response

# 显示测试输出（默认被捕获）
cargo test -- --nocapture
```

**预期输出：**

```
   Compiling zeroclaw v0.1.7
    Finished `test` profile [unoptimized + debuginfo] target(s) in 2m 30s
     Running unittests src/main.rs (target/debug/deps/zeroclaw-abc123)

running 47 tests
test config::tests::test_default_config ... ok
test config::tests::test_env_override ... ok
test agent::tests::test_message_parse ... ok
...
test result: ok. 47 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

### 从源码直接运行（不安装）

```bash
# cargo run = cargo build + 执行（Dev 模式）
cargo run -- agent -m "Hello from source!"

# 以 Release 模式运行
cargo run --release -- agent -m "Hello from source (release)!"

# 带 Feature 运行
cargo run --release --features "hardware" -- agent -m "Hello with hardware!"
```

> **前端类比：** `cargo run` = `npx ts-node src/index.ts`——编译并立即执行，不安装到全局。

### 安装到 ~/.cargo/bin（替换全局版本）

```bash
# 从当前源码安装到 ~/.cargo/bin
cargo install --path . --force --locked

# --path .    → 从当前目录安装（而非 crates.io）
# --force     → 覆盖已有版本
# --locked    → 使用 Cargo.lock 锁定的依赖版本

# 带 Feature 安装
cargo install --path . --force --locked --features "channel-lark,memory-postgres"

# 验证
zeroclaw --version
which zeroclaw
# ~/.cargo/bin/zeroclaw
```

### 增量编译技巧

```bash
# 首次编译慢（~2-15 min），后续修改只编译变更部分
cargo build           # 首次: ~2 min
# 修改一个文件后...
cargo build           # 增量: ~5-10s  ← 快很多！

# ⚠️ 切换 Feature 会触发大量重编译
cargo build --features "hardware"       # 首次带 Feature: 重新编译
cargo build --features "hardware"       # 第二次: 增量编译

# 💡 清理缓存（磁盘空间不足时）
cargo clean                              # 删除整个 target/ 目录
du -sh target/                           # 查看 target 目录大小（可能 2-6 GB）
```

> **前端类比：** `cargo build` 的增量编译 = webpack 的 `cache.type: 'filesystem'` 持久化缓存——首次慢，后续只编译变更部分。

---

## 前端类比总结

| 源码构建操作 | 前端等价操作 |
|-------------|-------------|
| `git clone + cargo build` | `git clone + npm install + npm run build` |
| `cargo build`（dev） | `npm run dev` / `webpack --mode development` |
| `cargo build --release` | `npm run build` / `webpack --mode production` |
| `cargo build --profile release-fast` | `next dev --turbo`（快速但非最优） |
| `--features "hardware"` | `npm install serialport`（可选原生模块） |
| `--all-features` | 把 `package.json` 所有 optional deps 都装上 |
| `opt-level = "z"` | `terser({ compress: true, mangle: true })` |
| `lto = "fat"` | scope hoisting + 跨模块内联 |
| `strip = true` | 不生成 source map |
| `cross build --target arm64` | — （JS 天然跨平台） |
| `cargo fmt` | `prettier --write` |
| `cargo clippy` | `eslint .` |
| `cargo test` | `jest` / `vitest run` |
| `cargo run` | `npx ts-node src/index.ts` |
| `cargo install --path .` | `npm link` / `npm install -g .` |
| `cargo clean` | `rm -rf node_modules/.cache` / `rm -rf .next` |
| `target/` 目录 | `node_modules/` + `.next/` + `dist/` |

---

## 常见构建问题排查

| # | 现象 | 原因 | 解决方案 |
|---|------|------|---------|
| 1 | `error: linker 'cc' not found` | 缺少 C 编译器 | `sudo apt install build-essential`（Linux）或 `xcode-select --install`（macOS） |
| 2 | `failed to run custom build command for 'openssl-sys'` | 缺少 OpenSSL 头文件 | `sudo apt install libssl-dev`。但 ZeroClaw 用 rustls，正常不会遇到 |
| 3 | `Killed (signal 9)` | 内存不足 | 用 `cargo build -j 1`（单线程编译减少内存）或增加 swap |
| 4 | `target/` 占用太多磁盘 | 编译缓存积累 | `cargo clean` 释放空间 |
| 5 | `error[E0658]: feature X is not stable` | Rust 版本过低 | `rustup update stable` 或检查 `rust-toolchain.toml` |
| 6 | 切换 Feature 后编译很慢 | 不同 Feature 组合的缓存不共享 | 正常现象；首次带新 Feature 编译会重新编译依赖 |
| 7 | `cross` 报错 `Docker not found` | Docker 未运行 | 启动 Docker Desktop |
| 8 | `cross` 拉取镜像失败 | 网络问题 | 配置 Docker 镜像加速或代理 |
| 9 | Windows 编译极慢 | Defender 实时扫描 | 将 `~/.cargo` 和 `target/` 加入 Defender 排除列表 |
| 10 | `error: profile 'release-fast' is not defined` | 自定义 Profile 需在 Cargo.toml 中定义 | 确保使用 ZeroClaw 仓库源码（非 `cargo install`） |

---

## 参考资料

| 来源 | 说明 |
|------|------|
| [来源: sourcecode/zeroclaw/Cargo.toml] | Feature Flags 完整定义、Release Profile 配置、MSRV |
| [来源: sourcecode/zeroclaw/Dockerfile] | Docker 多阶段构建、ARG FEATURES 参数化构建 |
| [来源: sourcecode/zeroclaw/.github/workflows/release.yml] | Release 构建矩阵（4 个 target triple） |
| [来源: sourcecode/zeroclaw/rust-toolchain.toml] | 项目级 Rust 版本锁定 |

---

> **上一篇：** `07_实战代码_场景2_环境变量与配置文件实战.md` — 环境变量与配置文件实战
>
> **下一篇：** `08_面试必问.md` — 安装与环境配置面试高频问题

---

**文件信息**
- 知识点: ZeroClaw 安装与环境配置
- 维度: 07_实战代码_场景3_源码构建与Feature定制
- 版本: v1.0
- 日期: 2026-03-11
