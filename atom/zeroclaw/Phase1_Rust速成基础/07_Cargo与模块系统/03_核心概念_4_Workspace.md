# Cargo 与模块系统 - 核心概念 4：Workspace

> **知识点**: Cargo 与模块系统
> **层级**: Phase1_Rust速成基础
> **维度**: 核心概念（Workspace 多 crate 管理、依赖继承、元数据共享与 monorepo 实践）
> **目标受众**: 有 TypeScript/前端经验但零 Rust 经验的开发者

---

## 一句话定义

> **Workspace 是 Cargo 管理多个相关 crate 的方式——像 pnpm workspace 一样共享依赖和锁文件，让 monorepo 中所有包用同一份 Cargo.lock 和 target/ 目录。**

---

## 一、什么是 Workspace？

### 1.1 定义

Workspace 是一组**共享 `Cargo.lock` 和 `target/` 目录**的相关 Rust 包（crate）集合 [^context7_cargo1]。它们住在同一个仓库里，由一个顶层 `Cargo.toml` 统一管理。

**核心价值**：
- **依赖版本一致**——所有成员 crate 使用同一份 `Cargo.lock`，不会出现 crate A 用 serde 1.0.200 而 crate B 用 serde 1.0.217 的情况
- **共享编译缓存**——共用 `target/` 目录，相同依赖只编译一次
- **统一管理**——一条命令测试/构建所有包

### 1.2 TypeScript 类比

如果你用过 pnpm workspace 或 npm workspaces，概念几乎一一对应 [^search1]：

```
Rust Workspace                    pnpm Workspace
────────────────────────────────────────────────────────────
根 Cargo.toml [workspace]         根 pnpm-workspace.yaml
workspace.members                 packages: ["packages/*"]
共享 Cargo.lock                   共享 pnpm-lock.yaml
共享 target/                      共享 node_modules/.pnpm/
workspace.dependencies            catalog: protocol（pnpm 9+）
cargo build --workspace           pnpm -r build
cargo test -p <name>              pnpm --filter <name> test
```

### 1.3 两种 Workspace 形态

Workspace 有两种组织方式 [^context7_cargo1]：

**1. 虚拟 Manifest（推荐）**——根 `Cargo.toml` 只有 `[workspace]`，没有 `[package]`：

```toml
# 根 Cargo.toml（虚拟 manifest）
[workspace]
members = ["crates/app", "crates/core", "crates/utils"]
resolver = "2"
```

**2. 根包 Workspace**——根 `Cargo.toml` 既是 workspace 又是一个包：

```toml
# 根 Cargo.toml（根包 workspace）——ZeroClaw 用的就是这种
[package]
name = "my-app"
version = "0.1.0"

[workspace]
members = ["crates/robot-kit"]
```

**类比**：虚拟 manifest 就像 pnpm workspace 中根目录的 `package.json` 没有自己的业务代码；根包 workspace 就像根 `package.json` 自身也是一个可发布的包 [^source1]。

### 1.4 什么时候需要 Workspace？

```
阶段                   推荐方式            信号
────────────────────────────────────────────────────────────────────
刚开始写               单 crate            代码 < 5000 行，一个人维护
模块越来越多           单 crate + 模块      lib.rs 有 10+ pub mod
编译太慢了             拆 crate            改一行等 30 秒编译
需要共享库             workspace           多个二进制共享核心逻辑
多团队协作             workspace           不同人负责不同 crate
```

**转折点信号**：增量编译超过 10 秒、有独立可复用的库代码、想发布子组件到 crates.io。

---

## 二、Workspace 配置详解

### 2.1 `[workspace]` 基本结构

```toml
[workspace]
members = [
    "crates/core",        # 精确路径
    "crates/utils",
    "tools/*",            # glob 模式——匹配 tools/ 下所有子目录
]
exclude = [
    "experiments/broken",  # 排除某些目录（glob 匹配到但不想纳入的）
]
resolver = "2"            # 推荐：使用 resolver v2（更智能的 feature 解析）
```

**TypeScript 类比**：`members` 就像 `pnpm-workspace.yaml` 中的 `packages` 列表，也支持 glob。

### 2.2 resolver 设置

resolver = "2" 让 Cargo 更智能地处理 feature 解析 [^context7_cargo1]：

```
resolver "1"（旧）                   resolver "2"（推荐）
──────────────────────────────────────────────────────────
dev-dependencies 的 features         dev-dependencies 的 features
会影响正常构建                       只在 cargo test 时启用

所有平台的 features 统一解析          按目标平台分别解析
```

### 2.3 完整虚拟 Manifest 示例

```toml
# 根 Cargo.toml（虚拟 manifest——推荐的 workspace 模式）
[workspace]
members = ["crates/app", "crates/core", "crates/api"]
resolver = "2"

[workspace.dependencies]
tokio = { version = "1.42", features = ["rt-multi-thread", "macros"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
tracing = "0.1"
core = { path = "crates/core" }    # 内部 crate 也可以在这里声明

[workspace.package]
version = "0.1.0"
edition = "2024"
license = "MIT OR Apache-2.0"
authors = ["Your Team"]
```

---

## 三、workspace.dependencies 依赖继承

这是 Workspace 最实用的功能——**集中管理依赖版本**，避免各个 crate 的版本漂移 [^context7_cargo1] [^search1]。

### 3.1 根 Cargo.toml 定义版本，成员 crate 继承

```toml
# 根 Cargo.toml
[workspace.dependencies]
tokio = { version = "1.42", features = ["rt-multi-thread", "macros"] }
serde = { version = "1.0", features = ["derive"] }
reqwest = { version = "0.12", default-features = false }
```

```toml
# crates/core/Cargo.toml
[package]
name = "my-core"
version.workspace = true          # 继承 workspace 版本
edition.workspace = true          # 继承 workspace edition

[dependencies]
tokio = { workspace = true }      # 继承版本和 features
serde.workspace = true            # 简写形式（TOML 的点号语法）
```

### 3.2 在成员中追加额外 features

继承不意味着锁死——成员 crate 可以在 workspace 定义的基础上**追加** features：

```toml
# crates/api/Cargo.toml
[dependencies]
reqwest = { workspace = true, features = ["json", "multipart"] }  # 额外启用

# crates/core/Cargo.toml
[dependencies]
reqwest = { workspace = true, features = ["stream"] }             # 不同的追加
```

**注意**：features 是**可加的**——最终 reqwest 会被编译为所有成员请求的 features 的并集 [^search1]。

### 3.3 与 pnpm catalog protocol 的对比

pnpm 9+ 引入了 `catalog:` 协议，理念完全一致：

```
Rust                                       pnpm 9+
──────────────────────────────────────────────────────────────────
[workspace.dependencies]                   catalog: (pnpm-workspace.yaml)
serde = { version = "1.0" }               "serde": "^1.0"
serde = { workspace = true }              "serde": "catalog:"
{ workspace = true, features = [...] }    （pnpm 不支持额外配置）
```

---

## 四、workspace.package 元数据继承

除了依赖版本，元数据也可以在 workspace 级别共享 [^context7_cargo1]：

```toml
# 根 Cargo.toml
[workspace.package]
version = "0.1.0"
edition = "2024"
license = "MIT OR Apache-2.0"
authors = ["Felix <felix@example.com>"]
rust-version = "1.87"
```

```toml
# crates/core/Cargo.toml（继承元数据）
[package]
name = "my-core"                  # name 不能继承——每个 crate 名字不同
version.workspace = true          # ← 继承 0.1.0
edition.workspace = true          # ← 继承 2024
license.workspace = true          # ← 继承 MIT OR Apache-2.0
description = "Core library"      # description 通常各 crate 不同，不继承
```

**可继承字段**：`version`、`edition`、`authors`、`license`、`repository`、`rust-version` 等。**不可继承**：`name`（每个 crate 必须独立命名）。`description`、`keywords` 等虽可继承但通常各 crate 不同。

**TypeScript 类比**：类似 monorepo 根目录的 `tsconfig.base.json` 定义 `compilerOptions`，子包通过 `extends` 继承。

---

## 五、Workspace 常用命令

```bash
# 构建
cargo build --workspace                  # 构建所有成员
cargo build -p my-core                   # 只构建某个 crate（-p = --package）
cargo build --workspace --exclude my-api # 构建所有但排除某个

# 测试
cargo test --workspace                   # 测试所有
cargo test -p my-core                    # 只测试某个 crate
cargo test -p my-core -- test_name       # 测试某个特定测试

# 检查 / 格式化 / lint
cargo check --workspace                  # 快速检查（不生成二进制）
cargo fmt --all                          # 格式化所有 crate
cargo clippy --workspace                 # lint 所有 crate

# 依赖管理
cargo tree --workspace                   # 查看依赖树
cargo update                             # 更新整个 workspace 的 Cargo.lock
```

### 与 npm workspace 命令对比

```
Cargo                                    npm/pnpm
──────────────────────────────────────────────────────────────────
cargo build --workspace                  pnpm -r build
cargo build -p my-core                   pnpm --filter my-core build
cargo test --workspace                   pnpm -r test
cargo test -p my-core                    pnpm --filter my-core test
cargo build --workspace --exclude api    pnpm -r --filter '!api' build
cargo check --workspace                  pnpm -r typecheck
cargo fmt --all                          pnpm -r format
cargo clippy --workspace                 pnpm -r lint
```

**关键区别**：Cargo 不需要在每个 crate 的 `Cargo.toml` 中定义 `scripts`——构建、测试、检查都是内置的 [^search1]。

---

## 六、ZeroClaw 的 Workspace 结构

### 6.1 实际布局

ZeroClaw 使用**根包 workspace**模式——根目录同时是主 crate 和 workspace 管理者 [^source1]：

```
zeroclaw/
├── Cargo.toml            # 根 workspace + zeroclaw 主 crate
├── Cargo.lock            # 整个 workspace 共享
├── src/                  # 主 crate 代码（30+ 模块）
│   ├── lib.rs
│   ├── main.rs
│   ├── agent/
│   ├── channels/
│   └── tools/
├── crates/               # 子 crate 目录
│   └── robot-kit/        # robot-kit 子 crate
│       ├── Cargo.toml
│       └── src/lib.rs
├── target/               # 整个 workspace 共享的编译输出
└── firmware/             # ⚠️ 这些 crate 不在 workspace 中！
    ├── zeroclaw-esp32/
    ├── zeroclaw-esp32-ui/
    └── zeroclaw-nucleo/
```

### 6.2 根 Cargo.toml 的 workspace 配置

```toml
# zeroclaw/Cargo.toml
[package]
name = "zeroclaw"
version = "0.1.7"
edition = "2024"

[workspace]
members = [
    ".",                    # 根目录自身（zeroclaw 主 crate）
    "crates/robot-kit",     # robot-kit 子 crate
]
resolver = "2"

[dependencies]
robot-kit = { path = "crates/robot-kit" }   # 路径依赖——workspace 内部引用
```

```rust
// src/lib.rs 中使用
use robot_kit::SafetyMonitor;   // 注意：Cargo 把连字符(-)转成下划线(_)
use robot_kit::DriveSystem;
```

### 6.3 Firmware crates 为什么不在 Workspace 中？

`firmware/` 目录下有三个独立 crate，**故意不放进 workspace** [^source1]：

```
原因                          解释
──────────────────────────────────────────────────────────────────────
不同的编译目标                固件编译为 ARM Cortex-M / Xtensa，主 crate 编译为 x86/ARM Linux
不同的依赖生态                固件用 embassy（嵌入式异步），主 crate 用 tokio
不同的 Cargo 配置             固件需要 `.cargo/config.toml` 指定链接器和 target
no_std 环境                   固件不使用 Rust 标准库
```

**类比**：就像 monorepo 中有个 React Native 子项目和一个 Node.js 后端——它们虽然在同一个 Git 仓库，但 `package.json` 和 `node_modules` 完全独立，不放进同一个 pnpm workspace。

### 6.4 为什么 ZeroClaw 选择 Workspace

共享依赖版本（统一 serde、tokio 版本）、统一 Cargo.lock（可重复构建）、共享编译缓存（robot-kit 编译一次后复用）、并行编译（robot-kit 独立编译不阻塞主 crate）、清晰的代码边界（robot-kit 有自己的 API 和 feature flags）。

---

## 七、Workspace vs 单 Crate 的决策

### 7.1 什么时候用什么

```
✅ 单 crate 够了：                        ✅ 需要 workspace：
  - 代码量 < 10,000 行                     - 有可复用的库代码（如 robot-kit）
  - 只有一个二进制或一个库                  - 多个二进制共享核心逻辑
  - 一个人维护                             - 增量编译太慢——拆分可并行编译
  - 增量编译时间 < 5 秒                     - 需要独立发布子组件到 crates.io
  - 没有需要独立发布的子组件                - 多人协作，各负责不同模块
```

### 7.2 拆分 crate 对编译时间的影响

这是从单 crate 迁移到 workspace 的最大动力 [^search1]：

```
场景                               单 crate           拆分成 workspace
───────────────────────────────────────────────────────────────────────
改了 robot-kit 的一行              重编译整个 crate    只重编译 robot-kit
增量编译（改一行核心代码）          ~15 秒             ~3 秒
首次全量编译                       串行编译            并行编译多个 crate
```

**原理**：Rust 的编译单元是 crate。一个 crate 内部改了任何文件，整个 crate 都要重新编译。拆成多个小 crate 后，只需重新编译被改动的那个，其余直接复用缓存 [^search1]。

**TypeScript 类比**：类似把一个巨大的 Next.js 应用拆分成 Turborepo monorepo——每个包独立构建，修改一个包时只重新构建它和依赖它的包。

### 7.3 决策流程图

```
开始 → 代码量 > 10,000 行？
       │
       ├── 否 → 单 crate ✅
       │
       └── 是 → 有可复用的库代码？
                │
                ├── 否 → 增量编译 > 10 秒？
                │         │
                │         ├── 否 → 单 crate ✅
                │         └── 是 → workspace ✅（为了编译速度）
                │
                └── 是 → workspace ✅（为了代码复用）
```

---

## 八、常见陷阱与最佳实践

### 陷阱 1：在成员中忘记 `workspace = true`

```toml
# ❌ 错误——直接写版本号，和 workspace 定义不一致
[dependencies]
serde = "1.0.200"

# ✅ 正确——继承 workspace 版本
[dependencies]
serde.workspace = true
```

**后果**：不同 crate 用不同版本，编译出两份 serde 代码，浪费编译时间和二进制体积。

### 陷阱 2：feature 统一的"意外启用"

```
crate-a 依赖 tokio = { workspace = true }                     # 只用 macros
crate-b 依赖 tokio = { workspace = true, features = ["fs"] }  # 追加 fs

→ 最终 tokio 编译时启用 macros + fs（取并集）
→ crate-a 的代码也能用 tokio::fs，但这不是它声明的！
```

**建议**：不要依赖 feature 统一的"意外启用"。每个 crate 显式声明自己需要的 features [^search1]。

### 陷阱 3：虚拟 manifest 中不能直接 `cargo run`

```bash
cargo run              # ❌ 报错——不知道运行哪个 crate
cargo run -p my-app    # ✅ 必须指定包名
```

ZeroClaw 不会遇到这个问题——它用的是根包 workspace，有 `[[bin]]`，可以直接 `cargo run`。

### 最佳实践总结

```
实践                                         原因
──────────────────────────────────────────────────────────────────
用 workspace.dependencies 集中版本管理       避免版本漂移
用 workspace.package 共享元数据              减少重复配置
子 crate 放在 crates/ 目录下                 清晰的项目结构
路径依赖引用内部 crate                       workspace 内互相引用的标准方式
不同编译目标的 crate 不放进 workspace        避免 feature/target 冲突
从单 crate 开始，需要时再拆                  过早拆分增加复杂度
```

---

## 总结

```
Cargo Workspace 核心要点：

  是什么？
    一组共享 Cargo.lock + target/ 的 crate 集合
    ≈ pnpm workspace / npm workspaces / Nx monorepo

  两种形态：
    虚拟 manifest   → 根 Cargo.toml 没有 [package]（纯管理者）
    根包 workspace  → 根 Cargo.toml 既是包又是管理者（ZeroClaw 用这种）

  三大继承机制：
    workspace.dependencies  → 集中依赖版本   → serde.workspace = true
    workspace.package       → 共享元数据     → version.workspace = true
    共享 Cargo.lock         → 版本一致性     → 自动

  常用命令：
    cargo build --workspace         → 构建所有
    cargo test -p <name>            → 测试某个 crate
    cargo check --workspace         → 快速检查所有

  何时使用：
    有可复用的库代码                → ✅ 拆成独立 crate
    增量编译太慢                    → ✅ 并行编译加速
    不同编译目标                    → ❌ 不要放进同一个 workspace

  ZeroClaw 的选择：
    zeroclaw（主 crate）+ robot-kit（子 crate）= 根包 workspace
    firmware crates 独立管理（不同编译目标）
```

---

> **下一步**: 阅读 `03_核心概念_5_Feature_Flags.md`，深入理解 Cargo 的条件编译机制——这是 Rust 独有的"编译时功能开关"，npm/webpack 的 DefinePlugin 只是它的弱化版。

---

**参考来源**

[^source1]: ZeroClaw 源码分析 — `reference/source_cargo_module_01.md`
[^context7_cargo1]: Cargo 官方文档 — `reference/context7_cargo_01.md`
[^search1]: Rust Cargo 社区最佳实践 — `reference/search_cargo_module_01.md`

---

**文件信息**
- 知识点: Cargo 与模块系统
- 维度: 03_核心概念_4_Workspace
- 版本: v1.0
- 日期: 2026-03-10
