# Cargo 与模块系统 - 核心概念 1：Cargo.toml 与包管理

> **知识点**: Cargo 与模块系统
> **层级**: Phase1_Rust速成基础
> **维度**: 核心概念（深入掌握 Cargo.toml 配置与包管理命令）
> **目标受众**: 有 TypeScript/前端经验但零 Rust 经验的开发者

---

## 一句话定义

> **Cargo.toml 是 Rust 项目的清单文件，定义包元数据、依赖和构建配置——类似 package.json。**

---

## 一、Cargo.toml 完整结构解析

### 1.1 `[package]` 部分——项目身份证

`[package]` 是每个 Cargo.toml 的核心，定义了"这个包是谁" [^context7_cargo1]。

```toml
[package]
name = "zeroclaw"              # 包名（= package.json 的 name）
version = "0.1.7"              # 语义化版本号（= version）
edition = "2024"               # Rust Edition（≈ tsconfig 的 target: "ES2024"）
rust-version = "1.87"          # 最低 Rust 编译器版本（≈ engines.node）
description = "AI Agent framework"  # 包描述（= description）
license = "MIT OR Apache-2.0"  # 许可证（= license）
authors = ["Felix <felix@example.com>"]  # 作者列表
repository = "https://github.com/user/zeroclaw"  # 仓库地址
keywords = ["ai", "agent"]     # 关键词（crates.io 搜索用，最多 5 个）
```

**TypeScript 对照**：

```
Cargo.toml [package]              package.json
─────────────────────────────────────────────────
name = "zeroclaw"                 "name": "zeroclaw"
version = "0.1.7"                 "version": "0.1.7"
edition = "2024"                  无直接对应（≈ tsconfig target）
rust-version = "1.87"             "engines": { "node": ">=20" }
license = "MIT OR Apache-2.0"     "license": "MIT"
```

**关键区别**：`edition` 控制语言特性但**不影响兼容性**——Edition 2024 的 crate 可以依赖 Edition 2018 的 crate，就像 ES2022 项目能用 ES2015 写的 npm 包 [^context7_cargo1]。

### 1.2 语义化版本（SemVer）规则

Rust 严格遵循 SemVer，和 npm 基本一致 [^context7_cargo1]：

```toml
[dependencies]
serde = "1.0.195"        # 等同于 ^1.0.195（≥1.0.195, <2.0.0）
tokio = "1.42"           # 等同于 ^1.42（≥1.42.0, <2.0.0）
rand = "0.8"             # 等同于 ^0.8（≥0.8.0, <0.9.0）← 注意 0.x 特殊规则！
```

**⚠️ 0.x 版本陷阱**：在 1.0 之前，最左边的非零数字被视为 major 版本。`"0.8.5"` 表示 ≥0.8.5, <0.9.0（MINOR 视为 breaking）[^context7_cargo1]。

### 1.3 Cargo.toml vs Cargo.lock

> **Cargo.toml = 你的意图（"我想要 serde 1.x"）**
> **Cargo.lock = 精确快照（"你实际用的是 serde 1.0.217"）**

```
                  Cargo.toml                    Cargo.lock
──────────────────────────────────────────────────────────────
谁写的？         开发者手动编辑                  Cargo 自动生成
内容            版本范围（"1.0"）              精确版本 + checksum
提交到 Git？    ✅ 必须                        ✅ 应用项目 / ❌ 库项目
npm 对应物      package.json                   package-lock.json
```

**为什么库不提交 Cargo.lock？** 和 npm 一样——库的 lock 文件会和最终应用的版本冲突。让应用的 Cargo.lock 统一管理 [^context7_cargo1] [^search1]。

---

## 二、Cargo 核心命令速查

### 2.1 创建项目

```bash
cargo new my-project          # ≈ npm init + mkdir（创建可执行程序，默认）
cargo new my-lib --lib        # 创建库
cargo init                    # ≈ npm init（在当前目录初始化）
```

TypeScript 需要 `mkdir + npm init + npm i typescript + tsc --init` 四步，Cargo **一步搞定** [^search1]。

### 2.2 构建与运行

```bash
cargo build                   # ≈ npm run build (dev)  → target/debug/
cargo build --release         # ≈ npm run build (prod) → target/release/
cargo run                     # ≈ npm start（构建并运行）
cargo run -- --port 8080      # -- 后面是传给程序的参数
```

### 2.3 测试与检查

```bash
cargo check                   # ≈ tsc --noEmit（只检查不编译，最快反馈！）
cargo test                    # ≈ npm test / jest
cargo clippy                  # ≈ eslint（Rust 官方 linter）
cargo fmt                     # ≈ prettier（代码格式化）
```

**💡 日常开发循环**：`cargo check`（写几行就跑）→ `cargo clippy`（提交前）→ `cargo test`（提交前）→ `cargo build --release`（发布时）。

### 2.4 依赖管理

```bash
cargo add serde --features derive       # ≈ npm install serde
cargo add --dev pretty_assertions       # ≈ npm install --save-dev
cargo remove serde                      # ≈ npm uninstall serde
cargo update                            # ≈ npm update
cargo doc --open                        # 生成 API 文档并在浏览器打开
```

### 2.5 命令速查表

```
命令                          npm 等价物                 用途
─────────────────────────────────────────────────────────────────
cargo new <name>              npm init + mkdir           创建新项目
cargo build                   npm run build (dev)        开发构建
cargo build --release         npm run build (prod)       生产构建
cargo run                     npm start                  构建并运行
cargo test                    npm test                   运行测试
cargo check                   tsc --noEmit               快速类型检查
cargo add <pkg>               npm install <pkg>          添加依赖
cargo remove <pkg>            npm uninstall <pkg>        移除依赖
cargo clippy                  npx eslint .               代码检查
cargo fmt                     npx prettier --write .     代码格式化
cargo doc --open              无直接对应                  生成文档
cargo publish                 npm publish                发布到 crates.io
cargo clean                   rm -rf node_modules/       清理构建产物
```

---

## 三、项目初始化实战

### 3.1 `--bin` vs `--lib` 生成的文件结构

```
my-app/（可执行程序）                my-lib/（库）
├── Cargo.toml                      ├── Cargo.toml
└── src/                            └── src/
    └── main.rs  ← 入口点               └── lib.rs  ← 库根
```

```rust
// main.rs 默认内容             // lib.rs 默认内容
fn main() {                    pub fn add(left: u64, right: u64) -> u64 {
    println!("Hello, world!");     left + right
}                              }
                               #[cfg(test)]
                               mod tests {
                                   use super::*;
                                   #[test]
                                   fn it_works() { assert_eq!(add(2, 2), 4); }
                               }
```

### 3.2 lib.rs vs main.rs——一个项目可以同时有两个！

ZeroClaw 就是 bin + lib 双模式 [^source1]：

```rust
// src/lib.rs — 库：定义所有模块和公共 API
pub mod agent;
pub mod config;
pub mod tools;
pub use config::Config;
pub use agent::{Agent, AgentBuilder};
```

```rust
// src/main.rs — 可执行程序：通过 crate 名引用库的 API
use zeroclaw::Config;
use zeroclaw::Agent;

#[tokio::main]
async fn main() {
    let config = Config::load().await;
    Agent::new(config).run().await;
}
```

```typescript
// TypeScript 对照——同样的模式
// src/index.ts（库入口）    →    export { Config } from './config';
// bin/cli.ts（程序入口）    →    import { Config } from '../src';
```

**为什么要双模式？** ① 库可被其他项目依赖 ② main.rs 和外部用户用同一套 API ③ 测试直接测库 API，不需启动程序。

### 3.3 完整项目目录约定

```
my-project/
├── Cargo.toml                 # 项目配置（= package.json）
├── Cargo.lock                 # 版本锁定（= package-lock.json）
├── src/
│   ├── main.rs                # 可执行入口
│   ├── lib.rs                 # 库入口
│   └── utils.rs               # 模块（需在 lib.rs 中 mod utils;）
├── tests/                     # 集成测试（cargo test 自动发现）
├── examples/                  # 示例（cargo run --example basic）
├── benches/                   # 性能基准（cargo bench）
└── build.rs                   # 构建脚本（编译前执行）
```

Cargo 对这些目录有**约定**，不需要额外配置——和 Jest 自动发现 `__tests__/` 的思路一样 [^context7_cargo1]。

---

## 四、Cargo 构建 Profile

### 4.1 dev vs release

```
                  dev（默认）                    release
──────────────────────────────────────────────────────────────
触发              cargo build                    cargo build --release
优化级别          0（不优化）                     3（完全优化）
编译速度          快（秒级）                      慢（分钟级）
产物位置          target/debug/                   target/release/
TypeScript 类比   开发模式（source map）           生产模式（uglify + tree-shake）
```

### 4.2 ZeroClaw 的 release profile——为嵌入式设备极致优化

```toml
[profile.release]
opt-level = "z"        # 极致体积优化（不是速度！取值："0"/"1"-"3"/"s"/"z"）
lto = "fat"            # 跨 crate 链接时优化（≈ 二进制层面的 tree-shaking）
codegen-units = 1      # 串行代码生成——更慢编译，更好优化
strip = true           # 移除调试符号（≈ 去掉 source map）
panic = "abort"        # panic 直接终止（省去 stack unwinding 代码）
```

**体积影响**：这些配置合计可节省 **50-70%** 的二进制体积，对树莓派等嵌入式部署至关重要 [^source1]。

### 4.3 自定义 Profile

```toml
# 服务器用——编译更快，优化足够
[profile.release-fast]
inherits = "release"   # 继承 release 所有设置
codegen-units = 8      # 覆盖：允许并行代码生成
```

```bash
cargo build --profile release-fast    # 产物：target/release-fast/zeroclaw
```

**何时用哪个？** 日常开发 → dev | CI 测试 → dev | 嵌入式部署 → release | 云服务器 → release-fast [^source1]

---

## 五、ZeroClaw 的 Cargo.toml 解读

完整解读 ZeroClaw 真实配置的每个部分 [^source1]：

### 5.1 Workspace + Package + 双模式

```toml
[workspace]
members = ["."]                    # workspace 成员（robot-kit 在 crates/ 下）
resolver = "2"                     # 更智能的 feature 解析

[package]
name = "zeroclaw"
version = "0.1.7"
edition = "2024"
rust-version = "1.87"
license = "MIT OR Apache-2.0"     # Rust 社区标准双许可证

[[bin]]                            # [[双方括号]] = TOML 数组表（可以有多个 bin）
name = "zeroclaw"
path = "src/main.rs"

[lib]
name = "zeroclaw"
path = "src/lib.rs"
```

**`[[bin]]` 为什么是双方括号？** TOML 的数组表语法——一个项目可以有多个可执行文件。等同于 `package.json` 的 `"bin": { "zeroclaw": "...", "zeroclaw-cli": "..." }`。

### 5.2 依赖分类

```toml
# 核心依赖（始终编译）
[dependencies]
tokio = { version = "1.42", features = ["rt-multi-thread", "macros"] }
serde = { version = "1.0", features = ["derive"] }

# 可选依赖（feature flag 控制）
nusb = { version = "0.1", optional = true }
matrix-sdk = { version = "0.7", optional = true }

# 开发依赖（仅测试）                          ≈ devDependencies
[dev-dependencies]
pretty_assertions = "1.4"
```

### 5.3 Feature Flags

```toml
[features]
default = []                                      # 默认最小构建
hardware = ["dep:nusb", "dep:tokio-serial"]       # USB + 串口
channel-matrix = ["dep:matrix-sdk"]               # Matrix 协议
observability-otel = ["dep:opentelemetry"]        # OpenTelemetry
```

**`dep:` 前缀**：新语法（推荐），让 feature 名和依赖名解耦。旧语法下 optional 依赖名自动成为 feature 名，容易冲突 [^context7_cargo1]。

```bash
cargo build                                       # 最小构建
cargo build --features "hardware,channel-matrix"  # 按需启用
cargo build --all-features                        # 全部功能
```

---

## 六、常见陷阱

### `cargo add` ≠ `npm install`

```bash
cargo add serde     # 只修改 Cargo.toml，不下载！
cargo build         # 这一步才下载 + 编译
# 没有 node_modules/——依赖缓存在 ~/.cargo/registry/
```

### `target/` 目录很大

```bash
du -sh target/      # 可能 4GB+（编译中间产物）
cargo clean         # 清理（≈ rm -rf node_modules）
# target/ 已在 .gitignore 中
```

---

## 七、从零开始的完整流程

```bash
cargo new zeroclaw-mini && cd zeroclaw-mini   # 1. 创建项目
cargo add tokio -F rt-multi-thread,macros     # 2. 添加依赖
cargo add serde -F derive
cargo check                                    # 3. 检查编译（最快反馈）
cargo clippy                                   # 4. 代码检查
cargo fmt                                      # 5. 格式化
cargo run                                      # 6. 运行
cargo test                                     # 7. 测试
cargo build --release                          # 8. 生产构建
```

---

## 总结

```
Cargo.toml 五大区域：
  [package]        → 项目身份证（name, version, edition）
  [dependencies]   → 运行时依赖 + 可选依赖
  [features]       → 编译时功能开关（零运行时成本）
  [profile.*]      → 构建优化配置（dev/release/自定义）
  [[bin]] + [lib]  → 产物类型（可执行文件 + 库）

与 npm 的关键区别：
  ✅ Cargo = npm + webpack + jest + eslint 的合体
  ✅ 没有 node_modules/，依赖全局缓存
  ✅ Feature flags 是编译时条件，不是运行时判断
  ✅ Profile 控制编译优化级别（前端无对应物）
  ✅ cargo check 是日常开发最常用的命令
```

---

> **下一步**: 阅读 `03_核心概念_2_模块路径与可见性.md`，深入理解 `mod`、`use`、`pub` 的路径解析规则和可见性控制。

---

**参考来源**

[^source1]: ZeroClaw 源码分析 — `reference/source_cargo_module_01.md`
[^context7_cargo1]: Cargo 官方文档 — `reference/context7_cargo_01.md`
[^search1]: Rust Cargo 社区最佳实践 — `reference/search_cargo_module_01.md`

---

**文件信息**
- 知识点: Cargo 与模块系统
- 维度: 03_核心概念_1_Cargo_toml与包管理
- 版本: v1.0
- 日期: 2026-03-10
