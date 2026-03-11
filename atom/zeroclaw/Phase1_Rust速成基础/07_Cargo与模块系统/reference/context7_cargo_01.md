---
type: context7_documentation
library: Cargo (doc.rust-lang.org)
version: stable 2025-2026
fetched_at: 2026-03-10
knowledge_point: 07_Cargo与模块系统
context7_query: Cargo.toml workspace dependencies features module system
---

# Context7 文档：Cargo 官方文档

## 文档来源
- 库名称：Cargo
- 版本：stable (2025-2026)
- 官方文档链接：https://doc.rust-lang.org/cargo/

## 关键信息提取

### 1. Cargo.toml 基本结构

```toml
[package]
name = "hello_world"
version = "0.1.0"
edition = "2024"

[dependencies]
time = "0.1.12"
regex = "0.1.41"
```

- `[package]` 部分包含必需的 `name` 字段和 `version` 字段（语义化版本）
- `edition` 指定 Rust 版本（2024 为最新）
- `[dependencies]` 部分列出外部依赖

### 2. Workspace 配置

#### 虚拟 Workspace（推荐）
```toml
[workspace]
members = ["crate1", "crate2"]

[workspace.dependencies]
cc = "1.0.73"
rand = "0.8.5"
regex = { version = "1.6.0", default-features = false, features = ["std"] }
```

#### 成员 crate 继承 workspace 依赖
```toml
# crate1/Cargo.toml
[package]
name = "crate1"
version = "0.2.0"

[dependencies]
regex = { workspace = true, features = ["unicode"] }

[build-dependencies]
cc.workspace = true

[dev-dependencies]
rand.workspace = true
```

### 3. Feature Flags

#### 基本定义
```toml
[features]
default = ["std"]
std = []
serde = ["dep:serde"]

[dependencies]
serde = { version = "1.0", optional = true }
```

#### 启用依赖的特性
```toml
[dependencies]
jpeg-decoder = { version = "0.1.20", default-features = false }

[features]
parallel = ["jpeg-decoder/rayon"]
```

#### 禁用默认特性并启用特定特性
```toml
[dependencies]
flate2 = { version = "1.0.3", default-features = false, features = ["zlib-rs"] }
```

#### Feature 机制说明
- Cargo "features" 提供条件编译和可选依赖的机制
- 包在 `[features]` 表中定义命名特性
- 每个特性可以启用或禁用
- 通过 `--features` 命令行标志启用
- crates.io 限制最多 300 个特性

### 4. 依赖管理

#### 开发依赖（仅用于测试/示例）
```toml
[dev-dependencies]
tempdir = "0.3"
```

#### 构建依赖
```toml
[build-dependencies]
cc = "1.0"
```

#### 路径依赖（workspace 内部）
```toml
[dependencies]
my-core = { path = "../core" }
```

### 5. Resolver

- resolver = "2" 或 "3" 用于更好的特性解析
- resolver 2 独立处理 dev-dependencies 的特性
- 推荐在 workspace 中使用 resolver = "2" 以上
