---
type: context7_documentation
library: Cargo (Rust Package Manager)
version: latest
fetched_at: 2026-03-11
knowledge_point: 09_ZeroClaw安装与环境配置
context7_query: "cargo install binary package features profile build configuration"
---

# Context7 文档：Cargo (Rust Package Manager)

## 文档来源
- 库名称：Cargo
- Context7 Library ID: /websites/doc_rust-lang_cargo
- 官方文档链接：https://doc.rust-lang.org/cargo/

## 关键信息提取

### cargo install 命令

`cargo install` 用于构建和安装 Rust crate，使其作为可执行二进制可用。

**安装来源：**
```bash
# 从 crates.io
cargo install ripgrep

# 指定版本
cargo install ripgrep@13.0.0

# 从本地路径
cargo install --path ./

# 从 Git 仓库
cargo install --git https://github.com/user/repo.git

# 列出已安装包
cargo install --list
```

### 构建配置

**目标架构和构建配置文件：**
```bash
# 指定目标架构
cargo install --target "x86_64-unknown-linux-gnu"

# 指定构建产物目录
cargo install --target-dir "/path/to/artifacts"

# 使用 debug 配置文件
cargo install --debug

# 使用特定配置文件
cargo install --profile "production"
```

### Feature Flags 管理

```bash
# 指定单个或多个 features
cargo install --features "feature1 feature2"

# 包级别的 feature
cargo install --features "package/feature-name"

# 启用所有 features
cargo install --all-features

# 禁用默认 features
cargo install --no-default-features
```

### Release Profile 配置

```toml
[profile.release]
strip = "debuginfo"  # 去除调试信息减小体积

# 自定义 rustc 标志
rustflags = [ "-C", "..." ]
```

### ZeroClaw 特有的 Profile 配置

```toml
[profile.release]
opt-level = "z"      # 体积优化（ZeroClaw 选择体积最小化）
lto = "fat"          # 链接时优化（最大化跨 crate 优化）
codegen-units = 1    # 单线程代码生成（低内存友好）
strip = true         # 去除所有调试符号
panic = "abort"      # 使用 abort 而非 unwind（减小体积）

[profile.release-fast]
inherits = "release"
codegen-units = 8    # 并行代码生成（需要 16GB+ RAM）
```

这些配置使 ZeroClaw 的 release 构建优化为体积最小（~8.8MB），
适合在资源受限的硬件上运行。
