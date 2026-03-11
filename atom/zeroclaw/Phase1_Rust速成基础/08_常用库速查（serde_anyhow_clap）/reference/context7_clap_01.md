---
type: context7_documentation
library: clap
version: "4.5"
fetched_at: 2026-03-11
knowledge_point: 08_常用库速查（serde/anyhow/clap）
context7_query: derive Parser Subcommand ValueEnum arguments attributes
---

# Context7 文档：clap

## 文档来源
- 库名称：clap
- 版本：4.5
- 官方文档链接：https://docs.rs/clap/latest/clap/

## 关键信息提取

### 1. Derive API 结构
clap 的 derive API 包含四种核心 trait：
- `Parser`: 解析参数到 struct（顶层入口）
- `Args`: 定义可复用的参数组
- `Subcommand`: 定义子命令枚举
- `ValueEnum`: 解析值到枚举

```rust
use clap::{Parser, Args, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "app", version, about)]
struct Cli {
    #[arg(short, long)]
    verbose: bool,

    #[command(flatten)]
    common: CommonArgs,

    #[command(subcommand)]
    command: Command,
}

#[derive(Args)]
struct CommonArgs {
    #[arg(long)]
    config: Option<String>,
}

#[derive(Subcommand)]
enum Command {
    Add { name: String },
    Remove { #[arg(short)] id: u32 },
}

#[derive(ValueEnum, Clone)]
enum LogLevel {
    Trace, Debug, Info, Warn, Error,
}
```

### 2. 属性体系
- `#[command(...)]`: 命令级属性（name, version, about, long_about）
- `#[arg(...)]`: 参数级属性（short, long, default_value, value_parser）
- `#[value(...)]`: ValueEnum 值属性（name, alias）
- `#[group(...)]`: 参数分组属性

### 3. 文档注释作为帮助信息
/// 注释自动成为帮助文本：
- struct 上的 /// 成为 about
- field 上的 /// 成为参数 help
- enum variant 上的 /// 成为子命令 help
