---
type: context7_documentation
library: clap (Rust CLI parser)
version: latest
fetched_at: 2026-03-11
knowledge_point: 10_Onboarding_Wizard与首次配置
context7_query: CLI subcommand definition arg derive builder
---

# Context7 文档：Clap (Rust CLI Parser)

## 文档来源
- 库名称：clap
- 版本：latest
- Context7 ID: /websites/rs_clap

## 关键信息提取

### 1. Derive 宏定义子命令

ZeroClaw 使用 `#[derive(Subcommand)]` 定义 CLI 命令，包括 `Onboard` 子命令：

```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Onboard {
        #[arg(long)]
        interactive: bool,
        #[arg(long)]
        force: bool,
        #[arg(long)]
        api_key: Option<String>,
        #[arg(long)]
        provider: Option<String>,
    },
}
```

### 2. Optional 参数

- 使用 `Option<T>` 包装使参数可选
- `#[arg(long)]` 定义长选项（如 `--interactive`）
- bool 类型参数自动成为 flag

### 3. 与 ZeroClaw Onboarding 的关联

ZeroClaw 的 `Onboard` 命令使用 clap derive 宏定义了以下参数：
- `--interactive` (bool flag) - 触发完整 wizard
- `--force` (bool flag) - 强制覆盖
- `--channels_only` (bool flag) - 仅修复 channels
- `--api_key` (Option<String>) - 快速模式 API key
- `--provider` (Option<String>) - Provider 名称
- `--model` (Option<String>) - 模型 ID
- `--memory` (Option<String>) - Memory 后端
