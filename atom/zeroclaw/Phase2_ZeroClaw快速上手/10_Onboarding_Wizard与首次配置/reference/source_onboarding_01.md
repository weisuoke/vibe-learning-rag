---
type: source_code_analysis
source: sourcecode/zeroclaw
analyzed_files:
  - src/onboard/mod.rs
  - src/onboard/wizard.rs
  - src/main.rs (Onboard command definition)
  - src/config/schema.rs (Config persistence)
analyzed_at: 2026-03-11
knowledge_point: 10_Onboarding_Wizard与首次配置
---

# 源码分析：ZeroClaw Onboarding Wizard 系统

## 分析的文件

- `src/onboard/mod.rs` - 模块导出，re-exports wizard 函数
- `src/onboard/wizard.rs` (263KB) - 核心 onboarding 逻辑
- `src/main.rs` (lines 137-163, 696-769) - CLI 命令定义与路由
- `src/config/schema.rs` - Config 结构体、save() 方法、加密管道

## 关键发现

### 1. 三种执行模式

#### A. Quick Setup（默认模式）- `zeroclaw onboard`
- 调用 `onboard::run_quick_setup()`
- 参数：`credential_override`, `provider`, `model_override`, `memory_backend`, `force`
- 流程：Provider 选择 → API Key 输入 → Model 选择 → Config 生成 → 工作区脚手架

#### B. Full Interactive Wizard - `zeroclaw onboard --interactive`
- 调用 `onboard::run_wizard(force)`
- 完整交互流程：工作区 → Provider → Model → Memory → Channel → Tunnel → Hardware → Tools → Config 保存

#### C. Channels Repair Wizard - `zeroclaw onboard --channels_only`
- 调用 `onboard::run_channels_repair_wizard()`
- 仅修复/重新配置 Channel

### 2. CLI 命令定义（main.rs）

```rust
Onboard {
    #[arg(long)]
    interactive: bool,           // 运行完整 wizard
    #[arg(long)]
    force: bool,                 // 覆盖现有配置无需确认
    #[arg(long)]
    channels_only: bool,         // 仅修复 channels
    #[arg(long)]
    api_key: Option<String>,     // API key 覆盖（快速模式）
    #[arg(long)]
    provider: Option<String>,    // Provider 名称（快速模式，默认 openrouter）
    #[arg(long)]
    model: Option<String>,       // Model ID 覆盖
    #[arg(long)]
    memory: Option<String>,      // Memory 后端（sqlite, lucid, markdown, none）
}
```

### 3. Provider 选择系统（50+ 个 Provider）

**分层菜单设计：**

| Tier | 类别 | Provider 列表 |
|------|------|-------------|
| Tier 0 | 推荐 | OpenRouter, Venice, Anthropic, OpenAI, Gemini, DeepSeek, Mistral, xAI, Perplexity |
| Tier 1 | 快速推理 | Groq, Fireworks, Novita, Together AI, NVIDIA NIM |
| Tier 2 | 网关/代理 | Vercel AI, Cloudflare AI, Astrai, Bedrock |
| Tier 3 | 特化 | Moonshot, GLM, MiniMax, Qwen, Qianfan, Z.AI, Cohere |
| Tier 4 | 本地/私有 | Ollama, llama.cpp, vLLM, SGLang, Osaurus |
| Tier 5 | 自定义 | 任何 OpenAI 兼容 API |

### 4. Provider 特定认证逻辑

- **Ollama**: 检测本地 vs 远程，提示远程 endpoint URL + 可选 API key
- **Gemini**: 优先检查 CLI 凭据 (`GeminiProvider::has_cli_credentials()`)，回退到 API key 或环境变量
- **Anthropic**: 检查 `ANTHROPIC_OAUTH_TOKEN` 或 `ANTHROPIC_API_KEY` 环境变量
- **Qwen OAuth**: 复用 `~/.qwen/oauth_creds.json` 凭据
- **Device flow (Copilot)**: 延迟认证到 `zeroclaw auth login --provider <name>`
- **Custom**: 提示输入 base URL、可选 API key、模型名称

### 5. 模型选择系统

- 如果 Provider 支持，从 API 获取实时模型列表（OpenRouter, Anthropic, Gemini, Ollama, OpenAI-compatible）
- 使用模型缓存（`~/.zeroclaw/workspace/.model_cache.json`）
- 如果获取失败，回退到精选默认列表

### 6. Config 生成与持久化

**Config 结构关键字段：**
- `api_key: Option<String>` - Provider API key
- `api_url: Option<String>` - Base URL override
- `default_provider: Option<String>` - Provider ID（默认: "openrouter"）
- `default_model: Option<String>` - Model ID
- `default_temperature: f64` - 温度设置（0.0-2.0, 默认: 0.7）

**保存流程（原子写入 + 加密）：**
1. 使用 `toml::to_string_pretty()` 序列化
2. 加密敏感字段（api_key, composio.api_key 等）通过 SecretStore
3. 创建带 UUID 后缀的临时文件
4. 写入 TOML 内容并 sync 到磁盘
5. 备份现有配置（.bak）
6. 原子重命名 temp → original
7. 设置文件权限（0600 仅所有者读写）

### 7. 工作区脚手架

创建文件：
- `BOOTSTRAP.md` - 启动说明
- `SOUL.md` - Agent 人格/灵魂定义
- `CONTEXT.md` - 项目上下文

设置默认项目上下文：用户名、时区、Agent 名称、通信风格。

### 8. 现有配置检测

如果 config 已存在且无 `--force`：
- 提供 3 个选项：完整 onboarding、仅更新 provider、取消
- `--force` 标志跳过确认直接覆盖

### 9. 环境变量支持

| 变量 | 用途 |
|------|------|
| `ZEROCLAW_WORKSPACE` | 覆盖工作区目录 |
| `ZEROCLAW_CONFIG_DIR` | 覆盖配置目录 |
| `ZEROCLAW_API_KEY` | 运行时覆盖 API key |
| `API_KEY` | 回退 API key |
| `GEMINI_API_KEY` | Gemini API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `ANTHROPIC_OAUTH_TOKEN` | Anthropic OAuth token |
| `ZEROCLAW_AUTOSTART_CHANNELS` | 信号自动启动 channels |

### 10. Full Wizard 额外配置项

- **Memory 后端选择**: sqlite（默认）, lucid, markdown, none
- **Channel 配置**: Telegram, Discord, Slack 等的交互式配置
- **Tunnel 配置**: ngrok, Cloudflare, Tailscale, custom, none
- **Hardware 配置**: 可选外设板配置
- **Tool 配置**: Composio API key, browser computer use, web search

## 代码片段

### 模块导出 (mod.rs)
```rust
pub use wizard::{
    run_wizard,
    run_channels_repair_wizard,
    run_quick_setup,
    run_models_list,
    run_models_refresh,
    run_models_set,
    run_models_status,
    run_models_refresh_all,
};
```

### 命令路由 (main.rs)
```rust
// Quick setup is default; --interactive triggers full wizard
// --channels_only runs repair wizard
match (interactive, channels_only) {
    (true, _) => onboard::run_wizard(force),
    (_, true) => onboard::run_channels_repair_wizard(),
    _ => onboard::run_quick_setup(api_key, provider, model, memory, force),
}
```

### 工作区路径解析
```rust
// 优先级：ZEROCLAW_CONFIG_DIR > ZEROCLAW_WORKSPACE > ~/.zeroclaw/
let config_dir = std::env::var("ZEROCLAW_CONFIG_DIR")
    .or_else(|_| std::env::var("ZEROCLAW_WORKSPACE"))
    .unwrap_or_else(|_| dirs::home_dir().join(".zeroclaw").to_string());
```
