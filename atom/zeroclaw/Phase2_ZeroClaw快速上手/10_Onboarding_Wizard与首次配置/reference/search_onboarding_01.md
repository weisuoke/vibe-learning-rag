---
type: search_result
search_query: ZeroClaw onboarding wizard setup config.toml 2025 2026
search_engine: grok-mcp
searched_at: 2026-03-11
knowledge_point: 10_Onboarding_Wizard与首次配置
---

# 搜索结果：ZeroClaw Onboarding Wizard Setup

## 搜索摘要

ZeroClaw 是一个轻量级 Rust AI Agent 运行时框架，其 onboarding wizard 是通过 `zeroclaw onboard` 命令启动的交互式配置引导系统。

## 关键信息提取

### 安装方式
1. **一行脚本**: `curl -LsSf https://raw.githubusercontent.com/zeroclaw-labs/zeroclaw/main/install.sh | bash`
2. **Homebrew**: `brew install zeroclaw`
3. **Cargo**: `cargo install --git https://github.com/zeroclaw-labs/zeroclaw`
4. **源码编译**: `git clone + cargo install`

### Onboarding 命令
- **快速模式（默认）**: `zeroclaw onboard --quick --api-key sk-or-... --provider openrouter`
- **交互式 wizard**: `zeroclaw onboard --interactive`
- **强制覆盖**: `zeroclaw onboard --force`
- **仅修复 channels**: `zeroclaw onboard --channels_only`

### 交互式 Wizard 9 步流程
1. 工作区设置（Workspace setup）
2. Provider 与 API Key 配置
3. Channel 配置（Telegram, Discord, Slack 等）
4. Tunnel 配置（ngrok, Cloudflare, Tailscale）
5. 工具模式与安全设置
6. Hardware 集成
7. Memory 后端选择（SQLite, Lucid, Markdown, None）
8. 项目上下文
9. 脚手架文件生成

### Config 路径解析优先级
1. `ZEROCLAW_WORKSPACE` 环境变量
2. `active_workspace.toml` 持久化标记
3. 默认 `~/.zeroclaw/config.toml`

### 2026 年新特性
- TOTP 注册（Step 10）在 `[security.otp]` section
- 覆盖保护提示（已有 config 时）
- Docker 兼容性修复

### Config 管理命令
- `zeroclaw config show` - 显示有效配置（secrets 已脱敏）
- `zeroclaw config set providers.default_model "new-model"` - 更新配置
- `zeroclaw config get security.otp.enabled` - 查询特定键

### 示例 config.toml 核心部分
```toml
[providers.default]
provider = "openrouter"
model = "openrouter/auto"
api_key = "sk-or-..."

[memory]
backend = "sqlite"

[security]
otp.enabled = true
sandbox.enabled = true

[runtime]
mode = "native"
```

## 相关链接
- [ZeroClaw GitHub](https://github.com/zeroclaw-labs/zeroclaw) - 主仓库
- docs/config-reference.md - 配置参考文档
- Wiki: 02.2-Onboarding-Wizard - Wizard 详细流程
