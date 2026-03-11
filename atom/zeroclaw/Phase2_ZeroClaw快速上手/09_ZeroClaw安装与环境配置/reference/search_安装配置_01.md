---
type: search_result
search_query: "ZeroClaw Rust AI agent install setup guide 2025 2026"
search_engine: grok-mcp
searched_at: 2026-03-11
knowledge_point: 09_ZeroClaw安装与环境配置
---

# 搜索结果：ZeroClaw 安装与环境配置

## 搜索摘要

ZeroClaw 是一个 100% Rust 编写的轻量级 AI Agent 运行时，核心特点：<5MB RAM、<10ms 启动、可插拔架构。

## 相关链接

- [GitHub 官方仓库](https://github.com/zeroclaw-labs/zeroclaw) - 主要代码和文档
- [ZeroClaw 官网](https://zeroclawlabs.ai/) - 一键安装脚本
- [社区安装指南 Gist](https://gist.github.com/Pyr0zen/1233d7b94f512d4f81a4b7a770e2ed2f) - VPS 部署详细步骤

## 关键信息提取

### 安装方式汇总（2026 最新）

1. **一键安装**（推荐）:
   ```bash
   curl -fsSL https://zeroclawlabs.ai/install.sh | bash
   # 或直接从 GitHub:
   curl -fsSL https://raw.githubusercontent.com/zeroclaw-labs/zeroclaw/main/install.sh | bash
   ```

2. **Homebrew**（macOS/Linux）:
   ```bash
   brew install zeroclaw
   ```

3. **Git 克隆 + Bootstrap**:
   ```bash
   git clone https://github.com/zeroclaw-labs/zeroclaw.git
   cd zeroclaw
   ./bootstrap.sh --prefer-prebuilt  # 低资源系统推荐
   ```

4. **Cargo Install**:
   ```bash
   cargo install zeroclaw
   ```

5. **Docker**:
   ```bash
   docker compose up -d
   ```

### 首次配置流程

1. 运行 onboard: `zeroclaw onboard --interactive`
2. 或非交互式: `zeroclaw onboard --api-key YOUR_KEY --provider openrouter`
3. 生成配置文件: `~/.zeroclaw/config.toml`
4. 测试: `zeroclaw agent -m "Hello from ZeroClaw!"`

### 运行模式

- 交互式对话: `zeroclaw agent`
- 后台守护: `zeroclaw daemon`
- Web 网关: `zeroclaw gateway` (默认 http://127.0.0.1:42617)
- 系统服务: `zeroclaw service install && zeroclaw service start`

### 2025-2026 更新要点

- 最新稳定版 v0.1.7（2026年2月）
- 新增 Android ARM 目标支持
- 更多 Provider（Groq, DeepSeek, Ollama）
- 加密密钥存储
- 单二进制部署

### 社区最佳实践

- VPS 推荐用于 24/7 运行，隔离性好
- 低资源系统使用 `--prefer-prebuilt` 避免编译
- 安全建议：保持 `workspace_only = true`
- 本地模型：配置 `default_provider = "ollama"` + `default_model = "llama3"`
