# Phase 2: ZeroClaw 快速上手 - 知识点列表

> 目标：作为用户掌握 ZeroClaw 的安装、配置和基本使用
> 学习时长：第 3-4 周
> 前置要求：Phase 1 完成

---

## 知识点列表

### 09. ZeroClaw 安装与环境配置
- cargo install、一键安装脚本、Homebrew、环境变量配置
- 前端类比：npm install -g
- ZeroClaw 场景：多种安装方式、版本管理、PATH 配置

### 10. Onboarding Wizard 与首次配置
- 交互式引导、Provider 选择、API Key 输入、config.toml 生成
- 前端类比：create-react-app 脚手架
- ZeroClaw 场景：首次运行 `zeroclaw onboard` 的完整流程

### 11. CLI 基础命令
- agent、daemon、config、channel、tools、doctor 子命令
- 前端类比：npm scripts（build/start/test）
- ZeroClaw 场景：日常使用的核心命令集

### 12. 第一个 Agent 对话
- CLI 模式交互、单次消息、多轮对话、系统提示词
- 前端类比：Postman 发 API 请求
- ZeroClaw 场景：`zeroclaw agent --message "..."` 的完整流程

### 13. Provider 配置与切换
- Ollama 本地模型、OpenRouter、Anthropic、自定义 endpoint
- 前端类比：切换 API 后端地址 (baseURL)
- ZeroClaw 场景：config.toml 中 [provider] 配置、免费 vs 付费方案

### 14. Channel 配对（Telegram 实战）
- 配对码机制、BotFather 创建 Bot、Webhook 设置、消息测试
- 前端类比：OAuth 扫码授权
- ZeroClaw 场景：完整的 Telegram Channel 配置实战

### 15. Gateway 启动与管理
- 端口配置、Webhook 路由、健康检查、localhost 绑定
- 前端类比：Express 服务器启动 (app.listen)
- ZeroClaw 场景：`zeroclaw gateway --port 18789` 的完整流程

### 16. 基础故障排查
- 日志查看、常见错误（API Key/网络/配置）、doctor 命令、社区求助
- 前端类比：Chrome DevTools 调试
- ZeroClaw 场景：`zeroclaw doctor`、日志级别、常见问题解决方案
