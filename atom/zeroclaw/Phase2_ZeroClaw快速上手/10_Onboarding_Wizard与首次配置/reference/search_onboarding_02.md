---
type: search_result
search_query: ZeroClaw AI agent setup configuration onboarding experience Reddit
search_engine: grok-mcp
searched_at: 2026-03-11
knowledge_point: 10_Onboarding_Wizard与首次配置
---

# 搜索结果：ZeroClaw 社区 Onboarding 体验

## 搜索摘要

Reddit 社区（r/LocalLLaMA, r/openclaw, r/zeroclawlabs）对 ZeroClaw onboarding 体验评价积极，主要亮点是快速（<1分钟）、轻量（3-5MB 二进制，~5MB RAM）。

## 关键信息提取

### 社区评价亮点

1. **安装体验**: "under a minute, feels like magic" - 从安装到首次对话不到5分钟
2. **硬件要求极低**: 可在 Raspberry Pi 2、Android/Termux 上运行
3. **安全性突出**: Rust 沙箱 + 显式 allow-lists（命令/目录白名单）
4. **对比 OpenClaw**: 启动快 400×，内存小 GBs → ~5MB
5. **TUI Wizard**: 文本菜单式交互，被称为 2026 "killer feature"

### 典型安装路径（Reddit 验证）
1. 安装 Rust（如果从源码编译）或用预编译二进制
2. 运行安装器或 clone + bootstrap
3. `zeroclaw onboard` - "不到一分钟，感觉像魔法"
4. 添加 channels（WhatsApp, Telegram 等）
5. `zeroclaw gateway --port 8080` 或 daemon 模式
6. `zeroclaw skills install clawhub:xxx`（带安全审计）

### WhatsApp Channel 配置示例（来自 Reddit）
```toml
[channels_config.whatsapp]
access_token = "EAABx..."
phone_number_id = "123456789012345"
verify_token = "my-secret-verify-token"
allowed_numbers = ["+1234567890"]
```

### 常见问题
- 源码编译时某些模型（Kimi/GLM）可能有问题
- Telegram bot 偶尔卡住
- Shell 命令需要额外调整
- 解决方案：使用预编译二进制 或 Ollama 本地模型

### Skills 系统
- 通过 `zeroclaw skills install clawhub:xxx` 安装
- 内置安全审计
- 生态系统正在快速发展

### 性能数据
- 启动时间: <10ms
- 二进制大小: 3-9MB
- 内存占用: ~5MB RAM
- 启动速度比 OpenClaw 快 400×

## 相关链接
- r/LocalLLaMA - OpenClaw vs ZeroClaw 对比讨论
- r/openclaw - ZeroClaw runtime 介绍帖
- r/zeroclawlabs - 官方社区 welcome post + newbie guide
