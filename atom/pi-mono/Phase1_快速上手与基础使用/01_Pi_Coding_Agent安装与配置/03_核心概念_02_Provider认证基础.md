# 核心概念 02：Provider 认证基础

> **知识点定位**：理解 Pi Coding Agent 的多 Provider 认证体系，掌握 API Key 和 OAuth 两种认证方式

---

## 一、认证体系概览

Pi Coding Agent 支持 **15+ LLM 提供商**，提供两种认证方式：

### 1.1 两种认证方式对比

| 认证方式 | 适用场景 | 优势 | 劣势 |
|---------|---------|------|------|
| **API Key** | 开发者、企业用户 | 灵活控制、按需付费、无限制 | 需要付费账户、管理密钥 |
| **OAuth 订阅** | 个人用户 | 使用现有订阅（Claude Pro/ChatGPT Plus）、无需额外费用 | 受订阅限制、速率限制 |

### 1.2 支持的 Provider 列表

**OAuth 订阅认证（5个）：**
- Anthropic Claude Pro/Max
- OpenAI ChatGPT Plus/Pro (Codex)
- GitHub Copilot
- Google Gemini CLI
- Google Antigravity

**API Key 认证（17个）：**
- Anthropic
- OpenAI
- Azure OpenAI
- Google Gemini
- Google Vertex AI
- Amazon Bedrock
- Mistral
- Groq
- Cerebras
- xAI (Grok)
- OpenRouter
- Vercel AI Gateway
- ZAI
- OpenCode Zen
- Hugging Face
- Kimi For Coding
- MiniMax

> **来源**：[Pi Coding Agent README](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/README.md) (2026-02)

---

## 二、API Key 认证详解

### 2.1 环境变量配置

最简单的认证方式是设置环境变量：

```bash
# Anthropic Claude
export ANTHROPIC_API_KEY=sk-ant-api03-...

# OpenAI GPT
export OPENAI_API_KEY=sk-proj-...

# xAI Grok
export XAI_API_KEY=xai-...

# Google Gemini
export GEMINI_API_KEY=...

# Groq
export GROQ_API_KEY=gsk_...
```

**完整环境变量列表：**

| Provider | 环境变量名 | 示例前缀 |
|----------|-----------|---------|
| Anthropic | `ANTHROPIC_API_KEY` | `sk-ant-` |
| OpenAI | `OPENAI_API_KEY` | `sk-proj-` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` | 自定义 |
| Google Gemini | `GEMINI_API_KEY` | `AI...` |
| Mistral | `MISTRAL_API_KEY` | `...` |
| Groq | `GROQ_API_KEY` | `gsk_` |
| Cerebras | `CEREBRAS_API_KEY` | `...` |
| xAI | `XAI_API_KEY` | `xai-` |
| OpenRouter | `OPENROUTER_API_KEY` | `sk-or-` |
| Vercel AI Gateway | `AI_GATEWAY_API_KEY` | `...` |
| ZAI | `ZAI_API_KEY` | `...` |
| OpenCode Zen | `OPENCODE_API_KEY` | `...` |
| Hugging Face | `HF_TOKEN` | `hf_` |
| Kimi For Coding | `KIMI_API_KEY` | `...` |
| MiniMax | `MINIMAX_API_KEY` | `...` |

> **来源**：[Pi Providers Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/providers.md) (2026-02)

### 2.2 Shell 配置持久化

**macOS/Linux (Bash)：**

```bash
# 编辑 ~/.bashrc 或 ~/.bash_profile
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc
source ~/.bashrc
```

**macOS/Linux (Zsh)：**

```bash
# 编辑 ~/.zshrc
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.zshrc
source ~/.zshrc
```

**Windows (PowerShell)：**

```powershell
# 临时设置
$env:ANTHROPIC_API_KEY="sk-ant-..."

# 永久设置（用户级别）
[System.Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY', 'sk-ant-...', 'User')
```

### 2.3 auth.json 文件配置

Pi 支持将 API Key 存储在 `~/.pi/agent/auth.json` 文件中：

```json
{
  "anthropic": {
    "type": "api_key",
    "key": "sk-ant-api03-..."
  },
  "openai": {
    "type": "api_key",
    "key": "sk-proj-..."
  },
  "xai": {
    "type": "api_key",
    "key": "xai-..."
  },
  "google": {
    "type": "api_key",
    "key": "AI..."
  }
}
```

**auth.json 的优势：**
- ✅ 集中管理多个 Provider 的密钥
- ✅ 文件权限自动设置为 `0600`（仅用户可读写）
- ✅ 优先级高于环境变量
- ✅ 支持高级密钥解析（见下文）

### 2.4 高级密钥解析

`auth.json` 的 `key` 字段支持三种格式：

#### 2.4.1 Shell 命令（推荐用于密钥管理工具）

```json
{
  "anthropic": {
    "type": "api_key",
    "key": "!security find-generic-password -ws 'anthropic'"
  }
}
```

**macOS Keychain 示例：**

```bash
# 存储密钥到 Keychain
security add-generic-password -a "$USER" -s "anthropic" -w "sk-ant-..."

# auth.json 配置
{
  "anthropic": {
    "type": "api_key",
    "key": "!security find-generic-password -ws 'anthropic'"
  }
}
```

**1Password CLI 示例：**

```json
{
  "openai": {
    "type": "api_key",
    "key": "!op read 'op://vault/OpenAI/credential'"
  }
}
```

> **注意**：命令输出会被缓存到进程生命周期结束，避免重复执行

#### 2.4.2 环境变量引用

```json
{
  "anthropic": {
    "type": "api_key",
    "key": "MY_ANTHROPIC_KEY"
  }
}
```

Pi 会自动读取 `$MY_ANTHROPIC_KEY` 环境变量的值。

#### 2.4.3 字面值（直接存储）

```json
{
  "anthropic": {
    "type": "api_key",
    "key": "sk-ant-api03-..."
  }
}
```

直接存储密钥，适合开发环境。

### 2.5 密钥解析优先级

Pi 按以下顺序解析 API Key：

1. **CLI 参数** `--api-key` 标志
2. **auth.json 文件** `~/.pi/agent/auth.json`
3. **环境变量** `ANTHROPIC_API_KEY` 等
4. **自定义 Provider** `models.json` 中的密钥

> **来源**：[Pi Providers Documentation - Resolution Order](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/providers.md#resolution-order) (2026-02)

---

## 三、OAuth 订阅认证详解

### 3.1 OAuth 认证流程

OAuth 认证允许使用现有订阅（如 Claude Pro、ChatGPT Plus）：

```bash
# 启动 pi
pi

# 在交互模式中输入
/login
```

**交互流程：**

```
1. 选择 Provider
   → Anthropic Claude Pro/Max
   → OpenAI ChatGPT Plus/Pro (Codex)
   → GitHub Copilot
   → Google Gemini CLI
   → Google Antigravity

2. 浏览器自动打开授权页面

3. 登录并授权

4. 返回终端，认证完成
```

### 3.2 Token 存储机制

OAuth Token 存储在 `~/.pi/agent/auth.json`：

```json
{
  "anthropic": {
    "type": "oauth",
    "accessToken": "sk-ant-sid01-...",
    "refreshToken": "...",
    "expiresAt": 1740000000000
  }
}
```

**Token 管理特性：**
- ✅ **自动刷新**：Token 过期时自动使用 `refreshToken` 刷新
- ✅ **安全存储**：文件权限 `0600`，仅用户可读写
- ✅ **跨会话持久化**：重启 pi 无需重新登录
- ✅ **多 Provider 共存**：同时存储多个 Provider 的 Token

> **来源**：[Pi Providers Documentation - Subscriptions](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/providers.md#subscriptions) (2026-02)

### 3.3 登出操作

```bash
pi
/logout
```

这会清除 `auth.json` 中的 OAuth Token，但保留 API Key 配置。

### 3.4 OAuth vs API Key 对比

| 维度 | OAuth 订阅 | API Key |
|------|-----------|---------|
| **成本** | 使用现有订阅（$20-200/月） | 按使用量付费（$0.003-0.075/1K tokens） |
| **速率限制** | 订阅限制（如 Claude Pro 每日消息数） | API 限制（RPM/TPM） |
| **模型访问** | 订阅包含的模型 | 所有 API 可用模型 |
| **使用场景** | 个人开发、学习 | 生产环境、高频使用 |
| **配置复杂度** | 简单（/login） | 中等（管理密钥） |

---

## 四、云服务 Provider 配置

### 4.1 Azure OpenAI

```bash
# 必需配置
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_BASE_URL=https://your-resource.openai.azure.com

# 或使用资源名称
export AZURE_OPENAI_RESOURCE_NAME=your-resource

# 可选配置
export AZURE_OPENAI_API_VERSION=2024-02-01
export AZURE_OPENAI_DEPLOYMENT_NAME_MAP=gpt-4=my-gpt4,gpt-4o=my-gpt4o
```

### 4.2 Amazon Bedrock

**方式 1：AWS Profile**

```bash
export AWS_PROFILE=your-profile
```

**方式 2：IAM Keys**

```bash
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-west-2  # 可选，默认 us-east-1
```

**方式 3：Bearer Token**

```bash
export AWS_BEARER_TOKEN_BEDROCK=...
```

**使用示例：**

```bash
pi --provider amazon-bedrock --model us.anthropic.claude-sonnet-4-20250514-v1:0
```

### 4.3 Google Vertex AI

使用 Application Default Credentials：

```bash
# 登录
gcloud auth application-default login

# 配置项目
export GOOGLE_CLOUD_PROJECT=your-project
export GOOGLE_CLOUD_LOCATION=us-central1
```

或使用服务账户：

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

> **来源**：[Pi Providers Documentation - Cloud Providers](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/providers.md#cloud-providers) (2026-02)

---

## 五、安全最佳实践

### 5.1 密钥存储安全

**❌ 不安全的做法：**

```bash
# 直接在命令行中暴露密钥
pi --api-key sk-ant-...

# 在代码中硬编码
const apiKey = "sk-ant-..."

# 提交到 Git
git add .env
```

**✅ 安全的做法：**

```bash
# 使用环境变量
export ANTHROPIC_API_KEY=sk-ant-...

# 使用密钥管理工具
security add-generic-password -a "$USER" -s "anthropic" -w "sk-ant-..."

# 使用 auth.json + .gitignore
echo "~/.pi/agent/auth.json" >> ~/.gitignore
```

### 5.2 文件权限检查

```bash
# 检查 auth.json 权限
ls -la ~/.pi/agent/auth.json
# 应该显示：-rw------- (0600)

# 如果权限不正确，手动修复
chmod 600 ~/.pi/agent/auth.json
```

### 5.3 密钥轮换策略

**定期轮换密钥（推荐每 90 天）：**

```bash
# 1. 生成新密钥（在 Provider 控制台）
# 2. 更新环境变量或 auth.json
# 3. 测试新密钥
pi --model claude-opus-4
# 4. 撤销旧密钥
```

### 5.4 团队协作安全

**不要共享个人 API Key，使用团队方案：**

- **Anthropic**：使用 Anthropic Workspaces
- **OpenAI**：使用 Organization API Keys
- **企业方案**：使用 Azure OpenAI 或 Amazon Bedrock

---

## 六、常见问题排查

### 6.1 认证失败

**问题：** `Authentication failed: Invalid API key`

**排查步骤：**

```bash
# 1. 检查环境变量是否设置
echo $ANTHROPIC_API_KEY

# 2. 检查密钥格式
# Anthropic: sk-ant-api03-...
# OpenAI: sk-proj-...

# 3. 检查 auth.json
cat ~/.pi/agent/auth.json

# 4. 测试密钥有效性
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-opus-4","max_tokens":10,"messages":[{"role":"user","content":"Hi"}]}'
```

### 6.2 OAuth Token 过期

**问题：** `OAuth token expired`

**解决方案：**

```bash
# 重新登录
pi
/logout
/login
```

### 6.3 Provider 不可用

**问题：** `Provider not found: anthropic`

**排查步骤：**

```bash
# 1. 检查 pi 版本
pi --version

# 2. 更新到最新版本
npm update -g @mariozechner/pi-coding-agent

# 3. 检查支持的 Provider
pi --help
```

---

## 七、实战示例

### 7.1 多 Provider 配置

```json
// ~/.pi/agent/auth.json
{
  "anthropic": {
    "type": "api_key",
    "key": "!security find-generic-password -ws 'anthropic'"
  },
  "openai": {
    "type": "oauth",
    "accessToken": "...",
    "refreshToken": "...",
    "expiresAt": 1740000000000
  },
  "xai": {
    "type": "api_key",
    "key": "XAI_KEY"
  },
  "google": {
    "type": "api_key",
    "key": "AI..."
  }
}
```

### 7.2 快速切换 Provider

```bash
# 启动时指定 Provider
pi --provider anthropic --model claude-opus-4

# 交互模式中切换
/model
# 选择不同 Provider 的模型
```

### 7.3 CI/CD 环境配置

```yaml
# .github/workflows/test.yml
name: Test with Pi
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '20'
      - name: Install Pi
        run: npm install -g @mariozechner/pi-coding-agent
      - name: Run tests
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          pi --print "Run all tests and report results"
```

---

## 八、总结

### 8.1 核心要点

1. **两种认证方式**：API Key（灵活付费）和 OAuth（使用订阅）
2. **15+ Provider 支持**：覆盖主流 LLM 服务商
3. **三种配置方式**：环境变量、auth.json、CLI 参数
4. **安全优先**：文件权限、密钥管理工具、定期轮换
5. **自动 Token 刷新**：OAuth Token 过期自动续期

### 8.2 最佳实践

- ✅ 使用密钥管理工具（Keychain、1Password）
- ✅ 设置 `auth.json` 权限为 `0600`
- ✅ 定期轮换 API Key（90 天）
- ✅ 团队使用企业方案，不共享个人密钥
- ✅ CI/CD 使用 Secrets 管理密钥

### 8.3 下一步

- 学习 **首次运行与交互**（核心概念 03）
- 掌握 **OAuth 订阅登录**（核心概念 04）
- 了解 **环境变量配置**（核心概念 05）

---

**参考资料：**
- [Pi Coding Agent README](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/README.md)
- [Pi Providers Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/providers.md)
- [Pi NPM Package](https://www.npmjs.com/package/@mariozechner/pi-coding-agent)

**文档版本**：v1.0 (2026-02-18)
