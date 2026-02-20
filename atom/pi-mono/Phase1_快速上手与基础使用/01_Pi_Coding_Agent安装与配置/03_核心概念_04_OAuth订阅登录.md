# 核心概念 04：OAuth 订阅登录

> **知识点定位**：深入理解 Pi Coding Agent 的 OAuth 认证机制，掌握订阅登录流程、Token 管理和会话持久化

---

## 一、OAuth 认证概述

### 1.1 什么是 OAuth 订阅登录

OAuth 订阅登录允许使用现有的 LLM 订阅服务（如 Claude Pro、ChatGPT Plus）来使用 Pi Coding Agent，无需额外的 API 费用。

**核心优势：**
- ✅ **零额外成本**：使用已有订阅，无需单独购买 API 额度
- ✅ **简化配置**：无需管理 API Key，浏览器登录即可
- ✅ **自动刷新**：Token 过期自动续期，无需手动干预
- ✅ **多设备同步**：Token 存储在本地，可在多台设备使用

**适用场景：**
- 个人开发者已有 Claude Pro/ChatGPT Plus 订阅
- 学习和实验，不需要高频 API 调用
- 不想管理 API Key 的用户

> **来源**：[Pi Providers Documentation - Subscriptions](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/providers.md#subscriptions) (2026-02)

### 1.2 支持的订阅服务

| Provider | 订阅名称 | 月费 | 特点 |
|----------|---------|------|------|
| **Anthropic** | Claude Pro | $20 | 优先访问 Claude Opus 4 |
| **Anthropic** | Claude Max | $200 | 更高速率限制 + 优先支持 |
| **OpenAI** | ChatGPT Plus | $20 | 访问 GPT-4o、o1 |
| **OpenAI** | ChatGPT Pro (Codex) | $200 | 无限 o1、更高速率 |
| **GitHub** | GitHub Copilot | $10 | 访问 GPT-4o、Claude Sonnet |
| **Google** | Gemini CLI | 免费 | 标准 Gemini 模型 |
| **Google** | Antigravity | 免费 | Gemini 3 + Claude + GPT-OSS |

> **注意**：Google 的两个服务都是免费的，但有速率限制

---

## 二、OAuth 登录流程详解

### 2.1 完整登录流程

```bash
# 1. 启动 Pi
pi

# 2. 输入登录命令
> /login

# 3. 选择 Provider
┌─────────────────────────────────────┐
│ Select provider:                    │
│ ✓ Anthropic Claude Pro/Max         │
│   OpenAI ChatGPT Plus/Pro           │
│   GitHub Copilot                    │
│   Google Gemini CLI                 │
│   Google Antigravity                │
└─────────────────────────────────────┘

# 4. 浏览器自动打开授权页面
Opening browser for authentication...
https://console.anthropic.com/oauth/authorize?...

# 5. 在浏览器中登录并授权
# (输入邮箱密码，点击"授权"按钮)

# 6. 返回终端，认证完成
✅ Logged in as user@example.com
✅ Token saved to ~/.pi/agent/auth.json
```

### 2.2 浏览器授权页面

**Anthropic 授权页面示例：**

```
┌─────────────────────────────────────────────┐
│ Anthropic                                   │
│                                             │
│ Pi Coding Agent 请求访问您的账户            │
│                                             │
│ 权限：                                      │
│ ✓ 使用 Claude API                          │
│ ✓ 访问您的订阅信息                          │
│                                             │
│ [授权] [取消]                               │
└─────────────────────────────────────────────┘
```

**OpenAI 授权页面示例：**

```
┌─────────────────────────────────────────────┐
│ OpenAI                                      │
│                                             │
│ Pi Coding Agent 请求访问                    │
│                                             │
│ 权限：                                      │
│ ✓ 使用 ChatGPT API (Codex)                 │
│ ✓ 读取您的订阅状态                          │
│                                             │
│ [允许] [拒绝]                               │
└─────────────────────────────────────────────┘
```

### 2.3 OAuth 流程技术细节

**标准 OAuth 2.0 授权码流程：**

```typescript
// 1. Pi 生成授权 URL
const authUrl = `https://provider.com/oauth/authorize?
  client_id=pi-coding-agent
  &redirect_uri=http://localhost:8080/callback
  &response_type=code
  &scope=api.read api.write
  &state=random_state_string`;

// 2. 打开浏览器
open(authUrl);

// 3. 启动本地服务器监听回调
const server = http.createServer((req, res) => {
  const code = new URL(req.url).searchParams.get('code');

  // 4. 用授权码换取 Access Token
  const tokens = await exchangeCodeForTokens(code);

  // 5. 保存 Token
  await saveTokens(tokens);

  res.end('Authentication successful! You can close this window.');
  server.close();
});

server.listen(8080);
```

**Token 交换请求：**

```http
POST /oauth/token HTTP/1.1
Host: provider.com
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code
&code=AUTH_CODE_FROM_CALLBACK
&client_id=pi-coding-agent
&client_secret=CLIENT_SECRET
&redirect_uri=http://localhost:8080/callback
```

**Token 响应：**

```json
{
  "access_token": "sk-ant-sid01-...",
  "refresh_token": "sk-ant-refresh-...",
  "expires_in": 3600,
  "token_type": "Bearer"
}
```

---

## 三、Token 存储机制

### 3.1 auth.json 文件结构

OAuth Token 存储在 `~/.pi/agent/auth.json`：

```json
{
  "anthropic": {
    "type": "oauth",
    "accessToken": "sk-ant-sid01-abcdef1234567890...",
    "refreshToken": "sk-ant-refresh-xyz9876543210...",
    "expiresAt": 1740000000000,
    "userId": "user@example.com",
    "scopes": ["api.read", "api.write"]
  },
  "openai": {
    "type": "oauth",
    "accessToken": "sess-...",
    "refreshToken": "refresh-...",
    "expiresAt": 1740003600000,
    "userId": "user@openai.com"
  }
}
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 认证类型，固定为 `"oauth"` |
| `accessToken` | string | 访问令牌，用于 API 调用 |
| `refreshToken` | string | 刷新令牌，用于获取新的 Access Token |
| `expiresAt` | number | 过期时间戳（毫秒） |
| `userId` | string | 用户标识（可选） |
| `scopes` | string[] | 授权范围（可选） |

### 3.2 文件权限与安全

```bash
# 检查文件权限
ls -la ~/.pi/agent/auth.json
# 输出：-rw------- 1 user staff 512 Feb 18 10:30 auth.json

# 权限说明
# -rw------- = 0600
# - 所有者可读写
# - 组用户无权限
# - 其他用户无权限
```

**安全特性：**
- ✅ Pi 自动设置文件权限为 `0600`
- ✅ 仅当前用户可读写
- ✅ Token 不会被其他用户或进程访问
- ✅ 建议将 `~/.pi/` 目录加入 `.gitignore`

### 3.3 Token 存储位置

**默认位置：**
```
~/.pi/agent/auth.json
```

**完整路径示例：**
- macOS/Linux: `/Users/username/.pi/agent/auth.json`
- Windows: `C:\Users\username\.pi\agent\auth.json`

**自定义位置（不推荐）：**
```bash
# 通过环境变量指定
export PI_AUTH_FILE=/custom/path/auth.json
pi
```

---

## 四、Token 自动刷新机制

### 4.1 刷新触发条件

Pi 会在以下情况自动刷新 Token：

1. **Token 即将过期**：距离过期时间少于 5 分钟
2. **API 调用失败**：收到 `401 Unauthorized` 响应
3. **启动时检查**：Pi 启动时检查 Token 有效性

### 4.2 刷新流程

```typescript
// Pi 内部刷新逻辑（简化版）
async function refreshTokenIfNeeded(provider: string) {
  const auth = await loadAuth(provider);

  // 检查是否需要刷新
  const now = Date.now();
  const expiresIn = auth.expiresAt - now;

  if (expiresIn < 5 * 60 * 1000) { // 少于 5 分钟
    console.log('Token expiring soon, refreshing...');

    // 调用 Provider 的刷新端点
    const response = await fetch('https://provider.com/oauth/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'refresh_token',
        refresh_token: auth.refreshToken,
        client_id: 'pi-coding-agent',
        client_secret: CLIENT_SECRET
      })
    });

    const newTokens = await response.json();

    // 更新 auth.json
    await saveAuth(provider, {
      ...auth,
      accessToken: newTokens.access_token,
      refreshToken: newTokens.refresh_token || auth.refreshToken,
      expiresAt: Date.now() + newTokens.expires_in * 1000
    });

    console.log('✅ Token refreshed successfully');
  }
}
```

### 4.3 刷新失败处理

**场景 1：Refresh Token 过期**

```
❌ Error: Refresh token expired
Please log in again: /login
```

**解决方案：**
```bash
pi
/logout
/login
```

**场景 2：网络错误**

```
❌ Error: Failed to refresh token (network error)
Retrying in 5 seconds...
```

**解决方案：**
- Pi 会自动重试 3 次
- 如果仍失败，提示用户检查网络

**场景 3：订阅过期**

```
❌ Error: Subscription expired
Please renew your subscription at https://provider.com/billing
```

**解决方案：**
- 续费订阅
- 或切换到 API Key 认证

---

## 五、会话持久化

### 5.1 跨会话 Token 复用

```bash
# 第一次登录
pi
/login
# 选择 Anthropic
✅ Logged in as user@example.com

# 退出 Pi
/quit

# 第二次启动（无需重新登录）
pi
# 自动使用保存的 Token
✅ Using saved credentials for Anthropic
```

### 5.2 多设备同步

**方案 1：手动同步 auth.json**

```bash
# 设备 A
scp ~/.pi/agent/auth.json user@device-b:~/.pi/agent/

# 设备 B
# 确保文件权限正确
chmod 600 ~/.pi/agent/auth.json
pi
```

**方案 2：使用云同步（不推荐）**

```bash
# 使用 Dropbox/iCloud 同步 ~/.pi/ 目录
# ⚠️ 安全风险：Token 可能被云服务访问
```

**最佳实践：**
- ✅ 每台设备独立登录
- ❌ 不要通过云同步共享 Token
- ✅ 使用密钥管理工具（如 1Password）存储 Refresh Token

### 5.3 Token 生命周期

| Provider | Access Token 有效期 | Refresh Token 有效期 |
|----------|-------------------|---------------------|
| Anthropic | 1 小时 | 30 天 |
| OpenAI | 1 小时 | 90 天 |
| GitHub Copilot | 8 小时 | 无限期（直到撤销） |
| Google | 1 小时 | 无限期（直到撤销） |

**注意事项：**
- Access Token 短期有效，频繁刷新
- Refresh Token 长期有效，需妥善保管
- 如果 30 天未使用，Anthropic Refresh Token 会过期

---

## 六、OAuth vs API Key 对比

### 6.1 成本对比

**OAuth 订阅：**
```
Claude Pro: $20/月
- 无限制使用（受订阅限制）
- 适合：每天 50-200 次对话

ChatGPT Plus: $20/月
- 无限制使用（受速率限制）
- 适合：每天 100-300 次对话
```

**API Key：**
```
Claude Opus 4:
- 输入：$15 / 1M tokens
- 输出：$75 / 1M tokens
- 适合：高频使用、生产环境

GPT-4o:
- 输入：$2.50 / 1M tokens
- 输出：$10 / 1M tokens
- 适合：成本敏感场景
```

### 6.2 速率限制对比

| Provider | OAuth 订阅限制 | API Key 限制 |
|----------|---------------|-------------|
| **Anthropic** | Claude Pro: 每 5 小时 45 条消息 | Tier 1: 50 RPM, 40K TPM |
| **OpenAI** | ChatGPT Plus: 每 3 小时 80 条消息 | Tier 1: 500 RPM, 30K TPM |
| **GitHub Copilot** | 未公开（较宽松） | N/A |

**RPM**: Requests Per Minute（每分钟请求数）
**TPM**: Tokens Per Minute（每分钟 Token 数）

### 6.3 使用场景建议

**选择 OAuth 订阅：**
- ✅ 个人开发和学习
- ✅ 已有订阅，不想额外付费
- ✅ 低频使用（每天 < 100 次对话）
- ✅ 不需要精确成本控制

**选择 API Key：**
- ✅ 生产环境部署
- ✅ 高频使用（每天 > 200 次对话）
- ✅ 需要精确成本控制
- ✅ 团队协作（多人共享 API Key）
- ✅ CI/CD 自动化

---

## 七、特定 Provider 配置

### 7.1 GitHub Copilot 配置

```bash
pi
/login
# 选择 GitHub Copilot

# 输入 GitHub 域名
Enter GitHub domain (press Enter for github.com):
> <直接回车>

# 或输入企业域名
> github.enterprise.com

# 浏览器打开授权页面
# 登录 GitHub 并授权
```

**模型启用检查：**

如果遇到 "model not supported" 错误：

```
1. 打开 VS Code
2. 打开 Copilot Chat
3. 点击模型选择器
4. 选择要使用的模型（如 GPT-4o）
5. 点击 "Enable"
6. 返回 Pi 重试
```

### 7.2 Google Providers 配置

**Gemini CLI（免费）：**

```bash
pi
/login
# 选择 Google Gemini CLI

# 浏览器打开 Google 登录
# 使用任何 Google 账户登录
✅ Logged in with free Gemini CLI access
```

**Antigravity（免费沙箱）：**

```bash
pi
/login
# 选择 Google Antigravity

# 访问沙箱环境
✅ Access to Gemini 3, Claude, GPT-OSS models
```

**付费 Cloud Code Assist：**

```bash
# 设置项目 ID
export GOOGLE_CLOUD_PROJECT=your-project-id

pi
/login
# 选择 Google Gemini CLI
✅ Using paid Cloud Code Assist
```

### 7.3 OpenAI Codex 配置

```bash
pi
/login
# 选择 OpenAI ChatGPT Plus/Pro

# 浏览器打开 OpenAI 登录
# 使用 ChatGPT Plus/Pro 账户登录
✅ Logged in with Codex access

# 可用模型
- gpt-4o
- o1
- o3-mini (如果有 Pro 订阅)
```

**注意事项：**
- Codex 仅供个人使用
- 生产环境请使用 OpenAI Platform API
- Pro 订阅提供无限 o1 访问

---

## 八、登出与撤销

### 8.1 登出操作

```bash
pi
/logout

┌─────────────────────────────────────┐
│ Select provider to logout:          │
│ ✓ Anthropic (user@example.com)     │
│   OpenAI (user@openai.com)          │
└─────────────────────────────────────┘

✅ Logged out from Anthropic
✅ Token removed from auth.json
```

### 8.2 撤销授权

**Anthropic：**
```
1. 访问 https://console.anthropic.com/settings/oauth
2. 找到 "Pi Coding Agent"
3. 点击 "Revoke Access"
```

**OpenAI：**
```
1. 访问 https://platform.openai.com/account/api-keys
2. 找到 "Pi Coding Agent"
3. 点击 "Revoke"
```

**GitHub：**
```
1. 访问 https://github.com/settings/applications
2. 找到 "Pi Coding Agent"
3. 点击 "Revoke"
```

### 8.3 清理本地 Token

```bash
# 手动删除 auth.json
rm ~/.pi/agent/auth.json

# 或只删除特定 Provider
# 编辑 auth.json，删除对应条目
```

---

## 九、故障排查

### 9.1 登录失败

**问题 1：浏览器未打开**

```
❌ Error: Failed to open browser
```

**解决方案：**
```bash
# 手动复制 URL 到浏览器
# Pi 会显示完整的授权 URL
https://console.anthropic.com/oauth/authorize?...
```

**问题 2：回调超时**

```
❌ Error: OAuth callback timeout
```

**解决方案：**
```bash
# 检查防火墙是否阻止 localhost:8080
# 或手动指定端口
export PI_OAUTH_PORT=9090
pi
/login
```

### 9.2 Token 刷新失败

**问题：Refresh Token 无效**

```
❌ Error: Invalid refresh token
```

**解决方案：**
```bash
# 重新登录
pi
/logout
/login
```

### 9.3 订阅状态检查

```bash
# 检查订阅是否有效
pi
/session

# 查看当前认证状态
cat ~/.pi/agent/auth.json | jq '.anthropic'
```

---

## 十、最佳实践

### 10.1 安全建议

1. **定期检查授权应用**
   ```bash
   # 每月检查一次授权列表
   # 撤销不再使用的应用
   ```

2. **不要共享 auth.json**
   ```bash
   # 添加到 .gitignore
   echo "~/.pi/agent/auth.json" >> ~/.gitignore
   ```

3. **使用独立账户**
   ```bash
   # 为开发工具创建独立的 Google/GitHub 账户
   # 避免使用主账户
   ```

### 10.2 多账户管理

```bash
# 方案 1：使用不同的 Provider
# Anthropic 用于工作
# OpenAI 用于个人项目

# 方案 2：使用环境变量切换
export PI_AUTH_FILE=~/.pi/work-auth.json
pi

export PI_AUTH_FILE=~/.pi/personal-auth.json
pi
```

### 10.3 成本优化

```bash
# 策略 1：OAuth 用于开发，API Key 用于生产
# 开发环境
pi  # 使用 OAuth

# 生产环境
export ANTHROPIC_API_KEY=sk-ant-...
pi

# 策略 2：混合使用
# 复杂任务用 Opus 4 (OAuth)
# 简单任务用 Sonnet 4 (API Key)
```

---

## 十一、总结

### 11.1 核心要点

1. **OAuth 登录**：`/login` 命令，浏览器授权，自动保存 Token
2. **Token 存储**：`~/.pi/agent/auth.json`，权限 `0600`，自动刷新
3. **会话持久化**：Token 跨会话复用，无需重复登录
4. **支持 5 个订阅服务**：Claude Pro/Max、ChatGPT Plus/Pro、GitHub Copilot、Google Gemini/Antigravity
5. **自动刷新**：Token 过期前 5 分钟自动刷新

### 11.2 OAuth vs API Key

| 维度 | OAuth 订阅 | API Key |
|------|-----------|---------|
| 成本 | 固定月费 $10-200 | 按使用量付费 |
| 配置 | 简单（/login） | 中等（管理密钥） |
| 速率 | 订阅限制 | API 限制（更高） |
| 场景 | 个人开发 | 生产环境 |

### 11.3 下一步

- 学习 **环境变量配置**（核心概念 05）
- 掌握 **项目级配置**（核心概念 06）
- 了解 **多 Provider 切换**（核心概念 08）

---

**参考资料：**
- [Pi Providers Documentation - Subscriptions](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/providers.md#subscriptions)
- [Pi OAuth Implementation](https://github.com/badlogic/pi-mono/blob/main/packages/ai/src/oauth.ts)
- [Pi Auth Storage](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/auth-storage.ts)

**文档版本**：v1.0 (2026-02-18)
