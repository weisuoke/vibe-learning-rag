# 实战代码：OAuth集成实战

## 场景描述

**需求：** 集成企业内部LLM服务，需要OAuth 2.0 Device Code Flow认证。

**背景：**
- 企业内部部署的LLM服务
- 使用企业SSO（Single Sign-On）
- 不能使用浏览器重定向（CLI环境）
- 需要Token自动刷新

**技术要求：**
- OAuth 2.0 Device Code Flow (RFC 8628)
- Token存储在`~/.pi/agent/auth.json`
- 自动Token刷新
- 支持`/login`命令

## OAuth Device Code Flow原理

### 流程图

```
用户                    pi-agent                OAuth服务器
 |                         |                         |
 |  /login company-llm     |                         |
 |------------------------>|                         |
 |                         |  请求设备代码            |
 |                         |------------------------>|
 |                         |<------------------------|
 |                         |  device_code, user_code |
 |                         |                         |
 |  显示user_code          |                         |
 |<------------------------|                         |
 |                         |                         |
 |  访问验证URL            |                         |
 |  输入user_code          |                         |
 |---------------------------------------->|         |
 |                         |                         |
 |                         |  轮询Token              |
 |                         |------------------------>|
 |                         |<------------------------|
 |                         |  authorization_pending  |
 |                         |                         |
 |  授权成功               |                         |
 |<----------------------------------------|         |
 |                         |                         |
 |                         |  轮询Token              |
 |                         |------------------------>|
 |                         |<------------------------|
 |                         |  access_token, refresh  |
 |                         |                         |
 |  登录成功               |                         |
 |<------------------------|                         |
```

### 关键步骤

1. **请求设备代码**：pi-agent向OAuth服务器请求device_code和user_code
2. **显示用户代码**：pi显示user_code和验证URL给用户
3. **用户授权**：用户在浏览器中访问验证URL，输入user_code
4. **轮询Token**：pi-agent轮询OAuth服务器，等待用户授权
5. **获取Token**：授权成功后，获取access_token和refresh_token
6. **存储凭证**：将Token存储到`~/.pi/agent/auth.json`

## 完整实现

### 步骤1：创建Extension目录

```bash
mkdir -p ~/.pi/agent/extensions/company-llm
cd ~/.pi/agent/extensions/company-llm
```

### 步骤2：初始化项目

```bash
# package.json
cat > package.json << 'EOF'
{
  "name": "company-llm-provider",
  "version": "1.0.0",
  "type": "module",
  "main": "index.ts",
  "dependencies": {
    "@mariozechner/pi-coding-agent": "latest",
    "@mariozechner/pi-ai": "latest"
  }
}
EOF

npm install
```

### 步骤3：实现OAuth Provider

```typescript
// index.ts
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import type {
  OAuthCredentials,
  OAuthLoginCallbacks,
} from "@mariozechner/pi-ai";

// OAuth配置
const OAUTH_CONFIG = {
  clientId: "pi-agent",
  authServer: "https://auth.company.com",
  scope: "llm.read llm.write",
};

// 请求设备代码
async function requestDeviceCode() {
  const response = await fetch(
    `${OAUTH_CONFIG.authServer}/device/code`,
    {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        client_id: OAUTH_CONFIG.clientId,
        scope: OAUTH_CONFIG.scope,
      }),
    }
  );

  if (!response.ok) {
    throw new Error(`Device code request failed: ${await response.text()}`);
  }

  const data = await response.json();
  return {
    device_code: data.device_code,
    user_code: data.user_code,
    verification_uri: data.verification_uri,
    verification_uri_complete: data.verification_uri_complete,
    expires_in: data.expires_in,
    interval: data.interval || 5,
  };
}

// 轮询Token
async function pollForToken(
  deviceCode: string,
  interval: number
): Promise<any> {
  let pollInterval = interval;

  while (true) {
    // 等待指定间隔
    await new Promise((resolve) => setTimeout(resolve, pollInterval * 1000));

    const response = await fetch(`${OAUTH_CONFIG.authServer}/token`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        grant_type: "urn:ietf:params:oauth:grant-type:device_code",
        device_code: deviceCode,
        client_id: OAUTH_CONFIG.clientId,
      }),
    });

    const data = await response.json();

    if (response.ok) {
      // 成功获取Token
      return data;
    } else if (data.error === "authorization_pending") {
      // 用户还未授权，继续轮询
      continue;
    } else if (data.error === "slow_down") {
      // 轮询太快，增加间隔
      pollInterval += 5;
      continue;
    } else if (data.error === "expired_token") {
      // 设备代码过期
      throw new Error("Device code expired. Please try again.");
    } else if (data.error === "access_denied") {
      // 用户拒绝授权
      throw new Error("Authorization denied by user.");
    } else {
      // 其他错误
      throw new Error(`Token request failed: ${data.error_description || data.error}`);
    }
  }
}

// 刷新Token
async function refreshAccessToken(refreshToken: string): Promise<any> {
  const response = await fetch(`${OAUTH_CONFIG.authServer}/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "refresh_token",
      refresh_token: refreshToken,
      client_id: OAUTH_CONFIG.clientId,
    }),
  });

  if (!response.ok) {
    throw new Error(`Token refresh failed: ${await response.text()}`);
  }

  return await response.json();
}

// OAuth Provider实现
const companyLLMOAuth = {
  name: "Company LLM (SSO)",

  async login(callbacks: OAuthLoginCallbacks): Promise<OAuthCredentials> {
    try {
      // 1. 请求设备代码
      const deviceCodeData = await requestDeviceCode();

      // 2. 显示用户代码
      callbacks.onDeviceCode({
        userCode: deviceCodeData.user_code,
        verificationUri: deviceCodeData.verification_uri,
      });

      // 可选：显示进度
      if (callbacks.onProgress) {
        callbacks.onProgress("Waiting for authorization...");
      }

      // 3. 轮询Token
      const tokenData = await pollForToken(
        deviceCodeData.device_code,
        deviceCodeData.interval
      );

      // 4. 返回凭证
      return {
        access: tokenData.access_token,
        refresh: tokenData.refresh_token,
        expires: Date.now() + tokenData.expires_in * 1000,
      };
    } catch (error) {
      throw new Error(
        `OAuth login failed: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  },

  async refreshToken(
    credentials: OAuthCredentials
  ): Promise<OAuthCredentials> {
    try {
      const tokenData = await refreshAccessToken(credentials.refresh);

      return {
        access: tokenData.access_token,
        refresh: tokenData.refresh_token || credentials.refresh,
        expires: Date.now() + tokenData.expires_in * 1000,
      };
    } catch (error) {
      throw new Error(
        `Token refresh failed: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  },

  getApiKey(credentials: OAuthCredentials): string {
    return credentials.access;
  },
};

// Extension入口
export default function (pi: ExtensionAPI) {
  pi.registerProvider("company-llm", {
    baseUrl: "https://ai.company.com/v1",
    api: "openai-completions",

    models: [
      {
        id: "company-gpt-4",
        name: "Company GPT-4 (Internal)",
        reasoning: true,
        input: ["text", "image"],
        contextWindow: 128000,
        maxTokens: 16384,
        cost: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
        },
      },
    ],

    oauth: companyLLMOAuth,
  });

  console.log("Company LLM Provider registered with OAuth support");
}
```

### 步骤4：配置Extension

```bash
# 在~/.pi/agent/settings.json中添加
{
  "extensions": [
    "~/.pi/agent/extensions/company-llm"
  ]
}
```

### 步骤5：测试OAuth登录

```bash
# 启动pi
pi

# 登录
/login company-llm

# 应该看到：
# Please visit: https://auth.company.com/device
# And enter code: ABCD-1234
# Waiting for authorization...

# 在浏览器中访问URL，输入代码
# 授权成功后，pi会显示：
# Login successful!

# 选择模型
/model company-llm/company-gpt-4

# 测试对话
Hello, can you hear me?
```

## 进阶功能

### 1. 添加超时处理

```typescript
async function pollForToken(
  deviceCode: string,
  interval: number,
  expiresIn: number = 900 // 15分钟
): Promise<any> {
  const startTime = Date.now();
  let pollInterval = interval;

  while (true) {
    // 检查超时
    if (Date.now() - startTime > expiresIn * 1000) {
      throw new Error("Device code expired. Please try again.");
    }

    await new Promise((resolve) => setTimeout(resolve, pollInterval * 1000));

    // ... 轮询逻辑
  }
}
```

### 2. 添加取消支持

```typescript
async function pollForToken(
  deviceCode: string,
  interval: number,
  signal?: AbortSignal
): Promise<any> {
  let pollInterval = interval;

  while (true) {
    // 检查取消
    if (signal?.aborted) {
      throw new Error("Login cancelled by user");
    }

    await new Promise((resolve) => setTimeout(resolve, pollInterval * 1000));

    // ... 轮询逻辑
  }
}

// 使用
const controller = new AbortController();
const tokenData = await pollForToken(
  deviceCode,
  interval,
  controller.signal
);
```

### 3. 添加重试机制

```typescript
async function requestWithRetry<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3
): Promise<T> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === maxRetries - 1) throw error;

      // 指数退避
      const delay = Math.pow(2, i) * 1000;
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
  throw new Error("Max retries exceeded");
}

// 使用
const deviceCodeData = await requestWithRetry(() => requestDeviceCode());
```

### 4. 添加日志

```typescript
const companyLLMOAuth = {
  async login(callbacks: OAuthLoginCallbacks): Promise<OAuthCredentials> {
    console.log("[OAuth] Starting login flow");

    const deviceCodeData = await requestDeviceCode();
    console.log(`[OAuth] Device code obtained: ${deviceCodeData.user_code}`);

    callbacks.onDeviceCode({
      userCode: deviceCodeData.user_code,
      verificationUri: deviceCodeData.verification_uri,
    });

    console.log("[OAuth] Polling for token...");
    const tokenData = await pollForToken(
      deviceCodeData.device_code,
      deviceCodeData.interval
    );

    console.log("[OAuth] Token obtained successfully");

    return {
      access: tokenData.access_token,
      refresh: tokenData.refresh_token,
      expires: Date.now() + tokenData.expires_in * 1000,
    };
  },
};
```

### 5. 添加二维码支持

```typescript
import qrcode from "qrcode-terminal";

async function login(callbacks: OAuthLoginCallbacks): Promise<OAuthCredentials> {
  const deviceCodeData = await requestDeviceCode();

  // 显示二维码
  if (deviceCodeData.verification_uri_complete) {
    console.log("\nScan this QR code with your mobile device:");
    qrcode.generate(deviceCodeData.verification_uri_complete, { small: true });
  }

  // 显示用户代码
  callbacks.onDeviceCode({
    userCode: deviceCodeData.user_code,
    verificationUri: deviceCodeData.verification_uri,
  });

  // ... 轮询逻辑
}
```

## 凭证管理

### auth.json结构

```json
{
  "company-llm": {
    "access": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refresh": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
    "expires": 1708531200000
  }
}
```

### 自动刷新机制

pi-mono会自动处理Token刷新：
1. 每次API调用前检查Token是否过期
2. 如果Token即将过期（提前5分钟），自动调用`refreshToken()`
3. 刷新成功后，更新`auth.json`
4. 对用户完全透明

### 手动管理凭证

```bash
# 查看凭证
cat ~/.pi/agent/auth.json

# 删除凭证（强制重新登录）
rm ~/.pi/agent/auth.json

# 或删除特定Provider的凭证
jq 'del(.["company-llm"])' ~/.pi/agent/auth.json > temp.json
mv temp.json ~/.pi/agent/auth.json
```

## 故障排查

### 问题1：设备代码过期

**错误：** `Device code expired`

**原因：** 用户在15分钟内未完成授权

**解决：** 重新运行`/login company-llm`

### 问题2：Token刷新失败

**错误：** `Token refresh failed`

**原因：** Refresh token过期或无效

**解决：**
```bash
# 删除凭证
rm ~/.pi/agent/auth.json

# 重新登录
pi
/login company-llm
```

### 问题3：网络错误

**错误：** `fetch failed`

**检查：**
```bash
# 测试OAuth服务器连接
curl https://auth.company.com/device/code \
  -X POST \
  -d "client_id=pi-agent&scope=llm.read llm.write"
```

### 问题4：授权被拒绝

**错误：** `Authorization denied by user`

**原因：** 用户在授权页面点击了"拒绝"

**解决：** 重新运行`/login`并点击"允许"

## 最佳实践

### 1. 安全性

**保护Client Secret：**
```typescript
// 如果需要Client Secret，使用环境变量
const CLIENT_SECRET = process.env.COMPANY_LLM_CLIENT_SECRET;
```

**HTTPS传输：**
```typescript
// 确保OAuth服务器使用HTTPS
const OAUTH_CONFIG = {
  authServer: "https://auth.company.com", // 必须是HTTPS
};
```

### 2. 用户体验

**清晰的指引：**
```typescript
callbacks.onDeviceCode({
  userCode: deviceCodeData.user_code,
  verificationUri: deviceCodeData.verification_uri,
});

if (callbacks.onProgress) {
  callbacks.onProgress(
    `Please visit ${deviceCodeData.verification_uri} and enter code: ${deviceCodeData.user_code}`
  );
}
```

**进度反馈：**
```typescript
if (callbacks.onProgress) {
  callbacks.onProgress("Waiting for authorization...");
}

// 轮询时更新进度
if (callbacks.onProgress) {
  callbacks.onProgress("Still waiting... (30s)");
}
```

### 3. 错误处理

**友好的错误消息：**
```typescript
catch (error) {
  if (error.message.includes("expired")) {
    throw new Error(
      "Device code expired. Please run /login again."
    );
  } else if (error.message.includes("denied")) {
    throw new Error(
      "Authorization denied. Please allow access and try again."
    );
  } else {
    throw new Error(
      `OAuth login failed: ${error.message}`
    );
  }
}
```

### 4. 测试

**Mock OAuth服务器：**
```typescript
// 用于测试的Mock服务器
const MOCK_OAUTH_SERVER = "http://localhost:3000";

const OAUTH_CONFIG = {
  authServer: process.env.NODE_ENV === "test"
    ? MOCK_OAUTH_SERVER
    : "https://auth.company.com",
};
```

## 参考资源

**OAuth 2.0规范：**
- RFC 8628: OAuth 2.0 Device Authorization Grant
- https://datatracker.ietf.org/doc/html/rfc8628

**Pi-mono文档：**
- https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/custom-provider.md

**OAuth服务器：**
- Okta: https://developer.okta.com/
- Auth0: https://auth0.com/
- Keycloak: https://www.keycloak.org/

## 总结

**OAuth集成的核心步骤：**
1. 实现`OAuthProviderInterface`接口
2. 实现Device Code Flow（login方法）
3. 实现Token刷新（refreshToken方法）
4. 实现API Key获取（getApiKey方法）
5. 注册到pi-mono

**关键点：**
- Device Code Flow适合CLI环境
- Token自动刷新对用户透明
- 凭证安全存储在auth.json
- 错误处理和用户体验很重要

**下一步：**
- 学习自定义API适配器（streamSimple）
- 学习完整的Provider开发流程
- 学习生产环境部署和监控
