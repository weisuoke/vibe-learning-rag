# 核心概念：自定义OAuth

## 概述

OAuth 2.0是现代Web应用的标准认证授权协议。在pi-mono中，自定义OAuth支持让你能够集成需要OAuth认证的企业LLM服务。

**核心价值：**
- **安全性**：Token自动管理，无需手动处理
- **用户体验**：Device Code Flow适合CLI环境
- **企业集成**：支持企业SSO和OAuth服务器

## OAuth 2.0基础

### 什么是OAuth 2.0？

OAuth 2.0是一个授权框架，允许第三方应用在用户授权的情况下访问用户的资源，而无需获取用户的密码。

**核心角色：**
- **Resource Owner（资源所有者）**：用户
- **Client（客户端）**：pi-coding-agent
- **Authorization Server（授权服务器）**：企业OAuth服务器
- **Resource Server（资源服务器）**：LLM API服务器

### OAuth 2.0流程类型

**1. Authorization Code Flow（授权码流程）**
- 适用于Web应用
- 需要浏览器重定向
- 最安全的流程

**2. Device Code Flow（设备码流程）**
- 适用于CLI/IoT设备
- 无需浏览器重定向
- **pi-mono推荐使用**

**3. Client Credentials Flow（客户端凭证流程）**
- 适用于服务器到服务器
- 无用户交互

**4. Implicit Flow（隐式流程）**
- 已废弃，不推荐使用

### 为什么选择Device Code Flow？

**CLI环境的挑战：**
- 无法打开浏览器窗口
- 无法接收HTTP回调
- 用户可能在远程服务器上

**Device Code Flow的优势：**
- 用户在任何设备上完成授权
- 无需本地HTTP服务器
- 适合无头环境

## OAuthProviderInterface接口

### 接口定义

```typescript
// packages/ai/src/utils/oauth/types.ts
export interface OAuthProviderInterface {
  readonly id: OAuthProviderId;
  readonly name: string;

  /** 运行登录流程，返回凭证 */
  login(callbacks: OAuthLoginCallbacks): Promise<OAuthCredentials>;

  /** 是否使用本地回调服务器 */
  usesCallbackServer?: boolean;

  /** 刷新过期凭证，返回更新后的凭证 */
  refreshToken(credentials: OAuthCredentials): Promise<OAuthCredentials>;

  /** 将凭证转换为API Key字符串 */
  getApiKey(credentials: OAuthCredentials): string;

  /** 可选：修改模型配置（如更新baseUrl） */
  modifyModels?(models: Model<Api>[], credentials: OAuthCredentials): Model<Api>[];
}
```

### 核心方法详解

#### 1. login() - 登录流程

**作用：** 执行OAuth登录，获取access_token和refresh_token

**参数：**
```typescript
interface OAuthLoginCallbacks {
  // 打开浏览器URL（用于Authorization Code Flow）
  onAuth(info: OAuthAuthInfo): void;

  // 显示设备代码（用于Device Code Flow）
  onDeviceCode(params: { userCode: string; verificationUri: string }): void;

  // 提示用户输入
  onPrompt(prompt: OAuthPrompt): Promise<string>;

  // 显示进度信息
  onProgress?(message: string): void;

  // 取消信号
  signal?: AbortSignal;
}
```

**返回值：**
```typescript
interface OAuthCredentials {
  refresh: string;   // Refresh Token
  access: string;    // Access Token
  expires: number;   // 过期时间戳（毫秒）
  [key: string]: unknown;  // 自定义字段
}
```

**实现示例：**
```typescript
async login(callbacks: OAuthLoginCallbacks): Promise<OAuthCredentials> {
  // 1. 请求设备代码
  const deviceCode = await requestDeviceCode();

  // 2. 显示用户代码
  callbacks.onDeviceCode({
    userCode: deviceCode.user_code,
    verificationUri: deviceCode.verification_uri,
  });

  // 3. 轮询Token
  const token = await pollForToken(deviceCode.device_code);

  // 4. 返回凭证
  return {
    access: token.access_token,
    refresh: token.refresh_token,
    expires: Date.now() + token.expires_in * 1000,
  };
}
```

#### 2. refreshToken() - 刷新Token

**作用：** 使用refresh_token获取新的access_token

**参数：**
```typescript
credentials: OAuthCredentials  // 当前凭证（包含refresh_token）
```

**返回值：**
```typescript
OAuthCredentials  // 更新后的凭证
```

**实现示例：**
```typescript
async refreshToken(credentials: OAuthCredentials): Promise<OAuthCredentials> {
  const response = await fetch('https://auth.example.com/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'refresh_token',
      refresh_token: credentials.refresh,
      client_id: 'pi-agent',
    }),
  });

  const data = await response.json();

  return {
    access: data.access_token,
    refresh: data.refresh_token || credentials.refresh,  // 有些服务器不返回新的refresh_token
    expires: Date.now() + data.expires_in * 1000,
  };
}
```

#### 3. getApiKey() - 获取API Key

**作用：** 从凭证中提取API Key（通常是access_token）

**参数：**
```typescript
credentials: OAuthCredentials
```

**返回值：**
```typescript
string  // API Key字符串
```

**实现示例：**
```typescript
getApiKey(credentials: OAuthCredentials): string {
  return credentials.access;
}
```

#### 4. modifyModels() - 修改模型配置（可选）

**作用：** 根据用户凭证动态修改模型配置

**使用场景：**
- 根据用户订阅级别显示不同模型
- 根据用户区域设置不同的baseUrl
- 根据用户权限启用/禁用功能

**实现示例：**
```typescript
modifyModels(models: Model<Api>[], credentials: OAuthCredentials): Model<Api>[] {
  // 从Token中解码用户信息
  const userInfo = decodeJWT(credentials.access);

  // 根据用户区域设置baseUrl
  return models.map(model => ({
    ...model,
    baseUrl: `https://${userInfo.region}.api.example.com/v1`,
  }));
}
```

## Device Code Flow详解

### 流程步骤

**步骤1：请求设备代码**

```typescript
async function requestDeviceCode() {
  const response = await fetch('https://auth.example.com/device/code', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      client_id: 'pi-agent',
      scope: 'llm.read llm.write',
    }),
  });

  const data = await response.json();
  return {
    device_code: data.device_code,              // 设备代码（用于轮询）
    user_code: data.user_code,                  // 用户代码（用户输入）
    verification_uri: data.verification_uri,    // 验证URL
    verification_uri_complete: data.verification_uri_complete,  // 完整URL（含user_code）
    expires_in: data.expires_in,                // 过期时间（秒）
    interval: data.interval || 5,               // 轮询间隔（秒）
  };
}
```

**响应示例：**
```json
{
  "device_code": "GmRhmhcxhwAzkoEqiMEg_DnyEysNkuNhszIySk9eS",
  "user_code": "WDJB-MJHT",
  "verification_uri": "https://example.com/device",
  "verification_uri_complete": "https://example.com/device?user_code=WDJB-MJHT",
  "expires_in": 900,
  "interval": 5
}
```

**步骤2：显示用户代码**

```typescript
callbacks.onDeviceCode({
  userCode: deviceCode.user_code,
  verificationUri: deviceCode.verification_uri,
});

// pi会显示：
// Please visit: https://example.com/device
// And enter code: WDJB-MJHT
```

**步骤3：轮询Token**

```typescript
async function pollForToken(deviceCode: string, interval: number): Promise<any> {
  let pollInterval = interval;

  while (true) {
    // 等待指定间隔
    await new Promise(resolve => setTimeout(resolve, pollInterval * 1000));

    // 请求Token
    const response = await fetch('https://auth.example.com/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'urn:ietf:params:oauth:grant-type:device_code',
        device_code: deviceCode,
        client_id: 'pi-agent',
      }),
    });

    const data = await response.json();

    if (response.ok) {
      // 成功获取Token
      return data;
    } else if (data.error === 'authorization_pending') {
      // 用户还未授权，继续轮询
      continue;
    } else if (data.error === 'slow_down') {
      // 轮询太快，增加间隔
      pollInterval += 5;
      continue;
    } else if (data.error === 'expired_token') {
      // 设备代码过期
      throw new Error('Device code expired. Please try again.');
    } else if (data.error === 'access_denied') {
      // 用户拒绝授权
      throw new Error('Authorization denied by user.');
    } else {
      // 其他错误
      throw new Error(`Token request failed: ${data.error_description || data.error}`);
    }
  }
}
```

**错误处理：**

| 错误代码 | 含义 | 处理方式 |
|---------|------|---------|
| `authorization_pending` | 用户还未授权 | 继续轮询 |
| `slow_down` | 轮询太快 | 增加轮询间隔 |
| `expired_token` | 设备代码过期 | 抛出错误，提示重新登录 |
| `access_denied` | 用户拒绝授权 | 抛出错误，提示用户 |

**步骤4：返回凭证**

```typescript
return {
  access: tokenData.access_token,
  refresh: tokenData.refresh_token,
  expires: Date.now() + tokenData.expires_in * 1000,
};
```

### 完整实现示例

```typescript
const myOAuthProvider: OAuthProviderInterface = {
  id: 'my-provider',
  name: 'My Provider (OAuth)',

  async login(callbacks: OAuthLoginCallbacks): Promise<OAuthCredentials> {
    try {
      // 1. 请求设备代码
      const deviceCode = await requestDeviceCode();

      // 2. 显示用户代码
      callbacks.onDeviceCode({
        userCode: deviceCode.user_code,
        verificationUri: deviceCode.verification_uri,
      });

      // 3. 显示进度
      if (callbacks.onProgress) {
        callbacks.onProgress('Waiting for authorization...');
      }

      // 4. 轮询Token
      const tokenData = await pollForToken(
        deviceCode.device_code,
        deviceCode.interval
      );

      // 5. 返回凭证
      return {
        access: tokenData.access_token,
        refresh: tokenData.refresh_token,
        expires: Date.now() + tokenData.expires_in * 1000,
      };
    } catch (error) {
      throw new Error(`OAuth login failed: ${error.message}`);
    }
  },

  async refreshToken(credentials: OAuthCredentials): Promise<OAuthCredentials> {
    const response = await fetch('https://auth.example.com/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'refresh_token',
        refresh_token: credentials.refresh,
        client_id: 'pi-agent',
      }),
    });

    const data = await response.json();

    return {
      access: data.access_token,
      refresh: data.refresh_token || credentials.refresh,
      expires: Date.now() + data.expires_in * 1000,
    };
  },

  getApiKey(credentials: OAuthCredentials): string {
    return credentials.access;
  },
};
```
