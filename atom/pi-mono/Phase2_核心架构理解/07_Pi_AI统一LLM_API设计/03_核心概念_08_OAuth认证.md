# 核心概念8：OAuth 与认证

> 理解如何统一处理不同 LLM Provider 的认证方式

---

## 概念定义

**OAuth 与认证**是指统一管理不同 Provider 的认证方式，包括 API Key、OAuth Token、订阅认证等，简化认证配置和管理。

**核心价值：**
- **统一接口**：一套认证 API 适配所有 Provider
- **安全存储**：安全管理认证凭据
- **自动刷新**：自动刷新过期的 Token
- **多种方式**：支持多种认证方式

---

## 核心实现

### 1. 认证配置

```typescript
/**
 * 认证配置
 */
type AuthConfig =
  | { type: 'api_key'; apiKey: string }
  | { type: 'oauth'; token: string; refreshToken?: string }
  | { type: 'subscription'; userId: string; subscriptionId: string }
  | { type: 'custom'; credentials: Record<string, string> };

/**
 * Provider 认证配置
 */
const AUTH_CONFIGS: Record<string, AuthConfig> = {
  openai: {
    type: 'api_key',
    apiKey: process.env.OPENAI_API_KEY!
  },
  anthropic: {
    type: 'api_key',
    apiKey: process.env.ANTHROPIC_API_KEY!
  },
  github: {
    type: 'oauth',
    token: process.env.GITHUB_TOKEN!
  }
};
```

### 2. 认证管理器

```typescript
/**
 * 认证管理器
 */
class AuthManager {
  private configs = new Map<string, AuthConfig>();

  /**
   * 注册认证配置
   */
  register(provider: string, config: AuthConfig): void {
    this.configs.set(provider, config);
  }

  /**
   * 获取认证配置
   */
  get(provider: string): AuthConfig | undefined {
    return this.configs.get(provider);
  }

  /**
   * 获取认证头
   */
  getHeaders(provider: string): Record<string, string> {
    const config = this.get(provider);
    if (!config) {
      throw new Error(`No auth config for provider: ${provider}`);
    }

    switch (config.type) {
      case 'api_key':
        return {
          'Authorization': `Bearer ${config.apiKey}`
        };

      case 'oauth':
        return {
          'Authorization': `Bearer ${config.token}`
        };

      case 'subscription':
        return {
          'X-User-ID': config.userId,
          'X-Subscription-ID': config.subscriptionId
        };

      case 'custom':
        return config.credentials;

      default:
        throw new Error(`Unknown auth type`);
    }
  }
}
```

### 3. OAuth 流程

```typescript
/**
 * OAuth 认证流程
 */
class OAuthFlow {
  /**
   * 获取授权 URL
   */
  getAuthorizationUrl(
    clientId: string,
    redirectUri: string,
    scope: string[]
  ): string {
    const params = new URLSearchParams({
      client_id: clientId,
      redirect_uri: redirectUri,
      scope: scope.join(' '),
      response_type: 'code'
    });

    return `https://provider.com/oauth/authorize?${params}`;
  }

  /**
   * 交换授权码为 Token
   */
  async exchangeCodeForToken(
    code: string,
    clientId: string,
    clientSecret: string,
    redirectUri: string
  ): Promise<{ accessToken: string; refreshToken: string }> {
    const response = await fetch('https://provider.com/oauth/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        grant_type: 'authorization_code',
        code,
        client_id: clientId,
        client_secret: clientSecret,
        redirect_uri: redirectUri
      })
    });

    const data = await response.json();
    return {
      accessToken: data.access_token,
      refreshToken: data.refresh_token
    };
  }

  /**
   * 刷新 Token
   */
  async refreshToken(
    refreshToken: string,
    clientId: string,
    clientSecret: string
  ): Promise<string> {
    const response = await fetch('https://provider.com/oauth/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        grant_type: 'refresh_token',
        refresh_token: refreshToken,
        client_id: clientId,
        client_secret: clientSecret
      })
    });

    const data = await response.json();
    return data.access_token;
  }
}
```

---

## 在 AI Agent 中的应用

### 场景1：多种认证方式

```typescript
async function multiAuthExample(): Promise<void> {
  const authManager = new AuthManager();

  // 1. API Key 认证（OpenAI）
  authManager.register('openai', {
    type: 'api_key',
    apiKey: process.env.OPENAI_API_KEY!
  });

  // 2. OAuth 认证（GitHub）
  authManager.register('github', {
    type: 'oauth',
    token: process.env.GITHUB_TOKEN!
  });

  // 3. 订阅认证（Pi）
  authManager.register('pi', {
    type: 'subscription',
    userId: 'user123',
    subscriptionId: 'sub456'
  });

  // 使用
  const headers = authManager.getHeaders('openai');
  console.log(headers);
}
```

### 场景2：自动刷新 Token

```typescript
class AutoRefreshAdapter implements ProviderAdapter {
  private oauth: OAuthFlow;
  private authConfig: AuthConfig;

  async complete(context: Context): Promise<Message> {
    try {
      return await this.callAPI(context);
    } catch (error) {
      if (error.status === 401) {
        // Token 过期，刷新
        await this.refreshAccessToken();
        return await this.callAPI(context);
      }
      throw error;
    }
  }

  private async refreshAccessToken(): Promise<void> {
    if (this.authConfig.type === 'oauth' && this.authConfig.refreshToken) {
      const newToken = await this.oauth.refreshToken(
        this.authConfig.refreshToken,
        CLIENT_ID,
        CLIENT_SECRET
      );
      this.authConfig.token = newToken;
    }
  }

  private async callAPI(context: Context): Promise<Message> {
    // 实际 API 调用
    return {} as Message;
  }

  // ... 其他方法
}
```

---

## 学习检查清单

- [ ] 理解不同认证方式的差异
- [ ] 能够实现认证管理器
- [ ] 能够处理 OAuth 流程
- [ ] 能够实现自动刷新 Token
- [ ] 能够安全存储认证凭据

---

## 参考资源

- [OAuth 2.0](https://oauth.net/2/)
- [OpenAI Authentication](https://platform.openai.com/docs/api-reference/authentication)
- [GitHub OAuth](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps)

---

**版本：** v1.0
**最后更新：** 2026-02-19
