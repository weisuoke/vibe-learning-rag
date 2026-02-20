# 实战代码 05：OAuth 登录流程

> **实战目标**：掌握 Pi Coding Agent 的 OAuth 登录完整流程，实现 Token 管理和自动刷新

---

## 一、OAuth 登录完整流程

### 1.1 交互式登录脚本

```bash
#!/bin/bash
# oauth-login-interactive.sh - 交互式 OAuth 登录

echo "🔐 Pi Coding Agent OAuth 登录"
echo ""

# 显示支持的 Provider
cat << 'EOF'
支持的 OAuth Provider:
1. Anthropic Claude Pro/Max ($20-200/月)
2. OpenAI ChatGPT Plus/Pro ($20-200/月)
3. GitHub Copilot ($10/月)
4. Google Gemini CLI (免费)
5. Google Antigravity (免费)

选择 Provider:
EOF

read -p "输入编号 (1-5): " choice

case $choice in
    1) provider="Anthropic Claude Pro/Max" ;;
    2) provider="OpenAI ChatGPT Plus/Pro" ;;
    3) provider="GitHub Copilot" ;;
    4) provider="Google Gemini CLI" ;;
    5) provider="Google Antigravity" ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "✅ 已选择: $provider"
echo ""
echo "步骤:"
echo "1. Pi 将打开浏览器"
echo "2. 登录并授权"
echo "3. 返回终端"
echo ""

read -p "按 Enter 键继续..." -r

# 启动 Pi 并执行登录
pi << 'EOF'
/login
EOF

# 验证登录
echo ""
echo "🔍 验证登录状态..."

if [ -f ~/.pi/agent/auth.json ]; then
    echo "✅ auth.json 文件已创建"

    # 检查 Token
    if jq -e '.anthropic.type == "oauth" or .openai.type == "oauth"' ~/.pi/agent/auth.json > /dev/null 2>&1; then
        echo "✅ OAuth Token 已保存"
        echo ""
        echo "🎉 登录成功！"
    else
        echo "⚠️  未找到 OAuth Token"
    fi
else
    echo "❌ 登录失败"
fi
```

### 1.2 自动化登录脚本

```typescript
// oauth-login-automated.ts - 自动化 OAuth 登录

import { spawn } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

interface OAuthConfig {
  provider: string;
  authFile: string;
}

async function oauthLogin(config: OAuthConfig): Promise<boolean> {
  return new Promise((resolve, reject) => {
    console.log(`🔐 开始 OAuth 登录: ${config.provider}`);

    // 启动 Pi 进程
    const pi = spawn('pi', [], {
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let output = '';

    pi.stdout.on('data', (data) => {
      output += data.toString();
      console.log(data.toString());
    });

    pi.stderr.on('data', (data) => {
      console.error(data.toString());
    });

    // 发送 /login 命令
    setTimeout(() => {
      pi.stdin.write('/login\n');
    }, 1000);

    // 等待登录完成
    pi.on('close', (code) => {
      if (code === 0) {
        // 验证 auth.json
        const authPath = path.join(
          process.env.HOME!,
          '.pi/agent/auth.json'
        );

        if (fs.existsSync(authPath)) {
          const auth = JSON.parse(fs.readFileSync(authPath, 'utf-8'));

          if (auth[config.provider]?.type === 'oauth') {
            console.log('✅ OAuth 登录成功');
            resolve(true);
          } else {
            console.log('❌ OAuth Token 未找到');
            resolve(false);
          }
        } else {
          console.log('❌ auth.json 文件不存在');
          resolve(false);
        }
      } else {
        reject(new Error(`Pi 进程退出，代码: ${code}`));
      }
    });
  });
}

// 使用示例
const config: OAuthConfig = {
  provider: 'anthropic',
  authFile: '~/.pi/agent/auth.json'
};

oauthLogin(config)
  .then((success) => {
    if (success) {
      console.log('🎉 登录完成');
      process.exit(0);
    } else {
      console.log('❌ 登录失败');
      process.exit(1);
    }
  })
  .catch((error) => {
    console.error('❌ 错误:', error.message);
    process.exit(1);
  });
```

---

## 二、Token 管理

### 2.1 Token 读取脚本

```typescript
// read-oauth-token.ts - 读取 OAuth Token

import * as fs from 'fs';
import * as path from 'path';

interface OAuthToken {
  type: 'oauth';
  accessToken: string;
  refreshToken: string;
  expiresAt: number;
  userId?: string;
  scopes?: string[];
}

interface AuthConfig {
  [provider: string]: OAuthToken | { type: 'api_key'; key: string };
}

function readOAuthToken(provider: string): OAuthToken | null {
  const authPath = path.join(
    process.env.HOME!,
    '.pi/agent/auth.json'
  );

  if (!fs.existsSync(authPath)) {
    console.log('❌ auth.json 文件不存在');
    return null;
  }

  const auth: AuthConfig = JSON.parse(
    fs.readFileSync(authPath, 'utf-8')
  );

  const providerAuth = auth[provider];

  if (!providerAuth) {
    console.log(`❌ 未找到 ${provider} 的配置`);
    return null;
  }

  if (providerAuth.type !== 'oauth') {
    console.log(`❌ ${provider} 不是 OAuth 认证`);
    return null;
  }

  return providerAuth as OAuthToken;
}

function isTokenExpired(token: OAuthToken): boolean {
  const now = Date.now();
  return token.expiresAt < now;
}

function getTokenExpiryTime(token: OAuthToken): string {
  const expiryDate = new Date(token.expiresAt);
  return expiryDate.toLocaleString();
}

// 使用示例
const provider = 'anthropic';
const token = readOAuthToken(provider);

if (token) {
  console.log(`✅ ${provider} OAuth Token:`);
  console.log(`- 类型: ${token.type}`);
  console.log(`- 过期时间: ${getTokenExpiryTime(token)}`);
  console.log(`- 是否过期: ${isTokenExpired(token) ? '是' : '否'}`);

  if (token.userId) {
    console.log(`- 用户: ${token.userId}`);
  }

  if (token.scopes) {
    console.log(`- 权限: ${token.scopes.join(', ')}`);
  }
}
```

### 2.2 Token 验证脚本

```bash
#!/bin/bash
# validate-oauth-token.sh - 验证 OAuth Token

validate_token() {
    local provider=$1
    local auth_file=~/.pi/agent/auth.json

    if [ ! -f "$auth_file" ]; then
        echo "❌ auth.json 文件不存在"
        return 1
    fi

    # 检查 Provider 配置
    if ! jq -e ".$provider" "$auth_file" > /dev/null 2>&1; then
        echo "❌ 未找到 $provider 的配置"
        return 1
    fi

    # 检查认证类型
    local auth_type=$(jq -r ".$provider.type" "$auth_file")
    if [ "$auth_type" != "oauth" ]; then
        echo "❌ $provider 不是 OAuth 认证 (类型: $auth_type)"
        return 1
    fi

    # 检查 Token 过期时间
    local expires_at=$(jq -r ".$provider.expiresAt" "$auth_file")
    local current_time=$(date +%s)000

    if [ "$expires_at" -lt "$current_time" ]; then
        echo "⚠️  Token 已过期"
        echo "过期时间: $(date -r $((expires_at / 1000)) '+%Y-%m-%d %H:%M:%S')"
        return 1
    fi

    echo "✅ Token 有效"
    echo "过期时间: $(date -r $((expires_at / 1000)) '+%Y-%m-%d %H:%M:%S')"
    return 0
}

# 验证所有 Provider
providers=("anthropic" "openai" "github-copilot" "google")

for provider in "${providers[@]}"; do
    echo "🔍 验证 $provider..."
    validate_token "$provider"
    echo ""
done
```

---

## 三、Token 刷新

### 3.1 手动刷新脚本

```typescript
// refresh-oauth-token.ts - 手动刷新 OAuth Token

import * as fs from 'fs';
import * as path from 'path';

interface OAuthToken {
  type: 'oauth';
  accessToken: string;
  refreshToken: string;
  expiresAt: number;
}

async function refreshToken(
  provider: string,
  refreshToken: string
): Promise<OAuthToken | null> {
  // 这里是简化示例，实际刷新逻辑由 Pi 内部处理
  console.log(`🔄 刷新 ${provider} Token...`);

  // 模拟 API 调用
  try {
    // 实际应该调用 Provider 的 Token 刷新端点
    const response = await fetch(`https://api.${provider}.com/oauth/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'refresh_token',
        refresh_token: refreshToken,
        client_id: 'pi-coding-agent'
      })
    });

    if (!response.ok) {
      throw new Error(`刷新失败: ${response.statusText}`);
    }

    const data = await response.json();

    return {
      type: 'oauth',
      accessToken: data.access_token,
      refreshToken: data.refresh_token || refreshToken,
      expiresAt: Date.now() + data.expires_in * 1000
    };
  } catch (error) {
    console.error('❌ 刷新失败:', error);
    return null;
  }
}

function updateAuthFile(provider: string, token: OAuthToken): void {
  const authPath = path.join(
    process.env.HOME!,
    '.pi/agent/auth.json'
  );

  const auth = JSON.parse(fs.readFileSync(authPath, 'utf-8'));
  auth[provider] = token;

  fs.writeFileSync(authPath, JSON.stringify(auth, null, 2));
  console.log('✅ Token 已更新');
}

// 使用示例
async function main() {
  const provider = 'anthropic';
  const authPath = path.join(
    process.env.HOME!,
    '.pi/agent/auth.json'
  );

  const auth = JSON.parse(fs.readFileSync(authPath, 'utf-8'));
  const currentToken = auth[provider] as OAuthToken;

  if (!currentToken || currentToken.type !== 'oauth') {
    console.log('❌ 未找到 OAuth Token');
    return;
  }

  const newToken = await refreshToken(provider, currentToken.refreshToken);

  if (newToken) {
    updateAuthFile(provider, newToken);
    console.log('🎉 Token 刷新成功');
  } else {
    console.log('❌ Token 刷新失败');
  }
}

main();
```

### 3.2 自动刷新监控

```typescript
// monitor-token-expiry.ts - 监控 Token 过期

import * as fs from 'fs';
import * as path from 'path';

interface OAuthToken {
  type: 'oauth';
  accessToken: string;
  refreshToken: string;
  expiresAt: number;
}

function checkTokenExpiry(provider: string): {
  expired: boolean;
  expiresIn: number;
  shouldRefresh: boolean;
} {
  const authPath = path.join(
    process.env.HOME!,
    '.pi/agent/auth.json'
  );

  const auth = JSON.parse(fs.readFileSync(authPath, 'utf-8'));
  const token = auth[provider] as OAuthToken;

  if (!token || token.type !== 'oauth') {
    throw new Error(`未找到 ${provider} 的 OAuth Token`);
  }

  const now = Date.now();
  const expiresIn = token.expiresAt - now;
  const fiveMinutes = 5 * 60 * 1000;

  return {
    expired: expiresIn <= 0,
    expiresIn,
    shouldRefresh: expiresIn < fiveMinutes
  };
}

function formatTime(ms: number): string {
  const minutes = Math.floor(ms / 60000);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) return `${days} 天`;
  if (hours > 0) return `${hours} 小时`;
  if (minutes > 0) return `${minutes} 分钟`;
  return '少于 1 分钟';
}

// 监控循环
function monitorTokens(providers: string[], interval: number = 60000) {
  console.log('🔍 开始监控 OAuth Token...');

  setInterval(() => {
    for (const provider of providers) {
      try {
        const status = checkTokenExpiry(provider);

        if (status.expired) {
          console.log(`❌ ${provider}: Token 已过期`);
        } else if (status.shouldRefresh) {
          console.log(
            `⚠️  ${provider}: Token 即将过期 (剩余 ${formatTime(status.expiresIn)})`
          );
        } else {
          console.log(
            `✅ ${provider}: Token 有效 (剩余 ${formatTime(status.expiresIn)})`
          );
        }
      } catch (error) {
        console.error(`❌ ${provider}: ${error.message}`);
      }
    }

    console.log('---');
  }, interval);
}

// 使用示例
const providers = ['anthropic', 'openai'];
monitorTokens(providers, 60000); // 每分钟检查一次
```

---

## 四、登出流程

### 4.1 登出脚本

```bash
#!/bin/bash
# oauth-logout.sh - OAuth 登出

logout_provider() {
    local provider=$1

    echo "🔓 登出 $provider..."

    # 启动 Pi 并执行登出
    pi << EOF
/logout
EOF

    # 验证登出
    if [ -f ~/.pi/agent/auth.json ]; then
        if jq -e ".$provider" ~/.pi/agent/auth.json > /dev/null 2>&1; then
            echo "⚠️  $provider 配置仍然存在"
        else
            echo "✅ $provider 已登出"
        fi
    else
        echo "✅ auth.json 已删除"
    fi
}

# 登出所有 Provider
logout_all() {
    echo "🔓 登出所有 Provider..."

    if [ -f ~/.pi/agent/auth.json ]; then
        # 备份 auth.json
        cp ~/.pi/agent/auth.json ~/.pi/agent/auth.json.backup
        echo "✅ 已备份 auth.json"

        # 删除 auth.json
        rm ~/.pi/agent/auth.json
        echo "✅ 已删除 auth.json"
    else
        echo "⚠️  auth.json 不存在"
    fi
}

# 使用示例
if [ "$1" = "all" ]; then
    logout_all
else
    logout_provider "${1:-anthropic}"
fi
```

### 4.2 清理 Token 脚本

```typescript
// clean-oauth-tokens.ts - 清理 OAuth Token

import * as fs from 'fs';
import * as path from 'path';

interface AuthConfig {
  [provider: string]: any;
}

function cleanExpiredTokens(): void {
  const authPath = path.join(
    process.env.HOME!,
    '.pi/agent/auth.json'
  );

  if (!fs.existsSync(authPath)) {
    console.log('⚠️  auth.json 不存在');
    return;
  }

  const auth: AuthConfig = JSON.parse(
    fs.readFileSync(authPath, 'utf-8')
  );

  const now = Date.now();
  let cleaned = 0;

  for (const [provider, config] of Object.entries(auth)) {
    if (config.type === 'oauth' && config.expiresAt < now) {
      console.log(`🧹 清理过期 Token: ${provider}`);
      delete auth[provider];
      cleaned++;
    }
  }

  if (cleaned > 0) {
    fs.writeFileSync(authPath, JSON.stringify(auth, null, 2));
    console.log(`✅ 已清理 ${cleaned} 个过期 Token`);
  } else {
    console.log('✅ 没有过期 Token');
  }
}

function removeProvider(provider: string): void {
  const authPath = path.join(
    process.env.HOME!,
    '.pi/agent/auth.json'
  );

  if (!fs.existsSync(authPath)) {
    console.log('⚠️  auth.json 不存在');
    return;
  }

  const auth: AuthConfig = JSON.parse(
    fs.readFileSync(authPath, 'utf-8')
  );

  if (auth[provider]) {
    delete auth[provider];
    fs.writeFileSync(authPath, JSON.stringify(auth, null, 2));
    console.log(`✅ 已删除 ${provider}`);
  } else {
    console.log(`⚠️  未找到 ${provider}`);
  }
}

// 使用示例
const command = process.argv[2];
const provider = process.argv[3];

if (command === 'clean') {
  cleanExpiredTokens();
} else if (command === 'remove' && provider) {
  removeProvider(provider);
} else {
  console.log('用法:');
  console.log('  node clean-oauth-tokens.ts clean');
  console.log('  node clean-oauth-tokens.ts remove <provider>');
}
```

---

## 五、故障排查

### 5.1 OAuth 诊断脚本

```bash
#!/bin/bash
# diagnose-oauth.sh - OAuth 故障诊断

echo "🔧 OAuth 故障诊断"
echo ""

# 1. 检查 auth.json 文件
echo "1️⃣ 检查 auth.json 文件:"
if [ -f ~/.pi/agent/auth.json ]; then
    echo "✅ auth.json 存在"

    # 检查文件权限
    perms=$(stat -f "%Lp" ~/.pi/agent/auth.json 2>/dev/null || stat -c "%a" ~/.pi/agent/auth.json 2>/dev/null)
    if [ "$perms" = "600" ]; then
        echo "✅ 文件权限正确 (600)"
    else
        echo "⚠️  文件权限不正确 ($perms)，应该是 600"
        echo "修复: chmod 600 ~/.pi/agent/auth.json"
    fi

    # 检查 JSON 格式
    if jq empty ~/.pi/agent/auth.json 2>/dev/null; then
        echo "✅ JSON 格式正确"
    else
        echo "❌ JSON 格式错误"
    fi
else
    echo "❌ auth.json 不存在"
fi

echo ""

# 2. 检查 OAuth Provider
echo "2️⃣ 检查 OAuth Provider:"
if [ -f ~/.pi/agent/auth.json ]; then
    providers=$(jq -r 'to_entries[] | select(.value.type == "oauth") | .key' ~/.pi/agent/auth.json 2>/dev/null)

    if [ -n "$providers" ]; then
        echo "OAuth Provider:"
        echo "$providers" | while read provider; do
            echo "- $provider"

            # 检查 Token 过期
            expires_at=$(jq -r ".$provider.expiresAt" ~/.pi/agent/auth.json)
            current_time=$(date +%s)000

            if [ "$expires_at" -gt "$current_time" ]; then
                echo "  ✅ Token 有效"
            else
                echo "  ⚠️  Token 已过期"
            fi
        done
    else
        echo "⚠️  未找到 OAuth Provider"
    fi
fi

echo ""

# 3. 检查浏览器
echo "3️⃣ 检查浏览器:"
if command -v open &> /dev/null; then
    echo "✅ 可以打开浏览器 (macOS)"
elif command -v xdg-open &> /dev/null; then
    echo "✅ 可以打开浏览器 (Linux)"
elif command -v start &> /dev/null; then
    echo "✅ 可以打开浏览器 (Windows)"
else
    echo "⚠️  无法自动打开浏览器"
fi

echo ""

# 4. 检查网络连接
echo "4️⃣ 检查网络连接:"
if ping -c 1 console.anthropic.com &> /dev/null; then
    echo "✅ 可以访问 Anthropic"
else
    echo "⚠️  无法访问 Anthropic"
fi

if ping -c 1 platform.openai.com &> /dev/null; then
    echo "✅ 可以访问 OpenAI"
else
    echo "⚠️  无法访问 OpenAI"
fi

echo ""
echo "✨ 诊断完成"
```

### 5.2 常见问题解决

```bash
#!/bin/bash
# fix-oauth-issues.sh - 修复 OAuth 常见问题

fix_permissions() {
    echo "🔧 修复文件权限..."
    chmod 600 ~/.pi/agent/auth.json
    echo "✅ 权限已修复"
}

fix_expired_token() {
    echo "🔧 修复过期 Token..."
    echo "请重新登录:"
    echo "  pi"
    echo "  /logout"
    echo "  /login"
}

fix_corrupted_json() {
    echo "🔧 修复损坏的 JSON..."

    if [ -f ~/.pi/agent/auth.json.backup ]; then
        cp ~/.pi/agent/auth.json.backup ~/.pi/agent/auth.json
        echo "✅ 已从备份恢复"
    else
        echo "⚠️  没有备份文件"
        echo "请删除 auth.json 并重新登录:"
        echo "  rm ~/.pi/agent/auth.json"
        echo "  pi"
        echo "  /login"
    fi
}

# 主菜单
cat << 'EOF'
🔧 OAuth 问题修复

选择问题:
1. 文件权限错误
2. Token 过期
3. JSON 格式损坏
4. 退出

EOF

read -p "选择 (1-4): " choice

case $choice in
    1) fix_permissions ;;
    2) fix_expired_token ;;
    3) fix_corrupted_json ;;
    4) exit 0 ;;
    *) echo "❌ 无效选择" ;;
esac
```

---

## 六、总结

### 6.1 OAuth 流程检查清单

- [ ] 订阅已激活（Claude Pro/ChatGPT Plus 等）
- [ ] 浏览器可以正常打开
- [ ] 网络连接正常
- [ ] /login 命令执行成功
- [ ] auth.json 文件已创建
- [ ] Token 已保存且有效
- [ ] 文件权限正确 (600)

### 6.2 快速参考

```bash
# OAuth 登录
pi
/login

# 检查 Token
cat ~/.pi/agent/auth.json | jq '.anthropic'

# 验证 Token
jq -r '.anthropic.expiresAt' ~/.pi/agent/auth.json

# 登出
pi
/logout

# 清理过期 Token
node clean-oauth-tokens.ts clean

# 修复权限
chmod 600 ~/.pi/agent/auth.json
```

---

**参考资料:**
- [Pi OAuth Implementation](https://github.com/badlogic/pi-mono/blob/main/packages/ai/src/oauth.ts)
- [Pi Auth Storage](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/auth-storage.ts)

**文档版本:** v1.0 (2026-02-18)
