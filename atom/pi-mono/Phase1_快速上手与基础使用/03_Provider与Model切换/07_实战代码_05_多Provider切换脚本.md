# 实战代码 05：多 Provider 切换脚本

> **自动化切换与 Fallback 实现**

---

## 自动切换脚本

### Bash 版本

```bash
#!/bin/bash
# switch-provider.sh

PROVIDER=$1
MODEL=$2

if [ -z "$PROVIDER" ]; then
  echo "Usage: $0 <provider> [model]"
  echo "Available providers: anthropic, openai, xai, ollama"
  exit 1
fi

# 启动 Pi
if [ -z "$MODEL" ]; then
  pi --provider "$PROVIDER"
else
  pi --provider "$PROVIDER" --model "$MODEL"
fi
```

### Node.js 版本

```typescript
// switch-provider.ts
import { spawn } from 'child_process';

interface ProviderConfig {
  provider: string;
  model?: string;
}

function switchProvider(config: ProviderConfig): Promise<void> {
  return new Promise((resolve, reject) => {
    const args = ['--provider', config.provider];
    if (config.model) {
      args.push('--model', config.model);
    }

    const pi = spawn('pi', args, {
      stdio: 'inherit'
    });

    pi.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Pi exited with code ${code}`));
      }
    });
  });
}

// 使用示例
switchProvider({
  provider: 'anthropic',
  model: 'claude-3-5-sonnet-20241022'
});
```

---

## Provider 健康检查

### 完整实现

```typescript
// health-check.ts
import fetch from 'node-fetch';

interface ProviderHealth {
  name: string;
  status: 'healthy' | 'unhealthy';
  latency?: number;
  error?: string;
}

class HealthChecker {
  private endpoints = {
    anthropic: 'https://api.anthropic.com',
    openai: 'https://api.openai.com',
    xai: 'https://api.x.ai',
    ollama: 'http://localhost:11434/api/tags'
  };

  async checkProvider(name: string): Promise<ProviderHealth> {
    const url = this.endpoints[name];
    if (!url) {
      return {
        name,
        status: 'unhealthy',
        error: 'Unknown provider'
      };
    }

    const start = Date.now();

    try {
      const response = await fetch(url, {
        method: 'GET',
        timeout: 5000
      });

      const latency = Date.now() - start;

      if (response.ok || response.status === 404) {
        return {
          name,
          status: 'healthy',
          latency
        };
      } else {
        return {
          name,
          status: 'unhealthy',
          error: `HTTP ${response.status}`
        };
      }
    } catch (error) {
      return {
        name,
        status: 'unhealthy',
        error: error.message
      };
    }
  }

  async checkAll(): Promise<ProviderHealth[]> {
    const providers = Object.keys(this.endpoints);
    return Promise.all(providers.map(p => this.checkProvider(p)));
  }

  async getHealthyProviders(): Promise<string[]> {
    const results = await this.checkAll();
    return results
      .filter(r => r.status === 'healthy')
      .map(r => r.name);
  }
}

// 使用示例
const checker = new HealthChecker();

const results = await checker.checkAll();
results.forEach(result => {
  const icon = result.status === 'healthy' ? '✅' : '❌';
  const latency = result.latency ? ` (${result.latency}ms)` : '';
  console.log(`${icon} ${result.name}${latency}`);
});
```

---

## Fallback 实现

### 简单 Fallback

```typescript
// simple-fallback.ts
async function callWithFallback(
  prompt: string,
  providers: string[]
): Promise<string> {
  for (const provider of providers) {
    try {
      console.log(`Trying ${provider}...`);
      const result = await callProvider(provider, prompt);
      console.log(`✅ ${provider} succeeded`);
      return result;
    } catch (error) {
      console.log(`❌ ${provider} failed: ${error.message}`);
    }
  }

  throw new Error('All providers failed');
}

// 使用示例
const result = await callWithFallback('Hello', [
  'anthropic',
  'openai',
  'ollama'
]);
```

### 智能 Fallback

```typescript
// smart-fallback.ts
interface FallbackStrategy {
  primary: string;
  fallbacks: string[];
  healthCheck: boolean;
}

class SmartFallback {
  private checker: HealthChecker;

  constructor() {
    this.checker = new HealthChecker();
  }

  async execute(
    prompt: string,
    strategy: FallbackStrategy
  ): Promise<string> {
    let providers = [strategy.primary, ...strategy.fallbacks];

    // 健康检查
    if (strategy.healthCheck) {
      const healthy = await this.checker.getHealthyProviders();
      providers = providers.filter(p => healthy.includes(p));

      if (providers.length === 0) {
        throw new Error('No healthy providers available');
      }
    }

    // 尝试调用
    for (const provider of providers) {
      try {
        return await this.callProvider(provider, prompt);
      } catch (error) {
        console.log(`${provider} failed, trying next...`);
      }
    }

    throw new Error('All providers failed');
  }

  private async callProvider(
    provider: string,
    prompt: string
  ): Promise<string> {
    // 实现 Provider 调用逻辑
    return `Response from ${provider}`;
  }
}

// 使用示例
const fallback = new SmartFallback();

const result = await fallback.execute('Hello', {
  primary: 'anthropic',
  fallbacks: ['openai', 'ollama'],
  healthCheck: true
});
```

---

## 会话管理

### 会话持久化

```typescript
// session-manager.ts
import { writeFileSync, readFileSync, existsSync } from 'fs';

interface Session {
  id: string;
  provider: string;
  model: string;
  messages: Array<{ role: string; content: string }>;
  createdAt: Date;
  updatedAt: Date;
}

class SessionManager {
  private sessionFile: string;

  constructor(sessionFile: string = '.pi-session.json') {
    this.sessionFile = sessionFile;
  }

  createSession(provider: string, model: string): Session {
    const session: Session = {
      id: Date.now().toString(),
      provider,
      model,
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date()
    };

    this.saveSession(session);
    return session;
  }

  loadSession(): Session | null {
    if (!existsSync(this.sessionFile)) {
      return null;
    }

    const data = readFileSync(this.sessionFile, 'utf-8');
    return JSON.parse(data);
  }

  saveSession(session: Session) {
    session.updatedAt = new Date();
    writeFileSync(this.sessionFile, JSON.stringify(session, null, 2));
  }

  addMessage(role: string, content: string) {
    const session = this.loadSession();
    if (!session) {
      throw new Error('No active session');
    }

    session.messages.push({ role, content });
    this.saveSession(session);
  }

  switchProvider(provider: string, model: string) {
    const session = this.loadSession();
    if (!session) {
      throw new Error('No active session');
    }

    session.provider = provider;
    session.model = model;
    this.saveSession(session);
  }
}

// 使用示例
const manager = new SessionManager();

// 创建会话
const session = manager.createSession('anthropic', 'claude-3-5-sonnet-20241022');

// 添加消息
manager.addMessage('user', 'Hello');
manager.addMessage('assistant', 'Hi there!');

// 切换 Provider
manager.switchProvider('openai', 'gpt-4o');
```

---

## 批量切换脚本

### 测试所有 Provider

```bash
#!/bin/bash
# test-all-providers.sh

PROVIDERS=("anthropic" "openai" "xai" "ollama")
TEST_PROMPT="Say 'OK'"

echo "Testing all providers..."
echo ""

for provider in "${PROVIDERS[@]}"; do
  echo "Testing $provider..."

  if pi --provider "$provider" <<< "$TEST_PROMPT" > /dev/null 2>&1; then
    echo "✅ $provider: OK"
  else
    echo "❌ $provider: FAILED"
  fi

  echo ""
done

echo "Test complete"
```

### 性能对比

```typescript
// benchmark-providers.ts
interface BenchmarkResult {
  provider: string;
  model: string;
  latency: number;
  success: boolean;
  error?: string;
}

class ProviderBenchmark {
  async benchmark(
    provider: string,
    model: string,
    prompt: string
  ): Promise<BenchmarkResult> {
    const start = Date.now();

    try {
      await this.callProvider(provider, model, prompt);
      const latency = Date.now() - start;

      return {
        provider,
        model,
        latency,
        success: true
      };
    } catch (error) {
      return {
        provider,
        model,
        latency: Date.now() - start,
        success: false,
        error: error.message
      };
    }
  }

  async benchmarkAll(
    configs: Array<{ provider: string; model: string }>,
    prompt: string
  ): Promise<BenchmarkResult[]> {
    return Promise.all(
      configs.map(c => this.benchmark(c.provider, c.model, prompt))
    );
  }

  private async callProvider(
    provider: string,
    model: string,
    prompt: string
  ): Promise<void> {
    // 实现调用逻辑
  }
}

// 使用示例
const benchmark = new ProviderBenchmark();

const results = await benchmark.benchmarkAll(
  [
    { provider: 'anthropic', model: 'claude-3-5-haiku-20241022' },
    { provider: 'openai', model: 'gpt-4o-mini' },
    { provider: 'ollama', model: 'llama3.1:8b' }
  ],
  'Hello'
);

// 排序并显示
results
  .sort((a, b) => a.latency - b.latency)
  .forEach(r => {
    const icon = r.success ? '✅' : '❌';
    console.log(`${icon} ${r.provider}: ${r.latency}ms`);
  });
```

---

## 自动化工作流

### 智能路由

```typescript
// smart-router.ts
class SmartRouter {
  private checker: HealthChecker;
  private fallback: SmartFallback;

  constructor() {
    this.checker = new HealthChecker();
    this.fallback = new SmartFallback();
  }

  async route(prompt: string): Promise<string> {
    // 1. 分类任务
    const complexity = this.classifyTask(prompt);

    // 2. 选择 Provider
    const provider = this.selectProvider(complexity);

    // 3. 执行（带 Fallback）
    return this.fallback.execute(prompt, {
      primary: provider,
      fallbacks: this.getFallbacks(provider),
      healthCheck: true
    });
  }

  private classifyTask(prompt: string): 'simple' | 'medium' | 'complex' {
    // 实现任务分类逻辑
    return 'medium';
  }

  private selectProvider(complexity: string): string {
    const mapping = {
      simple: 'anthropic',
      medium: 'anthropic',
      complex: 'anthropic'
    };
    return mapping[complexity];
  }

  private getFallbacks(primary: string): string[] {
    const fallbacks = {
      anthropic: ['openai', 'ollama'],
      openai: ['anthropic', 'ollama'],
      ollama: ['anthropic', 'openai']
    };
    return fallbacks[primary] || [];
  }
}

// 使用示例
const router = new SmartRouter();
const result = await router.route('Refactor this component');
```

---

## 监控脚本

### 实时监控

```bash
#!/bin/bash
# monitor-providers.sh

while true; do
  clear
  echo "=== Provider Health Monitor ==="
  echo ""
  echo "Timestamp: $(date)"
  echo ""

  # 检查每个 Provider
  for provider in anthropic openai xai ollama; do
    if node -e "
      const { HealthChecker } = require('./health-check');
      const checker = new HealthChecker();
      checker.checkProvider('$provider').then(r => {
        const icon = r.status === 'healthy' ? '✅' : '❌';
        const latency = r.latency ? \` (\${r.latency}ms)\` : '';
        console.log(\`\${icon} $provider\${latency}\`);
      });
    " 2>/dev/null; then
      :
    else
      echo "❌ $provider (error)"
    fi
  done

  echo ""
  echo "Press Ctrl+C to exit"

  sleep 10
done
```

---

## 完整示例

### 集成系统

```typescript
// provider-manager.ts
class ProviderManager {
  private checker: HealthChecker;
  private fallback: SmartFallback;
  private router: SmartRouter;
  private session: SessionManager;

  constructor() {
    this.checker = new HealthChecker();
    this.fallback = new SmartFallback();
    this.router = new SmartRouter();
    this.session = new SessionManager();
  }

  async execute(prompt: string): Promise<string> {
    // 1. 智能路由
    const result = await this.router.route(prompt);

    // 2. 记录会话
    this.session.addMessage('user', prompt);
    this.session.addMessage('assistant', result);

    return result;
  }

  async healthCheck(): Promise<void> {
    const results = await this.checker.checkAll();
    results.forEach(r => {
      const icon = r.status === 'healthy' ? '✅' : '❌';
      console.log(`${icon} ${r.name}`);
    });
  }

  async benchmark(): Promise<void> {
    const benchmark = new ProviderBenchmark();
    const results = await benchmark.benchmarkAll(
      [
        { provider: 'anthropic', model: 'claude-3-5-haiku-20241022' },
        { provider: 'openai', model: 'gpt-4o-mini' }
      ],
      'Hello'
    );

    results.forEach(r => {
      console.log(`${r.provider}: ${r.latency}ms`);
    });
  }
}

// 使用示例
const manager = new ProviderManager();

// 健康检查
await manager.healthCheck();

// 执行任务
const result = await manager.execute('Refactor this code');

// 性能测试
await manager.benchmark();
```

---

**记住**：自动化切换提高可靠性，Fallback 机制保障服务可用性。
