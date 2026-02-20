# 实战代码 07：自定义 Provider 实现

> **实战目标**：实现完整的自定义 Provider 配置，支持 Ollama、企业代理和 OpenRouter 等场景

---

## 一、Ollama 本地模型配置

### 1.1 安装和启动 Ollama

```bash
#!/bin/bash
# setup-ollama.sh - 安装和配置 Ollama

# macOS 安装
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "📦 安装 Ollama (macOS)..."
    brew install ollama
fi

# Linux 安装
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "📦 安装 Ollama (Linux)..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# 启动 Ollama 服务
echo "🚀 启动 Ollama 服务..."
ollama serve &

# 等待服务启动
sleep 3

# 拉取模型
echo "📥 拉取模型..."
ollama pull llama3.1:8b
ollama pull qwen2.5-coder:7b
ollama pull deepseek-r1:7b

# 验证安装
echo "✅ 验证安装..."
ollama list

echo "🎉 Ollama 配置完成"
```

### 1.2 配置 models.json

```bash
#!/bin/bash
# configure-ollama.sh - 配置 Ollama Provider

mkdir -p ~/.pi/agent

cat > ~/.pi/agent/models.json << 'EOF'
{
  "providers": {
    "ollama": {
      "baseUrl": "http://localhost:11434/v1",
      "api": "openai-completions",
      "apiKey": "ollama",
      "models": [
        {
          "id": "llama3.1:8b",
          "name": "Llama 3.1 8B (Local)",
          "contextWindow": 128000,
          "maxTokens": 32000,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        },
        {
          "id": "qwen2.5-coder:7b",
          "name": "Qwen 2.5 Coder 7B",
          "contextWindow": 32768,
          "maxTokens": 8192,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        },
        {
          "id": "deepseek-r1:7b",
          "name": "DeepSeek R1 7B",
          "reasoning": true,
          "contextWindow": 64000,
          "maxTokens": 8192,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
EOF

echo "✅ Ollama Provider 配置完成"
```

### 1.3 测试 Ollama

```bash
#!/bin/bash
# test-ollama.sh - 测试 Ollama 配置

echo "🔍 测试 Ollama..."

# 测试 API
response=$(curl -s http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 10
  }')

if echo "$response" | grep -q "choices"; then
    echo "✅ Ollama API 正常"
    echo "响应: $(echo $response | jq -r '.choices[0].message.content')"
else
    echo "❌ Ollama API 异常"
    echo "错误: $response"
fi

# 测试 Pi
echo ""
echo "🏖️ 测试 Pi with Ollama..."
pi --provider ollama --model llama3.1:8b --print "生成一个 Hello World 函数"
```

---

## 二、企业代理配置

### 2.1 企业代理 models.json

```json
// ~/.pi/agent/models.json - 企业代理配置
{
  "providers": {
    "corp-proxy": {
      "baseUrl": "https://llm-proxy.corp.example.com/v1",
      "api": "anthropic-messages",
      "apiKey": "!aws secretsmanager get-secret-value --secret-id anthropic-key --query SecretString --output text",
      "headers": {
        "x-corp-auth": "CORP_AUTH_TOKEN",
        "x-department": "engineering",
        "x-cost-center": "12345"
      },
      "models": [
        {
          "id": "claude-opus-4",
          "name": "Claude Opus 4 (Corp Proxy)",
          "contextWindow": 200000,
          "maxTokens": 16384,
          "cost": {
            "input": 15,
            "output": 75,
            "cacheRead": 1.5,
            "cacheWrite": 18.75
          }
        },
        {
          "id": "claude-sonnet-4",
          "name": "Claude Sonnet 4 (Corp Proxy)",
          "contextWindow": 200000,
          "maxTokens": 16384,
          "cost": {
            "input": 3,
            "output": 15,
            "cacheRead": 0.3,
            "cacheWrite": 3.75
          }
        }
      ]
    }
  }
}
```

### 2.2 企业代理配置脚本

```bash
#!/bin/bash
# setup-corp-proxy.sh - 配置企业代理

# 1. 配置环境变量
export CORP_AUTH_TOKEN=$(aws secretsmanager get-secret-value \
  --secret-id corp-auth-token \
  --query SecretString \
  --output text)

# 2. 创建 models.json
mkdir -p ~/.pi/agent

cat > ~/.pi/agent/models.json << EOF
{
  "providers": {
    "corp-proxy": {
      "baseUrl": "https://llm-proxy.corp.example.com/v1",
      "api": "anthropic-messages",
      "apiKey": "!aws secretsmanager get-secret-value --secret-id anthropic-key --query SecretString --output text",
      "headers": {
        "x-corp-auth": "$CORP_AUTH_TOKEN",
        "x-department": "engineering"
      },
      "models": [
        {
          "id": "claude-opus-4",
          "name": "Claude Opus 4 (Corp)"
        }
      ]
    }
  }
}
EOF

echo "✅ 企业代理配置完成"
```

### 2.3 测试企业代理

```bash
#!/bin/bash
# test-corp-proxy.sh - 测试企业代理

echo "🔍 测试企业代理..."

# 测试连接
if curl -s -o /dev/null -w "%{http_code}" https://llm-proxy.corp.example.com/health | grep -q "200"; then
    echo "✅ 代理服务可访问"
else
    echo "❌ 代理服务不可访问"
    exit 1
fi

# 测试 Pi
pi --provider corp-proxy --model claude-opus-4 --print "测试企业代理"
```

---

## 三、OpenRouter 配置

### 3.1 OpenRouter models.json

```json
// ~/.pi/agent/models.json - OpenRouter 配置
{
  "providers": {
    "openrouter": {
      "baseUrl": "https://openrouter.ai/api/v1",
      "api": "openai-completions",
      "apiKey": "OPENROUTER_API_KEY",
      "models": [
        {
          "id": "anthropic/claude-sonnet-4",
          "name": "Claude Sonnet 4 (OpenRouter)",
          "contextWindow": 200000,
          "maxTokens": 16384,
          "cost": {
            "input": 3,
            "output": 15,
            "cacheRead": 0.3,
            "cacheWrite": 3.75
          },
          "compat": {
            "openRouterRouting": {
              "order": ["anthropic"],
              "fallbacks": ["openai"]
            }
          }
        },
        {
          "id": "openai/gpt-4o",
          "name": "GPT-4o (OpenRouter)",
          "contextWindow": 128000,
          "maxTokens": 16384,
          "cost": {
            "input": 2.5,
            "output": 10,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        },
        {
          "id": "google/gemini-2.0-flash-exp",
          "name": "Gemini 2.0 Flash (OpenRouter)",
          "contextWindow": 1000000,
          "maxTokens": 8192,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
```

### 3.2 OpenRouter 配置脚本

```bash
#!/bin/bash
# setup-openrouter.sh - 配置 OpenRouter

# 1. 获取 API Key
cat << 'EOF'
📝 获取 OpenRouter API Key

步骤:
1. 访问 https://openrouter.ai/keys
2. 创建 API Key
3. 复制 API Key (sk-or-...)
EOF

read -p "输入 OpenRouter API Key: " api_key

# 2. 配置环境变量
export OPENROUTER_API_KEY=$api_key
echo "export OPENROUTER_API_KEY=$api_key" >> ~/.bashrc

# 3. 创建 models.json
mkdir -p ~/.pi/agent

cat > ~/.pi/agent/models.json << 'EOF'
{
  "providers": {
    "openrouter": {
      "baseUrl": "https://openrouter.ai/api/v1",
      "api": "openai-completions",
      "apiKey": "OPENROUTER_API_KEY",
      "models": [
        {
          "id": "anthropic/claude-sonnet-4",
          "name": "Claude Sonnet 4 (OpenRouter)"
        },
        {
          "id": "openai/gpt-4o",
          "name": "GPT-4o (OpenRouter)"
        }
      ]
    }
  }
}
EOF

echo "✅ OpenRouter 配置完成"
```

---

## 四、Vercel AI Gateway 配置

### 4.1 Vercel AI Gateway models.json

```json
// ~/.pi/agent/models.json - Vercel AI Gateway 配置
{
  "providers": {
    "vercel-gateway": {
      "baseUrl": "https://ai-gateway.vercel.sh/v1",
      "api": "openai-completions",
      "apiKey": "AI_GATEWAY_API_KEY",
      "models": [
        {
          "id": "moonshotai/kimi-k2.5",
          "name": "Kimi K2.5 (Fireworks)",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": 262144,
          "maxTokens": 262144,
          "cost": {
            "input": 0.6,
            "output": 3,
            "cacheRead": 0,
            "cacheWrite": 0
          },
          "compat": {
            "vercelGatewayRouting": {
              "only": ["fireworks", "novita"],
              "order": ["fireworks", "novita"]
            }
          }
        }
      ]
    }
  }
}
```

---

## 五、完整配置示例

### 5.1 多 Provider 完整配置

```json
// ~/.pi/agent/models.json - 完整配置
{
  "providers": {
    "ollama": {
      "baseUrl": "http://localhost:11434/v1",
      "api": "openai-completions",
      "apiKey": "ollama",
      "models": [
        {
          "id": "llama3.1:8b",
          "name": "Llama 3.1 8B (Local)",
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    },
    "corp-proxy": {
      "baseUrl": "https://llm-proxy.corp.example.com/v1",
      "api": "anthropic-messages",
      "apiKey": "!aws secretsmanager get-secret-value --secret-id anthropic-key --query SecretString --output text",
      "headers": {
        "x-corp-auth": "CORP_AUTH_TOKEN"
      },
      "models": [
        {
          "id": "claude-opus-4",
          "name": "Claude Opus 4 (Corp)"
        }
      ]
    },
    "openrouter": {
      "baseUrl": "https://openrouter.ai/api/v1",
      "api": "openai-completions",
      "apiKey": "OPENROUTER_API_KEY",
      "models": [
        {
          "id": "anthropic/claude-sonnet-4",
          "name": "Claude Sonnet 4 (OpenRouter)"
        }
      ]
    }
  }
}
```

### 5.2 配置管理脚本

```typescript
// manage-providers.ts - Provider 配置管理

import * as fs from 'fs';
import * as path from 'path';

interface ProviderConfig {
  baseUrl: string;
  api: string;
  apiKey: string;
  headers?: Record<string, string>;
  models: ModelConfig[];
}

interface ModelConfig {
  id: string;
  name: string;
  contextWindow?: number;
  maxTokens?: number;
  cost?: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
  };
}

interface ModelsConfig {
  providers: Record<string, ProviderConfig>;
}

class ProviderManager {
  private configPath: string;
  private config: ModelsConfig;

  constructor() {
    this.configPath = path.join(
      process.env.HOME!,
      '.pi/agent/models.json'
    );
    this.config = this.loadConfig();
  }

  private loadConfig(): ModelsConfig {
    if (fs.existsSync(this.configPath)) {
      return JSON.parse(fs.readFileSync(this.configPath, 'utf-8'));
    }
    return { providers: {} };
  }

  private saveConfig(): void {
    fs.writeFileSync(
      this.configPath,
      JSON.stringify(this.config, null, 2)
    );
  }

  addProvider(name: string, config: ProviderConfig): void {
    this.config.providers[name] = config;
    this.saveConfig();
    console.log(`✅ 已添加 Provider: ${name}`);
  }

  removeProvider(name: string): void {
    if (this.config.providers[name]) {
      delete this.config.providers[name];
      this.saveConfig();
      console.log(`✅ 已删除 Provider: ${name}`);
    } else {
      console.log(`⚠️  Provider 不存在: ${name}`);
    }
  }

  listProviders(): void {
    console.log('📋 已配置的 Provider:');
    for (const [name, config] of Object.entries(this.config.providers)) {
      console.log(`\n${name}:`);
      console.log(`  Base URL: ${config.baseUrl}`);
      console.log(`  API: ${config.api}`);
      console.log(`  Models: ${config.models.length}`);
      config.models.forEach(model => {
        console.log(`    - ${model.name} (${model.id})`);
      });
    }
  }

  addModel(provider: string, model: ModelConfig): void {
    if (!this.config.providers[provider]) {
      console.log(`❌ Provider 不存在: ${provider}`);
      return;
    }

    this.config.providers[provider].models.push(model);
    this.saveConfig();
    console.log(`✅ 已添加模型: ${model.name} 到 ${provider}`);
  }
}

// 使用示例
const manager = new ProviderManager();

// 添加 Ollama Provider
manager.addProvider('ollama', {
  baseUrl: 'http://localhost:11434/v1',
  api: 'openai-completions',
  apiKey: 'ollama',
  models: [
    {
      id: 'llama3.1:8b',
      name: 'Llama 3.1 8B (Local)',
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
    }
  ]
});

// 列出所有 Provider
manager.listProviders();
```

---

## 六、验证和测试

### 6.1 Provider 验证脚本

```bash
#!/bin/bash
# validate-providers.sh - 验证 Provider 配置

echo "🔍 验证 Provider 配置..."

# 1. 检查 models.json
if [ ! -f ~/.pi/agent/models.json ]; then
    echo "❌ models.json 不存在"
    exit 1
fi

echo "✅ models.json 存在"

# 2. 验证 JSON 格式
if jq empty ~/.pi/agent/models.json 2>/dev/null; then
    echo "✅ JSON 格式正确"
else
    echo "❌ JSON 格式错误"
    exit 1
fi

# 3. 列出所有 Provider
echo ""
echo "📋 已配置的 Provider:"
jq -r '.providers | keys[]' ~/.pi/agent/models.json

# 4. 验证每个 Provider
jq -r '.providers | keys[]' ~/.pi/agent/models.json | while read provider; do
    echo ""
    echo "🔍 验证 $provider..."

    # 检查必需字段
    base_url=$(jq -r ".providers.$provider.baseUrl" ~/.pi/agent/models.json)
    api=$(jq -r ".providers.$provider.api" ~/.pi/agent/models.json)
    model_count=$(jq -r ".providers.$provider.models | length" ~/.pi/agent/models.json)

    echo "  Base URL: $base_url"
    echo "  API: $api"
    echo "  Models: $model_count"

    # 测试连接
    if curl -s -o /dev/null -w "%{http_code}" "$base_url" | grep -q "200\|404"; then
        echo "  ✅ 可以访问"
    else
        echo "  ⚠️  无法访问"
    fi
done

echo ""
echo "✨ 验证完成"
```

### 6.2 模型测试脚本

```bash
#!/bin/bash
# test-all-models.sh - 测试所有模型

echo "🧪 测试所有模型..."

# 获取所有 Provider 和模型
jq -r '.providers | to_entries[] | "\(.key):\(.value.models[].id)"' ~/.pi/agent/models.json | while IFS=: read provider model; do
    echo ""
    echo "🔍 测试 $provider - $model..."

    # 使用 Pi 测试
    if pi --provider "$provider" --model "$model" --print "Hi" 2>/dev/null; then
        echo "✅ $provider - $model 正常"
    else
        echo "❌ $provider - $model 失败"
    fi
done

echo ""
echo "✨ 测试完成"
```

---

## 七、故障排查

### 7.1 Provider 诊断脚本

```bash
#!/bin/bash
# diagnose-provider.sh - Provider 故障诊断

provider=$1

if [ -z "$provider" ]; then
    echo "用法: ./diagnose-provider.sh <provider>"
    exit 1
fi

echo "🔧 诊断 Provider: $provider"
echo ""

# 1. 检查配置
echo "1️⃣ 检查配置:"
if jq -e ".providers.$provider" ~/.pi/agent/models.json > /dev/null 2>&1; then
    echo "✅ Provider 配置存在"

    # 显示配置
    echo ""
    echo "配置详情:"
    jq ".providers.$provider" ~/.pi/agent/models.json
else
    echo "❌ Provider 配置不存在"
    exit 1
fi

echo ""

# 2. 检查 Base URL
echo "2️⃣ 检查 Base URL:"
base_url=$(jq -r ".providers.$provider.baseUrl" ~/.pi/agent/models.json)
echo "Base URL: $base_url"

if curl -s -o /dev/null -w "%{http_code}" "$base_url" | grep -q "200\|404"; then
    echo "✅ 可以访问"
else
    echo "❌ 无法访问"
fi

echo ""

# 3. 检查 API Key
echo "3️⃣ 检查 API Key:"
api_key_config=$(jq -r ".providers.$provider.apiKey" ~/.pi/agent/models.json)

if [[ $api_key_config == !* ]]; then
    echo "API Key 类型: Shell 命令"
    echo "命令: ${api_key_config:1}"
elif [[ $api_key_config =~ ^[A-Z_]+$ ]]; then
    echo "API Key 类型: 环境变量"
    echo "变量: $api_key_config"
    if [ -n "${!api_key_config}" ]; then
        echo "✅ 环境变量已设置"
    else
        echo "❌ 环境变量未设置"
    fi
else
    echo "API Key 类型: 字面值"
    echo "✅ API Key 已配置"
fi

echo ""

# 4. 检查模型
echo "4️⃣ 检查模型:"
model_count=$(jq -r ".providers.$provider.models | length" ~/.pi/agent/models.json)
echo "模型数量: $model_count"

jq -r ".providers.$provider.models[].name" ~/.pi/agent/models.json | while read model_name; do
    echo "  - $model_name"
done

echo ""
echo "✨ 诊断完成"
```

---

## 八、总结

### 8.1 配置检查清单

- [ ] models.json 已创建
- [ ] JSON 格式正确
- [ ] 所有 Provider 已配置
- [ ] Base URL 可访问
- [ ] API Key 已设置
- [ ] 模型列表完整
- [ ] 配置已测试验证

### 8.2 快速参考

```bash
# 创建 models.json
mkdir -p ~/.pi/agent
cat > ~/.pi/agent/models.json << 'EOF'
{"providers":{"ollama":{"baseUrl":"http://localhost:11434/v1","api":"openai-completions","apiKey":"ollama","models":[{"id":"llama3.1:8b","name":"Llama 3.1 8B"}]}}}
EOF

# 验证配置
jq empty ~/.pi/agent/models.json

# 列出 Provider
jq -r '.providers | keys[]' ~/.pi/agent/models.json

# 测试 Provider
pi --provider ollama --model llama3.1:8b --print "Hi"
```

---

**参考资料:**
- [Pi Models Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/models.md)
- [Pi Custom Provider Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/custom-provider.md)

**文档版本:** v1.0 (2026-02-18)
