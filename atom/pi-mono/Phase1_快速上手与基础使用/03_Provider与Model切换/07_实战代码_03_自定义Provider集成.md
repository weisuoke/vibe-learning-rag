# 实战代码 03：自定义 Provider 集成

> **集成 Ollama、LM Studio、OpenRouter 等自定义 Provider**

---

## Ollama 本地部署

### 安装 Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# 启动服务
ollama serve
```

### 下载模型

```bash
# 下载 Llama 3.1 8B
ollama pull llama3.1:8b

# 下载 Code Llama 13B
ollama pull codellama:13b

# 验证
ollama list
```

### Pi 配置

```json
// ~/.pi/agent/models.json
{
  "providers": {
    "ollama": {
      "apiType": "openai-compatible",
      "baseUrl": "http://localhost:11434",
      "models": {
        "llama3.1:8b": {
          "id": "llama3.1:8b",
          "name": "Llama 3.1 8B (Local)",
          "contextWindow": 131072,
          "maxOutput": 32768,
          "cost": {
            "input": 0.0,
            "output": 0.0
          },
          "tags": ["local", "free", "offline"]
        },
        "codellama:13b": {
          "id": "codellama:13b",
          "name": "Code Llama 13B (Local)",
          "contextWindow": 16384,
          "maxOutput": 4096,
          "cost": {
            "input": 0.0,
            "output": 0.0
          },
          "tags": ["local", "free", "coding"]
        }
      }
    }
  }
}
```

### 验证脚本

```bash
#!/bin/bash
# verify-ollama.sh

echo "Verifying Ollama setup..."

# 检查 Ollama 服务
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
  echo "❌ Ollama service not running"
  echo "Run: ollama serve"
  exit 1
fi

# 检查模型
MODELS=$(curl -s http://localhost:11434/api/tags | jq -r '.models[].name')

if [ -z "$MODELS" ]; then
  echo "❌ No models found"
  echo "Run: ollama pull llama3.1:8b"
  exit 1
fi

echo "✅ Ollama is running"
echo "Available models:"
echo "$MODELS" | sed 's/^/  - /'

# 测试 Pi 集成
echo ""
echo "Testing Pi integration..."
pi --provider ollama --model llama3.1:8b <<EOF
Say "Ollama OK"
EOF
```

---

## LM Studio 集成

### 安装 LM Studio

1. 下载：https://lmstudio.ai/
2. 安装并启动
3. 下载模型（如 Llama 3.1 8B）
4. 启动本地服务器（端口 1234）

### Pi 配置

```json
// ~/.pi/agent/models.json
{
  "providers": {
    "lmstudio": {
      "apiType": "openai-compatible",
      "baseUrl": "http://localhost:1234",
      "models": {
        "llama-3.1-8b": {
          "id": "llama-3.1-8b",
          "name": "Llama 3.1 8B (LM Studio)",
          "contextWindow": 131072,
          "maxOutput": 32768,
          "cost": {
            "input": 0.0,
            "output": 0.0
          },
          "tags": ["local", "free", "gui"]
        }
      }
    }
  }
}
```

### 验证脚本

```bash
#!/bin/bash
# verify-lmstudio.sh

echo "Verifying LM Studio setup..."

# 检查服务
if ! curl -s http://localhost:1234/v1/models > /dev/null; then
  echo "❌ LM Studio server not running"
  echo "Start server in LM Studio app"
  exit 1
fi

# 获取模型列表
MODELS=$(curl -s http://localhost:1234/v1/models | jq -r '.data[].id')

echo "✅ LM Studio is running"
echo "Available models:"
echo "$MODELS" | sed 's/^/  - /'
```

---

## OpenRouter 集成

### 获取 API Key

1. 访问：https://openrouter.ai/
2. 注册账号
3. 获取 API Key

### Pi 配置

```json
// ~/.pi/agent/models.json
{
  "providers": {
    "openrouter": {
      "apiType": "openai-compatible",
      "baseUrl": "https://openrouter.ai/api/v1",
      "models": {
        "anthropic/claude-3.5-sonnet": {
          "id": "anthropic/claude-3.5-sonnet",
          "name": "Claude 3.5 Sonnet (OpenRouter)",
          "contextWindow": 200000,
          "maxOutput": 8192,
          "cost": {
            "input": 3.0,
            "output": 15.0
          }
        },
        "openai/gpt-4o": {
          "id": "openai/gpt-4o",
          "name": "GPT-4o (OpenRouter)",
          "contextWindow": 128000,
          "maxOutput": 16384,
          "cost": {
            "input": 2.5,
            "output": 10.0
          }
        }
      }
    }
  }
}

// ~/.pi/agent/auth.json
{
  "openrouter": {
    "apiKey": "sk-or-v1-YOUR_KEY_HERE"
  }
}
```

---

## Azure OpenAI 集成

### 配置

```json
// ~/.pi/agent/models.json
{
  "providers": {
    "azure-openai": {
      "apiType": "openai-completions",
      "baseUrl": "https://YOUR_RESOURCE.openai.azure.com",
      "headers": {
        "api-key": "${AZURE_OPENAI_API_KEY}"
      },
      "models": {
        "gpt-4o": {
          "id": "gpt-4o",
          "name": "GPT-4o (Azure)",
          "contextWindow": 128000,
          "maxOutput": 16384,
          "cost": {
            "input": 2.5,
            "output": 10.0
          }
        }
      }
    }
  }
}
```

### 环境变量

```bash
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
```

---

## 完整集成脚本

### 自动化设置

```bash
#!/bin/bash
# setup-custom-providers.sh

set -e

echo "🚀 Setting up custom providers..."

# 1. Ollama
echo ""
echo "1. Setting up Ollama..."

if command -v ollama &> /dev/null; then
  echo "✅ Ollama installed"

  # 启动服务（后台）
  if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama service..."
    ollama serve &
    sleep 2
  fi

  # 下载模型
  if ! ollama list | grep -q "llama3.1:8b"; then
    echo "Downloading llama3.1:8b..."
    ollama pull llama3.1:8b
  fi

  echo "✅ Ollama configured"
else
  echo "⚠️  Ollama not installed"
  echo "Install: brew install ollama"
fi

# 2. 配置 Pi
echo ""
echo "2. Configuring Pi..."

mkdir -p ~/.pi/agent

# 添加 Ollama 到 models.json
cat >> ~/.pi/agent/models.json <<'EOF'
{
  "providers": {
    "ollama": {
      "apiType": "openai-compatible",
      "baseUrl": "http://localhost:11434",
      "models": {
        "llama3.1:8b": {
          "id": "llama3.1:8b",
          "name": "Llama 3.1 8B (Local)",
          "contextWindow": 131072,
          "maxOutput": 32768,
          "cost": { "input": 0.0, "output": 0.0 },
          "tags": ["local", "free"]
        }
      }
    }
  }
}
EOF

echo "✅ Pi configured"

# 3. 验证
echo ""
echo "3. Verifying setup..."

if curl -s http://localhost:11434/api/tags > /dev/null; then
  echo "✅ Ollama service is running"
else
  echo "❌ Ollama service not running"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Test with:"
echo "  pi --provider ollama --model llama3.1:8b"
```

---

## 健康检查脚本

```typescript
// health-check.ts
import fetch from 'node-fetch';

interface ProviderHealth {
  name: string;
  url: string;
  status: 'healthy' | 'unhealthy' | 'unknown';
  latency?: number;
  error?: string;
}

async function checkProvider(
  name: string,
  url: string
): Promise<ProviderHealth> {
  const start = Date.now();

  try {
    const response = await fetch(url, {
      method: 'GET',
      timeout: 5000
    });

    const latency = Date.now() - start;

    if (response.ok) {
      return {
        name,
        url,
        status: 'healthy',
        latency
      };
    } else {
      return {
        name,
        url,
        status: 'unhealthy',
        error: `HTTP ${response.status}`
      };
    }
  } catch (error) {
    return {
      name,
      url,
      status: 'unhealthy',
      error: error.message
    };
  }
}

async function checkAllProviders() {
  console.log('Checking provider health...\n');

  const providers = [
    { name: 'Ollama', url: 'http://localhost:11434/api/tags' },
    { name: 'LM Studio', url: 'http://localhost:1234/v1/models' },
    { name: 'Anthropic', url: 'https://api.anthropic.com' },
    { name: 'OpenAI', url: 'https://api.openai.com' }
  ];

  const results = await Promise.all(
    providers.map(p => checkProvider(p.name, p.url))
  );

  results.forEach(result => {
    const icon = result.status === 'healthy' ? '✅' : '❌';
    const latency = result.latency ? ` (${result.latency}ms)` : '';
    const error = result.error ? ` - ${result.error}` : '';

    console.log(`${icon} ${result.name}${latency}${error}`);
  });

  const healthyCount = results.filter(r => r.status === 'healthy').length;
  console.log(`\n${healthyCount}/${results.length} providers healthy`);
}

checkAllProviders();
```

---

## 项目模板

### 本地开发模板

```json
// project-local/.pi/settings.json
{
  "defaultModel": "llama3.1:8b",
  "scopedModels": [
    "llama3.1:8b",
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022"
  ]
}
```

### 混合模式模板

```json
// project-hybrid/.pi/settings.json
{
  "defaultModel": "llama3.1:8b",
  "scopedModels": [
    "llama3.1:8b",
    "claude-3-5-haiku-20241022",
    "gpt-4o-mini"
  ]
}
```

---

## 故障排查

### Ollama 问题

```bash
# 检查服务状态
pgrep -x "ollama"

# 重启服务
pkill ollama
ollama serve

# 检查端口
lsof -i :11434

# 查看日志
ollama logs
```

### LM Studio 问题

```bash
# 检查端口
lsof -i :1234

# 测试连接
curl http://localhost:1234/v1/models
```

---

**记住**：本地模型免费但性能有限，适合开发测试；云端模型付费但质量更高，适合生产环境。
