# 实战代码 01：基础 Provider 配置

> **完整的 Provider 配置示例与验证脚本**

---

## Anthropic (Claude) 配置

### 完整配置

```json
// ~/.pi/agent/models.json
{
  "providers": {
    "anthropic": {
      "apiType": "anthropic-messages",
      "baseUrl": "https://api.anthropic.com",
      "models": {
        "claude-3-5-sonnet-20241022": {
          "id": "claude-3-5-sonnet-20241022",
          "name": "Claude 3.5 Sonnet",
          "contextWindow": 200000,
          "maxOutput": 8192,
          "reasoning": true,
          "cost": {
            "input": 3.0,
            "output": 15.0
          },
          "tags": ["coding", "analysis", "reasoning"]
        },
        "claude-3-5-haiku-20241022": {
          "id": "claude-3-5-haiku-20241022",
          "name": "Claude 3.5 Haiku",
          "contextWindow": 200000,
          "maxOutput": 8192,
          "cost": {
            "input": 0.8,
            "output": 4.0
          },
          "tags": ["fast", "cheap", "coding"]
        },
        "claude-opus-4-20250514": {
          "id": "claude-opus-4-20250514",
          "name": "Claude Opus 4",
          "contextWindow": 200000,
          "maxOutput": 8192,
          "reasoning": true,
          "cost": {
            "input": 15.0,
            "output": 75.0
          },
          "tags": ["powerful", "reasoning", "complex"]
        }
      }
    }
  }
}
```

### 认证配置

```json
// ~/.pi/agent/auth.json
{
  "anthropic": {
    "apiKey": "sk-ant-api03-YOUR_KEY_HERE"
  }
}
```

### 设置脚本

```bash
#!/bin/bash
# setup-anthropic.sh

echo "Setting up Anthropic Provider..."

# 创建目录
mkdir -p ~/.pi/agent

# 配置 models.json
cat > ~/.pi/agent/models.json <<'EOF'
{
  "providers": {
    "anthropic": {
      "apiType": "anthropic-messages",
      "baseUrl": "https://api.anthropic.com",
      "models": {
        "claude-3-5-sonnet-20241022": {
          "id": "claude-3-5-sonnet-20241022",
          "name": "Claude 3.5 Sonnet",
          "contextWindow": 200000,
          "maxOutput": 8192,
          "reasoning": true,
          "cost": { "input": 3.0, "output": 15.0 }
        },
        "claude-3-5-haiku-20241022": {
          "id": "claude-3-5-haiku-20241022",
          "name": "Claude 3.5 Haiku",
          "contextWindow": 200000,
          "maxOutput": 8192,
          "cost": { "input": 0.8, "output": 4.0 }
        }
      }
    }
  }
}
EOF

# 配置 auth.json
cat > ~/.pi/agent/auth.json <<EOF
{
  "anthropic": {
    "apiKey": "${ANTHROPIC_API_KEY}"
  }
}
EOF

# 设置权限
chmod 600 ~/.pi/agent/auth.json

echo "✅ Anthropic Provider configured"
echo "Run 'pi --provider anthropic' to test"
```

---

## OpenAI (GPT) 配置

### 完整配置

```json
// ~/.pi/agent/models.json
{
  "providers": {
    "openai": {
      "apiType": "openai-completions",
      "baseUrl": "https://api.openai.com",
      "models": {
        "gpt-4-turbo": {
          "id": "gpt-4-turbo",
          "name": "GPT-4 Turbo",
          "contextWindow": 128000,
          "maxOutput": 4096,
          "cost": {
            "input": 10.0,
            "output": 30.0
          },
          "tags": ["powerful", "general"]
        },
        "gpt-4o": {
          "id": "gpt-4o",
          "name": "GPT-4o",
          "contextWindow": 128000,
          "maxOutput": 16384,
          "cost": {
            "input": 2.5,
            "output": 10.0
          },
          "tags": ["multimodal", "balanced"]
        },
        "gpt-4o-mini": {
          "id": "gpt-4o-mini",
          "name": "GPT-4o Mini",
          "contextWindow": 128000,
          "maxOutput": 16384,
          "cost": {
            "input": 0.15,
            "output": 0.6
          },
          "tags": ["cheap", "fast"]
        }
      }
    }
  }
}
```

### 认证配置

```json
// ~/.pi/agent/auth.json
{
  "openai": {
    "apiKey": "sk-proj-YOUR_KEY_HERE"
  }
}
```

---

## xAI (Grok) 配置

### 完整配置

```json
// ~/.pi/agent/models.json
{
  "providers": {
    "xai": {
      "apiType": "openai-completions",
      "baseUrl": "https://api.x.ai",
      "models": {
        "grok-2-1212": {
          "id": "grok-2-1212",
          "name": "Grok 2",
          "contextWindow": 131072,
          "maxOutput": 32768,
          "cost": {
            "input": 2.0,
            "output": 10.0
          },
          "tags": ["realtime", "long-context"]
        }
      }
    }
  }
}
```

---

## 多 Provider 配置

### 完整配置

```json
// ~/.pi/agent/models.json
{
  "providers": {
    "anthropic": {
      "apiType": "anthropic-messages",
      "baseUrl": "https://api.anthropic.com",
      "models": {
        "claude-3-5-sonnet-20241022": { ... },
        "claude-3-5-haiku-20241022": { ... }
      }
    },
    "openai": {
      "apiType": "openai-completions",
      "baseUrl": "https://api.openai.com",
      "models": {
        "gpt-4o": { ... },
        "gpt-4o-mini": { ... }
      }
    },
    "xai": {
      "apiType": "openai-completions",
      "baseUrl": "https://api.x.ai",
      "models": {
        "grok-2-1212": { ... }
      }
    }
  }
}
```

### 认证配置

```json
// ~/.pi/agent/auth.json
{
  "anthropic": {
    "apiKey": "sk-ant-api03-..."
  },
  "openai": {
    "apiKey": "sk-proj-..."
  },
  "xai": {
    "apiKey": "xai-..."
  }
}
```

---

## 验证脚本

### 验证配置文件

```bash
#!/bin/bash
# verify-config.sh

echo "Verifying Pi configuration..."

# 检查配置文件是否存在
if [ ! -f ~/.pi/agent/models.json ]; then
  echo "❌ models.json not found"
  exit 1
fi

if [ ! -f ~/.pi/agent/auth.json ]; then
  echo "❌ auth.json not found"
  exit 1
fi

# 检查 JSON 语法
if ! jq . ~/.pi/agent/models.json > /dev/null 2>&1; then
  echo "❌ models.json has invalid JSON syntax"
  exit 1
fi

if ! jq . ~/.pi/agent/auth.json > /dev/null 2>&1; then
  echo "❌ auth.json has invalid JSON syntax"
  exit 1
fi

# 检查文件权限
AUTH_PERMS=$(stat -f "%A" ~/.pi/agent/auth.json)
if [ "$AUTH_PERMS" != "600" ]; then
  echo "⚠️  auth.json permissions are $AUTH_PERMS (should be 600)"
  echo "Run: chmod 600 ~/.pi/agent/auth.json"
fi

echo "✅ Configuration files are valid"
```

### 测试 Provider 连接

```bash
#!/bin/bash
# test-providers.sh

echo "Testing Provider connections..."

# 测试 Anthropic
echo "Testing Anthropic..."
pi --provider anthropic --model claude-3-5-haiku-20241022 <<EOF
Say "Anthropic OK"
EOF

# 测试 OpenAI
echo "Testing OpenAI..."
pi --provider openai --model gpt-4o-mini <<EOF
Say "OpenAI OK"
EOF

echo "✅ All providers tested"
```

---

## 故障排查

### 问题 1：API Key 无效

```bash
# 检查 API Key 格式
cat ~/.pi/agent/auth.json | jq .

# 验证 API Key
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{
    "model": "claude-3-5-haiku-20241022",
    "max_tokens": 10,
    "messages": [{"role": "user", "content": "Hi"}]
  }'
```

### 问题 2：文件权限错误

```bash
# 检查权限
ls -la ~/.pi/agent/auth.json

# 修复权限
chmod 600 ~/.pi/agent/auth.json
```

### 问题 3：配置未生效

```bash
# 重载配置
pi
> /reload

# 验证配置
> /model
```

---

## 完整设置脚本

```bash
#!/bin/bash
# setup-all-providers.sh

set -e

echo "🚀 Setting up Pi Providers..."

# 创建目录
mkdir -p ~/.pi/agent

# 配置 models.json
cat > ~/.pi/agent/models.json <<'EOF'
{
  "providers": {
    "anthropic": {
      "apiType": "anthropic-messages",
      "baseUrl": "https://api.anthropic.com",
      "models": {
        "claude-3-5-sonnet-20241022": {
          "id": "claude-3-5-sonnet-20241022",
          "name": "Claude 3.5 Sonnet",
          "contextWindow": 200000,
          "maxOutput": 8192,
          "cost": { "input": 3.0, "output": 15.0 }
        },
        "claude-3-5-haiku-20241022": {
          "id": "claude-3-5-haiku-20241022",
          "name": "Claude 3.5 Haiku",
          "contextWindow": 200000,
          "maxOutput": 8192,
          "cost": { "input": 0.8, "output": 4.0 }
        }
      }
    },
    "openai": {
      "apiType": "openai-completions",
      "baseUrl": "https://api.openai.com",
      "models": {
        "gpt-4o": {
          "id": "gpt-4o",
          "name": "GPT-4o",
          "contextWindow": 128000,
          "maxOutput": 16384,
          "cost": { "input": 2.5, "output": 10.0 }
        },
        "gpt-4o-mini": {
          "id": "gpt-4o-mini",
          "name": "GPT-4o Mini",
          "contextWindow": 128000,
          "maxOutput": 16384,
          "cost": { "input": 0.15, "output": 0.6 }
        }
      }
    }
  }
}
EOF

# 配置 auth.json
cat > ~/.pi/agent/auth.json <<EOF
{
  "anthropic": {
    "apiKey": "${ANTHROPIC_API_KEY}"
  },
  "openai": {
    "apiKey": "${OPENAI_API_KEY}"
  }
}
EOF

# 设置权限
chmod 600 ~/.pi/agent/auth.json

# 配置 settings.json
cat > ~/.pi/agent/settings.json <<'EOF'
{
  "defaultModel": "claude-3-5-haiku-20241022",
  "scopedModels": [
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "gpt-4o"
  ]
}
EOF

echo "✅ All providers configured"
echo ""
echo "Next steps:"
echo "1. Run 'pi' to start"
echo "2. Press Ctrl+P to cycle through models"
echo "3. Run '/session' to verify"
```

---

## 使用示例

```bash
# 1. 设置环境变量
export ANTHROPIC_API_KEY="sk-ant-api03-..."
export OPENAI_API_KEY="sk-proj-..."

# 2. 运行设置脚本
bash setup-all-providers.sh

# 3. 启动 Pi
pi

# 4. 测试切换
> Hello
# 按 Ctrl+P 切换模型
> Hello again

# 5. 查看会话信息
> /session
```

---

**记住**：配置一次，长期使用。保护好 auth.json 文件权限。
