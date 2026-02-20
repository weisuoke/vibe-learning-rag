# 核心概念 07：自定义 Provider

> **知识点定位**：掌握通过 models.json 配置自定义 Provider 和模型，支持 Ollama、vLLM、LM Studio 等本地模型和企业代理

---

## 一、自定义 Provider 概述

### 1.1 什么是自定义 Provider

自定义 Provider 允许将 Pi Coding Agent 连接到任何 OpenAI 兼容的 LLM 服务，包括：

**本地模型：**
- Ollama
- LM Studio
- vLLM
- llama.cpp

**企业代理：**
- 内部 API 网关
- 负载均衡器
- 缓存代理

**第三方服务：**
- OpenRouter
- Vercel AI Gateway
- Portkey

> **来源**：[Pi Models Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/models.md) (2026-02)

### 1.2 为什么需要自定义 Provider

**场景示例：**

```
场景 1：本地开发
- 使用 Ollama 运行 Llama 3.1 8B
- 零成本，完全离线
- 适合实验和学习

场景 2：企业环境
- 通过企业代理访问 Claude API
- 统一认证和审计
- 符合安全合规要求

场景 3：成本优化
- 使用 OpenRouter 自动选择最便宜的 Provider
- 设置 Fallback 策略
- 降低 API 成本
```

---

## 二、models.json 配置文件

### 2.1 配置文件位置

```
~/.pi/agent/models.json
```

**完整路径：**
- macOS/Linux: `/Users/username/.pi/agent/models.json`
- Windows: `C:\Users\username\.pi\agent\models.json`

### 2.2 最小配置示例

```json
{
  "providers": {
    "ollama": {
      "baseUrl": "http://localhost:11434/v1",
      "api": "openai-completions",
      "apiKey": "ollama",
      "models": [
        { "id": "llama3.1:8b" },
        { "id": "qwen2.5-coder:7b" }
      ]
    }
  }
}
```

**字段说明：**
- `baseUrl`: API 端点 URL
- `api`: API 类型（见下文）
- `apiKey`: API 密钥（Ollama 忽略此字段，但必须提供）
- `models`: 模型列表，最少只需 `id` 字段

### 2.3 完整配置示例

```json
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
          "reasoning": false,
          "input": ["text"],
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
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 32768,
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

### 2.4 热重载

```bash
# 编辑 models.json
vim ~/.pi/agent/models.json

# 在 Pi 中打开模型选择器（自动重新加载）
pi
/model

# 或使用 /reload 命令
/reload
```

**无需重启 Pi**，配置会在打开 `/model` 时自动重新加载。

---

## 三、支持的 API 类型

### 3.1 API 类型列表

| API 类型 | 描述 | 兼容性 |
|---------|------|--------|
| `openai-completions` | OpenAI Chat Completions API | 最广泛支持 |
| `openai-responses` | OpenAI Responses API | OpenAI 专用 |
| `anthropic-messages` | Anthropic Messages API | Anthropic 专用 |
| `google-generative-ai` | Google Generative AI | Google 专用 |

### 3.2 API 类型选择

**openai-completions（推荐）：**
```json
{
  "api": "openai-completions"
}
```

**适用于：**
- Ollama
- LM Studio
- vLLM
- llama.cpp
- 大多数 OpenAI 兼容服务

**anthropic-messages：**
```json
{
  "api": "anthropic-messages"
}
```

**适用于：**
- Anthropic API 代理
- 企业 Claude 网关

**google-generative-ai：**
```json
{
  "api": "google-generative-ai"
}
```

**适用于：**
- Google Gemini API 代理
- Vertex AI 网关

### 3.3 Provider 级别 vs 模型级别

**Provider 级别（所有模型使用相同 API）：**
```json
{
  "providers": {
    "ollama": {
      "api": "openai-completions",
      "models": [...]
    }
  }
}
```

**模型级别（每个模型使用不同 API）：**
```json
{
  "providers": {
    "mixed": {
      "baseUrl": "http://localhost:8080/v1",
      "models": [
        {
          "id": "llama3.1:8b",
          "api": "openai-completions"
        },
        {
          "id": "claude-proxy",
          "api": "anthropic-messages"
        }
      ]
    }
  }
}
```

---

## 四、Provider 配置详解

### 4.1 Provider 配置字段

| 字段 | 必需 | 描述 |
|------|------|------|
| `baseUrl` | 是 | API 端点 URL |
| `api` | 是 | API 类型 |
| `apiKey` | 是 | API 密钥（支持值解析） |
| `headers` | 否 | 自定义 HTTP 头 |
| `authHeader` | 否 | 自动添加 `Authorization: Bearer <apiKey>` |
| `models` | 是 | 模型列表 |
| `modelOverrides` | 否 | 覆盖内置模型配置 |

### 4.2 API Key 值解析

`apiKey` 和 `headers` 字段支持三种格式：

#### 4.2.1 Shell 命令

```json
{
  "apiKey": "!security find-generic-password -ws 'anthropic'"
}
```

**示例：**

```json
{
  "providers": {
    "anthropic-proxy": {
      "baseUrl": "https://proxy.example.com/v1",
      "api": "anthropic-messages",
      "apiKey": "!op read 'op://vault/Anthropic/credential'",
      "models": [...]
    }
  }
}
```

#### 4.2.2 环境变量引用

```json
{
  "apiKey": "MY_API_KEY"
}
```

Pi 会读取 `$MY_API_KEY` 环境变量的值。

#### 4.2.3 字面值

```json
{
  "apiKey": "sk-ant-..."
}
```

直接使用提供的值。

### 4.3 自定义 HTTP 头

```json
{
  "providers": {
    "portkey-proxy": {
      "baseUrl": "https://api.portkey.ai/v1",
      "api": "anthropic-messages",
      "apiKey": "ANTHROPIC_API_KEY",
      "headers": {
        "x-portkey-api-key": "PORTKEY_API_KEY",
        "x-portkey-virtual-key": "!op read 'op://vault/Portkey/virtual-key'",
        "x-portkey-metadata": "{\"environment\":\"production\"}"
      },
      "models": [...]
    }
  }
}
```

**Headers 也支持值解析：**
- Shell 命令：`"!command"`
- 环境变量：`"ENV_VAR"`
- 字面值：`"literal-value"`

### 4.4 自动认证头

```json
{
  "providers": {
    "custom": {
      "baseUrl": "http://localhost:8080/v1",
      "api": "openai-completions",
      "apiKey": "my-secret-key",
      "authHeader": true,
      "models": [...]
    }
  }
}
```

**效果：**
Pi 会自动添加 `Authorization: Bearer my-secret-key` 头。

---

## 五、模型配置详解

### 5.1 模型配置字段

| 字段 | 必需 | 默认值 | 描述 |
|------|------|--------|------|
| `id` | 是 | — | 模型标识符（传递给 API） |
| `name` | 否 | `id` | 模型选择器中的显示名称 |
| `api` | 否 | Provider 的 `api` | 覆盖 Provider 的 API 类型 |
| `reasoning` | 否 | `false` | 是否支持扩展思考 |
| `input` | 否 | `["text"]` | 输入类型：`["text"]` 或 `["text", "image"]` |
| `contextWindow` | 否 | `128000` | 上下文窗口大小（tokens） |
| `maxTokens` | 否 | `16384` | 最大输出 tokens |
| `cost` | 否 | 全为 0 | 成本配置（每百万 tokens） |

### 5.2 最小模型配置

```json
{
  "models": [
    { "id": "llama3.1:8b" }
  ]
}
```

**默认值：**
- `name`: `"llama3.1:8b"`
- `reasoning`: `false`
- `input`: `["text"]`
- `contextWindow`: `128000`
- `maxTokens`: `16384`
- `cost`: 全为 0

### 5.3 完整模型配置

```json
{
  "models": [
    {
      "id": "llama3.1:8b",
      "name": "Llama 3.1 8B (Local)",
      "reasoning": false,
      "input": ["text"],
      "contextWindow": 128000,
      "maxTokens": 32000,
      "cost": {
        "input": 0,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0
      }
    }
  ]
}
```

### 5.4 多模态模型配置

```json
{
  "models": [
    {
      "id": "llava:13b",
      "name": "LLaVA 13B (Vision)",
      "input": ["text", "image"],
      "contextWindow": 4096,
      "maxTokens": 2048
    }
  ]
}
```

**`input` 字段：**
- `["text"]`: 仅文本
- `["text", "image"]`: 文本 + 图片

### 5.5 推理模型配置

```json
{
  "models": [
    {
      "id": "deepseek-r1:70b",
      "name": "DeepSeek R1 70B",
      "reasoning": true,
      "contextWindow": 64000,
      "maxTokens": 8192
    }
  ]
}
```

**`reasoning: true` 效果：**
- 启用扩展思考模式
- 支持 `reasoning_effort` 参数
- 显示思考过程

---

## 六、常见配置场景

### 6.1 Ollama 配置

```json
{
  "providers": {
    "ollama": {
      "baseUrl": "http://localhost:11434/v1",
      "api": "openai-completions",
      "apiKey": "ollama",
      "models": [
        {
          "id": "llama3.1:8b",
          "name": "Llama 3.1 8B"
        },
        {
          "id": "qwen2.5-coder:7b",
          "name": "Qwen 2.5 Coder 7B"
        },
        {
          "id": "deepseek-r1:7b",
          "name": "DeepSeek R1 7B",
          "reasoning": true
        }
      ]
    }
  }
}
```

**使用方式：**
```bash
# 启动 Ollama
ollama serve

# 拉取模型
ollama pull llama3.1:8b

# 使用 Pi
pi --provider ollama --model llama3.1:8b
```

### 6.2 LM Studio 配置

```json
{
  "providers": {
    "lm-studio": {
      "baseUrl": "http://localhost:1234/v1",
      "api": "openai-completions",
      "apiKey": "lm-studio",
      "models": [
        {
          "id": "local-model",
          "name": "LM Studio Local Model"
        }
      ]
    }
  }
}
```

**使用方式：**
```bash
# 启动 LM Studio 服务器
# 在 LM Studio GUI 中启用 Local Server

# 使用 Pi
pi --provider lm-studio --model local-model
```

### 6.3 vLLM 配置

```json
{
  "providers": {
    "vllm": {
      "baseUrl": "http://localhost:8000/v1",
      "api": "openai-completions",
      "apiKey": "vllm",
      "models": [
        {
          "id": "meta-llama/Llama-3.1-8B-Instruct",
          "name": "Llama 3.1 8B (vLLM)",
          "contextWindow": 128000,
          "maxTokens": 32000
        }
      ]
    }
  }
}
```

**启动 vLLM：**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8000
```

### 6.4 企业代理配置

```json
{
  "providers": {
    "corp-proxy": {
      "baseUrl": "https://llm-proxy.corp.example.com/v1",
      "api": "anthropic-messages",
      "apiKey": "!aws secretsmanager get-secret-value --secret-id anthropic-key --query SecretString --output text",
      "headers": {
        "x-corp-auth": "CORP_AUTH_TOKEN",
        "x-department": "engineering"
      },
      "models": [
        {
          "id": "claude-opus-4",
          "name": "Claude Opus 4 (Corp Proxy)"
        }
      ]
    }
  }
}
```

### 6.5 OpenRouter 配置

```json
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
        }
      ]
    }
  }
}
```

### 6.6 Vercel AI Gateway 配置

```json
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

## 七、覆盖内置 Provider

### 7.1 覆盖 Base URL

```json
{
  "providers": {
    "anthropic": {
      "baseUrl": "https://my-proxy.example.com/v1"
    }
  }
}
```

**效果：**
- 所有 Anthropic 内置模型仍然可用
- 请求发送到自定义代理
- OAuth 和 API Key 认证继续工作

### 7.2 合并自定义模型

```json
{
  "providers": {
    "anthropic": {
      "baseUrl": "https://my-proxy.example.com/v1",
      "models": [
        {
          "id": "claude-custom",
          "name": "Claude Custom Model"
        }
      ]
    }
  }
}
```

**合并语义：**
- 保留所有内置模型
- 添加自定义模型
- 如果 `id` 冲突，自定义模型替换内置模型

### 7.3 Per-model Overrides

```json
{
  "providers": {
    "openrouter": {
      "modelOverrides": {
        "anthropic/claude-sonnet-4": {
          "name": "Claude Sonnet 4 (Bedrock Route)",
          "compat": {
            "openRouterRouting": {
              "only": ["amazon-bedrock"]
            }
          }
        }
      }
    }
  }
}
```

**支持的覆盖字段：**
- `name`
- `reasoning`
- `input`
- `cost`（部分）
- `contextWindow`
- `maxTokens`
- `headers`
- `compat`

---

## 八、OpenAI 兼容性配置

### 8.1 compat 字段

```json
{
  "providers": {
    "local-llm": {
      "baseUrl": "http://localhost:8080/v1",
      "api": "openai-completions",
      "compat": {
        "supportsUsageInStreaming": false,
        "maxTokensField": "max_tokens"
      },
      "models": [...]
    }
  }
}
```

### 8.2 compat 字段列表

| 字段 | 描述 |
|------|------|
| `supportsStore` | 支持 `store` 字段 |
| `supportsDeveloperRole` | 使用 `developer` 角色而非 `system` |
| `supportsReasoningEffort` | 支持 `reasoning_effort` 参数 |
| `supportsUsageInStreaming` | 支持 `stream_options: { include_usage: true }` |
| `maxTokensField` | 使用 `max_completion_tokens` 或 `max_tokens` |
| `openRouterRouting` | OpenRouter 路由配置 |
| `vercelGatewayRouting` | Vercel AI Gateway 路由配置 |

### 8.3 常见兼容性问题

**问题 1：不支持流式 Usage**

```json
{
  "compat": {
    "supportsUsageInStreaming": false
  }
}
```

**问题 2：maxTokens 字段名称不同**

```json
{
  "compat": {
    "maxTokensField": "max_tokens"
  }
}
```

**问题 3：不支持 Developer Role**

```json
{
  "compat": {
    "supportsDeveloperRole": false
  }
}
```

---

## 九、故障排查

### 9.1 模型未显示

**问题：** 配置后模型未出现在 `/model` 选择器中

**排查步骤：**

```bash
# 1. 检查 models.json 语法
cat ~/.pi/agent/models.json | jq .

# 2. 检查 Pi 日志
PI_DEBUG=1 pi
/model

# 3. 验证 baseUrl 可访问
curl http://localhost:11434/v1/models

# 4. 重新加载配置
pi
/reload
/model
```

### 9.2 API 调用失败

**问题：** `Error: Failed to call model`

**排查步骤：**

```bash
# 1. 测试 API 端点
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "Hi"}]
  }'

# 2. 检查 API Key
echo $OPENROUTER_API_KEY

# 3. 检查 compat 配置
# 添加 supportsUsageInStreaming: false
```

### 9.3 认证失败

**问题：** `401 Unauthorized`

**解决方案：**

```json
{
  "providers": {
    "custom": {
      "apiKey": "!echo $MY_API_KEY",
      "authHeader": true
    }
  }
}
```

---

## 十、最佳实践

### 10.1 配置组织

```json
{
  "providers": {
    "ollama-local": {
      "baseUrl": "http://localhost:11434/v1",
      "api": "openai-completions",
      "apiKey": "ollama",
      "models": [...]
    },
    "ollama-remote": {
      "baseUrl": "http://192.168.1.100:11434/v1",
      "api": "openai-completions",
      "apiKey": "ollama",
      "models": [...]
    },
    "corp-proxy": {
      "baseUrl": "https://llm-proxy.corp.example.com/v1",
      "api": "anthropic-messages",
      "apiKey": "!aws secretsmanager get-secret-value ...",
      "models": [...]
    }
  }
}
```

### 10.2 安全建议

```json
{
  "providers": {
    "secure": {
      "apiKey": "!op read 'op://vault/item/credential'",
      "headers": {
        "x-secret": "!security find-generic-password -ws 'secret'"
      }
    }
  }
}
```

**不要：**
- ❌ 在 models.json 中硬编码密钥
- ❌ 提交 models.json 到 Git

**应该：**
- ✅ 使用 Shell 命令读取密钥
- ✅ 使用环境变量引用
- ✅ 添加 models.json 到 .gitignore

### 10.3 成本追踪

```json
{
  "models": [
    {
      "id": "llama3.1:8b",
      "cost": {
        "input": 0,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0
      }
    },
    {
      "id": "claude-opus-4",
      "cost": {
        "input": 15,
        "output": 75,
        "cacheRead": 1.5,
        "cacheWrite": 18.75
      }
    }
  ]
}
```

**成本单位：** 每百万 tokens 的美元

---

## 十一、总结

### 11.1 核心要点

1. **models.json 位置**：`~/.pi/agent/models.json`
2. **最小配置**：`baseUrl`, `api`, `apiKey`, `models[].id`
3. **API 类型**：`openai-completions`（最兼容）
4. **值解析**：Shell 命令、环境变量、字面值
5. **热重载**：打开 `/model` 自动重新加载

### 11.2 配置模板

```json
{
  "providers": {
    "provider-name": {
      "baseUrl": "http://localhost:8080/v1",
      "api": "openai-completions",
      "apiKey": "your-key",
      "models": [
        {
          "id": "model-id",
          "name": "Model Display Name"
        }
      ]
    }
  }
}
```

### 11.3 下一步

- 学习 **多 Provider 切换**（核心概念 08）
- 掌握 **实战代码示例**（实战代码 01-08）
- 了解 **Extensions 开发**（Phase 3）

---

**参考资料：**
- [Pi Models Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/models.md)
- [Pi Custom Provider Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/custom-provider.md)
- [Pi Providers Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/providers.md)

**文档版本**：v1.0 (2026-02-18)
