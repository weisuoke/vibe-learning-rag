# 核心概念 01：Provider 定义与配置

> **理解 Provider 的本质：统一 API 抽象层**

---

## 什么是 Provider？

### 定义

**Provider** 是 Pi Coding Agent 中对 LLM 服务提供商的抽象层。它定义了：
- **API 端点**：如何连接到 LLM 服务
- **认证方式**：如何验证身份（API Key、OAuth、Shell Command）
- **模型列表**：该 Provider 支持哪些模型
- **请求格式**：如何构造 API 请求
- **响应解析**：如何解析 API 响应

### 为什么需要 Provider 抽象？

**问题**：不同 LLM 服务商的 API 格式完全不同

```typescript
// Anthropic API
const anthropicRequest = {
  model: "claude-3-5-sonnet-20241022",
  messages: [{ role: "user", content: "Hello" }],
  max_tokens: 1024
};

// OpenAI API
const openaiRequest = {
  model: "gpt-4-turbo",
  messages: [{ role: "user", content: "Hello" }],
  max_completion_tokens: 1024
};

// Google API
const googleRequest = {
  model: "models/gemini-2.0-flash-exp",
  contents: [{ role: "user", parts: [{ text: "Hello" }] }],
  generationConfig: { maxOutputTokens: 1024 }
};
```

**解决方案**：Provider 抽象层统一接口

```typescript
// Pi 统一接口
const piRequest = {
  provider: "anthropic", // 或 "openai" 或 "google"
  model: "claude-3-5-sonnet-20241022",
  messages: [{ role: "user", content: "Hello" }],
  maxTokens: 1024
};
```

---

## Provider 的结构

### models.json 配置文件

Pi 使用 `models.json` 文件定义所有 Provider 和模型。

**文件位置**：
- **全局配置**：`~/.pi/agent/models.json`
- **项目配置**：`.pi/models.json`（可选，覆盖全局配置）

**基本结构**：

```json
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
          }
        }
      }
    }
  }
}
```

### 关键字段解析

#### 1. apiType（API 类型）

定义 Provider 使用的 API 协议。

**支持的类型**：

| apiType | 说明 | 适用 Provider |
|---------|------|---------------|
| `anthropic-messages` | Anthropic Messages API | Anthropic (Claude) |
| `openai-completions` | OpenAI Chat Completions API | OpenAI, xAI, Groq, Mistral, Cohere |
| `google-generative-ai` | Google Generative AI API | Google (Gemini) |
| `openai-compatible` | OpenAI 兼容 API | Ollama, LM Studio, OpenRouter |

**示例**：

```json
{
  "providers": {
    "anthropic": {
      "apiType": "anthropic-messages"
    },
    "openai": {
      "apiType": "openai-completions"
    },
    "google": {
      "apiType": "google-generative-ai"
    },
    "ollama": {
      "apiType": "openai-compatible"
    }
  }
}
```

#### 2. baseUrl（API 端点）

定义 API 的基础 URL。

**默认值**：
- Anthropic: `https://api.anthropic.com`
- OpenAI: `https://api.openai.com`
- Google: `https://generativelanguage.googleapis.com`

**自定义端点**（用于代理或自托管）：

```json
{
  "providers": {
    "anthropic": {
      "apiType": "anthropic-messages",
      "baseUrl": "https://your-proxy.com/anthropic"
    },
    "ollama": {
      "apiType": "openai-compatible",
      "baseUrl": "http://localhost:11434"
    }
  }
}
```

#### 3. models（模型列表）

定义该 Provider 支持的所有模型。

**模型配置字段**：

```json
{
  "models": {
    "claude-3-5-sonnet-20241022": {
      "id": "claude-3-5-sonnet-20241022",           // 模型 ID（必需）
      "name": "Claude 3.5 Sonnet",                  // 显示名称
      "contextWindow": 200000,                      // 上下文窗口（tokens）
      "maxOutput": 8192,                            // 最大输出（tokens）
      "reasoning": true,                            // 是否支持推理
      "cost": {                                     // 成本（$/MTok）
        "input": 3.0,
        "output": 15.0
      },
      "tags": ["coding", "analysis"],               // 标签（可选）
      "deprecated": false                           // 是否已弃用
    }
  }
}
```

**字段说明**：

- **id**：模型的唯一标识符，必须与 API 调用时使用的 ID 一致
- **name**：用户友好的显示名称
- **contextWindow**：模型支持的最大上下文长度（输入 + 输出）
- **maxOutput**：单次请求的最大输出 tokens
- **reasoning**：是否支持推理模式（如 Claude 的 extended thinking）
- **cost**：每百万 tokens 的成本（美元）
  - `input`：输入 tokens 成本
  - `output`：输出 tokens 成本
- **tags**：用于分类和搜索的标签
- **deprecated**：标记已弃用的模型

---

## 认证方式

Pi 支持 3 种认证方式：

### 1. API Key（最常用）

**配置文件**：`~/.pi/agent/auth.json`

```json
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

**安全性**：
- 文件权限必须是 `0600`（仅所有者可读写）
- 不要将 `auth.json` 提交到版本控制
- 使用 `.gitignore` 排除该文件

**设置脚本**：

```bash
#!/bin/bash
# setup-auth.sh

AUTH_FILE="$HOME/.pi/agent/auth.json"

# 创建目录
mkdir -p "$(dirname "$AUTH_FILE")"

# 写入配置
cat > "$AUTH_FILE" <<EOF
{
  "anthropic": {
    "apiKey": "$ANTHROPIC_API_KEY"
  },
  "openai": {
    "apiKey": "$OPENAI_API_KEY"
  }
}
EOF

# 设置权限
chmod 600 "$AUTH_FILE"

echo "✅ Auth configured at $AUTH_FILE"
```

### 2. 环境变量

Pi 会自动读取以下环境变量：

```bash
# Anthropic
export ANTHROPIC_API_KEY="sk-ant-api03-..."

# OpenAI
export OPENAI_API_KEY="sk-proj-..."

# xAI
export XAI_API_KEY="xai-..."

# Google
export GOOGLE_API_KEY="AIza..."
```

**优先级**：环境变量 < auth.json < CLI 参数

### 3. Shell Command（高级）

用于动态获取 API Key（如从密钥管理服务）。

**配置示例**：

```json
{
  "providers": {
    "anthropic": {
      "apiType": "anthropic-messages",
      "auth": {
        "type": "shell",
        "command": "aws secretsmanager get-secret-value --secret-id anthropic-api-key --query SecretString --output text"
      }
    }
  }
}
```

**使用场景**：
- AWS Secrets Manager
- HashiCorp Vault
- 1Password CLI
- 企业密钥管理系统

---

## 内置 Provider

Pi 内置支持 15+ Provider，开箱即用。

### 主流云服务

#### 1. Anthropic (Claude)

```json
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
        },
        "claude-opus-4-20250514": {
          "id": "claude-opus-4-20250514",
          "name": "Claude Opus 4",
          "contextWindow": 200000,
          "maxOutput": 8192,
          "reasoning": true,
          "cost": { "input": 15.0, "output": 75.0 }
        }
      }
    }
  }
}
```

**特点**：
- 最强代码能力
- 支持 extended thinking（推理模式）
- 200K 上下文窗口

#### 2. OpenAI (GPT)

```json
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
          "cost": { "input": 10.0, "output": 30.0 }
        },
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
```

**特点**：
- 通用能力强
- 多模态支持（图像、音频）
- 128K 上下文窗口

#### 3. xAI (Grok)

```json
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
          "cost": { "input": 2.0, "output": 10.0 }
        },
        "grok-2-vision-1212": {
          "id": "grok-2-vision-1212",
          "name": "Grok 2 Vision",
          "contextWindow": 32768,
          "maxOutput": 16384,
          "cost": { "input": 2.0, "output": 10.0 }
        }
      }
    }
  }
}
```

**特点**：
- 实时信息（X 平台集成）
- 长上下文支持
- 多模态能力

#### 4. Google (Gemini)

```json
{
  "providers": {
    "google": {
      "apiType": "google-generative-ai",
      "baseUrl": "https://generativelanguage.googleapis.com",
      "models": {
        "gemini-2.0-flash-exp": {
          "id": "models/gemini-2.0-flash-exp",
          "name": "Gemini 2.0 Flash",
          "contextWindow": 1000000,
          "maxOutput": 8192,
          "cost": { "input": 0.0, "output": 0.0 }
        },
        "gemini-1.5-pro": {
          "id": "models/gemini-1.5-pro",
          "name": "Gemini 1.5 Pro",
          "contextWindow": 2000000,
          "maxOutput": 8192,
          "cost": { "input": 1.25, "output": 5.0 }
        }
      }
    }
  }
}
```

**特点**：
- 超长上下文（2M tokens）
- 免费额度（Flash 模型）
- 多模态支持

### 其他云服务

#### 5. Mistral

```json
{
  "providers": {
    "mistral": {
      "apiType": "openai-completions",
      "baseUrl": "https://api.mistral.ai",
      "models": {
        "mistral-large-latest": {
          "id": "mistral-large-latest",
          "name": "Mistral Large",
          "contextWindow": 128000,
          "maxOutput": 4096,
          "cost": { "input": 2.0, "output": 6.0 }
        }
      }
    }
  }
}
```

#### 6. Cohere

```json
{
  "providers": {
    "cohere": {
      "apiType": "openai-completions",
      "baseUrl": "https://api.cohere.ai",
      "models": {
        "command-r-plus": {
          "id": "command-r-plus",
          "name": "Command R+",
          "contextWindow": 128000,
          "maxOutput": 4096,
          "cost": { "input": 2.5, "output": 10.0 }
        }
      }
    }
  }
}
```

#### 7. Groq

```json
{
  "providers": {
    "groq": {
      "apiType": "openai-completions",
      "baseUrl": "https://api.groq.com",
      "models": {
        "llama-3.1-70b-versatile": {
          "id": "llama-3.1-70b-versatile",
          "name": "Llama 3.1 70B",
          "contextWindow": 131072,
          "maxOutput": 32768,
          "cost": { "input": 0.59, "output": 0.79 }
        }
      }
    }
  }
}
```

**特点**：超快推理速度（LPU 加速）

---

## 自定义 Provider

Pi 支持添加自定义 Provider，用于：
- 本地模型（Ollama、LM Studio）
- 聚合服务（OpenRouter）
- 企业私有部署
- 自托管模型

### 示例：Ollama

```json
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
          "cost": { "input": 0.0, "output": 0.0 }
        },
        "codellama:13b": {
          "id": "codellama:13b",
          "name": "Code Llama 13B (Local)",
          "contextWindow": 16384,
          "maxOutput": 4096,
          "cost": { "input": 0.0, "output": 0.0 }
        }
      }
    }
  }
}
```

### 示例：OpenRouter

```json
{
  "providers": {
    "openrouter": {
      "apiType": "openai-compatible",
      "baseUrl": "https://openrouter.ai/api",
      "models": {
        "anthropic/claude-3.5-sonnet": {
          "id": "anthropic/claude-3.5-sonnet",
          "name": "Claude 3.5 Sonnet (OpenRouter)",
          "contextWindow": 200000,
          "maxOutput": 8192,
          "cost": { "input": 3.0, "output": 15.0 }
        }
      }
    }
  }
}
```

---

## 配置验证

### 验证配置文件

```bash
# 检查配置文件语法
cat ~/.pi/agent/models.json | jq .

# 检查认证配置
cat ~/.pi/agent/auth.json | jq .

# 检查文件权限
ls -la ~/.pi/agent/auth.json
# 应该显示：-rw------- (600)
```

### 测试 Provider 连接

```bash
# 启动 Pi 并指定 Provider
pi --provider anthropic --model claude-3-5-sonnet-20241022

# 在 Pi 中测试
> /model
# 应该显示所有可用模型

> /session
# 应该显示当前使用的 Provider 和模型
```

---

## 常见问题

### Q1: 如何添加新的 Provider？

**答**：编辑 `~/.pi/agent/models.json`，添加新的 Provider 配置：

```json
{
  "providers": {
    "your-provider": {
      "apiType": "openai-compatible",
      "baseUrl": "https://your-api.com",
      "models": {
        "your-model": {
          "id": "your-model",
          "name": "Your Model",
          "contextWindow": 8192,
          "maxOutput": 2048,
          "cost": { "input": 1.0, "output": 2.0 }
        }
      }
    }
  }
}
```

然后在 `auth.json` 中添加认证：

```json
{
  "your-provider": {
    "apiKey": "your-api-key"
  }
}
```

### Q2: 如何覆盖内置 Provider 的配置？

**答**：在项目根目录创建 `.pi/models.json`，只需定义要覆盖的部分：

```json
{
  "providers": {
    "anthropic": {
      "baseUrl": "https://your-proxy.com/anthropic"
    }
  }
}
```

### Q3: 如何禁用某个 Provider？

**答**：在 `models.json` 中设置 `disabled: true`：

```json
{
  "providers": {
    "openai": {
      "disabled": true
    }
  }
}
```

### Q4: 如何使用企业代理？

**答**：设置 `baseUrl` 和自定义 headers：

```json
{
  "providers": {
    "anthropic": {
      "apiType": "anthropic-messages",
      "baseUrl": "https://corporate-proxy.com/anthropic",
      "headers": {
        "X-Corporate-Auth": "your-token"
      }
    }
  }
}
```

---

## 最佳实践

### 1. 配置分离

- **全局配置**：常用 Provider 和模型
- **项目配置**：项目特定的 Provider（如本地模型）
- **认证分离**：API Key 单独存储在 `auth.json`

### 2. 成本标注

为所有模型配置准确的 `cost` 字段，便于成本监控：

```json
{
  "models": {
    "claude-3-5-sonnet-20241022": {
      "cost": {
        "input": 3.0,   // $/MTok
        "output": 15.0  // $/MTok
      }
    }
  }
}
```

### 3. 标签分类

使用 `tags` 字段对模型分类：

```json
{
  "models": {
    "claude-3-5-sonnet-20241022": {
      "tags": ["coding", "analysis", "reasoning"]
    },
    "gpt-4o-mini": {
      "tags": ["cheap", "fast", "general"]
    }
  }
}
```

### 4. 版本管理

使用明确的模型版本号，避免使用 `latest`：

```typescript
// ✅ 推荐
"claude-3-5-sonnet-20241022"

// ❌ 不推荐
"claude-3-5-sonnet-latest"
```

---

## 下一步

- **Model 选择机制**：阅读 [03_核心概念_02_Model选择机制.md](./03_核心概念_02_Model选择机制.md)
- **实战配置**：阅读 [07_实战代码_01_基础Provider配置.md](./07_实战代码_01_基础Provider配置.md)

---

**记住**：Provider 配置是 Pi 多模型能力的基础，理解配置结构是掌握 Provider 切换的第一步。
