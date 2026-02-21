# 核心概念：models.json配置

## 概述

`models.json`是pi-mono中最简单的Provider集成方式，通过声明式配置即可添加自定义LLM Provider和模型，无需编写任何代码。这种方式适用于80%的场景，特别是OpenAI兼容的API。

**文件位置：**
- 用户级：`~/.pi/agent/models.json`（所有项目共享）
- 项目级：`.pi/agent/models.json`（仅当前项目）

**配置优先级：**
1. 项目级配置（`.pi/agent/models.json`）
2. 用户级配置（`~/.pi/agent/models.json`）
3. 内置默认配置

**热重载：**
- 每次打开`/model`命令时自动重新加载
- 无需重启pi即可生效
- 编辑配置后立即可用

## 文件结构

### 基本结构

```json
{
  "providers": {
    "provider-id": {
      "baseUrl": "https://api.example.com/v1",
      "apiKey": "API_KEY_ENV_VAR",
      "api": "openai-completions",
      "models": [
        {
          "id": "model-id",
          "name": "Model Display Name",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 128000,
          "maxTokens": 16384,
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

### 完整Schema

```typescript
interface ModelsConfig {
  providers: {
    [providerId: string]: ProviderConfig
  }
}

interface ProviderConfig {
  // 必需字段（定义models时）
  baseUrl: string              // API端点URL
  apiKey: string               // API Key或环境变量名
  api: ApiType                 // API类型

  // 可选字段
  headers?: Record<string, string>  // 自定义请求头
  authHeader?: boolean         // 是否自动添加Authorization头
  models?: ModelConfig[]       // 模型列表
  modelOverrides?: Record<string, Partial<ModelConfig>>  // 覆盖内置模型
  compat?: CompatConfig        // OpenAI兼容性配置
}

interface ModelConfig {
  // 必需字段
  id: string                   // 模型ID（传递给API）

  // 可选字段（有默认值）
  name?: string                // 显示名称（默认=id）
  api?: ApiType                // 覆盖Provider的api
  reasoning?: boolean          // 是否支持推理（默认=false）
  input?: ("text" | "image")[] // 输入类型（默认=["text"]）
  contextWindow?: number       // 上下文窗口（默认=128000）
  maxTokens?: number           // 最大输出Token（默认=16384）
  cost?: CostConfig            // 成本配置（默认=全0）
  headers?: Record<string, string>  // 模型特定请求头
  compat?: CompatConfig        // 模型特定兼容性配置
}

interface CostConfig {
  input: number        // 输入成本（$/百万Token）
  output: number       // 输出成本（$/百万Token）
  cacheRead: number    // 缓存读取成本（$/百万Token）
  cacheWrite: number   // 缓存写入成本（$/百万Token）
}

type ApiType =
  | "openai-completions"      // OpenAI Chat Completions API
  | "openai-responses"        // OpenAI Responses API
  | "anthropic-messages"      // Anthropic Messages API
  | "google-generative-ai"    // Google Generative AI
  | "google-vertex"           // Google Vertex AI
  | "bedrock-converse-stream" // Amazon Bedrock
  | "azure-openai-responses"  // Azure OpenAI
```

## Provider配置详解

### 1. baseUrl（API端点）

**作用：** 指定LLM API的基础URL

**示例：**
```json
{
  "providers": {
    "ollama": {
      "baseUrl": "http://localhost:11434/v1"
    },
    "vllm": {
      "baseUrl": "http://your-server:8000/v1"
    },
    "proxy": {
      "baseUrl": "https://proxy.company.com/openai/v1"
    }
  }
}
```

**注意事项：**
- 必须包含协议（http://或https://）
- 通常以`/v1`结尾（OpenAI兼容API）
- 本地服务使用`http://localhost`
- 生产环境必须使用`https://`

### 2. apiKey（API密钥）

**作用：** 指定API密钥或环境变量名

**三种格式：**

**格式1：环境变量名（推荐）**
```json
{
  "apiKey": "MY_API_KEY"
}
```
pi会读取环境变量`$MY_API_KEY`的值。

**格式2：Shell命令**
```json
{
  "apiKey": "!security find-generic-password -ws 'anthropic'"
}
```
以`!`开头，执行Shell命令并使用stdout作为API Key。

**格式3：字面值（不推荐）**
```json
{
  "apiKey": "sk-ant-api03-..."
}
```
直接使用字面值，不安全，仅用于测试。

**最佳实践：**
```bash
# 设置环境变量
export OLLAMA_API_KEY=ollama
export ANTHROPIC_API_KEY=sk-ant-...

# 或在~/.bashrc或~/.zshrc中添加
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.zshrc
```

### 3. api（API类型）

**作用：** 指定使用哪种流式处理实现

**常用类型：**

| API类型 | 适用场景 | 示例Provider |
|---------|---------|-------------|
| `openai-completions` | OpenAI兼容API（最常用） | Ollama, vLLM, LiteLLM, OpenRouter |
| `anthropic-messages` | Anthropic格式 | Anthropic Claude |
| `google-generative-ai` | Google格式 | Google Gemini |
| `openai-responses` | OpenAI Responses API | OpenAI GPT-4 |
| `bedrock-converse-stream` | AWS Bedrock | Amazon Bedrock |

**判断标准：**
- API文档说"OpenAI兼容" → `openai-completions`
- Anthropic格式 → `anthropic-messages`
- Google格式 → `google-generative-ai`
- 完全自定义 → 使用Extension实现streamSimple

**示例：**
```json
{
  "providers": {
    "ollama": {
      "api": "openai-completions"  // Ollama提供OpenAI兼容API
    },
    "anthropic": {
      "api": "anthropic-messages"  // Anthropic自己的格式
    }
  }
}
```

### 4. headers（自定义请求头）

**作用：** 添加自定义HTTP请求头

**示例：**
```json
{
  "providers": {
    "proxy": {
      "baseUrl": "https://proxy.example.com/v1",
      "apiKey": "MY_API_KEY",
      "headers": {
        "X-Portkey-API-Key": "PORTKEY_API_KEY",
        "X-Custom-Header": "value",
        "X-Secret": "!op read 'op://vault/item/secret'"
      }
    }
  }
}
```

**值解析规则：**
- 环境变量名：`"PORTKEY_API_KEY"` → 读取`$PORTKEY_API_KEY`
- Shell命令：`"!op read ..."` → 执行命令并使用输出
- 字面值：`"value"` → 直接使用

**常见用途：**
- 代理认证：`X-Portkey-API-Key`, `X-Gateway-Auth`
- 路由控制：`X-Provider`, `X-Model-Route`
- 追踪标识：`X-Request-ID`, `X-Trace-ID`

### 5. authHeader（自动Authorization头）

**作用：** 自动添加`Authorization: Bearer <apiKey>`请求头

**示例：**
```json
{
  "providers": {
    "custom-api": {
      "baseUrl": "https://api.example.com",
      "apiKey": "MY_API_KEY",
      "authHeader": true,  // 自动添加Authorization头
      "api": "openai-completions"
    }
  }
}
```

**等价于：**
```json
{
  "headers": {
    "Authorization": "Bearer MY_API_KEY"
  }
}
```

**使用场景：**
- 自定义API需要Bearer token认证
- 但不是标准的OpenAI/Anthropic格式

### 6. compat（兼容性配置）

**作用：** 处理OpenAI兼容API的差异

**常用选项：**

```json
{
  "compat": {
    "supportsUsageInStreaming": false,      // 不支持流式usage
    "supportsDeveloperRole": false,         // 不支持developer角色
    "supportsReasoningEffort": false,       // 不支持reasoning_effort
    "maxTokensField": "max_tokens",         // 使用max_tokens而非max_completion_tokens
    "requiresToolResultName": true,         // 工具结果需要name字段
    "requiresAssistantAfterToolResult": true,  // 工具结果后需要assistant消息
    "requiresThinkingAsText": true,         // thinking作为text处理
    "requiresMistralToolIds": true,         // 工具ID必须是9位字母数字
    "thinkingFormat": "qwen"                // 使用enable_thinking: true
  }
}
```

**示例：**
```json
{
  "providers": {
    "qwen": {
      "baseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "apiKey": "QWEN_API_KEY",
      "api": "openai-completions",
      "compat": {
        "supportsDeveloperRole": false,
        "thinkingFormat": "qwen"
      }
    }
  }
}
```

## Model配置详解

### 1. id（模型ID）

**作用：** 传递给API的模型标识符

**格式：** 通常为`provider/model-name`或直接`model-name`

**示例：**
```json
{
  "models": [
    { "id": "llama3.1:8b" },           // Ollama格式
    { "id": "gpt-4" },                 // OpenAI格式
    { "id": "claude-sonnet-4" },       // Anthropic格式
    { "id": "ollama/llama3.2" }        // 自定义格式
  ]
}
```

**注意：**
- `id`必须与API期望的模型名称完全一致
- 不同Provider的命名规则不同
- 可以通过API文档或`curl`测试确认

### 2. name（显示名称）

**作用：** 在`/model`选择器中显示的名称

**默认值：** 如果不指定，使用`id`

**示例：**
```json
{
  "models": [
    {
      "id": "llama3.1:8b",
      "name": "Llama 3.1 8B (Local)"  // 更友好的显示名称
    },
    {
      "id": "gpt-4",
      "name": "GPT-4 (OpenAI)"
    }
  ]
}
```

### 3. reasoning（推理能力）

**作用：** 模型是否支持扩展思考（extended thinking）

**默认值：** `false`

**示例：**
```json
{
  "models": [
    {
      "id": "claude-opus-4",
      "reasoning": true  // 支持推理
    },
    {
      "id": "llama3.1:8b",
      "reasoning": false  // 不支持推理
    }
  ]
}
```

**影响：**
- `reasoning: true`时，pi会发送`thinking`参数
- 用户可以使用`/reasoning`命令控制推理级别

### 4. input（输入类型）

**作用：** 模型支持的输入类型

**默认值：** `["text"]`

**可选值：** `["text"]`或`["text", "image"]`

**示例：**
```json
{
  "models": [
    {
      "id": "gpt-4-vision",
      "input": ["text", "image"]  // 支持图像输入
    },
    {
      "id": "llama3.1:8b",
      "input": ["text"]  // 仅支持文本
    }
  ]
}
```

### 5. contextWindow（上下文窗口）

**作用：** 模型的最大上下文长度（Token数）

**默认值：** `128000`

**示例：**
```json
{
  "models": [
    {
      "id": "claude-opus-4",
      "contextWindow": 200000  // 200K上下文
    },
    {
      "id": "llama3.1:8b",
      "contextWindow": 128000  // 128K上下文
    }
  ]
}
```

**注意：**
- 影响pi的上下文管理策略
- 超过限制时pi会自动压缩历史

### 6. maxTokens（最大输出Token）

**作用：** 模型单次响应的最大Token数

**默认值：** `16384`

**示例：**
```json
{
  "models": [
    {
      "id": "claude-opus-4",
      "maxTokens": 64000  // 最大64K输出
    },
    {
      "id": "llama3.1:8b",
      "maxTokens": 32000  // 最大32K输出
    }
  ]
}
```

### 7. cost（成本配置）

**作用：** 用于成本追踪和显示

**默认值：** 全部为0（本地模型）

**单位：** 美元/百万Token

**示例：**
```json
{
  "models": [
    {
      "id": "gpt-4",
      "cost": {
        "input": 30.0,      // $30/百万输入Token
        "output": 60.0,     // $60/百万输出Token
        "cacheRead": 3.0,   // $3/百万缓存读取Token
        "cacheWrite": 37.5  // $37.5/百万缓存写入Token
      }
    },
    {
      "id": "ollama/llama3.1:8b",
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

## 实际配置示例

### 示例1：Ollama本地模型（最简单）

```json
{
  "providers": {
    "ollama": {
      "baseUrl": "http://localhost:11434/v1",
      "apiKey": "ollama",
      "api": "openai-completions",
      "models": [
        { "id": "llama3.1:8b" },
        { "id": "qwen2.5-coder:7b" },
        { "id": "deepseek-coder:6.7b" }
      ]
    }
  }
}
```

**特点：**
- 最小配置，仅指定`id`
- 其他字段使用默认值
- `apiKey`可以是任意值（Ollama不验证）

### 示例2：vLLM高性能部署

```json
{
  "providers": {
    "vllm": {
      "baseUrl": "http://your-vllm-server:8000/v1",
      "apiKey": "vllm",
      "api": "openai-completions",
      "models": [
        {
          "id": "deepseek-coder-33b",
          "name": "DeepSeek Coder 33B (vLLM)",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 16384,
          "maxTokens": 4096,
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

### 示例3：企业代理路由

```json
{
  "providers": {
    "openai": {
      "baseUrl": "https://proxy.company.com/openai/v1",
      "headers": {
        "X-Company-Auth": "COMPANY_AUTH_TOKEN"
      }
    }
  }
}
```

**特点：**
- 覆盖内置Provider的`baseUrl`
- 添加自定义认证头
- 保留所有内置模型

### 示例4：OpenRouter多模型聚合

```json
{
  "providers": {
    "openrouter": {
      "baseUrl": "https://openrouter.ai/api/v1",
      "apiKey": "OPENROUTER_API_KEY",
      "api": "openai-completions",
      "models": [
        {
          "id": "anthropic/claude-3.5-sonnet",
          "name": "Claude 3.5 Sonnet (OpenRouter)",
          "reasoning": false,
          "input": ["text", "image"],
          "contextWindow": 200000,
          "maxTokens": 8192,
          "cost": {
            "input": 3.0,
            "output": 15.0,
            "cacheRead": 0,
            "cacheWrite": 0
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

### 示例5：多Provider混合配置

```json
{
  "providers": {
    "ollama": {
      "baseUrl": "http://localhost:11434/v1",
      "apiKey": "ollama",
      "api": "openai-completions",
      "models": [
        { "id": "llama3.1:8b", "name": "Llama 3.1 8B (Local)" },
        { "id": "qwen2.5-coder:7b", "name": "Qwen 2.5 Coder 7B (Local)" }
      ]
    },
    "vllm": {
      "baseUrl": "http://gpu-server:8000/v1",
      "apiKey": "vllm",
      "api": "openai-completions",
      "models": [
        {
          "id": "deepseek-coder-33b",
          "name": "DeepSeek Coder 33B (GPU)",
          "contextWindow": 16384,
          "maxTokens": 4096
        }
      ]
    },
    "openai": {
      "baseUrl": "https://proxy.company.com/openai/v1",
      "headers": {
        "X-Company-Auth": "COMPANY_AUTH_TOKEN"
      }
    }
  }
}
```

## 高级功能

### 1. 覆盖内置Provider

**场景：** 通过代理访问内置Provider

**配置：**
```json
{
  "providers": {
    "anthropic": {
      "baseUrl": "https://proxy.company.com/anthropic"
    }
  }
}
```

**效果：**
- 所有内置Anthropic模型保留
- 所有请求通过代理路由
- OAuth认证继续工作

### 2. 模型覆盖（modelOverrides）

**场景：** 修改内置模型的特定属性

**配置：**
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

**支持的字段：**
- `name`, `reasoning`, `input`, `cost`, `contextWindow`, `maxTokens`, `headers`, `compat`

### 3. 模型合并

**场景：** 在内置Provider中添加自定义模型

**配置：**
```json
{
  "providers": {
    "anthropic": {
      "models": [
        {
          "id": "claude-custom-model",
          "name": "Claude Custom Model",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": 200000,
          "maxTokens": 64000
        }
      ]
    }
  }
}
```

**合并规则：**
- 内置模型保留
- 自定义模型按`id`合并
- 相同`id`时，自定义模型覆盖内置模型

## 最佳实践

### 1. 安全性

**✅ 推荐：**
```json
{
  "apiKey": "MY_API_KEY"  // 环境变量
}
```

```bash
export MY_API_KEY=sk-...
```

**❌ 不推荐：**
```json
{
  "apiKey": "sk-ant-api03-..."  // 硬编码
}
```

### 2. 组织结构

**按用途分组：**
```json
{
  "providers": {
    // 本地开发
    "ollama": { ... },

    // 生产环境
    "vllm-prod": { ... },

    // 测试环境
    "vllm-test": { ... }
  }
}
```

### 3. 命名规范

**清晰的显示名称：**
```json
{
  "models": [
    {
      "id": "llama3.1:8b",
      "name": "Llama 3.1 8B (Local, Fast)"
    },
    {
      "id": "deepseek-coder-33b",
      "name": "DeepSeek Coder 33B (GPU, Slow)"
    }
  ]
}
```

### 4. 成本追踪

**准确的成本配置：**
```json
{
  "cost": {
    "input": 3.0,
    "output": 15.0,
    "cacheRead": 0.3,
    "cacheWrite": 3.75
  }
}
```

**用途：**
- pi会显示每次请求的成本
- 帮助控制API调用开销

### 5. 文档注释

**使用注释说明配置：**
```json
{
  "providers": {
    "ollama": {
      // 本地Ollama服务，用于快速原型开发
      "baseUrl": "http://localhost:11434/v1",
      "apiKey": "ollama",
      "api": "openai-completions",
      "models": [
        {
          "id": "llama3.1:8b",
          // 8B模型，速度快但能力有限
          "name": "Llama 3.1 8B (Local)"
        }
      ]
    }
  }
}
```

**注意：** JSON标准不支持注释，但pi会忽略注释行。

## 常见问题

### Q1: models.json不生效？

**检查清单：**
1. 文件路径正确？（`~/.pi/agent/models.json`或`.pi/agent/models.json`）
2. JSON格式正确？（使用`jq`验证：`jq . models.json`）
3. 重新打开`/model`？（配置在打开`/model`时加载）

### Q2: API Key无法读取？

**检查：**
```bash
# 检查环境变量
echo $MY_API_KEY

# 检查Shell命令
security find-generic-password -ws 'anthropic'
```

### Q3: 模型不显示在/model列表中？

**可能原因：**
- `id`字段缺失
- Provider配置不完整（缺少`baseUrl`、`apiKey`或`api`）
- JSON语法错误

### Q4: 如何测试配置？

**步骤：**
```bash
# 1. 启动pi
pi

# 2. 打开模型选择器
/model

# 3. 选择自定义模型
# 4. 发送测试消息
Hello, can you hear me?
```

### Q5: 如何调试API调用？

**方法1：查看pi日志**
```bash
# pi会输出API请求和响应
# 查找错误信息
```

**方法2：使用curl测试**
```bash
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## 总结

**models.json的核心价值：**
1. **零代码配置**：无需编写TypeScript代码
2. **热重载**：编辑后立即生效
3. **灵活性**：支持从简单到复杂的各种场景
4. **安全性**：支持环境变量和Shell命令
5. **可维护性**：声明式配置易于理解和修改

**适用场景：**
- ✅ OpenAI兼容API（Ollama、vLLM、LiteLLM）
- ✅ 代理路由
- ✅ 自定义请求头
- ✅ 覆盖内置Provider

**不适用场景：**
- ❌ 需要OAuth认证（使用Extension）
- ❌ 非标准API格式（使用Extension + streamSimple）
- ❌ 复杂的请求/响应转换（使用Extension）

**下一步：**
- 如果models.json无法满足需求，学习Extension开发
- 如果需要OAuth认证，学习OAuth Provider接口
- 如果需要自定义API适配，学习streamSimple实现
