# Extensions 扩展开发基础 - 实战代码 07：Provider定制扩展

> 通过 registerProvider API 注册自定义 LLM Provider

---

## 1. 场景介绍

### 1.1 为什么需要 Provider 定制扩展？

在 AI Agent 开发中，可能需要使用不同的 LLM Provider：

**场景 1：使用本地模型**
```
需求：使用 Ollama 运行本地模型
问题：pi-mono 默认只支持云端 API
解决：注册 Ollama Provider
```

**场景 2：使用自定义 API**
```
需求：使用公司内部的 LLM API
问题：API 格式与 OpenAI 不兼容
解决：注册自定义 Provider，实现流式解析
```

**场景 3：模型切换**
```
需求：在不同任务中使用不同模型
问题：需要手动修改配置
解决：注册多个 Provider，动态切换
```

### 1.2 Provider 定制的核心概念

**ProviderConfig 接口**：
```typescript
interface ProviderConfig {
  id: string;                    // Provider ID（唯一标识）
  name: string;                  // 显示名称
  models: ModelConfig[];         // 支持的模型列表
  apiKeyEnvVar?: string;         // API Key 环境变量名
  baseURL?: string;              // API 基础 URL
  headers?: Record<string, string>;  // 自定义请求头
  stream?: StreamFunction;       // 自定义流式解析
  oauth?: OAuthConfig;           // OAuth 配置（可选）
}

interface ModelConfig {
  id: string;                    // 模型 ID
  name: string;                  // 显示名称
  contextWindow: number;         // 上下文窗口大小
  maxOutputTokens: number;       // 最大输出 token 数
  supportsTools: boolean;        // 是否支持工具调用
  supportsVision: boolean;       // 是否支持视觉
  thinkingLevel?: number;        // 思考级别（0-3）
}
```

**StreamFunction 接口**：
```typescript
type StreamFunction = (
  response: Response,            // HTTP 响应
  signal: AbortSignal            // 取消信号
) => AsyncIterable<StreamChunk>;

interface StreamChunk {
  type: "text" | "tool_call" | "error";
  content?: string;              // 文本内容
  toolCall?: ToolCall;           // 工具调用
  error?: string;                // 错误信息
}
```

### 1.3 技术要点

- **registerProvider**：注册自定义 Provider
- **模型配置**：定义模型的能力和限制
- **流式解析**：实现自定义的 SSE 解析逻辑
- **OAuth 集成**：可选的 OAuth 认证流程
- **错误处理**：处理 API 错误和网络问题

---

## 2. 完整实现

### 2.1 基础版本：OpenAI 兼容 Provider

```typescript
/**
 * OpenAI-Compatible Provider Extension
 *
 * 功能：
 * - 注册 OpenAI 兼容的 Provider
 * - 支持自定义 API 端点
 * - 支持多个模型
 */

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  pi.registerProvider({
    id: "custom-openai",
    name: "Custom OpenAI",
    apiKeyEnvVar: "CUSTOM_OPENAI_API_KEY",
    baseURL: "https://api.custom-openai.com/v1",

    models: [
      {
        id: "gpt-4-turbo",
        name: "GPT-4 Turbo",
        contextWindow: 128000,
        maxOutputTokens: 4096,
        supportsTools: true,
        supportsVision: true,
      },
      {
        id: "gpt-3.5-turbo",
        name: "GPT-3.5 Turbo",
        contextWindow: 16385,
        maxOutputTokens: 4096,
        supportsTools: true,
        supportsVision: false,
      },
    ],
  });
}
```

### 2.2 进阶版本：Ollama Provider

```typescript
/**
 * Ollama Provider Extension
 *
 * 功能：
 * - 注册 Ollama 本地模型 Provider
 * - 自动检测可用模型
 * - 自定义流式解析
 */

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  // Ollama 默认运行在 localhost:11434
  const ollamaBaseURL = process.env.OLLAMA_BASE_URL || "http://localhost:11434";

  // 自定义流式解析（Ollama 使用 JSON Lines 格式）
  async function* parseOllamaStream(
    response: Response,
    signal: AbortSignal
  ): AsyncIterable<any> {
    const reader = response.body?.getReader();
    if (!reader) throw new Error("No response body");

    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        if (signal.aborted) {
          reader.cancel();
          break;
        }

        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;

          try {
            const data = JSON.parse(line);

            // Ollama 格式：{ "model": "...", "response": "...", "done": false }
            if (data.response) {
              yield {
                type: "text",
                content: data.response,
              };
            }

            if (data.done) {
              break;
            }
          } catch (error) {
            console.error("Failed to parse Ollama response:", error);
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  pi.registerProvider({
    id: "ollama",
    name: "Ollama (Local)",
    baseURL: ollamaBaseURL,

    models: [
      {
        id: "llama3.1:8b",
        name: "Llama 3.1 8B",
        contextWindow: 128000,
        maxOutputTokens: 4096,
        supportsTools: false,
        supportsVision: false,
      },
      {
        id: "llama3.1:70b",
        name: "Llama 3.1 70B",
        contextWindow: 128000,
        maxOutputTokens: 4096,
        supportsTools: false,
        supportsVision: false,
      },
      {
        id: "codellama:13b",
        name: "Code Llama 13B",
        contextWindow: 16384,
        maxOutputTokens: 4096,
        supportsTools: false,
        supportsVision: false,
      },
    ],

    stream: parseOllamaStream,
  });
}
```

### 2.3 高级版本：完整自定义 Provider

```typescript
/**
 * Custom Provider Extension - 完整自定义 Provider
 *
 * 功能：
 * - 自定义 API 格式
 * - 自定义流式解析
 * - 工具调用支持
 * - 错误处理
 * - OAuth 认证（可选）
 */

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  // 自定义流式解析
  async function* parseCustomStream(
    response: Response,
    signal: AbortSignal
  ): AsyncIterable<any> {
    const reader = response.body?.getReader();
    if (!reader) throw new Error("No response body");

    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        if (signal.aborted) {
          reader.cancel();
          break;
        }

        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // 处理 SSE 格式：data: {...}\n\n
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;

          // 移除 "data: " 前缀
          const dataLine = line.replace(/^data:\s*/, "");

          // 跳过特殊标记
          if (dataLine === "[DONE]") {
            break;
          }

          try {
            const data = JSON.parse(dataLine);

            // 解析文本内容
            if (data.choices?.[0]?.delta?.content) {
              yield {
                type: "text",
                content: data.choices[0].delta.content,
              };
            }

            // 解析工具调用
            if (data.choices?.[0]?.delta?.tool_calls) {
              for (const toolCall of data.choices[0].delta.tool_calls) {
                yield {
                  type: "tool_call",
                  toolCall: {
                    id: toolCall.id,
                    name: toolCall.function?.name,
                    arguments: toolCall.function?.arguments,
                  },
                };
              }
            }

            // 检查错误
            if (data.error) {
              yield {
                type: "error",
                error: data.error.message || "Unknown error",
              };
              break;
            }
          } catch (error) {
            console.error("Failed to parse stream chunk:", error);
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  pi.registerProvider({
    id: "custom-llm",
    name: "Custom LLM Provider",
    apiKeyEnvVar: "CUSTOM_LLM_API_KEY",
    baseURL: "https://api.custom-llm.com/v1",

    // 自定义请求头
    headers: {
      "X-Custom-Header": "value",
      "User-Agent": "pi-mono/1.0",
    },

    models: [
      {
        id: "custom-model-large",
        name: "Custom Model Large",
        contextWindow: 200000,
        maxOutputTokens: 8192,
        supportsTools: true,
        supportsVision: true,
        thinkingLevel: 2,  // 支持思考模式
      },
      {
        id: "custom-model-fast",
        name: "Custom Model Fast",
        contextWindow: 128000,
        maxOutputTokens: 4096,
        supportsTools: true,
        supportsVision: false,
        thinkingLevel: 0,  // 不支持思考模式
      },
    ],

    stream: parseCustomStream,

    // OAuth 配置（可选）
    oauth: {
      authorizationURL: "https://auth.custom-llm.com/oauth/authorize",
      tokenURL: "https://auth.custom-llm.com/oauth/token",
      clientId: process.env.CUSTOM_LLM_CLIENT_ID || "",
      clientSecret: process.env.CUSTOM_LLM_CLIENT_SECRET || "",
      scopes: ["llm.read", "llm.write"],
    },
  });
}
```

---

## 3. 代码解析

### 3.1 registerProvider API

```typescript
pi.registerProvider({
  id: "provider-id",             // 唯一标识
  name: "Provider Name",         // 显示名称
  apiKeyEnvVar: "API_KEY_VAR",   // API Key 环境变量
  baseURL: "https://api.example.com/v1",  // API 基础 URL
  headers: { /* 自定义请求头 */ },
  models: [ /* 模型配置 */ ],
  stream: customStreamParser,    // 自定义流式解析
  oauth: { /* OAuth 配置 */ },
});
```

**注册后的效果：**
- Provider 出现在模型选择列表中
- 可以通过 `--model provider-id/model-id` 使用
- 可以在 `models.json` 中配置

### 3.2 模型配置

```typescript
{
  id: "model-id",                // 模型 ID
  name: "Model Name",            // 显示名称
  contextWindow: 128000,         // 上下文窗口（tokens）
  maxOutputTokens: 4096,         // 最大输出（tokens）
  supportsTools: true,           // 是否支持工具调用
  supportsVision: true,          // 是否支持视觉输入
  thinkingLevel: 2,              // 思考级别（0-3）
}
```

**thinkingLevel 说明：**
- `0`：不支持思考模式
- `1`：基础思考（简单推理）
- `2`：中级思考（复杂推理）
- `3`：高级思考（深度推理）

### 3.3 流式解析

```typescript
async function* parseStream(
  response: Response,
  signal: AbortSignal
): AsyncIterable<StreamChunk> {
  const reader = response.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      // 检查取消
      if (signal.aborted) {
        reader.cancel();
        break;
      }

      // 读取数据
      const { done, value } = await reader.read();
      if (done) break;

      // 解码
      buffer += decoder.decode(value, { stream: true });

      // 解析（根据 API 格式）
      // ...

      // 生成 chunk
      yield {
        type: "text",
        content: "...",
      };
    }
  } finally {
    reader.releaseLock();
  }
}
```

**StreamChunk 类型：**
```typescript
// 文本内容
{ type: "text", content: "Hello" }

// 工具调用
{
  type: "tool_call",
  toolCall: {
    id: "call_123",
    name: "get_weather",
    arguments: '{"city": "Beijing"}',
  },
}

// 错误
{ type: "error", error: "API error message" }
```

### 3.4 OAuth 集成

```typescript
oauth: {
  authorizationURL: "https://auth.example.com/oauth/authorize",
  tokenURL: "https://auth.example.com/oauth/token",
  clientId: process.env.CLIENT_ID || "",
  clientSecret: process.env.CLIENT_SECRET || "",
  scopes: ["read", "write"],
}
```

**OAuth 流程：**
1. 用户首次使用时，pi-mono 打开浏览器进行授权
2. 用户授权后，获取 access token
3. 后续请求自动使用 access token
4. Token 过期时自动刷新

---

## 4. 变体与扩展

### 4.1 动态模型发现

```typescript
export default function (pi: ExtensionAPI) {
  // 从 API 获取可用模型
  async function fetchAvailableModels(): Promise<any[]> {
    try {
      const response = await fetch("http://localhost:11434/api/tags");
      const data = await response.json();

      return data.models.map((model: any) => ({
        id: model.name,
        name: model.name,
        contextWindow: 128000,
        maxOutputTokens: 4096,
        supportsTools: false,
        supportsVision: false,
      }));
    } catch (error) {
      console.error("Failed to fetch models:", error);
      return [];
    }
  }

  // 异步注册 Provider
  fetchAvailableModels().then((models) => {
    pi.registerProvider({
      id: "ollama",
      name: "Ollama (Local)",
      baseURL: "http://localhost:11434",
      models,
    });
  });
}
```

### 4.2 多 Provider 管理

```typescript
export default function (pi: ExtensionAPI) {
  const providers = [
    {
      id: "openai",
      name: "OpenAI",
      apiKeyEnvVar: "OPENAI_API_KEY",
      baseURL: "https://api.openai.com/v1",
      models: [/* ... */],
    },
    {
      id: "anthropic",
      name: "Anthropic",
      apiKeyEnvVar: "ANTHROPIC_API_KEY",
      baseURL: "https://api.anthropic.com/v1",
      models: [/* ... */],
    },
    {
      id: "ollama",
      name: "Ollama",
      baseURL: "http://localhost:11434",
      models: [/* ... */],
    },
  ];

  // 注册所有 Providers
  for (const provider of providers) {
    pi.registerProvider(provider);
  }

  // 注册切换命令
  pi.registerCommand("switch-provider", {
    description: "Switch to a different provider",
    handler: async (args, ctx) => {
      const providerId = args.trim();

      if (!providerId) {
        const list = providers.map((p) => `${p.id}: ${p.name}`).join("\n");
        ctx.ui.notify(`Available providers:\n${list}`, "info");
        return;
      }

      const provider = providers.find((p) => p.id === providerId);
      if (!provider) {
        ctx.ui.notify(`Provider not found: ${providerId}`, "error");
        return;
      }

      ctx.ui.notify(`Switched to ${provider.name}`, "success");
      // 注意：实际切换需要修改配置或重启
    },
  });
}
```

### 4.3 Provider 健康检查

```typescript
export default function (pi: ExtensionAPI) {
  async function checkProviderHealth(baseURL: string): Promise<boolean> {
    try {
      const response = await fetch(`${baseURL}/health`, {
        method: "GET",
        signal: AbortSignal.timeout(5000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  pi.registerCommand("check-providers", {
    description: "Check provider health status",
    handler: async (_args, ctx) => {
      const providers = [
        { name: "OpenAI", url: "https://api.openai.com/v1" },
        { name: "Ollama", url: "http://localhost:11434" },
      ];

      const results: string[] = [];

      for (const provider of providers) {
        const healthy = await checkProviderHealth(provider.url);
        const status = healthy ? "✓ Healthy" : "✗ Unavailable";
        results.push(`${provider.name}: ${status}`);
      }

      ctx.ui.notify(`Provider Status:\n${results.join("\n")}`, "info");
    },
  });
}
```

### 4.4 自定义错误处理

```typescript
async function* parseStreamWithErrorHandling(
  response: Response,
  signal: AbortSignal
): AsyncIterable<any> {
  // 检查 HTTP 状态
  if (!response.ok) {
    const errorText = await response.text();
    yield {
      type: "error",
      error: `HTTP ${response.status}: ${errorText}`,
    };
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    yield { type: "error", error: "No response body" };
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      if (signal.aborted) {
        reader.cancel();
        yield { type: "error", error: "Request cancelled" };
        break;
      }

      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // 解析逻辑...
      try {
        // ...
      } catch (parseError: any) {
        console.error("Parse error:", parseError);
        yield {
          type: "error",
          error: `Parse error: ${parseError.message}`,
        };
      }
    }
  } catch (error: any) {
    yield {
      type: "error",
      error: `Stream error: ${error.message}`,
    };
  } finally {
    reader.releaseLock();
  }
}
```

---

## 5. 错误处理

### 5.1 API Key 缺失

```typescript
export default function (pi: ExtensionAPI) {
  const apiKey = process.env.CUSTOM_API_KEY;

  if (!apiKey) {
    console.warn("CUSTOM_API_KEY not set, provider may not work");
  }

  pi.registerProvider({
    id: "custom",
    name: "Custom Provider",
    apiKeyEnvVar: "CUSTOM_API_KEY",
    // ...
  });
}
```

### 5.2 网络错误

```typescript
async function* parseStreamSafely(
  response: Response,
  signal: AbortSignal
): AsyncIterable<any> {
  try {
    // 流式解析逻辑...
    yield* parseStream(response, signal);
  } catch (error: any) {
    if (error.name === "AbortError") {
      yield { type: "error", error: "Request cancelled" };
    } else if (error.message.includes("network")) {
      yield { type: "error", error: "Network error" };
    } else {
      yield { type: "error", error: error.message };
    }
  }
}
```

### 5.3 格式错误

```typescript
async function* parseStreamRobust(
  response: Response,
  signal: AbortSignal
): AsyncIterable<any> {
  const reader = response.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // 尝试解析
      try {
        const data = JSON.parse(buffer);
        buffer = "";  // 清空 buffer

        // 验证数据格式
        if (!data.choices || !Array.isArray(data.choices)) {
          console.warn("Invalid response format:", data);
          continue;
        }

        // 处理数据...
      } catch (parseError) {
        // 可能是不完整的 JSON，继续读取
        continue;
      }
    }
  } finally {
    reader.releaseLock();
  }
}
```

---

## 6. 最佳实践

### 6.1 Provider ID 命名

```typescript
// ✓ 好：清晰的 ID
pi.registerProvider({ id: "ollama-local", name: "Ollama (Local)" });
pi.registerProvider({ id: "openai-azure", name: "Azure OpenAI" });

// ✗ 差：模糊的 ID
pi.registerProvider({ id: "provider1", name: "Provider" });
```

### 6.2 模型配置

```typescript
// ✓ 好：准确的配置
{
  id: "gpt-4-turbo",
  contextWindow: 128000,        // 实际值
  maxOutputTokens: 4096,        // 实际值
  supportsTools: true,          // 实际支持
}

// ✗ 差：不准确的配置
{
  id: "gpt-4-turbo",
  contextWindow: 999999,        // 夸大
  maxOutputTokens: 999999,      // 夸大
  supportsTools: true,          // 实际不支持
}
```

### 6.3 错误处理

```typescript
// ✓ 好：详细的错误信息
yield {
  type: "error",
  error: `API error: ${response.status} ${response.statusText}`,
};

// ✗ 差：模糊的错误信息
yield { type: "error", error: "Error" };
```

### 6.4 流式解析

```typescript
// ✓ 好：支持取消
while (true) {
  if (signal.aborted) {
    reader.cancel();
    break;
  }
  // ...
}

// ✗ 差：不支持取消
while (true) {
  // 无法取消
}
```

---

## 7. 总结

### 7.1 核心要点

1. **registerProvider**：注册自定义 LLM Provider
2. **模型配置**：定义模型的能力和限制
3. **流式解析**：实现自定义的 SSE 解析逻辑
4. **错误处理**：处理 API 错误和网络问题
5. **OAuth 集成**：可选的 OAuth 认证流程

### 7.2 扩展方向

- **动态模型发现**：从 API 获取可用模型
- **多 Provider 管理**：注册和切换多个 Providers
- **健康检查**：检查 Provider 可用性
- **自定义错误处理**：提供友好的错误信息
- **性能优化**：缓存、重试、超时控制

### 7.3 实际应用

Provider 定制扩展适用于：
- **本地模型**：使用 Ollama、LM Studio 等本地模型
- **自定义 API**：集成公司内部的 LLM API
- **多云部署**：同时使用多个云服务商
- **成本优化**：根据任务选择不同价格的模型
- **隐私保护**：使用本地模型处理敏感数据

---

**下一步：** 学习 [07_实战代码_08_完整扩展实战.md](./07_实战代码_08_完整扩展实战.md)，了解如何构建一个完整的生产级扩展。
