# Extensions 扩展开发基础 - 核心概念 05：Provider 与模型管理

> 深入理解自定义 Provider 注册、OAuth 集成和模型管理

---

## 概述

本文档深入讲解 Extensions 的 Provider 和模型管理机制，包括：
- registerProvider API 和 ProviderConfig
- OAuth 认证流程集成
- 自定义流式响应实现
- 模型管理和思考级别控制

---

## 1. Provider 概念

### 1.1 什么是 Provider？

**Provider 是 LLM API 的抽象层：**

```
pi-mono
  ↓
Provider 抽象层
  ↓
├── Anthropic Provider → Claude API
├── OpenAI Provider → GPT API
├── Google Provider → Gemini API
└── Custom Provider → 任何 LLM API
```

**Provider 的职责：**
1. **认证**：处理 API 密钥、OAuth 等认证方式
2. **请求转换**：将统一的请求格式转换为特定 API 格式
3. **流式响应**：实现流式响应的解析和处理
4. **模型管理**：定义可用的模型列表和配置

### 1.2 为什么需要自定义 Provider？

**使用场景：**
1. **集成企业内部 LLM API**
2. **使用本地模型**（Ollama、LM Studio）
3. **实现 API 代理和缓存**
4. **添加自定义认证逻辑**
5. **支持新的 LLM 服务**

---

## 2. registerProvider API

### 2.1 ProviderConfig 接口

**完整的 ProviderConfig 接口：**

```typescript
export interface ProviderConfig {
  // 基本信息
  id: string;                    // Provider ID（唯一标识）
  name: string;                  // Provider 名称

  // 模型列表
  models: ModelConfig[];

  // 流式响应实现
  createStream(
    request: StreamRequest,
    signal: AbortSignal
  ): Promise<AsyncIterable<StreamChunk>>;

  // 可选：OAuth 配置
  oauth?: OAuthConfig;

  // 可选：自定义配置
  config?: Record<string, unknown>;
}

export interface ModelConfig {
  id: string;                    // 模型 ID
  name: string;                  // 模型名称
  contextWindow: number;         // 上下文窗口大小
  maxOutputTokens?: number;      // 最大输出 tokens
  supportsThinking?: boolean;    // 是否支持思考模式
  thinkingLevels?: string[];     // 支持的思考级别
}
```

### 2.2 基本 Provider 注册

**注册一个简单的 Provider：**

```typescript
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  pi.registerProvider({
    id: "my-provider",
    name: "My Custom Provider",

    models: [
      {
        id: "my-model-1",
        name: "My Model 1",
        contextWindow: 128000,
        maxOutputTokens: 4096,
      },
      {
        id: "my-model-2",
        name: "My Model 2",
        contextWindow: 200000,
        maxOutputTokens: 8192,
        supportsThinking: true,
        thinkingLevels: ["low", "medium", "high"],
      },
    ],

    async createStream(request, signal) {
      // 实现流式响应
      return createCustomStream(request, signal);
    },
  });
}
```

---

## 3. 流式响应实现

### 3.1 StreamRequest 和 StreamChunk

**StreamRequest 接口：**

```typescript
export interface StreamRequest {
  model: string;                 // 模型 ID
  messages: Message[];           // 消息列表
  temperature?: number;          // 温度参数
  maxTokens?: number;            // 最大 tokens
  thinkingLevel?: string;        // 思考级别
  tools?: ToolDefinition[];      // 可用工具
  systemPrompt?: string;         // 系统提示词
}
```

**StreamChunk 接口：**

```typescript
export type StreamChunk =
  | { type: "text"; text: string }
  | { type: "thinking"; text: string }
  | { type: "tool_call"; toolCallId: string; toolName: string; input: unknown }
  | { type: "tool_result"; toolCallId: string; result: ToolResult }
  | { type: "error"; error: string }
  | { type: "done" };
```

### 3.2 实现 OpenAI 兼容的 Provider

**完整示例：**

```typescript
export default function (pi: ExtensionAPI) {
  pi.registerProvider({
    id: "openai-compatible",
    name: "OpenAI Compatible API",

    models: [
      {
        id: "gpt-4",
        name: "GPT-4",
        contextWindow: 128000,
        maxOutputTokens: 4096,
      },
    ],

    async createStream(request, signal) {
      // 1. 转换请求格式
      const openaiRequest = {
        model: request.model,
        messages: request.messages.map(msg => ({
          role: msg.role,
          content: msg.content,
        })),
        temperature: request.temperature ?? 0.7,
        max_tokens: request.maxTokens ?? 4096,
        stream: true,
      };

      // 2. 发送请求
      const response = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`,
        },
        body: JSON.stringify(openaiRequest),
        signal,
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      // 3. 解析流式响应
      return parseOpenAIStream(response.body);
    },
  });
}

// 解析 OpenAI 流式响应
async function* parseOpenAIStream(
  body: ReadableStream<Uint8Array>
): AsyncIterable<StreamChunk> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim() || line.startsWith(":")) continue;
        if (line === "data: [DONE]") {
          yield { type: "done" };
          return;
        }

        if (line.startsWith("data: ")) {
          const data = JSON.parse(line.slice(6));
          const delta = data.choices[0]?.delta;

          if (delta?.content) {
            yield { type: "text", text: delta.content };
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
```

### 3.3 实现 Ollama Provider

**本地模型 Provider：**

```typescript
export default function (pi: ExtensionAPI) {
  pi.registerProvider({
    id: "ollama",
    name: "Ollama Local",

    models: [
      {
        id: "llama3",
        name: "Llama 3",
        contextWindow: 8192,
        maxOutputTokens: 2048,
      },
      {
        id: "mistral",
        name: "Mistral",
        contextWindow: 8192,
        maxOutputTokens: 2048,
      },
    ],

    async createStream(request, signal) {
      const response = await fetch("http://localhost:11434/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: request.model,
          messages: request.messages,
          stream: true,
        }),
        signal,
      });

      return parseOllamaStream(response.body);
    },
  });
}

async function* parseOllamaStream(
  body: ReadableStream<Uint8Array>
): AsyncIterable<StreamChunk> {
  const reader = body.getReader();
  const decoder = new TextDecoder();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const text = decoder.decode(value);
      const lines = text.split("\n").filter(l => l.trim());

      for (const line of lines) {
        const data = JSON.parse(line);

        if (data.message?.content) {
          yield { type: "text", text: data.message.content };
        }

        if (data.done) {
          yield { type: "done" };
          return;
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
```

---

## 4. OAuth 集成

### 4.1 OAuthConfig 接口

**OAuth 配置接口：**

```typescript
export interface OAuthConfig {
  // OAuth 端点
  authorizationUrl: string;      // 授权 URL
  tokenUrl: string;              // Token URL

  // 客户端信息
  clientId: string;
  clientSecret?: string;

  // 作用域
  scopes: string[];

  // 可选：设备流
  deviceCodeUrl?: string;

  // 可选：刷新 token
  refreshTokenUrl?: string;
}
```

### 4.2 实现 OAuth Provider

**完整的 OAuth Provider 示例：**

```typescript
export default function (pi: ExtensionAPI) {
  pi.registerProvider({
    id: "oauth-provider",
    name: "OAuth Provider",

    models: [
      {
        id: "model-1",
        name: "Model 1",
        contextWindow: 128000,
      },
    ],

    // OAuth 配置
    oauth: {
      authorizationUrl: "https://auth.example.com/oauth/authorize",
      tokenUrl: "https://auth.example.com/oauth/token",
      clientId: "your-client-id",
      clientSecret: "your-client-secret",
      scopes: ["read", "write"],
    },

    async createStream(request, signal) {
      // 获取 access token（由 pi-mono 自动处理）
      const accessToken = await getAccessToken();

      const response = await fetch("https://api.example.com/chat", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(request),
        signal,
      });

      return parseStream(response.body);
    },
  });
}
```

### 4.3 设备流 OAuth

**适用于 CLI 工具的设备流：**

```typescript
pi.registerProvider({
  id: "device-flow-provider",
  name: "Device Flow Provider",

  models: [...],

  oauth: {
    authorizationUrl: "https://auth.example.com/device/authorize",
    tokenUrl: "https://auth.example.com/oauth/token",
    deviceCodeUrl: "https://auth.example.com/device/code",
    clientId: "your-client-id",
    scopes: ["read", "write"],
  },

  async createStream(request, signal) {
    // pi-mono 会自动处理设备流认证
    // 1. 请求 device code
    // 2. 显示用户码和验证 URL
    // 3. 轮询 token 端点
    // 4. 获取 access token

    const accessToken = await getAccessToken();
    // 使用 access token 调用 API
  },
});
```

---

## 5. 工具调用支持

### 5.1 转换工具定义

**将 pi-mono 的工具定义转换为 API 格式：**

```typescript
async createStream(request, signal) {
  // 转换工具定义
  const tools = request.tools?.map(tool => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: convertTypeBoxToJSONSchema(tool.parameters),
    },
  }));

  const apiRequest = {
    model: request.model,
    messages: request.messages,
    tools,
    tool_choice: "auto",
  };

  const response = await fetch(apiUrl, {
    method: "POST",
    headers: { ... },
    body: JSON.stringify(apiRequest),
    signal,
  });

  return parseStreamWithTools(response.body);
}
```

### 5.2 处理工具调用

**解析和处理工具调用：**

```typescript
async function* parseStreamWithTools(
  body: ReadableStream<Uint8Array>
): AsyncIterable<StreamChunk> {
  const reader = body.getReader();
  const decoder = new TextDecoder();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const text = decoder.decode(value);
      const data = JSON.parse(text);

      // 文本内容
      if (data.delta?.content) {
        yield { type: "text", text: data.delta.content };
      }

      // 工具调用
      if (data.delta?.tool_calls) {
        for (const toolCall of data.delta.tool_calls) {
          yield {
            type: "tool_call",
            toolCallId: toolCall.id,
            toolName: toolCall.function.name,
            input: JSON.parse(toolCall.function.arguments),
          };
        }
      }

      // 完成
      if (data.finish_reason === "stop") {
        yield { type: "done" };
        return;
      }
    }
  } finally {
    reader.releaseLock();
  }
}
```

---

## 6. 思考模式支持

### 6.1 配置思考级别

**定义支持思考模式的模型：**

```typescript
pi.registerProvider({
  id: "thinking-provider",
  name: "Thinking Provider",

  models: [
    {
      id: "thinking-model",
      name: "Thinking Model",
      contextWindow: 200000,
      maxOutputTokens: 8192,
      supportsThinking: true,
      thinkingLevels: ["low", "medium", "high", "extended"],
    },
  ],

  async createStream(request, signal) {
    // 根据思考级别调整参数
    const thinkingBudget = getThinkingBudget(request.thinkingLevel);

    const apiRequest = {
      model: request.model,
      messages: request.messages,
      thinking_budget: thinkingBudget,
    };

    const response = await fetch(apiUrl, {
      method: "POST",
      headers: { ... },
      body: JSON.stringify(apiRequest),
      signal,
    });

    return parseThinkingStream(response.body);
  },
});

function getThinkingBudget(level?: string): number {
  switch (level) {
    case "low": return 1000;
    case "medium": return 5000;
    case "high": return 10000;
    case "extended": return 20000;
    default: return 5000;
  }
}
```

### 6.2 解析思考内容

**区分思考内容和输出内容：**

```typescript
async function* parseThinkingStream(
  body: ReadableStream<Uint8Array>
): AsyncIterable<StreamChunk> {
  const reader = body.getReader();
  const decoder = new TextDecoder();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const text = decoder.decode(value);
      const data = JSON.parse(text);

      // 思考内容
      if (data.type === "thinking") {
        yield { type: "thinking", text: data.content };
      }

      // 输出内容
      if (data.type === "content") {
        yield { type: "text", text: data.content };
      }

      if (data.done) {
        yield { type: "done" };
        return;
      }
    }
  } finally {
    reader.releaseLock();
  }
}
```

---

## 7. 错误处理

### 7.1 API 错误处理

**处理各种 API 错误：**

```typescript
async createStream(request, signal) {
  try {
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: { ... },
      body: JSON.stringify(request),
      signal,
    });

    // 检查 HTTP 状态
    if (!response.ok) {
      const error = await response.json();
      throw new Error(`API error: ${error.message}`);
    }

    return parseStream(response.body);
  } catch (error) {
    // 取消错误
    if (error.name === "AbortError") {
      return (async function* () {
        yield { type: "error", error: "Request cancelled" };
      })();
    }

    // 网络错误
    if (error instanceof TypeError) {
      return (async function* () {
        yield { type: "error", error: "Network error" };
      })();
    }

    // 其他错误
    return (async function* () {
      yield { type: "error", error: error.message };
    })();
  }
}
```

### 7.2 流式响应错误处理

**在流式响应中处理错误：**

```typescript
async function* parseStream(
  body: ReadableStream<Uint8Array>
): AsyncIterable<StreamChunk> {
  const reader = body.getReader();
  const decoder = new TextDecoder();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      try {
        const text = decoder.decode(value);
        const data = JSON.parse(text);

        // 检查错误
        if (data.error) {
          yield { type: "error", error: data.error.message };
          return;
        }

        // 处理正常数据
        if (data.content) {
          yield { type: "text", text: data.content };
        }
      } catch (parseError) {
        // JSON 解析错误
        console.error("Parse error:", parseError);
        yield { type: "error", error: "Failed to parse response" };
        return;
      }
    }

    yield { type: "done" };
  } catch (error) {
    yield { type: "error", error: error.message };
  } finally {
    reader.releaseLock();
  }
}
```

---

## 8. 实际应用示例

### 8.1 企业内部 API Provider

```typescript
export default function (pi: ExtensionAPI) {
  pi.registerProvider({
    id: "company-llm",
    name: "Company Internal LLM",

    models: [
      {
        id: "company-model-v1",
        name: "Company Model v1",
        contextWindow: 100000,
        maxOutputTokens: 4096,
      },
    ],

    async createStream(request, signal) {
      // 使用公司内部认证
      const token = await getCompanyToken();

      const response = await fetch("https://internal-api.company.com/chat", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`,
          "X-Company-ID": process.env.COMPANY_ID,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: request.model,
          messages: request.messages,
          max_tokens: request.maxTokens,
        }),
        signal,
      });

      return parseCompanyStream(response.body);
    },
  });
}
```

### 8.2 API 代理和缓存 Provider

```typescript
export default function (pi: ExtensionAPI) {
  const cache = new Map<string, string>();

  pi.registerProvider({
    id: "cached-provider",
    name: "Cached Provider",

    models: [...],

    async createStream(request, signal) {
      // 生成缓存键
      const cacheKey = JSON.stringify({
        model: request.model,
        messages: request.messages,
      });

      // 检查缓存
      if (cache.has(cacheKey)) {
        const cached = cache.get(cacheKey);
        return (async function* () {
          yield { type: "text", text: cached };
          yield { type: "done" };
        })();
      }

      // 调用实际 API
      const response = await fetch(apiUrl, { ... });
      const stream = parseStream(response.body);

      // 缓存响应
      let fullResponse = "";
      return (async function* () {
        for await (const chunk of stream) {
          if (chunk.type === "text") {
            fullResponse += chunk.text;
          }
          yield chunk;
        }
        cache.set(cacheKey, fullResponse);
      })();
    },
  });
}
```

### 8.3 多模型聚合 Provider

```typescript
export default function (pi: ExtensionAPI) {
  pi.registerProvider({
    id: "aggregated-provider",
    name: "Aggregated Provider",

    models: [
      { id: "fast-model", name: "Fast Model", contextWindow: 8192 },
      { id: "smart-model", name: "Smart Model", contextWindow: 128000 },
      { id: "cheap-model", name: "Cheap Model", contextWindow: 4096 },
    ],

    async createStream(request, signal) {
      // 根据模型 ID 路由到不同的后端
      switch (request.model) {
        case "fast-model":
          return callFastAPI(request, signal);
        case "smart-model":
          return callSmartAPI(request, signal);
        case "cheap-model":
          return callCheapAPI(request, signal);
        default:
          throw new Error(`Unknown model: ${request.model}`);
      }
    },
  });
}
```

---

## 9. 最佳实践

### 9.1 错误处理

```typescript
// ✅ 好：完整的错误处理
async createStream(request, signal) {
  try {
    const response = await fetch(apiUrl, { ... });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(`API error: ${error.message}`);
    }

    return parseStream(response.body);
  } catch (error) {
    console.error("Provider error:", error);
    return (async function* () {
      yield { type: "error", error: error.message };
    })();
  }
}

// ❌ 不好：没有错误处理
async createStream(request, signal) {
  const response = await fetch(apiUrl, { ... });
  return parseStream(response.body);
}
```

### 9.2 取消支持

```typescript
// ✅ 好：支持取消
async createStream(request, signal) {
  const response = await fetch(apiUrl, {
    method: "POST",
    body: JSON.stringify(request),
    signal, // 传递 signal
  });

  return parseStream(response.body, signal);
}

async function* parseStream(body, signal) {
  const reader = body.getReader();

  try {
    while (true) {
      // 检查取消
      if (signal.aborted) {
        break;
      }

      const { done, value } = await reader.read();
      if (done) break;

      // 处理数据
    }
  } finally {
    reader.releaseLock();
  }
}
```

### 9.3 性能优化

```typescript
// ✅ 好：流式解析，逐块处理
async function* parseStream(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    // 逐行处理
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      yield processLine(line);
    }
  }
}

// ❌ 不好：等待全部数据
async function* parseStream(body) {
  const reader = body.getReader();
  let fullText = "";

  // 等待全部数据
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    fullText += new TextDecoder().decode(value);
  }

  // 一次性处理
  yield { type: "text", text: fullText };
}
```

---

## 10. 总结

### 核心要点

1. **Provider 抽象**：统一的 LLM API 抽象层
2. **registerProvider**：注册自定义 Provider
3. **流式响应**：实现 AsyncIterable<StreamChunk>
4. **OAuth 集成**：支持 OAuth 认证流程
5. **工具调用**：转换和处理工具调用
6. **思考模式**：支持思考级别配置
7. **错误处理**：完整的错误处理和取消支持

### 下一步

- **核心概念 06**：扩展加载与热重载
- **核心概念 07**：扩展开发模式
- **实战代码**：Provider 定制扩展

---

**版本**: v1.0
**最后更新**: 2026-02-20
**适用于**: pi-mono 2025-2026 版本
