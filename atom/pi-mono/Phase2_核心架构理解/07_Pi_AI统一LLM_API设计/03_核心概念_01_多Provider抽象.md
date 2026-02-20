# 核心概念1：多 Provider 抽象

> 理解如何通过 Provider Adapter 模式统一 25+ LLM Provider 的接口

---

## 概念定义

**多 Provider 抽象**是指通过 Adapter Pattern（适配器模式）将不同 LLM Provider 的 API 统一为一套标准接口，使开发者可以用相同的代码调用不同的模型。

**核心价值：**
- **开发效率**：写一次代码，支持所有 Provider
- **灵活切换**：一行配置切换 Provider，无需修改业务代码
- **降低耦合**：业务逻辑与 Provider 实现解耦
- **易于扩展**：添加新 Provider 无需修改现有代码

---

## 第一性原理

### 问题的本质

**核心问题：** 不同 LLM Provider 的 API 格式完全不同，如何统一？

```typescript
// OpenAI API
const openai = new OpenAI();
const response = await openai.chat.completions.create({
  model: 'gpt-4o-mini',
  messages: [{ role: 'user', content: 'Hello' }]
});

// Anthropic API
const anthropic = new Anthropic();
const response = await anthropic.messages.create({
  model: 'claude-opus-4',
  messages: [{ role: 'user', content: [{ type: 'text', text: 'Hello' }] }]
});

// Google API
const genai = new GoogleGenerativeAI();
const model = genai.getGenerativeModel({ model: 'gemini-2.0-flash' });
const response = await model.generateContent('Hello');
```

**问题分析：**
1. **方法名不同**：`create()` vs `generateContent()`
2. **参数格式不同**：消息结构、内容格式
3. **响应格式不同**：返回值结构
4. **认证方式不同**：API Key、OAuth、订阅

### 设计模式：Adapter Pattern

**Adapter Pattern 的核心思想：**
> 将一个类的接口转换为客户端期望的另一个接口，使原本不兼容的类可以一起工作。

```
Client (业务代码)
  ↓ 调用统一接口
Target Interface (统一接口)
  ↓ 实现
Adapter (适配器)
  ↓ 调用
Adaptee (被适配的类，如 OpenAI SDK)
```

**应用到 LLM API：**

```
Agent Core (业务代码)
  ↓ 调用 complete()
ProviderAdapter Interface (统一接口)
  ↓ 实现
OpenAIAdapter / AnthropicAdapter / GoogleAdapter
  ↓ 调用
OpenAI SDK / Anthropic SDK / Google SDK
```

---

## 核心实现

### 1. 定义统一接口

```typescript
/**
 * Provider Adapter 统一接口
 * 所有 Provider 必须实现这个接口
 */
interface ProviderAdapter {
  /**
   * 同步调用（等待完整响应）
   * @param context 对话上下文
   * @returns 完整的响应消息
   */
  complete(context: Context): Promise<Message>;

  /**
   * 流式调用（实时返回增量内容）
   * @param context 对话上下文
   * @returns 异步生成器，产生流式事件
   */
  stream(context: Context): AsyncGenerator<StreamEvent>;

  /**
   * Provider 名称
   */
  readonly name: string;

  /**
   * 支持的功能
   */
  readonly capabilities: {
    streaming: boolean;
    toolCalling: boolean;
    vision: boolean;
  };
}

/**
 * 对话上下文（统一格式）
 */
interface Context {
  systemPrompt?: string;
  messages: Message[];
  tools?: Tool[];
  temperature?: number;
  maxTokens?: number;
}

/**
 * 消息（统一格式）
 */
interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string | ContentBlock[];
}

/**
 * 流式事件（统一格式）
 */
type StreamEvent =
  | { type: 'start'; model: string }
  | { type: 'delta'; delta: { content: string } }
  | { type: 'end'; usage: TokenUsage };
```

### 2. 实现 OpenAI Adapter

```typescript
import OpenAI from 'openai';

/**
 * OpenAI Provider Adapter
 * 将 OpenAI API 适配为统一接口
 */
class OpenAIAdapter implements ProviderAdapter {
  readonly name = 'openai';
  readonly capabilities = {
    streaming: true,
    toolCalling: true,
    vision: true
  };

  private client: OpenAI;

  constructor(apiKey: string) {
    this.client = new OpenAI({ apiKey });
  }

  /**
   * 同步调用
   */
  async complete(context: Context): Promise<Message> {
    // 1. 转换为 OpenAI 格式
    const openaiRequest = this.toOpenAIFormat(context);

    // 2. 调用 OpenAI API
    const response = await this.client.chat.completions.create(openaiRequest);

    // 3. 转换为统一格式
    return this.toPiFormat(response);
  }

  /**
   * 流式调用
   */
  async *stream(context: Context): AsyncGenerator<StreamEvent> {
    // 1. 转换为 OpenAI 格式
    const openaiRequest = this.toOpenAIFormat(context);
    openaiRequest.stream = true;

    // 2. 调用 OpenAI Streaming API
    const stream = await this.client.chat.completions.create(openaiRequest);

    // 3. 发送 start 事件
    yield { type: 'start', model: openaiRequest.model };

    // 4. 转换流式响应
    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta;
      if (delta?.content) {
        yield {
          type: 'delta',
          delta: { content: delta.content }
        };
      }
    }

    // 5. 发送 end 事件
    yield {
      type: 'end',
      usage: {
        inputTokens: 0,  // OpenAI 流式响应不返回 usage
        outputTokens: 0,
        totalTokens: 0
      }
    };
  }

  /**
   * 转换为 OpenAI 格式
   */
  private toOpenAIFormat(context: Context): OpenAI.ChatCompletionCreateParams {
    return {
      model: 'gpt-4o-mini',
      messages: context.messages.map(msg => ({
        role: msg.role,
        content: typeof msg.content === 'string'
          ? msg.content
          : msg.content.map(block => {
              if (block.type === 'text') {
                return { type: 'text', text: block.text };
              } else if (block.type === 'image') {
                return {
                  type: 'image_url',
                  image_url: { url: block.source.url }
                };
              }
              return block;
            })
      })),
      temperature: context.temperature,
      max_tokens: context.maxTokens,
      tools: context.tools?.map(tool => ({
        type: 'function',
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters
        }
      }))
    };
  }

  /**
   * 转换为统一格式
   */
  private toPiFormat(response: OpenAI.ChatCompletion): Message {
    const choice = response.choices[0];
    return {
      role: 'assistant',
      content: choice.message.content || '',
      toolCalls: choice.message.tool_calls?.map(tc => ({
        id: tc.id,
        name: tc.function.name,
        input: JSON.parse(tc.function.arguments)
      }))
    };
  }
}
```

### 3. 实现 Anthropic Adapter

```typescript
import Anthropic from '@anthropic-ai/sdk';

/**
 * Anthropic Provider Adapter
 * 将 Anthropic API 适配为统一接口
 */
class AnthropicAdapter implements ProviderAdapter {
  readonly name = 'anthropic';
  readonly capabilities = {
    streaming: true,
    toolCalling: true,
    vision: true
  };

  private client: Anthropic;

  constructor(apiKey: string) {
    this.client = new Anthropic({ apiKey });
  }

  async complete(context: Context): Promise<Message> {
    const anthropicRequest = this.toAnthropicFormat(context);
    const response = await this.client.messages.create(anthropicRequest);
    return this.toPiFormat(response);
  }

  async *stream(context: Context): AsyncGenerator<StreamEvent> {
    const anthropicRequest = this.toAnthropicFormat(context);
    anthropicRequest.stream = true;

    const stream = await this.client.messages.create(anthropicRequest);

    yield { type: 'start', model: anthropicRequest.model };

    for await (const event of stream) {
      if (event.type === 'content_block_delta') {
        if (event.delta.type === 'text_delta') {
          yield {
            type: 'delta',
            delta: { content: event.delta.text }
          };
        }
      } else if (event.type === 'message_stop') {
        yield {
          type: 'end',
          usage: {
            inputTokens: event.message.usage.input_tokens,
            outputTokens: event.message.usage.output_tokens,
            totalTokens: event.message.usage.input_tokens + event.message.usage.output_tokens
          }
        };
      }
    }
  }

  private toAnthropicFormat(context: Context): Anthropic.MessageCreateParams {
    return {
      model: 'claude-opus-4',
      system: context.systemPrompt,
      messages: context.messages.map(msg => ({
        role: msg.role === 'user' ? 'user' : 'assistant',
        content: typeof msg.content === 'string'
          ? msg.content
          : msg.content.map(block => {
              if (block.type === 'text') {
                return { type: 'text', text: block.text };
              } else if (block.type === 'image') {
                return {
                  type: 'image',
                  source: {
                    type: 'url',
                    url: block.source.url
                  }
                };
              }
              return block;
            })
      })),
      temperature: context.temperature,
      max_tokens: context.maxTokens || 4096,
      tools: context.tools?.map(tool => ({
        name: tool.name,
        description: tool.description,
        input_schema: tool.parameters
      }))
    };
  }

  private toPiFormat(response: Anthropic.Message): Message {
    const textContent = response.content
      .filter(block => block.type === 'text')
      .map(block => block.text)
      .join('');

    const toolCalls = response.content
      .filter(block => block.type === 'tool_use')
      .map(block => ({
        id: block.id,
        name: block.name,
        input: block.input
      }));

    return {
      role: 'assistant',
      content: textContent,
      toolCalls: toolCalls.length > 0 ? toolCalls : undefined
    };
  }
}
```

### 4. Adapter 注册与管理

```typescript
/**
 * Adapter Registry
 * 管理所有 Provider Adapter
 */
class AdapterRegistry {
  private adapters = new Map<string, ProviderAdapter>();

  /**
   * 注册 Adapter
   */
  register(adapter: ProviderAdapter): void {
    this.adapters.set(adapter.name, adapter);
  }

  /**
   * 获取 Adapter
   */
  get(provider: string): ProviderAdapter {
    const adapter = this.adapters.get(provider);
    if (!adapter) {
      throw new Error(`Provider "${provider}" not found`);
    }
    return adapter;
  }

  /**
   * 列出所有 Provider
   */
  list(): string[] {
    return Array.from(this.adapters.keys());
  }
}

// 全局注册表
const registry = new AdapterRegistry();

// 注册 Adapter
registry.register(new OpenAIAdapter(process.env.OPENAI_API_KEY!));
registry.register(new AnthropicAdapter(process.env.ANTHROPIC_API_KEY!));

// 使用
const adapter = registry.get('openai');
const response = await adapter.complete(context);
```

---

## 在 AI Agent 中的应用

### 场景1：多模型策略

```typescript
/**
 * 根据任务复杂度选择模型
 */
async function smartComplete(task: string, context: Context): Promise<Message> {
  // 简单任务：使用快速便宜的模型
  if (isSimpleTask(task)) {
    const adapter = registry.get('openai');
    return adapter.complete(context);
  }

  // 复杂任务：使用强大的模型
  const adapter = registry.get('anthropic');
  return adapter.complete(context);
}

function isSimpleTask(task: string): boolean {
  // 简单启发式：短任务 = 简单任务
  return task.length < 100;
}
```

### 场景2：容错降级

```typescript
/**
 * 主 Provider 故障时自动切换备用
 */
async function resilientComplete(
  providers: string[],
  context: Context
): Promise<Message> {
  for (const provider of providers) {
    try {
      const adapter = registry.get(provider);
      return await adapter.complete(context);
    } catch (error) {
      console.error(`Provider ${provider} failed:`, error);
      // 继续尝试下一个 Provider
    }
  }

  throw new Error('All providers failed');
}

// 使用
const response = await resilientComplete(
  ['openai', 'anthropic', 'google'],
  context
);
```

### 场景3：A/B 测试

```typescript
/**
 * 对比不同模型的效果
 */
async function abTest(context: Context): Promise<void> {
  const providers = ['openai', 'anthropic'];

  const results = await Promise.all(
    providers.map(async provider => {
      const adapter = registry.get(provider);
      const start = Date.now();
      const response = await adapter.complete(context);
      const latency = Date.now() - start;

      return {
        provider,
        response: response.content,
        latency,
        tokens: response.usage?.totalTokens || 0
      };
    })
  );

  // 对比结果
  console.table(results);
}
```

---

## 设计权衡

### 优点

1. **统一接口**
   - 业务代码与 Provider 解耦
   - 切换 Provider 无需修改代码

2. **易于扩展**
   - 添加新 Provider 只需实现 Adapter
   - 无需修改现有代码

3. **灵活组合**
   - 可以实现多模型策略
   - 可以实现容错降级

### 缺点

1. **抽象开销**
   - 格式转换有轻微性能开销（< 1ms）
   - 但远小于网络延迟（> 50ms）

2. **功能限制**
   - 统一接口只包含通用功能
   - Provider 特有功能需要通过 options 传递

3. **维护成本**
   - 每个 Provider 需要单独维护 Adapter
   - Provider API 变更需要更新 Adapter

---

## 实际案例（2025-2026）

### 案例1：GitHub Copilot Workspace

**背景：** GitHub Copilot Workspace 支持多个 LLM Provider（GPT-4、Claude、Gemini）

**实现：**
- 使用 Adapter Pattern 统一接口
- 根据任务类型自动选择模型
- 代码生成用 GPT-4，代码审查用 Claude

**效果：**
- 开发效率提升 40%
- 成本降低 30%（智能选择模型）

**来源：** [GitHub Blog - Copilot Workspace](https://github.blog/2025-11-15-copilot-workspace-multi-model/) (2025-11-15)

---

### 案例2：Vercel AI SDK

**背景：** Vercel AI SDK 是 TypeScript 的多 Provider 库

**实现：**
```typescript
import { openai } from '@ai-sdk/openai';
import { anthropic } from '@ai-sdk/anthropic';
import { generateText } from 'ai';

// 统一接口
const result1 = await generateText({
  model: openai('gpt-4o'),
  prompt: 'Hello'
});

const result2 = await generateText({
  model: anthropic('claude-opus-4'),
  prompt: 'Hello'
});
```

**特点：**
- 统一的 `generateText()` 接口
- 支持 10+ Provider
- 类型安全（TypeScript）

**来源：** [Vercel AI SDK](https://sdk.vercel.ai/) (2026-01-20)

---

### 案例3：LangChain.js

**背景：** LangChain.js 支持 50+ LLM Provider

**实现：**
```typescript
import { ChatOpenAI } from '@langchain/openai';
import { ChatAnthropic } from '@langchain/anthropic';

// 统一接口
const openai = new ChatOpenAI({ model: 'gpt-4o' });
const anthropic = new ChatAnthropic({ model: 'claude-opus-4' });

// 相同的调用方式
const result1 = await openai.invoke('Hello');
const result2 = await anthropic.invoke('Hello');
```

**特点：**
- 统一的 `invoke()` 接口
- 支持链式组合
- 支持流式响应

**来源：** [LangChain.js Docs](https://js.langchain.com/) (2026-02-10)

---

## 学习检查清单

完成本概念学习后，你应该能够：

- [ ] 理解 Adapter Pattern 的核心思想
- [ ] 理解 ProviderAdapter 接口的设计
- [ ] 能够实现一个简单的 Provider Adapter
- [ ] 理解格式转换的实现（to/from）
- [ ] 理解 Adapter Registry 的作用
- [ ] 能够使用多 Provider 策略
- [ ] 能够实现容错降级
- [ ] 理解设计权衡（优点/缺点）

---

## 参考资源

### 官方文档
- [pi-ai README](https://github.com/badlogic/pi-mono/blob/main/packages/pi-ai/README.md) - 2026-02-18
- [pi-ai Provider 实现](https://github.com/badlogic/pi-mono/tree/main/packages/pi-ai/src/providers) - 源码

### 设计模式
- [Adapter Pattern](https://refactoring.guru/design-patterns/adapter) - 适配器模式详解
- [Strategy Pattern](https://refactoring.guru/design-patterns/strategy) - 策略模式详解

### 相关项目
- [Vercel AI SDK](https://sdk.vercel.ai/) - TypeScript 多 Provider 库
- [LangChain.js](https://js.langchain.com/) - JavaScript LLM 框架
- [LiteLLM](https://github.com/BerriAI/litellm) - Python 多 Provider 库

### 行业实践
- [GitHub Blog - Copilot Workspace](https://github.blog/2025-11-15-copilot-workspace-multi-model/) - 2025-11-15
- [Unified LLM Spec](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md) - 2026-02-09

---

**版本：** v1.0
**最后更新：** 2026-02-19
