# 实战代码1：基础 Provider 适配器

> 从零实现一个简单的 Provider Adapter

---

## 目标

实现一个完整可运行的 OpenAI Provider Adapter，理解 Adapter Pattern 的核心实现。

---

## 完整代码

```typescript
/**
 * 基础 Provider 适配器实现
 * 演示：如何实现一个简单的 OpenAI Adapter
 */

import OpenAI from 'openai';

// ===== 1. 定义统一接口 =====

/**
 * 统一的上下文格式
 */
interface Context {
  systemPrompt?: string;
  messages: Message[];
  temperature?: number;
  maxTokens?: number;
}

/**
 * 统一的消息格式
 */
interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

/**
 * Provider Adapter 接口
 */
interface ProviderAdapter {
  complete(context: Context): Promise<Message>;
  stream(context: Context): AsyncGenerator<StreamEvent>;
}

/**
 * 流式事件
 */
type StreamEvent =
  | { type: 'start'; model: string }
  | { type: 'delta'; content: string }
  | { type: 'end'; usage: { totalTokens: number } };

// ===== 2. 实现 OpenAI Adapter =====

class OpenAIAdapter implements ProviderAdapter {
  private client: OpenAI;
  private model: string;

  constructor(apiKey: string, model: string = 'gpt-4o-mini') {
    this.client = new OpenAI({ apiKey });
    this.model = model;
  }

  /**
   * 同步调用
   */
  async complete(context: Context): Promise<Message> {
    // 1. 转换为 OpenAI 格式
    const messages: OpenAI.ChatCompletionMessageParam[] = [];

    if (context.systemPrompt) {
      messages.push({
        role: 'system',
        content: context.systemPrompt
      });
    }

    for (const msg of context.messages) {
      messages.push({
        role: msg.role,
        content: msg.content
      });
    }

    // 2. 调用 OpenAI API
    const response = await this.client.chat.completions.create({
      model: this.model,
      messages,
      temperature: context.temperature,
      max_tokens: context.maxTokens
    });

    // 3. 转换为统一格式
    return {
      role: 'assistant',
      content: response.choices[0].message.content || ''
    };
  }

  /**
   * 流式调用
   */
  async *stream(context: Context): AsyncGenerator<StreamEvent> {
    // 1. 转换为 OpenAI 格式
    const messages: OpenAI.ChatCompletionMessageParam[] = [];

    if (context.systemPrompt) {
      messages.push({
        role: 'system',
        content: context.systemPrompt
      });
    }

    for (const msg of context.messages) {
      messages.push({
        role: msg.role,
        content: msg.content
      });
    }

    // 2. 调用 OpenAI Streaming API
    const stream = await this.client.chat.completions.create({
      model: this.model,
      messages,
      temperature: context.temperature,
      max_tokens: context.maxTokens,
      stream: true
    });

    // 3. 发送 start 事件
    yield { type: 'start', model: this.model };

    // 4. 转换流式响应
    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content;
      if (content) {
        yield { type: 'delta', content };
      }
    }

    // 5. 发送 end 事件
    yield { type: 'end', usage: { totalTokens: 0 } };
  }
}

// ===== 3. 使用示例 =====

async function main() {
  // 创建 Adapter
  const adapter = new OpenAIAdapter(
    process.env.OPENAI_API_KEY!,
    'gpt-4o-mini'
  );

  // 构建上下文
  const context: Context = {
    systemPrompt: 'You are a helpful assistant.',
    messages: [
      { role: 'user', content: 'Explain TypeScript in one sentence.' }
    ],
    temperature: 0.7
  };

  console.log('=== 同步调用 ===');
  const response = await adapter.complete(context);
  console.log(response.content);

  console.log('\n=== 流式调用 ===');
  for await (const event of adapter.stream(context)) {
    if (event.type === 'start') {
      console.log(`Stream started (model: ${event.model})`);
    } else if (event.type === 'delta') {
      process.stdout.write(event.content);
    } else if (event.type === 'end') {
      console.log(`\nStream ended`);
    }
  }
}

// 运行
main().catch(console.error);
```

---

## 运行步骤

```bash
# 1. 安装依赖
npm install openai

# 2. 设置 API Key
export OPENAI_API_KEY=sk-...

# 3. 运行
npx tsx basic-adapter.ts
```

---

## 预期输出

```
=== 同步调用 ===
TypeScript is a statically typed superset of JavaScript that compiles to plain JavaScript.

=== 流式调用 ===
Stream started (model: gpt-4o-mini)
TypeScript is a statically typed superset of JavaScript that compiles to plain JavaScript.
Stream ended
```

---

## 关键点解析

### 1. 格式转换

```typescript
// Pi AI 格式 → OpenAI 格式
const messages: OpenAI.ChatCompletionMessageParam[] = [];

if (context.systemPrompt) {
  messages.push({
    role: 'system',
    content: context.systemPrompt
  });
}

for (const msg of context.messages) {
  messages.push({
    role: msg.role,
    content: msg.content
  });
}
```

### 2. 流式处理

```typescript
// 使用 AsyncGenerator
async *stream(context: Context): AsyncGenerator<StreamEvent> {
  // 发送事件
  yield { type: 'start', model: this.model };

  // 处理流
  for await (const chunk of stream) {
    yield { type: 'delta', content: chunk.content };
  }

  yield { type: 'end', usage: { totalTokens: 0 } };
}
```

### 3. 错误处理

```typescript
try {
  const response = await this.client.chat.completions.create({...});
} catch (error) {
  if (error instanceof OpenAI.APIError) {
    console.error(`OpenAI API Error: ${error.message}`);
  }
  throw error;
}
```

---

## 扩展练习

### 练习1：添加 Anthropic Adapter

实现一个 Anthropic Adapter，使用相同的接口。

### 练习2：添加重试机制

在 API 调用失败时自动重试。

### 练习3：添加缓存

缓存相同请求的响应。

---

**版本：** v1.0
**最后更新：** 2026-02-19
