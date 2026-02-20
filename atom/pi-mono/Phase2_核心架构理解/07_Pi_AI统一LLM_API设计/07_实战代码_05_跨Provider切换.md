# 实战代码5：跨 Provider 切换

> 实现对话中途切换 Provider

---

## 目标

在对话过程中无缝切换不同的 LLM Provider。

---

## 完整代码

```typescript
/**
 * 跨 Provider 切换
 * 演示：对话中途切换模型
 */

// ===== 1. 对话历史管理 =====

interface Message {
  role: 'user' | 'assistant';
  content: string;
  metadata?: {
    provider?: string;
    model?: string;
  };
}

class ConversationHistory {
  private messages: Message[] = [];

  add(message: Message): void {
    this.messages.push(message);
  }

  getAll(): Message[] {
    return [...this.messages];
  }

  toContext(): { messages: Message[] } {
    return { messages: this.getAll() };
  }
}

// ===== 2. Provider 切换器 =====

class ProviderSwitcher {
  private history = new ConversationHistory();

  async chat(
    provider: string,
    model: string,
    userMessage: string
  ): Promise<string> {
    // 添加用户消息
    this.history.add({
      role: 'user',
      content: userMessage
    });

    // 调用 Provider（模拟）
    const response = await this.callProvider(provider, model);

    // 添加助手消息
    this.history.add({
      role: 'assistant',
      content: response,
      metadata: { provider, model }
    });

    return response;
  }

  private async callProvider(provider: string, model: string): Promise<string> {
    // 模拟 API 调用
    await new Promise(resolve => setTimeout(resolve, 100));
    return `Response from ${provider}/${model}`;
  }

  getHistory(): Message[] {
    return this.history.getAll();
  }
}

// ===== 3. 使用示例 =====

async function main() {
  const switcher = new ProviderSwitcher();

  console.log('=== 对话示例：切换 Provider ===\n');

  // 1. 使用 OpenAI 开始对话
  console.log('User: Explain TypeScript');
  const response1 = await switcher.chat('openai', 'gpt-4o-mini', 'Explain TypeScript');
  console.log(`OpenAI: ${response1}\n`);

  // 2. 切换到 Anthropic 继续对话
  console.log('User: Give me a code example');
  const response2 = await switcher.chat('anthropic', 'claude-opus-4', 'Give me a code example');
  console.log(`Anthropic: ${response2}\n`);

  // 3. 切换到 Google 继续对话
  console.log('User: Explain the benefits');
  const response3 = await switcher.chat('google', 'gemini-2.0-flash', 'Explain the benefits');
  console.log(`Google: ${response3}\n`);

  // 4. 显示完整历史
  console.log('=== 完整对话历史 ===\n');
  const history = switcher.getHistory();
  history.forEach((msg, i) => {
    const provider = msg.metadata?.provider || 'user';
    console.log(`[${i + 1}] ${msg.role} (${provider}): ${msg.content}`);
  });
}

main().catch(console.error);
```

---

## 运行输出

```
=== 对话示例：切换 Provider ===

User: Explain TypeScript
OpenAI: Response from openai/gpt-4o-mini

User: Give me a code example
Anthropic: Response from anthropic/claude-opus-4

User: Explain the benefits
Google: Response from google/gemini-2.0-flash

=== 完整对话历史 ===

[1] user (user): Explain TypeScript
[2] assistant (openai): Response from openai/gpt-4o-mini
[3] user (user): Give me a code example
[4] assistant (anthropic): Response from anthropic/claude-opus-4
[5] user (user): Explain the benefits
[6] assistant (google): Response from google/gemini-2.0-flash
```

---

## 关键点

### 1. 对话历史保留
所有消息都保存在统一格式中，切换 Provider 不影响历史。

### 2. 元数据记录
记录每条消息使用的 Provider 和模型。

### 3. 无缝切换
切换 Provider 只需修改参数，业务逻辑不变。

---

**版本：** v1.0
**最后更新：** 2026-02-19
