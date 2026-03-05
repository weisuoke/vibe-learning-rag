# 核心概念 3：Agent 响应流程

> **理解 OpenClaw Agent 如何自动响应消息**

---

## 概念定义

**Agent 响应流程（Agent Response Flow）** 是 OpenClaw 中实现智能自动回复的核心机制。Agent 监听来自各个通道的入站消息，根据预定义的规则或 AI 模型自动生成响应，并通过 Gateway 发送回复。

**核心价值**：
- 🤖 **自动化**：无需人工干预，自动处理消息
- 🧠 **智能化**：支持 AI 模型（如 GPT-4）生成响应
- 🔄 **实时性**：消息到达立即处理和响应
- 🌐 **多通道**：统一的 Agent 处理所有通道的消息

---

## 为什么需要 Agent？

### 问题场景

假设你想创建一个客服机器人，需要：

1. **监听消息**：实时接收用户发送的消息
2. **理解意图**：分析用户的问题
3. **生成回复**：根据问题生成合适的答案
4. **发送响应**：将答案发送回用户

**传统方案**：
```typescript
// 手动轮询检查新消息
setInterval(async () => {
  const messages = await checkNewMessages();
  for (const msg of messages) {
    const response = await generateResponse(msg);
    await sendMessage(response);
  }
}, 5000); // 每5秒检查一次
```

**问题**：
- 延迟高（最多5秒才能响应）
- 资源浪费（大部分时间没有新消息）
- 代码复杂（需要管理状态和错误）

### OpenClaw 的解决方案：Agent 系统

**核心思想**：事件驱动的自动响应

```
入站消息 → Agent 监听 → 处理逻辑 → 生成响应 → 发送回复
```

**优势**：
- ✅ 实时响应（消息到达立即处理）
- ✅ 资源高效（只在有消息时处理）
- ✅ 易于扩展（支持多个 Agent）
- ✅ 配置驱动（无需编写代码）

---

## Agent 响应流程详解

### 完整流程图

```
┌─────────────────────────────────────────────────────────────┐
│                   OpenClaw Agent 响应流程                    │
└─────────────────────────────────────────────────────────────┘

用户（发送者）          Gateway              Agent              LLM
     │                    │                    │                 │
     │  1. 发送消息        │                    │                 │
     ├───────────────────>│                    │                 │
     │  "今天天气怎么样？"  │                    │                 │
     │                    │                    │                 │
     │                    │  2. 入站消息事件    │                 │
     │                    ├───────────────────>│                 │
     │                    │  InboundMessage    │                 │
     │                    │                    │                 │
     │                    │                    │  3. 检查是否处理 │
     │                    │                    │  canHandle()?   │
     │                    │                    │  ✓ Yes          │
     │                    │                    │                 │
     │                    │                    │  4. 调用 LLM    │
     │                    │                    ├────────────────>│
     │                    │                    │  "今天天气..."   │
     │                    │                    │                 │
     │                    │                    │  5. LLM 响应    │
     │                    │                    │<────────────────┤
     │                    │                    │  "今天晴天..."   │
     │                    │                    │                 │
     │                    │  6. 生成响应消息    │                 │
     │                    │<───────────────────┤                 │
     │                    │  OutboundMessage   │                 │
     │                    │                    │                 │
     │  7. 接收回复        │                    │                 │
     │<───────────────────┤                    │                 │
     │  "今天晴天，温度25°C" │                    │                 │
     │                    │                    │                 │
```

### 步骤详解

#### 步骤 1：用户发送消息

用户通过任何已配对的通道（WhatsApp、Telegram、Discord）发送消息：

```
用户 → WhatsApp: "今天天气怎么样？"
```

#### 步骤 2：Gateway 接收入站消息

Gateway 监听所有通道的入站消息，并触发事件：

```typescript
// src/infra/inbound/message-listener.ts:45
export class MessageListener {
  private gateway: Gateway;
  private agents: Agent[] = [];

  constructor(gateway: Gateway) {
    this.gateway = gateway;
    this.loadAgents();
  }

  // 启动监听
  start(): void {
    this.gateway.on('message', async (message: InboundMessage) => {
      await this.handleInboundMessage(message);
    });
  }

  // 处理入站消息
  private async handleInboundMessage(message: InboundMessage): Promise<void> {
    console.log(`Received message from ${message.sender}: ${message.text}`);

    // 遍历所有 Agent，找到能处理该消息的 Agent
    for (const agent of this.agents) {
      if (await agent.canHandle(message)) {
        await this.processWithAgent(agent, message);
        break; // 只由第一个匹配的 Agent 处理
      }
    }
  }
}
```

**入站消息结构**：
```typescript
interface InboundMessage {
  id: string;                    // 消息 ID
  channelType: string;           // 通道类型（whatsapp、telegram 等）
  channelId: string;             // 通道 ID
  sender: string;                // 发送者标识
  senderName?: string;           // 发送者名称
  text: string;                  // 消息文本
  timestamp: number;             // 时间戳
  chatType: 'direct' | 'group';  // 聊天类型
  isGroupMessage: boolean;       // 是否群组消息
  attachments?: Attachment[];    // 附件
  metadata?: Record<string, any>; // 元数据
}
```

#### 步骤 3：Agent 检查是否处理

每个 Agent 实现 `canHandle()` 方法，决定是否处理该消息：

```typescript
// src/agents/agent-interface.ts:23
export interface Agent {
  name: string;
  canHandle(message: InboundMessage): Promise<boolean>;
  handle(message: InboundMessage): Promise<AgentResponse>;
}

// 示例：天气 Agent
export class WeatherAgent implements Agent {
  name = 'weather-agent';

  async canHandle(message: InboundMessage): Promise<boolean> {
    // 检查消息是否包含天气相关关键词
    const keywords = ['天气', 'weather', '温度', 'temperature'];
    return keywords.some(keyword =>
      message.text.toLowerCase().includes(keyword)
    );
  }

  async handle(message: InboundMessage): Promise<AgentResponse> {
    // 获取天气信息
    const weather = await this.getWeather();
    return {
      text: `今天${weather.condition}，温度${weather.temp}°C`,
      format: 'text'
    };
  }

  private async getWeather(): Promise<Weather> {
    // 调用天气 API
    // ...
  }
}
```

**Agent 类型**：

1. **规则型 Agent**：基于关键词或正则表达式匹配
2. **AI 型 Agent**：使用 LLM（如 GPT-4）处理所有消息
3. **混合型 Agent**：结合规则和 AI

#### 步骤 4：Agent 处理消息

Agent 的 `handle()` 方法处理消息并生成响应：

```typescript
// src/agents/ai-agent.ts:56
export class AIAgent implements Agent {
  name = 'ai-agent';
  private llm: LLMClient;

  constructor(config: AIAgentConfig) {
    this.llm = new LLMClient({
      apiKey: config.apiKey,
      model: config.model || 'gpt-4'
    });
  }

  async canHandle(message: InboundMessage): Promise<boolean> {
    // AI Agent 处理所有消息
    return true;
  }

  async handle(message: InboundMessage): Promise<AgentResponse> {
    // 构建 LLM 提示词
    const prompt = this.buildPrompt(message);

    // 调用 LLM
    const response = await this.llm.chat({
      messages: [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: prompt }
      ],
      temperature: 0.7,
      max_tokens: 500
    });

    return {
      text: response.content,
      format: 'text',
      metadata: {
        model: this.llm.model,
        tokens: response.usage.total_tokens
      }
    };
  }

  private buildPrompt(message: InboundMessage): string {
    return `User: ${message.text}\n\nPlease provide a helpful response.`;
  }
}
```

#### 步骤 5：生成响应消息

Agent 返回响应后，系统将其转换为出站消息：

```typescript
// src/infra/inbound/message-listener.ts:89
private async processWithAgent(
  agent: Agent,
  message: InboundMessage
): Promise<void> {
  try {
    // 调用 Agent 处理
    const response = await agent.handle(message);

    // 转换为出站消息
    const outboundMessage: OutboundMessage = {
      channelType: message.channelType,
      channelId: message.sender, // 回复给发送者
      text: response.text,
      format: response.format || 'text',
      replyTo: message.id, // 标记为回复
      timestamp: Date.now()
    };

    // 发送响应
    await this.sendResponse(outboundMessage);

    console.log(`Agent ${agent.name} responded to message ${message.id}`);
  } catch (error) {
    console.error(`Agent ${agent.name} failed to process message:`, error);
    // 可选：发送错误消息给用户
    await this.sendErrorResponse(message, error);
  }
}
```

#### 步骤 6：发送响应

响应通过 Gateway 发送回用户：

```typescript
// src/infra/inbound/message-listener.ts:123
private async sendResponse(message: OutboundMessage): Promise<void> {
  const sendService = new OutboundSendService();
  await sendService.send(message);
}
```

---

## Agent 配置

### 配置文件格式

```yaml
# ~/.openclaw/agents.yaml
agents:
  # AI Agent（使用 GPT-4）
  - name: ai-assistant
    type: ai
    enabled: true
    priority: 10
    config:
      provider: openai
      model: gpt-4
      apiKey: ${OPENAI_API_KEY}
      systemPrompt: "You are a helpful assistant."
      temperature: 0.7
      maxTokens: 500

  # 天气 Agent（规则型）
  - name: weather-bot
    type: rule
    enabled: true
    priority: 20
    config:
      keywords:
        - "天气"
        - "weather"
        - "温度"
      apiUrl: "https://api.weather.com"
      apiKey: ${WEATHER_API_KEY}

  # 客服 Agent（混合型）
  - name: customer-service
    type: hybrid
    enabled: true
    priority: 15
    config:
      rules:
        - pattern: "订单.*"
          action: "查询订单"
        - pattern: "退款.*"
          action: "处理退款"
      fallbackToAI: true
      aiModel: gpt-4
```

**配置字段说明**：
- `name`：Agent 名称（唯一标识）
- `type`：Agent 类型（ai、rule、hybrid）
- `enabled`：是否启用
- `priority`：优先级（数字越小优先级越高）
- `config`：Agent 特定配置

### 启动 Agent

```bash
# 使用默认配置启动
openclaw agent start

# 使用自定义配置文件
openclaw agent start --config ./my-agents.yaml

# 只启动特定 Agent
openclaw agent start --agent ai-assistant

# 后台运行
openclaw agent start --daemon
```

**输出示例**：
```
✓ Loading agent configuration from ~/.openclaw/agents.yaml
✓ Loaded 3 agents: ai-assistant, weather-bot, customer-service
✓ Starting agents...
✓ Agent ai-assistant started (priority: 10)
✓ Agent customer-service started (priority: 15)
✓ Agent weather-bot started (priority: 20)
✓ All agents running
✓ Listening for messages...
```

---

## Agent 类型详解

### 1. 规则型 Agent（Rule-based Agent）

**特点**：
- 基于关键词或正则表达式匹配
- 响应速度快
- 适合固定场景

**示例**：
```typescript
// src/agents/rule-agent.ts:34
export class RuleAgent implements Agent {
  name = 'rule-agent';
  private rules: Rule[];

  constructor(config: RuleAgentConfig) {
    this.rules = config.rules;
  }

  async canHandle(message: InboundMessage): Promise<boolean> {
    return this.rules.some(rule =>
      this.matchRule(rule, message.text)
    );
  }

  async handle(message: InboundMessage): Promise<AgentResponse> {
    const matchedRule = this.rules.find(rule =>
      this.matchRule(rule, message.text)
    );

    if (!matchedRule) {
      return { text: 'Sorry, I cannot help with that.' };
    }

    return {
      text: matchedRule.response,
      format: 'text'
    };
  }

  private matchRule(rule: Rule, text: string): boolean {
    if (rule.pattern) {
      const regex = new RegExp(rule.pattern, 'i');
      return regex.test(text);
    }
    if (rule.keywords) {
      return rule.keywords.some(keyword =>
        text.toLowerCase().includes(keyword.toLowerCase())
      );
    }
    return false;
  }
}
```

**配置示例**：
```yaml
agents:
  - name: faq-bot
    type: rule
    config:
      rules:
        - keywords: ["营业时间", "工作时间"]
          response: "我们的营业时间是周一至周五 9:00-18:00"
        - pattern: "订单.*查询"
          response: "请提供您的订单号，我会帮您查询"
```

### 2. AI 型 Agent（AI-powered Agent）

**特点**：
- 使用 LLM（如 GPT-4）生成响应
- 能处理复杂问题
- 响应更自然

**示例**：
```typescript
// src/agents/ai-agent.ts:78
export class AIAgent implements Agent {
  name = 'ai-agent';
  private llm: LLMClient;
  private conversationHistory: Map<string, Message[]> = new Map();

  async handle(message: InboundMessage): Promise<AgentResponse> {
    // 获取对话历史
    const history = this.conversationHistory.get(message.sender) || [];

    // 构建消息列表
    const messages = [
      { role: 'system', content: this.systemPrompt },
      ...history,
      { role: 'user', content: message.text }
    ];

    // 调用 LLM
    const response = await this.llm.chat({ messages });

    // 保存对话历史
    history.push(
      { role: 'user', content: message.text },
      { role: 'assistant', content: response.content }
    );
    this.conversationHistory.set(message.sender, history);

    return {
      text: response.content,
      format: 'text'
    };
  }
}
```

**配置示例**：
```yaml
agents:
  - name: gpt4-assistant
    type: ai
    config:
      provider: openai
      model: gpt-4
      systemPrompt: |
        You are a helpful customer service assistant.
        Be polite, concise, and professional.
      temperature: 0.7
      maxTokens: 500
      conversationMemory: true
```

### 3. 混合型 Agent（Hybrid Agent）

**特点**：
- 结合规则和 AI
- 规则优先，AI 兜底
- 平衡速度和灵活性

**示例**：
```typescript
// src/agents/hybrid-agent.ts:45
export class HybridAgent implements Agent {
  name = 'hybrid-agent';
  private ruleAgent: RuleAgent;
  private aiAgent: AIAgent;

  async handle(message: InboundMessage): Promise<AgentResponse> {
    // 先尝试规则匹配
    if (await this.ruleAgent.canHandle(message)) {
      return await this.ruleAgent.handle(message);
    }

    // 规则不匹配，使用 AI
    return await this.aiAgent.handle(message);
  }
}
```

---

## 实际应用场景

### 场景 1：智能客服机器人

```yaml
# customer-service-agent.yaml
agents:
  - name: customer-service
    type: hybrid
    config:
      rules:
        - keywords: ["订单", "order"]
          action: query_order
        - keywords: ["退款", "refund"]
          action: process_refund
        - keywords: ["投诉", "complaint"]
          action: escalate_to_human
      fallbackToAI: true
      aiConfig:
        model: gpt-4
        systemPrompt: |
          You are a customer service representative.
          Be helpful, empathetic, and professional.
```

**启动**：
```bash
openclaw agent start --config customer-service-agent.yaml
```

**效果**：
```
用户: "我想查询订单"
Agent: "请提供您的订单号"

用户: "订单号是 12345"
Agent: "您的订单 12345 状态：已发货，预计明天送达"

用户: "为什么这么慢？"
Agent: "非常抱歉给您带来不便。由于物流高峰期，配送时间略有延迟。我们会尽快为您送达。"
```

### 场景 2：多语言支持

```yaml
agents:
  - name: multilingual-assistant
    type: ai
    config:
      model: gpt-4
      systemPrompt: |
        You are a multilingual assistant.
        Detect the user's language and respond in the same language.
        Supported languages: English, Chinese, Spanish, French.
```

**效果**：
```
用户: "Hello, how are you?"
Agent: "Hello! I'm doing great, thank you for asking. How can I help you today?"

用户: "你好，今天天气怎么样？"
Agent: "你好！今天天气晴朗，温度适宜。有什么我可以帮助你的吗？"
```

### 场景 3：任务自动化

```yaml
agents:
  - name: task-automation
    type: rule
    config:
      rules:
        - pattern: "提醒我.*"
          action: create_reminder
        - pattern: "设置闹钟.*"
          action: set_alarm
        - pattern: "添加待办.*"
          action: add_todo
```

**效果**：
```
用户: "提醒我明天下午3点开会"
Agent: "✓ 已设置提醒：明天下午3点开会"

用户: "添加待办：买牛奶"
Agent: "✓ 已添加待办事项：买牛奶"
```

---

## Agent 管理命令

### 查看 Agent 状态

```bash
openclaw agent status
```

**输出**：
```
┌──────────────────┬─────────┬──────────┬────────────────┐
│ Agent Name       │ Type    │ Status   │ Messages       │
├──────────────────┼─────────┼──────────┼────────────────┤
│ ai-assistant     │ AI      │ ✓ Running│ 127 processed  │
│ weather-bot      │ Rule    │ ✓ Running│ 45 processed   │
│ customer-service │ Hybrid  │ ✓ Running│ 89 processed   │
└──────────────────┴─────────┴──────────┴────────────────┘
```

### 停止 Agent

```bash
# 停止所有 Agent
openclaw agent stop

# 停止特定 Agent
openclaw agent stop --agent ai-assistant
```

### 重启 Agent

```bash
# 重启所有 Agent
openclaw agent restart

# 重启特定 Agent
openclaw agent restart --agent ai-assistant
```

### 查看 Agent 日志

```bash
# 查看所有 Agent 日志
openclaw agent logs

# 查看特定 Agent 日志
openclaw agent logs --agent ai-assistant

# 实时查看日志
openclaw agent logs --follow
```

---

## 高级特性

### 1. Agent 优先级

多个 Agent 可以同时运行，优先级决定处理顺序：

```yaml
agents:
  - name: urgent-handler
    priority: 1  # 最高优先级
  - name: normal-handler
    priority: 10
  - name: fallback-handler
    priority: 100  # 最低优先级
```

**处理逻辑**：
```typescript
// 按优先级排序
const sortedAgents = agents.sort((a, b) => a.priority - b.priority);

// 依次尝试处理
for (const agent of sortedAgents) {
  if (await agent.canHandle(message)) {
    await agent.handle(message);
    break; // 只由第一个匹配的 Agent 处理
  }
}
```

### 2. 对话上下文管理

AI Agent 可以记住对话历史：

```typescript
export class AIAgent implements Agent {
  private conversationHistory: Map<string, Message[]> = new Map();
  private maxHistoryLength = 10;

  async handle(message: InboundMessage): Promise<AgentResponse> {
    // 获取对话历史
    const history = this.conversationHistory.get(message.sender) || [];

    // 限制历史长度
    if (history.length > this.maxHistoryLength) {
      history.splice(0, history.length - this.maxHistoryLength);
    }

    // 构建消息列表
    const messages = [
      { role: 'system', content: this.systemPrompt },
      ...history,
      { role: 'user', content: message.text }
    ];

    // 调用 LLM
    const response = await this.llm.chat({ messages });

    // 保存对话历史
    history.push(
      { role: 'user', content: message.text },
      { role: 'assistant', content: response.content }
    );
    this.conversationHistory.set(message.sender, history);

    return { text: response.content };
  }
}
```

### 3. Agent 链（Agent Chaining）

多个 Agent 可以协作处理复杂任务：

```typescript
export class AgentChain {
  private agents: Agent[];

  async process(message: InboundMessage): Promise<AgentResponse> {
    let currentMessage = message;

    for (const agent of this.agents) {
      const response = await agent.handle(currentMessage);

      // 将响应作为下一个 Agent 的输入
      currentMessage = {
        ...currentMessage,
        text: response.text
      };
    }

    return { text: currentMessage.text };
  }
}
```

**示例**：
```yaml
agentChains:
  - name: translation-chain
    agents:
      - name: translator
        type: ai
        config:
          systemPrompt: "Translate to English"
      - name: responder
        type: ai
        config:
          systemPrompt: "Provide a helpful response"
```

### 4. 错误处理和重试

```typescript
export class ResilientAgent implements Agent {
  private maxRetries = 3;
  private retryDelay = 1000;

  async handle(message: InboundMessage): Promise<AgentResponse> {
    let lastError: Error;

    for (let i = 0; i < this.maxRetries; i++) {
      try {
        return await this.processMessage(message);
      } catch (error) {
        lastError = error;
        console.error(`Attempt ${i + 1} failed:`, error);

        if (i < this.maxRetries - 1) {
          await this.delay(this.retryDelay * (i + 1));
        }
      }
    }

    // 所有重试都失败，返回错误消息
    return {
      text: 'Sorry, I encountered an error. Please try again later.',
      metadata: { error: lastError.message }
    };
  }
}
```

---

## 性能优化

### 1. 异步处理

```typescript
// 不阻塞主线程
async handleInboundMessage(message: InboundMessage): Promise<void> {
  // 立即返回，后台处理
  setImmediate(async () => {
    for (const agent of this.agents) {
      if (await agent.canHandle(message)) {
        await this.processWithAgent(agent, message);
        break;
      }
    }
  });
}
```

### 2. 缓存响应

```typescript
export class CachedAgent implements Agent {
  private cache: Map<string, AgentResponse> = new Map();
  private cacheTTL = 5 * 60 * 1000; // 5分钟

  async handle(message: InboundMessage): Promise<AgentResponse> {
    const cacheKey = this.getCacheKey(message);
    const cached = this.cache.get(cacheKey);

    if (cached && !this.isCacheExpired(cached)) {
      return cached;
    }

    const response = await this.processMessage(message);
    this.cache.set(cacheKey, response);

    return response;
  }
}
```

### 3. 并行处理

```typescript
// 多个 Agent 并行检查
const canHandleResults = await Promise.all(
  agents.map(agent => agent.canHandle(message))
);

const matchedAgent = agents.find((_, index) => canHandleResults[index]);
```

---

## 总结

Agent 响应流程是 OpenClaw 实现智能自动回复的核心：

**核心流程**：
```
入站消息 → Agent 监听 → 检查处理 → 生成响应 → 发送回复
```

**Agent 类型**：
- **规则型**：快速、固定场景
- **AI 型**：灵活、复杂问题
- **混合型**：平衡速度和灵活性

**关键特性**：
- 事件驱动，实时响应
- 支持多个 Agent 协作
- 配置驱动，易于扩展
- 对话上下文管理

**下一步**：
- [03_核心概念_4_多通道支持](./03_核心概念_4_多通道支持.md) - 了解 OpenClaw 支持的所有通道
- [07_实战代码_场景4_Agent交互](./07_实战代码_场景4_Agent交互.md) - 实战演练 Agent 交互

---

**版本**: v1.0
**最后更新**: 2026-02-22
**阅读时间**: 20分钟
**难度**: ⭐⭐⭐⭐☆
