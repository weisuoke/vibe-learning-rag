# 实战代码 - 场景4：Agent 交互

> **完整的 TypeScript 代码示例：实现智能自动回复系统**

---

## 场景概述

本场景演示如何实现 Agent 自动响应系统，包括：
- Agent 配置和启动
- 规则型 Agent（关键词匹配）
- AI 型 Agent（GPT-4 集成）
- 混合型 Agent（规则 + AI）
- 对话上下文管理
- Agent 优先级和路由

**难度**：⭐⭐⭐⭐⭐（专家级）
**预计时间**：30分钟
**前置要求**：已完成场景1-3，了解 LLM API

---

## 完整代码

### 文件结构

```
agent-interaction/
├── package.json
├── tsconfig.json
├── src/
│   ├── index.ts                  # 主程序
│   ├── agents/
│   │   ├── base-agent.ts         # Agent 基类
│   │   ├── rule-agent.ts         # 规则型 Agent
│   │   ├── ai-agent.ts           # AI 型 Agent
│   │   └── hybrid-agent.ts       # 混合型 Agent
│   ├── agent-manager.ts          # Agent 管理器
│   └── message-listener.ts       # 消息监听器
├── config/
│   └── agents.yaml               # Agent 配置
└── .env
```

### 1. src/agents/base-agent.ts

```typescript
/**
 * Agent 基类
 * 定义所有 Agent 的通用接口
 */

export interface InboundMessage {
  id: string;
  channelType: string;
  sender: string;
  text: string;
  timestamp: number;
}

export interface AgentResponse {
  text: string;
  format?: 'text' | 'markdown' | 'html';
  metadata?: Record<string, any>;
}

export abstract class BaseAgent {
  name: string;
  priority: number;
  enabled: boolean;

  constructor(name: string, priority: number = 10) {
    this.name = name;
    this.priority = priority;
    this.enabled = true;
  }

  /**
   * 检查是否可以处理该消息
   */
  abstract canHandle(message: InboundMessage): Promise<boolean>;

  /**
   * 处理消息并生成响应
   */
  abstract handle(message: InboundMessage): Promise<AgentResponse>;

  /**
   * 获取 Agent 信息
   */
  getInfo(): { name: string; priority: number; enabled: boolean } {
    return {
      name: this.name,
      priority: this.priority,
      enabled: this.enabled
    };
  }
}
```

### 2. src/agents/rule-agent.ts

```typescript
/**
 * 规则型 Agent
 * 基于关键词和正则表达式匹配
 */

import { BaseAgent, InboundMessage, AgentResponse } from './base-agent.js';

interface Rule {
  keywords?: string[];
  pattern?: string;
  response: string;
}

export class RuleAgent extends BaseAgent {
  private rules: Rule[];

  constructor(name: string, rules: Rule[], priority: number = 10) {
    super(name, priority);
    this.rules = rules;
  }

  async canHandle(message: InboundMessage): Promise<boolean> {
    return this.rules.some(rule => this.matchRule(rule, message.text));
  }

  async handle(message: InboundMessage): Promise<AgentResponse> {
    const matchedRule = this.rules.find(rule =>
      this.matchRule(rule, message.text)
    );

    if (!matchedRule) {
      return {
        text: 'Sorry, I cannot help with that.',
        format: 'text'
      };
    }

    return {
      text: matchedRule.response,
      format: 'text',
      metadata: {
        agentType: 'rule',
        matchedRule: matchedRule.keywords || matchedRule.pattern
      }
    };
  }

  private matchRule(rule: Rule, text: string): boolean {
    const lowerText = text.toLowerCase();

    // 关键词匹配
    if (rule.keywords) {
      return rule.keywords.some(keyword =>
        lowerText.includes(keyword.toLowerCase())
      );
    }

    // 正则表达式匹配
    if (rule.pattern) {
      const regex = new RegExp(rule.pattern, 'i');
      return regex.test(text);
    }

    return false;
  }
}
```

### 3. src/agents/ai-agent.ts

```typescript
/**
 * AI 型 Agent
 * 使用 LLM（如 GPT-4）生成响应
 */

import { BaseAgent, InboundMessage, AgentResponse } from './base-agent.js';

interface Message {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface LLMConfig {
  apiKey: string;
  model: string;
  systemPrompt: string;
  temperature?: number;
  maxTokens?: number;
}

export class AIAgent extends BaseAgent {
  private config: LLMConfig;
  private conversationHistory: Map<string, Message[]> = new Map();
  private maxHistoryLength = 10;

  constructor(name: string, config: LLMConfig, priority: number = 100) {
    super(name, priority);
    this.config = config;
  }

  async canHandle(message: InboundMessage): Promise<boolean> {
    // AI Agent 可以处理所有消息
    return true;
  }

  async handle(message: InboundMessage): Promise<AgentResponse> {
    // 获取对话历史
    const history = this.conversationHistory.get(message.sender) || [];

    // 构建消息列表
    const messages: Message[] = [
      { role: 'system', content: this.config.systemPrompt },
      ...history,
      { role: 'user', content: message.text }
    ];

    // 调用 LLM
    const response = await this.callLLM(messages);

    // 保存对话历史
    history.push(
      { role: 'user', content: message.text },
      { role: 'assistant', content: response }
    );

    // 限制历史长度
    if (history.length > this.maxHistoryLength * 2) {
      history.splice(0, history.length - this.maxHistoryLength * 2);
    }

    this.conversationHistory.set(message.sender, history);

    return {
      text: response,
      format: 'text',
      metadata: {
        agentType: 'ai',
        model: this.config.model,
        historyLength: history.length
      }
    };
  }

  private async callLLM(messages: Message[]): Promise<string> {
    // 模拟 LLM API 调用
    // 实际应用中应该调用真实的 API（如 OpenAI）
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.config.apiKey}`
      },
      body: JSON.stringify({
        model: this.config.model,
        messages,
        temperature: this.config.temperature || 0.7,
        max_tokens: this.config.maxTokens || 500
      })
    });

    const data = await response.json();
    return data.choices[0].message.content;
  }

  /**
   * 清除对话历史
   */
  clearHistory(sender?: string): void {
    if (sender) {
      this.conversationHistory.delete(sender);
    } else {
      this.conversationHistory.clear();
    }
  }
}
```

### 4. src/agents/hybrid-agent.ts

```typescript
/**
 * 混合型 Agent
 * 结合规则和 AI，规则优先，AI 兜底
 */

import { BaseAgent, InboundMessage, AgentResponse } from './base-agent.js';
import { RuleAgent } from './rule-agent.js';
import { AIAgent } from './ai-agent.js';

export class HybridAgent extends BaseAgent {
  private ruleAgent: RuleAgent;
  private aiAgent: AIAgent;

  constructor(
    name: string,
    ruleAgent: RuleAgent,
    aiAgent: AIAgent,
    priority: number = 50
  ) {
    super(name, priority);
    this.ruleAgent = ruleAgent;
    this.aiAgent = aiAgent;
  }

  async canHandle(message: InboundMessage): Promise<boolean> {
    // 混合型 Agent 总是可以处理（因为有 AI 兜底）
    return true;
  }

  async handle(message: InboundMessage): Promise<AgentResponse> {
    // 先尝试规则匹配
    if (await this.ruleAgent.canHandle(message)) {
      const response = await this.ruleAgent.handle(message);
      return {
        ...response,
        metadata: {
          ...response.metadata,
          agentType: 'hybrid',
          handledBy: 'rule'
        }
      };
    }

    // 规则不匹配，使用 AI
    const response = await this.aiAgent.handle(message);
    return {
      ...response,
      metadata: {
        ...response.metadata,
        agentType: 'hybrid',
        handledBy: 'ai'
      }
    };
  }
}
```

### 5. src/agent-manager.ts

```typescript
/**
 * Agent 管理器
 * 管理多个 Agent 的注册、路由和执行
 */

import { BaseAgent, InboundMessage, AgentResponse } from './agents/base-agent.js';

export class AgentManager {
  private agents: BaseAgent[] = [];

  /**
   * 注册 Agent
   */
  register(agent: BaseAgent): void {
    this.agents.push(agent);
    // 按优先级排序（数字越小优先级越高）
    this.agents.sort((a, b) => a.priority - b.priority);
    console.log(`✓ Registered agent: ${agent.name} (priority: ${agent.priority})`);
  }

  /**
   * 处理消息
   */
  async processMessage(message: InboundMessage): Promise<AgentResponse | null> {
    console.log(`\n📨 Processing message from ${message.sender}: "${message.text}"`);

    // 遍历所有 Agent，找到第一个可以处理的
    for (const agent of this.agents) {
      if (!agent.enabled) {
        continue;
      }

      if (await agent.canHandle(message)) {
        console.log(`✓ Matched agent: ${agent.name}`);

        try {
          const response = await agent.handle(message);
          console.log(`✓ Response generated: "${response.text.substring(0, 50)}..."`);
          return response;
        } catch (error) {
          console.error(`✗ Agent ${agent.name} failed:`, error);
          continue; // 尝试下一个 Agent
        }
      }
    }

    console.log('⚠️  No agent could handle this message');
    return null;
  }

  /**
   * 获取所有 Agent 信息
   */
  getAgents(): Array<{ name: string; priority: number; enabled: boolean }> {
    return this.agents.map(agent => agent.getInfo());
  }

  /**
   * 启用/禁用 Agent
   */
  setAgentEnabled(name: string, enabled: boolean): void {
    const agent = this.agents.find(a => a.name === name);
    if (agent) {
      agent.enabled = enabled;
      console.log(`✓ Agent ${name} ${enabled ? 'enabled' : 'disabled'}`);
    }
  }
}
```

### 6. src/message-listener.ts

```typescript
/**
 * 消息监听器
 * 监听入站消息并触发 Agent 处理
 */

import { exec } from 'child_process';
import { promisify } from 'util';
import { AgentManager } from './agent-manager.js';
import { InboundMessage } from './agents/base-agent.js';

const execAsync = promisify(exec);

export class MessageListener {
  private agentManager: AgentManager;
  private isListening = false;

  constructor(agentManager: AgentManager) {
    this.agentManager = agentManager;
  }

  /**
   * 启动监听
   */
  async start(): Promise<void> {
    console.log('👂 Starting message listener...\n');
    this.isListening = true;

    // 模拟消息监听（实际应该监听 Gateway 事件）
    while (this.isListening) {
      try {
        // 检查新消息
        const messages = await this.checkNewMessages();

        for (const message of messages) {
          // 处理消息
          const response = await this.agentManager.processMessage(message);

          if (response) {
            // 发送响应
            await this.sendResponse(message, response);
          }
        }

        // 等待一段时间再检查
        await this.delay(5000);
      } catch (error) {
        console.error('✗ Listener error:', error);
        await this.delay(5000);
      }
    }
  }

  /**
   * 停止监听
   */
  stop(): void {
    console.log('\n🛑 Stopping message listener...');
    this.isListening = false;
  }

  /**
   * 检查新消息
   */
  private async checkNewMessages(): Promise<InboundMessage[]> {
    // 实际应该从 Gateway 获取新消息
    // 这里返回空数组作为示例
    return [];
  }

  /**
   * 发送响应
   */
  private async sendResponse(
    originalMessage: InboundMessage,
    response: any
  ): Promise<void> {
    try {
      await execAsync(
        `openclaw message send "${response.text}" --channel ${originalMessage.channelType}`
      );
      console.log('✓ Response sent\n');
    } catch (error) {
      console.error('✗ Failed to send response:', error);
    }
  }

  /**
   * 延迟函数
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
```

### 7. src/index.ts

```typescript
/**
 * OpenClaw Agent 交互示例
 * 演示完整的 Agent 系统
 */

import 'dotenv/config';
import { AgentManager } from './agent-manager.js';
import { RuleAgent } from './agents/rule-agent.js';
import { AIAgent } from './agents/ai-agent.js';
import { HybridAgent } from './agents/hybrid-agent.js';
import { InboundMessage } from './agents/base-agent.js';

async function main() {
  console.log('🎯 OpenClaw Agent Interaction Example\n');

  // 1. 创建 Agent 管理器
  const manager = new AgentManager();

  // 2. 创建规则型 Agent（FAQ Bot）
  const faqAgent = new RuleAgent(
    'faq-bot',
    [
      {
        keywords: ['营业时间', '工作时间', '几点开门'],
        response: '我们的营业时间是周一至周五 9:00-18:00'
      },
      {
        keywords: ['地址', '位置', '在哪里'],
        response: '我们的地址是：北京市朝阳区xxx路xxx号'
      },
      {
        keywords: ['电话', '联系方式'],
        response: '客服电话：400-123-4567'
      }
    ],
    1 // 最高优先级
  );

  // 3. 创建 AI 型 Agent（GPT-4）
  const aiAgent = new AIAgent(
    'gpt4-assistant',
    {
      apiKey: process.env.OPENAI_API_KEY || '',
      model: 'gpt-4',
      systemPrompt: 'You are a helpful customer service assistant. Be polite and concise.',
      temperature: 0.7,
      maxTokens: 500
    },
    100 // 最低优先级（兜底）
  );

  // 4. 创建混合型 Agent
  const orderRuleAgent = new RuleAgent(
    'order-rules',
    [
      {
        pattern: '订单.*查询',
        response: '请提供您的订单号，我会帮您查询'
      },
      {
        pattern: '订单.*取消',
        response: '请提供订单号，我会帮您处理取消'
      }
    ],
    50
  );

  const hybridAgent = new HybridAgent(
    'order-assistant',
    orderRuleAgent,
    aiAgent,
    50
  );

  // 5. 注册所有 Agent
  manager.register(faqAgent);
  manager.register(hybridAgent);
  manager.register(aiAgent);

  console.log('\n📋 Registered agents:');
  manager.getAgents().forEach(agent => {
    console.log(`  - ${agent.name} (priority: ${agent.priority})`);
  });
  console.log();

  // 6. 测试场景
  await testScenarios(manager);
}

async function testScenarios(manager: AgentManager) {
  console.log('🧪 Testing scenarios:\n');

  // 场景1：FAQ 问题（规则型 Agent 处理）
  console.log('='.repeat(60));
  console.log('场景1：FAQ 问题');
  console.log('='.repeat(60));
  await manager.processMessage({
    id: 'msg1',
    channelType: 'whatsapp',
    sender: '+1234567890',
    text: '你们的营业时间是什么？',
    timestamp: Date.now()
  });

  await delay(1000);

  // 场景2：订单查询（混合型 Agent 处理 - 规则匹配）
  console.log('\n' + '='.repeat(60));
  console.log('场景2：订单查询');
  console.log('='.repeat(60));
  await manager.processMessage({
    id: 'msg2',
    channelType: 'telegram',
    sender: '@user123',
    text: '我想查询订单状态',
    timestamp: Date.now()
  });

  await delay(1000);

  // 场景3：复杂问题（AI Agent 处理）
  console.log('\n' + '='.repeat(60));
  console.log('场景3：复杂问题');
  console.log('='.repeat(60));
  await manager.processMessage({
    id: 'msg3',
    channelType: 'discord',
    sender: 'user#5678',
    text: '我的订单已经3天了还没发货，这是怎么回事？',
    timestamp: Date.now()
  });

  await delay(1000);

  // 场景4：对话上下文（AI Agent 记忆）
  console.log('\n' + '='.repeat(60));
  console.log('场景4：对话上下文');
  console.log('='.repeat(60));
  await manager.processMessage({
    id: 'msg4',
    channelType: 'whatsapp',
    sender: '+1234567890',
    text: '今天天气怎么样？',
    timestamp: Date.now()
  });

  await delay(1000);

  await manager.processMessage({
    id: 'msg5',
    channelType: 'whatsapp',
    sender: '+1234567890',
    text: '那明天呢？', // 依赖上下文
    timestamp: Date.now()
  });
}

function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// 运行主函数
main().catch(console.error);
```

---

## 配置文件

### config/agents.yaml

```yaml
agents:
  # FAQ Bot（规则型）
  - name: faq-bot
    type: rule
    priority: 1
    enabled: true
    rules:
      - keywords: ["营业时间", "工作时间"]
        response: "我们的营业时间是周一至周五 9:00-18:00"
      - keywords: ["地址", "位置"]
        response: "我们的地址是：北京市朝阳区xxx路xxx号"

  # Order Assistant（混合型）
  - name: order-assistant
    type: hybrid
    priority: 50
    enabled: true
    rules:
      - pattern: "订单.*查询"
        response: "请提供您的订单号"
    fallbackToAI: true

  # GPT-4 Assistant（AI 型）
  - name: gpt4-assistant
    type: ai
    priority: 100
    enabled: true
    config:
      model: gpt-4
      systemPrompt: "You are a helpful assistant."
      temperature: 0.7
```

---

## 运行步骤

### 1. 安装依赖

```bash
npm install
```

### 2. 配置环境变量

```bash
# .env
OPENAI_API_KEY=your_openai_api_key
```

### 3. 运行示例

```bash
npm start
```

**预期输出**：
```
🎯 OpenClaw Agent Interaction Example

✓ Registered agent: faq-bot (priority: 1)
✓ Registered agent: order-assistant (priority: 50)
✓ Registered agent: gpt4-assistant (priority: 100)

📋 Registered agents:
  - faq-bot (priority: 1)
  - order-assistant (priority: 50)
  - gpt4-assistant (priority: 100)

🧪 Testing scenarios:

============================================================
场景1：FAQ 问题
============================================================

📨 Processing message from +1234567890: "你们的营业时间是什么？"
✓ Matched agent: faq-bot
✓ Response generated: "我们的营业时间是周一至周五 9:00-18:00"

============================================================
场景2：订单查询
============================================================

📨 Processing message from @user123: "我想查询订单状态"
✓ Matched agent: order-assistant
✓ Response generated: "请提供您的订单号，我会帮您查询"

============================================================
场景3：复杂问题
============================================================

📨 Processing message from user#5678: "我的订单已经3天了还没发货..."
✓ Matched agent: gpt4-assistant
✓ Response generated: "非常抱歉给您带来不便。让我帮您查询一下订单状态..."

============================================================
场景4：对话上下文
============================================================

📨 Processing message from +1234567890: "今天天气怎么样？"
✓ Matched agent: gpt4-assistant
✓ Response generated: "今天天气晴朗，温度适宜..."

📨 Processing message from +1234567890: "那明天呢？"
✓ Matched agent: gpt4-assistant
✓ Response generated: "根据天气预报，明天也是晴天..."
```

---

## 最佳实践

### 1. Agent 优先级设计

```typescript
// 优先级规则：
// 1-10: 高优先级（FAQ、紧急处理）
// 11-50: 中优先级（业务逻辑）
// 51-100: 低优先级（AI 兜底）

const priorities = {
  urgent: 1,
  faq: 5,
  business: 25,
  ai: 100
};
```

### 2. 对话上下文管理

```typescript
// 限制历史长度避免 token 超限
private maxHistoryLength = 10;

// 定期清理过期对话
setInterval(() => {
  this.cleanupExpiredConversations();
}, 3600000); // 每小时清理一次
```

### 3. 错误处理和降级

```typescript
try {
  return await this.aiAgent.handle(message);
} catch (error) {
  // AI 失败，降级到规则
  return await this.ruleAgent.handle(message);
}
```

---

## 总结

本示例演示了 OpenClaw Agent 系统的完整实现：

1. ✅ 规则型 Agent（快速响应）
2. ✅ AI 型 Agent（智能处理）
3. ✅ 混合型 Agent（规则 + AI）
4. ✅ Agent 优先级和路由
5. ✅ 对话上下文管理

**关键要点**：
- Agent 按优先级处理消息
- 规则型 Agent 响应快，AI 型 Agent 灵活
- 混合型 Agent 平衡速度和智能
- 对话上下文提升用户体验

**下一步**：
- [场景5：故障排查](./07_实战代码_场景5_故障排查.md) - 诊断和解决问题
- [面试必问](./08_面试必问.md) - 深入理解 Agent 机制

---

**版本**: v1.0
**最后更新**: 2026-02-22
**代码行数**: ~600行
**难度**: ⭐⭐⭐⭐⭐
