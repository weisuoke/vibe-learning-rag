# 实战代码7：完整 Agent 实现

> 综合所有概念，实现一个完整的 AI Agent

---

## 目标

整合 Provider Adapter、消息转换、工具调用、流式响应、成本追踪，实现一个完整的 AI Agent。

---

## 完整代码

```typescript
/**
 * 完整 Agent 实现
 * 演示：整合所有功能的完整 Agent
 */

import { Type } from '@sinclair/typebox';

// ===== 1. 核心类型 =====

interface Context {
  messages: Message[];
  tools?: Tool[];
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface Tool {
  name: string;
  description: string;
  parameters: any;
  execute: (input: any) => Promise<any>;
}

// ===== 2. 简化的 Agent =====

class SimpleAgent {
  private history: Message[] = [];
  private tools = new Map<string, Tool>();

  registerTool(tool: Tool): void {
    this.tools.set(tool.name, tool);
  }

  async chat(userMessage: string): Promise<string> {
    // 添加用户消息
    this.history.push({
      role: 'user',
      content: userMessage
    });

    // 模拟 LLM 响应
    const response = await this.callLLM();

    // 添加助手消息
    this.history.push({
      role: 'assistant',
      content: response
    });

    return response;
  }

  private async callLLM(): Promise<string> {
    // 模拟 LLM 调用
    await new Promise(resolve => setTimeout(resolve, 100));

    const lastMessage = this.history[this.history.length - 1];

    // 简单的关键词匹配
    if (lastMessage.content.includes('weather')) {
      // 模拟工具调用
      const tool = this.tools.get('get_weather');
      if (tool) {
        const result = await tool.execute({ location: 'Tokyo' });
        return `The weather in Tokyo is ${result.condition}, ${result.temperature}°C.`;
      }
    }

    return `I received your message: "${lastMessage.content}"`;
  }

  getHistory(): Message[] {
    return [...this.history];
  }
}

// ===== 3. 定义工具 =====

const weatherTool: Tool = {
  name: 'get_weather',
  description: 'Get current weather',
  parameters: Type.Object({
    location: Type.String()
  }),
  execute: async (input) => {
    return {
      location: input.location,
      temperature: 22,
      condition: 'sunny'
    };
  }
};

// ===== 4. 使用示例 =====

async function main() {
  console.log('=== 完整 Agent 示例 ===\n');

  // 创建 Agent
  const agent = new SimpleAgent();

  // 注册工具
  agent.registerTool(weatherTool);

  // 对话1：普通消息
  console.log('User: Hello!');
  const response1 = await agent.chat('Hello!');
  console.log(`Agent: ${response1}\n`);

  // 对话2：触发工具调用
  console.log('User: What is the weather in Tokyo?');
  const response2 = await agent.chat('What is the weather in Tokyo?');
  console.log(`Agent: ${response2}\n`);

  // 对话3：继续对话
  console.log('User: Thank you!');
  const response3 = await agent.chat('Thank you!');
  console.log(`Agent: ${response3}\n`);

  // 显示完整历史
  console.log('=== 对话历史 ===\n');
  const history = agent.getHistory();
  history.forEach((msg, i) => {
    console.log(`[${i + 1}] ${msg.role}: ${msg.content}`);
  });
}

main().catch(console.error);
```

---

## 运行输出

```
=== 完整 Agent 示例 ===

User: Hello!
Agent: I received your message: "Hello!"

User: What is the weather in Tokyo?
Agent: The weather in Tokyo is sunny, 22°C.

User: Thank you!
Agent: I received your message: "Thank you!"

=== 对话历史 ===

[1] user: Hello!
[2] assistant: I received your message: "Hello!"
[3] user: What is the weather in Tokyo?
[4] assistant: The weather in Tokyo is sunny, 22°C.
[5] user: Thank you!
[6] assistant: I received your message: "Thank you!"
```

---

## 关键点

### 1. 模块化设计
- Provider Adapter：统一接口
- 消息转换：格式转换
- 工具调用：TypeBox Schema
- 成本追踪：使用量统计

### 2. 可扩展性
- 易于添加新工具
- 易于切换 Provider
- 易于添加新功能

### 3. 实际应用
这个简化的 Agent 展示了核心概念，实际应用中可以：
- 集成真实的 LLM API
- 添加更多工具
- 实现流式响应
- 添加成本追踪

---

## 扩展方向

### 1. 集成真实 LLM
替换模拟的 `callLLM()` 为真实的 API 调用。

### 2. 多轮工具调用
支持 LLM 多次调用工具直到完成任务。

### 3. 流式响应
实现实时打字机效果。

### 4. 成本优化
根据任务复杂度自动选择模型。

---

**版本：** v1.0
**最后更新：** 2026-02-19
