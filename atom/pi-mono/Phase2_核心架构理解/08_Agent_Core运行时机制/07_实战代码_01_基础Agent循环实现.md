# 实战代码 01：基础 Agent 循环实现

> 从零实现一个最小可用的 Agent 循环，理解核心机制

---

## 代码目标

实现一个最小的 Agent 循环，包含：
- LLM 调用
- 工具检测
- 工具执行
- 上下文更新
- 循环终止

**代码长度：** ~150 行
**运行环境：** Node.js 18+, TypeScript

---

## 完整代码

```typescript
/**
 * 基础 Agent 循环实现
 * 演示：最小可用的 Agent Core 机制
 */

import Anthropic from '@anthropic-ai/sdk';
import fs from 'fs/promises';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

// ===== 1. 类型定义 =====

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface ToolCall {
  id: string;
  name: string;
  input: any;
}

interface ToolResult {
  output: string;
  details?: any;
}

// ===== 2. 工具定义 =====

const tools = [
  {
    name: 'read_file',
    description: 'Read file contents',
    input_schema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'File path to read' }
      },
      required: ['path']
    }
  },
  {
    name: 'write_file',
    description: 'Write file contents',
    input_schema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'File path to write' },
        content: { type: 'string', description: 'Content to write' }
      },
      required: ['path', 'content']
    }
  },
  {
    name: 'bash',
    description: 'Execute bash command',
    input_schema: {
      type: 'object',
      properties: {
        command: { type: 'string', description: 'Command to execute' }
      },
      required: ['command']
    }
  }
];

// ===== 3. 工具执行函数 =====

async function executeTool(toolCall: ToolCall): Promise<ToolResult> {
  console.log(`\n🔧 Executing tool: ${toolCall.name}`);
  console.log(`   Input:`, JSON.stringify(toolCall.input, null, 2));

  try {
    switch (toolCall.name) {
      case 'read_file': {
        const content = await fs.readFile(toolCall.input.path, 'utf-8');
        return {
          output: `File content:\n${content}`,
          details: { path: toolCall.input.path, size: content.length }
        };
      }

      case 'write_file': {
        await fs.writeFile(toolCall.input.path, toolCall.input.content);
        return {
          output: `File written: ${toolCall.input.path}`,
          details: { path: toolCall.input.path, bytes: toolCall.input.content.length }
        };
      }

      case 'bash': {
        const { stdout, stderr } = await execAsync(toolCall.input.command);
        return {
          output: stdout || stderr || 'Command executed',
          details: { command: toolCall.input.command }
        };
      }

      default:
        return {
          output: `Error: Unknown tool: ${toolCall.name}`,
          details: { error: 'Unknown tool' }
        };
    }
  } catch (error: any) {
    return {
      output: `Error: ${error.message}`,
      details: { error: error.stack }
    };
  }
}

// ===== 4. Agent 循环 =====

async function agentLoop(userMessage: string): Promise<void> {
  // 初始化 Anthropic 客户端
  const client = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY
  });

  // 初始化上下文
  const messages: Message[] = [
    { role: 'user', content: userMessage }
  ];

  console.log(`\n🤖 Agent started`);
  console.log(`📝 User: ${userMessage}\n`);

  let iterationCount = 0;

  // 主循环：loop until done
  while (true) {
    iterationCount++;
    console.log(`\n=== Iteration ${iterationCount} ===`);

    // 步骤 1：调用 LLM
    console.log(`\n💭 Calling LLM...`);

    const response = await client.messages.create({
      model: 'claude-opus-4',
      max_tokens: 4096,
      messages,
      tools
    });

    // 提取助手响应
    const assistantMessage = response.content
      .filter((block: any) => block.type === 'text')
      .map((block: any) => block.text)
      .join('\n');

    console.log(`\n🤖 Assistant: ${assistantMessage}`);

    // 将助手响应追加到上下文
    messages.push({
      role: 'assistant',
      content: response.content
    } as any);

    // 步骤 2：检测工具调用
    const toolCalls = response.content.filter(
      (block: any) => block.type === 'tool_use'
    );

    if (toolCalls.length === 0) {
      // 无工具调用 → 任务完成
      console.log(`\n✅ Task completed (no tool calls)`);
      console.log(`📊 Total iterations: ${iterationCount}`);
      break;
    }

    console.log(`\n🔍 Found ${toolCalls.length} tool call(s)`);

    // 步骤 3：执行工具
    const toolResults: any[] = [];

    for (const toolCall of toolCalls) {
      const result = await executeTool({
        id: toolCall.id,
        name: toolCall.name,
        input: toolCall.input
      });

      console.log(`   ✓ Result: ${result.output.substring(0, 100)}...`);

      // 构造工具结果消息
      toolResults.push({
        type: 'tool_result',
        tool_use_id: toolCall.id,
        content: result.output
      });
    }

    // 步骤 4：更新上下文
    messages.push({
      role: 'user',
      content: toolResults
    } as any);

    // 继续下一轮迭代
  }

  console.log(`\n🎉 Agent finished\n`);
}

// ===== 5. 主函数 =====

async function main() {
  // 检查 API key
  if (!process.env.ANTHROPIC_API_KEY) {
    console.error('Error: ANTHROPIC_API_KEY not set');
    process.exit(1);
  }

  // 运行 Agent
  try {
    await agentLoop('创建一个 hello.ts 文件，内容是 console.log("hello")');
  } catch (error) {
    console.error('Agent error:', error);
    process.exit(1);
  }
}

// 运行
main();
```

---

## 代码详解

### 1. 类型定义

```typescript
interface Message {
  role: 'user' | 'assistant';
  content: string;
}
```

**说明：**
- 简化的消息格式
- 实际 Anthropic API 的 content 可以是数组

---

### 2. 工具定义

```typescript
const tools = [
  {
    name: 'read_file',
    description: 'Read file contents',
    input_schema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'File path to read' }
      },
      required: ['path']
    }
  },
  // ...
];
```

**说明：**
- 使用 Anthropic 的工具格式（JSON Schema）
- 3 个基础工具：read_file, write_file, bash
- 简化版，实际 Pi-mono 有更完善的验证

---

### 3. 工具执行

```typescript
async function executeTool(toolCall: ToolCall): Promise<ToolResult> {
  try {
    switch (toolCall.name) {
      case 'read_file': {
        const content = await fs.readFile(toolCall.input.path, 'utf-8');
        return { output: `File content:\n${content}` };
      }
      // ...
    }
  } catch (error: any) {
    return { output: `Error: ${error.message}` };
  }
}
```

**关键点：**
- 所有错误都返回，不抛异常
- 返回 `{ output, details }` 结构
- 异步执行（async/await）

---

### 4. Agent 循环

```typescript
while (true) {
  // 1. 调用 LLM
  const response = await client.messages.create({ messages, tools });

  // 2. 检测工具调用
  const toolCalls = response.content.filter(block => block.type === 'tool_use');

  if (toolCalls.length === 0) {
    break;  // 任务完成
  }

  // 3. 执行工具
  for (const toolCall of toolCalls) {
    const result = await executeTool(toolCall);
    toolResults.push(result);
  }

  // 4. 更新上下文
  messages.push({ role: 'user', content: toolResults });
}
```

**关键点：**
- `while (true)`：无限循环，由 LLM 决定终止
- 无 max-steps 限制
- 工具结果追加到上下文

---

## 运行示例

### 准备环境

```bash
# 1. 安装依赖
npm install @anthropic-ai/sdk

# 2. 设置 API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. 运行代码
npx tsx basic-agent-loop.ts
```

---

### 预期输出

```
🤖 Agent started
📝 User: 创建一个 hello.ts 文件，内容是 console.log("hello")

=== Iteration 1 ===

💭 Calling LLM...

🤖 Assistant: 我来创建这个文件

🔍 Found 1 tool call(s)

🔧 Executing tool: write_file
   Input: {
  "path": "hello.ts",
  "content": "console.log(\"hello\")"
}
   ✓ Result: File written: hello.ts

=== Iteration 2 ===

💭 Calling LLM...

🤖 Assistant: 文件已创建完成

✅ Task completed (no tool calls)
📊 Total iterations: 2

🎉 Agent finished
```

---

## 关键洞察

### 1. Loop Until Done

```typescript
while (true) {
  const response = await llm.call();
  if (!response.toolCalls) break;  // LLM 决定停止
}
```

**洞察：**
- 循环由 LLM 控制，不是代码控制
- 无需 max-steps，信任模型

---

### 2. 错误即反馈

```typescript
try {
  const result = await executeTool(toolCall);
  return result;
} catch (error) {
  return { output: `Error: ${error.message}` };  // 不抛异常
}
```

**洞察：**
- 错误返回给 LLM，不中断循环
- LLM 看到错误后会自我纠正

---

### 3. 上下文累积

```typescript
messages.push({ role: 'assistant', content: response.content });
messages.push({ role: 'user', content: toolResults });
```

**洞察：**
- 每次迭代都追加消息
- 上下文累积是 Agent 的"记忆"

---

## 扩展练习

### 练习 1：添加日志

```typescript
// 在循环中添加详细日志
console.log(`Context size: ${messages.length} messages`);
console.log(`Tokens used: ${response.usage.input_tokens + response.usage.output_tokens}`);
```

---

### 练习 2：添加超时

```typescript
const MAX_ITERATIONS = 10;
let iterationCount = 0;

while (true) {
  iterationCount++;

  if (iterationCount > MAX_ITERATIONS) {
    console.warn('Max iterations reached');
    break;
  }

  // ...
}
```

---

### 练习 3：添加事件流

```typescript
import { EventEmitter } from 'events';

const emitter = new EventEmitter();

// 发出事件
emitter.emit('message', { content: assistantMessage });
emitter.emit('tool_call', { name: toolCall.name });
emitter.emit('tool_result', { output: result.output });

// 监听事件
emitter.on('message', (msg) => console.log('Message:', msg.content));
emitter.on('tool_call', (call) => console.log('Tool call:', call.name));
```

---

## 与 Pi-mono 的对比

| 特性 | 本示例 | Pi-mono |
|------|--------|---------|
| **工具数量** | 3 个 | 4 个（read/write/edit/bash） |
| **验证** | 无 | TypeBox + AJV |
| **状态持久化** | 无 | JSONL 追加日志 |
| **事件流** | 无 | EventEmitter |
| **Provider 抽象** | 直接用 Anthropic | 统一 Provider 接口 |
| **代码行数** | ~150 行 | ~1000 行（完整实现） |

**本示例的价值：**
- 展示核心机制（~150 行）
- 可运行、可理解
- 为理解 Pi-mono 打基础

---

## 常见问题

### Q1: 为什么用 `while (true)`？

**A:** 信任 LLM 能自主终止。前沿模型（Claude Opus 4）理解"任务完成"的语义，会在适当时候停止工具调用。

---

### Q2: 如果 LLM 真的无限循环怎么办？

**A:** 实际生产中可以添加监控（如 50 次迭代警告），但不强制终止。Pi-mono 数千次任务证明无限循环不会发生。

---

### Q3: 为什么错误不抛异常？

**A:** 错误是 LLM 的学习信号。返回错误给 LLM，它会自我纠正。抛异常会中断循环，失去自我纠正的机会。

---

### Q4: 上下文会不会太长？

**A:** 会。实际应用中需要 Compaction（压缩历史消息）。本示例为了简单省略了这部分。

---

## 总结

**本示例展示了 Agent Core 的核心机制：**

1. **Loop Until Done**：无限循环，LLM 决定终止
2. **工具执行**：检测 → 执行 → 返回结果
3. **上下文累积**：每次迭代追加消息
4. **错误处理**：返回错误给 LLM，不抛异常

**关键代码：**
```typescript
while (true) {
  const response = await llm.call(messages, tools);
  if (!response.toolCalls) break;

  for (const call of response.toolCalls) {
    const result = await executeTool(call);
    messages.push({ role: 'user', content: result.output });
  }
}
```

**下一步：** 学习工具注册与调用（实战代码 02）。
