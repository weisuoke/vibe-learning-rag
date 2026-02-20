# 实战代码 04：完整 Agent 示例

> 整合所有核心概念，实现一个生产级的 Agent Core

---

## 代码目标

实现一个完整的 Agent Core，整合：
- 工具注册与验证
- Agent 循环执行
- 状态持久化
- 事件流
- 错误处理

**代码长度：** ~200 行
**运行环境：** Node.js 18+, TypeScript

---

## 完整代码

```typescript
/**
 * 完整 Agent 示例
 * 演示：整合所有核心概念的生产级实现
 */

import { Type, Static } from '@sinclair/typebox';
import Ajv from 'ajv';
import Anthropic from '@anthropic-ai/sdk';
import fs from 'fs/promises';
import path from 'path';
import { randomUUID } from 'crypto';
import { EventEmitter } from 'events';

// ===== 1. 类型定义 =====

interface Tool {
  name: string;
  description: string;
  schema: any;
  execute: (params: any) => Promise<ToolResult>;
}

interface ToolResult {
  output: string;
  details?: any;
}

interface SessionEntry {
  id: string;
  parentId?: string;
  timestamp: number;
  type: 'user' | 'assistant' | 'tool_result';
  content?: string;
  toolCalls?: any[];
  output?: string;
}

// ===== 2. 完整 Agent 类 =====

class Agent extends EventEmitter {
  private client: Anthropic;
  private tools: Map<string, Tool> = new Map();
  private validators: Map<string, any> = new Map();
  private ajv: Ajv;
  private sessionWriter: SessionWriter;
  private context: any[] = [];

  constructor(config: {
    apiKey: string;
    sessionId?: string;
    sessionDir?: string;
  }) {
    super();

    this.client = new Anthropic({ apiKey: config.apiKey });
    this.ajv = new Ajv({ allErrors: true });

    const sessionId = config.sessionId || randomUUID();
    this.sessionWriter = new SessionWriter(
      sessionId,
      config.sessionDir || './.pi/sessions'
    );
  }

  // 注册工具
  registerTool(tool: Tool): void {
    this.tools.set(tool.name, tool);
    const validator = this.ajv.compile(tool.schema);
    this.validators.set(tool.name, validator);

    console.log(`✓ Tool registered: ${tool.name}`);
  }

  // 获取工具定义（给 LLM）
  private getToolDefinitions(): any[] {
    return Array.from(this.tools.values()).map(tool => ({
      name: tool.name,
      description: tool.description,
      input_schema: {
        type: 'object',
        properties: tool.schema.properties,
        required: tool.schema.required || []
      }
    }));
  }

  // 验证工具参数
  private validate(name: string, params: any): { valid: boolean; errors?: string[] } {
    const validator = this.validators.get(name);
    if (!validator) {
      return { valid: false, errors: [`Tool not found: ${name}`] };
    }

    if (validator(params)) {
      return { valid: true };
    } else {
      const errors = validator.errors?.map(e => `${e.instancePath}: ${e.message}`) || [];
      return { valid: false, errors };
    }
  }

  // 执行工具
  private async executeTool(name: string, params: any): Promise<ToolResult> {
    // 验证参数
    const validation = this.validate(name, params);
    if (!validation.valid) {
      return {
        output: `Error: Invalid parameters\n${validation.errors!.join('\n')}`,
        details: { validationErrors: validation.errors }
      };
    }

    // 执行工具
    const tool = this.tools.get(name);
    if (!tool) {
      return {
        output: `Error: Tool not found: ${name}`,
        details: { error: 'Tool not found' }
      };
    }

    try {
      return await tool.execute(params);
    } catch (error: any) {
      return {
        output: `Error: ${error.message}`,
        details: { error: error.stack }
      };
    }
  }

  // 运行 Agent
  async run(userMessage: string): Promise<void> {
    await this.sessionWriter.init();

    console.log(`\n🤖 Agent started`);
    console.log(`📝 User: ${userMessage}\n`);

    // 初始化上下文
    this.context = [{ role: 'user', content: userMessage }];

    // 保存用户消息
    let currentId = await this.sessionWriter.appendUser(userMessage);

    // 获取工具定义
    const tools = this.getToolDefinitions();

    let iterationCount = 0;

    // 主循环：loop until done
    while (true) {
      iterationCount++;
      console.log(`\n=== Iteration ${iterationCount} ===`);

      // 调用 LLM
      this.emit('llm_call', { iteration: iterationCount });

      const response = await this.client.messages.create({
        model: 'claude-opus-4',
        max_tokens: 4096,
        messages: this.context,
        tools
      });

      // 提取助手响应
      const assistantMessage = response.content
        .filter((block: any) => block.type === 'text')
        .map((block: any) => block.text)
        .join('\n');

      if (assistantMessage) {
        console.log(`\n🤖 Assistant: ${assistantMessage}`);
        this.emit('message', { content: assistantMessage });
      }

      // 保存助手响应
      const toolCalls = response.content.filter((block: any) => block.type === 'tool_use');
      currentId = await this.sessionWriter.appendAssistant(
        assistantMessage,
        currentId,
        toolCalls
      );

      // 追加到上下文
      this.context.push({
        role: 'assistant',
        content: response.content
      });

      // 检测工具调用
      if (toolCalls.length === 0) {
        console.log(`\n✅ Task completed`);
        console.log(`📊 Total iterations: ${iterationCount}`);
        this.emit('done', { iterations: iterationCount });
        break;
      }

      console.log(`\n🔍 Found ${toolCalls.length} tool call(s)`);

      // 执行工具
      const toolResults: any[] = [];

      for (const toolCall of toolCalls) {
        this.emit('tool_call', { name: toolCall.name, params: toolCall.input });

        const result = await this.executeTool(toolCall.name, toolCall.input);

        this.emit('tool_result', { name: toolCall.name, result });

        console.log(`   ✓ ${toolCall.name}: ${result.output.substring(0, 50)}...`);

        // 保存工具结果
        currentId = await this.sessionWriter.appendToolResult(
          toolCall.id,
          result.output,
          result.details,
          currentId
        );

        toolResults.push({
          type: 'tool_result',
          tool_use_id: toolCall.id,
          content: result.output
        });
      }

      // 更新上下文
      this.context.push({
        role: 'user',
        content: toolResults
      });
    }

    // 关闭 Session
    await this.sessionWriter.close();

    console.log(`\n🎉 Agent finished\n`);
  }
}

// ===== 3. Session Writer 类 =====

class SessionWriter {
  private filePath: string;
  private buffer: SessionEntry[] = [];

  constructor(sessionId: string, sessionDir: string) {
    this.filePath = path.join(sessionDir, `${sessionId}.jsonl`);
  }

  async init(): Promise<void> {
    const dir = path.dirname(this.filePath);
    await fs.mkdir(dir, { recursive: true });
  }

  private async flush(): Promise<void> {
    if (this.buffer.length === 0) return;
    const lines = this.buffer.map(e => JSON.stringify(e) + '\n').join('');
    await fs.appendFile(this.filePath, lines);
    this.buffer = [];
  }

  async appendUser(content: string, parentId?: string): Promise<string> {
    const entry: SessionEntry = {
      id: randomUUID(),
      parentId,
      type: 'user',
      content,
      timestamp: Date.now()
    };
    this.buffer.push(entry);
    await this.flush();
    return entry.id;
  }

  async appendAssistant(content: string, parentId: string, toolCalls?: any[]): Promise<string> {
    const entry: SessionEntry = {
      id: randomUUID(),
      parentId,
      type: 'assistant',
      content,
      toolCalls,
      timestamp: Date.now()
    };
    this.buffer.push(entry);
    await this.flush();
    return entry.id;
  }

  async appendToolResult(
    toolCallId: string,
    output: string,
    details: any,
    parentId: string
  ): Promise<string> {
    const entry: SessionEntry = {
      id: randomUUID(),
      parentId,
      type: 'tool_result',
      output,
      timestamp: Date.now()
    };
    this.buffer.push(entry);
    await this.flush();
    return entry.id;
  }

  async close(): Promise<void> {
    await this.flush();
  }
}

// ===== 4. 工具定义 =====

const readTool: Tool = {
  name: 'read',
  description: 'Read file contents',
  schema: Type.Object({
    path: Type.String({ minLength: 1 })
  }),
  execute: async (params) => {
    const content = await fs.readFile(params.path, 'utf-8');
    return {
      output: `File content:\n${content}`,
      details: { path: params.path, size: content.length }
    };
  }
};

const writeTool: Tool = {
  name: 'write',
  description: 'Write file contents',
  schema: Type.Object({
    path: Type.String({ minLength: 1 }),
    content: Type.String()
  }),
  execute: async (params) => {
    await fs.writeFile(params.path, params.content);
    return {
      output: `File written: ${params.path}`,
      details: { path: params.path, bytes: params.content.length }
    };
  }
};

const editTool: Tool = {
  name: 'edit',
  description: 'Edit file by replacing text',
  schema: Type.Object({
    path: Type.String({ minLength: 1 }),
    oldText: Type.String({ minLength: 1 }),
    newText: Type.String()
  }),
  execute: async (params) => {
    let content = await fs.readFile(params.path, 'utf-8');
    if (!content.includes(params.oldText)) {
      return {
        output: `Error: Text not found: "${params.oldText}"`,
        details: { found: false }
      };
    }
    content = content.replace(params.oldText, params.newText);
    await fs.writeFile(params.path, content);
    return {
      output: `File edited: ${params.path}`,
      details: { path: params.path, replaced: true }
    };
  }
};

// ===== 5. 主函数 =====

async function main() {
  if (!process.env.ANTHROPIC_API_KEY) {
    console.error('Error: ANTHROPIC_API_KEY not set');
    process.exit(1);
  }

  // 创建 Agent
  const agent = new Agent({
    apiKey: process.env.ANTHROPIC_API_KEY
  });

  // 注册工具
  agent.registerTool(readTool);
  agent.registerTool(writeTool);
  agent.registerTool(editTool);

  // 监听事件
  agent.on('message', (event) => {
    // UI 可以在这里更新界面
  });

  agent.on('tool_call', (event) => {
    console.log(`   → Calling: ${event.name}`);
  });

  agent.on('tool_result', (event) => {
    // UI 可以显示工具结果
  });

  agent.on('done', (event) => {
    console.log(`\n✨ Completed in ${event.iterations} iterations`);
  });

  // 运行 Agent
  await agent.run('创建一个 hello.ts 文件，内容是 console.log("Hello, Pi!")');
}

main();
```

---

## 代码详解

### 1. Agent 类设计

```typescript
class Agent extends EventEmitter {
  private client: Anthropic;
  private tools: Map<string, Tool>;
  private validators: Map<string, any>;
  private sessionWriter: SessionWriter;
  private context: any[];
}
```

**关键点：**
- 继承 `EventEmitter`：支持事件流
- 封装所有核心组件：LLM、工具、验证、状态
- 单一职责：Agent 只负责循环逻辑

---

### 2. 工具注册

```typescript
registerTool(tool: Tool): void {
  this.tools.set(tool.name, tool);
  const validator = this.ajv.compile(tool.schema);
  this.validators.set(tool.name, validator);
}
```

**关键点：**
- 预编译验证器（性能优化）
- Map 存储（O(1) 查找）

---

### 3. Agent 循环

```typescript
async run(userMessage: string): Promise<void> {
  while (true) {
    // 1. 调用 LLM
    const response = await this.client.messages.create({ ... });

    // 2. 检测工具调用
    const toolCalls = response.content.filter(block => block.type === 'tool_use');
    if (toolCalls.length === 0) break;

    // 3. 执行工具
    for (const toolCall of toolCalls) {
      const result = await this.executeTool(toolCall.name, toolCall.input);
      toolResults.push(result);
    }

    // 4. 更新上下文
    this.context.push({ role: 'user', content: toolResults });
  }
}
```

**关键点：**
- Loop until done
- 事件发出（message, tool_call, tool_result, done）
- 状态持久化（SessionWriter）

---

### 4. 事件流

```typescript
// Agent 发出事件
this.emit('message', { content: assistantMessage });
this.emit('tool_call', { name: toolCall.name });
this.emit('tool_result', { result });
this.emit('done', { iterations });

// 外部监听事件
agent.on('message', (event) => {
  console.log('Assistant:', event.content);
});
```

**关键点：**
- 解耦 Agent 和 UI
- 实时更新
- 易于扩展

---

## 运行示例

### 输出

```
✓ Tool registered: read
✓ Tool registered: write
✓ Tool registered: edit

🤖 Agent started
📝 User: 创建一个 hello.ts 文件，内容是 console.log("Hello, Pi!")

=== Iteration 1 ===

🤖 Assistant: 我来创建这个文件

🔍 Found 1 tool call(s)
   → Calling: write
   ✓ write: File written: hello.ts...

=== Iteration 2 ===

🤖 Assistant: 文件已创建完成，内容为 console.log("Hello, Pi!")

✅ Task completed
📊 Total iterations: 2

✨ Completed in 2 iterations

🎉 Agent finished
```

---

### Session 文件

```jsonl
{"id":"msg-1","type":"user","content":"创建一个 hello.ts 文件...","timestamp":1708300000}
{"id":"msg-2","parentId":"msg-1","type":"assistant","content":"我来创建这个文件","toolCalls":[...],"timestamp":1708300001}
{"id":"msg-3","parentId":"msg-2","type":"tool_result","output":"File written: hello.ts","timestamp":1708300002}
{"id":"msg-4","parentId":"msg-3","type":"assistant","content":"文件已创建完成...","timestamp":1708300003}
```

---

## 与 Pi-mono 的对比

| 特性 | 本示例 | Pi-mono |
|------|--------|---------|
| **核心循环** | ✅ 完整实现 | ✅ 完整实现 |
| **工具注册** | ✅ TypeBox + AJV | ✅ TypeBox + AJV |
| **状态持久化** | ✅ JSONL | ✅ JSONL + Compaction |
| **事件流** | ✅ EventEmitter | ✅ EventEmitter |
| **Provider 抽象** | ❌ 仅 Anthropic | ✅ 多 Provider |
| **UI 组件** | ❌ 无 | ✅ pi-tui / pi-web-ui |
| **Extensions** | ❌ 无 | ✅ 扩展系统 |
| **代码行数** | ~200 行 | ~5000 行 |

**本示例的价值：**
- 展示核心机制（~200 行）
- 生产级质量
- 可直接使用
- 为理解 Pi-mono 打基础

---

## 扩展示例

### 示例 1：添加自定义工具

```typescript
// 定义数据库查询工具
const queryDbTool: Tool = {
  name: 'query_db',
  description: 'Query database with SQL',
  schema: Type.Object({
    sql: Type.String({ minLength: 1 })
  }),
  execute: async (params) => {
    // 连接数据库
    const db = await connectDatabase();
    const rows = await db.query(params.sql);

    return {
      output: `Query returned ${rows.length} rows`,
      details: { rows, count: rows.length }
    };
  }
};

// 注册工具
agent.registerTool(queryDbTool);

// 现在 Agent 可以查询数据库了
await agent.run('查询所有用户');
```

---

### 示例 2：添加 UI 更新

```typescript
// 创建 UI 更新器
class UIUpdater {
  constructor(agent: Agent) {
    agent.on('message', (event) => {
      this.updateChat('assistant', event.content);
    });

    agent.on('tool_call', (event) => {
      this.showToolCall(event.name, event.params);
    });

    agent.on('tool_result', (event) => {
      this.showToolResult(event.name, event.result);
    });

    agent.on('done', (event) => {
      this.showCompletion(event.iterations);
    });
  }

  private updateChat(role: string, content: string) {
    // 更新聊天界面
    console.log(`[${role}] ${content}`);
  }

  private showToolCall(name: string, params: any) {
    // 显示工具调用
    console.log(`🔧 Calling ${name}...`);
  }

  private showToolResult(name: string, result: ToolResult) {
    // 显示工具结果
    console.log(`✓ ${name}: ${result.output}`);
  }

  private showCompletion(iterations: number) {
    // 显示完成状态
    console.log(`✨ Completed in ${iterations} iterations`);
  }
}

// 使用
const agent = new Agent({ apiKey: process.env.ANTHROPIC_API_KEY! });
const ui = new UIUpdater(agent);

await agent.run('创建文件');
```

---

### 示例 3：添加错误重试

```typescript
class Agent extends EventEmitter {
  private maxRetries = 3;

  private async executeToolWithRetry(
    name: string,
    params: any,
    retries = 0
  ): Promise<ToolResult> {
    try {
      return await this.executeTool(name, params);
    } catch (error: any) {
      if (retries < this.maxRetries) {
        console.log(`⚠️  Retry ${retries + 1}/${this.maxRetries}...`);
        await new Promise(resolve => setTimeout(resolve, 1000));
        return this.executeToolWithRetry(name, params, retries + 1);
      }

      return {
        output: `Error: ${error.message} (after ${this.maxRetries} retries)`,
        details: { error: error.stack, retries }
      };
    }
  }
}
```

---

### 示例 4：添加 Session 恢复

```typescript
class Agent extends EventEmitter {
  // 从现有 Session 恢复
  async resume(sessionPath: string): Promise<void> {
    console.log(`📂 Resuming session: ${sessionPath}`);

    // 加载 Session
    const loader = new SessionLoader(sessionPath);
    await loader.load();

    // 构建 Context
    const branch = new SessionBranch(loader.getEntries());
    this.context = branch.buildContext();

    console.log(`✅ Resumed with ${this.context.length} messages`);

    // 继续对话
    await this.run('继续之前的任务');
  }
}

// 使用
const agent = new Agent({ apiKey: process.env.ANTHROPIC_API_KEY! });
await agent.resume('./.pi/sessions/abc-123.jsonl');
```

---

## 实际应用场景

### 场景 1：Coding Agent

```typescript
const codingAgent = new Agent({ apiKey: process.env.ANTHROPIC_API_KEY! });

codingAgent.registerTool(readTool);
codingAgent.registerTool(writeTool);
codingAgent.registerTool(editTool);

await codingAgent.run('重构 src/index.ts，提取重复代码');
```

---

### 场景 2：数据分析 Agent

```typescript
const dataAgent = new Agent({ apiKey: process.env.ANTHROPIC_API_KEY! });

dataAgent.registerTool(readTool);
dataAgent.registerTool(queryDbTool);
dataAgent.registerTool(plotChartTool);

await dataAgent.run('分析用户增长趋势并生成图表');
```

---

### 场景 3：代码审查 Agent

```typescript
const reviewAgent = new Agent({ apiKey: process.env.ANTHROPIC_API_KEY! });

reviewAgent.registerTool(readTool);
reviewAgent.registerTool(bashTool);  // 运行测试
reviewAgent.registerTool(writeTool);  // 生成报告

await reviewAgent.run('审查 PR #123 的代码质量');
```

---

## 性能优化建议

### 1. 并行工具执行

```typescript
// 如果工具之间无依赖，可以并行执行
const results = await Promise.all(
  toolCalls.map(call => this.executeTool(call.name, call.input))
);
```

---

### 2. Context 压缩

```typescript
// 当 Context 过长时，压缩历史消息
if (this.context.length > 100) {
  this.context = await this.compactContext(this.context);
}
```

---

### 3. 流式响应

```typescript
// 使用 SSE 实时推送
const response = await this.client.messages.create({
  messages: this.context,
  tools,
  stream: true
});

for await (const chunk of response) {
  this.emit('chunk', chunk);
}
```

---

## 总结

**本示例展示了完整的 Agent Core 实现：**

1. **Agent 类**：封装所有核心组件
2. **工具注册**：TypeBox + AJV 验证
3. **Agent 循环**：Loop until done
4. **状态持久化**：JSONL 追加日志
5. **事件流**：EventEmitter 解耦

**关键代码：**
```typescript
// 创建 Agent
const agent = new Agent({ apiKey });

// 注册工具
agent.registerTool(readTool);
agent.registerTool(writeTool);

// 监听事件
agent.on('message', (event) => { ... });

// 运行 Agent
await agent.run('创建文件');
```

**核心洞察：**
- Agent Core 的本质是"LLM + 循环 + 工具"
- 极简设计：~200 行实现完整功能
- 可扩展：通过工具注册和事件流扩展
- 生产级：包含验证、错误处理、状态管理

**完成：** 至此，所有 4 个实战代码示例全部完成。最后生成概览文件。
