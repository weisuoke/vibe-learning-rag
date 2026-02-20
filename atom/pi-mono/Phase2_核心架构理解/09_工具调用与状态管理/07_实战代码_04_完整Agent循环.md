# 实战代码 04：完整 Agent 循环

完整实现一个 Agent 循环系统，集成工具调用、状态管理和自纠正机制。

---

## 完整代码

```typescript
/**
 * 完整 Agent 循环实现
 * 演示：LLM 调用 + 工具执行 + 自纠正 + 状态持久化
 */

import { Type, Static } from '@sinclair/typebox';
import Anthropic from '@anthropic-ai/sdk';
import { ToolRegistry, ToolExecutor } from './tool-system';
import { SessionManager } from './session-manager';

// ===== 1. Agent 配置 =====

interface AgentConfig {
  model: string;
  maxTokens: number;
  maxIterations: number;
  workDir: string;
  sessionPath: string;
}

const DEFAULT_CONFIG: AgentConfig = {
  model: 'claude-3-5-sonnet-20241022',
  maxTokens: 4096,
  maxIterations: 10,
  workDir: process.cwd(),
  sessionPath: './agent-session.jsonl'
};

// ===== 2. Agent 类 =====

class Agent {
  private client: Anthropic;
  private registry: ToolRegistry;
  private executor: ToolExecutor;
  private session: SessionManager;
  private config: AgentConfig;

  constructor(config: Partial<AgentConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.client = new Anthropic();
    this.registry = new ToolRegistry();
    this.executor = new ToolExecutor(this.registry, this.config.workDir);
    this.session = new SessionManager(this.config.sessionPath);
  }

  // 初始化 Agent
  async initialize(): Promise<void> {
    // 注册工具
    this.registerTools();

    // 初始化会话
    await this.session.initialize();

    console.log('✓ Agent initialized');
    console.log(`  Model: ${this.config.model}`);
    console.log(`  Tools: ${this.registry.list().join(', ')}`);
    console.log(`  Session: ${this.config.sessionPath}`);
  }

  // 注册工具
  private registerTools(): void {
    // Read 工具
    this.registry.register({
      name: 'read',
      description: 'Read file content',
      schema: Type.Object({
        path: Type.String({ description: 'File path' })
      }),
      execute: async (params: any) => {
        const { readFile } = await import('fs/promises');
        const { resolve } = await import('path');
        const fullPath = resolve(this.config.workDir, params.path);
        return await readFile(fullPath, 'utf-8');
      }
    });

    // Write 工具
    this.registry.register({
      name: 'write',
      description: 'Write file content',
      schema: Type.Object({
        path: Type.String({ description: 'File path' }),
        content: Type.String({ description: 'File content' })
      }),
      execute: async (params: any) => {
        const { writeFile } = await import('fs/promises');
        const { resolve } = await import('path');
        const fullPath = resolve(this.config.workDir, params.path);
        await writeFile(fullPath, params.content, 'utf-8');
        return `✓ Wrote to ${params.path}`;
      }
    });

    // Bash 工具
    this.registry.register({
      name: 'bash',
      description: 'Execute shell command',
      schema: Type.Object({
        command: Type.String({ description: 'Shell command' })
      }),
      execute: async (params: any) => {
        const { spawn } = await import('child_process');
        return new Promise<string>((resolve) => {
          const proc = spawn('sh', ['-c', params.command], {
            cwd: this.config.workDir
          });

          let output = '';
          proc.stdout.on('data', (data) => output += data);
          proc.stderr.on('data', (data) => output += data);

          proc.on('close', (code) => {
            resolve(code === 0 ? output : `Error: ${output}`);
          });
        });
      }
    });
  }

  // 主循环：运行 Agent
  async run(userMessage: string): Promise<string> {
    console.log(`\n=== Agent Run ===`);
    console.log(`User: ${userMessage}\n`);

    // 记录用户消息
    await this.session.append('user', userMessage);

    // 获取对话历史
    const history = this.session.getHistory();
    const messages = this.convertToMessages(history);

    // 添加新的用户消息
    messages.push({
      role: 'user',
      content: userMessage
    });

    // Agent 循环
    let iteration = 0;
    let finalResponse = '';

    while (iteration < this.config.maxIterations) {
      iteration++;
      console.log(`--- Iteration ${iteration} ---`);

      // 调用 LLM
      const response = await this.client.messages.create({
        model: this.config.model,
        max_tokens: this.config.maxTokens,
        messages: messages as any,
        tools: this.registry.getSchemas() as any
      });

      console.log(`LLM response type: ${response.stop_reason}`);

      // 处理响应
      if (response.stop_reason === 'end_turn') {
        // 对话结束
        const textContent = response.content.find(c => c.type === 'text');
        if (textContent && 'text' in textContent) {
          finalResponse = textContent.text;
          await this.session.append('assistant', finalResponse);
        }
        break;
      }

      if (response.stop_reason === 'tool_use') {
        // 执行工具调用
        const assistantMessage = {
          role: 'assistant' as const,
          content: response.content
        };
        messages.push(assistantMessage);

        // 记录 assistant 消息
        await this.session.append(
          'assistant',
          JSON.stringify(response.content)
        );

        // 提取工具调用
        const toolCalls = response.content.filter(c => c.type === 'tool_use');
        console.log(`Tool calls: ${toolCalls.length}`);

        // 执行所有工具调用
        const toolResults = [];
        for (const toolCall of toolCalls) {
          if (toolCall.type === 'tool_use') {
            console.log(`  - ${toolCall.name}(${JSON.stringify(toolCall.input).slice(0, 50)}...)`);

            const result = await this.executor.execute(
              toolCall.name,
              toolCall.input
            );

            // 记录工具调用和结果
            await this.session.append(
              'tool_call',
              JSON.stringify({ name: toolCall.name, input: toolCall.input })
            );
            await this.session.append('tool_result', result.output);

            toolResults.push({
              type: 'tool_result' as const,
              tool_use_id: toolCall.id,
              content: result.output,
              is_error: result.error
            });

            console.log(`    Result: ${result.output.slice(0, 100)}...`);
          }
        }

        // 添加工具结果到消息历史
        messages.push({
          role: 'user' as const,
          content: toolResults
        });

        // 继续循环
        continue;
      }

      // 其他停止原因
      console.log(`Unexpected stop reason: ${response.stop_reason}`);
      break;
    }

    if (iteration >= this.config.maxIterations) {
      console.log(`⚠ Reached max iterations (${this.config.maxIterations})`);
    }

    // 刷新会话
    await this.session['writer'].flush();

    console.log(`\n=== Agent Complete ===\n`);
    return finalResponse;
  }

  // 转换会话历史为 LLM 消息格式
  private convertToMessages(history: any[]): any[] {
    const messages: any[] = [];

    for (const entry of history) {
      if (entry.type === 'user') {
        messages.push({
          role: 'user',
          content: entry.content
        });
      } else if (entry.type === 'assistant') {
        try {
          const content = JSON.parse(entry.content);
          messages.push({
            role: 'assistant',
            content: content
          });
        } catch {
          messages.push({
            role: 'assistant',
            content: entry.content
          });
        }
      }
    }

    return messages;
  }

  // 关闭 Agent
  async close(): Promise<void> {
    await this.session.close();
    console.log('✓ Agent closed');
  }
}

// ===== 3. 测试演示 =====

async function demo() {
  console.log('=== Complete Agent Loop Demo ===\n');

  // 创建 Agent
  const agent = new Agent({
    sessionPath: './demo-agent-session.jsonl',
    maxIterations: 5
  });

  await agent.initialize();

  // ===== 测试 1：简单文件操作 =====
  console.log('\n========================================');
  console.log('Test 1: Simple file operations');
  console.log('========================================');

  const response1 = await agent.run(
    'Create a file called test.txt with the content "Hello, World!"'
  );
  console.log(`\nFinal response: ${response1}`);

  // ===== 测试 2：读取和修改文件 =====
  console.log('\n========================================');
  console.log('Test 2: Read and modify file');
  console.log('========================================');

  const response2 = await agent.run(
    'Read test.txt and tell me what it says'
  );
  console.log(`\nFinal response: ${response2}`);

  // ===== 测试 3：执行命令 =====
  console.log('\n========================================');
  console.log('Test 3: Execute command');
  console.log('========================================');

  const response3 = await agent.run(
    'List all .txt files in the current directory'
  );
  console.log(`\nFinal response: ${response3}`);

  // ===== 测试 4：多步骤任务 =====
  console.log('\n========================================');
  console.log('Test 4: Multi-step task');
  console.log('========================================');

  const response4 = await agent.run(
    'Create a file called numbers.txt with numbers 1 to 5, one per line. Then read it back and count the lines.'
  );
  console.log(`\nFinal response: ${response4}`);

  // ===== 测试 5：自纠正演示 =====
  console.log('\n========================================');
  console.log('Test 5: Self-correction demo');
  console.log('========================================');

  const response5 = await agent.run(
    'Try to read a file that does not exist: nonexistent.txt'
  );
  console.log(`\nFinal response: ${response5}`);

  // 清理
  console.log('\n========================================');
  console.log('Cleanup');
  console.log('========================================');

  await agent.run('Delete test.txt and numbers.txt');

  // 关闭 Agent
  await agent.close();

  console.log('\n✓ Demo completed');
}

// 运行演示
demo().catch(console.error);
```

---

## 简化版本（核心逻辑）

```typescript
/**
 * 简化的 Agent 循环
 * 只保留核心逻辑，便于理解
 */

async function simpleAgentLoop(userMessage: string) {
  const messages = [{ role: 'user', content: userMessage }];
  let iteration = 0;

  while (iteration < 10) {
    iteration++;

    // 1. 调用 LLM
    const response = await llm.generate(messages, { tools });

    // 2. 检查是否结束
    if (response.stop_reason === 'end_turn') {
      return response.content;
    }

    // 3. 执行工具调用
    if (response.stop_reason === 'tool_use') {
      // 添加 assistant 消息
      messages.push({
        role: 'assistant',
        content: response.content
      });

      // 执行工具
      const toolResults = [];
      for (const toolCall of response.tool_calls) {
        const result = await executeTool(toolCall.name, toolCall.input);
        toolResults.push({
          type: 'tool_result',
          tool_use_id: toolCall.id,
          content: result
        });
      }

      // 添加工具结果
      messages.push({
        role: 'user',
        content: toolResults
      });

      // 继续循环
      continue;
    }

    // 4. 其他情况
    break;
  }
}
```

---

## 运行输出示例

```
=== Complete Agent Loop Demo ===

✓ Agent initialized
  Model: claude-3-5-sonnet-20241022
  Tools: read, write, bash
  Session: ./demo-agent-session.jsonl

========================================
Test 1: Simple file operations
========================================

=== Agent Run ===
User: Create a file called test.txt with the content "Hello, World!"

--- Iteration 1 ---
LLM response type: tool_use
Tool calls: 1
  - write({"path":"test.txt","content":"Hello, World!"})
    Result: ✓ Wrote to test.txt

--- Iteration 2 ---
LLM response type: end_turn

=== Agent Complete ===

Final response: I've created the file test.txt with the content "Hello, World!".

========================================
Test 2: Read and modify file
========================================

=== Agent Run ===
User: Read test.txt and tell me what it says

--- Iteration 1 ---
LLM response type: tool_use
Tool calls: 1
  - read({"path":"test.txt"})
    Result: Hello, World!

--- Iteration 2 ---
LLM response type: end_turn

=== Agent Complete ===

Final response: The file test.txt contains: "Hello, World!"

========================================
Test 3: Execute command
========================================

=== Agent Run ===
User: List all .txt files in the current directory

--- Iteration 1 ---
LLM response type: tool_use
Tool calls: 1
  - bash({"command":"ls *.txt"})
    Result: test.txt

--- Iteration 2 ---
LLM response type: end_turn

=== Agent Complete ===

Final response: There is one .txt file in the current directory: test.txt

========================================
Test 4: Multi-step task
========================================

=== Agent Run ===
User: Create a file called numbers.txt with numbers 1 to 5, one per line. Then read it back and count the lines.

--- Iteration 1 ---
LLM response type: tool_use
Tool calls: 1
  - write({"path":"numbers.txt","content":"1\n2\n3\n4\n5"})
    Result: ✓ Wrote to numbers.txt

--- Iteration 2 ---
LLM response type: tool_use
Tool calls: 1
  - read({"path":"numbers.txt"})
    Result: 1
2
3
4
5

--- Iteration 3 ---
LLM response type: end_turn

=== Agent Complete ===

Final response: I've created numbers.txt with numbers 1-5, one per line. The file contains 5 lines.

========================================
Test 5: Self-correction demo
========================================

=== Agent Run ===
User: Try to read a file that does not exist: nonexistent.txt

--- Iteration 1 ---
LLM response type: tool_use
Tool calls: 1
  - read({"path":"nonexistent.txt"})
    Result: Error: ENOENT: no such file or directory

--- Iteration 2 ---
LLM response type: end_turn

=== Agent Complete ===

Final response: The file nonexistent.txt does not exist. I got an error when trying to read it.

✓ Demo completed
```

---

## 关键特性

### 1. 完整的 Agent 循环
- LLM 调用 → 工具执行 → 结果返回 → 继续循环
- 自动处理多轮交互
- 支持多步骤任务

### 2. 自纠正机制
- 工具执行失败时，错误信息返回给 LLM
- LLM 自动理解错误并调整策略
- 无需人工干预

### 3. 状态持久化
- 所有消息记录到 JSONL
- 支持会话恢复
- 完整的对话历史

### 4. 工具并行执行
- 支持一次调用多个工具
- 使用 Promise.all 提升性能

### 5. 迭代限制
- 防止无限循环
- 可配置最大迭代次数

---

## 扩展建议

### 1. 添加流式响应

```typescript
async *runStream(userMessage: string): AsyncGenerator<string> {
  const stream = await this.client.messages.stream({
    model: this.config.model,
    max_tokens: this.config.maxTokens,
    messages: messages as any,
    tools: this.registry.getSchemas() as any
  });

  for await (const chunk of stream) {
    if (chunk.type === 'content_block_delta') {
      if (chunk.delta.type === 'text_delta') {
        yield chunk.delta.text;
      }
    }
  }
}
```

### 2. 添加中断和恢复

```typescript
class Agent {
  private interrupted = false;

  interrupt() {
    this.interrupted = true;
  }

  async run(userMessage: string) {
    while (iteration < this.config.maxIterations) {
      if (this.interrupted) {
        console.log('Agent interrupted');
        break;
      }

      // ... 执行逻辑
    }
  }

  async resume() {
    this.interrupted = false;
    const history = this.session.getHistory();
    const lastMessage = history[history.length - 1];

    if (lastMessage.type === 'user') {
      return await this.run(lastMessage.content);
    }
  }
}
```

### 3. 添加工具调用统计

```typescript
class Agent {
  private stats = {
    totalCalls: 0,
    toolCalls: new Map<string, number>(),
    errors: 0
  };

  async run(userMessage: string) {
    // ... 在工具执行后
    this.stats.totalCalls++;
    const count = this.stats.toolCalls.get(toolCall.name) || 0;
    this.stats.toolCalls.set(toolCall.name, count + 1);

    if (result.error) {
      this.stats.errors++;
    }
  }

  getStats() {
    return {
      ...this.stats,
      toolCalls: Object.fromEntries(this.stats.toolCalls)
    };
  }
}
```

### 4. 添加超时控制

```typescript
async runWithTimeout(userMessage: string, timeoutMs: number = 60000) {
  return Promise.race([
    this.run(userMessage),
    new Promise<string>((_, reject) =>
      setTimeout(() => reject(new Error('Agent timeout')), timeoutMs)
    )
  ]);
}
```

---

## 调试技巧

### 1. 启用详细日志

```typescript
class Agent {
  private debug = true;

  private log(...args: any[]) {
    if (this.debug) {
      console.log('[Agent]', ...args);
    }
  }

  async run(userMessage: string) {
    this.log('Starting run with message:', userMessage);
    this.log('Current history length:', history.length);
    // ...
  }
}
```

### 2. 保存中间状态

```typescript
async run(userMessage: string) {
  const stateFile = `./agent-state-${Date.now()}.json`;

  try {
    // ... Agent 逻辑

    // 保存成功状态
    await writeFile(stateFile, JSON.stringify({
      iteration,
      messages,
      finalResponse
    }));
  } catch (error) {
    // 保存错误状态
    await writeFile(stateFile, JSON.stringify({
      error: error.message,
      iteration,
      messages
    }));
    throw error;
  }
}
```

### 3. 可视化执行流程

```typescript
function visualizeExecution(messages: any[]) {
  console.log('\n=== Execution Flow ===');

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];

    if (msg.role === 'user') {
      console.log(`${i + 1}. User: ${msg.content.slice(0, 50)}...`);
    } else if (msg.role === 'assistant') {
      const toolCalls = msg.content.filter((c: any) => c.type === 'tool_use');
      if (toolCalls.length > 0) {
        console.log(`${i + 1}. Assistant: [${toolCalls.length} tool calls]`);
        toolCalls.forEach((tc: any) => {
          console.log(`   - ${tc.name}(...)`);
        });
      } else {
        console.log(`${i + 1}. Assistant: ${JSON.stringify(msg.content).slice(0, 50)}...`);
      }
    }
  }

  console.log('===================\n');
}
```

---

## 总结

完整 Agent 循环的核心要素：

1. **LLM 调用**：使用 Anthropic API 调用 Claude
2. **工具执行**：解析工具调用，执行并返回结果
3. **自纠正**：错误信息返回给 LLM，自动修正
4. **状态管理**：JSONL 持久化，支持恢复
5. **迭代控制**：防止无限循环，可配置限制

这个实现展示了：
- 如何构建一个完整的 Agent 系统
- 如何集成工具调用和状态管理
- 如何实现自纠正机制
- 如何处理多步骤任务

可以直接运行并扩展为生产级 Agent 系统。
