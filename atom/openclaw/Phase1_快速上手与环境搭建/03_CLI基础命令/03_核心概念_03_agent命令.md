# 03_核心概念_03_agent命令

## 命令概述

`openclaw agent` 是 OpenClaw 的 AI 交互命令，用于直接与 Agent 系统对话。Agent 基于 pi-agent-core 构建，支持多种 AI 模型（Claude、GPT-4、Gemini 等），可以执行复杂任务、调用工具、管理上下文。

---

## 命令语法

```bash
openclaw agent [options]
```

---

## 主命令：Agent 交互

### 基础用法

```bash
# 发送消息给 Agent
openclaw agent --message "Ship checklist"

# 指定思考级别
openclaw agent --message "Analyze this code" --thinking high

# 不回传到通道（仅本地输出）
openclaw agent --message "Quick question" --deliver=false
```

### 常用选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--message <text>` | 发送给 Agent 的消息 | 必需 |
| `--thinking <level>` | 思考级别（low, medium, high） | medium |
| `--deliver` | 是否回传到通道 | true |
| `--session <id>` | 指定会话 ID | 当前会话 |
| `--model <name>` | 指定模型 | 配置的默认模型 |

### 思考级别说明

| 级别 | 说明 | 适用场景 |
|------|------|----------|
| `low` | 快速响应，简单推理 | 简单问答、快速查询 |
| `medium` | 平衡速度和质量 | 日常对话、一般任务 |
| `high` | 深度思考，复杂推理 | 复杂问题、代码分析、决策 |

### 示例

```bash
# 简单问答（低思考级别）
openclaw agent --message "What's the weather?" --thinking low

# 代码分析（高思考级别）
openclaw agent --message "Review this TypeScript code" --thinking high

# 本地测试（不回传）
openclaw agent --message "Test response" --deliver=false
```

---

## Agent 系统架构

### 核心组件

```
┌─────────────────────────────────────────┐
│           OpenClaw Gateway              │
├─────────────────────────────────────────┤
│  ┌───────────────────────────────────┐  │
│  │      Agent System                 │  │
│  │  ┌─────────────────────────────┐  │  │
│  │  │   pi-agent-core             │  │  │
│  │  │   - Context Management      │  │  │
│  │  │   - Tool Calling            │  │  │
│  │  │   - Memory System           │  │  │
│  │  └─────────────────────────────┘  │  │
│  │  ┌─────────────────────────────┐  │  │
│  │  │   pi-ai (Unified API)       │  │  │
│  │  │   - Anthropic (Claude)      │  │  │
│  │  │   - OpenAI (GPT-4)          │  │  │
│  │  │   - Google (Gemini)         │  │  │
│  │  └─────────────────────────────┘  │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### Agent 工作流程

```typescript
// Agent 处理流程
1. 接收消息 → 2. 加载上下文 → 3. 调用模型 → 4. 执行工具 → 5. 返回响应
```

---

## 使用场景

### 场景 1: 快速问答

```bash
# 简单查询
openclaw agent --message "List all active channels" --thinking low

# 配置查询
openclaw agent --message "Show current model config" --thinking low

# 状态检查
openclaw agent --message "Gateway status" --thinking low
```

**优势**：
- 快速响应（< 2 秒）
- 低成本（少量 tokens）
- 适合简单任务

---

### 场景 2: 代码分析与生成

```bash
# 代码审查
openclaw agent --message "Review this TypeScript function for security issues" --thinking high

# 代码生成
openclaw agent --message "Generate a REST API endpoint for user authentication" --thinking high

# 重构建议
openclaw agent --message "Suggest refactoring for this legacy code" --thinking high
```

**优势**：
- 深度分析
- 考虑边界情况
- 提供详细建议

---

### 场景 3: 任务规划与执行

```bash
# 项目规划
openclaw agent --message "Create a deployment checklist for production" --thinking high

# 故障排查
openclaw agent --message "Debug: Gateway not responding on port 18789" --thinking high

# 性能优化
openclaw agent --message "Analyze and optimize this database query" --thinking high
```

**优势**：
- 系统化思考
- 多步骤规划
- 考虑依赖关系

---

### 场景 4: 本地测试（不回传）

```bash
# 测试 Agent 响应
openclaw agent --message "Test message" --deliver=false

# 调试配置
openclaw agent --message "Check model availability" --deliver=false

# 快速实验
openclaw agent --message "Explain quantum computing" --deliver=false
```

**优势**：
- 不污染通道消息
- 快速测试
- 本地调试

---

## TypeScript 集成

### 示例 1: Agent 客户端封装

```typescript
// src/agent-client.ts
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export interface AgentOptions {
  message: string;
  thinking?: 'low' | 'medium' | 'high';
  deliver?: boolean;
  session?: string;
  model?: string;
}

export interface AgentResponse {
  content: string;
  model: string;
  tokens: {
    input: number;
    output: number;
  };
  thinkingTime: number;
}

export class AgentClient {
  /**
   * 发送消息给 Agent
   */
  async send(options: AgentOptions): Promise<AgentResponse> {
    const args = this.buildArgs(options);
    const command = `openclaw agent ${args}`;

    console.log(`Sending to Agent: ${options.message}`);
    const startTime = Date.now();

    try {
      const { stdout, stderr } = await execAsync(command);
      const thinkingTime = Date.now() - startTime;

      if (stderr) {
        console.warn('Agent warning:', stderr);
      }

      return this.parseResponse(stdout, thinkingTime);
    } catch (error) {
      throw new Error(`Agent error: ${error.message}`);
    }
  }

  /**
   * 快速问答（低思考级别）
   */
  async quickAsk(message: string): Promise<string> {
    const response = await this.send({
      message,
      thinking: 'low',
      deliver: false,
    });
    return response.content;
  }

  /**
   * 深度分析（高思考级别）
   */
  async deepAnalyze(message: string): Promise<string> {
    const response = await this.send({
      message,
      thinking: 'high',
      deliver: false,
    });
    return response.content;
  }

  /**
   * 构建命令参数
   */
  private buildArgs(options: AgentOptions): string {
    const args: string[] = [];

    args.push(`--message "${options.message}"`);

    if (options.thinking) {
      args.push(`--thinking ${options.thinking}`);
    }

    if (options.deliver === false) {
      args.push('--deliver=false');
    }

    if (options.session) {
      args.push(`--session ${options.session}`);
    }

    if (options.model) {
      args.push(`--model ${options.model}`);
    }

    return args.join(' ');
  }

  /**
   * 解析 Agent 响应
   */
  private parseResponse(stdout: string, thinkingTime: number): AgentResponse {
    // 简化解析逻辑（实际需要根据输出格式调整）
    return {
      content: stdout.trim(),
      model: 'claude-opus-4',
      tokens: {
        input: 0,
        output: 0,
      },
      thinkingTime,
    };
  }
}
```

### 示例 2: Agent 任务队列

```typescript
// src/agent-queue.ts
import { EventEmitter } from 'events';
import { AgentClient, AgentOptions } from './agent-client';

interface Task {
  id: string;
  options: AgentOptions;
  priority: number;
  createdAt: Date;
}

export class AgentQueue extends EventEmitter {
  private client = new AgentClient();
  private queue: Task[] = [];
  private processing = false;
  private concurrency = 1; // Agent 通常串行处理

  /**
   * 添加任务到队列
   */
  async enqueue(options: AgentOptions, priority = 0): Promise<string> {
    const task: Task = {
      id: this.generateId(),
      options,
      priority,
      createdAt: new Date(),
    };

    this.queue.push(task);
    this.queue.sort((a, b) => b.priority - a.priority);

    this.emit('task-added', task);

    if (!this.processing) {
      this.processQueue();
    }

    return task.id;
  }

  /**
   * 处理队列
   */
  private async processQueue(): Promise<void> {
    if (this.processing || this.queue.length === 0) {
      return;
    }

    this.processing = true;

    while (this.queue.length > 0) {
      const task = this.queue.shift()!;
      this.emit('task-started', task);

      try {
        const response = await this.client.send(task.options);
        this.emit('task-completed', { task, response });
      } catch (error) {
        this.emit('task-failed', { task, error });
      }
    }

    this.processing = false;
  }

  /**
   * 获取队列状态
   */
  getStatus() {
    return {
      queueLength: this.queue.length,
      processing: this.processing,
      tasks: this.queue.map(t => ({
        id: t.id,
        message: t.options.message,
        priority: t.priority,
      })),
    };
  }

  private generateId(): string {
    return `task-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}

// 使用示例
const queue = new AgentQueue();

queue.on('task-completed', ({ task, response }) => {
  console.log(`✅ Task ${task.id} completed`);
  console.log(`Response: ${response.content}`);
});

queue.on('task-failed', ({ task, error }) => {
  console.error(`❌ Task ${task.id} failed: ${error.message}`);
});

// 添加任务
await queue.enqueue({
  message: 'Analyze system logs',
  thinking: 'high',
}, 10); // 高优先级

await queue.enqueue({
  message: 'Generate report',
  thinking: 'medium',
}, 5); // 中优先级
```

### 示例 3: Agent 会话管理

```typescript
// src/agent-session.ts
import { AgentClient } from './agent-client';

export class AgentSession {
  private client = new AgentClient();
  private sessionId: string;
  private history: Array<{ role: 'user' | 'agent'; content: string }> = [];

  constructor(sessionId?: string) {
    this.sessionId = sessionId || this.generateSessionId();
  }

  /**
   * 发送消息（保持会话上下文）
   */
  async send(message: string, thinking: 'low' | 'medium' | 'high' = 'medium'): Promise<string> {
    // 添加到历史
    this.history.push({ role: 'user', content: message });

    // 发送到 Agent
    const response = await this.client.send({
      message,
      thinking,
      session: this.sessionId,
      deliver: false,
    });

    // 添加响应到历史
    this.history.push({ role: 'agent', content: response.content });

    return response.content;
  }

  /**
   * 获取会话历史
   */
  getHistory() {
    return this.history;
  }

  /**
   * 清空会话
   */
  clear() {
    this.history = [];
  }

  /**
   * 导出会话
   */
  export() {
    return {
      sessionId: this.sessionId,
      history: this.history,
      createdAt: new Date(),
    };
  }

  private generateSessionId(): string {
    return `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}

// 使用示例
const session = new AgentSession();

// 多轮对话
await session.send('What is OpenClaw?', 'low');
await session.send('How do I configure channels?', 'medium');
await session.send('Show me a code example', 'high');

// 查看历史
console.log(session.getHistory());
```

---

## 高级用法

### 1. 多模型切换

```bash
# 使用 Claude Opus 4
openclaw agent --message "Complex analysis" --model claude-opus-4

# 使用 GPT-4
openclaw agent --message "Quick summary" --model gpt-4

# 使用 Gemini
openclaw agent --message "Generate ideas" --model gemini-pro
```

### 2. 会话管理

```bash
# 创建新会话
openclaw agent --message "Start new project" --session project-alpha

# 继续会话
openclaw agent --message "Continue from last step" --session project-alpha

# 查看会话
openclaw sessions
```

### 3. 工具调用

Agent 可以自动调用工具（如果配置了 Skills）：

```bash
# Agent 会自动调用文件系统工具
openclaw agent --message "List files in workspace"

# Agent 会自动调用浏览器工具
openclaw agent --message "Take a screenshot of example.com"

# Agent 会自动调用代码执行工具
openclaw agent --message "Run this Python script"
```

---

## 常见问题

### Q1: Agent 响应很慢怎么办？

**原因**：
- 思考级别设置为 `high`
- 模型负载高
- 网络延迟

**解决方案**：
```bash
# 降低思考级别
openclaw agent --message "Quick question" --thinking low

# 切换到更快的模型
openclaw agent --message "Quick question" --model claude-haiku

# 检查网络连接
openclaw gateway health
```

---

### Q2: Agent 无法调用工具

**原因**：
- Skills 未安装
- 权限不足
- 工具配置错误

**解决方案**：
```bash
# 检查 Skills
openclaw skills list

# 安装 Skills
openclaw onboard --install-skills

# 检查配置
openclaw config get skills.enabled
```

---

### Q3: 如何限制 Agent 的 Token 使用？

**解决方案**：
```bash
# 配置 Token 限制
openclaw config set agent.maxTokens 4000

# 配置成本限制
openclaw config set agent.maxCostPerRequest 0.10

# 查看当前配置
openclaw config get agent
```

---

### Q4: Agent 响应不准确

**原因**：
- 思考级别太低
- 上下文不足
- 模型选择不当

**解决方案**：
```bash
# 提高思考级别
openclaw agent --message "Detailed analysis" --thinking high

# 提供更多上下文
openclaw agent --message "Based on previous conversation, analyze..."

# 切换到更强的模型
openclaw agent --message "Complex task" --model claude-opus-4
```

---

## 最佳实践

### 1. 根据任务选择思考级别

```bash
# ✅ 简单任务用 low
openclaw agent --message "What's 2+2?" --thinking low

# ✅ 复杂任务用 high
openclaw agent --message "Design a distributed system" --thinking high

# ❌ 简单任务用 high（浪费资源）
openclaw agent --message "Hello" --thinking high
```

### 2. 本地测试用 --deliver=false

```bash
# ✅ 测试时不回传
openclaw agent --message "Test" --deliver=false

# ❌ 测试时回传（污染通道）
openclaw agent --message "Test"
```

### 3. 使用会话保持上下文

```bash
# ✅ 使用会话
openclaw agent --message "Start task" --session task-1
openclaw agent --message "Continue" --session task-1

# ❌ 不使用会话（丢失上下文）
openclaw agent --message "Start task"
openclaw agent --message "Continue"
```

### 4. 监控 Token 使用

```typescript
// 记录 Token 使用
const response = await client.send({
  message: 'Analyze this',
  thinking: 'high',
});

console.log(`Tokens used: ${response.tokens.input + response.tokens.output}`);
```

---

## 性能优化

### 1. 批量处理

```typescript
// 使用队列批量处理
const queue = new AgentQueue();

const tasks = [
  'Task 1',
  'Task 2',
  'Task 3',
];

for (const task of tasks) {
  await queue.enqueue({ message: task, thinking: 'low' });
}
```

### 2. 缓存响应

```typescript
// 缓存常见问题的响应
const cache = new Map<string, string>();

async function cachedAsk(message: string): Promise<string> {
  if (cache.has(message)) {
    return cache.get(message)!;
  }

  const response = await client.quickAsk(message);
  cache.set(message, response);
  return response;
}
```

### 3. 并发限制

```typescript
// 限制并发请求
const semaphore = new Semaphore(3); // 最多 3 个并发

async function sendWithLimit(message: string) {
  await semaphore.acquire();
  try {
    return await client.send({ message, thinking: 'low' });
  } finally {
    semaphore.release();
  }
}
```

---

## 下一步

- 学习 **channels 命令** → `03_核心概念_04_channels_status命令.md`
- 学习 **config 命令** → `03_核心概念_05_config命令.md`
- 学习 **实战场景** → `07_实战代码_场景1_基础命令使用.md`

---

## 参考资料

- [OpenClaw GitHub](https://github.com/openclaw/openclaw)
- [Agent 系统文档](https://docs.openclaw.ai/agent)
- [pi-agent-core](https://github.com/mariozechner/pi-agent-core)
- [CLI 官方文档](https://docs.openclaw.ai/cli)
