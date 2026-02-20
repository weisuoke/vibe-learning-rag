# 核心概念 01：Agent 执行循环初始化

> 深入理解 Agent Core 如何启动和初始化执行循环

---

## 概念定义

**Agent 执行循环初始化**是指创建 Agent 实例、配置 LLM Provider、注册工具、设置系统提示的过程，为后续的循环执行做好准备。

**一句话：** 初始化就是给 Agent 装上"大脑"（LLM）、"工具箱"（Tools）和"记忆"（Context）。

---

## 初始化的核心组件

### 1. Agent 类实例化

**最小实现：**

```typescript
import { Agent } from '@mariozechner/pi-agent-core';
import { AnthropicProvider } from '@mariozechner/pi-ai';

// 创建 Agent 实例
const agent = new Agent({
  // 1. LLM Provider（大脑）
  provider: new AnthropicProvider({
    apiKey: process.env.ANTHROPIC_API_KEY,
    model: 'claude-opus-4'
  }),

  // 2. 系统提示（指令）
  systemPrompt: 'You are a helpful coding assistant.',

  // 3. 工具注册（工具箱）
  tools: builtInTools,  // read, write, edit, bash

  // 4. 初始上下文（记忆）
  context: []
});
```

**关键点：**
- **Provider**：统一的 LLM 接口，支持 Anthropic、OpenAI、Google 等
- **System Prompt**：定义 Agent 的角色和行为
- **Tools**：可用的工具列表
- **Context**：初始对话历史（通常为空）

---

### 2. Model 选择策略

**Pi-mono 的 Model 选择哲学：**

```typescript
// 不同任务选择不同模型
const models = {
  // 复杂任务：使用 Opus（最强推理）
  complex: 'claude-opus-4',

  // 简单任务：使用 Sonnet（平衡性能）
  simple: 'claude-sonnet-4',

  // 快速响应：使用 Haiku（最快速度）
  fast: 'claude-haiku-4'
};

// 示例：根据任务复杂度选择
const agent = new Agent({
  provider: new AnthropicProvider({
    model: taskComplexity === 'high' ? models.complex : models.simple
  })
});
```

**选择建议：**
- **Opus**：代码重构、架构设计、复杂调试
- **Sonnet**：日常编码、文件操作、测试编写
- **Haiku**：简单查询、文档生成、快速响应

---

### 3. 工具注册表初始化

**Pi-mono 的 4 个核心工具：**

```typescript
import { Type } from '@sinclair/typebox';

// 1. Read 工具
const readTool = {
  name: 'read',
  description: 'Read file contents',
  schema: Type.Object({
    path: Type.String({ description: 'File path to read' }),
    lines: Type.Optional(Type.Object({
      start: Type.Number(),
      end: Type.Number()
    }))
  }),
  execute: async (params) => {
    const content = await fs.readFile(params.path, 'utf-8');
    return {
      output: `File content:\n${content}`,
      details: { path: params.path, size: content.length }
    };
  }
};

// 2. Write 工具
const writeTool = {
  name: 'write',
  description: 'Write or overwrite file',
  schema: Type.Object({
    path: Type.String(),
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

// 3. Edit 工具
const editTool = {
  name: 'edit',
  description: 'Edit file with search/replace',
  schema: Type.Object({
    path: Type.String(),
    oldText: Type.String(),
    newText: Type.String()
  }),
  execute: async (params) => {
    let content = await fs.readFile(params.path, 'utf-8');
    content = content.replace(params.oldText, params.newText);
    await fs.writeFile(params.path, content);
    return {
      output: `File edited: ${params.path}`,
      details: { path: params.path, replaced: true }
    };
  }
};

// 4. Bash 工具
const bashTool = {
  name: 'bash',
  description: 'Execute bash command',
  schema: Type.Object({
    command: Type.String()
  }),
  execute: async (params) => {
    const { stdout, stderr } = await exec(params.command);
    return {
      output: stdout || stderr,
      details: { command: params.command, exitCode: 0 }
    };
  }
};

// 注册到工具表
const toolRegistry = new Map([
  ['read', readTool],
  ['write', writeTool],
  ['edit', editTool],
  ['bash', bashTool]
]);
```

**工具注册表的作用：**
- **名称映射**：工具名 → 工具定义
- **Schema 验证**：TypeBox schema → AJV 验证器
- **执行函数**：参数 → 执行结果

---

### 4. 初始上下文设置

**Context 是 Agent 的"记忆"：**

```typescript
interface Message {
  role: 'user' | 'assistant' | 'tool';
  content: string;
  toolCalls?: ToolCall[];
  toolCallId?: string;
}

// 初始化时的上下文
const initialContext: Message[] = [
  // 可选：添加系统消息
  {
    role: 'assistant',
    content: 'I am ready to help you with coding tasks.'
  }
];

// 或者从 Session 恢复
const restoredContext = await loadSession('session-id');
```

**Context 的三种初始化方式：**

1. **空上下文**（新会话）
   ```typescript
   context: []
   ```

2. **预设上下文**（带示例）
   ```typescript
   context: [
     { role: 'user', content: 'Create a TypeScript file' },
     { role: 'assistant', content: 'I will create it for you' }
   ]
   ```

3. **恢复上下文**（继续会话）
   ```typescript
   context: await loadSessionContext('session-123.jsonl')
   ```

---

## Pi-mono 的极简初始化哲学

### 1. 无 max-steps 限制

**Pi-mono 不设置最大迭代次数：**

```typescript
// ❌ 其他框架
const agent = new Agent({
  maxIterations: 10,  // 限制最多 10 次循环
  timeout: 30000      // 30 秒超时
});

// ✅ Pi-mono
const agent = new Agent({
  // 无 maxIterations
  // 无 timeout
  // 信任模型自己知道何时停止
});
```

**为什么不需要？**
- 前沿模型理解"任务完成"的语义
- 模型会在任务完成时自然停止工具调用
- 人为限制反而破坏 Agent 的自主性

---

### 2. 无 Plan Mode

**Pi-mono 不预设执行计划：**

```typescript
// ❌ 其他框架：预设执行计划
const agent = new Agent({
  planMode: true,  // 先规划，再执行
  steps: [
    'Read file',
    'Analyze content',
    'Generate report'
  ]
});

// ✅ Pi-mono：动态决策
const agent = new Agent({
  // 无 planMode
  // LLM 自己决定每一步
});
```

**为什么不需要？**
- LLM 能根据当前状态动态决策
- 预设计划无法应对不确定性
- 动态决策更灵活

---

### 3. 仅 4 个工具

**Pi-mono 只提供 4 个核心工具：**

```typescript
// ✅ Pi-mono：极简工具集
const builtInTools = [
  'read',   // 读取文件
  'write',  // 写入文件
  'edit',   // 编辑文件
  'bash'    // 执行命令
];

// ❌ 其他框架：大量预置工具
const tools = [
  'read_file', 'write_file', 'edit_file', 'delete_file',
  'create_directory', 'list_directory', 'move_file', 'copy_file',
  'git_commit', 'git_push', 'npm_install', 'run_test',
  // ... 50+ 工具
];
```

**为什么 4 个够用？**
- 前沿模型理解编码上下文
- `bash` 可以执行任意命令（git、npm 等）
- 组合能力 > 工具数量

---

## 初始化流程详解

### 完整初始化代码

```typescript
import { Agent } from '@mariozechner/pi-agent-core';
import { AnthropicProvider } from '@mariozechner/pi-ai';
import { Type } from '@sinclair/typebox';
import Ajv from 'ajv';

// ===== 1. 创建 LLM Provider =====
const provider = new AnthropicProvider({
  apiKey: process.env.ANTHROPIC_API_KEY,
  model: 'claude-opus-4',
  // 可选配置
  temperature: 0.7,
  maxTokens: 4096
});

// ===== 2. 定义工具 Schema =====
const toolSchemas = {
  read: Type.Object({
    path: Type.String()
  }),
  write: Type.Object({
    path: Type.String(),
    content: Type.String()
  }),
  edit: Type.Object({
    path: Type.String(),
    oldText: Type.String(),
    newText: Type.String()
  }),
  bash: Type.Object({
    command: Type.String()
  })
};

// ===== 3. 创建 AJV 验证器 =====
const ajv = new Ajv();
const validators = new Map();
for (const [name, schema] of Object.entries(toolSchemas)) {
  validators.set(name, ajv.compile(schema));
}

// ===== 4. 注册工具执行函数 =====
const toolExecutors = new Map([
  ['read', async (params) => {
    const content = await fs.readFile(params.path, 'utf-8');
    return { output: content, details: { path: params.path } };
  }],
  ['write', async (params) => {
    await fs.writeFile(params.path, params.content);
    return { output: 'File written', details: { path: params.path } };
  }],
  ['edit', async (params) => {
    let content = await fs.readFile(params.path, 'utf-8');
    content = content.replace(params.oldText, params.newText);
    await fs.writeFile(params.path, content);
    return { output: 'File edited', details: { path: params.path } };
  }],
  ['bash', async (params) => {
    const { stdout } = await exec(params.command);
    return { output: stdout, details: { command: params.command } };
  }]
]);

// ===== 5. 创建 Agent 实例 =====
const agent = new Agent({
  provider,
  systemPrompt: 'You are a helpful coding assistant. Use the available tools to complete tasks.',
  tools: {
    schemas: toolSchemas,
    validators,
    executors: toolExecutors
  },
  context: []  // 空上下文，新会话
});

// ===== 6. 设置事件监听器 =====
agent.on('message', (msg) => {
  console.log('Assistant:', msg.content);
});

agent.on('tool_call', (call) => {
  console.log(`Calling ${call.name}...`);
});

agent.on('tool_result', (result) => {
  console.log('Result:', result.output);
});

// ===== 7. Agent 已就绪，可以开始执行 =====
console.log('Agent initialized and ready!');
```

---

## 初始化的关键决策

### 决策 1：选择 Provider

**支持的 Provider：**

```typescript
// Anthropic（推荐）
import { AnthropicProvider } from '@mariozechner/pi-ai';
const provider = new AnthropicProvider({ model: 'claude-opus-4' });

// OpenAI
import { OpenAIProvider } from '@mariozechner/pi-ai';
const provider = new OpenAIProvider({ model: 'gpt-4' });

// Google
import { GoogleProvider } from '@mariozechner/pi-ai';
const provider = new GoogleProvider({ model: 'gemini-pro' });

// GitHub Models
import { GitHubProvider } from '@mariozechner/pi-ai';
const provider = new GitHubProvider({ model: 'gpt-4' });
```

**选择建议：**
- **Anthropic**：最佳编码能力，支持 thinking trace
- **OpenAI**：广泛支持，生态丰富
- **Google**：多模态能力强
- **GitHub**：免费额度，适合测试

---

### 决策 2：System Prompt 设计

**好的 System Prompt 应该：**

```typescript
// ✅ 清晰、简洁、具体
const goodPrompt = `
You are a coding assistant. Use the available tools to:
- Read and write files
- Edit existing code
- Run commands

When the task is complete, respond without calling any tools.
`;

// ❌ 冗长、模糊、过度限制
const badPrompt = `
You are an advanced AI coding assistant with extensive knowledge...
You must always follow these 20 rules...
Never do X, Y, Z...
`;
```

**Pi-mono 的 System Prompt 原则：**
- **简洁**：只说必要的信息
- **明确**：清楚定义角色和工具
- **信任**：不过度限制模型行为

---

### 决策 3：是否恢复 Session

**三种初始化场景：**

```typescript
// 场景 1：新会话
const agent = new Agent({
  provider,
  context: []  // 空上下文
});

// 场景 2：继续会话
const context = await loadSession('session-123.jsonl');
const agent = new Agent({
  provider,
  context  // 恢复上下文
});

// 场景 3：分支会话
const context = await loadSessionBranch('session-123.jsonl', 'msg-5');
const agent = new Agent({
  provider,
  context  // 从分支点恢复
});
```

---

## 初始化的性能考虑

### 1. 懒加载工具

**只在需要时加载工具：**

```typescript
// ❌ 一次性加载所有工具
const tools = loadAllTools();  // 加载 50+ 工具

// ✅ 按需加载
const coreTools = loadCoreTools();  // 仅 4 个工具
agent.registerTool('custom', customTool);  // 按需添加
```

---

### 2. Provider 连接池

**复用 Provider 实例：**

```typescript
// ✅ 单例 Provider
const provider = new AnthropicProvider({ ... });

const agent1 = new Agent({ provider });
const agent2 = new Agent({ provider });  // 复用同一个 Provider
```

---

### 3. Schema 预编译

**预编译 TypeBox Schema：**

```typescript
// ✅ 初始化时编译
const ajv = new Ajv();
const validators = new Map();
for (const [name, schema] of Object.entries(toolSchemas)) {
  validators.set(name, ajv.compile(schema));  // 预编译
}

// 运行时直接使用
const validate = validators.get('read');
if (!validate(params)) {
  return { output: `Error: ${validate.errors}` };
}
```

---

## 实际应用场景

### 场景 1：Coding Agent

```typescript
const codingAgent = new Agent({
  provider: new AnthropicProvider({ model: 'claude-opus-4' }),
  systemPrompt: 'You are a coding assistant. Use read/write/edit/bash tools.',
  tools: builtInTools,
  context: []
});
```

---

### 场景 2：数据分析 Agent

```typescript
const dataAgent = new Agent({
  provider: new AnthropicProvider({ model: 'claude-sonnet-4' }),
  systemPrompt: 'You are a data analyst. Query databases and generate charts.',
  tools: [...builtInTools, queryDbTool, plotChartTool],
  context: []
});
```

---

### 场景 3：代码审查 Agent

```typescript
const reviewAgent = new Agent({
  provider: new AnthropicProvider({ model: 'claude-opus-4' }),
  systemPrompt: 'You are a code reviewer. Read code, run tests, generate reports.',
  tools: ['read', 'bash', 'write'],  // 仅 3 个工具
  context: []
});
```

---

## 总结

**Agent 执行循环初始化的核心要点：**

1. **Provider 选择**：根据任务选择合适的 LLM
2. **工具注册**：Pi-mono 仅 4 个核心工具
3. **System Prompt**：简洁、明确、信任模型
4. **Context 初始化**：新会话、恢复会话、分支会话
5. **极简哲学**：无 max-steps、无 plan mode、信任模型

**关键洞察：**
- 初始化越简单，Agent 越灵活
- 信任模型的自主性，不过度限制
- 4 个工具 + 组合能力 = 无限可能

**下一步：** 理解初始化后，Agent 如何进入循环迭代机制（核心概念 02）。
