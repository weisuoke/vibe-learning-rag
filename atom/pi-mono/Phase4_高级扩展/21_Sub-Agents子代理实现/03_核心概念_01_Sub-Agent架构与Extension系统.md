# 核心概念 01：Sub-Agent 架构与 Extension 系统

## 概述

**Sub-Agents 是通过 pi-mono 的 Extension 系统实现的工具，展示了 Extension 架构的强大扩展能力。**

本文档深入解析 Sub-Agent 的架构设计、与 Extension 系统的集成方式，以及如何通过 Extension API 实现多 Agent 协作。

---

## 1. Sub-Agent 在 Extension 架构中的位置

### 1.1 Extension 系统概览

**Extension 是 pi-mono 的核心扩展机制，允许开发者注册自定义工具。**

```typescript
// Extension 的基本结构
export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "tool-name",
    description: "Tool description",
    parameters: ParametersSchema,
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      // 工具实现
    }
  });
}
```

**关键组件**：
- **ExtensionAPI**: 提供 `registerTool()` 方法
- **Tool Definition**: 工具的元数据和实现
- **Execute Function**: 工具的执行逻辑
- **Context (ctx)**: 提供运行时上下文（cwd, ui, callTool 等）

### 1.2 Sub-Agent 作为 Extension

**Sub-Agent 本身是一个 Extension，注册了名为 `subagent` 的工具。**

```typescript
// sourcecode/pi-mono/packages/coding-agent/examples/extensions/subagent/index.ts
export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "subagent",
    label: "Subagent",
    description: "Delegate tasks to specialized subagents...",
    parameters: SubagentParams,
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      // Sub-Agent 实现
    }
  });
}
```

**架构层次**：
```
pi-mono 核心
  ↓
Extension 系统
  ↓
subagent Extension
  ↓
独立的 pi 进程（Sub-Agents）
```

---

## 2. SubagentParams Schema

### 2.1 参数定义

**SubagentParams 使用 TypeBox 定义，支持三种执行模式。**

```typescript
import { Type } from "@sinclair/typebox";

const SubagentParams = Type.Object({
  // Single Mode
  agent: Type.Optional(Type.String({
    description: "Name of the agent to invoke (for single mode)"
  })),
  task: Type.Optional(Type.String({
    description: "Task to delegate (for single mode)"
  })),

  // Parallel Mode
  tasks: Type.Optional(Type.Array(TaskItem, {
    description: "Array of {agent, task} for parallel execution"
  })),

  // Chain Mode
  chain: Type.Optional(Type.Array(ChainItem, {
    description: "Array of {agent, task} for sequential execution"
  })),

  // Configuration
  agentScope: Type.Optional(AgentScopeSchema),
  confirmProjectAgents: Type.Optional(Type.Boolean({
    description: "Prompt before running project-local agents. Default: true.",
    default: true
  })),
  cwd: Type.Optional(Type.String({
    description: "Working directory for the agent process (single mode)"
  }))
});
```

### 2.2 模式互斥验证

**工具执行时会验证只能使用一种模式。**

```typescript
const hasChain = (params.chain?.length ?? 0) > 0;
const hasTasks = (params.tasks?.length ?? 0) > 0;
const hasSingle = Boolean(params.agent && params.task);
const modeCount = Number(hasChain) + Number(hasTasks) + Number(hasSingle);

if (modeCount !== 1) {
  return {
    content: [{
      type: "text",
      text: "Invalid parameters. Provide exactly one mode."
    }]
  };
}
```

**设计理念**：明确的模式选择，避免歧义。

---

## 3. Extension API 集成

### 3.1 核心 API 方法

**Sub-Agent Extension 使用以下 ExtensionAPI 功能：**

| API 方法 | 用途 | 示例 |
|---------|------|------|
| `pi.registerTool()` | 注册 subagent 工具 | 工具注册 |
| `ctx.cwd` | 获取当前工作目录 | 传递给子进程 |
| `ctx.ui.confirm()` | 显示确认对话框 | 项目 agent 确认 |
| `ctx.hasUI` | 检查是否有 UI | 决定是否显示确认 |
| `signal` | 中断信号 | Ctrl+C 传播 |
| `onUpdate` | 流式更新回调 | 实时进度显示 |

### 3.2 工具调用流程

```typescript
// 1. 用户请求
"Use scout to find authentication code"

// 2. pi 识别并调用 subagent 工具
await ctx.callTool("subagent", {
  agent: "scout",
  task: "Find authentication code"
});

// 3. subagent Extension 执行
async execute(toolCallId, params, signal, onUpdate, ctx) {
  // 3.1 发现 agents
  const agents = discoverAgents(ctx.cwd, agentScope);

  // 3.2 安全检查（如果需要）
  if (needsConfirmation) {
    const ok = await ctx.ui.confirm(...);
    if (!ok) return { content: [{ type: "text", text: "Canceled" }] };
  }

  // 3.3 启动 Sub-Agent 进程
  const result = await runSingleAgent(...);

  // 3.4 返回结果
  return {
    content: [{ type: "text", text: result.output }],
    details: { mode: "single", results: [result] }
  };
}
```

---

## 4. 进程管理架构

### 4.1 进程启动

**Sub-Agent 通过 Node.js `child_process.spawn()` 启动独立的 `pi` 进程。**

```typescript
import { spawn } from "node:child_process";

const proc = spawn('pi', [
  '--mode', 'json',           // JSON 输出模式
  '--model', agent.model,     // 指定模型
  '--tools', agent.tools.join(','),  // 工具列表
  '--no-session',             // 不保存 session
  `Task: ${task}`             // 任务描述
], {
  cwd: cwd ?? ctx.cwd,        // 工作目录
  shell: false,               // 不使用 shell
  stdio: ['ignore', 'pipe', 'pipe']  // stdin 忽略，stdout/stderr 管道
});
```

**关键参数**：
- `--mode json`: 结构化输出，便于解析
- `--no-session`: 不持久化 session，避免污染
- `stdio: ['ignore', 'pipe', 'pipe']`: 捕获输出

### 4.2 进程通信

**通过 stdout 接收 JSON 事件流。**

```typescript
let buffer = "";

proc.stdout.on('data', (data) => {
  buffer += data.toString();
  const lines = buffer.split("\n");
  buffer = lines.pop() || "";

  for (const line of lines) {
    if (!line.trim()) continue;

    try {
      const event = JSON.parse(line);
      processEvent(event);
    } catch {
      // 忽略非 JSON 行
    }
  }
});
```

**事件类型**：
- `message_end`: Assistant 消息完成
- `tool_result_end`: 工具调用结果
- 其他事件（忽略）

### 4.3 进程生命周期

```typescript
// 1. 启动
const proc = spawn('pi', args);

// 2. 监听输出
proc.stdout.on('data', handleData);
proc.stderr.on('data', handleError);

// 3. 监听退出
proc.on('close', (code) => {
  resolve(code ?? 0);
});

// 4. 错误处理
proc.on('error', () => {
  resolve(1);
});

// 5. 中断支持
if (signal) {
  signal.addEventListener('abort', () => {
    proc.kill('SIGTERM');
    setTimeout(() => {
      if (!proc.killed) proc.kill('SIGKILL');
    }, 5000);
  }, { once: true });
}
```

---

## 5. 2025-2026 架构模式对比

### 5.1 业界主流模式

根据 2025-2026 最新研究，多 Agent 系统有以下主流架构模式：

[Source: Azure AI Agent Orchestration Patterns](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns)

| 模式 | 描述 | 适用场景 |
|------|------|---------|
| **Orchestrator-Worker** | 中央编排器协调多个工作者 | 企业系统、复杂任务分解 |
| **Hierarchical** | 多层代理层次结构 | 大规模系统、清晰职责链 |
| **Sequential Pipeline** | 代理顺序执行 | 线性工作流、步骤依赖 |
| **Concurrent/Parallel** | 代理并行执行 | 独立任务、提高效率 |
| **Event-Driven** | 事件驱动通信 | 松耦合、可扩展系统 |

[Source: The Orchestration of Multi-Agent Systems (2026 arXiv)](https://arxiv.org/html/2601.13671v1)

### 5.2 Pi-mono Sub-Agent 的模式定位

**Pi-mono Sub-Agent 实现了 Orchestrator-Worker + Sequential Pipeline + Concurrent 的混合模式。**

```typescript
// Orchestrator-Worker: 主 Agent 作为编排器
主 Agent (Orchestrator)
  ↓ 委托任务
Sub-Agent 1 (Worker: Scout)
Sub-Agent 2 (Worker: Planner)
Sub-Agent 3 (Worker: Worker)

// Sequential Pipeline: Chain 模式
Scout → Planner → Worker
  ↓       ↓        ↓
输出1 → 输出2 → 最终输出

// Concurrent: Parallel 模式
Scout 1 ┐
Scout 2 ├→ 并行执行
Scout 3 ┘
```

**优势**：
- ✅ 灵活性：支持多种编排模式
- ✅ 简单性：通过进程隔离实现，无需复杂的消息队列
- ✅ 可靠性：进程崩溃不影响主 Agent
- ✅ 可扩展性：易于添加新的 Agent 类型

---

## 6. 与其他 Extension 的对比

### 6.1 普通 Extension vs Sub-Agent Extension

| 维度 | 普通 Extension | Sub-Agent Extension |
|------|---------------|-------------------|
| **执行方式** | 同步/异步函数 | 启动独立进程 |
| **上下文** | 共享主 Agent 上下文 | 独立上下文 |
| **AI 能力** | 无（纯工具） | 有（完整 LLM） |
| **成本** | 无 | 有（LLM API 调用） |
| **复杂度** | 低 | 高 |
| **适用场景** | 简单工具 | 复杂任务委托 |

**示例对比**：

```typescript
// 普通 Extension: 简单的文件搜索工具
pi.registerTool({
  name: "search-files",
  async execute(toolCallId, params, signal, onUpdate, ctx) {
    const files = await glob(params.pattern);
    return { content: [{ type: "text", text: files.join("\n") }] };
  }
});

// Sub-Agent Extension: 智能代码分析
pi.registerTool({
  name: "subagent",
  async execute(toolCallId, params, signal, onUpdate, ctx) {
    // 启动独立的 AI Agent
    const proc = spawn('pi', ['--mode', 'json', ...]);
    // Sub-Agent 会自主决策、调用工具、分析结果
    return result;
  }
});
```

### 6.2 设计权衡

**何时使用普通 Extension**：
- ✅ 确定性逻辑（如文件操作、API 调用）
- ✅ 低延迟要求
- ✅ 成本敏感
- ✅ 简单任务

**何时使用 Sub-Agent**：
- ✅ 需要 AI 推理和决策
- ✅ 复杂的多步骤任务
- ✅ 需要上下文隔离
- ✅ 专业化分工

---

## 7. 架构优势与限制

### 7.1 架构优势

#### 1. 进程隔离带来的好处

```typescript
// 主 Agent 上下文
{
  contextTokens: 150000,  // 接近上限
  messages: [...200 条消息]
}

// Sub-Agent 上下文（独立）
{
  contextTokens: 5000,    // 全新开始
  messages: [任务描述]
}
```

**优势**：
- ✅ 突破上下文限制：每个 Sub-Agent 有完整的上下文窗口
- ✅ 避免污染：子任务的细节不占用主 Agent 上下文
- ✅ 故障隔离：Sub-Agent 崩溃不影响主 Agent

#### 2. Extension 系统的灵活性

```typescript
// 易于扩展：添加新的 Agent 类型
~/.pi/agent/agents/
├── scout.md       // 快速侦查
├── planner.md     // 计划生成
├── reviewer.md    // 代码审查
├── tester.md      // 测试执行（新增）
└── documenter.md  // 文档生成（新增）
```

**优势**：
- ✅ 热重载：`/reload` 即可加载新 Agent
- ✅ 配置驱动：通过 Markdown 文件定义
- ✅ 无需修改代码：Extension 代码保持不变

#### 3. 与 pi-mono 生态集成

**Sub-Agent 继承 pi-mono 的所有能力**：
- ✅ 工具系统：read, grep, find, bash 等
- ✅ Provider 支持：Anthropic, OpenAI, 自定义 Provider
- ✅ Session 管理：可选的 session 持久化
- ✅ UI 渲染：pi-tui 的终端 UI

### 7.2 架构限制

#### 1. 性能开销

```typescript
// 启动开销
进程启动时间：~500ms - 1s
模型加载时间：~200ms - 500ms
总开销：~700ms - 1.5s

// 对比：函数调用
函数调用时间：<1ms
```

**影响**：
- ⚠️ 短任务不适合：开销大于收益
- ⚠️ 频繁调用：累积开销显著

#### 2. 资源消耗

```typescript
// 每个 Sub-Agent 进程
内存：~50-100MB
CPU：取决于任务复杂度
API 调用：独立计费
```

**影响**：
- ⚠️ 并发限制：最多 4 个并发（避免资源耗尽）
- ⚠️ 成本：每个 Sub-Agent 都产生 LLM API 费用

#### 3. 通信限制

```typescript
// 只能通过文本传递信息
主 Agent → Sub-Agent: 任务描述（文本）
Sub-Agent → 主 Agent: 输出结果（文本）

// 无法传递：
// - 复杂对象
// - 二进制数据
// - 函数引用
```

**影响**：
- ⚠️ 信息丢失：工具调用细节不会传递
- ⚠️ 序列化开销：需要将结果格式化为文本

---

## 8. 最佳实践

### 8.1 Extension 设计

**1. 清晰的参数 Schema**

```typescript
// ✅ 好的设计：明确的参数定义
const SubagentParams = Type.Object({
  agent: Type.Optional(Type.String({
    description: "Name of the agent to invoke (for single mode)"
  })),
  // ... 详细的描述和类型
});

// ❌ 不好的设计：模糊的参数
const Params = Type.Object({
  data: Type.Any()  // 太宽泛
});
```

**2. 完善的错误处理**

```typescript
// ✅ 好的设计：详细的错误信息
if (!agent) {
  const available = agents.map(a => `"${a.name}"`).join(", ");
  return {
    content: [{
      type: "text",
      text: `Unknown agent: "${agentName}". Available: ${available}.`
    }]
  };
}

// ❌ 不好的设计：模糊的错误
return { content: [{ type: "text", text: "Error" }] };
```

**3. 流式更新支持**

```typescript
// ✅ 好的设计：实时进度反馈
const emitUpdate = () => {
  if (onUpdate) {
    onUpdate({
      content: [{ type: "text", text: currentOutput }],
      details: { progress: currentProgress }
    });
  }
};

// 在关键节点调用
emitUpdate();
```

### 8.2 Agent 定义

**1. 专业化分工**

```markdown
<!-- ✅ 好的设计：明确的职责 -->
---
name: scout
description: Fast codebase reconnaissance
tools: read, grep, find, ls
model: claude-haiku-4-5
---

You are a fast reconnaissance agent.
Focus on: quickly locating files, returning compressed context.
Do NOT: implement changes, write detailed analysis.
```

**2. 合适的工具集**

```markdown
<!-- ✅ Scout: 只读工具 -->
tools: read, grep, find, ls

<!-- ✅ Worker: 完整工具集 -->
tools: (all default tools)

<!-- ❌ 不好：Scout 有写权限 -->
tools: read, grep, find, ls, write, edit
```

### 8.3 使用模式

**1. 选择合适的模式**

```typescript
// ✅ 独立任务 → Parallel
{ tasks: [
  { agent: "scout", task: "Find frontend" },
  { agent: "scout", task: "Find backend" }
]}

// ✅ 依赖任务 → Chain
{ chain: [
  { agent: "scout", task: "Find code" },
  { agent: "planner", task: "Plan based on {previous}" }
]}

// ❌ 依赖任务用 Parallel（错误）
{ tasks: [
  { agent: "scout", task: "Find code" },
  { agent: "planner", task: "Plan based on code" }  // 无法获取 scout 的结果
]}
```

**2. 任务描述清晰**

```typescript
// ✅ 好的任务描述
{
  agent: "scout",
  task: "Find all authentication-related code in src/auth/. " +
        "Focus on login, logout, and token validation functions. " +
        "Return file paths and key function names."
}

// ❌ 不好的任务描述
{
  agent: "scout",
  task: "Find auth code"  // 太模糊
}
```

---

## 9. 未来演进方向

### 9.1 基于 2025-2026 趋势

根据最新研究，多 Agent 系统的演进方向：

[Source: Multi-Agent AI Orchestration: Enterprise Strategy for 2025-2026](https://www.onabout.ai/p/mastering-multi-agent-orchestration-architectures-patterns-roi-benchmarks-for-2025-2026)

**1. 统一编排层**

```typescript
// 未来可能的架构
class UnifiedOrchestrator {
  async orchestrate(plan: Plan) {
    // 统一的规划、策略、状态管理
    for (const step of plan.steps) {
      await this.executeStep(step);
    }
  }
}
```

**2. 事件驱动架构**

[Source: Four Design Patterns for Event-Driven, Multi-Agent Systems](https://www.confluent.io/blog/event-driven-multi-agent-systems/)

```typescript
// 未来可能的实现
eventBus.on('task-completed', (event) => {
  // 触发下一个 Agent
  nextAgent.execute(event.result);
});
```

**3. 更智能的 Agent 发现**

```typescript
// 未来可能的功能
const agent = await discoverBestAgent({
  task: "Refactor authentication",
  requirements: {
    tools: ["read", "write", "bash"],
    expertise: ["security", "refactoring"],
    cost: "low"
  }
});
```

### 9.2 Pi-mono 的潜在改进

**1. Agent 能力声明**

```markdown
---
name: scout
capabilities:
  - code-search
  - file-analysis
  - pattern-matching
performance:
  speed: fast
  cost: low
---
```

**2. 结果缓存**

```typescript
// 避免重复执行相同任务
const cached = await cache.get(taskHash);
if (cached) return cached;
```

**3. 更丰富的通信协议**

```typescript
// 支持结构化数据传递
interface AgentResult {
  text: string;
  structured: {
    files: string[];
    functions: FunctionInfo[];
    metrics: Metrics;
  };
}
```

---

## 10. 总结

### 核心要点

1. **Extension 架构**：Sub-Agent 是 Extension 系统的高级应用
2. **进程隔离**：通过独立进程实现上下文隔离
3. **灵活编排**：支持 Single、Parallel、Chain 三种模式
4. **业界对齐**：符合 2025-2026 主流多 Agent 架构模式
5. **可扩展性**：易于添加新的 Agent 类型和能力

### 关键洞察

- **Sub-Agent ≠ 函数调用**：是独立的 AI 进程，有完整的推理能力
- **Extension 是基础**：Sub-Agent 的所有能力都基于 Extension API
- **架构权衡**：性能开销换取隔离性和专业化
- **未来趋势**：向统一编排层和事件驱动架构演进

### 学习路径

1. ✅ 理解 Extension 系统的基本概念
2. ✅ 掌握 Sub-Agent 的三种执行模式
3. → 深入学习 Agent 发现与配置机制
4. → 探索进程隔离与上下文管理
5. → 实践代理间通信协议

---

**参考资源**：
- [Azure AI Agent Orchestration Patterns](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns)
- [The Orchestration of Multi-Agent Systems (2026 arXiv)](https://arxiv.org/html/2601.13671v1)
- [Google ADK Multi-Agent Patterns](https://developers.googleblog.com/developers-guide-to-multi-agent-patterns-in-adk/)
- [Pi-mono Extension Documentation](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent/docs/extensions.md)
