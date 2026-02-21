# 实战代码 11：多 Server 协作

> **编排多个 MCP 服务器，实现复杂的工作流和协作**

---

## 概述

多服务器协作是构建复杂 AI 工作流的关键。本文实现多个 MCP 服务器之间的编排、上下文共享和冲突解决。

```
多服务器协作核心：
├─ 工具编排 → 顺序执行 + 并行执行
├─ 上下文共享 → 数据传递 + 状态管理
├─ 冲突解决 → 优先级 + 命名空间
└─ 工作流 → 端到端场景
```

**本质**：多服务器协作是将独立的 MCP 服务器组合成协同工作的系统，通过标准化的编排模式实现复杂的业务逻辑。

---

## 工具编排

### 顺序执行模式

```typescript
import type { ExtensionAPI } from '@mariozechner/pi-coding-agent';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { Type } from '@sinclair/typebox';

export default function (pi: ExtensionAPI) {
  const clients = new Map<string, Client>();

  pi.on('session_start', async (_event, ctx) => {
    // 初始化多个服务器
    await initializeServers(clients, ctx);

    // 注册编排工具
    pi.registerTool({
      name: 'analyze_and_fix_code',
      label: 'Analyze and Fix Code',
      description: 'Read code from filesystem, analyze with GitHub, fix issues',
      parameters: Type.Object({
        file_path: Type.String({ description: 'File path to analyze' }),
      }),

      async execute(toolCallId, params, signal, onUpdate, ctx) {
        try {
          // 步骤 1: 从 Filesystem 读取文件
          onUpdate({ type: 'text', text: 'Step 1: Reading file...' });
          const fsClient = clients.get('filesystem')!;
          const fileResult = await fsClient.callTool({
            name: 'read_file',
            arguments: { path: params.file_path },
          });

          const fileContent = fileResult.content[0].text;

          // 步骤 2: 使用 GitHub 搜索相似代码
          onUpdate({ type: 'text', text: 'Step 2: Searching similar code...' });
          const githubClient = clients.get('github')!;
          const searchResult = await githubClient.callTool({
            name: 'search_code',
            arguments: { query: `language:typescript ${params.file_path}` },
          });

          // 步骤 3: 分析并生成修复建议
          onUpdate({ type: 'text', text: 'Step 3: Analyzing code...' });
          const analysis = analyzeCode(fileContent, searchResult);

          // 步骤 4: 写回修复后的代码
          if (analysis.needsFix) {
            onUpdate({ type: 'text', text: 'Step 4: Writing fixed code...' });
            await fsClient.callTool({
              name: 'write_file',
              arguments: {
                path: params.file_path,
                content: analysis.fixedCode,
              },
            });
          }

          return {
            content: [{
              type: 'text',
              text: `Analysis complete. ${analysis.needsFix ? 'Code fixed.' : 'No issues found.'}`,
            }],
            details: {
              steps: ['read', 'search', 'analyze', 'write'],
              analysis,
            },
          };
        } catch (error) {
          return {
            content: [{
              type: 'text',
              text: `Orchestration error: ${error}`,
            }],
            isError: true,
          };
        }
      },
    });
  });
}

function analyzeCode(content: string, searchResult: any): any {
  // 分析逻辑
  return {
    needsFix: false,
    fixedCode: content,
    issues: [],
  };
}

async function initializeServers(
  clients: Map<string, Client>,
  ctx: any
): Promise<void> {
  const servers = [
    { id: 'filesystem', command: 'npx', args: ['-y', '@modelcontextprotocol/server-filesystem', '/projects'] },
    { id: 'github', command: 'npx', args: ['-y', '@modelcontextprotocol/server-github'], env: { GITHUB_TOKEN: process.env.GITHUB_TOKEN } },
  ];

  for (const config of servers) {
    const client = new Client({ name: `${config.id}-client`, version: '1.0.0' });
    const transport = new StdioClientTransport({
      command: config.command,
      args: config.args,
      env: config.env,
    });
    await client.connect(transport);
    clients.set(config.id, client);
  }
}
```

### 并行执行模式

```typescript
pi.registerTool({
  name: 'multi_source_search',
  label: 'Multi-Source Search',
  description: 'Search across GitHub, filesystem, and database in parallel',
  parameters: Type.Object({
    query: Type.String({ description: 'Search query' }),
  }),

  async execute(toolCallId, params, signal, onUpdate, ctx) {
    try {
      // 并行执行多个搜索
      const [githubResults, filesystemResults, dbResults] = await Promise.all([
        clients.get('github')!.callTool({
          name: 'search_code',
          arguments: { query: params.query },
        }),
        clients.get('filesystem')!.callTool({
          name: 'search_files',
          arguments: { pattern: params.query },
        }),
        clients.get('postgres')!.callTool({
          name: 'query',
          arguments: { sql: `SELECT * FROM docs WHERE content LIKE '%${params.query}%'` },
        }),
      ]);

      // 合并结果
      const combinedResults = {
        github: githubResults.content,
        filesystem: filesystemResults.content,
        database: dbResults.content,
      };

      return {
        content: [{
          type: 'text',
          text: JSON.stringify(combinedResults, null, 2),
        }],
        details: {
          sources: ['github', 'filesystem', 'database'],
          totalResults: combinedResults.github.length +
                       combinedResults.filesystem.length +
                       combinedResults.database.length,
        },
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Parallel search error: ${error}`,
        }],
        isError: true,
      };
    }
  },
});
```

---

## 上下文共享

### 数据传递模式

```typescript
export class ContextManager {
  private context = new Map<string, any>();

  /**
   * 设置上下文
   */
  set(key: string, value: any): void {
    this.context.set(key, value);
  }

  /**
   * 获取上下文
   */
  get(key: string): any {
    return this.context.get(key);
  }

  /**
   * 清除上下文
   */
  clear(): void {
    this.context.clear();
  }
}

// 使用示例
const contextManager = new ContextManager();

pi.registerTool({
  name: 'workflow_step1',
  label: 'Workflow Step 1',
  description: 'Read file and store in context',
  parameters: Type.Object({
    file_path: Type.String(),
  }),

  async execute(toolCallId, params, signal, onUpdate, ctx) {
    const fsClient = clients.get('filesystem')!;
    const result = await fsClient.callTool({
      name: 'read_file',
      arguments: { path: params.file_path },
    });

    // 存储到上下文
    contextManager.set('file_content', result.content[0].text);
    contextManager.set('file_path', params.file_path);

    return {
      content: [{
        type: 'text',
        text: 'File content stored in context',
      }],
    };
  },
});

pi.registerTool({
  name: 'workflow_step2',
  label: 'Workflow Step 2',
  description: 'Process file content from context',
  parameters: Type.Object({}),

  async execute(toolCallId, params, signal, onUpdate, ctx) {
    // 从上下文获取数据
    const fileContent = contextManager.get('file_content');
    const filePath = contextManager.get('file_path');

    if (!fileContent) {
      return {
        content: [{
          type: 'text',
          text: 'No file content in context. Run workflow_step1 first.',
        }],
        isError: true,
      };
    }

    // 处理数据
    const processed = processContent(fileContent);

    return {
      content: [{
        type: 'text',
        text: `Processed ${filePath}: ${processed.length} lines`,
      }],
    };
  },
});

function processContent(content: string): string[] {
  return content.split('\n');
}
```

### 状态管理模式

```typescript
export class WorkflowState {
  private state: {
    currentStep: number;
    steps: string[];
    results: Map<string, any>;
    errors: Error[];
  };

  constructor() {
    this.state = {
      currentStep: 0,
      steps: [],
      results: new Map(),
      errors: [],
    };
  }

  /**
   * 添加步骤
   */
  addStep(stepName: string): void {
    this.state.steps.push(stepName);
  }

  /**
   * 记录结果
   */
  recordResult(stepName: string, result: any): void {
    this.state.results.set(stepName, result);
    this.state.currentStep++;
  }

  /**
   * 记录错误
   */
  recordError(error: Error): void {
    this.state.errors.push(error);
  }

  /**
   * 获取状态
   */
  getState(): any {
    return {
      ...this.state,
      progress: `${this.state.currentStep}/${this.state.steps.length}`,
      hasErrors: this.state.errors.length > 0,
    };
  }

  /**
   * 重置状态
   */
  reset(): void {
    this.state = {
      currentStep: 0,
      steps: [],
      results: new Map(),
      errors: [],
    };
  }
}
```

---

## 冲突解决

### 命名空间隔离

```typescript
export class NamespacedToolRegistry {
  private tools = new Map<string, Map<string, any>>();

  /**
   * 注册工具（带命名空间）
   */
  registerTool(namespace: string, toolName: string, tool: any): void {
    if (!this.tools.has(namespace)) {
      this.tools.set(namespace, new Map());
    }
    this.tools.get(namespace)!.set(toolName, tool);
  }

  /**
   * 获取工具（带命名空间）
   */
  getTool(namespace: string, toolName: string): any {
    return this.tools.get(namespace)?.get(toolName);
  }

  /**
   * 获取完全限定名
   */
  getQualifiedName(namespace: string, toolName: string): string {
    return `${namespace}:${toolName}`;
  }
}

// 使用示例
const registry = new NamespacedToolRegistry();

// 注册 filesystem 的 read_file
registry.registerTool('filesystem', 'read_file', fsReadFileTool);

// 注册 github 的 read_file（不同实现）
registry.registerTool('github', 'read_file', githubReadFileTool);

// 调用时使用完全限定名
const tool = registry.getTool('filesystem', 'read_file');
```

### 优先级管理

```typescript
export class PriorityManager {
  private priorities = new Map<string, number>();

  /**
   * 设置服务器优先级
   */
  setPriority(serverId: string, priority: number): void {
    this.priorities.set(serverId, priority);
  }

  /**
   * 获取优先级
   */
  getPriority(serverId: string): number {
    return this.priorities.get(serverId) || 0;
  }

  /**
   * 按优先级排序服务器
   */
  sortByPriority(serverIds: string[]): string[] {
    return serverIds.sort((a, b) => {
      return this.getPriority(b) - this.getPriority(a);
    });
  }
}

// 使用示例
const priorityManager = new PriorityManager();

// 设置优先级（数字越大优先级越高）
priorityManager.setPriority('filesystem', 10);
priorityManager.setPriority('github', 5);
priorityManager.setPriority('postgres', 3);

// 当多个服务器都能处理同一请求时，选择优先级最高的
const servers = ['github', 'filesystem', 'postgres'];
const sorted = priorityManager.sortByPriority(servers);
// 结果: ['filesystem', 'github', 'postgres']
```

### 冲突检测与解决

```typescript
export class ConflictResolver {
  /**
   * 检测工具名称冲突
   */
  detectConflicts(servers: Map<string, Client>): Map<string, string[]> {
    const toolNames = new Map<string, string[]>();

    for (const [serverId, client] of servers.entries()) {
      // 假设我们有工具列表
      const tools = ['read_file', 'write_file', 'search'];

      for (const toolName of tools) {
        if (!toolNames.has(toolName)) {
          toolNames.set(toolName, []);
        }
        toolNames.get(toolName)!.push(serverId);
      }
    }

    // 返回有冲突的工具（被多个服务器提供）
    const conflicts = new Map<string, string[]>();
    for (const [toolName, serverIds] of toolNames.entries()) {
      if (serverIds.length > 1) {
        conflicts.set(toolName, serverIds);
      }
    }

    return conflicts;
  }

  /**
   * 解决冲突（使用命名空间）
   */
  resolveConflicts(
    conflicts: Map<string, string[]>,
    pi: ExtensionAPI
  ): void {
    for (const [toolName, serverIds] of conflicts.entries()) {
      console.warn(`Conflict detected for tool: ${toolName}`);
      console.warn(`Provided by: ${serverIds.join(', ')}`);
      console.warn(`Using namespaced names: ${serverIds.map(id => `${id}:${toolName}`).join(', ')}`);
    }
  }
}
```

---

## 完整工作流示例

### 代码审查工作流

```typescript
pi.registerTool({
  name: 'code_review_workflow',
  label: 'Code Review Workflow',
  description: 'Complete code review workflow across multiple servers',
  parameters: Type.Object({
    file_path: Type.String({ description: 'File to review' }),
  }),

  async execute(toolCallId, params, signal, onUpdate, ctx) {
    const workflow = new WorkflowState();
    workflow.addStep('read_file');
    workflow.addStep('search_similar');
    workflow.addStep('check_database');
    workflow.addStep('generate_report');

    try {
      // 步骤 1: 读取文件
      onUpdate({ type: 'text', text: '📖 Reading file...' });
      const fsClient = clients.get('filesystem')!;
      const fileResult = await fsClient.callTool({
        name: 'read_file',
        arguments: { path: params.file_path },
      });
      workflow.recordResult('read_file', fileResult);

      // 步骤 2: 搜索相似代码
      onUpdate({ type: 'text', text: '🔍 Searching similar code...' });
      const githubClient = clients.get('github')!;
      const searchResult = await githubClient.callTool({
        name: 'search_code',
        arguments: { query: `filename:${params.file_path}` },
      });
      workflow.recordResult('search_similar', searchResult);

      // 步骤 3: 检查数据库中的代码质量记录
      onUpdate({ type: 'text', text: '💾 Checking quality records...' });
      const dbClient = clients.get('postgres')!;
      const dbResult = await dbClient.callTool({
        name: 'query',
        arguments: {
          sql: `SELECT * FROM code_quality WHERE file_path = '${params.file_path}'`,
        },
      });
      workflow.recordResult('check_database', dbResult);

      // 步骤 4: 生成审查报告
      onUpdate({ type: 'text', text: '📝 Generating report...' });
      const report = generateReviewReport(workflow.getState());
      workflow.recordResult('generate_report', report);

      return {
        content: [{
          type: 'text',
          text: `Code review complete!\n\n${report}`,
        }],
        details: workflow.getState(),
      };
    } catch (error) {
      workflow.recordError(error as Error);
      return {
        content: [{
          type: 'text',
          text: `Workflow error: ${error}`,
        }],
        isError: true,
        details: workflow.getState(),
      };
    }
  },
});

function generateReviewReport(state: any): string {
  return `
Code Review Report
==================
File: ${state.results.get('read_file')?.content[0]?.text?.split('\n')[0] || 'Unknown'}
Similar files found: ${state.results.get('search_similar')?.content?.length || 0}
Quality score: ${state.results.get('check_database')?.content[0]?.text || 'N/A'}
Status: ${state.hasErrors ? '❌ Failed' : '✅ Passed'}
  `.trim();
}
```

---

## 总结

### 核心要点

1. **工具编排**：顺序执行 + 并行执行模式
2. **上下文共享**：ContextManager + WorkflowState
3. **冲突解决**：命名空间隔离 + 优先级管理
4. **工作流**：端到端场景编排
5. **错误处理**：完整的状态跟踪和错误记录

### 关键约束

- ✅ 使用命名空间避免工具名称冲突
- ✅ 实现上下文管理器共享数据
- ✅ 使用优先级管理器处理冲突
- ✅ 完整的工作流状态跟踪
- ✅ 并行执行提高性能

### 下一步

- 阅读 [07_实战代码_12_故障排查与优化](./07_实战代码_12_故障排查与优化.md) 学习故障排查

---

**参考资源**：
- [MCP Servers Repository](https://github.com/modelcontextprotocol/servers)
- [The Best MCP Servers for Developers in 2026](https://www.builder.io/blog/best-mcp-servers-2026)
