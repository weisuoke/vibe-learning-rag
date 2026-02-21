# 实战代码 10：常用 Server 集成

> **集成 Filesystem、Database、API 等常用 MCP 服务器**

---

## 概述

常用 MCP 服务器集成是构建实用 AI 工具的基础。本文实现 Filesystem、Database 和 API 服务器的完整集成示例。

```
常用服务器集成核心：
├─ Filesystem → 文件读写 + 目录管理
├─ Database → SQL 查询 + 数据访问
├─ API → GitHub + Stripe 集成
└─ 完整示例 → 端到端集成
```

[Source: The Best MCP Servers for Developers in 2026](https://www.builder.io/blog/best-mcp-servers-2026)

---

## Filesystem Server 集成

### 配置

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/username/projects"
      ]
    }
  }
}
```

### 扩展集成

创建 `extensions/filesystem-integration.ts`:

```typescript
import type { ExtensionAPI } from '@mariozechner/pi-coding-agent';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { Type } from '@sinclair/typebox';

export default function (pi: ExtensionAPI) {
  let fsClient: Client | null = null;

  pi.on('session_start', async (_event, ctx) => {
    try {
      fsClient = new Client({
        name: 'filesystem-client',
        version: '1.0.0',
      });

      const transport = new StdioClientTransport({
        command: 'npx',
        args: [
          '-y',
          '@modelcontextprotocol/server-filesystem',
          '/Users/username/projects',
        ],
      });

      await fsClient.connect(transport);

      // 注册文件读取工具
      pi.registerTool({
        name: 'read_project_file',
        label: 'Read Project File',
        description: 'Read a file from the projects directory',
        parameters: Type.Object({
          path: Type.String({ description: 'Relative file path' }),
        }),

        async execute(toolCallId, params, signal, onUpdate, ctx) {
          try {
            const result = await fsClient!.callTool({
              name: 'read_file',
              arguments: { path: params.path },
            });

            return {
              content: result.content,
              details: { tool: 'filesystem', operation: 'read' },
            };
          } catch (error) {
            return {
              content: [{
                type: 'text',
                text: `Error reading file: ${error}`,
              }],
              isError: true,
            };
          }
        },
      });

      // 注册文件写入工具
      pi.registerTool({
        name: 'write_project_file',
        label: 'Write Project File',
        description: 'Write content to a file',
        parameters: Type.Object({
          path: Type.String({ description: 'Relative file path' }),
          content: Type.String({ description: 'File content' }),
        }),

        async execute(toolCallId, params, signal, onUpdate, ctx) {
          try {
            const result = await fsClient!.callTool({
              name: 'write_file',
              arguments: {
                path: params.path,
                content: params.content,
              },
            });

            return {
              content: result.content,
              details: { tool: 'filesystem', operation: 'write' },
            };
          } catch (error) {
            return {
              content: [{
                type: 'text',
                text: `Error writing file: ${error}`,
              }],
              isError: true,
            };
          }
        },
      });

      ctx.ui.notify('Filesystem server connected', 'success');
    } catch (error) {
      ctx.ui.notify(`Filesystem connection failed: ${error}`, 'error');
    }
  });

  pi.on('session_shutdown', async () => {
    if (fsClient) {
      await fsClient.close();
    }
  });
}
```

[Source: MCP Servers Repository](https://github.com/modelcontextprotocol/servers)

---

## Database Server 集成

### PostgreSQL 配置

```json
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "POSTGRES_CONNECTION_STRING": "${POSTGRES_CONNECTION_STRING}"
      }
    }
  }
}
```

### 扩展集成

创建 `extensions/database-integration.ts`:

```typescript
import type { ExtensionAPI } from '@mariozechner/pi-coding-agent';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { Type } from '@sinclair/typebox';

export default function (pi: ExtensionAPI) {
  let dbClient: Client | null = null;

  pi.on('session_start', async (_event, ctx) => {
    try {
      dbClient = new Client({
        name: 'postgres-client',
        version: '1.0.0',
      });

      const transport = new StdioClientTransport({
        command: 'npx',
        args: ['-y', '@modelcontextprotocol/server-postgres'],
        env: {
          POSTGRES_CONNECTION_STRING: process.env.POSTGRES_CONNECTION_STRING,
        },
      });

      await dbClient.connect(transport);

      // 注册查询工具
      pi.registerTool({
        name: 'query_database',
        label: 'Query Database',
        description: 'Execute a SQL query (read-only)',
        parameters: Type.Object({
          query: Type.String({ description: 'SQL query' }),
        }),

        async execute(toolCallId, params, signal, onUpdate, ctx) {
          try {
            const result = await dbClient!.callTool({
              name: 'query',
              arguments: { sql: params.query },
            });

            return {
              content: result.content,
              details: { tool: 'postgres', operation: 'query' },
            };
          } catch (error) {
            return {
              content: [{
                type: 'text',
                text: `Query error: ${error}`,
              }],
              isError: true,
            };
          }
        },
      });

      // 注册表结构查询工具
      pi.registerTool({
        name: 'describe_table',
        label: 'Describe Table',
        description: 'Get table schema information',
        parameters: Type.Object({
          table: Type.String({ description: 'Table name' }),
        }),

        async execute(toolCallId, params, signal, onUpdate, ctx) {
          try {
            const result = await dbClient!.callTool({
              name: 'describe_table',
              arguments: { table: params.table },
            });

            return {
              content: result.content,
              details: { tool: 'postgres', operation: 'describe' },
            };
          } catch (error) {
            return {
              content: [{
                type: 'text',
                text: `Describe error: ${error}`,
              }],
              isError: true,
            };
          }
        },
      });

      ctx.ui.notify('Database server connected', 'success');
    } catch (error) {
      ctx.ui.notify(`Database connection failed: ${error}`, 'error');
    }
  });

  pi.on('session_shutdown', async () => {
    if (dbClient) {
      await dbClient.close();
    }
  });
}
```

[Source: The Best MCP Servers for Developers in 2026](https://www.builder.io/blog/best-mcp-servers-2026)

---

## API Server 集成

### GitHub 配置

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

### 扩展集成

创建 `extensions/github-integration.ts`:

```typescript
import type { ExtensionAPI } from '@mariozechner/pi-coding-agent';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { Type } from '@sinclair/typebox';

export default function (pi: ExtensionAPI) {
  let githubClient: Client | null = null;

  pi.on('session_start', async (_event, ctx) => {
    try {
      githubClient = new Client({
        name: 'github-client',
        version: '1.0.0',
      });

      const transport = new StdioClientTransport({
        command: 'npx',
        args: ['-y', '@modelcontextprotocol/server-github'],
        env: {
          GITHUB_TOKEN: process.env.GITHUB_TOKEN,
        },
      });

      await githubClient.connect(transport);

      // 注册代码搜索工具
      pi.registerTool({
        name: 'search_github_code',
        label: 'Search GitHub Code',
        description: 'Search code in GitHub repositories',
        parameters: Type.Object({
          query: Type.String({ description: 'Search query' }),
          repo: Type.Optional(Type.String({ description: 'Repository (owner/repo)' })),
        }),

        async execute(toolCallId, params, signal, onUpdate, ctx) {
          try {
            const result = await githubClient!.callTool({
              name: 'search_code',
              arguments: params,
            });

            return {
              content: result.content,
              details: { tool: 'github', operation: 'search_code' },
            };
          } catch (error) {
            return {
              content: [{
                type: 'text',
                text: `Search error: ${error}`,
              }],
              isError: true,
            };
          }
        },
      });

      // 注册 PR 查询工具
      pi.registerTool({
        name: 'list_pull_requests',
        label: 'List Pull Requests',
        description: 'List pull requests in a repository',
        parameters: Type.Object({
          owner: Type.String({ description: 'Repository owner' }),
          repo: Type.String({ description: 'Repository name' }),
          state: Type.Optional(Type.String({ description: 'PR state (open/closed/all)' })),
        }),

        async execute(toolCallId, params, signal, onUpdate, ctx) {
          try {
            const result = await githubClient!.callTool({
              name: 'list_pull_requests',
              arguments: params,
            });

            return {
              content: result.content,
              details: { tool: 'github', operation: 'list_prs' },
            };
          } catch (error) {
            return {
              content: [{
                type: 'text',
                text: `PR list error: ${error}`,
              }],
              isError: true,
            };
          }
        },
      });

      ctx.ui.notify('GitHub server connected', 'success');
    } catch (error) {
      ctx.ui.notify(`GitHub connection failed: ${error}`, 'error');
    }
  });

  pi.on('session_shutdown', async () => {
    if (githubClient) {
      await githubClient.close();
    }
  });
}
```

[Source: MCP Servers Repository](https://github.com/modelcontextprotocol/servers)

---

## 完整集成示例

### 多服务器扩展

创建 `extensions/multi-server.ts`:

```typescript
import type { ExtensionAPI } from '@mariozechner/pi-coding-agent';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

export default function (pi: ExtensionAPI) {
  const clients = new Map<string, Client>();

  const servers = [
    {
      id: 'filesystem',
      command: 'npx',
      args: ['-y', '@modelcontextprotocol/server-filesystem', '/projects'],
    },
    {
      id: 'postgres',
      command: 'npx',
      args: ['-y', '@modelcontextprotocol/server-postgres'],
      env: { POSTGRES_CONNECTION_STRING: process.env.POSTGRES_CONNECTION_STRING },
    },
    {
      id: 'github',
      command: 'npx',
      args: ['-y', '@modelcontextprotocol/server-github'],
      env: { GITHUB_TOKEN: process.env.GITHUB_TOKEN },
    },
  ];

  pi.on('session_start', async (_event, ctx) => {
    for (const config of servers) {
      try {
        const client = new Client({
          name: `${config.id}-client`,
          version: '1.0.0',
        });

        const transport = new StdioClientTransport({
          command: config.command,
          args: config.args,
          env: config.env,
        });

        await client.connect(transport);
        clients.set(config.id, client);

        const tools = await client.listTools();
        ctx.ui.notify(
          `${config.id}: ${tools.tools.length} tools`,
          'success'
        );
      } catch (error) {
        ctx.ui.notify(`${config.id} failed: ${error}`, 'error');
      }
    }
  });

  pi.on('session_shutdown', async () => {
    for (const client of clients.values()) {
      await client.close();
    }
    clients.clear();
  });
}
```

---

## 总结

### 核心要点

1. **Filesystem**：文件读写 + 目录管理
2. **Database**：SQL 查询 + 表结构查询
3. **API**：GitHub 代码搜索 + PR 管理
4. **多服务器**：统一管理多个 MCP 服务器
5. **错误处理**：完整的 try-catch 和用户通知

### 关键约束

- ✅ 使用环境变量存储敏感信息
- ✅ 完整的错误处理和日志
- ✅ 优雅的连接管理和清理
- ✅ 工具注册和参数验证
- ✅ 用户友好的通知

### 下一步

- 阅读 [07_实战代码_11_多Server协作](./07_实战代码_11_多Server协作.md) 学习多服务器协作
- 阅读 [07_实战代码_12_故障排查与优化](./07_实战代码_12_故障排查与优化.md) 学习故障排查

---

**参考资源**：
- [MCP Servers Repository](https://github.com/modelcontextprotocol/servers)
- [The Best MCP Servers for Developers in 2026](https://www.builder.io/blog/best-mcp-servers-2026)
