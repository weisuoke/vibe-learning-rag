# 实战代码 04：高级 Server 特性

> **实现 Resources、Prompts、Subscriptions 等高级 MCP 特性**

---

## 概述

除了基本的工具（Tools）功能，MCP 服务器还支持资源（Resources）、提示（Prompts）和订阅（Subscriptions）等高级特性。本文将实现一个完整的文档管理 MCP 服务器，展示这些高级特性的使用。

```
高级 Server 特性：
├─ Resources → 文件式数据访问
├─ Prompts → 预定义提示模板
├─ Subscriptions → 实时变更通知
└─ Sampling → LLM 采样请求
```

**本质**：高级特性让 MCP 服务器不仅能提供工具调用，还能提供结构化数据、提示模板和实时更新，构建更丰富的 AI 应用。

---

## 完整服务器实现

### 项目初始化

```bash
# 创建项目
mkdir docs-mcp-server
cd docs-mcp-server

# 初始化并安装依赖
npm init -y
npm install @modelcontextprotocol/sdk chokidar
npm install -D typescript @types/node

# 创建目录结构
mkdir src docs
touch src/server.ts
```

### 服务器实现

创建 `src/server.ts`:

```typescript
#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
  ListPromptsRequestSchema,
  GetPromptRequestSchema,
  SubscribeRequestSchema,
  UnsubscribeRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import * as fs from 'fs/promises';
import * as path from 'path';
import chokidar from 'chokidar';

// 创建服务器实例
const server = new Server({
  name: 'docs-mcp-server',
  version: '1.0.0',
}, {
  capabilities: {
    tools: {},
    resources: {
      subscribe: true,  // 支持订阅
      listChanged: true, // 支持变更通知
    },
    prompts: {},
  },
});

// 文档目录
const DOCS_DIR = process.env.DOCS_DIR || path.join(process.cwd(), 'docs');

// 订阅管理
const subscriptions = new Map<string, Set<string>>();
let watcher: chokidar.FSWatcher | null = null;

// ==================== Resources ====================

/**
 * 列出所有资源
 */
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  try {
    const files = await fs.readdir(DOCS_DIR);
    const resources = await Promise.all(
      files
        .filter(file => file.endsWith('.md') || file.endsWith('.txt'))
        .map(async (file) => {
          const filePath = path.join(DOCS_DIR, file);
          const stats = await fs.stat(filePath);
          return {
            uri: `file:///${file}`,
            name: file,
            description: `Document: ${file}`,
            mimeType: file.endsWith('.md') ? 'text/markdown' : 'text/plain',
            annotations: {
              audience: ['user'],
              priority: 0.5,
              lastModified: stats.mtime.toISOString(),
            },
          };
        })
    );

    return { resources };
  } catch (error) {
    console.error('Error listing resources:', error);
    return { resources: [] };
  }
});

/**
 * 读取资源内容
 */
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const uri = request.params.uri;
  const fileName = uri.replace('file:///', '');
  const filePath = path.join(DOCS_DIR, fileName);

  try {
    const content = await fs.readFile(filePath, 'utf-8');
    return {
      contents: [
        {
          uri,
          mimeType: fileName.endsWith('.md') ? 'text/markdown' : 'text/plain',
          text: content,
        },
      ],
    };
  } catch (error) {
    throw new Error(`Failed to read resource: ${error}`);
  }
});

// ==================== Prompts ====================

/**
 * 列出所有提示模板
 */
server.setRequestHandler(ListPromptsRequestSchema, async () => {
  return {
    prompts: [
      {
        name: 'summarize_doc',
        description: 'Summarize a document',
        arguments: [
          {
            name: 'doc_name',
            description: 'Name of the document to summarize',
            required: true,
          },
        ],
      },
      {
        name: 'compare_docs',
        description: 'Compare two documents',
        arguments: [
          {
            name: 'doc1',
            description: 'First document name',
            required: true,
          },
          {
            name: 'doc2',
            description: 'Second document name',
            required: true,
          },
        ],
      },
      {
        name: 'extract_keywords',
        description: 'Extract keywords from a document',
        arguments: [
          {
            name: 'doc_name',
            description: 'Document name',
            required: true,
          },
          {
            name: 'count',
            description: 'Number of keywords to extract',
            required: false,
          },
        ],
      },
    ],
  };
});

/**
 * 获取提示模板内容
 */
server.setRequestHandler(GetPromptRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    case 'summarize_doc': {
      const docName = args?.doc_name as string;
      const filePath = path.join(DOCS_DIR, docName);
      const content = await fs.readFile(filePath, 'utf-8');

      return {
        description: `Summarize the document: ${docName}`,
        messages: [
          {
            role: 'user',
            content: {
              type: 'text',
              text: `Please summarize the following document:\n\n${content}`,
            },
          },
        ],
      };
    }

    case 'compare_docs': {
      const doc1 = args?.doc1 as string;
      const doc2 = args?.doc2 as string;
      const content1 = await fs.readFile(path.join(DOCS_DIR, doc1), 'utf-8');
      const content2 = await fs.readFile(path.join(DOCS_DIR, doc2), 'utf-8');

      return {
        description: `Compare documents: ${doc1} and ${doc2}`,
        messages: [
          {
            role: 'user',
            content: {
              type: 'text',
              text: `Please compare these two documents and highlight the key differences:\n\nDocument 1 (${doc1}):\n${content1}\n\nDocument 2 (${doc2}):\n${content2}`,
            },
          },
        ],
      };
    }

    case 'extract_keywords': {
      const docName = args?.doc_name as string;
      const count = (args?.count as number) || 10;
      const content = await fs.readFile(path.join(DOCS_DIR, docName), 'utf-8');

      return {
        description: `Extract ${count} keywords from ${docName}`,
        messages: [
          {
            role: 'user',
            content: {
              type: 'text',
              text: `Please extract the top ${count} keywords from this document:\n\n${content}`,
            },
          },
        ],
      };
    }

    default:
      throw new Error(`Unknown prompt: ${name}`);
  }
});

// ==================== Subscriptions ====================

/**
 * 订阅资源变更
 */
server.setRequestHandler(SubscribeRequestSchema, async (request) => {
  const { uri } = request.params;

  // 添加订阅
  if (!subscriptions.has(uri)) {
    subscriptions.set(uri, new Set());
  }
  subscriptions.get(uri)!.add(request.params._meta?.sessionId || 'default');

  // 启动文件监听器（如果尚未启动）
  if (!watcher) {
    watcher = chokidar.watch(DOCS_DIR, {
      persistent: true,
      ignoreInitial: true,
    });

    watcher.on('change', async (filePath) => {
      const fileName = path.basename(filePath);
      const uri = `file:///${fileName}`;

      if (subscriptions.has(uri)) {
        console.error(`Resource changed: ${uri}`);
        // 发送变更通知
        await server.notification({
          method: 'notifications/resources/list_changed',
          params: {},
        });
      }
    });

    watcher.on('add', async (filePath) => {
      console.error(`Resource added: ${path.basename(filePath)}`);
      await server.notification({
        method: 'notifications/resources/list_changed',
        params: {},
      });
    });

    watcher.on('unlink', async (filePath) => {
      console.error(`Resource removed: ${path.basename(filePath)}`);
      await server.notification({
        method: 'notifications/resources/list_changed',
        params: {},
      });
    });
  }

  console.error(`Subscribed to: ${uri}`);
  return {};
});

/**
 * 取消订阅
 */
server.setRequestHandler(UnsubscribeRequestSchema, async (request) => {
  const { uri } = request.params;
  const sessionId = request.params._meta?.sessionId || 'default';

  if (subscriptions.has(uri)) {
    subscriptions.get(uri)!.delete(sessionId);
    if (subscriptions.get(uri)!.size === 0) {
      subscriptions.delete(uri);
    }
  }

  console.error(`Unsubscribed from: ${uri}`);
  return {};
});

// ==================== Tools ====================

/**
 * 列出工具
 */
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: 'create_document',
      description: 'Create a new document',
      inputSchema: {
        type: 'object',
        properties: {
          name: {
            type: 'string',
            description: 'Document name (with .md or .txt extension)',
          },
          content: {
            type: 'string',
            description: 'Document content',
          },
        },
        required: ['name', 'content'],
      },
    },
    {
      name: 'update_document',
      description: 'Update an existing document',
      inputSchema: {
        type: 'object',
        properties: {
          name: {
            type: 'string',
            description: 'Document name',
          },
          content: {
            type: 'string',
            description: 'New content',
          },
        },
        required: ['name', 'content'],
      },
    },
    {
      name: 'delete_document',
      description: 'Delete a document',
      inputSchema: {
        type: 'object',
        properties: {
          name: {
            type: 'string',
            description: 'Document name',
          },
        },
        required: ['name'],
      },
    },
  ],
}));

/**
 * 执行工具
 */
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case 'create_document': {
        const { name: docName, content } = args as any;
        const filePath = path.join(DOCS_DIR, docName);
        await fs.writeFile(filePath, content, 'utf-8');
        return {
          content: [
            {
              type: 'text',
              text: `Document created: ${docName}`,
            },
          ],
        };
      }

      case 'update_document': {
        const { name: docName, content } = args as any;
        const filePath = path.join(DOCS_DIR, docName);
        await fs.writeFile(filePath, content, 'utf-8');
        return {
          content: [
            {
              type: 'text',
              text: `Document updated: ${docName}`,
            },
          ],
        };
      }

      case 'delete_document': {
        const { name: docName } = args as any;
        const filePath = path.join(DOCS_DIR, docName);
        await fs.unlink(filePath);
        return {
          content: [
            {
              type: 'text',
              text: `Document deleted: ${docName}`,
            },
          ],
        };
      }

      default:
        return {
          content: [
            {
              type: 'text',
              text: `Unknown tool: ${name}`,
            },
          ],
          isError: true,
        };
    }
  } catch (error) {
    return {
      content: [
        {
          type: 'text',
          text: `Error: ${error instanceof Error ? error.message : String(error)}`,
        },
      ],
      isError: true,
    };
  }
});

// ==================== 启动服务器 ====================

async function main() {
  // 确保文档目录存在
  await fs.mkdir(DOCS_DIR, { recursive: true });

  const transport = new StdioServerTransport();
  await server.connect(transport);

  console.error('Docs MCP Server running on stdio');
  console.error(`Docs directory: ${DOCS_DIR}`);
  console.error('Features: Tools, Resources, Prompts, Subscriptions');
}

// 优雅关闭
process.on('SIGINT', async () => {
  if (watcher) {
    await watcher.close();
  }
  process.exit(0);
});

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
```

---

## 特性详解

### Resources（资源）

**用途**：提供文件式数据访问，类似于 API 响应或文件内容。

**关键点**：
- URI 格式：`file:///filename.md`
- 支持 MIME 类型
- 可添加注解（audience、priority、lastModified）

**使用场景**：
- 文档库访问
- 配置文件读取
- API 响应缓存

### Prompts（提示模板）

**用途**：预定义的提示模板，帮助用户完成特定任务。

**关键点**：
- 支持参数化
- 返回完整的消息列表
- 可包含系统提示和用户提示

**使用场景**：
- 文档摘要
- 代码审查
- 数据分析

### Subscriptions（订阅）

**用途**：实时监听资源变更，发送通知。

**关键点**：
- 使用 `chokidar` 监听文件系统
- 发送 `notifications/resources/list_changed` 通知
- 管理订阅状态

**使用场景**：
- 文件变更监听
- 实时数据更新
- 协作编辑

---

## 测试与使用

### 本地测试

```bash
# 构建项目
npm run build

# 创建测试文档
mkdir docs
echo "# Test Document" > docs/test.md

# 启动服务器
DOCS_DIR=./docs npm start
```

### Claude Desktop 配置

```json
{
  "mcpServers": {
    "docs": {
      "command": "node",
      "args": [
        "/absolute/path/to/docs-mcp-server/dist/server.js"
      ],
      "env": {
        "DOCS_DIR": "/absolute/path/to/docs"
      }
    }
  }
}
```

### 测试查询

```
1. "List all available documents" (Resources)
2. "Read the content of test.md" (Resources)
3. "Use the summarize_doc prompt for test.md" (Prompts)
4. "Create a new document called notes.md" (Tools)
5. "Subscribe to changes in test.md" (Subscriptions)
```

---

## 最佳实践

### Resource URI 设计

```typescript
// ✅ 清晰的 URI 结构
const uri = `file:///${category}/${filename}`;
const uri = `https://api.example.com/data/${id}`;
const uri = `git://github.com/user/repo/path/to/file`;

// ✅ 使用 URI 模板
const uriTemplate = 'file:///{category}/{filename}';
```

### Prompt 参数验证

```typescript
// ✅ 验证必需参数
if (!args?.doc_name) {
  throw new Error('doc_name parameter is required');
}

// ✅ 提供默认值
const count = (args?.count as number) || 10;

// ✅ 类型检查
if (typeof args?.doc_name !== 'string') {
  throw new Error('doc_name must be a string');
}
```

### Subscription 管理

```typescript
// ✅ 使用 Map 管理订阅
const subscriptions = new Map<string, Set<string>>();

// ✅ 清理空订阅
if (subscriptions.get(uri)!.size === 0) {
  subscriptions.delete(uri);
}

// ✅ 优雅关闭监听器
process.on('SIGINT', async () => {
  if (watcher) {
    await watcher.close();
  }
  process.exit(0);
});
```

### 错误处理

```typescript
// ✅ 资源不存在处理
try {
  const content = await fs.readFile(filePath, 'utf-8');
  return { contents: [{ uri, text: content }] };
} catch (error) {
  if ((error as any).code === 'ENOENT') {
    throw new Error(`Resource not found: ${uri}`);
  }
  throw error;
}
```

---

## 总结

### 核心要点

1. **Resources**：提供文件式数据访问，支持 URI 和 MIME 类型
2. **Prompts**：预定义提示模板，支持参数化
3. **Subscriptions**：实时监听资源变更，发送通知
4. **Tools**：基本的 CRUD 操作
5. **集成**：所有特性可在同一服务器中共存

### 关键约束

- ✅ Resource URI 必须唯一且有意义
- ✅ Prompt 参数必须验证
- ✅ Subscription 需要管理订阅状态
- ✅ 文件监听器需要优雅关闭
- ✅ 所有特性都需要错误处理

### 下一步

- 阅读 [07_实战代码_05_基础Extension开发](./07_实战代码_05_基础Extension开发.md) 学习扩展开发
- 阅读 [07_实战代码_06_MCP_Client集成](./07_实战代码_06_MCP_Client集成.md) 学习客户端集成

---

**参考资源**：
- [MCP Resources Specification](https://modelcontextprotocol.io/docs/concepts/resources)
- [MCP Prompts Specification](https://modelcontextprotocol.io/docs/concepts/prompts)
- [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
