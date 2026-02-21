# 实战代码 09：发现与配置 Servers

> **发现和配置 MCP 服务器，构建强大的工具生态系统**

---

## 概述

MCP 服务器发现和配置是构建 AI 工具生态系统的第一步。本文介绍如何发现优质的 MCP 服务器，以及如何在 pi-mono 中正确配置它们。

```
服务器发现与配置核心：
├─ 发现渠道 → 官方注册表 + 社区列表
├─ 配置模式 → settings.json + 环境变量
├─ 认证管理 → GitHub OAuth + API Keys
└─ 服务器管理 → 安装 + 更新 + 移除
```

**本质**：服务器发现与配置是连接 AI 能力与真实世界工具的桥梁，通过标准化的配置模式让 AI 能够安全、高效地访问各种服务。

[Source: The Best MCP Servers for Developers in 2026](https://www.builder.io/blog/best-mcp-servers-2026)

---

## 发现 MCP 服务器

### 官方渠道

**1. MCP 官方注册表**

```bash
# 访问官方注册表
https://registry.modelcontextprotocol.io/

# API 文档
https://registry.modelcontextprotocol.io/docs
```

官方注册表提供：
- 搜索和浏览已发布的服务器
- 服务器元数据（版本、作者、许可证）
- 安装说明和配置示例
- 社区评分和反馈

[Source: MCP Registry](https://github.com/modelcontextprotocol/registry)

**2. MCP 服务器参考仓库**

```bash
# 官方参考实现
https://github.com/modelcontextprotocol/servers

# 包含的参考服务器：
- Everything: 测试服务器（prompts + resources + tools）
- Fetch: 网页内容抓取
- Filesystem: 安全文件操作
- Git: Git 仓库管理
- Memory: 知识图谱记忆系统
- Sequential Thinking: 结构化推理
- Time: 时间和时区转换
```

[Source: MCP Servers Repository](https://github.com/modelcontextprotocol/servers)

**3. 社区精选列表**

```bash
# Awesome MCP Servers
https://mcpservers.org/

# Glama.ai 市场（带可视化预览）
https://glama.ai/mcp/servers

# MCP Market 排行榜
https://mcpmarket.com/leaderboards
```

[Source: The Best MCP Servers for Developers in 2026](https://www.builder.io/blog/best-mcp-servers-2026)

### 服务器分类

**按功能分类**：

| 类别 | 典型服务器 | 用途 |
|------|-----------|------|
| **Learn** | Context7, Brave Search, GPT Researcher | 文档检索、网页搜索、深度研究 |
| **Create** | Figma, Magic UI, Builder Fusion | 设计转代码、UI 组件库 |
| **Build** | Filesystem, Git, GitHub, E2B | 文件操作、版本控制、代码执行 |
| **Data** | PostgreSQL, MongoDB, Supabase, Stripe | 数据库访问、支付集成 |
| **Test** | Playwright, Chrome DevTools, BrowserStack | 浏览器自动化、跨浏览器测试 |
| **Deploy** | Vercel, Netlify | 部署和托管 |
| **Run** | Sentry, Datadog, Last9 | 错误追踪、可观测性 |
| **Work** | Slack, Linear, Jira, Notion | 沟通、项目管理、知识库 |
| **Automate** | n8n, Zapier, Pipedream | 工作流自动化 |
| **Brain** | Sequential Thinking, Knowledge Graph | 推理增强、长期记忆 |

[Source: The Best MCP Servers for Developers in 2026](https://www.builder.io/blog/best-mcp-servers-2026)

### 搜索策略

```typescript
/**
 * 服务器搜索策略
 */
export interface ServerSearchCriteria {
  // 按需求搜索
  category: 'learn' | 'create' | 'build' | 'data' | 'test' | 'deploy' | 'run' | 'work' | 'automate' | 'brain';

  // 按成熟度筛选
  maturity: 'official' | 'production-ready' | 'community';

  // 按语言筛选
  language?: 'typescript' | 'python' | 'go' | 'rust';

  // 按许可证筛选
  license?: 'MIT' | 'Apache-2.0' | 'GPL-3.0';
}

/**
 * 搜索示例
 */
// 1. 需要文件操作？→ Filesystem (官方参考)
// 2. 需要数据库访问？→ PostgreSQL, MongoDB (生产就绪)
// 3. 需要浏览器自动化？→ Playwright (官方集成)
// 4. 需要错误追踪？→ Sentry (官方集成)
```

---

## 配置模式

### settings.json 基础配置

**全局配置** (`~/.pi/agent/settings.json`):

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
    },
    "git": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-git",
        "--repository",
        "/Users/username/projects/my-repo"
      ]
    }
  }
}
```

**项目配置** (`.pi/settings.json`):

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

[Source: Pi-mono Settings Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/settings.md)

### 多服务器配置

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
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "postgres": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-e",
        "POSTGRES_CONNECTION_STRING=${POSTGRES_CONNECTION_STRING}",
        "mcp/postgres"
      ]
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "${BRAVE_API_KEY}"
      }
    }
  }
}
```

### 环境变量管理

创建 `.env`:

```bash
# GitHub
GITHUB_TOKEN=ghp_your_token_here

# Database
POSTGRES_CONNECTION_STRING=postgresql://user:password@localhost:5432/dbname

# Search
BRAVE_API_KEY=your_brave_api_key

# Observability
SENTRY_DSN=https://your-sentry-dsn
DATADOG_API_KEY=your_datadog_api_key
```

**加载环境变量**:

```typescript
import dotenv from 'dotenv';

// 加载 .env 文件
dotenv.config();

// 验证必需的环境变量
const requiredEnvVars = [
  'GITHUB_TOKEN',
  'POSTGRES_CONNECTION_STRING',
];

for (const envVar of requiredEnvVars) {
  if (!process.env[envVar]) {
    throw new Error(`Missing required environment variable: ${envVar}`);
  }
}
```

### 认证配置

**GitHub OAuth**:

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

**API Keys**:

```json
{
  "mcpServers": {
    "stripe": {
      "command": "npx",
      "args": ["-y", "@stripe/mcp-server"],
      "env": {
        "STRIPE_API_KEY": "${STRIPE_API_KEY}",
        "STRIPE_WEBHOOK_SECRET": "${STRIPE_WEBHOOK_SECRET}"
      }
    }
  }
}
```

**DNS/HTTP 验证** (用于自定义域名):

```bash
# DNS 验证
# 添加 TXT 记录到你的域名
_mcp-challenge.yourdomain.com TXT "verification-token"

# HTTP 验证
# 在你的网站根目录创建文件
/.well-known/mcp-verification.txt
```

[Source: MCP Registry](https://github.com/modelcontextprotocol/registry)

---

## 常用服务器示例

### Filesystem Server

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/username/projects",
        "/Users/username/documents"
      ]
    }
  }
}
```

**功能**：
- 读取文件内容
- 写入文件
- 列出目录
- 搜索文件
- 创建/删除文件和目录

**安全配置**：
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "--allowed-directories",
        "/Users/username/projects",
        "--read-only"
      ]
    }
  }
}
```

[Source: MCP Servers Repository](https://github.com/modelcontextprotocol/servers)

### Git Server

```json
{
  "mcpServers": {
    "git": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-git",
        "--repository",
        "/Users/username/projects/my-repo"
      ]
    }
  }
}
```

**功能**：
- 查看提交历史
- 查看分支
- 查看文件差异
- 创建分支
- 提交更改

[Source: MCP Servers Repository](https://github.com/modelcontextprotocol/servers)

### PostgreSQL Server

```json
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "POSTGRES_CONNECTION_STRING": "postgresql://user:password@localhost:5432/dbname"
      }
    }
  }
}
```

**功能**：
- 执行 SQL 查询（只读）
- 查看表结构
- 查看索引
- 查看约束

**安全最佳实践**：
```sql
-- 创建只读用户
CREATE USER mcp_readonly WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE mydb TO mcp_readonly;
GRANT USAGE ON SCHEMA public TO mcp_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO mcp_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO mcp_readonly;
```

[Source: The Best MCP Servers for Developers in 2026](https://www.builder.io/blog/best-mcp-servers-2026)

### GitHub Server

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

**功能**：
- 搜索代码
- 查看 Pull Requests
- 查看 Issues
- 查看提交历史
- 创建 Issues

**Token 权限**：
```bash
# 创建 GitHub Personal Access Token
# 需要的权限：
- repo (完整仓库访问)
- read:org (读取组织信息)
```

[Source: MCP Servers Repository](https://github.com/modelcontextprotocol/servers)

### Brave Search Server

```json
{
  "mcpServers": {
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "${BRAVE_API_KEY}"
      }
    }
  }
}
```

**功能**：
- 网页搜索
- 新闻搜索
- 图片搜索
- 视频搜索

[Source: The Best MCP Servers for Developers in 2026](https://www.builder.io/blog/best-mcp-servers-2026)

---

## 服务器管理

### 安装服务器

**使用 pi 命令**:

```bash
# 全局安装
pi install npm:@modelcontextprotocol/server-filesystem

# 项目安装
pi install -l npm:@modelcontextprotocol/server-postgres

# 从 Git 安装
pi install git:github.com/user/mcp-server

# 从本地路径安装
pi install /path/to/local/server
```

[Source: Pi Packages Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/packages.md)

**使用 npx（临时使用）**:

```bash
# 不安装，直接使用
pi -e npm:@modelcontextprotocol/server-filesystem
```

### 列出已安装服务器

```bash
# 列出所有已安装的包
pi list

# 输出示例：
# Global packages:
#   npm:@modelcontextprotocol/server-filesystem@1.0.0
#   npm:@modelcontextprotocol/server-git@1.0.0
#
# Project packages (.pi/settings.json):
#   npm:@modelcontextprotocol/server-postgres@1.0.0
```

### 更新服务器

```bash
# 更新所有非固定版本的包
pi update

# 更新特定包
npm update -g @modelcontextprotocol/server-filesystem
```

### 移除服务器

```bash
# 移除全局包
pi remove npm:@modelcontextprotocol/server-filesystem

# 移除项目包
pi remove -l npm:@modelcontextprotocol/server-postgres
```

### 故障排查

**检查服务器状态**:

```typescript
/**
 * 健康检查脚本
 */
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

async function checkServerHealth(serverConfig: any): Promise<boolean> {
  try {
    const client = new Client({
      name: 'health-check',
      version: '1.0.0',
    });

    const transport = new StdioClientTransport({
      command: serverConfig.command,
      args: serverConfig.args,
      env: serverConfig.env,
    });

    await client.connect(transport);

    // 尝试列出工具
    const tools = await client.listTools();
    console.log(`✓ Server healthy: ${tools.tools.length} tools available`);

    await client.close();
    return true;
  } catch (error) {
    console.error(`✗ Server unhealthy:`, error);
    return false;
  }
}
```

**常见问题**:

```typescript
// 1. 命令未找到
// 解决：检查 npx 是否安装，或使用完整路径
{
  "command": "/usr/local/bin/node",
  "args": ["/path/to/server/index.js"]
}

// 2. 环境变量未加载
// 解决：确保 .env 文件存在且格式正确
dotenv.config({ path: '/absolute/path/to/.env' });

// 3. 权限错误
// 解决：检查文件/目录权限
chmod +x /path/to/server/index.js

// 4. 端口冲突
// 解决：更改服务器端口
{
  "env": {
    "PORT": "8081"
  }
}
```

### 日志和调试

```bash
# 启用调试日志
DEBUG=mcp:* pi

# 查看服务器输出
tail -f ~/.pi/agent/logs/mcp-server.log

# 测试服务器连接
npx @modelcontextprotocol/inspector \
  npx -y @modelcontextprotocol/server-filesystem /path/to/dir
```

---

## 最佳实践

### 安全配置

```typescript
// ✅ 使用只读模式
{
  "filesystem": {
    "args": ["--read-only", "/path/to/dir"]
  }
}

// ✅ 限制访问范围
{
  "filesystem": {
    "args": ["--allowed-directories", "/safe/path"]
  }
}

// ✅ 使用环境变量存储敏感信息
{
  "env": {
    "API_KEY": "${API_KEY}"  // 从 .env 加载
  }
}

// ✅ 为每个服务器使用独立的 API key
// 避免使用同一个 key 访问多个服务
```

[Source: The Best MCP Servers for Developers in 2026](https://www.builder.io/blog/best-mcp-servers-2026)

### 配置组织

```json
{
  "mcpServers": {
    // 开发工具
    "filesystem": { ... },
    "git": { ... },
    "github": { ... },

    // 数据访问
    "postgres": { ... },
    "mongodb": { ... },

    // 外部服务
    "stripe": { ... },
    "slack": { ... },

    // 可观测性
    "sentry": { ... },
    "datadog": { ... }
  }
}
```

### 性能优化

```typescript
// ✅ 按需启动服务器
// 不要一次性启动所有服务器

// ✅ 使用缓存
// 对频繁访问的数据启用缓存

// ✅ 限制并发连接
{
  "env": {
    "MAX_CONNECTIONS": "10"
  }
}
```

---

## 总结

### 核心要点

1. **发现渠道**：官方注册表 + 社区列表 + 分类搜索
2. **配置模式**：settings.json + 环境变量 + 认证管理
3. **常用服务器**：Filesystem, Git, GitHub, PostgreSQL, Brave Search
4. **服务器管理**：安装、更新、移除、健康检查
5. **安全最佳实践**：只读模式、限制范围、环境变量、独立 API keys

### 关键约束

- ✅ 从官方渠道发现服务器
- ✅ 使用 settings.json 管理配置
- ✅ 环境变量存储敏感信息
- ✅ 只读模式和权限限制
- ✅ 定期更新和健康检查

### 下一步

- 阅读 [07_实战代码_10_常用Server集成](./07_实战代码_10_常用Server集成.md) 学习常用服务器
- 阅读 [07_实战代码_11_多Server协作](./07_实战代码_11_多Server协作.md) 学习多服务器协作

---

**参考资源**：
- [MCP Registry](https://registry.modelcontextprotocol.io/)
- [MCP Servers Repository](https://github.com/modelcontextprotocol/servers)
- [The Best MCP Servers for Developers in 2026](https://www.builder.io/blog/best-mcp-servers-2026)
- [Pi Packages Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/packages.md)
