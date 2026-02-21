# 核心概念 07：Resource 管理

> **深入理解 MCP Resource 的提供模式、动态生成和订阅机制**

---

## 概述

MCP Resource 提供了标准化的方式让服务器向客户端暴露数据资源，支持静态资源、动态生成、订阅更新和缓存策略。

```
Resource 管理核心模式：
├─ 资源提供模式 → 静态资源、动态生成、模板化
├─ 订阅机制 → 实时更新通知
├─ 缓存策略 → 性能优化
└─ URI 设计 → 标准化资源标识
```

**本质**：Resource 是 MCP Server 暴露给客户端的数据源，通过 URI 唯一标识，支持文本和二进制内容，为 AI 模型提供上下文信息。

[Source: Resources - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-06-18/server/resources)

---

## 资源提供模式

### 静态资源提供

**基本资源定义**：

```typescript
interface Resource {
  uri: string;               // 唯一标识符（RFC3986）
  name: string;              // 资源名称
  title?: string;            // 可选的人类可读名称
  description?: string;      // 可选的描述
  mimeType?: string;         // 可选的 MIME 类型
  size?: number;             // 可选的字节大小
  icons?: Icon[];            // 可选的图标数组
}
```

**TypeScript 实现**：

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";

const server = new McpServer({
  name: "file-server",
  version: "1.0.0"
});

// 注册资源列表处理器
server.registerResource(
  "project-files",
  {
    uri: "file:///project/README.md",
    name: "README.md",
    title: "Project Documentation",
    description: "Main project documentation file",
    mimeType: "text/markdown"
  },
  async () => {
    const content = await fs.readFile("/project/README.md", "utf-8");
    return {
      contents: [
        {
          uri: "file:///project/README.md",
          mimeType: "text/markdown",
          text: content
        }
      ]
    };
  }
);
```

**Python 实现**：

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("file-server")

@mcp.resource("file:///project/README.md")
async def get_readme() -> str:
    """Get project README file."""
    with open("/project/README.md", "r") as f:
        return f.read()
```

[Source: Resources - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-06-18/server/resources)

### 资源模板（动态生成）

**URI 模板**：

资源模板使用 [RFC6570 URI 模板](https://datatracker.ietf.org/doc/html/rfc6570)允许服务器暴露参数化资源。

**示例**：

```typescript
server.registerResourceTemplate(
  {
    uriTemplate: "file:///{path}",
    name: "Project Files",
    title: "📁 Project Files",
    description: "Access files in the project directory",
    mimeType: "application/octet-stream"
  },
  async ({ path }) => {
    // 验证路径安全性
    if (!isPathSafe(path)) {
      throw new Error("Invalid path");
    }

    const fullPath = `/project/${path}`;
    const content = await fs.readFile(fullPath, "utf-8");
    const mimeType = getMimeType(fullPath);

    return {
      contents: [
        {
          uri: `file:///${path}`,
          mimeType,
          text: content
        }
      ]
    };
  }
);
```

**URI 模板语法**：

```
file:///{path}                    # 简单变量
file:///{+path}                   # 保留字符扩展
file:///users/{user_id}/files     # 路径段
file:///search{?query,limit}      # 查询参数
```

[Source: Resources - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-06-18/server/resources)

### 资源内容类型

#### 文本内容

```json
{
  "uri": "file:///example.txt",
  "mimeType": "text/plain",
  "text": "Resource content"
}
```

#### 二进制内容

```json
{
  "uri": "file:///example.png",
  "mimeType": "image/png",
  "blob": "base64-encoded-data"
}
```

**TypeScript 实现**：

```typescript
// 文本资源
return {
  contents: [
    {
      uri: "file:///document.md",
      mimeType: "text/markdown",
      text: markdownContent
    }
  ]
};

// 二进制资源
return {
  contents: [
    {
      uri: "file:///image.png",
      mimeType: "image/png",
      blob: Buffer.from(imageData).toString("base64")
    }
  ]
};
```

---

## 订阅机制

### 能力声明

服务器必须声明 `subscribe` 能力以支持订阅：

```json
{
  "capabilities": {
    "resources": {
      "subscribe": true,
      "listChanged": true
    }
  }
}
```

**能力组合**：

```typescript
// 两者都不支持
{ "capabilities": { "resources": {} } }

// 仅支持订阅
{ "capabilities": { "resources": { "subscribe": true } } }

// 仅支持列表变更通知
{ "capabilities": { "resources": { "listChanged": true } } }

// 两者都支持
{ "capabilities": { "resources": { "subscribe": true, "listChanged": true } } }
```

[Source: Resources - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-06-18/server/resources)

### 订阅实现

**客户端订阅请求**：

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "resources/subscribe",
  "params": {
    "uri": "file:///project/src/main.rs"
  }
}
```

**服务器更新通知**：

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": {
    "uri": "file:///project/src/main.rs"
  }
}
```

**TypeScript 实现**：

```typescript
import { watch } from "fs/promises";

// 订阅管理器
class SubscriptionManager {
  private subscriptions = new Map<string, Set<string>>();
  private watchers = new Map<string, AbortController>();

  async subscribe(uri: string, clientId: string) {
    if (!this.subscriptions.has(uri)) {
      this.subscriptions.set(uri, new Set());
      await this.startWatching(uri);
    }

    this.subscriptions.get(uri)!.add(clientId);
  }

  async unsubscribe(uri: string, clientId: string) {
    const clients = this.subscriptions.get(uri);
    if (clients) {
      clients.delete(clientId);
      if (clients.size === 0) {
        this.stopWatching(uri);
        this.subscriptions.delete(uri);
      }
    }
  }

  private async startWatching(uri: string) {
    const filePath = uriToPath(uri);
    const controller = new AbortController();
    this.watchers.set(uri, controller);

    try {
      const watcher = watch(filePath, { signal: controller.signal });
      for await (const event of watcher) {
        if (event.eventType === "change") {
          await this.notifyClients(uri);
        }
      }
    } catch (error) {
      if (error.name !== "AbortError") {
        console.error(`Watch error for ${uri}:`, error);
      }
    }
  }

  private stopWatching(uri: string) {
    const controller = this.watchers.get(uri);
    if (controller) {
      controller.abort();
      this.watchers.delete(uri);
    }
  }

  private async notifyClients(uri: string) {
    const clients = this.subscriptions.get(uri);
    if (clients) {
      for (const clientId of clients) {
        await server.sendNotification({
          method: "notifications/resources/updated",
          params: { uri }
        });
      }
    }
  }
}

const subscriptionManager = new SubscriptionManager();

// 处理订阅请求
server.setRequestHandler("resources/subscribe", async ({ uri }) => {
  await subscriptionManager.subscribe(uri, clientId);
  return {};
});

// 处理取消订阅请求
server.setRequestHandler("resources/unsubscribe", async ({ uri }) => {
  await subscriptionManager.unsubscribe(uri, clientId);
  return {};
});
```

### 列表变更通知

当可用资源列表发生变化时，服务器发送通知：

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/list_changed"
}
```

**实现示例**：

```typescript
class ResourceRegistry {
  private resources = new Map<string, Resource>();

  async addResource(resource: Resource) {
    this.resources.set(resource.uri, resource);
    await this.notifyListChanged();
  }

  async removeResource(uri: string) {
    this.resources.delete(uri);
    await this.notifyListChanged();
  }

  private async notifyListChanged() {
    await server.sendNotification({
      method: "notifications/resources/list_changed"
    });
  }
}
```

---

## 缓存策略

### 客户端缓存

**基于注解的缓存提示**：

```typescript
{
  uri: "file:///project/README.md",
  name: "README.md",
  mimeType: "text/markdown",
  annotations: {
    lastModified: "2025-01-12T15:00:58Z",
    priority: 0.8
  }
}
```

**缓存实现**：

```typescript
class ResourceCache {
  private cache = new Map<string, CachedResource>();

  async get(uri: string): Promise<ResourceContent | null> {
    const cached = this.cache.get(uri);
    if (!cached) return null;

    // 检查是否过期
    if (this.isExpired(cached)) {
      this.cache.delete(uri);
      return null;
    }

    return cached.content;
  }

  async set(uri: string, content: ResourceContent, lastModified?: string) {
    this.cache.set(uri, {
      content,
      lastModified: lastModified ? new Date(lastModified) : new Date(),
      cachedAt: new Date()
    });
  }

  private isExpired(cached: CachedResource): boolean {
    const now = new Date();
    const age = now.getTime() - cached.cachedAt.getTime();
    const maxAge = 5 * 60 * 1000; // 5 分钟
    return age > maxAge;
  }
}
```

### 服务器端缓存

**内存缓存**：

```typescript
class ServerResourceCache {
  private cache = new LRUCache<string, ResourceContent>({
    max: 100,
    ttl: 1000 * 60 * 5 // 5 分钟
  });

  async getResource(uri: string): Promise<ResourceContent> {
    // 检查缓存
    const cached = this.cache.get(uri);
    if (cached) {
      return cached;
    }

    // 加载资源
    const content = await this.loadResource(uri);

    // 缓存结果
    this.cache.set(uri, content);

    return content;
  }

  private async loadResource(uri: string): Promise<ResourceContent> {
    const filePath = uriToPath(uri);
    const text = await fs.readFile(filePath, "utf-8");
    const mimeType = getMimeType(filePath);

    return {
      uri,
      mimeType,
      text
    };
  }
}
```

---

## 注解（Annotations）

### 注解类型

```typescript
interface Annotations {
  audience?: ("user" | "assistant")[];  // 目标受众
  priority?: number;                     // 重要性（0.0-1.0）
  lastModified?: string;                 // ISO 8601 时间戳
}
```

**使用示例**：

```typescript
{
  uri: "file:///project/README.md",
  name: "README.md",
  mimeType: "text/markdown",
  annotations: {
    audience: ["user"],           // 仅用户可见
    priority: 0.8,                // 高优先级
    lastModified: "2026-02-21T15:00:58Z"
  }
}
```

**用途**：

1. **受众过滤**：根据目标受众过滤资源
2. **优先级排序**：优先选择高优先级资源
3. **缓存控制**：基于修改时间决定是否重新加载

[Source: Resources - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-06-18/server/resources)

---

## URI 设计模式

### 常见 URI Scheme

#### https://

用于表示 Web 上可用的资源。服务器**应该**仅在客户端能够自行从 Web 获取资源时使用此 scheme。

```typescript
{
  uri: "https://api.example.com/data/users.json",
  name: "Users Data",
  mimeType: "application/json"
}
```

#### file://

用于标识类似文件系统的资源。资源不需要映射到实际的物理文件系统。

```typescript
{
  uri: "file:///project/src/main.rs",
  name: "main.rs",
  mimeType: "text/x-rust"
}
```

**XDG MIME 类型**：

MCP 服务器**可以**使用 XDG MIME 类型（如 `inode/directory`）标识非常规文件。

```typescript
{
  uri: "file:///project/src",
  name: "src",
  mimeType: "inode/directory"
}
```

#### git://

Git 版本控制集成：

```typescript
{
  uri: "git://repo/main/src/main.rs",
  name: "main.rs (main branch)",
  mimeType: "text/x-rust"
}
```

#### 自定义 URI Scheme

自定义 URI scheme **必须**符合 RFC3986：

```typescript
{
  uri: "database://users/table/customers",
  name: "Customers Table",
  mimeType: "application/json"
}

{
  uri: "api://github/repos/owner/repo",
  name: "GitHub Repository",
  mimeType: "application/json"
}
```

[Source: Resources - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-06-18/server/resources)

---

## 错误处理

### 标准错误码

```typescript
// 资源未找到
{
  "jsonrpc": "2.0",
  "id": 5,
  "error": {
    "code": -32002,
    "message": "Resource not found",
    "data": {
      "uri": "file:///nonexistent.txt"
    }
  }
}

// 内部错误
{
  "jsonrpc": "2.0",
  "id": 6,
  "error": {
    "code": -32603,
    "message": "Internal error",
    "data": {
      "details": "Failed to read file"
    }
  }
}
```

### 错误处理实现

```typescript
server.setRequestHandler("resources/read", async ({ uri }) => {
  try {
    // 验证 URI
    if (!isValidUri(uri)) {
      throw {
        code: -32602,
        message: "Invalid URI format",
        data: { uri }
      };
    }

    // 检查权限
    if (!hasPermission(uri)) {
      throw {
        code: -32001,
        message: "Permission denied",
        data: { uri }
      };
    }

    // 读取资源
    const content = await readResource(uri);

    return {
      contents: [content]
    };
  } catch (error) {
    if (error.code === "ENOENT") {
      throw {
        code: -32002,
        message: "Resource not found",
        data: { uri }
      };
    }

    throw {
      code: -32603,
      message: "Internal error",
      data: { details: error.message }
    };
  }
});
```

---

## 安全考虑

### URI 验证

```typescript
function isValidUri(uri: string): boolean {
  try {
    const parsed = new URL(uri);

    // 检查 scheme
    const allowedSchemes = ["file", "https", "git", "database"];
    if (!allowedSchemes.includes(parsed.protocol.slice(0, -1))) {
      return false;
    }

    // 检查路径遍历
    if (parsed.pathname.includes("..")) {
      return false;
    }

    return true;
  } catch {
    return false;
  }
}
```

### 访问控制

```typescript
class AccessControl {
  private permissions = new Map<string, Set<string>>();

  grantAccess(userId: string, uri: string) {
    if (!this.permissions.has(userId)) {
      this.permissions.set(userId, new Set());
    }
    this.permissions.get(userId)!.add(uri);
  }

  hasAccess(userId: string, uri: string): boolean {
    const userPerms = this.permissions.get(userId);
    if (!userPerms) return false;

    // 检查精确匹配
    if (userPerms.has(uri)) return true;

    // 检查通配符匹配
    for (const perm of userPerms) {
      if (perm.endsWith("/*") && uri.startsWith(perm.slice(0, -1))) {
        return true;
      }
    }

    return false;
  }
}
```

### 数据编码

```typescript
// 正确编码二进制数据
function encodeBinaryResource(data: Buffer): string {
  return data.toString("base64");
}

// 正确解码二进制数据
function decodeBinaryResource(encoded: string): Buffer {
  return Buffer.from(encoded, "base64");
}
```

[Source: Resources - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-06-18/server/resources)

---

## 总结

### 核心要点

1. **资源提供模式**：静态资源、动态生成（URI 模板）、订阅更新
2. **订阅机制**：实时通知资源变更，支持文件监控
3. **缓存策略**：客户端和服务器端缓存，基于注解优化
4. **URI 设计**：标准 scheme（https、file、git）+ 自定义 scheme
5. **安全控制**：URI 验证、访问控制、数据编码

### 关键约束

- ✅ 资源 URI：必须符合 RFC3986
- ✅ 能力声明：必须声明 resources 能力
- ✅ 订阅支持：可选，需要声明 subscribe 能力
- ✅ 列表变更：可选，需要声明 listChanged 能力
- ✅ 安全验证：必须验证所有 URI 和权限

### 下一步

- 阅读 [03_核心概念_08_测试与调试](./03_核心概念_08_测试与调试.md) 了解测试策略
- 阅读 [07_实战代码_03_API包装器](./07_实战代码_03_API包装器.md) 查看完整实现

---

**参考资源**：
- [Source: Resources - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-06-18/server/resources)
