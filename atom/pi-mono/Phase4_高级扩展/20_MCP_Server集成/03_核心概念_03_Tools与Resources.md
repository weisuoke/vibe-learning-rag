# 核心概念 03：Tools 与 Resources

> **深入理解 MCP 的核心功能：工具定义、资源管理和参数验证**

---

## 概述

MCP 协议的核心价值在于两个关键抽象：**Tools（工具）**和 **Resources（资源）**。

```
MCP 核心抽象：
├─ Tools（工具）→ 可执行的操作（如查询数据库、调用 API）
└─ Resources（资源）→ 可访问的数据（如文件内容、配置信息）
```

**本质区别**：
- **Tools 是动词**：执行动作，改变状态
- **Resources 是名词**：提供数据，作为上下文

[Source: Tools - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)
[Source: Resources - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/server/resources)

---

## Tools（工具）

### 用户交互模型

Tools 在 MCP 中设计为**模型控制**（model-controlled）：

```
工具发现与调用流程：
1. LLM 查询可用工具（tools/list）
2. LLM 根据上下文理解选择合适的工具
3. LLM 自主决定调用哪个工具
4. 人类在循环中确认（human-in-the-loop）
```

**安全要求**：

为了信任和安全，**应该**始终有人类在循环中，能够拒绝工具调用。

应用程序**应该**：
- 提供 UI 清楚显示哪些工具暴露给 AI 模型
- 在工具被调用时插入清晰的视觉指示器
- 向用户呈现确认提示，确保人类在循环中

[Source: Tools - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)

### 能力声明

支持工具的服务器**必须**声明 `tools` 能力：

```json
{
  "capabilities": {
    "tools": {
      "listChanged": true
    }
  }
}
```

**能力说明**：
- `listChanged`: 服务器是否会在可用工具列表变更时发出通知

### 协议消息

#### 列出工具（tools/list）

**请求**：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {
    "cursor": "optional-cursor-value"
  }
}
```

**响应**：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {
        "name": "search_users",
        "title": "User Search Tool",
        "description": "Search user profiles by name, email, or ID. Returns basic user information including name, email, and registration date. Use this when you need to find specific users or verify user existence.",
        "inputSchema": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "Search term: user name (partial match), email (exact), or user ID (exact)"
            },
            "limit": {
              "type": "number",
              "description": "Maximum results (default: 10, max: 100)",
              "default": 10
            }
          },
          "required": ["query"]
        },
        "icons": [
          {
            "src": "https://example.com/search-icon.png",
            "mimeType": "image/png",
            "sizes": ["48x48"]
          }
        ]
      }
    ]
  }
}
```

[Source: Tools - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)

#### 调用工具（tools/call）

**请求**：

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "search_users",
    "arguments": {
      "query": "john",
      "limit": 5
    }
  }
}
```

**响应**：

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Found 2 users:\n1. John Doe (john.doe@example.com)\n2. John Smith (john.smith@example.com)"
      }
    ],
    "isError": false
  }
}
```

#### 工具列表变更通知

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/tools/list_changed"
}
```

### 工具定义

**Tool 数据类型**：

```typescript
interface Tool {
  name: string;              // 唯一标识符
  title?: string;            // 可选的人类可读名称
  description: string;       // 功能描述（给 AI 看）
  icons?: Icon[];            // 可选的图标数组
  inputSchema: JSONSchema;   // 参数定义（JSON Schema）
  outputSchema?: JSONSchema; // 可选的输出验证
  annotations?: object;      // 可选的行为属性
}
```

**工具名称规范**：

- **应该**在 1-128 个字符之间（包含）
- **应该**被视为区分大小写
- **应该**仅包含以下字符：大小写 ASCII 字母（A-Z, a-z）、数字（0-9）、下划线（_）、连字符（-）、点（.）
- **不应该**包含空格、逗号或其他特殊字符
- **应该**在服务器内唯一

**有效工具名称示例**：
- `getUser`
- `DATA_EXPORT_v2`
- `admin.tools.list`

[Source: Tools - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)

### 输入 Schema 设计

**JSON Schema 规范**：

- 遵循 JSON Schema 使用指南
- 默认为 2020-12（如果没有 `$schema` 字段）
- **必须**是有效的 JSON Schema 对象（不能为 null）

**无参数工具的正确定义**：

```json
// ✅ 推荐：显式仅接受空对象
{
  "name": "get_current_time",
  "description": "Returns the current server time",
  "inputSchema": {
    "type": "object",
    "additionalProperties": false
  }
}

// ✅ 可接受：接受任何对象
{
  "name": "get_current_time",
  "description": "Returns the current server time",
  "inputSchema": {
    "type": "object"
  }
}
```

**参数验证示例**：

```json
{
  "name": "create_user",
  "description": "Create a new user account",
  "inputSchema": {
    "type": "object",
    "properties": {
      "email": {
        "type": "string",
        "format": "email",
        "description": "User email address"
      },
      "age": {
        "type": "integer",
        "minimum": 18,
        "maximum": 120,
        "description": "User age (must be 18+)"
      },
      "role": {
        "type": "string",
        "enum": ["user", "admin", "moderator"],
        "default": "user",
        "description": "User role"
      }
    },
    "required": ["email"]
  }
}
```

### 工具结果类型

工具结果可以包含多种内容类型：

#### 文本内容

```json
{
  "type": "text",
  "text": "Tool result text"
}
```

#### 图像内容

```json
{
  "type": "image",
  "data": "base64-encoded-data",
  "mimeType": "image/png",
  "annotations": {
    "audience": ["user"],
    "priority": 0.9
  }
}
```

#### 资源链接

```json
{
  "type": "resource_link",
  "uri": "file:///project/src/main.rs",
  "name": "main.rs",
  "description": "Primary application entry point",
  "mimeType": "text/x-rust"
}
```

#### 结构化内容

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"temperature\": 22.5, \"conditions\": \"Partly cloudy\"}"
      }
    ],
    "structuredContent": {
      "temperature": 22.5,
      "conditions": "Partly cloudy",
      "humidity": 65
    }
  }
}
```

[Source: Tools - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)

### 输出 Schema 验证

工具可以提供输出 schema 用于结构化结果验证：

```json
{
  "name": "get_weather_data",
  "inputSchema": { /* ... */ },
  "outputSchema": {
    "type": "object",
    "properties": {
      "temperature": {
        "type": "number",
        "description": "Temperature in celsius"
      },
      "conditions": {
        "type": "string",
        "description": "Weather conditions"
      }
    },
    "required": ["temperature", "conditions"]
  }
}
```

**要求**：
- 如果提供了输出 schema，服务器**必须**提供符合此 schema 的结构化结果
- 客户端**应该**根据此 schema 验证结构化结果

### 错误处理

**两种错误报告机制**：

1. **协议错误**：标准 JSON-RPC 错误
   - 未知工具
   - 格式错误的请求
   - 服务器错误

2. **工具执行错误**：在结果中使用 `isError: true`
   - API 失败
   - 输入验证错误
   - 业务逻辑错误

**示例**：

```json
// 协议错误
{
  "jsonrpc": "2.0",
  "id": 3,
  "error": {
    "code": -32602,
    "message": "Unknown tool: invalid_tool_name"
  }
}

// 工具执行错误
{
  "jsonrpc": "2.0",
  "id": 4,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Invalid date: must be in the future. Current date is 2026-02-21."
      }
    ],
    "isError": true
  }
}
```

**处理建议**：
- 客户端**应该**将工具执行错误提供给 LLM 以启用自我纠正
- 客户端**可以**将协议错误提供给 LLM，但这些错误不太可能导致成功恢复

[Source: Tools - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)

---

## Resources（资源）

### 用户交互模型

Resources 在 MCP 中设计为**应用驱动**（application-driven）：

```
资源使用模式：
├─ UI 元素显示（树形或列表视图）
├─ 搜索和过滤可用资源
├─ 基于启发式或 AI 模型选择的自动上下文包含
└─ 任何适合应用需求的接口模式
```

协议本身不强制任何特定的用户交互模型。

[Source: Resources - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/server/resources)

### 能力声明

支持资源的服务器**必须**声明 `resources` 能力：

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

**能力说明**：
- `subscribe`: 客户端是否可以订阅单个资源的变更通知
- `listChanged`: 服务器是否会在可用资源列表变更时发出通知

**可选组合**：

```json
// 两者都不支持
{ "capabilities": { "resources": {} } }

// 仅支持订阅
{ "capabilities": { "resources": { "subscribe": true } } }

// 仅支持列表变更通知
{ "capabilities": { "resources": { "listChanged": true } } }
```

### 协议消息

#### 列出资源（resources/list）

**请求**：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "resources/list",
  "params": {
    "cursor": "optional-cursor-value"
  }
}
```

**响应**：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resources": [
      {
        "uri": "file:///project/src/main.rs",
        "name": "main.rs",
        "title": "Rust Application Main File",
        "description": "Primary application entry point",
        "mimeType": "text/x-rust",
        "icons": [
          {
            "src": "https://example.com/rust-icon.png",
            "mimeType": "image/png",
            "sizes": ["48x48"]
          }
        ]
      }
    ]
  }
}
```

#### 读取资源（resources/read）

**请求**：

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "resources/read",
  "params": {
    "uri": "file:///project/src/main.rs"
  }
}
```

**响应**：

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "contents": [
      {
        "uri": "file:///project/src/main.rs",
        "mimeType": "text/x-rust",
        "text": "fn main() {\n  println!(\"Hello world!\");\n}"
      }
    ]
  }
}
```

[Source: Resources - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/server/resources)

#### 资源模板（Resource Templates）

资源模板允许服务器使用 URI 模板暴露参数化资源：

**请求**：

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "resources/templates/list"
}
```

**响应**：

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "resourceTemplates": [
      {
        "uriTemplate": "file:///{path}",
        "name": "Project Files",
        "title": "📁 Project Files",
        "description": "Access files in the project directory",
        "mimeType": "application/octet-stream"
      }
    ]
  }
}
```

#### 订阅资源变更

**订阅请求**：

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

**更新通知**：

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": {
    "uri": "file:///project/src/main.rs"
  }
}
```

### 资源定义

**Resource 数据类型**：

```typescript
interface Resource {
  uri: string;               // 唯一标识符
  name: string;              // 资源名称
  title?: string;            // 可选的人类可读名称
  description?: string;      // 可选的描述
  icons?: Icon[];            // 可选的图标数组
  mimeType?: string;         // 可选的 MIME 类型
  size?: number;             // 可选的字节大小
}
```

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

### 注解（Annotations）

资源支持可选注解，提供关于如何使用或显示资源的提示：

```typescript
interface Annotations {
  audience?: ("user" | "assistant")[];  // 目标受众
  priority?: number;                     // 重要性（0.0-1.0）
  lastModified?: string;                 // ISO 8601 时间戳
}
```

**示例**：

```json
{
  "uri": "file:///project/README.md",
  "name": "README.md",
  "mimeType": "text/markdown",
  "annotations": {
    "audience": ["user"],
    "priority": 0.8,
    "lastModified": "2026-02-21T15:00:58Z"
  }
}
```

**用途**：
- 根据目标受众过滤资源
- 优先选择哪些资源包含在上下文中
- 显示修改时间或按最近性排序

[Source: Resources - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/server/resources)

### 常见 URI Scheme

#### https://

用于表示 Web 上可用的资源。服务器**应该**仅在客户端能够自行从 Web 获取和加载资源时使用此 scheme。

#### file://

用于标识类似文件系统的资源。但是，资源不需要映射到实际的物理文件系统。

MCP 服务器**可以**使用 XDG MIME 类型（如 `inode/directory`）标识 `file://` 资源，以表示没有标准 MIME 类型的非常规文件（如目录）。

#### git://

Git 版本控制集成。

#### 自定义 URI Scheme

自定义 URI scheme **必须**符合 RFC3986。

---

## Tools vs Resources 对比

### 核心区别

| 维度 | Tools | Resources |
|------|-------|-----------|
| **本质** | 操作（动词） | 数据（名词） |
| **用途** | 执行动作 | 提供上下文 |
| **交互模型** | 模型控制 | 应用驱动 |
| **示例** | 查询数据库、发送邮件 | 文件内容、配置信息 |
| **变更通知** | tools/list_changed | resources/list_changed |
| **订阅** | 不支持 | 支持（可选） |
| **结果类型** | 多种（文本、图像、结构化） | 文本或二进制 |
| **Schema** | inputSchema + outputSchema | 无（通过 MIME 类型） |

### 使用场景

**使用 Tools 当**：
- ✅ 需要执行操作（查询、创建、更新、删除）
- ✅ 需要与外部系统交互（API、数据库）
- ✅ 需要 AI 自主决策何时调用
- ✅ 需要参数验证和输出验证

**使用 Resources 当**：
- ✅ 需要提供静态或半静态数据
- ✅ 需要为 AI 提供上下文信息
- ✅ 需要应用控制何时加载
- ✅ 需要订阅数据变更

---

## 安全考虑

### Tools 安全

服务器**必须**：
- 验证所有工具输入
- 实现适当的访问控制
- 限制工具调用速率
- 清理工具输出

客户端**应该**：
- 在敏感操作上提示用户确认
- 在调用服务器前向用户显示工具输入
- 在传递给 LLM 前验证工具结果
- 实现工具调用超时
- 记录工具使用以供审计

### Resources 安全

服务器**必须**：
- 验证所有资源 URI
- 为敏感资源实现访问控制
- 正确编码二进制数据
- 在操作前检查资源权限

[Source: Tools - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)
[Source: Resources - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/server/resources)

---

## 实现最佳实践

### 工具描述的黄金法则

```typescript
// ❌ 错误：描述过于简单
{
  name: "query",
  description: "Query data"
}

// ✅ 正确：描述详细清晰
{
  name: "search_user_profiles",
  description: "Searches user profiles by name, email, or ID. Returns basic user information including name, email, and registration date. Use this when you need to find specific users or verify user existence. Maximum 100 results per query."
}
```

### 参数验证最佳实践

```typescript
// ✅ 正确：详细的参数描述和验证
{
  inputSchema: {
    type: "object",
    properties: {
      query: {
        type: "string",
        minLength: 1,
        maxLength: 100,
        description: "Search term: user name (partial match), email (exact), or user ID (exact)"
      },
      limit: {
        type: "integer",
        minimum: 1,
        maximum: 100,
        default: 10,
        description: "Maximum number of results (default: 10, max: 100)"
      }
    },
    required: ["query"]
  }
}
```

### 资源 URI 设计

```typescript
// ✅ 正确：标准化的 URI
"file:///project/src/main.rs"
"https://example.com/api/data"
"git://repo/branch/file"
"custom://namespace/resource"

// ❌ 错误：非标准 URI
"file.txt"
"/path/to/file"
"relative/path"
```

---

## 总结

### 核心要点

1. **Tools 是操作，Resources 是数据**：清晰的职责分离
2. **详细描述至关重要**：AI 依赖描述来理解和选择工具
3. **参数验证必不可少**：使用 JSON Schema 确保输入正确
4. **安全始终第一**：人类在循环中，验证所有输入输出
5. **注解提供元数据**：帮助客户端更好地使用资源

### 关键约束

- ✅ 工具名称：1-128 字符，区分大小写，仅字母数字和 _-.
- ✅ 输入 Schema：必须是有效的 JSON Schema 对象
- ✅ 资源 URI：必须符合 RFC3986
- ✅ 人类确认：敏感操作需要用户批准

### 下一步

- 阅读 [03_核心概念_04_安全与认证](./03_核心概念_04_安全与认证.md) 了解安全机制
- 阅读 [07_实战代码_01_简单MCP_Server](./07_实战代码_01_简单MCP_Server.md) 查看完整实现

---

**参考资源**：
- [Source: Tools - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)
- [Source: Resources - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/server/resources)
