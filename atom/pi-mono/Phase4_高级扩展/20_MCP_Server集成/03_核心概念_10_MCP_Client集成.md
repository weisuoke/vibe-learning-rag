# 核心概念 10：MCP Client 集成

> **深入理解 MCP Client 的实现模式、连接管理和生命周期处理**

---

## 概述

MCP Client 是连接 MCP Server 的客户端实现，负责管理服务器连接、工具发现、请求处理和生命周期管理。

```
MCP Client 核心能力：
├─ 服务器连接 → stdio、HTTP 传输管理
├─ 工具发现 → 列出可用工具和资源
├─ 请求处理 → 工具调用和结果处理
└─ 生命周期管理 → 初始化、运行、清理
```

**本质**：MCP Client 是 AI 应用与 MCP Server 之间的桥梁，通过标准化的协议实现工具发现、调用和结果处理，让 LLM 能够安全地访问外部工具和数据源。

[Source: Build an MCP client](https://modelcontextprotocol.io/docs/develop/build-client)

---

## 客户端架构

### 基本结构

**Python 实现**：

```python
import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

class MCPClient:
    def __init__(self):
        # 初始化会话和客户端对象
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
```

**TypeScript 实现**:

```typescript
import { Anthropic } from "@anthropic-ai/sdk";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

class MCPClient {
  private mcp: Client;
  private anthropic: Anthropic;
  private transport: StdioClientTransport | null = null;
  private tools: Tool[] = [];

  constructor() {
    this.anthropic = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY,
    });
    this.mcp = new Client({
      name: "mcp-client-cli",
      version: "1.0.0"
    });
  }
}
```

[Source: Build an MCP client](https://modelcontextprotocol.io/docs/develop/build-client)

---

## 服务器连接管理

### stdio 传输连接

**Python 实现**：

```python
async def connect_to_server(self, server_script_path: str):
    """连接到 MCP 服务器

    Args:
        server_script_path: 服务器脚本路径（.py 或 .js）
    """
    is_python = server_script_path.endswith('.py')
    is_js = server_script_path.endswith('.js')
    if not (is_python or is_js):
        raise ValueError("Server script must be a .py or .js file")

    command = "python" if is_python else "node"
    server_params = StdioServerParameters(
        command=command,
        args=[server_script_path],
        env=None
    )

    # 建立 stdio 传输
    stdio_transport = await self.exit_stack.enter_async_context(
        stdio_client(server_params)
    )
    self.stdio, self.write = stdio_transport

    # 创建客户端会话
    self.session = await self.exit_stack.enter_async_context(
        ClientSession(self.stdio, self.write)
    )

    # 初始化会话
    await self.session.initialize()

    # 列出可用工具
    response = await self.session.list_tools()
    tools = response.tools
    print("Connected to server with tools:", [tool.name for tool in tools])
```

**TypeScript 实现**：

```typescript
async connectToServer(serverScriptPath: string) {
  try {
    const isJs = serverScriptPath.endsWith(".js");
    const isPy = serverScriptPath.endsWith(".py");
    if (!isJs && !isPy) {
      throw new Error("Server script must be a .js or .py file");
    }

    const command = isPy
      ? process.platform === "win32" ? "python" : "python3"
      : process.execPath;

    this.transport = new StdioClientTransport({
      command,
      args: [serverScriptPath],
    });

    await this.mcp.connect(this.transport);

    const toolsResult = await this.mcp.listTools();
    this.tools = toolsResult.tools.map((tool) => ({
      name: tool.name,
      description: tool.description,
      input_schema: tool.inputSchema,
    }));

    console.log(
      "Connected to server with tools:",
      this.tools.map(({ name }) => name)
    );
  } catch (e) {
    console.log("Failed to connect to MCP server: ", e);
    throw e;
  }
}
```

[Source: Build an MCP client](https://modelcontextprotocol.io/docs/develop/build-client)

---

## 查询处理逻辑

### 工具调用流程

**Python 实现**：

```python
async def process_query(self, query: str) -> str:
    """使用 Claude 和可用工具处理查询"""
    messages = [{"role": "user", "content": query}]

    # 获取可用工具
    response = await self.session.list_tools()
    available_tools = [{
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.inputSchema
    } for tool in response.tools]

    # 初始 Claude API 调用
    response = self.anthropic.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=messages,
        tools=available_tools
    )

    # 处理响应和工具调用
    final_text = []
    assistant_message_content = []

    for content in response.content:
        if content.type == 'text':
            final_text.append(content.text)
            assistant_message_content.append(content)
        elif content.type == 'tool_use':
            tool_name = content.name
            tool_args = content.input

            # 执行工具调用
            result = await self.session.call_tool(tool_name, tool_args)
            final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

            assistant_message_content.append(content)
            messages.append({
                "role": "assistant",
                "content": assistant_message_content
            })
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": content.id,
                    "content": result.content
                }]
            })

            # 获取 Claude 的下一个响应
            response = self.anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=messages,
                tools=available_tools
            )

            final_text.append(response.content[0].text if response.content else "")

    return "\n".join(final_text)
```

[Source: Build an MCP client](https://modelcontextprotocol.io/docs/develop/build-client)

---

## 最佳实践

### 1. 错误处理

**连接错误处理**：

```python
async def connect_to_server(self, server_script_path: str):
    try:
        # 连接逻辑
        await self.session.initialize()
    except FileNotFoundError:
        raise ValueError(f"Server script not found: {server_script_path}")
    except ConnectionRefusedError:
        raise ValueError("Failed to connect to server")
    except Exception as e:
        raise ValueError(f"Connection error: {str(e)}")
```

**工具调用错误处理**：

```python
async def process_query(self, query: str) -> str:
    try:
        result = await self.session.call_tool(tool_name, tool_args)
        return result.content
    except Exception as e:
        return f"Tool execution failed: {str(e)}"
```

### 2. 资源管理

**使用 AsyncExitStack 管理资源**：

```python
class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()

    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()

async def main():
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()
```

### 3. 安全考虑

**API 密钥管理**：

```python
# ✅ 正确：使用环境变量
from dotenv import load_dotenv
load_dotenv()

anthropic = Anthropic()  # 自动从 ANTHROPIC_API_KEY 读取

# ❌ 错误：硬编码 API 密钥
anthropic = Anthropic(api_key="sk-ant-...")
```

**服务器脚本验证**：

```python
async def connect_to_server(self, server_script_path: str):
    # 验证文件类型
    if not (server_script_path.endswith('.py') or server_script_path.endswith('.js')):
        raise ValueError("Server script must be a .py or .js file")

    # 验证文件存在
    if not os.path.exists(server_script_path):
        raise FileNotFoundError(f"Server script not found: {server_script_path}")
```

[Source: Implementing MCP: Tips and Pitfalls](https://nearform.com/digital-community/implementing-model-context-protocol-mcp-tips-tricks-and-pitfalls/)

---

## 常见问题排查

### 服务器路径问题

**问题**：`FileNotFoundError` 或 `Connection refused`

**解决方案**：

```bash
# 使用相对路径
python client.py ./server/weather.py

# 使用绝对路径
python client.py /Users/username/projects/mcp-server/weather.py

# Windows 路径（两种格式都可以）
python client.py C:/projects/mcp-server/weather.py
python client.py C:\\projects\\mcp-server\\weather.py
```

### 响应超时

**问题**：首次响应可能需要 30 秒

**原因**：
- 服务器初始化
- Claude 处理查询
- 工具执行

**解决方案**：

```python
# 增加超时时间
response = self.anthropic.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1000,
    messages=messages,
    tools=available_tools,
    timeout=60.0  # 60 秒超时
)
```

### 常见错误消息

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `FileNotFoundError` | 服务器路径错误 | 检查服务器路径 |
| `Connection refused` | 服务器未运行 | 确保服务器正在运行 |
| `Tool execution failed` | 工具环境变量未设置 | 验证所需环境变量 |
| `Timeout error` | 请求超时 | 增加客户端超时配置 |

[Source: Build an MCP client](https://modelcontextprotocol.io/docs/develop/build-client)

---

## 高级模式

### 多服务器管理

**元服务器模式**：

```python
class MultiServerClient:
    def __init__(self):
        self.servers = {}
        self.anthropic = Anthropic()

    async def connect_to_servers(self, server_configs):
        """连接到多个服务器"""
        for name, config in server_configs.items():
            client = MCPClient()
            await client.connect_to_server(config['path'])
            self.servers[name] = client

    async def route_query(self, query: str, server_name: str):
        """路由查询到特定服务器"""
        if server_name not in self.servers:
            raise ValueError(f"Server {server_name} not found")

        return await self.servers[server_name].process_query(query)
```

### 流式响应

**实现进度更新**：

```python
async def process_query_streaming(self, query: str):
    """流式处理查询"""
    messages = [{"role": "user", "content": query}]

    with self.anthropic.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=messages,
        tools=available_tools
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
```

[Source: Implementing MCP: Tips and Pitfalls](https://nearform.com/digital-community/implementing-model-context-protocol-mcp-tips-tricks-and-pitfalls/)

---

## 总结

### 核心要点

1. **客户端架构**：使用 ClientSession 管理服务器连接
2. **连接管理**：支持 stdio 和 HTTP 传输
3. **查询处理**：工具发现、调用和结果处理
4. **错误处理**：连接错误、工具执行错误、超时处理
5. **资源管理**：使用 AsyncExitStack 确保资源清理

### 关键约束

- ✅ 使用环境变量存储 API 密钥
- ✅ 验证服务器脚本路径和类型
- ✅ 实现完整的错误处理
- ✅ 使用 AsyncExitStack 管理资源
- ✅ 处理首次响应超时

### 下一步

- 阅读 [03_核心概念_11_工具包装与转换](./03_核心概念_11_工具包装与转换.md) 了解工具包装
- 阅读 [07_实战代码_06_MCP_Client集成](./07_实战代码_06_MCP_Client集成.md) 查看完整实现

---

**参考资源**：
- [Source: Build an MCP client](https://modelcontextprotocol.io/docs/develop/build-client)
- [Source: Implementing MCP: Tips and Pitfalls](https://nearform.com/digital-community/implementing-model-context-protocol-mcp-tips-tricks-and-pitfalls/)
