# 实战代码 01：简单 MCP Server

> **从零构建一个完整可运行的 MCP 天气服务器**

---

## 概述

本文将从零开始构建一个简单但完整的 MCP 服务器，实现天气查询功能。通过这个实战项目，你将学会 MCP 服务器的基本结构、工具注册、stdio 传输和测试方法。

```
简单 MCP Server 核心：
├─ 项目初始化 → TypeScript + MCP SDK
├─ 工具实现 → get_alerts + get_forecast
├─ stdio 传输 → 标准输入输出通信
└─ Claude Desktop 集成 → 测试与调试
```

**本质**：MCP 服务器是一个通过 stdio 与 AI 应用通信的 Node.js 程序，通过注册工具让 LLM 能够调用外部 API 获取实时数据。

[Source: Build an MCP server](https://modelcontextprotocol.io/docs/develop/build-server)
[Source: How to Build a Custom MCP Server with TypeScript](https://www.freecodecamp.org/news/how-to-build-a-custom-mcp-server-with-typescript-a-handbook-for-developers/)

---

## 项目初始化

### 系统要求

- Node.js 16 或更高版本
- npm 或 yarn 包管理器
- TypeScript 基础知识

### 创建项目

```bash
# 创建项目目录
mkdir weather-mcp-server
cd weather-mcp-server

# 初始化 npm 项目
npm init -y

# 安装依赖
npm install @modelcontextprotocol/sdk zod@3
npm install -D @types/node typescript

# 创建源代码目录
mkdir src
touch src/index.ts
```

### 配置 package.json

```json
{
  "name": "weather-mcp-server",
  "version": "1.0.0",
  "type": "module",
  "bin": {
    "weather": "./build/index.js"
  },
  "scripts": {
    "build": "tsc && chmod 755 build/index.js",
    "dev": "tsc --watch"
  },
  "files": ["build"],
  "dependencies": {
    "@modelcontextprotocol/sdk": "^1.0.0",
    "zod": "^3.0.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "typescript": "^5.0.0"
  }
}
```

### 配置 TypeScript

创建 `tsconfig.json`：

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "Node16",
    "moduleResolution": "Node16",
    "outDir": "./build",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules"]
}
```

[Source: Build an MCP server](https://modelcontextprotocol.io/docs/develop/build-server)

---

## 完整服务器实现

### 导入依赖和常量定义

创建 `src/index.ts`：

```typescript
#!/usr/bin/env node

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

// API 配置
const NWS_API_BASE = "https://api.weather.gov";
const USER_AGENT = "weather-mcp-server/1.0";

// 创建 MCP 服务器实例
const server = new McpServer({
  name: "weather",
  version: "1.0.0",
});
```

**关键点**：

- `#!/usr/bin/env node`：shebang，让文件可直接执行
- `McpServer`：MCP 服务器核心类
- `StdioServerTransport`：stdio 传输层
- `zod`：参数验证库

### 类型定义

```typescript
// 天气警报特征
interface AlertFeature {
  properties: {
    event?: string;
    areaDesc?: string;
    severity?: string;
    status?: string;
    headline?: string;
    description?: string;
    instruction?: string;
  };
}

// 警报响应
interface AlertsResponse {
  features: AlertFeature[];
}

// 点位响应
interface PointsResponse {
  properties: {
    forecast?: string;
    forecastHourly?: string;
  };
}

// 预报周期
interface ForecastPeriod {
  name?: string;
  temperature?: number;
  temperatureUnit?: string;
  windSpeed?: string;
  windDirection?: string;
  shortForecast?: string;
  detailedForecast?: string;
}

// 预报响应
interface ForecastResponse {
  properties: {
    periods: ForecastPeriod[];
  };
}
```

### 辅助函数

```typescript
// HTTP 请求辅助函数
async function makeNWSRequest<T>(url: string): Promise<T | null> {
  const headers = {
    "User-Agent": USER_AGENT,
    Accept: "application/geo+json",
  };

  try {
    const response = await fetch(url, { headers });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return (await response.json()) as T;
  } catch (error) {
    console.error("Error making NWS request:", error);
    return null;
  }
}

// 格式化警报信息
function formatAlert(feature: AlertFeature): string {
  const props = feature.properties;
  return [
    `Event: ${props.event || "Unknown"}`,
    `Area: ${props.areaDesc || "Unknown"}`,
    `Severity: ${props.severity || "Unknown"}`,
    `Status: ${props.status || "Unknown"}`,
    `Headline: ${props.headline || "No headline"}`,
    "---",
  ].join("\n");
}

// 格式化预报信息
function formatForecast(period: ForecastPeriod): string {
  return [
    `${period.name || "Unknown"}:`,
    `Temperature: ${period.temperature || "Unknown"}°${period.temperatureUnit || "F"}`,
    `Wind: ${period.windSpeed || "Unknown"} ${period.windDirection || ""}`,
    `${period.shortForecast || "No forecast available"}`,
    "---",
  ].join("\n");
}
```

**关键点**：

- `makeNWSRequest`：通用 HTTP 请求函数，带错误处理
- `formatAlert` / `formatForecast`：格式化 API 响应为可读文本
- 使用 `console.error` 而非 `console.log`（stdio 服务器规则）

### 工具注册：获取天气警报

```typescript
// 注册 get_alerts 工具
server.registerTool(
  "get_alerts",
  {
    description: "Get weather alerts for a US state",
    inputSchema: {
      state: z
        .string()
        .length(2)
        .describe("Two-letter US state code (e.g. CA, NY)"),
    },
  },
  async ({ state }) => {
    const stateCode = state.toUpperCase();
    const alertsUrl = `${NWS_API_BASE}/alerts?area=${stateCode}`;

    console.error(`Fetching alerts for ${stateCode}...`);

    const alertsData = await makeNWSRequest<AlertsResponse>(alertsUrl);

    if (!alertsData) {
      return {
        content: [
          {
            type: "text",
            text: "Failed to retrieve alerts data",
          },
        ],
      };
    }

    const features = alertsData.features || [];

    if (features.length === 0) {
      return {
        content: [
          {
            type: "text",
            text: `No active alerts for ${stateCode}`,
          },
        ],
      };
    }

    const formattedAlerts = features.map(formatAlert);
    const alertsText = `Active alerts for ${stateCode}:\n\n${formattedAlerts.join("\n")}`;

    return {
      content: [
        {
          type: "text",
          text: alertsText,
        },
      ],
    };
  },
);
```

**关键点**：

- `registerTool`：注册工具的核心方法
- `inputSchema`：使用 Zod 定义参数验证规则
- `async ({ state })`：工具执行函数，接收验证后的参数
- 返回格式：`{ content: [{ type: "text", text: "..." }] }`

### 工具注册：获取天气预报

```typescript
// 注册 get_forecast 工具
server.registerTool(
  "get_forecast",
  {
    description: "Get weather forecast for a location",
    inputSchema: {
      latitude: z
        .number()
        .min(-90)
        .max(90)
        .describe("Latitude of the location"),
      longitude: z
        .number()
        .min(-180)
        .max(180)
        .describe("Longitude of the location"),
    },
  },
  async ({ latitude, longitude }) => {
    console.error(`Fetching forecast for ${latitude}, ${longitude}...`);

    // 第一步：获取网格点数据
    const pointsUrl = `${NWS_API_BASE}/points/${latitude.toFixed(4)},${longitude.toFixed(4)}`;
    const pointsData = await makeNWSRequest<PointsResponse>(pointsUrl);

    if (!pointsData) {
      return {
        content: [
          {
            type: "text",
            text: `Failed to retrieve grid point data for coordinates: ${latitude}, ${longitude}. This location may not be supported by the NWS API (only US locations are supported).`,
          },
        ],
      };
    }

    const forecastUrl = pointsData.properties?.forecast;

    if (!forecastUrl) {
      return {
        content: [
          {
            type: "text",
            text: "Failed to get forecast URL from grid point data",
          },
        ],
      };
    }

    // 第二步：获取预报数据
    const forecastData = await makeNWSRequest<ForecastResponse>(forecastUrl);

    if (!forecastData) {
      return {
        content: [
          {
            type: "text",
            text: "Failed to retrieve forecast data",
          },
        ],
      };
    }

    const periods = forecastData.properties?.periods || [];

    if (periods.length === 0) {
      return {
        content: [
          {
            type: "text",
            text: "No forecast periods available",
          },
        ],
      };
    }

    // 格式化预报周期（只显示前 5 个）
    const formattedForecast = periods.slice(0, 5).map(formatForecast);
    const forecastText = `Forecast for ${latitude}, ${longitude}:\n\n${formattedForecast.join("\n")}`;

    return {
      content: [
        {
          type: "text",
          text: forecastText,
        },
      ],
    };
  },
);
```

**关键点**：

- 两步 API 调用：先获取网格点，再获取预报
- 完整的错误处理：每个步骤都检查结果
- 限制输出：只显示前 5 个预报周期

### 启动服务器

```typescript
// 主函数
async function main() {
  // 创建 stdio 传输层
  const transport = new StdioServerTransport();

  // 连接服务器到传输层
  await server.connect(transport);

  // 日志输出到 stderr（不能用 stdout）
  console.error("Weather MCP Server running on stdio");
  console.error("Available tools: get_alerts, get_forecast");
}

// 启动服务器
main().catch((error) => {
  console.error("Fatal error in main():", error);
  process.exit(1);
});
```

**关键点**：

- `StdioServerTransport`：使用标准输入输出通信
- `server.connect(transport)`：连接服务器到传输层
- 错误处理：捕获并记录致命错误

[Source: Build an MCP server](https://modelcontextprotocol.io/docs/develop/build-server)

---

## 构建和测试

### 构建项目

```bash
# 构建 TypeScript
npm run build

# 验证构建输出
ls -la build/
# 应该看到 index.js 文件
```

### 本地测试

```bash
# 直接运行（用于调试）
node build/index.js

# 应该看到：
# Weather MCP Server running on stdio
# Available tools: get_alerts, get_forecast
```

### 配置 Claude Desktop

编辑 `~/Library/Application Support/Claude/claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "weather": {
      "command": "node",
      "args": [
        "/ABSOLUTE/PATH/TO/weather-mcp-server/build/index.js"
      ]
    }
  }
}
```

**重要**：

- 使用绝对路径（运行 `pwd` 获取）
- Windows 用户使用双反斜杠或正斜杠
- 保存后重启 Claude Desktop

### 测试工具调用

在 Claude Desktop 中测试：

```
1. 点击 "+" 图标
2. 悬停在 "Connectors" 菜单
3. 应该看到 "weather" 服务器
4. 测试查询：
   - "What's the weather in Sacramento?"
   - "What are the active weather alerts in Texas?"
   - "Get the forecast for latitude 37.7749, longitude -122.4194"
```

[Source: Build an MCP server](https://modelcontextprotocol.io/docs/develop/build-server)

---

## 调试技巧

### 查看日志

```bash
# macOS: 查看 Claude Desktop 日志
tail -n 20 -f ~/Library/Logs/Claude/mcp*.log

# 查看服务器特定日志
tail -f ~/Library/Logs/Claude/mcp-server-weather.log
```

### 常见问题

**问题 1：服务器未显示在 Claude Desktop**

```bash
# 检查配置文件语法
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json | jq .

# 检查路径是否正确
ls -la /ABSOLUTE/PATH/TO/weather-mcp-server/build/index.js

# 完全重启 Claude Desktop（Cmd+Q）
```

**问题 2：工具调用失败**

```typescript
// 在代码中添加更多日志
console.error(`Request URL: ${alertsUrl}`);
console.error(`Response:`, JSON.stringify(alertsData, null, 2));
```

**问题 3：stdio 通信错误**

```typescript
// ❌ 错误：使用 console.log
console.log("Debug message");

// ✅ 正确：使用 console.error
console.error("Debug message");
```

---

## 最佳实践

### 日志规范

```typescript
// ✅ 正确：所有日志输出到 stderr
console.error("Server started");
console.error("Processing request:", params);

// ❌ 错误：stdout 会破坏 JSON-RPC 通信
console.log("Server started");
```

### 错误处理

```typescript
// ✅ 完整的错误处理
async function safeFetch(url: string) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Fetch error:", error);
    return null;
  }
}

// ❌ 缺少错误处理
async function unsafeFetch(url: string) {
  const response = await fetch(url);
  return await response.json();
}
```

### 参数验证

```typescript
// ✅ 使用 Zod 验证
inputSchema: {
  state: z
    .string()
    .length(2)
    .regex(/^[A-Z]{2}$/)
    .describe("Two-letter US state code"),
}

// ✅ 运行时验证
async ({ state }) => {
  const stateCode = state.toUpperCase();
  if (!/^[A-Z]{2}$/.test(stateCode)) {
    return {
      content: [{
        type: "text",
        text: "Invalid state code format"
      }]
    };
  }
  // ...
}
```

### 响应格式

```typescript
// ✅ 标准响应格式
return {
  content: [
    {
      type: "text",
      text: "Response text here"
    }
  ]
};

// ✅ 错误响应
return {
  content: [
    {
      type: "text",
      text: "Error: Failed to fetch data"
    }
  ],
  isError: true
};
```

[Source: Build an MCP server](https://modelcontextprotocol.io/docs/develop/build-server)

---

## 扩展练习

### 添加更多工具

```typescript
// 练习 1：添加当前天气工具
server.registerTool(
  "get_current_weather",
  {
    description: "Get current weather conditions",
    inputSchema: {
      latitude: z.number(),
      longitude: z.number(),
    },
  },
  async ({ latitude, longitude }) => {
    // 实现当前天气查询
    // 提示：使用 /points/{lat},{lon}/stations/observations/latest
  },
);

// 练习 2：添加多日预报工具
server.registerTool(
  "get_extended_forecast",
  {
    description: "Get 7-day forecast",
    inputSchema: {
      latitude: z.number(),
      longitude: z.number(),
      days: z.number().min(1).max(7).default(7),
    },
  },
  async ({ latitude, longitude, days }) => {
    // 实现多日预报
  },
);
```

### 添加缓存

```typescript
// 简单的内存缓存
const cache = new Map<string, { data: any; timestamp: number }>();
const CACHE_TTL = 5 * 60 * 1000; // 5 分钟

async function cachedFetch<T>(url: string): Promise<T | null> {
  const cached = cache.get(url);

  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    console.error("Cache hit:", url);
    return cached.data as T;
  }

  const data = await makeNWSRequest<T>(url);

  if (data) {
    cache.set(url, { data, timestamp: Date.now() });
  }

  return data;
}
```

---

## 总结

### 核心要点

1. **项目结构**：TypeScript + MCP SDK + Zod 验证
2. **工具注册**：`server.registerTool()` 注册工具
3. **stdio 传输**：使用 `StdioServerTransport` 通信
4. **日志规范**：所有日志输出到 stderr
5. **错误处理**：完整的 try-catch 和空值检查

### 关键约束

- ✅ 使用 `console.error` 而非 `console.log`
- ✅ 返回标准格式：`{ content: [{ type: "text", text: "..." }] }`
- ✅ 使用 Zod 验证参数
- ✅ 完整的错误处理
- ✅ 绝对路径配置 Claude Desktop

### 下一步

- 阅读 [07_实战代码_02_数据库连接器](./07_实战代码_02_数据库连接器.md) 学习数据库集成
- 阅读 [07_实战代码_03_API包装器](./07_实战代码_03_API包装器.md) 学习 API 包装

---

**参考资源**：
- [Source: Build an MCP server](https://modelcontextprotocol.io/docs/develop/build-server)
- [Source: How to Build a Custom MCP Server with TypeScript](https://www.freecodecamp.org/news/how-to-build-a-custom-mcp-server-with-typescript-a-handbook-for-developers/)
- [Source: MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
