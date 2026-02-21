# Extensions 扩展开发基础 - 核心概念 01：TypeScript 扩展基础

> 深入理解 Extensions 的 TypeScript 模块结构、类型系统和加载机制

---

## 概述

本文档深入讲解 Extensions 的 TypeScript 基础，包括：
- Extension 模块的结构和工厂模式
- TypeScript 类型系统和类型安全
- 模块依赖和虚拟模块
- jiti 动态加载机制

---

## 1. Extension 模块结构

### 1.1 基本结构

**Extension 是一个 TypeScript 模块，导出一个默认函数：**

```typescript
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

// 默认导出函数（工厂函数）
export default function (pi: ExtensionAPI) {
  // 扩展初始化代码
  console.log("Extension loaded");

  // 注册功能
  pi.registerTool({ ... });
  pi.registerCommand("cmd", { ... });

  // 监听事件
  pi.on("tool_call", async (event, ctx) => { ... });
}
```

**关键要素：**

1. **默认导出**：必须使用 `export default`
2. **函数签名**：`(pi: ExtensionAPI) => void`
3. **同步执行**：函数体同步执行（但可以注册异步处理器）
4. **无返回值**：函数不需要返回值

### 1.2 工厂模式

**Extension 使用工厂模式：**

```typescript
// 工厂函数模式
export default function extensionFactory(pi: ExtensionAPI) {
  // 1. 私有状态（闭包）
  let privateState = {};

  // 2. 私有函数
  const privateHelper = () => {
    // 只在扩展内部可见
  };

  // 3. 注册公共接口
  pi.registerTool({
    name: "my_tool",
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      // 可以访问私有状态和函数
      privateHelper();
      return { content: [...], details: {} };
    },
  });

  // 4. 可选：返回清理函数
  return () => {
    // 清理资源（如果需要）
    console.log("Extension unloaded");
  };
}
```

**工厂模式的优势：**

1. **封装**：私有状态和函数不会泄露到全局
2. **闭包**：工具和事件处理器可以访问私有状态
3. **清理**：可以返回清理函数（用于热重载）
4. **模块化**：每个 Extension 是独立的模块

### 1.3 Extension 生命周期

```
加载阶段
  ↓
1. 发现 Extension 文件
   - 扫描 ~/.pi/agent/extensions/
   - 扫描 .pi/extensions/
   - 或通过 --extension 参数指定
  ↓
2. 使用 jiti 加载 TypeScript 文件
   - 动态编译 TypeScript
   - 执行模块代码
  ↓
3. 调用默认导出函数
   - 传入 ExtensionAPI
   - 执行初始化代码
  ↓
4. 注册功能
   - 工具、命令、快捷键
   - 事件监听器
  ↓
运行阶段
  ↓
5. 响应事件
   - 工具调用
   - Session 事件
   - 用户输入
  ↓
卸载阶段（热重载）
  ↓
6. 调用清理函数（如果有）
   - 清理资源
   - 移除监听器
  ↓
7. 重新加载
   - 清除模块缓存
   - 重新执行加载阶段
```

**代码示例：**

```typescript
export default function (pi: ExtensionAPI) {
  console.log("[Lifecycle] Extension loading");

  // 初始化阶段
  let state = { initialized: true };

  // 注册阶段
  pi.on("session_start", async (event, ctx) => {
    console.log("[Lifecycle] Session started");
  });

  // 清理阶段（可选）
  return () => {
    console.log("[Lifecycle] Extension unloading");
    state = null;
  };
}
```

---

## 2. TypeScript 类型系统

### 2.1 ExtensionAPI 类型定义

**完整的 ExtensionAPI 接口：**

```typescript
export interface ExtensionAPI {
  // 工具注册
  registerTool(definition: ToolDefinition): void;

  // 命令注册
  registerCommand(name: string, config: CommandConfig): void;

  // 快捷键注册
  registerShortcut(key: KeyId, handler: ShortcutHandler): void;

  // 标志注册
  registerFlag(name: string, config: FlagConfig): void;

  // 事件监听
  on<E extends EventType>(
    event: E,
    handler: EventHandler<E>
  ): void;

  off<E extends EventType>(
    event: E,
    handler: EventHandler<E>
  ): void;

  // Provider 管理
  registerProvider(config: ProviderConfig): void;

  // 消息渲染
  registerMessageRenderer(
    matcher: MessageMatcher,
    renderer: MessageRenderer
  ): void;

  // 查询 API
  getCommands(): SlashCommandInfo[];

  // 事件总线
  events: EventEmitter;

  // 发送消息
  sendUserMessage(text: string): Promise<void>;
}
```

### 2.2 核心类型

**ToolDefinition 类型：**

```typescript
export interface ToolDefinition {
  name: string;
  label?: string;
  description: string;
  parameters: TSchema;  // TypeBox schema

  execute(
    toolCallId: string,
    params: unknown,
    signal: AbortSignal,
    onUpdate: (update: ToolUpdate) => void,
    ctx: ExtensionContext
  ): Promise<ToolResult>;

  // 可选：自定义渲染
  renderCall?(
    args: unknown,
    theme: Theme
  ): Component;

  renderResult?(
    result: ToolResult,
    state: { expanded: boolean },
    theme: Theme
  ): Component;
}
```

**EventHandler 类型：**

```typescript
export type EventHandler<E extends EventType> = (
  event: EventData<E>,
  ctx: ExtensionContext
) => Promise<EventResult<E> | void>;

// 事件结果类型
export type EventResult<E extends EventType> =
  | { block: true; reason: string }
  | { transform: EventData<E> }
  | void;
```

**ExtensionContext 类型：**

```typescript
export interface ExtensionContext {
  // UI API
  ui: ExtensionUIContext;

  // Session 管理
  sessionManager: SessionManager;

  // 执行命令
  exec(command: string, options?: ExecOptions): Promise<ExecResult>;

  // 是否有 UI
  hasUI: boolean;

  // 关闭 pi
  shutdown(): Promise<void>;
}
```

### 2.3 类型安全的好处

**1. 编译时错误检查：**

```typescript
// ❌ 编译错误：缺少必需字段
pi.registerTool({
  name: "my_tool",
  // 缺少 description 和 parameters
});

// ✅ 编译通过
pi.registerTool({
  name: "my_tool",
  description: "My tool",
  parameters: Type.Object({}),
  async execute(toolCallId, params, signal, onUpdate, ctx) {
    return { content: [], details: {} };
  },
});
```

**2. IDE 自动补全：**

```typescript
export default function (pi: ExtensionAPI) {
  // 输入 pi. 后，IDE 会显示所有可用方法
  pi.registerTool({ ... });
  pi.registerCommand("cmd", { ... });
  pi.on("tool_call", async (event, ctx) => {
    // 输入 event. 后，IDE 会显示事件的所有字段
    console.log(event.toolName);
    console.log(event.input);

    // 输入 ctx. 后，IDE 会显示上下文的所有方法
    await ctx.ui.confirm("Title", "Message");
  });
}
```

**3. 类型推导：**

```typescript
pi.on("tool_call", async (event, ctx) => {
  // event 的类型自动推导为 ToolCallEvent
  // TypeScript 知道 event 有 toolName, input, toolCallId 等字段

  if (event.toolName === "bash") {
    // event.input 的类型自动推导为 BashInput
    // TypeScript 知道 input 有 command 字段
    const command = event.input.command;
  }
});
```

---

## 3. 模块依赖管理

### 3.1 导入 npm 包

**Extensions 可以导入 npm 包：**

```typescript
// 导入标准库
import fs from "fs";
import path from "path";
import os from "os";

// 导入 npm 包
import axios from "axios";
import { z } from "zod";

// 导入 pi-mono 包
import { Type } from "@sinclair/typebox";
import { StringEnum } from "@mariozechner/pi-ai";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  // 使用导入的包
  const homeDir = os.homedir();
  const configPath = path.join(homeDir, ".pi", "config.json");

  pi.registerTool({
    name: "fetch_data",
    description: "Fetch data from API",
    parameters: Type.Object({
      url: Type.String(),
    }),
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      const { url } = params as { url: string };
      const response = await axios.get(url);
      return {
        content: [{ type: "text", text: JSON.stringify(response.data) }],
        details: { data: response.data },
      };
    },
  });
}
```

### 3.2 Extension 的 package.json

**如果 Extension 需要额外的依赖，可以创建 package.json：**

```
my-extension/
├── package.json
├── index.ts
└── utils.ts
```

**package.json：**

```json
{
  "name": "my-extension",
  "version": "1.0.0",
  "type": "module",
  "dependencies": {
    "axios": "^1.6.0",
    "zod": "^3.22.0"
  }
}
```

**index.ts：**

```typescript
import axios from "axios";
import { z } from "zod";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { helperFunction } from "./utils.js";

export default function (pi: ExtensionAPI) {
  // 使用依赖
  pi.registerTool({ ... });
}
```

**安装依赖：**

```bash
cd my-extension
npm install

# 然后加载扩展
pi --extension ./my-extension/index.ts
```

### 3.3 虚拟模块

**pi-mono 提供了一些虚拟模块，可以直接导入：**

```typescript
// 虚拟模块：@mariozechner/pi-coding-agent
import type {
  ExtensionAPI,
  ExtensionContext,
  ToolDefinition,
  CommandConfig,
  EventType,
} from "@mariozechner/pi-coding-agent";

// 虚拟模块：@mariozechner/pi-ai
import { StringEnum } from "@mariozechner/pi-ai";
import { Type } from "@sinclair/typebox";

// 虚拟模块：@mariozechner/pi-tui
import { Text, Box, matchesKey } from "@mariozechner/pi-tui";
```

**这些模块不需要安装，jiti 会自动解析。**

---

## 4. jiti 动态加载机制

### 4.1 什么是 jiti？

**jiti 是一个 TypeScript 运行时加载器：**

- 动态编译 TypeScript 到 JavaScript
- 支持 ESM 和 CommonJS
- 支持 TypeBox、Zod 等库
- 无需预编译

**为什么使用 jiti？**

1. **开发体验**：无需编译步骤，直接运行 TypeScript
2. **热重载**：可以清除模块缓存，重新加载
3. **类型安全**：保留 TypeScript 的类型检查
4. **兼容性**：支持各种 npm 包

### 4.2 加载流程

```
Extension 文件（.ts）
  ↓
jiti 加载器
  ↓
1. 读取文件内容
  ↓
2. 使用 esbuild 编译 TypeScript
  ↓
3. 解析 import 语句
  ↓
4. 递归加载依赖
  ↓
5. 执行模块代码
  ↓
6. 返回 exports
  ↓
调用默认导出函数
```

### 4.3 模块缓存

**jiti 使用 Node.js 的模块缓存：**

```typescript
// 第一次加载
const ext1 = await jiti.import("./my-extension.ts");
// 模块被缓存

// 第二次加载（从缓存）
const ext2 = await jiti.import("./my-extension.ts");
// ext1 === ext2

// 热重载：清除缓存
delete require.cache[require.resolve("./my-extension.ts")];
// 重新加载
const ext3 = await jiti.import("./my-extension.ts");
// ext3 是新的实例
```

### 4.4 热重载实现

**pi-mono 的热重载机制：**

```typescript
class ExtensionManager {
  private extensions: Map<string, Extension> = new Map();
  private jiti: Jiti;

  async reload() {
    // 1. 卸载所有扩展
    for (const ext of this.extensions.values()) {
      if (ext.dispose) {
        await ext.dispose();
      }
    }
    this.extensions.clear();

    // 2. 清除模块缓存
    for (const key of Object.keys(require.cache)) {
      if (key.includes("/extensions/")) {
        delete require.cache[key];
      }
    }

    // 3. 重新加载扩展
    const extensionFiles = await this.discoverExtensions();
    for (const file of extensionFiles) {
      await this.loadExtension(file);
    }
  }

  async loadExtension(filePath: string) {
    // 使用 jiti 加载
    const module = await this.jiti.import(filePath);
    const factory = module.default;

    if (typeof factory !== "function") {
      throw new Error(`Extension ${filePath} must export a default function`);
    }

    // 调用工厂函数
    const dispose = factory(this.extensionAPI);

    // 保存扩展
    this.extensions.set(filePath, { factory, dispose });
  }
}
```

---

## 5. 最佳实践

### 5.1 模块组织

**推荐的文件结构：**

```
my-extension/
├── index.ts          # 主入口
├── types.ts          # 类型定义
├── utils.ts          # 工具函数
├── tools/            # 工具定义
│   ├── tool1.ts
│   └── tool2.ts
├── commands/         # 命令定义
│   ├── cmd1.ts
│   └── cmd2.ts
└── package.json      # 依赖（可选）
```

**index.ts：**

```typescript
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { registerTools } from "./tools/index.js";
import { registerCommands } from "./commands/index.js";

export default function (pi: ExtensionAPI) {
  // 注册工具
  registerTools(pi);

  // 注册命令
  registerCommands(pi);

  // 监听事件
  pi.on("session_start", async (event, ctx) => {
    console.log("Session started");
  });
}
```

### 5.2 类型定义

**将类型定义放在单独的文件中：**

```typescript
// types.ts
export interface TodoItem {
  id: number;
  text: string;
  done: boolean;
}

export interface TodoDetails {
  todos: TodoItem[];
  nextId: number;
}

export interface TodoParams {
  action: "list" | "add" | "toggle" | "clear";
  text?: string;
  id?: number;
}
```

**在主文件中导入：**

```typescript
// index.ts
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import type { TodoItem, TodoDetails, TodoParams } from "./types.js";

export default function (pi: ExtensionAPI) {
  let todos: TodoItem[] = [];

  pi.registerTool({
    name: "todo",
    // 使用类型
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      const { action, text, id } = params as TodoParams;
      // ...
      return {
        content: [...],
        details: { todos, nextId } as TodoDetails,
      };
    },
  });
}
```

### 5.3 错误处理

**在 Extension 中处理错误：**

```typescript
export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "my_tool",
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      try {
        // 执行逻辑
        const result = await doSomething(params);
        return {
          content: [{ type: "text", text: "Success" }],
          details: { result },
        };
      } catch (error) {
        // 捕获错误
        console.error("Tool execution failed:", error);
        return {
          content: [
            {
              type: "text",
              text: `Error: ${error.message}`,
            },
          ],
          details: { error: error.message },
        };
      }
    },
  });

  // 事件监听器的错误处理
  pi.on("tool_call", async (event, ctx) => {
    try {
      // 处理事件
      if (shouldBlock(event)) {
        return { block: true, reason: "Blocked" };
      }
    } catch (error) {
      console.error("Event handler failed:", error);
      // 不阻止事件继续
    }
  });
}
```

### 5.4 性能优化

**避免阻塞主线程：**

```typescript
export default function (pi: ExtensionAPI) {
  // ❌ 不好：同步的重计算
  pi.on("tool_call", async (event, ctx) => {
    const result = expensiveComputation(); // 阻塞
    console.log(result);
  });

  // ✅ 好：异步执行
  pi.on("tool_call", async (event, ctx) => {
    // 使用 Promise 异步执行
    const result = await Promise.resolve().then(() => {
      return expensiveComputation();
    });
    console.log(result);
  });

  // ✅ 更好：使用 Worker（如果需要）
  pi.on("tool_call", async (event, ctx) => {
    // 在后台线程执行
    const result = await runInWorker(expensiveComputation);
    console.log(result);
  });
}
```

---

## 6. 调试技巧

### 6.1 使用 console.log

```typescript
export default function (pi: ExtensionAPI) {
  console.log("[Extension] Loading");

  pi.on("tool_call", async (event, ctx) => {
    console.log("[Extension] Tool called:", event.toolName);
    console.log("[Extension] Input:", event.input);
  });

  pi.registerTool({
    name: "my_tool",
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      console.log("[Extension] Tool executing:", params);
      const result = await doSomething(params);
      console.log("[Extension] Tool result:", result);
      return { content: [...], details: {} };
    },
  });
}
```

### 6.2 使用 VS Code Debugger

**launch.json：**

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "node",
      "request": "launch",
      "name": "Debug pi with Extension",
      "program": "${workspaceFolder}/node_modules/.bin/pi",
      "args": [
        "--extension",
        "${workspaceFolder}/my-extension.ts"
      ],
      "console": "integratedTerminal",
      "skipFiles": ["<node_internals>/**"]
    }
  ]
}
```

**在代码中设置断点：**

```typescript
export default function (pi: ExtensionAPI) {
  pi.on("tool_call", async (event, ctx) => {
    // 设置断点在这里
    debugger;
    console.log(event);
  });
}
```

### 6.3 错误追踪

```typescript
export default function (pi: ExtensionAPI) {
  // 全局错误处理
  process.on("unhandledRejection", (error) => {
    console.error("[Extension] Unhandled rejection:", error);
  });

  pi.registerTool({
    name: "my_tool",
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      try {
        return await doSomething(params);
      } catch (error) {
        // 记录完整的错误堆栈
        console.error("[Extension] Error:", error);
        console.error("[Extension] Stack:", error.stack);
        throw error;
      }
    },
  });
}
```

---

## 7. 总结

### 核心要点

1. **模块结构**：Extension 是导出默认函数的 TypeScript 模块
2. **工厂模式**：使用闭包封装私有状态和函数
3. **类型安全**：完整的 TypeScript 类型定义，编译时错误检查
4. **模块依赖**：可以导入 npm 包和虚拟模块
5. **jiti 加载**：动态编译 TypeScript，支持热重载
6. **最佳实践**：模块组织、类型定义、错误处理、性能优化

### 下一步

- **核心概念 02**：事件监听与响应
- **核心概念 03**：工具与命令注册
- **实战代码**：编写完整的 Extension

---

**版本**: v1.0
**最后更新**: 2026-02-20
**适用于**: pi-mono 2025-2026 版本
