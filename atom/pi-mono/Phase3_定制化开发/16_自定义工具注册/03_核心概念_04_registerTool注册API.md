# 核心概念 04：registerTool 注册 API

> **工具的"注册中心" - 让框架知道工具的存在**

## 概述

registerTool 是 Extension 向框架注册工具的 API。理解注册时机、注册流程和注册后的生命周期，是开发 Extension 的关键。

---

## API 签名

### 源码位置

```
sourcecode/pi-mono/packages/coding-agent/src/core/extensions/types.ts:961
```

### 函数签名

```typescript
function registerTool(
  context: ExtensionContext,
  tool: ToolDefinition
): void
```

### 参数说明

**context**: ExtensionContext
- Extension 的执行上下文
- 在 Extension 的 activate 函数中可用

**tool**: ToolDefinition
- 完整的工具定义
- 包含 name, description, parameters, execute 等字段

**返回值**: void
- 无返回值
- 注册成功不会有提示
- 注册失败会抛出错误

---

## 注册时机

### 在 Extension 的 activate 中注册

```typescript
export const myExtension: Extension = {
  name: 'my-extension',
  version: '1.0.0',
  
  // ✅ 在 activate 函数中注册工具
  async activate(context) {
    context.registerTool({
      name: 'my-tool',
      description: 'My tool',
      parameters: Type.Object({ /* ... */ }),
      execute: async (params) => { /* ... */ }
    });
    
    context.logger.info('Tool registered');
  }
};
```

### 为什么在 activate 中？

**Extension 生命周期：**
```
1. 加载 Extension
   ↓
2. 调用 activate 函数  ← 在这里注册工具
   ↓
3. Extension 激活完成
   ↓
4. 工具可用（下次 LLM 调用时）
```

**原因：**
- activate 是 Extension 的初始化阶段
- 确保工具在 Extension 使用前就绪
- 框架会等待所有 Extension 的 activate 完成

---

## 注册流程

### 完整流程

```
用户启动 pi-mono
  ↓
框架加载所有 Extension
  ↓
框架调用每个 Extension 的 activate 函数
  ↓
Extension 调用 context.registerTool()
  ↓
框架收集所有注册的工具
  ↓
框架生成工具列表（JSON Schema 格式）
  ↓
在下一次 LLM 调用时，将工具列表作为 tools 参数发送
  ↓
LLM 可以看到并调用这些工具
```

### 源码实现

```typescript
// sourcecode/pi-mono/packages/coding-agent/src/core/extensions/loader.ts:150-155

class ExtensionContext {
  private tools: Map<string, ToolDefinition> = new Map();
  
  registerTool(tool: ToolDefinition): void {
    // 验证工具定义
    if (!tool.name) {
      throw new Error('Tool name is required');
    }
    
    // 存储工具（同名工具会被覆盖）
    this.tools.set(tool.name, tool);
    
    // 记录日志
    this.logger.debug(`Tool registered: ${tool.name}`);
  }
  
  getAllTools(): ToolDefinition[] {
    return Array.from(this.tools.values());
  }
}
```

---

## 注册示例

### 示例 1：注册单个工具

```typescript
export const simpleExtension: Extension = {
  name: 'simple-extension',
  version: '1.0.0',
  
  async activate(context) {
    context.registerTool({
      name: 'hello',
      description: 'Say hello to someone',
      parameters: Type.Object({
        name: Type.String({ description: 'Name of the person' })
      }),
      execute: async ({ name }) => {
        return { content: `Hello, ${name}!` };
      }
    });
    
    context.logger.info('Hello tool registered');
  }
};
```

### 示例 2：注册多个工具

```typescript
export const multiToolExtension: Extension = {
  name: 'multi-tool-extension',
  version: '1.0.0',
  
  async activate(context) {
    // 工具 1: Hello
    context.registerTool({
      name: 'hello',
      description: 'Say hello',
      parameters: Type.Object({
        name: Type.String()
      }),
      execute: async ({ name }) => {
        return { content: `Hello, ${name}!` };
      }
    });
    
    // 工具 2: Goodbye
    context.registerTool({
      name: 'goodbye',
      description: 'Say goodbye',
      parameters: Type.Object({
        name: Type.String()
      }),
      execute: async ({ name }) => {
        return { content: `Goodbye, ${name}!` };
      }
    });
    
    // 工具 3: Calculate
    context.registerTool({
      name: 'calculate',
      description: 'Perform arithmetic operations',
      parameters: Type.Object({
        operation: Type.Union([
          Type.Literal('add'),
          Type.Literal('subtract')
        ]),
        a: Type.Number(),
        b: Type.Number()
      }),
      execute: async ({ operation, a, b }) => {
        const result = operation === 'add' ? a + b : a - b;
        return { content: String(result) };
      }
    });
    
    context.logger.info('3 tools registered');
  }
};
```

### 示例 3：从外部模块导入工具

```typescript
// tools/weather.ts
export const weatherTool: ToolDefinition = {
  name: 'get-weather',
  description: 'Get current weather for a city',
  parameters: Type.Object({
    city: Type.String({ description: 'City name' })
  }),
  execute: async ({ city }, context) => {
    const weather = await fetchWeather(city);
    return { content: `Weather in ${city}: ${weather}` };
  }
};

// tools/search.ts
export const searchTool: ToolDefinition = {
  name: 'search-code',
  description: 'Search for code patterns',
  parameters: Type.Object({
    query: Type.String({ description: 'Search query' })
  }),
  execute: async ({ query }, context) => {
    const results = await searchCode(query);
    return { content: results.join('\n') };
  }
};

// extension.ts
import { weatherTool } from './tools/weather';
import { searchTool } from './tools/search';

export const myExtension: Extension = {
  name: 'my-extension',
  version: '1.0.0',
  
  async activate(context) {
    // 注册导入的工具
    context.registerTool(weatherTool);
    context.registerTool(searchTool);
    
    context.logger.info('Tools registered from external modules');
  }
};
```

### 示例 4：动态注册工具

```typescript
export const dynamicExtension: Extension = {
  name: 'dynamic-extension',
  version: '1.0.0',
  
  async activate(context) {
    // 读取配置
    const config = await loadConfig();
    
    // 根据配置动态注册工具
    if (config.enableWeather) {
      context.registerTool(weatherTool);
    }
    
    if (config.enableSearch) {
      context.registerTool(searchTool);
    }
    
    // 为每个 API 端点注册工具
    for (const endpoint of config.apiEndpoints) {
      context.registerTool({
        name: `call-${endpoint.name}`,
        description: `Call ${endpoint.name} API`,
        parameters: Type.Object({
          data: Type.Any()
        }),
        execute: async ({ data }) => {
          const result = await callAPI(endpoint.url, data);
          return { content: JSON.stringify(result) };
        }
      });
    }
    
    context.logger.info('Dynamic tools registered');
  }
};
```

---

## 注册后的生命周期

### 工具的状态

```
注册 → 收集 → 发送给 LLM → 可被调用
```

**1. 注册阶段**
- Extension 调用 context.registerTool()
- 工具被存储在 ExtensionContext 中

**2. 收集阶段**
- 所有 Extension 的 activate 完成后
- 框架收集所有注册的工具
- 生成工具列表

**3. 发送给 LLM**
- 在下一次 LLM 调用时
- 将工具列表作为 tools 参数发送
- LLM 可以"看到"这些工具

**4. 可被调用**
- LLM 根据用户输入决定是否调用工具
- 框架接收 LLM 的调用指令
- 框架执行对应的工具

### 工具的可见性

```typescript
// 用户启动 pi-mono
$ pi-mono

// Extension 加载并注册工具
[INFO] Extension 'my-extension' activated
[INFO] Tool 'my-tool' registered

// 用户查看可用工具
> /tools
Available tools:
- my-tool: My tool description

// 用户发送消息
> 使用 my-tool 做某事

// LLM 调用工具
[INFO] Tool 'my-tool' called with params: {...}
[INFO] Tool 'my-tool' returned: {...}
```

---

## 常见问题

### Q1: 工具注册后为什么不立即可用？

**A**: 工具注册后需要等待：
1. 所有 Extension 的 activate 完成
2. 框架生成工具列表
3. 下一次 LLM 调用时发送工具列表

**解决方案：**
- 如果是新添加的 Extension，执行 `/reload`
- 执行 `/tools` 查看工具列表

### Q2: 同名工具会怎样？

**A**: 后注册的工具会覆盖先注册的工具

```typescript
// Extension A
context.registerTool({
  name: 'my-tool',
  description: 'Tool from A',
  // ...
});

// Extension B（后加载）
context.registerTool({
  name: 'my-tool',  // 同名！
  description: 'Tool from B',
  // ...
});

// 结果：只有 Extension B 的工具可用
```

**最佳实践：**
- 使用唯一的工具名称
- 使用命名空间（如 `myext:tool-name`）

### Q3: 可以在 activate 之外注册工具吗？

**A**: 不推荐

```typescript
// ❌ 不推荐：在 activate 之外注册
export const badExtension: Extension = {
  name: 'bad-extension',
  version: '1.0.0',
  
  async activate(context) {
    // 延迟注册
    setTimeout(() => {
      context.registerTool(myTool);  // 可能太晚了
    }, 1000);
  }
};

// ✅ 推荐：在 activate 中立即注册
export const goodExtension: Extension = {
  name: 'good-extension',
  version: '1.0.0',
  
  async activate(context) {
    context.registerTool(myTool);  // 立即注册
  }
};
```

### Q4: 如何调试工具注册？

**A**: 使用日志和 /tools 命令

```typescript
export const myExtension: Extension = {
  async activate(context) {
    // 注册前记录日志
    context.logger.info('Registering tool:', myTool.name);
    
    context.registerTool(myTool);
    
    // 注册后记录日志
    context.logger.info('Tool registered successfully');
    
    // 验证工具列表
    const tools = context.getAllTools();  // 如果有这个方法
    context.logger.info('Total tools:', tools.length);
  }
};

// 在 CLI 中查看工具列表
> /tools
```

### Q5: 可以动态卸载工具吗？

**A**: Pi-mono 目前不支持动态卸载工具

**替代方案：**
- 在 execute 函数中检查条件，决定是否执行
- 使用 Extension 的 deactivate 钩子（如果有）

```typescript
const execute = async (params, context) => {
  // 检查条件
  if (!isToolEnabled()) {
    return { content: 'Tool is currently disabled' };
  }
  
  // 执行逻辑
  // ...
};
```

---

## 最佳实践

### 1. 在 activate 中立即注册

```typescript
// ✅ 正确
async activate(context) {
  context.registerTool(myTool);
}

// ❌ 错误：延迟注册
async activate(context) {
  setTimeout(() => {
    context.registerTool(myTool);
  }, 1000);
}
```

### 2. 使用唯一的工具名称

```typescript
// ✅ 正确：使用命名空间
context.registerTool({
  name: 'myext:search',
  // ...
});

// ❌ 不好：通用名称
context.registerTool({
  name: 'search',  // 可能与其他 Extension 冲突
  // ...
});
```

### 3. 记录日志

```typescript
// ✅ 正确
async activate(context) {
  context.registerTool(tool1);
  context.registerTool(tool2);
  context.logger.info('2 tools registered');
}
```

### 4. 组织工具定义

```typescript
// ✅ 正确：分离工具定义
// tools/weather.ts
export const weatherTool: ToolDefinition = { /* ... */ };

// tools/search.ts
export const searchTool: ToolDefinition = { /* ... */ };

// extension.ts
import { weatherTool } from './tools/weather';
import { searchTool } from './tools/search';

export const myExtension: Extension = {
  async activate(context) {
    context.registerTool(weatherTool);
    context.registerTool(searchTool);
  }
};
```

### 5. 错误处理

```typescript
// ✅ 正确
async activate(context) {
  try {
    context.registerTool(myTool);
    context.logger.info('Tool registered');
  } catch (error) {
    context.logger.error('Failed to register tool:', error);
    // 不要抛出错误，避免影响其他 Extension
  }
}
```

---

## 2025-2026 最新趋势

### 1. 工具注册验证

```typescript
// 2025-2026 年，框架增加了工具注册验证
context.registerTool(myTool, {
  validate: true,  // 验证工具定义
  checkConflicts: true  // 检查名称冲突
});
```

### 2. 工具分组

```typescript
// 2025-2026 年，支持工具分组
context.registerToolGroup({
  name: 'file-operations',
  tools: [readFileTool, writeFileTool, deleteFileTool]
});
```

### 3. 工具权限

```typescript
// 2025-2026 年，支持工具权限控制
context.registerTool(myTool, {
  permissions: ['read:files', 'write:files']
});
```

---

## 实战案例

### 案例 1：天气查询 Extension

```typescript
import { Type } from '@sinclair/typebox';
import type { Extension } from '@pi-mono/coding-agent';

export const weatherExtension: Extension = {
  name: 'weather-extension',
  version: '1.0.0',
  description: 'Provides weather information',
  
  async activate(context) {
    context.registerTool({
      name: 'get-weather',
      description: 'Get current weather for a city',
      parameters: Type.Object({
        city: Type.String({
          description: 'City name (e.g., Beijing, Shanghai)'
        }),
        units: Type.Optional(Type.Union([
          Type.Literal('celsius'),
          Type.Literal('fahrenheit')
        ], {
          description: 'Temperature units',
          default: 'celsius'
        }))
      }),
      execute: async ({ city, units = 'celsius' }, ctx) => {
        try {
          ctx.logger.info(`Fetching weather for ${city}`);
          
          const weather = await fetchWeather(city, units);
          
          return {
            content: `Weather in ${city}: ${weather.temperature}°${units === 'celsius' ? 'C' : 'F'}, ${weather.condition}`,
            metadata: {
              city,
              temperature: weather.temperature,
              condition: weather.condition,
              units
            }
          };
        } catch (error) {
          ctx.logger.error('Weather API error:', error);
          return {
            content: `Failed to get weather for ${city}: ${error.message}`,
            metadata: { error: true }
          };
        }
      }
    });
    
    context.logger.info('Weather tool registered');
  }
};
```

### 案例 2：多工具 Extension

```typescript
export const utilsExtension: Extension = {
  name: 'utils-extension',
  version: '1.0.0',
  
  async activate(context) {
    // 工具 1: Base64 编码
    context.registerTool({
      name: 'base64-encode',
      description: 'Encode text to base64',
      parameters: Type.Object({
        text: Type.String()
      }),
      execute: async ({ text }) => {
        const encoded = Buffer.from(text).toString('base64');
        return { content: encoded };
      }
    });
    
    // 工具 2: Base64 解码
    context.registerTool({
      name: 'base64-decode',
      description: 'Decode base64 to text',
      parameters: Type.Object({
        encoded: Type.String()
      }),
      execute: async ({ encoded }) => {
        try {
          const decoded = Buffer.from(encoded, 'base64').toString('utf-8');
          return { content: decoded };
        } catch (error) {
          return { content: `Error: Invalid base64 string` };
        }
      }
    });
    
    // 工具 3: UUID 生成
    context.registerTool({
      name: 'generate-uuid',
      description: 'Generate a random UUID',
      parameters: Type.Object({}),
      execute: async () => {
        const uuid = crypto.randomUUID();
        return { content: uuid };
      }
    });
    
    context.logger.info('3 utility tools registered');
  }
};
```

---

## 总结

**registerTool API 的关键要点：**

1. **签名**：`context.registerTool(tool: ToolDefinition): void`
2. **时机**：在 Extension 的 activate 函数中
3. **流程**：注册 → 收集 → 发送给 LLM → 可被调用
4. **注意**：同名工具会被覆盖

**最佳实践：**
- 在 activate 中立即注册
- 使用唯一的工具名称
- 记录日志
- 组织工具定义
- 错误处理

**常见问题：**
- 工具注册后不立即可用（需要 /reload）
- 同名工具会被覆盖
- 不推荐在 activate 之外注册
- 使用 /tools 命令调试

理解 registerTool API 的使用方式和最佳实践，你就能正确地注册自定义工具，扩展 AI Agent 的能力！
