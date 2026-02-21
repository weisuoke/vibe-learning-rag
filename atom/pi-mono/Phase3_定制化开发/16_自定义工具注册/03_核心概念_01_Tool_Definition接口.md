# 核心概念 01：Tool Definition 接口

> **工具的"身份证" - 定义工具的完整描述**

## 概述

Tool Definition 是自定义工具注册的核心接口，它完整描述了一个工具的所有信息：名称、用途、参数、执行逻辑、以及可选的自定义渲染。

**类比：**
- Tool Definition = 工具的"身份证"
- 包含工具的所有必要信息
- LLM 和框架都依赖这个接口

---

## 接口定义

### 源码位置

```
sourcecode/pi-mono/packages/coding-agent/src/core/extensions/types.ts:335-359
```

### 完整接口

```typescript
export interface ToolDefinition {
  // 必需字段
  name: string;
  description: string;
  parameters: TSchema;
  execute: (
    params: any,
    context: ExtensionContext
  ) => Promise<AgentToolResult>;
  
  // 可选字段
  label?: string;
  renderCall?: (params: any) => React.ReactNode;
  renderResult?: (result: AgentToolResult) => React.ReactNode;
}
```

---

## 字段详解

### 1. name（必需）

**类型：** `string`

**作用：** 工具的唯一标识符

**要求：**
- 必须唯一（同名工具会被覆盖）
- 建议使用 kebab-case（如 `get-weather`）
- 不要使用空格或特殊字符
- 应该简洁且描述性强

**示例：**
```typescript
// ✅ 好的命名
name: 'read-file'
name: 'search-code'
name: 'get-weather'

// ❌ 不好的命名
name: 'readFile'        // camelCase（不推荐）
name: 'read file'       // 包含空格
name: 'tool1'           // 不描述性
name: 'read_file_from_disk_and_return_content'  // 太长
```

**LLM 如何使用：**
```json
// LLM 生成的工具调用
{
  "name": "read-file",
  "parameters": {
    "path": "file.txt"
  }
}
```

---

### 2. description（必需）

**类型：** `string`

**作用：** 工具的功能描述，帮助 LLM 理解何时调用这个工具

**要求：**
- 清晰描述工具的用途
- 使用简洁的英文（LLM 更容易理解）
- 包含关键信息（输入、输出、限制）
- 不要过于简短或过于冗长

**示例：**
```typescript
// ✅ 好的描述
description: 'Read the contents of a file from the filesystem'
description: 'Search for code patterns in the codebase using regex'
description: 'Get current weather information for a specified city'

// ❌ 不好的描述
description: 'Read file'  // 太简短
description: 'This tool reads files'  // 废话
description: 'Read the contents of a file from the filesystem, supporting various encodings including UTF-8, UTF-16, ASCII, and more, with error handling for missing files, permission issues, and large files'  // 太冗长
```

**最佳实践：**
```typescript
// 包含关键信息
description: 'Search files by pattern. Returns file paths matching the glob pattern.'

// 说明限制
description: 'Get weather for a city. Requires valid city name in English.'

// 说明输出
description: 'Calculate arithmetic operations. Returns the numeric result.'
```

**LLM 如何使用：**
```
用户: 读取 config.json 文件的内容
LLM 思考: 
  - 需要读取文件
  - 查看可用工具
  - 找到 "read-file" 工具
  - description 说明它可以读取文件
  - 决定调用这个工具
```

---

### 3. parameters（必需）

**类型：** `TSchema`（TypeBox Schema）

**作用：** 定义工具的参数结构和类型

**要求：**
- 使用 TypeBox 定义
- 每个参数都要有 description
- 使用适当的类型约束
- 标记可选参数

**基本示例：**
```typescript
import { Type } from '@sinclair/typebox';

parameters: Type.Object({
  path: Type.String({
    description: 'File path to read'
  })
})
```

**复杂示例：**
```typescript
parameters: Type.Object({
  // 必需参数
  query: Type.String({
    description: 'Search query'
  }),
  
  // 可选参数
  fileType: Type.Optional(Type.Union([
    Type.Literal('js'),
    Type.Literal('ts'),
    Type.Literal('json')
  ], {
    description: 'File type filter'
  })),
  
  // 带约束的参数
  maxResults: Type.Optional(Type.Number({
    description: 'Maximum number of results',
    minimum: 1,
    maximum: 100,
    default: 10
  })),
  
  // 布尔参数
  caseSensitive: Type.Optional(Type.Boolean({
    description: 'Case sensitive search',
    default: false
  }))
})
```

**LLM 如何使用：**
```json
// LLM 根据 parameters 生成调用
{
  "name": "search-code",
  "parameters": {
    "query": "function.*async",
    "fileType": "ts",
    "maxResults": 20
  }
}
```

---

### 4. execute（必需）

**类型：** `(params: any, context: ExtensionContext) => Promise<AgentToolResult>`

**作用：** 工具的执行逻辑

**签名详解：**
```typescript
execute: async (
  params: any,                    // 已验证的参数
  context: ExtensionContext       // 执行上下文
) => Promise<AgentToolResult>     // 返回结果
```

**参数说明：**

**params**：
- 已经通过 TypeBox 验证的参数
- 类型安全（如果使用 TypeScript）
- 可以直接使用，无需再次验证

**context**：
- `context.logger`: 日志记录器
- `context.session`: 会话状态存储
- `context.signal`: AbortSignal（用于取消操作）
- `context.onUpdate`: 流式更新回调

**返回值：**
```typescript
interface AgentToolResult {
  content: string;           // 主要内容（必需）
  metadata?: {               // 元数据（可选）
    [key: string]: any;
  };
}
```

**基本示例：**
```typescript
execute: async ({ path }, context) => {
  try {
    const content = await fs.readFile(path, 'utf-8');
    return { content };
  } catch (error) {
    context.logger.error('Failed to read file:', error);
    return { content: `Error: ${error.message}` };
  }
}
```

**完整示例：**
```typescript
execute: async ({ query, maxResults = 10 }, context) => {
  const { logger, signal, onUpdate } = context;
  
  try {
    // 记录日志
    logger.info('Searching for:', query);
    
    // 检查取消
    if (signal.aborted) {
      return { content: 'Search cancelled' };
    }
    
    // 执行搜索
    onUpdate?.({ content: 'Searching...' });
    const results = await searchFiles(query, { signal });
    
    // 限制结果数量
    const limited = results.slice(0, maxResults);
    
    // 返回结果
    return {
      content: limited.map(r => r.path).join('\n'),
      metadata: {
        totalResults: results.length,
        returnedResults: limited.length
      }
    };
  } catch (error) {
    logger.error('Search failed:', error);
    return {
      content: `Error: ${error.message}`,
      metadata: { error: true }
    };
  }
}
```

---

### 5. label（可选）

**类型：** `string`

**作用：** 工具的显示名称（用于 UI）

**默认值：** 如果不提供，使用 `name`

**示例：**
```typescript
{
  name: 'read-file',
  label: 'Read File',  // UI 显示 "Read File"
  // ...
}
```

**使用场景：**
- name 是技术标识符（kebab-case）
- label 是用户友好的显示名称（Title Case）

---

### 6. renderCall（可选）

**类型：** `(params: any) => React.ReactNode`

**作用：** 自定义工具调用时的显示

**默认行为：** 显示工具名称和参数的 JSON

**示例：**
```typescript
renderCall: (params) => {
  return `Reading file: ${params.path}`;
}
```

**使用 pi-tui 组件：**
```typescript
import { Box, Text } from '@pi-mono/pi-tui';

renderCall: (params) => {
  return (
    <Box flexDirection="column">
      <Text bold>Reading File</Text>
      <Text color="gray">Path: {params.path}</Text>
    </Box>
  );
}
```

---

### 7. renderResult（可选）

**类型：** `(result: AgentToolResult) => React.ReactNode`

**作用：** 自定义工具结果的显示

**默认行为：** 显示 result.content

**示例：**
```typescript
renderResult: (result) => {
  const lines = result.content.split('\n').length;
  return `File content (${lines} lines):\n${result.content}`;
}
```

**使用 pi-tui 组件：**
```typescript
renderResult: (result) => {
  return (
    <Box flexDirection="column">
      <Text color="green">✓ File read successfully</Text>
      <Text>{result.content}</Text>
      {result.metadata?.fileSize && (
        <Text color="gray">Size: {result.metadata.fileSize} bytes</Text>
      )}
    </Box>
  );
}
```

---

## 完整示例

### 示例 1：简单文件读取工具

```typescript
import { Type } from '@sinclair/typebox';
import type { ToolDefinition } from '@pi-mono/coding-agent';
import * as fs from 'fs/promises';

const readFileTool: ToolDefinition = {
  name: 'read-file',
  description: 'Read the contents of a file from the filesystem',
  parameters: Type.Object({
    path: Type.String({
      description: 'File path to read'
    })
  }),
  execute: async ({ path }, context) => {
    try {
      const content = await fs.readFile(path, 'utf-8');
      return {
        content,
        metadata: {
          fileSize: content.length,
          timestamp: new Date().toISOString()
        }
      };
    } catch (error) {
      context.logger.error('Failed to read file:', error);
      return {
        content: `Error reading file: ${error.message}`,
        metadata: { error: true }
      };
    }
  }
};
```

### 示例 2：带自定义渲染的搜索工具

```typescript
import { Type } from '@sinclair/typebox';
import { Box, Text } from '@pi-mono/pi-tui';

const searchTool: ToolDefinition = {
  name: 'search-code',
  label: 'Search Code',
  description: 'Search for code patterns in the codebase using regex',
  
  parameters: Type.Object({
    query: Type.String({
      description: 'Search query (regex pattern)'
    }),
    fileType: Type.Optional(Type.Union([
      Type.Literal('js'),
      Type.Literal('ts'),
      Type.Literal('json')
    ], {
      description: 'File type filter'
    })),
    maxResults: Type.Optional(Type.Number({
      description: 'Maximum number of results',
      minimum: 1,
      maximum: 100,
      default: 10
    }))
  }),
  
  execute: async ({ query, fileType, maxResults = 10 }, context) => {
    const { logger, signal, onUpdate } = context;
    
    try {
      logger.info('Searching for:', query);
      
      if (signal.aborted) {
        return { content: 'Search cancelled' };
      }
      
      onUpdate?.({ content: 'Searching files...' });
      
      // 模拟搜索
      const results = await searchFiles(query, fileType, { signal });
      const limited = results.slice(0, maxResults);
      
      return {
        content: limited.map(r => `${r.path}:${r.line}`).join('\n'),
        metadata: {
          totalResults: results.length,
          returnedResults: limited.length,
          query,
          fileType
        }
      };
    } catch (error) {
      logger.error('Search failed:', error);
      return {
        content: `Error: ${error.message}`,
        metadata: { error: true }
      };
    }
  },
  
  renderCall: (params) => {
    return (
      <Box flexDirection="column">
        <Text bold>🔍 Searching Code</Text>
        <Text>Query: {params.query}</Text>
        {params.fileType && <Text>Type: {params.fileType}</Text>}
        <Text color="gray">Max results: {params.maxResults || 10}</Text>
      </Box>
    );
  },
  
  renderResult: (result) => {
    if (result.metadata?.error) {
      return (
        <Box>
          <Text color="red">✗ {result.content}</Text>
        </Box>
      );
    }
    
    const count = result.metadata?.returnedResults || 0;
    const total = result.metadata?.totalResults || 0;
    
    return (
      <Box flexDirection="column">
        <Text color="green">✓ Found {count} results (total: {total})</Text>
        <Text>{result.content}</Text>
      </Box>
    );
  }
};
```

---

## 实战应用

### 来自 pi-mono 的真实示例

#### 1. question.ts - 交互式问题工具

```typescript
// sourcecode/pi-mono/packages/coding-agent/examples/extensions/question.ts

export const questionTool: ToolDefinition = {
  name: 'ask-question',
  description: 'Ask the user a question with multiple choice options',
  
  parameters: Type.Object({
    question: Type.String({
      description: 'The question to ask'
    }),
    options: Type.Array(Type.Object({
      label: Type.String(),
      value: Type.String()
    }), {
      description: 'Available options'
    })
  }),
  
  execute: async ({ question, options }, context) => {
    // 显示问题并等待用户选择
    const answer = await showQuestion(question, options);
    return {
      content: `User selected: ${answer}`,
      metadata: { question, answer }
    };
  },
  
  renderCall: (params) => {
    return (
      <Box flexDirection="column">
        <Text bold>{params.question}</Text>
        {params.options.map((opt, i) => (
          <Text key={i}>  {i + 1}. {opt.label}</Text>
        ))}
      </Box>
    );
  }
};
```

#### 2. todo.ts - 状态管理工具

```typescript
// sourcecode/pi-mono/packages/coding-agent/examples/extensions/todo.ts

export const addTaskTool: ToolDefinition = {
  name: 'add-task',
  description: 'Add a new task to the todo list',
  
  parameters: Type.Object({
    task: Type.String({
      description: 'Task description'
    })
  }),
  
  execute: async ({ task }, context) => {
    // 从 session 读取任务列表
    const tasks = context.session.get('tasks') || [];
    
    // 添加新任务
    const newTask = {
      id: Date.now(),
      task,
      completed: false
    };
    tasks.push(newTask);
    
    // 保存到 session
    context.session.set('tasks', tasks);
    
    return {
      content: `Task added: ${task}`,
      metadata: { taskId: newTask.id }
    };
  }
};
```

---

## 2025-2026 最新趋势

### 1. MCP (Model Context Protocol) 集成

```typescript
// 2025-2026 年，工具定义越来越标准化
// MCP 提供了标准的工具描述格式

import { MCPTool } from '@modelcontextprotocol/sdk';

// Pi-mono Tool Definition 可以转换为 MCP Tool
function toMCPTool(tool: ToolDefinition): MCPTool {
  return {
    name: tool.name,
    description: tool.description,
    inputSchema: typeboxToJsonSchema(tool.parameters)
  };
}
```

### 2. 类型安全增强

```typescript
// 2025-2026 年，TypeScript 类型推导更强大

import { Type, Static } from '@sinclair/typebox';

const params = Type.Object({
  city: Type.String(),
  units: Type.Union([
    Type.Literal('celsius'),
    Type.Literal('fahrenheit')
  ])
});

type Params = Static<typeof params>;
// Params = { city: string; units: 'celsius' | 'fahrenheit' }

const tool: ToolDefinition = {
  name: 'get-weather',
  description: 'Get weather',
  parameters: params,
  execute: async (params: Params, context) => {
    // params 是类型安全的
    const { city, units } = params;
    // ...
  }
};
```

### 3. 流式输出

```typescript
// 2025-2026 年，流式输出成为标准

const tool: ToolDefinition = {
  name: 'generate-report',
  description: 'Generate a detailed report',
  parameters: Type.Object({
    topic: Type.String()
  }),
  execute: async ({ topic }, context) => {
    const { onUpdate } = context;
    
    // 流式更新进度
    onUpdate?.({ content: 'Collecting data...' });
    const data = await collectData(topic);
    
    onUpdate?.({ content: 'Analyzing data...' });
    const analysis = await analyzeData(data);
    
    onUpdate?.({ content: 'Generating report...' });
    const report = await generateReport(analysis);
    
    return { content: report };
  }
};
```

---

## 最佳实践总结

### 1. 命名规范
- ✅ 使用 kebab-case
- ✅ 简洁且描述性强
- ✅ 避免缩写和特殊字符

### 2. 描述规范
- ✅ 清晰描述用途
- ✅ 包含关键信息
- ✅ 使用简洁英文

### 3. 参数规范
- ✅ 每个参数都有 description
- ✅ 使用适当的类型约束
- ✅ 标记可选参数
- ✅ 提供默认值

### 4. 执行规范
- ✅ 必须是 async 函数
- ✅ 使用 try-catch 处理错误
- ✅ 记录日志
- ✅ 支持取消操作
- ✅ 提供进度反馈

### 5. 渲染规范
- ✅ 使用 pi-tui 组件
- ✅ 突出关键信息
- ✅ 保持一致的 UI 风格

---

## 常见问题

### Q1: name 和 label 有什么区别？

**A**: 
- `name`: 技术标识符，用于 LLM 调用（kebab-case）
- `label`: 显示名称，用于 UI（Title Case）

### Q2: description 应该多详细？

**A**: 
- 简洁但完整
- 包含关键信息（输入、输出、限制）
- 1-2 句话即可

### Q3: 如何处理复杂参数？

**A**: 
- 使用嵌套的 Type.Object
- 每个字段都要有 description
- 使用 Type.Optional 标记可选字段

### Q4: execute 函数可以是同步的吗？

**A**: 
- 不可以，必须是 async 函数
- 即使逻辑是同步的，也要声明为 async

### Q5: 如何测试 Tool Definition？

**A**: 
```typescript
// 单元测试
describe('readFileTool', () => {
  it('should read file content', async () => {
    const result = await readFileTool.execute(
      { path: 'test.txt' },
      mockContext
    );
    expect(result.content).toBe('file content');
  });
});
```

---

## 总结

Tool Definition 是自定义工具注册的核心接口，包含：

**必需字段：**
1. **name**: 工具标识符
2. **description**: 工具描述
3. **parameters**: 参数定义（TypeBox Schema）
4. **execute**: 执行逻辑（async 函数）

**可选字段：**
5. **label**: 显示名称
6. **renderCall**: 自定义调用显示
7. **renderResult**: 自定义结果显示

理解 Tool Definition 接口，你就能设计出清晰、类型安全、用户友好的自定义工具！
