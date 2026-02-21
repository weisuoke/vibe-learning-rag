# 核心概念 03：Execute 执行函数

> **工具的"执行引擎" - 实现工具的实际逻辑**

## 概述

execute 函数是工具的核心，它定义了工具被调用时的实际行为。理解 execute 函数的签名、参数、返回值和最佳实践，是编写高质量工具的关键。

---

## 函数签名

### 完整签名

```typescript
type ExecuteFunction = (
  params: any,                    // 工具参数（已通过 TypeBox 验证）
  context: ExtensionContext       // 执行上下文
) => Promise<AgentToolResult>;    // 返回结果
```

### 关键要点

1. **必须是 async 函数**：即使逻辑是同步的
2. **params 已验证**：TypeBox 已经验证过参数类型
3. **context 提供资源**：logger, session, signal, onUpdate 等
4. **返回 Promise**：异步返回 AgentToolResult

---

## 参数详解

### 1. params（工具参数）

**类型：** `any`（实际类型由 TypeBox Schema 定义）

**特点：**
- 已经通过 TypeBox 验证
- 类型安全（如果使用 TypeScript 类型推导）
- 可以直接使用，无需再次验证

**类型推导示例：**

```typescript
import { Type, Static } from '@sinclair/typebox';

const params = Type.Object({
  path: Type.String(),
  encoding: Type.Optional(Type.Union([
    Type.Literal('utf-8'),
    Type.Literal('ascii')
  ]))
});

type Params = Static<typeof params>;
// Params = { path: string; encoding?: 'utf-8' | 'ascii' }

const execute = async (params: Params, context) => {
  // params 是类型安全的
  const { path, encoding = 'utf-8' } = params;
  // ...
};
```

**访问参数：**

```typescript
// 解构赋值（推荐）
const execute = async ({ path, maxResults = 10 }, context) => {
  // 直接使用 path 和 maxResults
};

// 对象访问
const execute = async (params, context) => {
  const path = params.path;
  const maxResults = params.maxResults || 10;
};
```

---

### 2. context（执行上下文）

**类型：** `ExtensionContext`

**提供的资源：**

#### context.logger

**类型：** `Logger`

**用途：** 记录日志

**方法：**
```typescript
context.logger.info('Info message');
context.logger.warn('Warning message');
context.logger.error('Error message');
context.logger.debug('Debug message');
```

**示例：**
```typescript
const execute = async ({ query }, context) => {
  context.logger.info('Searching for:', query);
  
  try {
    const results = await search(query);
    context.logger.info('Found results:', results.length);
    return { content: JSON.stringify(results) };
  } catch (error) {
    context.logger.error('Search failed:', error);
    return { content: `Error: ${error.message}` };
  }
};
```

#### context.session

**类型：** `SessionStorage`

**用途：** 存储会话状态

**方法：**
```typescript
// 存储
context.session.set('key', value);

// 读取
const value = context.session.get('key');

// 删除
context.session.delete('key');

// 清空
context.session.clear();
```

**示例：**
```typescript
const execute = async ({ task }, context) => {
  // 读取任务列表
  const tasks = context.session.get('tasks') || [];
  
  // 添加新任务
  tasks.push({ id: Date.now(), task });
  
  // 保存
  context.session.set('tasks', tasks);
  
  return { content: `Task added. Total: ${tasks.length}` };
};
```

#### context.signal

**类型：** `AbortSignal`

**用途：** 支持取消操作

**使用方式：**
```typescript
const execute = async (params, context) => {
  const { signal } = context;
  
  // 检查是否已取消
  if (signal.aborted) {
    return { content: 'Operation cancelled' };
  }
  
  // 传递给异步操作
  const data = await fetch(url, { signal });
  
  // 再次检查
  if (signal.aborted) {
    return { content: 'Operation cancelled' };
  }
  
  return { content: data };
};
```

**监听取消事件：**
```typescript
const execute = async (params, context) => {
  const { signal } = context;
  
  signal.addEventListener('abort', () => {
    context.logger.info('Operation cancelled by user');
  });
  
  // ...
};
```

#### context.onUpdate

**类型：** `((update: AgentToolResult) => void) | undefined`

**用途：** 流式更新进度

**使用方式：**
```typescript
const execute = async (params, context) => {
  const { onUpdate } = context;
  
  // 发送进度更新
  onUpdate?.({ content: 'Starting...' });
  await step1();
  
  onUpdate?.({ content: 'Processing...' });
  await step2();
  
  onUpdate?.({ content: 'Finishing...' });
  await step3();
  
  return { content: 'Complete!' };
};
```

**注意：** onUpdate 可能是 undefined，使用可选链调用

---

## 返回值详解

### AgentToolResult 结构

```typescript
interface AgentToolResult {
  content: string;           // 主要内容（必需）
  metadata?: {               // 元数据（可选）
    [key: string]: any;
  };
}
```

### content 字段

**类型：** `string`

**用途：** 工具的主要输出，会被发送给 LLM

**要求：**
- 必须是字符串
- 应该包含工具执行的结果
- 如果是结构化数据，使用 JSON.stringify()

**示例：**

```typescript
// 简单文本
return { content: 'File read successfully' };

// 结构化数据
return { content: JSON.stringify({ count: 10, items: [...] }) };

// 多行文本
return { content: `
File: ${path}
Size: ${size} bytes
Modified: ${modified}
`.trim() };

// 错误信息
return { content: `Error: ${error.message}` };
```

### metadata 字段

**类型：** `{ [key: string]: any }`

**用途：** 附加信息，不会直接发送给 LLM，但可以被其他 Extension 使用

**使用场景：**
- 性能指标（执行时间、资源使用）
- 错误标记
- 结构化数据
- 调试信息

**示例：**

```typescript
return {
  content: 'Search complete',
  metadata: {
    totalResults: 100,
    returnedResults: 10,
    executionTime: 1234,
    query: 'search term'
  }
};

// 错误标记
return {
  content: 'Error: File not found',
  metadata: {
    error: true,
    errorType: 'NotFound',
    errorCode: 404
  }
};
```

---

## 实战示例

### 示例 1：文件读取工具

```typescript
import * as fs from 'fs/promises';

const execute = async ({ path }, context) => {
  const { logger, signal } = context;
  
  try {
    logger.info('Reading file:', path);
    
    // 检查取消
    if (signal.aborted) {
      return { content: 'Operation cancelled' };
    }
    
    // 读取文件
    const content = await fs.readFile(path, 'utf-8');
    
    // 再次检查取消
    if (signal.aborted) {
      return { content: 'Operation cancelled' };
    }
    
    logger.info('File read successfully');
    
    return {
      content,
      metadata: {
        fileSize: content.length,
        path,
        timestamp: new Date().toISOString()
      }
    };
  } catch (error) {
    logger.error('Failed to read file:', error);
    
    return {
      content: `Error reading file: ${error.message}`,
      metadata: {
        error: true,
        errorType: error.code,
        path
      }
    };
  }
};
```

### 示例 2：API 调用工具

```typescript
const execute = async ({ url, method = 'GET', headers, body }, context) => {
  const { logger, signal, onUpdate } = context;
  
  try {
    logger.info(`${method} ${url}`);
    
    onUpdate?.({ content: 'Sending request...' });
    
    const response = await fetch(url, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
      signal  // 支持取消
    });
    
    onUpdate?.({ content: 'Receiving response...' });
    
    const data = await response.text();
    
    logger.info('Response received:', response.status);
    
    return {
      content: data,
      metadata: {
        status: response.status,
        statusText: response.statusText,
        headers: Object.fromEntries(response.headers.entries()),
        url
      }
    };
  } catch (error) {
    if (error.name === 'AbortError') {
      logger.info('Request cancelled');
      return { content: 'Request cancelled by user' };
    }
    
    logger.error('Request failed:', error);
    return {
      content: `Error: ${error.message}`,
      metadata: { error: true }
    };
  }
};
```

### 示例 3：状态管理工具

```typescript
const execute = async ({ action, taskId, task }, context) => {
  const { session, logger } = context;
  
  // 读取任务列表
  const tasks = session.get('tasks') || [];
  
  switch (action) {
    case 'add':
      const newTask = {
        id: Date.now(),
        task,
        completed: false,
        createdAt: new Date().toISOString()
      };
      tasks.push(newTask);
      session.set('tasks', tasks);
      logger.info('Task added:', newTask.id);
      return {
        content: `Task added: ${task}`,
        metadata: { taskId: newTask.id }
      };
      
    case 'list':
      const list = tasks.map(t => 
        `${t.completed ? '✓' : '○'} ${t.task}`
      ).join('\n');
      return {
        content: list || 'No tasks',
        metadata: { count: tasks.length }
      };
      
    case 'complete':
      const taskToComplete = tasks.find(t => t.id === taskId);
      if (taskToComplete) {
        taskToComplete.completed = true;
        session.set('tasks', tasks);
        logger.info('Task completed:', taskId);
        return { content: 'Task completed' };
      }
      return { content: 'Task not found' };
      
    case 'delete':
      const index = tasks.findIndex(t => t.id === taskId);
      if (index !== -1) {
        tasks.splice(index, 1);
        session.set('tasks', tasks);
        logger.info('Task deleted:', taskId);
        return { content: 'Task deleted' };
      }
      return { content: 'Task not found' };
      
    default:
      return { content: `Unknown action: ${action}` };
  }
};
```

### 示例 4：长时间运行的工具

```typescript
const execute = async ({ items }, context) => {
  const { logger, signal, onUpdate } = context;
  
  const results = [];
  const total = items.length;
  
  for (let i = 0; i < total; i++) {
    // 检查取消
    if (signal.aborted) {
      logger.info('Processing cancelled');
      return {
        content: `Cancelled after processing ${i}/${total} items`,
        metadata: {
          cancelled: true,
          processed: i,
          total
        }
      };
    }
    
    // 更新进度
    const progress = Math.round((i / total) * 100);
    onUpdate?.({ 
      content: `Processing ${i + 1}/${total} (${progress}%)` 
    });
    
    // 处理项目
    try {
      const result = await processItem(items[i]);
      results.push(result);
    } catch (error) {
      logger.error(`Failed to process item ${i}:`, error);
      results.push({ error: error.message });
    }
  }
  
  logger.info('Processing complete');
  
  return {
    content: `Processed ${total} items`,
    metadata: {
      total,
      successful: results.filter(r => !r.error).length,
      failed: results.filter(r => r.error).length,
      results
    }
  };
};
```

---

## 最佳实践

### 1. 始终使用 async

```typescript
// ✅ 正确
const execute = async (params, context) => {
  return { content: 'Result' };
};

// ❌ 错误
const execute = (params, context) => {
  return { content: 'Result' };
};
```

### 2. 使用 try-catch 处理错误

```typescript
// ✅ 正确
const execute = async (params, context) => {
  try {
    const result = await operation();
    return { content: result };
  } catch (error) {
    context.logger.error('Operation failed:', error);
    return { content: `Error: ${error.message}` };
  }
};

// ❌ 错误（未处理错误）
const execute = async (params, context) => {
  const result = await operation();  // 可能抛出错误
  return { content: result };
};
```

### 3. 记录日志

```typescript
// ✅ 正确
const execute = async (params, context) => {
  context.logger.info('Starting operation');
  const result = await operation();
  context.logger.info('Operation complete');
  return { content: result };
};
```

### 4. 支持取消操作

```typescript
// ✅ 正确
const execute = async (params, context) => {
  const { signal } = context;
  
  if (signal.aborted) {
    return { content: 'Cancelled' };
  }
  
  const result = await operation({ signal });
  
  if (signal.aborted) {
    return { content: 'Cancelled' };
  }
  
  return { content: result };
};
```

### 5. 提供进度反馈

```typescript
// ✅ 正确（长时间操作）
const execute = async (params, context) => {
  const { onUpdate } = context;
  
  onUpdate?.({ content: 'Step 1/3...' });
  await step1();
  
  onUpdate?.({ content: 'Step 2/3...' });
  await step2();
  
  onUpdate?.({ content: 'Step 3/3...' });
  await step3();
  
  return { content: 'Complete!' };
};
```

### 6. 使用 metadata 存储结构化数据

```typescript
// ✅ 正确
return {
  content: 'Found 10 results',
  metadata: {
    count: 10,
    results: [...],
    executionTime: 1234
  }
};

// ❌ 不好（所有信息都在 content 中）
return {
  content: JSON.stringify({
    message: 'Found 10 results',
    count: 10,
    results: [...]
  })
};
```

### 7. 友好的错误信息

```typescript
// ✅ 正确
return { 
  content: 'Error: File not found. Please check the file path.' 
};

// ❌ 不好
return { 
  content: 'Error: ENOENT' 
};
```

---

## 常见错误

### 错误 1：忘记 async

```typescript
// ❌ 错误
const execute = ({ path }, context) => {
  const content = fs.readFileSync(path, 'utf-8');
  return { content };
};

// ✅ 正确
const execute = async ({ path }, context) => {
  const content = await fs.readFile(path, 'utf-8');
  return { content };
};
```

### 错误 2：不处理错误

```typescript
// ❌ 错误
const execute = async ({ url }, context) => {
  const response = await fetch(url);  // 可能失败
  const data = await response.text();
  return { content: data };
};

// ✅ 正确
const execute = async ({ url }, context) => {
  try {
    const response = await fetch(url);
    const data = await response.text();
    return { content: data };
  } catch (error) {
    return { content: `Error: ${error.message}` };
  }
};
```

### 错误 3：忽略 AbortSignal

```typescript
// ❌ 错误（不支持取消）
const execute = async ({ items }, context) => {
  for (const item of items) {
    await processItem(item);  // 无法取消
  }
  return { content: 'Done' };
};

// ✅ 正确
const execute = async ({ items }, context) => {
  const { signal } = context;
  
  for (const item of items) {
    if (signal.aborted) {
      return { content: 'Cancelled' };
    }
    await processItem(item);
  }
  return { content: 'Done' };
};
```

### 错误 4：返回非字符串 content

```typescript
// ❌ 错误
return { content: { result: 'data' } };  // content 必须是字符串

// ✅ 正确
return { content: JSON.stringify({ result: 'data' }) };
```

---

## 2025-2026 最新趋势

### 1. 流式输出成为标准

```typescript
// 2025-2026 年，流式输出被广泛使用
const execute = async (params, context) => {
  const { onUpdate } = context;
  
  // 实时更新
  for await (const chunk of streamData()) {
    onUpdate?.({ content: chunk });
  }
  
  return { content: 'Complete' };
};
```

### 2. 结构化输出

```typescript
// 2025-2026 年，工具返回结构化数据
return {
  content: JSON.stringify({
    type: 'search_result',
    data: {
      query: 'search term',
      results: [...]
    }
  }),
  metadata: {
    schema: 'search_result_v1'
  }
};
```

### 3. 性能监控

```typescript
// 2025-2026 年，性能监控成为标准
const execute = async (params, context) => {
  const startTime = performance.now();
  
  try {
    const result = await operation();
    const duration = performance.now() - startTime;
    
    context.logger.info('Execution time:', duration, 'ms');
    
    return {
      content: result,
      metadata: {
        executionTime: duration,
        timestamp: new Date().toISOString()
      }
    };
  } catch (error) {
    const duration = performance.now() - startTime;
    context.logger.error('Failed after', duration, 'ms');
    throw error;
  }
};
```

---

## 总结

**execute 函数的关键要点：**

1. **签名**：`async (params, context) => Promise<AgentToolResult>`
2. **params**：已验证的参数，可以直接使用
3. **context**：提供 logger, session, signal, onUpdate
4. **返回值**：`{ content: string, metadata?: any }`

**最佳实践：**
- 始终使用 async
- 使用 try-catch 处理错误
- 记录日志
- 支持取消操作（signal）
- 提供进度反馈（onUpdate）
- 使用 metadata 存储结构化数据
- 友好的错误信息

理解 execute 函数的设计和最佳实践，你就能编写出健壮、用户友好的工具！
