# 核心概念 01：JSONL 格式与追加写入

> 深入理解 Pi-mono 使用 JSONL 格式和追加写入机制的设计原理

---

## JSONL 格式定义

### 什么是 JSONL？

**JSONL（JSON Lines）** 是一种文本格式，每行包含一个独立的 JSON 对象。

**基本格式：**

```jsonl
{"id": "1", "type": "user", "content": "Hello"}
{"id": "2", "type": "assistant", "content": "Hi there"}
{"id": "3", "type": "user", "content": "How are you?"}
```

**关键特性：**
- 每行一个完整的 JSON 对象
- 行与行之间独立，可以单独解析
- 使用换行符（`\n`）分隔
- 支持流式处理

### JSONL vs JSON Array

| 特性 | JSONL | JSON Array |
|------|-------|------------|
| **格式** | 每行一个对象 | 数组包含所有对象 |
| **追加** | ✅ 直接追加新行 | ❌ 需要修改整个文件 |
| **解析** | ✅ 逐行解析 | ❌ 需要解析整个文件 |
| **内存** | ✅ 流式处理，内存友好 | ❌ 需要加载整个数组 |
| **错误恢复** | ✅ 单行错误不影响其他行 | ❌ 格式错误导致整个文件无法解析 |
| **可读性** | ✅ 易于查看和编辑 | ⚠️ 大文件难以查看 |

**示例对比：**

**JSON Array：**
```json
{
  "messages": [
    {"id": "1", "content": "Hello"},
    {"id": "2", "content": "Hi"}
  ]
}
```

**JSONL：**
```jsonl
{"id": "1", "content": "Hello"}
{"id": "2", "content": "Hi"}
```

---

## 为什么选择 JSONL？

### 1. 追加写入的天然支持

**问题：** 如何高效地添加新消息？

**JSON Array 方案：**
```typescript
// 需要读取整个文件
const data = JSON.parse(fs.readFileSync('session.json', 'utf-8'));
// 修改数组
data.messages.push(newMessage);
// 写回整个文件
fs.writeFileSync('session.json', JSON.stringify(data, null, 2));
```

**性能问题：**
- 文件越大，读写越慢
- 需要加载整个文件到内存
- 并发写入容易冲突

**JSONL 方案：**
```typescript
// 直接追加新行
fs.appendFileSync('session.jsonl', JSON.stringify(newMessage) + '\n');
```

**性能优势：**
- O(1) 复杂度，与文件大小无关
- 不需要读取现有内容
- 支持并发追加（使用文件锁）

### 2. 流式处理

**JSONL 支持逐行读取：**

```typescript
import * as readline from 'readline';
import * as fs from 'fs';

const rl = readline.createInterface({
  input: fs.createReadStream('session.jsonl'),
  crlfDelay: Infinity
});

for await (const line of rl) {
  const entry = JSON.parse(line);
  console.log(entry);
  // 处理每条消息，不需要加载整个文件
}
```

**优势：**
- 内存占用恒定，不随文件大小增长
- 可以处理超大文件（GB 级别）
- 支持实时处理（tail -f）

### 3. 错误恢复

**JSONL 的容错性：**

```jsonl
{"id": "1", "content": "Hello"}
{"id": "2", "content": "Hi"}
{"id": "3", "content": "Invalid JSON  // 这行有错误
{"id": "4", "content": "Still works"}
```

**处理方式：**
```typescript
for await (const line of rl) {
  try {
    const entry = JSON.parse(line);
    // 处理有效的行
  } catch (error) {
    console.error('Invalid line:', line);
    // 跳过错误行，继续处理
  }
}
```

**JSON Array 的问题：**
- 一个语法错误导致整个文件无法解析
- 需要手动修复才能恢复

---

## Pi-mono 的文件结构

### SessionHeader（第一行）

**第一行是 Session 的元数据：**

```typescript
interface SessionHeader {
  type: "session";
  version?: number;
  id: string;           // Session ID
  timestamp: string;    // ISO 8601 格式
  cwd: string;          // 工作目录
  parentSession?: string; // 父 Session ID（如果是分支）
}
```

**示例：**
```jsonl
{"type":"session","id":"abc12345","timestamp":"2024-01-01T10:00:00.000Z","cwd":"/home/user/project"}
```

### SessionEntry（后续行）

**后续行是会话中的各种条目：**

```typescript
interface SessionEntryBase {
  type: string;         // 条目类型
  id: string;           // 8-char hex ID
  parentId: string | null; // 父条目 ID
  timestamp: string;    // ISO 8601 格式
}

// 用户消息
interface UserMessage extends SessionEntryBase {
  type: "user";
  content: string;
}

// AI 响应
interface AssistantMessage extends SessionEntryBase {
  type: "assistant";
  content: string;
}

// 工具调用
interface ToolUse extends SessionEntryBase {
  type: "tool_use";
  name: string;
  input: any;
}

// 工具结果
interface ToolResult extends SessionEntryBase {
  type: "tool_result";
  tool_use_id: string;
  content: string;
}
```

**示例文件：**
```jsonl
{"type":"session","id":"root","timestamp":"2024-01-01T10:00:00.000Z","cwd":"/project"}
{"type":"user","id":"msg1","parentId":"root","timestamp":"2024-01-01T10:00:01.000Z","content":"Hello"}
{"type":"assistant","id":"msg2","parentId":"msg1","timestamp":"2024-01-01T10:00:02.000Z","content":"Hi"}
{"type":"tool_use","id":"tool1","parentId":"msg2","timestamp":"2024-01-01T10:00:03.000Z","name":"read_file","input":{"path":"test.ts"}}
{"type":"tool_result","id":"result1","parentId":"tool1","timestamp":"2024-01-01T10:00:04.000Z","tool_use_id":"tool1","content":"file content"}
```

---

## 追加写入机制

### 基本追加写入

**最简单的实现：**

```typescript
import * as fs from 'fs';

function appendEntry(filePath: string, entry: SessionEntry): void {
  const line = JSON.stringify(entry) + '\n';
  fs.appendFileSync(filePath, line, 'utf-8');
}
```

**使用示例：**
```typescript
const entry: UserMessage = {
  type: 'user',
  id: generateId(),
  parentId: 'msg1',
  timestamp: new Date().toISOString(),
  content: 'Hello world'
};

appendEntry('session.jsonl', entry);
```

### 懒刷新（Lazy Flush）

**问题：** 频繁的磁盘写入影响性能

**解决方案：** 使用缓冲区，批量写入

```typescript
class SessionWriter {
  private buffer: string[] = [];
  private flushTimer: NodeJS.Timeout | null = null;
  private readonly flushInterval = 1000; // 1秒

  constructor(private filePath: string) {}

  append(entry: SessionEntry): void {
    const line = JSON.stringify(entry);
    this.buffer.push(line);

    // 设置延迟刷新
    if (!this.flushTimer) {
      this.flushTimer = setTimeout(() => this.flush(), this.flushInterval);
    }
  }

  flush(): void {
    if (this.buffer.length === 0) return;

    const content = this.buffer.join('\n') + '\n';
    fs.appendFileSync(this.filePath, content, 'utf-8');

    this.buffer = [];
    this.flushTimer = null;
  }

  close(): void {
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
    }
    this.flush();
  }
}
```

**使用示例：**
```typescript
const writer = new SessionWriter('session.jsonl');

// 添加多条消息
writer.append(entry1);
writer.append(entry2);
writer.append(entry3);

// 自动在 1 秒后批量写入
// 或手动刷新
writer.flush();

// 关闭时确保所有数据写入
writer.close();
```

### 性能优势

**对比测试：**

```typescript
// 方案 A：JSON Array（每次重写整个文件）
function benchmarkJsonArray(count: number): number {
  const start = Date.now();
  const data = { messages: [] };

  for (let i = 0; i < count; i++) {
    data.messages.push({ id: i, content: `Message ${i}` });
    fs.writeFileSync('test.json', JSON.stringify(data));
  }

  return Date.now() - start;
}

// 方案 B：JSONL（追加写入）
function benchmarkJsonl(count: number): number {
  const start = Date.now();

  for (let i = 0; i < count; i++) {
    const line = JSON.stringify({ id: i, content: `Message ${i}` }) + '\n';
    fs.appendFileSync('test.jsonl', line);
  }

  return Date.now() - start;
}

// 测试结果（1000 条消息）
console.log('JSON Array:', benchmarkJsonArray(1000), 'ms'); // ~5000ms
console.log('JSONL:', benchmarkJsonl(1000), 'ms');          // ~100ms
```

**结论：** JSONL 追加写入比 JSON Array 快 50 倍以上。

---

## 手写实现：JSONL 读写器

### 完整的 JSONL 管理器

```typescript
import * as fs from 'fs';
import * as readline from 'readline';
import { randomBytes } from 'crypto';

// 类型定义
interface SessionEntry {
  type: string;
  id: string;
  parentId: string | null;
  timestamp: string;
  [key: string]: any;
}

class JsonlManager {
  private filePath: string;
  private writeBuffer: string[] = [];
  private flushTimer: NodeJS.Timeout | null = null;

  constructor(filePath: string) {
    this.filePath = filePath;
  }

  // 生成 8-char hex ID
  private generateId(): string {
    return randomBytes(4).toString('hex');
  }

  // 追加条目
  append(entry: Omit<SessionEntry, 'id' | 'timestamp'>): string {
    const id = this.generateId();
    const fullEntry: SessionEntry = {
      ...entry,
      id,
      timestamp: new Date().toISOString()
    };

    const line = JSON.stringify(fullEntry);
    this.writeBuffer.push(line);

    // 延迟刷新
    if (!this.flushTimer) {
      this.flushTimer = setTimeout(() => this.flush(), 1000);
    }

    return id;
  }

  // 立即刷新到磁盘
  flush(): void {
    if (this.writeBuffer.length === 0) return;

    const content = this.writeBuffer.join('\n') + '\n';
    fs.appendFileSync(this.filePath, content, 'utf-8');

    this.writeBuffer = [];
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }
  }

  // 读取所有条目
  async readAll(): Promise<SessionEntry[]> {
    if (!fs.existsSync(this.filePath)) {
      return [];
    }

    const entries: SessionEntry[] = [];
    const rl = readline.createInterface({
      input: fs.createReadStream(this.filePath),
      crlfDelay: Infinity
    });

    for await (const line of rl) {
      if (line.trim()) {
        try {
          const entry = JSON.parse(line);
          entries.push(entry);
        } catch (error) {
          console.error('Failed to parse line:', line, error);
        }
      }
    }

    return entries;
  }

  // 流式读取（内存友好）
  async *readStream(): AsyncGenerator<SessionEntry> {
    if (!fs.existsSync(this.filePath)) {
      return;
    }

    const rl = readline.createInterface({
      input: fs.createReadStream(this.filePath),
      crlfDelay: Infinity
    });

    for await (const line of rl) {
      if (line.trim()) {
        try {
          yield JSON.parse(line);
        } catch (error) {
          console.error('Failed to parse line:', line, error);
        }
      }
    }
  }

  // 关闭并刷新
  close(): void {
    this.flush();
  }
}
```

### 使用示例

```typescript
async function example() {
  const manager = new JsonlManager('session.jsonl');

  // 追加条目
  const rootId = manager.append({
    type: 'session',
    parentId: null,
    cwd: '/project'
  });

  const msg1Id = manager.append({
    type: 'user',
    parentId: rootId,
    content: 'Hello'
  });

  const msg2Id = manager.append({
    type: 'assistant',
    parentId: msg1Id,
    content: 'Hi there!'
  });

  // 立即刷新
  manager.flush();

  // 读取所有条目
  const entries = await manager.readAll();
  console.log('Total entries:', entries.length);

  // 流式读取（内存友好）
  for await (const entry of manager.readStream()) {
    console.log('Entry:', entry.type, entry.id);
  }

  // 关闭
  manager.close();
}
```

---

## 2025-2026 最佳实践

### 1. 流式处理大文件

**问题：** 会话文件可能非常大（数万条消息）

**解决方案：** 使用流式处理，避免一次性加载

```typescript
async function processLargeSession(filePath: string): Promise<void> {
  const rl = readline.createInterface({
    input: fs.createReadStream(filePath),
    crlfDelay: Infinity
  });

  let count = 0;
  for await (const line of rl) {
    const entry = JSON.parse(line);

    // 处理每条消息
    if (entry.type === 'user') {
      count++;
    }
  }

  console.log('Total user messages:', count);
}
```

**参考：**
- [Medium - Understanding JSONL](https://medium.com/@kahila.boulbaba.pro/understanding-jsonl-an-efficient-approach-for-structured-data-ad763589d7ec)

### 2. 错误恢复与验证

**问题：** 文件可能损坏或包含无效行

**解决方案：** 跳过无效行，记录错误

```typescript
async function readWithValidation(filePath: string): Promise<SessionEntry[]> {
  const entries: SessionEntry[] = [];
  const errors: Array<{ line: number; content: string; error: string }> = [];

  const rl = readline.createInterface({
    input: fs.createReadStream(filePath),
    crlfDelay: Infinity
  });

  let lineNumber = 0;
  for await (const line of rl) {
    lineNumber++;

    if (!line.trim()) continue;

    try {
      const entry = JSON.parse(line);

      // 验证必需字段
      if (!entry.type || !entry.id) {
        throw new Error('Missing required fields');
      }

      entries.push(entry);
    } catch (error) {
      errors.push({
        line: lineNumber,
        content: line,
        error: error.message
      });
    }
  }

  if (errors.length > 0) {
    console.warn(`Found ${errors.length} invalid lines:`, errors);
  }

  return entries;
}
```

### 3. 并发写入控制

**问题：** 多个进程同时写入可能冲突

**解决方案：** 使用文件锁

```typescript
import * as lockfile from 'proper-lockfile';

class SafeJsonlWriter {
  private filePath: string;

  constructor(filePath: string) {
    this.filePath = filePath;
  }

  async append(entry: SessionEntry): Promise<void> {
    // 获取文件锁
    const release = await lockfile.lock(this.filePath, {
      retries: {
        retries: 5,
        minTimeout: 100,
        maxTimeout: 1000
      }
    });

    try {
      // 安全地追加
      const line = JSON.stringify(entry) + '\n';
      fs.appendFileSync(this.filePath, line, 'utf-8');
    } finally {
      // 释放锁
      await release();
    }
  }
}
```

---

## 在 Pi-mono 中的应用

### SessionManager 的 JSONL 使用

**文件位置：** `sourcecode/pi-mono/packages/coding-agent/src/core/session-manager.ts`

**核心方法：**

```typescript
// 追加新条目
private appendMessage(entry: SessionEntry): void {
  const line = JSON.stringify(entry);

  // 添加到内存缓冲区
  this.writeBuffer.push(line);

  // 延迟刷新
  if (!this.flushTimer) {
    this.flushTimer = setTimeout(() => this.flush(), 1000);
  }
}

// 刷新到磁盘
private flush(): void {
  if (this.writeBuffer.length === 0) return;

  const content = this.writeBuffer.join('\n') + '\n';
  fs.appendFileSync(this.sessionFile, content, 'utf-8');

  this.writeBuffer = [];
  this.flushTimer = null;
}

// 加载会话
private loadSession(): SessionEntry[] {
  if (!fs.existsSync(this.sessionFile)) {
    return [];
  }

  const content = fs.readFileSync(this.sessionFile, 'utf-8');
  const lines = content.split('\n').filter(line => line.trim());

  return lines.map(line => JSON.parse(line));
}
```

**设计要点：**
1. **懒刷新**：批量写入，减少磁盘 I/O
2. **简单加载**：小文件直接全部加载到内存
3. **无锁设计**：单进程写入，无需文件锁

---

## 关键要点总结

1. **JSONL 格式**：每行一个 JSON 对象，支持追加写入
2. **追加写入**：O(1) 复杂度，性能优秀
3. **流式处理**：内存友好，支持大文件
4. **错误恢复**：单行错误不影响其他行
5. **懒刷新**：批量写入，减少磁盘 I/O
6. **极简设计**：不需要数据库，不需要复杂的存储引擎

---

**下一步**：理解树形结构与 id/parentId 设计 → `03_核心概念_02_树形结构与id_parentId设计.md`
