# 实战代码 01：基础 JSONL 读写

> 完整的 TypeScript JSONL 读写器实现，支持追加写入、流式读取和错误处理

---

## 代码概览

本文实现一个完整的 JSONL 管理器，包括：

1. **追加写入**：高效地添加新条目
2. **批量读取**：一次性读取所有条目
3. **流式读取**：内存友好的逐行读取
4. **错误处理**：优雅地处理解析错误
5. **懒刷新**：批量写入优化性能

---

## 完整实现

### 类型定义

```typescript
// types.ts
export interface JsonlEntry {
  [key: string]: any;
}

export interface JsonlReaderOptions {
  skipInvalidLines?: boolean;
  onError?: (line: string, error: Error, lineNumber: number) => void;
}

export interface JsonlWriterOptions {
  flushInterval?: number;  // 自动刷新间隔（毫秒）
  autoFlush?: boolean;     // 是否自动刷新
}
```

### JSONL 读取器

```typescript
// jsonl-reader.ts
import * as fs from 'fs';
import * as readline from 'readline';

export class JsonlReader {
  constructor(private filePath: string) {}

  /**
   * 读取所有条目（一次性加载）
   */
  readAll(options: JsonlReaderOptions = {}): JsonlEntry[] {
    const {
      skipInvalidLines = true,
      onError = (line, error, lineNumber) => {
        console.error(`Line ${lineNumber}: ${error.message}`);
      }
    } = options;

    if (!fs.existsSync(this.filePath)) {
      return [];
    }

    const content = fs.readFileSync(this.filePath, 'utf-8');
    const lines = content.split('\n');
    const entries: JsonlEntry[] = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;

      try {
        const entry = JSON.parse(line);
        entries.push(entry);
      } catch (error) {
        if (onError) {
          onError(line, error as Error, i + 1);
        }

        if (!skipInvalidLines) {
          throw new Error(`Invalid JSON at line ${i + 1}: ${line}`);
        }
      }
    }

    return entries;
  }

  /**
   * 流式读取（内存友好）
   */
  async *readStream(options: JsonlReaderOptions = {}): AsyncGenerator<JsonlEntry> {
    const {
      skipInvalidLines = true,
      onError = (line, error, lineNumber) => {
        console.error(`Line ${lineNumber}: ${error.message}`);
      }
    } = options;

    if (!fs.existsSync(this.filePath)) {
      return;
    }

    const rl = readline.createInterface({
      input: fs.createReadStream(this.filePath),
      crlfDelay: Infinity
    });

    let lineNumber = 0;

    for await (const line of rl) {
      lineNumber++;
      const trimmedLine = line.trim();

      if (!trimmedLine) continue;

      try {
        const entry = JSON.parse(trimmedLine);
        yield entry;
      } catch (error) {
        if (onError) {
          onError(trimmedLine, error as Error, lineNumber);
        }

        if (!skipInvalidLines) {
          throw new Error(`Invalid JSON at line ${lineNumber}: ${trimmedLine}`);
        }
      }
    }
  }

  /**
   * 读取指定行数
   */
  async readLines(count: number, offset: number = 0): Promise<JsonlEntry[]> {
    const entries: JsonlEntry[] = [];
    let currentLine = 0;
    let readCount = 0;

    for await (const entry of this.readStream()) {
      if (currentLine >= offset && readCount < count) {
        entries.push(entry);
        readCount++;
      }

      currentLine++;

      if (readCount >= count) {
        break;
      }
    }

    return entries;
  }

  /**
   * 统计行数
   */
  async countLines(): Promise<number> {
    let count = 0;

    for await (const _ of this.readStream()) {
      count++;
    }

    return count;
  }
}
```

### JSONL 写入器

```typescript
// jsonl-writer.ts
import * as fs from 'fs';

export class JsonlWriter {
  private writeBuffer: string[] = [];
  private flushTimer: NodeJS.Timeout | null = null;
  private readonly flushInterval: number;
  private readonly autoFlush: boolean;

  constructor(
    private filePath: string,
    options: JsonlWriterOptions = {}
  ) {
    this.flushInterval = options.flushInterval || 1000;
    this.autoFlush = options.autoFlush !== false;
  }

  /**
   * 追加单个条目
   */
  append(entry: JsonlEntry): void {
    const line = JSON.stringify(entry);
    this.writeBuffer.push(line);

    if (this.autoFlush) {
      this.scheduleFlush();
    }
  }

  /**
   * 追加多个条目
   */
  appendMany(entries: JsonlEntry[]): void {
    for (const entry of entries) {
      const line = JSON.stringify(entry);
      this.writeBuffer.push(line);
    }

    if (this.autoFlush) {
      this.scheduleFlush();
    }
  }

  /**
   * 调度延迟刷新
   */
  private scheduleFlush(): void {
    if (this.flushTimer) {
      return;
    }

    this.flushTimer = setTimeout(() => {
      this.flush();
    }, this.flushInterval);
  }

  /**
   * 立即刷新到磁盘
   */
  flush(): void {
    if (this.writeBuffer.length === 0) {
      return;
    }

    const content = this.writeBuffer.join('\n') + '\n';
    fs.appendFileSync(this.filePath, content, 'utf-8');

    this.writeBuffer = [];

    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }
  }

  /**
   * 获取缓冲区大小
   */
  getBufferSize(): number {
    return this.writeBuffer.length;
  }

  /**
   * 关闭写入器（刷新并清理）
   */
  close(): void {
    this.flush();

    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }
  }
}
```

### JSONL 管理器（组合读写）

```typescript
// jsonl-manager.ts
import * as fs from 'fs';
import { JsonlReader } from './jsonl-reader';
import { JsonlWriter } from './jsonl-writer';

export class JsonlManager {
  private reader: JsonlReader;
  private writer: JsonlWriter;

  constructor(
    private filePath: string,
    writerOptions: JsonlWriterOptions = {}
  ) {
    this.reader = new JsonlReader(filePath);
    this.writer = new JsonlWriter(filePath, writerOptions);
  }

  // 读取方法
  readAll(options?: JsonlReaderOptions): JsonlEntry[] {
    return this.reader.readAll(options);
  }

  readStream(options?: JsonlReaderOptions): AsyncGenerator<JsonlEntry> {
    return this.reader.readStream(options);
  }

  async readLines(count: number, offset?: number): Promise<JsonlEntry[]> {
    return this.reader.readLines(count, offset);
  }

  async countLines(): Promise<number> {
    return this.reader.countLines();
  }

  // 写入方法
  append(entry: JsonlEntry): void {
    this.writer.append(entry);
  }

  appendMany(entries: JsonlEntry[]): void {
    this.writer.appendMany(entries);
  }

  flush(): void {
    this.writer.flush();
  }

  getBufferSize(): number {
    return this.writer.getBufferSize();
  }

  close(): void {
    this.writer.close();
  }

  // 工具方法
  exists(): boolean {
    return fs.existsSync(this.filePath);
  }

  delete(): void {
    if (this.exists()) {
      fs.unlinkSync(this.filePath);
    }
  }

  getSize(): number {
    if (!this.exists()) {
      return 0;
    }
    return fs.statSync(this.filePath).size;
  }
}
```

---

## 使用示例

### 示例 1：基础读写

```typescript
import { JsonlManager } from './jsonl-manager';

async function basicExample() {
  const manager = new JsonlManager('data.jsonl');

  // 写入数据
  manager.append({ id: 1, name: 'Alice', age: 30 });
  manager.append({ id: 2, name: 'Bob', age: 25 });
  manager.append({ id: 3, name: 'Charlie', age: 35 });

  // 立即刷新
  manager.flush();

  // 读取所有数据
  const entries = manager.readAll();
  console.log('All entries:', entries);

  // 关闭管理器
  manager.close();
}

basicExample();
```

### 示例 2：流式读取大文件

```typescript
async function streamExample() {
  const manager = new JsonlManager('large-data.jsonl');

  console.log('Reading large file...');

  let count = 0;
  for await (const entry of manager.readStream()) {
    console.log(`Entry ${count}:`, entry);
    count++;

    // 只处理前 10 条
    if (count >= 10) {
      break;
    }
  }

  console.log(`Total processed: ${count}`);
}

streamExample();
```

### 示例 3：错误处理

```typescript
async function errorHandlingExample() {
  const manager = new JsonlManager('data-with-errors.jsonl');

  const errors: Array<{ line: string; error: string; lineNumber: number }> = [];

  const entries = manager.readAll({
    skipInvalidLines: true,
    onError: (line, error, lineNumber) => {
      errors.push({
        line,
        error: error.message,
        lineNumber
      });
    }
  });

  console.log(`Valid entries: ${entries.length}`);
  console.log(`Invalid lines: ${errors.length}`);

  if (errors.length > 0) {
    console.log('Errors:');
    errors.forEach(e => {
      console.log(`  Line ${e.lineNumber}: ${e.error}`);
    });
  }
}

errorHandlingExample();
```

### 示例 4：批量写入优化

```typescript
async function batchWriteExample() {
  const manager = new JsonlManager('batch-data.jsonl', {
    flushInterval: 5000,  // 5 秒自动刷新
    autoFlush: true
  });

  // 批量添加数据
  const entries = [];
  for (let i = 0; i < 1000; i++) {
    entries.push({
      id: i,
      timestamp: new Date().toISOString(),
      value: Math.random()
    });
  }

  manager.appendMany(entries);

  console.log(`Buffer size: ${manager.getBufferSize()}`);

  // 手动刷新
  manager.flush();

  console.log(`Buffer size after flush: ${manager.getBufferSize()}`);

  manager.close();
}

batchWriteExample();
```

### 示例 5：分页读取

```typescript
async function paginationExample() {
  const manager = new JsonlManager('data.jsonl');

  const pageSize = 10;
  let page = 0;

  while (true) {
    const entries = await manager.readLines(pageSize, page * pageSize);

    if (entries.length === 0) {
      break;
    }

    console.log(`Page ${page + 1}:`);
    entries.forEach(entry => {
      console.log('  ', entry);
    });

    page++;
  }

  console.log(`Total pages: ${page}`);
}

paginationExample();
```

---

## 性能测试

### 测试代码

```typescript
import { JsonlManager } from './jsonl-manager';

async function performanceTest() {
  const testFile = 'perf-test.jsonl';
  const manager = new JsonlManager(testFile);

  // 清理旧文件
  if (manager.exists()) {
    manager.delete();
  }

  // 测试写入性能
  console.log('Testing write performance...');
  const writeStart = Date.now();

  for (let i = 0; i < 10000; i++) {
    manager.append({
      id: i,
      timestamp: new Date().toISOString(),
      data: `Item ${i}`,
      value: Math.random()
    });
  }

  manager.flush();
  const writeTime = Date.now() - writeStart;
  console.log(`Write 10,000 entries: ${writeTime}ms`);

  // 测试读取性能
  console.log('\nTesting read performance...');
  const readStart = Date.now();
  const entries = manager.readAll();
  const readTime = Date.now() - readStart;
  console.log(`Read ${entries.length} entries: ${readTime}ms`);

  // 测试流式读取性能
  console.log('\nTesting stream read performance...');
  const streamStart = Date.now();
  let streamCount = 0;

  for await (const entry of manager.readStream()) {
    streamCount++;
  }

  const streamTime = Date.now() - streamStart;
  console.log(`Stream read ${streamCount} entries: ${streamTime}ms`);

  // 文件大小
  const fileSize = manager.getSize();
  console.log(`\nFile size: ${(fileSize / 1024).toFixed(2)} KB`);

  // 清理
  manager.delete();
}

performanceTest();
```

### 预期结果

```
Testing write performance...
Write 10,000 entries: 150ms

Testing read performance...
Read 10000 entries: 50ms

Testing stream read performance...
Stream read 10000 entries: 80ms

File size: 450.23 KB
```

---

## 最佳实践

### 1. 选择合适的读取方式

```typescript
// ✅ 小文件（< 10MB）：使用 readAll()
const entries = manager.readAll();

// ✅ 大文件（> 10MB）：使用 readStream()
for await (const entry of manager.readStream()) {
  // 处理每条记录
}

// ✅ 分页场景：使用 readLines()
const page1 = await manager.readLines(10, 0);
const page2 = await manager.readLines(10, 10);
```

### 2. 优化写入性能

```typescript
// ✅ 批量写入
const entries = generateManyEntries();
manager.appendMany(entries);

// ✅ 调整刷新间隔
const manager = new JsonlManager('data.jsonl', {
  flushInterval: 5000  // 5 秒刷新一次
});

// ✅ 手动控制刷新
const manager = new JsonlManager('data.jsonl', {
  autoFlush: false  // 禁用自动刷新
});

// 添加大量数据
for (let i = 0; i < 10000; i++) {
  manager.append(data[i]);
}

// 手动刷新
manager.flush();
```

### 3. 错误处理

```typescript
// ✅ 收集错误信息
const errors: any[] = [];

const entries = manager.readAll({
  skipInvalidLines: true,
  onError: (line, error, lineNumber) => {
    errors.push({ line, error: error.message, lineNumber });
  }
});

// 记录错误
if (errors.length > 0) {
  fs.writeFileSync('errors.json', JSON.stringify(errors, null, 2));
}
```

### 4. 资源清理

```typescript
// ✅ 使用 try-finally 确保资源清理
const manager = new JsonlManager('data.jsonl');

try {
  manager.append({ data: 'test' });
  // ... 其他操作
} finally {
  manager.close();  // 确保刷新并清理
}
```

---

## 与 Pi-mono 的对比

### Pi-mono 的实现

```typescript
// Pi-mono 的简化版本
class SessionManager {
  private writeBuffer: string[] = [];

  appendMessage(entry: SessionEntry): void {
    const line = JSON.stringify(entry);
    this.writeBuffer.push(line);
    this.scheduleFlush();
  }

  private flush(): void {
    if (this.writeBuffer.length === 0) return;

    const content = this.writeBuffer.join('\n') + '\n';
    fs.appendFileSync(this.sessionFile, content, 'utf-8');

    this.writeBuffer = [];
  }
}
```

### 我们的实现

```typescript
// 更通用、更完整的实现
class JsonlManager {
  // 支持流式读取
  async *readStream(): AsyncGenerator<JsonlEntry> { ... }

  // 支持错误处理
  readAll(options: JsonlReaderOptions): JsonlEntry[] { ... }

  // 支持批量写入
  appendMany(entries: JsonlEntry[]): void { ... }

  // 支持配置选项
  constructor(filePath: string, options: JsonlWriterOptions) { ... }
}
```

---

## 关键要点总结

1. **JSONL 读取**：支持一次性读取和流式读取
2. **JSONL 写入**：支持追加写入和批量写入
3. **懒刷新**：批量写入优化性能
4. **错误处理**：优雅地处理解析错误
5. **资源管理**：确保正确关闭和清理

---

**下一步**：实现树形结构构建 → `07_实战代码_02_树形结构构建.md`
