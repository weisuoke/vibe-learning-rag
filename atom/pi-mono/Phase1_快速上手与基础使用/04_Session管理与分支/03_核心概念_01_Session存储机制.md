# Session 管理与分支 - 核心概念 01：Session 存储机制

> 深入理解 Pi Session 的 JSONL 存储格式、文件组织和持久化机制

---

## 概述

Pi Session 使用 **JSONL（JSON Lines）追加日志**作为存储格式，这是一种简单、高效、可靠的持久化方案。本文将详细讲解：

- JSONL 文件格式
- 存储位置与文件命名
- 追加写入机制
- 读取与解析
- 文件管理与清理

---

## 1. JSONL 文件格式

### 1.1 什么是 JSONL？

**JSONL（JSON Lines）** 是一种文本格式，每行一个 JSON 对象。

**格式规范：**
```jsonl
{"id":"1","type":"message","content":"Hello"}
{"id":"2","type":"message","content":"World"}
{"id":"3","type":"message","content":"!"}
```

**特点：**
- ✅ 每行一个完整的 JSON 对象
- ✅ 行与行之间用换行符 `\n` 分隔
- ✅ 每行可以独立解析
- ✅ 支持追加写入

### 1.2 为什么选择 JSONL？

**对比其他格式：**

| 格式 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **JSON** | 结构清晰 | 需要重写整个文件 | 小型配置文件 |
| **JSONL** | 支持追加，高效 | 不支持嵌套结构 | 日志、事件流 |
| **CSV** | 简单，通用 | 不支持复杂数据 | 表格数据 |
| **Protocol Buffers** | 高效，类型安全 | 需要schema定义 | 高性能RPC |

**JSONL 的优势：**
```typescript
// JSON - 需要重写整个文件
const data = { messages: [] };
data.messages.push(newMessage);
fs.writeFileSync('session.json', JSON.stringify(data));
// 问题：每次都要重写整个文件

// JSONL - 只追加新行
fs.appendFileSync('session.jsonl', JSON.stringify(newMessage) + '\n');
// 优势：O(1) 写入，不需要读取现有内容
```

### 1.3 Pi Session 的 JSONL 结构

**Entry 类型：**

```typescript
// Session Entry 的基础结构
interface SessionEntry {
  id: string;              // 唯一标识
  parentId?: string;       // 父节点 ID（用于树形结构）
  type: EntryType;         // 条目类型
  timestamp: number;       // 时间戳
  [key: string]: any;      // 其他字段（根据类型不同）
}

// Entry 类型枚举
type EntryType =
  | 'message'              // 用户或 Agent 消息
  | 'tool_call'            // 工具调用
  | 'tool_result'          // 工具结果
  | 'compaction'           // 压缩条目
  | 'branch_summary'       // 分支摘要
  | 'metadata'             // 元数据
  | 'fork_metadata';       // Fork 元数据
```

**示例 JSONL 文件：**

```jsonl
{"id":"1","type":"message","role":"user","content":"帮我实现登录功能","timestamp":1708329600000}
{"id":"2","parentId":"1","type":"message","role":"assistant","content":"好的，我来实现...","timestamp":1708329601000}
{"id":"3","parentId":"2","type":"tool_call","tool":"write","args":{"file":"auth.ts","content":"..."},"timestamp":1708329602000}
{"id":"4","parentId":"3","type":"tool_result","tool":"write","result":"File created","timestamp":1708329603000}
{"id":"5","parentId":"1","type":"message","role":"user","content":"改用 JWT 认证","timestamp":1708329700000}
{"id":"6","parentId":"5","type":"message","role":"assistant","content":"好的，改用 JWT...","timestamp":1708329701000}
```

**树形结构：**
```
1: 帮我实现登录功能
├─ 2: 好的，我来实现...
│  ├─ 3: [tool_call] write
│  │  └─ 4: [tool_result] File created
└─ 5: 改用 JWT 认证
   └─ 6: 好的，改用 JWT...
```

---

## 2. 存储位置与文件命名

### 2.1 默认存储位置

**Pi Session 存储目录：**
```bash
~/.pi/sessions/
```

**完整路径示例：**
```bash
# macOS/Linux
/Users/username/.pi/sessions/

# Windows
C:\Users\username\.pi\sessions\
```

### 2.2 文件命名规则

**命名格式：**
```
{sessionId}.jsonl
```

**示例：**
```bash
~/.pi/sessions/
├── abc123.jsonl
├── def456.jsonl
├── feature-login.jsonl
└── bugfix-auth.jsonl
```

**Session ID 生成：**
```typescript
// 生成随机 Session ID
function generateSessionId(): string {
  return Math.random().toString(36).substring(2, 15);
}

// 或使用 UUID
import { v4 as uuidv4 } from 'uuid';
function generateSessionId(): string {
  return uuidv4();
}

// 或使用自定义名称
function generateSessionId(name?: string): string {
  if (name) {
    // 清理名称，移除非法字符
    return name.replace(/[^a-zA-Z0-9-_]/g, '-');
  }
  return Math.random().toString(36).substring(2, 15);
}
```

### 2.3 文件结构

**目录结构：**
```bash
~/.pi/
├── sessions/              # Session 存储目录
│   ├── abc123.jsonl      # Session 文件
│   ├── def456.jsonl
│   └── ...
├── settings.json         # 全局设置
└── extensions/           # 扩展目录
```

---

## 3. 追加写入机制

### 3.1 追加写入原理

**核心特点：**
- ✅ 只追加，不修改
- ✅ O(1) 写入复杂度
- ✅ 原子性操作
- ✅ 崩溃安全

**实现示例：**

```typescript
import fs from 'fs/promises';
import path from 'path';

class SessionStorage {
  private sessionDir: string;
  private sessionFile: string;

  constructor(sessionId: string) {
    this.sessionDir = path.join(process.env.HOME!, '.pi', 'sessions');
    this.sessionFile = path.join(this.sessionDir, `${sessionId}.jsonl`);
  }

  // 追加条目
  async append(entry: SessionEntry): Promise<void> {
    // 1. 确保目录存在
    await fs.mkdir(this.sessionDir, { recursive: true });

    // 2. 序列化为 JSON
    const line = JSON.stringify(entry) + '\n';

    // 3. 追加到文件
    await fs.appendFile(this.sessionFile, line, 'utf-8');

    // 注意：
    // - appendFile 是原子操作
    // - 即使进程崩溃，已写入的数据不会丢失
    // - 不需要读取现有内容
  }

  // 批量追加
  async appendBatch(entries: SessionEntry[]): Promise<void> {
    const lines = entries.map(e => JSON.stringify(e) + '\n').join('');
    await fs.appendFile(this.sessionFile, lines, 'utf-8');
  }
}
```

### 3.2 原子性保证

**文件系统的原子性：**

```typescript
// appendFile 在大多数文件系统上是原子的
// 这意味着：
// 1. 写入要么完全成功，要么完全失败
// 2. 不会出现部分写入的情况
// 3. 并发写入是安全的（在同一进程内）

// 示例：并发追加
async function concurrentAppend() {
  const storage = new SessionStorage('abc123');

  // 并发追加多个条目
  await Promise.all([
    storage.append({ id: '1', type: 'message', content: 'A' }),
    storage.append({ id: '2', type: 'message', content: 'B' }),
    storage.append({ id: '3', type: 'message', content: 'C' })
  ]);

  // 结果：所有条目都会被正确写入
  // 顺序可能不确定，但不会丢失或损坏
}
```

### 3.3 性能优化

**批量写入：**

```typescript
class OptimizedSessionStorage extends SessionStorage {
  private writeBuffer: SessionEntry[] = [];
  private flushTimer: NodeJS.Timeout | null = null;

  // 添加到缓冲区
  async append(entry: SessionEntry): Promise<void> {
    this.writeBuffer.push(entry);

    // 定时刷新
    if (!this.flushTimer) {
      this.flushTimer = setTimeout(() => this.flush(), 100);
    }

    // 缓冲区满时立即刷新
    if (this.writeBuffer.length >= 10) {
      await this.flush();
    }
  }

  // 刷新缓冲区
  private async flush(): Promise<void> {
    if (this.writeBuffer.length === 0) return;

    const entries = this.writeBuffer.splice(0);
    await this.appendBatch(entries);

    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }
  }
}
```

---

## 4. 读取与解析

### 4.1 读取 JSONL 文件

**基础读取：**

```typescript
class SessionStorage {
  // 读取所有条目
  async readAll(): Promise<SessionEntry[]> {
    try {
      const content = await fs.readFile(this.sessionFile, 'utf-8');

      // 按行分割
      const lines = content.split('\n');

      // 解析每行
      const entries: SessionEntry[] = [];
      for (const line of lines) {
        if (line.trim()) {
          try {
            entries.push(JSON.parse(line));
          } catch (err) {
            console.error('Failed to parse line:', line, err);
            // 跳过损坏的行
          }
        }
      }

      return entries;
    } catch (err) {
      if ((err as any).code === 'ENOENT') {
        // 文件不存在，返回空数组
        return [];
      }
      throw err;
    }
  }

  // 流式读取（适用于大文件）
  async *readStream(): AsyncGenerator<SessionEntry> {
    const stream = fs.createReadStream(this.sessionFile, 'utf-8');
    const readline = require('readline');
    const rl = readline.createInterface({ input: stream });

    for await (const line of rl) {
      if (line.trim()) {
        try {
          yield JSON.parse(line);
        } catch (err) {
          console.error('Failed to parse line:', line, err);
        }
      }
    }
  }
}
```

### 4.2 构建树形结构

**从扁平列表构建树：**

```typescript
interface TreeNode extends SessionEntry {
  children: TreeNode[];
}

class SessionStorage {
  // 构建树形结构
  buildTree(entries: SessionEntry[]): TreeNode[] {
    // 1. 创建节点映射
    const nodeMap = new Map<string, TreeNode>();
    entries.forEach(entry => {
      nodeMap.set(entry.id, { ...entry, children: [] });
    });

    // 2. 建立父子关系
    const roots: TreeNode[] = [];
    entries.forEach(entry => {
      const node = nodeMap.get(entry.id)!;

      if (entry.parentId) {
        const parent = nodeMap.get(entry.parentId);
        if (parent) {
          parent.children.push(node);
        } else {
          // 父节点不存在，作为根节点
          roots.push(node);
        }
      } else {
        // 没有父节点，是根节点
        roots.push(node);
      }
    });

    return roots;
  }

  // 获取到指定节点的路径
  getPathToNode(entries: SessionEntry[], targetId: string): SessionEntry[] {
    const nodeMap = new Map(entries.map(e => [e.id, e]));
    const path: SessionEntry[] = [];

    let currentId: string | undefined = targetId;
    while (currentId) {
      const node = nodeMap.get(currentId);
      if (!node) break;

      path.unshift(node);
      currentId = node.parentId;
    }

    return path;
  }
}
```

### 4.3 查询与过滤

**常见查询操作：**

```typescript
class SessionStorage {
  // 按类型过滤
  async filterByType(type: EntryType): Promise<SessionEntry[]> {
    const entries = await this.readAll();
    return entries.filter(e => e.type === type);
  }

  // 按时间范围过滤
  async filterByTimeRange(
    startTime: number,
    endTime: number
  ): Promise<SessionEntry[]> {
    const entries = await this.readAll();
    return entries.filter(
      e => e.timestamp >= startTime && e.timestamp <= endTime
    );
  }

  // 搜索内容
  async search(keyword: string): Promise<SessionEntry[]> {
    const entries = await this.readAll();
    return entries.filter(e => {
      const content = JSON.stringify(e).toLowerCase();
      return content.includes(keyword.toLowerCase());
    });
  }

  // 获取最近 N 条消息
  async getRecentMessages(count: number): Promise<SessionEntry[]> {
    const entries = await this.readAll();
    const messages = entries.filter(e => e.type === 'message');
    return messages.slice(-count);
  }
}
```

---

## 5. 文件管理与清理

### 5.1 列出所有 Session

**列出 Session 文件：**

```typescript
class SessionManager {
  private sessionsDir: string;

  constructor() {
    this.sessionsDir = path.join(process.env.HOME!, '.pi', 'sessions');
  }

  // 列出所有 Session
  async listSessions(): Promise<SessionInfo[]> {
    await fs.mkdir(this.sessionsDir, { recursive: true });

    const files = await fs.readdir(this.sessionsDir);
    const sessions: SessionInfo[] = [];

    for (const file of files) {
      if (file.endsWith('.jsonl')) {
        const sessionId = file.replace('.jsonl', '');
        const filePath = path.join(this.sessionsDir, file);
        const stats = await fs.stat(filePath);

        sessions.push({
          id: sessionId,
          filePath,
          size: stats.size,
          lastModified: stats.mtime.getTime(),
          created: stats.birthtime.getTime()
        });
      }
    }

    // 按最后修改时间排序
    sessions.sort((a, b) => b.lastModified - a.lastModified);

    return sessions;
  }
}

interface SessionInfo {
  id: string;
  filePath: string;
  size: number;
  lastModified: number;
  created: number;
}
```

### 5.2 删除 Session

**删除操作：**

```typescript
class SessionManager {
  // 删除单个 Session
  async deleteSession(sessionId: string): Promise<void> {
    const filePath = path.join(this.sessionsDir, `${sessionId}.jsonl`);

    try {
      await fs.unlink(filePath);
      console.log(`Session ${sessionId} deleted`);
    } catch (err) {
      if ((err as any).code === 'ENOENT') {
        console.log(`Session ${sessionId} not found`);
      } else {
        throw err;
      }
    }
  }

  // 删除旧 Session
  async deleteOldSessions(olderThanDays: number): Promise<number> {
    const sessions = await this.listSessions();
    const cutoffTime = Date.now() - olderThanDays * 24 * 60 * 60 * 1000;

    let deletedCount = 0;
    for (const session of sessions) {
      if (session.lastModified < cutoffTime) {
        await this.deleteSession(session.id);
        deletedCount++;
      }
    }

    return deletedCount;
  }

  // 清理所有 Session
  async clearAllSessions(): Promise<void> {
    const sessions = await this.listSessions();

    for (const session of sessions) {
      await this.deleteSession(session.id);
    }

    console.log(`Cleared ${sessions.length} sessions`);
  }
}
```

### 5.3 备份与恢复

**备份 Session：**

```typescript
class SessionManager {
  // 备份 Session
  async backupSession(sessionId: string, backupDir: string): Promise<string> {
    const sourceFile = path.join(this.sessionsDir, `${sessionId}.jsonl`);
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const backupFile = path.join(
      backupDir,
      `${sessionId}-${timestamp}.jsonl`
    );

    await fs.mkdir(backupDir, { recursive: true });
    await fs.copyFile(sourceFile, backupFile);

    console.log(`Session backed up to ${backupFile}`);
    return backupFile;
  }

  // 恢复 Session
  async restoreSession(backupFile: string, sessionId?: string): Promise<string> {
    if (!sessionId) {
      sessionId = path.basename(backupFile, '.jsonl').split('-')[0];
    }

    const targetFile = path.join(this.sessionsDir, `${sessionId}.jsonl`);
    await fs.copyFile(backupFile, targetFile);

    console.log(`Session restored to ${targetFile}`);
    return sessionId;
  }

  // 导出 Session 为 JSON
  async exportSession(sessionId: string, outputFile: string): Promise<void> {
    const storage = new SessionStorage(sessionId);
    const entries = await storage.readAll();

    const exportData = {
      sessionId,
      exportTime: Date.now(),
      entryCount: entries.length,
      entries
    };

    await fs.writeFile(outputFile, JSON.stringify(exportData, null, 2));
    console.log(`Session exported to ${outputFile}`);
  }
}
```

---

## 6. 实际应用示例

### 示例 1：完整的 Session 管理器

```typescript
import fs from 'fs/promises';
import path from 'path';

class CompleteSessionManager {
  private sessionsDir: string;

  constructor(sessionsDir?: string) {
    this.sessionsDir = sessionsDir || path.join(
      process.env.HOME!,
      '.pi',
      'sessions'
    );
  }

  // 创建新 Session
  async createSession(name?: string): Promise<Session> {
    const sessionId = name || this.generateSessionId();
    const storage = new SessionStorage(sessionId);

    // 添加初始元数据
    await storage.append({
      id: this.generateEntryId(),
      type: 'metadata',
      sessionId,
      created: Date.now(),
      version: '1.0'
    });

    return new Session(sessionId, storage);
  }

  // 加载 Session
  async loadSession(sessionId: string): Promise<Session> {
    const storage = new SessionStorage(sessionId);
    const entries = await storage.readAll();

    if (entries.length === 0) {
      throw new Error(`Session ${sessionId} not found`);
    }

    return new Session(sessionId, storage);
  }

  // 列出所有 Session
  async listSessions(): Promise<SessionInfo[]> {
    await fs.mkdir(this.sessionsDir, { recursive: true });

    const files = await fs.readdir(this.sessionsDir);
    const sessions: SessionInfo[] = [];

    for (const file of files) {
      if (file.endsWith('.jsonl')) {
        const sessionId = file.replace('.jsonl', '');
        const filePath = path.join(this.sessionsDir, file);
        const stats = await fs.stat(filePath);

        // 读取第一条记录获取元数据
        const storage = new SessionStorage(sessionId);
        const entries = await storage.readAll();
        const metadata = entries.find(e => e.type === 'metadata');

        sessions.push({
          id: sessionId,
          name: metadata?.name || sessionId,
          filePath,
          size: stats.size,
          entryCount: entries.length,
          lastModified: stats.mtime.getTime(),
          created: stats.birthtime.getTime()
        });
      }
    }

    sessions.sort((a, b) => b.lastModified - a.lastModified);
    return sessions;
  }

  private generateSessionId(): string {
    return Math.random().toString(36).substring(2, 15);
  }

  private generateEntryId(): string {
    return Math.random().toString(36).substring(2, 15);
  }
}

// Session 类
class Session {
  constructor(
    public readonly id: string,
    private storage: SessionStorage
  ) {}

  // 追加消息
  async appendMessage(role: 'user' | 'assistant', content: string): Promise<string> {
    const entryId = this.generateEntryId();

    await this.storage.append({
      id: entryId,
      type: 'message',
      role,
      content,
      timestamp: Date.now()
    });

    return entryId;
  }

  // 获取上下文
  async getContext(): Promise<SessionEntry[]> {
    return await this.storage.readAll();
  }

  // 构建树
  async getTree(): Promise<TreeNode[]> {
    const entries = await this.storage.readAll();
    return this.storage.buildTree(entries);
  }

  private generateEntryId(): string {
    return Math.random().toString(36).substring(2, 15);
  }
}
```

### 示例 2：使用 Session 管理器

```typescript
async function main() {
  const manager = new CompleteSessionManager();

  // 1. 创建新 Session
  const session = await manager.createSession('my-project');
  console.log(`Created session: ${session.id}`);

  // 2. 添加消息
  await session.appendMessage('user', '帮我实现登录功能');
  await session.appendMessage('assistant', '好的，我来实现...');

  // 3. 获取上下文
  const context = await session.getContext();
  console.log(`Context has ${context.length} entries`);

  // 4. 列出所有 Session
  const sessions = await manager.listSessions();
  console.log(`Total sessions: ${sessions.length}`);
  sessions.forEach(s => {
    console.log(`- ${s.name} (${s.entryCount} entries, ${s.size} bytes)`);
  });

  // 5. 加载已有 Session
  const loadedSession = await manager.loadSession(session.id);
  const tree = await loadedSession.getTree();
  console.log(`Tree has ${tree.length} roots`);
}

main().catch(console.error);
```

---

## 7. 性能考虑

### 7.1 读取性能

**问题：**
- JSONL 文件需要全文件扫描
- 大文件读取慢

**优化方案：**

```typescript
// 1. 使用流式读取
async function streamRead(sessionId: string) {
  const storage = new SessionStorage(sessionId);

  for await (const entry of storage.readStream()) {
    // 处理每个条目
    console.log(entry);
  }
}

// 2. 缓存最近的条目
class CachedSessionStorage extends SessionStorage {
  private cache: SessionEntry[] = [];
  private cacheSize = 100;

  async append(entry: SessionEntry): Promise<void> {
    await super.append(entry);

    // 更新缓存
    this.cache.push(entry);
    if (this.cache.length > this.cacheSize) {
      this.cache.shift();
    }
  }

  async getRecentEntries(count: number): Promise<SessionEntry[]> {
    if (count <= this.cache.length) {
      return this.cache.slice(-count);
    }

    // 缓存不足，读取文件
    const entries = await this.readAll();
    return entries.slice(-count);
  }
}
```

### 7.2 写入性能

**优化方案：**

```typescript
// 批量写入
class BatchSessionStorage extends SessionStorage {
  private buffer: SessionEntry[] = [];
  private flushInterval = 100; // ms

  async append(entry: SessionEntry): Promise<void> {
    this.buffer.push(entry);

    if (this.buffer.length >= 10) {
      await this.flush();
    }
  }

  private async flush(): Promise<void> {
    if (this.buffer.length === 0) return;

    const entries = this.buffer.splice(0);
    await this.appendBatch(entries);
  }
}
```

---

## 学习检查清单

完成本节学习后，你应该能够：

- [ ] 理解 JSONL 文件格式
- [ ] 理解 Pi Session 的存储位置
- [ ] 理解追加写入机制
- [ ] 实现 Session 的读取与解析
- [ ] 实现树形结构的构建
- [ ] 实现 Session 的管理与清理
- [ ] 理解性能优化策略

---

**版本：** v1.0
**最后更新：** 2026-02-18
**维护者：** Claude Code
