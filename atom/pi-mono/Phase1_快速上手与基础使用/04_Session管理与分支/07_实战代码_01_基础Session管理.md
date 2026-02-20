# Session 管理与分支 - 实战代码 01：基础 Session 管理

> 完整可运行的 TypeScript 代码示例，演示基础 Session 管理功能

---

## 场景：日常开发工作流

实现一个完整的 Session 管理系统，支持：
- 创建和恢复会话
- 添加消息
- 命名和查询会话
- 列出所有会话

---

## 完整代码实现

```typescript
/**
 * Session 管理实战示例
 * 演示：创建、恢复、命名、查询会话的完整工作流
 */

import fs from 'fs/promises';
import path from 'path';

// ===== 1. 类型定义 =====

interface SessionEntry {
  id: string;
  parentId?: string;
  type: 'message' | 'metadata' | 'tool_call' | 'tool_result';
  timestamp: number;
  [key: string]: any;
}

interface MessageEntry extends SessionEntry {
  type: 'message';
  role: 'user' | 'assistant';
  content: string;
}

interface SessionInfo {
  id: string;
  name?: string;
  filePath: string;
  size: number;
  entryCount: number;
  lastModified: number;
  created: number;
}

// ===== 2. Session 存储类 =====

class SessionStorage {
  private sessionFile: string;

  constructor(sessionId: string, sessionsDir: string) {
    this.sessionFile = path.join(sessionsDir, `${sessionId}.jsonl`);
  }

  // 追加条目
  async append(entry: SessionEntry): Promise<void> {
    const line = JSON.stringify(entry) + '\n';
    await fs.appendFile(this.sessionFile, line, 'utf-8');
  }

  // 读取所有条目
  async readAll(): Promise<SessionEntry[]> {
    try {
      const content = await fs.readFile(this.sessionFile, 'utf-8');
      return content
        .split('\n')
        .filter(line => line.trim())
        .map(line => JSON.parse(line));
    } catch (err: any) {
      if (err.code === 'ENOENT') {
        return [];
      }
      throw err;
    }
  }

  // 检查文件是否存在
  async exists(): Promise<boolean> {
    try {
      await fs.access(this.sessionFile);
      return true;
    } catch {
      return false;
    }
  }
}

// ===== 3. Session 类 =====

class Session {
  private storage: SessionStorage;
  private currentNodeId: string = '';

  constructor(
    public readonly id: string,
    private sessionsDir: string
  ) {
    this.storage = new SessionStorage(id, sessionsDir);
  }

  // 初始化会话
  async initialize(): Promise<void> {
    const exists = await this.storage.exists();

    if (!exists) {
      // 创建新会话，添加初始元数据
      await this.storage.append({
        id: this.generateId(),
        type: 'metadata',
        sessionId: this.id,
        created: Date.now(),
        timestamp: Date.now()
      });
    }
  }

  // 添加消息
  async appendMessage(role: 'user' | 'assistant', content: string): Promise<string> {
    const entryId = this.generateId();

    const entry: MessageEntry = {
      id: entryId,
      parentId: this.currentNodeId || undefined,
      type: 'message',
      role,
      content,
      timestamp: Date.now()
    };

    await this.storage.append(entry);
    this.currentNodeId = entryId;

    return entryId;
  }

  // 获取所有消息
  async getMessages(): Promise<MessageEntry[]> {
    const entries = await this.storage.readAll();
    return entries.filter(e => e.type === 'message') as MessageEntry[];
  }

  // 获取上下文（用于 LLM）
  async getContext(): Promise<Array<{ role: string; content: string }>> {
    const messages = await this.getMessages();
    return messages.map(m => ({
      role: m.role,
      content: m.content
    }));
  }

  // 命名会话
  async setName(name: string): Promise<void> {
    await this.storage.append({
      id: this.generateId(),
      type: 'metadata',
      action: 'rename',
      name,
      timestamp: Date.now()
    });
  }

  // 获取会话名称
  async getName(): Promise<string | undefined> {
    const entries = await this.storage.readAll();
    const metadataEntries = entries.filter(e => e.type === 'metadata');

    // 找到最后一个命名操作
    for (let i = metadataEntries.length - 1; i >= 0; i--) {
      if (metadataEntries[i].name) {
        return metadataEntries[i].name;
      }
    }

    return undefined;
  }

  private generateId(): string {
    return Math.random().toString(36).substring(2, 15);
  }
}

// ===== 4. Session 管理器 =====

class SessionManager {
  private sessionsDir: string;

  constructor(sessionsDir?: string) {
    this.sessionsDir = sessionsDir || path.join(
      process.env.HOME || process.env.USERPROFILE || '.',
      '.pi',
      'sessions'
    );
  }

  // 创建新会话
  async createSession(name?: string): Promise<Session> {
    // 确保目录存在
    await fs.mkdir(this.sessionsDir, { recursive: true });

    const sessionId = name || this.generateSessionId();
    const session = new Session(sessionId, this.sessionsDir);

    await session.initialize();

    if (name) {
      await session.setName(name);
    }

    return session;
  }

  // 加载会话
  async loadSession(sessionId: string): Promise<Session> {
    const session = new Session(sessionId, this.sessionsDir);
    const storage = new SessionStorage(sessionId, this.sessionsDir);

    if (!(await storage.exists())) {
      throw new Error(`Session ${sessionId} not found`);
    }

    return session;
  }

  // 列出所有会话
  async listSessions(): Promise<SessionInfo[]> {
    await fs.mkdir(this.sessionsDir, { recursive: true });

    const files = await fs.readdir(this.sessionsDir);
    const sessions: SessionInfo[] = [];

    for (const file of files) {
      if (file.endsWith('.jsonl')) {
        const sessionId = file.replace('.jsonl', '');
        const filePath = path.join(this.sessionsDir, file);

        try {
          const stats = await fs.stat(filePath);
          const storage = new SessionStorage(sessionId, this.sessionsDir);
          const entries = await storage.readAll();

          // 提取名称
          const metadataEntries = entries.filter(e => e.type === 'metadata');
          let name: string | undefined;
          for (let i = metadataEntries.length - 1; i >= 0; i--) {
            if (metadataEntries[i].name) {
              name = metadataEntries[i].name;
              break;
            }
          }

          sessions.push({
            id: sessionId,
            name,
            filePath,
            size: stats.size,
            entryCount: entries.length,
            lastModified: stats.mtime.getTime(),
            created: stats.birthtime.getTime()
          });
        } catch (err) {
          console.error(`Error reading session ${sessionId}:`, err);
        }
      }
    }

    // 按最后修改时间排序
    sessions.sort((a, b) => b.lastModified - a.lastModified);

    return sessions;
  }

  // 删除会话
  async deleteSession(sessionId: string): Promise<void> {
    const filePath = path.join(this.sessionsDir, `${sessionId}.jsonl`);

    try {
      await fs.unlink(filePath);
      console.log(`Session ${sessionId} deleted`);
    } catch (err: any) {
      if (err.code === 'ENOENT') {
        console.log(`Session ${sessionId} not found`);
      } else {
        throw err;
      }
    }
  }

  private generateSessionId(): string {
    return Math.random().toString(36).substring(2, 15);
  }
}

// ===== 5. 使用示例 =====

async function main() {
  console.log('=== Session 管理实战示例 ===\n');

  const manager = new SessionManager();

  // 1. 创建新会话
  console.log('1. 创建新会话');
  const session = await manager.createSession('my-project');
  console.log(`   创建会话: ${session.id}\n`);

  // 2. 添加消息
  console.log('2. 添加消息');
  await session.appendMessage('user', '帮我实现用户登录功能');
  await session.appendMessage('assistant', '好的，我来实现用户登录功能...');
  await session.appendMessage('user', '使用 JWT 认证');
  await session.appendMessage('assistant', '好的，使用 JWT 认证实现...');
  console.log('   已添加 4 条消息\n');

  // 3. 获取上下文
  console.log('3. 获取上下文');
  const context = await session.getContext();
  console.log(`   上下文包含 ${context.length} 条消息:`);
  context.forEach((msg, i) => {
    console.log(`   ${i + 1}. ${msg.role}: ${msg.content.substring(0, 50)}...`);
  });
  console.log();

  // 4. 命名会话
  console.log('4. 命名会话');
  await session.setName('feature-user-login');
  const name = await session.getName();
  console.log(`   会话名称: ${name}\n`);

  // 5. 列出所有会话
  console.log('5. 列出所有会话');
  const sessions = await manager.listSessions();
  console.log(`   总共 ${sessions.length} 个会话:`);
  sessions.forEach((s, i) => {
    console.log(`   ${i + 1}. ${s.name || s.id}`);
    console.log(`      - 条目数: ${s.entryCount}`);
    console.log(`      - 大小: ${s.size} bytes`);
    console.log(`      - 最后修改: ${new Date(s.lastModified).toLocaleString()}`);
  });
  console.log();

  // 6. 恢复会话
  console.log('6. 恢复会话');
  const loadedSession = await manager.loadSession(session.id);
  const loadedMessages = await loadedSession.getMessages();
  console.log(`   恢复会话: ${session.id}`);
  console.log(`   包含 ${loadedMessages.length} 条消息\n`);

  // 7. 创建第二个会话
  console.log('7. 创建第二个会话');
  const session2 = await manager.createSession('feature-payment');
  await session2.appendMessage('user', '实现支付功能');
  await session2.appendMessage('assistant', '好的，我来实现支付功能...');
  console.log(`   创建会话: ${session2.id}\n`);

  // 8. 再次列出所有会话
  console.log('8. 再次列出所有会话');
  const allSessions = await manager.listSessions();
  console.log(`   总共 ${allSessions.length} 个会话:`);
  allSessions.forEach((s, i) => {
    console.log(`   ${i + 1}. ${s.name || s.id} (${s.entryCount} entries)`);
  });
  console.log();

  console.log('=== 示例完成 ===');
}

// 运行示例
main().catch(console.error);
```

---

## 运行输出示例

```
=== Session 管理实战示例 ===

1. 创建新会话
   创建会话: my-project

2. 添加消息
   已添加 4 条消息

3. 获取上下文
   上下文包含 4 条消息:
   1. user: 帮我实现用户登录功能...
   2. assistant: 好的，我来实现用户登录功能......
   3. user: 使用 JWT 认证...
   4. assistant: 好的，使用 JWT 认证实现......

4. 命名会话
   会话名称: feature-user-login

5. 列出所有会话
   总共 1 个会话:
   1. feature-user-login
      - 条目数: 6
      - 大小: 512 bytes
      - 最后修改: 2026-02-18 15:00:00

6. 恢复会话
   恢复会话: my-project
   包含 4 条消息

7. 创建第二个会话
   创建会话: feature-payment

8. 再次列出所有会话
   总共 2 个会话:
   1. feature-payment (3 entries)
   2. feature-user-login (6 entries)

=== 示例完成 ===
```

---

## 应用场景

### 场景 1：日常开发工作流

```typescript
async function dailyWorkflow() {
  const manager = new SessionManager();

  // 早上开始工作
  const session = await manager.createSession('daily-work');

  // 任务 1
  await session.appendMessage('user', '修复登录 bug');
  await session.appendMessage('assistant', '已修复登录 bug');

  // 任务 2
  await session.appendMessage('user', '实现新功能');
  await session.appendMessage('assistant', '已实现新功能');

  // 下班前保存
  await session.setName(`work-${new Date().toISOString().split('T')[0]}`);

  // 第二天继续
  const sessions = await manager.listSessions();
  const yesterdaySession = sessions.find(s => s.name?.includes('2026-02-17'));

  if (yesterdaySession) {
    const resumed = await manager.loadSession(yesterdaySession.id);
    const context = await resumed.getContext();
    console.log('昨天的工作:', context);
  }
}
```

### 场景 2：多项目管理

```typescript
async function multiProjectManagement() {
  const manager = new SessionManager();

  // 项目 A
  const projectA = await manager.createSession('project-a-backend');
  await projectA.appendMessage('user', '实现 API');

  // 项目 B
  const projectB = await manager.createSession('project-b-frontend');
  await projectB.appendMessage('user', '实现 UI');

  // 项目 C
  const projectC = await manager.createSession('project-c-database');
  await projectC.appendMessage('user', '设计数据库');

  // 查看所有项目
  const sessions = await manager.listSessions();
  console.log('当前项目:');
  sessions.forEach(s => {
    console.log(`- ${s.name} (${s.entryCount} entries)`);
  });
}
```

### 场景 3：会话清理

```typescript
async function sessionCleanup() {
  const manager = new SessionManager();

  // 列出所有会话
  const sessions = await manager.listSessions();

  // 删除 30 天前的会话
  const thirtyDaysAgo = Date.now() - 30 * 24 * 60 * 60 * 1000;

  for (const session of sessions) {
    if (session.lastModified < thirtyDaysAgo) {
      console.log(`删除旧会话: ${session.name || session.id}`);
      await manager.deleteSession(session.id);
    }
  }
}
```

---

## 扩展功能

### 扩展 1：搜索会话

```typescript
class SessionManager {
  // 搜索会话
  async searchSessions(keyword: string): Promise<SessionInfo[]> {
    const sessions = await this.listSessions();
    const results: SessionInfo[] = [];

    for (const sessionInfo of sessions) {
      // 搜索名称
      if (sessionInfo.name?.toLowerCase().includes(keyword.toLowerCase())) {
        results.push(sessionInfo);
        continue;
      }

      // 搜索内容
      const session = await this.loadSession(sessionInfo.id);
      const messages = await session.getMessages();

      for (const message of messages) {
        if (message.content.toLowerCase().includes(keyword.toLowerCase())) {
          results.push(sessionInfo);
          break;
        }
      }
    }

    return results;
  }
}

// 使用示例
async function searchExample() {
  const manager = new SessionManager();

  const results = await manager.searchSessions('登录');
  console.log(`找到 ${results.length} 个相关会话:`);
  results.forEach(s => {
    console.log(`- ${s.name || s.id}`);
  });
}
```

### 扩展 2：导出会话

```typescript
class SessionManager {
  // 导出会话为 JSON
  async exportSession(sessionId: string, outputFile: string): Promise<void> {
    const session = await this.loadSession(sessionId);
    const storage = new SessionStorage(sessionId, this.sessionsDir);
    const entries = await storage.readAll();

    const exportData = {
      sessionId,
      name: await session.getName(),
      exportTime: Date.now(),
      entryCount: entries.length,
      entries
    };

    await fs.writeFile(outputFile, JSON.stringify(exportData, null, 2));
    console.log(`会话已导出到: ${outputFile}`);
  }

  // 导入会话
  async importSession(inputFile: string): Promise<Session> {
    const content = await fs.readFile(inputFile, 'utf-8');
    const data = JSON.parse(content);

    const session = await this.createSession(data.sessionId);

    // 写入所有条目
    const storage = new SessionStorage(data.sessionId, this.sessionsDir);
    for (const entry of data.entries) {
      await storage.append(entry);
    }

    return session;
  }
}
```

### 扩展 3：会话统计

```typescript
class SessionManager {
  // 获取会话统计信息
  async getStatistics(): Promise<SessionStatistics> {
    const sessions = await this.listSessions();

    let totalEntries = 0;
    let totalSize = 0;
    let totalMessages = 0;

    for (const sessionInfo of sessions) {
      totalEntries += sessionInfo.entryCount;
      totalSize += sessionInfo.size;

      const session = await this.loadSession(sessionInfo.id);
      const messages = await session.getMessages();
      totalMessages += messages.length;
    }

    return {
      totalSessions: sessions.length,
      totalEntries,
      totalSize,
      totalMessages,
      avgEntriesPerSession: totalEntries / sessions.length,
      avgMessagesPerSession: totalMessages / sessions.length
    };
  }
}

interface SessionStatistics {
  totalSessions: number;
  totalEntries: number;
  totalSize: number;
  totalMessages: number;
  avgEntriesPerSession: number;
  avgMessagesPerSession: number;
}

// 使用示例
async function statisticsExample() {
  const manager = new SessionManager();
  const stats = await manager.getStatistics();

  console.log('会话统计:');
  console.log(`- 总会话数: ${stats.totalSessions}`);
  console.log(`- 总条目数: ${stats.totalEntries}`);
  console.log(`- 总大小: ${(stats.totalSize / 1024).toFixed(2)} KB`);
  console.log(`- 总消息数: ${stats.totalMessages}`);
  console.log(`- 平均每会话条目数: ${stats.avgEntriesPerSession.toFixed(2)}`);
  console.log(`- 平均每会话消息数: ${stats.avgMessagesPerSession.toFixed(2)}`);
}
```

---

## 测试代码

```typescript
import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import fs from 'fs/promises';
import path from 'path';

describe('SessionManager', () => {
  let manager: SessionManager;
  let testDir: string;

  beforeEach(async () => {
    // 创建临时测试目录
    testDir = path.join(process.cwd(), '.test-sessions');
    await fs.mkdir(testDir, { recursive: true });
    manager = new SessionManager(testDir);
  });

  afterEach(async () => {
    // 清理测试目录
    await fs.rm(testDir, { recursive: true, force: true });
  });

  it('should create a new session', async () => {
    const session = await manager.createSession('test-session');
    expect(session.id).toBe('test-session');

    const sessions = await manager.listSessions();
    expect(sessions.length).toBe(1);
    expect(sessions[0].id).toBe('test-session');
  });

  it('should append messages to session', async () => {
    const session = await manager.createSession('test-session');

    await session.appendMessage('user', 'Hello');
    await session.appendMessage('assistant', 'Hi');

    const messages = await session.getMessages();
    expect(messages.length).toBe(2);
    expect(messages[0].content).toBe('Hello');
    expect(messages[1].content).toBe('Hi');
  });

  it('should load existing session', async () => {
    const session = await manager.createSession('test-session');
    await session.appendMessage('user', 'Test message');

    const loadedSession = await manager.loadSession('test-session');
    const messages = await loadedSession.getMessages();

    expect(messages.length).toBe(1);
    expect(messages[0].content).toBe('Test message');
  });

  it('should set and get session name', async () => {
    const session = await manager.createSession('test-session');
    await session.setName('My Project');

    const name = await session.getName();
    expect(name).toBe('My Project');
  });

  it('should delete session', async () => {
    await manager.createSession('test-session');

    let sessions = await manager.listSessions();
    expect(sessions.length).toBe(1);

    await manager.deleteSession('test-session');

    sessions = await manager.listSessions();
    expect(sessions.length).toBe(0);
  });
});
```

---

## 学习检查清单

完成本节学习后，你应该能够：

- [ ] 实现 SessionStorage 类
- [ ] 实现 Session 类
- [ ] 实现 SessionManager 类
- [ ] 创建和恢复会话
- [ ] 添加和获取消息
- [ ] 命名和查询会话
- [ ] 列出和删除会话
- [ ] 扩展功能（搜索、导出、统计）
- [ ] 编写测试代码

---

**版本：** v1.0
**最后更新：** 2026-02-18
**维护者：** Claude Code
