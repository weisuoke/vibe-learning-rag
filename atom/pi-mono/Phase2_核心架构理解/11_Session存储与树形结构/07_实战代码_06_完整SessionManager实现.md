# 实战代码 06：完整 SessionManager 实现

> 完整的 TypeScript SessionManager 实现，整合所有功能模块

---

## 代码概览

本文实现完整的 SessionManager 类，整合：

1. **JSONL 读写**：基础文件操作
2. **树形结构**：构建和遍历
3. **分支管理**：创建和切换分支
4. **上下文构建**：提取消息序列
5. **分支摘要**：自动生成摘要

---

## 完整实现

```typescript
// session-manager.ts
import * as fs from 'fs';
import { randomBytes } from 'crypto';
import { JsonlManager } from './jsonl-manager';
import { TreeBuilder } from './tree-builder';
import { SessionContextBuilder } from './session-context-builder';
import { BranchSummarizer } from './branch-summarizer';
import {
  SessionEntry,
  TreeNode,
  Message,
  ContextOptions,
  BranchInfo,
  LLMClient
} from './types';

export class SessionManager {
  private jsonlManager: JsonlManager;
  private treeBuilder: TreeBuilder;
  private contextBuilder: SessionContextBuilder;
  private summarizer: BranchSummarizer | null = null;
  private leafId: string;
  private usedIds = new Set<string>();

  constructor(
    private sessionFile: string,
    llmClient?: LLMClient
  ) {
    this.jsonlManager = new JsonlManager(sessionFile);

    // 加载或创建会话
    const entries = this.jsonlManager.readAll();

    if (entries.length === 0) {
      // 创建新会话
      const root = this.createRootEntry();
      this.jsonlManager.append(root);
      this.jsonlManager.flush();

      this.leafId = root.id;
      this.usedIds.add(root.id);
      this.treeBuilder = new TreeBuilder([root]);
    } else {
      // 加载现有会话
      this.leafId = entries[entries.length - 1].id;
      entries.forEach(e => this.usedIds.add(e.id));
      this.treeBuilder = new TreeBuilder(entries);
    }

    // 初始化上下文构建器
    this.contextBuilder = new SessionContextBuilder(this);

    // 初始化摘要生成器（如果提供了 LLM 客户端）
    if (llmClient) {
      this.summarizer = new BranchSummarizer(this, llmClient);
    }
  }

  // ==================== 基础操作 ====================

  /**
   * 生成唯一 ID
   */
  private generateId(): string {
    let id: string;
    let attempts = 0;

    do {
      id = randomBytes(4).toString('hex');
      attempts++;

      if (attempts >= 10) {
        throw new Error('Failed to generate unique ID');
      }
    } while (this.usedIds.has(id));

    this.usedIds.add(id);
    return id;
  }

  /**
   * 创建根条目
   */
  private createRootEntry(): SessionEntry {
    return {
      type: 'session',
      id: this.generateId(),
      parentId: null,
      timestamp: new Date().toISOString(),
      cwd: process.cwd()
    };
  }

  /**
   * 添加条目
   */
  private appendEntry(entry: SessionEntry): void {
    this.jsonlManager.append(entry);
    this.jsonlManager.flush();

    // 重新加载树
    const entries = this.jsonlManager.readAll();
    this.treeBuilder = new TreeBuilder(entries);
  }

  // ==================== 消息操作 ====================

  /**
   * 添加用户消息
   */
  addUserMessage(content: string): string {
    return this.addMessage(content, 'user');
  }

  /**
   * 添加助手消息
   */
  addAssistantMessage(content: string): string {
    return this.addMessage(content, 'assistant');
  }

  /**
   * 添加消息
   */
  addMessage(content: string, type: 'user' | 'assistant'): string {
    const id = this.generateId();
    const entry: SessionEntry = {
      type,
      id,
      parentId: this.leafId,
      timestamp: new Date().toISOString(),
      content
    };

    this.appendEntry(entry);
    this.leafId = id;

    return id;
  }

  /**
   * 添加工具调用
   */
  addToolUse(name: string, input: any): string {
    const id = this.generateId();
    const entry: SessionEntry = {
      type: 'tool_use',
      id,
      parentId: this.leafId,
      timestamp: new Date().toISOString(),
      name,
      input
    };

    this.appendEntry(entry);
    this.leafId = id;

    return id;
  }

  /**
   * 添加工具结果
   */
  addToolResult(toolUseId: string, content: string): string {
    const id = this.generateId();
    const entry: SessionEntry = {
      type: 'tool_result',
      id,
      parentId: this.leafId,
      timestamp: new Date().toISOString(),
      tool_use_id: toolUseId,
      content
    };

    this.appendEntry(entry);
    this.leafId = id;

    return id;
  }

  // ==================== 分支操作 ====================

  /**
   * 创建分支
   */
  branch(branchFromId: string): void {
    const entry = this.treeBuilder.getEntry(branchFromId);
    if (!entry) {
      throw new Error(`Entry not found: ${branchFromId}`);
    }

    this.leafId = branchFromId;
  }

  /**
   * 创建分支并生成摘要
   */
  async branchWithSummary(branchFromId: string): Promise<string> {
    if (!this.summarizer) {
      throw new Error('LLM client not provided');
    }

    // 生成摘要
    const result = await this.summarizer.generateSummary(branchFromId);

    // 创建摘要条目
    const summaryId = this.generateId();
    const summaryEntry: SessionEntry = {
      type: 'branch_summary',
      id: summaryId,
      parentId: branchFromId,
      timestamp: new Date().toISOString(),
      summary: result.summary,
      branchFromId,
      readFiles: result.readFiles,
      modifiedFiles: result.modifiedFiles
    };

    this.appendEntry(summaryEntry);

    // 移动叶子指针
    this.branch(branchFromId);

    return summaryId;
  }

  /**
   * 获取所有分支
   */
  getAllBranches(): BranchInfo[] {
    const leaves = this.treeBuilder.getLeaves();
    const branches: BranchInfo[] = [];

    for (const leaf of leaves) {
      const branch = this.treeBuilder.getBranch(leaf.id);
      const messages = branch.filter(e => e.type === 'user' || e.type === 'assistant');
      const lastMessage = messages[messages.length - 1];

      branches.push({
        id: leaf.id,
        leafId: leaf.id,
        depth: branch.length,
        messageCount: messages.length,
        lastMessage: lastMessage?.content,
        createdAt: leaf.timestamp
      });
    }

    return branches;
  }

  /**
   * 切换到指定分支
   */
  switchToBranch(branchId: string): void {
    const branches = this.getAllBranches();
    const branch = branches.find(b => b.id === branchId);

    if (!branch) {
      throw new Error(`Branch not found: ${branchId}`);
    }

    this.leafId = branch.leafId;
  }

  // ==================== 树操作 ====================

  /**
   * 获取当前分支（从叶子到根）
   */
  getBranch(fromId?: string): SessionEntry[] {
    return this.treeBuilder.getBranch(fromId || this.leafId);
  }

  /**
   * 获取完整树
   */
  getTree(): TreeNode | null {
    return this.treeBuilder.buildFullTree();
  }

  /**
   * 获取条目
   */
  getEntry(id: string): SessionEntry | undefined {
    return this.treeBuilder.getEntry(id);
  }

  /**
   * 获取子节点
   */
  getChildren(id: string): SessionEntry[] {
    return this.treeBuilder.getChildren(id);
  }

  /**
   * 获取父节点
   */
  getParent(id: string): SessionEntry | null {
    return this.treeBuilder.getParent(id);
  }

  /**
   * 获取叶子节点
   */
  getLeaves(): SessionEntry[] {
    return this.treeBuilder.getLeaves();
  }

  // ==================== 上下文操作 ====================

  /**
   * 构建会话上下文
   */
  buildContext(options?: ContextOptions): Message[] {
    return this.contextBuilder.buildContext(options);
  }

  /**
   * 获取上下文统计
   */
  getContextStats(messages: Message[]) {
    return this.contextBuilder.getStats(messages);
  }

  /**
   * 格式化为 API 格式
   */
  formatForAPI(messages: Message[]): any[] {
    return this.contextBuilder.formatForAPI(messages);
  }

  // ==================== 查询操作 ====================

  /**
   * 获取当前叶子 ID
   */
  getLeafId(): string {
    return this.leafId;
  }

  /**
   * 获取会话文件路径
   */
  getSessionFile(): string {
    return this.sessionFile;
  }

  /**
   * 获取所有条目
   */
  getAllEntries(): SessionEntry[] {
    return this.jsonlManager.readAll();
  }

  /**
   * 获取树统计信息
   */
  getTreeStats() {
    return this.treeBuilder.getStats();
  }

  // ==================== 可视化 ====================

  /**
   * 可视化当前分支
   */
  visualizeCurrentBranch(): void {
    const branch = this.getBranch();

    console.log('\n=== Current Branch ===');
    for (let i = 0; i < branch.length; i++) {
      const prefix = '  '.repeat(i);
      const entry = branch[i];
      const marker = entry.id === this.leafId ? ' ← current' : '';

      this.printEntry(entry, prefix, marker);
    }
    console.log('');
  }

  /**
   * 可视化完整树
   */
  visualizeFullTree(): void {
    const tree = this.getTree();
    if (!tree) {
      console.log('No tree to visualize');
      return;
    }

    console.log('\n=== Full Tree ===');
    this.renderNode(tree, '', true);
    console.log('');
  }

  private renderNode(node: TreeNode, prefix: string, isLast: boolean): void {
    const connector = isLast ? '└── ' : '├── ';
    const marker = node.entry.id === this.leafId ? ' ← current' : '';

    console.log(prefix + connector + this.getNodeLabel(node.entry) + marker);

    const childPrefix = prefix + (isLast ? '    ' : '│   ');
    const children = node.children;

    for (let i = 0; i < children.length; i++) {
      const isLastChild = i === children.length - 1;
      this.renderNode(children[i], childPrefix, isLastChild);
    }
  }

  private printEntry(entry: SessionEntry, prefix: string, marker: string): void {
    if (entry.type === 'session') {
      console.log(`${prefix}└─ [Session]${marker}`);
    } else {
      const content = entry.content?.substring(0, 40) || '';
      const suffix = (entry.content?.length || 0) > 40 ? '...' : '';
      console.log(`${prefix}└─ [${entry.type}] ${content}${suffix}${marker}`);
    }
  }

  private getNodeLabel(entry: SessionEntry): string {
    if (entry.type === 'session') {
      return '[Session]';
    } else if (entry.type === 'branch_summary') {
      return `[Summary] ${entry.summary?.substring(0, 30)}...`;
    } else {
      const content = entry.content?.substring(0, 30) || '';
      const suffix = (entry.content?.length || 0) > 30 ? '...' : '';
      return `[${entry.type}] ${content}${suffix}`;
    }
  }

  // ==================== 工具方法 ====================

  /**
   * 关闭管理器
   */
  close(): void {
    this.jsonlManager.close();
  }

  /**
   * 刷新到磁盘
   */
  flush(): void {
    this.jsonlManager.flush();
  }

  /**
   * 检查文件是否存在
   */
  exists(): boolean {
    return this.jsonlManager.exists();
  }

  /**
   * 获取文件大小
   */
  getFileSize(): number {
    return this.jsonlManager.getSize();
  }
}
```

---

## 使用示例

### 示例 1：基础会话管理

```typescript
import { SessionManager } from './session-manager';

function basicExample() {
  const manager = new SessionManager('my-session.jsonl');

  // 添加对话
  manager.addUserMessage('Hello');
  manager.addAssistantMessage('Hi there!');
  manager.addUserMessage('How are you?');
  manager.addAssistantMessage('I\'m doing well, thanks!');

  // 可视化当前分支
  manager.visualizeCurrentBranch();

  // 获取统计信息
  const stats = manager.getTreeStats();
  console.log('Tree stats:', stats);

  manager.close();
}

basicExample();
```

### 示例 2：分支管理

```typescript
async function branchExample() {
  const manager = new SessionManager('session.jsonl');

  // 初始对话
  manager.addUserMessage('帮我实现排序');
  manager.addAssistantMessage('好的，用快速排序');
  const branchPoint = manager.getLeafId();

  manager.addUserMessage('继续');
  manager.addAssistantMessage('这是快速排序实现...');

  // 创建分支
  manager.branch(branchPoint);
  manager.addUserMessage('用归并排序');
  manager.addAssistantMessage('这是归并排序实现...');

  // 可视化树
  manager.visualizeFullTree();

  // 获取所有分支
  const branches = manager.getAllBranches();
  console.log(`Total branches: ${branches.length}`);

  manager.close();
}

branchExample();
```

### 示例 3：上下文构建

```typescript
function contextExample() {
  const manager = new SessionManager('session.jsonl');

  // 添加对话
  manager.addUserMessage('Hello');
  manager.addAssistantMessage('Hi!');
  manager.addUserMessage('How are you?');
  manager.addAssistantMessage('Good!');

  // 构建上下文
  const messages = manager.buildContext({
    maxMessages: 10,
    includeSummaries: true
  });

  console.log('Context messages:', messages.length);

  // 格式化为 API 调用
  const apiMessages = manager.formatForAPI(messages);
  console.log('API format:', JSON.stringify(apiMessages, null, 2));

  manager.close();
}

contextExample();
```

### 示例 4：完整的 AI 对话循环

```typescript
import { AnthropicClient } from './llm-client';

async function aiConversationExample() {
  const llmClient = new AnthropicClient(process.env.ANTHROPIC_API_KEY!);
  const manager = new SessionManager('ai-session.jsonl', llmClient);

  // 用户输入
  const userInput = '请解释什么是快速排序';
  manager.addUserMessage(userInput);

  // 构建上下文
  const context = manager.buildContext({ maxTokens: 4000 });

  // 调用 LLM
  const response = await llmClient.complete({
    messages: manager.formatForAPI(context),
    max_tokens: 1000
  });

  // 保存 AI 响应
  manager.addAssistantMessage(response.content);

  console.log('AI:', response.content);

  manager.close();
}

aiConversationExample();
```

### 示例 5：工具调用跟踪

```typescript
function toolTrackingExample() {
  const manager = new SessionManager('tool-session.jsonl');

  // 用户请求
  manager.addUserMessage('读取 config.json 文件');

  // 工具调用
  const toolUseId = manager.addToolUse('read_file', {
    file_path: 'config.json'
  });

  // 工具结果
  manager.addToolResult(toolUseId, '{"port": 3000, "host": "localhost"}');

  // AI 响应
  manager.addAssistantMessage('配置文件内容如上所示');

  // 可视化
  manager.visualizeCurrentBranch();

  manager.close();
}

toolTrackingExample();
```

---

## 高级用法

### 扩展 1：自动保存

```typescript
class AutoSaveSessionManager extends SessionManager {
  private saveTimer: NodeJS.Timeout | null = null;

  constructor(sessionFile: string, llmClient?: LLMClient) {
    super(sessionFile, llmClient);
    this.startAutoSave();
  }

  private startAutoSave(): void {
    this.saveTimer = setInterval(() => {
      this.flush();
      console.log('Auto-saved session');
    }, 30000); // 每 30 秒保存一次
  }

  close(): void {
    if (this.saveTimer) {
      clearInterval(this.saveTimer);
    }
    super.close();
  }
}
```

### 扩展 2：事件监听

```typescript
type SessionEvent = 'message_added' | 'branch_created' | 'context_built';

class EventEmittingSessionManager extends SessionManager {
  private listeners = new Map<SessionEvent, Array<(data: any) => void>>();

  on(event: SessionEvent, callback: (data: any) => void): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event)!.push(callback);
  }

  private emit(event: SessionEvent, data: any): void {
    const callbacks = this.listeners.get(event) || [];
    for (const callback of callbacks) {
      callback(data);
    }
  }

  addMessage(content: string, type: 'user' | 'assistant'): string {
    const id = super.addMessage(content, type);
    this.emit('message_added', { id, content, type });
    return id;
  }

  branch(branchFromId: string): void {
    super.branch(branchFromId);
    this.emit('branch_created', { branchFromId });
  }
}
```

### 扩展 3：会话快照

```typescript
class SnapshotSessionManager extends SessionManager {
  createSnapshot(): string {
    const snapshotFile = `${this.getSessionFile()}.snapshot.${Date.now()}`;
    const entries = this.getAllEntries();

    fs.writeFileSync(
      snapshotFile,
      entries.map(e => JSON.stringify(e)).join('\n'),
      'utf-8'
    );

    return snapshotFile;
  }

  restoreSnapshot(snapshotFile: string): void {
    const content = fs.readFileSync(snapshotFile, 'utf-8');
    fs.writeFileSync(this.getSessionFile(), content, 'utf-8');

    // 重新加载
    const entries = this.getAllEntries();
    this.treeBuilder = new TreeBuilder(entries);
    this.leafId = entries[entries.length - 1].id;
  }
}
```

---

## 与 Pi-mono 的对比

### Pi-mono 的实现

```typescript
// Pi-mono session-manager.ts (简化版)
class SessionManager {
  private entries: SessionEntry[] = [];
  private leafId: string;

  addMessage(content: string, type: string): string {
    const id = this.generateId();
    const entry = { type, id, parentId: this.leafId, content, timestamp: new Date().toISOString() };
    this.entries.push(entry);
    this.appendToFile(entry);
    this.leafId = id;
    return id;
  }

  branch(branchFromId: string): void {
    this.leafId = branchFromId;
  }

  buildContext(): Message[] {
    const branch = this.getBranch(this.leafId);
    return branch.filter(e => e.type === 'user' || e.type === 'assistant')
      .map(e => ({ role: e.type, content: e.content }));
  }
}
```

### 我们的实现

```typescript
// 更完整、更模块化的实现
class SessionManager {
  // 模块化设计
  private jsonlManager: JsonlManager;
  private treeBuilder: TreeBuilder;
  private contextBuilder: SessionContextBuilder;
  private summarizer: BranchSummarizer;

  // 完整的 API
  addUserMessage(content: string): string { ... }
  addAssistantMessage(content: string): string { ... }
  addToolUse(name: string, input: any): string { ... }
  addToolResult(toolUseId: string, content: string): string { ... }

  // 高级功能
  async branchWithSummary(branchFromId: string): Promise<string> { ... }
  getAllBranches(): BranchInfo[] { ... }
  switchToBranch(branchId: string): void { ... }

  // 可视化
  visualizeCurrentBranch(): void { ... }
  visualizeFullTree(): void { ... }
}
```

---

## 关键要点总结

1. **模块化设计**：将功能分解为独立模块
2. **完整的 API**：提供所有必需的操作方法
3. **易于扩展**：支持继承和事件监听
4. **生产就绪**：包含错误处理和资源管理
5. **可视化支持**：提供友好的调试界面

---

**下一步**：实现 Web 界面集成 → `07_实战代码_07_Web界面集成示例.md`
