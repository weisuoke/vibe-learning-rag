# 实战代码 04：Session 上下文构建

> 完整的 TypeScript Session 上下文构建实现，从树形结构提取消息序列用于 LLM 调用

---

## 代码概览

本文实现完整的 Session 上下文构建器，包括：

1. **消息提取**：从树形结构提取消息序列
2. **上下文过滤**：只保留相关的消息类型
3. **Compaction 处理**：处理压缩后的消息
4. **Branch Summary 处理**：处理分支摘要
5. **上下文优化**：控制上下文长度和质量

---

## 完整实现

### 类型定义

```typescript
// types.ts
export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface ContextOptions {
  includeSystem?: boolean;
  includeSummaries?: boolean;
  maxMessages?: number;
  maxTokens?: number;
}

export interface ContextStats {
  totalMessages: number;
  userMessages: number;
  assistantMessages: number;
  systemMessages: number;
  estimatedTokens: number;
}
```

### Session 上下文构建器

```typescript
// session-context-builder.ts
import { SessionEntry, Message, ContextOptions, ContextStats } from './types';
import { BranchManager } from './branch-manager';

export class SessionContextBuilder {
  constructor(private manager: BranchManager) {}

  /**
   * 构建 Session 上下文
   */
  buildContext(options: ContextOptions = {}): Message[] {
    const {
      includeSystem = false,
      includeSummaries = true,
      maxMessages,
      maxTokens
    } = options;

    // 获取当前分支
    const branch = this.manager.getCurrentBranch();

    // 提取消息
    let messages = this.extractMessages(branch, includeSystem, includeSummaries);

    // 应用限制
    if (maxMessages) {
      messages = this.limitByCount(messages, maxMessages);
    }

    if (maxTokens) {
      messages = this.limitByTokens(messages, maxTokens);
    }

    return messages;
  }

  /**
   * 从分支提取消息
   */
  private extractMessages(
    branch: SessionEntry[],
    includeSystem: boolean,
    includeSummaries: boolean
  ): Message[] {
    const messages: Message[] = [];

    for (const entry of branch) {
      // 跳过 session 类型
      if (entry.type === 'session') {
        continue;
      }

      // 处理用户和助手消息
      if (entry.type === 'user' || entry.type === 'assistant') {
        messages.push({
          role: entry.type,
          content: entry.content || ''
        });
      }

      // 处理系统消息
      if (entry.type === 'system' && includeSystem) {
        messages.push({
          role: 'system',
          content: entry.content || ''
        });
      }

      // 处理分支摘要
      if (entry.type === 'branch_summary' && includeSummaries) {
        messages.push({
          role: 'system',
          content: `[Branch Summary] ${entry.summary}`
        });
      }

      // 处理 Compaction
      if (entry.type === 'compaction' && includeSummaries) {
        messages.push({
          role: 'system',
          content: `[Compacted] ${entry.summary}`
        });
      }
    }

    return messages;
  }

  /**
   * 按消息数量限制
   */
  private limitByCount(messages: Message[], maxMessages: number): Message[] {
    if (messages.length <= maxMessages) {
      return messages;
    }

    // 保留最近的消息
    return messages.slice(-maxMessages);
  }

  /**
   * 按 Token 数量限制
   */
  private limitByTokens(messages: Message[], maxTokens: number): Message[] {
    let totalTokens = 0;
    const result: Message[] = [];

    // 从后往前遍历，保留最近的消息
    for (let i = messages.length - 1; i >= 0; i--) {
      const message = messages[i];
      const tokens = this.estimateTokens(message.content);

      if (totalTokens + tokens > maxTokens) {
        break;
      }

      result.unshift(message);
      totalTokens += tokens;
    }

    return result;
  }

  /**
   * 估算 Token 数量（简单估算：1 token ≈ 4 字符）
   */
  private estimateTokens(text: string): number {
    return Math.ceil(text.length / 4);
  }

  /**
   * 获取上下文统计信息
   */
  getStats(messages: Message[]): ContextStats {
    const stats: ContextStats = {
      totalMessages: messages.length,
      userMessages: 0,
      assistantMessages: 0,
      systemMessages: 0,
      estimatedTokens: 0
    };

    for (const message of messages) {
      if (message.role === 'user') {
        stats.userMessages++;
      } else if (message.role === 'assistant') {
        stats.assistantMessages++;
      } else if (message.role === 'system') {
        stats.systemMessages++;
      }

      stats.estimatedTokens += this.estimateTokens(message.content);
    }

    return stats;
  }

  /**
   * 格式化为 LLM API 格式
   */
  formatForAPI(messages: Message[]): any[] {
    return messages.map(msg => ({
      role: msg.role,
      content: msg.content
    }));
  }

  /**
   * 格式化为可读文本
   */
  formatAsText(messages: Message[]): string {
    return messages
      .map(msg => `[${msg.role.toUpperCase()}] ${msg.content}`)
      .join('\n\n');
  }
}
```

### 上下文优化器

```typescript
// context-optimizer.ts
import { Message, SessionEntry } from './types';

export class ContextOptimizer {
  /**
   * 压缩重复内容
   */
  deduplicateMessages(messages: Message[]): Message[] {
    const seen = new Set<string>();
    const result: Message[] = [];

    for (const message of messages) {
      const key = `${message.role}:${message.content}`;

      if (!seen.has(key)) {
        seen.add(key);
        result.push(message);
      }
    }

    return result;
  }

  /**
   * 合并连续的系统消息
   */
  mergeSystemMessages(messages: Message[]): Message[] {
    const result: Message[] = [];
    let systemBuffer: string[] = [];

    for (const message of messages) {
      if (message.role === 'system') {
        systemBuffer.push(message.content);
      } else {
        // 输出累积的系统消息
        if (systemBuffer.length > 0) {
          result.push({
            role: 'system',
            content: systemBuffer.join('\n\n')
          });
          systemBuffer = [];
        }

        result.push(message);
      }
    }

    // 处理末尾的系统消息
    if (systemBuffer.length > 0) {
      result.push({
        role: 'system',
        content: systemBuffer.join('\n\n')
      });
    }

    return result;
  }

  /**
   * 截断过长的消息
   */
  truncateMessages(messages: Message[], maxLength: number): Message[] {
    return messages.map(msg => ({
      ...msg,
      content: msg.content.length > maxLength
        ? msg.content.substring(0, maxLength) + '...'
        : msg.content
    }));
  }

  /**
   * 移除空消息
   */
  removeEmptyMessages(messages: Message[]): Message[] {
    return messages.filter(msg => msg.content.trim().length > 0);
  }

  /**
   * 智能压缩上下文
   */
  smartCompress(messages: Message[], targetTokens: number): Message[] {
    // 1. 移除空消息
    let result = this.removeEmptyMessages(messages);

    // 2. 去重
    result = this.deduplicateMessages(result);

    // 3. 合并系统消息
    result = this.mergeSystemMessages(result);

    // 4. 如果还是太长，截断旧消息
    const estimatedTokens = this.estimateTotalTokens(result);
    if (estimatedTokens > targetTokens) {
      const ratio = targetTokens / estimatedTokens;
      const keepCount = Math.floor(result.length * ratio);
      result = result.slice(-keepCount);
    }

    return result;
  }

  private estimateTotalTokens(messages: Message[]): number {
    return messages.reduce((sum, msg) => sum + Math.ceil(msg.content.length / 4), 0);
  }
}
```

### Compaction 处理器

```typescript
// compaction-handler.ts
import { SessionEntry, Message } from './types';

export interface CompactionEntry extends SessionEntry {
  type: 'compaction';
  summary: string;
  compactedRange: {
    startId: string;
    endId: string;
  };
  originalMessageCount: number;
}

export class CompactionHandler {
  /**
   * 检测是否有 Compaction
   */
  hasCompaction(branch: SessionEntry[]): boolean {
    return branch.some(entry => entry.type === 'compaction');
  }

  /**
   * 提取 Compaction 信息
   */
  extractCompactions(branch: SessionEntry[]): CompactionEntry[] {
    return branch.filter(entry => entry.type === 'compaction') as CompactionEntry[];
  }

  /**
   * 应用 Compaction 到消息列表
   */
  applyCompaction(messages: Message[], compactions: CompactionEntry[]): Message[] {
    if (compactions.length === 0) {
      return messages;
    }

    // 简化实现：将 Compaction 作为系统消息插入
    const result: Message[] = [];

    for (const compaction of compactions) {
      result.push({
        role: 'system',
        content: `[Compacted ${compaction.originalMessageCount} messages] ${compaction.summary}`
      });
    }

    // 添加 Compaction 之后的消息
    result.push(...messages);

    return result;
  }

  /**
   * 创建 Compaction 条目
   */
  createCompaction(
    messages: Message[],
    summary: string,
    startId: string,
    endId: string
  ): CompactionEntry {
    return {
      type: 'compaction',
      id: this.generateId(),
      parentId: endId,
      timestamp: new Date().toISOString(),
      summary,
      compactedRange: {
        startId,
        endId
      },
      originalMessageCount: messages.length
    };
  }

  private generateId(): string {
    return Math.random().toString(36).substring(2, 10);
  }
}
```

### Branch Summary 处理器

```typescript
// branch-summary-handler.ts
import { SessionEntry, Message } from './types';

export interface BranchSummaryEntry extends SessionEntry {
  type: 'branch_summary';
  summary: string;
  branchFromId: string;
  readFiles?: string[];
  modifiedFiles?: string[];
}

export class BranchSummaryHandler {
  /**
   * 检测是否有 Branch Summary
   */
  hasBranchSummary(branch: SessionEntry[]): boolean {
    return branch.some(entry => entry.type === 'branch_summary');
  }

  /**
   * 提取 Branch Summary
   */
  extractBranchSummaries(branch: SessionEntry[]): BranchSummaryEntry[] {
    return branch.filter(entry => entry.type === 'branch_summary') as BranchSummaryEntry[];
  }

  /**
   * 应用 Branch Summary 到消息列表
   */
  applyBranchSummary(messages: Message[], summaries: BranchSummaryEntry[]): Message[] {
    if (summaries.length === 0) {
      return messages;
    }

    const result: Message[] = [];

    for (const summary of summaries) {
      let content = `[Branch Summary] ${summary.summary}`;

      if (summary.readFiles && summary.readFiles.length > 0) {
        content += `\n\nRead files: ${summary.readFiles.join(', ')}`;
      }

      if (summary.modifiedFiles && summary.modifiedFiles.length > 0) {
        content += `\n\nModified files: ${summary.modifiedFiles.join(', ')}`;
      }

      result.push({
        role: 'system',
        content
      });
    }

    result.push(...messages);

    return result;
  }
}
```

---

## 使用示例

### 示例 1：基础上下文构建

```typescript
import { BranchManager } from './branch-manager';
import { SessionContextBuilder } from './session-context-builder';

function basicContextExample() {
  const manager = new BranchManager('session.jsonl');
  const builder = new SessionContextBuilder(manager);

  // 构建上下文
  const messages = builder.buildContext();

  console.log('=== Session Context ===');
  console.log(builder.formatAsText(messages));

  // 获取统计信息
  const stats = builder.getStats(messages);
  console.log('\n=== Context Stats ===');
  console.log(`Total messages: ${stats.totalMessages}`);
  console.log(`User messages: ${stats.userMessages}`);
  console.log(`Assistant messages: ${stats.assistantMessages}`);
  console.log(`Estimated tokens: ${stats.estimatedTokens}`);

  manager.close();
}

basicContextExample();
```

### 示例 2：限制上下文长度

```typescript
function limitedContextExample() {
  const manager = new BranchManager('session.jsonl');
  const builder = new SessionContextBuilder(manager);

  // 限制消息数量
  const messages1 = builder.buildContext({
    maxMessages: 10
  });
  console.log(`Limited to 10 messages: ${messages1.length} messages`);

  // 限制 Token 数量
  const messages2 = builder.buildContext({
    maxTokens: 1000
  });
  const stats = builder.getStats(messages2);
  console.log(`Limited to 1000 tokens: ${stats.estimatedTokens} tokens`);

  manager.close();
}

limitedContextExample();
```

### 示例 3：优化上下文

```typescript
import { ContextOptimizer } from './context-optimizer';

function optimizeContextExample() {
  const manager = new BranchManager('session.jsonl');
  const builder = new SessionContextBuilder(manager);
  const optimizer = new ContextOptimizer();

  // 构建原始上下文
  let messages = builder.buildContext();
  console.log(`Original: ${messages.length} messages`);

  // 去重
  messages = optimizer.deduplicateMessages(messages);
  console.log(`After dedup: ${messages.length} messages`);

  // 合并系统消息
  messages = optimizer.mergeSystemMessages(messages);
  console.log(`After merge: ${messages.length} messages`);

  // 智能压缩
  messages = optimizer.smartCompress(messages, 2000);
  const stats = builder.getStats(messages);
  console.log(`After compress: ${stats.estimatedTokens} tokens`);

  manager.close();
}

optimizeContextExample();
```

### 示例 4：处理 Branch Summary

```typescript
import { BranchSummaryHandler } from './branch-summary-handler';

function branchSummaryExample() {
  const manager = new BranchManager('session.jsonl');
  const builder = new SessionContextBuilder(manager);
  const summaryHandler = new BranchSummaryHandler();

  // 获取当前分支
  const branch = manager.getCurrentBranch();

  // 检查是否有 Branch Summary
  if (summaryHandler.hasBranchSummary(branch)) {
    const summaries = summaryHandler.extractBranchSummaries(branch);
    console.log(`Found ${summaries.length} branch summaries`);

    for (const summary of summaries) {
      console.log(`\nSummary: ${summary.summary}`);
      if (summary.readFiles) {
        console.log(`Read files: ${summary.readFiles.join(', ')}`);
      }
      if (summary.modifiedFiles) {
        console.log(`Modified files: ${summary.modifiedFiles.join(', ')}`);
      }
    }
  }

  manager.close();
}

branchSummaryExample();
```

### 示例 5：格式化为 LLM API 调用

```typescript
async function llmApiExample() {
  const manager = new BranchManager('session.jsonl');
  const builder = new SessionContextBuilder(manager);

  // 构建上下文
  const messages = builder.buildContext({
    maxTokens: 4000,
    includeSummaries: true
  });

  // 格式化为 API 格式
  const apiMessages = builder.formatForAPI(messages);

  // 调用 LLM API（示例）
  const response = await callLLMAPI({
    model: 'claude-3-opus-20240229',
    messages: apiMessages,
    max_tokens: 1000
  });

  console.log('LLM Response:', response);

  manager.close();
}

// 模拟 LLM API 调用
async function callLLMAPI(params: any): Promise<string> {
  // 实际实现中，这里会调用真实的 LLM API
  return 'This is a simulated response';
}

llmApiExample();
```

---

## 高级功能

### 功能 1：上下文缓存

```typescript
class CachedContextBuilder extends SessionContextBuilder {
  private cache = new Map<string, Message[]>();

  buildContext(options: ContextOptions = {}): Message[] {
    const cacheKey = JSON.stringify(options);

    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }

    const messages = super.buildContext(options);
    this.cache.set(cacheKey, messages);

    return messages;
  }

  clearCache(): void {
    this.cache.clear();
  }
}
```

### 功能 2：上下文模板

```typescript
interface ContextTemplate {
  systemPrompt?: string;
  includeFiles?: boolean;
  includeHistory?: boolean;
  maxHistoryMessages?: number;
}

class TemplatedContextBuilder extends SessionContextBuilder {
  buildFromTemplate(template: ContextTemplate): Message[] {
    const messages: Message[] = [];

    // 添加系统提示
    if (template.systemPrompt) {
      messages.push({
        role: 'system',
        content: template.systemPrompt
      });
    }

    // 添加历史消息
    if (template.includeHistory) {
      const history = this.buildContext({
        maxMessages: template.maxHistoryMessages
      });
      messages.push(...history);
    }

    return messages;
  }
}
```

### 功能 3：上下文分析

```typescript
class ContextAnalyzer {
  analyzeContext(messages: Message[]): {
    conversationTurns: number;
    avgMessageLength: number;
    longestMessage: number;
    shortestMessage: number;
  } {
    const lengths = messages.map(m => m.content.length);

    return {
      conversationTurns: Math.floor(messages.length / 2),
      avgMessageLength: lengths.reduce((a, b) => a + b, 0) / lengths.length,
      longestMessage: Math.max(...lengths),
      shortestMessage: Math.min(...lengths)
    };
  }

  findKeywords(messages: Message[], topN: number = 10): string[] {
    const words = new Map<string, number>();

    for (const message of messages) {
      const tokens = message.content.toLowerCase().split(/\W+/);

      for (const token of tokens) {
        if (token.length > 3) {
          words.set(token, (words.get(token) || 0) + 1);
        }
      }
    }

    return Array.from(words.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, topN)
      .map(([word]) => word);
  }
}
```

---

## 与 Pi-mono 的对比

### Pi-mono 的实现

```typescript
// Pi-mono session-manager.ts
class SessionManager {
  buildSessionContext(): Message[] {
    const branch = this.getBranch(this.leafId);
    const messages: Message[] = [];

    for (const entry of branch) {
      if (entry.type === 'user' || entry.type === 'assistant') {
        messages.push({
          role: entry.type,
          content: entry.content
        });
      }
    }

    return messages;
  }
}
```

### 我们的实现

```typescript
// 更完整、更灵活的实现
class SessionContextBuilder {
  // 支持多种选项
  buildContext(options: ContextOptions): Message[] { ... }

  // 支持上下文优化
  smartCompress(messages, targetTokens): Message[] { ... }

  // 支持 Compaction 和 Branch Summary
  applyCompaction(messages, compactions): Message[] { ... }
  applyBranchSummary(messages, summaries): Message[] { ... }

  // 支持多种格式化
  formatForAPI(messages): any[] { ... }
  formatAsText(messages): string { ... }
}
```

---

## 关键要点总结

1. **消息提取**：从树形结构提取相关消息
2. **上下文限制**：按数量或 Token 限制上下文长度
3. **上下文优化**：去重、合并、压缩
4. **Compaction 处理**：处理压缩后的历史消息
5. **Branch Summary 处理**：包含分支摘要信息

---

**下一步**：实现分支摘要生成 → `07_实战代码_05_分支摘要生成.md`
