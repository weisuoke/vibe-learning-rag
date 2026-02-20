# Session 管理与分支 - 核心概念 05：Compaction 压缩机制

> 理解 Compaction 如何优化大型 Session，同时保留完整历史

---

## 概述

Compaction（压缩）是 Pi Session 管理的关键优化机制：
- 优化内存中的上下文大小
- 保留 JSONL 文件中的完整历史
- 提升 LLM 调用性能
- 降低 Token 消耗成本

---

## 1. 为什么需要 Compaction？

### 1.1 问题：上下文窗口限制

```typescript
// LLM 的上下文窗口限制
const contextLimits = {
  'claude-opus-4': 200000,      // 200K tokens
  'claude-sonnet-4': 200000,    // 200K tokens
  'gpt-4-turbo': 128000,        // 128K tokens
  'gpt-4': 8192                 // 8K tokens
};

// 问题：长对话会超出限制
const longSession = {
  entries: 500,                  // 500 条消息
  averageTokensPerEntry: 500,    // 每条 500 tokens
  totalTokens: 250000            // 总共 250K tokens
};

// 超出了大多数模型的限制！
```

### 1.2 问题：性能和成本

```typescript
// 每次 LLM 调用都需要传入完整上下文
async function callLLM(context: Message[]) {
  const response = await llm.generate({
    messages: context,  // 完整上下文
    model: 'claude-opus-4'
  });

  // 问题：
  // 1. 上下文越大，响应越慢
  // 2. Token 消耗越多，成本越高
  // 3. 很多旧消息可能不再相关
}
```

---

## 2. Compaction 工作原理

### 2.1 核心思想

**Compaction = 总结旧消息 + 保留新消息 + 完整历史不变**

```typescript
// Compaction 前
const beforeCompaction = {
  context: [
    { id: '1', content: '消息1' },
    { id: '2', content: '消息2' },
    // ... 100 条消息 ...
    { id: '100', content: '消息100' }
  ],
  totalTokens: 50000
};

// Compaction 后
const afterCompaction = {
  // 内存中的上下文（用于 LLM）
  context: [
    { type: 'summary', content: '前90条消息的摘要' },  // 总结
    { id: '91', content: '消息91' },                  // 保留
    // ... 最近10条消息 ...
    { id: '100', content: '消息100' }
  ],
  totalTokens: 10000,  // 减少了 80%

  // JSONL 文件（完整历史）
  jsonlFile: [
    { id: '1', content: '消息1' },
    // ... 所有100条消息 ...
    { id: '100', content: '消息100' },
    { type: 'compaction', summary: '...' }  // 压缩记录
  ]
};
```

### 2.2 TypeScript 实现

```typescript
interface CompactionOptions {
  keepRecentMessages: number;    // 保留最近 N 条消息
  strategy: 'summarize' | 'truncate';  // 压缩策略
  summaryModel?: string;         // 用于生成摘要的模型
}

class SessionCompaction {
  async compact(
    sessionId: string,
    options: CompactionOptions
  ): Promise<CompactionResult> {
    // 1. 读取完整历史
    const allEntries = await this.storage.readAll();
    console.log(`完整历史: ${allEntries.length} 条`);

    // 2. 分离新旧消息
    const recentEntries = allEntries.slice(-options.keepRecentMessages);
    const oldEntries = allEntries.slice(0, -options.keepRecentMessages);

    // 3. 生成摘要
    let summary: string;
    if (options.strategy === 'summarize') {
      summary = await this.generateSummary(oldEntries, options.summaryModel);
    } else {
      summary = `[Truncated ${oldEntries.length} messages]`;
    }

    // 4. 创建压缩条目
    const compactionEntry = {
      id: this.generateId(),
      type: 'compaction',
      timestamp: Date.now(),
      summary,
      compressedCount: oldEntries.length,
      strategy: options.strategy
    };

    // 5. 追加到 JSONL（不删除旧条目）
    await this.storage.append(compactionEntry);

    // 6. 返回优化后的上下文
    return {
      context: [
        { type: 'summary', content: summary },
        ...recentEntries
      ],
      fullHistory: allEntries,
      compactionEntry
    };
  }

  private async generateSummary(
    entries: SessionEntry[],
    model?: string
  ): Promise<string> {
    // 使用 LLM 生成摘要
    const messages = entries
      .filter(e => e.type === 'message')
      .map(e => `${e.role}: ${e.content}`)
      .join('\n\n');

    const response = await this.llm.generate({
      messages: [
        {
          role: 'user',
          content: `请总结以下对话的关键内容：\n\n${messages}`
        }
      ],
      model: model || 'claude-haiku-4'  // 使用快速模型
    });

    return response.content;
  }
}
```

---

## 3. 压缩策略

### 3.1 策略 1：Summarize（总结）

**优点：**
- ✅ 保留关键信息
- ✅ 上下文连贯
- ✅ LLM 仍能理解历史

**缺点：**
- ❌ 需要调用 LLM 生成摘要
- ❌ 有一定成本

**实现：**

```typescript
class SummarizeStrategy {
  async compress(entries: SessionEntry[]): Promise<string> {
    // 1. 提取关键信息
    const keyPoints = this.extractKeyPoints(entries);

    // 2. 生成结构化摘要
    const summary = `
## 对话摘要

**讨论主题：** ${keyPoints.topics.join(', ')}

**关键决策：**
${keyPoints.decisions.map(d => `- ${d}`).join('\n')}

**实现内容：**
${keyPoints.implementations.map(i => `- ${i}`).join('\n')}

**待解决问题：**
${keyPoints.issues.map(i => `- ${i}`).join('\n')}
    `.trim();

    return summary;
  }

  private extractKeyPoints(entries: SessionEntry[]) {
    // 简单的关键词提取
    const topics = new Set<string>();
    const decisions: string[] = [];
    const implementations: string[] = [];
    const issues: string[] = [];

    for (const entry of entries) {
      if (entry.type !== 'message') continue;

      const content = entry.content.toLowerCase();

      // 提取主题
      if (content.includes('实现') || content.includes('开发')) {
        topics.add('功能开发');
      }
      if (content.includes('bug') || content.includes('错误')) {
        topics.add('Bug修复');
      }

      // 提取决策
      if (content.includes('决定') || content.includes('选择')) {
        decisions.push(entry.content.substring(0, 100));
      }

      // 提取实现
      if (entry.role === 'assistant' && content.includes('完成')) {
        implementations.push(entry.content.substring(0, 100));
      }

      // 提取问题
      if (content.includes('问题') || content.includes('待解决')) {
        issues.push(entry.content.substring(0, 100));
      }
    }

    return {
      topics: Array.from(topics),
      decisions,
      implementations,
      issues
    };
  }
}
```

### 3.2 策略 2：Truncate（截断）

**优点：**
- ✅ 简单快速
- ✅ 无需额外成本

**缺点：**
- ❌ 丢失历史上下文
- ❌ LLM 无法理解之前的讨论

**实现：**

```typescript
class TruncateStrategy {
  async compress(entries: SessionEntry[]): Promise<string> {
    return `[Truncated ${entries.length} messages from history]`;
  }
}
```

### 3.3 策略对比

| 策略 | 信息保留 | 成本 | 速度 | 适用场景 |
|------|---------|------|------|---------|
| **Summarize** | 高 | 中 | 慢 | 长期项目，需要上下文 |
| **Truncate** | 低 | 无 | 快 | 临时会话，不需要历史 |

---

## 4. 自动压缩触发

### 4.1 触发条件

```typescript
interface AutoCompactionConfig {
  maxEntries?: number;        // 最大条目数
  maxTokens?: number;         // 最大 Token 数
  maxSize?: number;           // 最大文件大小（字节）
  checkInterval?: number;     // 检查间隔（毫秒）
}

class AutoCompaction {
  private config: AutoCompactionConfig;

  constructor(config: AutoCompactionConfig) {
    this.config = {
      maxEntries: 100,
      maxTokens: 50000,
      maxSize: 1024 * 1024,  // 1MB
      checkInterval: 60000,   // 1分钟
      ...config
    };
  }

  async shouldCompact(sessionId: string): Promise<boolean> {
    const info = await this.sessionManager.getSessionInfo(sessionId);

    // 检查条目数
    if (this.config.maxEntries && info.entryCount > this.config.maxEntries) {
      return true;
    }

    // 检查文件大小
    if (this.config.maxSize && info.size > this.config.maxSize) {
      return true;
    }

    // 检查 Token 数
    if (this.config.maxTokens) {
      const tokens = await this.estimateTokens(sessionId);
      if (tokens > this.config.maxTokens) {
        return true;
      }
    }

    return false;
  }

  async autoCompact(sessionId: string): Promise<void> {
    if (await this.shouldCompact(sessionId)) {
      console.log(`Auto-compacting session ${sessionId}`);

      await this.compaction.compact(sessionId, {
        keepRecentMessages: 20,
        strategy: 'summarize'
      });
    }
  }

  // 启动自动压缩
  startAutoCompaction(sessionId: string): NodeJS.Timeout {
    return setInterval(async () => {
      await this.autoCompact(sessionId);
    }, this.config.checkInterval);
  }
}
```

### 4.2 手动压缩

```bash
# 在 pi 交互界面中
/compact

# 或指定参数
/compact --keep 20 --strategy summarize
```

---

## 5. 访问完整历史

### 5.1 通过 /tree 访问

```bash
# 打开 tree 视图
/tree

# 可以看到所有历史节点，包括被压缩的
├─ 1: User: 消息1
├─ 2: Assistant: 消息2
# ... 所有消息 ...
├─ 90: User: 消息90
├─ [Compaction] 前90条消息已压缩
├─ 91: User: 消息91
# ... 最近消息 ...
└─ 100: User: 消息100
```

### 5.2 编程访问

```typescript
class HistoryAccess {
  // 获取完整历史（包括被压缩的）
  async getFullHistory(sessionId: string): Promise<SessionEntry[]> {
    // 直接读取 JSONL 文件
    const allEntries = await this.storage.readAll();

    // 过滤掉 compaction 条目
    return allEntries.filter(e => e.type !== 'compaction');
  }

  // 获取压缩前的特定节点
  async getCompressedNode(sessionId: string, nodeId: string): Promise<SessionEntry | null> {
    const allEntries = await this.getFullHistory(sessionId);
    return allEntries.find(e => e.id === nodeId) || null;
  }

  // 重建到特定节点的完整上下文
  async rebuildFullContext(sessionId: string, nodeId: string): Promise<SessionEntry[]> {
    const allEntries = await this.getFullHistory(sessionId);
    const path = this.getPathToNode(allEntries, nodeId);
    return path;
  }
}
```

---

## 6. 实战示例

### 示例 1：长期项目的 Compaction 工作流

```typescript
async function longTermProjectWorkflow() {
  const manager = new SessionManager();
  const compaction = new SessionCompaction();

  // 1. 创建项目会话
  const session = await manager.createNewSession('long-term-project');

  // 2. 工作一段时间（添加很多消息）
  for (let i = 0; i < 100; i++) {
    await session.appendMessage('user', `任务 ${i}`);
    await session.appendMessage('assistant', `完成任务 ${i}`);
  }

  console.log('会话有 200 条消息');

  // 3. 检查是否需要压缩
  const autoCompaction = new AutoCompaction({
    maxEntries: 100
  });

  if (await autoCompaction.shouldCompact(session.id)) {
    console.log('需要压缩');

    // 4. 执行压缩
    const result = await compaction.compact(session.id, {
      keepRecentMessages: 20,
      strategy: 'summarize'
    });

    console.log(`压缩完成:`);
    console.log(`- 原始条目: ${result.fullHistory.length}`);
    console.log(`- 优化后上下文: ${result.context.length}`);
    console.log(`- 摘要: ${result.compactionEntry.summary.substring(0, 100)}...`);
  }

  // 5. 继续工作（使用优化后的上下文）
  await session.appendMessage('user', '新任务');
  const context = await session.getContext();
  console.log(`当前上下文大小: ${context.length} 条`);

  // 6. 需要时访问完整历史
  const historyAccess = new HistoryAccess();
  const fullHistory = await historyAccess.getFullHistory(session.id);
  console.log(`完整历史: ${fullHistory.length} 条`);
}
```

### 示例 2：智能压缩策略

```typescript
class SmartCompaction {
  async smartCompact(sessionId: string): Promise<void> {
    const info = await this.sessionManager.getSessionInfo(sessionId);

    // 根据会话大小选择策略
    let strategy: 'summarize' | 'truncate';
    let keepRecent: number;

    if (info.entryCount < 50) {
      // 小会话，不压缩
      return;
    } else if (info.entryCount < 200) {
      // 中等会话，保留较多消息
      strategy = 'summarize';
      keepRecent = 50;
    } else {
      // 大会话，激进压缩
      strategy = 'summarize';
      keepRecent = 20;
    }

    await this.compaction.compact(sessionId, {
      keepRecentMessages: keepRecent,
      strategy
    });
  }
}
```

### 示例 3：分层压缩

```typescript
class LayeredCompaction {
  async layeredCompact(sessionId: string): Promise<void> {
    const allEntries = await this.storage.readAll();

    // 分层：最近、中期、旧
    const recent = allEntries.slice(-20);      // 最近 20 条
    const middle = allEntries.slice(-100, -20); // 中期 80 条
    const old = allEntries.slice(0, -100);      // 旧消息

    // 旧消息：高度压缩
    const oldSummary = await this.generateSummary(old, 'brief');

    // 中期消息：中度压缩
    const middleSummary = await this.generateSummary(middle, 'detailed');

    // 最近消息：不压缩
    const context = [
      { type: 'summary', level: 'old', content: oldSummary },
      { type: 'summary', level: 'middle', content: middleSummary },
      ...recent
    ];

    return context;
  }

  private async generateSummary(
    entries: SessionEntry[],
    level: 'brief' | 'detailed'
  ): Promise<string> {
    const prompt = level === 'brief'
      ? '用一段话总结以下对话'
      : '详细总结以下对话的关键内容';

    // 调用 LLM 生成摘要
    return await this.llm.generate({ prompt, entries });
  }
}
```

---

## 7. 最佳实践

### 7.1 何时压缩

```typescript
const compactionBestPractices = {
  // 推荐的压缩时机
  triggers: {
    entryCount: 100,      // 超过 100 条消息
    tokens: 50000,        // 超过 50K tokens
    fileSize: 1024 * 1024 // 超过 1MB
  },

  // 保留消息数量
  keepRecent: {
    shortSession: 50,     // 短会话保留 50 条
    mediumSession: 30,    // 中等会话保留 30 条
    longSession: 20       // 长会话保留 20 条
  },

  // 压缩策略选择
  strategySelection: {
    important: 'summarize',  // 重要会话使用总结
    temporary: 'truncate'    // 临时会话使用截断
  }
};
```

### 7.2 压缩频率

```typescript
const compactionFrequency = {
  // 自动压缩间隔
  autoCompaction: {
    checkInterval: 60000,    // 每分钟检查一次
    minInterval: 300000      // 最少 5 分钟压缩一次
  },

  // 手动压缩建议
  manual: {
    afterMilestone: true,    // 里程碑后压缩
    beforeBranch: true,      // 分支前压缩
    dailyCleanup: true       // 每日清理
  }
};
```

### 7.3 性能优化

```typescript
class CompactionOptimization {
  // 批量压缩
  async batchCompact(sessionIds: string[]): Promise<void> {
    await Promise.all(
      sessionIds.map(id => this.compaction.compact(id, {
        keepRecentMessages: 20,
        strategy: 'summarize'
      }))
    );
  }

  // 后台压缩
  async backgroundCompact(sessionId: string): Promise<void> {
    // 不阻塞主流程
    setImmediate(async () => {
      await this.compaction.compact(sessionId, {
        keepRecentMessages: 20,
        strategy: 'summarize'
      });
    });
  }

  // 增量压缩
  async incrementalCompact(sessionId: string): Promise<void> {
    const lastCompaction = await this.getLastCompaction(sessionId);

    // 只压缩上次压缩后的新消息
    const newEntries = await this.getEntriesSince(sessionId, lastCompaction.timestamp);

    if (newEntries.length > 50) {
      await this.compaction.compact(sessionId, {
        keepRecentMessages: 20,
        strategy: 'summarize'
      });
    }
  }
}
```

---

## 学习检查清单

完成本节学习后，你应该能够：

- [ ] 理解为什么需要 Compaction
- [ ] 理解 Compaction 的工作原理
- [ ] 理解 Summarize 和 Truncate 两种策略
- [ ] 配置自动压缩触发条件
- [ ] 使用 /compact 手动压缩
- [ ] 通过 /tree 访问完整历史
- [ ] 实现自定义的压缩策略
- [ ] 应用压缩最佳实践

---

**版本：** v1.0
**最后更新：** 2026-02-18
**维护者：** Claude Code
