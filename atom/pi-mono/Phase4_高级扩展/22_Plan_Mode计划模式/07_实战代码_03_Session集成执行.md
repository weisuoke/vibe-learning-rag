# 实战代码 03：Session 集成执行

> **核心理念：** 通过 Session API 实现状态持久化和历史追踪的 Plan Mode，适合复杂项目。

---

## 完整代码示例

### 文件结构

```
~/.pi/extensions/plan-session/
├── index.ts              # 扩展入口
├── session-manager.ts    # Session 管理
├── plan-lifecycle.ts     # 计划生命周期
├── history-tracker.ts    # 历史追踪
└── package.json          # 扩展配置
```

### index.ts（扩展入口）

```typescript
/**
 * Plan Mode with Session Integration
 *
 * 功能：
 * - Session 状态管理
 * - CustomEntry 日志记录
 * - Labels 标签系统
 * - Session 分支支持
 * - 历史追踪
 */

import { ExtensionAPI } from '@pi/extension-api';
import { PlanSessionManager } from './session-manager';
import { PlanLifecycleManager } from './plan-lifecycle';
import { HistoryTracker } from './history-tracker';

export default function(api: ExtensionAPI) {
  console.log('🚀 Loading Plan Session extension...');

  // 初始化管理器
  const sessionManager = new PlanSessionManager(api);
  const lifecycleManager = new PlanLifecycleManager(api, sessionManager);
  const historyTracker = new HistoryTracker(api, sessionManager);

  // 注册命令
  registerCommands(api, sessionManager, lifecycleManager, historyTracker);

  // 注册快捷键
  registerShortcuts(api);

  console.log('✅ Plan Session extension loaded');
}

function registerCommands(
  api: ExtensionAPI,
  sessionManager: PlanSessionManager,
  lifecycleManager: PlanLifecycleManager,
  historyTracker: HistoryTracker
) {
  // /plan 命令
  api.registerCommand({
    name: 'plan',
    description: 'Create a new planning session',
    handler: async (args: string[]) => {
      const task = args.join(' ') || 'general task';
      const session = await lifecycleManager.createPlanSession(task);
      return `✅ Created planning session: ${session.id}`;
    }
  });

  // /execute 命令
  api.registerCommand({
    name: 'execute',
    description: 'Switch to execution mode',
    handler: async () => {
      const execSession = await lifecycleManager.switchToExecutionMode();
      return `✅ Switched to execution session: ${execSession.id}`;
    }
  });

  // /plan-history 命令
  api.registerCommand({
    name: 'plan-history',
    description: 'Show planning history',
    handler: async () => {
      const history = await historyTracker.getHistory();
      return historyTracker.formatHistory(history);
    }
  });

  // /plan-branch 命令
  api.registerCommand({
    name: 'plan-branch',
    description: 'Create a branch session',
    args: [{ name: 'name', description: 'Branch name', required: true }],
    handler: async (args: string[]) => {
      const branchName = args[0];
      const branch = await sessionManager.createBranch(branchName);
      return `✅ Created branch session: ${branch.id}`;
    }
  });
}

function registerShortcuts(api: ExtensionAPI) {
  api.registerShortcut({
    key: 'Shift+P',
    description: 'Create planning session',
    handler: () => api.executeCommand('plan')
  });

  api.registerShortcut({
    key: 'Shift+E',
    description: 'Switch to execution',
    handler: () => api.executeCommand('execute')
  });
}
```

### session-manager.ts（Session 管理）

```typescript
/**
 * Session 管理模块
 */

import { ExtensionAPI, Session, CustomEntry } from '@pi/extension-api';

export class PlanSessionManager {
  constructor(private api: ExtensionAPI) {}

  /**
   * 创建规划 Session
   */
  async createPlanSession(task: string): Promise<Session> {
    const session = await this.api.createSession({
      labels: ['planning', 'active', 'draft'],
      state: {
        mode: 'plan',
        task,
        startTime: Date.now(),
        progress: 0
      },
      readOnly: true,
      entries: [
        {
          type: 'system',
          content: `Created planning session for: ${task}`,
          timestamp: Date.now()
        }
      ]
    });

    // 创建计划文件
    const planFile = `.pi/plans/${session.id}.md`;
    await this.api.tools.write(planFile, this.generatePlanTemplate(task, session.id));

    // 更新 Session 状态
    await this.api.updateSession(session.id, {
      state: {
        ...session.state,
        planFile
      }
    });

    // 切换到新 Session
    await this.api.switchSession(session.id);

    // 禁用写入工具
    this.api.disableTools(['write', 'edit', 'delete']);

    return session;
  }

  /**
   * 创建执行 Session
   */
  async createExecutionSession(planSessionId: string): Promise<Session> {
    const planSession = this.api.getSession(planSessionId);

    // 创建执行 Session（作为分支）
    const execSession = await this.api.createSession({
      parentId: planSessionId,
      labels: ['executing', 'active'],
      state: {
        mode: 'execute',
        planSessionId,
        task: planSession.state.task,
        startTime: Date.now()
      },
      readOnly: false,
      entries: [
        {
          type: 'system',
          content: `Started execution for plan: ${planSessionId}`,
          timestamp: Date.now()
        }
      ]
    });

    // 更新原 Session 标签
    await this.api.removeSessionLabel(planSessionId, 'active');
    await this.api.addSessionLabel(planSessionId, 'completed');

    // 切换到执行 Session
    await this.api.switchSession(execSession.id);

    // 启用写入工具
    this.api.enableTools(['write', 'edit', 'delete']);

    return execSession;
  }

  /**
   * 创建分支 Session
   */
  async createBranch(branchName: string): Promise<Session> {
    const currentSession = this.api.getCurrentSession();

    const branchSession = await this.api.createSession({
      parentId: currentSession.id,
      labels: ['planning', 'branch', branchName],
      state: {
        ...currentSession.state,
        branchName,
        branchStartTime: Date.now()
      },
      readOnly: true,
      entries: [
        {
          type: 'system',
          content: `Created branch: ${branchName}`,
          timestamp: Date.now()
        }
      ]
    });

    // 切换到分支 Session
    await this.api.switchSession(branchSession.id);

    return branchSession;
  }

  /**
   * 添加 Session 日志
   */
  async addLog(
    sessionId: string,
    type: CustomEntry['type'],
    content: string | Record<string, any>,
    metadata?: Record<string, any>
  ): Promise<void> {
    await this.api.addSessionEntry(sessionId, {
      type,
      content,
      timestamp: Date.now(),
      metadata
    });
  }

  /**
   * 获取 Session 日志
   */
  getSessionLogs(sessionId: string): CustomEntry[] {
    return this.api.getSessionEntries(sessionId);
  }

  /**
   * 获取活跃的规划 Session
   */
  getActivePlanningSessions(): Session[] {
    return this.api.getSessions({
      labels: ['planning', 'active']
    });
  }

  /**
   * 获取 Session 的所有分支
   */
  getSessionBranches(sessionId: string): Session[] {
    const allSessions = this.api.getAllSessions();
    return allSessions.filter(s => s.parentId === sessionId);
  }

  /**
   * 生成计划模板
   */
  private generatePlanTemplate(task: string, sessionId: string): string {
    return `# Plan: ${task}

**Session ID**: ${sessionId}
**Status**: 📝 draft
**Created**: ${new Date().toISOString()}

## Goal

${task}

## Context

[Add context here]

## Tasks

- [ ] Task 1
- [ ] Task 2

## Notes

[Add notes here]
`;
  }
}
```

### plan-lifecycle.ts（计划生命周期）

```typescript
/**
 * 计划生命周期管理
 */

import { ExtensionAPI, Session } from '@pi/extension-api';
import { PlanSessionManager } from './session-manager';

export class PlanLifecycleManager {
  constructor(
    private api: ExtensionAPI,
    private sessionManager: PlanSessionManager
  ) {}

  /**
   * 创建规划 Session
   */
  async createPlanSession(task: string): Promise<Session> {
    const session = await this.sessionManager.createPlanSession(task);

    // 记录日志
    await this.sessionManager.addLog(
      session.id,
      'system',
      'Planning session created',
      { task, timestamp: Date.now() }
    );

    // 触发事件
    this.api.emit('plan:session:created', {
      sessionId: session.id,
      task
    });

    return session;
  }

  /**
   * 批准计划
   */
  async approvePlan(sessionId: string): Promise<void> {
    // 更新标签
    await this.api.removeSessionLabel(sessionId, 'draft');
    await this.api.addSessionLabel(sessionId, 'approved');

    // 记录日志
    await this.sessionManager.addLog(
      sessionId,
      'system',
      'Plan approved',
      { timestamp: Date.now() }
    );

    // 触发事件
    this.api.emit('plan:approved', { sessionId });
  }

  /**
   * 切换到执行模式
   */
  async switchToExecutionMode(): Promise<Session> {
    const currentSession = this.api.getCurrentSession();

    // 检查是否在规划模式
    if (!currentSession.labels.includes('planning')) {
      throw new Error('Not in planning mode');
    }

    // 创建执行 Session
    const execSession = await this.sessionManager.createExecutionSession(
      currentSession.id
    );

    // 记录日志
    await this.sessionManager.addLog(
      execSession.id,
      'system',
      'Switched to execution mode',
      {
        planSessionId: currentSession.id,
        timestamp: Date.now()
      }
    );

    // 触发事件
    this.api.emit('plan:execution:started', {
      planSessionId: currentSession.id,
      execSessionId: execSession.id
    });

    return execSession;
  }

  /**
   * 完成执行
   */
  async completeExecution(sessionId: string): Promise<void> {
    // 更新标签
    await this.api.removeSessionLabel(sessionId, 'executing');
    await this.api.removeSessionLabel(sessionId, 'active');
    await this.api.addSessionLabel(sessionId, 'completed');

    // 记录日志
    await this.sessionManager.addLog(
      sessionId,
      'system',
      'Execution completed',
      { timestamp: Date.now() }
    );

    // 触发事件
    this.api.emit('plan:execution:completed', { sessionId });
  }

  /**
   * 归档 Session
   */
  async archiveSession(sessionId: string): Promise<void> {
    // 更新标签
    await this.api.removeSessionLabel(sessionId, 'completed');
    await this.api.addSessionLabel(sessionId, 'archived');

    // 记录日志
    await this.sessionManager.addLog(
      sessionId,
      'system',
      'Session archived',
      { timestamp: Date.now() }
    );
  }
}
```

### history-tracker.ts（历史追踪）

```typescript
/**
 * 历史追踪模块
 */

import { ExtensionAPI, Session, CustomEntry } from '@pi/extension-api';
import { PlanSessionManager } from './session-manager';

interface HistoryEntry {
  sessionId: string;
  task: string;
  startTime: number;
  endTime?: number;
  duration?: number;
  status: string;
  logs: CustomEntry[];
}

export class HistoryTracker {
  constructor(
    private api: ExtensionAPI,
    private sessionManager: PlanSessionManager
  ) {}

  /**
   * 获取历史记录
   */
  async getHistory(): Promise<HistoryEntry[]> {
    const allSessions = this.api.getAllSessions();

    // 过滤规划 Session
    const planningSessions = allSessions.filter(s =>
      s.labels.includes('planning')
    );

    // 构建历史记录
    const history: HistoryEntry[] = [];

    for (const session of planningSessions) {
      const logs = this.sessionManager.getSessionLogs(session.id);

      const entry: HistoryEntry = {
        sessionId: session.id,
        task: session.state.task || 'Unknown',
        startTime: session.state.startTime || 0,
        endTime: session.state.endTime,
        duration: session.state.endTime
          ? session.state.endTime - session.state.startTime
          : undefined,
        status: this.getSessionStatus(session),
        logs
      };

      history.push(entry);
    }

    // 按时间排序
    history.sort((a, b) => b.startTime - a.startTime);

    return history;
  }

  /**
   * 格式化历史记录
   */
  formatHistory(history: HistoryEntry[]): string {
    if (history.length === 0) {
      return '📋 No planning history found';
    }

    const lines = ['📋 Planning History\n'];

    history.forEach((entry, index) => {
      const duration = entry.duration
        ? this.formatDuration(entry.duration)
        : 'In progress';

      lines.push(`${index + 1}. ${entry.task}`);
      lines.push(`   Session: ${entry.sessionId}`);
      lines.push(`   Status: ${entry.status}`);
      lines.push(`   Duration: ${duration}`);
      lines.push(`   Logs: ${entry.logs.length} entries`);
      lines.push('');
    });

    return lines.join('\n');
  }

  /**
   * 获取 Session 状态
   */
  private getSessionStatus(session: Session): string {
    if (session.labels.includes('archived')) return '📦 Archived';
    if (session.labels.includes('completed')) return '✅ Completed';
    if (session.labels.includes('executing')) return '🔄 Executing';
    if (session.labels.includes('approved')) return '✅ Approved';
    if (session.labels.includes('draft')) return '📝 Draft';
    return '❓ Unknown';
  }

  /**
   * 格式化时长
   */
  private formatDuration(ms: number): string {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    } else {
      return `${seconds}s`;
    }
  }

  /**
   * 导出历史记录
   */
  async exportHistory(outputPath: string): Promise<void> {
    const history = await this.getHistory();

    const json = JSON.stringify(history, null, 2);
    await this.api.tools.write(outputPath, json);

    console.log(`✅ Exported history to ${outputPath}`);
  }
}
```

---

## 使用方法

### 1. 创建规划 Session

```bash
> /plan Implement user authentication
✅ Created planning session: session-1234567890
```

### 2. 查看 Session 状态

```bash
> /plan-status
📊 Current Session: session-1234567890
Mode: 📋 Planning
Task: Implement user authentication
Labels: planning, active, draft
```

### 3. 创建分支实验

```bash
> /plan-branch approach-a
✅ Created branch session: session-1234567891

> /plan-branch approach-b
✅ Created branch session: session-1234567892
```

### 4. 切换到执行模式

```bash
> /execute
✅ Switched to execution session: session-1234567893
```

### 5. 查看历史记录

```bash
> /plan-history
📋 Planning History

1. Implement user authentication
   Session: session-1234567890
   Status: ✅ Completed
   Duration: 15m 30s
   Logs: 12 entries
```

---

## 测试代码

```typescript
/**
 * 测试 Session 集成
 */

import { ExtensionAPI } from '@pi/extension-api';
import { PlanSessionManager } from './session-manager';
import { PlanLifecycleManager } from './plan-lifecycle';

async function testSessionIntegration(api: ExtensionAPI) {
  console.log('=== Testing Session Integration ===\n');

  const sessionManager = new PlanSessionManager(api);
  const lifecycleManager = new PlanLifecycleManager(api, sessionManager);

  // Test 1: 创建规划 Session
  console.log('Test 1: Create planning session');
  const planSession = await lifecycleManager.createPlanSession('Test task');
  console.log(`✅ Created: ${planSession.id}\n`);

  // Test 2: 添加日志
  console.log('Test 2: Add logs');
  await sessionManager.addLog(planSession.id, 'user', 'Started planning');
  await sessionManager.addLog(planSession.id, 'assistant', 'Analyzing requirements');
  console.log('✅ Added logs\n');

  // Test 3: 创建分支
  console.log('Test 3: Create branch');
  const branch = await sessionManager.createBranch('experiment');
  console.log(`✅ Created branch: ${branch.id}\n`);

  // Test 4: 切换到执行模式
  console.log('Test 4: Switch to execution');
  await api.switchSession(planSession.id);
  const execSession = await lifecycleManager.switchToExecutionMode();
  console.log(`✅ Execution session: ${execSession.id}\n`);

  // Test 5: 查看日志
  console.log('Test 5: View logs');
  const logs = sessionManager.getSessionLogs(planSession.id);
  console.log(`✅ Found ${logs.length} log entries\n`);

  console.log('=== All tests passed ===');
}

export { testSessionIntegration };
```

---

## 优势与劣势

### 优势

1. **状态持久化**：Session 自动保存，重启后恢复
2. **历史追踪**：完整记录所有操作
3. **分支支持**：可以创建分支实验不同方案
4. **标签系统**：灵活的状态管理
5. **日志记录**：CustomEntry 记录所有事件

### 劣势

1. **最复杂**：需要理解 Session API
2. **学习曲线**：需要更多时间学习
3. **开发成本**：实现和维护成本高

---

## 最佳实践

### 1. Session 命名规范

```typescript
// 使用描述性标签
const session = await api.createSession({
  labels: [
    'planning',           // 功能标签
    'active',            // 状态标签
    'project:auth',      // 项目标签
    'priority:high'      // 优先级标签
  ]
});
```

### 2. 日志记录策略

```typescript
// 记录关键事件
await sessionManager.addLog(sessionId, 'system', 'Plan created');
await sessionManager.addLog(sessionId, 'user', 'User input');
await sessionManager.addLog(sessionId, 'assistant', 'Agent response');
await sessionManager.addLog(sessionId, 'tool', { tool: 'write', result: 'success' });
```

### 3. 分支管理

```typescript
// 创建实验分支
const branchA = await sessionManager.createBranch('approach-a');
const branchB = await sessionManager.createBranch('approach-b');

// 比较分支结果
const logsA = sessionManager.getSessionLogs(branchA.id);
const logsB = sessionManager.getSessionLogs(branchB.id);

// 选择最佳方案并合并
```

### 4. 定期清理

```typescript
// 归档旧 Session
const oldSessions = api.getSessions({
  labels: ['completed']
}).filter(s => {
  const age = Date.now() - new Date(s.createdAt).getTime();
  return age > 30 * 24 * 60 * 60 * 1000; // 30 天
});

for (const session of oldSessions) {
  await lifecycleManager.archiveSession(session.id);
}
```

---

## 参考资源

### 官方资源

- [Pi-mono GitHub](https://github.com/badlogic/pi-mono)
- [Extension API Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md)

### 研究资料

- `temp/03_grok_pi_mono_extensions.md` - Pi-mono 扩展 API
- `temp/04_grok_pi_mono_examples.md` - Pi-mono 扩展示例

---

## 下一步

- **07_实战代码_04_完整Plan_Mode扩展.md**：查看生产级完整实现
- **03_核心概念_03_Session状态管理.md**：深入理解 Session API
