# 核心概念 03：Session 状态管理

> **核心理念：** Session 是 pi-mono 管理复杂状态的机制，通过 CustomEntry、labels 和分支实现状态隔离和持久化。

---

## Session 概览

### 什么是 Session？

Session（会话）是 pi-mono 中管理代理状态的容器，提供：

1. **状态隔离**：不同 Session 之间状态独立
2. **状态持久化**：Session 状态自动保存
3. **历史追踪**：记录 Session 的所有操作
4. **分支管理**：支持创建分支 Session
5. **标签系统**：通过 labels 组织 Session

### Session 的核心概念

```typescript
interface Session {
  // 会话标识
  id: string;

  // 会话标签
  labels: string[];

  // 会话状态
  state: Record<string, any>;

  // 自定义条目
  entries: CustomEntry[];

  // 父会话（如果是分支）
  parentId?: string;

  // 创建时间
  createdAt: string;

  // 最后更新时间
  updatedAt: string;
}
```

---

## Session 管理 API

### 创建 Session

```typescript
// 基本用法
const session = await api.createSession({
  labels: ['planning'],
  state: { mode: 'plan' }
});

console.log(session.id); // 'session-1234567890'
```

### SessionConfig 接口

```typescript
interface SessionConfig {
  // 会话标签
  labels?: string[];

  // 初始状态
  state?: Record<string, any>;

  // 是否只读
  readOnly?: boolean;

  // 父会话 ID（创建分支时）
  parentId?: string;

  // 自定义条目
  entries?: CustomEntry[];
}
```

### 完整示例：创建 Plan Mode Session

```typescript
// 创建规划会话
async function createPlanSession(task: string) {
  const session = await api.createSession({
    labels: ['planning', 'active'],
    state: {
      mode: 'plan',
      task,
      startTime: Date.now()
    },
    readOnly: true, // 规划模式下只读
    entries: [
      {
        type: 'system',
        content: `Entered plan mode for: ${task}`,
        timestamp: Date.now()
      }
    ]
  });

  return session;
}

// 使用
const planSession = await createPlanSession('Implement auth system');
console.log(planSession.id); // 'session-1234567890'
console.log(planSession.labels); // ['planning', 'active']
```

### 切换 Session

```typescript
// 切换到指定 Session
await api.switchSession(sessionId);

// 获取当前 Session
const currentSession = api.getCurrentSession();
console.log(currentSession.id);
```

### 获取 Session

```typescript
// 通过 ID 获取
const session = api.getSession(sessionId);

// 通过标签获取
const planningSessions = api.getSessionsByLabel('planning');

// 获取所有 Session
const allSessions = api.getAllSessions();
```

### 更新 Session

```typescript
// 更新 Session 状态
await api.updateSession(sessionId, {
  state: {
    ...session.state,
    progress: 0.5
  }
});

// 添加标签
await api.addSessionLabel(sessionId, 'approved');

// 移除标签
await api.removeSessionLabel(sessionId, 'active');
```

### 删除 Session

```typescript
// 删除 Session
await api.deleteSession(sessionId);

// 删除所有带特定标签的 Session
const sessions = api.getSessionsByLabel('archived');
for (const session of sessions) {
  await api.deleteSession(session.id);
}
```

---

## CustomEntry（自定义条目）

### 什么是 CustomEntry？

CustomEntry 是 Session 中的记录单元，用于存储：

1. **系统消息**：代理的系统级消息
2. **用户输入**：用户的输入记录
3. **代理输出**：代理的输出记录
4. **工具调用**：工具调用的记录
5. **自定义数据**：任意自定义数据

### CustomEntry 接口

```typescript
interface CustomEntry {
  // 条目类型
  type: 'system' | 'user' | 'assistant' | 'tool' | 'custom';

  // 条目内容
  content: string | Record<string, any>;

  // 时间戳
  timestamp: number;

  // 元数据
  metadata?: Record<string, any>;
}
```

### 添加 CustomEntry

```typescript
// 添加系统消息
await api.addSessionEntry(sessionId, {
  type: 'system',
  content: 'Entered plan mode',
  timestamp: Date.now()
});

// 添加用户输入
await api.addSessionEntry(sessionId, {
  type: 'user',
  content: 'Create a plan for auth system',
  timestamp: Date.now()
});

// 添加代理输出
await api.addSessionEntry(sessionId, {
  type: 'assistant',
  content: 'Here is the plan...',
  timestamp: Date.now()
});

// 添加工具调用记录
await api.addSessionEntry(sessionId, {
  type: 'tool',
  content: {
    tool: 'createPlan',
    params: { goal: 'auth system' },
    result: { success: true }
  },
  timestamp: Date.now(),
  metadata: {
    duration: 1500
  }
});
```

### 查询 CustomEntry

```typescript
// 获取所有条目
const entries = api.getSessionEntries(sessionId);

// 按类型过滤
const systemEntries = entries.filter(e => e.type === 'system');

// 按时间范围过滤
const recentEntries = entries.filter(e =>
  e.timestamp > Date.now() - 3600000 // 最近1小时
);
```

### 完整示例：Plan Mode 日志记录

```typescript
// 记录规划过程
async function logPlanningProcess(sessionId: string) {
  // 1. 记录开始
  await api.addSessionEntry(sessionId, {
    type: 'system',
    content: 'Planning started',
    timestamp: Date.now()
  });

  // 2. 记录用户输入
  await api.addSessionEntry(sessionId, {
    type: 'user',
    content: 'Implement user authentication',
    timestamp: Date.now()
  });

  // 3. 记录代理分析
  await api.addSessionEntry(sessionId, {
    type: 'assistant',
    content: 'Analyzing requirements...',
    timestamp: Date.now()
  });

  // 4. 记录工具调用
  await api.addSessionEntry(sessionId, {
    type: 'tool',
    content: {
      tool: 'createPlan',
      params: { goal: 'user authentication' },
      result: { planId: 'plan-001' }
    },
    timestamp: Date.now()
  });

  // 5. 记录完成
  await api.addSessionEntry(sessionId, {
    type: 'system',
    content: 'Planning completed',
    timestamp: Date.now(),
    metadata: {
      duration: 5000,
      tasksCreated: 4
    }
  });
}
```

---

## Labels（标签系统）

### 什么是 Labels？

Labels 是 Session 的分类标签，用于：

1. **组织 Session**：按功能、状态、项目分类
2. **快速查找**：通过标签快速找到相关 Session
3. **批量操作**：对带特定标签的 Session 批量操作
4. **状态管理**：用标签表示 Session 状态

### 常用标签模式

```typescript
// 功能标签
const FUNCTION_LABELS = {
  PLANNING: 'planning',
  EXECUTING: 'executing',
  REVIEWING: 'reviewing',
  DEBUGGING: 'debugging'
};

// 状态标签
const STATUS_LABELS = {
  ACTIVE: 'active',
  PAUSED: 'paused',
  COMPLETED: 'completed',
  FAILED: 'failed',
  ARCHIVED: 'archived'
};

// 项目标签
const PROJECT_LABELS = {
  AUTH_SYSTEM: 'project:auth-system',
  API_REFACTOR: 'project:api-refactor',
  UI_REDESIGN: 'project:ui-redesign'
};

// 优先级标签
const PRIORITY_LABELS = {
  HIGH: 'priority:high',
  MEDIUM: 'priority:medium',
  LOW: 'priority:low'
};
```

### 标签操作

```typescript
// 创建带标签的 Session
const session = await api.createSession({
  labels: ['planning', 'active', 'project:auth-system', 'priority:high']
});

// 添加标签
await api.addSessionLabel(session.id, 'approved');

// 移除标签
await api.removeSessionLabel(session.id, 'active');
await api.addSessionLabel(session.id, 'completed');

// 查询带特定标签的 Session
const activePlanningSessions = api.getSessions({
  labels: ['planning', 'active']
});

// 批量更新标签
const sessions = api.getSessionsByLabel('active');
for (const session of sessions) {
  await api.removeSessionLabel(session.id, 'active');
  await api.addSessionLabel(session.id, 'paused');
}
```

### 完整示例：Plan Mode 状态管理

```typescript
// Plan Mode 生命周期管理
class PlanModeSessionManager {
  // 创建规划会话
  async createPlanSession(task: string) {
    const session = await api.createSession({
      labels: ['planning', 'active', 'draft'],
      state: {
        mode: 'plan',
        task,
        startTime: Date.now()
      }
    });

    await api.addSessionEntry(session.id, {
      type: 'system',
      content: `Created plan session for: ${task}`,
      timestamp: Date.now()
    });

    return session;
  }

  // 批准计划
  async approvePlan(sessionId: string) {
    await api.removeSessionLabel(sessionId, 'draft');
    await api.addSessionLabel(sessionId, 'approved');

    await api.addSessionEntry(sessionId, {
      type: 'system',
      content: 'Plan approved',
      timestamp: Date.now()
    });
  }

  // 开始执行
  async startExecution(sessionId: string) {
    await api.removeSessionLabel(sessionId, 'planning');
    await api.addSessionLabel(sessionId, 'executing');

    await api.updateSession(sessionId, {
      state: {
        ...api.getSession(sessionId).state,
        mode: 'execute',
        executionStartTime: Date.now()
      }
    });
  }

  // 完成执行
  async completeExecution(sessionId: string) {
    await api.removeSessionLabel(sessionId, 'executing');
    await api.removeSessionLabel(sessionId, 'active');
    await api.addSessionLabel(sessionId, 'completed');

    await api.addSessionEntry(sessionId, {
      type: 'system',
      content: 'Execution completed',
      timestamp: Date.now()
    });
  }

  // 归档会话
  async archiveSession(sessionId: string) {
    await api.removeSessionLabel(sessionId, 'completed');
    await api.addSessionLabel(sessionId, 'archived');
  }

  // 查询活跃的规划会话
  getActivePlanningSessions() {
    return api.getSessions({
      labels: ['planning', 'active']
    });
  }

  // 查询待批准的计划
  getDraftPlans() {
    return api.getSessions({
      labels: ['planning', 'draft']
    });
  }
}
```

---

## Session 分支

### 什么是 Session 分支？

Session 分支允许从现有 Session 创建新的独立 Session，用于：

1. **实验性修改**：在分支中尝试不同方案
2. **并行开发**：同时进行多个方向的探索
3. **回滚机制**：保留原始 Session，可随时回退
4. **A/B 测试**：比较不同实现方案

### 创建分支

```typescript
// 从现有 Session 创建分支
const branchSession = await api.createSession({
  parentId: originalSessionId,
  labels: ['planning', 'branch', 'experiment'],
  state: {
    ...api.getSession(originalSessionId).state,
    branchName: 'alternative-approach'
  }
});

console.log(branchSession.parentId); // originalSessionId
```

### 分支管理

```typescript
// 获取 Session 的所有分支
function getSessionBranches(sessionId: string): Session[] {
  const allSessions = api.getAllSessions();
  return allSessions.filter(s => s.parentId === sessionId);
}

// 合并分支（手动实现）
async function mergeBranch(branchId: string, targetId: string) {
  const branch = api.getSession(branchId);
  const target = api.getSession(targetId);

  // 合并状态
  await api.updateSession(targetId, {
    state: {
      ...target.state,
      ...branch.state
    }
  });

  // 合并条目
  const branchEntries = api.getSessionEntries(branchId);
  for (const entry of branchEntries) {
    await api.addSessionEntry(targetId, entry);
  }

  // 标记分支为已合并
  await api.addSessionLabel(branchId, 'merged');
  await api.addSessionLabel(branchId, 'archived');
}
```

### 完整示例：Plan Mode 分支实验

```typescript
// 实验不同的规划方案
async function experimentWithPlanApproaches(originalPlanId: string) {
  // 创建方案 A 分支
  const approachA = await api.createSession({
    parentId: originalPlanId,
    labels: ['planning', 'branch', 'approach-a'],
    state: {
      ...api.getSession(originalPlanId).state,
      approach: 'monolithic'
    }
  });

  // 创建方案 B 分支
  const approachB = await api.createSession({
    parentId: originalPlanId,
    labels: ['planning', 'branch', 'approach-b'],
    state: {
      ...api.getSession(originalPlanId).state,
      approach: 'microservices'
    }
  });

  // 在方案 A 中规划
  await api.switchSession(approachA.id);
  await api.addSessionEntry(approachA.id, {
    type: 'assistant',
    content: 'Planning monolithic approach...',
    timestamp: Date.now()
  });

  // 在方案 B 中规划
  await api.switchSession(approachB.id);
  await api.addSessionEntry(approachB.id, {
    type: 'assistant',
    content: 'Planning microservices approach...',
    timestamp: Date.now()
  });

  // 比较两种方案
  const comparison = {
    approachA: {
      complexity: 'low',
      scalability: 'medium',
      cost: 'low'
    },
    approachB: {
      complexity: 'high',
      scalability: 'high',
      cost: 'high'
    }
  };

  // 选择方案 A
  await mergeBranch(approachA.id, originalPlanId);

  // 归档方案 B
  await api.addSessionLabel(approachB.id, 'rejected');
  await api.addSessionLabel(approachB.id, 'archived');
}
```

---

## 状态持久化

### 自动持久化

Pi-mono 自动将 Session 状态持久化到文件系统：

```
.pi/sessions/
├── session-1234567890.json
├── session-1234567891.json
└── index.json
```

### Session 文件格式

```json
{
  "id": "session-1234567890",
  "labels": ["planning", "active"],
  "state": {
    "mode": "plan",
    "task": "Implement auth system",
    "startTime": 1708516800000
  },
  "entries": [
    {
      "type": "system",
      "content": "Entered plan mode",
      "timestamp": 1708516800000
    }
  ],
  "parentId": null,
  "createdAt": "2026-02-21T10:00:00Z",
  "updatedAt": "2026-02-21T10:30:00Z"
}
```

### 索引文件

```json
{
  "sessions": [
    {
      "id": "session-1234567890",
      "labels": ["planning", "active"],
      "createdAt": "2026-02-21T10:00:00Z",
      "updatedAt": "2026-02-21T10:30:00Z"
    }
  ],
  "lastUpdated": "2026-02-21T10:30:00Z"
}
```

### 手动保存和加载

```typescript
// 保存 Session 到文件
async function saveSession(session: Session) {
  const filepath = `.pi/sessions/${session.id}.json`;
  await api.tools.write(filepath, JSON.stringify(session, null, 2));
}

// 从文件加载 Session
async function loadSession(sessionId: string): Promise<Session> {
  const filepath = `.pi/sessions/${sessionId}.json`;
  const content = await api.tools.read(filepath);
  return JSON.parse(content);
}

// 导出 Session
async function exportSession(sessionId: string, outputPath: string) {
  const session = api.getSession(sessionId);
  await api.tools.write(outputPath, JSON.stringify(session, null, 2));
}

// 导入 Session
async function importSession(inputPath: string): Promise<Session> {
  const content = await api.tools.read(inputPath);
  const session = JSON.parse(content);

  // 创建新 Session
  return await api.createSession({
    labels: session.labels,
    state: session.state,
    entries: session.entries
  });
}
```

---

## 完整示例：Plan Mode Session 管理

### 文件结构

```
~/.pi/extensions/plan-mode-session/
├── index.ts          # 扩展入口
├── session.ts        # Session 管理
├── lifecycle.ts      # 生命周期管理
└── persistence.ts    # 持久化管理
```

### session.ts

```typescript
import { ExtensionAPI, Session } from '@pi/extension-api';

export class PlanModeSessionManager {
  constructor(private api: ExtensionAPI) {}

  // 创建规划会话
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
          content: `Created plan session for: ${task}`,
          timestamp: Date.now()
        }
      ]
    });

    // 创建计划文件
    const planFile = `.pi/plans/${session.id}.md`;
    await this.api.tools.write(planFile, `# Plan: ${task}\n\n`);

    // 更新状态
    await this.api.updateSession(session.id, {
      state: {
        ...session.state,
        planFile
      }
    });

    return session;
  }

  // 切换到执行模式
  async switchToExecutionMode(sessionId: string): Promise<Session> {
    const session = this.api.getSession(sessionId);

    // 创建执行会话（分支）
    const execSession = await this.api.createSession({
      parentId: sessionId,
      labels: ['executing', 'active'],
      state: {
        mode: 'execute',
        planSessionId: sessionId,
        task: session.state.task,
        startTime: Date.now()
      },
      readOnly: false
    });

    // 更新原会话标签
    await this.api.removeSessionLabel(sessionId, 'active');
    await this.api.addSessionLabel(sessionId, 'completed');

    return execSession;
  }

  // 获取活跃的规划会话
  getActivePlanningSessions(): Session[] {
    return this.api.getSessions({
      labels: ['planning', 'active']
    });
  }

  // 获取会话历史
  getSessionHistory(sessionId: string): CustomEntry[] {
    return this.api.getSessionEntries(sessionId);
  }
}
```

---

## 最佳实践

### 1. Session 命名和标签

- **使用描述性标签**：`planning`、`executing`、`reviewing`
- **状态标签**：`active`、`paused`、`completed`
- **项目标签**：`project:auth-system`
- **优先级标签**：`priority:high`

### 2. CustomEntry 使用

- **记录关键事件**：状态变化、工具调用、错误
- **添加元数据**：duration、result、error
- **保持简洁**：只记录必要信息

### 3. 分支管理

- **明确分支目的**：实验、并行开发、回滚
- **及时合并或归档**：避免分支过多
- **保留原始 Session**：作为回退点

### 4. 状态持久化

- **定期备份**：导出重要 Session
- **清理旧 Session**：归档或删除不需要的 Session
- **版本控制**：将 `.pi/sessions/` 纳入 Git

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

- **04_最小可用.md**：学习 20% 核心知识实现 80% 功能
- **07_实战代码_03_Session集成执行.md**：查看完整的 Session 集成代码
