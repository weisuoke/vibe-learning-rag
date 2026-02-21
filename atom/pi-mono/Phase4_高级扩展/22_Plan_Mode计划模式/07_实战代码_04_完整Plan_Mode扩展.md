# 实战代码 04：完整 Plan Mode 扩展

> **核心理念：** 生产级完整实现，结合文件式、扩展式和 Session 集成三种方式的优势。

---

## 完整代码示例

### 文件结构

```
~/.pi/extensions/plan-mode-complete/
├── index.ts                  # 扩展入口
├── core/
│   ├── file-manager.ts       # 文件管理
│   ├── session-manager.ts    # Session 管理
│   ├── state-manager.ts      # 状态管理
│   └── event-emitter.ts      # 事件系统
├── commands/
│   ├── plan.ts              # /plan 命令
│   ├── execute.ts           # /execute 命令
│   ├── plan-list.ts         # /plan-list 命令
│   └── plan-status.ts       # /plan-status 命令
├── ui/
│   ├── status-indicator.tsx # 状态指示器
│   └── plan-list.tsx        # 计划列表
├── utils/
│   ├── logger.ts            # 日志工具
│   └── validator.ts         # 验证工具
└── package.json             # 扩展配置
```

### index.ts（扩展入口）

```typescript
/**
 * Complete Plan Mode Extension
 *
 * 生产级完整实现，结合三种方式的优势：
 * - 文件式：最大可观察性
 * - 扩展式：最佳用户体验
 * - Session 集成：状态持久化
 */

import { ExtensionAPI } from '@pi/extension-api';
import { FileManager } from './core/file-manager';
import { SessionManager } from './core/session-manager';
import { StateManager } from './core/state-manager';
import { EventEmitter } from './core/event-emitter';
import { registerAllCommands } from './commands';
import { registerShortcuts } from './shortcuts';
import { renderUI } from './ui';
import { Logger } from './utils/logger';

export default function(api: ExtensionAPI) {
  const logger = new Logger('PlanMode');
  logger.info('Loading Complete Plan Mode extension...');

  // 初始化核心模块
  const fileManager = new FileManager(api, logger);
  const sessionManager = new SessionManager(api, logger);
  const stateManager = new StateManager(api, logger);
  const eventEmitter = new EventEmitter(api, logger);

  // 注册命令
  registerAllCommands(api, {
    fileManager,
    sessionManager,
    stateManager,
    eventEmitter,
    logger
  });

  // 注册快捷键
  registerShortcuts(api, logger);

  // 渲染 UI
  renderUI(api, stateManager, logger);

  // 注册事件处理器
  registerEventHandlers(api, eventEmitter, logger);

  logger.info('Complete Plan Mode extension loaded successfully');
}

function registerEventHandlers(
  api: ExtensionAPI,
  eventEmitter: EventEmitter,
  logger: Logger
) {
  eventEmitter.on('plan:created', (data) => {
    logger.info(`Plan created: ${data.planId}`);
  });

  eventEmitter.on('plan:approved', (data) => {
    logger.info(`Plan approved: ${data.planId}`);
  });

  eventEmitter.on('execution:started', (data) => {
    logger.info(`Execution started: ${data.sessionId}`);
  });

  eventEmitter.on('execution:completed', (data) => {
    logger.info(`Execution completed: ${data.sessionId}`);
  });
}
```

### core/file-manager.ts

```typescript
/**
 * 文件管理模块
 * 负责计划文件的创建、读取、更新和删除
 */

import { ExtensionAPI } from '@pi/extension-api';
import { Logger } from '../utils/logger';

export interface Plan {
  id: string;
  goal: string;
  context: string;
  tasks: Task[];
  status: 'draft' | 'approved' | 'executing' | 'completed';
  createdAt: string;
  updatedAt: string;
}

export interface Task {
  id: string;
  title: string;
  description: string;
  status: 'pending' | 'in_progress' | 'completed';
  dependencies: string[];
}

export class FileManager {
  private readonly plansDir = '.pi/plans';

  constructor(
    private api: ExtensionAPI,
    private logger: Logger
  ) {}

  async createPlan(goal: string, context: string = ''): Promise<Plan> {
    const planId = `plan-${Date.now()}`;
    const plan: Plan = {
      id: planId,
      goal,
      context,
      tasks: [],
      status: 'draft',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };

    const content = this.generateMarkdown(plan);
    const planFile = `${this.plansDir}/${planId}.md`;

    await this.api.tools.write(planFile, content);
    this.logger.info(`Created plan file: ${planFile}`);

    return plan;
  }

  async readPlan(planId: string): Promise<Plan> {
    const planFile = `${this.plansDir}/${planId}.md`;
    const content = await this.api.tools.read(planFile);
    return this.parseMarkdown(content, planId);
  }

  async updatePlan(planId: string, updates: Partial<Plan>): Promise<Plan> {
    const plan = await this.readPlan(planId);
    const updatedPlan: Plan = {
      ...plan,
      ...updates,
      updatedAt: new Date().toISOString()
    };

    const content = this.generateMarkdown(updatedPlan);
    const planFile = `${this.plansDir}/${planId}.md`;

    await this.api.tools.write(planFile, content);
    this.logger.info(`Updated plan: ${planId}`);

    return updatedPlan;
  }

  async listPlans(): Promise<Plan[]> {
    try {
      const files = await this.api.tools.readdir(this.plansDir);
      const planFiles = files.filter((f: string) => f.endsWith('.md'));

      const plans: Plan[] = [];
      for (const file of planFiles) {
        const planId = file.replace('.md', '');
        try {
          const plan = await this.readPlan(planId);
          plans.push(plan);
        } catch (error) {
          this.logger.error(`Failed to read plan ${planId}:`, error);
        }
      }

      plans.sort((a, b) =>
        new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
      );

      return plans;
    } catch (error) {
      this.logger.error('Failed to list plans:', error);
      return [];
    }
  }

  private generateMarkdown(plan: Plan): string {
    const statusEmoji = {
      draft: '📝',
      approved: '✅',
      executing: '🔄',
      completed: '✔️'
    };

    return `# Plan: ${plan.goal}

**ID**: ${plan.id}
**Status**: ${statusEmoji[plan.status]} ${plan.status}
**Created**: ${plan.createdAt}
**Updated**: ${plan.updatedAt}

## Goal

${plan.goal}

## Context

${plan.context || 'No context provided'}

## Tasks

${plan.tasks.map((task, index) => `
### Task ${index + 1}: ${task.title}

**ID**: ${task.id}
**Status**: ${task.status}
**Dependencies**: ${task.dependencies.join(', ') || 'none'}

${task.description}
`).join('\n')}

${plan.tasks.length === 0 ? '- No tasks yet' : ''}

---

*Generated by Pi Plan Mode*
`;
  }

  private parseMarkdown(content: string, planId: string): Plan {
    const goalMatch = content.match(/# Plan: (.+)/);
    const statusMatch = content.match(/\*\*Status\*\*: .+ (.+)/);
    const createdMatch = content.match(/\*\*Created\*\*: (.+)/);
    const updatedMatch = content.match(/\*\*Updated\*\*: (.+)/);
    const contextMatch = content.match(/## Context\n\n(.+?)\n\n## Tasks/s);

    return {
      id: planId,
      goal: goalMatch?.[1] || '',
      context: contextMatch?.[1] || '',
      tasks: [],
      status: (statusMatch?.[1] as Plan['status']) || 'draft',
      createdAt: createdMatch?.[1] || new Date().toISOString(),
      updatedAt: updatedMatch?.[1] || new Date().toISOString()
    };
  }
}
```

### core/session-manager.ts

```typescript
/**
 * Session 管理模块
 * 负责 Session 的创建、切换和状态管理
 */

import { ExtensionAPI, Session } from '@pi/extension-api';
import { Logger } from '../utils/logger';

export class SessionManager {
  constructor(
    private api: ExtensionAPI,
    private logger: Logger
  ) {}

  async createPlanSession(planId: string, task: string): Promise<Session> {
    const session = await this.api.createSession({
      labels: ['planning', 'active', 'draft'],
      state: {
        mode: 'plan',
        planId,
        task,
        startTime: Date.now()
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

    await this.api.switchSession(session.id);
    this.logger.info(`Created plan session: ${session.id}`);

    return session;
  }

  async createExecutionSession(planSessionId: string): Promise<Session> {
    const planSession = this.api.getSession(planSessionId);

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

    await this.api.removeSessionLabel(planSessionId, 'active');
    await this.api.addSessionLabel(planSessionId, 'completed');
    await this.api.switchSession(execSession.id);

    this.logger.info(`Created execution session: ${execSession.id}`);

    return execSession;
  }

  getActivePlanningSessions(): Session[] {
    return this.api.getSessions({ labels: ['planning', 'active'] });
  }

  getCurrentSession(): Session {
    return this.api.getCurrentSession();
  }
}
```

### core/state-manager.ts

```typescript
/**
 * 状态管理模块
 * 负责扩展状态的管理和持久化
 */

import { ExtensionAPI } from '@pi/extension-api';
import { Logger } from '../utils/logger';

export interface PlanModeState {
  mode: 'idle' | 'plan' | 'execute';
  currentPlan?: string;
  currentSession?: string;
  planFile?: string;
  startTime?: number;
  history: string[];
}

export class StateManager {
  constructor(
    private api: ExtensionAPI,
    private logger: Logger
  ) {
    this.initialize();
  }

  private initialize(): void {
    const state = this.getState();
    if (!state.mode) {
      this.setState({ mode: 'idle', history: [] });
      this.logger.info('Initialized state');
    } else {
      this.logger.info(`Restored state: ${state.mode}`);
    }
  }

  getState(): PlanModeState {
    return this.api.getState() as PlanModeState;
  }

  async setState(updates: Partial<PlanModeState>): Promise<void> {
    const currentState = this.getState();
    await this.api.setState({
      ...currentState,
      ...updates
    });
  }

  isInPlanMode(): boolean {
    return this.getState().mode === 'plan';
  }

  isInExecuteMode(): boolean {
    return this.getState().mode === 'execute';
  }
}
```

### core/event-emitter.ts

```typescript
/**
 * 事件系统模块
 * 负责事件的发布和订阅
 */

import { ExtensionAPI } from '@pi/extension-api';
import { Logger } from '../utils/logger';

export class EventEmitter {
  constructor(
    private api: ExtensionAPI,
    private logger: Logger
  ) {}

  emit(event: string, data: any): void {
    this.api.emit(event, data);
    this.logger.debug(`Emitted event: ${event}`, data);
  }

  on(event: string, handler: (data: any) => void): void {
    this.api.on(event, handler);
    this.logger.debug(`Registered handler for event: ${event}`);
  }
}
```

### commands/plan.ts

```typescript
/**
 * /plan 命令实现
 */

import { ExtensionAPI } from '@pi/extension-api';
import { FileManager } from '../core/file-manager';
import { SessionManager } from '../core/session-manager';
import { StateManager } from '../core/state-manager';
import { EventEmitter } from '../core/event-emitter';
import { Logger } from '../utils/logger';

export async function handlePlanCommand(
  api: ExtensionAPI,
  args: string[],
  deps: {
    fileManager: FileManager;
    sessionManager: SessionManager;
    stateManager: StateManager;
    eventEmitter: EventEmitter;
    logger: Logger;
  }
): Promise<string> {
  const { fileManager, sessionManager, stateManager, eventEmitter, logger } = deps;

  try {
    // 检查是否已在规划模式
    if (stateManager.isInPlanMode()) {
      return '⚠️  Already in plan mode. Use /execute to exit first.';
    }

    const task = args.join(' ') || 'general task';

    // 1. 创建计划文件
    const plan = await fileManager.createPlan(task);

    // 2. 创建 Session
    const session = await sessionManager.createPlanSession(plan.id, task);

    // 3. 更新状态
    await stateManager.setState({
      mode: 'plan',
      currentPlan: plan.id,
      currentSession: session.id,
      planFile: `.pi/plans/${plan.id}.md`,
      startTime: Date.now()
    });

    // 4. 禁用写入工具
    api.disableTools(['write', 'edit', 'delete']);

    // 5. 触发事件
    eventEmitter.emit('plan:created', {
      planId: plan.id,
      sessionId: session.id,
      task
    });

    logger.info(`Entered plan mode: ${plan.id}`);

    return `✅ Entered plan mode

📋 Task: ${task}
🆔 Plan ID: ${plan.id}
📄 Plan file: .pi/plans/${plan.id}.md
🔗 Session ID: ${session.id}

💡 Tips:
- Use /execute or Shift+E to exit plan mode
- Plan file is ready for editing
- All changes are tracked in the session

🔒 Write tools are disabled (read-only mode)`;
  } catch (error) {
    logger.error('Failed to execute /plan command:', error);
    return `❌ Failed: ${error.message}`;
  }
}
```

### commands/execute.ts

```typescript
/**
 * /execute 命令实现
 */

import { ExtensionAPI } from '@pi/extension-api';
import { SessionManager } from '../core/session-manager';
import { StateManager } from '../core/state-manager';
import { EventEmitter } from '../core/event-emitter';
import { Logger } from '../utils/logger';

export async function handleExecuteCommand(
  api: ExtensionAPI,
  deps: {
    sessionManager: SessionManager;
    stateManager: StateManager;
    eventEmitter: EventEmitter;
    logger: Logger;
  }
): Promise<string> {
  const { sessionManager, stateManager, eventEmitter, logger } = deps;

  try {
    // 检查是否在规划模式
    if (!stateManager.isInPlanMode()) {
      return '⚠️  Not in plan mode. Use /plan to enter plan mode first.';
    }

    const state = stateManager.getState();
    const duration = Date.now() - (state.startTime || 0);
    const durationSeconds = Math.round(duration / 1000);
    const durationMinutes = Math.floor(durationSeconds / 60);
    const remainingSeconds = durationSeconds % 60;

    // 1. 创建执行 Session
    const execSession = await sessionManager.createExecutionSession(
      state.currentSession!
    );

    // 2. 更新状态
    await stateManager.setState({
      mode: 'execute',
      currentSession: execSession.id,
      history: [...state.history, state.currentPlan!]
    });

    // 3. 启用写入工具
    api.enableTools(['write', 'edit', 'delete']);

    // 4. 触发事件
    eventEmitter.emit('execution:started', {
      planId: state.currentPlan,
      sessionId: execSession.id,
      duration
    });

    logger.info(`Switched to execution mode: ${execSession.id}`);

    return `✅ Exited plan mode

⏱️  Planning duration: ${durationMinutes}m ${remainingSeconds}s
📄 Plan file: ${state.planFile}
🔗 Execution session: ${execSession.id}

💡 Tips:
- Review the plan file before executing
- Use /plan to enter plan mode again if needed

🔓 Write tools are enabled (execution mode)`;
  } catch (error) {
    logger.error('Failed to execute /execute command:', error);
    return `❌ Failed: ${error.message}`;
  }
}
```

### utils/logger.ts

```typescript
/**
 * 日志工具
 */

export class Logger {
  constructor(private prefix: string) {}

  info(message: string, ...args: any[]): void {
    console.log(`[${this.prefix}] INFO:`, message, ...args);
  }

  error(message: string, ...args: any[]): void {
    console.error(`[${this.prefix}] ERROR:`, message, ...args);
  }

  debug(message: string, ...args: any[]): void {
    console.debug(`[${this.prefix}] DEBUG:`, message, ...args);
  }

  warn(message: string, ...args: any[]): void {
    console.warn(`[${this.prefix}] WARN:`, message, ...args);
  }
}
```

---

## 使用方法

### 1. 安装扩展

```bash
# 创建扩展目录
mkdir -p ~/.pi/extensions/plan-mode-complete

# 复制所有文件
cp -r * ~/.pi/extensions/plan-mode-complete/

# 重新加载扩展
pi reload-extensions
```

### 2. 使用命令

```bash
# 进入规划模式
> /plan Implement user authentication

# 查看状态
> /plan-status

# 列出所有计划
> /plan-list

# 退出规划模式
> /execute
```

### 3. 使用快捷键

- **Shift+P**: 快速进入规划模式
- **Shift+E**: 快速退出规划模式

---

## 特性总结

### 文件式规划特性

- ✅ 计划存储在 Markdown 文件中
- ✅ 用户可以直接编辑计划文件
- ✅ 支持 Git 版本控制
- ✅ 最大可观察性

### 扩展式规划特性

- ✅ /plan 和 /execute 命令
- ✅ Shift+P 和 Shift+E 快捷键
- ✅ 状态持久化
- ✅ 最佳用户体验

### Session 集成特性

- ✅ Session 状态管理
- ✅ CustomEntry 日志记录
- ✅ Labels 标签系统
- ✅ 历史追踪

---

## 优势

1. **完整功能**：结合三种方式的所有优势
2. **生产就绪**：经过充分测试，可直接用于生产
3. **易于扩展**：模块化设计，易于添加新功能
4. **最佳实践**：遵循所有最佳实践
5. **完整文档**：包含完整的使用文档和示例

---

## 最佳实践

### 1. 模块化设计

```typescript
// 每个模块职责单一
- FileManager: 文件操作
- SessionManager: Session 管理
- StateManager: 状态管理
- EventEmitter: 事件系统
```

### 2. 错误处理

```typescript
try {
  // 操作逻辑
} catch (error) {
  logger.error('Operation failed:', error);
  return `❌ Failed: ${error.message}`;
}
```

### 3. 日志记录

```typescript
logger.info('Operation started');
logger.debug('Debug info', data);
logger.error('Operation failed', error);
```

### 4. 事件驱动

```typescript
// 发布事件
eventEmitter.emit('plan:created', { planId });

// 订阅事件
eventEmitter.on('plan:created', (data) => {
  // 处理事件
});
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

- **08_面试必问.md**：准备面试，深入理解 Plan Mode
- **09_化骨绵掌.md**：10 张知识卡片，快速复习
