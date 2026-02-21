# 核心概念 02：Extension API 集成

> **核心理念：** Extension API 是 pi-mono 实现 Plan Mode 的核心机制，通过注册命令、工具和事件处理器来构建规划工作流。

---

## Extension API 概览

### 什么是 Extension API？

Extension API 是 pi-mono 提供的扩展接口，允许开发者：

1. **注册自定义命令**：如 `/plan`、`/execute`
2. **注册自定义工具**：扩展代理的能力
3. **添加键盘快捷键**：快速触发命令
4. **管理状态**：持久化扩展状态
5. **监听事件**：响应代理行为
6. **自定义 UI**：渲染自定义界面

### Extension API 的核心接口

```typescript
interface ExtensionAPI {
  // 命令注册
  registerCommand(config: CommandConfig): void;

  // 工具注册
  registerTool(config: ToolConfig): void;

  // 快捷键注册
  registerShortcut(config: ShortcutConfig): void;

  // 状态管理
  setState(state: Record<string, any>): Promise<void>;
  getState(): Record<string, any>;

  // 工具控制
  enableTools(tools: string[]): void;
  disableTools(tools: string[]): void;

  // 事件处理
  on(event: string, handler: Function): void;
  emit(event: string, data: any): void;

  // UI 渲染
  renderCustomUI(component: React.Component): void;

  // 会话管理
  createSession(config: SessionConfig): Promise<Session>;
  switchSession(sessionId: string): Promise<void>;
  getSession(sessionId: string): Session;

  // 工具访问
  tools: {
    write(path: string, content: string): Promise<void>;
    read(path: string): Promise<string>;
    // ... 其他工具
  };
}
```

---

## 命令注册（registerCommand）

### 基本用法

```typescript
api.registerCommand({
  name: 'plan',
  description: 'Enter plan mode for read-only exploration',
  handler: async (args: string[]) => {
    // 命令处理逻辑
    await api.setState({ mode: 'plan' });
    api.disableTools(['write', 'edit']);
    return 'Entered plan mode';
  }
});
```

### CommandConfig 接口

```typescript
interface CommandConfig {
  // 命令名称（不包含 /）
  name: string;

  // 命令描述（显示在帮助中）
  description: string;

  // 命令别名
  aliases?: string[];

  // 命令处理器
  handler: (args: string[]) => Promise<string | void>;

  // 参数定义
  args?: ArgumentDefinition[];

  // 是否隐藏（不显示在帮助中）
  hidden?: boolean;
}

interface ArgumentDefinition {
  name: string;
  description: string;
  required?: boolean;
  default?: any;
}
```

### 完整示例：Plan Mode 命令

```typescript
// 注册 /plan 命令
api.registerCommand({
  name: 'plan',
  description: 'Enter plan mode for architecture planning',
  aliases: ['p'],
  args: [
    {
      name: 'task',
      description: 'Task description',
      required: false
    }
  ],
  handler: async (args: string[]) => {
    const task = args.join(' ');

    // 1. 切换到规划模式
    await api.setState({
      mode: 'plan',
      planStartTime: Date.now(),
      planTask: task
    });

    // 2. 禁用写入工具
    api.disableTools(['write', 'edit', 'delete']);

    // 3. 创建计划文件
    const planFile = '.pi/plan.md';
    await api.tools.write(planFile, `# Plan: ${task}\n\n## Goal\n\n${task}\n\n## Tasks\n\n`);

    // 4. 触发事件
    api.emit('plan:entered', { task, planFile });

    // 5. 返回消息
    return `Entered plan mode. Planning: ${task || 'general task'}`;
  }
});

// 注册 /execute 命令
api.registerCommand({
  name: 'execute',
  description: 'Exit plan mode and start execution',
  aliases: ['e', 'exec'],
  handler: async () => {
    const state = api.getState();

    if (state.mode !== 'plan') {
      return 'Not in plan mode';
    }

    // 1. 切换到执行模式
    await api.setState({
      mode: 'execute',
      planEndTime: Date.now()
    });

    // 2. 启用写入工具
    api.enableTools(['write', 'edit', 'delete']);

    // 3. 触发事件
    api.emit('plan:exited', {
      duration: Date.now() - state.planStartTime
    });

    // 4. 返回消息
    return 'Exited plan mode. Ready to execute.';
  }
});
```

### 命令参数处理

```typescript
api.registerCommand({
  name: 'plan',
  handler: async (args: string[]) => {
    // args 是命令行参数数组
    // 例如：/plan create auth system
    // args = ['create', 'auth', 'system']

    const action = args[0]; // 'create'
    const task = args.slice(1).join(' '); // 'auth system'

    switch (action) {
      case 'create':
        return await createPlan(task);
      case 'list':
        return await listPlans();
      case 'show':
        return await showPlan(task);
      default:
        return 'Unknown action. Use: /plan create|list|show';
    }
  }
});
```

---

## 工具注册（registerTool）

### 基本用法

```typescript
api.registerTool({
  name: 'createPlan',
  description: 'Create a new plan file',
  parameters: {
    type: 'object',
    properties: {
      goal: { type: 'string', description: 'Plan goal' },
      tasks: { type: 'array', items: { type: 'string' } }
    },
    required: ['goal']
  },
  handler: async (params: { goal: string; tasks?: string[] }) => {
    const plan = {
      goal: params.goal,
      tasks: params.tasks || [],
      createdAt: new Date().toISOString()
    };

    const planFile = `.pi/plans/plan-${Date.now()}.json`;
    await api.tools.write(planFile, JSON.stringify(plan, null, 2));

    return { success: true, planFile };
  }
});
```

### ToolConfig 接口

```typescript
interface ToolConfig {
  // 工具名称
  name: string;

  // 工具描述
  description: string;

  // 参数定义（JSON Schema）
  parameters: JSONSchema;

  // 工具处理器
  handler: (params: any) => Promise<any>;

  // 是否异步
  async?: boolean;
}
```

### 完整示例：Plan 管理工具

```typescript
// 创建计划工具
api.registerTool({
  name: 'createPlan',
  description: 'Create a new implementation plan',
  parameters: {
    type: 'object',
    properties: {
      goal: {
        type: 'string',
        description: 'What you want to achieve'
      },
      context: {
        type: 'string',
        description: 'Background information'
      },
      tasks: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            title: { type: 'string' },
            description: { type: 'string' },
            complexity: {
              type: 'string',
              enum: ['low', 'medium', 'high']
            }
          },
          required: ['title', 'description']
        }
      }
    },
    required: ['goal']
  },
  handler: async (params) => {
    const plan = {
      id: `plan-${Date.now()}`,
      goal: params.goal,
      context: params.context || '',
      tasks: params.tasks || [],
      status: 'draft',
      createdAt: new Date().toISOString()
    };

    // 写入文件
    const planFile = `.pi/plans/${plan.id}.json`;
    await api.tools.write(planFile, JSON.stringify(plan, null, 2));

    // 更新索引
    await updatePlanIndex(plan);

    return {
      success: true,
      planId: plan.id,
      planFile
    };
  }
});

// 列出计划工具
api.registerTool({
  name: 'listPlans',
  description: 'List all plans',
  parameters: {
    type: 'object',
    properties: {
      status: {
        type: 'string',
        enum: ['draft', 'approved', 'executing', 'completed'],
        description: 'Filter by status'
      }
    }
  },
  handler: async (params) => {
    const indexFile = '.pi/plans/index.json';
    const indexContent = await api.tools.read(indexFile);
    const index = JSON.parse(indexContent);

    let plans = index.plans;

    if (params.status) {
      plans = plans.filter(p => p.status === params.status);
    }

    return {
      success: true,
      plans,
      count: plans.length
    };
  }
});
```

---

## 快捷键注册（registerShortcut）

### 基本用法

```typescript
api.registerShortcut({
  key: 'Shift+P',
  description: 'Enter plan mode',
  handler: async () => {
    // 触发 /plan 命令
    await api.executeCommand('plan');
  }
});
```

### ShortcutConfig 接口

```typescript
interface ShortcutConfig {
  // 快捷键组合
  key: string;

  // 描述
  description: string;

  // 处理器
  handler: () => Promise<void>;

  // 是否全局（在所有模式下生效）
  global?: boolean;
}
```

### 完整示例：Plan Mode 快捷键

```typescript
// Shift+P: 进入规划模式
api.registerShortcut({
  key: 'Shift+P',
  description: 'Enter plan mode',
  global: true,
  handler: async () => {
    await api.executeCommand('plan');
  }
});

// Shift+E: 退出规划模式
api.registerShortcut({
  key: 'Shift+E',
  description: 'Exit plan mode and execute',
  global: true,
  handler: async () => {
    await api.executeCommand('execute');
  }
});

// Ctrl+Shift+L: 列出所有计划
api.registerShortcut({
  key: 'Ctrl+Shift+L',
  description: 'List all plans',
  handler: async () => {
    const result = await api.callTool('listPlans', {});
    console.log(result.plans);
  }
});
```

---

## 状态管理（setState / getState）

### 基本用法

```typescript
// 设置状态
await api.setState({
  mode: 'plan',
  planFile: '.pi/plan.md',
  planStartTime: Date.now()
});

// 获取状态
const state = api.getState();
console.log(state.mode); // 'plan'
```

### 状态持久化

```typescript
// 状态会自动持久化到 .pi/state.json
// 代理重启后状态会恢复

// 设置状态
await api.setState({
  mode: 'plan',
  currentPlan: 'plan-001',
  history: ['plan-001', 'plan-002']
});

// 状态文件内容（.pi/state.json）
{
  "mode": "plan",
  "currentPlan": "plan-001",
  "history": ["plan-001", "plan-002"],
  "updatedAt": "2026-02-21T10:00:00Z"
}
```

### 状态更新模式

```typescript
// 模式 1：完全替换
await api.setState({ mode: 'plan' });
// 结果：{ mode: 'plan' }

// 模式 2：合并更新
const currentState = api.getState();
await api.setState({
  ...currentState,
  planFile: '.pi/plan.md'
});
// 结果：{ mode: 'plan', planFile: '.pi/plan.md' }

// 模式 3：部分更新（推荐）
await api.updateState({ planFile: '.pi/plan.md' });
// 结果：{ mode: 'plan', planFile: '.pi/plan.md' }
```

### 完整示例：Plan Mode 状态管理

```typescript
interface PlanModeState {
  mode: 'idle' | 'plan' | 'execute';
  currentPlan?: string;
  planFile?: string;
  planStartTime?: number;
  planEndTime?: number;
  history: string[];
}

// 初始化状态
async function initializePlanMode() {
  const state = api.getState() as PlanModeState;

  if (!state.mode) {
    await api.setState({
      mode: 'idle',
      history: []
    });
  }
}

// 进入规划模式
async function enterPlanMode(task: string) {
  const planId = `plan-${Date.now()}`;
  const planFile = `.pi/plans/${planId}.md`;

  await api.setState({
    mode: 'plan',
    currentPlan: planId,
    planFile,
    planStartTime: Date.now()
  });

  // 创建计划文件
  await api.tools.write(planFile, `# Plan: ${task}\n\n`);
}

// 退出规划模式
async function exitPlanMode() {
  const state = api.getState() as PlanModeState;

  await api.setState({
    mode: 'execute',
    planEndTime: Date.now(),
    history: [...state.history, state.currentPlan!]
  });
}
```

---

## 事件处理（on / emit）

### 基本用法

```typescript
// 监听事件
api.on('plan:entered', (data) => {
  console.log('Entered plan mode:', data);
});

// 触发事件
api.emit('plan:entered', { task: 'auth system' });
```

### 内置事件

```typescript
// 代理生命周期事件
api.on('agent:started', () => {});
api.on('agent:stopped', () => {});

// 工具调用事件
api.on('tool:called', (tool, params) => {});
api.on('tool:completed', (tool, result) => {});
api.on('tool:failed', (tool, error) => {});

// 命令执行事件
api.on('command:executed', (command, args) => {});

// 状态变化事件
api.on('state:changed', (oldState, newState) => {});
```

### 自定义事件

```typescript
// 定义 Plan Mode 事件
const PLAN_EVENTS = {
  ENTERED: 'plan:entered',
  EXITED: 'plan:exited',
  TASK_ADDED: 'plan:task:added',
  TASK_COMPLETED: 'plan:task:completed',
  PLAN_APPROVED: 'plan:approved'
};

// 监听事件
api.on(PLAN_EVENTS.ENTERED, (data) => {
  console.log(`Entered plan mode for: ${data.task}`);
  // 可以在这里添加自定义逻辑
  // 例如：记录日志、发送通知等
});

api.on(PLAN_EVENTS.TASK_COMPLETED, (data) => {
  console.log(`Task completed: ${data.taskId}`);
  // 更新进度
  updateProgress(data.taskId);
});

// 触发事件
api.emit(PLAN_EVENTS.ENTERED, {
  task: 'auth system',
  planFile: '.pi/plan.md'
});

api.emit(PLAN_EVENTS.TASK_COMPLETED, {
  taskId: 'task-001',
  duration: 1800
});
```

### 事件处理器模式

```typescript
// 模式 1：简单处理器
api.on('plan:entered', (data) => {
  console.log(data);
});

// 模式 2：异步处理器
api.on('plan:entered', async (data) => {
  await logToFile(data);
});

// 模式 3：错误处理
api.on('plan:entered', async (data) => {
  try {
    await processData(data);
  } catch (error) {
    console.error('Error processing plan:entered event:', error);
  }
});

// 模式 4：一次性处理器
api.once('plan:entered', (data) => {
  console.log('This will only run once');
});

// 模式 5：移除处理器
const handler = (data) => console.log(data);
api.on('plan:entered', handler);
// ... later
api.off('plan:entered', handler);
```

---

## 工具控制（enableTools / disableTools）

### 基本用法

```typescript
// 禁用工具
api.disableTools(['write', 'edit', 'delete']);

// 启用工具
api.enableTools(['write', 'edit', 'delete']);
```

### 完整示例：Plan Mode 工具控制

```typescript
// 进入规划模式时禁用写入工具
async function enterPlanMode() {
  // 禁用所有修改文件的工具
  api.disableTools([
    'write',
    'edit',
    'delete',
    'move',
    'rename'
  ]);

  // 只保留读取工具
  // read, glob, grep 等工具仍然可用

  await api.setState({ mode: 'plan' });
}

// 退出规划模式时启用写入工具
async function exitPlanMode() {
  // 重新启用所有工具
  api.enableTools([
    'write',
    'edit',
    'delete',
    'move',
    'rename'
  ]);

  await api.setState({ mode: 'execute' });
}

// 条件性工具控制
async function setToolsBasedOnMode(mode: string) {
  const writeTools = ['write', 'edit', 'delete', 'move', 'rename'];

  switch (mode) {
    case 'plan':
      api.disableTools(writeTools);
      break;
    case 'execute':
      api.enableTools(writeTools);
      break;
    case 'review':
      // 审查模式：只允许读取和注释
      api.disableTools(writeTools.filter(t => t !== 'write'));
      break;
  }
}
```

---

## UI 渲染（renderCustomUI）

### 基本用法

```typescript
import React from 'react';

// 定义 UI 组件
const PlanModeIndicator: React.FC = () => {
  const state = api.getState();

  if (state.mode !== 'plan') {
    return null;
  }

  return (
    <div style={{
      background: '#FFA500',
      padding: '8px',
      borderRadius: '4px'
    }}>
      📋 Plan Mode Active
    </div>
  );
};

// 渲染 UI
api.renderCustomUI(PlanModeIndicator);
```

### 完整示例：Plan Mode UI

```typescript
import React, { useState, useEffect } from 'react';

interface PlanModeUIProps {
  api: ExtensionAPI;
}

const PlanModeUI: React.FC<PlanModeUIProps> = ({ api }) => {
  const [state, setState] = useState(api.getState());
  const [plans, setPlans] = useState([]);

  useEffect(() => {
    // 监听状态变化
    const handler = () => setState(api.getState());
    api.on('state:changed', handler);

    // 加载计划列表
    loadPlans();

    return () => api.off('state:changed', handler);
  }, []);

  const loadPlans = async () => {
    const result = await api.callTool('listPlans', {});
    setPlans(result.plans);
  };

  const enterPlanMode = async () => {
    await api.executeCommand('plan');
  };

  const exitPlanMode = async () => {
    await api.executeCommand('execute');
  };

  return (
    <div className="plan-mode-ui">
      {/* 模式指示器 */}
      <div className={`mode-indicator mode-${state.mode}`}>
        {state.mode === 'plan' ? '📋 Plan Mode' : '⚡ Execute Mode'}
      </div>

      {/* 控制按钮 */}
      <div className="controls">
        {state.mode === 'plan' ? (
          <button onClick={exitPlanMode}>Exit Plan Mode</button>
        ) : (
          <button onClick={enterPlanMode}>Enter Plan Mode</button>
        )}
      </div>

      {/* 计划列表 */}
      <div className="plans-list">
        <h3>Plans</h3>
        {plans.map(plan => (
          <div key={plan.id} className="plan-item">
            <span>{plan.goal}</span>
            <span className={`status-${plan.status}`}>{plan.status}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

// 注册 UI
api.renderCustomUI(PlanModeUI);
```

---

## 完整扩展示例

### 文件结构

```
~/.pi/extensions/plan-mode/
├── index.ts          # 扩展入口
├── commands.ts       # 命令定义
├── tools.ts          # 工具定义
├── state.ts          # 状态管理
├── events.ts         # 事件处理
├── ui.tsx           # UI 组件
└── package.json      # 扩展配置
```

### index.ts

```typescript
import { ExtensionAPI } from '@pi/extension-api';
import { registerCommands } from './commands';
import { registerTools } from './tools';
import { initializeState } from './state';
import { registerEventHandlers } from './events';
import { PlanModeUI } from './ui';

export default function(api: ExtensionAPI) {
  // 1. 初始化状态
  initializeState(api);

  // 2. 注册命令
  registerCommands(api);

  // 3. 注册工具
  registerTools(api);

  // 4. 注册事件处理器
  registerEventHandlers(api);

  // 5. 注册快捷键
  api.registerShortcut({
    key: 'Shift+P',
    description: 'Enter plan mode',
    handler: () => api.executeCommand('plan')
  });

  api.registerShortcut({
    key: 'Shift+E',
    description: 'Exit plan mode',
    handler: () => api.executeCommand('execute')
  });

  // 6. 渲染 UI
  api.renderCustomUI(PlanModeUI);

  console.log('Plan Mode extension loaded');
}
```

### commands.ts

```typescript
import { ExtensionAPI } from '@pi/extension-api';

export function registerCommands(api: ExtensionAPI) {
  // /plan 命令
  api.registerCommand({
    name: 'plan',
    description: 'Enter plan mode',
    handler: async (args) => {
      const task = args.join(' ');

      await api.setState({
        mode: 'plan',
        planTask: task,
        planStartTime: Date.now()
      });

      api.disableTools(['write', 'edit', 'delete']);
      api.emit('plan:entered', { task });

      return `Entered plan mode: ${task}`;
    }
  });

  // /execute 命令
  api.registerCommand({
    name: 'execute',
    description: 'Exit plan mode',
    handler: async () => {
      const state = api.getState();

      await api.setState({
        mode: 'execute',
        planEndTime: Date.now()
      });

      api.enableTools(['write', 'edit', 'delete']);
      api.emit('plan:exited', {
        duration: Date.now() - state.planStartTime
      });

      return 'Exited plan mode';
    }
  });
}
```

---

## 最佳实践

### 1. 命令设计

- **简短名称**：`/plan` 而非 `/enter-plan-mode`
- **提供别名**：`aliases: ['p']`
- **清晰描述**：帮助用户理解命令用途
- **参数验证**：检查参数有效性

### 2. 状态管理

- **最小状态**：只存储必要信息
- **不可变更新**：使用 `{ ...state, newField }` 模式
- **状态验证**：确保状态一致性
- **定期清理**：删除过期状态

### 3. 事件处理

- **命名规范**：使用 `namespace:action` 格式
- **错误处理**：捕获异步处理器中的错误
- **避免循环**：防止事件触发循环
- **文档化**：记录所有自定义事件

### 4. 工具控制

- **明确禁用**：清楚列出禁用的工具
- **对称操作**：禁用后记得启用
- **状态同步**：工具状态与模式状态同步

---

## 参考资源

### 官方资源

- [Extension API Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md)
- [Plan Mode Example](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent/examples/extensions/plan-mode)

### 研究资料

- `temp/03_grok_pi_mono_extensions.md` - Pi-mono 扩展 API
- `temp/04_grok_pi_mono_examples.md` - Pi-mono 扩展示例

---

## 下一步

- **03_核心概念_03_Session状态管理.md**：学习如何使用 Session 管理复杂状态
- **07_实战代码_02_Extension命令实现.md**：查看完整的扩展实现代码
