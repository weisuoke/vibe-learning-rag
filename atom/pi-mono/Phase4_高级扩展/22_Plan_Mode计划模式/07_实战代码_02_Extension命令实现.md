# 实战代码 02：Extension 命令实现

> **核心理念：** 通过 Extension API 实现 /plan 和 /execute 命令，提供最佳用户体验的 Plan Mode。

---

## 完整代码示例

### 文件结构

```
~/.pi/extensions/plan-mode/
├── index.ts              # 扩展入口
├── commands.ts           # 命令实现
├── state.ts              # 状态管理
├── events.ts             # 事件处理
├── package.json          # 扩展配置
└── README.md            # 扩展文档
```

### package.json

```json
{
  "name": "@pi/plan-mode",
  "version": "1.0.0",
  "description": "Plan Mode extension for Pi coding agent",
  "main": "index.ts",
  "keywords": ["pi", "extension", "plan-mode"],
  "author": "Your Name",
  "license": "MIT"
}
```

### index.ts（扩展入口）

```typescript
/**
 * Plan Mode Extension
 *
 * 功能：
 * - /plan 命令：进入规划模式
 * - /execute 命令：退出规划模式
 * - Shift+P 快捷键：快速进入规划模式
 * - Shift+E 快捷键：快速退出规划模式
 * - 状态持久化
 * - 事件通知
 */

import { ExtensionAPI } from '@pi/extension-api';
import { registerCommands } from './commands';
import { initializeState, PlanModeState } from './state';
import { registerEventHandlers } from './events';

export default function(api: ExtensionAPI) {
  console.log('🚀 Loading Plan Mode extension...');

  // 1. 初始化状态
  initializeState(api);

  // 2. 注册命令
  registerCommands(api);

  // 3. 注册事件处理器
  registerEventHandlers(api);

  // 4. 注册快捷键
  registerShortcuts(api);

  console.log('✅ Plan Mode extension loaded');
}

/**
 * 注册快捷键
 */
function registerShortcuts(api: ExtensionAPI) {
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
      await api.executeCommand('plan-list');
    }
  });

  console.log('✅ Registered shortcuts: Shift+P, Shift+E, Ctrl+Shift+L');
}
```

### state.ts（状态管理）

```typescript
/**
 * 状态管理模块
 */

import { ExtensionAPI } from '@pi/extension-api';

export interface PlanModeState {
  // 当前模式
  mode: 'idle' | 'plan' | 'execute';

  // 当前计划
  currentPlan?: string;

  // 计划文件路径
  planFile?: string;

  // 规划开始时间
  planStartTime?: number;

  // 规划结束时间
  planEndTime?: number;

  // 历史记录
  history: string[];
}

/**
 * 初始化状态
 */
export function initializeState(api: ExtensionAPI): void {
  const state = api.getState() as PlanModeState;

  if (!state.mode) {
    api.setState({
      mode: 'idle',
      history: []
    });
    console.log('✅ Initialized Plan Mode state');
  } else {
    console.log(`✅ Restored Plan Mode state: ${state.mode}`);
  }
}

/**
 * 获取当前状态
 */
export function getCurrentState(api: ExtensionAPI): PlanModeState {
  return api.getState() as PlanModeState;
}

/**
 * 更新状态
 */
export async function updateState(
  api: ExtensionAPI,
  updates: Partial<PlanModeState>
): Promise<void> {
  const currentState = getCurrentState(api);
  await api.setState({
    ...currentState,
    ...updates
  });
}

/**
 * 检查是否在规划模式
 */
export function isInPlanMode(api: ExtensionAPI): boolean {
  const state = getCurrentState(api);
  return state.mode === 'plan';
}

/**
 * 检查是否在执行模式
 */
export function isInExecuteMode(api: ExtensionAPI): boolean {
  const state = getCurrentState(api);
  return state.mode === 'execute';
}
```

### commands.ts（命令实现）

```typescript
/**
 * 命令实现模块
 */

import { ExtensionAPI } from '@pi/extension-api';
import { getCurrentState, updateState, isInPlanMode } from './state';

/**
 * 注册所有命令
 */
export function registerCommands(api: ExtensionAPI): void {
  // /plan 命令
  api.registerCommand({
    name: 'plan',
    description: 'Enter plan mode for architecture planning',
    aliases: ['p'],
    args: [
      {
        name: 'task',
        description: 'Task description (optional)',
        required: false
      }
    ],
    handler: async (args: string[]) => {
      return await handlePlanCommand(api, args);
    }
  });

  // /execute 命令
  api.registerCommand({
    name: 'execute',
    description: 'Exit plan mode and start execution',
    aliases: ['e', 'exec'],
    handler: async () => {
      return await handleExecuteCommand(api);
    }
  });

  // /plan-list 命令
  api.registerCommand({
    name: 'plan-list',
    description: 'List all plans',
    aliases: ['pl'],
    handler: async () => {
      return await handlePlanListCommand(api);
    }
  });

  // /plan-show 命令
  api.registerCommand({
    name: 'plan-show',
    description: 'Show plan details',
    args: [
      {
        name: 'planId',
        description: 'Plan ID',
        required: true
      }
    ],
    handler: async (args: string[]) => {
      return await handlePlanShowCommand(api, args[0]);
    }
  });

  // /plan-status 命令
  api.registerCommand({
    name: 'plan-status',
    description: 'Show current plan mode status',
    handler: async () => {
      return await handlePlanStatusCommand(api);
    }
  });

  console.log('✅ Registered commands: /plan, /execute, /plan-list, /plan-show, /plan-status');
}

/**
 * 处理 /plan 命令
 */
async function handlePlanCommand(
  api: ExtensionAPI,
  args: string[]
): Promise<string> {
  const task = args.join(' ') || 'general task';

  // 检查是否已经在规划模式
  if (isInPlanMode(api)) {
    return '⚠️  Already in plan mode. Use /execute to exit first.';
  }

  // 生成计划 ID 和文件路径
  const planId = `plan-${Date.now()}`;
  const planFile = `.pi/plans/${planId}.md`;

  // 更新状态
  await updateState(api, {
    mode: 'plan',
    currentPlan: planId,
    planFile,
    planStartTime: Date.now()
  });

  // 禁用写入工具
  api.disableTools(['write', 'edit', 'delete', 'move', 'rename']);

  // 创建计划文件
  const planContent = generatePlanTemplate(task, planId);
  await api.tools.write(planFile, planContent);

  // 触发事件
  api.emit('plan:entered', {
    task,
    planId,
    planFile,
    timestamp: Date.now()
  });

  // 返回消息
  return `✅ Entered plan mode

📋 Task: ${task}
🆔 Plan ID: ${planId}
📄 Plan file: ${planFile}

💡 Tips:
- Use /execute or Shift+E to exit plan mode
- Use /plan-status to check current status
- Plan file is created and ready for editing

🔒 Write tools are disabled (read-only mode)`;
}

/**
 * 处理 /execute 命令
 */
async function handleExecuteCommand(api: ExtensionAPI): Promise<string> {
  const state = getCurrentState(api);

  // 检查是否在规划模式
  if (!isInPlanMode(api)) {
    return '⚠️  Not in plan mode. Use /plan to enter plan mode first.';
  }

  // 计算规划时长
  const duration = Date.now() - (state.planStartTime || 0);
  const durationSeconds = Math.round(duration / 1000);
  const durationMinutes = Math.floor(durationSeconds / 60);
  const remainingSeconds = durationSeconds % 60;

  // 更新状态
  await updateState(api, {
    mode: 'execute',
    planEndTime: Date.now(),
    history: [...state.history, state.currentPlan!]
  });

  // 启用写入工具
  api.enableTools(['write', 'edit', 'delete', 'move', 'rename']);

  // 触发事件
  api.emit('plan:exited', {
    planId: state.currentPlan,
    duration,
    timestamp: Date.now()
  });

  // 返回消息
  return `✅ Exited plan mode

⏱️  Planning duration: ${durationMinutes}m ${remainingSeconds}s
📄 Plan file: ${state.planFile}

💡 Tips:
- Review the plan file before executing
- Use /plan to enter plan mode again if needed

🔓 Write tools are enabled (execution mode)`;
}

/**
 * 处理 /plan-list 命令
 */
async function handlePlanListCommand(api: ExtensionAPI): Promise<string> {
  try {
    // 读取计划目录
    const plansDir = '.pi/plans';
    const files = await api.tools.readdir(plansDir);

    // 过滤 .md 文件
    const planFiles = files.filter((f: string) => f.endsWith('.md'));

    if (planFiles.length === 0) {
      return '📋 No plans found. Use /plan to create a new plan.';
    }

    // 读取所有计划的基本信息
    const plans = [];
    for (const file of planFiles) {
      const planId = file.replace('.md', '');
      const planPath = `${plansDir}/${file}`;
      const content = await api.tools.read(planPath);

      // 提取基本信息
      const goalMatch = content.match(/# Plan: (.+)/);
      const statusMatch = content.match(/\*\*Status\*\*: .+ (.+)/);

      plans.push({
        id: planId,
        goal: goalMatch?.[1] || 'Unknown',
        status: statusMatch?.[1] || 'unknown',
        file: planPath
      });
    }

    // 生成列表
    const list = plans.map((p, i) =>
      `${i + 1}. ${p.id}\n   Goal: ${p.goal}\n   Status: ${p.status}\n   File: ${p.file}`
    ).join('\n\n');

    return `📋 Plans (${plans.length} total)\n\n${list}\n\n💡 Use /plan-show <planId> to view details`;
  } catch (error) {
    return `❌ Failed to list plans: ${error.message}`;
  }
}

/**
 * 处理 /plan-show 命令
 */
async function handlePlanShowCommand(
  api: ExtensionAPI,
  planId: string
): Promise<string> {
  try {
    const planFile = `.pi/plans/${planId}.md`;
    const content = await api.tools.read(planFile);

    // 提取关键信息
    const goalMatch = content.match(/# Plan: (.+)/);
    const statusMatch = content.match(/\*\*Status\*\*: .+ (.+)/);
    const createdMatch = content.match(/\*\*Created\*\*: (.+)/);
    const tasksMatch = content.match(/## Tasks\n\n([\s\S]+?)\n\n##/);

    return `📋 Plan Details

🆔 ID: ${planId}
🎯 Goal: ${goalMatch?.[1] || 'Unknown'}
📊 Status: ${statusMatch?.[1] || 'unknown'}
📅 Created: ${createdMatch?.[1] || 'Unknown'}

📄 Full content:
${content}`;
  } catch (error) {
    return `❌ Failed to show plan: ${error.message}`;
  }
}

/**
 * 处理 /plan-status 命令
 */
async function handlePlanStatusCommand(api: ExtensionAPI): Promise<string> {
  const state = getCurrentState(api);

  if (state.mode === 'idle') {
    return `📊 Plan Mode Status

Mode: 🟢 Idle
Status: Ready to start planning

💡 Use /plan to enter plan mode`;
  }

  if (state.mode === 'plan') {
    const duration = Date.now() - (state.planStartTime || 0);
    const durationSeconds = Math.round(duration / 1000);
    const durationMinutes = Math.floor(durationSeconds / 60);
    const remainingSeconds = durationSeconds % 60;

    return `📊 Plan Mode Status

Mode: 📋 Planning
Current Plan: ${state.currentPlan}
Plan File: ${state.planFile}
Duration: ${durationMinutes}m ${remainingSeconds}s

🔒 Write tools are disabled (read-only mode)

💡 Use /execute or Shift+E to exit plan mode`;
  }

  if (state.mode === 'execute') {
    return `📊 Plan Mode Status

Mode: ⚡ Executing
Last Plan: ${state.currentPlan}
History: ${state.history.length} plans

🔓 Write tools are enabled (execution mode)

💡 Use /plan to enter plan mode again`;
  }

  return '❌ Unknown mode';
}

/**
 * 生成计划模板
 */
function generatePlanTemplate(task: string, planId: string): string {
  return `# Plan: ${task}

**ID**: ${planId}
**Status**: 📝 draft
**Created**: ${new Date().toISOString()}
**Updated**: ${new Date().toISOString()}

## Goal

${task}

## Context

[Add background information, constraints, and assumptions here]

## Tasks

### Task 1: [Task Title]

**ID**: task-001
**Status**: ⏳ pending
**Dependencies**: none

[Task description]

### Task 2: [Task Title]

**ID**: task-002
**Status**: ⏳ pending
**Dependencies**: task-001

[Task description]

## Progress

- Total tasks: 2
- Completed: 0
- In progress: 0
- Pending: 2

## Notes

[Add any additional notes here]

---

*Generated by Pi Plan Mode Extension*
`;
}
```

### events.ts（事件处理）

```typescript
/**
 * 事件处理模块
 */

import { ExtensionAPI } from '@pi/extension-api';

/**
 * 注册事件处理器
 */
export function registerEventHandlers(api: ExtensionAPI): void {
  // 监听进入规划模式事件
  api.on('plan:entered', (data: any) => {
    console.log(`📋 Entered plan mode: ${data.task}`);
    console.log(`   Plan ID: ${data.planId}`);
    console.log(`   Plan file: ${data.planFile}`);

    // 可以在这里添加自定义逻辑
    // 例如：记录日志、发送通知等
  });

  // 监听退出规划模式事件
  api.on('plan:exited', (data: any) => {
    const durationSeconds = Math.round(data.duration / 1000);
    console.log(`⚡ Exited plan mode: ${data.planId}`);
    console.log(`   Duration: ${durationSeconds}s`);

    // 可以在这里添加自定义逻辑
    // 例如：记录统计、清理临时文件等
  });

  // 监听状态变化事件
  api.on('state:changed', (oldState: any, newState: any) => {
    if (oldState.mode !== newState.mode) {
      console.log(`🔄 Mode changed: ${oldState.mode} → ${newState.mode}`);
    }
  });

  console.log('✅ Registered event handlers');
}
```

---

## 使用方法

### 1. 安装扩展

```bash
# 创建扩展目录
mkdir -p ~/.pi/extensions/plan-mode

# 复制文件
cp index.ts ~/.pi/extensions/plan-mode/
cp commands.ts ~/.pi/extensions/plan-mode/
cp state.ts ~/.pi/extensions/plan-mode/
cp events.ts ~/.pi/extensions/plan-mode/
cp package.json ~/.pi/extensions/plan-mode/

# 重启 Pi 或重新加载扩展
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

# 查看计划详情
> /plan-show plan-1234567890

# 退出规划模式
> /execute
```

### 3. 使用快捷键

- **Shift+P**: 快速进入规划模式
- **Shift+E**: 快速退出规划模式
- **Ctrl+Shift+L**: 列出所有计划

---

## 测试代码

```typescript
/**
 * 测试扩展命令实现
 */

import { ExtensionAPI } from '@pi/extension-api';

async function testPlanModeExtension(api: ExtensionAPI) {
  console.log('=== Testing Plan Mode Extension ===\n');

  // Test 1: 进入规划模式
  console.log('Test 1: Enter plan mode');
  await api.executeCommand('plan', ['Test task']);
  console.log('✅ Entered plan mode\n');

  // Test 2: 检查状态
  console.log('Test 2: Check status');
  await api.executeCommand('plan-status');
  console.log('✅ Status checked\n');

  // Test 3: 列出计划
  console.log('Test 3: List plans');
  await api.executeCommand('plan-list');
  console.log('✅ Plans listed\n');

  // Test 4: 退出规划模式
  console.log('Test 4: Exit plan mode');
  await api.executeCommand('execute');
  console.log('✅ Exited plan mode\n');

  // Test 5: 再次检查状态
  console.log('Test 5: Check status again');
  await api.executeCommand('plan-status');
  console.log('✅ Status checked\n');

  console.log('=== All tests passed ===');
}

export { testPlanModeExtension };
```

---

## 扩展功能

### 1. 计划审批流程

```typescript
// 添加审批命令
api.registerCommand({
  name: 'plan-approve',
  description: 'Approve the current plan',
  handler: async () => {
    const state = getCurrentState(api);

    if (!state.currentPlan) {
      return '❌ No active plan to approve';
    }

    // 更新计划状态
    const planFile = state.planFile!;
    const content = await api.tools.read(planFile);
    const updatedContent = content.replace(
      /\*\*Status\*\*: 📝 draft/,
      '**Status**: ✅ approved'
    );
    await api.tools.write(planFile, updatedContent);

    return `✅ Plan ${state.currentPlan} approved`;
  }
});
```

### 2. 计划模板系统

```typescript
// 添加模板命令
api.registerCommand({
  name: 'plan-from-template',
  description: 'Create plan from template',
  args: [
    {
      name: 'template',
      description: 'Template name',
      required: true
    },
    {
      name: 'task',
      description: 'Task description',
      required: true
    }
  ],
  handler: async (args: string[]) => {
    const templateName = args[0];
    const task = args.slice(1).join(' ');

    // 读取模板
    const templateFile = `.pi/templates/${templateName}.md`;
    const template = await api.tools.read(templateFile);

    // 替换占位符
    const planId = `plan-${Date.now()}`;
    const planContent = template
      .replace(/{{TASK}}/g, task)
      .replace(/{{PLAN_ID}}/g, planId)
      .replace(/{{DATE}}/g, new Date().toISOString());

    // 创建计划
    const planFile = `.pi/plans/${planId}.md`;
    await api.tools.write(planFile, planContent);

    return `✅ Created plan from template: ${planId}`;
  }
});
```

### 3. 计划统计

```typescript
// 添加统计命令
api.registerCommand({
  name: 'plan-stats',
  description: 'Show plan statistics',
  handler: async () => {
    const plansDir = '.pi/plans';
    const files = await api.tools.readdir(plansDir);
    const planFiles = files.filter((f: string) => f.endsWith('.md'));

    const stats = {
      total: planFiles.length,
      draft: 0,
      approved: 0,
      executing: 0,
      completed: 0
    };

    for (const file of planFiles) {
      const content = await api.tools.read(`${plansDir}/${file}`);
      if (content.includes('📝 draft')) stats.draft++;
      if (content.includes('✅ approved')) stats.approved++;
      if (content.includes('🔄 executing')) stats.executing++;
      if (content.includes('✔️ completed')) stats.completed++;
    }

    return `📊 Plan Statistics

Total: ${stats.total}
Draft: ${stats.draft}
Approved: ${stats.approved}
Executing: ${stats.executing}
Completed: ${stats.completed}`;
  }
});
```

---

## 优势与劣势

### 优势

1. **最佳用户体验**：集成的 /plan 和 /execute 命令
2. **快捷键支持**：Shift+P 和 Shift+E 快速切换
3. **状态持久化**：自动保存和恢复状态
4. **事件通知**：可扩展的事件系统
5. **完整功能**：列表、查看、状态检查等

### 劣势

1. **需要创建扩展**：比文件式规划复杂
2. **需要理解 API**：需要学习 Extension API
3. **维护成本**：需要维护扩展代码

---

## 最佳实践

### 1. 错误处理

```typescript
async function handlePlanCommand(api: ExtensionAPI, args: string[]): Promise<string> {
  try {
    // 命令逻辑
    const task = args.join(' ');
    // ...
    return '✅ Success';
  } catch (error) {
    console.error('Failed to execute /plan command:', error);
    return `❌ Failed: ${error.message}`;
  }
}
```

### 2. 状态验证

```typescript
async function handleExecuteCommand(api: ExtensionAPI): Promise<string> {
  const state = getCurrentState(api);

  // 验证状态
  if (!isInPlanMode(api)) {
    return '⚠️  Not in plan mode';
  }

  if (!state.currentPlan) {
    return '❌ No active plan';
  }

  // 执行逻辑
  // ...
}
```

### 3. 日志记录

```typescript
api.on('plan:entered', (data: any) => {
  // 记录到文件
  const logFile = '.pi/logs/plan-mode.log';
  const logEntry = `[${new Date().toISOString()}] Entered plan mode: ${data.task}\n`;
  api.tools.append(logFile, logEntry);
});
```

---

## 参考资源

### 官方资源

- [Pi-mono GitHub](https://github.com/badlogic/pi-mono)
- [Extension API Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md)
- [Plan Mode Example](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent/examples/extensions/plan-mode)

### 研究资料

- `temp/03_grok_pi_mono_extensions.md` - Pi-mono 扩展 API
- `temp/04_grok_pi_mono_examples.md` - Pi-mono 扩展示例

---

## 下一步

- **07_实战代码_03_Session集成执行.md**：学习如何使用 Session 管理复杂状态
- **03_核心概念_02_Extension_API集成.md**：深入理解 Extension API
