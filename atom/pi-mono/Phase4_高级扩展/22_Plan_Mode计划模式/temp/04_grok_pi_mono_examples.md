# Pi-mono Extension Examples & Plan Mode (2025-2026)

**Fetched:** 2026-02-21

## Official Pi-mono Extension Examples

### 1. Pi-mono Main Repository
**Source:** https://github.com/badlogic/pi-mono

Pi-mono 是 AI 代理工具包核心仓库，包含 coding-agent 及其扩展系统示例，支持 TypeScript 自定义扩展和 plan mode 示例。

### 2. Extension Documentation
**Source:** https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md

官方扩展文档，详细说明如何创建扩展，包括注册工具、命令和自定义渲染，包含 plan mode 等高级示例链接。

### 3. Extensions Examples Directory
**Source:** https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent/examples/extensions

官方提供 50+ 个扩展示例，包括 plan-mode、subagent、todo、tools、commands 等，适合学习和直接使用。

**Available Examples:**
- `plan-mode/`: Claude Code style read-only exploration
- `subagent/`: Sub-agent implementation
- `todo/`: Task management
- `tools/`: Custom tool examples
- `commands/`: Custom command examples
- `shortcuts/`: Keyboard shortcut examples
- `ui/`: Custom UI rendering

### 4. Plan Mode Example Extension
**Source:** https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent/examples/extensions/plan-mode

Claude Code 风格的只读探索模式实现，通过 /plan 命令切换，支持安全代码分析而不修改文件，2026 年更新。

**Key Features:**
- `/plan` command to enter plan mode
- `Shift+P` keyboard shortcut
- Read-only exploration
- Disables write/edit tools
- `/execute` or `Shift+E` to exit

**Implementation Structure:**
```
plan-mode/
├── index.ts          # Main extension entry
├── commands.ts       # Command handlers
├── state.ts          # State management
└── ui.ts            # UI components
```

### 5. Pi-coding-agent README
**Source:** https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/README.md

Pi-coding-agent 核心说明，介绍无内置 plan mode 但可通过扩展实现，列出示例扩展如 sub-agents 和 plan mode。

**Key Points:**
- No built-in plan mode by design
- Extensions provide flexibility
- Example implementations available
- Community contributions welcome

### 6. CHANGELOG Updates
**Source:** https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/CHANGELOG.md

更新日志显示 plan-mode 示例增强，包括 /plan 命令和 Shift+P 快捷键，适用于 2026 年版本。

**Recent Updates (2026):**
- Enhanced plan-mode example with better state management
- Added keyboard shortcuts for quick mode switching
- Improved extension reload support
- Better error handling in extensions
- Session integration improvements

### 7. Pi Official Website
**Source:** https://shittycodingagent.ai/

Pi 官方站点，展示扩展系统和示例，包括 plan mode、subagent、sandbox 等，强调自定义工作流。

**Community Packages:**
- `@juanibiapina/pi-plan`: Community plan mode extension
- Various plan mode implementations
- Extension templates and starters

### 8. Pi: The Minimal Agent (2026)
**Source:** https://lucumr.pocoo.org/2026/1/31/pi

2026 年文章讨论 Pi 代理及其扩展系统，提及 plan mode 替代方案和社区扩展示例。

**Key Insights:**
- Pi's philosophy: minimal core, extensible periphery
- Plan mode as extension pattern
- Community-driven feature development
- Observability through files

### 9. Pi-mono: The Minimalist AI Coding Assistant (2026)
**Source:** https://medium.com/@ai-engineering-trend/pi-mono-the-minimalist-ai-coding-assistant-behind-openclaw-bd3ccc0a1b04

2026 年 Medium 文章分析 pi-mono 极简设计和扩展系统，可自行构建 plan mode 等功能。

**Design Philosophy:**
- Extreme minimalism in core
- Extension-first architecture
- No forced workflows
- User control paramount

### 10. Building Custom Agent Frameworks (2026)
**Source:** https://nader.substack.com/p/how-to-build-a-custom-agent-framework

详细指南介绍 pi-mono 包结构和扩展使用，包括 plan mode 等功能通过扩展实现的方法。

**Implementation Guide:**
- Package structure overview
- Extension development workflow
- Plan mode implementation patterns
- Testing and debugging extensions

## Extension Development Patterns

### Pattern 1: Simple Command Extension
```typescript
// Minimal plan mode command
export default function(api: ExtensionAPI) {
  api.registerCommand({
    name: 'plan',
    description: 'Enter plan mode',
    handler: async () => {
      await api.setState({ mode: 'plan' });
      api.disableTools(['write', 'edit']);
      return 'Entered plan mode';
    }
  });
}
```

### Pattern 2: Stateful Extension
```typescript
// Plan mode with state persistence
export default function(api: ExtensionAPI) {
  let planState = { active: false, planFile: null };

  api.registerCommand({
    name: 'plan',
    handler: async () => {
      planState.active = true;
      planState.planFile = '.pi/plan.md';
      await api.setState(planState);
      api.disableTools(['write', 'edit']);
    }
  });

  api.registerCommand({
    name: 'execute',
    handler: async () => {
      planState.active = false;
      await api.setState(planState);
      api.enableTools(['write', 'edit']);
    }
  });
}
```

### Pattern 3: UI-Enhanced Extension
```typescript
// Plan mode with custom UI
export default function(api: ExtensionAPI) {
  api.registerCommand({
    name: 'plan',
    handler: async () => {
      await api.setState({ mode: 'plan' });
      api.renderCustomUI(PlanModeIndicator);
      api.disableTools(['write', 'edit']);
    }
  });
}
```

## Best Practices from Examples

1. **Clear Entry/Exit**: Always provide clear commands to enter and exit plan mode
2. **State Management**: Persist plan state to session for recovery
3. **Tool Control**: Explicitly disable/enable tools based on mode
4. **UI Feedback**: Provide visual indicators for current mode
5. **Keyboard Shortcuts**: Add shortcuts for quick mode switching
6. **Error Handling**: Handle edge cases gracefully
7. **Documentation**: Document extension behavior clearly

## Relevance to Documentation

These examples demonstrate:
1. **Three implementation approaches**: Command-based, file-based, session-integrated
2. **Best practices**: From official pi-mono examples
3. **Real-world patterns**: Used in production extensions
4. **Community adoption**: Multiple implementations exist
5. **Evolution**: 2026 updates show active development
