# Pi-mono Extension API & Plan Mode Implementation (2025-2026)

**Fetched:** 2026-02-21

## Official Pi-mono Resources

### 1. Pi-mono Main Repository
**Source:** https://github.com/badlogic/pi-mono

Pi-mono 是 AI 代理工具包核心仓库，包含 coding-agent 及其扩展系统示例，支持 TypeScript 自定义扩展和 plan mode 示例。

**Key Components:**
- `packages/coding-agent/`: Core agent implementation
- `packages/coding-agent/examples/extensions/`: 50+ extension examples
- `packages/coding-agent/docs/extensions.md`: Official extension documentation

### 2. Extension API Documentation
**Source:** https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md

官方扩展文档，详细说明如何创建扩展，包括注册工具、命令和自定义渲染，包含 plan mode 等高级示例链接。

**Extension API Capabilities:**
- Register custom commands
- Register custom tools
- Add keyboard shortcuts
- Custom UI rendering
- Event handlers
- State persistence

### 3. Plan Mode Example Extension
**Source:** https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent/examples/extensions/plan-mode

Claude Code 风格的只读探索模式实现，通过 /plan 命令切换，支持安全代码分析而不修改文件，2026 年更新。

**Features:**
- `/plan` command to enter plan mode
- `Shift+P` keyboard shortcut
- Read-only exploration
- Safe code analysis without modifications
- Exit plan mode with `/execute` or `Shift+E`

**Implementation Highlights:**
```typescript
// Register plan mode command
api.registerCommand({
  name: 'plan',
  description: 'Enter plan mode for read-only exploration',
  handler: async () => {
    // Switch to plan mode
    await api.setState({ mode: 'plan' });
    // Disable write tools
    api.disableTools(['write', 'edit']);
  }
});
```

### 4. Pi-mono README
**Source:** https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/README.md

Pi-coding-agent 核心说明，介绍无内置 plan mode 但可通过扩展实现，列出示例扩展如 sub-agents 和 plan mode。

**Philosophy:**
- **No built-in plan mode**: Avoid black-box orchestration
- **Extension-based**: Build what you need
- **Observability**: File-based state
- **Composability**: Mix and match extensions

### 5. CHANGELOG - Plan Mode Updates
**Source:** https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/CHANGELOG.md

更新日志显示 plan-mode 示例增强，包括 /plan 命令和 Shift+P 快捷键，适用于 2026 年版本。

**Recent Updates (2026):**
- Enhanced plan-mode example
- Added keyboard shortcuts
- Improved state management
- Better error handling
- Extension reload support

### 6. Pi Official Website
**Source:** https://shittycodingagent.ai/

Pi 官方站点，展示扩展系统和示例，包括 plan mode、subagent、sandbox 等，强调自定义工作流。

**Community Extensions:**
- `@juanibiapina/pi-plan`: Plan mode extension
- Various plan mode implementations
- Community-contributed patterns

### 7. Pi: The Minimal Agent (Armin Ronacher, 2026)
**Source:** https://lucumr.pocoo.org/2026/1/31/pi

2026 年文章讨论 Pi 代理及其扩展系统，提及 plan mode 替代方案和社区扩展示例。

**Key Insights:**
- Pi's minimalist philosophy
- Extension system design
- Plan mode alternatives
- Community ecosystem

### 8. Pi-mono: The Minimalist AI Coding Assistant (Medium, 2026)
**Source:** https://medium.com/@ai-engineering-trend/pi-mono-the-minimalist-ai-coding-assistant-behind-openclaw-bd3ccc0a1b04

2026 年 Medium 文章分析 pi-mono 极简设计和扩展系统，可自行构建 plan mode 等功能。

**Design Principles:**
- Extreme minimalism
- No forced features
- User-controlled workflows
- Extension-first architecture

### 9. Building Custom Agent Frameworks (Nader, 2026)
**Source:** https://nader.substack.com/p/how-to-build-a-custom-agent-framework

详细指南介绍 pi-mono 包结构和扩展使用，包括 plan mode 等功能通过扩展实现的方法。

**Implementation Guide:**
- Package structure
- Extension development
- Plan mode implementation patterns
- Best practices

## Extension API Structure

### Core API Methods

```typescript
interface ExtensionAPI {
  // Command registration
  registerCommand(config: CommandConfig): void;

  // Tool registration
  registerTool(config: ToolConfig): void;

  // Keyboard shortcuts
  registerShortcut(config: ShortcutConfig): void;

  // State management
  setState(state: Record<string, any>): Promise<void>;
  getState(): Record<string, any>;

  // Tool control
  enableTools(tools: string[]): void;
  disableTools(tools: string[]): void;

  // Event handlers
  on(event: string, handler: Function): void;

  // UI rendering
  renderCustomUI(component: React.Component): void;
}
```

### Plan Mode Implementation Patterns

**Pattern 1: Simple Command-Based**
```typescript
api.registerCommand({
  name: 'plan',
  handler: async () => {
    await api.setState({ mode: 'plan' });
    api.disableTools(['write', 'edit']);
  }
});
```

**Pattern 2: File-Based Planning**
```typescript
api.registerCommand({
  name: 'plan',
  handler: async () => {
    const planFile = '.pi/plan.md';
    await api.tools.write(planFile, '# Plan\n\n');
    await api.setState({ planFile });
  }
});
```

**Pattern 3: Session-Integrated**
```typescript
api.registerCommand({
  name: 'plan',
  handler: async () => {
    const session = await api.createSession({
      label: 'planning',
      readOnly: true
    });
    await api.switchSession(session.id);
  }
});
```

## Why Pi-mono Doesn't Have Built-in Plan Mode

### From Mario's Blog (2025)
**Source:** https://mariozechner.at/posts/2025-11-30-pi-coding-agent

**Key Reasons:**
1. **No black-box orchestration**: Users should see and control planning
2. **Observability**: File-based plans are transparent
3. **Flexibility**: Different users need different planning approaches
4. **Composability**: Extensions can combine planning patterns
5. **Simplicity**: Core agent stays minimal

**Quote:**
> "I don't want to force a planning mode on users. Instead, I provide the primitives to build any planning workflow you need. Write plans to files, use extensions, or integrate with CLI tools - the choice is yours."

## Community Implementations

### oh-my-pi Fork
- Added `/plan` command
- Integrated planner agents
- File-based plan storage
- See: `04_grok_oh_my_pi.md` for details

### @juanibiapina/pi-plan
- NPM package for plan mode
- Read-only exploration
- Analysis mode support

## Best Practices for Plan Mode Extensions

1. **Use file-based plans**: Maximum observability
2. **Disable write tools**: Prevent accidental modifications
3. **Clear entry/exit**: `/plan` and `/execute` commands
4. **State persistence**: Save plan state to session
5. **Keyboard shortcuts**: Quick mode switching
6. **Custom UI**: Visual indicators for plan mode
7. **Event handlers**: React to mode changes

## Relevance to Documentation

This research shows that:
1. Plan Mode is **intentionally not built-in**
2. Extensions provide **three implementation approaches**
3. Pi-mono's philosophy emphasizes **user control and observability**
4. Community has created **multiple plan mode implementations**
5. Official example demonstrates **best practices**
