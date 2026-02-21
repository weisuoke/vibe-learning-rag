# Oh-my-pi Plan Command Implementation (2025-2026)

**Fetched:** 2026-02-21

## Oh-my-pi Overview

### 1. Oh-my-pi Main Repository
**Source:** https://github.com/can1357/oh-my-pi

Oh-my-pi 是终端AI编码代理工具，支持/plan命令切换计划模式，用于在实施前生成架构规划，包含完整源代码实现。

**Key Features:**
- `/plan` slash command for plan mode
- CLI `--plan` parameter
- Plan mode switching
- Architecture planning before implementation
- Fork of pi-mono with additional features

### 2. Oh-my-pi README
**Source:** https://github.com/can1357/oh-my-pi/blob/main/README.md

项目自述文件，详细说明/plan slash命令功能、plan模式切换及CLI参数如--plan的使用方式，包含安装和命令实现细节。

**Plan Mode Features:**
- **`/plan` command**: Enter plan mode for architecture planning
- **`--plan` CLI flag**: Start session in plan mode
- **Read-only exploration**: Analyze code without modifications
- **Architecture generation**: Create implementation plans
- **Exit with `/execute`**: Switch back to execution mode

**Usage:**
```bash
# Start in plan mode
omp --plan "Design a new authentication system"

# Or use /plan command in session
> /plan
```

### 3. Oh-my-pi AGENTS.md
**Source:** https://github.com/can1357/oh-my-pi/blob/main/AGENTS.md

代理实现文档，聚焦packages/coding-agent包，解释plan相关代理逻辑和整体架构源代码结构。

**Agent Architecture:**
- **Planner Agent**: Dedicated planning agent
- **Architect Agent**: Architecture design agent
- **Executor Agent**: Implementation agent
- **Coordinator**: Orchestrates between agents

**Plan Mode Logic:**
```
User Request → Planner Agent → Architecture Plan → User Approval → Executor Agent
```

### 4. Oh-my-pi CHANGELOG
**Source:** https://github.com/can1357/oh-my-pi/blob/main/packages/coding-agent/CHANGELOG.md

更新日志，记录plan模式、plan agent、architect-plan等功能的添加、移除与实现变更历史。

**Key Changes:**
- **Added**: `/plan` command implementation
- **Added**: Planner agent with architecture focus
- **Added**: `--plan` CLI parameter
- **Modified**: Plan mode state management
- **Removed**: Some experimental planning features
- **Enhanced**: Plan-to-execution transition

### 5. Oh-my-pi Custom Slash Commands
**Source:** https://github.com/can1357/oh-my-pi/blob/main/README.md

文档描述/plan作为slash命令的实现，支持TypeScript自定义命令，位于~/.omp/agent/commands/路径下。

**Custom Command Structure:**
```
~/.omp/agent/commands/
├── plan.ts          # Plan mode command
├── execute.ts       # Execute mode command
└── ...
```

**Implementation Pattern:**
```typescript
// ~/.omp/agent/commands/plan.ts
export default {
  name: 'plan',
  description: 'Enter plan mode for architecture planning',
  handler: async (api, args) => {
    // Switch to plan mode
    await api.setState({ mode: 'plan' });

    // Load planner agent
    await api.loadAgent('planner');

    // Disable write tools
    api.disableTools(['write', 'edit', 'delete']);

    return 'Entered plan mode. I will help you design the architecture.';
  }
};
```

### 6. @oh-my-pi/subagents Package
**Source:** https://www.npmjs.com/package/@oh-my-pi/subagents

包含planner.md和architect-plan.md等规划代理实现，用于任务分解和计划生成的核心源代码模块。

**Subagent Types:**
- **planner.md**: General planning agent
- **architect-plan.md**: Architecture-focused planning
- **task-decomposer.md**: Task breakdown agent
- **reviewer.md**: Plan review agent

**Planner Agent Prompt Structure:**
```markdown
# Planner Agent

You are a planning specialist. Your role is to:
1. Analyze user requirements
2. Break down into subtasks
3. Identify dependencies
4. Create implementation plan
5. Document assumptions

## Planning Process
1. Understand the goal
2. Explore codebase
3. Design architecture
4. Create step-by-step plan
5. Present for approval

## Output Format
- Clear task breakdown
- Dependency graph
- Implementation order
- Risk assessment
```

### 7. Oh-my-pi Plan Mode Extension Example
**Source:** https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent/examples/extensions/plan-mode

原pi-mono仓库中plan mode扩展示例代码，oh-my-pi fork继承，支持计划模式下的无编辑思考实现参考。

**Inherited Features:**
- Read-only exploration
- Tool disabling
- State management
- Mode switching

### 8. Oh-my-pi CLI Reference
**Source:** https://github.com/can1357/oh-my-pi/blob/main/README.md

CLI参数说明，包括--plan指定模型用于计划模式，体现plan命令的配置与执行实现细节。

**CLI Parameters:**
```bash
# Start in plan mode
omp --plan "Task description"

# Specify model for planning
omp --plan --model gpt-4 "Complex task"

# Plan mode with specific agent
omp --plan --agent architect "Design system"
```

## Implementation Comparison: Pi-mono vs. Oh-my-pi

| Aspect | Pi-mono | Oh-my-pi |
|--------|---------|----------|
| **Plan Mode** | Extension example | Built-in command |
| **Approach** | User implements | Pre-implemented |
| **Philosophy** | Minimal core | Feature-rich |
| **Flexibility** | Maximum | Opinionated |
| **Agents** | Optional | Integrated |
| **CLI Support** | Extension-based | Native `--plan` |

## Oh-my-pi Plan Mode Workflow

### 1. Enter Plan Mode
```bash
> /plan
Entered plan mode. I will help you design the architecture.
```

### 2. Planning Phase
- Agent explores codebase (read-only)
- Analyzes requirements
- Designs architecture
- Creates implementation plan
- Presents plan for approval

### 3. Plan Output
```markdown
# Implementation Plan

## Goal
[Clear statement of objective]

## Architecture
[High-level design]

## Tasks
1. Task 1: [Description]
   - Subtask 1.1
   - Subtask 1.2
2. Task 2: [Description]
   ...

## Dependencies
[Dependency graph]

## Risks
[Potential issues]
```

### 4. Approval & Execution
```bash
> /execute
Exiting plan mode. Ready to implement the plan.
```

## Key Differences from Pi-mono

### Oh-my-pi Additions:
1. **Built-in `/plan` command**: No extension needed
2. **Dedicated planner agents**: Specialized planning prompts
3. **CLI `--plan` flag**: Start in plan mode directly
4. **Integrated workflow**: Seamless plan-to-execution
5. **Opinionated structure**: Pre-defined planning process

### Pi-mono Philosophy:
1. **Extension-based**: User chooses implementation
2. **Minimal core**: No forced features
3. **Maximum flexibility**: Any planning pattern
4. **Observability**: File-based plans
5. **User control**: Explicit mode switching

## Relevance to Documentation

Oh-my-pi demonstrates:
1. **One implementation approach**: Built-in command pattern
2. **Agent integration**: Dedicated planner agents
3. **CLI support**: Native plan mode flag
4. **Opinionated workflow**: Pre-defined planning process
5. **Community fork**: Alternative to pi-mono's minimalism

**Key Insight:** Oh-my-pi shows what a "batteries-included" plan mode looks like, while pi-mono provides the primitives to build any planning approach you need.
