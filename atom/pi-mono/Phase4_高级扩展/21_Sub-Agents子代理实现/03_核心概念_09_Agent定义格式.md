# 核心概念 09：Agent 定义格式

## 1. 基本格式

### 1.1 Markdown + YAML Frontmatter

```markdown
---
name: agent-name
description: Agent description
tools: tool1, tool2, tool3
model: model-name
---

System prompt goes here.
```

### 1.2 必需字段

| 字段 | 必需 | 类型 | 说明 |
|------|------|------|------|
| `name` | ✅ | string | Agent 唯一标识 |
| `description` | ✅ | string | Agent 功能描述 |
| `tools` | ❌ | string | 逗号分隔的工具列表 |
| `model` | ❌ | string | 使用的模型 |

---

## 2. 示例 Agents

### 2.1 Scout Agent

```markdown
---
name: scout
description: Fast codebase reconnaissance
tools: read, grep, find, ls, bash
model: claude-haiku-4-5
---

You are a fast reconnaissance agent. Your job is to:
1. Quickly locate relevant files and code
2. Return compressed, actionable context
3. Use efficient search strategies

Keep your output concise and structured.
```

### 2.2 Planner Agent

```markdown
---
name: planner
description: Creates implementation plans
tools: read, grep, find, ls
model: claude-sonnet-4
---

You are a planning agent. Your job is to:
1. Analyze the codebase context
2. Create detailed implementation plans
3. Break down complex tasks into steps

Output format:
## Plan
1. Step 1: ...
2. Step 2: ...
```

### 2.3 Worker Agent

```markdown
---
name: worker
description: General-purpose implementation
---

You are a general-purpose worker agent.
You have access to all default tools.
Implement changes as requested.
```

---

## 3. 工具配置

### 3.1 可用工具

```typescript
// 常用工具
read, write, edit    // 文件操作
grep, find, ls       // 搜索
bash                 // 命令执行
```

### 3.2 工具集设计

```markdown
<!-- 只读 Agent -->
tools: read, grep, find, ls

<!-- 完整权限 Agent -->
tools: (all default)  // 或不指定 tools 字段
```

---

## 4. 模型选择

### 4.1 模型对比

| 模型 | 速度 | 成本 | 适用场景 |
|------|------|------|---------|
| `claude-haiku-4-5` | 快 | 低 | 快速侦查 |
| `claude-sonnet-4` | 中 | 中 | 计划、审查 |
| `claude-opus-4` | 慢 | 高 | 复杂推理 |

### 4.2 模型配置

```markdown
<!-- 快速任务 -->
model: claude-haiku-4-5

<!-- 复杂任务 -->
model: claude-sonnet-4

<!-- 不指定：使用默认模型 -->
```

---

## 5. 系统提示设计

### 5.1 清晰的职责

```markdown
You are a [role] agent. Your job is to:
1. [Primary responsibility]
2. [Secondary responsibility]
3. [Tertiary responsibility]

Focus on: [What to focus on]
Do NOT: [What to avoid]
```

### 5.2 输出格式

```markdown
Output format:
- [Format requirement 1]
- [Format requirement 2]

Example:
[Example output]
```

---

## 6. 文件位置

### 6.1 用户级别

```bash
~/.pi/agent/agents/
├── scout.md
├── planner.md
├── reviewer.md
└── worker.md
```

### 6.2 项目级别

```bash
.pi/agents/
├── custom-scout.md
└── project-specific.md
```

---

## 7. 最佳实践

### 7.1 命名规范

```markdown
✅ scout, planner, worker
❌ agent1, my-agent
```

### 7.2 描述清晰

```markdown
✅ description: Fast codebase reconnaissance
❌ description: Helper
```

### 7.3 最小权限

```markdown
✅ tools: read, grep, find
❌ tools: read, write, edit, bash  # 对于 scout
```

---

## 8. 总结

### 核心要点

1. **Markdown + YAML**：简单格式
2. **必需字段**：name, description
3. **工具配置**：最小权限原则
4. **模型选择**：根据任务复杂度
5. **系统提示**：清晰的职责和格式

---

**参考资源**：
- [Pi-mono Agent 示例](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent/examples/extensions/subagent/agents)
