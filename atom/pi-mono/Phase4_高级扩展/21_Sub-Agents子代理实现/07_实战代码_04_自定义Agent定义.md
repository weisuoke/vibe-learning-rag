# 自定义Agent定义 - 实战代码

> **知识点**: Sub-Agents 子代理实现 - 自定义 Agent 定义与配置
> **难度**: ⭐⭐⭐⭐
> **前置知识**: Agent 发现与配置、Markdown + YAML frontmatter

---

## 1. 场景描述

自定义 Agent 定义是 Sub-Agents 系统的核心能力,允许开发者创建专业化的代理来处理特定任务。每个 Agent 通过 Markdown 文件 + YAML frontmatter 定义其能力、工具集和行为模式。

**典型应用场景**:
- 创建专业化代理(scout、planner、worker、reviewer)
- 定义领域特定代理(security-auditor、performance-optimizer)
- 配置代理工具集和模型选择
- 实现代理角色分工和协作

根据 [Reddit 社区的实际案例](https://www.reddit.com/r/LangChain/comments/1llw60o/most_people_think_one_ai_agent_can_handle),将 1 个通用代理拆分为 13 个专业化代理后,准确率显著提升、可调试性更好、整体系统更健壮。[Stack-AI 2026 年指南](https://www.stack-ai.com/blog/the-2026-guide-to-agentic-workflow-architectures)强调专业化代理是生产环境的最佳实践。

---

## 2. 核心概念

### 2.1 Agent 定义格式

Pi-mono 使用 **Markdown + YAML frontmatter** 格式定义 Agent:

```markdown
---
name: agent-name
description: Agent description
tools: tool1, tool2, tool3
model: claude-haiku-4-5
---

System prompt content here...
```

这种格式与 [ragapp/agentfiles](https://github.com/ragapp/agentfiles) 的 YAML 配置理念一致,是 2025-2026 年的行业标准。

### 2.2 Agent 字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `name` | string | ✅ | Agent 唯一标识符 |
| `description` | string | ✅ | Agent 功能描述 |
| `tools` | string | ✅ | 逗号分隔的工具列表 |
| `model` | string | ❌ | 模型选择(默认继承) |

### 2.3 Agent 发现路径

- **User-level**: `~/.pi/agent/agents/*.md`
- **Project-level**: `.pi/agents/*.md`

根据 [2026 年最佳实践](https://www.stack-ai.com/blog/the-2026-guide-to-agentic-workflow-architectures),应该明确定义代理范围和权限边界。

---

## 3. 完整代码示例

### 3.1 Scout Agent 定义

```markdown
---
name: scout
description: Fast codebase reconnaissance agent for quick exploration
tools: read, grep, glob, ls, bash
model: claude-haiku-4-5
---

# Scout Agent

You are a fast reconnaissance agent specialized in quickly exploring codebases.

## Your Mission

Rapidly gather information about:
- File structure and organization
- Key components and their locations
- Technology stack and dependencies
- Code patterns and conventions

## Guidelines

1. **Speed over depth**: Quick scans, not deep analysis
2. **Pattern recognition**: Identify common structures
3. **Concise reporting**: Bullet points, not essays
4. **Tool efficiency**: Use grep for patterns, glob for files

## Output Format

```
## Findings
- [Key discovery 1]
- [Key discovery 2]

## Locations
- Component X: path/to/file.ts:123
- Feature Y: path/to/feature/

## Recommendations
- [Next step 1]
- [Next step 2]
```

## Constraints

- Maximum 2 minutes execution time
- Focus on high-value information
- Avoid reading large files completely
- Use grep with context (-C 3) for code snippets
```

### 3.2 Planner Agent 定义

```markdown
---
name: planner
description: Strategic planning agent for implementation roadmaps
tools: read, grep, glob
model: claude-sonnet-4-5
---

# Planner Agent

You are a strategic planning agent specialized in creating detailed implementation plans.

## Your Mission

Create comprehensive plans that include:
- Step-by-step implementation sequence
- File modifications required
- Potential risks and mitigations
- Testing strategy

## Guidelines

1. **Thorough analysis**: Read relevant files completely
2. **Risk assessment**: Identify breaking changes
3. **Dependency mapping**: Understand component relationships
4. **Validation strategy**: Define success criteria

## Output Format

```
## Implementation Plan

### Phase 1: Preparation
- [ ] Task 1: Description
- [ ] Task 2: Description

### Phase 2: Core Changes
- [ ] Task 3: Description (file: path/to/file.ts:123)
- [ ] Task 4: Description

### Phase 3: Testing & Validation
- [ ] Task 5: Description

## Risk Assessment
- Risk 1: Description → Mitigation
- Risk 2: Description → Mitigation

## Success Criteria
- [ ] Criterion 1
- [ ] Criterion 2
```

## Constraints

- Read all relevant files before planning
- Consider backward compatibility
- Estimate complexity (simple/medium/complex)
- Flag high-risk changes
```

### 3.3 Worker Agent 定义

```markdown
---
name: worker
description: Implementation agent for executing code changes
tools: read, write, edit, bash
model: claude-sonnet-4-5
---

# Worker Agent

You are an implementation agent specialized in executing code changes.

## Your Mission

Implement changes by:
- Writing new code files
- Editing existing code
- Running tests and validation
- Fixing issues that arise

## Guidelines

1. **Follow the plan**: Stick to provided specifications
2. **Test as you go**: Validate each change
3. **Clean code**: Follow project conventions
4. **Error handling**: Gracefully handle failures

## Output Format

```
## Changes Made

### File: path/to/file.ts
- Added function `foo()` at line 123
- Modified class `Bar` at line 456

### File: path/to/test.ts
- Added test cases for `foo()`

## Validation
✅ Tests passed
✅ Linting passed
✅ Type checking passed

## Issues Encountered
- Issue 1: Description → Resolution
```

## Constraints

- Always read files before editing
- Use Edit tool for modifications (not Write)
- Run tests after significant changes
- Report all errors immediately
```

### 3.4 Reviewer Agent 定义

```markdown
---
name: reviewer
description: Code review agent for quality assurance
tools: read, grep, bash
model: claude-sonnet-4-5
---

# Reviewer Agent

You are a code review agent specialized in quality assurance.

## Your Mission

Review code for:
- Correctness and logic errors
- Code quality and maintainability
- Security vulnerabilities
- Performance issues
- Test coverage

## Guidelines

1. **Thorough review**: Read all changed files
2. **Constructive feedback**: Explain issues clearly
3. **Prioritize issues**: Critical → Major → Minor
4. **Suggest fixes**: Provide concrete solutions

## Output Format

```
## Review Summary
- Files reviewed: 5
- Issues found: 3 critical, 2 major, 5 minor

## Critical Issues
1. **Security**: SQL injection vulnerability
   - Location: path/to/file.ts:123
   - Fix: Use parameterized queries

## Major Issues
1. **Performance**: N+1 query problem
   - Location: path/to/file.ts:456
   - Fix: Use batch loading

## Minor Issues
1. **Style**: Inconsistent naming
   - Location: path/to/file.ts:789
   - Fix: Rename to camelCase

## Approval Status
❌ Changes required before merge
```

## Constraints

- Review all modified files
- Run linting and type checking
- Check test coverage
- Verify security best practices
```

### 3.5 TypeScript Agent 创建工具

```typescript
import { ExtensionAPI } from "@pi-mono/extension-api";
import * as fs from "fs/promises";
import * as path from "path";
import { z } from "zod";

// Agent 定义 Schema
const AgentDefinitionSchema = z.object({
  name: z.string().regex(/^[a-z0-9-]+$/).describe("Agent name (lowercase, hyphens)"),
  description: z.string().min(10).max(200).describe("Agent description"),
  tools: z.array(z.string()).min(1).describe("List of tools"),
  model: z.enum(["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"]).optional(),
  systemPrompt: z.string().min(100).describe("System prompt content"),
  scope: z.enum(["user", "project"]).default("project"),
});

type AgentDefinition = z.infer<typeof AgentDefinitionSchema>;

/**
 * Agent 定义生成器
 *
 * 参考:
 * - https://github.com/ragapp/agentfiles (YAML agent definition)
 * - https://www.stack-ai.com/blog/the-2026-guide-to-agentic-workflow-architectures (2026 best practices)
 */
export class AgentDefinitionGenerator {
  private cwd: string;

  constructor(cwd: string) {
    this.cwd = cwd;
  }

  /**
   * 创建 Agent 定义文件
   */
  async createAgent(definition: AgentDefinition): Promise<string> {
    const agentDir = definition.scope === "user"
      ? path.join(process.env.HOME || "~", ".pi", "agent", "agents")
      : path.join(this.cwd, ".pi", "agents");

    // 确保目录存在
    await fs.mkdir(agentDir, { recursive: true });

    const agentPath = path.join(agentDir, `${definition.name}.md`);

    // 检查是否已存在
    try {
      await fs.access(agentPath);
      throw new Error(`Agent ${definition.name} already exists at ${agentPath}`);
    } catch (error: any) {
      if (error.code !== "ENOENT") throw error;
    }

    // 生成 Markdown 内容
    const content = this.generateMarkdown(definition);

    // 写入文件
    await fs.writeFile(agentPath, content, "utf-8");

    return agentPath;
  }

  /**
   * 生成 Markdown 内容
   */
  private generateMarkdown(definition: AgentDefinition): string {
    const frontmatter = [
      "---",
      `name: ${definition.name}`,
      `description: ${definition.description}`,
      `tools: ${definition.tools.join(", ")}`,
    ];

    if (definition.model) {
      frontmatter.push(`model: ${definition.model}`);
    }

    frontmatter.push("---", "");

    return frontmatter.join("\n") + definition.systemPrompt;
  }

  /**
   * 验证 Agent 定义
   */
  async validateAgent(agentPath: string): Promise<{
    valid: boolean;
    errors: string[];
  }> {
    const errors: string[] = [];

    try {
      const content = await fs.readFile(agentPath, "utf-8");

      // 检查 frontmatter
      if (!content.startsWith("---")) {
        errors.push("Missing YAML frontmatter");
      }

      // 提取 frontmatter
      const match = content.match(/^---\n([\s\S]*?)\n---/);
      if (!match) {
        errors.push("Invalid frontmatter format");
        return { valid: false, errors };
      }

      // 解析 YAML (简单实现)
      const frontmatter = match[1];
      const lines = frontmatter.split("\n");
      const fields: Record<string, string> = {};

      for (const line of lines) {
        const [key, ...valueParts] = line.split(":");
        if (key && valueParts.length > 0) {
          fields[key.trim()] = valueParts.join(":").trim();
        }
      }

      // 验证必需字段
      if (!fields.name) errors.push("Missing required field: name");
      if (!fields.description) errors.push("Missing required field: description");
      if (!fields.tools) errors.push("Missing required field: tools");

      // 验证 name 格式
      if (fields.name && !/^[a-z0-9-]+$/.test(fields.name)) {
        errors.push("Invalid name format (use lowercase and hyphens)");
      }

      // 检查 system prompt
      const systemPrompt = content.substring(match[0].length).trim();
      if (systemPrompt.length < 100) {
        errors.push("System prompt too short (minimum 100 characters)");
      }

    } catch (error: any) {
      errors.push(`Failed to read agent file: ${error.message}`);
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }

  /**
   * 列出所有可用 Agents
   */
  async listAgents(scope?: "user" | "project" | "both"): Promise<Array<{
    name: string;
    description: string;
    path: string;
    scope: "user" | "project";
  }>> {
    const agents: Array<{
      name: string;
      description: string;
      path: string;
      scope: "user" | "project";
    }> = [];

    const dirs: Array<{ path: string; scope: "user" | "project" }> = [];

    if (scope === "user" || scope === "both" || !scope) {
      dirs.push({
        path: path.join(process.env.HOME || "~", ".pi", "agent", "agents"),
        scope: "user",
      });
    }

    if (scope === "project" || scope === "both" || !scope) {
      dirs.push({
        path: path.join(this.cwd, ".pi", "agents"),
        scope: "project",
      });
    }

    for (const dir of dirs) {
      try {
        const files = await fs.readdir(dir.path);
        for (const file of files) {
          if (file.endsWith(".md")) {
            const agentPath = path.join(dir.path, file);
            const content = await fs.readFile(agentPath, "utf-8");
            const match = content.match(/^---\n([\s\S]*?)\n---/);

            if (match) {
              const frontmatter = match[1];
              const nameMatch = frontmatter.match(/name:\s*(.+)/);
              const descMatch = frontmatter.match(/description:\s*(.+)/);

              if (nameMatch && descMatch) {
                agents.push({
                  name: nameMatch[1].trim(),
                  description: descMatch[1].trim(),
                  path: agentPath,
                  scope: dir.scope,
                });
              }
            }
          }
        }
      } catch (error) {
        // Directory doesn't exist, skip
      }
    }

    return agents;
  }
}

/**
 * Extension 注册函数
 */
export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "create_custom_agent",
    description: "Create a custom agent definition file",
    parameters: AgentDefinitionSchema,
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      const generator = new AgentDefinitionGenerator(ctx.cwd);

      onUpdate({
        type: "progress",
        content: `Creating agent: ${params.name}`,
      });

      const agentPath = await generator.createAgent(params);

      onUpdate({
        type: "progress",
        content: `✅ Agent created at: ${agentPath}`,
      });

      // 验证创建的 agent
      const validation = await generator.validateAgent(agentPath);

      if (!validation.valid) {
        throw new Error(`Agent validation failed:\n${validation.errors.join("\n")}`);
      }

      return {
        success: true,
        agentPath,
        message: `Agent ${params.name} created successfully`,
      };
    },
  });

  pi.registerTool({
    name: "list_custom_agents",
    description: "List all available custom agents",
    parameters: z.object({
      scope: z.enum(["user", "project", "both"]).optional(),
    }),
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      const generator = new AgentDefinitionGenerator(ctx.cwd);
      const agents = await generator.listAgents(params.scope);

      return {
        agents,
        count: agents.length,
      };
    },
  });
}
```

---

## 4. 代码解析

### 4.1 核心设计决策

**1. Markdown + YAML frontmatter 格式**

这种格式的优势:
- ✅ 人类可读,易于编辑
- ✅ 版本控制友好
- ✅ 支持丰富的 system prompt 内容
- ✅ 与 [ragapp/agentfiles](https://github.com/ragapp/agentfiles) 等主流框架一致

**2. 专业化代理设计**

根据 [Reddit 案例](https://www.reddit.com/r/LangChain/comments/1llw60o/most_people_think_one_ai_agent_can_handle),专业化代理的优势:
- 每个代理 prompt 可以极致优化
- 工具集精简,减少混乱
- 更好的可调试性
- 更高的准确率

**3. 工具集限定**

每个代理只能访问其定义的工具,实现最小特权原则。这符合 [2026 年安全最佳实践](https://www.stack-ai.com/blog/the-2026-guide-to-agentic-workflow-architectures)。

### 4.2 Agent 角色分工

| Agent | 工具集 | 模型 | 职责 |
|-------|--------|------|------|
| Scout | read, grep, glob, ls, bash | Haiku | 快速侦察 |
| Planner | read, grep, glob | Sonnet | 战略规划 |
| Worker | read, write, edit, bash | Sonnet | 执行实现 |
| Reviewer | read, grep, bash | Sonnet | 质量审查 |

---

## 5. 实际案例分析

### 案例 1: 1 个通用代理 → 13 个专业化代理

**来源**: [Reddit - LangChain Community](https://www.reddit.com/r/LangChain/comments/1llw60o/most_people_think_one_ai_agent_can_handle)

**背景**: 博客内容自动化系统

**拆分前**: 1 个通用 content-agent
- 负责研究、写作、SEO、图片生成
- 上下文窗口经常溢出
- 工具冲突频繁
- 输出质量不稳定

**拆分后**: 13 个专业化代理
- research-agent: 收集信息
- writer-agent: 创作内容
- seo-agent: 优化排名
- image-agent: 生成图片
- editor-agent: 审查改进
- publisher-agent: 管理发布
- ...

**结果**:
- ✅ 准确率提升 40%
- ✅ 可调试性显著改善
- ✅ 系统更健壮

### 案例 2: 5 代理系统用于 AI 咨询公司

**来源**: [Reddit - AI Agents Community](https://www.reddit.com/r/AI_Agents/comments/1r7d4yv/real_example_from_my_setup_i_run_a_5agent_system)

**系统架构**:
1. **intake-agent**: 客户需求分析
2. **research-agent**: 技术调研
3. **proposal-agent**: 方案撰写
4. **review-agent**: 质量审查
5. **delivery-agent**: 交付管理

**关键设计**:
- 共享任务板协调
- 明确的交接协议
- 每个代理独立的工具集

**效果**:
- 可靠性大幅提高
- 客户满意度提升
- 团队协作更顺畅

### 案例 3: ragapp/agentfiles YAML 配置

**来源**: [GitHub - ragapp/agentfiles](https://github.com/ragapp/agentfiles)

**News Reporter Agent 示例**:

```yaml
agents:
  - name: news_reporter
    role: News Reporter
    goal: Generate comprehensive reports about current events
    backstory: You are an experienced journalist
    tools:
      - WebSearch
      - ContentGenerator
```

**特点**:
- 单文件定义
- 工具集成简单
- UI 配置界面
- 支持多模型提供商

---

## 6. 最佳实践

### 6.1 Agent 命名规范

```typescript
// ✅ 好的命名
"scout"           // 简短、描述性
"security-auditor" // 连字符分隔
"performance-optimizer"

// ❌ 不好的命名
"ScoutAgent"      // 不要用 PascalCase
"scout_agent"     // 不要用下划线
"agent1"          // 不要用数字编号
```

### 6.2 工具集最小化

根据 [2026 年最佳实践](https://www.stack-ai.com/blog/the-2026-guide-to-agentic-workflow-architectures):

```markdown
# ✅ 好的工具集 - 精简且专注
---
tools: read, grep, glob
---

# ❌ 不好的工具集 - 过于宽泛
---
tools: read, write, edit, grep, glob, bash, ls, find, ...
---
```

### 6.3 System Prompt 结构化

```markdown
# Agent Name

## Your Mission
[Clear, concise mission statement]

## Guidelines
1. [Guideline 1]
2. [Guideline 2]

## Output Format
```
[Expected output structure]
```

## Constraints
- [Constraint 1]
- [Constraint 2]
```

### 6.4 模型选择策略

| 任务类型 | 推荐模型 | 原因 |
|---------|---------|------|
| 快速侦察 | Haiku | 速度快、成本低 |
| 战略规划 | Sonnet | 平衡性能和成本 |
| 复杂实现 | Sonnet | 代码质量高 |
| 深度审查 | Sonnet | 分析能力强 |

---

## 7. 常见问题

### Q1: Agent 定义应该放在 user-level 还是 project-level?

**A**: 根据使用场景选择:

- **User-level** (`~/.pi/agent/agents/`): 通用代理,跨项目使用
- **Project-level** (`.pi/agents/`): 项目特定代理,团队共享

**安全考虑**: Project-level agents 需要用户确认才能执行(除非设置 `confirmProjectAgents: false`)。

### Q2: 如何避免工具集过于宽泛?

**A**: 遵循最小特权原则:

```typescript
// 分析代理需要的工具
const scoutNeeds = ["read", "grep", "glob"]; // 只读
const workerNeeds = ["read", "write", "edit"]; // 读写
const reviewerNeeds = ["read", "grep", "bash"]; // 读 + 验证
```

### Q3: System prompt 应该多长?

**A**: 根据 [2026 年指南](https://www.stack-ai.com/blog/the-2026-guide-to-agentic-workflow-architectures):

- **最小**: 100 字符(太短无法有效指导)
- **推荐**: 200-500 字符(清晰且聚焦)
- **最大**: 1000 字符(超过会影响性能)

---

## 8. 扩展阅读

### 核心资源

1. **[ragapp/agentfiles](https://github.com/ragapp/agentfiles)**
   - YAML 单文件定义 AI 代理
   - 实际示例和最佳实践

2. **[Stack-AI 2026 Guide](https://www.stack-ai.com/blog/the-2026-guide-to-agentic-workflow-architectures)**
   - 四种工作流架构
   - Agent 设计最佳实践

3. **[Reddit - 1→13 Specialized Agents](https://www.reddit.com/r/LangChain/comments/1llw60o/most_people_think_one_ai_agent_can_handle)**
   - 真实专业化案例
   - 社区讨论和经验

4. **[Reddit - 5-Agent System](https://www.reddit.com/r/AI_Agents/comments/1r7d4yv/real_example_from_my_setup_i_run_a_5agent_system)**
   - AI 咨询公司实战
   - 协调机制设计

### 相关知识点

- `03_核心概念_02_Agent发现与配置.md` - Agent 发现机制
- `03_核心概念_09_Agent定义格式.md` - 格式详解
- `07_实战代码_05_Agent发现与注册.md` - 发现与注册实现

---

**版本**: v1.0
**最后更新**: 2026-02-21
**作者**: Claude Code
**审核**: 基于 2025-2026 年最新实践
