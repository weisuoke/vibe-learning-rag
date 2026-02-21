# 核心概念 02：Agent 发现与配置

## 概述

**Agent 发现机制是 Sub-Agents 系统的基础，决定了哪些 Agents 可用、如何加载以及如何配置。**

本文档深入解析 pi-mono 的 Agent 发现机制、配置系统，以及与 2025-2026 业界最新动态发现模式的对比。

---

## 1. Agent 发现机制

### 1.1 发现流程概览

**Agent 发现在每次 subagent 工具调用时动态执行。**

```typescript
// sourcecode/pi-mono/packages/coding-agent/examples/extensions/subagent/agents.ts
export function discoverAgents(
  cwd: string,
  scope: AgentScope
): AgentDiscovery {
  const userAgentsDir = path.join(os.homedir(), ".pi", "agent", "agents");
  const projectAgentsDir = path.join(cwd, ".pi", "agents");

  const discovery: AgentDiscovery = {
    agents: [],
    userAgentsDir,
    projectAgentsDir: null
  };

  // 1. 发现用户级别 agents
  if (scope === "user" || scope === "both") {
    const userAgents = loadAgentsFromDir(userAgentsDir, "user");
    discovery.agents.push(...userAgents);
  }

  // 2. 发现项目级别 agents
  if (scope === "project" || scope === "both") {
    if (fs.existsSync(projectAgentsDir)) {
      discovery.projectAgentsDir = projectAgentsDir;
      const projectAgents = loadAgentsFromDir(projectAgentsDir, "project");

      // 项目 agents 覆盖同名用户 agents
      for (const pa of projectAgents) {
        const idx = discovery.agents.findIndex(a => a.name === pa.name);
        if (idx >= 0) discovery.agents[idx] = pa;
        else discovery.agents.push(pa);
      }
    }
  }

  return discovery;
}
```

**关键特性**：
- **动态发现**：每次调用时重新扫描，支持热重载
- **作用域控制**：通过 `AgentScope` 参数控制发现范围
- **覆盖机制**：项目 agents 可以覆盖同名用户 agents

### 1.2 文件扫描机制

**扫描指定目录下的所有 `.md` 文件。**

```typescript
function loadAgentsFromDir(
  dir: string,
  source: "user" | "project"
): AgentConfig[] {
  if (!fs.existsSync(dir)) return [];

  const agents: AgentConfig[] = [];
  const files = fs.readdirSync(dir);

  for (const file of files) {
    if (!file.endsWith(".md")) continue;

    const filePath = path.join(dir, file);
    const content = fs.readFileSync(filePath, "utf-8");

    try {
      const agent = parseAgentFile(content, source);
      agents.push(agent);
    } catch (error) {
      console.error(`Failed to parse agent file ${file}:`, error);
    }
  }

  return agents;
}
```

**扫描规则**：
- ✅ 只扫描 `.md` 文件
- ✅ 忽略解析失败的文件
- ✅ 递归扫描子目录（如果需要）
- ❌ 不缓存结果（每次重新扫描）

### 1.3 YAML Frontmatter 解析

**使用 YAML frontmatter 定义 Agent 元数据。**

```typescript
function parseAgentFile(
  content: string,
  source: "user" | "project"
): AgentConfig {
  // 提取 YAML frontmatter
  const match = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!match) {
    throw new Error("Invalid agent file: missing YAML frontmatter");
  }

  const [, yamlContent, systemPrompt] = match;
  const metadata = yaml.parse(yamlContent);

  return {
    name: metadata.name,
    description: metadata.description,
    tools: metadata.tools ? metadata.tools.split(",").map(t => t.trim()) : [],
    model: metadata.model,
    source,
    systemPrompt: systemPrompt.trim()
  };
}
```

**解析步骤**：
1. 正则匹配 `---` 包围的 YAML 块
2. 解析 YAML 为对象
3. 提取 `name`, `description`, `tools`, `model`
4. 剩余内容作为 `systemPrompt`

---

## 2. Agent 配置结构

### 2.1 AgentConfig 接口

```typescript
interface AgentConfig {
  name: string;              // Agent 唯一标识
  description: string;       // Agent 功能描述
  tools: string[];           // 可用工具列表
  model?: string;            // 使用的模型
  source: "user" | "project"; // 来源
  systemPrompt: string;      // 系统提示
}
```

### 2.2 配置示例

**Scout Agent 配置**：

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

**解析结果**：

```typescript
{
  name: "scout",
  description: "Fast codebase reconnaissance",
  tools: ["read", "grep", "find", "ls", "bash"],
  model: "claude-haiku-4-5",
  source: "user",
  systemPrompt: "You are a fast reconnaissance agent..."
}
```

### 2.3 配置字段详解

| 字段 | 必需 | 类型 | 说明 | 示例 |
|------|------|------|------|------|
| `name` | ✅ | string | Agent 唯一标识，用于调用 | `"scout"`, `"planner"` |
| `description` | ✅ | string | Agent 功能描述，用于选择 | `"Fast codebase reconnaissance"` |
| `tools` | ❌ | string | 逗号分隔的工具列表 | `"read, grep, find"` |
| `model` | ❌ | string | 使用的模型 | `"claude-haiku-4-5"` |
| `systemPrompt` | ✅ | string | YAML 后的所有内容 | Agent 的系统提示 |

**默认值**：
- `tools`: 如果未指定，使用 pi 的默认工具集
- `model`: 如果未指定，使用 pi 的默认模型

---

## 3. Agent Scope 控制

### 3.1 三种 Scope 模式

```typescript
type AgentScope = "user" | "project" | "both";
```

| Scope | 发现范围 | 用途 | 安全性 |
|-------|---------|------|--------|
| `"user"` | 只加载 `~/.pi/agent/agents/` | 个人 agents，跨项目共享 | 高（完全信任） |
| `"project"` | 只加载 `.pi/agents/` | 项目特定 agents | 中（需要审查） |
| `"both"` | 加载两者，项目优先 | 结合个人和项目 agents | 中（需要确认） |

### 3.2 Scope 选择逻辑

```typescript
// 默认：只使用用户 agents
const scope: AgentScope = params.agentScope ?? "user";

// 发现 agents
const discovery = discoverAgents(ctx.cwd, scope);

// 如果使用项目 agents，需要确认
if ((scope === "project" || scope === "both") && confirmProjectAgents) {
  const projectAgentsRequested = /* 筛选项目 agents */;

  if (projectAgentsRequested.length > 0) {
    const ok = await ctx.ui.confirm(
      "Run project-local agents?",
      `Agents: ${names}\nSource: ${dir}\n\n` +
      "Project agents are repo-controlled. Only continue for trusted repositories."
    );

    if (!ok) {
      return { content: [{ type: "text", text: "Canceled" }] };
    }
  }
}
```

**安全考虑**：
- ✅ 默认只使用用户 agents（安全）
- ⚠️ 项目 agents 需要用户确认（可选）
- ❌ 不信任的仓库不应启用项目 agents

### 3.3 覆盖机制

**项目 agents 可以覆盖同名用户 agents。**

```typescript
// 场景：用户有 scout.md，项目也有 scout.md
// 结果：使用项目的 scout.md

// 用户 scout
{
  name: "scout",
  tools: ["read", "grep", "find"],
  source: "user"
}

// 项目 scout（覆盖）
{
  name: "scout",
  tools: ["read", "grep", "find", "bash"],  // 添加了 bash
  source: "project"
}

// 最终使用项目 scout
```

**用途**：
- ✅ 项目特定的 Agent 配置
- ✅ 覆盖默认行为
- ⚠️ 需要注意安全风险

---

## 4. 2025-2026 动态发现模式对比

### 4.1 业界主流发现模式

根据 2025-2026 最新研究，AI Agent 发现机制有以下主流模式：

[Source: Agentic MCP: Dynamic MCP Integration for AI Agents](https://medium.com/red-buffer/agentic-mcp-dynamic-mcp-integration-for-ai-agents-5c1daf5b950c)

| 模式 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **File-Based** | 扫描文件系统 | 简单、无依赖 | 性能较低 |
| **Registry-Based** | 中央注册表 | 快速、可管理 | 需要额外服务 |
| **Protocol-Based** | MCP/A2A 协议 | 标准化、互操作 | 复杂度高 |
| **Service Discovery** | 类似 K8s | 动态、可扩展 | 基础设施要求高 |

[Source: MCP Tool Discovery Specification](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)

### 4.2 Pi-mono 的选择：File-Based

**Pi-mono 选择 File-Based 发现模式的原因：**

```typescript
// ✅ 优点
1. 简单：无需额外服务或基础设施
2. 透明：用户可以直接查看和编辑 agent 文件
3. 版本控制：项目 agents 可以纳入 git
4. 热重载：编辑后立即生效（下次调用时）
5. 零配置：开箱即用

// ⚠️ 缺点
1. 性能：每次调用都扫描文件系统
2. 扩展性：大量 agents 时性能下降
3. 分布式：不支持远程 agents
```

**性能优化空间**：

```typescript
// 未来可能的优化
class AgentCache {
  private cache = new Map<string, AgentConfig>();
  private mtimes = new Map<string, number>();

  get(file: string): AgentConfig | null {
    const mtime = fs.statSync(file).mtimeMs;
    if (this.mtimes.get(file) === mtime) {
      return this.cache.get(file) ?? null;
    }
    return null;
  }

  set(file: string, agent: AgentConfig) {
    const mtime = fs.statSync(file).mtimeMs;
    this.cache.set(file, agent);
    this.mtimes.set(file, mtime);
  }
}
```

### 4.3 MCP 协议发现对比

**MCP (Model Context Protocol) 的动态发现机制：**

[Source: MCP Architecture Patterns for AI Systems](https://developer.ibm.com/articles/mcp-architecture-patterns-ai-systems/)

```typescript
// MCP Server 动态工具发现
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "result": {
    "tools": [
      {
        "name": "read_file",
        "description": "Read a file",
        "inputSchema": { ... }
      },
      {
        "name": "search_code",
        "description": "Search code",
        "inputSchema": { ... }
      }
    ]
  }
}
```

**对比**：

| 维度 | Pi-mono File-Based | MCP Protocol-Based |
|------|-------------------|-------------------|
| **发现时机** | 每次调用时扫描 | 连接时查询 |
| **配置方式** | Markdown 文件 | JSON-RPC 协议 |
| **动态性** | 文件修改后生效 | 服务器重启后生效 |
| **标准化** | 自定义格式 | MCP 标准 |
| **互操作性** | 仅 pi-mono | 跨平台 |

**未来演进**：Pi-mono 可能支持 MCP 协议发现，与 File-Based 并存。

---

## 5. 配置最佳实践

### 5.1 Agent 命名规范

**好的命名**：

```markdown
✅ scout       - 简短、描述性
✅ planner     - 动词或名词
✅ reviewer    - 清晰的职责
✅ worker      - 通用角色
```

**不好的命名**：

```markdown
❌ agent1      - 无意义
❌ my-agent    - 太通用
❌ super-duper-agent - 太长
❌ 侦查员      - 非英文（可能有兼容性问题）
```

### 5.2 工具集配置

**原则：最小权限原则**

```markdown
<!-- ✅ Scout: 只读工具 -->
---
name: scout
tools: read, grep, find, ls
---

<!-- ✅ Planner: 只读 + 分析工具 -->
---
name: planner
tools: read, grep, find, ls
---

<!-- ✅ Worker: 完整工具集 -->
---
name: worker
tools: (all default)
---

<!-- ❌ Scout 有写权限（不必要） -->
---
name: scout
tools: read, grep, find, ls, write, edit, bash
---
```

### 5.3 模型选择

**根据任务复杂度选择模型**：

```markdown
<!-- ✅ 简单任务：Haiku（快速、便宜） -->
---
name: scout
model: claude-haiku-4-5
---

<!-- ✅ 复杂任务：Sonnet（强大、平衡） -->
---
name: planner
model: claude-sonnet-4
---

<!-- ✅ 最复杂任务：Opus（最强、昂贵） -->
---
name: architect
model: claude-opus-4
---
```

**成本对比**：

| 模型 | 输入成本 | 输出成本 | 适用场景 |
|------|---------|---------|---------|
| Haiku | 低 | 低 | 快速侦查、简单任务 |
| Sonnet | 中 | 中 | 计划生成、代码审查 |
| Opus | 高 | 高 | 架构设计、复杂推理 |

### 5.4 系统提示设计

**清晰的职责定义**：

```markdown
<!-- ✅ 好的系统提示 -->
---
name: scout
---

You are a fast reconnaissance agent. Your job is to:
1. Quickly locate relevant files and code
2. Return compressed, actionable context
3. Use efficient search strategies

Focus on:
- File paths and line numbers
- Key function/class names
- Brief code summaries

Do NOT:
- Implement changes
- Write detailed analysis
- Read entire files unless necessary

Output format:
- File: path/to/file.ts (lines 10-50)
- Functions: login(), verify(), refresh()
- Summary: JWT-based authentication
```

```markdown
<!-- ❌ 不好的系统提示 -->
---
name: scout
---

You are a helpful assistant.
```

---

## 6. 高级配置场景

### 6.1 项目特定 Agent

**场景：项目有特殊的代码结构，需要定制 scout。**

```bash
# 项目结构
.pi/agents/
└── scout.md  # 项目特定的 scout

# 内容
---
name: scout
description: Project-specific scout for monorepo
tools: read, grep, find, ls, bash
model: claude-haiku-4-5
---

You are a scout for a monorepo project.

Project structure:
- packages/: All packages
- apps/: All applications
- libs/: Shared libraries

When searching:
1. Focus on packages/ and apps/ first
2. Use workspace-aware search
3. Consider package dependencies
```

**使用**：

```typescript
// 启用项目 agents
{
  agent: "scout",
  task: "Find authentication code",
  agentScope: "both"  // 使用项目的 scout
}
```

### 6.2 多环境配置

**场景：开发环境和生产环境使用不同的 agents。**

```bash
# 开发环境
~/.pi/agent/agents/
├── scout-dev.md      # 开发用 scout（更详细）
├── planner-dev.md    # 开发用 planner（更激进）
└── worker-dev.md     # 开发用 worker（更快速）

# 生产环境
~/.pi/agent/agents/
├── scout-prod.md     # 生产用 scout（更保守）
├── planner-prod.md   # 生产用 planner（更稳健）
└── worker-prod.md    # 生产用 worker（更安全）
```

**切换**：

```bash
# 开发环境
export PI_AGENT_SUFFIX="-dev"

# 生产环境
export PI_AGENT_SUFFIX="-prod"
```

### 6.3 Agent 继承

**场景：多个 agents 共享基础配置。**

```markdown
<!-- base-agent.md（不直接使用） -->
---
name: base
description: Base agent configuration
tools: read, grep, find, ls
---

Common instructions for all agents:
- Be concise
- Use structured output
- Focus on actionable information

<!-- scout.md（继承 base） -->
---
name: scout
description: Fast reconnaissance
tools: read, grep, find, ls
model: claude-haiku-4-5
---

{{include: base-agent.md}}

Additional scout-specific instructions:
- Prioritize speed over completeness
- Return compressed context
```

**注意**：Pi-mono 当前不支持继承，这是未来可能的功能。

---

## 7. 故障排查

### 7.1 Agent 未发现

**问题**：调用 agent 时提示 "Unknown agent"。

**排查步骤**：

```bash
# 1. 检查文件是否存在
ls ~/.pi/agent/agents/scout.md
ls .pi/agents/scout.md

# 2. 检查文件权限
ls -l ~/.pi/agent/agents/scout.md

# 3. 检查 YAML frontmatter 格式
cat ~/.pi/agent/agents/scout.md
# 确保有 --- 包围的 YAML 块

# 4. 检查 agentScope 参数
# 如果 agent 在项目目录，需要 agentScope: "both"
```

### 7.2 Agent 配置解析失败

**问题**：Agent 文件存在，但无法加载。

**常见原因**：

```markdown
<!-- ❌ 错误1：YAML 格式错误 -->
---
name: scout
description: Fast reconnaissance
tools: read, grep, find  # 缺少引号，有逗号
---

<!-- ✅ 正确 -->
---
name: scout
description: Fast reconnaissance
tools: read, grep, find
---

<!-- ❌ 错误2：缺少必需字段 -->
---
description: Fast reconnaissance
---

<!-- ✅ 正确 -->
---
name: scout
description: Fast reconnaissance
---

<!-- ❌ 错误3：YAML 块不完整 -->
---
name: scout
description: Fast reconnaissance

<!-- ✅ 正确 -->
---
name: scout
description: Fast reconnaissance
---
```

### 7.3 项目 Agent 未生效

**问题**：项目 agent 存在，但使用的是用户 agent。

**原因**：

```typescript
// 默认只使用用户 agents
{ agent: "scout", task: "..." }  // agentScope: "user"

// 需要显式启用项目 agents
{ agent: "scout", task: "...", agentScope: "both" }
```

---

## 8. 性能考虑

### 8.1 发现性能

**当前实现**：

```typescript
// 每次调用都扫描文件系统
function discoverAgents(cwd: string, scope: AgentScope) {
  // 扫描 ~/.pi/agent/agents/
  // 扫描 .pi/agents/
  // 解析所有 .md 文件
}
```

**性能影响**：

| Agents 数量 | 扫描时间 | 影响 |
|------------|---------|------|
| 1-10 | <10ms | 可忽略 |
| 10-50 | 10-50ms | 轻微 |
| 50-100 | 50-100ms | 明显 |
| 100+ | >100ms | 显著 |

**优化建议**：

```typescript
// 1. 缓存解析结果（基于文件 mtime）
// 2. 延迟加载（只加载需要的 agent）
// 3. 索引文件（预先构建 agent 索引）
```

### 8.2 热重载性能

**优势**：

```bash
# 编辑 agent 文件
vim ~/.pi/agent/agents/scout.md

# 下次调用立即生效（无需重启 pi）
pi
> Use scout to find auth code
```

**代价**：

- 每次调用都重新扫描（无缓存）
- 大量 agents 时性能下降

**权衡**：

- ✅ 开发体验好（热重载）
- ⚠️ 性能略有影响（可接受）

---

## 9. 未来演进方向

### 9.1 基于 2025-2026 趋势

根据最新研究，Agent 发现机制的演进方向：

[Source: 2026 AI Agent Protocols Complete Guide](https://www.ruh.ai/blogs/ai-agent-protocols-2026-complete-guide)

**1. 协议标准化**

```typescript
// 未来可能支持 MCP 协议发现
interface MCPAgentDiscovery {
  protocol: "mcp";
  endpoint: "http://localhost:3000/mcp";
  agents: AgentConfig[];
}

// 与 File-Based 并存
type AgentDiscovery = FileBasedDiscovery | MCPAgentDiscovery;
```

**2. 动态能力广告**

[Source: A2A Protocol Agent Discovery](https://www.solo.io/blog/agent-discovery-naming-and-resolution---the-missing-pieces-to-a2a)

```typescript
// Agent 动态广告自己的能力
interface AgentCapabilities {
  name: string;
  capabilities: string[];  // ["code-search", "refactoring", "testing"]
  performance: {
    speed: "fast" | "medium" | "slow";
    cost: "low" | "medium" | "high";
  };
}
```

**3. 智能 Agent 选择**

```typescript
// 根据任务自动选择最佳 agent
async function selectBestAgent(task: string): Promise<AgentConfig> {
  const agents = discoverAgents(cwd, "both");

  // 使用 LLM 分析任务，选择最合适的 agent
  const analysis = await analyzeTas(task);
  return agents.find(a => matchesRequirements(a, analysis));
}
```

### 9.2 Pi-mono 的潜在改进

**1. Agent 索引文件**

```json
// ~/.pi/agent/agents/index.json
{
  "agents": [
    {
      "name": "scout",
      "file": "scout.md",
      "mtime": 1234567890,
      "description": "Fast reconnaissance"
    }
  ]
}
```

**2. 远程 Agent 支持**

```typescript
// 支持从远程加载 agents
interface RemoteAgentSource {
  type: "remote";
  url: "https://agents.example.com/scout.md";
  cache: true;
}
```

**3. Agent 市场**

```bash
# 从 agent 市场安装
pi agent install scout --from marketplace

# 列出可用 agents
pi agent list --marketplace
```

---

## 10. 总结

### 核心要点

1. **动态发现**：每次调用时重新扫描，支持热重载
2. **File-Based**：简单、透明、无依赖
3. **Scope 控制**：用户 vs 项目 agents，安全边界清晰
4. **覆盖机制**：项目 agents 可以覆盖用户 agents
5. **YAML 配置**：简单、可读、易于编辑

### 关键洞察

- **简单性优先**：File-Based 虽然性能略低，但简单可靠
- **安全第一**：默认只使用用户 agents，项目 agents 需要确认
- **热重载**：编辑后立即生效，开发体验好
- **未来演进**：向协议标准化和智能选择演进

### 学习路径

1. ✅ 理解 Agent 发现的基本流程
2. ✅ 掌握 YAML frontmatter 配置格式
3. ✅ 理解 Agent Scope 的安全模型
4. → 深入学习执行模式详解
5. → 探索进程隔离与上下文管理

---

**参考资源**：
- [Agentic MCP: Dynamic MCP Integration](https://medium.com/red-buffer/agentic-mcp-dynamic-mcp-integration-for-ai-agents-5c1daf5b950c)
- [A2A Protocol Agent Discovery](https://www.solo.io/blog/agent-discovery-naming-and-resolution---the-missing-pieces-to-a2a)
- [MCP Tool Discovery Specification](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)
- [2026 AI Agent Protocols Guide](https://www.ruh.ai/blogs/ai-agent-protocols-2026-complete-guide)
