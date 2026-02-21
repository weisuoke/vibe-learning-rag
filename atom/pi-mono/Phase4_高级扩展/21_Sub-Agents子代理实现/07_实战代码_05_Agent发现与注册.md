# Agent发现与注册 - 实战代码

> **知识点**: Sub-Agents 子代理实现 - Agent 发现与注册机制
> **难度**: ⭐⭐⭐⭐
> **前置知识**: 自定义 Agent 定义、文件系统操作

---

## 1. 场景描述

Agent 发现与注册是 Sub-Agents 系统的核心基础设施,负责在运行时动态发现可用的代理并将其注册到系统中。这使得系统能够灵活地扩展新代理,支持热重载,并实现代理的生命周期管理。

**典型应用场景**:
- 启动时自动发现所有可用代理
- 运行时动态加载新代理(热重载)
- 多层级代理发现(user-level + project-level)
- 代理能力查询和匹配

根据 [IETF AI Agent Discovery and Invocation Protocol](https://datatracker.ietf.org/doc/draft-cui-ai-agent-discovery-invocation/01),标准化的代理发现机制是实现跨平台互操作的关键。[arXiv Agent Name Service 论文](https://arxiv.org/abs/2505.10609)提出了基于 DNS 的分布式代理目录,支持 PKI 证书验证和能力感知解析。

---

## 2. 核心概念

### 2.1 发现路径

Pi-mono 支持两级代理发现:

| 级别 | 路径 | 作用域 | 信任级别 |
|------|------|--------|---------|
| User-level | `~/.pi/agent/agents/*.md` | 跨项目 | 高(用户安装) |
| Project-level | `.pi/agents/*.md` | 项目内 | 中(需确认) |

### 2.2 Agent Scope 控制

```typescript
type AgentScope = "user" | "project" | "both";

// 发现指定范围的代理
const discovery = discoverAgents(cwd, "both");
```

### 2.3 与行业标准的对应

| Pi-mono 概念 | IETF AIDIP | arXiv ANS |
|-------------|-----------|-----------|
| `discoverAgents()` | Discovery Mechanism | Agent Resolution |
| Agent metadata | Agent Metadata Spec | JSON Schema |
| User/Project scope | Trust boundaries | PKI Certificates |
| Hot reload | Dynamic loading | Registration renewal |

---

## 3. 完整代码示例

### 3.1 Agent 发现实现

```typescript
import * as fs from "fs/promises";
import * as path from "path";
import { z } from "zod";

// Agent 元数据 Schema
const AgentMetadataSchema = z.object({
  name: z.string(),
  description: z.string(),
  tools: z.string(),
  model: z.string().optional(),
});

interface Agent {
  name: string;
  description: string;
  tools: string[];
  model?: string;
  source: "user" | "project";
  path: string;
  systemPrompt: string;
}

interface AgentDiscovery {
  agents: Agent[];
  userAgents: Agent[];
  projectAgents: Agent[];
  errors: Array<{ path: string; error: string }>;
}

/**
 * Agent 发现器
 *
 * 参考:
 * - https://datatracker.ietf.org/doc/draft-cui-ai-agent-discovery-invocation/01 (IETF AIDIP)
 * - https://arxiv.org/abs/2505.10609 (arXiv ANS)
 */
export class AgentDiscoverer {
  private cwd: string;
  private cache: Map<string, Agent> = new Map();

  constructor(cwd: string) {
    this.cwd = cwd;
  }

  /**
   * 发现所有可用代理
   */
  async discoverAgents(scope: "user" | "project" | "both" = "both"): Promise<AgentDiscovery> {
    const discovery: AgentDiscovery = {
      agents: [],
      userAgents: [],
      projectAgents: [],
      errors: [],
    };

    // User-level agents
    if (scope === "user" || scope === "both") {
      const userDir = path.join(process.env.HOME || "~", ".pi", "agent", "agents");
      const userAgents = await this.discoverFromDirectory(userDir, "user");
      discovery.userAgents = userAgents.agents;
      discovery.errors.push(...userAgents.errors);
    }

    // Project-level agents
    if (scope === "project" || scope === "both") {
      const projectDir = path.join(this.cwd, ".pi", "agents");
      const projectAgents = await this.discoverFromDirectory(projectDir, "project");
      discovery.projectAgents = projectAgents.agents;
      discovery.errors.push(...projectAgents.errors);
    }

    // 合并所有代理
    discovery.agents = [...discovery.userAgents, ...discovery.projectAgents];

    // 更新缓存
    for (const agent of discovery.agents) {
      this.cache.set(agent.name, agent);
    }

    return discovery;
  }

  /**
   * 从目录发现代理
   */
  private async discoverFromDirectory(
    dir: string,
    source: "user" | "project"
  ): Promise<{ agents: Agent[]; errors: Array<{ path: string; error: string }> }> {
    const agents: Agent[] = [];
    const errors: Array<{ path: string; error: string }> = [];

    try {
      await fs.access(dir);
    } catch {
      return { agents, errors };
    }

    try {
      const files = await fs.readdir(dir);

      for (const file of files) {
        if (!file.endsWith(".md")) continue;

        const agentPath = path.join(dir, file);

        try {
          const agent = await this.parseAgentFile(agentPath, source);
          agents.push(agent);
        } catch (error: any) {
          errors.push({
            path: agentPath,
            error: error.message,
          });
        }
      }
    } catch (error: any) {
      errors.push({
        path: dir,
        error: `Failed to read directory: ${error.message}`,
      });
    }

    return { agents, errors };
  }

  /**
   * 解析 Agent 文件
   */
  private async parseAgentFile(filePath: string, source: "user" | "project"): Promise<Agent> {
    const content = await fs.readFile(filePath, "utf-8");

    // 提取 frontmatter
    const match = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
    if (!match) {
      throw new Error("Invalid agent file format: missing frontmatter");
    }

    const [, frontmatter, systemPrompt] = match;

    // 解析 YAML frontmatter (简单实现)
    const metadata: Record<string, string> = {};
    for (const line of frontmatter.split("\n")) {
      const [key, ...valueParts] = line.split(":");
      if (key && valueParts.length > 0) {
        metadata[key.trim()] = valueParts.join(":").trim();
      }
    }

    // 验证必需字段
    const validated = AgentMetadataSchema.parse(metadata);

    return {
      name: validated.name,
      description: validated.description,
      tools: validated.tools.split(",").map(t => t.trim()),
      model: validated.model,
      source,
      path: filePath,
      systemPrompt: systemPrompt.trim(),
    };
  }

  /**
   * 按名称查找代理
   */
  async findAgent(name: string): Promise<Agent | null> {
    // 先查缓存
    if (this.cache.has(name)) {
      return this.cache.get(name)!;
    }

    // 重新发现
    await this.discoverAgents();
    return this.cache.get(name) || null;
  }

  /**
   * 按能力查找代理
   */
  async findAgentsByCapability(capability: string): Promise<Agent[]> {
    const discovery = await this.discoverAgents();
    return discovery.agents.filter(agent =>
      agent.description.toLowerCase().includes(capability.toLowerCase())
    );
  }

  /**
   * 热重载 - 重新发现代理
   */
  async reload(): Promise<AgentDiscovery> {
    this.cache.clear();
    return this.discoverAgents();
  }
}

/**
 * Agent 注册表
 *
 * 参考:
 * - https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/1780 (MCP dynamic loading)
 */
export class AgentRegistry {
  private agents: Map<string, Agent> = new Map();
  private discoverer: AgentDiscoverer;

  constructor(cwd: string) {
    this.discoverer = new AgentDiscoverer(cwd);
  }

  /**
   * 初始化注册表
   */
  async initialize(scope: "user" | "project" | "both" = "both"): Promise<void> {
    const discovery = await this.discoverer.discoverAgents(scope);

    for (const agent of discovery.agents) {
      this.register(agent);
    }

    if (discovery.errors.length > 0) {
      console.warn(`Agent discovery errors: ${discovery.errors.length}`);
      for (const error of discovery.errors) {
        console.warn(`  ${error.path}: ${error.error}`);
      }
    }
  }

  /**
   * 注册代理
   */
  register(agent: Agent): void {
    if (this.agents.has(agent.name)) {
      throw new Error(`Agent ${agent.name} already registered`);
    }
    this.agents.set(agent.name, agent);
  }

  /**
   * 注销代理
   */
  unregister(name: string): boolean {
    return this.agents.delete(name);
  }

  /**
   * 获取代理
   */
  get(name: string): Agent | undefined {
    return this.agents.get(name);
  }

  /**
   * 列出所有代理
   */
  list(): Agent[] {
    return Array.from(this.agents.values());
  }

  /**
   * 按来源过滤
   */
  listBySource(source: "user" | "project"): Agent[] {
    return this.list().filter(agent => agent.source === source);
  }

  /**
   * 搜索代理
   */
  search(query: string): Agent[] {
    const lowerQuery = query.toLowerCase();
    return this.list().filter(agent =>
      agent.name.toLowerCase().includes(lowerQuery) ||
      agent.description.toLowerCase().includes(lowerQuery)
    );
  }

  /**
   * 热重载
   */
  async reload(): Promise<void> {
    this.agents.clear();
    await this.initialize();
  }
}

/**
 * Extension 注册函数
 */
export default function (pi: ExtensionAPI) {
  const registry = new AgentRegistry(pi.cwd);

  // 初始化时发现代理
  registry.initialize("both").catch(console.error);

  pi.registerTool({
    name: "list_agents",
    description: "List all available agents",
    parameters: z.object({
      source: z.enum(["user", "project", "all"]).optional(),
    }),
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      let agents: Agent[];

      if (params.source === "user") {
        agents = registry.listBySource("user");
      } else if (params.source === "project") {
        agents = registry.listBySource("project");
      } else {
        agents = registry.list();
      }

      return {
        agents: agents.map(a => ({
          name: a.name,
          description: a.description,
          tools: a.tools,
          source: a.source,
        })),
        count: agents.length,
      };
    },
  });

  pi.registerTool({
    name: "search_agents",
    description: "Search agents by name or description",
    parameters: z.object({
      query: z.string(),
    }),
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      const agents = registry.search(params.query);

      return {
        agents: agents.map(a => ({
          name: a.name,
          description: a.description,
          source: a.source,
        })),
        count: agents.length,
      };
    },
  });

  pi.registerTool({
    name: "reload_agents",
    description: "Reload all agents (hot reload)",
    parameters: z.object({}),
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      onUpdate({ type: "progress", content: "Reloading agents..." });

      await registry.reload();

      const agents = registry.list();

      onUpdate({ type: "progress", content: `✅ Reloaded ${agents.length} agents` });

      return {
        success: true,
        count: agents.length,
      };
    },
  });
}
```

---

## 4. 代码解析

### 4.1 核心设计决策

**1. 两级发现机制**

```typescript
// User-level: 跨项目共享
~/.pi/agent/agents/*.md

// Project-level: 项目特定
.pi/agents/*.md
```

这种设计符合 [IETF AIDIP](https://datatracker.ietf.org/doc/draft-cui-ai-agent-discovery-invocation/01) 的信任边界原则。

**2. 缓存机制**

```typescript
private cache: Map<string, Agent> = new Map();
```

减少文件系统访问,提高性能。参考 [MCP 动态加载讨论](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/1780)。

**3. 错误容忍**

```typescript
errors: Array<{ path: string; error: string }>;
```

单个代理解析失败不影响其他代理,符合生产环境最佳实践。

---

## 5. 实际案例分析

### 案例 1: IETF AI Agent Discovery and Invocation Protocol

**来源**: [IETF draft-cui-ai-agent-discovery-invocation-01](https://datatracker.ietf.org/doc/draft-cui-ai-agent-discovery-invocation/01)

**核心机制**:
- Agent Registry (Discovery Service)
- 基于能力的发现
- RESTful 调用接口
- OAuth 2.0 认证

**Pi-mono 对应**:
- `AgentRegistry` = Agent Registry
- `findAgentsByCapability()` = Capability-based search
- `discoverAgents()` = Discovery mechanism

### 案例 2: arXiv Agent Name Service (ANS)

**来源**: [arXiv:2505.10609](https://arxiv.org/abs/2505.10609)

**关键创新**:
- DNS-based 分布式目录
- PKI 证书验证
- 协议适配器层 (A2A, MCP, ACP)
- 能力感知解析

**启示**:
- 分层命名空间
- 安全身份验证
- 多协议支持

### 案例 3: MCP 动态工具加载

**来源**: [GitHub MCP Discussion #1780](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/1780)

**核心模式**:
- 渐进式披露
- 按需加载
- 两通道模式 (指令 + 数据)

**应用**:
```typescript
// 仅加载相关代理
const relevant = await findAgentsByCapability("security");
```

---

## 6. 最佳实践

### 6.1 发现策略

```typescript
// ✅ 启动时发现
await registry.initialize("both");

// ✅ 按需查找
const agent = await registry.get("scout");

// ❌ 每次都重新发现
// 性能差,应使用缓存
```

### 6.2 错误处理

```typescript
// ✅ 容错设计
const discovery = await discoverAgents();
if (discovery.errors.length > 0) {
  console.warn("Some agents failed to load");
}

// ❌ 失败即中断
// 单个代理失败不应影响整体
```

### 6.3 热重载时机

```typescript
// ✅ 用户明确请求时
await registry.reload();

// ✅ 文件系统监听
fs.watch(".pi/agents", () => registry.reload());

// ❌ 每次调用前重载
// 性能开销大
```

---

## 7. 常见问题

### Q1: User-level 和 Project-level 代理冲突怎么办?

**A**: Project-level 优先:

```typescript
// 先查 project,再查 user
const projectAgent = projectAgents.find(a => a.name === name);
const userAgent = userAgents.find(a => a.name === name);
return projectAgent || userAgent;
```

### Q2: 如何实现代理版本管理?

**A**: 在 frontmatter 添加 version 字段:

```markdown
---
name: scout
version: 1.2.0
---
```

### Q3: 如何实现代理依赖管理?

**A**: 参考 [IETF AIDIP](https://datatracker.ietf.org/doc/draft-cui-ai-agent-discovery-invocation/01) 的依赖声明:

```yaml
dependencies:
  - name: base-agent
    version: ">=1.0.0"
```

---

## 8. 扩展阅读

### 核心资源

1. **[IETF AIDIP](https://datatracker.ietf.org/doc/draft-cui-ai-agent-discovery-invocation/01)**
   - 标准化协议
   - Agent 元数据规范

2. **[arXiv ANS](https://arxiv.org/abs/2505.10609)**
   - DNS-based 目录
   - PKI 证书验证

3. **[GitHub MCP #1780](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/1780)**
   - 动态工具加载
   - 渐进式披露

4. **[JetBrains ACP Registry](https://blog.jetbrains.com/ai/2026/01/acp-agent-registry)**
   - IDE 集成
   - 代理市场

### 相关知识点

- `03_核心概念_02_Agent发现与配置.md` - 发现机制详解
- `07_实战代码_04_自定义Agent定义.md` - Agent 定义格式
- `07_实战代码_06_安全与确认流程.md` - 信任边界

---

**版本**: v1.0
**最后更新**: 2026-02-21
**作者**: Claude Code
**审核**: 基于 2025-2026 年最新实践
