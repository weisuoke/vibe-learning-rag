# Pi-mono 与 Claude Code/Codex-CLI 协同工作调研

> 技术可行性分析与集成方案设计

**调研日期**: 2026-02-18
**版本**: v1.0
**状态**: 完成

---

## 执行摘要

### 核心结论

**三个工具可以通过多种方式协同工作**：

- **Pi-mono**: 高度可扩展，通过 Extension 系统可调用外部工具
- **Codex-CLI**: 支持 MCP Server 模式和非交互式 CLI 调用
- **Claude Code**: 支持非交互式 CLI 调用和 MCP Client 模式

### 推荐集成策略

**MCP 协议 + CLI 包装器的混合方案**

- **短期方案**: CLI 包装器（立即可用，1-2天实施）
- **长期方案**: MCP 协议集成（标准化，3-5天实施）
- **最佳实践**: Pi-mono 作为主控制器，按需委托给专家工具

### 关键价值

1. **灵活性**: 根据任务选择最合适的工具
2. **效率**: 自动化任务分发，减少手动切换
3. **标准化**: 通过 MCP 协议实现可扩展的集成
4. **成本优化**: Pi-mono 处理简单任务，降低 API 成本

---

## 目录

1. [技术可行性分析](#第一部分技术可行性分析)
2. [集成方案设计](#第二部分集成方案设计)
3. [个人工作流配置](#第三部分个人工作流配置)
4. [实施步骤](#第四部分实施步骤)
5. [验证计划](#第五部分验证计划)
6. [关键文件清单](#第六部分关键文件清单)
7. [风险与缓解](#第七部分风险与缓解)

---

## 第一部分：技术可行性分析

### 1.1 Pi-mono 的扩展能力

#### ✅ Extension 系统

**能力**：
- 完整的 TypeScript/JavaScript 扩展 API
- 工具注册：`registerTool()` 可注册自定义工具
- 外部调用：可通过 Node.js `child_process` 调用任何 CLI 工具
- 热重载：`/reload` 命令支持无需重启的扩展更新

**示例代码**：

```typescript
// Extension 示例：调用外部工具
import { execSync } from 'child_process';
import { z } from 'zod';

export default function(api: ExtensionAPI) {
  api.registerTool({
    name: "call_claude_code",
    description: "Call Claude Code for complex tasks",
    schema: z.object({
      task: z.string()
    })
  }, async ({ task }) => {
    const result = execSync(
      `claude --print "${task}" --output-format json`
    );
    return JSON.parse(result.toString());
  });
}
```

#### ✅ MCP 集成

- **方式**: 通过 Extension 包装 MCP Server
- **社区方案**: `pi-mcp-adapter` 已实现
- **灵活性**: 用户可选择是否使用 MCP（符合极简哲学）

#### ✅ Sub-Agent 机制

- **架构**: 支持微服务式的多 Agent 协作
- **外部调用**: 可通过 API 调用外部 AI Agent
- **实现方式**: 在 Extension 中封装外部 Agent SDK

### 1.2 Codex-CLI 的集成能力

#### ✅ 非交互式模式

```bash
# 基本调用
codex exec "task description"

# JSON 输出
codex exec --json "task" | jq

# 结构化输出
codex exec --output-schema ./schema.json "task"

# 会话恢复
codex exec resume --last
```

#### ✅ MCP Server 模式

```bash
# 作为 MCP Server 运行
codex mcp-server

# 其他工具可通过 MCP 协议调用 Codex 能力
```

#### ✅ Codex SDK

```typescript
import { Codex } from "@openai/codex-sdk";

const codex = new Codex();
const thread = codex.startThread();
const result = await thread.run("task");
```

### 1.3 Claude Code 的集成能力

#### ✅ 非交互式 CLI

```bash
# 基本调用
claude --print "prompt" --output-format json

# 会话管理
claude --print "prompt" --session-id <uuid>
claude --continue  # 继续上次会话
claude --resume <id>  # 恢复指定会话

# 预算控制
claude --max-budget-usd 5.0

# 结构化输出
claude --json-schema schema.json
```

**输出格式**：

```json
{
  "result": "响应内容",
  "session_id": "uuid",
  "total_cost_usd": 0.05,
  "usage": {
    "input_tokens": 100,
    "output_tokens": 200
  }
}
```

#### ✅ MCP Client 模式

```bash
# 加载 MCP 配置
claude --mcp-config path/to/config.json

# 严格模式（仅使用指定的 MCP 服务器）
claude --strict-mcp-config --mcp-config custom.json
```

#### ✅ 状态共享

- **配置目录**: `~/.claude/`
- **会话历史**: `~/.claude/history.jsonl`（JSONL 格式）
- **环境配置**: `~/.claude/settings.json`
- **计划文件**: `~/.claude/plans/`
- **任务列表**: `~/.claude/tasks/`

#### ❌ 不支持的功能

- **MCP Server 模式**: Claude Code 不能作为 MCP Server 暴露能力
- **SDK**: 没有 Python/JavaScript SDK

---

## 第二部分：集成方案设计

### 2.1 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    用户工作流                              │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Pi-mono    │    │ Claude Code  │    │  Codex-CLI   │
│  (主控制器)   │    │  (专家助手)   │    │  (专家助手)   │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        │                   │                   │
        │    ┌──────────────┴──────────────┐    │
        │    │                             │    │
        ▼    ▼                             ▼    ▼
┌─────────────────┐                ┌─────────────────┐
│  MCP 生态系统    │                │  CLI 包装器      │
│  (标准协议)      │                │  (直接调用)      │
└─────────────────┘                └─────────────────┘
```

### 2.2 集成方案矩阵

| 方案 | 技术路径 | 优势 | 劣势 | 推荐度 |
|------|---------|------|------|--------|
| **方案 A** | Pi-mono Extension + CLI 包装器 | 简单直接，立即可用 | 无上下文共享 | ⭐⭐⭐⭐⭐ |
| **方案 B** | MCP 协议集成 | 标准化，可扩展 | 需要配置 | ⭐⭐⭐⭐ |
| **方案 C** | Codex SDK 桥接 | 灵活控制 | 需要开发 | ⭐⭐⭐ |
| **方案 D** | 状态文件共享 | 可共享上下文 | 耦合度高 | ⭐⭐ |

### 2.3 推荐方案：混合架构

**核心思路**: Pi-mono 作为主控制器，通过 Extension 系统调用其他工具

#### 方案 A：CLI 包装器（立即可用）

**实现步骤**：

**1. 创建 Pi-mono Extension**: `~/.pi/extensions/ai-tools-bridge/`

```typescript
// ~/.pi/extensions/ai-tools-bridge/index.ts
import { execSync } from 'child_process';
import { z } from 'zod';

export default function(api: ExtensionAPI) {
  // 注册 Claude Code 工具
  api.registerTool({
    name: "call_claude_code",
    description: "Call Claude Code for complex refactoring or architectural tasks",
    schema: z.object({
      task: z.string().describe("Task description"),
      session_id: z.string().optional().describe("Session ID to continue"),
      max_budget: z.number().optional().describe("Max budget in USD")
    })
  }, async ({ task, session_id, max_budget }) => {
    let cmd = `claude --print "${task}" --output-format json`;
    if (session_id) cmd += ` --session-id ${session_id}`;
    if (max_budget) cmd += ` --max-budget-usd ${max_budget}`;

    const result = execSync(cmd, { encoding: 'utf-8' });
    const parsed = JSON.parse(result);

    return {
      result: parsed.result,
      session_id: parsed.session_id,
      cost: parsed.total_cost_usd
    };
  });

  // 注册 Codex-CLI 工具
  api.registerTool({
    name: "call_codex",
    description: "Call OpenAI Codex for specialized coding tasks",
    schema: z.object({
      task: z.string().describe("Task description"),
      output_schema: z.string().optional().describe("JSON schema for structured output")
    })
  }, async ({ task, output_schema }) => {
    let cmd = `codex exec --json "${task}"`;
    if (output_schema) cmd += ` --output-schema ${output_schema}`;

    const result = execSync(cmd, { encoding: 'utf-8' });
    return JSON.parse(result);
  });
}
```

**2. 安装 Extension**

```bash
# 创建目录
mkdir -p ~/.pi/extensions/ai-tools-bridge

# 复制代码
cp index.ts ~/.pi/extensions/ai-tools-bridge/

# 在 pi 中重载
pi
/reload
```

**3. 使用示例**

```bash
pi
> 帮我重构这个复杂的类，使用 Claude Code 来做
# Pi 会自动调用 call_claude_code 工具

> 用 Codex 生成一个符合 OpenAPI 规范的 API 定义
# Pi 会自动调用 call_codex 工具
```

#### 方案 B：MCP 协议集成（标准化）

**实现步骤**：

**1. 启动 Codex MCP Server**

```bash
# 在后台运行 Codex MCP Server
codex mcp-server &
```

**2. 创建 Pi-mono MCP Extension**

```typescript
// ~/.pi/extensions/mcp-bridge/index.ts
import { MCPClient } from '@modelcontextprotocol/sdk';

export default function(api: ExtensionAPI) {
  // 连接到 Codex MCP Server
  const codexClient = new MCPClient('http://localhost:3000');

  // 动态注册 Codex 的所有工具
  codexClient.listTools().then(tools => {
    tools.forEach(tool => {
      api.registerTool({
        name: `codex_${tool.name}`,
        description: tool.description,
        schema: tool.inputSchema
      }, async (params) => {
        return await codexClient.callTool(tool.name, params);
      });
    });
  });
}
```

**3. 配置 Claude Code 使用 Pi-mono 的 MCP 服务**

```json
// ~/.claude/mcp-config.json
{
  "mcpServers": {
    "pi-mono-tools": {
      "command": "pi-mcp-server",
      "args": ["--port", "3001"]
    }
  }
}
```

```bash
# 启动 Claude Code 并加载 MCP 配置
claude --mcp-config ~/.claude/mcp-config.json
```

### 2.4 任务路由策略

**根据任务类型选择合适的工具**：

| 任务类型 | 推荐工具 | 理由 |
|---------|---------|------|
| 快速原型开发 | Pi-mono | 极简工具集，快速迭代 |
| 复杂重构 | Claude Code | 强大的代码理解和重构能力 |
| API 设计 | Codex-CLI | OpenAI 模型擅长结构化输出 |
| 多文件协调 | Claude Code | 更好的上下文管理 |
| 单文件快速修改 | Pi-mono | 最小化开销 |
| 架构设计 | Claude Code | Plan Mode 支持 |

**实现任务路由**：

```typescript
// ~/.pi/extensions/task-router/index.ts
export default function(api: ExtensionAPI) {
  api.registerTool({
    name: "smart_delegate",
    description: "Intelligently delegate task to the best AI tool",
    schema: z.object({
      task: z.string(),
      task_type: z.enum(['prototype', 'refactor', 'api_design', 'architecture'])
    })
  }, async ({ task, task_type }) => {
    const routing = {
      'prototype': 'pi-mono',
      'refactor': 'claude-code',
      'api_design': 'codex',
      'architecture': 'claude-code'
    };

    const tool = routing[task_type];

    if (tool === 'claude-code') {
      return await callClaudeCode(task);
    } else if (tool === 'codex') {
      return await callCodex(task);
    } else {
      return { message: "Handling with pi-mono directly" };
    }
  });
}
```

---

## 第三部分：个人工作流配置

### 3.1 日常开发场景配置

#### 场景 1：快速代码修改（Pi-mono）

```bash
# 启动 pi
pi

# 快速修改
> 修复这个 bug
> 添加一个新函数
> 重命名这个变量
```

**优势**：
- 极简工具集（read/write/edit/bash）
- 快速响应
- 低 token 消耗

#### 场景 2：复杂重构（Claude Code）

```bash
# 启动 Claude Code
claude

# 复杂任务
> 重构整个认证模块
> 将这个类拆分成多个文件
> 优化这个算法的性能
```

**优势**：
- 强大的代码理解
- Plan Mode 支持
- 多文件协调

#### 场景 3：API 设计（Codex-CLI）

```bash
# 非交互式调用
codex exec --output-schema openapi.json "设计一个用户管理 API"

# 或在 pi 中调用
pi
> 用 Codex 设计一个 RESTful API
```

**优势**：
- 结构化输出
- OpenAI 模型擅长 API 设计
- 可集成到 CI/CD

### 3.2 推荐工作流

#### 工作流 1：Pi-mono 为主，按需委托

```
1. 日常开发使用 Pi-mono
   ↓
2. 遇到复杂任务时，Pi 自动识别并询问
   ↓
3. 用户确认后，Pi 调用 Claude Code 或 Codex
   ↓
4. 结果返回给 Pi，继续工作
```

**配置**：

```typescript
// ~/.pi/extensions/auto-delegate/index.ts
export default function(api: ExtensionAPI) {
  // 监听用户输入
  api.on('user_message', async (message) => {
    // 检测复杂任务关键词
    const complexKeywords = ['重构', '架构', '设计', '优化'];
    const isComplex = complexKeywords.some(kw => message.includes(kw));

    if (isComplex) {
      // 询问用户是否委托
      const response = await api.askUser({
        question: "这个任务看起来比较复杂，是否使用 Claude Code 处理？",
        options: ["是", "否，继续用 Pi"]
      });

      if (response === "是") {
        // 调用 Claude Code
        const result = await callClaudeCode(message);
        return result;
      }
    }
  });
}
```

#### 工作流 2：并行使用多个工具

```bash
# 终端 1：Pi-mono（快速迭代）
pi

# 终端 2：Claude Code（复杂任务）
claude

# 终端 3：Codex-CLI（API 设计）
codex exec "设计 API"
```

**状态同步**：

```bash
# 共享 git 仓库
# 每个工具都在同一个项目目录工作
# 通过 git 同步状态

# Pi-mono 修改文件
pi
> 修改 user.ts

# Claude Code 看到变化
claude
> 基于最新的 user.ts 重构认证模块
```

### 3.3 配置文件模板

#### Pi-mono 配置

```json
// .pi/settings.json
{
  "provider": "anthropic",
  "model": "claude-opus-4",
  "extensions": [
    "ai-tools-bridge",
    "task-router",
    "auto-delegate"
  ],
  "context_files": [
    "README.md",
    "ARCHITECTURE.md"
  ]
}
```

#### Claude Code 配置

```json
// ~/.claude/settings.json
{
  "env": {
    "ANTHROPIC_MODEL": "claude-opus-4.6"
  },
  "enabledPlugins": {
    "superpowers-marketplace": true
  }
}
```

#### MCP 配置（共享）

```json
// ~/mcp-config.json
{
  "mcpServers": {
    "codex": {
      "command": "codex",
      "args": ["mcp-server"]
    },
    "context7": {
      "command": "npx",
      "args": ["-y", "@context7/mcp-server"]
    },
    "figma": {
      "command": "npx",
      "args": ["-y", "@figma/mcp-server-figma"]
    }
  }
}
```

### 3.4 快捷命令配置

```bash
# ~/.bashrc 或 ~/.zshrc

# 快速启动 Pi-mono
alias pi='pi'

# 快速启动 Claude Code
alias cc='claude'

# 快速启动 Codex
alias cx='codex exec'

# 启动 Pi-mono 并加载 MCP 配置
alias pi-mcp='pi --mcp-config ~/mcp-config.json'

# 启动 Claude Code 并加载 MCP 配置
alias cc-mcp='claude --mcp-config ~/mcp-config.json'

# 非交互式调用 Claude Code
function cc-exec() {
  claude --print "$1" --output-format json | jq -r '.result'
}

# 非交互式调用 Codex
function cx-exec() {
  codex exec --json "$1" | jq
}
```

---

## 第四部分：实施步骤

### 阶段 1：基础集成（1-2 天）

**目标**：实现 Pi-mono 调用 Claude Code 和 Codex-CLI

**步骤**：

1. ✅ 确认所有工具已安装
   ```bash
   pi --version
   claude --version
   codex --version
   ```

2. ✅ 创建 Pi-mono Extension 目录
   ```bash
   mkdir -p ~/.pi/extensions/ai-tools-bridge
   ```

3. ✅ 实现 CLI 包装器 Extension
   - 复制上面的 `index.ts` 代码
   - 添加错误处理
   - 添加日志记录

4. ✅ 测试基本调用
   ```bash
   pi
   /reload
   > 测试调用 Claude Code
   ```

### 阶段 2：MCP 集成（3-5 天）

**目标**：通过 MCP 协议实现标准化集成

**步骤**：

1. ✅ 启动 Codex MCP Server
   ```bash
   codex mcp-server --port 3000
   ```

2. ✅ 创建 MCP Bridge Extension
   - 实现 MCP 客户端
   - 动态注册工具
   - 处理连接错误

3. ✅ 配置 Claude Code MCP Client
   - 创建 MCP 配置文件
   - 测试连接
   - 验证工具调用

4. ✅ 测试端到端流程
   ```bash
   # 终端 1：启动 Codex MCP Server
   codex mcp-server

   # 终端 2：启动 Pi-mono
   pi

   # 终端 3：启动 Claude Code
   claude --mcp-config ~/mcp-config.json
   ```

### 阶段 3：任务路由（2-3 天）

**目标**：实现智能任务分发

**步骤**：

1. ✅ 实现任务分类器
   - 关键词检测
   - 任务复杂度评估
   - 工具能力匹配

2. ✅ 实现自动委托逻辑
   - 用户确认机制
   - 结果聚合
   - 错误回退

3. ✅ 测试不同场景
   - 简单修改 → Pi-mono
   - 复杂重构 → Claude Code
   - API 设计 → Codex-CLI

### 阶段 4：工作流优化（持续）

**目标**：根据实际使用优化配置

**步骤**：

1. ✅ 收集使用数据
   - 任务类型分布
   - 工具选择准确率
   - 响应时间

2. ✅ 优化路由策略
   - 调整关键词
   - 优化阈值
   - 添加新规则

3. ✅ 改进用户体验
   - 减少确认步骤
   - 优化输出格式
   - 添加快捷命令

---

## 第五部分：验证计划

### 5.1 功能验证

**测试用例 1：Pi-mono 调用 Claude Code**

```bash
# 在 Pi-mono 中
pi
> 使用 Claude Code 重构 src/auth.ts 模块

# 预期结果：
# 1. Pi 识别到需要调用 Claude Code
# 2. 执行 CLI 调用
# 3. 返回重构结果
# 4. 应用到代码库
```

**测试用例 2：Pi-mono 调用 Codex-CLI**

```bash
# 在 Pi-mono 中
pi
> 用 Codex 设计一个用户管理 API，输出 OpenAPI 规范

# 预期结果：
# 1. Pi 调用 Codex
# 2. 返回结构化的 OpenAPI JSON
# 3. 保存到文件
```

**测试用例 3：MCP 协议集成**

```bash
# 启动 Codex MCP Server
codex mcp-server

# 在 Claude Code 中
claude --mcp-config ~/mcp-config.json
> 列出可用的 MCP 工具

# 预期结果：
# 显示 Codex 提供的工具列表
```

### 5.2 性能验证

**指标**：

- **响应时间**：CLI 调用 < 5s，MCP 调用 < 3s
- **成功率**：工具调用成功率 > 95%
- **成本**：每次调用成本 < $0.10

**测试方法**：

```bash
# 性能测试脚本
#!/bin/bash

echo "测试 Pi-mono 调用 Claude Code"
time pi --print "简单任务" --call-claude-code

echo "测试 Pi-mono 调用 Codex"
time pi --print "简单任务" --call-codex

echo "测试 MCP 调用"
time claude --mcp-config ~/mcp-config.json --print "列出工具"
```

### 5.3 用户体验验证

**评估标准**：

- ✅ 工具切换是否流畅
- ✅ 错误提示是否清晰
- ✅ 结果是否符合预期
- ✅ 配置是否简单

**用户测试**：

1. 邀请 3-5 位开发者测试
2. 收集反馈
3. 迭代优化

---

## 第六部分：关键文件清单

### 需要创建的文件

1. **Pi-mono Extensions**
   - `~/.pi/extensions/ai-tools-bridge/index.ts`
   - `~/.pi/extensions/mcp-bridge/index.ts`
   - `~/.pi/extensions/task-router/index.ts`
   - `~/.pi/extensions/auto-delegate/index.ts`

2. **配置文件**
   - `~/.pi/settings.json`
   - `~/mcp-config.json`
   - `~/.bashrc` 或 `~/.zshrc`（快捷命令）

3. **测试脚本**
   - `~/test-integration.sh`
   - `~/benchmark.sh`

4. **文档**
   - `~/docs/integration-guide.md`
   - `~/docs/troubleshooting.md`

### 需要修改的文件

- 无（所有集成通过 Extension 实现，不修改源码）

---

## 第七部分：风险与缓解

### 风险 1：工具版本不兼容

**缓解措施**：
- 在 Extension 中检测版本
- 提供降级方案
- 文档中明确版本要求

### 风险 2：MCP 连接不稳定

**缓解措施**：
- 实现重连机制
- 添加超时处理
- 提供 CLI 包装器作为备选

### 风险 3：成本超支

**缓解措施**：
- 设置预算限制
- 监控 API 调用
- 优化任务路由

### 风险 4：上下文丢失

**缓解措施**：
- 通过 git 同步状态
- 共享配置文件
- 实现会话恢复

---

## 总结

### 核心价值

1. **灵活性**：根据任务选择最合适的工具
2. **效率**：自动化任务分发，减少手动切换
3. **标准化**：通过 MCP 协议实现可扩展的集成
4. **成本优化**：Pi-mono 处理简单任务，降低 API 成本

### 下一步行动

1. ✅ 实现基础 CLI 包装器 Extension
2. ✅ 测试 Pi-mono 调用 Claude Code
3. ✅ 测试 Pi-mono 调用 Codex-CLI
4. ✅ 配置 MCP 集成
5. ✅ 优化任务路由策略
6. ✅ 编写使用文档

### 预期成果

一个统一的 AI 编码助手工作流，能够：
- 在 Pi-mono 中无缝调用 Claude Code 和 Codex-CLI
- 根据任务类型自动选择最佳工具
- 通过 MCP 协议实现标准化集成
- 提供流畅的开发体验

---

**文档结束**

