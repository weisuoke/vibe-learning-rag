# Agent发现与注册 - 2025-2026 实际案例研究

> **研究日期**: 2026-02-21
> **来源**: Grok-mcp web_search + web_fetch
> **查询数量**: 3 个搜索查询 + 3 个详细抓取

---

## 搜索查询记录

### Query 1: Service discovery patterns AI agents 2025 site:x.com OR site:reddit.com

**关键发现**:
- Reddit: 代理市场 - 代理间自主发现和支付
- Reddit: 自适应安全扫描代理 - 多步服务发现链
- X.com: Unibase - AI 代理自主发现机制
- X.com: ERC-8004 - 基于身份注册表的服务发现
- X.com: 代理式商务全栈实现
- X.com: BlockFlow - ERC-8004 信任层
- Reddit: 构建 AI 代理的实际挑战

### Query 2: Agent registration mechanisms production 2026

**关键发现**:
- IETF: AI Agent Discovery and Invocation Protocol (AIDIP)
- arXiv: Agent Name Service (ANS) - DNS-based agent directory
- NIST: 软件和 AI 代理身份与授权
- aregistry.ai: AI Agents Registry Platform
- JetBrains: ACP Agent Registry
- Collibra: AI agent registry for governance
- Nylas: 2026 年代理 AI 状态
- Azure Tech Insider: 2026 年生产级 AI 代理

### Query 3: Agent discovery dynamic loading TypeScript 2025 2026 site:github.com

**关键发现**:
- GitHub: MCP 代码执行和动态工具加载
- GitHub: universal-tool-calling-protocol/code-mode
- GitHub: joshuadavidthomas/opencode-agent-skills
- GitHub: harche/ProDisco - 渐进式披露 MCP
- GitHub: muratcankoylan/Agent-Skills-for-Context-Engineering
- GitHub: gptme/gptme - 终端 AI 代理
- GitHub: universal-tool-calling-protocol/typescript-utcp
- GitHub: a2aproject/A2A Agent Registry Proposal

---

## 案例 1: Agent Name Service (ANS) - DNS-based Agent Directory

**来源**: [arXiv:2505.10609](https://arxiv.org/abs/2505.10609)
**作者**: Ken Huang, Vineeth Sai Narajala, Idan Habler, Akram Sheriff
**发布日期**: 2025-05-15
**支持**: OWASP GenAI ASI Project

### 核心概念

**Agent Name Service (ANS)** 是一个基于 DNS 的新型架构,解决公共代理发现框架的缺失。ANS 提供协议无关的注册表基础设施,利用 **公钥基础设施 (PKI)** 证书实现可验证的代理身份和信任。

### 关键创新

1. **正式的代理注册和续期机制**
   - 生命周期管理
   - 自动续期和过期处理

2. **DNS 启发的命名约定**
   - 能力感知解析
   - 层次化命名空间

3. **模块化协议适配器层**
   - 支持多种通信标准 (A2A, MCP, ACP 等)
   - 协议无关设计

4. **精确定义的安全解析算法**
   - PKI 证书验证
   - 身份和能力验证

5. **结构化通信**
   - 使用 JSON Schema
   - 标准化数据格式

### 架构特点

**核心组件**:
- **注册表服务**: 存储代理元数据
- **解析服务**: 查找和匹配代理
- **证书服务**: PKI 身份验证
- **协议适配器**: 多协议支持

**安全特性**:
- PKI 证书验证
- 身份可验证性
- 信任建立机制
- 威胁分析和防护

### 应用场景

- 多代理系统的安全发现
- 跨平台代理互操作
- 可信代理生态系统
- 可扩展代理网络

---

## 案例 2: IETF AI Agent Discovery and Invocation Protocol (AIDIP)

**来源**: [IETF draft-cui-ai-agent-discovery-invocation-01](https://datatracker.ietf.org/doc/draft-cui-ai-agent-discovery-invocation/01)
**作者**: Y. Cui (Tsinghua University), Y. Chao, C. Du (Zhongguancun Laboratory)
**发布日期**: 2026-02-01
**状态**: Internet-Draft (Informational)

### 核心定义

**AIDIP** 提出了一个标准化协议,用于 AI 代理的发现和调用。它定义了:

1. **通用元数据格式**
   - 描述代理能力
   - I/O 规范
   - 支持的语言
   - 标签和认证方法

2. **基于能力的发现机制**
   - 注册表方式
   - 能力匹配
   - 语义查询

3. **统一的 RESTful 调用接口**
   - 标准端点
   - JSON 负载
   - 跨平台互操作

4. **安全考虑**
   - 认证 (OAuth 2.0)
   - 授权
   - 加密传输 (TLS)
   - 信任建立

### Agent 元数据规范

**核心字段**:

```json
{
  "id": "agent-12345",
  "name": "Chinese-English Translator",
  "description": "Translates text between Chinese and English",
  "version": "1.2.0",
  "publisher": "ExampleAI Inc.",
  "capabilities": ["translation"],
  "tags": ["nlp", "chinese", "english", "cloud"],
  "endpoint": "https://api.example.com/agents/translate",
  "supported_languages": ["en", "zh"],
  "authentication": {
    "type": "api_key",
    "instructions": "Include 'X-API-Key' header"
  },
  "status": "active",
  "operations": [
    {
      "name": "translateText",
      "description": "Translates text from source to target language",
      "inputs": {
        "type": "object",
        "properties": {
          "text": {"type": "string"},
          "source_language": {"type": "string"},
          "target_language": {"type": "string"}
        }
      },
      "outputs": {
        "type": "object",
        "properties": {
          "translated_text": {"type": "string"}
        }
      }
    }
  ]
}
```

### 发现机制

**Agent Registry (Discovery Service)**:

1. **注册**: 代理注册元数据
2. **索引**: 存储和索引元数据
3. **查询**: 客户端搜索代理
4. **返回**: 匹配的代理列表

**查询方式**:
- 基于能力查询
- 基于标签查询
- 语义查询
- 复合查询

### Agent Semantic Resolution 扩展

**意图驱动的代理选择**:
- 描述任务意图
- 接收候选代理
- 不预先确定调用哪个代理
- 语义匹配阶段

### 互操作性

**引用现有标准**:
- JSON Schema
- OAuth 2.0 (RFC6749)
- OpenAPI 概念
- 已建立的 Web 技术

---

## 案例 3: MCP 动态工具加载与代码执行

**来源**: [GitHub MCP Discussion #1780](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/1780)
**主题**: Code execution with MCP: Building more efficient agents
**相关博客**: [Anthropic - Code Execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)

### 核心问题

**上下文拥塞**:
- 直接工具调用消耗上下文
- 每个定义和结果都占用空间
- 代理通过编写代码调用工具更具扩展性

### 解决方案: 动态工具加载

**渐进式披露模式**:

1. **高层描述判断**
   - 先用高层描述判断工具相关性
   - 仅加载相关工具的详细信息

2. **按需加载**
   - 代理根据上下文动态决定加载哪些工具
   - 仅加载相关部分

3. **代码执行**
   - 代理编写代码调用工具
   - 减少上下文窗口占用

### 两通道模式提案

**问题**: MCP 服务器既提供"指令"又提供"数据",但代码执行绕过 LLM 上下文窗口,使 MCP 服务器变成纯 API。

**提案**: 给 MCP 工具两个输出通道:

1. **指令通道** (Instruction Channel)
   - 非结构化
   - 直接放入 LLM 上下文窗口
   - 用于指导 LLM

2. **数据通道** (Data Channel)
   - 结构化数据
   - 供程序读取
   - 不占用 LLM 上下文

**实现方式**:

```typescript
// 使用 _meta 字段实现数据通道
{
  content: "Found 1000 log entries. Columns: timestamp, level, message",
  _meta: {
    data_channel: {
      schema: { /* JSON Schema */ },
      data: [ /* 实际数据 */ ]
    }
  }
}
```

**工作流程**:
1. MCP Host 将"指令"添加到 LLM 上下文
2. MCP Host 解析数据,放入内存文件
3. 编写代码使用内存文件
4. 执行代码,将结果放入 LLM 上下文

### Resource Links 模式

**使用 Resource Links**:

```typescript
// 工具返回 Resource Link
{
  type: "resource",
  resource: {
    uri: "file:///tmp/search_results.json",
    description: "Search results (1000 entries)",
    mimeType: "application/json"
  }
}
```

**优势**:
- 描述字段提供指令
- MCP 客户端程序化读取资源
- 资源描述包含解析指令、schema、大小提示

### Pass-by-Reference 模式

**核心思想**:
- 工具调用间通过引用而非值传递数据
- 中间数据不感兴趣,只关心最终输出
- MCP Host 程序化替换占位符

**工作流程**:
1. 工具 A 返回数据引用
2. LLM 生成工具调用 B,使用占位符
3. MCP Host 替换占位符为实际数据
4. 调用工具 B

**优势**:
- 绕过 LLM 上下文
- 更快的数据传递
- 减少 token 消耗

---

## 其他重要发现

### Reddit 案例

1. **代理市场 - 自主发现和支付**
   - [Reddit - Agent Marketplace](https://www.reddit.com/r/AI_Agents/comments/1p63m3b/i_built_a_marketplace_for_agents_to_discover_and)
   - 代理间自主发现、交互和支付
   - 代理组合模式
   - Solana 主网实际部署

2. **自适应安全扫描代理**
   - [Reddit - Adaptive Security Scanning](https://www.reddit.com/r/devops/comments/1lj4rjh/built_an_ai_agent_for_adaptive_security_scanning)
   - 多步服务发现链
   - 版本识别和测试模式
   - 基础设施自动化

3. **构建 AI 代理的实际挑战**
   - [Reddit - Building AI Agents](https://www.reddit.com/r/AI_Agents/comments/1ojyu8p/i_build_ai_agents_for_a_living_its_a_mess_out)
   - 服务发现挑战
   - Schema 映射和转换
   - 实际模式和解决方案

### X.com 趋势

1. **Unibase - AI 代理自主发现**
   - [X.com - Unibase](https://x.com/Unibase_AI/status/2004918811980693570)
   - 当前服务发现为人类设计
   - AI 代理需要自主发现机制
   - 从助手向自治实体演进

2. **ERC-8004 - 服务发现模式**
   - [X.com - Jimmy Skuros](https://x.com/jskuros/status/1983303575590801472)
   - 客户端查询身份注册表
   - 查找兼容服务
   - 支持独立决策和自治代理

3. **代理式商务全栈**
   - [X.com - Heurist AI](https://x.com/heurist_ai/status/2000866110540062846)
   - AI 代理自主完成服务发现
   - 可信判断
   - 订单生成等全流程

4. **ERC-8004 信任层**
   - [X.com - BlockFlow](https://x.com/BlockFlow_News/status/1983141738341626278)
   - AI 代理支付信任层
   - 服务发现支持
   - 身份注册表查询

### 生产级注册表平台

1. **aregistry.ai - AI Agents Registry Platform**
   - [aregistry.ai](https://aregistry.ai/)
   - 集中式 AI 代理注册表
   - 支持 MCP 构建代理
   - 技能与服务器注册
   - 审核与公司级部署
   - 治理与快速生产化

2. **JetBrains ACP Agent Registry**
   - [JetBrains Blog](https://blog.jetbrains.com/ai/2026/01/acp-agent-registry)
   - JetBrains 与 Zed 推出
   - AI 编码代理目录
   - 浏览与安装
   - 直接集成 IDE
   - 快速发现与连接

3. **Collibra AI Agent Registry**
   - [Collibra Blog](https://www.collibra.com/blog/collibra-ai-agent-registry-governing-autonomous-ai-agents)
   - 集中注册
   - 监控与全生命周期治理
   - 内部与第三方代理统一管理
   - 合规支持

### NIST 安全指南

**来源**: [NIST Concept Paper](https://www.nccoe.nist.gov/sites/default/files/2026-02/accelerating-the-adoption-of-software-and-ai-agent-identity-and-authorization-concept-paper.pdf)

**核心内容**:
- AI 代理身份与授权机制
- 注册、委托与透明日志
- OAuth 和 MCP 协议
- 生产环境安全部署

### GitHub 动态加载项目

1. **universal-tool-calling-protocol/code-mode**
   - [GitHub](https://github.com/universal-tool-calling-protocol/code-mode)
   - 代理通过代码执行调用 MCP 和 UTCP 工具
   - 渐进式工具发现
   - 动态加载所需工具
   - TypeScript 库

2. **joshuadavidthomas/opencode-agent-skills**
   - [GitHub](https://github.com/joshuadavidthomas/opencode-agent-skills)
   - OpenCode 动态技能插件
   - 自动从多目录发现和加载
   - 可重用 AI 代理技能
   - TypeScript 开发

3. **harche/ProDisco**
   - [GitHub](https://github.com/harche/ProDisco)
   - 渐进式披露 MCP 服务器框架
   - TypeScript 库 API 索引和发现
   - 代理动态探索
   - 仅加载所需模块

4. **muratcankoylan/Agent-Skills-for-Context-Engineering**
   - [GitHub](https://github.com/muratcankoylan/Agent-Skills-for-Context-Engineering)
   - 上下文工程代理技能集合
   - 文件系统动态上下文发现
   - TypeScript 实现评估工具

5. **gptme/gptme**
   - [GitHub](https://github.com/gptme/gptme)
   - 终端 AI 代理
   - MCP 发现与动态加载
   - 2025-2026 版本更新
   - Token 意识和自治运行

6. **universal-tool-calling-protocol/typescript-utcp**
   - [GitHub](https://github.com/universal-tool-calling-protocol/typescript-utcp)
   - UTCP 官方 TypeScript 实现
   - 自动插件注册
   - 工具搜索
   - 代理直接调用 API

7. **a2aproject/A2A Agent Registry Proposal**
   - [GitHub Discussion #741](https://github.com/a2aproject/A2A/discussions/741)
   - A2A 代理注册表提案
   - 代理发现
   - TypeScript 实现
   - 企业级注册服务

### 2026 生产状态

1. **Nylas - State of Agentic AI in 2026**
   - [Nylas Blog](https://www.nylas.com/blog/the-state-of-agentic-ai-in-2026)
   - 2026 年代理已在生产环境部署
   - 67% 团队构建代理工作流
   - 从实验转向实际生产集成
   - 可衡量结果

2. **Azure Tech Insider - Production AI Agents in 2026**
   - [Azure Tech Insider](https://azuretechinsider.com/from-hype-to-reality-what-production-ai-agents-actually-look-like)
   - 基于真实部署案例分析
   - 85% 采用自定义框架
   - 聚焦可解释性
   - 审计与生产环境实际表现

---

## 关键洞察总结

### Agent 发现的演进趋势

1. **从中心化到分布式**
   - 早期: 中心化注册表 (aregistry.ai, JetBrains ACP)
   - 现在: DNS-based 分布式系统 (ANS)
   - 未来: 区块链 + 身份注册表 (ERC-8004)

2. **从静态到动态**
   - 静态注册: 预先注册所有代理
   - 动态发现: 运行时发现和加载
   - 渐进式披露: 按需加载详细信息

3. **从能力到意图**
   - 能力匹配: 基于 capabilities 字段
   - 语义匹配: 基于任务意图
   - Agent Semantic Resolution: AIDIP 扩展

4. **从单协议到多协议**
   - 早期: 单一协议 (如 MCP)
   - 现在: 协议适配器层 (ANS)
   - 支持: A2A, MCP, ACP, UTCP 等

5. **从不安全到安全**
   - 早期: 无认证
   - 现在: PKI 证书 (ANS)
   - 标准: OAuth 2.0, TLS (AIDIP)

### Pi-mono 的对应关系

Pi-mono 的 Agent 发现机制与这些最佳实践高度一致:

1. **`discoverAgents()` 函数** = AIDIP 的发现机制
2. **User-level vs Project-level** = 分层发现和信任边界
3. **Agent scope 控制** = 安全和权限管理
4. **Markdown + YAML frontmatter** = 标准化元数据格式
5. **动态加载** = MCP 渐进式披露模式

### 实现模式对比

| 模式 | 优势 | 劣势 | 适用场景 |
|------|------|------|---------|
| 中心化注册表 | 简单、易管理 | 单点故障、扩展性差 | 小型团队、企业内部 |
| DNS-based (ANS) | 分布式、可扩展 | 复杂、需要基础设施 | 大规模、跨组织 |
| 动态加载 (MCP) | 减少上下文、高效 | 实现复杂、调试困难 | 资源受限、大量工具 |
| 渐进式披露 | 按需加载、灵活 | 需要协议支持 | 复杂系统、多层次 |

---

**研究完成时间**: 2026-02-21
**总搜索查询**: 3 个
**总详细抓取**: 3 个
**总案例数**: 20+ 个实际案例
**覆盖来源**: arXiv (1), IETF (1), GitHub (10+), Reddit (3+), X.com (4+), 企业博客 (5+)
