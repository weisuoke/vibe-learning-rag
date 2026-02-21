# 安全与确认流程 - 2025-2026 实际案例研究

> **研究日期**: 2026-02-21
> **来源**: Grok-mcp web_search + web_fetch
> **查询数量**: 3 个搜索查询 + 2 个成功抓取

---

## 搜索查询记录

### Query 1: AI agent security confirmation flow TypeScript 2025 2026 site:github.com

**关键发现**:
- ruvnet/claude-flow - 企业级安全,提示注入防护
- ruvnet/agentic-flow - 生产就绪编排,213个MCP工具
- VoltAgent/ai-agent-examples - JWT认证示例
- forter/trusted-agentic-commerce-protocol - JWS+JWE安全认证
- westonbrown/Cyber-AutoAgent - 工具确认提示
- mondaycom/agent-tool-protocol - 安全沙箱执行
- awslabs/agent-squad - 多代理管理框架
- PredicateSystems/sdk-typescript - 验证优先运行时

### Query 2: Agent trust boundaries security patterns 2025 site:x.com OR site:reddit.com

**关键发现**:
- Reddit: 2025年AI agent安全事件分析
- Reddit: 代理AI可信度问题
- Reddit: AI代理已被入侵警告
- X.com: 2026年安全焦点转向信任边界
- X.com: 企业agentic AI安全架构
- X.com: Onchain agents可编程信任
- X.com: AI agent自治边界
- X.com: 代理架构中的信任边界

### Query 3: User consent patterns AI agents production 2026

**关键发现**:
- Smashing Magazine: 代理AI的UX模式
- Curity: 用户同意最佳实践
- Protecto.ai: AI对用户同意的影响
- Orochi Network: 2026隐私趋势
- OneTrust: 2026隐私、AI与信任
- Anthropic: 测量AI代理自治
- Redis: 2026 AI代理架构
- Oso: AI代理授权最佳实践

---

## 案例 1: Anthropic - 测量AI代理自治的实际模式

**来源**: [Anthropic Research - Measuring AI agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy)
**作者**: Miles McCain, Thomas Millar, Saffron Huang, et al.
**发布日期**: 2026-02-18

### 核心研究发现

Anthropic 分析了数百万次人机交互(Claude Code + 公共API),研究人们如何实际使用代理。

#### 1. Claude Code 自主工作时间增长

**关键数据**:
- 99.9th percentile turn duration 在3个月内几乎翻倍
- 从 <25 分钟增长到 >45 分钟
- 增长平滑跨越模型发布,不仅是能力提升
- 现有模型能力超过实际使用的自主程度

**启示**: 存在显著的"部署悬垂"(deployment overhang),模型能处理的自主程度超过实际行使的。

#### 2. 经验用户的自动批准模式

**新用户 vs 经验用户**:
- **新用户**: ~20% 的会话使用完全自动批准
- **经验用户**: >40% 的会话使用完全自动批准
- **增长模式**: 随经验逐渐增加,表明信任的稳定积累

**关键洞察**: 经验用户从逐步审查转向自主监控,仅在需要时干预。

#### 3. 中断率随经验增加

**反直觉发现**:
- **新用户**: 5% 的 turn 中断 Claude
- **经验用户**: 9% 的 turn 中断 Claude
- **解释**: 经验用户让 Claude 自主工作,但在出错时更主动干预

**两种监督策略**:
1. **新用户**: 逐步批准每个操作(事前控制)
2. **经验用户**: 自主运行 + 主动中断(事中控制)

#### 4. 代理主动暂停 vs 人类中断

**关键发现**:
- 在最复杂任务上,Claude Code 主动暂停请求澄清的频率是人类中断的 2 倍以上
- 代理主动停止是部署系统中的重要监督形式

**启示**: 有效监督不需要批准每个操作,而是能在关键时刻干预。

#### 5. 代理在风险领域的使用

**公共API使用分布**:
- **软件工程**: 近50%的代理活动
- **新兴领域**: 医疗、金融、网络安全
- **风险评估**: 大多数操作低风险且可逆

**结论**: 代理在风险领域使用,但尚未大规模部署。

### 关键指标

| 指标 | 新用户 | 经验用户 | 变化 |
|------|--------|---------|------|
| 自动批准率 | ~20% | >40% | +100% |
| 中断率 | 5% | 9% | +80% |
| Turn 时长(99.9th) | <25 min | >45 min | +80% |

### 对Pi-mono的启示

1. **渐进式授权**: 默认需要批准,允许用户逐步启用自动批准
2. **主动暂停**: 代理在不确定时应主动请求澄清
3. **监控而非批准**: 经验用户倾向于监控 + 干预,而非逐步批准
4. **信任积累**: 信任是通过经验逐步建立的,不是一次性授予的

---

## 案例 2: Smashing Magazine - 代理AI的UX模式

**来源**: [Smashing Magazine - Designing For Agentic AI](https://www.smashingmagazine.com/2026/02/designing-agentic-ai-practical-ux-patterns)
**作者**: Victor Yocco
**发布日期**: 2026-02-11

### 6个核心UX模式

#### 1. Intent Preview (意图预览)

**定义**: 在代理执行任何重要操作前,用户必须清楚了解即将发生什么。

**心理基础**: 呈现计划减少认知负荷,消除惊讶,给用户验证代理理解的时刻。

**有效Intent Preview的要素**:
- **清晰简洁**: 用简单语言总结主要操作和结果
- **顺序步骤**: 多步操作应概述关键阶段
- **明确用户操作**: 提供清晰的选择集(Proceed / Edit Plan / Handle it Myself)

**示例**:
```
Proposed Plan for Your Trip Disruption
I've detected that your 10:05 AM flight has been canceled. Here's what I plan to do:
1. Cancel Flight UA456 - Process refund
2. Rebook on Flight DL789 - 2:30 PM non-stop
3. Update Hotel Reservation - Notify late arrival
4. Email Updated Itinerary - Send to you and Jane Doe

[ Proceed with this Plan ] [ Edit Plan ] [ Handle it Myself ]
```

**何时优先使用**:
- 不可逆操作(删除数据)
- 财务交易
- 信息共享
- 重大变更

**成功指标**:
- **接受率**: Plans Accepted Without Edit / Total Plans > 85%
- **覆盖频率**: Handle it Myself Clicks / Total Plans < 10%
- **回忆准确性**: 用户能正确列出计划步骤

#### 2. Autonomy Dial (自主拨盘)

**定义**: 渐进式授权机制,允许用户调整代理的自主程度。

**实现方式**:
- **Level 1**: 建议(Suggest) - 仅提供建议
- **Level 2**: 草稿(Draft) - 准备操作但不执行
- **Level 3**: 执行并通知(Execute & Notify) - 执行后通知
- **Level 4**: 完全自主(Fully Autonomous) - 自主执行

**心理基础**: 给用户控制感,允许根据任务风险和信任程度调整。

#### 3. Explainable Rationale (可解释理由)

**定义**: 代理工作时保持透明,展示"为什么"。

**实现方式**:
- 显示决策理由
- 引用数据来源
- 解释选择逻辑

#### 4. Confidence Signal (置信度信号)

**定义**: 显示代理对其操作的确定程度。

**实现方式**:
- 高置信度(>90%): 绿色指示器
- 中置信度(70-90%): 黄色指示器
- 低置信度(<70%): 红色指示器 + 请求人工审查

#### 5. Action Audit & Undo (操作审计与撤销)

**定义**: 为错误提供安全网。

**实现方式**:
- 完整操作日志
- 一键撤销
- 批量回滚

#### 6. Escalation Pathway (升级路径)

**定义**: 高模糊性时刻的处理机制。

**实现方式**:
- 自动检测高风险操作
- 暂停并请求人工审查
- 提供上下文和建议

### 成功指标汇总

| 模式 | 关键指标 | 目标值 |
|------|---------|--------|
| Intent Preview | 接受率 | >85% |
| Intent Preview | 覆盖频率 | <10% |
| Autonomy Dial | 拨盘调整频率 | 监控趋势 |
| Confidence Signal | 低置信度操作准确率 | >95% |
| Action Audit | 撤销使用率 | <5% |
| Escalation | 升级准确率 | >90% |

---

## 其他重要发现

### GitHub 安全项目

1. **ruvnet/claude-flow**
   - [GitHub](https://github.com/ruvnet/claude-flow)
   - 企业级安全
   - 提示注入防护
   - 输入验证
   - 威胁检测 AIDefence
   - 安全审计代理

2. **forter/trusted-agentic-commerce-protocol**
   - [GitHub](https://github.com/forter/trusted-agentic-commerce-protocol)
   - JWS+JWE 安全认证
   - 代理身份验证
   - 关系确认

3. **mondaycom/agent-tool-protocol**
   - [GitHub](https://github.com/mondaycom/agent-tool-protocol)
   - 安全沙箱 V8 VM
   - TypeScript/JavaScript 执行
   - 来源跟踪防御提示注入

4. **VoltAgent/ai-agent-examples**
   - [GitHub](https://github.com/VoltAgent/ai-agent-examples)
   - JWT 令牌验证
   - 角色访问控制
   - 多租户认证流程

### Reddit 安全讨论

1. **2025年AI agent安全事件分析**
   - [Reddit](https://www.reddit.com/r/cybersecurity/comments/1r79rye/)
   - 区分真实、夸大内容
   - OWASP 多代理攻击模式
   - 框架中的范围机制

2. **代理AI可信度问题**
   - [Reddit](https://www.reddit.com/r/AI_Agents/comments/1p1c522/)
   - 代理不会故意违反权限
   - 设计导致边界模糊
   - 信任失效

3. **AI代理已被入侵警告**
   - [Reddit](https://www.reddit.com/r/AI_Agents/comments/1o7xuhf/)
   - 接入真实数据后易被入侵
   - 传统安全工具难以应对
   - 需要严格信任边界

### X.com 安全趋势

1. **2026年安全焦点转向信任边界**
   - [X.com](https://x.com/chandika/status/2025120075607605660)
   - 而非输出过滤
   - 提示注入转为执行问题
   - 需要硬权限和出口控制

2. **Onchain agents可编程信任**
   - [X.com](https://x.com/FAIR_Blockchain/status/1988313033723220329)
   - 智能合约实现
   - 强制执行托管规则
   - 支出限额和白名单

3. **AI agent自治边界**
   - [X.com](https://x.com/Scobleizer/status/2018832685511459208)
   - 定义清晰边界
   - 定期审查权限
   - 降低风险

### 2026最佳实践资源

1. **Curity - User Consent Best Practices**
   - [Curity Blog](https://curity.io/blog/user-consent-best-practices-in-the-age-of-ai-agents)
   - 细粒度、时限同意
   - 代理视为第三方应用
   - 明确授权
   - 授予和撤销访问指南

2. **Oso - Best Practices of Authorizing AI Agents**
   - [Oso Learn](https://www.osohq.com/learn/best-practices-of-authorizing-ai-agents)
   - OAuth 2.1 和 MCP 协议
   - 安全授权
   - 动态权限
   - 同意管理
   - 避免静态访问控制

3. **Redis - AI Agent Architecture 2026**
   - [Redis Blog](https://redis.io/blog/ai-agent-architecture)
   - 授权、观测性
   - 人类监督路径
   - 高风险决策中的用户同意

---

## 关键洞察总结

### 用户同意模式的演进

1. **从逐步批准到自主监控**
   - 新用户: 逐步批准每个操作
   - 经验用户: 自主运行 + 主动中断
   - 信任通过经验逐步建立

2. **从事前控制到事中控制**
   - 事前: 批准每个操作(新用户)
   - 事中: 监控 + 干预(经验用户)
   - 事后: 审计 + 撤销(所有用户)

3. **从单一模式到渐进式授权**
   - Level 1: 建议
   - Level 2: 草稿
   - Level 3: 执行并通知
   - Level 4: 完全自主

### 信任边界的关键要素

1. **明确的权限范围**
   - 定义代理可以做什么
   - 定义代理不能做什么
   - 定义需要确认的操作

2. **透明的决策过程**
   - 显示决策理由
   - 引用数据来源
   - 解释选择逻辑

3. **有效的监督机制**
   - 代理主动暂停
   - 用户主动中断
   - 操作审计与撤销

4. **渐进式信任建立**
   - 从低自主开始
   - 根据经验增加自主
   - 允许用户调整自主程度

### Pi-mono的对应关系

| Pi-mono 概念 | Anthropic 研究 | Smashing Magazine |
|-------------|---------------|-------------------|
| `confirmProjectAgents` | 新用户逐步批准 | Intent Preview |
| Agent scope (user/project) | 信任边界 | Trust boundaries |
| Streaming updates | 透明性 | Explainable Rationale |
| Stop reason 传播 | 代理主动暂停 | Escalation Pathway |
| Usage tracking | 监控 | Action Audit |

### 实现优先级

**高优先级**(不可逆/财务/共享):
1. Intent Preview - 操作前确认
2. Action Audit & Undo - 操作后撤销
3. Escalation Pathway - 高风险升级

**中优先级**(复杂任务):
1. Autonomy Dial - 渐进式授权
2. Confidence Signal - 置信度显示

**低优先级**(增强体验):
1. Explainable Rationale - 决策解释

---

**研究完成时间**: 2026-02-21
**总搜索查询**: 3 个
**总详细抓取**: 2 个成功
**总案例数**: 15+ 个实际案例
**覆盖来源**: Anthropic (1), Smashing Magazine (1), GitHub (8+), Reddit (3+), X.com (3+), 企业博客 (5+)
