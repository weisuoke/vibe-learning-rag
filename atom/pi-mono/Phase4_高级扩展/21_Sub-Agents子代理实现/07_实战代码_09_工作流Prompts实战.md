# 工作流Prompts实战 - 实战代码

> **知识点**: Sub-Agents 子代理实现 - 工作流Prompts设计与优化
> **难度**: ⭐⭐⭐⭐⭐
> **前置知识**: 基础单代理执行、链式工作流、提示工程基础

---

## 1. 场景描述

工作流Prompts是Sub-Agents系统的核心，决定了代理的行为质量、协作效率和输出可靠性。优秀的Prompt设计能让代理准确理解任务、高效协作、产出符合预期的结果。

**典型应用场景**:
- Scout-and-Plan工作流（侦察与规划）
- Implement-and-Review工作流（实现与审查）
- Parallel-Specialist工作流（并行专家）
- Iterative-Refinement工作流（迭代精化）

根据 [Lakera 2026 Prompt Engineering Guide](https://www.lakera.ai/blog/prompt-engineering-guide)，清晰的结构和上下文比巧妙的措辞更重要。[PromptEngineering.org 2026 Playbook](https://promptengineering.org/agents-at-work-the-2026-playbook-for-building-reliable-agentic-workflows) 强调：Structured Outputs + Verification 是通往可靠、低漂移代理的最快路径。

---

## 2. 核心概念

### 2.1 Prompt组件

| 组件 | 作用 | 示例 |
|------|------|------|
| **Role** | 定义代理身份 | "你是一个代码审查专家" |
| **Context** | 提供背景信息 | "当前项目使用TypeScript" |
| **Task** | 明确任务目标 | "审查以下代码的安全性" |
| **Constraints** | 设置约束条件 | "只关注OWASP Top 10漏洞" |
| **Format** | 指定输出格式 | "以JSON格式返回结果" |
| **Examples** | 提供示例 | "示例输出：{...}" |

### 2.2 工作流模式

根据 [Anthropic Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system):

1. **从代理视角思考** - 代理需要什么信息？
2. **任务委托** - 明确每个代理的职责
3. **根据复杂度扩展** - 简单查询用简单代理

### 2.3 ReAct范式

**参考**: [ReAct Paper (arXiv:2210.03629)](https://arxiv.org/abs/2210.03629)

**核心思想**: 交替推理和行动，减少幻觉

```
Thought: 我需要搜索文档
Action: search_documents(query="...")
Observation: 找到3个相关文档
Thought: 现在我需要提取关键信息
Action: extract_info(doc_id=1)
Observation: 提取成功
```

---

## 3. 完整代码示例

### 3.1 Scout-and-Plan工作流

```typescript
import { ExtensionAPI } from "@pi-mono/extension-api";

/**
 * Scout-and-Plan工作流Prompts
 * 
 * 参考:
 * - https://www.lakera.ai/blog/prompt-engineering-guide
 * - https://promptengineering.org/agents-at-work-the-2026-playbook
 */

export const ScoutPrompt = `
你是一个快速侦察代理（Scout Agent），专门负责快速扫描代码库并识别关键信息。

## 你的职责
1. 快速浏览目标目录和文件
2. 识别关键文件、依赖关系和架构模式
3. 生成简洁的侦察报告

## 可用工具
- read: 读取文件内容
- grep: 搜索代码模式
- find: 查找文件
- ls: 列出目录内容

## 输出格式
以JSON格式返回侦察结果：
\`\`\`json
{
  "key_files": ["file1.ts", "file2.ts"],
  "dependencies": ["dep1", "dep2"],
  "architecture_patterns": ["pattern1", "pattern2"],
  "recommendations": ["建议1", "建议2"]
}
\`\`\`

## 约束条件
- 侦察时间不超过30秒
- 只关注与任务相关的文件
- 不要深入阅读文件内容，只需识别关键信息

## 当前任务
{task_description}

## 目标目录
{target_directory}
`;

export const PlannerPrompt = `
你是一个规划代理（Planner Agent），负责基于侦察结果制定详细的执行计划。

## 你的职责
1. 分析Scout Agent的侦察报告
2. 将目标分解为可执行的步骤
3. 为每个步骤分配合适的Worker Agent
4. 定义步骤间的依赖关系

## 输入
侦察报告：
{scout_report}

## 输出格式
以JSON格式返回执行计划：
\`\`\`json
{
  "goal": "任务目标",
  "steps": [
    {
      "id": 1,
      "description": "步骤描述",
      "agent": "worker-type",
      "dependencies": [],
      "success_criteria": "成功标准"
    }
  ],
  "estimated_complexity": "low|medium|high"
}
\`\`\`

## 规划原则
1. 步骤应该是原子性的（单一职责）
2. 明确定义每步的成功标准
3. 考虑步骤间的依赖关系
4. 为复杂任务预留验证步骤

## 当前目标
{goal_description}
`;

export const WorkerPrompt = `
你是一个执行代理（Worker Agent），负责执行计划中的具体步骤。

## 你的职责
1. 严格按照计划执行分配的步骤
2. 使用指定的工具完成任务
3. 验证输出是否满足成功标准
4. 报告执行结果和遇到的问题

## 当前步骤
{step_description}

## 成功标准
{success_criteria}

## 可用工具
{available_tools}

## 输出格式
以JSON格式返回执行结果：
\`\`\`json
{
  "status": "success|failed|partial",
  "output": "执行结果",
  "artifacts": ["生成的文件或数据"],
  "issues": ["遇到的问题"],
  "next_steps": ["建议的后续步骤"]
}
\`\`\`

## 执行原则
1. 严格遵循成功标准
2. 遇到问题立即报告，不要猜测
3. 保持输出格式一致
4. 记录所有重要决策
`;

export const ReviewerPrompt = `
你是一个审查代理（Reviewer Agent），负责验证Worker Agent的执行结果。

## 你的职责
1. 检查输出是否满足成功标准
2. 验证代码质量、安全性和最佳实践
3. 提供改进建议
4. 决定是否需要返工

## 审查维度
1. **正确性**: 输出是否符合预期？
2. **完整性**: 是否遗漏了关键部分？
3. **质量**: 代码/文档质量如何？
4. **安全性**: 是否存在安全隐患？
5. **可维护性**: 是否易于理解和维护？

## 输入
执行结果：
{worker_output}

成功标准：
{success_criteria}

## 输出格式
以JSON格式返回审查结果：
\`\`\`json
{
  "approved": true|false,
  "score": 0-100,
  "findings": [
    {
      "severity": "critical|high|medium|low",
      "category": "correctness|completeness|quality|security|maintainability",
      "description": "问题描述",
      "suggestion": "改进建议"
    }
  ],
  "decision": "approve|revise|reject",
  "revision_instructions": "返工指令（如果需要）"
}
\`\`\`

## 审查原则
1. 客观公正，基于事实
2. 提供具体、可操作的建议
3. 区分关键问题和次要问题
4. 鼓励最佳实践
`;
```

### 3.2 Implement-and-Review工作流

```typescript
/**
 * Implement-and-Review工作流Prompts
 * 
 * 适用场景：需要质量保证的代码生成
 */

export const ImplementerPrompt = `
你是一个实现代理（Implementer Agent），负责编写高质量的代码。

## 你的职责
1. 根据需求编写代码
2. 遵循项目的编码规范
3. 添加必要的注释和文档
4. 确保代码可测试

## 编码规范
{coding_standards}

## 当前任务
{task_description}

## 技术栈
{tech_stack}

## 输出格式
以JSON格式返回实现结果：
\`\`\`json
{
  "files": [
    {
      "path": "文件路径",
      "content": "文件内容",
      "description": "文件说明"
    }
  ],
  "tests": [
    {
      "path": "测试文件路径",
      "content": "测试内容"
    }
  ],
  "documentation": "实现说明"
}
\`\`\`

## 实现原则
1. 代码简洁、可读
2. 遵循SOLID原则
3. 避免过度工程
4. 考虑边界情况
5. 编写可测试的代码
`;

export const CodeReviewerPrompt = `
你是一个代码审查专家（Code Reviewer Agent），负责审查代码质量。

## 审查清单

### 1. 代码质量
- [ ] 代码是否清晰易读？
- [ ] 是否遵循项目编码规范？
- [ ] 是否有适当的注释？
- [ ] 是否避免了代码重复？

### 2. 安全性
- [ ] 是否存在SQL注入风险？
- [ ] 是否存在XSS漏洞？
- [ ] 是否正确处理用户输入？
- [ ] 是否安全存储敏感信息？

### 3. 性能
- [ ] 是否存在性能瓶颈？
- [ ] 是否有不必要的计算？
- [ ] 是否正确使用缓存？

### 4. 可测试性
- [ ] 代码是否易于测试？
- [ ] 是否有足够的测试覆盖？
- [ ] 测试是否有意义？

## 输入
实现结果：
{implementation}

## 输出格式
以JSON格式返回审查结果：
\`\`\`json
{
  "overall_score": 0-100,
  "issues": [
    {
      "file": "文件路径",
      "line": 行号,
      "severity": "critical|high|medium|low",
      "category": "quality|security|performance|testability",
      "description": "问题描述",
      "suggestion": "改进建议",
      "code_snippet": "相关代码片段"
    }
  ],
  "positive_aspects": ["优点1", "优点2"],
  "decision": "approve|request_changes|reject"
}
\`\`\`

## 审查原则
参考 OWASP Top 10 和项目编码规范
`;

export const FixerPrompt = `
你是一个修复代理（Fixer Agent），负责根据审查反馈修复代码问题。

## 你的职责
1. 理解审查反馈
2. 修复发现的问题
3. 保持代码风格一致
4. 验证修复效果

## 审查反馈
{review_feedback}

## 原始代码
{original_code}

## 输出格式
以JSON格式返回修复结果：
\`\`\`json
{
  "fixed_files": [
    {
      "path": "文件路径",
      "content": "修复后的内容",
      "changes": "修改说明"
    }
  ],
  "addressed_issues": ["已解决的问题ID"],
  "remaining_issues": ["未解决的问题及原因"]
}
\`\`\`

## 修复原则
1. 只修复审查中指出的问题
2. 不要引入新的问题
3. 保持代码风格一致
4. 添加注释说明修复原因
`;
```

### 3.3 Structured Outputs集成

```typescript
import { z } from "zod";

/**
 * 使用Zod定义结构化输出Schema
 * 
 * 参考:
 * - https://platform.openai.com/docs/guides/structured-outputs
 * - https://docs.anthropic.com/claude/docs/structured-outputs
 */

// Scout报告Schema
export const ScoutReportSchema = z.object({
  key_files: z.array(z.string()).describe("关键文件列表"),
  dependencies: z.array(z.string()).describe("依赖关系"),
  architecture_patterns: z.array(z.string()).describe("架构模式"),
  recommendations: z.array(z.string()).describe("建议"),
});

// 执行计划Schema
export const ExecutionPlanSchema = z.object({
  goal: z.string().describe("任务目标"),
  steps: z.array(
    z.object({
      id: z.number().describe("步骤ID"),
      description: z.string().describe("步骤描述"),
      agent: z.string().describe("负责的代理类型"),
      dependencies: z.array(z.number()).describe("依赖的步骤ID"),
      success_criteria: z.string().describe("成功标准"),
    })
  ),
  estimated_complexity: z.enum(["low", "medium", "high"]).describe("预估复杂度"),
});

// 审查结果Schema
export const ReviewResultSchema = z.object({
  approved: z.boolean().describe("是否批准"),
  score: z.number().min(0).max(100).describe("评分"),
  findings: z.array(
    z.object({
      severity: z.enum(["critical", "high", "medium", "low"]).describe("严重程度"),
      category: z.string().describe("问题类别"),
      description: z.string().describe("问题描述"),
      suggestion: z.string().describe("改进建议"),
    })
  ),
  decision: z.enum(["approve", "revise", "reject"]).describe("决策"),
  revision_instructions: z.string().optional().describe("返工指令"),
});

/**
 * 集成到Subagent Extension
 */
export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "subagent_workflow",
    description: "Execute structured workflow with prompts",
    parameters: z.object({
      workflow: z.enum(["scout-and-plan", "implement-and-review"]),
      task: z.string(),
      context: z.record(z.any()).optional(),
    }),
    
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      const { workflow, task, context = {} } = params;
      
      if (workflow === "scout-and-plan") {
        // 1. Scout阶段
        const scoutPrompt = ScoutPrompt
          .replace("{task_description}", task)
          .replace("{target_directory}", context.target_directory || ".");
        
        const scoutResult = await executeAgent({
          agent: "scout",
          prompt: scoutPrompt,
          outputSchema: ScoutReportSchema,
        });
        
        onUpdate?.({
          type: "progress",
          content: `✅ Scout完成：发现${scoutResult.key_files.length}个关键文件`,
        });
        
        // 2. Plan阶段
        const plannerPrompt = PlannerPrompt
          .replace("{scout_report}", JSON.stringify(scoutResult, null, 2))
          .replace("{goal_description}", task);
        
        const plan = await executeAgent({
          agent: "planner",
          prompt: plannerPrompt,
          outputSchema: ExecutionPlanSchema,
        });
        
        onUpdate?.({
          type: "progress",
          content: `✅ Plan完成：生成${plan.steps.length}个执行步骤`,
        });
        
        // 3. Execute阶段
        const results = [];
        for (const step of plan.steps) {
          const workerPrompt = WorkerPrompt
            .replace("{step_description}", step.description)
            .replace("{success_criteria}", step.success_criteria)
            .replace("{available_tools}", JSON.stringify(ctx.availableTools));
          
          const result = await executeAgent({
            agent: step.agent,
            prompt: workerPrompt,
          });
          
          results.push(result);
          
          onUpdate?.({
            type: "progress",
            content: `✅ 步骤${step.id}完成：${step.description}`,
          });
        }
        
        // 4. Review阶段
        const reviewPrompt = ReviewerPrompt
          .replace("{worker_output}", JSON.stringify(results, null, 2))
          .replace("{success_criteria}", plan.steps.map(s => s.success_criteria).join("\n"));
        
        const review = await executeAgent({
          agent: "reviewer",
          prompt: reviewPrompt,
          outputSchema: ReviewResultSchema,
        });
        
        return {
          type: "success" as const,
          output: {
            scout: scoutResult,
            plan,
            results,
            review,
          },
        };
      }
      
      // 其他工作流...
      throw new Error(`Unknown workflow: ${workflow}`);
    },
  });
}

async function executeAgent(config: any): Promise<any> {
  // 实际的代理执行逻辑
  // 这里简化为示例
  return {};
}
```

---

## 4. 代码解析

### 4.1 Prompt模板化

**关键技术**:
- 使用占位符 `{variable}` 实现动态替换
- 分离角色、任务、约束、格式等组件
- 支持多语言和多模型

**示例**:
```typescript
const prompt = ScoutPrompt
  .replace("{task_description}", task)
  .replace("{target_directory}", dir);
```

### 4.2 Structured Outputs

**优势**:
- 100% Schema合规性
- 类型安全
- 自动验证
- 下游系统可直接解析

**实现**:
```typescript
const result = await executeAgent({
  prompt: scoutPrompt,
  outputSchema: ScoutReportSchema, // Zod schema
});
```

### 4.3 工作流编排

**模式**:
1. Scout → Plan → Execute → Review（顺序执行）
2. 每阶段输出作为下一阶段输入
3. 使用 `onUpdate` 提供实时进度
4. 结构化输出确保数据一致性

---

## 5. 实际案例分析

### 案例1: Anthropic多代理研究系统

**来源**: [Anthropic Engineering Blog](https://www.anthropic.com/engineering/multi-agent-research-system)

**系统架构**:
- Research Coordinator: 协调整体研究流程
- Literature Scout: 搜索相关文献
- Data Analyst: 分析数据
- Report Writer: 撰写研究报告

**关键经验**:
1. **从代理视角思考**: 代理需要什么信息才能完成任务？
2. **任务委托**: 明确每个代理的职责边界
3. **根据复杂度扩展**: 简单查询用简单代理，复杂查询用多代理协作

### 案例2: 发票对账自动化

**来源**: [PromptEngineering.org Playbook](https://promptengineering.org/agents-at-work-the-2026-playbook)

**业务目标**: 每周五下午4点前将上周发票与账本对账，并将异常邮件发送给应付账款

**工作流Prompts**:

```typescript
const ParsePrompt = `
解析发票文件 {invoice_file}，提取以下字段：
- invoice_id
- vendor
- amount
- date
- line_items

输出JSON格式，严格遵循Schema。
`;

const MatchPrompt = `
将发票 {invoice_id} 与采购订单匹配：
1. 查找对应的PO
2. 验证金额一致性
3. 检查日期合理性

返回匹配结果和置信度。
`;

const VerifyPrompt = `
验证总额，检查：
1. 舍入规则
2. 税务计算
3. 折扣应用

如果发现不一致，标记为异常。
`;
```

**控制措施**:
- ERP API速率限制
- 最小权限API密钥（NIST AC-6）
- 运行成本上限

### 案例3: Lakera提示工程最佳实践

**来源**: [Lakera 2026 Guide](https://www.lakera.ai/blog/prompt-engineering-guide)

**核心发现**:
1. **清晰结构 > 巧妙措辞**: 大多数Prompt失败来自歧义，而非模型限制
2. **不同模型不同格式**: 没有通用最佳实践
3. **安全风险**: Prompt工程也是潜在的安全风险（对抗性提示）

**推荐技术**:
- Chain-of-Thought Reasoning
- Format Constraints
- Prompt Scaffolding
- Multi-Turn Memory Prompting

---

## 6. 最佳实践

### 6.1 Prompt设计原则

**清晰性**:
```typescript
// ❌ 模糊
"分析这个文件"

// ✅ 清晰
"分析 src/index.ts，识别：1) 导出的函数 2) 依赖的模块 3) 潜在的性能问题"
```

**结构化**:
```typescript
// ✅ 使用JSON Schema
const prompt = `
输出格式：
\`\`\`json
{
  "functions": ["func1", "func2"],
  "dependencies": ["dep1", "dep2"],
  "issues": [{"type": "performance", "description": "..."}]
}
\`\`\`
`;
```

**上下文管理**:
```typescript
// ✅ 只包含必要上下文
const prompt = `
当前项目：TypeScript + React
编码规范：Airbnb Style Guide
任务：${task}
`;
```

### 6.2 工作流组合策略

**何时使用静态工作流**:
- 流程可预测（如CI/CD）
- 任务顺序固定
- 需要确定性

**何时使用代理工作流**:
- 输入数据混乱
- 需要自适应规划
- 分支逻辑复杂

**混合方法**:
```typescript
// 使用Airflow处理固定管道
// 让代理处理需要适应的部分
if (isComplexQuery) {
  return await agentWorkflow(query);
} else {
  return await staticPipeline(query);
}
```

### 6.3 验证策略

**三层验证**:
1. **输入验证**: Schema检查、范围验证
2. **中间验证**: 每步输出检查、健全性测试
3. **最终验证**: 完整性检查、业务规则验证

```typescript
// 输入验证
const input = InputSchema.parse(rawInput);

// 中间验证
for (const step of steps) {
  const result = await executeStep(step);
  StepResultSchema.parse(result); // 验证
}

// 最终验证
const output = FinalOutputSchema.parse(aggregatedResults);
```

---

## 7. 常见问题

### Q1: 如何处理Prompt漂移？

**A**: 使用Structured Outputs强制格式：
```typescript
const schema = z.object({
  field1: z.string(),
  field2: z.number(),
});

const result = await executeAgent({
  prompt,
  outputSchema: schema, // 强制Schema合规
});
```

### Q2: 如何优化Prompt长度？

**A**: 
1. 使用Prompt Compression技术
2. 只包含必要上下文
3. 使用语义压缩（摘要嵌入）
4. 利用Prompt Caching

### Q3: 如何处理多语言Prompt？

**A**:
```typescript
const prompts = {
  en: EnglishPrompt,
  zh: ChinesePrompt,
};

const prompt = prompts[language] || prompts.en;
```

### Q4: 如何测试Prompt质量？

**A**:
1. 单元测试：验证输出格式
2. 集成测试：验证工作流完整性
3. A/B测试：比较不同Prompt版本
4. 人工评估：抽查输出质量

### Q5: 如何防止Prompt注入？

**A**:
1. 输入验证和清理
2. 使用Prompt Scaffolding
3. 分离用户输入和系统指令
4. 参考OWASP LLM Top 10

---

## 8. 扩展阅读

### 核心资源

1. [Lakera Prompt Engineering Guide 2026](https://www.lakera.ai/blog/prompt-engineering-guide) - 2026年提示工程终极指南
2. [PromptEngineering.org 2026 Playbook](https://promptengineering.org/agents-at-work-the-2026-playbook) - 可靠代理工作流实战手册
3. [Anthropic Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system) - 多代理研究系统构建经验
4. [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) - 结构化输出文档
5. [Claude Structured Outputs](https://docs.anthropic.com/claude/docs/structured-outputs) - Claude结构化输出

### 学术论文

6. [ReAct Paper (arXiv:2210.03629)](https://arxiv.org/abs/2210.03629) - ReAct范式论文
7. [LangChain Plan-and-Execute](https://python.langchain.com/docs/tutorials/plan-and-execute/) - 规划与执行模式

### 相关知识点

- `03_核心概念_09_Agent定义格式.md` - Agent定义规范
- `07_实战代码_03_链式工作流实现.md` - 链式工作流
- `07_实战代码_08_错误处理与恢复.md` - 错误处理
- `07_实战代码_10_生产部署与最佳实践.md` - 生产部署

---

**版本**: v1.0
**最后更新**: 2026-02-21
**作者**: Claude Code (基于2025-2026最新实践)
