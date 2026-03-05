---
type: search_result
search_query: agentevals trajectory evaluation + CoT prompting enhancement
search_engine: grok-mcp
searched_at: 2026-03-01
knowledge_point: 04_Agent推理与决策
---

# 搜索结果：Agent 评估与 CoT 增强

## Agent 评估搜索摘要

### 1. LangSmith 轨迹评估官方指南
- **来源**: https://docs.langchain.com/langsmith/trajectory-evals
- agentevals 包官方指南
- 支持 trajectory match 和 LLM judge 方法
- 测试 ReAct 等代理的执行序列和质量

### 2. langchain-ai/agentevals GitHub
- **来源**: https://github.com/langchain-ai/agentevals
- 现成代理轨迹评估器
- trajectory match、LLM-as-judge、graph trajectory
- 支持 ReAct 代理质量度量

### 3. 深度代理评估经验 (LangChain Blog, 2025)
- **来源**: https://blog.langchain.com/evaluating-deep-agents-our-learnings
- 轨迹评估、工具调用序列
- 最终响应质量指标
- 适用于 ReAct 代理优化

### 4. AI Agent 工程状态报告 (LangChain, 2025-2026)
- **来源**: https://www.langchain.com/state-of-agent-engineering
- 离线/在线评估采用率
- LLM-as-judge 使用趋势
- 轨迹评估等代理质量度量

### 5. ReAct 代理基准测试 (LangChain Blog, 2025)
- **来源**: https://blog.langchain.com/react-agent-benchmarking
- 工具数量和指令对性能影响
- 轨迹准确率等指标
- 模型质量比较

### 6. Multi-turn Evals (LangSmith, 2025)
- **来源**: https://blog.langchain.com/insights-agent-multiturn-evals-langsmith
- Multi-turn Evals 评估完整代理对话轨迹
- 语义结果和工具调用路径
- ReAct 代理质量监控

## CoT 增强搜索摘要

### 1. LangChain 高级提示指南
- **来源**: https://bridgephase.com/insights/advanced-prompting-with-langchain
- Few-Shot Prompting 和 CoT 提示技术
- Few-Shot CoT 示例实现
- 提升复杂推理能力

### 2. Few-Shot Prompting 最佳实践
- **来源**: https://www.digitalocean.com/community/tutorials/_few-shot-prompting-techniques-examples-best-practices
- FewShotPromptTemplate 使用指南
- OpenAI API 和 LangChain 实现

### 3. CoT 提示详解
- **来源**: https://www.codecademy.com/article/chain-of-thought-cot-prompting
- Zero-Shot CoT、Few-Shot CoT、Auto-CoT
- LangChain 中的实现示例

### 4. Few-Shot 提升工具调用性能 (LangChain Blog)
- **来源**: https://blog.langchain.com/few-shot-prompting-to-improve-tool-calling-performance
- Few-Shot Prompting 显著提升工具调用准确性
- ReAct 工作流中的推理准确性提升

### 5. 提示优化探索 (LangChain Blog)
- **来源**: https://blog.langchain.com/exploring-prompt-optimization
- Few-Shot Prompting 在提示优化中的应用
- Meta-Prompting 结合提升推理性能

## 关键信息提取

### 评估维度
1. **轨迹评估 (Trajectory Evaluation)**：评估 agent 的推理路径是否合理
2. **工具调用准确性**：是否选择了正确的工具和参数
3. **最终响应质量**：最终答案的正确性和完整性
4. **效率指标**：推理步骤数、token 消耗、延迟

### CoT 增强技术
1. **Zero-Shot CoT**：添加 "Let's think step by step" 即可
2. **Few-Shot CoT**：提供推理示例，引导 LLM 模仿
3. **Auto-CoT**：自动生成推理链示例
4. **Few-Shot Tool Calling**：提供工具调用示例，提升准确性
