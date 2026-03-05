# CallbackHandler回调系统 - 生成计划

## 数据来源记录

### 源码分析
- ✅ reference/source_callback_01.md - CallbackHandler 核心架构分析
  - Mixin 架构设计（6个 Mixin）
  - BaseCallbackHandler 设计
  - 回调管理器架构
  - 内置回调处理器
  - 回调参数标准化
  - 异步支持

### Context7 官方文档
- ✅ reference/context7_langchain_01.md - LangChain 可观测性与追踪
  - LangSmith 追踪系统
  - @traceable 装饰器
  - OpenAI 客户端包装
  - 事件流（astream_events, astream_log）
- ✅ reference/context7_langchain_02.md - LangChain 自定义回调处理器
  - 自定义异步回调处理器
  - 回调处理器与 LLMChain 集成
  - 第三方回调处理器集成
  - Agent 回调监控

### 网络搜索
- ✅ reference/search_callback_01.md - LangChain 可观测性与追踪最佳实践
  - 可观测性工具生态（Langfuse, LangSmith, Phoenix, AgentReplay）
  - 最佳实践主题
  - 生产环境集成
- ✅ reference/search_callback_02.md - LangChain 自定义 CallbackHandler 教程
  - 自定义 CallbackHandler 实现模式
  - 常见使用场景（流式输出、成本追踪、可观测性集成、工具调用监控）
  - 常见问题和解决方案

### 待抓取链接（将由第三方工具自动保存到 reference/）
- [ ] https://github.com/pinecone-io/examples/blob/master/learn/generation/langchain/handbook/09-langchain-streaming/09-langchain-streaming.ipynb
- [ ] https://github.com/FareedKhan-dev/agentic-rag
- [ ] https://github.com/langfuse/langfuse-docs/blob/main/cookbook/integration_azure_openai_langchain.ipynb
- [ ] https://github.com/langfuse/langfuse/issues/6761
- [ ] https://github.com/orgs/langfuse/discussions/10711
- [ ] https://gist.github.com/ninely/88485b2e265d852d3feb8bd115065b1a
- [ ] https://github.com/orgs/langfuse/discussions/11934
- [ ] https://www.reddit.com/r/LangChain/comments/19enjxr/streaming_local_llm_with_fastapi_llamacpp_and/
- [ ] https://github.com/langfuse/langfuse-docs/blob/main/cookbook/example_langgraph_agents.ipynb
- [ ] https://github.com/vysotin/agentic_evals_docs
- [ ] https://github.com/alirezadir/Agentic-AI-Systems/blob/main/03_system_design/evals/agentic-ai-evals.md
- [ ] https://github.com/agentreplay/agentreplay
- [ ] https://github.com/FareedKhan-dev/production-grade-agentic-system
- [ ] https://github.com/Harmeet10000/langchain-fastapi-production
- [ ] https://www.reddit.com/r/LangChain/comments/1neh5sw/what_are_the_best_open_source_llm_observability/
- [ ] https://www.reddit.com/r/LangChain/comments/1f6kl5z/whats_more_important_observability_or_evaluations/

## 文件清单

### 基础维度文件
- [ ] 00_概览.md
- [ ] 01_30字核心.md
- [ ] 02_第一性原理.md

### 核心概念文件（按使用场景组织：基础→自定义→可观测性→生产）

#### 基础使用
- [ ] 03_核心概念_1_回调接口与Mixin架构.md - 回调接口设计、6个Mixin职责划分 [来源: 源码]
- [ ] 03_核心概念_2_内置回调处理器.md - StdOutCallbackHandler、StreamingStdOutCallbackHandler [来源: 源码]
- [ ] 03_核心概念_3_回调管理器系统.md - CallbackManager、RunManager、回调传递 [来源: 源码]
- [ ] 03_核心概念_4_回调生命周期.md - on_*_start/end/error/new_token 方法 [来源: 源码]

#### 自定义回调
- [ ] 03_核心概念_5_自定义回调处理器实现.md - 继承BaseCallbackHandler、实现特定方法 [来源: Context7 + 网络]
- [ ] 03_核心概念_6_异步回调处理器.md - AsyncCallbackHandler、异步方法实现 [来源: 源码 + Context7]

#### 可观测性集成
- [ ] 03_核心概念_7_LangSmith追踪系统.md - @traceable装饰器、wrap_openai [来源: Context7]
- [ ] 03_核心概念_8_事件流与追踪.md - astream_events、astream_log、trace_as_chain_group [来源: Context7]
- [ ] 03_核心概念_9_第三方可观测性平台.md - Langfuse、Phoenix、AgentReplay [来源: 网络]

#### 生产环境（新增独立概念）
- [ ] 03_核心概念_10_LangGraph回调处理.md - 状态机回调、Trace链接、Span管理 [来源: 网络]
- [ ] 03_核心概念_11_Agent工具调用追踪.md - 工具调用监控、参数传递、推理过程记录 [来源: Context7 + 网络]

### 基础维度文件（续）
- [ ] 04_最小可用.md
- [ ] 05_双重类比.md
- [ ] 06_反直觉点.md

### 实战代码文件（按用户选择的场景）

#### 场景1：基础使用和自定义日志（推荐）
- [ ] 07_实战代码_场景1_内置回调处理器使用.md - StdOutCallbackHandler、StreamingStdOutCallbackHandler [来源: 源码 + Context7]
- [ ] 07_实战代码_场景2_自定义日志回调实现.md - 继承BaseCallbackHandler、记录LLM调用详情 [来源: Context7 + 网络]

#### 场景2：流式输出处理（推荐）
- [ ] 07_实战代码_场景3_FastAPI流式回调集成.md - 异步流式回调、AsyncIterator [来源: 网络]
- [ ] 07_实战代码_场景4_本地LLM流式输出.md - Llama.cpp集成、流式输出队列管理 [来源: 网络]

#### 场景3：成本追踪和性能监控（推荐）
- [ ] 07_实战代码_场景5_Token用量统计回调.md - 自定义TokenCostCallbackHandler [来源: 网络]
- [ ] 07_实战代码_场景6_延迟和成本计算.md - 端到端性能监控 [来源: 网络]

#### 场景4：可观测性集成（可选）
- [ ] 07_实战代码_场景7_Langfuse集成实战.md - Langfuse CallbackHandler配置 [来源: 网络]
- [ ] 07_实战代码_场景8_LangSmith追踪实战.md - @traceable装饰器使用 [来源: Context7]

### 基础维度文件（续）
- [ ] 08_面试必问.md
- [ ] 09_化骨绵掌.md
- [ ] 10_一句话总结.md

## 知识点拆解方案（基于用户反馈）

### 核心概念组织方式
**选择**：按使用场景（基础→自定义→可观测性→生产）

**理由**：
- 从简单使用到复杂场景，逐步进阶
- 适合实战导向的学习路径
- 符合初学者的认知曲线

### 实战场景选择
**选择**：
1. 基础使用和自定义日志（推荐）✅
2. 流式输出处理（推荐）✅
3. 成本追踪和性能监控（推荐）✅
4. 可观测性集成（可选）

**理由**：
- 覆盖最常见的生产环境需求
- 提供完整的代码示例
- 平衡基础和进阶内容

### LangGraph 和 Agent 回调处理
**选择**：作为独立核心概念深入讲解（推荐）

**理由**：
- LangGraph 是 2025-2026 的重要趋势
- 值得单独讲解，不应简化
- 提供完整的技术深度

### 并发环境回调管理
**选择**：作为反直觉点重点讲解（推荐）

**理由**：
- 并发冲突是常见陷阱
- 适合放在反直觉点强调
- 提供实用的解决方案

## 核心概念详细说明

### 1. 回调接口与Mixin架构
**内容**：
- BaseCallbackHandler 的设计哲学
- 6个 Mixin 的职责划分：
  - RetrieverManagerMixin
  - LLMManagerMixin
  - ChainManagerMixin
  - ToolManagerMixin
  - CallbackManagerMixin
  - RunManagerMixin
- 回调方法的标准化参数（run_id, parent_run_id, tags, metadata）

**数据来源**：源码分析

### 2. 内置回调处理器
**内容**：
- StdOutCallbackHandler：标准输出回调
- StreamingStdOutCallbackHandler：流式输出回调
- FileCallbackHandler：文件回调
- UsageMetadataCallbackHandler：使用统计回调

**数据来源**：源码分析

### 3. 回调管理器系统
**内容**：
- CallbackManager 的作用
- RunManager 的层次结构
- 回调的传递和继承机制
- trace_as_chain_group 上下文管理器

**数据来源**：源码分析

### 4. 回调生命周期
**内容**：
- on_*_start 方法：LLM、Chain、Tool、Retriever 开始时触发
- on_*_end 方法：结束时触发
- on_*_error 方法：错误时触发
- on_llm_new_token 方法：流式输出时触发

**数据来源**：源码分析

### 5. 自定义回调处理器实现
**内容**：
- 继承 BaseCallbackHandler
- 实现特定回调方法
- 在初始化时或运行时传入
- 多个回调处理器组合

**数据来源**：Context7 + 网络

### 6. 异步回调处理器
**内容**：
- AsyncCallbackHandler 的使用
- 异步方法实现（async def）
- FastAPI 集成示例

**数据来源**：源码 + Context7

### 7. LangSmith追踪系统
**内容**：
- @traceable 装饰器
- wrap_openai 包装
- 流式模型追踪
- 自定义 metadata 和 tags

**数据来源**：Context7

### 8. 事件流与追踪
**内容**：
- astream_events：流式事件监听
- astream_log：回调日志流
- trace_as_chain_group：链组追踪
- 过滤选项（include_names, include_types, include_tags）

**数据来源**：Context7

### 9. 第三方可观测性平台
**内容**：
- Langfuse：最常用的开源平台
- LangSmith：官方平台
- Phoenix：开源替代方案
- AgentReplay：本地调试工具

**数据来源**：网络搜索

### 10. LangGraph回调处理（新增独立概念）
**内容**：
- LangGraph 状态机回调
- Trace 链接问题和解决方案
- Span 管理和嵌套
- @observe 装饰器使用

**数据来源**：网络搜索

### 11. Agent工具调用追踪（新增独立概念）
**内容**：
- 工具调用监控
- on_tool_start/on_tool_end 方法重写
- 工具参数传递
- Agent 推理过程记录

**数据来源**：Context7 + 网络

## 实战场景详细说明

### 场景1：内置回调处理器使用
**内容**：
- StdOutCallbackHandler 基础使用
- StreamingStdOutCallbackHandler 流式输出
- 在 LLM 和 Chain 中传递回调

**代码示例**：完整可运行的 Python 代码

### 场景2：自定义日志回调实现
**内容**：
- 继承 BaseCallbackHandler
- 实现 on_llm_start/on_llm_end
- 记录 LLM 调用详情
- 自定义日志格式

**代码示例**：完整可运行的 Python 代码

### 场景3：FastAPI流式回调集成
**内容**：
- 自定义 AsyncIteratorCallbackHandler
- FastAPI 流式响应
- 队列管理
- 仅追加 final answer

**代码示例**：完整可运行的 Python 代码

### 场景4：本地LLM流式输出
**内容**：
- Llama.cpp 集成
- 自定义流式回调
- 问题解决和调试

**代码示例**：完整可运行的 Python 代码

### 场景5：Token用量统计回调
**内容**：
- 自定义 TokenCostCallbackHandler
- 实时追踪 token 用量
- 计算成本
- 端到端延迟统计

**代码示例**：完整可运行的 Python 代码

### 场景6：延迟和成本计算
**内容**：
- 性能监控回调
- 延迟计算
- 成本估算
- 生产环境优化

**代码示例**：完整可运行的 Python 代码

### 场景7：Langfuse集成实战
**内容**：
- Langfuse CallbackHandler 初始化
- 追踪配置
- Prompt 版本控制
- 评估集成

**代码示例**：完整可运行的 Python 代码

### 场景8：LangSmith追踪实战
**内容**：
- @traceable 装饰器使用
- wrap_openai 包装
- 自定义 metadata
- 追踪查看

**代码示例**：完整可运行的 Python 代码

## 反直觉点详细说明

### 反直觉点1：并发环境回调管理（重点）
**错误观点**：回调处理器可以在多个请求之间重用

**为什么错**：
- 回调处理器可能包含状态
- 并发请求会导致状态冲突
- trace_id 混乱

**正确理解**：
- 每个请求新建 CallbackHandler 实例
- 避免重用导致冲突
- 使用 last_trace_id 获取追踪 ID
- 调用 flush() 确保数据持久化

**代码示例**：Django/FastAPI 并发处理

### 反直觉点2：LangGraph trace 链接问题
**错误观点**：LangGraph 会自动正确链接 trace

**为什么错**：
- 自动生成的 trace 可能链接不正确
- Prompt 链接可能丢失

**正确理解**：
- 使用 @observe 装饰器
- 正确配置 prompt 链接
- 使用 start_as_current_span

**代码示例**：LangGraph trace 链接修复

### 反直觉点3：工具调用数据获取
**错误观点**：默认回调方法会返回工具参数

**为什么错**：
- 默认实现不返回工具参数
- 需要重写 on_tool_start/on_tool_end

**正确理解**：
- 重写 on_tool_start 方法
- 从 kwargs 中提取工具参数
- 适用于 OpenAI function call agent

**代码示例**：自定义工具调用回调

## 生成进度

- [x] 阶段一：Plan 生成
  - [x] 1.1 Brainstorm 分析
  - [x] 1.2 多源数据收集（源码 + Context7 + 网络）
  - [x] 1.3 用户确认拆解方案
  - [x] 1.4 Plan 最终确定
- [x] 阶段二：补充调研（针对需要更多资料的部分）
  - [x] 2.1 识别需要补充资料的部分
  - [x] 2.2 执行补充调研（已通过 Grok-mcp 搜索完成）
  - [x] 2.3 生成抓取任务文件（FETCH_TASK.json）
  - [x] 2.4 更新 PLAN.md
  - [x] 2.5 输出抓取任务提示
- [ ] 阶段三：文档生成（读取 reference/ 中的所有资料）

## 数据来源统计

- **源码分析**：1 个文件
- **Context7 文档**：2 个文件
- **网络搜索**：2 个文件
- **待抓取链接**：16 个（将由第三方工具自动保存）

## 文件数量统计

- **基础维度文件**：10 个
- **核心概念文件**：11 个
- **实战代码文件**：8 个
- **总计**：29 个文件

## 预计文件长度

- **基础维度文件**：每个 50-100 行
- **核心概念文件**：每个 300-500 行
- **实战代码文件**：每个 300-500 行
- **总计**：约 7000-10000 行

## 质量保证

- ✅ 所有核心概念都有明确的数据来源
- ✅ 实战代码场景基于用户选择
- ✅ LangGraph 和 Agent 作为独立概念深入讲解
- ✅ 并发环境回调管理作为反直觉点重点讲解
- ✅ 按使用场景组织核心概念（基础→自定义→可观测性→生产）
- ✅ 覆盖 2025-2026 最新内容

## 下一步操作

1. **阶段二：补充调研**
   - 等待第三方工具抓取 16 个链接
   - 抓取完成后，所有内容将自动保存到 `reference/` 目录

2. **阶段三：文档生成**
   - 读取 `reference/` 中的所有资料
   - 按照 PLAN.md 中的文件清单生成文档
   - 每个文件生成后，更新 PLAN.md 中的文件清单（标记 ✓）

## 备注

- 本 PLAN 基于源码分析、Context7 官方文档和网络搜索结果生成
- 所有数据来源都已标注
- 用户反馈已充分考虑
- 知识点拆解方案已优化
