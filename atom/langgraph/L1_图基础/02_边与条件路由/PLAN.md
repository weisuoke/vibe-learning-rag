# 02_边与条件路由 - 生成计划

## 数据来源记录

### 源码分析
- ✓ reference/source_edges_01_add_edge.md - add_edge 方法实现分析
- ✓ reference/source_edges_02_add_conditional_edges.md - add_conditional_edges 方法实现分析

### Context7 官方文档
- ✓ reference/context7_langgraph_01_edges_routing.md - LangGraph 边与条件路由官方文档

### 网络搜索
- ✓ reference/search_edges_01_github.md - GitHub 教程和示例搜索结果
- ✓ reference/search_edges_02_reddit.md - Reddit 社区讨论搜索结果

### 待抓取链接（将由第三方工具自动保存到 reference/）
#### High 优先级（GitHub）
- [ ] https://github.com/Coding-Crashkurse/LangGraph-Tutorial/blob/main/langgraph.ipynb
- [ ] https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/langgraph-tutorial.ipynb
- [ ] https://github.com/philschmid/gemini-samples/blob/main/guides/langgraph-react-agent.ipynb
- [ ] https://github.com/langchain-ai/langchain-academy/blob/main/module-1/simple-graph.ipynb

#### Medium 优先级（GitHub）
- [ ] https://github.com/pinecone-io/examples/blob/master/learn/generation/langchain/langgraph/00-langgraph-intro.ipynb
- [ ] https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/customer_support_agent_langgraph.ipynb

#### High 优先级（Reddit）
- [ ] https://www.reddit.com/r/LangChain/comments/1kpkybb/why_are_people_choosing_langgraph_pydanticai_for/
- [ ] https://www.reddit.com/r/LangChain/comments/1kwxunp/any_interesting_project_in_langgraph/

#### Medium 优先级（Reddit）
- [ ] https://www.reddit.com/r/AI_Agents/comments/1pxu5un/should_i_use_langgraph_to_build_my_ai_agent/
- [ ] https://www.reddit.com/r/LangGraph/comments/1o5wjof/langgraph_related_problem/
- [ ] https://www.reddit.com/r/AI_Agents/comments/1qf88fe/yall_are_overcomplicating_langgraph_and_burning/

## 文件清单

### 基础维度文件
- [ ] 00_概览.md
- [ ] 01_30字核心.md
- [ ] 02_第一性原理.md

### 核心概念文件（基于源码 + Context7 + 网络调研）
- [ ] 03_核心概念_1_边的基础概念.md - 什么是边，边的作用，边的类型 [来源: 源码]
- [ ] 03_核心概念_2_普通边add_edge.md - 固定路由的语法和用法 [来源: 源码 + Context7]
- [ ] 03_核心概念_3_条件边add_conditional_edges.md - 动态路由的语法和用法 [来源: 源码 + Context7]
- [ ] 03_核心概念_4_路由函数设计.md - 路由函数的签名、输入输出、逻辑编写 [来源: Context7 + 网络]
- [ ] 03_核心概念_5_路由映射path_map.md - 路由映射的定义和使用 [来源: Context7]
- [ ] 03_核心概念_6_多分支路由.md - 多个目标节点的路由决策 [来源: Context7]
- [ ] 03_核心概念_7_边的执行机制.md - 边的执行时机、状态传递、错误处理 [来源: 源码]
- [ ] 03_核心概念_8_边的最佳实践.md - 何时用普通边/条件边，设计原则，常见模式 [来源: Context7 + 网络]

### 基础维度文件（续）
- [ ] 04_最小可用.md
- [ ] 05_双重类比.md
- [ ] 06_反直觉点.md

### 实战代码文件（基于源码 + Context7 + 网络调研）
- [ ] 07_实战代码_场景1_基础线性流程.md - 普通边的基础用法 [来源: Context7]
- [ ] 07_实战代码_场景2_简单条件分支.md - if-else 决策 [来源: Context7]
- [ ] 07_实战代码_场景3_多路由决策.md - switch-case 模式 [来源: Context7]
- [ ] 07_实战代码_场景4_复杂状态路由.md - 实际业务场景 [来源: Context7 + 网络]
- [ ] 07_实战代码_场景5_错误处理与回退路由.md - 错误处理和重试机制 [来源: Context7]
- [ ] 07_实战代码_场景6_审批流程实战.md - 人机协作场景 [来源: Context7]

### 基础维度文件（续）
- [ ] 08_面试必问.md
- [ ] 09_化骨绵掌.md
- [ ] 10_一句话总结.md

## 知识点拆解详情

### 核心概念详细说明

#### 1. 边的基础概念
**内容范围**：
- 边的定义：连接节点的有向路径
- 边的作用：控制图的执行流程
- 边的类型：普通边 vs 条件边
- 边的存储：edges 集合 vs waiting_edges 集合

**数据来源**：
- 源码：`state.py` 中的 `add_edge` 方法实现
- 源码：`StateGraph` 类的 `edges` 和 `waiting_edges` 属性

#### 2. 普通边（add_edge）
**内容范围**：
- 方法签名：`add_edge(start_key: str | list[str], end_key: str)`
- 单起始节点：固定路由，直接连接
- 多起始节点：等待所有节点完成（AND 逻辑）
- 验证规则：END 不能作为起始节点，START 不能作为结束节点
- 快捷方法：`set_entry_point`, `set_finish_point`

**数据来源**：
- 源码：`state.py:785-838` 的 `add_edge` 方法
- Context7：基础用法示例

#### 3. 条件边（add_conditional_edges）
**内容范围**：
- 方法签名：`add_conditional_edges(source, path, path_map)`
- 路由函数：接受状态，返回目标节点名称
- 动态决策：基于状态值选择执行路径
- BranchSpec：条件边的规范对象
- 类型提示：使用 `Literal` 明确路由目标

**数据来源**：
- 源码：`state.py:839-887` 的 `add_conditional_edges` 方法
- 源码：`_branch.py` 的 `BranchSpec` 类
- Context7：动态路由示例

#### 4. 路由函数设计
**内容范围**：
- 函数签名：`def route_func(state: State) -> Literal[...]`
- 输入参数：state（当前状态）
- 返回值类型：单个节点名称或多个节点名称
- 路由逻辑：清晰的条件判断
- 最佳实践：避免复杂逻辑，添加日志，处理边界情况

**数据来源**：
- Context7：路由函数设计模式
- 网络：社区最佳实践

#### 5. 路由映射（path_map）
**内容范围**：
- dict 形式：`{"key": "node_name"}`
- list 形式：`["node1", "node2"]`
- None 形式：依赖类型提示
- 映射作用：将路由函数返回值映射到节点名称

**数据来源**：
- 源码：`_branch.py` 的 `BranchSpec.from_path` 方法
- Context7：path_map 使用示例

#### 6. 多分支路由
**内容范围**：
- 多目标节点：路由函数返回多个节点名称
- 并行执行：多个目标节点并行执行
- 汇聚节点：等待多个节点完成
- 应用场景：多步推理、并行处理

**数据来源**：
- Context7：多路由决策示例
- 网络：复杂路由案例

#### 7. 边的执行机制
**内容范围**：
- 执行时机：节点完成后立即执行
- 状态传递：通过 ChannelWrite 传递状态
- 错误处理：边执行失败的处理
- 编译验证：图编译时的边验证

**数据来源**：
- 源码：`state.py` 的 `validate` 方法
- 源码：`_branch.py` 的 `_route` 和 `_aroute` 方法

#### 8. 边的最佳实践
**内容范围**：
- 何时使用普通边：固定流程、线性执行
- 何时使用条件边：动态决策、分支逻辑
- 设计原则：简单清晰、避免过度复杂化
- 常见模式：简单分支、多路由、循环路由、入口路由、多层路由
- 常见误区：过度复杂化、性能问题、维护困难

**数据来源**：
- Context7：设计模式和最佳实践
- 网络：社区经验和常见问题

### 实战代码场景详细说明

#### 场景1：基础线性流程
**内容范围**：
- START → node1 → node2 → END
- 使用 `add_edge` 连接节点
- 完整可运行代码
- 输出结果展示

**数据来源**：
- Context7：基础示例

#### 场景2：简单条件分支
**内容范围**：
- if-else 决策逻辑
- 路由函数设计
- path_map 使用
- 完整可运行代码

**数据来源**：
- Context7：条件分支示例

#### 场景3：多路由决策
**内容范围**：
- switch-case 模式
- 多个目标节点
- 使用 `Literal` 类型提示
- 完整可运行代码

**数据来源**：
- Context7：多路由示例

#### 场景4：复杂状态路由
**内容范围**：
- 实际业务场景（RAG 工作流）
- 多层条件路由
- 状态管理
- 完整可运行代码

**数据来源**：
- Context7：RAG 工作流示例
- 网络：生产级应用案例

#### 场景5：错误处理与回退路由
**内容范围**：
- 错误检测
- 重试机制
- 回退路由
- 完整可运行代码

**数据来源**：
- Context7：错误处理示例
- 网络：错误处理最佳实践

#### 场景6：审批流程实战
**内容范围**：
- 人机协作场景
- 审批决策路由
- 状态更新
- 完整可运行代码

**数据来源**：
- Context7：人机循环示例
- 网络：审批流程案例

## 生成进度
- [x] 阶段一：Plan 生成
  - [x] 1.1 Brainstorm 分析
  - [x] 1.2 多源数据收集（源码 + Context7 + 网络）
  - [x] 1.3 用户确认拆解方案
  - [x] 1.4 Plan 最终确定
- [x] 阶段二：补充调研（跳过 - 数据已充足）
  - 决策：当前已收集 5 个高质量资料文件，覆盖所有核心概念和实战场景，无需补充调研
- [x] 阶段三：文档生成（读取 reference/ 中的所有资料）
  - [x] 3.1 读取所有 reference/ 资料
  - [x] 3.2 按顺序生成文档（23个文件全部完成）
  - [x] 3.3 最终验证

## 数据覆盖度分析

### 源码分析覆盖
- ✓ add_edge 方法实现（完整）
- ✓ add_conditional_edges 方法实现（完整）
- ✓ BranchSpec 类（完整）
- ✓ 边的存储结构（edges, waiting_edges, branches）
- ✓ 边的验证逻辑（validate 方法）

### Context7 官方文档覆盖
- ✓ 动态路由示例
- ✓ RAG 工作流示例
- ✓ 多层条件路由示例
- ✓ 路由函数设计模式
- ✓ 最佳实践和常见错误

### 网络搜索覆盖
- ✓ GitHub 教程和示例（8个）
- ✓ Reddit 社区讨论（5个）
- ✓ 生产级应用案例
- ✓ 常见问题和解决方案
- ✓ 最佳实践和误区

### 需要补充的部分
根据当前数据收集情况，以下部分可能需要补充：
1. **高级路由模式**：循环路由、嵌套路由的更多实例
2. **性能优化**：大规模图中的边优化策略
3. **调试技巧**：如何调试复杂的条件路由
4. **测试方法**：如何测试条件边的各个分支

**决策**：当前数据已足够生成高质量文档，可以跳过阶段二的补充调研，直接进入阶段三的文档生成。如果在生成过程中发现需要更多资料，可以随时补充。

## 下一步操作

### 选项 A：跳过阶段二，直接进入阶段三（推荐）
**理由**：
- 已收集 5 个高质量资料文件
- 覆盖了所有 8 个核心概念
- 包含 6 个实战场景的参考资料
- 源码分析完整，官方文档充足

**操作**：
1. 更新 PLAN.md 标记阶段二为"跳过"
2. 开始阶段三：文档生成

### 选项 B：执行阶段二，补充调研
**理由**：
- 需要更多高级路由模式的实例
- 需要更多性能优化的案例
- 需要更多调试和测试的方法

**操作**：
1. 识别需要补充的部分
2. 执行补充调研
3. 生成抓取任务文件

## 推荐方案

**推荐选项 A：跳过阶段二，直接进入阶段三**

**原因**：
1. **数据充足**：已收集的资料覆盖了所有核心概念和实战场景
2. **质量高**：源码分析完整，官方文档权威，社区讨论丰富
3. **效率高**：可以立即开始文档生成，避免等待抓取任务
4. **灵活性**：如果在生成过程中发现需要更多资料，可以随时补充

**执行计划**：
1. 更新 PLAN.md 标记阶段二为"跳过（数据已充足）"
2. 创建资料索引文件 `reference/INDEX.md`
3. 开始阶段三：按顺序生成文档
