# 06_条件分支策略 - 生成计划

## 数据来源记录

### 源码分析
- ✓ reference/source_conditional_branching_01.md - StateGraph.add_conditional_edges()、BranchSpec、Send、Command 源码分析

### 网络搜索
- ✓ reference/search_conditional_branching_01.md - 条件分支策略综合搜索结果

### 抓取内容
- ✓ reference/fetch_map_reduce_01.md - Send API Map-Reduce 并行分支教程
- ✓ reference/fetch_command_routing_01.md - Command 动态路由入门指南
- ✓ reference/fetch_best_practices_01.md - LangGraph 最佳实践（条件分支相关）

## 文件清单

### 基础维度文件
- [x] 00_概览.md
- [x] 01_30字核心.md
- [x] 02_第一性原理.md

### 核心概念文件（6个，基于源码 + 网络调研）
- [x] 03_核心概念_1_add_conditional_edges基础.md - API签名、路由函数、path_map三种形式 [来源: 源码]
- [x] 03_核心概念_2_Literal类型推断与路径映射.md - 自动推断机制、显式path_map、可视化影响 [来源: 源码]
- [x] 03_核心概念_3_多路分支与优先级路由.md - 多目标路由、Supervisor模式、优先级策略 [来源: 源码+网络]
- [x] 03_核心概念_4_Send动态路由与MapReduce.md - Send类、并行分支、状态隔离、结果聚合 [来源: 源码+网络]
- [x] 03_核心概念_5_Command高级路由控制.md - goto+update、跨图导航、与条件边对比 [来源: 源码+网络]
- [x] 03_核心概念_6_回退机制与循环守卫.md - 错误路由、max_steps、降级策略、重试模式 [来源: 网络]

### 基础维度文件（续）
- [x] 04_最小可用.md
- [x] 05_双重类比.md
- [x] 06_反直觉点.md

### 实战代码文件（9个场景）
- [x] 07_实战代码_场景1_基础条件路由.md - 简单if-else分支 + Literal类型推断 [来源: 源码]
- [x] 07_实战代码_场景2_多路分支与Supervisor.md - 意图分类→专家代理路由 [来源: 网络]
- [x] 07_实战代码_场景3_Send并行MapReduce.md - Tree of Thoughts风格并行处理 [来源: 网络]
- [x] 07_实战代码_场景4_Command动态路由.md - 运行时决策+状态更新 [来源: 网络]
- [x] 07_实战代码_场景5_错误处理与回退.md - 重试+降级+循环守卫完整模式 [来源: 网络]
- [x] 07_实战代码_场景6_RAG条件路由.md - Self-RAG/Adaptive RAG检索质量判断路由 [来源: 源码+网络]
- [x] 07_实战代码_场景7_人机循环条件分支.md - interrupt+Command审批分支 [来源: 源码]
- [x] 07_实战代码_场景8_多Agent协作路由.md - Agent间handoff与结果质量路由 [来源: 网络]
- [x] 07_实战代码_场景9_LLM驱动智能路由.md - LLM分类意图→不同处理流程(ReAct) [来源: 源码]

### 基础维度文件（续）
- [x] 08_面试必问.md
- [x] 09_化骨绵掌.md
- [x] 10_一句话总结.md

## 生成进度
- [x] 阶段一：Plan 生成
  - [x] 1.1 Brainstorm 分析
  - [x] 1.2 多源数据收集（源码 + 网络）
  - [x] 1.3 用户确认拆解方案
  - [x] 1.4 Plan 最终确定
- [x] 阶段二：补充调研
- [x] 阶段三：文档生成（24个文件全部完成）
