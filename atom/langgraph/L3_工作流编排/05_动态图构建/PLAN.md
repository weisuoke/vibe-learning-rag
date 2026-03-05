# 05_动态图构建 - 生成计划

## 数据来源记录

### 源码分析
- ✓ reference/source_dynamic_graph_01.md - StateGraph/Send/Command/Pregel 核心机制分析

### Context7 官方文档
- ✓ reference/context7_langgraph_01.md - Send/Command/条件边/并行执行官方文档

### 网络搜索
- ✓ reference/search_dynamic_graph_01.md - 社区资料与实践案例

### 抓取内容
- ✓ reference/fetch_command_routing_01.md - Medium: Command() 动态路由入门指南
- ✓ reference/fetch_conditional_edges_01.md - Dev.to: 条件边与工具调用代理

## 文件清单

### 基础维度文件
- [x] 00_概览.md
- [x] 01_30字核心.md
- [x] 02_第一性原理.md

### 核心概念文件（基于源码 + Context7 + 网络调研）
- [x] 03_核心概念_1_条件边与路由函数.md - add_conditional_edges + path 函数 [来源: 源码/Context7]
- [x] 03_核心概念_2_Send_API与Map_Reduce.md - 运行时动态并行任务 [来源: 源码/Context7]
- [x] 03_核心概念_3_Command_API与动态导航.md - 状态更新+路由控制 [来源: 源码/Context7]
- [x] 03_核心概念_4_LLM驱动的智能路由.md - 结构化输出+AI决策 [来源: Context7/网络]
- [x] 03_核心概念_5_图构建器模式与编译.md - StateGraph builder pattern [来源: 源码]

### 基础维度文件（续）
- [x] 04_最小可用.md
- [x] 05_双重类比.md
- [x] 06_反直觉点.md

### 实战代码文件（基于源码 + Context7 + 网络调研）
- [x] 07_实战代码_场景1_基础条件路由.md - 多路分支与回退 [来源: Context7]
- [x] 07_实战代码_场景2_Send_Map_Reduce.md - 动态并行任务处理 [来源: 源码/Context7]
- [x] 07_实战代码_场景3_Command多代理切换.md - 跨图导航 [来源: 源码/Context7]
- [x] 07_实战代码_场景4_LLM智能路由系统.md - 完整动态决策流 [来源: Context7/网络]

### 基础维度文件（续）
- [x] 08_面试必问.md
- [x] 09_化骨绵掌.md
- [x] 10_一句话总结.md

## 生成进度
- [x] 阶段一：Plan 生成
  - [x] 1.1 Brainstorm 分析
  - [x] 1.2 多源数据收集（源码 + Context7 + 网络）
  - [x] 1.3 用户确认拆解方案
  - [x] 1.4 Plan 最终确定
- [x] 阶段二：补充调研（针对需要更多资料的部分）
- [x] 阶段三：文档生成（读取 reference/ 中的所有资料）
