# 02_子图（Subgraph）与模块化 - 生成计划

## 数据来源记录

### 源码分析
- ✓ reference/source_subgraph_01.md - StateGraph、add_node、Command、Send、input/output_schema 核心实现分析

### Context7 官方文档
- ✓ reference/context7_langgraph_01.md - 子图集成方式、状态传递、Command.PARENT、流式输出、Checkpointer

### 网络搜索
- ✓ reference/search_subgraph_01.md - 社区最佳实践、多代理系统、性能考虑

### 待抓取链接
- [ ] https://dev.to/sreeni5018/langgraph-subgraphs-a-guide-to-modular-ai-agents-development-31ob
- [ ] https://dev.to/jamesli/building-complex-ai-workflows-with-langgraph-a-detailed-explanation-of-subgraph-architecture-1dj5

## 知识点拆解

### 核心概念分析

子图（Subgraph）是 LangGraph 中将复杂工作流拆分为可复用模块的核心机制。基于源码和官方文档，拆解为以下核心概念：

1. **子图定义与编译** - StateGraph 如何定义子图，input_schema/output_schema 的作用
2. **共享状态键集成** - 父子图通过共享 key 自动传递状态
3. **状态转换集成** - 通过包装函数手动转换不同 schema 之间的状态
4. **Command.PARENT 跨图通信** - 子图向父图发送命令，实现动态路由
5. **Send 动态分发** - 使用 Send 将不同输入分发到子图（map-reduce 模式）
6. **子图 Checkpointer 与持久化** - 子图的 checkpoint 继承与独立记忆

### 实战场景分析

1. **基础子图创建与调用** - 最简单的子图集成示例
2. **多代理 Handoff 系统** - 使用 Command.PARENT 实现代理切换
3. **Map-Reduce 并行子图** - 使用 Send 实现并行处理
4. **嵌套子图与流式监控** - 多层嵌套 + subgraphs=True 流式输出

## 文件清单

### 基础维度文件
- [x] 00_概览.md
- [x] 01_30字核心.md
- [x] 02_第一性原理.md

### 核心概念文件（基于源码 + Context7 + 网络调研）
- [x] 03_核心概念_1_子图定义与编译.md - input_schema/output_schema [来源: 源码/Context7]
- [x] 03_核心概念_2_共享状态键集成.md - 自动状态传递 [来源: Context7]
- [x] 03_核心概念_3_状态转换集成.md - 包装函数手动转换 [来源: Context7]
- [x] 03_核心概念_4_Command跨图通信.md - Command.PARENT [来源: 源码/Context7]
- [x] 03_核心概念_5_Send动态分发.md - map-reduce 模式 [来源: 源码]
- [x] 03_核心概念_6_子图Checkpointer.md - 持久化与独立记忆 [来源: Context7]

### 基础维度文件（续）
- [x] 04_最小可用.md
- [x] 05_双重类比.md
- [x] 06_反直觉点.md

### 实战代码文件（基于源码 + Context7 + 网络调研）
- [x] 07_实战代码_场景1_基础子图创建与调用.md [来源: 源码/Context7]
- [x] 07_实战代码_场景2_多代理Handoff系统.md [来源: Context7/网络]
- [x] 07_实战代码_场景3_MapReduce并行子图.md [来源: 源码]
- [x] 07_实战代码_场景4_嵌套子图与流式监控.md [来源: Context7]

### 基础维度文件（续）
- [x] 08_面试必问.md
- [x] 09_化骨绵掌.md
- [x] 10_一句话总结.md

## 文件总数

- 基础维度文件：8 个
- 核心概念文件：6 个
- 实战代码文件：4 个
- **总计：18 个文件**

## 生成进度
- [x] 阶段一：Plan 生成
  - [x] 1.1 Brainstorm 分析
  - [x] 1.2 多源数据收集（源码 + Context7 + 网络）
  - [x] 1.3 用户确认拆解方案
  - [x] 1.4 Plan 最终确定
- [x] 阶段二：补充调研（现有资料已充足，跳过抓取）
- [x] 阶段三：文档生成（全部 18 个文件已完成）
