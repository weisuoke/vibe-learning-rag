# 04_人机循环（Human-in-the-loop） - 生成计划

## 数据来源记录

### 源码分析
- ✓ reference/source_hitl_01.md - types.py (interrupt/Command/Interrupt), errors.py (GraphInterrupt), prebuilt/interrupt.py (HumanInterrupt)
- ✓ reference/source_hitl_02.md - _algo.py (should_interrupt), _loop.py (PregelLoop), main.py (Pregel), state.py (compile参数)

### Context7 官方文档
- ✓ reference/context7_langgraph_01.md - LangGraph HITL 官方文档（中断/恢复/审批/流式处理）

### 网络搜索
- ✓ reference/search_hitl_01.md - 2025-2026 HITL 教程、博客、社区讨论、开源项目

### 待抓取链接（将由第三方工具自动保存到 reference/）
- [ ] https://medium.com/the-advanced-school-of-ai/human-in-the-loop-with-langgraph-mastering-interrupts-and-commands-9e1cf2183ae3
- [ ] https://dev.to/jamesbmour/interrupts-and-commands-in-langgraph-building-human-in-the-loop-workflows-4ngl
- [ ] https://blog.langchain.com/making-it-easier-to-build-human-in-the-loop-agents-with-interrupt
- [ ] https://www.reddit.com/r/LangGraph/comments/1ldiqtg/i_am_struggling_with_langgraphs_humanintheloop

## 文件清单

### 基础维度文件
- [x] 00_概览.md (569行)
- [x] 01_30字核心.md (372行)
- [x] 02_第一性原理.md (491行)

### 核心概念文件（基于源码 + Context7 + 网络调研）
- [x] 03_核心概念_1_interrupt函数.md (748行)
- [x] 03_核心概念_2_Command恢复执行.md (709行)
- [x] 03_核心概念_3_静态断点配置.md (768行)
- [x] 03_核心概念_4_Checkpoint状态持久化.md (660行)
- [x] 03_核心概念_5_结构化人机交互.md (736行)
- [x] 03_核心概念_6_多中断点与匹配.md (596行)

### 基础维度文件（续）
- [x] 04_最小可用.md (563行)
- [x] 05_双重类比.md (635行)
- [x] 06_反直觉点.md (451行)

### 实战代码文件（基于源码 + Context7 + 网络调研）
- [x] 07_实战代码_场景1_基础中断与恢复.md (475行)
- [x] 07_实战代码_场景2_审批工作流.md (875行)
- [x] 07_实战代码_场景3_工具调用审批.md (932行)
- [x] 07_实战代码_场景4_用户反馈循环.md (824行)
- [x] 07_实战代码_场景5_生产级HITL系统.md (886行)

### 基础维度文件（续）
- [x] 08_面试必问.md (387行)
- [x] 09_化骨绵掌.md (471行)
- [x] 10_一句话总结.md (105行)

## 核心概念详细说明

### 概念1: interrupt() 函数
- 动态中断机制，在节点内部任意位置调用
- 传递JSON可序列化的value给客户端
- 第一次调用抛出GraphInterrupt异常
- 恢复时从节点开头重新执行
- 官方推荐的HITL方式（替代旧的interrupt_before/after）

### 概念2: Command(resume=...) 恢复执行
- Command是恢复中断的核心原语
- 支持单值恢复和按ID恢复（字典形式）
- 可组合resume + goto + update
- 支持invoke/stream/ainvoke/astream四种恢复方式
- 必须使用相同的thread_id

### 概念3: 静态断点 interrupt_before/after
- compile()时配置的静态断点
- interrupt_before: 节点执行前暂停
- interrupt_after: 节点执行后暂停
- 支持"*"通配符（所有节点）
- stream/invoke运行时也可覆盖
- 灵活性低于interrupt()，但适合简单场景

### 概念4: Checkpoint 状态持久化
- 中断依赖checkpointer保存状态
- MemorySaver（开发）vs PostgresSaver（生产）
- StateSnapshot包含interrupts字段
- get_state()/get_state_history()检查中断状态
- thread_id是状态恢复的关键标识

### 概念5: 结构化人机交互
- HumanInterruptConfig定义允许的操作（ignore/respond/edit/accept）
- HumanInterrupt封装请求（action_request + config + description）
- HumanResponse封装响应（type + args）
- Agent Inbox模式：标准化的人机交互协议
- 注意：已迁移到langchain.agents.interrupt

### 概念6: 多中断点与匹配
- 单节点内可有多个interrupt()调用
- resume值按顺序与interrupt配对
- 已匹配的interrupt直接返回缓存值
- 未匹配的interrupt抛出新的GraphInterrupt
- 作用域限定在特定任务内

## 实战场景详细说明

### 场景1: 基础中断与恢复
- 最简单的interrupt+Command模式
- 单节点单中断点
- 演示完整的暂停→检查→恢复流程
- 包含stream和invoke两种方式

### 场景2: 审批工作流
- 多节点审批流程（申请→审批→执行/拒绝）
- interrupt()传递结构化审批信息
- Command(goto=...)条件路由
- 多级审批链

### 场景3: 工具调用审批
- Agent调用工具前的人工确认
- 展示工具名称和参数供审批
- 支持approve/reject/edit三种操作
- 结合LangChain工具使用

### 场景4: 用户反馈循环
- 迭代式内容生成+人工反馈
- 多轮interrupt实现反馈循环
- 用户可选择"满意"退出或"修改"继续
- 状态累积记录修改历史

### 场景5: 生产级HITL系统
- FastAPI后端 + 中断管理
- PostgresSaver持久化
- 异步处理和超时控制
- 完整的API端点设计

## 生成进度
- [x] 阶段一：Plan 生成
  - [x] 1.1 Brainstorm 分析
  - [x] 1.2 多源数据收集（源码 + Context7 + 网络）
  - [x] 1.3 用户确认拆解方案
  - [x] 1.4 Plan 最终确定
- [x] 阶段二：补充调研（针对需要更多资料的部分）
- [x] 阶段三：文档生成（读取 reference/ 中的所有资料）
  - [x] Batch 1: 00~02 + 03_概念1~3 (6文件)
  - [x] Batch 2: 03_概念4~6 + 04 + 05 (5文件)
  - [x] Batch 3: 06 + 07_场景1~4 (5文件)
  - [x] Batch 4: 07_场景5 + 08~10 (4文件)
  - 总计：20个文件，12,382行
