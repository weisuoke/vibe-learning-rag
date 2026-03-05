# 04_Agent推理与决策 - 生成计划

## 数据来源记录

### 源码分析
- ✓ reference/source_react_agent_01.md - ReAct agent 工厂函数、输出解析器、wiki_prompt、scratchpad 格式化
- ✓ reference/source_output_parsers_02.md - 5种输出解析器对比（ReAct/JSON/Tools/XML/MRKL）
- ✓ reference/source_reasoning_strategies_03.md - MRKL prompt、ZeroShotAgent、tool_calling_agent、推理策略对比

### Context7 官方文档
- ✓ reference/context7_langchain_01.md - ReAct LCEL 链构建、推理循环、create_agent、结构化输出

### 网络搜索
- ✓ reference/search_agent_reasoning_01.md - Twitter/X + Reddit + GitHub 搜索结果汇总

### 待抓取链接（将由第三方工具自动保存到 reference/）
- [ ] https://x.com/Aurimas_Gr/status/2002047650011705725 - ReAct Agent 模式详解
- [ ] https://www.reddit.com/r/LangChain/comments/1iqrfyy/ - ReAct vs 函数调用讨论
- [ ] https://www.reddit.com/r/LangChain/comments/1kc61kj/ - ReAct vs Function Calling 选择
- [ ] https://www.reddit.com/r/LangChain/comments/1ffe38x/ - Tool Calling vs Structured Chat vs ReAct

## 文件清单

### 基础维度文件
- [ ] 00_概览.md
- [ ] 01_30字核心.md
- [ ] 02_第一性原理.md

### 核心概念文件（基于源码 + Context7 + 网络调研）
- [ ] 03_核心概念_1_ReAct推理模式.md - Thought→Action→Observation循环，create_react_agent实现 [来源: 源码/Context7]
- [ ] 03_核心概念_2_思维链与Scratchpad.md - CoT原理，format_scratchpad机制，推理历史累积 [来源: 源码]
- [ ] 03_核心概念_3_输出解析与决策提取.md - 5种OutputParser对比，AgentAction/AgentFinish决策 [来源: 源码]
- [ ] 03_核心概念_4_推理提示工程.md - MRKL/ReAct/JSON/XML格式，prompt设计要素 [来源: 源码/Context7]
- [ ] 03_核心概念_5_推理策略对比与选型.md - ReAct vs Tool Calling vs XML，选型决策树 [来源: 源码/网络]

### 基础维度文件（续）
- [ ] 04_最小可用.md
- [ ] 05_双重类比.md
- [ ] 06_反直觉点.md

### 实战代码文件（基于源码 + Context7 + 网络调研）
- [ ] 07_实战代码_场景1_ReAct_Agent实战.md - 构建完整ReAct agent，显式推理过程 [来源: 源码/Context7]
- [ ] 07_实战代码_场景2_MRKL零样本推理.md - ZeroShot agent，零样本工具选择 [来源: 源码]
- [ ] 07_实战代码_场景3_思维链增强Agent.md - CoT增强技术，few-shot推理示例 [来源: 网络]
- [ ] 07_实战代码_场景4_自定义输出解析器.md - 自定义OutputParser，错误自修正 [来源: 源码]
- [ ] 07_实战代码_场景5_推理质量评估.md - agentevals评估，LangSmith追踪 [来源: 网络/GitHub]

### 基础维度文件（续）
- [ ] 08_面试必问.md
- [ ] 09_化骨绵掌.md
- [ ] 10_一句话总结.md

## 生成批次计划

### 批次1（3个subagent × 2个文档 = 6个文档）
- subagent A: 00_概览.md + 01_30字核心.md
- subagent B: 02_第一性原理.md + 03_核心概念_1_ReAct推理模式.md
- subagent C: 03_核心概念_2_思维链与Scratchpad.md + 03_核心概念_3_输出解析与决策提取.md

### 批次2（3个subagent × 2个文档 = 6个文档）
- subagent D: 03_核心概念_4_推理提示工程.md + 03_核心概念_5_推理策略对比与选型.md
- subagent E: 04_最小可用.md + 05_双重类比.md
- subagent F: 06_反直觉点.md + 07_实战代码_场景1_ReAct_Agent实战.md

### 批次3（3个subagent × 2个文档 = 6个文档）
- subagent G: 07_实战代码_场景2_MRKL零样本推理.md + 07_实战代码_场景3_思维链增强Agent.md
- subagent H: 07_实战代码_场景4_自定义输出解析器.md + 07_实战代码_场景5_推理质量评估.md
- subagent I: 08_面试必问.md + 09_化骨绵掌.md

### 批次4（1个subagent × 1个文档）
- subagent J: 10_一句话总结.md

## 生成进度
- [x] 阶段一：Plan 生成
  - [x] 1.1 Brainstorm 分析
  - [x] 1.2 多源数据收集（源码 + Context7 + 网络）
  - [x] 1.3 用户确认拆解方案
  - [x] 1.4 Plan 最终确定
- [ ] 阶段二：补充调研（针对需要更多资料的部分）
- [ ] 阶段三：文档生成（读取 reference/ 中的所有资料）
