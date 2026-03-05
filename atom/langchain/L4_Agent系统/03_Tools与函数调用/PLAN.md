# 03_Tools与函数调用 - 生成计划

## 数据来源记录

### 源码分析
- ✓ reference/source_tools_core_01.md - Tools 核心系统完整分析（base.py, convert.py, structured.py, simple.py, render.py, tool.py, function_calling.py）

### Context7 官方文档
- ✓ reference/context7_langchain_tools_01.md - LangChain Tools & Function Calling 官方文档

### 网络搜索
- （暂无需要，源码 + Context7 已覆盖所有核心内容）

### 待抓取链接
- （暂无需要）

## 文件清单

### 基础维度文件（第一部分）
- [x] 00_概览.md
- [x] 01_30字核心.md
- [x] 02_第一性原理.md

### 核心概念文件（基于源码 + Context7 调研）
- [x] 03_核心概念_1_Tool定义三种方式.md - @tool装饰器、StructuredTool.from_function()、BaseTool子类 [来源: 源码]
- [x] 03_核心概念_2_函数调用协议.md - bind_tools()、模型生成tool_calls、格式转换 [来源: 源码+Context7]
- [x] 03_核心概念_3_ToolCall与ToolMessage消息流.md - 请求→执行→响应消息链路 [来源: 源码]
- [x] 03_核心概念_4_Tool_Schema与参数验证.md - Pydantic BaseModel、JSON Schema、parse_docstring [来源: 源码]
- [x] 03_核心概念_5_工具选择策略.md - LLMToolSelectorMiddleware、动态工具过滤 [来源: Context7]
- [x] 03_核心概念_6_高级Tool特性.md - response_format、InjectedToolArg、extras、return_direct [来源: 源码]

### 基础维度文件（第二部分）
- [x] 04_最小可用.md
- [x] 05_双重类比.md
- [x] 06_反直觉点.md

### 实战代码文件（基于源码 + Context7 调研）
- [x] 07_实战代码_场景1_基础Tool定义与调用.md - 三种方式定义Tool并直接调用 [来源: 源码+Context7]
- [x] 07_实战代码_场景2_函数调用完整流程.md - bind_tools + 模型调用 + ToolMessage回传 [来源: Context7]
- [x] 07_实战代码_场景3_自定义Schema与参数验证.md - Pydantic模型、docstring解析 [来源: 源码]
- [x] 07_实战代码_场景4_动态工具选择与Agent集成.md - Middleware模式 + Agent工具使用 [来源: Context7]

### 基础维度文件（第三部分）
- [x] 08_面试必问.md
- [x] 09_化骨绵掌.md
- [x] 10_一句话总结.md

## 生成进度
- [x] 阶段一：Plan 生成
  - [x] 1.1 Brainstorm 分析
  - [x] 1.2 多源数据收集（源码 + Context7）
  - [x] 1.3 用户确认拆解方案
  - [x] 1.4 Plan 最终确定
- [x] 阶段二：补充调研（源码 + Context7 已充分覆盖，跳过）
- [x] 阶段三：文档生成
  - [x] 批次1: 00_概览 + 01_30字核心 + 02_第一性原理
  - [x] 批次2: 03_核心概念_1 + 03_核心概念_2
  - [x] 批次3: 03_核心概念_3 + 03_核心概念_4
  - [x] 批次4: 03_核心概念_5 + 03_核心概念_6
  - [x] 批次5: 04_最小可用 + 05_双重类比 + 06_反直觉点
  - [x] 批次6: 07_实战代码_场景1 + 07_实战代码_场景2
  - [x] 批次7: 07_实战代码_场景3 + 07_实战代码_场景4
  - [x] 批次8: 08_面试必问 + 09_化骨绵掌 + 10_一句话总结
