# Agent类型与选择 - 生成计划

**知识点**: Agent类型与选择 (06_Agent类型与选择)
**层级**: L4_Agent系统
**生成时间**: 2026-03-02
**目标**: 实用选择指南 + 迁移 + 故障排查的全面覆盖

---

## 数据来源记录

### 源码分析
- ✅ reference/source_agent_types_01.md - agent_types.py 枚举定义分析
- ✅ reference/source_agent_init_02.md - __init__.py 导出接口分析
- ✅ reference/source_agents_core_03.md - langchain_core/agents.py 核心抽象

### Context7 官方文档
- ✅ reference/context7_langchain_agent_types_01.txt - Agent 类型和 create_agent API
- ✅ reference/context7_langchain_agent_comparison_02.txt - Agent 对比和使用场景
- ✅ reference/context7_langchain_structured_chat_03.txt - Structured Chat 和工具调用

### 网络搜索（待执行）
- [ ] 搜索: "langchain agent types comparison 2025 2026"
- [ ] 搜索: "OpenAI Functions vs ReAct agent when to use"
- [ ] 搜索: "langchain structured chat agent migration"
- [ ] 搜索: "langchain create_agent 2026 best practices"

### 待抓取链接（将由第三方工具自动保存到 reference/）
待网络搜索完成后生成

---

## 核心发现与洞察

### 1. 2026 年 Agent 生态现状

**关键变化**:
- ✅ `create_agent()` 成为 2026 年推荐的统一 API
- ⚠️ 传统 `AgentType` 枚举已标记为 deprecated (0.1.0 → 1.0 移除)
- ✅ 三大主流类型仍然活跃: OpenAI Functions, ReAct, Structured Chat
- 🆕 Tool Calling Agent 作为新一代通用方案

**源码证据**:
```python
# sourcecode/langchain/libs/langchain/langchain_classic/agents/agent_types.py
@deprecated("0.1.0", message=AGENT_DEPRECATION_WARNING, removal="1.0")
class AgentType(str, Enum):
    OPENAI_FUNCTIONS = "openai-functions"
    ZERO_SHOT_REACT_DESCRIPTION = "zero-shot-react-description"
    STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION = "structured-chat-zero-shot-react-description"
```

### 2. 实用选择决策树（核心内容）

基于源码和官方文档，提炼出以下决策逻辑:

```
是否使用支持函数调用的模型 (OpenAI, Anthropic)?
├─ 是 → 工具是否需要复杂参数 (多个输入/嵌套结构)?
│   ├─ 是 → Structured Chat Agent (多输入工具)
│   └─ 否 → OpenAI Functions / Tool Calling Agent (推荐)
└─ 否 → 使用开源模型 (Llama, Mistral)?
    └─ ReAct Agent (基于 prompt 的推理)
```

### 3. 三大类型核心特征对比

| 特征 | OpenAI Functions | ReAct | Structured Chat |
|------|------------------|-------|-----------------|
| **模型要求** | 支持函数调用 | 任何 LLM | 支持函数调用 |
| **工具参数** | 简单参数 | 简单参数 | 复杂/多输入 |
| **推理方式** | 结构化函数调用 | Thought-Action-Observation | 结构化 + 多输入 |
| **可靠性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **灵活性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **成本** | 中等 | 较高 (更多 token) | 中等 |
| **2026 状态** | 稳定 | 稳定 | 稳定 |

---

## 文件清单

### 基础维度文件
- [ ] 00_概览.md
- [ ] 01_30字核心.md
- [ ] 02_第一性原理.md

### 核心概念文件（基于源码 + Context7 + 网络调研）
- [ ] 03_核心概念_1_OpenAI_Functions_Agent.md - 函数调用型 Agent 原理与适用场景 [来源: 源码 + Context7]
- [ ] 03_核心概念_2_ReAct_Agent.md - 推理-行动循环型 Agent 原理与适用场景 [来源: 源码 + Context7]
- [ ] 03_核心概念_3_Structured_Chat_Agent.md - 结构化对话型 Agent 原理与适用场景 [来源: 源码 + Context7]
- [ ] 03_核心概念_4_Tool_Calling_Agent.md - 2026 新一代工具调用 Agent [来源: Context7 + 网络]
- [ ] 03_核心概念_5_create_agent统一API.md - 2026 推荐的 Agent 创建方式 [来源: Context7 + 源码]
- [ ] 03_核心概念_6_Agent选择决策树.md - 实用选择指南和决策流程 [来源: 综合分析]

### 基础维度文件（续）
- [ ] 04_最小可用.md
- [ ] 05_双重类比.md
- [ ] 06_反直觉点.md

### 实战代码文件（基于源码 + Context7 + 网络调研）
- [ ] 07_实战代码_场景1_从零构建OpenAI_Functions_Agent.md - 完整示例 [来源: Context7 + 源码]
- [ ] 07_实战代码_场景2_从零构建ReAct_Agent.md - 完整示例 [来源: Context7 + 源码]
- [ ] 07_实战代码_场景3_从零构建Structured_Chat_Agent.md - 完整示例 [来源: Context7 + 源码]
- [ ] 07_实战代码_场景4_使用create_agent统一API.md - 2026 推荐方式 [来源: Context7]
- [ ] 07_实战代码_场景5_Agent类型迁移实战.md - 从旧 API 迁移到新 API [来源: 源码 + 网络]
- [ ] 07_实战代码_场景6_Agent故障排查与类型切换.md - 诊断和解决常见问题 [来源: 网络 + 实践]
- [ ] 07_实战代码_场景7_多Agent类型对比测试.md - 性能和效果对比 [来源: 综合]

### 基础维度文件（续）
- [ ] 08_面试必问.md
- [ ] 09_化骨绵掌.md
- [ ] 10_一句话总结.md

---

## 内容重点与特色

### 1. 实用选择指南（核心）
- **决策树**: 基于模型能力、工具复杂度、成本预算的选择流程
- **场景映射**: 10+ 真实场景 → 推荐 Agent 类型
- **对比表格**: 多维度对比（性能、成本、可靠性、开发体验）

### 2. 迁移指南
- **从 AgentType 枚举迁移到 create_agent()**
- **从 initialize_agent() 迁移到现代 API**
- **从旧版 OpenAI Functions 迁移到 Tool Calling**
- **代码示例**: Before/After 对比

### 3. 故障排查
- **常见问题诊断**: Agent 不调用工具、调用错误工具、无限循环
- **类型切换策略**: 何时从 ReAct 切换到 OpenAI Functions
- **调试技巧**: 使用 verbose=True、LangSmith 追踪

### 4. 2026 最佳实践
- **优先使用 create_agent()** 而非 initialize_agent()
- **Tool Calling Agent 作为默认选择**
- **避免使用已弃用的 AgentType 枚举**
- **使用 LangSmith 监控 Agent 行为**

---

## 双重类比设计

### 前端开发类比
- **OpenAI Functions Agent** = React 组件 (声明式、结构化)
- **ReAct Agent** = jQuery (命令式、灵活但冗长)
- **Structured Chat Agent** = Vue 组件 (结构化 + 灵活性平衡)
- **Tool Calling Agent** = React Hooks (现代、统一、推荐)

### 日常生活类比
- **OpenAI Functions Agent** = 点菜系统 (菜单选择、结构化)
- **ReAct Agent** = 自由对话点菜 (描述需求、服务员理解)
- **Structured Chat Agent** = 定制菜单 (复杂需求、多选项)
- **Agent 选择** = 选择交通工具 (根据距离、预算、时间)

---

## 反直觉点预告

1. **"OpenAI Functions 只能用于 OpenAI 模型"** ❌
   - 实际: Anthropic Claude 也支持函数调用

2. **"ReAct Agent 已过时"** ❌
   - 实际: 开源模型场景下仍是最佳选择

3. **"Structured Chat 总是比 OpenAI Functions 好"** ❌
   - 实际: 简单工具用 OpenAI Functions 更高效

4. **"create_agent() 是全新 API"** ❌
   - 实际: 是对现有 create_*_agent() 的统一封装

5. **"Agent 类型一旦选定就不能更改"** ❌
   - 实际: 可以动态切换，只需更改创建函数

---

## 面试必问预告

1. **OpenAI Functions Agent 和 ReAct Agent 的核心区别是什么？**
2. **什么场景下必须使用 Structured Chat Agent？**
3. **如何从 initialize_agent() 迁移到 create_agent()？**
4. **Agent 不调用工具时如何诊断和解决？**
5. **2026 年推荐的 Agent 创建方式是什么？为什么？**

---

## 生成进度

### 阶段一：Plan 生成
- [x] 1.1 Brainstorm 分析
- [x] 1.2 多源数据收集（源码 + Context7）
- [x] 1.3 网络搜索补充（跳过，现有资料充足）
- [x] 1.4 用户确认拆解方案
- [x] 1.5 Plan 最终确定

### 阶段二：补充调研
- [x] 2.1 识别需要补充资料的部分（现有资料充足）
- [x] 2.2 执行补充调研（已完成）
- [x] 2.3 生成抓取任务文件（跳过）
- [x] 2.4 等待抓取完成（跳过）

### 阶段三：文档生成 ⬅️ 当前阶段
- [ ] 3.1 读取所有 reference/ 资料
- [ ] 3.2 生成基础维度文件（第一部分）
- [ ] 3.3 生成核心概念文件（6个）
- [ ] 3.4 生成基础维度文件（第二部分）
- [ ] 3.5 生成实战代码文件（7个）
- [ ] 3.6 生成基础维度文件（第三部分）
- [ ] 3.7 最终验证

---

## 质量保证

### 代码示例要求
- ✅ 所有代码必须基于 Python 3.13+
- ✅ 使用 langchain 最新 API (2026)
- ✅ 包含完整的导入语句
- ✅ 可直接运行（假设已配置 API key）
- ✅ 每个示例 100-200 行

### 引用规范
- 源码引用: `[来源: sourcecode/langchain/libs/langchain/langchain_classic/agents/agent_types.py]`
- Context7 引用: `[来源: reference/context7_langchain_agent_types_01.txt | LangChain 官方文档]`
- 网络引用: `[来源: reference/search_agent_types_01.md]`

### 文件长度控制
- 目标: 每个文件 300-500 行
- 超过 500 行自动拆分
- 核心概念文件: 400-500 行（深度讲解）
- 实战代码文件: 300-400 行（代码 + 注释）

---

**下一步**: 执行网络搜索，补充社区实践案例和最新讨论
