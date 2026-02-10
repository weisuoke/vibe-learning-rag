# Phase4_AI_Agent开发 知识点列表

> 将 LangChain/LangGraph 集成到 FastAPI，构建生产级 AI Agent 后端

---

## 知识点清单

1. **LangChain LCEL** - LangChain 表达式语言，链式调用 LLM 和工具
2. **Agent执行器** - Agent 的工具调用循环，ReAct 模式
3. **对话记忆管理** - 对话历史存储和上下文管理
4. **RAG检索链** - 向量检索 + LLM 生成的完整流程
5. **流式输出集成** - 将 LangChain 流式输出通过 FastAPI 返回
6. **自定义Tool** - 为 Agent 创建自定义工具（API 调用、数据库查询等）

---

## 学习顺序建议

```
LangChain LCEL → RAG检索链 → 对话记忆管理 → Agent执行器 → 自定义Tool → 流式输出集成
      ↓            ↓            ↓              ↓           ↓            ↓
   链式调用      RAG基础      上下文管理      工具调用    扩展能力      实时响应
```

**为什么是这个顺序？**

1. **LangChain LCEL（1）**：LangChain 的核心语法，链式调用基础
2. **RAG检索链（2）**：最常用的 AI Agent 模式，检索 + 生成
3. **对话记忆管理（3）**：多轮对话的上下文维护
4. **Agent执行器（4）**：更高级的 Agent 模式，自主决策工具调用
5. **自定义Tool（5）**：扩展 Agent 能力，调用外部 API 或数据库
6. **流式输出集成（6）**：提升用户体验，实时显示 AI 生成内容

---

## 与 AI Agent 开发的关系

| 知识点 | 在 AI Agent 后端中的应用 | 关键产出 |
|--------|-------------------------|----------|
| LangChain LCEL | 构建 Prompt → LLM → 解析的链 | 可组合的 AI 流程 |
| Agent执行器 | 自主调用工具的智能 Agent | 自动化任务执行 |
| 对话记忆管理 | 多轮对话上下文维护 | 连贯的对话体验 |
| RAG检索链 | 文档问答、知识库检索 | 基于知识的回答 |
| 流式输出集成 | 实时显示 AI 生成内容 | 流畅的用户体验 |
| 自定义Tool | 调用数据库、API、计算器等 | 扩展 Agent 能力 |

---

## 前置知识

- ✅ Phase1_Python基础强化（异步编程）
- ✅ Phase2_FastAPI核心（流式响应、后台任务）
- ✅ Phase3_数据库层（向量检索pgvector）
- ✅ RAG 基础概念（Embedding、向量检索、Prompt Engineering）

## 后续学习

- → Phase5_生产级实践（错误处理、长任务处理）
- → Phase6_部署与架构（容器化部署）
- → RAG 进阶优化（ReRank、Query改写）

---

## 学习检查清单

完成本阶段后，你应该能够：

- [ ] 用 LCEL 构建 Prompt → LLM → Parser 链
- [ ] 实现基础的 RAG 检索链
- [ ] 使用 ConversationBufferMemory 管理对话历史
- [ ] 创建 Agent 并配置工具
- [ ] 自定义 Tool 让 Agent 调用外部 API
- [ ] 将 LangChain 流式输出通过 FastAPI 返回给前端

---

**版本：** v1.0
**最后更新：** 2026-02-10
