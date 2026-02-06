# L2_LLM核心 知识点列表

> 掌握大语言模型的核心原理与实战技能，为 RAG 开发奠定坚实基础

---

## 知识点清单

1. **Transformer直觉理解** - 理解大模型的工作原理，知其所以然
2. **大模型API调用** - 掌握 OpenAI/Anthropic 等主流 API 的实战使用
3. **Token与Context Window** - 理解 LLM 的核心限制，这是 RAG 设计的关键约束
4. **Prompt Engineering基础** - 掌握提示词工程的核心技巧（角色设定、指令清晰、格式控制）
5. **Prompt Engineering进阶** - 掌握 Few-shot、Chain-of-Thought、结构化输出等高级技巧

---

## 学习顺序建议

```
Transformer直觉理解 → 大模型API调用 → Token与Context Window → Prompt基础 → Prompt进阶
       ↓                   ↓                    ↓                 ↓            ↓
   理解原理            学会使用              理解限制          基础交互      高级技巧
```

**为什么是这个顺序？**

1. **先原理后实践**：理解 Transformer 的注意力机制，才能明白为什么 Prompt 设计如此重要
2. **先使用后优化**：学会 API 调用后，才能在实践中理解 Token 限制的影响
3. **Token 是 RAG 的核心约束**：Context Window 大小直接决定了 RAG 能注入多少检索内容
4. **Prompt 是 RAG 的核心交互**：RAG 的最终输出质量很大程度取决于 Prompt 设计

---

## 与 RAG 开发的关系

| 知识点 | 在 RAG 中的应用 |
|--------|----------------|
| Transformer直觉理解 | 理解为什么 LLM 能理解上下文、为什么位置很重要 |
| 大模型API调用 | RAG 系统的生成模块核心，调用 LLM 生成最终答案 |
| Token与Context Window | 决定能注入多少检索内容、如何分配 Token 预算 |
| Prompt Engineering基础 | 设计 RAG 的系统提示词、用户问题模板 |
| Prompt Engineering进阶 | 优化 RAG 输出质量、处理复杂推理任务 |

---

## 前置知识

- ✅ L1_NLP基础（BPE分词、Embedding、语义相似度）

## 后续学习

- → L3_RAG核心流程（文档加载、分块、检索、生成）

---

**版本：** v1.0
**最后更新：** 2025-02-05
