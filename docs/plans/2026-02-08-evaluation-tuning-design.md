# 07_评估与调优 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate 15 markdown files for the "评估与调优" knowledge point covering RAG system evaluation metrics and optimization methods.

**Architecture:** Content generation following the established pattern from `06_幻觉检测与缓解/`. Each file follows the CLAUDE.md specification with 10 dimensions split across 15 files (3 core concept files + 4 practical code files + 8 standard dimension files).

**Tech Stack:** Markdown, Python code examples (openai, ragas, langchain, chromadb, sentence-transformers)

**Reference files:** `atom/L4_RAG进阶优化/06_幻觉检测与缓解/` (same structure pattern)

---

## Batch 1: Simple Dimensions (independent, can run in parallel)

### Task 1: Create directory + 01_30字核心.md

**Files:**
- Create: `atom/L4_RAG进阶优化/07_评估与调优/01_30字核心.md`

**Content spec:**
- Header: `# 30字核心` with subtitle `> 用一句话说清评估与调优的本质`
- Core definition (~30 chars): **评估与调优是通过量化指标衡量 RAG 系统检索与生成质量，并基于数据驱动持续优化的系统方法。**
- Sections: 核心定义, 为什么对 RAG 系统至关重要, 核心要点, 一句话记住
- "为什么重要" covers: (1) RAG 系统需要量化质量 (2) 评估驱动优化闭环 (3) 检索侧 vs 生成侧评估
- Include a simple ASCII diagram showing evaluation loop: 检索→生成→评估→优化→检索
- Include a comparison table: 没有评估 vs 有评估
- Include pseudocode showing evaluation flow
- Target: ~100 lines
- Style: Match `06_幻觉检测与缓解/01_30字核心.md` exactly

### Task 2: Create 15_一句话总结.md

**Files:**
- Create: `atom/L4_RAG进阶优化/07_评估与调优/15_一句话总结.md`

**Content spec:**
- Header: `# 一句话总结` with subtitle `> 综合总结评估与调优的核心要点`
- Core summary: **评估与调优是 RAG 系统质量保障的核心方法论，通过 RAGAS 框架统一评估、检索侧指标（Precision/Recall/MRR/NDCG）衡量召回质量、生成侧指标（Faithfulness/Relevance/Correctness）衡量回答质量，三维评估驱动数据化优化，让 RAG 系统从"能用"升级到"好用"。**
- Sections: 核心总结, 三个核心技术 (RAGAS/检索侧/生成侧), 为什么三个技术缺一不可, 在 RAG 开发中的价值 (comparison table), 实施要点 (分层策略 + 成本效果表 + 关键指标), 常见误区, 最终要点, 下一步学习
- Target: ~150 lines
- Style: Match `06_幻觉检测与缓解/15_一句话总结.md` exactly

---

## Batch 2: Foundation Dimensions (independent, can run in parallel)

### Task 3: Create 02_第一性原理.md

**Files:**
- Create: `atom/L4_RAG进阶优化/07_评估与调优/02_第一性原理.md`

**Content spec:**
- Header: `# 第一性原理` with subtitle `> 回到评估与调优的最基础真理，从源头思考问题`
- Structure:
  1. 什么是第一性原理？(brief intro)
  2. 评估与调优的第一性原理
     - 最基础的定义: **评估 = 用量化指标衡量系统输出与期望之间的差距**
     - 为什么需要评估与调优？核心问题：RAG 系统的质量不可见
     - 三层价值: 可量化（数字化质量）、可比较（A/B 对比）、可优化（数据驱动改进）
  3. 从第一性原理推导 RAG 评估体系
     - 推理链: 系统有输入输出 → 输出有好坏 → 好坏需要标准 → 标准需要量化 → 量化需要指标 → 指标分检索和生成 → 综合指标驱动优化
  4. 评估的本质：反馈回路
     - 控制论视角：评估是系统的"感知器官"
  5. 一句话总结第一性原理
- Include ASCII diagrams, code pseudocode, comparison tables
- Target: ~300 lines

### Task 4: Create 06_最小可用.md

**Files:**
- Create: `atom/L4_RAG进阶优化/07_评估与调优/06_最小可用.md`

**Content spec:**
- Header: `# 最小可用` with subtitle `> 掌握哪 20% 就能解决 80% 问题`
- 5 core knowledge points:
  1. 构建评估数据集（question + ground_truth + contexts + answer）
  2. 用 RAGAS 跑一次端到端评估（5行代码）
  3. 理解4个核心指标（faithfulness, answer_relevancy, context_precision, context_recall）
  4. 检索质量快速诊断（Precision@K + Recall@K）
  5. 基于评估结果的优化方向判断
- Each point: 概念说明 + Python code + 应用场景
- End with "这些知识足以" checklist
- Target: ~300 lines

### Task 5: Create 07_双重类比.md

**Files:**
- Create: `atom/L4_RAG进阶优化/07_评估与调优/07_双重类比.md`

**Content spec:**
- Header: `# 双重类比` with subtitle `> 用前端开发和日常生活的类比理解评估与调优`
- 5 analogies:
  1. RAGAS 框架 → 前端: Lighthouse/PageSpeed → 生活: 体检报告
  2. Precision@K → 前端: 搜索结果相关率 → 生活: 钓鱼命中率
  3. Recall@K → 前端: 搜索覆盖率 → 生活: 考试知识覆盖率
  4. Faithfulness → 前端: 数据绑定一致性 → 生活: 新闻报道忠实度
  5. 评估驱动优化 → 前端: A/B Testing → 生活: 运动员训练数据分析
- Each analogy: 前端类比 + 日常生活类比 + Python code comparison
- End with summary comparison table
- Target: ~350 lines

### Task 6: Create 08_反直觉点.md

**Files:**
- Create: `atom/L4_RAG进阶优化/07_评估与调优/08_反直觉点.md`

**Content spec:**
- Header: `# 反直觉点` with subtitle `> 评估与调优最容易错在哪`
- 3 misconceptions:
  1. ❌ "评估指标越高越好" → 正确: 指标之间存在 trade-off（precision vs recall），需要根据场景平衡
  2. ❌ "有了 RAGAS 就不需要人工评估" → 正确: 自动评估有盲区（创意性、用户满意度），需要人机结合
  3. ❌ "检索好了生成自然就好" → 正确: 检索质量是必要不充分条件，生成侧有独立的失败模式（幻觉、格式错误、信息遗漏）
- Each: 错误观点 + 为什么错 + 为什么人们容易这样错 + 正确理解(with code)
- Target: ~350 lines

### Task 7: Create 13_面试必问.md

**Files:**
- Create: `atom/L4_RAG进阶优化/07_评估与调优/13_面试必问.md`

**Content spec:**
- Header: `# 面试必问` with subtitle `> 如果被问到评估与调优，怎么答出彩`
- 2 interview questions:
  1. "如何评估一个 RAG 系统的效果？"
  2. "RAG 系统上线后效果不好，你会怎么排查和优化？"
- Each: 普通回答(❌) + 出彩回答(✅) + 为什么出彩
- 出彩回答 structure: 多层次（原理/指标/实践）+ 具体例子 + RAG 联系
- Target: ~300 lines

---

## Batch 3: Core Concepts (independent, can run in parallel)

### Task 8: Create 03_核心概念_RAGAS评估框架.md

**Files:**
- Create: `atom/L4_RAG进阶优化/07_评估与调优/03_核心概念_RAGAS评估框架.md`

**Content spec:**
- Header: `# 核心概念：RAGAS 评估框架`
- Sections:
  1. RAGAS 是什么（一句话定义 + 详细解释）
  2. RAGAS 的4个核心指标:
     - Faithfulness（忠实度）: 生成答案是否忠于检索上下文
     - Answer Relevancy（答案相关性）: 答案是否回答了问题
     - Context Precision（上下文精确度）: 检索的上下文中相关内容排名是否靠前
     - Context Recall（上下文召回率）: 检索的上下文是否覆盖了 ground truth
  3. 每个指标: 定义 + 计算公式(文字描述) + Python 代码示例 + 直觉理解
  4. RAGAS 评估数据集格式（question, answer, contexts, ground_truth）
  5. 快速上手代码（pip install ragas + evaluate）
  6. RAGAS 的局限性
  7. 在 RAG 开发中的应用
- Target: ~400 lines

### Task 9: Create 04_核心概念_检索侧评估指标.md

**Files:**
- Create: `atom/L4_RAG进阶优化/07_评估与调优/04_核心概念_检索侧评估指标.md`

**Content spec:**
- Header: `# 核心概念：检索侧评估指标`
- Sections:
  1. 为什么需要单独评估检索质量
  2. 核心指标详解:
     - Precision@K: 前K个结果中相关的比例（公式 + 手写实现 + 例子）
     - Recall@K: 前K个结果覆盖了多少相关文档（公式 + 手写实现 + 例子）
     - MRR (Mean Reciprocal Rank): 第一个相关结果的排名倒数（公式 + 手写实现 + 例子）
     - NDCG (Normalized Discounted Cumulative Gain): 考虑排名位置的综合指标（公式 + 手写实现 + 例子）
     - Hit Rate: 是否至少命中一个相关文档
  3. 指标对比表（适用场景、优缺点）
  4. 如何选择指标
  5. 在 RAG 开发中的应用
- All formulas explained in plain Chinese + code implementation
- Target: ~400 lines

### Task 10: Create 05_核心概念_生成侧评估指标.md

**Files:**
- Create: `atom/L4_RAG进阶优化/07_评估与调优/05_核心概念_生成侧评估指标.md`

**Content spec:**
- Header: `# 核心概念：生成侧评估指标`
- Sections:
  1. 为什么需要单独评估生成质量
  2. 核心指标详解:
     - Faithfulness（忠实度）: LLM 是否忠于检索内容（NLI-based + LLM-as-judge）
     - Answer Relevancy（答案相关性）: 答案是否切题（embedding similarity + LLM-as-judge）
     - Answer Correctness（答案正确性）: 答案是否与 ground truth 一致（F1 + semantic similarity）
     - Answer Completeness（答案完整性）: 答案是否覆盖了所有要点
  3. LLM-as-Judge 方法详解（prompt 设计 + 评分标准 + 多评委投票）
  4. 自动评估 vs 人工评估的对比
  5. 指标对比表
  6. 在 RAG 开发中的应用
- Target: ~400 lines

---

## Batch 4: Practical Code (independent, can run in parallel)

### Task 11: Create 09_实战代码_RAGAS端到端评估.md

**Files:**
- Create: `atom/L4_RAG进阶优化/07_评估与调优/09_实战代码_RAGAS端到端评估.md`

**Content spec:**
- Complete runnable Python example using RAGAS
- Sections:
  1. 环境准备（pip install ragas datasets langchain-openai）
  2. 构建评估数据集（手动构建 + 从文件加载）
  3. 配置 RAGAS 评估（选择指标、配置 LLM）
  4. 运行评估并解读结果
  5. 结果可视化（文本表格输出）
  6. 常见问题排查
- Code must handle API key via dotenv
- Include mock/simulated output for reference
- Target: ~200 lines

### Task 12: Create 10_实战代码_检索质量评估.md

**Files:**
- Create: `atom/L4_RAG进阶优化/07_评估与调优/10_实战代码_检索质量评估.md`

**Content spec:**
- Complete runnable Python example for retrieval evaluation
- Sections:
  1. 手写实现 Precision@K, Recall@K, MRR, NDCG, Hit Rate
  2. 构建检索评估数据集（query + relevant_docs + retrieved_docs）
  3. 批量评估多个查询
  4. 结果分析与可视化（文本表格）
  5. 与 RAG 系统集成的评估
- Pure Python implementation (no external eval library needed)
- Target: ~200 lines

### Task 13: Create 11_实战代码_生成质量评估.md

**Files:**
- Create: `atom/L4_RAG进阶优化/07_评估与调优/11_实战代码_生成质量评估.md`

**Content spec:**
- Complete runnable Python example for generation evaluation
- Sections:
  1. LLM-as-Judge 实现（使用 OpenAI API）
  2. Faithfulness 评估（检查答案与上下文一致性）
  3. Answer Relevancy 评估（检查答案与问题相关性）
  4. Answer Correctness 评估（与 ground truth 对比）
  5. 多维度综合评分
  6. 批量评估与结果汇总
- Uses openai library for LLM-as-judge
- Target: ~200 lines

### Task 14: Create 12_实战代码_自动化评估Pipeline.md

**Files:**
- Create: `atom/L4_RAG进阶优化/07_评估与调优/12_实战代码_自动化评估Pipeline.md`

**Content spec:**
- Complete runnable Python example for automated evaluation pipeline
- Sections:
  1. 端到端评估 Pipeline 设计
  2. 数据集管理（加载、缓存、版本控制）
  3. 多维度评估执行器
  4. 结果存储与对比（JSON 输出）
  5. A/B 测试框架（对比两个 RAG 配置）
  6. 评估报告生成（文本格式）
- Combines retrieval + generation evaluation
- Target: ~200 lines

---

## Batch 5: Knowledge Cards

### Task 15: Create 14_化骨绵掌.md

**Files:**
- Create: `atom/L4_RAG进阶优化/07_评估与调优/14_化骨绵掌.md`

**Content spec:**
- 10 knowledge cards, each ~200 words, 2-minute read
- Progressive structure:
  1. 直觉理解：为什么需要评估 RAG 系统
  2. 形式化定义：评估指标的数学本质
  3. RAGAS 框架：4个核心指标速览
  4. Precision 与 Recall：检索质量的两个维度
  5. MRR 与 NDCG：排序质量的衡量
  6. Faithfulness：生成忠实度评估
  7. Answer Relevancy：答案相关性评估
  8. LLM-as-Judge：用 LLM 评估 LLM
  9. 评估驱动优化：从指标到行动
  10. 总结与延伸：构建评估体系的最佳实践
- Each card: 一句话 + 举例/代码 + 应用
- Target: ~400 lines

---

## Execution Notes

- **Parallel batches:** Tasks within each batch are independent and can be dispatched in parallel
- **Dependencies:** Batch 1 must complete first (creates directory), then Batches 2-4 can run in parallel, Batch 5 last
- **Style reference:** All files must match the style of `atom/L4_RAG进阶优化/06_幻觉检测与缓解/`
- **CLAUDE.md compliance:** Follow all 10 dimensions as specified in CLAUDE.md
- **Code quality:** All Python code must be complete, runnable, with comments and print output
- **Length control:** Each file 300-500 lines max; code files ~200 lines
