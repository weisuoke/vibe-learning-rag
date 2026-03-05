# ExampleSelector示例选择 - 生成计划

**生成时间：** 2026-02-26
**知识点层级：** L3_组件生态
**知识点编号：** 12
**目标受众：** 初学者 + 进阶学习者（全面覆盖）

---

## 数据来源记录

### 源码分析 ✅
- ✓ reference/source_example_selector_01.md - ExampleSelector 核心实现分析
  - BaseExampleSelector 抽象接口
  - SemanticSimilarityExampleSelector（语义相似度选择器）
  - MaxMarginalRelevanceExampleSelector（MMR 选择器）
  - LengthBasedExampleSelector（基于长度的选择器）
  - NGramOverlapExampleSelector（N-gram 重叠选择器）

### Context7 官方文档 ✅
- ✓ reference/context7_langchain_01.md - LangChain ExampleSelector 官方文档
  - 语义搜索在示例选择中的应用
  - find_similar 函数实现
  - LangSmith 中的 Few-shot 示例占位符
  - 为什么使用语义相似度选择示例
  - Few-Shot Prompting 与情景记忆

### 网络搜索 ✅
- ✓ reference/search_example_selector_01.md - FewShotPromptTemplate 教程搜索
  - FewShotPromptTemplate 核心用法
  - ExampleSelector 类型对比
  - Token 限制场景应用
  - 社区最佳实践

- ✓ reference/search_example_selector_02.md - 2025-2026 最新趋势搜索
  - 语义相似度选择成为主流
  - 混合策略（语义 + 长度）
  - 性能优化关注点
  - 常见误区总结

### 待抓取链接（将由第三方工具自动保存到 reference/）

**High 优先级（7 个）：**
- [ ] https://medium.com/donato-story/exploring-few-shot-prompts-with-langchain-852f27ea4e1d
  - 标签：核心概念_2_SemanticSimilarityExampleSelector
  - 原因：详细解释动态 few-shot 示例选择，包含代码演示

- [ ] https://github.com/gkamradt/langchain-tutorials/blob/main/LangChain%20Cookbook%20Part%201%20-%20Fundamentals.ipynb
  - 标签：实战场景_1_静态vs动态示例选择对比
  - 原因：完整的 Jupyter Notebook 示例

- [ ] https://www.pinecone.io/learn/series/langchain/langchain-prompt-templates
  - 标签：核心概念_3_LengthBasedExampleSelector
  - 原因：Pinecone 官方教程，介绍 LengthBasedExampleSelector

- [ ] https://pub.aimind.so/langchain-in-chains-6-example-selectors-310f47b4cdf3
  - 标签：核心概念_6_FewShotPromptTemplate集成
  - 原因：专门讲解 Example Selectors 的教程

- [ ] https://www.sandgarden.com/learn/few-shot-prompting
  - 标签：实战场景_2_语义相似度选择
  - 原因：Few-shot prompting 最佳实践

- [ ] https://www.swarnendu.de/blog/langchain-best-practices
  - 标签：核心概念_7_混合策略
  - 原因：2025 年 LangChain 最佳实践

- [ ] https://gist.github.com/sakha1370/cf4663c759099ae6c66e8ab3aa4b7a52
  - 标签：实战场景_6_RAG系统中的Few-shot优化
  - 原因：2025-2026 年最新更新笔记

**Medium 优先级（3 个）：**
- [ ] https://www.newline.co/@zaoyang/dynamic-prompts-with-langchain-templates--71d0c244
  - 标签：核心概念_6_FewShotPromptTemplate集成
  - 原因：动态 Prompt 构建教程

- [ ] https://dev.to/aiengineering/a-beginners-guide-to-few-shot-prompting-in-langchain-2ilm
  - 标签：实战场景_1_静态vs动态示例选择对比
  - 原因：初学者友好的教程

- [ ] https://github.com/whitesmith/langchain-semantic-length-example-selector
  - 标签：核心概念_7_混合策略
  - 原因：社区扩展项目，结合语义相似度和长度限制

---

## 文件清单

### 基础维度文件（第一部分）
- [ ] 00_概览.md
- [ ] 01_30字核心.md
- [ ] 02_第一性原理.md

### 核心概念文件（7 个，基于源码 + Context7 + 网络调研）
- [ ] 03_核心概念_1_BaseExampleSelector抽象接口.md
  - 统一的示例选择接口设计
  - 两个核心方法：add_example() 和 select_examples()
  - 异步支持
  - [来源: 源码 + Context7]

- [ ] 03_核心概念_2_SemanticSimilarityExampleSelector.md
  - 基于语义相似度的动态选择（最常用）
  - VectorStore 集成
  - from_examples() 类方法
  - [来源: 源码 + Context7 + 网络]

- [ ] 03_核心概念_3_LengthBasedExampleSelector.md
  - 基于长度限制的选择（Token 控制）
  - max_length 参数
  - 自定义长度计算函数
  - [来源: 源码 + 网络]

- [ ] 03_核心概念_4_MaxMarginalRelevanceExampleSelector.md
  - MMR 选择器（平衡相关性和多样性）
  - fetch_k 参数
  - 论文支持：https://arxiv.org/pdf/2211.13892.pdf
  - [来源: 源码]

- [ ] 03_核心概念_5_NGramOverlapExampleSelector.md
  - N-gram 重叠选择器（轻量级）
  - 不依赖向量化
  - 适合文本相似度场景
  - [来源: 源码]

- [ ] 03_核心概念_6_FewShotPromptTemplate集成.md
  - 如何在 Prompt 中使用 ExampleSelector
  - 静态示例 vs 动态示例
  - LangSmith 中的 {{few_shot_examples}} 占位符
  - [来源: Context7 + 网络]

- [ ] 03_核心概念_7_混合策略.md
  - 语义相似度 + 长度限制（2025-2026 新趋势）
  - 社区扩展项目
  - 最佳实践
  - [来源: 网络]

### 基础维度文件（第二部分）
- [ ] 04_最小可用.md
- [ ] 05_双重类比.md
- [ ] 06_反直觉点.md

### 实战代码文件（6 个，基于源码 + Context7 + 网络调研）
- [ ] 07_实战代码_场景1_静态vs动态示例选择对比.md
  - 固定示例 vs ExampleSelector
  - 性能对比
  - 适用场景分析
  - [来源: Context7 + 网络]

- [ ] 07_实战代码_场景2_语义相似度选择（问答系统）.md
  - SemanticSimilarityExampleSelector 实战
  - Chroma 向量存储集成
  - 问答系统示例
  - [来源: 源码 + Context7 + 网络]

- [ ] 07_实战代码_场景3_长度限制选择（Token控制）.md
  - LengthBasedExampleSelector 实战
  - Token 限制场景
  - 自定义长度计算
  - [来源: 源码 + 网络]

- [ ] 07_实战代码_场景4_MMR选择（提高多样性）.md
  - MaxMarginalRelevanceExampleSelector 实战
  - 多样性优化
  - fetch_k 参数调优
  - [来源: 源码]

- [ ] 07_实战代码_场景5_混合策略（语义+长度）.md
  - 语义相似度 + 长度限制结合
  - 社区扩展项目使用
  - 生产环境最佳实践
  - [来源: 网络]

- [ ] 07_实战代码_场景6_RAG系统中的Few-shot优化.md
  - RAG 系统中的 Few-shot learning
  - 动态示例选择优化 Prompt
  - 完整的 RAG + ExampleSelector 示例
  - [来源: Context7 + 网络]

### 基础维度文件（第三部分）
- [ ] 08_面试必问.md
- [ ] 09_化骨绵掌.md
- [ ] 10_一句话总结.md

---

## 知识点拆解方案

### 核心概念（3 个重点）
1. **BaseExampleSelector 抽象接口** - 统一的示例选择接口设计
2. **SemanticSimilarityExampleSelector** - 基于语义相似度的动态选择（最常用）
3. **LengthBasedExampleSelector** - 基于长度限制的选择（Token 控制）

### 扩展概念（全面覆盖）
4. **MaxMarginalRelevanceExampleSelector** - MMR 选择器（平衡相关性和多样性）
5. **NGramOverlapExampleSelector** - N-gram 重叠选择器（轻量级）
6. **FewShotPromptTemplate 集成** - 如何在 Prompt 中使用 ExampleSelector
7. **混合策略** - 语义相似度 + 长度限制（2025-2026 新趋势）

### 实战场景（6 个）
1. 静态 vs 动态示例选择对比
2. 语义相似度选择（问答系统）
3. 长度限制选择（Token 控制）
4. MMR 选择（提高多样性）
5. 混合策略（语义 + 长度）
6. RAG 系统中的 Few-shot 优化

---

## 生成进度

### 阶段一：Plan 生成 ✅
- [x] 1.1 Brainstorm 分析
  - 用户选择：全面覆盖（平衡）
- [x] 1.2 多源数据收集（源码 + Context7 + 网络）
  - 源码分析：1 个文件
  - Context7 文档：1 个文件
  - 网络搜索：2 个文件
  - **总计：4 个资料文件**
- [x] 1.3 用户确认拆解方案
  - 用户确认：完全符合，继续执行
- [x] 1.4 Plan 最终确定
  - PLAN.md 已生成

### 阶段二：补充调研（针对需要更多资料的部分）
- [x] 2.1 识别需要补充资料的部分
  - 核心概念_6_FewShotPromptTemplate集成
  - 核心概念_7_混合策略
  - 实战场景_1_静态vs动态示例选择对比
  - 实战场景_6_RAG系统中的Few-shot优化
- [x] 2.2 执行补充调研（已通过网络搜索完成）
- [x] 2.3 生成抓取任务文件（FETCH_TASK.json）
  - 总 URL 数：10 个
  - High 优先级：7 个
  - Medium 优先级：3 个
- [x] 2.4 更新 PLAN.md
- [x] 2.5 输出抓取任务提示
- [ ] 2.6 检查抓取完成状态
- [ ] 2.7 更新 PLAN.md
- [ ] 2.8 生成资料索引文件（reference/INDEX.md）

### 阶段三：文档生成（读取 reference/ 资料）
- [ ] 3.1 读取所有 reference/ 资料
- [ ] 3.2 按顺序生成文档
  - [ ] 基础维度文件（第一部分）：3 个
  - [ ] 核心概念文件：7 个
  - [ ] 基础维度文件（第二部分）：3 个
  - [ ] 实战代码文件：6 个
  - [ ] 基础维度文件（第三部分）：3 个
  - **总计：22 个文件**
- [ ] 3.3 生成规范检查
- [ ] 3.4 最终验证

---

## 质量标准

### 内容要求
- ✅ 所有内容基于 2025-2026 年最新资料
- ✅ 代码必须完整可运行（Python 3.13+）
- ✅ 每个文件 300-500 行
- ✅ 联系 AI Agent 开发实际应用
- ✅ 初学者友好 + 源码深度

### 数据来源优先级
1. **源码分析**（第一优先级）✅
2. **Context7 官方文档**（第二优先级）✅
3. **Grok-mcp 网络搜索**（第三优先级）✅
4. **Grok-mcp 网络抓取**（第四优先级）⏳

### 引用规范
- 源码引用：`[来源: sourcecode/langchain/<文件路径>]`
- Context7 引用：`[来源: reference/context7_*.md | LangChain 官方文档]`
- 搜索结果引用：`[来源: reference/search_*.md]`
- 抓取内容引用：`[来源: reference/fetch_*.md | <原始URL>]`

---

## 关键技术点

### 从源码中提取
- BaseExampleSelector 抽象接口设计
- SemanticSimilarityExampleSelector 实现细节
- MaxMarginalRelevanceExampleSelector MMR 算法
- LengthBasedExampleSelector 长度计算
- 异步支持机制

### 从 Context7 中提取
- 语义搜索在示例选择中的应用
- find_similar 函数手动实现
- LangSmith 中的 Few-shot 示例管理
- 为什么需要动态选择示例

### 从网络搜索中提取
- 2025-2026 年最新趋势
- 混合策略（语义 + 长度）
- 性能优化最佳实践
- 常见误区总结

---

## 下一步操作

### 立即执行
1. **阶段二：补充调研**
   - 识别需要更多资料的部分
   - 生成 FETCH_TASK.json
   - 等待第三方工具抓取

2. **阶段三：文档生成**
   - 抓取完成后，开始文档生成
   - 使用 subagent 批量生成文件
   - 严格遵循 300-500 行长度限制

### 预期输出
- **22 个内容文件**（00_概览.md ~ 10_一句话总结.md）
- **1 个 FETCH_TASK.json**（抓取任务文件）
- **1 个 FETCH_REPORT.md**（抓取报告，由第三方工具生成）
- **1 个 reference/INDEX.md**（资料索引）

---

**版本：** v1.0
**最后更新：** 2026-02-26
**维护者：** Claude Code
