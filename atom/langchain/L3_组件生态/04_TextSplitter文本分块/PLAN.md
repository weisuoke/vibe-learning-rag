# 04_TextSplitter文本分块 - 生成计划

## 数据来源记录

### 源码分析（已完成）
- ✓ reference/source_textsplitter_01_base.md - TextSplitter 基类架构分析
- ✓ reference/source_textsplitter_02_character.md - CharacterTextSplitter 和 RecursiveCharacterTextSplitter 分析
- ✓ reference/source_textsplitter_03_markdown.md - Markdown 文本分块器分析
- ✓ reference/source_textsplitter_04_html.md - HTML 文本分块器分析

### Context7 官方文档（已完成）
- ✓ reference/context7_langchain_01.md - LangChain TextSplitter 官方文档

### 网络搜索（已完成）
- ✓ reference/search_textsplitter_01.md - LangChain TextSplitter 最佳实践（2025-2026）

### 数据来源统计
- 源码分析：4 个文件
- Context7 文档：1 个文件
- 网络搜索：1 个文件
- **总计：6 个资料文件**

## 文件清单

### 基础维度文件（10个）
- [ ] 00_概览.md
- [ ] 01_30字核心.md
- [ ] 02_第一性原理.md
- [ ] 04_最小可用.md
- [ ] 05_双重类比.md
- [ ] 06_反直觉点.md
- [ ] 08_面试必问.md
- [ ] 09_化骨绵掌.md
- [ ] 10_一句话总结.md

### 核心概念文件（9个，基于源码 + Context7 + 网络调研）
- [ ] 03_核心概念_1_TextSplitter基类架构.md - 参数、方法、_merge_splits算法 [来源: 源码]
- [ ] 03_核心概念_2_CharacterTextSplitter.md - 基于分隔符的简单分块 [来源: 源码]
- [ ] 03_核心概念_3_RecursiveCharacterTextSplitter.md - 递归分块算法（最常用） [来源: 源码+Context7+网络]
- [ ] 03_核心概念_4_TokenTextSplitter.md - 基于 token 计数的分块 [来源: 源码+Context7]
- [ ] 03_核心概念_5_HTMLHeaderTextSplitter.md - 保留 HTML 结构的分块 [来源: 源码]
- [ ] 03_核心概念_6_MarkdownHeaderTextSplitter.md - 保留 Markdown 层级的分块 [来源: 源码]
- [ ] 03_核心概念_7_PythonCodeTextSplitter.md - 代码语法感知分块 [来源: 源码]
- [ ] 03_核心概念_8_RecursiveJsonSplitter.md - JSON 结构分块 [来源: 源码]
- [ ] 03_核心概念_9_语义分块与NLP分块器.md - NLTKTextSplitter, SpacyTextSplitter [来源: 源码]

### 实战代码文件（6个，基于源码 + Context7 + 网络调研）
- [ ] 07_实战代码_场景1_RAG文档分块管道.md - 完整的文档加载→分块→向量化流程 [来源: Context7+网络]
- [ ] 07_实战代码_场景2_代码仓库分析.md - 使用 PythonCodeTextSplitter 分析代码 [来源: 源码]
- [ ] 07_实战代码_场景3_网页内容提取.md - 使用 HTMLHeaderTextSplitter 保留网页结构 [来源: 源码]
- [ ] 07_实战代码_场景4_Markdown文档处理.md - 使用 MarkdownHeaderTextSplitter 保留层级 [来源: 源码]
- [ ] 07_实战代码_场景5_Token优化分块.md - 使用 TokenTextSplitter 精确控制 token 数量 [来源: Context7]
- [ ] 07_实战代码_场景6_分块策略对比.md - 对比不同分块器的效果和性能测试 [来源: 网络]

**总计：25 个文件**

## 核心概念数据来源映射

### 1. TextSplitter基类架构
**数据来源**：
- 源码：source_textsplitter_01_base.md
- 关键内容：chunk_size, chunk_overlap, _merge_splits() 算法, length_function 机制

### 2. CharacterTextSplitter
**数据来源**：
- 源码：source_textsplitter_02_character.md
- 关键内容：separator 参数, 正则支持, 零宽度断言处理

### 3. RecursiveCharacterTextSplitter（最重要）
**数据来源**：
- 源码：source_textsplitter_02_character.md
- Context7：context7_langchain_01.md
- 网络：search_textsplitter_01.md
- 关键内容：递归算法, separators 优先级, 最佳实践配置（chunk_size=512/1000）

### 4. TokenTextSplitter
**数据来源**：
- 源码：source_textsplitter_01_base.md
- Context7：context7_langchain_01.md
- 关键内容：tiktoken 集成, token 计数, split_text_on_tokens() 函数

### 5. HTMLHeaderTextSplitter
**数据来源**：
- 源码：source_textsplitter_04_html.md
- 关键内容：HTML 标签识别, 层级元数据, BeautifulSoup 集成

### 6. MarkdownHeaderTextSplitter
**数据来源**：
- 源码：source_textsplitter_03_markdown.md
- 关键内容：标题层级跟踪, 代码块检测, 自定义标题模式

### 7. PythonCodeTextSplitter
**数据来源**：
- 源码：source_textsplitter_02_character.md（from_language 方法）
- 关键内容：语言特定分隔符, 20+ 种语言支持

### 8. RecursiveJsonSplitter
**数据来源**：
- 源码：需要补充（未在已读取的文件中）
- 关键内容：JSON 结构分块

### 9. 语义分块与NLP分块器
**数据来源**：
- 源码：需要补充（未在已读取的文件中）
- 关键内容：NLTKTextSplitter, SpacyTextSplitter

## 实战场景数据来源映射

### 场景1：RAG文档分块管道
**数据来源**：
- Context7：context7_langchain_01.md（完整流程示例）
- 网络：search_textsplitter_01.md（最佳实践）
- 关键内容：DocumentLoader → TextSplitter → Embedding → VectorStore

### 场景2：代码仓库分析
**数据来源**：
- 源码：source_textsplitter_02_character.md（from_language 方法）
- 关键内容：PythonCodeTextSplitter, 语法感知分块

### 场景3：网页内容提取
**数据来源**：
- 源码：source_textsplitter_04_html.md
- 关键内容：HTMLHeaderTextSplitter, 元数据提取

### 场景4：Markdown文档处理
**数据来源**：
- 源码：source_textsplitter_03_markdown.md
- 关键内容：MarkdownHeaderTextSplitter, 层级保留

### 场景5：Token优化分块
**数据来源**：
- Context7：context7_langchain_01.md
- 关键内容：from_tiktoken_encoder, 成本优化

### 场景6：分块策略对比
**数据来源**：
- 网络：search_textsplitter_01.md（基准测试）
- 关键内容：7种分块策略对比, 性能测试

## 关键发现总结

### 推荐配置（2025-2026）
- **chunk_size**: 512 tokens（基准测试最佳）/ 1000 characters（官方推荐）
- **chunk_overlap**: 150-200 characters（15-20% 重叠率）
- **默认分块器**: RecursiveCharacterTextSplitter

### 常见问题
1. 边界偏移问题：使用 Jaccard 匹配修复
2. 复杂 PDF 处理：结合 MarkdownHeaderTextSplitter
3. 分隔符优化：添加句子级分隔符

### 2025-2026 新趋势
- OpenSearch 集成 RecursiveCharacterTextSplitter
- 自动优化工具（rag-chunk）
- 基于实际数据的参数推荐

## 需要补充的内容

### 源码分析（可选）
- RecursiveJsonSplitter 实现
- NLTKTextSplitter 和 SpacyTextSplitter 实现

**决策**：这两个分块器使用频率较低，可以在核心概念文件中简要介绍，不需要单独的源码分析文件。

## 生成进度

### 阶段一：Plan 生成（已完成）
- [x] 1.1 Brainstorm 分析
- [x] 1.2 多源数据收集（源码 + Context7 + 网络）
- [ ] 1.3 用户确认拆解方案
- [ ] 1.4 Plan 最终确定

### 阶段二：补充调研（跳过）
- 决策：现有资料已足够，不需要额外的网络抓取

### 阶段三：文档生成（待执行）
- [ ] 3.1 读取所有 reference/ 资料
- [ ] 3.2 按顺序生成文档（25个文件）
- [ ] 3.3 最终验证

## 文档生成顺序

### 第一批：基础维度（第一部分）
1. 00_概览.md
2. 01_30字核心.md
3. 02_第一性原理.md

### 第二批：核心概念（9个文件）
4. 03_核心概念_1_TextSplitter基类架构.md
5. 03_核心概念_2_CharacterTextSplitter.md
6. 03_核心概念_3_RecursiveCharacterTextSplitter.md
7. 03_核心概念_4_TokenTextSplitter.md
8. 03_核心概念_5_HTMLHeaderTextSplitter.md
9. 03_核心概念_6_MarkdownHeaderTextSplitter.md
10. 03_核心概念_7_PythonCodeTextSplitter.md
11. 03_核心概念_8_RecursiveJsonSplitter.md
12. 03_核心概念_9_语义分块与NLP分块器.md

### 第三批：基础维度（第二部分）
13. 04_最小可用.md
14. 05_双重类比.md
15. 06_反直觉点.md

### 第四批：实战代码（6个文件）
16. 07_实战代码_场景1_RAG文档分块管道.md
17. 07_实战代码_场景2_代码仓库分析.md
18. 07_实战代码_场景3_网页内容提取.md
19. 07_实战代码_场景4_Markdown文档处理.md
20. 07_实战代码_场景5_Token优化分块.md
21. 07_实战代码_场景6_分块策略对比.md

### 第五批：基础维度（第三部分）
22. 08_面试必问.md
23. 09_化骨绵掌.md
24. 10_一句话总结.md

## 质量保证

### 每个文件生成后检查
- [ ] 文件长度：300-500 行
- [ ] 代码完整性：所有代码可运行
- [ ] 引用来源：包含明确的引用
- [ ] 技术深度：原理 + 实现 + 应用

### 最终验证
- [ ] 所有25个文件已生成
- [ ] 文件长度符合规范
- [ ] 引用来源完整
- [ ] 代码示例可运行

---

**生成时间**：2026-02-25
**知识点**：04_TextSplitter文本分块
**层级**：L3_组件生态
**总文件数**：25个
**资料文件数**：6个
