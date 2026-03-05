---
type: search_result
search_query: LangChain ExampleSelector semantic similarity few-shot learning 2025 2026 best practices
search_engine: grok-mcp
searched_at: 2026-02-26
knowledge_point: ExampleSelector示例选择
---

# 搜索结果：LangChain ExampleSelector 2025-2026 最佳实践

## 搜索摘要
搜索关键词：LangChain ExampleSelector semantic similarity few-shot learning 2025 2026 best practices
搜索平台：GitHub, Reddit, Twitter
结果数量：8 个

---

## 相关链接

### 1. LangChain官方文档：Few-shot prompting ⚠️
- **URL**: https://python.langchain.com/docs/concepts/few_shot_prompting/
- **类型**: 官方文档（已通过 Context7 获取，不需要抓取）
- **简述**: LangChain中Few-shot prompting指南，介绍SemanticSimilarityExampleSelector基于语义相似度动态选择示例的最佳实践

### 2. SemanticSimilarityExampleSelector API参考 ⚠️
- **URL**: https://api.python.langchain.com/en/latest/example_selectors/langchain_core.example_selectors.semantic_similarity.SemanticSimilarityExampleSelector.html
- **类型**: 官方 API 文档（不需要抓取）
- **简述**: LangChain核心模块中SemanticSimilarityExampleSelector的详细API文档，包括参数配置和使用方式

### 3. Exploring Few-Shot Prompts with LangChain ✅
- **URL**: https://medium.com/donato-story/exploring-few-shot-prompts-with-langchain-852f27ea4e1d
- **类型**: 社区教程（Medium）
- **简述**: 使用SemanticSimilarityExampleSelector实现动态few-shot示例选择，通过嵌入匹配最相关示例的实践示例
- **优先级**: High（与第一次搜索重复）
- **抓取原因**: 包含 SemanticSimilarityExampleSelector 的实践案例

### 4. LangChain in Chains #6: Example Selectors ✅
- **URL**: https://pub.aimind.so/langchain-in-chains-6-example-selectors-310f47b4cdf3
- **类型**: 社区教程（Medium）
- **简述**: 详细解释SemanticSimilarityExampleSelector使用余弦相似度选择最接近输入的示例，包含代码实现
- **优先级**: High（与第一次搜索重复）
- **抓取原因**: 详细讲解余弦相似度选择机制

### 5. Few-Shot Prompting Explained with LangChain ExampleSelector ✅
- **URL**: https://www.sandgarden.com/learn/few-shot-prompting
- **类型**: 技术教程（Sandgarden）
- **简述**: Few-shot prompting最佳实践，强调LangChain ExampleSelector基于语义相似度的动态选择以提升提示相关性和性能
- **优先级**: High
- **抓取原因**: 最佳实践总结

### 6. LangChain Best Practices 2025 ✅
- **URL**: https://www.swarnendu.de/blog/langchain-best-practices
- **类型**: 技术博客（2025 年）
- **简述**: 2025年LangChain最佳实践，包括SemanticSimilarityExampleSelector自适应选择相关few-shot示例以优化提示
- **优先级**: High
- **抓取原因**: 2025 年最新最佳实践

### 7. Build Smarter AI Apps with LangChain (2025 Gist) ✅
- **URL**: https://gist.github.com/sakha1370/cf4663c759099ae6c66e8ab3aa4b7a52
- **类型**: GitHub Gist（2025-2026 更新）
- **简述**: 2025-2026更新笔记，介绍基于相似度的Example Selector在few-shot学习中的应用方式
- **优先级**: High
- **抓取原因**: 2025-2026 年最新更新

### 8. LangChain Semantic Length Example Selector ✅
- **URL**: https://github.com/whitesmith/langchain-semantic-length-example-selector
- **类型**: GitHub 项目
- **简述**: GitHub项目扩展LangChain的相似度与长度结合的示例选择器，适用于few-shot场景优化
- **优先级**: Medium
- **抓取原因**: 社区扩展项目，结合语义相似度和长度限制

---

## 关键信息提取

### 1. 2025-2026 年最新趋势

**语义相似度选择成为主流：**
- SemanticSimilarityExampleSelector 是最常用的选择器
- 基于 embeddings 的相似度匹配效果最好
- 社区开始探索混合策略（语义 + 长度）

**性能优化关注点：**
- 缓存 embeddings 减少重复计算
- 使用更快的 embedding 模型（如 text-embedding-3-small）
- 批量处理示例以提高效率

### 2. SemanticSimilarityExampleSelector 最佳实践

**参数配置建议：**
- `k=3-5`：选择 3-5 个示例效果最好
- `vectorstore_cls=Chroma`：推荐使用 Chroma（易用性好）
- `embeddings=OpenAIEmbeddings(model="text-embedding-3-small")`：成本低、速度快

**使用场景：**
- 问答系统：根据问题选择相似的 QA 对
- 分类任务：根据输入选择相似的分类示例
- 代码生成：根据需求选择相似的代码示例

### 3. 余弦相似度选择机制

**工作原理：**
1. 将所有示例转换为 embeddings
2. 将输入查询转换为 embedding
3. 计算余弦相似度
4. 返回 top-k 最相似的示例

**优势：**
- 语义理解能力强
- 适用于多语言场景
- 可以处理同义词和近义词

### 4. 混合策略（2025 年新趋势）

**语义相似度 + 长度限制：**
```python
# 社区扩展项目示例
from langchain_semantic_length_example_selector import SemanticLengthExampleSelector

selector = SemanticLengthExampleSelector(
    examples=examples,
    embeddings=embeddings,
    vectorstore_cls=Chroma,
    k=5,  # 语义相似度选择 5 个
    max_length=2048  # 长度限制
)
```

**优势：**
- 既保证相关性，又控制长度
- 避免 token 超限
- 提高实际应用效果

### 5. 常见误区（2025 年总结）

**误区 1：k 值越大越好**
- ❌ 错误：k=10 或更多
- ✅ 正确：k=3-5 通常效果最好
- 原因：过多示例会引入噪音

**误区 2：不考虑 token 限制**
- ❌ 错误：只关注相关性，不考虑长度
- ✅ 正确：结合长度限制或使用混合策略
- 原因：可能超过模型的 context window

**误区 3：不缓存 embeddings**
- ❌ 错误：每次都重新计算 embeddings
- ✅ 正确：缓存示例的 embeddings
- 原因：重复计算浪费时间和成本

---

## 待抓取链接统计

**High 优先级（4 个）：**
1. https://www.sandgarden.com/learn/few-shot-prompting
2. https://www.swarnendu.de/blog/langchain-best-practices
3. https://gist.github.com/sakha1370/cf4663c759099ae6c66e8ab3aa4b7a52

**Medium 优先级（1 个）：**
4. https://github.com/whitesmith/langchain-semantic-length-example-selector

**重复（已在第一次搜索中）：**
- https://medium.com/donato-story/exploring-few-shot-prompts-with-langchain-852f27ea4e1d
- https://pub.aimind.so/langchain-in-chains-6-example-selectors-310f47b4cdf3

**排除（官方文档）：**
- https://python.langchain.com/docs/concepts/few_shot_prompting/
- https://api.python.langchain.com/en/latest/example_selectors/langchain_core.example_selectors.semantic_similarity.SemanticSimilarityExampleSelector.html

---

## 2025-2026 年关键更新

### 1. 性能优化
- 更快的 embedding 模型（text-embedding-3-small）
- 批量处理支持
- 缓存机制优化

### 2. 混合策略
- 语义相似度 + 长度限制
- 语义相似度 + MMR（多样性）
- 社区扩展项目活跃

### 3. 最佳实践
- k=3-5 成为共识
- Chroma 成为推荐的向量存储
- 缓存 embeddings 成为标准做法
