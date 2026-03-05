---
type: search_result
search_query: WAND algorithm sparse vector inverted index 2025 2026
search_engine: grok-mcp
searched_at: 2026-02-25
knowledge_point: 03_稀疏向量与BM25深入
---

# 搜索结果：WAND 算法与稀疏向量倒排索引

## 搜索摘要

搜索关键词：WAND algorithm sparse vector inverted index 2025 2026
搜索平台：GitHub, Reddit
搜索结果数：6

## 相关链接

1. [questions about sparse index · milvus-io/milvus](https://github.com/milvus-io/milvus/discussions/34806)
   - SPARSE_WAND索引与SPARSE_INVERTED_INDEX共享倒排结构，但额外维护元数据，支持稀疏向量高效WAND查询处理。

2. [IR-unipi: Information Retrieval Course](https://github.com/rossanoventurini/IR-unipi)
   - 2025年课程涵盖倒排索引、WAND、Block Max-WAND查询处理及稀疏向量近似kNN检索。

3. [v0.3.0 · timescale/pg_textsearch](https://github.com/timescale/pg_textsearch/discussions/127)
   - 引入Block-Max WAND优化，实现稀疏关键词倒排索引top-k查询4倍加速，支持BM25稀疏向量排名。

4. [docs/about/changelog.md at latest · timescale/docs](https://github.com/timescale/docs/blob/latest/about/changelog.md)
   - 发布Block MAX-WAND算法，大幅提升关键词搜索性能，适用于稀疏向量倒排索引的排名查询。

5. [Releases · milvus-io/milvus-sdk-cpp](https://github.com/milvus-io/milvus-sdk-cpp/releases)
   - 2026年1月更新支持SPARSE_WAND索引及稀疏向量，集成WAND算法用于高效倒排索引检索。

6. [[Bug] Standalone doesn't start up · milvus-io/milvus](https://github.com/milvus-io/milvus/issues/45629)
   - 2025年11月讨论中列出SPARSE_WAND作为核心稀疏向量倒排索引类型，支持WAND查询优化。

## 关键信息提取

### WAND 算法原理

**WAND（Weak-AND）算法**：
- 用于高效的 top-k 查询处理
- 基于倒排索引的查询优化算法
- 通过上界估计跳过不可能进入 top-k 的文档

**Block Max-WAND**：
- WAND 算法的优化版本
- 维护每个块的最大分数
- 实现 4 倍加速（根据 Timescale 的测试）

### Milvus SPARSE_WAND 索引

**与 SPARSE_INVERTED_INDEX 的关系**：
- 共享倒排索引结构
- 额外维护元数据（用于 WAND 查询优化）
- 支持高效的 top-k 查询

**适用场景**：
- 性能优先的场景
- 大规模稀疏向量检索
- 需要快速 top-k 查询

### 倒排索引结构

**基础结构**：
- 词项 → 文档列表
- 每个文档包含词频信息
- 支持快速查找包含特定词项的文档

**WAND 优化**：
- 维护上界信息
- 支持跳跃式遍历
- 减少不必要的文档评分

### 性能提升

**Timescale pg_textsearch**：
- Block-Max WAND 实现 4 倍加速
- 适用于 BM25 稀疏向量排名
- 支持关键词搜索优化

**Milvus 实现**：
- 2026 年 1 月更新支持 SPARSE_WAND
- 集成 WAND 算法用于高效检索
- 支持稀疏向量 top-k 查询

---

## 下一步调研方向

1. **WAND 算法详解**：深入理解 WAND 算法的工作原理
2. **Block Max-WAND**：Block Max-WAND 的优化机制
3. **性能对比**：SPARSE_INVERTED_INDEX vs SPARSE_WAND 性能对比
4. **元数据维护**：SPARSE_WAND 额外维护的元数据内容
5. **上界估计**：如何计算和使用上界信息
