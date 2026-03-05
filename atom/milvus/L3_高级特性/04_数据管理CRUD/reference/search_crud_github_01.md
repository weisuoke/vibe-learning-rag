---
type: search_result
search_query: Milvus CRUD operations Insert Delete Upsert Query Search 2025 2026
search_engine: grok-mcp
platform: GitHub
searched_at: 2026-02-25
knowledge_point: 04_数据管理CRUD
---

# 搜索结果：Milvus CRUD 操作（GitHub）

## 搜索摘要

在 GitHub 上搜索 Milvus CRUD 操作相关内容，获取了8个相关资源，包括官方 CheatSheet、第三方增强库、已知 Bug、社区讨论等，涵盖了 Insert、Upsert、Delete、Query、Search 等操作的实现细节和常见问题。

## 相关链接

### 1. Milvus CheatSheet CRUD操作指南
- **URL**: https://github.com/milvus-io/bootcamp/blob/master/bootcamp/MilvusCheatSheet.md
- **描述**: Milvus操作速查表，详细示例Insert、Upsert、Query、Search等CRUD方法与一致性设置
- **优先级**: High
- **内容类型**: 官方文档
- **需要抓取**: 是（虽然是官方文档，但是 CheatSheet 包含实践案例）

### 2. MilvusPlus Milvus CRUD增强库
- **URL**: https://github.com/dromara/MilvusPlus
- **描述**: 类似MyBatis-Plus的Milvus工具库，通过Mapper实现插入、删除、更新、查询等强大CRUD操作
- **优先级**: High
- **内容类型**: 第三方库
- **需要抓取**: 是

### 3. Upsert未完全移除旧数据Bug
- **URL**: https://github.com/milvus-io/milvus/issues/38947
- **描述**: 2025年1月Issue，Upsert操作导致数据重复，影响后续Query和Search结果的讨论
- **优先级**: High
- **内容类型**: Bug报告
- **需要抓取**: 是

### 4. 多次Upsert Delete后Search错误
- **URL**: https://github.com/milvus-io/milvus/issues/43315
- **描述**: 2025年7月Bug，多次CRUD操作后向量搜索返回错误结果的分析与复现
- **优先级**: High
- **内容类型**: Bug报告
- **需要抓取**: 是

### 5. Insert Upsert Delete操作问题
- **URL**: https://github.com/milvus-io/milvus/discussions/28374
- **描述**: Milvus v2.3+中插入、更新、删除操作的负载、可见性和一致性问题详细讨论
- **优先级**: High
- **内容类型**: 社区讨论
- **需要抓取**: 是

### 6. Upsert仅修改元数据讨论
- **URL**: https://github.com/milvus-io/milvus/discussions/37282
- **描述**: 探讨动态字段下Upsert是否支持仅更新元数据而不变更向量，支持v2.5部分更新
- **优先级**: Medium
- **内容类型**: 社区讨论
- **需要抓取**: 是

### 7. pymilvus Upsert Query不一致
- **URL**: https://github.com/milvus-io/milvus-lite/issues/249
- **描述**: 2025年Issue，Upsert后Query统计实体数未考虑覆盖，导致数据不一致的问题
- **优先级**: Medium
- **内容类型**: Bug报告
- **需要抓取**: 是

### 8. Milvus Insert与Search机制
- **URL**: https://github.com/milvus-io/milvus/discussions/30157
- **描述**: 解释插入、Upsert、Delete请求队列处理及Growing索引与Search实时性机制
- **优先级**: High
- **内容类型**: 技术讨论
- **需要抓取**: 是

## 关键信息提取

### 1. Milvus CheatSheet（官方速查表）

**核心内容**：
- Insert、Upsert、Query、Search 等 CRUD 方法的详细示例
- 一致性级别设置
- 常用操作的快速参考

**价值**：
- 官方推荐的最佳实践
- 完整的代码示例
- 快速上手指南

### 2. MilvusPlus 增强库

**核心特性**：
- 类似 MyBatis-Plus 的设计理念
- 通过 Mapper 实现 CRUD 操作
- 简化 Milvus 客户端使用

**功能**：
- 插入（Insert）
- 删除（Delete）
- 更新（Update/Upsert）
- 查询（Query）
- 自动映射

**优势**：
- 降低学习成本
- 提供更高层次的抽象
- 支持链式调用

### 3. Upsert 未完全移除旧数据 Bug（Issue #38947）

**问题描述**：
- Upsert 操作后，旧数据未被完全移除
- 导致数据重复
- 影响后续 Query 和 Search 结果

**影响版本**：
- Milvus 2.5.x（2025年1月报告）

**影响范围**：
- Upsert 操作
- Query 结果准确性
- Search 结果准确性

**状态**：
- 2025年1月报告
- 需要关注官方修复进度

### 4. 多次 Upsert Delete 后 Search 错误（Issue #43315）

**问题描述**：
- 多次执行 Upsert 和 Delete 操作后
- 向量搜索返回错误结果
- 可能与数据压缩（Compaction）有关

**影响版本**：
- Milvus 2.5.x（2025年7月报告）

**复现条件**：
- 多次 Upsert 操作
- 多次 Delete 操作
- 执行 Search 查询

**潜在原因**：
- Compaction 机制问题
- 索引更新不及时
- 数据一致性问题

### 5. Insert Upsert Delete 操作问题（Discussion #28374）

**讨论主题**：
- Milvus v2.3+ 中的 CRUD 操作问题
- 负载管理
- 数据可见性
- 一致性保证

**关键问题**：
1. **负载问题**：
   - 高并发插入时的性能瓶颈
   - 批量操作的最佳实践

2. **可见性问题**：
   - 插入后数据何时可见
   - Growing Segment 和 Sealed Segment 的区别
   - Flush 操作的时机

3. **一致性问题**：
   - Strong、Bounded、Eventually 三种一致性级别
   - 一致性级别对性能的影响
   - 如何选择合适的一致性级别

### 6. Upsert 仅修改元数据讨论（Discussion #37282）

**讨论主题**：
- 动态字段下的 Upsert 操作
- 是否支持仅更新元数据而不变更向量
- Milvus v2.5 的部分更新功能

**核心问题**：
- 向量字段是否可以不更新
- 仅更新标量字段的性能优化
- 部分更新的实现方式

**结论**：
- Milvus v2.5 支持部分更新
- 可以仅更新标量字段
- 向量字段可以保持不变

**应用场景**：
- 元数据更新（如标签、分类）
- 向量不变，仅更新属性
- 减少 Embedding 计算开销

### 7. pymilvus Upsert Query 不一致（Issue #249）

**问题描述**：
- Upsert 后 Query 统计实体数不正确
- 未考虑 Upsert 的覆盖操作
- 导致数据不一致

**影响版本**：
- Milvus Lite（2025年报告）

**问题原因**：
- Upsert 操作会先删除旧数据，再插入新数据
- Query 统计时可能计算了删除的数据
- 需要等待 Compaction 完成

**解决方案**：
- 使用 Flush 操作
- 等待 Compaction 完成
- 使用正确的一致性级别

### 8. Milvus Insert 与 Search 机制（Discussion #30157）

**核心内容**：
- Insert、Upsert、Delete 请求队列处理机制
- Growing 索引与 Search 实时性
- 数据写入流程详解

**Insert 流程**：
1. 数据写入 WAL（Write-Ahead Log）
2. 数据进入 Growing Segment
3. Growing Segment 达到阈值后 Seal
4. Sealed Segment 构建索引
5. 索引加载到内存

**Search 实时性**：
- Growing Segment：实时可搜索（brute-force）
- Sealed Segment：索引搜索（高性能）
- 一致性级别影响可见性

**队列处理**：
- 请求队列（Request Queue）
- 批量处理（Batch Processing）
- 异步执行（Async Execution）

## 总结

GitHub 社区资源揭示了 Milvus CRUD 操作的以下关键点：

1. **官方 CheatSheet 是最佳参考**：包含完整的 CRUD 操作示例和最佳实践
2. **MilvusPlus 提供更高层次抽象**：简化 CRUD 操作，降低学习成本
3. **Upsert 操作存在已知 Bug**：需要关注官方修复进度，谨慎使用
4. **数据一致性是核心挑战**：需要理解 Growing/Sealed Segment 机制
5. **部分更新功能（v2.5+）**：支持仅更新标量字段，优化性能
6. **Compaction 机制很重要**：影响删除操作的空间回收和数据一致性
7. **批量操作是最佳实践**：性能远优于单条操作
8. **一致性级别需要权衡**：Strong 一致性保证准确性，Eventually 一致性提升性能

## 需要深入抓取的链接

根据优先级和内容类型，以下链接需要深入抓取：

**High 优先级**（6个）：
1. https://github.com/milvus-io/bootcamp/blob/master/bootcamp/MilvusCheatSheet.md
2. https://github.com/dromara/MilvusPlus
3. https://github.com/milvus-io/milvus/issues/38947
4. https://github.com/milvus-io/milvus/issues/43315
5. https://github.com/milvus-io/milvus/discussions/28374
6. https://github.com/milvus-io/milvus/discussions/30157

**Medium 优先级**（2个）：
7. https://github.com/milvus-io/milvus/discussions/37282
8. https://github.com/milvus-io/milvus-lite/issues/249

**排除的链接**：
- 无（所有链接都包含有价值的信息）
