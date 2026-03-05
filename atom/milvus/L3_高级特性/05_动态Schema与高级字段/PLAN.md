# 05_动态Schema与高级字段 - 生成计划

## 数据来源记录

### 源码分析
- ✅ reference/source_动态Schema_01.md - 动态Schema与字段管理源码分析
  - 分析文件：dynamic_field_test.go, add_field_test.go, ddl_callbacks_alter_collection_add_field.go, plan.proto
  - 关键发现：EnableDynamicField机制、AddCollectionField实现、向量类型定义、GIS函数支持

### Context7 官方文档
- ✅ reference/context7_pymilvus_01.md - PyMilvus API 文档
  - 库：pymilvus (master, 2026-01-22)
  - 内容：FieldSchema、CollectionSchema、MilvusClient、默认值支持
- ✅ reference/context7_milvus_02.md - Milvus 官方文档
  - 库：Milvus v2.6.x (2025-11-17)
  - 内容：AddCollectionField API、nullable字段、默认值、向量类型、空间数据类型

### 网络搜索
- ✅ reference/search_动态Schema_01.md - 动态Schema与AddCollectionField生产经验
  - 平台：GitHub Issues, Releases, Changelog
  - 内容：生产环境问题、性能影响、最佳实践
- ✅ reference/search_Int8向量_02.md - Int8向量量化与性能优化
  - 平台：Reddit, GitHub
  - 内容：RaBitQ+SQ8量化、VectorDBBench、内存优化策略
- ✅ reference/search_RAG多租户_03.md - RAG元数据扩展与多租户Schema演化
  - 平台：GitHub, Reddit, Twitter
  - 内容：多租户架构、元数据管理、Schema演化模式

### 待抓取链接
无需抓取（已通过源码分析和Context7获取足够资料）

---

## 文件清单

### 基础维度文件
- [x] 00_概览.md
- [x] 01_30字核心.md
- [x] 02_第一性原理.md

### 核心概念文件（基于源码 + Context7 + 网络调研）
- [x] 03_核心概念_1_动态Schema基础.md - EnableDynamicField、$meta字段、使用场景 [来源: 源码+Context7]
- [x] 03_核心概念_2_动态字段数据操作.md - 插入、查询、过滤动态字段 [来源: 源码+Context7]
- [x] 03_核心概念_3_动态Schema最佳实践.md - 性能影响、使用建议 [来源: 网络+Context7]
- [x] 03_核心概念_4_Int8向量类型.md - 内存优化、量化方法、精度权衡 [来源: 源码+网络]
- [x] 03_核心概念_5_Float16_BFloat16向量.md - 半精度向量、应用场景 [来源: 源码+Context7]
- [x] 03_核心概念_6_空间数据类型.md - GIS函数、WKT格式、地理位置检索 [来源: 源码]
- [x] 03_核心概念_7_字段添加机制.md - AddCollectionField、nullable、默认值 [来源: 源码+Context7+网络]

### 基础维度文件（续）
- [x] 04_最小可用.md
- [x] 05_双重类比.md
- [x] 06_反直觉点.md

### 实战代码文件（基于源码 + Context7 + 网络调研）
- [x] 07_实战代码_场景1_基础动态Schema操作.md - 启用、插入、查询 [来源: Context7]
- [x] 07_实战代码_场景2_RAG元数据扩展.md - 动态添加文档元数据 [来源: 网络]
- [x] 07_实战代码_场景3_渐进式Schema演化.md - AddCollectionField、向后兼容 [来源: Context7+网络]
- [x] 07_实战代码_场景4_Int8向量实战.md - 量化、性能对比 [来源: 网络]
- [x] 07_实战代码_场景5_多租户元数据管理.md - 元数据过滤、权限控制 [来源: 网络]
- [x] 07_实战代码_场景6_字段添加实战.md - AddCollectionField完整流程 [来源: Context7+网络]
- [x] 07_实战代码_场景7_生产级Schema管理.md - 并发控制、错误处理 [来源: 网络]

### 基础维度文件（续）
- [x] 08_面试必问.md
- [x] 09_化骨绵掌.md
- [x] 10_一句话总结.md

---

## 核心概念详细拆解

### 核心概念1：动态Schema基础
**来源**：源码分析 + Context7文档

**内容要点**：
- EnableDynamicField 启用机制
- $meta 字段的隐式存储
- 动态字段的数据类型支持
- 与固定Schema的对比

**关键代码**：
```python
schema = client.create_schema(
    auto_id=False,
    enable_dynamic_field=True  # 启用动态字段
)
```

---

### 核心概念2：动态字段数据操作
**来源**：源码分析 + Context7文档

**内容要点**：
- 插入带有动态字段的数据
- 查询动态字段
- 过滤动态字段
- 动态字段的索引限制

**关键代码**：
```python
# 插入动态字段
data = [{
    "id": 1,
    "vector": [0.1] * 768,
    "title": "Document 1",  # 固定字段
    "author": "Alice",      # 动态字段
    "tags": ["AI", "ML"]    # 动态字段
}]

# 查询动态字段
results = client.search(
    collection_name="documents",
    data=[query_vector],
    filter='author == "Alice"',  # 过滤动态字段
    output_fields=["title", "author", "tags"]
)
```

---

### 核心概念3：动态Schema最佳实践
**来源**：网络搜索 + Context7文档

**内容要点**：
- 性能影响分析（group_by慢、聚合慢）
- 适用场景（元数据不确定、多租户、快速迭代）
- 不适用场景（高频聚合、固定Schema）
- 优化建议（核心字段固定、业务字段动态）

**生产经验**：
- 动态字段的group_by可能有bug（Issue #47438）
- 需要在应用层处理缺失字段
- 性能可能不如固定Schema

---

### 核心概念4：Int8向量类型
**来源**：源码分析 + 网络搜索

**内容要点**：
- Int8向量的定义和存储格式
- 量化方法（SQ8标量量化）
- 内存节省（75%）
- 精度损失（5-10%召回率下降）
- 维度限制（必须是8的倍数）

**生产案例**：
- Reddit用户使用RaBitQ+SQ8节省75%成本
- VectorDBBench性能基准测试
- 大规模向量存储的标准方案

**关键代码**：
```python
schema.add_field(
    field_name="vector",
    datatype=DataType.INT8_VECTOR,
    dim=128  # 必须是8的倍数
)
```

---

### 核心概念5：Float16/BFloat16向量
**来源**：源码分析 + Context7文档

**内容要点**：
- Float16：半精度浮点向量，内存占用为Float32的1/2
- BFloat16：Brain Float16向量，保留Float32的指数范围
- 应用场景：平衡精度和内存
- 精度说明：Float16适合带宽受限场景，BFloat16适合深度学习

**向量类型对比**：
| 类型 | 内存占用 | 精度 | 适用场景 |
|------|---------|------|---------|
| Float32 | 100% | 高 | 高精度需求 |
| Float16 | 50% | 中 | 带宽受限 |
| BFloat16 | 50% | 中 | 深度学习 |
| Int8 | 25% | 低 | 大规模存储 |

---

### 核心概念6：空间数据类型
**来源**：源码分析

**内容要点**：
- GIS函数支持（9种空间关系操作）
- WKT格式（Well-Known Text）
- 地理位置检索
- 距离计算（DWithin操作）

**GIS操作**：
- Equals、Touches、Overlaps、Crosses
- Contains、Intersects、Within
- DWithin（距离内）、STIsValid（验证有效性）

**当前状态**：
- Geolocation数据类型正在开发中（under development）
- 源码中已有GISFunctionFilterExpr实现

---

### 核心概念7：字段添加机制
**来源**：源码分析 + Context7文档 + 网络搜索

**内容要点**：
- AddCollectionField API
- nullable字段（必需）
- 默认值设置
- Schema版本管理
- 并发冲突处理

**生产问题**：
- HybridSearch失败（Issue #43003）
- 查询新增字段时断言错误（Issue #45318）
- 并发insert冲突（Issue #41858）
- Parquet导入失败（Issue #41755）

**最佳实践**：
- 添加字段前暂停写入
- 添加字段后重新加载Collection
- 测试所有查询类型
- 更新数据导入流程

---

## 实战场景详细拆解

### 场景1：基础动态Schema操作
**来源**：Context7文档

**内容**：
- 创建启用动态Schema的Collection
- 插入带有动态字段的数据
- 查询和过滤动态字段
- 验证动态字段的存储

**代码长度**：300-400行

---

### 场景2：RAG元数据扩展
**来源**：网络搜索

**内容**：
- 文档元数据的动态添加
- 元数据过滤查询
- 元数据更新策略
- 多版本元数据共存

**应用场景**：
- 文档标题、作者、日期
- 业务分类、标签
- 权限元数据

**代码长度**：350-450行

---

### 场景3：渐进式Schema演化
**来源**：Context7文档 + 网络搜索

**内容**：
- 使用AddCollectionField添加新字段
- 设置nullable和默认值
- 处理已存在数据
- 向后兼容性保证

**生产经验**：
- 停止写入再添加字段
- 重新加载Collection
- 测试所有查询类型

**代码长度**：400-500行

---

### 场景4：Int8向量实战
**来源**：网络搜索

**内容**：
- Float32向量量化为Int8
- 创建Int8向量Collection
- 性能对比测试（内存、QPS、召回率）
- RaBitQ+SQ8组合量化

**生产案例**：
- 75%内存节省
- 4x性能提升
- 5-10%召回率下降

**代码长度**：400-500行

---

### 场景5：多租户元数据管理
**来源**：网络搜索

**内容**：
- Collection级别隔离
- Partition级别隔离
- 元数据过滤隔离
- 权限控制实现

**架构对比**：
- 小规模（<100租户）：Collection隔离
- 中规模（100-10K租户）：Partition隔离
- 大规模（>10K租户）：元数据过滤

**代码长度**：450-500行

---

### 场景6：字段添加实战
**来源**：Context7文档 + 网络搜索

**内容**：
- AddCollectionField完整流程
- 并发控制策略
- 错误处理和回滚
- 生产环境部署

**生产问题处理**：
- HybridSearch失败的workaround
- 并发insert冲突的解决
- Parquet导入失败的修复

**代码长度**：400-500行

---

### 场景7：生产级Schema管理
**来源**：网络搜索

**内容**：
- Schema版本管理
- 多版本共存策略
- 数据迁移方案
- 监控和告警

**最佳实践**：
- 使用维护窗口
- 准备回滚方案
- 定期备份数据
- 监控关键指标

**代码长度**：450-500行

---

## 生成进度

### 阶段一：Plan 生成
- [x] 1.1 Brainstorm 分析
- [x] 1.2 多源数据收集
  - [x] 源码分析（1个文件）
  - [x] Context7文档（2个文件）
  - [x] 网络搜索（3个文件）
- [x] 1.3 用户确认拆解方案
- [x] 1.4 Plan 最终确定

### 阶段二：补充调研
- [x] 无需补充调研（资料已足够）

### 阶段三：文档生成
- [x] 基础维度文件（第一部分）
  - [x] 00_概览.md
  - [x] 01_30字核心.md
  - [x] 02_第一性原理.md
- [x] 核心概念文件（7个）
  - [x] 03_核心概念_1_动态Schema基础.md
  - [x] 03_核心概念_2_动态字段数据操作.md
  - [x] 03_核心概念_3_动态Schema最佳实践.md
  - [x] 03_核心概念_4_Int8向量类型.md
  - [x] 03_核心概念_5_Float16_BFloat16向量.md
  - [x] 03_核心概念_6_空间数据类型.md
  - [x] 03_核心概念_7_字段添加机制.md
- [x] 基础维度文件（第二部分）
  - [x] 04_最小可用.md
  - [x] 05_双重类比.md
  - [x] 06_反直觉点.md
- [x] 实战代码文件（7个）
  - [x] 07_实战代码_场景1_基础动态Schema操作.md
  - [x] 07_实战代码_场景2_RAG元数据扩展.md
  - [x] 07_实战代码_场景3_渐进式Schema演化.md
  - [x] 07_实战代码_场景4_Int8向量实战.md
  - [x] 07_实战代码_场景5_多租户元数据管理.md
  - [x] 07_实战代码_场景6_字段添加实战.md
  - [x] 07_实战代码_场景7_生产级Schema管理.md
- [x] 基础维度文件（第三部分）
  - [x] 08_面试必问.md
  - [x] 09_化骨绵掌.md
  - [x] 10_一句话总结.md

---

## 关键发现总结

### 1. 动态Schema的核心价值
- **灵活性**：无需修改Schema即可添加字段
- **快速迭代**：适合需求不确定的场景
- **多租户支持**：不同租户不同字段需求

### 2. 动态Schema的性能代价
- **group_by问题**：对缺失字段处理错误（Issue #47438）
- **聚合性能**：nullable字段聚合慢4倍（Issue #47566）
- **查询性能**：动态字段无法创建索引

### 3. AddCollectionField的生产问题
- **HybridSearch失败**：添加字段后HybridSearch可能失败
- **并发冲突**：与并发insert操作冲突
- **Parquet导入**：需要更新导入流程

### 4. Int8向量的实用价值
- **成本节省**：75%内存节省（生产案例）
- **性能提升**：4x查询性能提升
- **精度损失**：5-10%召回率下降（可接受）

### 5. 多租户架构选择
- **小规模**：Collection级别隔离（<100租户）
- **中规模**：Partition级别隔离（100-10K租户）
- **大规模**：元数据过滤隔离（>10K租户）

---

## 质量保证

### 数据来源质量
- ✅ 源码分析：直接来自Milvus官方仓库
- ✅ Context7文档：官方文档，最新版本
- ✅ 网络搜索：GitHub Issues、Reddit、Twitter，2025-2026年资料

### 内容覆盖度
- ✅ 动态Schema：完全覆盖（源码+Context7+网络）
- ✅ AddCollectionField：完全覆盖（源码+Context7+网络）
- ✅ Int8向量：完全覆盖（源码+网络）
- ✅ 空间数据类型：部分覆盖（源码，开发中）
- ✅ 多租户架构：完全覆盖（网络）

### 实战场景质量
- ✅ 所有场景都有生产案例支持
- ✅ 所有代码都基于官方API
- ✅ 所有最佳实践都有社区验证

---

## 下一步操作

### 阶段二：补充调研
**结论**：无需补充调研，资料已足够

**原因**：
- 源码分析已覆盖核心实现
- Context7文档已覆盖官方API
- 网络搜索已覆盖生产经验
- 空间数据类型虽在开发中，但源码已有实现

### 阶段三：文档生成
**准备工作**：
- ✅ 所有资料已保存到reference/目录
- ✅ 核心概念和实战场景已明确
- ✅ 文件清单已确定

**生成策略**：
- 使用subagent批量生成
- 每个文件300-500行
- 严格遵循引用规范
- 逐个文件生成并验证

**开始时间**：等待用户确认后开始

---

## 预计文件统计

- **总文件数**：24个
- **基础维度文件**：10个
- **核心概念文件**：7个
- **实战代码文件**：7个
- **预计总行数**：8,000-10,000行
- **预计生成时间**：根据subagent性能而定

---

**Plan 生成完成时间**：2026-02-25
**文档生成完成时间**：2026-03-01
**Plan 版本**：v1.1
**状态**：✅ 全部完成（24/24 文件）
