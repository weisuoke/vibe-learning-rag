# 03_稀疏向量与BM25深入 - 生成计划

## 数据来源记录

### 源码分析
- ✓ reference/source_sparse_bm25_01.md - Milvus 稀疏向量与 BM25 源码分析
  - client/index/sparse.go - 稀疏向量索引定义
  - client/entity/sparse.go - 稀疏向量数据结构
  - internal/util/function/bm25_function.go - BM25 函数实现
  - pkg/util/bm25/bm25.go - BM25 工具函数

### Context7 官方文档
- ✓ reference/context7_pymilvus_01.md - PyMilvus 稀疏向量与 BM25 官方文档
  - BM25 全文搜索
  - 稀疏向量索引
  - 混合检索与权重调整

### 网络搜索
- ✓ reference/search_hybrid_weight_01.md - 混合检索权重调优
- ✓ reference/search_wand_algorithm_01.md - WAND 算法与稀疏向量倒排索引
- ✓ reference/search_bm25_practices_01.md - Milvus BM25 稀疏向量实现最佳实践

### 待抓取链接（将由第三方工具自动保存到 reference/）
暂无需要抓取的链接（已通过源码和官方文档获取足够信息）

---

## 知识点拆解方案

### 定位：平衡型（算法原理 + 实战应用）

基于源码分析、Context7 官方文档和网络搜索结果，本知识点将采用**平衡型**定位：
- 深入讲解 BM25 算法的数学原理和稀疏向量的底层实现
- 提供丰富的实战代码和应用场景
- 覆盖从理论到实践的完整路径

---

## 文件清单

### 基础维度文件
- [ ] 00_概览.md
- [ ] 01_30字核心.md
- [ ] 02_第一性原理.md

### 核心概念文件（基于源码 + Context7 + 网络调研）

#### 1. BM25 算法原理
- [ ] 03_核心概念_1_BM25算法数学原理.md
  - TF-IDF 基础
  - BM25 公式推导
  - 参数 k1 和 b 的含义
  - 文档长度归一化
  - [来源: 源码 + Context7]

#### 2. 稀疏向量数据结构
- [ ] 03_核心概念_2_稀疏向量数据结构.md
  - 稀疏向量的定义
  - positions + values 存储格式
  - 序列化与反序列化
  - 维度计算规则
  - [来源: 源码 client/entity/sparse.go]

#### 3. 稀疏向量索引类型
- [ ] 03_核心概念_3_稀疏向量索引类型.md
  - SPARSE_INVERTED_INDEX（倒排索引）
  - SPARSE_WAND（WAND 算法）
  - 索引参数：drop_ratio_build 和 drop_ratio_search
  - 性能对比
  - [来源: 源码 + 网络搜索]

#### 4. WAND 算法深入
- [ ] 03_核心概念_4_WAND算法深入.md
  - WAND（Weak-AND）算法原理
  - Block Max-WAND 优化
  - 上界估计机制
  - 跳跃式遍历
  - [来源: 网络搜索 search_wand_algorithm_01.md]

#### 5. BM25 Function 实现
- [ ] 03_核心概念_5_BM25_Function实现.md
  - Analyzer 分词器
  - 哈希函数（token → uint32）
  - 词频统计
  - 并发处理机制
  - [来源: 源码 internal/util/function/bm25_function.go]

#### 6. 混合检索策略
- [ ] 03_核心概念_6_混合检索策略.md
  - RRFRanker（Reciprocal Rank Fusion）
  - WeightedRanker（加权融合）
  - 权重调优方法
  - 适用场景对比
  - [来源: Context7 + 网络搜索]

#### 7. Milvus 2.6 新特性
- [ ] 03_核心概念_7_Milvus_2.6_BM25新特性.md
  - slop 参数（短语匹配）
  - 动态统计更新
  - 服务器端计算
  - 性能提升
  - [来源: 网络搜索 search_bm25_practices_01.md]

### 基础维度文件（续）
- [ ] 04_最小可用.md
- [ ] 05_双重类比.md
- [ ] 06_反直觉点.md

### 实战代码文件（基于源码 + Context7 + 网络调研）

#### 场景 1：BM25 全文搜索基础
- [ ] 07_实战代码_场景1_BM25全文搜索基础.md
  - 创建 BM25 Function
  - 配置 SPARSE_INVERTED_INDEX
  - 插入文本数据（自动生成稀疏向量）
  - 执行全文搜索
  - [来源: Context7 context7_pymilvus_01.md]

#### 场景 2：稀疏向量手动操作
- [ ] 07_实战代码_场景2_稀疏向量手动操作.md
  - 手动创建稀疏向量
  - 稀疏向量序列化
  - 稀疏向量检索
  - drop_ratio 参数调优
  - [来源: Context7 + 源码]

#### 场景 3：混合检索实战
- [ ] 07_实战代码_场景3_混合检索实战.md
  - Dense 向量 + Sparse 向量
  - RRFRanker 使用
  - WeightedRanker 权重调整
  - 性能对比测试
  - [来源: Context7 + 网络搜索]

#### 场景 4：BM25 参数调优
- [ ] 07_实战代码_场景4_BM25参数调优.md
  - bm25_k1 参数实验
  - bm25_b 参数实验
  - drop_ratio 调优
  - 评估指标（Recall、Precision）
  - [来源: Context7 + 网络搜索]

#### 场景 5：生产级 RAG 系统
- [ ] 07_实战代码_场景5_生产级RAG系统.md
  - 向量检索 + BM25 混合
  - 动态文档更新
  - 权重自适应调整
  - 性能监控
  - [来源: 网络搜索 search_bm25_practices_01.md]

### 基础维度文件（续）
- [ ] 08_面试必问.md
- [ ] 09_化骨绵掌.md
- [ ] 10_一句话总结.md

---

## 核心技术点总结

### 从源码中提取的关键信息

1. **稀疏向量数据结构**：
   - positions[]（uint32）+ values[]（float32）
   - 序列化：8 字节/元素
   - 自动排序：按 position 升序
   - 维度：max(positions) + 1

2. **BM25 实现机制**：
   - Analyzer 分词
   - HashString2LessUint32 哈希
   - 词频统计：map[uint32]float32
   - 并发处理：8 个 goroutine

3. **稀疏向量索引**：
   - SPARSE_INVERTED_INDEX：倒排索引
   - SPARSE_WAND：WAND 算法
   - drop_ratio：控制索引大小和搜索精度

### 从 Context7 中提取的关键信息

1. **BM25 参数**：
   - bm25_k1：默认 1.2（词频饱和参数）
   - bm25_b：默认 0.75（文档长度归一化参数）

2. **混合检索策略**：
   - RRFRanker：无需手动调整权重
   - WeightedRanker：需要手动设置权重

3. **性能优化**：
   - drop_ratio_build：减少索引大小
   - drop_ratio_search：加速搜索

### 从网络搜索中提取的关键信息

1. **WAND 算法**：
   - Block Max-WAND：4 倍加速
   - 上界估计：跳过不可能进入 top-k 的文档

2. **权重调优经验**：
   - 实验驱动：根据具体用例进行实验
   - WeightedRanker 适合精确控制
   - RRF 适合快速上手

3. **Milvus 2.6 新特性**：
   - slop 参数：支持短语匹配
   - 动态统计更新：实时更新 BM25 统计信息

---

## 生成进度

- [x] 阶段一：Plan 生成
  - [x] 1.1 Brainstorm 分析
  - [x] 1.2 多源数据收集（源码 + Context7 + 网络）
    - [x] A. 源码分析
    - [x] B. Context7 官方文档查询
    - [x] C. Grok-mcp 网络搜索
    - [x] D. 数据整合
  - [x] 1.3 用户确认拆解方案
  - [x] 1.4 Plan 最终确定
- [x] 阶段二：补充调研（已跳过 - 资料充足）
- [ ] 阶段三：文档生成（读取 reference/ 中的所有资料）

---

## 文件长度控制

- **目标长度**：每个文件 300-500 行
- **核心概念文件**：7 个文件，每个文件详细讲解一个技术点
- **实战代码文件**：5 个场景，每个场景一个完整示例
- **基础维度文件**：10 个文件，按照原子化模板生成

---

## 资料充足性确认

✅ **当前资料已充足，可直接进入阶段三**

- 源码分析：覆盖数据结构、实现机制、索引类型
- Context7 文档：覆盖 API 使用、参数配置、混合检索
- 网络搜索：覆盖权重调优、WAND 算法、最佳实践

**拆解方案要点**：
1. **7 个核心概念**：覆盖 BM25 算法、稀疏向量、索引类型、WAND 算法、混合检索等
2. **5 个实战场景**：从基础到生产级，逐步深入
3. **平衡型定位**：既有算法原理，又有实战应用
4. **数据来源明确**：所有内容都有明确的数据来源（源码/Context7/网络）
