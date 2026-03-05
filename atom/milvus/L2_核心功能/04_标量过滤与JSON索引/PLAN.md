# 标量过滤与JSON索引 - 文档生成计划

> 生成时间：2026-02-24
> 状态：数据收集完成，待执行文档生成

---

## 一、知识点概述

### 知识点名称
标量过滤与JSON索引

### 知识点描述
- 实现向量检索与标量条件的混合查询
- 使用 JSON Path Index 进行高效元数据过滤
- **2026核心特性**：JSON Path Index - 100x 嵌套 JSON 查询性能提升

### 前置知识
- 向量索引类型
- 相似度度量
- 混合检索

### 目标位置
`atom/milvus/L2_核心功能/04_标量过滤与JSON索引/`

---

## 二、核心概念拆分（7个）

### 1. 过滤表达式语法
**内容**：
- 比较操作符：==, !=, >, <, >=, <=
- 范围操作符：IN, LIKE
- NULL 操作符：IS NULL, IS NOT NULL
- 表达式优先级

**数据来源**：
- 源码：CompareExpr.cpp, BinaryRangeExpr.cpp, TermExpr.cpp, NullExpr.cpp
- Context7：context7_milvus_boolean_operators.md
- 测试：test_milvus_client_scalar_filtering.py

### 2. 布尔运算符
**内容**：
- AND / OR / NOT 操作符
- 复合条件组合
- 优先级与括号使用
- 位图操作优化

**数据来源**：
- 源码：LogicalBinaryExpr.cpp
- Context7：context7_milvus_boolean_operators.md
- 网络：search_web_results.md

### 3. 数据类型支持
**内容**：
- 标量类型：INT8, INT16, INT32, INT64, BOOL, FLOAT, DOUBLE, VARCHAR
- JSON 类型：嵌套对象、数组
- ARRAY 类型：INT32, INT64, VARCHAR 数组
- 类型转换与兼容性

**数据来源**：
- 测试：test_milvus_client_scalar_filtering.py
- Context7：context7_pymilvus_scalar_filtering.md

### 4. JSON Path 语法
**内容**：
- 路径表达式：`json_field['a']['b']`
- 数组访问：`json_field['a'][0]`
- 嵌套访问：`json_field['a'][0]['b']`
- 根路径访问

**数据来源**：
- 源码：JsonIndexBuilder.cpp
- Context7：context7_milvus_json_index.md
- 测试：test_milvus_client_json_path_index.py

### 5. JSON 索引类型
**内容**：
- JsonFlatIndex：扁平索引，适用于简单查询
- JsonInvertedIndex：倒排索引，适用于复杂查询
- 索引构建流程
- 性能对比

**数据来源**：
- 源码：JsonFlatIndex.cpp, JsonInvertedIndex.cpp
- Context7：context7_milvus_json_index.md
- 网络：search_web_results.md（100x 性能提升）

### 6. 标量索引优化
**内容**：
- 索引类型：INVERTED, BITMAP, STL_SORT, AUTOINDEX, TRIE, NGRAM
- 索引选择策略
- 性能优化技巧
- 30倍性能提升验证

**数据来源**：
- 测试：test_milvus_client_scalar_filtering.py
- 网络：search_web_results.md（30倍性能提升）

### 7. 混合查询执行策略
**内容**：
- Pre-filtering：在向量搜索前过滤
- Post-filtering：在向量搜索后过滤
- Hybrid：混合执行策略
- 选择性与性能权衡

**数据来源**：
- 源码：CompareExpr.cpp
- Context7：context7_pymilvus_scalar_filtering.md
- 网络：search_web_results.md（arXiv 2026 论文）

---

## 三、实战代码场景（6个）

### 场景1：基础标量过滤
**目标**：实现简单比较和范围查询

**技术栈**：
- pymilvus
- 比较操作符
- 范围操作符

**核心代码**：
```python
# 简单比较
filter = "age > 25"

# 范围查询
filter = "price >= 100 AND price <= 500"

# IN 操作符
filter = "color IN ['red', 'blue', 'green']"
```

**数据来源**：
- Context7：context7_pymilvus_scalar_filtering.md
- 测试：test_milvus_client_scalar_filtering.py

### 场景2：复合条件查询
**目标**：使用 AND/OR/NOT 组合复杂条件

**技术栈**：
- 布尔运算符
- 括号优先级
- NULL 操作符

**核心代码**：
```python
filter = '''
    age > 25 AND (city == '北京' OR city == '上海')
    AND tags IS NOT NULL
'''
```

**数据来源**：
- Context7：context7_milvus_boolean_operators.md
- 网络：search_web_results.md

### 场景3：标量索引优化
**目标**：对比不同索引类型的性能

**技术栈**：
- INVERTED, BITMAP, STL_SORT, AUTOINDEX
- 性能测试
- 索引选择策略

**核心代码**：
```python
# 创建不同索引类型
index_params.add_index(
    field_name="age",
    index_type="INVERTED"
)

# 性能对比测试
# 30倍性能提升验证
```

**数据来源**：
- 测试：test_milvus_client_scalar_filtering.py
- 网络：search_web_results.md

### 场景4：JSON字段过滤
**目标**：实现嵌套 JSON 和数组过滤

**技术栈**：
- JSON Path 语法
- JSON_CONTAINS
- ARRAY_CONTAINS

**核心代码**：
```python
# 嵌套 JSON 过滤
filter = "metadata['category'] == 'electronics'"

# 数组过滤
filter = "ARRAY_CONTAINS(tags, 'tech')"
```

**数据来源**：
- Context7：context7_milvus_json_index.md
- 测试：test_milvus_client_json_path_index.py

### 场景5：JSON Path Index 性能对比
**目标**：验证 100x 性能提升

**技术栈**：
- JsonFlatIndex vs JsonInvertedIndex
- 性能测试
- 大规模数据验证

**核心代码**：
```python
# 创建 JSON Path Index
index_params.add_index(
    field_name="metadata",
    index_type="INVERTED",
    params={
        "json_path": "metadata['category']",
        "json_cast_type": "varchar"
    }
)

# 性能对比测试
# 100x 性能提升验证
```

**数据来源**：
- 源码：JsonInvertedIndex.cpp
- Context7：context7_milvus_json_index.md
- 网络：search_web_results.md

### 场景6：RAG 混合检索
**目标**：向量+标量+JSON 综合应用

**技术栈**：
- LangChain
- OpenAI
- Milvus 混合检索

**核心流程**：
1. 文档加载与分块
2. 向量化与元数据提取
3. 混合检索（向量+标量+JSON）
4. 上下文注入
5. LLM 生成

**数据来源**：
- Context7：context7_pymilvus_scalar_filtering.md
- 网络：search_web_results.md（LlamaIndex 教程）

---

## 四、文件清单

### 基础维度文件（10个）
- [x] `00_概览.md`
- [ ] `01_30字核心.md`
- [ ] `02_第一性原理.md`
- [ ] `04_最小可用.md`
- [ ] `05_双重类比.md`
- [ ] `06_反直觉点.md`
- [ ] `08_面试必问.md`
- [ ] `09_化骨绵掌.md`
- [ ] `10_一句话总结.md`

### 核心概念文件（7个）
- [ ] `03_核心概念_1_过滤表达式语法.md`
- [ ] `03_核心概念_2_布尔运算符.md`
- [ ] `03_核心概念_3_数据类型支持.md`
- [ ] `03_核心概念_4_JSON_Path语法.md`
- [ ] `03_核心概念_5_JSON索引类型.md`
- [ ] `03_核心概念_6_标量索引优化.md`
- [ ] `03_核心概念_7_混合查询执行策略.md`

### 实战代码文件（6个）
- [ ] `07_实战代码_场景1_基础标量过滤.md`
- [ ] `07_实战代码_场景2_复合条件查询.md`
- [ ] `07_实战代码_场景3_标量索引优化.md`
- [ ] `07_实战代码_场景4_JSON字段过滤.md`
- [ ] `07_实战代码_场景5_JSON_Path_Index性能对比.md`
- [ ] `07_实战代码_场景6_RAG混合检索.md`

### Reference 文件（7个）
- [x] `reference/source_json_index.md`
- [x] `reference/source_expression.md`
- [x] `reference/source_test_files.md`
- [x] `reference/context7_pymilvus_scalar_filtering.md`
- [x] `reference/context7_milvus_json_index.md`
- [x] `reference/context7_milvus_boolean_operators.md`
- [x] `reference/search_web_results.md`

**总计**：23个主文件 + 7个参考文件 = 30个文件

---

## 五、数据来源记录

### A. 源码分析（已完成）
- [x] JsonFlatIndex.cpp
- [x] JsonInvertedIndex.cpp
- [x] JsonIndexBuilder.cpp
- [x] CompareExpr.cpp
- [x] LogicalBinaryExpr.cpp
- [x] test_milvus_client_json_path_index.py
- [x] test_milvus_client_scalar_filtering.py

**保存位置**：
- `reference/source_json_index.md`
- `reference/source_expression.md`
- `reference/source_test_files.md`

### B. Context7 官方文档（已完成）
- [x] pymilvus - scalar filtering
- [x] milvus - JSON Path Index
- [x] milvus - boolean operators

**保存位置**：
- `reference/context7_pymilvus_scalar_filtering.md`
- `reference/context7_milvus_json_index.md`
- `reference/context7_milvus_boolean_operators.md`

### C. Grok-mcp 网络搜索（已完成）
- [x] milvus scalar filtering 2026
- [x] milvus json path index performance
- [x] milvus boolean expression tutorial

**保存位置**：
- `reference/search_web_results.md`

---

## 六、核心发现总结

### 1. 性能提升数据
- **标量索引**：点查询性能提升 **30倍**（1百万记录数据集）
- **JSON Path Index**：嵌套 JSON 查询性能提升 **100倍**
- **混合搜索**：显著缩短延迟，适用于高选择性查询

### 2. 2026 年最佳实践
1. **索引优化**：
   - 为常用标量字段创建索引
   - 使用 JSON Path Index 处理嵌套 JSON
   - 选择合适的索引类型（INVERTED, AUTOINDEX）

2. **查询优化**：
   - 使用 filter templating 减少解析开销
   - 简化表达式避免全表扫描
   - 按过滤选择性优化执行策略

3. **混合搜索**：
   - 标准过滤 vs 迭代过滤
   - Pre-filtering vs Post-filtering
   - 结合 LLM 生成复杂过滤表达式

### 3. 技术亮点
- **simdjson**：高性能 JSON 解析
- **位图操作**：TargetBitmap 优化
- **Chunked Segment**：分块存储优化
- **类型转换**：STRING_TO_DOUBLE 等转换函数

---

## 七、生成进度

### 阶段一：数据收集（已完成 ✓）
- [x] 步骤 1.1：Brainstorm 初步分析
- [x] 步骤 1.2：多源数据收集
  - [x] A. 源码分析
  - [x] B. Context7 官方文档
  - [x] C. Grok-mcp 网络搜索
- [ ] 步骤 1.3：用户确认拆解方案
- [ ] 步骤 1.4：生成最终 PLAN.md（当前文件）

### 阶段二：补充调研（待执行）
- [ ] 识别需要补充资料的部分
- [ ] 执行补充调研
- [ ] 生成抓取任务文件 FETCH_TASK.json
- [ ] 更新 PLAN.md

### 阶段三：文档生成（待执行）
- [ ] 读取所有 reference/ 资料
- [ ] 按顺序生成文档（23个文件）
- [ ] 最终验证

---

## 八、质量标准

### 文件长度限制
- 基础维度文件：300-500 行
- 核心概念文件：300-500 行
- 实战代码文件：600-700 行

### 引用格式
- 源码：`[来源: sourcecode/milvus/<文件路径>]`
- Context7：`[来源: reference/context7_<库名>_<序号>.md | <库名> 官方文档]`
- 搜索：`[来源: reference/search_<知识点简称>_<序号>.md]`

### 内容要求
- 初学者友好（简单语言、丰富示例）
- 代码可运行（Python）
- 双重类比（前端 + 日常生活）
- 与 Milvus 2.6 特性紧密结合
- 速成高效（20%核心 + 80%效果）

---

## 九、下一步行动

### 立即执行
1. 等待用户确认拆解方案
2. 如有需要，执行阶段二补充调研
3. 开始阶段三文档生成

### 生成顺序
1. 基础维度文件（第一部分）：00, 01, 02
2. 核心概念文件（7个）：03_1 到 03_7
3. 基础维度文件（第二部分）：04, 05, 06
4. 实战代码文件（6个）：07_1 到 07_6
5. 基础维度文件（第三部分）：08, 09, 10

---

## 十、参考资源

### 官方文档
1. Scalar Index: https://milvus.io/docs/scalar_index.md
2. Filtered Search: https://milvus.io/docs/filtered-search.md
3. Boolean Expressions: https://milvus.io/docs/boolean.md
4. JSON Indexing: https://milvus.io/docs/json-indexing.md

### 博客文章
1. Milvus 2.6 Introduction: https://milvus.io/blog/introduce-milvus-2-6-built-for-scale-designed-to-reduce-costs.md
2. Debug Slow Requests: https://milvus.io/blog/how-to-debug-slow-requests-in-milvus.md

### 学术论文
1. Filtered ANN Search (arXiv 2026): https://arxiv.org/html/2602.11443v1

### 教程
1. LlamaIndex Hybrid Search: https://milvus.io/docs/llamaindex_milvus_hybrid_search.md
2. LLM Generate Filters: https://milvus.io/docs/generating_milvus_query_filter_expressions.md

---

**计划版本**：v1.0
**创建时间**：2026-02-24
**状态**：数据收集完成，待用户确认后执行文档生成
