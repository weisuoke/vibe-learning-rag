# Milvus 标量过滤语法官方文档

> 来源：Context7 - /websites/milvus_io

---

## 1. 逻辑操作符（AND, OR, NOT）

**来源**：https://milvus.io/docs/basic-operators

### 功能说明

演示如何使用逻辑操作符组合多个过滤条件，创建复杂的查询。

### AND 操作符

```sql
filter = 'price > 100 AND stock > 50'
```

**说明**：
- 同时满足多个条件
- 所有条件都必须为真

### OR 操作符

```sql
filter = 'color == "red" OR color == "blue"'
```

**说明**：
- 满足任一条件即可
- 至少一个条件为真

### NOT 操作符

```sql
filter = 'NOT color == "green"'
```

**说明**：
- 否定条件
- 排除特定值

---

## 2. IS NULL 和 IS NOT NULL 操作符

**来源**：https://milvus.io/docs/basic-operators

### 功能说明

根据标量字段是否包含 NULL 值进行过滤。

### IS NULL

```sql
filter = 'description IS NULL'
```

**说明**：
- 查找缺失或未定义的值
- 空字符串不被视为 NULL（对于 VARCHAR 字段）

### IS NOT NULL

```sql
filter = 'description IS NOT NULL'
```

**说明**：
- 查找已定义的值
- 排除 NULL 值

### 组合使用

```sql
filter = 'description IS NOT NULL AND price > 10'
```

**说明**：
- 可以与其他条件组合
- 确保字段有值且满足其他条件

---

## 3. 关系操作符

**来源**：https://milvus.io/docs/v2.2.x/boolean

### 支持的操作符

Milvus 支持以下关系操作符，用于比较两个表达式：

| 操作符 | 说明 | 示例 |
|--------|------|------|
| `<` | 小于 | `a < b` |
| `>` | 大于 | `a > b` |
| `==` | 等于 | `a == b` |
| `!=` | 不等于 | `a != b` |
| `<=` | 小于等于 | `a <= b` |
| `>=` | 大于等于 | `a >= b` |

### 返回值

所有关系操作符返回布尔值（true 或 false）。

---

## 4. Milvus vs Elasticsearch 布尔查询对比

**来源**：https://milvus.io/docs/elasticsearch-queries-to-milvus

### Elasticsearch 查询

```python
resp = client.search(
    query={
        "bool": {
            "filter": {
                "term": {
                    "user": "kimchy"
                }
            },
            "filter": {
                "term": {
                    "tags": "production"
                }
            }
        }
    },
)
```

### Milvus 等价查询

```python
filter = 'user like "%kimchy%" AND ARRAY_CONTAINS(tags, "production")'

res = client.query(
    collection_name="my_collection",
    filter=filter,
    output_fields=["id", "user", "age", "tags"]
)
```

### 关键差异

1. **语法风格**：
   - Elasticsearch：结构化 JSON 查询
   - Milvus：SQL 风格的字符串表达式

2. **数组操作**：
   - Elasticsearch：使用 `term` 查询
   - Milvus：使用 `ARRAY_CONTAINS` 函数

3. **字符串匹配**：
   - Elasticsearch：精确匹配
   - Milvus：支持 `LIKE` 模糊匹配

---

## 5. 过滤功能总览

**来源**：https://milvus.io/docs/boolean

### 支持的操作符类型

Milvus 支持以下几种基本操作符用于数据过滤：

#### 1. 比较操作符

- `==`, `!=`, `>`, `<`, `>=`, `<=`
- 用于数值或文本字段的过滤

#### 2. 范围过滤

- `IN`：匹配特定值集合
- `LIKE`：模式匹配

#### 3. 算术操作符

- `+`, `-`, `*`, `/`, `%`, `**`
- 用于涉及数值字段的计算

#### 4. 逻辑操作符

- `AND`, `OR`, `NOT`
- 将多个条件组合成复杂表达式

#### 5. NULL 操作符

- `IS NULL`, `IS NOT NULL`
- 根据字段是否包含 NULL 值进行过滤

---

## 总结

### 标量过滤语法完整示例

```python
# 1. 比较操作符
"age > 25"
"city == '北京'"
"price >= 100 AND price <= 500"

# 2. 逻辑操作符
"age > 25 AND city == '北京'"
"color == 'red' OR color == 'blue'"
"NOT status == 'deleted'"

# 3. IN 操作符
"id IN [1, 2, 3, 4, 5]"
"color IN ['red', 'blue', 'green']"

# 4. LIKE 操作符
"name LIKE 'John%'"
"email LIKE '%@gmail.com'"

# 5. NULL 操作符
"description IS NULL"
"tags IS NOT NULL"

# 6. 数组操作符
"ARRAY_CONTAINS(tags, 'tech')"
"ARRAY_LENGTH(items) > 5"

# 7. JSON 操作符
"JSON_CONTAINS(metadata, 'category')"
"metadata['price'] < 1000"

# 8. 复合条件
"age > 25 AND (city == '北京' OR city == '上海') AND tags IS NOT NULL"
"price >= 100 AND price <= 500 AND stock > 0 AND NOT status == 'deleted'"
```

### 操作符优先级

1. **算术操作符**：`**` > `*`, `/`, `%` > `+`, `-`
2. **比较操作符**：`==`, `!=`, `>`, `<`, `>=`, `<=`
3. **逻辑操作符**：`NOT` > `AND` > `OR`

### 使用建议

1. **使用括号明确优先级**：
   ```python
   "age > 25 AND (city == '北京' OR city == '上海')"
   ```

2. **避免过于复杂的表达式**：
   - 将复杂条件拆分为多个简单条件
   - 使用索引优化常用过滤条件

3. **注意数据类型匹配**：
   - 字符串使用单引号或双引号
   - 数值不需要引号
   - 布尔值使用 `true` 或 `false`

4. **NULL 值处理**：
   - 空字符串 `""` 不等于 NULL
   - 使用 `IS NULL` 而不是 `== NULL`

5. **性能优化**：
   - 为常用的过滤字段创建索引
   - 将选择性高的条件放在前面
   - 避免在大字段上使用 LIKE 模糊匹配

---

## 与 Elasticsearch 的对比

| 特性 | Milvus | Elasticsearch |
|------|--------|---------------|
| 查询语法 | SQL 风格字符串 | JSON 结构化查询 |
| 字符串匹配 | LIKE 模糊匹配 | term/match 查询 |
| 数组操作 | ARRAY_CONTAINS | term 查询 |
| JSON 操作 | JSON_CONTAINS | nested 查询 |
| 逻辑组合 | AND/OR/NOT | bool 查询 |
| 性能 | 向量检索优化 | 全文搜索优化 |

---

## 实际应用场景

### 1. 电商场景

```python
# 查找特定价格范围、有库存、特定品牌的商品
filter = '''
    price >= 100 AND price <= 500
    AND stock > 0
    AND brand IN ['Apple', 'Samsung', 'Huawei']
    AND NOT status == 'discontinued'
'''
```

### 2. 用户画像场景

```python
# 查找特定年龄段、特定城市、有特定标签的用户
filter = '''
    age >= 25 AND age <= 35
    AND city IN ['北京', '上海', '深圳']
    AND ARRAY_CONTAINS(interests, 'tech')
    AND vip_level IS NOT NULL
'''
```

### 3. 内容推荐场景

```python
# 查找特定类别、高评分、最近发布的内容
filter = '''
    category == 'technology'
    AND rating >= 4.0
    AND publish_date > '2026-01-01'
    AND tags IS NOT NULL
'''
```

### 4. RAG 场景

```python
# 查找特定主题、特定语言、高质量的文档
filter = '''
    topic IN ['AI', 'ML', 'NLP']
    AND language == 'zh'
    AND quality_score >= 0.8
    AND metadata['source'] != 'unknown'
'''
```
