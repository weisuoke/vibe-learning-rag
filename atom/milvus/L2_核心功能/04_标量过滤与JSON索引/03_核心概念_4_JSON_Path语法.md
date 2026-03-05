# 核心概念 4：JSON Path 语法

> **目标**：掌握 Milvus JSON Path 的完整语法和使用方法

---

## 概述

JSON Path 是访问 JSON 字段内部数据的路径表达式，支持单层访问、嵌套访问、数组访问等多种模式。

**核心价值**：
- 灵活访问嵌套数据
- 支持数组元素访问
- 类型安全的路径表达式
- 高效的索引优化

[来源: reference/context7_milvus_json_index.md | Milvus 官方文档]

---

## 1. 基础语法

### 1.1 单层访问

**语法**：
```python
json_field['key']
```

**示例**：
```python
# 访问 category 字段
"metadata['category'] == 'electronics'"

# 访问 price 字段
"metadata['price'] >= 100"

# 访问 status 字段
"metadata['status'] != 'deleted'"
```

**JSON 数据示例**：
```json
{
  "metadata": {
    "category": "electronics",
    "price": 299.99,
    "status": "active"
  }
}
```

### 1.2 引号规则

**双引号与单引号**：
```python
# 使用双引号（推荐）
"metadata[\"category\"] == 'electronics'"

# 使用单引号
"metadata['category'] == 'electronics'"
```

**注意事项**：
- 外层使用双引号时，内层使用单引号或转义双引号
- 外层使用单引号时，内层使用双引号或转义单引号
- 保持一致性，避免混淆

[来源: reference/context7_milvus_json_index.md | Milvus 官方文档]

---

## 2. 嵌套访问

### 2.1 多层嵌套

**语法**：
```python
json_field['level1']['level2']['level3']
```

**示例**：
```python
# 两层嵌套
"metadata['product']['brand'] == 'Apple'"

# 三层嵌套
"metadata['user']['profile']['age'] > 25"

# 四层嵌套
"metadata['data']['nested']['deep']['value'] != null"
```

**JSON 数据示例**：
```json
{
  "metadata": {
    "product": {
      "brand": "Apple",
      "model": "iPhone 15"
    },
    "user": {
      "profile": {
        "age": 30,
        "city": "Beijing"
      }
    }
  }
}
```

### 2.2 嵌套访问规则

**路径构建**：
```python
# 逐层访问
metadata → metadata['product'] → metadata['product']['brand']

# 完整路径
"metadata['product']['brand']"
```

**错误示例**：
```python
# ❌ 错误：跳过中间层
"metadata['brand']"  # 无法访问到 product.brand

# ✅ 正确：完整路径
"metadata['product']['brand']"
```

[来源: reference/context7_milvus_json_index.md | Milvus 官方文档]

---

## 3. 数组访问

### 3.1 数组索引访问

**语法**：
```python
json_field['array'][index]
```

**示例**：
```python
# 访问第一个元素
"metadata['tags'][0] == 'hot'"

# 访问第二个元素
"metadata['items'][1] == 'laptop'"

# 访问最后一个元素（需要知道长度）
"metadata['history'][9] != null"
```

**JSON 数据示例**：
```json
{
  "metadata": {
    "tags": ["hot", "new", "sale"],
    "items": ["phone", "laptop", "tablet"],
    "history": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  }
}
```

### 3.2 数组元素嵌套访问

**语法**：
```python
json_field['array'][index]['key']
```

**示例**：
```python
# 访问数组元素的属性
"metadata['reviews'][0]['rating'] >= 4.0"

# 访问数组元素的嵌套属性
"metadata['orders'][0]['items'][0]['price'] < 1000"
```

**JSON 数据示例**：
```json
{
  "metadata": {
    "reviews": [
      {"rating": 4.5, "comment": "Great product"},
      {"rating": 5.0, "comment": "Excellent"}
    ],
    "orders": [
      {
        "items": [
          {"name": "Phone", "price": 999},
          {"name": "Case", "price": 29}
        ]
      }
    ]
  }
}
```

### 3.3 数组操作函数

**ARRAY_CONTAINS**：
```python
# 检查数组是否包含特定值
"ARRAY_CONTAINS(metadata['tags'], 'tech')"

# 检查嵌套数组
"ARRAY_CONTAINS(metadata['categories'], 'electronics')"
```

**ARRAY_LENGTH**：
```python
# 检查数组长度
"ARRAY_LENGTH(metadata['tags']) > 5"

# 检查嵌套数组长度
"ARRAY_LENGTH(metadata['items']) >= 10"
```

[来源: reference/source_test_files.md | test_milvus_client_json_path_index.py]

---

## 4. 根路径访问

### 4.1 根路径语法

**语法**：
```python
json_field
```

**示例**：
```python
# 访问整个 JSON 字段
"metadata != null"

# 检查 JSON 字段是否存在
"metadata IS NOT NULL"
```

**JSON 数据示例**：
```json
{
  "metadata": {
    "category": "electronics",
    "price": 299.99
  }
}
```

### 4.2 根路径索引

**创建根路径索引**：
```python
# 为整个 JSON 字段创建索引
index_params.add_index(
    field_name="metadata",
    index_type="INVERTED",
    params={
        "json_path": "metadata",  # 根路径
        "json_cast_type": "json"
    }
)
```

[来源: reference/source_test_files.md | test_milvus_client_json_path_index.py]

---

## 5. JSON Path Index 创建

### 5.1 基础索引创建

**语法**：
```python
index_params.add_index(
    field_name="json_field_name",
    index_type="INVERTED",  # 或 AUTOINDEX
    params={
        "json_path": "path_expression",
        "json_cast_type": "data_type"
    }
)
```

**示例**：
```python
# 单层路径索引
index_params.add_index(
    field_name="metadata",
    index_type="INVERTED",
    params={
        "json_path": "metadata['category']",
        "json_cast_type": "varchar"
    }
)

# 嵌套路径索引
index_params.add_index(
    field_name="metadata",
    index_type="INVERTED",
    params={
        "json_path": "metadata['product']['brand']",
        "json_cast_type": "varchar"
    }
)

# 数组路径索引
index_params.add_index(
    field_name="metadata",
    index_type="INVERTED",
    params={
        "json_path": "metadata['tags']",
        "json_cast_type": "array_varchar"
    }
)
```

### 5.2 支持的 Cast 类型

**标量类型**：
- `bool`：布尔值
- `double`：数值（整数或浮点数）
- `varchar`：字符串值

**数组类型**：
- `array_bool`：布尔数组
- `array_double`：数值数组
- `array_varchar`：字符串数组

**JSON 类型**：
- `json`：保持 JSON 类型

**示例**：
```python
# 布尔类型
index_params.add_index(
    field_name="metadata",
    index_type="INVERTED",
    params={
        "json_path": "metadata['active']",
        "json_cast_type": "bool"
    }
)

# 数值类型
index_params.add_index(
    field_name="metadata",
    index_type="INVERTED",
    params={
        "json_path": "metadata['price']",
        "json_cast_type": "double"
    }
)

# 字符串数组
index_params.add_index(
    field_name="metadata",
    index_type="INVERTED",
    params={
        "json_path": "metadata['tags']",
        "json_cast_type": "array_varchar"
    }
)
```

[来源: reference/context7_milvus_json_index.md | Milvus 官方文档]

---

## 6. 类型转换函数

### 6.1 STRING_TO_DOUBLE

**功能**：将字符串表示的数字转换为双精度浮点数

**语法**：
```python
index_params.add_index(
    field_name="metadata",
    index_type="INVERTED",
    params={
        "json_path": "metadata['string_price']",
        "json_cast_type": "double",
        "json_cast_function": "STRING_TO_DOUBLE"
    }
)
```

**使用场景**：
```python
# JSON 数据
{
  "metadata": {
    "string_price": "99.99"  # 字符串格式的数字
  }
}

# 创建索引时转换为 double
index_params.add_index(
    field_name="metadata",
    index_type="INVERTED",
    params={
        "json_path": "metadata['string_price']",
        "json_cast_type": "double",
        "json_cast_function": "STRING_TO_DOUBLE"
    }
)

# 查询时可以使用数值比较
"metadata['string_price'] >= 50.0"
```

### 6.2 转换规则

**成功转换**：
```python
"99.99" → 99.99
"100" → 100.0
"-15.5" → -15.5
```

**转换失败**（跳过）：
```python
"abc" → 跳过
"" → 跳过
null → 跳过
```

**注意事项**：
- 转换函数不区分大小写
- 转换失败的值会被跳过，不会报错
- `json_cast_type` 必须与函数输出类型匹配

[来源: reference/context7_milvus_json_index.md | Milvus 官方文档]

---

## 7. 实战示例

### 7.1 电商场景

**JSON 数据结构**：
```json
{
  "metadata": {
    "product": {
      "category": "electronics",
      "brand": "Apple",
      "model": "iPhone 15"
    },
    "pricing": {
      "base": 999.99,
      "discount": 0.1,
      "final": 899.99
    },
    "inventory": {
      "stock": 100,
      "warehouse": "Beijing"
    },
    "tags": ["hot", "new", "5G"]
  }
}
```

**索引创建**：
```python
# 产品类别索引
index_params.add_index(
    field_name="metadata",
    index_type="INVERTED",
    params={
        "json_path": "metadata['product']['category']",
        "json_cast_type": "varchar"
    }
)

# 品牌索引
index_params.add_index(
    field_name="metadata",
    index_type="INVERTED",
    params={
        "json_path": "metadata['product']['brand']",
        "json_cast_type": "varchar"
    }
)

# 价格索引
index_params.add_index(
    field_name="metadata",
    index_type="INVERTED",
    params={
        "json_path": "metadata['pricing']['final']",
        "json_cast_type": "double"
    }
)

# 标签数组索引
index_params.add_index(
    field_name="metadata",
    index_type="INVERTED",
    params={
        "json_path": "metadata['tags']",
        "json_cast_type": "array_varchar"
    }
)
```

**查询示例**：
```python
filter = '''
    metadata['product']['category'] == 'electronics'
    AND metadata['product']['brand'] IN ['Apple', 'Samsung']
    AND metadata['pricing']['final'] >= 500 AND metadata['pricing']['final'] <= 1000
    AND metadata['inventory']['stock'] > 0
    AND ARRAY_CONTAINS(metadata['tags'], 'hot')
'''
```

### 7.2 用户画像场景

**JSON 数据结构**：
```json
{
  "metadata": {
    "user": {
      "profile": {
        "age": 30,
        "city": "Beijing",
        "vip_level": 3
      },
      "behavior": {
        "total_purchase": 15000,
        "last_login": 1640995200
      }
    },
    "interests": ["tech", "gaming", "travel"],
    "orders": [
      {"id": 1, "amount": 999},
      {"id": 2, "amount": 1299}
    ]
  }
}
```

**索引创建**：
```python
# 年龄索引
index_params.add_index(
    field_name="metadata",
    index_type="INVERTED",
    params={
        "json_path": "metadata['user']['profile']['age']",
        "json_cast_type": "double"
    }
)

# 城市索引
index_params.add_index(
    field_name="metadata",
    index_type="INVERTED",
    params={
        "json_path": "metadata['user']['profile']['city']",
        "json_cast_type": "varchar"
    }
)

# 兴趣数组索引
index_params.add_index(
    field_name="metadata",
    index_type="INVERTED",
    params={
        "json_path": "metadata['interests']",
        "json_cast_type": "array_varchar"
    }
)
```

**查询示例**：
```python
filter = '''
    metadata['user']['profile']['age'] >= 25 AND metadata['user']['profile']['age'] <= 35
    AND metadata['user']['profile']['city'] IN ['Beijing', 'Shanghai']
    AND metadata['user']['behavior']['total_purchase'] > 10000
    AND ARRAY_CONTAINS(metadata['interests'], 'tech')
'''
```

---

## 8. 常见错误

### 8.1 路径错误

**错误示例**：
```python
# ❌ 错误：路径不完整
"metadata['brand']"  # 应该是 metadata['product']['brand']

# ❌ 错误：使用点号语法
"metadata.product.brand"  # Milvus 不支持点号语法

# ✅ 正确
"metadata['product']['brand']"
```

### 8.2 引号错误

**错误示例**：
```python
# ❌ 错误：引号不匹配
"metadata["category"] == 'electronics'"  # 内外引号冲突

# ✅ 正确：转义双引号
"metadata[\"category\"] == 'electronics'"

# ✅ 正确：使用单引号
"metadata['category'] == 'electronics'"
```

### 8.3 类型错误

**错误示例**：
```python
# ❌ 错误：cast_type 与实际类型不匹配
index_params.add_index(
    field_name="metadata",
    index_type="INVERTED",
    params={
        "json_path": "metadata['price']",  # 实际是数值
        "json_cast_type": "varchar"  # 错误地指定为字符串
    }
)

# ✅ 正确
index_params.add_index(
    field_name="metadata",
    index_type="INVERTED",
    params={
        "json_path": "metadata['price']",
        "json_cast_type": "double"  # 正确的类型
    }
)
```

---

## 9. 性能优化

### 9.1 索引策略

**为常用路径创建索引**：
```python
# 高频查询路径
index_params.add_index(
    field_name="metadata",
    index_type="INVERTED",
    params={
        "json_path": "metadata['category']",
        "json_cast_type": "varchar"
    }
)

# 低频查询路径（可以不建索引）
# metadata['rare_field'] 不需要索引
```

**性能提升**：
- 有索引：100x 性能提升
- 无索引：全表扫描

### 9.2 查询优化

**选择性高的条件在前**：
```python
# 好的写法
filter = '''
    metadata['category'] == 'electronics'
    AND metadata['brand'] == 'Apple'
    AND metadata['price'] >= 100
'''

# 不好的写法
filter = '''
    metadata['price'] >= 100
    AND metadata['brand'] == 'Apple'
    AND metadata['category'] == 'electronics'
'''
```

---

## 10. 类比理解

### 前端开发类比

**JavaScript 对象访问**：
```javascript
// JavaScript
const brand = product.metadata.product.brand;
const price = product.metadata.pricing.final;
const firstTag = product.metadata.tags[0];

// Milvus JSON Path
"metadata['product']['brand']"
"metadata['pricing']['final']"
"metadata['tags'][0]"
```

### Python 字典访问

**Python 字典**：
```python
# Python
brand = metadata['product']['brand']
price = metadata['pricing']['final']
first_tag = metadata['tags'][0]

# Milvus JSON Path（语法相同）
"metadata['product']['brand']"
"metadata['pricing']['final']"
"metadata['tags'][0]"
```

---

## 11. 扩展阅读

### 相关概念
- **数据类型支持** → 03_核心概念_3_数据类型支持.md
- **JSON 索引类型** → 03_核心概念_5_JSON索引类型.md
- **标量索引优化** → 03_核心概念_6_标量索引优化.md

### 实战代码
- **JSON字段过滤** → 07_实战代码_场景4_JSON字段过滤.md
- **JSON Path Index性能对比** → 07_实战代码_场景5_JSON_Path_Index性能对比.md

---

## 一句话总结

JSON Path 语法支持单层、嵌套、数组等多种访问模式，通过创建 JSON Path Index 可实现100倍性能提升，是处理复杂元数据的核心技术。

---

**下一步**：学习 [03_核心概念_5_JSON索引类型.md](./03_核心概念_5_JSON索引类型.md)，了解索引实现原理
