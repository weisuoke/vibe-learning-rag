# Milvus JSON Path Index 官方文档

> 来源：Context7 - /websites/milvus_io

---

## 1. JSON Path Index 创建（多语言示例）

**来源**：https://milvus.io/docs/use-json-fields

### 功能说明

演示如何为 JSON 字段的特定路径创建索引，支持多种数据类型和数组。

### 支持的 JSON Cast 类型

| Cast Type | Description | Example JSON Value |
|-----------|-------------|-------------------|
| `bool` | 布尔值 | `true`, `false` |
| `double` | 数值（整数或浮点数） | `42`, `99.99`, `-15.5` |
| `varchar` | 字符串值 | `"electronics"`, `"BrandA"` |
| `array_bool` | 布尔数组 | `[true, false, true]` |
| `array_double` | 数值数组 | `[1.2, 3.14, 42]` |
| `array_varchar` | 字符串数组 | `["tag1", "tag2", "tag3"]` |

**注意**：数组应包含相同类型的元素以获得最佳索引性能。

### Python 示例

```python
# 索引 category 字段为字符串
index_params.add_index(
    field_name="metadata",
    index_type="AUTOINDEX",  # 必须设置为 AUTOINDEX 或 INVERTED
    index_name="category_index",  # 唯一索引名称
    params={
        "json_path": "metadata[\"category\"]",  # JSON 键的路径
        "json_cast_type": "varchar"  # 数据转换类型
    }
)

# 索引 tags 数组为字符串数组
index_params.add_index(
    field_name="metadata",
    index_type="AUTOINDEX",
    index_name="tags_array_index",
    params={
        "json_path": "metadata[\"tags\"]",
        "json_cast_type": "array_varchar"
    }
)
```

### Java 示例

```java
Map<String,Object> extraParams1 = new HashMap<>();
extraParams1.put("json_path", "metadata[\"category\"]");
extraParams1.put("json_cast_type", "varchar");
indexParams.add(IndexParam.builder()
        .fieldName("metadata")
        .indexName("category_index")
        .indexType(IndexParam.IndexType.AUTOINDEX)
        .extraParams(extraParams1)
        .build());

Map<String,Object> extraParams2 = new HashMap<>();
extraParams2.put("json_path", "metadata[\"tags\"]");
extraParams2.put("json_cast_type", "array_varchar");
indexParams.add(IndexParam.builder()
        .fieldName("metadata")
        .indexName("tags_array_index")
        .indexType(IndexParam.IndexType.AUTOINDEX)
        .extraParams(extraParams2)
        .build());
```

### NodeJS 示例

```javascript
const indexParams = [
  {
    collection_name: "product_catalog",
    field_name: "metadata",
    index_name: "category_index",
    index_type: "AUTOINDEX",  // 也可以使用 "INVERTED"
    extra_params: {
      json_path: 'metadata["category"]',
      json_cast_type: "varchar",
    },
  },
  {
    collection_name: "product_catalog",
    field_name: "metadata",
    index_name: "tags_array_index",
    index_type: "AUTOINDEX",
    extra_params: {
      json_path: 'metadata["tags"]',
      json_cast_type: "array_varchar",
    },
  },
];
```

### Go 示例

```go
jsonIndex1 := index.NewJSONPathIndex(index.AUTOINDEX, "varchar", `metadata["category"]`)
    .WithIndexName("category_index")
jsonIndex2 := index.NewJSONPathIndex(index.AUTOINDEX, "array_varchar", `metadata["tags"]`)
    .WithIndexName("tags_array_index")

indexOpt1 := milvusclient.NewCreateIndexOption("product_catalog", "metadata", jsonIndex1)
indexOpt2 := milvusclient.NewCreateIndexOption("product_catalog", "metadata", jsonIndex2)
```

---

## 2. INVERTED 索引创建

**来源**：https://milvus.io/docs/inverted

### 功能说明

演示如何在 JSON 字段的特定路径上创建 INVERTED 索引。

### Python 示例

```python
# 构建索引参数
index_params.add_index(
    field_name="metadata",                    # JSON 字段名称
    index_type="INVERTED",
    index_name="metadata_category_index",
    params={
        "json_path": "metadata[\"category\"]",    # JSON 键的路径
        "json_cast_type": "varchar"              # 索引时转换的数据类型
    }
)

# 创建索引
client.create_index(
    collection_name="my_collection",  # 替换为你的集合名称
    index_params=index_params
)
```

### 关键特性

1. **INVERTED 索引**：适用于高效的精确匹配和范围查询
2. **json_path**：指定要索引的 JSON 键路径
3. **json_cast_type**：指定索引时的数据类型转换

---

## 3. 类型转换函数（STRING_TO_DOUBLE）

**来源**：https://milvus.io/docs/use-json-fields

### 功能说明

演示如何使用 STRING_TO_DOUBLE 转换函数将字符串表示的数字转换为双精度浮点数进行索引。

### 使用场景

当 JSON 字段包含数字字符串时，可以使用转换函数进行索引。

### Python 示例

```python
# 将字符串数字转换为 double 进行索引
index_params.add_index(
    field_name="metadata",
    index_type="AUTOINDEX",  # 必须设置为 AUTOINDEX 或 INVERTED
    index_name="string_to_double_index",  # 唯一索引名称
    params={
        "json_path": "metadata[\"string_price\"]",  # 要索引的 JSON 键路径
        "json_cast_type": "double",  # 数据转换类型
        "json_cast_function": "STRING_TO_DOUBLE"  # 转换函数；不区分大小写
    }
)
```

### Java 示例

```java
Map<String,Object> extraParams3 = new HashMap<>();
extraParams3.put("json_path", "metadata[\"string_price\"]");
extraParams3.put("json_cast_type", "double");
extraParams3.put("json_cast_function", "STRING_TO_DOUBLE");
indexParams.add(IndexParam.builder()
        .fieldName("metadata")
        .indexName("string_to_double_index")
        .indexType(IndexParam.IndexType.AUTOINDEX)
        .extraParams(extraParams3)
        .build());
```

### NodeJS 示例

```javascript
indexParams.push({
  collection_name: "product_catalog",
  field_name: "metadata",
  index_name: "string_to_double_index",
  index_type: "AUTOINDEX",  // 也可以使用 "INVERTED"
  extra_params: {
    json_path: 'metadata["string_price"]',
    json_cast_type: "double",
    json_cast_function: "STRING_TO_DOUBLE",  // 不区分大小写
  },
});
```

### Go 示例

```go
jsonIndex3 := index.NewJSONPathIndex(index.AUTOINDEX, "double", `metadata["string_price"]`)
                    .WithIndexName("string_to_double_index")

indexOpt3 := milvusclient.NewCreateIndexOption("product_catalog", "metadata", jsonIndex3)
```

### cURL 示例

```bash
export stringToDoubleIndex='{
  "fieldName": "metadata",
  "indexName": "string_to_double_index",
  "params": {
    "index_type": "AUTOINDEX",
    "json_path": "metadata[\"string_price\"]",
    "json_cast_type": "double",
    "json_cast_function": "STRING_TO_DOUBLE"
  }
}'
```

### 关键特性

1. **json_cast_type 必需**：必须与函数输出类型匹配
2. **转换失败跳过**：转换失败的值会被跳过
3. **不区分大小写**：函数名称不区分大小写

---

## 4. 深度嵌套 JSON Path 索引

**来源**：https://milvus.io/docs/enable-dynamic-field

### Python 示例

```python
index_params.add_index(
    field_name="dynamic_json",
    index_type="AUTOINDEX",  # 必须设置为 AUTOINDEX 或 INVERTED
    index_name="json_nested_index",  # 唯一索引名称
    params={
        "json_cast_type": "double",
        "json_path": "dynamic_json['nested']['value']"
    }
)
```

### JavaScript 示例

```javascript
const indexParams = [
    {
      collection_name: 'my_collection',
      field_name: 'overview',
      index_name: 'overview_index',
      index_type: 'AUTOINDEX',
      metric_type: 'NONE',
      params: {
        json_path: 'overview',
        json_cast_type: 'varchar',
      },
    },
    {
      collection_name: 'my_collection',
      field_name: 'words',
      index_name: 'words_index',
      index_type: 'AUTOINDEX',
      metric_type: 'NONE',
      params: {
        json_path: 'words',
        json_cast_type: 'double',
      },
    },
    {
      collection_name: 'my_collection',
      field_name: 'dynamic_json',
      index_name: 'json_varchar_index',
      index_type: 'AUTOINDEX',
      metric_type: 'NONE',
      params: {
        json_cast_type: 'varchar',
        json_path: "dynamic_json['varchar']",
      },
    },
    {
      collection_name: 'my_collection',
      field_name: 'dynamic_json',
      index_name: 'json_nested_index',
      index_type: 'AUTOINDEX',
      metric_type: 'NONE',
      params: {
        json_cast_type: 'double',
        json_path: "dynamic_json['nested']['value']",
      },
    },
  ];
```

### 关键特性

1. **嵌套访问**：支持多层嵌套的 JSON 路径
2. **动态字段**：支持动态字段的 JSON 索引
3. **多索引**：同一个 JSON 字段可以创建多个索引

---

## 总结

### JSON Path Index 核心特性

1. **索引类型**：
   - `AUTOINDEX`：自动选择最佳索引类型
   - `INVERTED`：倒排索引，适用于精确匹配和范围查询

2. **数据类型支持**：
   - 标量类型：`bool`, `double`, `varchar`
   - 数组类型：`array_bool`, `array_double`, `array_varchar`

3. **类型转换函数**：
   - `STRING_TO_DOUBLE`：将字符串转换为 double
   - 转换失败的值会被跳过

4. **JSON Path 语法**：
   - 单层访问：`metadata["category"]`
   - 嵌套访问：`dynamic_json['nested']['value']`
   - 数组访问：`metadata["tags"]`

5. **性能优化**：
   - 2026 核心特性：JSON Path Index - 100x 嵌套 JSON 查询性能提升
   - 适用于复杂的元数据过滤场景

### 使用建议

1. **选择合适的索引类型**：
   - 精确匹配：使用 `INVERTED` 索引
   - 自动优化：使用 `AUTOINDEX`

2. **正确设置 json_cast_type**：
   - 必须与实际数据类型匹配
   - 数组类型使用 `array_*` 前缀

3. **使用类型转换函数**：
   - 当数据类型不匹配时使用转换函数
   - 注意转换失败的处理

4. **多索引策略**：
   - 同一个 JSON 字段可以创建多个索引
   - 为不同的查询场景创建不同的索引
