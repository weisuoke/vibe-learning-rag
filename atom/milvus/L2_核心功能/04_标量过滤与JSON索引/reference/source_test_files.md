# 测试文件分析

> 来源：sourcecode/milvus/tests/python_client/

---

## test_milvus_client_json_path_index.py

**文件路径**：`milvus_client/test_milvus_client_json_path_index.py`

### 测试场景

**测试类**：`TestMilvusClientInsertJsonPathIndexValid`

### 核心测试用例

#### 1. 插入后创建 JSON Path Index

```python
@pytest.mark.tags(CaseLabel.L1)
@pytest.mark.parametrize("enable_dynamic_field", [True, False])
def test_milvus_client_insert_before_json_path_index(
    self, enable_dynamic_field, supported_json_cast_type, supported_varchar_scalar_index):
    """
    target: test insert and then create json path index
    method: create json path index after insert
    steps: 1. create schema
           2. create collection
           3. insert
           4. prepare json path index params with parameter "json_cast_type" and "json_path"
           5. create index
    expected: insert and create json path index successfully
    """
```

### 支持的参数

#### 1. 索引类型
```python
@pytest.fixture(scope="function", params=["INVERTED"])
def supported_varchar_scalar_index(self, request):
    yield request.param
```

#### 2. JSON Cast 类型
```python
@pytest.fixture(scope="function", params=["BOOL", "Double", "Varchar", "json"])
def supported_json_cast_type(self, request):
    yield request.param
```

### 测试数据分布

#### 1. 嵌套 JSON 对象
```python
rows = [{
    default_primary_key_field_name: i,
    default_vector_field_name: vectors[i],
    default_string_field_name: str(i),
    json_field_name: {'a': {"b": i}}
} for i in range(default_nb)]
```

#### 2. 简单数值
```python
rows = [{
    default_primary_key_field_name: i,
    default_vector_field_name: vectors[i],
    default_string_field_name: str(i),
    json_field_name: i
} for i in range(default_nb, default_nb+10)]
```

#### 3. 空对象
```python
rows = [{
    default_primary_key_field_name: i,
    default_vector_field_name: vectors[i],
    default_string_field_name: str(i),
    json_field_name: {}
} for i in range(default_nb+10, default_nb+20)]
```

#### 4. 数组
```python
rows = [{
    default_primary_key_field_name: i,
    default_vector_field_name: vectors[i],
    default_string_field_name: str(i),
    json_field_name: {'a': [1, 2, 3]}
} for i in range(default_nb + 20, default_nb + 30)]
```

#### 5. 嵌套数组对象
```python
rows = [{
    default_primary_key_field_name: i,
    default_vector_field_name: vectors[i],
    default_string_field_name: str(i),
    json_field_name: {'a': [{'b': 1}, 2, 3]}
} for i in range(default_nb + 20, default_nb + 30)]
```

#### 6. NULL 值
```python
rows = [{
    default_primary_key_field_name: i,
    default_vector_field_name: vectors[i],
    default_string_field_name: str(i),
    json_field_name: {'a': [{'b': None}, 2, 3]}
} for i in range(default_nb + 30, default_nb + 40)]
```

### JSON Path 索引创建

#### 1. 嵌套路径
```python
index_params.add_index(
    field_name=json_field_name,
    index_name=index_name,
    index_type="INVERTED",
    params={
        "json_cast_type": "Double",  # BOOL, Double, Varchar, json
        "json_path": f"{json_field_name}['a']['b']"
    }
)
```

#### 2. 单层路径
```python
index_params.add_index(
    field_name=json_field_name,
    index_name=index_name + '1',
    index_type="INVERTED",
    params={
        "json_cast_type": supported_json_cast_type,
        "json_path": f"{json_field_name}['a']"
    }
)
```

#### 3. 根路径
```python
index_params.add_index(
    field_name=json_field_name,
    index_name=index_name + '2',
    index_type="INVERTED",
    params={
        "json_cast_type": supported_json_cast_type,
        "json_path": f"{json_field_name}"
    }
)
```

#### 4. 数组元素路径
```python
index_params.add_index(
    field_name=json_field_name,
    index_name=index_name + '3',
    index_type="INVERTED",
    params={
        "json_cast_type": supported_json_cast_type,
        "json_path": f"{json_field_name}['a'][0]['b']"
    }
)
```

#### 5. 数组索引路径
```python
index_params.add_index(
    field_name=json_field_name,
    index_name=index_name + '4',
    index_type="INVERTED",
    params={
        "json_cast_type": supported_json_cast_type,
        "json_path": f"{json_field_name}['a'][0]"
    }
)
```

### 索引验证

```python
self.describe_index(client, collection_name, index_name,
                    check_task=CheckTasks.check_describe_index_property,
                    check_items={
                        "json_cast_type": supported_json_cast_type,
                        "json_path": f"{json_field_name}['a']['b']",
                        "index_type": supported_varchar_scalar_index,
                        "field_name": json_field_name,
                        "index_name": index_name
                    })
```

---

## test_milvus_client_scalar_filtering.py

**文件路径**：`milvus_client/expressions/test_milvus_client_scalar_filtering.py`

### 测试类

**测试类**：`TestScalarExpressionFilteringOptimized`

### 核心特性

1. **单集合全数据类型测试**：在一个集合中测试所有数据类型和索引类型
2. **全面的操作符覆盖**：比较、范围、NULL 操作符
3. **索引类型对比测试**：确保不同索引类型的一致性
4. **自动数据保存**：测试失败时自动保存数据用于调试
5. **LIKE 模式测试**：支持转义字符
6. **批量插入**：使用 ct.default_nb 批量大小
7. **默认 100,000 条记录**：全面覆盖测试

### 支持的数据类型和索引类型

```python
data_types_config = {
    # Integer types - support INVERTED, BITMAP, STL_SORT, AUTOINDEX
    DataType.INT8: ["no_index", "inverted", "bitmap", "stl_sort", "autoindex"],
    DataType.INT16: ["no_index", "inverted", "bitmap", "stl_sort", "autoindex"],
    DataType.INT32: ["no_index", "inverted", "bitmap", "stl_sort", "autoindex"],
    DataType.INT64: ["no_index", "inverted", "bitmap", "stl_sort", "autoindex"],

    # BOOL - supports INVERTED, BITMAP, AUTOINDEX
    DataType.BOOL: ["no_index", "inverted", "bitmap", "autoindex"],

    # Float types - support INVERTED, STL_SORT, AUTOINDEX
    DataType.FLOAT: ["no_index", "inverted", "stl_sort", "autoindex"],
    DataType.DOUBLE: ["no_index", "inverted", "stl_sort", "autoindex"],

    # VARCHAR - supports index types
    DataType.VARCHAR: ["no_index", "inverted", "bitmap", "trie", "ngram", "autoindex"],

    # JSON - supports INVERTED, NGRAM, AUTOINDEX
    DataType.JSON: ["no_index", "inverted", "ngram", "autoindex"],

    # ARRAY types - support depends on element type
    (DataType.ARRAY, DataType.INT32): ["no_index", "inverted", "bitmap", "autoindex"],
    (DataType.ARRAY, DataType.INT64): ["no_index", "inverted", "bitmap", "autoindex"],
    (DataType.ARRAY, DataType.VARCHAR): ["no_index", "inverted", "bitmap", "autoindex"],
}
```

### 操作符定义

```python
# 比较操作符
comparison_operators = ["==", "!=", ">", "<", ">=", "<="]

# 范围操作符
range_operators = ["IN", "LIKE"]

# NULL 操作符
null_operators = ["IS NULL", "IS NOT NULL"]
```

### Schema 创建

```python
def create_comprehensive_schema_with_index_types(self, client, enable_dynamic_field: bool = False):
    """
    Create comprehensive schema with all data types and their supported index types.

    Args:
        client: Milvus client instance
        enable_dynamic_field: Whether to enable dynamic field support

    Returns:
        Tuple of (schema, field_mapping, index_configs)
    """
    schema = self.create_schema(client, enable_dynamic_field=enable_dynamic_field)[0]
    schema.add_field(default_primary_key_field_name, DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field(default_vector_field_name, DataType.FLOAT_VECTOR, dim=default_dim)

    field_mapping = {}
    index_configs = {}

    # Add fields for each data type with their supported index types
    for data_type, supported_indexes in data_types_config.items():
        if isinstance(data_type, tuple):
            # Array type
            array_type, element_type = data_type
            base_name = f"array_{element_type.name.lower()}"

            for index_type in supported_indexes:
                field_name = f"{base_name}_{index_type}"
                if element_type == DataType.VARCHAR:
                    schema.add_field(
                        field_name, DataType.ARRAY,
                        element_type=element_type,
                        max_length=100, max_capacity=10,
                        nullable=True
                    )
                else:
                    schema.add_field(
                        field_name, DataType.ARRAY,
                        element_type=element_type,
                        max_capacity=10,
                        nullable=True
                    )
                field_mapping[field_name] = data_type
                index_configs[field_name] = index_type
        else:
            # Scalar type
            base_name = data_type.name.lower()

            for index_type in supported_indexes:
                field_name = f"{base_name}_{index_type}"
                if data_type == DataType.VARCHAR:
                    schema.add_field(field_name, data_type, max_length=100, nullable=True)
                else:
                    schema.add_field(field_name, data_type, nullable=True)
                field_mapping[field_name] = data_type
                index_configs[field_name] = index_type

    return schema, field_mapping, index_configs
```

### 随机数据生成

```python
def generate_random_scalar_value(self, data_type: DataType, need_none: bool = True) -> Any:
    """
    Generate random scalar values for different data types with 10% chance of None.

    Args:
        data_type: The data type to generate value for
        need_none: Whether to include None values (10% probability)

    Returns:
        Random value of the specified data type
    """
    # 10% chance of None if need_none is True
    if need_none and random.random() < 0.1:
        return None

    # Generate random values based on data type
    # ...
```

---

## 关键测试场景总结

### 1. JSON Path Index 测试

**测试覆盖**：
- ✅ 嵌套 JSON 对象访问：`json_field['a']['b']`
- ✅ 单层路径访问：`json_field['a']`
- ✅ 根路径访问：`json_field`
- ✅ 数组元素访问：`json_field['a'][0]['b']`
- ✅ 数组索引访问：`json_field['a'][0]`
- ✅ NULL 值处理
- ✅ 空对象处理
- ✅ 多种 json_cast_type：BOOL, Double, Varchar, json

### 2. 标量过滤测试

**测试覆盖**：
- ✅ 所有标量数据类型：INT8, INT16, INT32, INT64, BOOL, FLOAT, DOUBLE, VARCHAR
- ✅ JSON 类型
- ✅ ARRAY 类型：INT32, INT64, VARCHAR
- ✅ 所有索引类型：no_index, inverted, bitmap, stl_sort, autoindex, trie, ngram
- ✅ 比较操作符：==, !=, >, <, >=, <=
- ✅ 范围操作符：IN, LIKE
- ✅ NULL 操作符：IS NULL, IS NOT NULL
- ✅ 10% NULL 值分布
- ✅ 100,000 条记录测试

### 3. 性能测试

**测试维度**：
- 索引类型对比（no_index vs inverted vs bitmap vs stl_sort vs autoindex）
- 数据类型对比
- 操作符性能对比
- 大规模数据测试（100,000 条记录）

---

## 使用示例

### 1. JSON Path Index 创建

```python
# 创建 JSON Path Index
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="json_field",
    index_name="json_index",
    index_type="INVERTED",
    params={
        "json_cast_type": "Double",  # BOOL, Double, Varchar, json
        "json_path": "json_field['a']['b']"
    }
)
client.create_index(collection_name, index_params)
```

### 2. 标量过滤查询

```python
# 比较操作符
"age > 25"
"city == '北京'"

# 范围操作符
"color IN ['red', 'blue', 'green']"
"name LIKE 'John%'"

# NULL 操作符
"description IS NULL"
"tags IS NOT NULL"

# 复合条件
"age > 25 AND (city == '北京' OR city == '上海')"
```

---

## 总结

1. **JSON Path Index**：支持多种路径格式和 cast 类型
2. **标量过滤**：全面支持所有数据类型和操作符
3. **索引优化**：多种索引类型可选
4. **NULL 处理**：完善的 NULL 值支持
5. **性能测试**：大规模数据测试验证
