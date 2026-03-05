# JSON索引实现源码分析

> 来源：sourcecode/milvus/internal/core/src/index/

---

## JsonFlatIndex 实现

**文件**：`JsonFlatIndex.cpp`

### 核心功能

1. **JSON Path 解析**：
   - 使用 `parse_json_pointer(nested_path_)` 解析 JSON 路径
   - 支持嵌套路径访问：`json_field['a']['b']`

2. **数据处理流程**：
```cpp
void JsonFlatIndex::build_index_for_json(
    const std::vector<std::shared_ptr<FieldDataBase>>& field_datas) {
    int64_t offset = 0;
    auto tokens = parse_json_pointer(nested_path_);
    simdjson::padded_string scratch_buffer(256);  // 复用缓冲区优化

    for (const auto& data : field_datas) {
        auto n = data->get_num_rows();
        for (int i = 0; i < n; i++) {
            // 处理 NULL 值
            if (schema_.nullable() && !data->is_valid(i)) {
                null_offset_.push_back(offset);
                wrapper_->add_json_array_data(nullptr, 0, offset++);
                continue;
            }

            // 检查路径是否存在
            auto json = static_cast<const Json*>(data->RawValue(i));
            auto exists = path_exists(json->dom_doc(), tokens);
            if (!exists || !json->exist(nested_path_)) {
                wrapper_->add_json_array_data(nullptr, 0, offset++);
                continue;
            }

            // 提取子路径数据
            if (nested_path_ == "") {
                wrapper_->add_json_data(json, 1, offset++);
            } else {
                auto doc = json->doc();
                auto res = doc.at_pointer(nested_path_);
                // ... 序列化并添加数据
            }
        }
    }
}
```

3. **性能优化**：
   - 使用 `simdjson::padded_string` 作为 scratch buffer
   - 动态调整缓冲区大小：`(str.size() + 1) * 2`
   - 避免重复的堆内存分配

---

## JsonInvertedIndex 实现

**文件**：`JsonInvertedIndex.cpp`

### 核心功能

1. **支持的数据类型**：
```cpp
template class JsonInvertedIndex<bool>;
template class JsonInvertedIndex<int64_t>;
template class JsonInvertedIndex<double>;
template class JsonInvertedIndex<std::string>;
```

2. **索引构建**：
```cpp
template <typename T>
void JsonInvertedIndex<T>::build_index_for_json(
    const std::vector<std::shared_ptr<FieldDataBase>>& field_datas) {

    ProcessJsonFieldData<T>(
        field_datas,
        this->schema_,
        nested_path_,
        cast_type_,
        cast_function_,
        // add data
        [this](const T* data, int64_t size, int64_t offset) {
            this->wrapper_->template add_array_data<T>(data, size, offset);
        },
        // handle null
        [this](int64_t offset) { this->null_offset_.push_back(offset); },
        // handle non exist
        [this](int64_t offset) { this->non_exist_offsets_.push_back(offset); },
        // handle error
        [this](const Json& json, const std::string& nested_path, simdjson::error_code error) {
            this->error_recorder_.Record(json, nested_path, error);
        }
    );
}
```

3. **Exists() 操作**：
```cpp
template <typename T>
TargetBitmap JsonInvertedIndex<T>::Exists() {
    int64_t count = this->Count();
    TargetBitmap bitset(count, true);

    // 标记不存在的偏移量
    auto end = std::lower_bound(
        non_exist_offsets_.begin(), non_exist_offsets_.end(), count);
    for (auto iter = non_exist_offsets_.begin(); iter != end; ++iter) {
        bitset.reset(*iter);
    }

    return bitset;
}
```

4. **向后兼容性**：
   - 支持 v2.5.x 数据格式
   - 如果没有 `non_exist_offset_file`，使用 `null_offset_` 作为后备

---

## JsonIndexBuilder 实现

**文件**：`JsonIndexBuilder.cpp`

### 核心功能

1. **ProcessJsonFieldData 模板函数**：
```cpp
template <typename T>
void ProcessJsonFieldData(
    const std::vector<std::shared_ptr<FieldDataBase>>& field_datas,
    const proto::schema::FieldSchema& schema,
    const std::string& nested_path,
    const JsonCastType& cast_type,
    JsonCastFunction cast_function,
    JsonDataAdder<T> data_adder,
    JsonNullAdder null_adder,
    JsonNonExistAdder non_exist_adder,
    JsonErrorRecorder error_recorder) {

    int64_t offset = 0;
    auto tokens = parse_json_pointer(nested_path);
    bool is_array = cast_type.data_type() == JsonCastType::DataType::ARRAY;

    for (const auto& data : field_datas) {
        auto n = data->get_num_rows();
        for (int64_t i = 0; i < n; i++) {
            auto json_column = static_cast<const Json*>(data->RawValue(i));

            // 处理 NULL
            if (schema.nullable() && !data->is_valid(i)) {
                non_exist_adder(offset);
                null_adder(offset);
                data_adder(nullptr, 0, offset++);
                continue;
            }

            // 检查路径存在性
            auto exists = path_exists(json_column->dom_doc(), tokens);
            if (!exists || !json_column->exist(nested_path)) {
                error_recorder(*json_column, nested_path, simdjson::NO_SUCH_FIELD);
                non_exist_adder(offset);
                data_adder(nullptr, 0, offset++);
                continue;
            }

            // 提取值
            folly::fbvector<T> values;
            if (is_array) {
                // 处理数组类型
                auto doc = json_column->dom_doc();
                auto array_res = doc.at_pointer(nested_path).get_array();
                if (array_res.error() == simdjson::SUCCESS) {
                    auto array_values = array_res.value();
                    for (auto value : array_values) {
                        auto val = value.template get<SIMDJSON_T>();
                        if (val.error() == simdjson::SUCCESS) {
                            values.push_back(static_cast<T>(val.value()));
                        }
                    }
                }
            } else {
                // 处理标量类型
                if (cast_function.match<T>()) {
                    auto res = JsonCastFunction::CastJsonValue<T>(
                        cast_function, *json_column, nested_path);
                    if (res.has_value()) {
                        values.push_back(res.value());
                    }
                } else {
                    value_result<SIMDJSON_T> res =
                        json_column->at<SIMDJSON_T>(nested_path);
                    if (res.error() == simdjson::SUCCESS) {
                        values.push_back(static_cast<T>(res.value()));
                    }
                }
            }

            data_adder(values.data(), values.size(), offset++);
        }
    }
}
```

2. **类型转换支持**：
   - 使用 `JsonCastType` 指定目标类型
   - 使用 `JsonCastFunction` 进行类型转换
   - 支持自定义转换函数

3. **错误处理**：
   - 记录 JSON 解析错误
   - 记录路径不存在错误
   - 使用 `error_recorder` 回调函数

---

## 关键技术点

### 1. simdjson 库使用

- **高性能 JSON 解析**：使用 simdjson 库进行快速 JSON 解析
- **DOM 模式**：使用 `dom_doc()` 获取 DOM 文档
- **JSON Pointer**：使用 `at_pointer()` 访问嵌套路径

### 2. 内存优化

- **Scratch Buffer**：复用缓冲区避免重复分配
- **folly::fbvector**：使用 Facebook 的高性能向量容器
- **Padded String**：使用 padded string 优化 SIMD 操作

### 3. NULL 和不存在值处理

- **null_offset_**：记录 NULL 值的偏移量
- **non_exist_offsets_**：记录路径不存在的偏移量
- **区分 NULL 和不存在**：v2.6+ 区分这两种情况

### 4. 向后兼容性

- **v2.5.x 兼容**：如果没有 `non_exist_offset_file`，使用 `null_offset_` 作为后备
- **Exists() 行为**：v2.6+ 的 Exists() 等价于 v2.5.x 的 IsNotNull()

---

## 性能特性

### 1. 100x 性能提升（2026 核心特性）

- **JSON Path Index**：通过倒排索引实现快速查询
- **嵌套 JSON 查询优化**：避免全表扫描
- **索引类型选择**：JsonFlatIndex vs JsonInvertedIndex

### 2. 索引构建优化

- **批量处理**：批量处理 field_datas
- **并行化**：支持多线程索引构建
- **增量更新**：支持增量索引更新

---

## 使用示例（从测试文件）

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

# 支持的 JSON Path 格式
# 1. 嵌套访问：json_field['a']['b']
# 2. 数组访问：json_field['a'][0]
# 3. 混合访问：json_field['a'][0]['b']
# 4. 根路径：json_field
```

---

## 总结

1. **JsonFlatIndex**：适用于简单查询，内存占用小
2. **JsonInvertedIndex**：适用于复杂查询，性能更高
3. **simdjson**：提供高性能 JSON 解析
4. **类型转换**：支持多种数据类型转换
5. **错误处理**：完善的错误记录和处理机制
6. **向后兼容**：支持 v2.5.x 数据格式
