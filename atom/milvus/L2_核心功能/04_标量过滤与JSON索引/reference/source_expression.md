# 表达式执行引擎源码分析

> 来源：sourcecode/milvus/internal/core/src/exec/expression/

---

## CompareExpr 实现

**文件**：`CompareExpr.cpp`

### 核心功能

1. **比较表达式执行**：
```cpp
template <typename OpType>
VectorPtr PhyCompareFilterExpr::ExecCompareExprDispatcher(OpType op, EvalCtx& context) {
    // 接收 offset input
    auto input = context.get_offset_input();
    if (has_offset_input_) {
        auto real_batch_size = input->size();
        if (real_batch_size == 0) {
            return nullptr;
        }

        // 创建结果向量
        auto res_vec = std::make_shared<ColumnVector>(
            TargetBitmap(real_batch_size, false),
            TargetBitmap(real_batch_size, true)
        );
        TargetBitmapView res(res_vec->GetRawData(), real_batch_size);
        TargetBitmapView valid_res(res_vec->GetValidRawData(), real_batch_size);

        // 获取数据边界
        auto left_data_barrier = segment_chunk_reader_.segment_->num_chunk_data(
            expr_->left_field_id_);
        auto right_data_barrier = segment_chunk_reader_.segment_->num_chunk_data(
            expr_->right_field_id_);

        // 处理每个 offset
        for (auto i = 0; i < real_batch_size; ++i) {
            auto offset = (*input)[i];

            // 获取 chunk ID 和 offset
            auto [left_chunk_id, left_chunk_offset] =
                get_chunk_id_and_offset(left_field_, left_data_barrier);
            auto [right_chunk_id, right_chunk_offset] =
                get_chunk_id_and_offset(right_field_, right_data_barrier);

            // 获取数据访问器
            auto left = segment_chunk_reader_.GetChunkDataAccessor(
                expr_->left_data_type_,
                expr_->left_field_id_,
                left_chunk_id,
                left_data_barrier,
                pinned_index_left_);
            auto right = segment_chunk_reader_.GetChunkDataAccessor(
                expr_->right_data_type_,
                expr_->right_field_id_,
                right_chunk_id,
                right_data_barrier,
                pinned_index_right_);

            // 执行比较操作
            // ...
        }
    }
}
```

2. **字符串表达式检测**：
```cpp
bool PhyCompareFilterExpr::IsStringExpr() {
    return expr_->left_data_type_ == DataType::VARCHAR ||
           expr_->right_data_type_ == DataType::VARCHAR;
}
```

3. **批量大小计算**：
```cpp
int64_t PhyCompareFilterExpr::GetNextBatchSize() {
    auto current_rows = GetCurrentRows();

    return current_rows + batch_size_ >= segment_chunk_reader_.active_count_
               ? segment_chunk_reader_.active_count_ - current_rows
               : batch_size_;
}
```

### 关键技术点

1. **Chunked Segment 处理**：
   - 支持 Growing Segment 和 Sealed Segment
   - 根据 segment 类型计算 chunk ID 和 offset
   - 使用 `get_chunk_by_offset()` 获取 chunk 信息

2. **数据访问器**：
   - 使用 `GetChunkDataAccessor()` 获取数据
   - 支持不同数据类型（VARCHAR, 数值类型）
   - 使用 pinned index 优化访问

3. **位图操作**：
   - 使用 `TargetBitmap` 存储结果
   - 使用 `TargetBitmapView` 进行位图操作
   - 支持 valid 位图（处理 NULL 值）

---

## LogicalBinaryExpr 实现

**文件**：`LogicalBinaryExpr.cpp`

### 核心功能

1. **逻辑二元表达式执行**：
```cpp
void PhyLogicalBinaryExpr::Eval(EvalCtx& context, VectorPtr& result) {
    tracer::AutoSpan span("PhyLogicalBinaryExpr::Eval", tracer::GetRootSpan());

    // 确保有两个输入
    AssertInfo(
        inputs_.size() == 2,
        "logical binary expr must have 2 inputs, but {} inputs are provided",
        inputs_.size());

    // 执行左右子表达式
    VectorPtr left;
    inputs_[0]->Eval(context, left);
    VectorPtr right;
    inputs_[1]->Eval(context, right);

    // 获取列向量
    auto lflat = GetColumnVector(left);
    auto rflat = GetColumnVector(right);
    auto size = left->size();

    // 创建位图视图
    TargetBitmapView lview(lflat->GetRawData(), size);
    TargetBitmapView rview(rflat->GetRawData(), size);

    // 执行逻辑操作
    if (expr_->op_type_ == expr::LogicalBinaryExpr::OpType::And) {
        LogicalElementFunc<LogicalOpType::And> func;
        func(lview, rview, size);
    } else if (expr_->op_type_ == expr::LogicalBinaryExpr::OpType::Or) {
        LogicalElementFunc<LogicalOpType::Or> func;
        func(lview, rview, size);
    } else {
        ThrowInfo(OpTypeInvalid,
                  "unsupported logical operator: {}",
                  expr_->GetOpTypeString());
    }

    // 处理 valid 位图
    TargetBitmapView lvalid_view(lflat->GetValidRawData(), size);
    TargetBitmapView rvalid_view(rflat->GetValidRawData(), size);
    LogicalElementFunc<LogicalOpType::Or> func;
    func(lvalid_view, rvalid_view, size);

    result = std::move(left);
}
```

### 关键技术点

1. **递归表达式执行**：
   - 先执行左右子表达式
   - 然后对结果执行逻辑操作
   - 支持表达式树的递归执行

2. **位图逻辑操作**：
   - AND 操作：`lview & rview`
   - OR 操作：`lview | rview`
   - 使用模板函数 `LogicalElementFunc` 实现

3. **Valid 位图处理**：
   - 对 valid 位图执行 OR 操作
   - 确保 NULL 值的正确处理
   - 保持结果的有效性信息

4. **性能优化**：
   - 使用 `std::move` 避免拷贝
   - 使用 `TargetBitmapView` 避免内存分配
   - 使用 tracer 进行性能追踪

---

## 表达式类型总结

从源码文件列表来看，Milvus 支持以下表达式类型：

### 1. 比较表达式
- **CompareExpr**：比较操作符（==, !=, >, <, >=, <=）
- **BinaryRangeExpr**：范围查询（BETWEEN）
- **BinaryArithOpEvalRangeExpr**：算术运算范围查询

### 2. 集合表达式
- **TermExpr**：IN 操作符
- **ExistsExpr**：EXISTS 操作符
- **NullExpr**：IS NULL / IS NOT NULL 操作符

### 3. 逻辑表达式
- **LogicalBinaryExpr**：AND / OR 操作符
- **LogicalUnaryExpr**：NOT 操作符

### 4. 其他表达式
- **AlwaysTrueExpr**：始终为真的表达式
- **CallExpr**：函数调用表达式
- **ColumnExpr**：列引用表达式
- **ConjunctExpr**：合取表达式

---

## 执行流程

### 1. 表达式树构建
```
用户查询字符串
    ↓
解析器（Parser）
    ↓
表达式树（Expression Tree）
    ↓
物理表达式（Physical Expression）
```

### 2. 表达式执行
```
PhyExpr::Eval(context, result)
    ↓
获取输入数据（offset input 或 batch input）
    ↓
执行子表达式（递归）
    ↓
执行当前表达式操作
    ↓
返回结果（TargetBitmap）
```

### 3. 结果合并
```
多个表达式结果
    ↓
位图逻辑操作（AND / OR）
    ↓
最终过滤结果
    ↓
应用到向量检索
```

---

## 性能优化技术

### 1. 批量处理
- 使用 batch_size 控制批量大小
- 减少函数调用开销
- 提高缓存命中率

### 2. 位图操作
- 使用 TargetBitmap 存储结果
- 使用 SIMD 指令加速位图操作
- 减少内存占用

### 3. Chunked Segment
- 支持分块存储
- 减少内存占用
- 支持增量更新

### 4. 数据访问优化
- 使用 pinned index 缓存数据访问器
- 减少重复的数据访问
- 提高数据访问效率

---

## 使用示例（从测试文件）

```python
# 比较表达式
"age > 25"
"city == '北京'"

# 范围表达式
"price >= 100 AND price <= 500"

# 集合表达式
"color IN ['red', 'blue', 'green']"

# NULL 表达式
"description IS NULL"
"tags IS NOT NULL"

# 逻辑表达式
"age > 25 AND (city == '北京' OR city == '上海')"
"NOT (status == 'deleted')"

# 复合表达式
"age > 25 AND city IN ['北京', '上海'] AND tags IS NOT NULL"
```

---

## 总结

1. **CompareExpr**：实现比较操作符，支持 chunked segment
2. **LogicalBinaryExpr**：实现逻辑操作符，支持表达式树递归执行
3. **位图操作**：使用 TargetBitmap 和 TargetBitmapView 优化性能
4. **批量处理**：使用 batch_size 控制批量大小
5. **性能优化**：使用 pinned index、SIMD 指令、chunked segment 等技术
