---
type: source_code_analysis
source: sourcecode/milvus
analyzed_files:
  - client/index/sparse.go
  - client/entity/sparse.go
  - internal/util/function/bm25_function.go
  - pkg/util/bm25/bm25.go
analyzed_at: 2026-02-25
knowledge_point: 03_稀疏向量与BM25深入
---

# 源码分析：Milvus 稀疏向量与 BM25 实现

## 分析的文件

### 1. client/index/sparse.go
**功能**：稀疏向量索引类型定义

**关键发现**：
- Milvus 支持两种稀疏向量索引类型：
  - `SPARSE_INVERTED_INDEX`：倒排索引
  - `SPARSE_WAND`：WAND（Weak-AND）算法
- 两种索引都支持 `drop_ratio` 参数（构建时和搜索时）

**代码片段**：
```go
// SPARSE_INVERTED_INDEX 索引
type sparseInvertedIndex struct {
    baseIndex
    dropRatio float64  // 丢弃比例参数
}

func NewSparseInvertedIndex(metricType MetricType, dropRatio float64) Index {
    return sparseInvertedIndex{
        baseIndex: baseIndex{
            metricType: metricType,
            indexType:  SparseInverted,
        },
        dropRatio: dropRatio,
    }
}

// SPARSE_WAND 索引
type sparseWANDIndex struct {
    baseIndex
    dropRatio float64
}

func NewSparseWANDIndex(metricType MetricType, dropRatio float64) Index {
    return sparseWANDIndex{
        baseIndex: baseIndex{
            metricType: metricType,
            indexType:  SparseWAND,
        },
        dropRatio: dropRatio,
    }
}

// 搜索参数
type sparseAnnParam struct {
    baseAnnParam
}

func (b sparseAnnParam) WithDropRatio(dropRatio float64) {
    b.WithExtraParam("drop_ratio_search", dropRatio)
}
```

**关键参数**：
- `drop_ratio_build`：构建索引时的丢弃比例
- `drop_ratio_search`：搜索时的丢弃比例

---

### 2. client/entity/sparse.go
**功能**：稀疏向量数据结构定义

**关键发现**：
- 稀疏向量使用 `positions` 和 `values` 两个数组存储
- 序列化格式：每个元素 8 字节（4 字节位置 + 4 字节值）
- 自动排序：按 position 升序排列
- 维度计算：`dim = max(positions) + 1`

**代码片段**：
```go
// 稀疏向量接口
type SparseEmbedding interface {
    Dim() int                                      // 维度
    Len() int                                      // 实际元素数量
    Get(idx int) (pos uint32, value float32, ok bool)  // 获取元素
    Serialize() []byte                             // 序列化
    FieldType() FieldType
}

// 稀疏向量实现
type sliceSparseEmbedding struct {
    positions []uint32  // 位置数组
    values    []float32 // 值数组
    dim       int       // 维度
    len       int       // 元素数量
}

// 序列化：每个元素 8 字节
func (e sliceSparseEmbedding) Serialize() []byte {
    row := make([]byte, 8*e.Len())
    for idx := 0; idx < e.Len(); idx++ {
        pos, value, _ := e.Get(idx)
        binary.LittleEndian.PutUint32(row[idx*8:], pos)
        binary.LittleEndian.PutUint32(row[idx*8+4:], math.Float32bits(value))
    }
    return row
}

// 创建稀疏向量：自动排序
func NewSliceSparseEmbedding(positions []uint32, values []float32) (SparseEmbedding, error) {
    if len(positions) != len(values) {
        return nil, errors.New("invalid sparse embedding input, positions shall have same number of values")
    }

    se := sliceSparseEmbedding{
        positions: positions,
        values:    values,
        len:       len(positions),
    }

    sort.Sort(se)  // 按 position 排序

    if se.len > 0 {
        se.dim = int(se.positions[se.len-1]) + 1  // 维度 = 最大位置 + 1
    }

    return se, nil
}
```

**数据结构特点**：
- **稀疏存储**：只存储非零元素
- **自动排序**：按位置升序排列
- **高效序列化**：固定 8 字节/元素
- **维度推断**：根据最大位置自动计算

---

### 3. internal/util/function/bm25_function.go
**功能**：BM25 函数实现

**关键发现**：
- BM25 使用 Analyzer 进行分词
- 将文本转换为稀疏向量（map[uint32]float32）
- 使用哈希函数将 token 转换为 uint32
- 支持并发处理（默认 8 个并发）
- 支持多 Analyzer 模式

**代码片段**：
```go
// BM25 函数运行器
type BM25FunctionRunner struct {
    mu          sync.RWMutex
    closed      bool
    tokenizer   analyzer.Analyzer  // 分词器
    schema      *schemapb.FunctionSchema
    outputField *schemapb.FieldSchema
    inputField  *schemapb.FieldSchema
    concurrency int  // 并发数（默认 8）
}

// 核心处理逻辑
func (v *BM25FunctionRunner) run(data []string, dst []map[uint32]float32) error {
    tokenizer, err := v.tokenizer.Clone()
    if err != nil {
        return err
    }
    defer tokenizer.Destroy()

    for i := 0; i < len(data); i++ {
        if len(data[i]) == 0 {
            dst[i] = map[uint32]float32{}
            continue
        }

        if !typeutil.IsUTF8(data[i]) {
            return merr.WrapErrParameterInvalidMsg("string data must be utf8 format: %v", data[i])
        }

        embeddingMap := map[uint32]float32{}
        tokenStream := tokenizer.NewTokenStream(data[i])
        defer tokenStream.Destroy()

        for tokenStream.Advance() {
            token := tokenStream.Token()
            // 使用哈希函数将 token 转换为 uint32
            hash := typeutil.HashString2LessUint32(token)
            embeddingMap[hash] += 1  // 词频统计
        }
        dst[i] = embeddingMap
    }
    return nil
}

// 批量处理：并发执行
func (v *BM25FunctionRunner) BatchRun(inputs ...any) ([]any, error) {
    text, ok := inputs[0].([]string)
    if !ok {
        return nil, errors.New("BM25 function batch input not string list")
    }

    rowNum := len(text)
    embedData := make([]map[uint32]float32, rowNum)
    wg := sync.WaitGroup{}

    errCh := make(chan error, v.concurrency)
    for i, j := 0, 0; i < v.concurrency && j < rowNum; i++ {
        start := j
        end := start + rowNum/v.concurrency
        if i < rowNum%v.concurrency {
            end += 1
        }
        wg.Add(1)
        go func() {
            defer wg.Done()
            err := v.run(text[start:end], embedData[start:end])
            if err != nil {
                errCh <- err
                return
            }
        }()
        j = end
    }

    wg.Wait()
    close(errCh)
    for err := range errCh {
        if err != nil {
            return nil, err
        }
    }

    return []any{buildSparseFloatArray(embedData)}, nil
}
```

**BM25 处理流程**：
1. **分词**：使用 Analyzer 将文本分词
2. **哈希**：将 token 转换为 uint32（使用 `HashString2LessUint32`）
3. **词频统计**：统计每个 token 的出现次数
4. **稀疏向量**：生成 `map[uint32]float32`
5. **并发处理**：默认 8 个并发 goroutine

**关键特性**：
- **UTF-8 验证**：确保输入文本是 UTF-8 编码
- **并发安全**：使用 RWMutex 保护
- **资源管理**：自动销毁 tokenizer 和 tokenStream
- **错误处理**：通过 channel 收集并发错误

---

### 4. pkg/util/bm25/bm25.go
**功能**：BM25 工具函数

**关键发现**：
- 提供稀疏向量字段数据构建函数
- 将 `SparseFloatArray` 转换为 `FieldData`

**代码片段**：
```go
func BuildSparseFieldData(field *schemapb.FieldSchema, sparseArray *schemapb.SparseFloatArray) *schemapb.FieldData {
    return &schemapb.FieldData{
        Type:      field.GetDataType(),
        FieldName: field.GetName(),
        Field: &schemapb.FieldData_Vectors{
            Vectors: &schemapb.VectorField{
                Dim: sparseArray.GetDim(),
                Data: &schemapb.VectorField_SparseFloatVector{
                    SparseFloatVector: sparseArray,
                },
            },
        },
        FieldId: field.GetFieldID(),
    }
}
```

---

## 核心技术总结

### 1. 稀疏向量数据结构
- **存储格式**：`positions[]` + `values[]`
- **序列化**：8 字节/元素（4 字节位置 + 4 字节值）
- **排序**：按位置升序
- **维度**：`max(positions) + 1`

### 2. BM25 实现原理
- **分词**：使用 Analyzer（可配置）
- **哈希**：token → uint32
- **词频统计**：`map[uint32]float32`
- **并发处理**：8 个 goroutine

### 3. 稀疏向量索引
- **SPARSE_INVERTED_INDEX**：倒排索引
- **SPARSE_WAND**：WAND 算法
- **drop_ratio**：控制索引大小和搜索精度

### 4. 性能优化
- **并发处理**：批量数据并发转换
- **资源池**：Analyzer 池化
- **内存优化**：稀疏存储

---

## 应用场景

### 1. 全文搜索
- 使用 BM25 将文本转换为稀疏向量
- 使用倒排索引加速检索

### 2. 混合检索
- 向量检索（语义相似度）
- BM25 检索（关键词匹配）
- 加权融合结果

### 3. 多语言支持
- 配置不同的 Analyzer
- 支持中文、英文等多种语言

---

## 关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `drop_ratio_build` | 构建索引时的丢弃比例 | - |
| `drop_ratio_search` | 搜索时的丢弃比例 | - |
| `analyzer_params` | 分词器参数（JSON） | `{}` |
| `concurrency` | BM25 并发数 | 8 |

---

## 下一步调研方向

1. **WAND 算法原理**：深入理解 WAND 算法的实现
2. **Analyzer 配置**：不同语言的分词器配置
3. **混合检索权重**：如何调整向量检索和 BM25 的权重
4. **性能对比**：SPARSE_INVERTED_INDEX vs SPARSE_WAND
5. **drop_ratio 调优**：如何选择合适的 drop_ratio 值
