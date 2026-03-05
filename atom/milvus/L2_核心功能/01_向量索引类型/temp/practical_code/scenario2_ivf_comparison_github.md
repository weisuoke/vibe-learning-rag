# Scenario 2: IVF Series Index Comparison from GitHub

## Search Results

### 1. zilliztech/VectorDBBench: Benchmark for vector databases
**URL**: https://github.com/zilliztech/VectorDBBench
**Description**: VectorDBBench provides performance and cost-effectiveness comparisons for vector databases, including tests for Milvus index types: IVF_FLAT, IVF_SQ8, and IVF_PQ with specific parameters like nprobes and m.

### 2. chap02_schema.md - milvus
**URL**: https://github.com/milvus-io/milvus/blob/master/docs/developer_guides/chap02_schema.md
**Description**: Describes Milvus indexes: IVF_PQ clusters and quantizes vectors for smaller index files than IVF_SQ8, but with accuracy loss compared to IVF_FLAT.

### 3. How much each index type consume memory?
**URL**: https://github.com/milvus-io/milvus/discussions/36327
**Description**: Discussion on Milvus index memory: IVF_FLAT/HNSW slightly larger than original vectors; IVF_SQ8/IVF_PQ use 25-30% of original size, with IVF_PQ having lower recall.

### 4. search performance can't be improved after trying several indexes
**URL**: https://github.com/milvus-io/milvus/discussions/17976
**Description**: User tested Milvus indexes: IVF_PQ, IVF_SQ8 show similar latency to IVF_FLAT on 4M vectors, discussing potential reasons for unchanged performance.

### 5. dromara/MilvusPlus
**URL**: https://github.com/dromara/MilvusPlus
**Description**: MilvusPlus tool describes indexes: IVF_FLAT for medium datasets, IVF_SQ8 sacrifices accuracy for speed on large datasets, IVF_PQ balances speed and accuracy for high-dimensional data.

### 6. Avoid query timeouts and out-of-memory errors
**URL**: https://github.com/milvus-io/milvus/discussions/37226
**Description**: Explains Milvus memory usage: IVF_SQ8/IVF_PQ require 50-60GB plus data size, compared to other indexes, to prevent timeouts and OOM errors.

### 7. Chapter 2 Milvus 核心概念：数据模型与索引体系
**URL**: https://github.com/datawhalechina/easy-vecdb/blob/main/docs/Milvus/chapter2/Milvus核心概念.md
**Description**: Compares Milvus IVF indexes: Memory usage IVF_PQ < IVF_SQ8 < IVF_FLAT; Recall rate opposite, with IVF_FLAT best, due to increasing quantization.

## Key Comparison Points

### Memory Usage
- **IVF_FLAT**: ~100% of original vector size (slightly larger)
- **IVF_SQ8**: ~25-30% of original vector size
- **IVF_PQ**: ~25-30% of original vector size (smallest)

### Recall Rate
- **IVF_FLAT**: Highest (best accuracy)
- **IVF_SQ8**: Medium
- **IVF_PQ**: Lowest (most accuracy loss)

### Use Cases
- **IVF_FLAT**: Medium datasets (1M-10M), balanced accuracy/speed
- **IVF_SQ8**: Large datasets, sacrifice accuracy for speed
- **IVF_PQ**: High-dimensional data, balance speed and accuracy

### Performance Characteristics
- All three use clustering (nlist parameter)
- Search controlled by nprobe parameter
- IVF_PQ has smallest index file but highest accuracy loss
- IVF_SQ8 uses scalar quantization (8-bit)
- IVF_PQ uses product quantization (more aggressive compression)
