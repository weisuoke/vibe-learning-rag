# RaBitQ Quantization Documentation

## Official Milvus Documentation

### IVF_RABITQ Index
- **URL**: https://milvus.io/docs/ivf-rabitq.md
- **Description**: Official Milvus documentation on IVF_RABITQ, a binary quantization index using RaBitQ for quantizing FP32 vectors to binary, optimizing memory in vector search scenarios.

### Release Notes with RaBitQ
- **URL**: https://milvus.io/docs/release_notes.md
- **Description**: Milvus 2.6.11 release notes highlighting RaBitQ 1-bit quantization for improved resource utilization, memory reduction, and search performance in 2026 updates.

## Key Features

### Memory Reduction
- **72% memory savings** compared to traditional indexes
- **4x performance boost** in query speed
- **<2% accuracy loss** - maintains high recall with minimal degradation

### Technical Implementation
- **1-bit quantization** with residual compensation
- Compresses vector indexes to **1/32 of original size**
- Supports multiple refinement options: SQ4/SQ6/SQ8

## Blog Articles

### Milvus 2.6 Preview
- **URL**: https://milvus.io/blog/milvus-26-preview-72-memory-reduction-without-compromising-recall-and-4x-faster-than-elasticsearch.md
- **Key Points**:
  - Optimal balance between memory usage and search quality
  - Primary index with RaBitQ quantization
  - Multiple refine options for different use cases

### Extreme Vector Compression
- **URL**: https://milvus.io/blog/bring-vector-compression-to-the-extreme-how-milvus-serves-3%C3%97-more-queries-with-rabitq.md
- **Key Points**:
  - Enables serving **3× more traffic** with lower memory cost
  - Novel technique for billion-scale vector search
  - Maintains search quality while reducing costs

### Introducing Milvus 2.6
- **URL**: https://milvus.io/blog/introduce-milvus-2-6-built-for-scale-designed-to-reduce-costs.md
- **Key Points**:
  - Built for billion-scale vector search
  - Traditional quantization forces trade-offs in search quality
  - RaBitQ breaks this limitation with minimal accuracy loss

## GitHub Resources

### RaBitQ-Library
- **URL**: https://github.com/VectorDB-NTU/RaBitQ-Library
- **Description**: Lightweight library implementing RaBitQ quantization, integrated with Milvus for memory-efficient vector search, supporting 4-7 bit quantization for high recall.

### Original RaBitQ Research
- **URL**: https://github.com/gaoj0017/RaBitQ
- **Description**: Implementation of RaBitQ from SIGMOD 2024, providing theoretical error bounds for high-dimensional vector quantization, enhancing memory optimization in ANN search including Milvus compatibility.

## Production Use Cases

### Optimizing Milvus Standalone
- **URL**: https://dev.to/ashutosh_kumar_e87e14143e/optimizing-milvus-standalone-for-production-achieving-72-memory-reduction-while-maintaining-1d6b
- **Key Points**:
  - Practical path for production deployments
  - Balance performance, accuracy, and resource usage
  - Real-world optimization strategies

### Upgrade Guide
- **URL**: https://milvus.io/blog/how-to-safely-upgrade-from-milvu-2-5-x-to-milvus-2-6-x.md
- **Key Points**:
  - New 1-bit quantization method in 2.6
  - Compresses vector indexes to 1/32 of original size
  - Safe migration strategies

## Technical Details

### Quantization Mechanism
- **Binary quantization**: Converts FP32 vectors to 1-bit representations
- **Residual compensation**: Maintains accuracy through intelligent refinement
- **Theoretical error bounds**: SIGMOD 2024 research provides mathematical guarantees

### Performance Characteristics
- **Memory**: 72% reduction (1/32 of original size)
- **Speed**: 4x faster queries
- **Accuracy**: <2% recall loss
- **Throughput**: 3× more queries per second

### Use Cases
- Billion-scale vector databases
- Cost-sensitive deployments
- High-throughput RAG systems
- Memory-constrained environments

## Integration with Milvus 2.6

### Index Types
- **IVF_RABITQ**: Primary index type using RaBitQ
- **Refine options**: SQ4, SQ6, SQ8 for different accuracy/speed trade-offs

### Configuration
- Available in Milvus 2.6.11+
- Supports both standalone and distributed deployments
- Compatible with existing Milvus APIs

## Comparison with Other Quantization Methods

### Traditional Quantization
- **SQ8**: 8-bit scalar quantization, 4x compression
- **PQ**: Product quantization, 8-32x compression
- **RaBitQ**: 1-bit quantization, 32x compression

### Advantages
- **Highest compression ratio**: 32x vs 4-8x for traditional methods
- **Minimal accuracy loss**: <2% vs 5-10% for PQ
- **Better performance**: 4x faster than traditional methods
- **Cost-effective**: Enables billion-scale deployments

## References

1. Milvus Official Documentation: https://milvus.io/docs/ivf-rabitq.md
2. RaBitQ Research Paper (SIGMOD 2024): https://github.com/gaoj0017/RaBitQ
3. Milvus 2.6 Release Notes: https://milvus.io/docs/release_notes.md
4. RaBitQ Library: https://github.com/VectorDB-NTU/RaBitQ-Library
