# IVF Index Overview - Milvus Official Documentation

Source: https://milvus.io/docs/index.md
Fetched: 2026-02-21

## Key Points

### IVF_FLAT
- **Classification**: N/A
- **Scenario**: High-speed query, Requires a recall rate as high as possible
- **Description**: Divides vector data into `nlist` cluster units, and then compares distances between the target input vector and the center of each cluster. Depending on the number of clusters the system is set to query (`nprobe`), similarity search results are returned based on comparisons between the target input and the vectors in the most similar cluster(s) only.

### IVF_SQ8
- **Classification**: Quantization-based index
- **Scenario**: Very high-speed query, Limited memory resources, Accepts minor compromise in recall rate
- **Description**: Combines IVF clustering with Scalar Quantization (SQ8). Uses 8-bit integers instead of 32-bit floating point numbers to store each dimension value, achieving 4:1 compression ratio.

### IVF_PQ
- **Classification**: Quantization-based index
- **Scenario**: High-speed query, Limited memory resources, Accepts minor compromise in recall rate
- **Description**: Combines IVF clustering with Product Quantization (PQ). Achieves much higher compression ratios (up to 64:1) by decomposing vectors into sub-vectors and using codebooks.

## Memory Comparison

For 128-dimensional float32 vectors:
- **Original**: 128 × 32 bits = 4,096 bits (512 bytes)
- **IVF_FLAT**: Same as original (no compression)
- **IVF_SQ8**: 128 × 8 bits = 1,024 bits (128 bytes) - 4:1 compression
- **IVF_PQ** (m=64, nbits=8): 64 × 8 bits = 512 bits (64 bytes) - 8:1 compression
