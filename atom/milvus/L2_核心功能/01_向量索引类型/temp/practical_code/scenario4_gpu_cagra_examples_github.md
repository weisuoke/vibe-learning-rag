# Scenario 4: GPU CAGRA Examples from GitHub

## Search Results

### 1. Milvus 主仓库 README
**URL**: https://github.com/milvus-io/milvus/blob/master/README.md
**Description**: Milvus 官方仓库文档，介绍支持 NVIDIA CAGRA 的 GPU 索引加速，用于高性能向量搜索，包括硬件加速细节。

### 2. Milvus GPU CAGRA 索引不支持错误讨论
**URL**: https://github.com/milvus-io/milvus/issues/34220
**Description**: 讨论 Milvus 2.4.5-gpu 版本中使用 GPU_CAGRA 索引时出现的 'index not supported' 错误及相关配置建议。

### 3. 无法构建 GPU_CAGRA 索引问题
**URL**: https://github.com/milvus-io/milvus/issues/38650
**Description**: Milvus 2.5.0-beta GPU 版本中构建 GPU_CAGRA 索引失败的 bug 报告，包含环境细节和排查信息。

### 4. Milvus SDK Go 版本发布记录
**URL**: https://github.com/milvus-io/milvus-sdk-go/releases
**Description**: Milvus Go SDK 发布日志，支持 GPUCagra 和 GPUBruteForce 索引类型，包含相关 PR 和 GPU 索引增强。

### 5. Milvus 使用 GPU_CAGRA 构建索引但搜索使用 CPU 的 bug
**URL**: https://github.com/milvus-io/milvus/issues/38986
**Description**: 报告 Milvus standalone GPU 模式下索引使用 GPU 但搜索 fallback 到 CPU 的问题，讨论性能瓶颈。

### 6. 建议使用 CAGRA 作为 milvus-gpu 默认 HNSW 构建
**URL**: https://github.com/milvus-io/milvus/issues/39615
**Description**: 功能请求讨论在 milvus-gpu 包中默认使用 CAGRA 构建索引，类似于 HNSW 的 GPU 加速方案。

### 7. BigVectorBench 基准测试仓库
**URL**: https://github.com/BenchCouncil/BigVectorBench
**Description**: 向量数据库性能基准套件，支持 milvus-gpu-cagra 等 GPU 算法，包括运行示例和 Milvus GPU CAGRA 测试。

### 8. GPU 图 ANN 基准测试仓库
**URL**: https://github.com/schencoding/gpu-graph-anns
**Description**: GPU 图加速近似最近邻搜索基准，包含 cuVS CAGRA 算法代码示例，支持多种语言实现。

## Key Insights

### GPU CAGRA Support
- **Milvus Version**: Supported since Milvus 2.4+
- **GPU Requirements**: NVIDIA GPU with CUDA support
- **Index Type**: GPU_CAGRA (graph-based index)
- **Performance**: Outperforms HNSW, especially for small batch sizes

### Common Issues
1. **Index Not Supported Error**: Requires milvus-gpu Docker image
2. **Build Failures**: Check CUDA version and GPU memory
3. **CPU Fallback**: Search may fallback to CPU if GPU resources insufficient
4. **Configuration**: Requires proper GPU device configuration

### Benchmarks
- **BigVectorBench**: Comprehensive GPU CAGRA benchmarks
- **Performance**: Significant speedup over CPU indexes
- **Scalability**: Handles large-scale datasets (50M+ vectors)
