# Scenario 2: IVF Index Best Practices and Parameter Tuning

## Search Results

### 1. Milvus IVF参数调优指南：nlist与nprobe最佳实践
**URL**: https://milvus.io/ai-quick-reference/how-can-the-parameters-of-an-ivf-index-like-the-number-of-clusters-nlist-and-the-number-of-probes-nprobe-be-tuned-to-achieve-a-target-recall-at-the-fastest-possible-query-speed
**Description**: 固定nlist在1000-10000范围，根据数据集大小逐步调整nprobe，通过测量recall和查询速度找到最佳平衡点，实现目标召回率下最快查询。

### 2. Milvus IVF_FLAT索引文档
**URL**: https://milvus.io/docs/ivf-flat.md
**Description**: IVF_FLAT将向量分为nlist个簇，nprobe决定搜索时考虑的簇数。nlist影响索引构建时间和精度，nprobe平衡速度与召回率，推荐根据场景调参。

### 3. Milvus索引参数选择最佳实践
**URL**: https://milvus.io/docs/performance_faq.md
**Description**: nlist推荐设置为4×√n（n为segment实体数），nprobe需通过实验权衡精度与性能，sift50m数据集测试显示不同nlist/nprobe组合的recall与QPS。

### 4. Milvus IVF索引参数选择与调优
**URL**: https://medium.com/vector-database/best-practices-for-setting-parameters-in-milvus-clients-9b8a8984d3dd
**Description**: nlist建议4*sqrt(n)，nprobe通过试错确定。nlist=4096且nprobe=128时搜索性能最佳，增大nlist减少桶内向量，nprobe增大提升精度但降低效率。

### 5. Milvus In-memory Index文档
**URL**: https://milvus.io/docs/index.md
**Description**: IVF_FLAT索引参数：nlist范围1-65536，默认128；nprobe范围1-nlist，默认8。调整nprobe可找到速度与精度的理想平衡点。

### 6. Milvus IVF_PQ索引文档
**URL**: https://milvus.io/docs/ivf-pq.md
**Description**: nprobe推荐与nlist成比例设置，范围[1, nlist]，较高值提升召回但增加查询延迟，建议根据需求实验调优。

### 7. Milvus IVF索引工作原理与HNSW对比
**URL**: https://milvus.io/blog/understanding-ivf-vector-index-how-It-works-and-when-to-choose-it-over-hnsw.md
**Description**: nlist决定簇的精细度，nprobe控制查询范围。调优nprobe时从小值开始实验，观察recall与延迟变化，适合对延迟不极致敏感的场景。

### 8. Milvus IVF索引聚类与参数调优
**URL**: https://mbrenndoerfer.com/writing/ivf-index-clustering-vector-search-partitioning
**Description**: nlist常用√n到4√n或16√n范围，平衡簇内扫描成本与质心比较开销；nprobe是精度-速度主要旋钮，高维空间需较高值以补偿边界效应。

## Parameter Tuning Best Practices

### nlist (Number of Clusters)
- **Formula**: 4 × √n (n = number of vectors in segment)
- **Range**: 1-65536
- **Default**: 128
- **Recommendations**:
  - Small datasets (< 1M): 1000-4096
  - Medium datasets (1M-10M): 4096-16384
  - Large datasets (> 10M): 16384-65536
  - Higher nlist = more clusters = faster search but longer build time

### nprobe (Number of Probes)
- **Range**: 1 to nlist
- **Default**: 8
- **Recommendations**:
  - Start with low values (8-16) and increase gradually
  - Typical production values: 32-128
  - Higher nprobe = better recall but slower search
  - Optimal: nprobe = nlist/8 to nlist/4 for balanced performance

### Tuning Strategy
1. Fix nlist based on dataset size (4√n)
2. Start with low nprobe (8-16)
3. Gradually increase nprobe while measuring recall and QPS
4. Find sweet spot where recall meets requirements with acceptable latency
5. Test with representative queries and data distribution

### Performance Benchmarks
- **nlist=4096, nprobe=128**: Best search performance for medium datasets
- **Recall vs Speed**: Linear trade-off controlled by nprobe
- **High-dimensional data**: May need higher nprobe to compensate for boundary effects
