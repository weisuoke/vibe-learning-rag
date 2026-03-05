# SCANN Index Documentation

## Official Milvus Documentation
- **URL**: https://milvus.io/docs/scann.md
- **Description**: Google ScaNN library support for large-scale vector similarity search

## Key Features
- **Score-aware quantization**: Improves IVFPQ with better accuracy
- **4-bit FastScan**: Fast distance computation
- **Balanced performance**: Speed vs accuracy trade-off
- **Google technology**: Based on ScaNN research

## Parameters
- **nlist**: Number of clusters (like IVF)
- **with_raw_data**: Store original vectors for refinement

## Performance
- **QPS**: Significantly higher than IVF_FLAT and IVF_PQ on Cohere1M dataset
- **Accuracy**: Better than IVF_PQ due to score-aware quantization
- **Memory**: Similar to IVF_PQ

## References
1. SCANN Documentation: https://milvus.io/docs/scann.md
2. ScaNN Introduction Blog: https://milvus.io/blog/a-brief-introduction-to-the-scann-index.md
