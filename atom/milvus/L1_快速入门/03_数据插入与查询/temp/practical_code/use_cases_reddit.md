---
source: Grok-mcp web search results
query: Milvus 2.6 Embedding Functions best practices 2026
platform: Reddit
fetched_at: 2026-02-21
---

# Reddit Discussions: Milvus 2.6 Embedding Functions Best Practices

## Search Results

### 1. Best practice for creating embeddings
**URL**: https://www.reddit.com/r/vectordatabase/comments/15rxk5d/best_practice_for_creating_embeddings
**Description**: Discussion on documented best practices for text embeddings, including optimal chunk sizes, removing new lines, and other preprocessing techniques suitable for Milvus integration.

### 2. How to Choose the Right Embedding Model for RAG - Milvus Blog
**URL**: https://www.reddit.com/r/LLMDevs/comments/1qg6hnh/how_to_choose_the_right_embedding_model_for_rag
**Description**: Guide to selecting appropriate embedding models for Retrieval-Augmented Generation systems using Milvus, with user discussions on LLM and embedding combinations.

### 3. Milvus Vector database
**URL**: https://www.reddit.com/r/LangChain/comments/1nkxjd1/milvus_vector_database
**Description**: Advice on avoiding general LLMs as embedders in Milvus; emphasizes that high parameter counts do not ensure effective embedding performance.

### 4. Milvus - Updating the Embeddings
**URL**: https://www.reddit.com/r/vectordatabase/comments/1dmnfob/milvus_updating_the_embeddings
**Description**: Exploration of methods to update embeddings in Milvus vector stores, including best practices and alternatives for dynamic data applications.

### 5. Slashed My RAG Startup Costs 75% with Milvus RaBitQ + SQ8 Quantization!
**URL**: https://www.reddit.com/r/Rag/comments/1pvrpzx/slashed_my_rag_startup_costs_75_with_milvus
**Description**: Practical approach to quantizing embeddings in Milvus using RaBitQ and SQ8 to significantly reduce memory usage and costs in RAG setups.

### 6. Milvus and RAG-system
**URL**: https://www.reddit.com/r/Rag/comments/1lkgplk/milvus_and_ragsystem
**Description**: Community insights on document processing for Milvus-based RAG systems, covering chunking strategies, embedding models, and supporting technologies.

### 7. Best Practices for Semantic Search on 200k vectors
**URL**: https://www.reddit.com/r/learnmachinelearning/comments/1acxy85/best_practices_for_semantic_search_on_200k
**Description**: Recommendations for managing large-scale embeddings in semantic search, including optimization techniques applicable to Milvus deployments.

## Key Community Insights

### Embedding Model Selection
- **Don't use general LLMs as embedders**: High parameter count ≠ good embedding quality
- **Specialized embedding models**: Use models designed for embeddings (e.g., BGE, Sentence Transformers)
- **Model size considerations**: Smaller specialized models often outperform larger general models
- **Domain-specific models**: Choose models trained on relevant domains

### Chunking Strategies
- **Optimal chunk size**: 256-512 tokens for most use cases
- **Overlap**: 10-20% overlap between chunks
- **Semantic boundaries**: Split at paragraph or sentence boundaries
- **Document structure**: Preserve document hierarchy when possible

### Cost Optimization
- **Quantization**: RaBitQ + SQ8 can reduce costs by 75%
- **Memory management**: Use quantization for large-scale deployments
- **Batch processing**: Process embeddings in batches to reduce API costs
- **Caching**: Cache frequently used embeddings

### Updating Embeddings
- **Upsert operation**: Use upsert instead of delete + insert
- **Incremental updates**: Update only changed documents
- **Version control**: Track embedding model versions
- **Reindexing strategy**: Plan for full reindexing when changing models

### RAG System Design
- **Hybrid search**: Combine dense and sparse embeddings
- **Reranking**: Add reranking step for better results
- **Metadata filtering**: Use scalar fields for pre-filtering
- **Multi-vector search**: Use multiple embedding types for different aspects

### Common Pitfalls
- Using general LLMs as embedding models
- Not considering chunk size impact on retrieval quality
- Ignoring quantization for large-scale deployments
- Not planning for embedding updates
- Overlooking metadata filtering opportunities
