---
source: Multiple GitHub repositories
title: Milvus Consistency Level in RAG Applications
fetched_at: 2026-02-21
---

# Milvus Consistency Level in RAG Applications

## 1. RAG with LlamaIndex

**Source**: https://github.com/milvus-io/bootcamp/blob/master/integration/llamaindex/rag_with_milvus_and_llamaindex.ipynb

### Key Insights

- When creating collections, you can specify `consistency_level` parameter
- Default consistency level is **Session**
- Session level ensures that writes within the same client connection are immediately visible to subsequent reads

### Example Usage

```python
from pymilvus import Collection

collection = Collection(
    name="llamaindex_rag",
    schema=schema,
    consistency_level="Session"  # Default for interactive RAG applications
)
```

## 2. Milvus Cheat Sheet - Consistency Recommendations

**Source**: https://github.com/milvus-io/bootcamp/blob/master/bootcamp/MilvusCheatSheet.md

### Four Consistency Levels Explained

1. **Strong**: All writes before request are visible
   - Highest latency
   - Use for: Financial data, critical operations

2. **Bounded**: Writes within staleness bound are visible
   - Default level
   - Good balance of consistency and performance
   - Use for: Most production scenarios

3. **Session**: Writes in same session are visible
   - Low overhead
   - Use for: Interactive applications, write-then-read patterns

4. **Eventually**: No guarantee on write visibility
   - Lowest latency, highest throughput
   - **Recommended for RAG scenarios** to get best performance
   - Use for: Analytics, non-critical reads

### RAG-Specific Recommendation

> **For typical usage (e.g. tables updated every few hours or daily), use "Eventually" consistency level for best performance.**

The cheat sheet explicitly recommends **Eventually** consistency for RAG applications because:
- RAG workloads are typically read-heavy
- Document updates are infrequent (batch updates)
- Slight staleness is acceptable for most queries
- Performance gains are significant

## 3. Build RAG with Milvus and Ollama

**Source**: https://github.com/milvus-io/bootcamp/blob/master/integration/build_RAG_with_milvus_and_ollama.ipynb

### Key Insights

- Demonstrates consistency level usage during data insertion
- Shows how to configure consistency level for local LLM integration
- Emphasizes the trade-off between consistency and search latency

## 4. Similarity Search Returning Null Result

**Source**: https://github.com/milvus-io/milvus/discussions/33809

### Problem

In NodeJS + Langchain + RAG application, similarity search returns empty results immediately after insertion.

### Root Cause

Consistency level controls data visibility for search/query operations:
- With **Eventually** consistency, newly inserted data may not be immediately visible
- The search was executed before the data was fully synchronized

### Solution

1. Use **Session** consistency for write-then-immediately-read patterns
2. Or use **Strong** consistency if immediate visibility is critical
3. Or add a small delay between insert and search when using Eventually

### Code Example

```javascript
// Problem: Empty results with Eventually consistency
await collection.insert(data);
const results = await collection.search(query); // May return empty

// Solution 1: Use Session consistency
const collection = new Collection({
    name: "rag_docs",
    consistency_level: "Session"
});

// Solution 2: Add delay with Eventually
await collection.insert(data);
await sleep(100); // Wait for synchronization
const results = await collection.search(query);
```

## 5. ReadTheDocs Zilliz LangChain RAG

**Source**: https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/readthedocs_zilliz_langchain.ipynb

### Key Insights

- Explains relationship between consistency level and CAP theorem
- Discusses impact on search latency
- Demonstrates consistency level configuration in LangChain integration

### CAP Theorem Connection

- **Strong** consistency: Prioritizes Consistency (C) in CAP
  - Higher latency due to synchronization
  - Suitable for critical data

- **Eventually** consistency: Prioritizes Availability (A) and Partition tolerance (P)
  - Lower latency
  - Suitable for RAG where slight staleness is acceptable

## Summary: Consistency Level Selection for RAG

| RAG Scenario | Recommended Level | Reason |
|--------------|-------------------|--------|
| **Batch document ingestion** | Eventually | Documents updated infrequently, performance critical |
| **Interactive Q&A** | Session | User expects to see their own updates immediately |
| **Real-time knowledge base** | Bounded | Balance between freshness and performance |
| **Financial/Legal RAG** | Strong | Data accuracy is critical |
| **High-throughput analytics** | Eventually | Maximum performance, staleness acceptable |

## Best Practices

1. **Default to Eventually for RAG**: Most RAG applications benefit from Eventually consistency
2. **Use Session for interactive apps**: When users write and immediately query
3. **Consider update frequency**: Infrequent updates favor Eventually
4. **Measure latency impact**: Test different levels to find the right balance
5. **Document your choice**: Make consistency level explicit in code comments
