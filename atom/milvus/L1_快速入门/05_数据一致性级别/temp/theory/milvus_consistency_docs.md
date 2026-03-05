---
source: https://milvus.io/docs/consistency.md
title: Consistency Level - Milvus Documentation
fetched_at: 2026-02-21
note: 原 consistency.html 已 404，现内容基于最新 consistency.md 页面完整提取
---

# Consistency Level

As a distributed vector database, Milvus offers multiple levels of consistency to ensure that each node or replica can access the same data during read and write operations. Currently, the supported levels of consistency include **Strong**, **Bounded**, **Eventually**, and **Session**, with **Bounded** being the default level of consistency used.

## Consistency Levels Illustrated

Milvus provides four types of consistency levels with different **GuaranteeTs**.

### Strong

The latest timestamp is used as the GuaranteeTs, and QueryNodes have to wait until the ServiceTime meets the GuaranteeTs before executing Search requests.

- **Guarantee**: All write operations completed before the search/query request are visible.
- **Performance impact**: Highest latency, strongest consistency.
- **Use case**: Scenarios requiring strict data freshness (e.g. financial, real-time inventory).

### Bounded

A user-specified time duration is used as the staleness bound. The default is **Bounded** with staleness bound of **0 seconds** (equivalent to strong in most cases but allows better performance).

- **Guarantee**: All write operations completed within the staleness bound before the search/query request are visible.
- **Default staleness bound**: 0s (可配置，通常几十到几百毫秒)
- **Performance**: Good balance between consistency and low latency.
- **Use case**: Most common production scenarios.

### Session

All write operations within the same session (same client connection) before the search/query request are visible.

- **Guarantee**: Session-level monotonic read — what you wrote in this session, you can read back.
- **Performance**: Very low overhead.
- **Use case**: Interactive applications, write-then-immediately-read patterns within one client.

### Eventually

No guarantee on the visibility of previous write operations.

- **Guarantee**: None — eventually consistent.
- **Performance**: Lowest latency, highest throughput.
- **Use case**: Analytics, non-critical read operations, maximum performance needed.

## Set Consistency Level upon Creating Collection

You can specify the default consistency level when creating a collection.

**Python**

```python
from pymilvus import Collection, utility

collection = Collection(
    name="book",
    schema=schema,
    consistency_level="Bounded"          # 可选: "Strong", "Bounded", "Session", "Eventually"
)
```

**Java (v2)**

```java
import io.milvus.v2.common.ConsistencyLevel;

CreateCollectionReq req = CreateCollectionReq.builder()
    .collectionName("book")
    .schema(schema)
    .consistencyLevel(ConsistencyLevel.Bounded)
    .build();
client.createCollection(req);
```

## Specify Consistency Level for Search/Query

You can override the collection-level consistency when performing search or query.

**Python example**

```python
res = collection.search(
    data=[query_vector],
    anns_field="embeddings",
    param=search_params,
    limit=10,
    consistency_level="Strong"   # 覆盖集合默认级别
)
```

**Java example**

```java
SearchReq searchReq = SearchReq.builder()
    .collectionName("book")
    .consistencyLevel(ConsistencyLevel.Strong)
    // ... other params
    .build();
```

## Consistency Guarantees Summary Table

| Consistency Level | GuaranteeTs Strategy                  | Read-your-writes (same session) | Monotonic reads | Visibility of previous writes | Latency Impact | Default |
|-------------------|----------------------------------------|----------------------------------|------------------|--------------------------------|----------------|---------|
| **Strong**        | Latest timestamp                      | Yes                              | Yes              | All writes before request      | Highest        | No      |
| **Bounded**       | Latest - staleness bound              | Usually                          | Usually          | Within bound                   | Medium         | **Yes** |
| **Session**       | Session-aware (max ts in session)     | Yes                              | Yes (in session) | Same session writes            | Low            | No      |
| **Eventually**    | No guarantee                          | No                               | No               | Eventually                     | Lowest         | No      |

> **Note**
> - The default consistency level is **Bounded** (with 0s staleness in most recent versions).
> - For single-replica deployment, Strong and Bounded usually behave similarly.
> - Consistency level only affects **search** and **query** operations. Insert/delete/update operations are always strongly consistent on the write path.

## Related Topics

- [Manage Collections → Consistency Level](/docs/manage-collections.md#Set-Consistency-Level)
- [Search & Query Parameters](/docs/search.md)
- [Data Consistency in Distributed Systems](https://en.wikipedia.org/wiki/Consistency_model) (external reference)
