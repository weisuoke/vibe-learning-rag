---
source: https://github.com/milvus-io/milvus/issues/40451
title: [Enhancement]: New Search Architecture Based On Streaming Service
fetched_at: 2026-02-21
status: open
author: chyezh
milestone: 3.0
labels: feature/streaming node, kind/enhancement
---

# [Enhancement]: New Search Architecture Based On Streaming Service

**Assignees**: None

**Labels**:
- feature/streaming node
- kind/enhancement

**Milestone**: [3.0](https://github.com/milvus-io/milvus/milestone/63)

### Is there an existing issue for this?
- I have searched the existing issues

### What would you like to be added?
New Search Architecture Based On Streaming Service

### Why is this needed?

Current Streaming Service supports a embedded querynode to implement search/query, also see [#38399](https://github.com/milvus-io/milvus/issues/38399).

But old delegator logic is too heavy for streaming node, and we cannot split the batch and streaming process based on current delegator arch completely:
- Cannot merge the flush process and search built process, so there're always double consuming from wal when recovery, double memory usage if collection is loaded.
- Cannot put all meta management of growing data on streaming node and make a light weight history meta coordinator.
- Cannot remove the forwarding RPC of the delegator to reduce streamingnode's work.

Here's the new distributed architecture for search/query process based on streaming service:

The query process is implemented as shown in the diagram above, following a two-phase query approach:

Coordinator will generate global versioned query view and sync the view to all streaming node and query node, and keep consistency by a cross-node state machine.

QueryNode will subscribe the pure delete stream from the streaming node and apply the delete request to all segment on it.

**Phase 1:** The Proxy generates a shard-level query plan using the highest version of the QueryView from the StreamingNode:
- Includes MVCC.
- Query optimization (BM25, segment filtering, etc.).
- Query view versioning.

**Phase 2:** The Proxy sends the query plan to both the StreamingNode and QueryNode:
- The StreamingNode and QueryNode execute all query operations on their respective segments based on the specified view version (similar to the current SearchSegments process, but using version numbers instead of a segment list).

**Final Step:** The Proxy reduces all results and returns them to the user.

During this process, if a node crashes or the view becomes invalid, the process is canceled and the query operation is retried.

Here's the **TODO list**:
- Versioned Data View to keep a view of historical data on a shard.
- Versioned Query View to keep a distributed loaded info of a loaded shard.
- [enhance: add query view proto, interface, event and utilities #40467](https://github.com/milvus-io/milvus/pull/40467)
- [enhance: add query view implementation of a shard at coord side #40518](https://github.com/milvus-io/milvus/pull/40518)
- [enhance: add grpc sycner client and coord syncer to make sync operation of qview #40521](https://github.com/milvus-io/milvus/pull/40521)
- New balancer to generate query view automatically by current cluster info.
- Cross node state machine to keep consistency query view between streaming node, query node and coord.
- Pure delete stream subscription start from any checkpoint.
- Segment loader scheduler on query node to act with query view.
- Client of new search architecture implementation for proxy.
- Server of new search architecture implementation for qn and sn.

### Anything else?
*No response*

---

**相关链接 / Related PRs mentioned**:
- https://github.com/milvus-io/milvus/issues/38399
- https://github.com/milvus-io/milvus/pull/40467
- https://github.com/milvus-io/milvus/pull/40518
- https://github.com/milvus-io/milvus/pull/40521

**Key Insights for Consistency Implementation**:

1. **Cross-node State Machine**: The architecture uses a cross-node state machine to maintain consistency between streaming node, query node, and coordinator. This is crucial for implementing different consistency levels.

2. **Versioned Query View**: The system maintains versioned query views that are synchronized across all nodes. This versioning mechanism is the foundation for implementing bounded and strong consistency.

3. **MVCC Support**: The architecture includes MVCC (Multi-Version Concurrency Control) in Phase 1, which allows for consistent reads at specific timestamps.

4. **Coordinator-driven Synchronization**: The coordinator generates global versioned query views and syncs them to all nodes, ensuring a consistent view of data across the cluster.

5. **Retry Mechanism**: If a node crashes or the view becomes invalid, the query is retried, which helps maintain consistency guarantees even during failures.
