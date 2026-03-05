# Worker Layer Sources

## Fetched: 2026-02-21

### Source 1: Milvus System Overview Developer Guide
**URL**: https://github.com/milvus-io/milvus/blob/master/docs/developer_guides/chap01_system_overview.md
**Description**: Detailed overview of Milvus architecture, including query nodes that manage indexes in memory and persist to disk for efficiency.

### Source 2: Milvus Main Repository
**URL**: https://github.com/milvus-io/milvus
**Description**: High-performance vector database with distributed, K8s-native architecture separating compute and storage for scalable queries.

### Source 3: Discussion on MMAP Structure
**URL**: https://github.com/milvus-io/milvus/discussions/33621
**Description**: Explains index building by index node and loading by query node, including file handling and memory mapping processes.

### Source 4: Bug Report on Milvus 2.6.0 Performance
**URL**: https://github.com/milvus-io/milvus/issues/43659
**Description**: Details Milvus 2.6 evolution to stream/batch-separation architecture, with streamingnode for growing data and querynode roles.

### Source 5: Enhancement for Streaming Service
**URL**: https://github.com/milvus-io/milvus/issues/33285
**Description**: Proposal to integrate streaming service, subscribing stream node in QueryNode and merging Indexnode into Datanode for better architecture.
