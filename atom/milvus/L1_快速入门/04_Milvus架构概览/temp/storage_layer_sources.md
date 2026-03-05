# Storage Layer Sources

## Fetched: 2026-02-21

### Source 1: CollectionCreate Timeout Bug
**URL**: https://github.com/milvus-io/milvus/issues/45007
**Description**: Milvus 2.6 bug: CollectionCreate times out despite healthy cluster, related to etcd path consistency in operator-based deployments.

### Source 2: AWS Multipart Upload Error
**URL**: https://github.com/milvus-io/milvus/issues/44853
**Description**: Milvus v2.6.x introduces multipart upload in storage layer, causing AWS NO_SUCH_UPLOAD error with S3-compatible object storage like MinIO.

### Source 3: Streamingnode MinIO Recovery Issue
**URL**: https://github.com/milvus-io/milvus/issues/43597
**Description**: Milvus streamingnode panics due to schema not found after MinIO pod disruption, as storage writer fails when object storage is unavailable.

### Source 4: External S3 Configuration
**URL**: https://github.com/milvus-io/milvus/discussions/46881
**Description**: Instructions for configuring Milvus standalone in Docker to use external S3 instead of MinIO, by editing milvus.yaml and mounting the config file.
