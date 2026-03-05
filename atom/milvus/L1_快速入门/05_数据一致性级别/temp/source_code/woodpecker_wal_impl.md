---
source: Multiple GitHub issues and discussions
title: Woodpecker WAL Implementation and Consistency
fetched_at: 2026-02-21
---

# Woodpecker WAL Implementation and Consistency

## Overview

Woodpecker is Milvus 2.6's new WAL (Write-Ahead Log) storage option that provides object storage-based high-throughput writes and efficient log services for cloud-native scenarios.

## Key Issues and Insights

### 1. Woodpecker as WAL Storage Option

**Source**: https://github.com/milvus-io/milvus/issues/40916

Woodpecker is introduced as a WAL storage option for Milvus, providing:
- Object storage-based high-throughput writes
- Efficient log service for cloud-native scenarios
- Support for streaming log output
- Embedded log sequential reading

### 2. Configuration in milvus.yaml

**Source**: https://github.com/milvus-io/milvus/blob/master/configs/milvus.yaml

Woodpecker-related configuration manages:
- Recent change operation logs
- Streaming log output
- Embedded log sequential reading
- WAL consistency-related settings

### 3. Consistency Challenges

**Issue**: Stale write.lock files causing 'storage not writable' after restart
**Source**: https://github.com/milvus-io/milvus/discussions/45494

This issue highlights the importance of proper lock management in Woodpecker WAL for maintaining consistency during restarts and recovery.

**Issue**: CP lag keeps increasing after node restart
**Source**: https://github.com/milvus-io/milvus/issues/43604

When using Woodpecker WAL, node restarts can cause checkpoint (cp) lag to increase continuously, affecting query recovery. This relates to WAL consistency and recovery mechanisms.

**Issue**: recoverFromStorageUnsafe error in standalone mode
**Source**: https://github.com/milvus-io/milvus/discussions/47584

Inconsistency between storage metadata and object storage in Woodpecker WAL can cause recovery failures, requiring etcd metadata cleanup to restore WAL consistency.

## Consistency Implementation Insights

### 1. Checkpoint Mechanism

Woodpecker WAL uses a checkpoint (cp) mechanism to track the progress of data synchronization. The checkpoint lag indicates how far behind the current state the persisted state is.

### 2. Lock Management

Write locks are used to ensure exclusive access during write operations. Proper cleanup of stale locks is crucial for maintaining consistency after failures.

### 3. Metadata Synchronization

Woodpecker maintains metadata in etcd that must stay synchronized with the actual data in object storage. Inconsistencies can lead to recovery failures.

### 4. Recovery Process

The recovery process (`recoverFromStorageUnsafe`) attempts to restore consistency by reading from both metadata and object storage. Failures in this process indicate consistency violations.

## Implications for Consistency Levels

### Strong Consistency

- Requires waiting for WAL writes to be fully synchronized
- Checkpoint must be up-to-date before reads
- Higher latency due to synchronization overhead

### Bounded Consistency

- Allows reads within a staleness bound (time window)
- Checkpoint lag within acceptable bounds
- Balances consistency and performance

### Eventually Consistency

- Reads can proceed without waiting for WAL synchronization
- Accepts checkpoint lag
- Lowest latency, highest throughput

## Code References

While the actual Golang implementation code is not directly accessible in these issues, the key components mentioned include:

1. **WAL Interface**: Manages write-ahead log operations
2. **Checkpoint Manager**: Tracks synchronization progress
3. **Lock Manager**: Handles write locks for consistency
4. **Recovery Manager**: Restores consistency after failures
5. **Metadata Synchronizer**: Keeps etcd and object storage in sync

## Best Practices

1. **Monitor Checkpoint Lag**: Keep track of cp lag to detect consistency issues early
2. **Proper Shutdown**: Ensure clean shutdown to avoid stale locks
3. **Metadata Backup**: Regularly backup etcd metadata for recovery
4. **Consistency Level Selection**: Choose appropriate consistency level based on use case
   - Strong: Critical data requiring immediate consistency
   - Bounded: Most production scenarios (default)
   - Eventually: High-throughput, non-critical reads
