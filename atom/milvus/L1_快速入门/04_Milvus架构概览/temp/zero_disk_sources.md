# Zero-Disk Architecture Sources

## Fetched: 2026-02-21

### Source 1: Milvus 2.6 Release Announcement
**URL**: https://milvus.io/blog/introduce-milvus-2-6-built-for-scale-designed-to-reduce-costs.md
**Description**: Milvus 2.6 发布公告，引入 Woodpecker 实现的 zero-disk architecture，实现经济高效的搜索新鲜度，所有日志数据存储于对象存储，无本地磁盘依赖。

### Source 2: Woodpecker Architecture Documentation
**URL**: https://milvus.io/docs/woodpecker_architecture.md
**Description**: Milvus 2.6 中 Woodpecker 的官方文档，详细说明 zero-disk architecture：日志数据存于对象存储，元数据由 etcd 管理，无本地磁盘依赖，降低运维开销。

### Source 3: Replacing Kafka/Pulsar with Woodpecker
**URL**: https://milvus.io/blog/we-replaced-kafka-pulsar-with-a-woodpecker-for-milvus.md
**Description**: 介绍 Milvus 2.6 用 Woodpecker 替换 Kafka/Pulsar，核心创新为 zero-disk architecture，所有日志直接写入对象存储，提升云原生效率。

### Source 4: Architecture Overview
**URL**: https://milvus.io/docs/architecture_overview.md
**Description**: Milvus 整体架构概述，强调 Woodpecker 采用 cloud-native zero-disk design，直接写入对象存储，实现存储与计算完全解耦。

### Source 5: Release Notes
**URL**: https://milvus.io/docs/release_notes.md
**Description**: Milvus 发布说明，涵盖 2.6 版本 Woodpecker 支持 zero-disk 模式，简化操作并提升性能。
