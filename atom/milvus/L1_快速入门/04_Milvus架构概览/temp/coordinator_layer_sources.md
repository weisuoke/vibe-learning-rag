# Coordinator Layer Sources

## Fetched: 2026-02-21

### Source 1: Milvus 2.6 Release Notes
**URL**: https://milvus.io/docs/release_notes.md
**Description**: Milvus 2.6 版本发布说明，介绍新架构变化，包括协调器合并为 MixCoord，RootCoord、DataCoord 和 QueryCoord 整合，简化系统设计。

### Source 2: Upgrade Guide from 2.5.x to 2.6.x
**URL**: https://milvus.io/blog/how-to-safely-upgrade-from-milvu-2-5-x-to-milvus-2-6-x.md
**Description**: Milvus 2.6 升级指南，详细说明协调器层架构演进：RootCoord、QueryCoord、DataCoord 合并为单一 MixCoord 组件，提升系统效率和可维护性。

### Source 3: GitHub Issue - Merge Coordinators
**URL**: https://github.com/milvus-io/milvus/issues/37764
**Description**: GitHub Issue 讨论将 RootCoord、DataCoord 和 QueryCoord 合并为单一组件的设计提案，旨在简化 Milvus 架构并减少冗余元数据管理。

### Source 4: Coordinator HA Documentation
**URL**: https://milvus.io/docs/coordinator_ha.md
**Description**: Milvus 协调器高可用文档，涵盖 RootCoord、QueryCoord、DataCoord 的主备机制，在 2.6 中这些协调器已整合为 MixCoord。

### Source 5: Milvus GitHub Releases
**URL**: https://github.com/milvus-io/milvus/releases
**Description**: Milvus GitHub Releases 页面，包含 2.6 系列版本更新记录，涉及 QueryCoord、RootCoord 等协调器组件的改进和修复。
