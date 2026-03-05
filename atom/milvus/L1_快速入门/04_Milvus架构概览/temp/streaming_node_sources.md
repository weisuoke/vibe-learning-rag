# Streaming Node Sources

## Fetched: 2026-02-21

### Source 1: Milvus Repository - Streaming Node Implementation
**URL**: https://github.com/milvus-io/milvus
**Description**: Milvus 核心仓库，Streaming Node 服务端实现位于 internal/streamingnode 或相关路径，支持 WAL 操作和增量数据处理，查看 configs/milvus.yaml 中的 streamingNode 配置部分

### Source 2: Milvus 2.6 Release Notes
**URL**: https://milvus.io/docs/release_notes.md
**Description**: Milvus 2.6 系列发布说明，详细描述 Streaming Node 从实验到 GA 的演进、架构变更（如 WAL 管理、MixCoord 集成），包含 PR 链接如 #46982 用于 streaming 服务稳定性

### Source 3: Use Woodpecker WAL Configuration
**URL**: https://milvus.io/docs/use-woodpecker.md
**Description**: Milvus 2.6 使用 Woodpecker 作为 WAL 的配置教程，包括 yaml 示例、Helm/K8s 部署 Streaming Node、Python 批量插入代码演示实时写入效果

### Source 4: Milvus configs/milvus.yaml
**URL**: https://github.com/milvus-io/milvus/blob/master/configs/milvus.yaml
**Description**: Milvus 默认配置文件，包含 streamingNode 部分详细参数如 IP、端口、日志、Woodpecker 集成设置，用于自定义 Streaming Node 行为

### Source 5: Milvus Architecture Overview
**URL**: https://milvus.io/docs/architecture_overview.md
**Description**: Milvus 官方架构文档，解释 Streaming Node 在 2.6 中的作用：处理增量摄取、查询委托、WAL 持久化，与 QueryNode/DataNode 协作
