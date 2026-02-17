# L5_生产实践 知识点列表

> 掌握 Milvus 2.6 的生产部署、监控和运维技能

---

## 知识点清单

1. **Docker部署** - 使用 Docker Compose 部署生产级 Milvus 2.6 实例（包括 Embedding Functions 配置）
2. **监控与健康检查** - 配置监控指标、告警和健康检查（包括 Streaming Node 监控）
3. **备份与恢复** - 实现数据备份、恢复和迁移策略（CDC + BulkInsert）
4. **Kubernetes部署** - 使用 K8s Operator 和 Helm Charts 部署 Milvus 2.6 集群
5. **安全与权限管理** - 配置认证、授权和数据加密
6. **高可用集群** - 实现多副本、故障转移和灾难恢复（Coord Merge 优化）

---

## 学习顺序建议

```
Docker部署 → 监控与健康检查 → 备份与恢复 → Kubernetes部署 → 安全与权限管理 → 高可用集群
    ↓             ↓              ↓              ↓                ↓                ↓
  环境搭建      运行监控        数据安全      云原生部署        访问控制          容错能力
```

---

## 前置知识

- ✅ L1_快速入门（所有知识点）
- ✅ L2_核心功能（所有知识点）
- ✅ Docker 基础知识
- 🔶 Kubernetes 基础（L4-L6 需要）

---

## 学习目标

完成本层级学习后，你将能够：
- ✅ 部署生产级 Milvus 2.6 实例（包括 Embedding Functions 配置）
- ✅ 配置 Woodpecker WAL 和 Streaming Node
- ✅ 配置监控和告警系统（包括 Streaming Node 监控）
- ✅ 实施备份和恢复策略（使用 CDC + BulkInsert）
- ✅ 处理常见的运维问题
- ✅ 在 Kubernetes 上部署和管理 Milvus 2.6 集群
- ✅ 配置安全认证和权限控制
- ✅ 构建高可用的 Milvus 架构（使用 Coord Merge）
- ✅ 实现故障自动恢复

---

## 2026 核心特性

### Milvus 2.6 部署特性
- **APT/YUM 包管理器**: 简化安装流程
- **Woodpecker WAL**: 零磁盘架构配置
- **Streaming Node**: 统一流式数据处理配置
- **Coord Merge**: 减少协调节点开销
- **Embedding Functions**: 配置多种 Embedding 提供商

### Milvus 2.6 运维特性
- **CDC + BulkInsert**: 高效数据迁移
- **100K Collections**: 大规模多租户部署
- **热冷分层存储**: 存储成本优化配置

---

## 预计学习时间

- 快速入门：3-4小时（核心3个知识点：Docker、监控、备份）
- 完整学习：12-15小时（全部6个知识点）

---

## 2026 学习重点

1. **Milvus 2.6 架构部署**：Woodpecker WAL + Streaming Node 配置
2. **Embedding Functions 配置**：生产环境 API 密钥管理
3. **CDC + BulkInsert**：高效数据迁移策略
4. **成本优化配置**：热冷分层存储、RaBitQ 量化
5. **大规模多租户**：100K collections 的部署实践

---

## 部署架构对比

| 部署方式 | 适用场景 | 优势 | 劣势 |
|----------|----------|------|------|
| Docker Compose | 开发/测试 | 快速部署 | 不支持自动扩展 |
| Kubernetes | 生产环境 | 自动扩展、高可用 | 配置复杂 |
| APT/YUM | 裸机部署 | 性能最优 | 运维复杂 |
| 云托管 | 快速上线 | 零运维 | 成本较高 |

---

**开始学习：** [01_Docker部署](./01_Docker部署/)
