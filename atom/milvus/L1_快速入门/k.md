# L1_快速入门 知识点列表

> 快速掌握 Milvus 2.6 的基本使用，从安装到第一次向量检索

---

## 知识点清单

1. **安装与连接** - 使用 Docker 部署 Milvus 2.6 并通过 Python SDK 建立连接
2. **Collection管理** - 创建、查看、删除 Collection，理解 Schema 定义，支持 100K collections
3. **数据插入与查询** - 使用 Embedding Functions 实现 Data-in, Data-out 模式，简化向量化流程
4. **Milvus架构概览** - 理解 Milvus 2.6 架构：Streaming Node、Woodpecker WAL、零磁盘架构
5. **数据一致性级别** - 掌握 Strong、Bounded、Eventually 三种一致性级别的选择

---

## 学习顺序建议

```
安装与连接 → Collection管理 → 数据插入与查询 → Milvus架构概览 → 数据一致性级别
    ↓              ↓              ↓                ↓                ↓
  环境准备      数据结构设计    Embedding自动化   理解2.6架构      性能权衡
```

---

## 前置知识

- ✅ 向量/Embedding 概念（atom/rag/L1_NLP基础/02_Embedding原理与选型）
- ✅ 语义相似度（atom/rag/L1_NLP基础/03_语义相似度）
- ✅ Python 基础

---

## 学习目标

完成本层级学习后，你将能够：
- ✅ 部署并连接到 Milvus 2.6 实例
- ✅ 创建包含向量字段的 Collection（支持 100K collections）
- ✅ 使用 Embedding Functions 直接插入原始文本，自动向量化
- ✅ 配置多种 Embedding 提供商（OpenAI、Cohere、Bedrock 等）
- ✅ 执行基本的相似度检索
- ✅ 理解 Milvus 2.6 的 Data-in, Data-out 工作流程
- ✅ 理解 Milvus 2.6 的分布式架构（Streaming Node + Woodpecker WAL）
- ✅ 根据场景选择合适的一致性级别

---

## 2026 核心特性

### Milvus 2.6 新特性（2026 年 2 月）
- **Embedding Functions**: 内置文本到向量转换，无需外部调用
- **Woodpecker WAL**: 零磁盘架构，搜索新鲜度优化
- **Streaming Node**: 统一流式数据处理
- **100K Collections**: 支持大规模多租户场景
- **APT/YUM 包管理器**: 简化安装流程

### 2026 学习重点
1. **Embedding Functions 是入门核心**：无需外部 Embedding 预处理
2. **Data-in, Data-out 模式**：直接插入原始文本，自动向量化
3. **简化的开发流程**：从 5 步简化到 3 步

---

## 预计学习时间

- 快速入门：2-3小时（核心3个知识点：安装、Collection、数据插入）
- 完整学习：5-6小时（全部5个知识点）

---

## 2026 vs 传统流程对比

### 传统流程（5步）
```
1. 加载文档
2. 调用外部 Embedding API
3. 创建 Collection
4. 插入向量
5. 检索
```

### 2026 Milvus 2.6 流程（3步）
```
1. 创建 Collection（配置 Embedding Function）
2. 插入原始文本（自动向量化）
3. 检索
```

---

**开始学习：** [01_安装与连接](./01_安装与连接/)
