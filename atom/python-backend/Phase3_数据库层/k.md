# Phase3_数据库层 知识点列表

> 掌握 SQLAlchemy ORM 和 PostgreSQL，为 AI Agent 提供数据持久化能力

---

## 知识点清单

1. **SQLAlchemy ORM基础** - Python 的 ORM 框架，类似 Prisma/TypeORM
2. **Session管理** - 数据库会话生命周期，类似数据库连接管理
3. **关系定义** - 表关系映射（一对多、多对多），类似外键关联
4. **数据库迁移Alembic** - 数据库版本控制，类似 Prisma Migrate
5. **连接池配置** - 数据库连接复用，提升性能
6. **向量检索pgvector** - PostgreSQL 向量扩展，RAG 系统的核心

---

## 学习顺序建议

```
SQLAlchemy ORM基础 → Session管理 → 关系定义 → 数据库迁移Alembic → 连接池配置 → 向量检索pgvector
        ↓              ↓          ↓              ↓                ↓              ↓
     模型定义        事务管理    表关联        版本控制          性能优化        向量检索
```

**为什么是这个顺序？**

1. **SQLAlchemy ORM基础（1）**：定义数据模型，CRUD 操作基础
2. **Session管理（2）**：理解事务和会话生命周期，避免常见错误
3. **关系定义（3）**：处理表之间的关联关系
4. **数据库迁移（4）**：管理数据库 schema 变更
5. **连接池配置（5）**：生产环境性能优化
6. **向量检索pgvector（6）**：AI Agent 的向量检索能力

---

## 与 AI Agent 开发的关系

| 知识点 | 在 AI Agent 后端中的应用 | 关键产出 |
|--------|-------------------------|----------|
| SQLAlchemy ORM基础 | 用户、对话、消息等数据模型 | 类型安全的数据操作 |
| Session管理 | 请求级别的数据库事务 | 数据一致性保证 |
| 关系定义 | 用户-对话-消息关系 | 复杂数据关系管理 |
| 数据库迁移Alembic | Schema 变更管理 | 可追溯的数据库版本 |
| 连接池配置 | 高并发请求处理 | 数据库性能优化 |
| 向量检索pgvector | 文档 Embedding 存储和检索 | RAG 向量检索能力 |

---

## 前置知识

- ✅ Phase1_Python基础强化（类型注解、上下文管理器）
- ✅ Phase2_FastAPI核心（依赖注入）
- ✅ SQL 基础（SELECT、INSERT、UPDATE、DELETE）
- ✅ 数据库基本概念（表、索引、外键）

## 后续学习

- → Phase4_AI_Agent开发（LangChain + 数据库集成）
- → Phase5_生产级实践（数据库监控、备份）
- → RAG 开发（向量检索优化）

---

## 学习检查清单

完成本阶段后，你应该能够：

- [ ] 用 SQLAlchemy 定义数据模型
- [ ] 在 FastAPI 中正确使用 Session
- [ ] 定义一对多、多对多关系
- [ ] 使用 Alembic 创建和应用迁移
- [ ] 配置数据库连接池
- [ ] 使用 pgvector 存储和检索向量

---

**版本：** v1.0
**最后更新：** 2026-02-10
