# 原子化知识点生成规范 - LangGraph 专用

> 本文档定义了为 LangGraph 学习项目生成原子化知识点文档的标准和要求

---

## 文档概述

**项目目标：** 为 LangGraph 深度学习构建完整的原子化知识体系

**核心理念：**
- **原子化**：每个知识点独立完整，可独立学习
- **全面覆盖**：知识点包含多个子概念时，全部详细讲解，不遗漏
- **实战导向**：所有知识点都要联系状态化工作流开发的实际应用
- **初学者友好**：假设零基础，用简单语言和丰富类比
- **速成高效**：抓住20%核心解决80%问题
- **双重类比**：同时提供前端开发类比 + 日常生活类比
- **Python 优先**：所有代码示例使用 Python 3.13+
- **源码理解**：深入理解框架设计和架构决策

---

## 模板引用

本文档基于通用原子化知识点模板：**`prompt/atom_template.md`**

LangGraph 的特殊要求在下方 **LangGraph 特定配置** 章节中定义。

---

## 生成流程

### 第一步：确认输入信息

在开始生成前，确认以下信息：

1. **知识点名称**：从 `atom/langgraph/[层级]/k.md` 中获取
2. **层级目录**：如 `L1_图基础`、`L2_状态管理`、`L3_工作流编排` 等
3. **目标受众**：初学者（默认）或进阶学习者
4. **文件位置**：`atom/langgraph/[层级]/[编号]_[知识点名称]/`

### 第二步：读取模板

**通用模板：** `prompt/atom_template.md` - 定义10个维度的标准结构
**简化接口：** `prompt/atom_knowledge.md` - 快速提示词接口

**10个必需维度：**
1. 【30字核心】
2. 【第一性原理】
3. 【3个核心概念】
4. 【最小可用】
5. 【双重类比】
6. 【反直觉点】
7. 【实战代码】
8. 【面试必问】
9. 【化骨绵掌】
10. 【一句话总结】

### 第三步：按规范生成内容

参考 `prompt/atom_template.md` 的详细规范，结合下方的 LangGraph 特定配置生成内容。

### 第四步：质量检查

使用 `prompt/atom_template.md` 中的检查清单验证质量。

---

## LangGraph 特定配置

### 应用场景强调

**每个部分都要联系状态化工作流开发实际应用：**
- ✅ 这个知识在 LangGraph 开发中如何体现？
- ✅ 为什么状态化工作流需要这个？
- ✅ 实际场景举例（多步推理、人机协作、复杂决策流、可恢复任务）

**重点强调：**
- 图结构与状态流转
- 条件路由与动态决策
- 持久化与断点续传
- 人机循环与审批流程
- 子图与模块化设计

### LangGraph 类比对照表

在【双重类比】维度中，优先使用以下类比：

| LangGraph 概念 | 前端类比 | 日常生活类比 |
|----------------|----------|--------------|
| StateGraph | React 状态机 | 流程图 |
| Node | 组件函数 | 工序站点 |
| State | Redux store | 工作台状态 |
| Edge | 路由规则 | 传送带 |
| Conditional Edge | 条件渲染 | 分拣机 |
| Reducer | Redux reducer | 状态更新规则 |
| Checkpoint | LocalStorage 快照 | 游戏存档 |
| Human-in-the-loop | 表单提交等待 | 审批流程 |
| Subgraph | 子组件 | 子流程 |
| Parallel Execution | Promise.all() | 多人并行工作 |
| Stream | Server-Sent Events | 实时进度播报 |
| Interrupt | 暂停执行 | 红灯停车 |

### 推荐库列表

在【实战代码】维度中，优先使用以下库：

| 用途 | 推荐库 |
|------|--------|
| 核心框架 | `langgraph` |
| 持久化 | `langgraph-checkpoint`, `langgraph-checkpoint-postgres` |
| LangChain 集成 | `langchain`, `langchain-openai` |
| 数据库 | `psycopg2`, `pymongo` |
| 可观测性 | `langsmith` |

### LangGraph 常见误区

在【反直觉点】维度中，可参考以下常见误区：

- "状态可以随意修改"（必须通过 Reducer 函数）
- "图执行是同步的"（支持异步节点）
- "Checkpoint 自动保存"（需要显式配置 Checkpointer）
- "所有节点必须返回完整状态"（可以返回部分更新）
- "条件边只能有两个分支"（可以有多个分支）
- "子图不能访问父图状态"（可以通过参数传递）
- "中断后无法恢复"（Checkpoint 支持断点续传）
- "LangGraph 只适合简单流程"（可以构建复杂的企业级工作流）

---

## Python 环境配置

### 环境要求

- **Python 版本**: 3.13+ (通过 asdf 管理)
- **包管理器**: uv

### 快速开始

```bash
# 1. 安装 Python
asdf install

# 2. 安装依赖
uv sync

# 3. 激活环境
source .venv/bin/activate

# 4. 运行示例
python examples/langgraph/basic_graph.py
```

### 可用的库

所有代码示例可以使用以下库：

| 用途 | 库名 |
|------|------|
| **核心框架** | `langgraph` |
| **持久化** | `langgraph-checkpoint`, `langgraph-checkpoint-postgres` |
| **LangChain** | `langchain`, `langchain-openai` |
| **数据库** | `psycopg2`, `pymongo` |
| **可观测性** | `langsmith` |
| **工具** | `python-dotenv` |

### 环境管理

```bash
# 添加新依赖
uv add <package-name>

# 添加开发依赖
uv add --dev <package-name>

# 更新依赖
uv sync --upgrade
```

### 配置 API 密钥

1. 复制环境变量模板：
   ```bash
   cp .env.example .env
   ```

2. 编辑 `.env` 文件，添加你的 API keys：
   ```bash
   OPENAI_API_KEY=your_key_here
   LANGSMITH_API_KEY=your_key_here
   LANGSMITH_TRACING=true
   # 数据库连接（可选）
   DATABASE_URL=postgresql://user:pass@localhost/dbname
   ```

3. 在代码中加载：
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```

---

## 文件组织规范

### 文件命名

**格式：** `[编号]_[知识点名称]/`（目录形式）

**示例：**
```
atom/langgraph/L1_图基础/01_StateGraph与节点定义/
atom/langgraph/L1_图基础/02_边与条件路由/
atom/langgraph/L2_状态管理/01_State_Schema与类型化状态/
atom/langgraph/L4_持久化与检查点/01_Checkpoint机制/
```

**编号规则：**
- 按学习顺序编号（01, 02, 03...）
- 反映知识点的依赖关系
- 与 `k.md` 中的顺序一致

### 目录结构

```
atom/
└── langgraph/                          # LangGraph 学习路径
    ├── L1_图基础/
    │   ├── k.md                        # 知识点列表
    │   ├── 01_StateGraph与节点定义/
    │   │   ├── 00_概览.md
    │   │   ├── 01_30字核心.md
    │   │   ├── 02_第一性原理.md
    │   │   ├── 03_核心概念.md
    │   │   ├── 04_最小可用.md
    │   │   ├── 05_双重类比.md
    │   │   ├── 06_反直觉点.md
    │   │   ├── 07_实战代码.md
    │   │   ├── 08_面试必问.md
    │   │   ├── 09_化骨绵掌.md
    │   │   └── 10_一句话总结.md
    │   └── ...
    ├── L2_状态管理/
    ├── L3_工作流编排/
    ├── L4_持久化与检查点/
    ├── L5_高级模式/
    └── L6_源码与架构/
```

---

## 快速启动模板

### 生成新知识点的步骤

1. **读取通用模板** (`prompt/atom_template.md`)
2. **读取本文档** (`CLAUDE_LANGGRAPH.md`) - LangGraph 特定配置
3. **读取简化接口** (`prompt/atom_knowledge.md`)
4. **读取知识点列表** (`atom/langgraph/[层级]/k.md`)
5. **确认目标知识点**（第几个）
6. **按规范生成内容**（10个维度）
7. **质量检查**（使用检查清单）
8. **保存文件**（`atom/langgraph/[层级]/[编号]_[知识点]/`）

### 提示词模板

```
根据 @prompt/atom_template.md 的通用规范和 @CLAUDE_LANGGRAPH.md 的 LangGraph 特定配置，为 @atom/langgraph/[层级]/k.md 中的第[N]个知识点 "[知识点名称]" 生成一个完整的学习文档。

要求：
- 按照10个维度完整生成
- 初学者友好
- 代码可运行（Python 3.13+）
- 双重类比（前端 + 日常生活）
- 与状态化工作流开发紧密结合
- 包含源码理解
- 强调与 LangChain 的集成

文件保存到：atom/langgraph/[层级]/[编号]_[知识点名称]/
```

---

## 核心原则总结

1. **原子化**：每个知识点独立完整
2. **全面覆盖**：知识点所有子概念都要讲到
3. **实战导向**：联系状态化工作流开发应用
4. **初学者友好**：简单语言 + 双重类比
5. **速成高效**：20%核心 + 80%效果
6. **代码可运行**：所有示例都能跑（Python 3.13+）
7. **体系完整**：10个维度全覆盖
8. **质量保证**：严格检查清单
9. **源码理解**：深入框架设计和架构

---

**版本：** v1.0 (LangGraph 专用版 - 基于通用模板)
**最后更新：** 2026-02-12
**维护者：** Claude Code

---

**记住：** 生成每个新知识点前，先读取 `prompt/atom_template.md` 和本文档！
