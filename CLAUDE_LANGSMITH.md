# 原子化知识点生成规范 - LangSmith 专用

> 本文档定义了为 LangSmith 学习项目生成原子化知识点文档的标准和要求

---

## 文档概述

**项目目标：** 为 LangSmith 深度学习构建完整的原子化知识体系

**核心理念：**
- **原子化**：每个知识点独立完整，可独立学习
- **全面覆盖**：知识点包含多个子概念时，全部详细讲解，不遗漏
- **实战导向**：所有知识点都要联系 AI 应用可观测性的实际应用
- **初学者友好**：假设零基础，用简单语言和丰富类比
- **速成高效**：抓住20%核心解决80%问题
- **双重类比**：同时提供前端开发类比 + 日常生活类比
- **Python 优先**：所有代码示例使用 Python 3.13+
- **生产导向**：强调生产环境的监控和调试

---

## 模板引用

本文档基于通用原子化知识点模板：**`prompt/atom_template.md`**

LangSmith 的特殊要求在下方 **LangSmith 特定配置** 章节中定义。

---

## 生成流程

### 第一步：确认输入信息

在开始生成前，确认以下信息：

1. **知识点名称**：从 `atom/langsmith/[层级]/k.md` 中获取
2. **层级目录**：如 `L1_可观测性基础`、`L2_调试与评估`、`L3_生产监控` 等
3. **目标受众**：初学者（默认）或进阶学习者
4. **文件位置**：`atom/langsmith/[层级]/[编号]_[知识点名称]/`

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

参考 `prompt/atom_template.md` 的详细规范，结合下方的 LangSmith 特定配置生成内容。

### 第四步：质量检查

使用 `prompt/atom_template.md` 中的检查清单验证质量。

---

## LangSmith 特定配置

### 应用场景强调

**每个部分都要联系 AI 应用可观测性实际应用：**
- ✅ 这个知识在 LangSmith 中如何体现？
- ✅ 为什么 AI 应用需要这个？
- ✅ 实际场景举例（调试 Agent、评估 RAG、监控生产、优化成本）

**重点强调：**
- Tracing 与调用链追踪
- Dataset 与评估体系
- 生产监控与告警
- 成本与性能优化
- 与 LangChain/LangGraph 的集成

### LangSmith 类比对照表

在【双重类比】维度中，优先使用以下类比：

| LangSmith 概念 | 前端类比 | 日常生活类比 |
|----------------|----------|--------------|
| Tracing | Chrome DevTools | 监控摄像头 |
| Run | HTTP 请求 | 一次任务执行 |
| Span | 函数调用栈 | 任务的子步骤 |
| Dataset | 测试用例集 | 考试题库 |
| Evaluator | 单元测试断言 | 阅卷老师 |
| Annotation | 代码注释 | 批注 |
| Feedback | 用户反馈 | 客户评价 |
| Monitoring | 性能监控面板 | 仪表盘 |
| Alert | 错误告警 | 报警器 |
| Insights Agent | 自动分析工具 | 智能助手 |
| Playground | 在线编辑器 | 实验室 |
| Prompt Hub | 代码片段库 | 模板库 |

### 推荐库列表

在【实战代码】维度中，优先使用以下库：

| 用途 | 推荐库 |
|------|--------|
| 核心框架 | `langsmith` |
| LangChain 集成 | `langchain`, `langchain-openai` |
| LangGraph 集成 | `langgraph` |
| 评估 | `langsmith` (内置评估器) |
| 工具 | `python-dotenv` |

### LangSmith 常见误区

在【反直觉点】维度中，可参考以下常见误区：

- "追踪严重影响性能"（开销很小，约1-2%）
- "只能追踪 LangChain"（支持任意 Python 代码）
- "评估只能在开发环境"（支持生产环境在线评估）
- "必须使用云服务"（支持自托管）
- "Tracing 数据会泄露隐私"（可以配置数据脱敏）
- "只能手动标注数据"（支持自动评估和 AI 辅助标注）
- "监控只能看历史数据"（支持实时监控和告警）
- "成本追踪不准确"（精确到每个 token）

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
python examples/langsmith/basic_tracing.py
```

### 可用的库

所有代码示例可以使用以下库：

| 用途 | 库名 |
|------|------|
| **核心框架** | `langsmith` |
| **LangChain** | `langchain`, `langchain-openai` |
| **LangGraph** | `langgraph` |
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
   LANGSMITH_PROJECT=your_project_name
   # 可选：自托管端点
   LANGSMITH_ENDPOINT=https://your-instance.com
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
atom/langsmith/L1_可观测性基础/01_Tracing基础与自动追踪/
atom/langsmith/L1_可观测性基础/02_Run与Span概念/
atom/langsmith/L2_调试与评估/01_Trace查看器与调试技巧/
atom/langsmith/L3_生产监控/01_实时监控与告警/
```

**编号规则：**
- 按学习顺序编号（01, 02, 03...）
- 反映知识点的依赖关系
- 与 `k.md` 中的顺序一致

### 目录结构

```
atom/
└── langsmith/                          # LangSmith 学习路径
    ├── L1_可观测性基础/
    │   ├── k.md                        # 知识点列表
    │   ├── 01_Tracing基础与自动追踪/
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
    ├── L2_调试与评估/
    ├── L3_生产监控/
    ├── L4_集成与扩展/
    └── L5_高级特性/
```

---

## 快速启动模板

### 生成新知识点的步骤

1. **读取通用模板** (`prompt/atom_template.md`)
2. **读取本文档** (`CLAUDE_LANGSMITH.md`) - LangSmith 特定配置
3. **读取简化接口** (`prompt/atom_knowledge.md`)
4. **读取知识点列表** (`atom/langsmith/[层级]/k.md`)
5. **确认目标知识点**（第几个）
6. **按规范生成内容**（10个维度）
7. **质量检查**（使用检查清单）
8. **保存文件**（`atom/langsmith/[层级]/[编号]_[知识点]/`）

### 提示词模板

```
根据 @prompt/atom_template.md 的通用规范和 @CLAUDE_LANGSMITH.md 的 LangSmith 特定配置，为 @atom/langsmith/[层级]/k.md 中的第[N]个知识点 "[知识点名称]" 生成一个完整的学习文档。

要求：
- 按照10个维度完整生成
- 初学者友好
- 代码可运行（Python 3.13+）
- 双重类比（前端 + 日常生活）
- 与 AI 应用可观测性紧密结合
- 强调生产环境应用
- 强调与 LangChain/LangGraph 的集成

文件保存到：atom/langsmith/[层级]/[编号]_[知识点名称]/
```

---

## 核心原则总结

1. **原子化**：每个知识点独立完整
2. **全面覆盖**：知识点所有子概念都要讲到
3. **实战导向**：联系 AI 应用可观测性
4. **初学者友好**：简单语言 + 双重类比
5. **速成高效**：20%核心 + 80%效果
6. **代码可运行**：所有示例都能跑（Python 3.13+）
7. **体系完整**：10个维度全覆盖
8. **质量保证**：严格检查清单
9. **生产导向**：强调生产环境监控和调试

---

**版本：** v1.0 (LangSmith 专用版 - 基于通用模板)
**最后更新：** 2026-02-12
**维护者：** Claude Code

---

**记住：** 生成每个新知识点前，先读取 `prompt/atom_template.md` 和本文档！
