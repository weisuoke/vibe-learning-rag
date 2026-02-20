# 原子化知识点生成规范 - LangChain 专用

> 本文档定义了为 LangChain 学习项目生成原子化知识点文档的标准和要求

---

## 文档概述

**项目目标：** 为 LangChain 深度学习构建完整的原子化知识体系

**核心理念：**
- **原子化**：每个知识点独立完整，可独立学习
- **全面覆盖**：知识点包含多个子概念时，全部详细讲解，不遗漏
- **实战导向**：所有知识点都要联系 AI Agent 开发的实际应用
- **初学者友好**：假设零基础，用简单语言和丰富类比
- **速成高效**：抓住20%核心解决80%问题
- **双重类比**：同时提供前端开发类比 + 日常生活类比
- **Python 优先**：所有代码示例使用 Python 3.13+
- **源码理解**：深入理解框架设计和架构决策

---

## 模板引用

本文档基于通用原子化知识点模板：**`prompt/atom_template.md`**

LangChain 的特殊要求在下方 **LangChain 特定配置** 章节中定义。

---

## 生成流程

### 第一步：确认输入信息

在开始生成前，确认以下信息：

1. **知识点名称**：从 `atom/langchain/[层级]/k.md` 中获取
2. **层级目录**：如 `L1_核心抽象`、`L2_LCEL表达式`、`L3_组件生态` 等
3. **目标受众**：初学者（默认）或进阶学习者
4. **文件位置**：`atom/langchain/[层级]/[编号]_[知识点名称]/`

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

参考 `prompt/atom_template.md` 的详细规范，结合下方的 LangChain 特定配置生成内容。

### 第四步：质量检查

使用 `prompt/atom_template.md` 中的检查清单验证质量。

---

## LangChain 特定配置

### 应用场景强调

**每个部分都要联系 AI Agent 开发实际应用：**
- ✅ 这个知识在 LangChain 开发中如何体现？
- ✅ 为什么 AI Agent 开发需要这个？
- ✅ 实际场景举例（对话机器人、RAG 应用、工作流自动化、多步推理）

**重点强调：**
- Runnable 协议与组合性
- LCEL 表达式的声明式编程
- Agent 的推理与工具调用
- Memory 与状态管理
- 可观测性与调试

**2025-2026 新增场景：**
- ✅ 工作流优先设计（LangGraph）
- ✅ 多模态应用（文本 + 图像 + 视频）
- ✅ 成本敏感的批处理任务
- ✅ 需要状态持久化的长时间任务
- ✅ 自助修复和自适应系统

**何时不使用 LangChain：**
- ❌ 极简单的单次 LLM 调用（直接用 SDK）
- ❌ 需要极致性能优化的场景
- ❌ 团队更熟悉其他框架（如 Semantic Kernel）

### LangChain 类比对照表

在【双重类比】维度中，优先使用以下类比：

| LangChain 概念 | 前端类比 | 日常生活类比 |
|----------------|----------|--------------|
| Runnable | Express 中间件 | 流水线工序 |
| LCEL (管道操作符) | RxJS 操作符链 | 工厂装配线 |
| ChatModel | Fetch API | 打电话咨询专家 |
| PromptTemplate | 模板字符串 | 填空题模板 |
| OutputParser | JSON.parse() | 翻译官 |
| RunnablePassthrough | 透传中间件 | 传话筒 |
| RunnableParallel | Promise.all() | 多人同时工作 |
| Chain | 函数组合 | 菜谱步骤 |
| Agent | 自主决策的 AI | 有助手的项目经理 |
| Tools | API 端点 | 工具箱 |
| Memory | LocalStorage | 笔记本 |
| Retriever | 搜索引擎 | 图书馆检索系统 |
| VectorStore | 数据库索引 | 按主题分类的档案柜 |
| LangGraph | 状态机 + React Flow | 项目管理看板 |
| StateGraph | Redux Store | 游戏存档系统 |
| Checkpointer | 数据库事务 | 游戏自动保存 |
| ConditionalEdge | 路由守卫 | 红绿灯路口 |
| MCP Server | GraphQL Gateway | 统一服务台 |
| Batch Processing | 队列系统 | 批量打印任务 |

### 推荐库列表

在【实战代码】维度中，优先使用以下库：

| 用途 | 推荐库 |
|------|--------|
| 核心框架 | `langchain`, `langchain-core` |
| LLM 集成 | `langchain-openai`, `langchain-anthropic` |
| 向量存储 | `langchain-chroma`, `langchain-community` |
| 文档加载 | `langchain-community` |
| 工具集成 | `langchain-community` |
| 可观测性 | `langsmith` |
| 工作流引擎 | `langgraph` |
| 成本优化 | `langasync` |
| MCP 集成 | `langsmith-mcp-server` |
| 多模态 | `langchain-google-genai` (4.0.0+) |

### LangChain 常见误区

在【反直觉点】维度中，可参考以下常见误区：

- "LCEL 比传统 Chain 慢"（实际上 LCEL 有优化的执行引擎）
- "Agent 一定比 Chain 好"（简单任务用 Chain 更高效）
- "Memory 会自动持久化"（需要显式配置持久化后端）
- "所有组件都必须用 LCEL"（可以混用传统方式）
- "Runnable 只能串行执行"（支持并行和条件路由）
- "LangChain 只适合简单应用"（企业级应用也在使用）
- "Tools 必须是函数"（可以是任何 Runnable）
- "ChatModel 和 LLM 是一回事"（ChatModel 专门处理对话格式）
- "LangGraph 只是 LangChain 的附加功能"（实际上是独立的工作流引擎）
- "序列化是安全的"（需要显式配置 allowed_objects）
- "环境变量会自动加载秘密"（2025 后默认为 false）
- "简单任务必须用 LangChain"（有时原生 API 更合适）
- "Agent 总是比 Chain 好"（工作流优先，代理是组件）
- "LangChain 不适合生产环境"（2025-2026 已有大量企业应用）

---

## LangGraph 与状态管理（2025-2026 新增）

### 核心概念
LangGraph 是 LangChain 生态系统中的工作流引擎，专为构建有状态的、循环的、多代理应用而设计。

### 关键特性
- **状态持久化**：通过 Checkpointer 实现状态保存和恢复
- **条件路由**：基于状态动态决定执行路径
- **循环支持**：允许代理多次迭代直到满足条件
- **多代理协作**：支持复杂的代理间通信模式

### 与传统 Chain 的区别
- Chain：线性、无状态、单向流动
- LangGraph：图状、有状态、支持循环和条件分支

### 适用场景
- 多步推理任务
- 需要回溯和重试的工作流
- 多代理协作系统
- 长时间运行的任务（需要检查点）

### 2026 年最新发展
- LangGraph v1 路线图活跃开发
- `config.configurable` → `context` API 迁移
- 自助修复代理循环模式
- 多模态工具集成（Gemini 2.5）

---

## 安全最佳实践（2025 更新）

### 2025 年关键安全修复
LangChain 在 2025 年修复了多个严重安全漏洞：

#### CVE-2025-68664: 序列化注入
- **问题**：`load()`/`loads()` 可被利用提取秘密
- **修复**：引入 `allowed_objects` 参数（默认 'core'）
- **迁移**：显式指定允许的对象类型

#### CVE-2025-64439: LangGraph 检查点 RCE
- **问题**：JSON 模式下的 JsonPlusSerializer 存在 RCE
- **修复**：限制反序列化函数执行
- **建议**：使用最新版本的 langgraph

#### CVE-2025-65106: 模板注入
- **问题**：Prompt 模板中的属性访问可被利用
- **修复**：限制模板中的对象访问
- **建议**：验证用户输入，避免直接插入模板

#### CVE-2025-6984: XXE 攻击
- **问题**：`etree.iterparse()` 未禁用外部实体引用
- **修复**：禁用 XML 外部实体
- **建议**：更新 langchain-community

### 安全编码实践

#### 1. 安全的序列化
```python
from langchain.load import load

# ❌ 不安全（旧版本）
chain = load(data)

# ✅ 安全（2025+）
chain = load(data, allowed_objects=['core'])
```

#### 2. 秘密管理
```python
from langchain.load import load

# ❌ 不安全（自动从环境加载）
chain = load(data, secretsFromEnv=True)

# ✅ 安全（显式管理秘密）
chain = load(data, secretsFromEnv=False)
# 使用专门的秘密管理工具
```

#### 3. 输入验证
```python
from langchain.prompts import PromptTemplate

# ❌ 不安全（直接插入用户输入）
prompt = PromptTemplate.from_template(user_input)

# ✅ 安全（验证和清理输入）
if validate_template(user_input):
    prompt = PromptTemplate.from_template(user_input)
```

#### 4. XML 解析
```python
# ✅ 使用最新版本的 langchain-community
# 自动禁用外部实体引用
```

---

## 成本优化策略（2026 新增）

### langasync：批处理降低成本 50%

#### 概述
langasync 是 2026 年社区推出的工具，通过批处理 API 调用降低 LLM 成本。

#### 核心原理
- 将多个 LCEL 链调用合并为批处理请求
- 支持 OpenAI 和 Anthropic 批处理 API
- 适用于评估和非实时任务

#### 使用示例
```python
from langasync import wrap_chain

# 原始 LCEL 链
chain = prompt | llm | parser

# 包装为批处理模式
async_chain = wrap_chain(chain, batch_size=10)

# 批量执行（成本降低 50%）
results = await async_chain.abatch(inputs)
```

#### 适用场景
- 批量评估和测试
- 数据标注任务
- 离线分析和报告生成
- 非实时的批量处理

#### 不适用场景
- 实时对话应用
- 需要即时响应的场景
- 单次查询

### 其他成本优化技巧

#### 1. 模型选择
```python
# 简单任务使用更便宜的模型
from langchain_openai import ChatOpenAI

cheap_llm = ChatOpenAI(model="gpt-4o-mini")  # 更便宜
expensive_llm = ChatOpenAI(model="gpt-4")    # 更强大

# 根据任务复杂度选择
chain = RunnableBranch(
    (is_simple_task, cheap_llm),
    expensive_llm
)
```

#### 2. 缓存策略
```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

# 启用缓存避免重复调用
set_llm_cache(InMemoryCache())
```

#### 3. Token 优化
```python
# 使用更短的 prompt
# 移除不必要的上下文
# 使用 token 计数器监控
```

---

## API 重大变更与迁移指南

### 2025-2026 重大变更

#### 1. load()/loads() 安全变更
**变更时间**：2025 年 12 月

**旧 API**：
```python
from langchain.load import load
chain = load(data)  # 自动加载所有对象
```

**新 API**：
```python
from langchain.load import load
chain = load(data, allowed_objects=['core'])  # 必须显式指定
```

**迁移步骤**：
1. 更新到最新版本
2. 添加 `allowed_objects` 参数
3. 测试所有序列化/反序列化代码

#### 2. secretsFromEnv 默认值变更
**变更时间**：2025 年 12 月

**旧行为**：默认 `True`（自动从环境加载）
**新行为**：默认 `False`（需要显式启用）

**迁移步骤**：
```python
# 如果依赖自动加载，需要显式启用
chain = load(data, secretsFromEnv=True)

# 推荐：使用专门的秘密管理
from langchain.pydantic_v1 import SecretStr
api_key = SecretStr(os.getenv("OPENAI_API_KEY"))
```

#### 3. LangGraph config.configurable → context
**变更时间**：2026 年（计划中）

**旧 API**：
```python
graph.invoke(
    input,
    config={"configurable": {"user_id": "123"}}
)
```

**新 API**：
```python
graph.invoke(
    input,
    context={"user_id": "123"}
)
```

**迁移步骤**：
1. 关注 LangGraph v1 发布
2. 使用 `context` 替代 `config.configurable`
3. 更新所有图调用代码

#### 4. Google Vertex AI 集成弃用
**变更时间**：2025 年

**旧包**：`langchain-google-vertexai`
**新包**：`langchain-google-genai` (4.0.0+)

**迁移步骤**：
```bash
# 卸载旧包
uv remove langchain-google-vertexai

# 安装新包
uv add langchain-google-genai
```

```python
# 旧导入
from langchain_google_vertexai import ChatVertexAI

# 新导入
from langchain_google_genai import ChatGoogleGenerativeAI
```

### 版本兼容性矩阵

| 功能 | LangChain 0.1.x | LangChain 0.2.x | LangChain 0.3.x (2025+) |
|------|-----------------|-----------------|-------------------------|
| load() 无参数 | ✅ | ✅ | ❌ 需要 allowed_objects |
| secretsFromEnv 默认 | True | True | False |
| LangGraph | 实验性 | 稳定 | 生产级 + v1 路线图 |
| MCP Server | ❌ | ❌ | ✅ (2026) |
| Google Vertex | ✅ | ✅ | ❌ 使用 GenAI |

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
python examples/langchain/basic_chain.py
```

### 可用的库

所有代码示例可以使用以下库：

| 用途 | 库名 |
|------|------|
| **核心框架** | `langchain`, `langchain-core` |
| **LLM 调用** | `langchain-openai`, `openai` |
| **向量存储** | `langchain-chroma`, `chromadb` |
| **文档处理** | `langchain-community` |
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
   # 可选：使用自定义 API 端点
   OPENAI_BASE_URL=https://your-proxy.com/v1
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
atom/langchain/L1_核心抽象/01_Runnable接口与LCEL基础/
atom/langchain/L1_核心抽象/02_ChatModel与PromptTemplate/
atom/langchain/L2_LCEL表达式/01_管道操作符/
atom/langchain/L4_Agent系统/01_Agent执行循环/
```

**编号规则：**
- 按学习顺序编号（01, 02, 03...）
- 反映知识点的依赖关系
- 与 `k.md` 中的顺序一致

### 目录结构

```
atom/
└── langchain/                          # LangChain 学习路径
    ├── L1_核心抽象/
    │   ├── k.md                        # 知识点列表
    │   ├── 01_Runnable接口与LCEL基础/
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
    ├── L2_LCEL表达式/
    ├── L3_组件生态/
    ├── L4_Agent系统/
    ├── L5_高级特性/
    ├── L6_源码与架构/
    └── L7_LangGraph与状态管理/          # 2025-2026 新增
        ├── k.md
        ├── 01_Graph工作流基础/
        ├── 02_状态持久化与检查点/
        ├── 03_条件路由与循环/
        └── 04_多代理协作模式/
```

---

## 快速启动模板

### 生成新知识点的步骤

1. **读取通用模板** (`prompt/atom_template.md`)
2. **读取本文档** (`CLAUDE_LANGCHAIN.md`) - LangChain 特定配置
3. **读取简化接口** (`prompt/atom_knowledge.md`)
4. **读取知识点列表** (`atom/langchain/[层级]/k.md`)
5. **确认目标知识点**（第几个）
6. **按规范生成内容**（10个维度）
7. **质量检查**（使用检查清单）
8. **保存文件**（`atom/langchain/[层级]/[编号]_[知识点]/`）

### 提示词模板

```
根据 @prompt/atom_template.md 的通用规范和 @CLAUDE_LANGCHAIN.md 的 LangChain 特定配置，为 @atom/langchain/[层级]/k.md 中的第[N]个知识点 "[知识点名称]" 生成一个完整的学习文档。

要求：
- 按照10个维度完整生成
- 初学者友好
- 代码可运行（Python 3.13+）
- 双重类比（前端 + 日常生活）
- 与 AI Agent 开发紧密结合
- 包含源码理解

文件保存到：atom/langchain/[层级]/[编号]_[知识点名称]/
```

---

## 核心原则总结

1. **原子化**：每个知识点独立完整
2. **全面覆盖**：知识点所有子概念都要讲到
3. **实战导向**：联系 AI Agent 开发应用
4. **初学者友好**：简单语言 + 双重类比
5. **速成高效**：20%核心 + 80%效果
6. **代码可运行**：所有示例都能跑（Python 3.13+）
7. **体系完整**：10个维度全覆盖
8. **质量保证**：严格检查清单
9. **源码理解**：深入框架设计和架构

---

**版本：** v2.0 (LangChain 专用版 - 2025-2026 更新)
**最后更新：** 2026-02-17
**维护者：** Claude Code

**重要变更：**
- 添加 LangGraph 与状态管理章节（L7）
- 添加 2025 安全最佳实践（CVE 修复指南）
- 添加成本优化策略（langasync 批处理）
- 添加 API 重大变更与迁移指南
- 更新类比对照表（新增 6 个概念）
- 更新推荐库列表（新增 4 个库）
- 更新常见误区（新增 6 个误区）
- 添加何时不使用 LangChain 的指导

---

**记住：** 生成每个新知识点前，先读取 `prompt/atom_template.md` 和本文档！
