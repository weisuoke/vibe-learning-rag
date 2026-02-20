# 原子化知识点生成规范 - Pi-mono 开发专用

> 本文档定义了为 Pi-mono AI Agent 工具包学习项目生成原子化知识点文档的标准和要求

---

## 文档概述

**项目目标：** 为 Pi-mono AI Agent 开发学习构建完整的原子化知识体系

**核心理念：**
- **原子化**：每个知识点独立完整，可独立学习
- **全面覆盖**：知识点包含多个子概念时，全部详细讲解，不遗漏
- **实战导向**：所有知识点都要联系 AI Agent 开发的实际应用
- **初学者友好**：假设零基础，用简单语言和丰富类比
- **速成高效**：抓住20%核心解决80%问题
- **双重类比**：同时提供 TypeScript/Node.js 类比 + 日常生活类比
- **TypeScript 优先**：所有代码示例使用 TypeScript/JavaScript
- **极简哲学**：体现 pi-mono 的极简设计理念

---

## 模板引用

本文档基于通用原子化知识点模板：**`prompt/atom_template.md`**

Pi-mono 开发的特殊要求在下方 **Pi-mono 开发特定配置** 章节中定义。

---

## 生成流程

### 第一步：确认输入信息

在开始生成前，确认以下信息：

1. **知识点名称**：从 `atom/pi-mono/[Phase]/k.md` 中获取
2. **Phase 目录**：如 `Phase1_快速上手与基础使用`、`Phase2_核心架构理解` 等
3. **目标受众**：有 TypeScript/Node.js 基础的开发者
4. **文件位置**：`atom/pi-mono/[Phase]/[编号]_[知识点名称]/`

### 第二步：读取模板

**通用模板：** `prompt/atom_template.md` - 定义10个维度的标准结构

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

参考 `prompt/atom_template.md` 的详细规范，结合下方的 Pi-mono 开发特定配置生成内容。

### 第四步：质量检查

使用 `prompt/atom_template.md` 中的检查清单验证质量。

---

## Pi-mono 开发特定配置

### 应用场景强调

**每个部分都要联系 AI Agent 开发实际应用：**
- ✅ 这个知识在 AI Agent 开发中如何体现？
- ✅ 为什么 AI Agent 开发需要这个？
- ✅ 实际场景举例（Coding Agent、Slack Bot、多 Agent 协作、自定义工具）

**重点强调：**
- Agent 运行时与工具调用
- LLM API 统一抽象
- Session 管理与状态持久化
- 扩展性与定制化（Extensions、Skills、Prompt Templates）
- 极简设计理念

### Pi-mono 类比对照表

在【双重类比】维度中，优先使用以下类比：

| Pi-mono 概念 | TypeScript/Node.js 类比 | 日常生活类比 |
|-------------|------------------------|--------------|
| **核心架构** |
| Pi AI 统一 API | Axios 拦截器统一 HTTP 请求 | 翻译器统一不同语言 |
| Agent Core | Express 中间件链 | 流水线工人 |
| Tool 工具 | REST API 端点 | 工具箱里的工具 |
| Session | Express Session | 聊天记录本 |
| Compaction | 日志轮转 | 整理笔记本 |
| JSONL 存储 | 追加日志文件 | 日记本逐行记录 |
| **定制化** |
| Prompt Template | Handlebars 模板 | 邮件模板 |
| Skill | npm 包 | 技能卡片 |
| Extension | Express 插件 | 浏览器扩展 |
| Event System | EventEmitter | 事件监听器 |
| **高级功能** |
| Provider | 数据库驱动 | 不同品牌的插座 |
| MCP Server | GraphQL 服务 | 外部服务接口 |
| Sub-Agent | 微服务 | 分包商 |
| Sandbox | Docker 容器 | 隔离的实验室 |
| **UI 组件** |
| pi-tui | Ink (React for CLI) | 终端界面 |
| Diff Rendering | Virtual DOM | 只更新变化部分 |
| Widget | React Component | UI 组件 |

### 推荐库列表

在【实战代码】维度中，优先使用以下库：

| 用途 | 推荐库 |
|------|--------|
| **开发环境** | Node.js 18+, npm, TypeScript |
| **包管理** | npm workspaces |
| **测试** | Jest, Vitest |
| **类型检查** | TypeScript, tsc |
| **代码规范** | ESLint, Prettier |
| **调试** | VS Code Debugger |
| **LLM API** | OpenAI SDK, Anthropic SDK |
| **终端 UI** | Ink (React for CLI) |
| **Web UI** | React, Lit |

### Pi-mono 常见误区

在【反直觉点】维度中，可参考以下常见误区：

**使用误区：**
- "Pi 只是 Claude Code 的简化版"（Pi 更极简但更可扩展）
- "Session 分支会创建新文件"（单文件树形结构）
- "Compaction 会丢失历史"（JSONL 保留完整历史）
- "Extensions 需要重启 pi"（/reload 热重载）

**架构误区：**
- "Pi AI 只支持 OpenAI 格式"（支持多种 API 格式）
- "工具调用是同步的"（可以异步）
- "Session 存储在内存中"（JSONL 文件持久化）
- "UI 组件必须用 React"（pi-tui 是自定义渲染）

**开发误区：**
- "Extension 必须用 JavaScript"（TypeScript 优先）
- "Skills 只能放在全局目录"（项目 .pi/skills/ 也可以）
- "自定义 Provider 需要修改源码"（models.json 配置）
- "MCP 集成很复杂"（Extension API 简化集成）

---

## TypeScript/Node.js 环境配置

### 环境要求

- **Node.js 版本**: 18+ (推荐 20+)
- **包管理器**: npm
- **TypeScript**: 5.0+
- **操作系统**: macOS, Linux, Windows (WSL)

### 快速开始

```bash
# 1. 全局安装 pi
npm install -g @mariozechner/pi-coding-agent

# 2. 配置 API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. 运行 pi
pi

# 4. 或使用订阅登录
pi
/login
```

### 开发环境搭建

```bash
# 1. 克隆仓库
git clone https://github.com/badlogic/pi-mono.git
cd pi-mono

# 2. 安装依赖
npm install

# 3. 构建所有包
npm run build

# 4. 运行测试
./test.sh

# 5. 从源码运行 pi
./pi-test.sh
```

### 可用的库

所有代码示例可以使用以下库：

| 用途 | 库名 |
|------|------|
| **LLM 调用** | `@anthropic-ai/sdk`, `openai` |
| **Agent 框架** | `@mariozechner/pi-agent-core` |
| **终端 UI** | `@mariozechner/pi-tui` |
| **Web UI** | `@mariozechner/pi-web-ui` |
| **工具** | `zod` (schema validation) |

### 环境管理

```bash
# 添加新依赖
npm install <package-name>

# 添加开发依赖
npm install --save-dev <package-name>

# 更新依赖
npm update
```

### 配置 API 密钥

1. 设置环境变量：
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   export OPENAI_API_KEY=sk-...
   ```

2. 或使用 pi 内置登录：
   ```bash
   pi
   /login
   ```

3. 或在项目中配置 `.pi/settings.json`：
   ```json
   {
     "provider": "anthropic",
     "model": "claude-opus-4"
   }
   ```

---

## 文件组织规范

### 文件命名

**格式：** `[编号]_[知识点名称]/`（目录形式）

**示例：**
```
atom/pi-mono/Phase1_快速上手与基础使用/01_Pi_Coding_Agent安装与配置/
atom/pi-mono/Phase1_快速上手与基础使用/02_交互模式与基础命令/
atom/pi-mono/Phase2_核心架构理解/07_Pi_AI统一LLM_API设计/
atom/pi-mono/Phase3_定制化开发/13_Prompt_Templates模板系统/
```

**编号规则：**
- 按学习顺序编号（01, 02, 03...）
- 反映知识点的依赖关系
- 与 `k.md` 中的顺序一致
- 跨 Phase 连续编号（Phase1: 01-06, Phase2: 07-12, Phase3: 13-18...）

### 目录结构

```
atom/
└── pi-mono/                                # Pi-mono 学习路径（6个阶段）
    ├── CLAUDE_PI_MONO.md                   # Pi-mono 特定配置文档
    ├── Phase1_快速上手与基础使用/
    │   ├── k.md                            # 知识点列表
    │   ├── 01_Pi_Coding_Agent安装与配置/
    │   │   ├── 00_概览.md
    │   │   ├── 01_30字核心.md
    │   │   ├── 02_第一性原理.md
    │   │   ├── 03_核心概念_01_npm全局安装.md
    │   │   ├── 03_核心概念_02_Provider认证.md
    │   │   ├── 03_核心概念_03_首次运行.md
    │   │   ├── 04_最小可用.md
    │   │   ├── 05_双重类比.md
    │   │   ├── 06_反直觉点.md
    │   │   ├── 07_实战代码_01_基础安装.md
    │   │   ├── 07_实战代码_02_配置Provider.md
    │   │   ├── 07_实战代码_03_项目配置.md
    │   │   ├── 08_面试必问.md
    │   │   ├── 09_化骨绵掌.md
    │   │   └── 10_一句话总结.md
    │   ├── 02_交互模式与基础命令/
    │   ├── 03_Provider与Model切换/
    │   ├── 04_Session管理与分支/
    │   ├── 05_Context_Files与项目配置/
    │   └── 06_基础工具使用/
    │
    ├── Phase2_核心架构理解/
    │   ├── k.md
    │   ├── 07_Pi_AI统一LLM_API设计/
    │   ├── 08_Agent_Core运行时机制/
    │   ├── 09_工具调用与状态管理/
    │   ├── 10_消息队列与流式响应/
    │   ├── 11_Session存储与树形结构/
    │   └── 12_Compaction压缩机制/
    │
    ├── Phase3_定制化开发/
    │   ├── k.md
    │   ├── 13_Prompt_Templates模板系统/
    │   ├── 14_Skills技能包开发/
    │   ├── 15_Extensions扩展开发基础/
    │   ├── 16_自定义工具注册/
    │   ├── 17_UI组件定制/
    │   └── 18_事件系统与钩子/
    │
    ├── Phase4_高级扩展/
    │   ├── k.md
    │   ├── 19_自定义Provider集成/
    │   ├── 20_MCP_Server集成/
    │   ├── 21_Sub_Agents子代理实现/
    │   ├── 22_Plan_Mode计划模式/
    │   ├── 23_权限控制与沙箱/
    │   └── 24_Git集成与自动提交/
    │
    ├── Phase5_实战项目/
    │   ├── k.md
    │   ├── 25_构建自定义Coding_Agent/
    │   ├── 26_Slack_Bot集成实战/
    │   ├── 27_Web界面集成/
    │   ├── 28_多Agent协作系统/
    │   ├── 29_生产环境部署/
    │   └── 30_性能优化与调试/
    │
    └── Phase6_贡献与优化/
        ├── k.md
        ├── 31_Monorepo架构深入/
        ├── 32_测试与质量保证/
        ├── 33_代码规范与Linting/
        ├── 34_贡献代码流程/
        ├── 35_社区最佳实践/
        └── 36_Pi_Packages发布/
```

---

## 快速启动模板

### 生成新知识点的步骤

1. **读取通用模板** (`prompt/atom_template.md`)
2. **读取本文档** (`atom/pi-mono/CLAUDE_PI_MONO.md`) - Pi-mono 特定配置
3. **读取知识点列表** (`atom/pi-mono/[Phase]/k.md`)
4. **确认目标知识点**（第几个）
5. **按规范生成内容**（10个维度）
6. **质量检查**（使用检查清单）
7. **保存文件**（`atom/pi-mono/[Phase]/[编号]_[知识点]/`）

### 提示词模板

```
根据 @prompt/atom_template.md 的通用规范和 @atom/pi-mono/CLAUDE_PI_MONO.md 的 Pi-mono 特定配置，为 @atom/pi-mono/[Phase]/k.md 中的第[N]个知识点 "[知识点名称]" 生成一个完整的学习文档。

要求：
- 按照10个维度完整生成
- 有 TypeScript/Node.js 基础的开发者友好
- 代码可运行（TypeScript/JavaScript）
- 双重类比（TypeScript/Node.js + 日常生活）
- 与 AI Agent 开发紧密结合
- 体现 pi-mono 的极简设计理念

文件保存到：atom/pi-mono/[Phase]/[编号]_[知识点名称]/
```

---

## 核心原则总结

1. **原子化**：每个知识点独立完整
2. **全面覆盖**：知识点所有子概念都要讲到
3. **实战导向**：联系 AI Agent 开发应用
4. **TypeScript 友好**：简单语言 + 双重类比（TypeScript/Node.js + 日常生活）
5. **速成高效**：20%核心 + 80%效果
6. **代码可运行**：所有示例都能跑（TypeScript/JavaScript）
7. **体系完整**：10个维度全覆盖
8. **质量保证**：严格检查清单
9. **极简哲学**：体现 pi-mono 的极简设计理念
10. **可扩展性**：强调 Extensions、Skills、Prompt Templates 的扩展能力

---

## 学习路径设计

### 用户背景

- **技术背景**：TypeScript 和 Node.js 熟练（日常使用，熟悉 monorepo）
- **LLM 基础**：LLM API 和 AI agent 开发有基础了解
- **主要语言**：主要使用 Python，但对 TypeScript 有扎实基础

### 学习目标

1. **使用者视角** - 把 pi-mono 作为日常编码助手工具使用
2. **架构理解** - 深入理解 pi-mono 的设计思想和技术架构
3. **构建者视角** - 基于 pi-mono 构建自己的 AI agent 应用

### 学习深度

**精通掌握（2-3个月）**

### 6个阶段，36个知识点

| Phase | 知识点数 | 学习时长 | 目标 |
|-------|---------|---------|------|
| Phase 1: 快速上手与基础使用 | 6 | 第 1-2 周 | 能够使用 pi coding agent 进行日常开发 |
| Phase 2: 核心架构理解 | 6 | 第 3-5 周 | 理解 pi-ai 和 pi-agent-core 的设计 |
| Phase 3: 定制化开发 | 6 | 第 6-8 周 | 掌握 Prompt Templates、Skills、Extensions |
| Phase 4: 高级扩展 | 6 | 第 9-10 周 | 实现高级功能（自定义 Provider、MCP、Sub-Agents）|
| Phase 5: 实战项目 | 6 | 第 11 周 | 基于 pi-mono 构建实际应用 |
| Phase 6: 贡献与优化 | 6 | 第 12 周 | 能够为 pi-mono 贡献代码 |

---

**版本：** v1.0 (Pi-mono 开发专用版 - 基于通用模板)
**最后更新：** 2026-02-17
**维护者：** Claude Code

---

**记住：** 生成每个新知识点前，先读取 `prompt/atom_template.md` 和本文档！
