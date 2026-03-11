# 原子化知识点生成规范 - ZeroClaw 开发专用

> 本文档定义了为 ZeroClaw（Rust AI Agent 运行时）学习项目生成原子化知识点文档的标准和要求

---

## 文档概述

**项目目标：** 为 ZeroClaw 使用、源码阅读与贡献构建完整的原子化知识体系

**核心理念：**
- **原子化**：每个知识点独立完整，可独立学习
- **全面覆盖**：知识点包含多个子概念时，全部详细讲解，不遗漏
- **实战导向**：所有知识点都要联系 ZeroClaw 开发的实际应用
- **初学者友好**：假设零 Rust 基础，有 TypeScript/前端经验，用简单语言和丰富类比
- **速成高效**：抓住20%核心解决80%问题
- **双重类比**：同时提供前端开发类比 + 日常生活类比
- **Rust 优先**：所有代码示例使用 Rust（附 TypeScript 对照说明）
- **Trait 驱动**：体现 ZeroClaw 的 Trait 驱动可插拔架构设计理念

---

## 模板引用

本文档基于通用原子化知识点模板：**`prompt/atom_template.md`**

ZeroClaw 开发的特殊要求在下方 **ZeroClaw 开发特定配置** 章节中定义。

---

## 生成流程

### 第一步：确认输入信息

在开始生成前，确认以下信息：

1. **知识点名称**：从 `atom/zeroclaw/[Phase]/k.md` 中获取
2. **Phase 目录**：如 `Phase1_Rust速成基础`、`Phase2_ZeroClaw快速上手` 等
3. **目标受众**：有 TypeScript/前端基础，零 Rust 基础的开发者
4. **文件位置**：`atom/zeroclaw/[Phase]/[编号]_[知识点名称]/`

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

参考 `prompt/atom_template.md` 的详细规范，结合下方的 ZeroClaw 开发特定配置生成内容。

### 第四步：质量检查

使用 `prompt/atom_template.md` 中的检查清单验证质量。

---

## ZeroClaw 开发特定配置

### 应用场景强调

**每个部分都要联系 ZeroClaw 开发实际应用：**
- ✅ 这个知识在 ZeroClaw 开发中如何体现？
- ✅ 为什么 ZeroClaw 开发需要这个？
- ✅ 实际场景举例（Agent 对话、多通道消息、工具执行、内存检索、安全控制）

**重点强调：**
- Trait 驱动架构与可插拔设计
- Agent 核心循环（ReAct 模式）
- Provider/Channel/Tool/Memory 四大子系统
- 安全模型与沙箱执行
- 性能优化（<5MB RAM, <10ms 启动）
- 与 OpenClaw 的对比和迁移

### ZeroClaw 类比对照表

在【双重类比】维度中，优先使用以下类比：

| ZeroClaw 概念 | 前端类比 | 日常生活类比 |
|---------------|---------|-------------|
| **Rust 基础** |
| 所有权 (Ownership) | 变量赋值后原值不可用 | 钥匙只能一个人拿 |
| 借用 (Borrowing) | const 引用 vs let 引用 | 借书（只读 vs 可批注） |
| Trait | TypeScript interface | 职业资格证（有证才能做） |
| Trait Object (`Box<dyn T>`) | 依赖注入容器 | 万能插座适配器 |
| Result/Option | try/catch + optional chaining | 快递签收（成功/拒收/空包裹） |
| async/await | JS Promise + Event Loop | 餐厅点菜后等叫号 |
| Cargo | npm/pnpm | 工具箱 + 说明书 |
| **核心架构** |
| Agent 循环 | Redux dispatch 循环 | 客服接线员处理流程 |
| AgentBuilder | React Context Provider | 组装电脑选配件 |
| Provider Trait | fetch API 适配器 | 翻译官（统一接口，多种语言） |
| Channel Trait | WebSocket 连接抽象 | 不同品牌的对讲机 |
| Tool Trait | React Hook 接口 | 瑞士军刀的刀片 |
| Memory Trait | localStorage + 搜索引擎 | 图书馆的索引系统 |
| ToolDispatcher | Redux middleware | 快递分拣中心 |
| config.toml | next.config.js | 汽车仪表盘设置 |
| **安全与运维** |
| SecurityPolicy | CORS + CSP | 门禁系统 |
| 配对认证 | OAuth 扫码登录 | 配对蓝牙耳机 |
| 沙箱执行 | iframe sandbox | 防爆实验室 |
| Gateway | Express + nginx | 酒店前台 |
| Daemon | pm2 进程管理 | 24小时值班保安 |
| Tunnel | ngrok 内网穿透 | 地下隧道直通 |
| **高级功能** |
| AIEOS 身份 | 用户 Profile 主题 | AI 的身份证 |
| Skill/SOP | npm scripts 编排 | 标准操作手册 |
| 混合检索 | Algolia 搜索 | 同时翻目录和搜关键词 |

### 推荐库列表

在【实战代码】维度中，优先使用以下库：

| 用途 | 推荐库 |
|------|--------|
| **异步运行时** | `tokio` |
| **HTTP 客户端** | `reqwest` (rustls) |
| **序列化** | `serde`, `serde_json`, `toml` |
| **错误处理** | `anyhow`, `thiserror` |
| **CLI 解析** | `clap` |
| **日志/追踪** | `tracing`, `tracing-subscriber` |
| **数据库** | `rusqlite` |
| **异步 Trait** | `async-trait` |
| **UUID** | `uuid` |
| **正则** | `regex` |

### ZeroClaw 常见误区

在【反直觉点】维度中，可参考以下常见误区：

**Rust 误区（Phase 1）：**
- "Rust 的所有权太难了，不适合初学者"（其实只需理解 3 条规则）
- "async/await 和 JavaScript 的完全一样"（Rust 的是惰性的，不调用不执行）
- "Trait 就是 interface"（Trait 可以有默认实现、可以为外部类型实现）

**架构误区：**
- "ZeroClaw 只是 OpenClaw 的 Rust 翻译"（全新的 Trait 驱动设计）
- "轻量意味着功能少"（70+ 个模块，功能比 OpenClaw 更全）
- "Rust 项目很难贡献"（实现一个 Trait 就是一个贡献）
- "单一二进制意味着不可扩展"（通过 Trait 实现完全可插拔）

**使用误区：**
- "必须用强大的硬件"（$10 硬件即可运行）
- "只支持 Anthropic 模型"（22+ Provider，含本地 Ollama）
- "内存系统需要外部向量数据库"（SQLite 内置向量+全文混合检索）
- "安全配置很复杂"（secure-by-default，零配置即安全）

---

## Rust 环境配置

### 环境要求

- **Rust 版本**: stable 最新版（通过 rustup 管理）
- **构建工具**: cargo
- **异步运行时**: tokio

### 快速开始

```bash
# 1. 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 2. 安装 ZeroClaw
cargo install zeroclaw
# 或使用一键脚本
curl -fsSL https://zeroclawlabs.ai/install.sh | bash

# 3. 首次配置
zeroclaw onboard

# 4. 运行 Agent
zeroclaw agent --message "Hello!"

# 5. 从源码构建
cd sourcecode/zeroclaw
cargo build --release
```

### 源码开发环境

```bash
# 1. 进入源码目录
cd sourcecode/zeroclaw

# 2. 构建（调试模式）
cargo build

# 3. 运行测试
cargo test

# 4. 运行特定模块测试
cargo test --lib providers

# 5. 代码检查
cargo clippy
cargo fmt --check

# 6. 从源码运行
cargo run -- agent --message "Hello!"
```

### 可用的库

所有代码示例可以使用以下库（来自 ZeroClaw 的 Cargo.toml）：

| 用途 | 库名 |
|------|------|
| **异步运行时** | `tokio` |
| **HTTP 客户端** | `reqwest` |
| **序列化** | `serde`, `serde_json`, `toml` |
| **错误处理** | `anyhow` |
| **CLI** | `clap` |
| **日志** | `tracing` |
| **数据库** | `rusqlite` |
| **异步 Trait** | `async-trait` |
| **工具** | `uuid`, `regex`, `chrono` |

### 配置 API 密钥

1. 运行 onboarding wizard：
   ```bash
   zeroclaw onboard
   ```

2. 或手动配置 `~/.zeroclaw/config.toml`：
   ```toml
   [provider]
   name = "ollama"  # 本地模型，免费
   model = "llama3"

   # 或使用 OpenRouter
   # name = "openrouter"
   # api_key = "sk-..."
   # model = "anthropic/claude-sonnet-4"
   ```

3. 环境变量方式：
   ```bash
   export OPENAI_API_KEY=sk-...
   export ANTHROPIC_API_KEY=sk-ant-...
   ```

---

## 文件组织规范

### 文件命名

**格式：** `[编号]_[知识点名称]/`（目录形式）

**示例：**
```
atom/zeroclaw/Phase1_Rust速成基础/01_所有权与借用/
atom/zeroclaw/Phase1_Rust速成基础/02_Struct与Enum/
atom/zeroclaw/Phase2_ZeroClaw快速上手/09_ZeroClaw安装与环境配置/
atom/zeroclaw/Phase3_核心架构理解/17_源码目录结构总览/
```

**编号规则：**
- 按学习顺序编号（01, 02, 03...）
- 跨 Phase 连续编号（Phase1: 01-08, Phase2: 09-16, ...）
- 反映知识点的依赖关系
- 与 `k.md` 中的顺序一致

### 目录结构

```
atom/
└── zeroclaw/                              # ZeroClaw 学习路径（8个阶段）
    ├── CLAUDE_ZEROCLAW.md                 # ZeroClaw 特定配置文档
    ├── Phase1_Rust速成基础/
    │   ├── k.md                           # 知识点列表
    │   ├── 01_所有权与借用/
    │   ├── 02_Struct与Enum/
    │   ├── 03_Trait与泛型/
    │   ├── 04_错误处理Result_Option/
    │   ├── 05_动态分发与Trait_Object/
    │   ├── 06_async_await与Tokio/
    │   ├── 07_Cargo与模块系统/
    │   └── 08_常用库速查/
    │
    ├── Phase2_ZeroClaw快速上手/
    │   ├── k.md
    │   ├── 09_ZeroClaw安装与环境配置/
    │   ├── 10_Onboarding_Wizard与首次配置/
    │   ├── 11_CLI基础命令/
    │   ├── 12_第一个Agent对话/
    │   ├── 13_Provider配置与切换/
    │   ├── 14_Channel配对Telegram实战/
    │   ├── 15_Gateway启动与管理/
    │   └── 16_基础故障排查/
    │
    ├── Phase3_核心架构理解/
    │   ├── k.md
    │   ├── 17_源码目录结构总览/
    │   ├── 18_Trait驱动架构设计/
    │   ├── 19_AgentBuilder与构建器模式/
    │   ├── 20_Agent核心循环ReAct/
    │   ├── 21_消息模型与数据流/
    │   ├── 22_配置系统详解/
    │   ├── 23_模块注册与工厂模式/
    │   ├── 24_错误传播与anyhow/
    │   ├── 25_日志与可观测性/
    │   └── 26_入口分析main_rs解读/
    │
    ├── Phase4_Provider与Channel系统/
    │   ├── k.md
    │   ├── 27_Provider_Trait详解/
    │   ├── 28_OpenAI_Compatible实现/
    │   ├── 29_Ollama本地模型集成/
    │   ├── 30_流式响应处理/
    │   ├── 31_Provider可靠性与降级/
    │   ├── 32_Channel_Trait详解/
    │   ├── 33_Telegram_Channel实现/
    │   ├── 34_Discord_Channel实现/
    │   ├── 35_通道路由与消息分发/
    │   └── 36_自定义Provider_Channel开发/
    │
    ├── Phase5_工具与内存系统/
    │   ├── k.md
    │   ├── 37_Tool_Trait详解/
    │   ├── 38_ToolDispatcher与工具调度/
    │   ├── 39_Shell工具实现/
    │   ├── 40_文件操作工具/
    │   ├── 41_HTTP与Browser工具/
    │   ├── 42_自定义Tool开发/
    │   ├── 43_Memory_Trait详解/
    │   ├── 44_SQLite混合检索引擎/
    │   ├── 45_Markdown_Memory后端/
    │   └── 46_Memory上下文管理与裁剪/
    │
    ├── Phase6_安全与运维/
    │   ├── k.md
    │   ├── 47_SecurityPolicy_Trait与安全模型/
    │   ├── 48_配对认证机制/
    │   ├── 49_沙箱执行/
    │   ├── 50_Secret_Store与密钥管理/
    │   ├── 51_Gateway架构与Webhook/
    │   ├── 52_Daemon守护进程/
    │   ├── 53_Cron调度系统/
    │   └── 54_隧道与远程访问/
    │
    ├── Phase7_扩展开发实战/
    │   ├── k.md
    │   ├── 55_开发环境搭建/
    │   ├── 56_测试体系与质量保证/
    │   ├── 57_完整实战_自定义Provider/
    │   ├── 58_完整实战_自定义Channel/
    │   ├── 59_完整实战_自定义Tool/
    │   ├── 60_完整实战_自定义Memory后端/
    │   ├── 61_Skill系统与SOP/
    │   └── 62_Python伴侣包开发/
    │
    └── Phase8_高级主题与社区贡献/
        ├── k.md
        ├── 63_AIEOS身份系统/
        ├── 64_RAG能力集成/
        ├── 65_Web_Dashboard/
        ├── 66_硬件与固件集成/
        ├── 67_性能优化与Benchmark/
        ├── 68_CI_CD与发布流程/
        ├── 69_PR工作流与代码审查/
        └── 70_社区参与与生态贡献/
```

---

## 快速启动模板

### 生成新知识点的步骤

1. **读取通用模板** (`prompt/atom_template.md`)
2. **读取本文档** (`atom/zeroclaw/CLAUDE_ZEROCLAW.md`) - ZeroClaw 特定配置
3. **读取知识点列表** (`atom/zeroclaw/[Phase]/k.md`)
4. **确认目标知识点**（第几个）
5. **读取 ZeroClaw 源码**（`sourcecode/zeroclaw/src/` 相关模块）
6. **使用 Grok-mcp 调研最新知识**（确保 2026 年最新）
7. **按规范生成内容**（10个维度）
8. **质量检查**（使用检查清单）
9. **保存文件**（`atom/zeroclaw/[Phase]/[编号]_[知识点]/`）

### 提示词模板

```
根据 @prompt/atom_template.md 的通用规范和 @atom/zeroclaw/CLAUDE_ZEROCLAW.md 的 ZeroClaw 特定配置，为 @atom/zeroclaw/[Phase]/k.md 中的第[N]个知识点 "[知识点名称]" 生成一个完整的学习文档。

要求：
- 按照10个维度完整生成
- 零 Rust 基础的前端开发者友好
- 代码可运行（Rust，附 TypeScript 对照）
- 双重类比（前端开发 + 日常生活）
- 与 ZeroClaw 源码紧密结合
- 体现 Trait 驱动架构设计理念
- 使用 Grok-mcp 确保 2026 年最新知识

文件保存到：atom/zeroclaw/[Phase]/[编号]_[知识点名称]/
```

---

## 特殊要求：Rust 代码示例

### Phase 1 代码规范

Phase 1 的 Rust 速成知识点需要特殊处理：

```rust
// ===== Rust 代码 =====
// 用途：展示所有权转移
fn take_ownership(s: String) {
    println!("{}", s);
} // s 在这里被释放

let name = String::from("ZeroClaw");
take_ownership(name);
// println!("{}", name); // ❌ 编译错误！所有权已转移
```

```typescript
// ===== TypeScript 对照 =====
// 同样的逻辑在 TS 中：
function takeOwnership(s: string) {
    console.log(s);
}

const name = "ZeroClaw";
takeOwnership(name);
console.log(name); // ✅ 正常！JS 没有所有权概念
```

### Phase 2-8 代码规范

从 Phase 2 开始，代码示例以 ZeroClaw 源码风格为主：

```rust
use async_trait::async_trait;
use anyhow::Result;

/// 自定义 Provider 示例
/// 对应源码：sourcecode/zeroclaw/src/providers/
#[async_trait]
impl Provider for MyProvider {
    async fn chat(&self, messages: &[Message]) -> Result<String> {
        // 实现细节
        Ok("Hello from MyProvider!".to_string())
    }
}
```

---

## 核心原则总结

1. **原子化**：每个知识点独立完整
2. **全面覆盖**：知识点所有子概念都要讲到
3. **实战导向**：联系 ZeroClaw 源码和开发应用
4. **前端友好**：用前端类比解释 Rust 和 ZeroClaw 概念
5. **速成高效**：20%核心 + 80%效果
6. **代码可运行**：所有示例都能跑（Rust + TypeScript 对照）
7. **体系完整**：10个维度全覆盖
8. **质量保证**：严格检查清单
9. **Trait 驱动**：体现 ZeroClaw 可插拔架构设计理念
10. **源码关联**：每个知识点标注对应的源码位置

---

## 实战案例资源

### 官方资源
- **GitHub**: https://github.com/zeroclaw-labs/zeroclaw
- **官网**: https://zeroclawlabs.ai
- **文档**: sourcecode/zeroclaw/docs/

### 源码关键路径
- **入口**: `sourcecode/zeroclaw/src/main.rs`
- **Agent 核心**: `sourcecode/zeroclaw/src/agent/`
- **Provider**: `sourcecode/zeroclaw/src/providers/`
- **Channel**: `sourcecode/zeroclaw/src/channels/`
- **Tool**: `sourcecode/zeroclaw/src/tools/`
- **Memory**: `sourcecode/zeroclaw/src/memory/`
- **Security**: `sourcecode/zeroclaw/src/security/`
- **Config**: `sourcecode/zeroclaw/src/config/`
- **Web Dashboard**: `sourcecode/zeroclaw/web/`
- **Python 包**: `sourcecode/zeroclaw/python/`
- **示例代码**: `sourcecode/zeroclaw/examples/`

---

**版本：** v1.0 (ZeroClaw 开发专用版 - 基于通用模板)
**最后更新：** 2026-03-09
**维护者：** Claude Code

---

**记住：** 生成每个新知识点前，先读取 `prompt/atom_template.md` 和本文档！
