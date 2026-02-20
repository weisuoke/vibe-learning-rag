# Phase 1: 快速上手与基础使用

> 第 1-2 周 | 6个知识点 | 目标：能够使用 pi coding agent 进行日常开发，理解基础工作流

---

## 知识点清单

| 编号 | 知识点名称 | 核心概念 | 应用场景 | 学习时长 |
|------|-----------|---------|---------|---------|
| 01 | Pi Coding Agent 安装与配置 | npm 全局安装、Provider 认证、首次运行 | 环境搭建、API 配置 | 30分钟 |
| 02 | 交互模式与基础命令 | 交互界面、核心命令、编辑器功能 | 日常使用、命令操作 | 1小时 |
| 03 | Provider 与 Model 切换 | Provider 列表、Model 选择、Scoped models | 多模型切换、成本优化 | 45分钟 |
| 04 | Session 管理与分支 | Session 存储、分支导航、Fork 与回溯 | 对话管理、历史追溯 | 1小时 |
| 05 | Context Files 与项目配置 | AGENTS.md、.pi/settings.json、SYSTEM.md | 项目定制、上下文管理 | 45分钟 |
| 06 | 基础工具使用 | read、write、edit、bash 四大工具 | 文件操作、命令执行 | 1小时 |

**总学习时长：** 约 5 小时

---

## 学习顺序建议

### 快速入门路径（2小时）
适合想快速上手的开发者：
1. **01 - Pi Coding Agent 安装与配置**（必学）
2. **02 - 交互模式与基础命令**（必学）
3. **06 - 基础工具使用**（必学）

### 完整学习路径（5小时）
按编号顺序学习所有知识点，建议：
- 第1天：01-03（掌握安装、命令、模型切换）
- 第2天：04-06（掌握 Session、配置、工具）

### 实战项目驱动路径
边学边用，推荐顺序：
1. 01 - 安装配置（立即可用）
2. 02 - 基础命令（日常操作）
3. 06 - 工具使用（实际开发）
4. 04 - Session 管理（需要时学习）
5. 03 - Model 切换（优化成本）
6. 05 - 项目配置（团队协作）

---

## 与 AI Agent 开发的关系

### 使用者视角
- **01-02**：掌握 pi 的基础使用，作为日常编码助手
- **03**：理解不同 LLM 的特点，选择合适的模型
- **04**：管理对话历史，追溯决策过程
- **05-06**：定制项目配置，提升开发效率

### 为后续学习打基础
- **Phase 2**：理解工具调用机制（基于 06）
- **Phase 3**：定制化开发（基于 05）
- **Phase 4**：高级扩展（基于 03）

---

## 学习检查清单

完成本 Phase 后，你应该能够：

### 基础操作
- [ ] 成功安装并运行 pi coding agent
- [ ] 配置至少一个 LLM Provider（Anthropic 或 OpenAI）
- [ ] 使用基础命令（/help、/model、/new、/settings）
- [ ] 在编辑器中引用文件（@file.ts）
- [ ] 切换不同的 Model

### Session 管理
- [ ] 创建新 Session
- [ ] 查看 Session 树形结构（/tree）
- [ ] Fork 到历史分支
- [ ] 理解 JSONL 存储格式

### 项目配置
- [ ] 创建 AGENTS.md 或 CLAUDE.md
- [ ] 配置 .pi/settings.json
- [ ] 理解 Context Files 加载机制

### 工具使用
- [ ] 使用 read 工具读取文件
- [ ] 使用 write 工具创建文件
- [ ] 使用 edit 工具修改文件
- [ ] 使用 bash 工具执行命令
- [ ] 理解工具调用流程

### 实战能力
- [ ] 使用 pi 完成一个简单的编码任务（如修复 bug）
- [ ] 使用 pi 生成代码（如创建新函数）
- [ ] 使用 pi 解释代码（如理解复杂逻辑）

---

## 双重类比速查表

| Pi 概念 | TypeScript/Node.js 类比 | 日常生活类比 |
|---------|------------------------|--------------|
| **安装与配置** |
| npm 全局安装 | `npm install -g` | 安装系统级应用 |
| Provider 认证 | API key 配置 | 登录账号 |
| .pi/settings.json | package.json | 项目配置文件 |
| **交互与命令** |
| 交互界面 | REPL (Node.js) | 对话窗口 |
| /命令 | CLI 命令 | 快捷键 |
| 编辑器 | VS Code 输入框 | 文本编辑器 |
| **Session 管理** |
| Session | Express Session | 聊天记录 |
| JSONL 存储 | 追加日志文件 | 日记本 |
| 分支 | Git 分支 | 平行宇宙 |
| Fork | Git checkout | 时光倒流 |
| **Context Files** |
| AGENTS.md | README.md | 项目说明书 |
| SYSTEM.md | .env | 环境变量 |
| Context 加载 | import 语句 | 读取参考资料 |
| **工具** |
| read 工具 | fs.readFile | 打开文件 |
| write 工具 | fs.writeFile | 保存文件 |
| edit 工具 | 文本替换 | 修改文档 |
| bash 工具 | child_process.exec | 运行命令 |

---

## 常见问题

### Q1: Pi 和 Claude Code 有什么区别？
**A:** Pi 是极简的 AI coding agent，专注核心功能；Claude Code 功能更丰富但更复杂。Pi 更适合快速上手和定制化开发。

### Q2: 必须使用 Anthropic API 吗？
**A:** 不是。Pi 支持多个 Provider（Anthropic、OpenAI、GitHub Copilot、Google 等），也可以使用订阅登录。

### Q3: Session 存储在哪里？
**A:** 默认存储在 `~/.pi/sessions/` 目录，每个 Session 是一个 JSONL 文件。

### Q4: 如何在团队中共享配置？
**A:** 将 `.pi/settings.json`、`AGENTS.md` 等文件提交到 Git 仓库，团队成员克隆后即可使用。

### Q5: 工具调用失败怎么办？
**A:** 检查文件路径、权限、命令语法。使用 Ctrl+O 展开工具输出查看详细错误信息。

---

## 下一步学习

完成 Phase 1 后，你已经掌握了 pi 的基础使用。接下来：

- **Phase 2: 核心架构理解** - 深入理解 pi-ai 和 pi-agent-core 的设计
- **Phase 3: 定制化开发** - 学习 Prompt Templates、Skills、Extensions
- **Phase 4: 高级扩展** - 实现自定义 Provider、MCP 集成、Sub-Agents

---

## 参考资源

- **官方文档**: https://github.com/badlogic/pi-mono
- **Discord 社区**: https://discord.gg/pi-mono
- **示例项目**: https://github.com/badlogic/pi-mono/tree/main/examples
- **API 文档**: https://github.com/badlogic/pi-mono/tree/main/packages

---

**版本：** v1.0
**最后更新：** 2026-02-17
**维护者：** Claude Code
