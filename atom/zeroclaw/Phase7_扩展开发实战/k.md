# Phase 7: 扩展开发实战 - 知识点列表

> 目标：动手实现完整的扩展，达到贡献代码的能力
> 学习时长：第 14-15 周
> 前置要求：Phase 1-6 完成

---

## 知识点列表

### 55. 开发环境搭建（从源码）
- git clone、cargo build、feature flags、IDE 配置（VS Code + rust-analyzer）
- 前端类比：git clone + pnpm install + 开发工具配置
- ZeroClaw 场景：从源码编译、调试运行、feature 条件编译

### 56. 测试体系与质量保证
- 单元测试（#[test]）、集成测试（tests/）、Fuzz 测试、Clippy/Rustfmt
- 前端类比：Vitest + Playwright + ESLint/Prettier
- ZeroClaw 场景：测试框架、CI 要求、代码质量标准

### 57. 完整实战：自定义 Provider
- 从零实现一个 LLM Provider（含流式响应）
- 前端类比：封装一个 API SDK
- ZeroClaw 场景：参考 OpenAI-Compatible，实现自定义模型对接

### 58. 完整实战：自定义 Channel
- 从零实现一个消息通道（如 LINE / 微信 / Matrix）
- 前端类比：写一个 WebSocket 服务端
- ZeroClaw 场景：参考 Telegram Channel，实现新的通信平台

### 59. 完整实战：自定义 Tool
- 从零实现一个复杂工具（如数据库查询 / API 集成）
- 前端类比：写一个 CLI 插件
- ZeroClaw 场景：参考 Shell Tool，实现带安全控制的自定义工具

### 60. 完整实战：自定义 Memory 后端
- 实现 Memory trait 对接 Redis / PostgreSQL / 外部向量数据库
- 前端类比：写一个缓存适配器（Redis → localStorage 接口）
- ZeroClaw 场景：参考 SQLite Memory，实现新的存储后端

### 61. Skill 系统与 SOP
- TOML 定义的 Skill、SOP（标准操作流程）、可复用行为模板
- 前端类比：npm scripts 编排 / GitHub Actions workflow
- ZeroClaw 场景：skills/ + sop/ — 声明式的 Agent 行为定义

### 62. Python 伴侣包开发
- zeroclaw_tools Python 包、LangGraph 集成、Discord Bot 示例
- 前端类比：npm 包发布
- ZeroClaw 场景：python/ — 用 Python 调用 ZeroClaw 工具
