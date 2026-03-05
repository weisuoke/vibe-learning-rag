# Phase 8: 源码贡献与高级主题

**目标：** 能够为 OpenClaw 贡献代码

**学习时长：** 第 17-18 周

**知识点数：** 10 个

---

## 知识点列表

### 73. 代码库结构深入分析
- 目录结构
- 模块划分
- 依赖关系
- 架构模式

### 74. Monorepo 架构（pnpm workspaces）
- Workspace 配置
- 包管理
- 依赖共享
- 构建顺序

### 75. 开发环境完整搭建
- 开发工具配置
- IDE 集成
- 调试配置
- 开发工作流

### 76. 测试框架（Vitest、E2E、Docker 测试）
- 单元测试
- E2E 测试
- Docker 测试
- 测试覆盖率
- 源码位置：`vitest.*.config.ts`, `scripts/test-*.sh`

### 77. 代码规范（Oxlint、Oxfmt、TypeScript）
- Oxlint 配置
- Oxfmt 格式化
- TypeScript 严格模式
- 代码审查标准

### 78. 构建系统（tsdown、打包流程）
- tsdown 配置
- 构建脚本
- 打包优化
- 源码位置：`scripts/`

### 79. 发布流程（npm、GitHub Release、Sparkle）
- 版本管理
- npm 发布
- GitHub Release
- Sparkle 更新（macOS）
- 源码位置：`docs/reference/RELEASING.md`

### 80. PR 工作流与代码审查
- PR 模板
- 代码审查流程
- CI/CD 集成
- 源码位置：`.github/pull_request_template.md`, `.agents/skills/PR_WORKFLOW.md`

### 81. 社区贡献指南
- 贡献流程
- Issue 模板
- 社区规范
- 源码位置：`CONTRIBUTING.md`

### 82. 高级调试与性能分析
- 调试技巧
- 性能分析工具
- 内存泄漏检测
- 性能瓶颈定位

---

**验证标准：**
- ✅ 运行完整测试套件
- ✅ 提交一个 PR
- ✅ 通过代码审查
- ✅ 理解发布流程
