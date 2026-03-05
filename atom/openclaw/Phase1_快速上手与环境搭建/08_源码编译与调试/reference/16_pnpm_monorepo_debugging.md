# pnpm Monorepo 调试技术

> 来源: Grok Web Search
> 查询时间: 2026-02-24

## 搜索结果

### 1. pnpm Workspaces 官方文档
**URL**: https://pnpm.io/workspaces
**描述**: pnpm monorepo 工作区配置与依赖解析指南，含 workspace: 协议及循环依赖排查

### 2. pnpm 同行依赖解析机制
**URL**: https://pnpm.io/how-peers-are-resolved
**描述**: 详解 monorepo 中 peer dependencies 多实例硬链接机制，调试版本冲突关键

### 3. Mastering pnpm Workspaces 2025
**URL**: https://blog.glen-thomas.com/software-engineering/2025/10/02/mastering-pnpm-workspaces-complete-guide-to-monorepo-management.html
**描述**: 2025 年 pnpm 单仓库管理指南，覆盖 phantom deps 与 hoisting 调试技巧

### 4. pnpm require.resolve 工作区问题
**URL**: https://github.com/pnpm/pnpm/issues/5237
**描述**: monorepo 中 require.resolve 无法找到其他 workspace 包的调试案例

### 5. pnpm workspace:* 依赖安装问题
**URL**: https://stackoverflow.com/questions/71378111/pnpm-workspace-dependencies
**描述**: Docker 等环境中 workspace 依赖版本匹配失败 ERR_PNPM_NO_MATCHING_VERSION 解决

### 6. pnpm 工作区包解析完全失效
**URL**: https://github.com/pnpm/pnpm/issues/10173
**描述**: pnpm v10 中 workspace 依赖解析导致 404 错误的最新 issue 分析

### 7. Monorepo Dependency Chaos pnpm
**URL**: https://dev.to/alex_aslam/monorepo-dependency-chaos-proven-hacks-to-keep-your-codebase-sane-and-your-team-happy-1957
**描述**: pnpm 单仓库依赖混乱实用 hacks，处理版本冲突与漂移问题

## 调试技巧

### 1. 依赖解析调试
```bash
# 查看依赖树
pnpm list --depth 1

# 查看特定包的依赖
pnpm why <package-name>

# 检查 workspace 包
pnpm list -r --depth 0
```

### 2. 常见问题排查
- ERR_PNPM_NO_MATCHING_VERSION: 检查 workspace: 协议
- require.resolve 失败: 检查 node_modules 结构
- Phantom dependencies: 使用 --shamefully-hoist=false

**文档版本**: 基于 2026-02-24 搜索结果
