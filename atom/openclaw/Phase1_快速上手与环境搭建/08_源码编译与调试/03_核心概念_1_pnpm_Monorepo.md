# 核心概念 1：pnpm Monorepo 工作区管理

> OpenClaw 使用 pnpm workspace 管理多包架构

---

## 概念介绍

### 什么是 pnpm Monorepo？

**Monorepo**：在单个代码仓库中管理多个相关包的开发模式。

**pnpm workspace**：pnpm 提供的 monorepo 管理方案，通过符号链接和内容寻址存储实现高效的依赖管理。

**OpenClaw 的 Monorepo 结构**：
```
openclaw/
├── package.json              # 根包（Gateway、CLI）
├── pnpm-workspace.yaml       # workspace 配置
├── ui/                       # UI 包
│   └── package.json
├── packages/                 # 通用包
│   ├── pkg-a/
│   └── pkg-b/
└── extensions/               # 扩展包
    ├── ext-1/
    └── ext-2/
```

---

## pnpm 工作原理

### 1. 内容寻址存储（Content-Addressable Storage）

**传统 npm 的问题**：
```
project1/node_modules/lodash/  (100MB)
project2/node_modules/lodash/  (100MB)
project3/node_modules/lodash/  (100MB)
→ 总共 300MB，重复存储
```

**pnpm 的解决方案**：
```
~/.pnpm-store/
  └── v3/
      └── files/
          └── 00/
              └── 1a2b3c...  (lodash 的实际文件，100MB)

project1/node_modules/
  └── .pnpm/
      └── lodash@4.17.21/
          └── node_modules/
              └── lodash → 硬链接到 store

project2/node_modules/
  └── .pnpm/
      └── lodash@4.17.21/
          └── node_modules/
              └── lodash → 硬链接到 store (共享)
```

**关键机制**：
1. **全局 store**：所有依赖存储在 `~/.pnpm-store/`
2. **硬链接**：从 store 到项目的 `.pnpm/` 目录
3. **符号链接**：从 `node_modules/` 到 `.pnpm/` 目录
4. **内容寻址**：相同内容只存储一次

**优势**：
- **节省空间**：相同依赖只存储一次
- **快速安装**：硬链接比复制快得多
- **严格依赖**：只能访问声明的依赖（避免幽灵依赖）

---

### 2. workspace 协议

**workspace 协议**：在 monorepo 中引用本地包的特殊协议。

**语法**：
```json
{
  "dependencies": {
    "@openclaw/core": "workspace:*",
    "@openclaw/ui": "workspace:^1.0.0"
  }
}
```

**协议类型**：
- `workspace:*`：使用当前版本
- `workspace:^`：使用兼容版本
- `workspace:~`：使用补丁版本

**OpenClaw 的使用**：
```json
// ui/package.json
{
  "dependencies": {
    "openclaw": "workspace:*"  // 引用根包
  }
}
```

**发布时的处理**：
```json
// 发布前
"dependencies": {
  "openclaw": "workspace:*"
}

// 发布后
"dependencies": {
  "openclaw": "2026.2.22"  // 自动替换为实际版本
}
```

---

### 3. 依赖提升（Hoisting）

**依赖提升**：将共享依赖提升到根 `node_modules/`，减少重复安装。

**pnpm 的策略**：
```
openclaw/
├── node_modules/
│   ├── .pnpm/                    # 实际依赖存储
│   │   ├── lodash@4.17.21/
│   │   ├── react@18.2.0/
│   │   └── typescript@5.9.3/
│   ├── lodash → .pnpm/lodash@4.17.21/node_modules/lodash
│   ├── react → .pnpm/react@18.2.0/node_modules/react
│   └── typescript → .pnpm/typescript@5.9.3/node_modules/typescript
├── ui/
│   └── node_modules/
│       └── react → ../.pnpm/react@18.2.0/node_modules/react
└── packages/
    └── pkg-a/
        └── node_modules/
            └── lodash → ../../.pnpm/lodash@4.17.21/node_modules/lodash
```

**配置选项**：
```yaml
# .npmrc
shamefully-hoist=false  # 禁用提升（严格模式）
public-hoist-pattern[]=*eslint*  # 提升特定包
```

---

## OpenClaw 的 Monorepo 配置

### 1. pnpm-workspace.yaml

```yaml
packages:
  - .           # 根包（Gateway、CLI）
  - ui          # UI 包
  - packages/*  # 通用包
  - extensions/* # 扩展包

onlyBuiltDependencies:
  - "@lydell/node-pty"
  - "@matrix-org/matrix-sdk-crypto-nodejs"
  - "@napi-rs/canvas"
  - "@whiskeysockets/baileys"
  - "authenticate-pam"
  - "esbuild"
  - "node-llama-cpp"
  - "protobufjs"
  - "sharp"
```

**关键配置**：
- **packages**：指定工作区包的目录模式
- **onlyBuiltDependencies**：仅编译这些原生依赖（优化安装性能）

---

### 2. package.json 配置

```json
{
  "packageManager": "pnpm@10.23.0",
  "pnpm": {
    "minimumReleaseAge": 2880,
    "overrides": {
      "hono": "4.11.10",
      "fast-xml-parser": "5.3.6",
      "minimatch": "10.2.1",
      "qs": "6.14.2",
      "@sinclair/typebox": "0.34.48",
      "tar": "7.5.9",
      "tough-cookie": "4.1.3"
    },
    "onlyBuiltDependencies": [
      "@lydell/node-pty",
      "@napi-rs/canvas",
      "sharp",
      "node-llama-cpp",
      "esbuild"
    ]
  }
}
```

**关键配置**：
- **packageManager**：锁定 pnpm 版本
- **minimumReleaseAge**：最小发布年龄（分钟）
- **overrides**：统一依赖版本、修复安全漏洞
- **onlyBuiltDependencies**：优化原生依赖编译

---

## 常用命令

### 1. 安装依赖

```bash
# 安装所有依赖
pnpm install

# CI 环境（冻结 lockfile）
pnpm install --frozen-lockfile

# 仅生产依赖
pnpm install --prod

# 递归安装（所有工作区）
pnpm install -r

# 跳过可选依赖
pnpm install --no-optional

# 离线安装（使用缓存）
pnpm install --prefer-offline
```

---

### 2. 添加依赖

```bash
# 添加到根包
pnpm add lodash

# 添加到特定包
pnpm --filter @openclaw/ui add react

# 添加到所有包
pnpm -r add lodash

# 添加工作区依赖
pnpm --filter @openclaw/ui add openclaw@workspace:*

# 添加开发依赖
pnpm add -D typescript

# 添加全局依赖
pnpm add -g openclaw
```

---

### 3. 运行脚本

```bash
# 运行根包脚本
pnpm build

# 运行所有包脚本
pnpm -r build

# 并行运行
pnpm -r --parallel dev

# 顺序运行
pnpm -r --sequential build

# 过滤运行
pnpm --filter "./packages/**" build

# 运行特定包
pnpm --filter @openclaw/ui build

# 从特定包恢复
pnpm -r --resume-from @openclaw/pkg-c build
```

---

### 4. 更新依赖

```bash
# 更新所有依赖
pnpm update

# 更新特定依赖
pnpm update lodash

# 更新到最新版本
pnpm update --latest

# 递归更新
pnpm -r update

# 交互式更新
pnpm update --interactive
```

---

### 5. 查看依赖

```bash
# 查看依赖树
pnpm list --depth 1

# 查看特定包的依赖
pnpm why lodash

# 查看工作区包
pnpm list -r --depth 0

# 查看过滤结果
pnpm --filter "./packages/**" list

# 查看过时依赖
pnpm outdated
```

---

## 调试技巧

### 1. 依赖解析调试

```bash
# 查看依赖树
pnpm list --depth 1

# 查看特定包的依赖
pnpm why <package-name>

# 检查 workspace 包
pnpm list -r --depth 0

# 查看依赖路径
pnpm list --depth Infinity | grep <package-name>
```

---

### 2. 常见问题排查

**问题 1：ERR_PNPM_NO_MATCHING_VERSION**

```bash
# 错误信息
ERR_PNPM_NO_MATCHING_VERSION  No matching version found for openclaw@workspace:*

# 解决方案
# 1. 检查 workspace 协议
pnpm list -r --depth 0

# 2. 重新安装
pnpm install --force

# 3. 清理缓存
pnpm store prune
pnpm install
```

---

**问题 2：require.resolve 失败**

```bash
# 错误信息
Error: Cannot find module '@openclaw/core'

# 解决方案
# 1. 检查 node_modules 结构
ls -la node_modules/.pnpm/

# 2. 重新链接
pnpm install --force

# 3. 检查 workspace 配置
cat pnpm-workspace.yaml
```

---

**问题 3：Phantom dependencies（幽灵依赖）**

```bash
# 问题：代码中使用了未声明的依赖

# 解决方案
# 1. 启用严格模式
echo "shamefully-hoist=false" >> .npmrc

# 2. 重新安装
pnpm install

# 3. 添加缺失的依赖
pnpm add <missing-package>
```

---

### 3. 性能优化

```bash
# 使用缓存
pnpm install --prefer-offline

# 跳过可选依赖
pnpm install --no-optional

# 仅安装生产依赖
pnpm install --prod

# 并行构建
pnpm -r --parallel build

# 清理未使用的依赖
pnpm store prune
```

---

## OpenClaw 特定用法

### 1. 构建流程

```bash
# 完整构建
pnpm build

# 开发模式
pnpm dev

# Gateway 开发
pnpm gateway:dev

# Gateway 监听
pnpm gateway:watch
```

---

### 2. 测试流程

```bash
# 所有测试
pnpm test

# 单元测试
pnpm test:fast

# E2E 测试
pnpm test:e2e

# Live 测试
pnpm test:live

# 覆盖率
pnpm test:coverage
```

---

### 3. 代码质量

```bash
# 类型检查 + lint + 格式检查
pnpm check

# 格式化
pnpm format

# Lint
pnpm lint

# 类型检查
pnpm typecheck
```

---

## 最佳实践

### 1. workspace 结构设计

```yaml
packages:
  - 'packages/*'    # 共享库
  - 'apps/*'        # 应用
  - 'tools/*'       # 工具
  - '!**/test/**'   # 排除测试
  - '!**/dist/**'   # 排除构建产物
```

---

### 2. 依赖管理策略

```json
{
  "pnpm": {
    "overrides": {
      "vulnerable-package": ">=safe-version"
    }
  }
}
```

---

### 3. 原生依赖优化

```yaml
onlyBuiltDependencies:
  - "esbuild"
  - "sharp"
  - "@napi-rs/canvas"
```

---

### 4. CI/CD 配置

```yaml
# .github/workflows/ci.yml
- name: Install dependencies
  run: pnpm install --frozen-lockfile

- name: Build
  run: pnpm -r build

- name: Test
  run: pnpm test
```

---

### 5. 脚本执行策略

```json
{
  "scripts": {
    "build": "pnpm -r --sequential build",
    "dev": "pnpm -r --parallel dev",
    "test": "pnpm -r test"
  }
}
```

---

## 与 OpenClaw 的关系

### 1. 为什么 OpenClaw 使用 pnpm monorepo？

**问题**：OpenClaw 有多个包需要管理
- 核心包（Gateway、CLI）
- UI 包（Web 界面）
- 通用包（工具库）
- 扩展包（Extension、Plugin）

**解决方案**：pnpm workspace
- 统一依赖管理
- 版本一致性
- 快速安装（符号链接）
- 节省磁盘空间（内容寻址）

---

### 2. OpenClaw 的包依赖关系

```
openclaw (根包)
  ├── ui (UI 包)
  │   └── depends on: openclaw@workspace:*
  ├── packages/
  │   ├── pkg-a
  │   └── pkg-b
  └── extensions/
      ├── ext-1
      │   └── depends on: openclaw@workspace:*
      └── ext-2
          └── depends on: openclaw@workspace:*
```

---

### 3. OpenClaw 的构建流程

```bash
# 1. 安装依赖
pnpm install

# 2. 构建所有包
pnpm -r build

# 3. 运行测试
pnpm test

# 4. 启动 Gateway
pnpm gateway:dev
```

---

## 核心洞察

### 1. pnpm 不是简单的包管理器

**表面**：安装依赖

**实际**：
- 内容寻址存储
- 符号链接优化
- workspace 协议
- 严格依赖管理

---

### 2. Monorepo 不是简单的多包管理

**表面**：多个 package.json

**实际**：
- 统一依赖版本
- 共享构建配置
- 协调发布流程
- 优化 CI/CD

---

### 3. workspace 协议的威力

**表面**：`workspace:*`

**实际**：
- 本地开发时使用符号链接
- 发布时自动替换为实际版本
- 避免循环依赖问题
- 支持版本约束

---

## 一句话总结

**pnpm monorepo 通过内容寻址存储和符号链接实现高效的多包管理，OpenClaw 使用 workspace 协议统一管理核心包、UI 包和扩展包。**

---

[来源: reference/04_pnpm_workspace_docs.md, reference/16_pnpm_monorepo_debugging.md, reference/01_source_code_analysis.md]
