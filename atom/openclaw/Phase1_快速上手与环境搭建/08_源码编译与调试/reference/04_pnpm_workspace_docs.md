# pnpm Workspace 官方文档

> 来源: Context7 - /pnpm/pnpm
> 查询时间: 2026-02-24
> 查询内容: pnpm workspace monorepo setup, pnpm-workspace.yaml configuration, onlyBuiltDependencies, scripts execution, and dependency management strategies

---

## 1. pnpm Workspace Configuration (pnpm-workspace.yaml)

**Source**: https://context7.com/pnpm/pnpm/llms.txt

Defines the structure and packages within a pnpm monorepo workspace. This YAML file specifies package directories using glob patterns, configures shared dependencies in a catalog, enables strict catalog mode, allows specific build processes, and defines dependency overrides.

```yaml
# pnpm-workspace.yaml
packages:
  # Include all packages in the packages directory
  - 'packages/*'
  # Include apps
  - 'apps/*'
  # Include specific directories
  - 'tools/*'
  - 'libs/*'
  # Exclude test directories
  - '!**/test/**'
  # Exclude example packages
  - '!**/examples/**'

# Configure catalog for shared dependencies
catalog:
  react: ^18.2.0
  typescript: ^5.0.0
  '@types/node': ^20.0.0

# Strict catalog mode
catalogMode: strict

# Allow specific builds during install
allowBuilds:
  esbuild: true
  node-gyp: true

# Override dependency versions
overrides:
  'lodash@<4.17.21': '>=4.17.21'
```

---

## 2. pnpm run: Execute Package Scripts

**Source**: https://context7.com/pnpm/pnpm/llms.txt

Runs scripts defined in `package.json`, with shorthands and options for conditional execution, workspace-wide execution, parallel or sequential runs, filtering, resuming from specific packages, and generating execution summaries.

```bash
# Run a script
pnpm run build

# Shorthand (omit 'run')
pnpm build

# Run script if it exists (don't fail if missing)
pnpm run --if-present test

# Run in all workspace packages
pnpm run -r build

# Run in parallel across all packages
pnpm run --parallel dev

# Run sequentially
pnpm run --sequential build

# Run in filtered packages
pnpm --filter "./packages/**" run build

# Resume from specific package
pnpm run -r --resume-from @myorg/pkg-c build

# Generate execution summary
pnpm run -r --report-summary build
```

---

## 3. Filtering Packages in pnpm Monorepos

**Source**: https://context7.com/pnpm/pnpm/llms.txt

Utilizes filters to target specific packages within a monorepo for commands like `install`, `run`, and `exec`. Supports filtering by package name, glob patterns, directories, dependencies, dependents, commit ranges, exclusions, and multiple filter criteria.

```bash
# Filter by package name
pnpm --filter @myorg/app build

# Filter by glob pattern
pnpm --filter "./packages/**" test

# Filter by directory
pnpm --filter ./packages/core build

# Filter dependencies of a package
pnpm --filter @myorg/app... build

# Filter dependents of a package
pnpm --filter ...@myorg/core build

# Filter by changed packages since commit/branch
pnpm --filter "...[origin/main]" test

# Exclude packages
pnpm --filter "!@myorg/docs" build

# Multiple filters
pnpm --filter @myorg/app --filter @myorg/api build

# Filter with workspace protocol
pnpm --filter "./packages/*" add lodash
```

---

## 4. pnpm install: Manage Project Dependencies

**Source**: https://context7.com/pnpm/pnpm/llms.txt

Installs project dependencies from `package.json`, generates `pnpm-lock.yaml`, and supports various options for CI environments, production/dev dependencies, offline installs, and workspace recursive installs.

```bash
# Install all dependencies
pnpm install

# Install with frozen lockfile (CI environments)
pnpm install --frozen-lockfile

# Install only production dependencies
pnpm install --prod

# Install only dev dependencies
pnpm install --dev

# Skip optional dependencies
pnpm install --no-optional

# Install recursively in all workspace packages
pnpm install --recursive

# Prefer offline installation (use cache)
pnpm install --prefer-offline

# Generate lockfile only, don't install
pnpm install --lockfile-only
```

---

## 5. Summary

**Source**: https://context7.com/pnpm/pnpm/llms.txt

For monorepo and workspace scenarios, pnpm excels with its powerful filtering system, shared lockfiles, workspace protocol for local package references, and catalog feature for centralized dependency version management. The extensive programmatic API through packages like `@pnpm/core`, `@pnpm/config`, and `@pnpm/lockfile.fs` enables building custom tooling, while hooks in `.pnpmfile.cjs` provide extensibility for modifying dependency resolution and package manifests. Integration with CI/CD systems is straightforward with frozen lockfile support, audit capabilities, and optimized offline installation modes.

---

## OpenClaw 项目中的 pnpm 配置

### pnpm-workspace.yaml

```yaml
packages:
  - .           # 根包
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

**关键特性**:
- **工作区包**: 根包、UI、packages、extensions
- **原生依赖**: 9 个需要编译的原生模块

### package.json 中的 pnpm 配置

```json
{
  "packageManager": "pnpm@10.23.0",
  "pnpm": {
    "minimumReleaseAge": 2880,
    "overrides": {
      "hono": "4.11.10",
      "fast-xml-parser": "5.3.6",
      "request": "npm:@cypress/request@3.0.10",
      "request-promise": "npm:@cypress/request-promise@5.0.0",
      "form-data": "2.5.4",
      "minimatch": "10.2.1",
      "qs": "6.14.2",
      "@sinclair/typebox": "0.34.48",
      "tar": "7.5.9",
      "tough-cookie": "4.1.3"
    },
    "onlyBuiltDependencies": [
      "@lydell/node-pty",
      "@matrix-org/matrix-sdk-crypto-nodejs",
      "@napi-rs/canvas",
      "@whiskeysockets/baileys",
      "authenticate-pam",
      "esbuild",
      "koffi",
      "node-llama-cpp",
      "protobufjs",
      "sharp"
    ]
  }
}
```

---

## 关键配置选项

### packages (工作区包)
- **类型**: `string[]`
- **说明**: 指定工作区包的目录模式
- **支持**: glob 模式、排除模式 (`!`)

### onlyBuiltDependencies (仅构建依赖)
- **类型**: `string[]`
- **说明**: 指定需要编译的原生依赖
- **用途**: 优化安装性能,只编译必要的原生模块

### overrides (依赖覆盖)
- **类型**: `Record<string, string>`
- **说明**: 覆盖依赖版本
- **用途**: 统一版本、修复安全漏洞

### catalog (依赖目录)
- **类型**: `Record<string, string>`
- **说明**: 集中管理共享依赖版本
- **用途**: 确保所有包使用相同版本

### catalogMode (目录模式)
- **类型**: `'strict' | 'loose'`
- **说明**: 强制使用 catalog 中的版本
- **默认**: `'loose'`

---

## 常用命令

### 安装依赖
```bash
# 安装所有依赖
pnpm install

# CI 环境 (冻结 lockfile)
pnpm install --frozen-lockfile

# 仅生产依赖
pnpm install --prod

# 递归安装 (所有工作区)
pnpm install -r
```

### 运行脚本
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
```

### 添加依赖
```bash
# 添加到根包
pnpm add lodash

# 添加到特定包
pnpm --filter @myorg/app add lodash

# 添加到所有包
pnpm -r add lodash

# 添加工作区依赖
pnpm --filter @myorg/app add @myorg/core@workspace:*
```

### 更新依赖
```bash
# 更新所有依赖
pnpm update

# 更新特定依赖
pnpm update lodash

# 更新到最新版本
pnpm update --latest

# 递归更新
pnpm -r update
```

---

## 最佳实践

### 1. 工作区结构
```yaml
packages:
  - 'packages/*'    # 共享库
  - 'apps/*'        # 应用
  - 'tools/*'       # 工具
  - '!**/test/**'   # 排除测试
```

### 2. 依赖管理
```json
{
  "pnpm": {
    "overrides": {
      "vulnerable-package": ">=safe-version"
    }
  }
}
```

### 3. 原生依赖优化
```yaml
onlyBuiltDependencies:
  - "esbuild"
  - "sharp"
  - "@napi-rs/canvas"
```

### 4. CI/CD 配置
```bash
# .github/workflows/ci.yml
- name: Install dependencies
  run: pnpm install --frozen-lockfile
```

### 5. 脚本执行
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

## 常见问题

### Q1: 如何解决依赖冲突?
```json
{
  "pnpm": {
    "overrides": {
      "conflicting-package": "specific-version"
    }
  }
}
```

### Q2: 如何加速安装?
```bash
# 使用缓存
pnpm install --prefer-offline

# 跳过可选依赖
pnpm install --no-optional

# 仅安装生产依赖
pnpm install --prod
```

### Q3: 如何调试工作区?
```bash
# 查看工作区包
pnpm list -r --depth 0

# 查看依赖树
pnpm list --depth 1

# 查看过滤结果
pnpm --filter "./packages/**" list
```

---

## OpenClaw 特定用法

### 构建流程
```bash
# 完整构建
pnpm build

# 开发模式
pnpm dev

# Gateway 开发
pnpm gateway:dev
```

### 测试流程
```bash
# 所有测试
pnpm test

# 单元测试
pnpm test:fast

# E2E 测试
pnpm test:e2e
```

### 代码质量
```bash
# 类型检查 + lint + 格式检查
pnpm check

# 格式化
pnpm format

# Lint
pnpm lint
```

---

**文档版本**: pnpm 10.23.0
**最后更新**: 2026-02-24
**来源**: Context7 官方文档
