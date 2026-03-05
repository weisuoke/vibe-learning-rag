# pnpm Workspace 官方文档

**来源**: Context7 - /pnpm/pnpm
**查询时间**: 2026-02-23
**用途**: 依赖管理层 - workspace 配置、依赖管理、构建脚本

---

## pnpm Workspace Configuration (pnpm-workspace.yaml)

**来源**: https://context7.com/pnpm/pnpm/llms.txt

定义 pnpm monorepo workspace 的结构和包。此 YAML 文件使用 glob 模式指定包目录，在 catalog 中配置共享依赖，启用严格 catalog 模式，允许特定构建过程，并定义依赖覆盖。

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

## pnpm install: 管理项目依赖

**来源**: https://context7.com/pnpm/pnpm/llms.txt

从 `package.json` 安装项目依赖，生成 `pnpm-lock.yaml`，支持 CI 环境、生产/开发依赖、离线安装和 workspace 递归安装等多种选项。

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

## 在 pnpm Monorepos 中过滤包

**来源**: https://context7.com/pnpm/pnpm/llms.txt

使用过滤器在 monorepo 中针对特定包执行命令，如 `install`、`run` 和 `exec`。支持按包名、glob 模式、目录、依赖、依赖者、提交范围、排除和多个过滤条件进行过滤。

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

## 构建 PNPM Artifacts

**来源**: https://github.com/pnpm/pnpm/blob/main/__utils__/build-artifacts/README.md

使用 pnpm 命令行界面执行 PNPM artifacts 的构建过程。假设 pnpm 已安装并配置。

```bash
pnpm build
```

---

## OpenClaw 项目中的应用

### package.json 依赖结构

OpenClaw 使用 pnpm workspace 管理 monorepo：

```json
{
  "name": "openclaw",
  "version": "2026.2.22",
  "engines": {
    "node": ">=22.12.0",
    "pnpm": "10.23.0"
  }
}
```

### 关键命令

```bash
# 安装依赖
pnpm install

# 构建 UI
pnpm ui:build

# 编译 TypeScript
pnpm build

# 全局链接
pnpm link --global

# 开发模式
pnpm dev
pnpm gateway:dev
pnpm gateway:watch

# 代码质量检查
pnpm check  # 格式检查 + 类型检查 + lint
pnpm test   # 运行测试
```

### 依赖审批机制

OpenClaw 使用 pnpm 的依赖审批机制：

```bash
pnpm approve-builds
```

---

## 最佳实践

1. **Frozen Lockfile**: CI 环境中使用 `--frozen-lockfile` 确保依赖一致性
2. **Workspace Protocol**: 使用 `workspace:*` 引用内部包
3. **Filter 命令**: 使用 `--filter` 针对特定包执行命令
4. **Catalog 模式**: 使用 catalog 统一管理共享依赖版本
5. **递归安装**: 使用 `--recursive` 在所有 workspace 包中安装依赖

---

**参考资料**:
- pnpm 官方文档: https://pnpm.io/
- Context7 pnpm 文档: https://context7.com/pnpm/pnpm/
