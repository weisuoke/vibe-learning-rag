# TypeScript 项目构建配置 - GitHub 资料

**来源**: Grok-mcp Web Search - GitHub
**查询时间**: 2026-02-23
**查询关键词**: "typescript project build configuration tsdown 2025 2026"
**用途**: 构建工具链层 - tsdown 编译器、构建配置

---

## 搜索结果概览

本文档收集了 GitHub 上关于 TypeScript 项目构建配置的最新资料，特别关注 tsdown 编译器的使用和配置。

---

## 1. rolldown/tsdown - TypeScript 库优雅打包器

**标题**: rolldown/tsdown - TypeScript库优雅打包器
**链接**: https://github.com/rolldown/tsdown
**来源**: GitHub 官方仓库

### 内容摘要

tsdown 官方 GitHub 仓库，由 Rolldown 驱动的 TypeScript 项目构建工具。支持高速打包、DTS 生成和现代配置选项，2025-2026 持续更新。

### 关键特性

1. **高性能编译**：
   - 基于 Rolldown 和 Oxc
   - 比传统 TypeScript 编译器快数倍
   - 支持增量编译

2. **声明文件生成**：
   - 自动生成 .d.ts 文件
   - 支持 declarationMap
   - 类型定义完整

3. **现代配置**：
   - 简洁的配置文件
   - 支持多入口
   - 灵活的输出选项

### 基础配置示例

```typescript
// tsdown.config.ts
import { defineConfig } from 'tsdown';

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm', 'cjs'],
  dts: true,
  sourcemap: true,
  clean: true,
});
```

### OpenClaw 应用

OpenClaw 使用 tsdown 作为主要编译器：

```json
{
  "scripts": {
    "build": "tsdown"
  },
  "devDependencies": {
    "tsdown": "^0.2.0"
  }
}
```

---

## 2. LuanRoger/ts-package-template - TS NPM 包模板

**标题**: LuanRoger/ts-package-template - TS NPM包模板
**链接**: https://github.com/LuanRoger/ts-package-template
**来源**: GitHub 社区模板

### 内容摘要

现代 TypeScript NPM 包构建模板，内置 tsdown.config.ts 和 tsconfig.build.json。提供完整项目构建配置示例，适用于 2025 开发。

### 项目结构

```
ts-package-template/
├── src/
│   └── index.ts
├── tsdown.config.ts
├── tsconfig.json
├── tsconfig.build.json
├── package.json
└── README.md
```

### tsdown.config.ts 配置

```typescript
import { defineConfig } from 'tsdown';

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm', 'cjs'],
  dts: {
    resolve: true,
  },
  sourcemap: true,
  clean: true,
  outDir: 'dist',
});
```

### tsconfig.build.json 配置

```json
{
  "extends": "./tsconfig.json",
  "compilerOptions": {
    "outDir": "dist",
    "declaration": true,
    "declarationMap": true
  },
  "include": ["src"],
  "exclude": ["node_modules", "dist", "**/*.test.ts"]
}
```

### 最佳实践

1. **分离配置**：
   - tsconfig.json - 开发配置
   - tsconfig.build.json - 构建配置

2. **输出格式**：
   - ESM 和 CJS 双格式输出
   - 支持 Node.js 和浏览器

3. **类型定义**：
   - 自动生成 .d.ts
   - 包含 source map

---

## 3. 0x80/mono-ts - TypeScript monorepo 实践

**标题**: 0x80/mono-ts - TypeScript monorepo实践
**链接**: https://github.com/0x80/mono-ts
**来源**: GitHub 社区实践

### 内容摘要

TypeScript 单仓库最佳实践仓库，每个包使用 tsdown.config.ts 定义构建。包含入口、输出、sourcemap 等配置，2025 更新。

### Monorepo 结构

```
mono-ts/
├── packages/
│   ├── core/
│   │   ├── src/
│   │   ├── tsdown.config.ts
│   │   └── package.json
│   ├── utils/
│   │   ├── src/
│   │   ├── tsdown.config.ts
│   │   └── package.json
│   └── cli/
│       ├── src/
│       ├── tsdown.config.ts
│       └── package.json
├── pnpm-workspace.yaml
└── package.json
```

### 每个包的 tsdown.config.ts

```typescript
// packages/core/tsdown.config.ts
import { defineConfig } from 'tsdown';

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm'],
  dts: true,
  sourcemap: true,
  outDir: 'dist',
  external: [
    // 外部依赖
    '@mono-ts/utils',
  ],
});
```

### pnpm-workspace.yaml

```yaml
packages:
  - 'packages/*'
```

### 构建脚本

```json
{
  "scripts": {
    "build": "pnpm -r build",
    "build:core": "pnpm --filter @mono-ts/core build",
    "dev": "pnpm -r --parallel dev"
  }
}
```

### OpenClaw 应用

OpenClaw 也是 monorepo 结构，可以参考这个配置：

```typescript
// OpenClaw 的 tsdown.config.ts
import { defineConfig } from 'tsdown';

export default defineConfig({
  entry: {
    index: 'src/index.ts',
    'plugin-sdk': 'src/plugin-sdk/index.ts',
  },
  format: ['esm'],
  dts: true,
  sourcemap: true,
  outDir: 'dist',
  clean: true,
});
```

---

## 4. jordanburke/typescript-library-template - TS 库模板

**标题**: jordanburke/typescript-library-template - TS库模板
**链接**: https://github.com/jordanburke/typescript-library-template
**来源**: GitHub 社区模板

### 内容摘要

现代 TypeScript 库模板，集成 tsdown 进行快速构建。标准化配置脚本，支持 2026 最佳实践和发布流程。

### 完整配置示例

```typescript
// tsdown.config.ts
import { defineConfig } from 'tsdown';

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm', 'cjs'],
  dts: {
    resolve: true,
    compilerOptions: {
      composite: false,
      incremental: false,
    },
  },
  sourcemap: true,
  clean: true,
  outDir: 'dist',
  splitting: false,
  treeshake: true,
  minify: false,
});
```

### package.json 配置

```json
{
  "name": "my-typescript-library",
  "version": "1.0.0",
  "type": "module",
  "main": "./dist/index.cjs",
  "module": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "import": {
        "types": "./dist/index.d.ts",
        "default": "./dist/index.js"
      },
      "require": {
        "types": "./dist/index.d.cts",
        "default": "./dist/index.cjs"
      }
    }
  },
  "files": ["dist"],
  "scripts": {
    "build": "tsdown",
    "dev": "tsdown --watch",
    "prepublishOnly": "pnpm build"
  }
}
```

### 发布流程

1. **构建**：
   ```bash
   pnpm build
   ```

2. **测试**：
   ```bash
   pnpm test
   ```

3. **发布**：
   ```bash
   pnpm publish
   ```

---

## tsdown 配置选项详解

### 基础选项

```typescript
export default defineConfig({
  // 入口文件
  entry: ['src/index.ts'],

  // 输出格式
  format: ['esm', 'cjs'],

  // 生成类型声明
  dts: true,

  // 生成 source map
  sourcemap: true,

  // 构建前清理输出目录
  clean: true,

  // 输出目录
  outDir: 'dist',
});
```

### 高级选项

```typescript
export default defineConfig({
  // 多入口配置
  entry: {
    index: 'src/index.ts',
    cli: 'src/cli.ts',
  },

  // DTS 配置
  dts: {
    resolve: true,
    compilerOptions: {
      composite: false,
    },
  },

  // 外部依赖
  external: ['react', 'react-dom'],

  // 代码分割
  splitting: true,

  // Tree shaking
  treeshake: true,

  // 压缩
  minify: false,

  // 目标环境
  target: 'es2020',

  // 平台
  platform: 'node',
});
```

---

## 最佳实践总结

### 1. 配置文件组织

```
project/
├── tsdown.config.ts      # 构建配置
├── tsconfig.json         # TypeScript 配置
├── tsconfig.build.json   # 构建专用配置
└── package.json          # 包配置
```

### 2. 输出格式选择

- **ESM only**: 现代项目推荐
- **ESM + CJS**: 兼容性最好
- **多入口**: 大型库推荐

### 3. 类型声明

- 始终生成 .d.ts
- 启用 declarationMap 便于调试
- 使用 dts.resolve 解析类型

### 4. 开发工作流

```json
{
  "scripts": {
    "dev": "tsdown --watch",
    "build": "tsdown",
    "type-check": "tsc --noEmit",
    "lint": "eslint src"
  }
}
```

### 5. Monorepo 配置

- 每个包独立的 tsdown.config.ts
- 使用 pnpm workspace
- 统一的构建脚本

---

## 常见问题

### Q: tsdown 与 tsc 的区别？

**回答**：
- **tsdown**: 快速打包工具，基于 Rolldown
- **tsc**: TypeScript 官方编译器
- **推荐**: 使用 tsdown 构建，tsc 做类型检查

### Q: 如何配置 watch 模式？

```bash
# 命令行
tsdown --watch

# 或在 package.json 中
{
  "scripts": {
    "dev": "tsdown --watch"
  }
}
```

### Q: 如何处理外部依赖？

```typescript
export default defineConfig({
  external: [
    // 所有 node_modules
    /node_modules/,
    // 特定包
    'react',
    'react-dom',
  ],
});
```

### Q: 如何生成多种格式？

```typescript
export default defineConfig({
  format: ['esm', 'cjs'],
  // package.json 中配置
  // "main": "./dist/index.cjs",
  // "module": "./dist/index.js",
});
```

---

## OpenClaw 构建流程

### 1. 开发模式

```bash
# 使用 run-node.mjs 脚本
pnpm dev

# 或直接使用 tsdown watch
pnpm gateway:watch
```

### 2. 生产构建

```bash
# 构建 UI
pnpm ui:build

# 编译 TypeScript
pnpm build

# 全局链接
pnpm link --global
```

### 3. 类型检查

```bash
# 运行类型检查
pnpm check
```

---

**参考资料**:
- tsdown GitHub: https://github.com/rolldown/tsdown
- Rolldown 官网: https://rolldown.rs/
- TypeScript 官方文档: https://www.typescriptlang.org/
