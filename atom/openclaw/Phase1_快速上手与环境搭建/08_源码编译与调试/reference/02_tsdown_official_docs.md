# tsdown 官方文档

> 来源: Context7 - /rolldown/tsdown
> 查询时间: 2026-02-24
> 查询内容: tsdown build configuration, entry points, platform settings, output directory, and usage guide for TypeScript bundling in monorepo projects

---

## 1. Programmatic TypeScript Library Builds with tsdown API

**Source**: https://context7.com/rolldown/tsdown/llms.txt

Integrate tsdown into your Node.js scripts for programmatic control over library builds. This API offers type safety and a comprehensive set of options for configuring entry points, formats, output directories, declaration files, and more.

```typescript
import { build } from 'tsdown'

// Basic build configuration
await build({
  entry: ['src/index.ts'],
  format: ['esm', 'cjs'],
  outDir: 'dist',
  dts: true,
})

// Advanced build with all options
await build({
  entry: {
    index: 'src/index.ts',
    cli: 'src/cli.ts',
  },
  format: ['esm', 'cjs'],
  outDir: 'dist',
  clean: true,
  dts: {
    bundle: true,
    resolve: true,
  },
  sourcemap: true,
  minify: false,
  treeshake: true,
  platform: 'node',
  target: 'node18',
  external: ['react', 'vue'],
  skipNodeModulesBundle: false,
  watch: false,
  onSuccess: async (config, signal) => {
    console.log('Build completed successfully!')
  },
})

// Build with error handling
try {
  await build({
    entry: ['src/index.ts'],
    format: ['esm'],
    logLevel: 'info',
    failOnWarn: true,
  })
} catch (error) {
  console.error('Build failed:', error)
  process.exit(1)
}
```

---

## 2. Build TypeScript Library with tsdown CLI

**Source**: https://context7.com/rolldown/tsdown/llms.txt

Use the tsdown command-line interface for quick bundling of TypeScript libraries. It supports various options for entry points, output formats, declaration files, minification, source maps, watch mode, and target environments.

```bash
# Basic build with default options (detects src/index.ts)
npx tsdown

# Build with specific entry files
npx tsdown src/index.ts src/cli.ts

# Build with multiple formats
npx tsdown --format esm,cjs --dts

# Build with minification and source maps
npx tsdown --minify --sourcemap

# Watch mode for development
npx tsdown --watch

# Build for specific target environment
npx tsdown --target es2020 --platform node

# Enable size reporting and validation
npx tsdown --report --publint --attw
```

---

## 3. Configure Monorepo Workspace Support in tsdown

**Source**: https://context7.com/rolldown/tsdown/llms.txt

Illustrates various ways to configure workspace support in tsdown's configuration file, including auto-detection and explicit inclusion/exclusion patterns.

```typescript
// Root tsdown.config.ts
import { defineConfig } from 'tsdown'

// Auto-detect workspace packages
export default defineConfig({
  workspace: true,
})

// Explicit workspace configuration
export default defineConfig({
  workspace: {
    include: ['packages/*', 'apps/*'],
    exclude: ['**/node_modules/**', '**/dist/**'],
    config: './tsdown.config.ts', // Load config from each package
  },
})

// Shared configuration for all packages
export default defineConfig({
  workspace: ['packages/*'],
  format: ['esm', 'cjs'],
  dts: true,
  clean: true,
})

// Per-package configuration (packages/foo/tsdown.config.ts)
export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm'],
  name: 'foo', // Display name in CLI output
})
```

---

## 4. Configure Target Environment for tsdown Builds

**Source**: https://context7.com/rolldown/tsdown/llms.txt

Explains how to specify JavaScript syntax targets and runtime environments for tsdown builds, including single/multiple targets, browser platform, and disabling transformations.

```typescript
// tsdown.config.ts
import { defineConfig } from 'tsdown'

// Single target
export default defineConfig({
  entry: ['src/index.ts'],
  target: 'es2020',
  platform: 'node',
})

// Multiple targets (intersection)
export default defineConfig({
  entry: ['src/index.ts'],
  target: ['node18', 'es2021'],
})

// Browser target
export default defineConfig({
  entry: ['src/index.ts'],
  target: 'es2020',
  platform: 'browser',
})

// Disable syntax transformations
export default defineConfig({
  entry: ['src/index.ts'],
  target: false,
})

// Auto-detect from package.json engines.node
// package.json: { "engines": { "node": ">=18" } }
export default defineConfig({
  entry: ['src/index.ts'],
  // target defaults to "node18"
})
```

---

## 5. tsdown Configuration File Setup

**Source**: https://context7.com/rolldown/tsdown/llms.txt

Define reusable and type-safe build configurations for your TypeScript libraries using `tsdown.config.ts`. This allows for complex setups, including multiple configurations and dynamic options based on environment variables or CLI arguments.

```typescript
// tsdown.config.ts
import { defineConfig } from 'tsdown'

export default defineConfig({
  entry: ['./src/index.ts'],
  format: ['esm', 'cjs'],
  outDir: 'dist',
  dts: true,
  clean: true,
  sourcemap: false,
  minify: false,
  treeshake: true,
  platform: 'node',
  target: 'node18',
  plugins: [],
})

// Multiple configurations
export default defineConfig([
  {
    entry: ['src/index.ts'],
    format: ['esm'],
    outDir: 'dist/esm',
  },
  {
    entry: ['src/cli.ts'],
    format: ['cjs'],
    outDir: 'dist/cjs',
    banner: {
      js: '#!/usr/bin/env node',
    },
  },
])

// Dynamic configuration
export default defineConfig((cliOptions) => {
  return {
    entry: ['src/index.ts'],
    format: ['esm', 'cjs'],
    minify: process.env.NODE_ENV === 'production',
    dts: cliOptions.dts ?? true,
  }
})
```

---

## 关键配置选项

### entry (入口点)
- **类型**: `string | string[] | Record<string, string>`
- **说明**: 指定构建入口文件
- **示例**:
  - 单入口: `'src/index.ts'`
  - 多入口: `['src/index.ts', 'src/cli.ts']`
  - 命名入口: `{ index: 'src/index.ts', cli: 'src/cli.ts' }`

### format (输出格式)
- **类型**: `('esm' | 'cjs')[]`
- **说明**: 指定输出模块格式
- **默认**: `['esm', 'cjs']`

### outDir (输出目录)
- **类型**: `string`
- **说明**: 指定构建产物输出目录
- **默认**: `'dist'`

### platform (平台)
- **类型**: `'node' | 'browser' | 'neutral'`
- **说明**: 指定目标运行平台
- **默认**: `'node'`

### target (目标环境)
- **类型**: `string | string[] | false`
- **说明**: 指定 JavaScript 语法目标
- **示例**: `'node18'`, `'es2020'`, `['node18', 'es2021']`

### dts (类型定义)
- **类型**: `boolean | { bundle?: boolean, resolve?: boolean }`
- **说明**: 是否生成 TypeScript 类型定义文件
- **默认**: `false`

### clean (清理)
- **类型**: `boolean`
- **说明**: 构建前是否清理输出目录
- **默认**: `false`

### sourcemap (源码映射)
- **类型**: `boolean | 'inline'`
- **说明**: 是否生成 source map
- **默认**: `false`

### minify (压缩)
- **类型**: `boolean`
- **说明**: 是否压缩输出代码
- **默认**: `false`

### treeshake (树摇)
- **类型**: `boolean`
- **说明**: 是否启用 tree-shaking
- **默认**: `true`

### external (外部依赖)
- **类型**: `string[]`
- **说明**: 指定不打包的外部依赖
- **示例**: `['react', 'vue']`

### watch (监听模式)
- **类型**: `boolean`
- **说明**: 是否启用文件监听模式
- **默认**: `false`

---

## OpenClaw 项目中的 tsdown 配置

```typescript
// sourcecode/openclaw/tsdown.config.ts
import { defineConfig } from "tsdown";

const env = {
  NODE_ENV: "production",
};

export default defineConfig([
  {
    entry: "src/index.ts",
    env,
    fixedExtension: false,
    platform: "node",
  },
  {
    entry: "src/entry.ts",
    env,
    fixedExtension: false,
    platform: "node",
  },
  {
    entry: "src/cli/daemon-cli.ts",
    env,
    fixedExtension: false,
    platform: "node",
  },
  {
    entry: "src/infra/warning-filter.ts",
    env,
    fixedExtension: false,
    platform: "node",
  },
  {
    entry: "src/plugin-sdk/index.ts",
    outDir: "dist/plugin-sdk",
    env,
    fixedExtension: false,
    platform: "node",
  },
  {
    entry: "src/plugin-sdk/account-id.ts",
    outDir: "dist/plugin-sdk",
    env,
    fixedExtension: false,
    platform: "node",
  },
  {
    entry: "src/extensionAPI.ts",
    env,
    fixedExtension: false,
    platform: "node",
  },
  {
    entry: ["src/hooks/bundled/*/handler.ts", "src/hooks/llm-slug-generator.ts"],
    env,
    fixedExtension: false,
    platform: "node",
  },
]);
```

**关键特点**:
- **多入口点**: 8 个独立入口点
- **平台**: 全部为 Node.js 平台
- **环境变量**: NODE_ENV=production
- **扩展名**: fixedExtension: false (动态扩展名)
- **输出目录**: 默认 dist/, plugin-sdk 单独输出到 dist/plugin-sdk

---

## 最佳实践

### 1. Monorepo 配置
```typescript
// 根目录 tsdown.config.ts
export default defineConfig({
  workspace: {
    include: ['packages/*', 'apps/*'],
    exclude: ['**/node_modules/**', '**/dist/**'],
  },
  format: ['esm', 'cjs'],
  dts: true,
  clean: true,
})
```

### 2. 多入口点配置
```typescript
export default defineConfig({
  entry: {
    index: 'src/index.ts',
    cli: 'src/cli.ts',
    utils: 'src/utils/index.ts',
  },
  format: ['esm', 'cjs'],
  dts: true,
})
```

### 3. 开发与生产配置
```typescript
export default defineConfig((options) => ({
  entry: ['src/index.ts'],
  format: ['esm', 'cjs'],
  minify: process.env.NODE_ENV === 'production',
  sourcemap: process.env.NODE_ENV === 'development',
  watch: options.watch ?? false,
}))
```

### 4. 类型定义打包
```typescript
export default defineConfig({
  entry: ['src/index.ts'],
  dts: {
    bundle: true,  // 打包所有类型定义到单个文件
    resolve: true, // 解析外部类型引用
  },
})
```

---

## 常见问题

### Q1: 如何调试 tsdown 构建?
```bash
# 启用详细日志
npx tsdown --logLevel debug

# 查看构建产物
npx tsdown --report
```

### Q2: 如何排除特定依赖?
```typescript
export default defineConfig({
  entry: ['src/index.ts'],
  external: ['react', 'react-dom', /^@myorg\//],
})
```

### Q3: 如何自定义输出文件名?
```typescript
export default defineConfig({
  entry: {
    'my-lib': 'src/index.ts',
    'my-lib.min': 'src/index.ts',
  },
  minify: true,
})
```

---

**文档版本**: tsdown 0.20.3
**最后更新**: 2026-02-24
**来源**: Context7 官方文档
