# 核心概念 2：tsdown 构建流程

> OpenClaw 使用 tsdown 实现高效的 TypeScript 构建

---

## 概念介绍

### 什么是 tsdown？

**tsdown**：基于 Rolldown（Rust 实现的 bundler）的 TypeScript 构建工具，专为库和应用开发设计。

**核心特性**：
- **快速构建**：Rust 实现，比 tsc 快 10-100 倍
- **Tree-shaking**：自动移除未使用的代码
- **代码分割**：支持多入口点独立构建
- **类型定义**：自动生成 .d.ts 文件
- **多格式输出**：支持 ESM、CJS 等格式

**OpenClaw 的使用**：
- 8 个独立入口点
- Node.js 平台
- 生产环境优化
- 动态扩展名

---

## TypeScript 编译原理

### 1. tsc vs bundler

**传统 tsc 的工作方式**：
```
src/index.ts
    ↓ [tsc 编译]
dist/index.js

src/utils.ts
    ↓ [tsc 编译]
dist/utils.js

src/types.ts
    ↓ [tsc 编译]
dist/types.js
```

**特点**：
- 逐文件编译
- 保留模块结构
- 不支持 tree-shaking
- 速度较慢

---

**tsdown（bundler）的工作方式**：
```
src/index.ts
src/utils.ts
src/types.ts
    ↓ [tsdown 构建]
    ↓ [依赖分析]
    ↓ [tree-shaking]
    ↓ [代码分割]
dist/index.js (打包后)
```

**特点**：
- 整体分析
- 移除未使用代码
- 代码优化
- 速度快

---

### 2. 编译流程对比

| 特性 | tsc | tsdown |
|------|-----|--------|
| **速度** | 慢 | 快（10-100倍） |
| **Tree-shaking** | ❌ | ✅ |
| **代码分割** | ❌ | ✅ |
| **压缩优化** | ❌ | ✅ |
| **多格式输出** | 需要配置 | 内置支持 |
| **类型定义** | ✅ | ✅ |
| **适用场景** | 库开发 | 库 + 应用 |

---

## tsdown 工作机制

### 1. 构建流程

```
TypeScript 源码 (src/)
    ↓
[1. 入口点分析]
    ↓
[2. 依赖图构建]
    ↓
[3. TypeScript 编译]
    ↓
[4. Tree-shaking]
    ↓
[5. 代码分割]
    ↓
[6. 压缩优化]
    ↓
[7. 生成 Source Maps]
    ↓
[8. 生成类型定义]
    ↓
JavaScript 代码 (dist/)
```

---

### 2. 核心机制

**依赖图构建**：
```typescript
// src/index.ts
import { foo } from './utils'
import { bar } from './helpers'

// tsdown 分析依赖关系
index.ts
  ├── utils.ts
  │   └── types.ts
  └── helpers.ts
      └── constants.ts
```

**Tree-shaking**：
```typescript
// utils.ts
export function used() { return 'used' }
export function unused() { return 'unused' }

// index.ts
import { used } from './utils'

// 构建后只包含 used()
// unused() 被移除
```

**代码分割**：
```typescript
// tsdown.config.ts
export default defineConfig({
  entry: {
    index: 'src/index.ts',
    cli: 'src/cli.ts',
  }
})

// 输出
dist/index.js
dist/cli.js
```

---

### 3. Rolldown 底层原理

**Rolldown**：Rust 实现的 JavaScript bundler，类似 Rollup 但更快。

**关键特性**：
- **并行处理**：多线程编译
- **增量构建**：只重新构建变更的文件
- **内存优化**：高效的内存管理
- **原生性能**：Rust 的性能优势

**性能对比**：
```
tsc:      100s
esbuild:  10s
tsdown:   5s (基于 Rolldown)
```

---

## OpenClaw 的构建配置

### 1. tsdown.config.ts

```typescript
import { defineConfig } from "tsdown";

const env = {
  NODE_ENV: "production",
};

export default defineConfig([
  // 入口点 1: 主入口
  {
    entry: "src/index.ts",
    env,
    fixedExtension: false,
    platform: "node",
  },

  // 入口点 2: 备用入口
  {
    entry: "src/entry.ts",
    env,
    fixedExtension: false,
    platform: "node",
  },

  // 入口点 3: CLI 守护进程
  {
    entry: "src/cli/daemon-cli.ts",
    env,
    fixedExtension: false,
    platform: "node",
  },

  // 入口点 4: 警告过滤
  {
    entry: "src/infra/warning-filter.ts",
    env,
    fixedExtension: false,
    platform: "node",
  },

  // 入口点 5: Plugin SDK
  {
    entry: "src/plugin-sdk/index.ts",
    outDir: "dist/plugin-sdk",
    env,
    fixedExtension: false,
    platform: "node",
  },

  // 入口点 6: 账户 ID
  {
    entry: "src/plugin-sdk/account-id.ts",
    outDir: "dist/plugin-sdk",
    env,
    fixedExtension: false,
    platform: "node",
  },

  // 入口点 7: Extension API
  {
    entry: "src/extensionAPI.ts",
    env,
    fixedExtension: false,
    platform: "node",
  },

  // 入口点 8: Hook 处理器（glob 模式）
  {
    entry: ["src/hooks/bundled/*/handler.ts", "src/hooks/llm-slug-generator.ts"],
    env,
    fixedExtension: false,
    platform: "node",
  },
]);
```

**关键配置**：
- **entry**：入口文件或 glob 模式
- **env**：环境变量注入
- **fixedExtension**：动态扩展名（.js/.mjs）
- **platform**：目标平台（node）
- **outDir**：输出目录（默认 dist/）

---

### 2. tsconfig.json

```json
{
  "compilerOptions": {
    "target": "es2023",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "lib": ["DOM", "DOM.Iterable", "ES2023", "ScriptHost"],
    "strict": true,
    "noEmit": true,
    "experimentalDecorators": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "resolveJsonModule": true,
    "skipLibCheck": true
  },
  "include": ["src/**/*", "ui/**/*", "extensions/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

**关键配置**：
- **target**: ES2023（现代 JavaScript）
- **module**: NodeNext（Node.js ESM）
- **moduleResolution**: NodeNext（Node.js 模块解析）
- **strict**: true（严格模式）
- **noEmit**: true（不生成文件，由 tsdown 负责）

---

### 3. 构建脚本

```json
{
  "scripts": {
    "build": "pnpm canvas:a2ui:bundle && tsdown && pnpm build:plugin-sdk:dts && ...",
    "build:watch": "tsdown --watch",
    "build:clean": "rm -rf dist && pnpm build"
  }
}
```

**完整构建流程**：
```bash
# 1. 打包 Canvas A2UI
pnpm canvas:a2ui:bundle

# 2. tsdown 构建
tsdown

# 3. 生成 plugin-sdk 类型定义
pnpm build:plugin-sdk:dts

# 4. 写入 plugin-sdk 入口类型
node --import tsx scripts/write-plugin-sdk-entry-dts.ts

# 5. 复制 Canvas 资源
node --import tsx scripts/canvas-a2ui-copy.ts

# 6. 复制 Hook 元数据
node --import tsx scripts/copy-hook-metadata.ts

# 7. 复制 HTML 模板
node --import tsx scripts/copy-export-html-templates.ts

# 8. 写入构建信息
node --import tsx scripts/write-build-info.ts

# 9. 写入 CLI 兼容性
node --import tsx scripts/write-cli-compat.ts
```

---

## 构建产物分析

### 1. 目录结构

```
dist/
├── index.js                    # 主入口
├── entry.js                    # 备用入口
├── daemon-cli.js               # CLI 守护进程
├── warning-filter.js           # 警告过滤
├── extensionAPI.js             # Extension API
├── plugin-sdk/                 # Plugin SDK
│   ├── index.js
│   ├── index.d.ts
│   ├── account-id.js
│   └── account-id.d.ts
└── hooks/                      # Hook 处理器
    ├── bundled/
    │   ├── hook1/
    │   │   └── handler.js
    │   └── hook2/
    │       └── handler.js
    └── llm-slug-generator.js
```

---

### 2. 输出格式

**ESM 格式**（默认）：
```javascript
// dist/index.js
export { Gateway } from './gateway.js'
export { Agent } from './agent.js'
```

**CJS 格式**（可选）：
```javascript
// dist/index.cjs
module.exports = {
  Gateway: require('./gateway.cjs').Gateway,
  Agent: require('./agent.cjs').Agent,
}
```

---

### 3. 类型定义

```typescript
// dist/plugin-sdk/index.d.ts
export interface PluginConfig {
  name: string
  version: string
  hooks: Hook[]
}

export class Plugin {
  constructor(config: PluginConfig)
  register(): void
}
```

---

### 4. Source Maps

```javascript
// dist/index.js
export { Gateway } from './gateway.js'
//# sourceMappingURL=index.js.map
```

```json
// dist/index.js.map
{
  "version": 3,
  "sources": ["../src/index.ts"],
  "mappings": "AAAA,OAAO,EAAE,OAAO,EAAE,MAAM,YAAY,CAAC",
  "names": []
}
```

---

## 构建优化

### 1. Tree-shaking 优化

**问题**：未使用的代码被打包

**解决方案**：
```typescript
// tsdown.config.ts
export default defineConfig({
  entry: ['src/index.ts'],
  treeshake: true,  // 启用 tree-shaking
})
```

**效果**：
```
构建前: 1.2MB
构建后: 800KB (减少 33%)
```

---

### 2. 代码分割优化

**问题**：单个大文件加载慢

**解决方案**：
```typescript
// tsdown.config.ts
export default defineConfig({
  entry: {
    index: 'src/index.ts',
    cli: 'src/cli.ts',
    'plugin-sdk': 'src/plugin-sdk/index.ts',
  },
})
```

**效果**：
```
单文件: 1.2MB
分割后:
  - index.js: 600KB
  - cli.js: 400KB
  - plugin-sdk.js: 200KB
```

---

### 3. 压缩优化

**问题**：代码体积大

**解决方案**：
```typescript
// tsdown.config.ts
export default defineConfig({
  entry: ['src/index.ts'],
  minify: process.env.NODE_ENV === 'production',
})
```

**效果**：
```
未压缩: 800KB
压缩后: 300KB (减少 62.5%)
```

---

### 4. Source Maps 优化

**开发环境**：
```typescript
export default defineConfig({
  entry: ['src/index.ts'],
  sourcemap: true,  // 完整 source maps
})
```

**生产环境**：
```typescript
export default defineConfig({
  entry: ['src/index.ts'],
  sourcemap: false,  // 禁用 source maps
})
```

---

## 最佳实践

### 1. Monorepo 配置

**根目录 tsdown.config.ts**：
```typescript
import { defineConfig } from 'tsdown'

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

**包级别 tsdown.config.ts**：
```typescript
import { defineConfig } from 'tsdown'

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm', 'cjs'],
  outDir: 'dist',
  dts: {
    bundle: true,
    resolve: true,
  },
})
```

---

### 2. Unbundle 模式

**适用场景**：
- 库开发
- 需要单文件导入
- Tree-shaking 优化

**配置**：
```typescript
export default defineConfig({
  entry: ['src/**/*.ts'],
  unbundle: true,
  outDir: 'dist',
  dts: true,
})
```

**效果**：
```
src/
├── index.ts
├── utils.ts
└── types.ts

dist/
├── index.js
├── utils.js
└── types.js
```

---

### 3. 类型声明优化

**isolatedDeclarations**：
```typescript
// tsconfig.json
{
  "compilerOptions": {
    "isolatedDeclarations": true
  }
}

// tsdown.config.ts
export default defineConfig({
  dts: {
    bundle: true,
    resolve: true,
  },
})
```

**效果**：
- 类型定义生成速度提升 200%
- 类型检查更快

---

### 4. 多格式输出

```typescript
export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm', 'cjs'],
  outDir: 'dist',
})
```

**输出**：
```
dist/
├── index.js      # ESM
└── index.cjs     # CJS
```

---

### 5. 外部依赖

```typescript
export default defineConfig({
  entry: ['src/index.ts'],
  external: [
    'react',
    'react-dom',
    /^@openclaw\//,  // 正则匹配
  ],
})
```

**效果**：
- 不打包外部依赖
- 减小包体积
- 避免重复打包

---

### 6. 环境变量注入

```typescript
export default defineConfig({
  entry: ['src/index.ts'],
  env: {
    NODE_ENV: 'production',
    API_URL: 'https://api.example.com',
  },
})
```

**代码中使用**：
```typescript
if (process.env.NODE_ENV === 'production') {
  console.log('Production mode')
}
```

---

## 与 OpenClaw 的关系

### 1. 为什么 OpenClaw 使用 tsdown？

**问题**：tsc 编译速度慢，不支持 tree-shaking

**解决方案**：tsdown
- **更快的构建速度**：Rust 实现，比 tsc 快 10-100 倍
- **tree-shaking**：移除未使用代码，减小包体积
- **代码分割**：8 个入口点独立构建
- **更好的优化**：压缩、混淆、sourcemap

---

### 2. OpenClaw 的 8 个入口点

**为什么需要多入口点**：
- **模块化加载**：按需加载，减少启动时间
- **独立构建**：Plugin SDK 可以单独发布
- **代码隔离**：Extension API 与核心代码分离

**入口点分类**：
```
核心入口:
  - src/index.ts (主入口)
  - src/entry.ts (备用入口)

CLI 入口:
  - src/cli/daemon-cli.ts (守护进程)

基础设施:
  - src/infra/warning-filter.ts (警告过滤)

扩展系统:
  - src/plugin-sdk/index.ts (Plugin SDK)
  - src/plugin-sdk/account-id.ts (账户 ID)
  - src/extensionAPI.ts (Extension API)

Hook 系统:
  - src/hooks/bundled/*/handler.ts (Hook 处理器)
```

---

### 3. OpenClaw 的构建流程

```bash
# 1. 安装依赖
pnpm install

# 2. 构建所有包
pnpm build

# 3. 运行测试
pnpm test

# 4. 启动 Gateway
pnpm gateway:dev
```

---

## 核心洞察

### 1. tsdown 不是简单的编译器

**表面**：TypeScript → JavaScript

**实际**：
- 依赖图分析
- Tree-shaking 优化
- 代码分割
- 压缩混淆
- Source maps 生成

---

### 2. 多入口点的威力

**表面**：多个 entry 配置

**实际**：
- 模块化加载
- 独立构建
- 代码隔离
- 按需加载

---

### 3. 构建不是一次性的

**表面**：运行 `pnpm build`

**实际**：
- 9 步构建流程
- 资源复制
- 元数据生成
- 构建信息写入

---

## 一句话总结

**tsdown 通过 Rolldown 实现高效的 TypeScript 构建，OpenClaw 使用 8 个入口点实现模块化构建和按需加载。**

---

[来源: reference/02_tsdown_official_docs.md, reference/08_tsdown_best_practices.md, reference/01_source_code_analysis.md]
