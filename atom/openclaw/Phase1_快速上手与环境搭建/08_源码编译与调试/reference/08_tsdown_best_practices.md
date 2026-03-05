# tsdown 构建配置最佳实践

> 来源: Grok Web Search
> 查询时间: 2026-02-24
> 查询内容: tsdown build configuration best practices TypeScript monorepo 2026

---

## 搜索结果

### 1. tsdown 官方入门指南

**URL**: https://tsdown.dev/guide/getting-started

**描述**: tsdown.config.ts 配置示例，包含 entry 入口、多格式输出和插件集成，适用于 TypeScript monorepo 库构建的最佳实践。

---

### 2. tsdown Unbundle 模式

**URL**: https://tsdown.dev/options/unbundle

**描述**: monorepo 场景下启用 unbundle 模式保留源模块结构，支持单个文件导入，避免全打包的最佳配置指南。

**关键特性**:
- 保留源文件结构
- 支持单文件导入
- 适合库开发

---

### 3. 完美 TS Monorepo tsdown 示例

**URL**: https://github.com/0x80/typescript-monorepo

**描述**: 使用 tsdown 和 Turborepo 的 monorepo 模板，每包独立 tsdown.config.ts 配置、dts 生成和项目引用最佳实践。

**项目结构**:
```
typescript-monorepo/
├── packages/
│   ├── core/
│   │   ├── tsdown.config.ts
│   │   └── src/
│   ├── utils/
│   │   ├── tsdown.config.ts
│   │   └── src/
│   └── types/
│       ├── tsdown.config.ts
│       └── src/
├── turbo.json
└── pnpm-workspace.yaml
```

---

### 4. tsdown DTS 声明文件选项

**URL**: https://tsdown.dev/options/dts

**描述**: monorepo 中 tsdown 生成类型声明的最佳实践，支持 declarationMap 和 isolatedDeclarations 优化开发体验。

**配置示例**:
```typescript
export default defineConfig({
  entry: ['src/index.ts'],
  dts: {
    bundle: true,
    resolve: true,
  },
})
```

---

### 5. 从 tsup 迁移至 tsdown

**URL**: https://alan.norbauer.com/articles/tsdown-bundler/

**描述**: TypeScript monorepo 项目迁移经验，性能提升 49%，isolatedDeclarations 设置和 unbundle 配置最佳实践。

**性能对比**:
- 构建速度: +49%
- d.ts 生成: +200%
- 内存使用: -30%

---

### 6. tsdown AI Skills 指南

**URL**: https://tsdown.dev/guide/skills

**描述**: 官方提供 monorepo workspace 构建配置提示，涵盖配置格式、选项和 TypeScript 库最佳实践。

---

### 7. tsdown GitHub 官方仓库

**URL**: https://github.com/rolldown/tsdown

**描述**: Rolldown 驱动的 tsdown 源代码仓库，包含 monorepo 支持示例、配置文档和社区讨论。

---

## 最佳实践总结

### 1. Monorepo 配置

**根目录 tsdown.config.ts**:
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

**包级别 tsdown.config.ts**:
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
  unbundle: false, // 库使用 unbundle: true
})
```

### 2. Unbundle 模式

**适用场景**:
- 库开发
- 需要单文件导入
- Tree-shaking 优化

**配置**:
```typescript
export default defineConfig({
  entry: ['src/**/*.ts'],
  unbundle: true,
  outDir: 'dist',
  dts: true,
})
```

### 3. 类型声明优化

**isolatedDeclarations**:
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

### 4. 多格式输出

```typescript
export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm', 'cjs'],
  outDir: 'dist',
  // ESM 输出到 dist/esm
  // CJS 输出到 dist/cjs
})
```

### 5. 性能优化

```typescript
export default defineConfig({
  entry: ['src/index.ts'],
  treeshake: true,
  minify: process.env.NODE_ENV === 'production',
  sourcemap: process.env.NODE_ENV === 'development',
  clean: true,
})
```

### 6. 外部依赖

```typescript
export default defineConfig({
  entry: ['src/index.ts'],
  external: [
    'react',
    'react-dom',
    /^@myorg\//,
  ],
})
```

---

## OpenClaw 项目应用

### 当前配置分析

```typescript
// sourcecode/openclaw/tsdown.config.ts
export default defineConfig([
  {
    entry: "src/index.ts",
    env: { NODE_ENV: "production" },
    fixedExtension: false,
    platform: "node",
  },
  // ... 8 个入口点
])
```

**特点**:
- 多入口点配置
- Node.js 平台
- 动态扩展名

### 优化建议

**1. 添加类型声明**:
```typescript
export default defineConfig([
  {
    entry: "src/index.ts",
    dts: true,
    outDir: "dist",
  },
])
```

**2. 启用 Tree-shaking**:
```typescript
export default defineConfig([
  {
    entry: "src/index.ts",
    treeshake: true,
    minify: true,
  },
])
```

**3. 添加 Source Maps**:
```typescript
export default defineConfig([
  {
    entry: "src/index.ts",
    sourcemap: true,
  },
])
```

---

## 常见问题

### Q1: 如何配置 Monorepo?
```typescript
// 根目录
export default defineConfig({
  workspace: ['packages/*'],
  format: ['esm', 'cjs'],
  dts: true,
})

// 包目录
export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm'],
})
```

### Q2: 如何优化构建速度?
```typescript
export default defineConfig({
  entry: ['src/index.ts'],
  dts: {
    bundle: true, // 打包类型定义
  },
  treeshake: true,
  clean: false, // 跳过清理
})
```

### Q3: 如何处理原生依赖?
```typescript
export default defineConfig({
  entry: ['src/index.ts'],
  external: [
    'sharp',
    'esbuild',
    '@napi-rs/canvas',
  ],
})
```

---

**文档版本**: 基于 2026-02-24 搜索结果
**来源**: tsdown 官方文档 + 社区最佳实践
