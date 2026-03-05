# TypeScript 构建性能优化

> 来源: Grok Web Search
> 查询时间: 2026-02-24

## 搜索结果

### 1. tsdown 官方指南
**URL**: https://tsdown.dev/guide/
**描述**: tsdown 优雅的 TS 库打包器，基于 Rolldown 实现极速构建、树摇和最小化，与 esbuild 比较优化 2026 年性能。

### 2. tsdown 性能基准测试
**URL**: https://tsdown.dev/advanced/benchmark
**描述**: tsdown 与 tsup 对比，标准构建快 2 倍，生成声明文件快至 8 倍，TypeScript 优化关键数据。

### 3. 从 tsup 迁移到 tsdown
**URL**: https://tsdown.dev/guide/migrate-from-tsup
**描述**: 从 esbuild 驱动的 tsup 迁移至 Rolldown tsdown 的完整指南，大幅提升 TypeScript 库构建速度。

### 4. esbuild 官方站点
**URL**: https://esbuild.github.io/
**描述**: Go 语言实现的极致快速 JS/TS 打包工具，提供 10-100 倍性能提升，是构建优化的基础。

### 5. tsdown GitHub 仓库
**URL**: https://github.com/rolldown/tsdown
**描述**: tsdown 开源仓库，包含最新优化配置、插件生态和 TypeScript 性能提升 issue 讨论。

### 6. 2026 年 npm 包构建实践
**URL**: https://medium.com/@pyyupsk/how-i-build-an-npm-package-in-2026-4bb1a4b88e11
**描述**: 2026 年现代栈推荐：使用 tsdown 实现 ESM/CJS 双格式、类型声明的快速 TypeScript 构建。

## 性能优化建议

### 1. 使用 tsdown 替代 tsup
- 构建速度提升 2 倍
- d.ts 生成提升 8 倍
- 更好的 TypeScript 支持

### 2. 启用 isolatedDeclarations
```json
{
  "compilerOptions": {
    "isolatedDeclarations": true
  }
}
```

### 3. 优化配置
```typescript
export default defineConfig({
  entry: ['src/index.ts'],
  treeshake: true,
  minify: true,
  dts: { bundle: true }
})
```

**文档版本**: 基于 2026-02-24 搜索结果
