# TypeScript 5.9.3 官方文档

**来源**: Context7 - /microsoft/typescript/v5.9.3
**查询时间**: 2026-02-23
**用途**: 构建工具链层 - tsconfig 配置、编译选项、模块系统

---

## TypeScript Compiler Options for NodeNext Module Resolution

**来源**: https://github.com/microsoft/typescript/blob/v5.9.3/tests/baselines/reference/nodeNextPackageSelfNameWithOutDirDeclDirNestedDirs.errors.txt

此 `tsconfig.json` 定义了项目的 TypeScript 编译器选项。它为模块解析指定 `nodenext`，将编译后的 JavaScript 输出到 `./dist`，并将声明文件 (`.d.ts`) 生成到 `./types`。

```json
{
  "compilerOptions": {
    "module": "nodenext",
    "outDir": "./dist",
    "declarationDir": "./types",
    "declaration": true
  }
}
```

---

## TypeScript Compiler Configuration Requirements for for-await

**来源**: https://github.com/microsoft/typescript/blob/v5.9.3/tests/baselines/reference/awaitUsingDeclarationsInForAwaitOf.3(target=es5).errors.txt

指定使用顶层 for-await 循环所需的 TypeScript 编译器选项。'module' 选项必须设置为以下之一：'es2022'、'esnext'、'system'、'node16'、'node18'、'node20'、'nodenext' 或 'preserve'。'target' 选项必须是 'es2017' 或更高版本才能启用 for-await 循环支持。

```json
{
  "compilerOptions": {
    "module": "es2022",
    "target": "es2017"
  }
}
```

---

## TypeScript tsconfig.json Module Resolution Configuration

**来源**: https://github.com/microsoft/typescript/blob/v5.9.3/tests/baselines/reference/cachedModuleResolution8.errors.txt

tsconfig.json 中的配置选项用于解决 TS2792 错误。将 moduleResolution 设置为 'nodenext' 以实现现代 Node.js 模块解析，或配置 paths 选项为自定义模块位置创建模块别名。

```json
{
  "compilerOptions": {
    "moduleResolution": "nodenext",
    "paths": {
      "foo": ["./path/to/foo"]
    }
  }
}
```

---

## Handle TypeScript TS2792 'Cannot find module' Errors in ES5 Target

**来源**: https://github.com/microsoft/typescript/blob/v5.9.3/tests/baselines/reference/esnextmodulekindWithES5Target9.errors.txt

此代码片段说明了各种 TypeScript 导入和导出语句，这些语句会导致 `TS2792` 错误，因为找不到模块 'mod'。这通常发生在模块解析策略（`moduleResolution` 编译器选项）未正确配置（例如，未设置为 `nodenext`）或 `tsconfig.json` 中缺少模块别名（`paths` 选项）时。错误建议检查这些编译器选项以解决问题，特别是在针对 ES5 时。

```typescript
import d from "mod";
import {a} from "mod";
import * as M from "mod";
export * from "mod";
export {b} from "mod"
```

---

## TypeScript Module Configuration Error TS5110

**来源**: https://github.com/microsoft/typescript/blob/v5.9.3/tests/baselines/reference/allowImportingTsExtensions(moduleresolution=nodenext).errors.txt

当 'moduleResolution' 设置为 'NodeNext' 但 'module' 选项也未设置为 'NodeNext' 时，会发生错误 TS5110。这两个选项必须一起配置才能正确进行 Node.js 模块解析。这是一个配置验证错误，会阻止编译。

```typescript
!!! error TS5110: Option 'module' must be set to 'NodeNext' when option 'moduleResolution' is set to 'NodeNext'.
```

---

## OpenClaw 项目中的应用

### tsconfig.json 配置

OpenClaw 使用以下 TypeScript 配置：

```json
{
  "compilerOptions": {
    "target": "ES2023",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "lib": ["ES2023"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "paths": {
      "openclaw/plugin-sdk": ["./src/plugin-sdk/index.ts"]
    }
  },
  "include": ["src/**/*", "ui/**/*", "extensions/**/*"],
  "exclude": ["node_modules", "dist", "**/*.test.ts"]
}
```

### 关键配置说明

1. **模块系统**：
   - `module: "NodeNext"` - 使用 Node.js 的 ESM 模块系统
   - `moduleResolution: "NodeNext"` - 使用 Node.js 的模块解析策略

2. **编译目标**：
   - `target: "ES2023"` - 编译到 ES2023 标准
   - `lib: ["ES2023"]` - 使用 ES2023 标准库

3. **输出配置**：
   - `outDir: "./dist"` - 输出目录
   - `declaration: true` - 生成 .d.ts 声明文件
   - `sourceMap: true` - 生成 source map

4. **路径映射**：
   - `paths` - 配置模块别名，用于 plugin-sdk

5. **严格模式**：
   - `strict: true` - 启用所有严格类型检查选项

---

## TypeScript 5.9.3 关键特性

### 1. 模块系统增强

- **NodeNext 模块解析**：完全支持 Node.js 的 ESM 和 CommonJS 互操作
- **路径映射**：支持复杂的模块别名配置
- **声明文件生成**：自动生成类型声明文件

### 2. 类型检查改进

- **更严格的类型推断**：减少类型错误
- **更好的错误消息**：更清晰的编译错误提示
- **性能优化**：更快的类型检查速度

### 3. 编译器选项

- **target: ES2023**：支持最新的 JavaScript 特性
- **module: NodeNext**：支持 Node.js 的模块系统
- **strict: true**：启用所有严格检查

---

## 最佳实践

### 1. 模块配置

```json
{
  "compilerOptions": {
    "module": "NodeNext",
    "moduleResolution": "NodeNext"
  }
}
```

**注意**：`module` 和 `moduleResolution` 必须同时设置为 `NodeNext`。

### 2. 路径映射

```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "openclaw/plugin-sdk": ["src/plugin-sdk/index.ts"]
    }
  }
}
```

### 3. 声明文件生成

```json
{
  "compilerOptions": {
    "declaration": true,
    "declarationMap": true,
    "declarationDir": "./types"
  }
}
```

### 4. Source Map 配置

```json
{
  "compilerOptions": {
    "sourceMap": true,
    "inlineSources": true
  }
}
```

---

## 常见问题

### Q: TS2792 错误如何解决？

**错误**：Cannot find module 'xxx'

**解决方案**：
1. 确保 `moduleResolution` 设置为 `nodenext`
2. 配置 `paths` 选项添加模块别名
3. 检查模块是否正确安装

```json
{
  "compilerOptions": {
    "moduleResolution": "nodenext",
    "paths": {
      "foo": ["./path/to/foo"]
    }
  }
}
```

### Q: TS5110 错误如何解决？

**错误**：Option 'module' must be set to 'NodeNext' when option 'moduleResolution' is set to 'NodeNext'

**解决方案**：同时设置 `module` 和 `moduleResolution` 为 `NodeNext`

```json
{
  "compilerOptions": {
    "module": "NodeNext",
    "moduleResolution": "NodeNext"
  }
}
```

### Q: 如何配置 for-await 循环？

**要求**：
- `module`: 'es2022', 'esnext', 'system', 'node16', 'node18', 'node20', 'nodenext', 或 'preserve'
- `target`: 'es2017' 或更高

```json
{
  "compilerOptions": {
    "module": "es2022",
    "target": "es2017"
  }
}
```

---

## OpenClaw 编译流程

### 1. 使用 tsdown 编译

OpenClaw 使用 tsdown 作为编译器：

```bash
# 编译项目
pnpm build

# 开发模式（watch）
pnpm dev
```

### 2. 编译输出

```
dist/
├── index.js          # 主入口
├── index.d.ts        # 类型声明
├── plugin-sdk/       # Plugin SDK
│   ├── index.js
│   └── index.d.ts
└── ...
```

### 3. 类型检查

```bash
# 运行类型检查
pnpm check
```

---

**参考资料**:
- TypeScript 官方文档: https://www.typescriptlang.org/docs/
- Context7 TypeScript 文档: https://context7.com/microsoft/typescript/
- TypeScript 5.9 发布说明: https://devblogs.microsoft.com/typescript/announcing-typescript-5-9/
