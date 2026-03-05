# oxlint 官方文档

> 来源: Context7 - /websites/oxc_rs_guide_usage
> 查询时间: 2026-02-24
> 查询内容: oxlint TypeScript linting configuration, type-aware linting, oxfmt formatting, and integration with VS Code

---

## 1. VS Code Enable Type-Aware Linting

**Source**: https://oxc.rs/docs/guide/usage/linter/editors

This JSON setting enables type-aware linting for Oxc within VS Code. It requires the `oxlint-tsgolint` to be installed and provides more accurate linting based on TypeScript types.

```json
{
  "oxc.typeAware": true
}
```

---

## 2. Configure Type-Aware Rules in Oxlint

**Source**: https://oxc.rs/docs/guide/usage/linter/type-aware

Example Oxlint configuration file (`.oxlintrc.json`) enabling the 'typescript' plugin and specifying rules like 'typescript/no-floating-promises' and 'typescript/no-unsafe-assignment'.

```json
{
  "plugins": ["typescript"],
  "rules": {
    "typescript/no-floating-promises": "error",
    "typescript/no-unsafe-assignment": "warn"
  }
}
```

---

## 3. Run Oxlint with Type-Aware Linting and Type Checking

**Source**: https://oxc.rs/docs/guide/usage/linter/type-aware

Enables both type-aware linting and TypeScript type checking using the `--type-aware --type-check` flags. This can replace a separate `tsc --noEmit` step.

```bash
oxlint --type-aware --type-check
```

---

## 4. Format Code using Oxfmt Node.js API

**Source**: https://oxc.rs/docs/guide/usage/formatter/quickstart

Demonstrates how to format code programmatically using the Oxfmt Node.js API, including specifying format options.

```ts
import { format, type FormatOptions } from "oxfmt";

const input = `let a=42;`;
const options: FormatOptions = {
  semi: false,
};

const { code } = await format("a.js", input, options);
console.log(code); // "let a = 42"
```

---

## 5. Type-Aware Linting - Running type-aware linting

**Source**: https://oxc.rs/docs/guide/usage/linter/type-aware

To run Oxlint with type-aware linting, you must pass the `--type-aware` flag. When enabled, Oxlint runs standard rules and type-aware rules in the `typescript/*` namespace. Type-aware linting is opt-in and does not run unless the flag is provided. In editor and LSP-based integrations like VS Code, type-aware linting can be enabled by setting the `typeAware` option to `true`, see the [Editors](./editors) page for more information.

---

## OpenClaw 项目中的 oxlint 配置

### package.json 中的脚本

```json
{
  "scripts": {
    "lint": "oxlint --type-aware",
    "lint:fix": "oxlint --type-aware --fix && pnpm format",
    "format": "oxfmt --write",
    "format:check": "oxfmt --check",
    "check": "pnpm format:check && pnpm tsgo && pnpm lint"
  },
  "devDependencies": {
    "oxfmt": "0.34.0",
    "oxlint": "^1.49.0",
    "oxlint-tsgolint": "^0.14.2"
  }
}
```

### VS Code 配置

```json
{
  "editor.formatOnSave": true,
  "[typescript]": {
    "editor.defaultFormatter": "oxc.oxc-vscode"
  },
  "[javascript]": {
    "editor.defaultFormatter": "oxc.oxc-vscode"
  },
  "[json]": {
    "editor.defaultFormatter": "oxc.oxc-vscode"
  }
}
```

---

## 关键特性

### 1. Type-Aware Linting (类型感知 Lint)

**说明**: 基于 TypeScript 类型系统进行更精确的代码检查

**启用方式**:
```bash
# 命令行
oxlint --type-aware

# VS Code
{
  "oxc.typeAware": true
}
```

**支持的规则**:
- `typescript/no-floating-promises` - 检测未处理的 Promise
- `typescript/no-unsafe-assignment` - 检测不安全的类型赋值
- `typescript/no-unsafe-call` - 检测不安全的函数调用
- `typescript/no-unsafe-member-access` - 检测不安全的成员访问

### 2. Type Checking (类型检查)

**说明**: 替代 `tsc --noEmit` 进行类型检查

**使用方式**:
```bash
# 同时进行 lint 和类型检查
oxlint --type-aware --type-check
```

**优势**:
- 更快的检查速度
- 统一的错误报告
- 减少工具链复杂度

### 3. Oxfmt Formatting (代码格式化)

**说明**: 快速的代码格式化工具

**命令行使用**:
```bash
# 格式化文件
oxfmt --write src/**/*.ts

# 检查格式
oxfmt --check src/**/*.ts

# 格式化所有文件
oxfmt --write
```

**Node.js API**:
```typescript
import { format } from "oxfmt";

const { code } = await format("file.ts", sourceCode, {
  semi: false,
  singleQuote: true,
  tabWidth: 2,
});
```

### 4. VS Code 集成

**安装扩展**:
```bash
code --install-extension oxc.oxc-vscode
```

**配置**:
```json
{
  "oxc.typeAware": true,
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "oxc.oxc-vscode"
}
```

---

## 常用命令

### Linting

```bash
# 基础 lint
oxlint

# Type-aware lint
oxlint --type-aware

# Type-aware lint + 类型检查
oxlint --type-aware --type-check

# 自动修复
oxlint --fix

# 指定文件
oxlint src/**/*.ts

# 忽略特定文件
oxlint --ignore-pattern "**/*.test.ts"
```

### Formatting

```bash
# 格式化文件
oxfmt --write

# 检查格式
oxfmt --check

# 格式化特定文件
oxfmt --write src/**/*.ts

# 查看差异
oxfmt --write && git --no-pager diff
```

---

## 配置文件

### .oxlintrc.json

```json
{
  "plugins": ["typescript", "react"],
  "rules": {
    "typescript/no-floating-promises": "error",
    "typescript/no-unsafe-assignment": "warn",
    "typescript/no-unsafe-call": "error",
    "typescript/no-unsafe-member-access": "warn",
    "react/jsx-key": "error"
  },
  "env": {
    "browser": true,
    "node": true,
    "es2021": true
  }
}
```

### .oxfmtrc.json

```json
{
  "semi": false,
  "singleQuote": true,
  "tabWidth": 2,
  "useTabs": false,
  "trailingComma": "es5",
  "bracketSpacing": true,
  "arrowParens": "always"
}
```

---

## 最佳实践

### 1. CI/CD 集成

```yaml
# .github/workflows/ci.yml
- name: Lint
  run: oxlint --type-aware

- name: Format check
  run: oxfmt --check
```

### 2. Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash
oxfmt --check || exit 1
oxlint --type-aware || exit 1
```

### 3. 性能优化

```bash
# 使用缓存
oxlint --cache

# 并行处理
oxlint --max-workers 4

# 仅检查变更文件
git diff --name-only | grep '\.ts$' | xargs oxlint
```

---

## 与其他工具对比

| 特性 | oxlint | ESLint | Biome |
|------|--------|--------|-------|
| 速度 | 极快 | 慢 | 快 |
| Type-aware | ✅ | ✅ | ❌ |
| 自动修复 | ✅ | ✅ | ✅ |
| 插件生态 | 有限 | 丰富 | 有限 |
| 配置复杂度 | 低 | 高 | 低 |

---

## 常见问题

### Q1: 如何启用 type-aware linting?
```bash
# 命令行
oxlint --type-aware

# VS Code
{
  "oxc.typeAware": true
}
```

### Q2: 如何自动修复问题?
```bash
oxlint --fix
```

### Q3: 如何忽略特定规则?
```json
{
  "rules": {
    "typescript/no-unsafe-assignment": "off"
  }
}
```

### Q4: 如何在 CI 中使用?
```bash
# 检查 lint
oxlint --type-aware

# 检查格式
oxfmt --check

# 失败时退出
oxlint --type-aware || exit 1
```

---

## OpenClaw 特定用法

### 检查代码质量

```bash
# 完整检查 (类型 + lint + 格式)
pnpm check

# 仅 lint
pnpm lint

# 仅格式检查
pnpm format:check
```

### 自动修复

```bash
# 修复 lint 问题并格式化
pnpm lint:fix

# 仅格式化
pnpm format
```

### CI/CD

```yaml
# .github/workflows/ci.yml
- name: Check types and lint and oxfmt
  run: pnpm check
```

---

**文档版本**: oxlint 1.49.0, oxfmt 0.34.0
**最后更新**: 2026-02-24
**来源**: Context7 官方文档
