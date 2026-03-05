# Vitest 官方文档

> 来源: Context7 - /websites/vitest_dev
> 查询时间: 2026-02-24
> 查询内容: vitest configuration files, test execution modes, coverage setup, worker pools (vmForks, forks), maxWorkers configuration, and debugging techniques

---

## 1. Configure maxWorkers with a Number in Vitest

**Source**: https://vitest.dev/config/maxworkers

Sets the maximum number of test workers to a fixed number (e.g., 4) in the Vitest configuration file or via the command line interface. This is useful for controlling resource usage during testing.

```javascript
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    maxWorkers: 4,
  },
})
```

```bash
vitest --maxWorkers=4
```

---

## 2. Switch to Forks Pool to Avoid Worker Termination Errors in Vitest

**Source**: https://vitest.dev/guide/common-errors

Offers a workaround for 'Failed to Terminate Worker' errors that occur when using NodeJS's `fetch` with the default `pool: 'threads'`. It suggests switching to `pool: 'forks'` or `pool: 'vmForks'` in the Vitest configuration.

```typescript
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    pool: 'forks',
  },
})
```

```bash
vitest --pool=forks
```

---

## 3. Switch Test Pool with CLI or Config

**Source**: https://vitest.dev/guide/improving-performance

Optimize test run times by switching the execution pool. The `'threads'` pool can offer better performance than the default `'forks'` pool in larger projects. Use the CLI flag `--pool=threads` or set `test.pool` to `'threads'` in the configuration.

```bash
vitest --pool=threads
```

```javascript
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    pool: 'threads',
  },
})
```

---

## 4. Debugging Vitest Browser Mode

**Source**: https://vitest.dev/guide/debugging

Commands and configuration to enable debugging for Vitest in browser mode. This allows you to inspect browser-specific test execution using Node.js inspector protocols.

```bash
vitest --inspect-brk --browser --no-file-parallelism
```

```typescript
import { defineConfig } from 'vitest/config'
import { playwright } from '@vitest/browser-playwright'

export default defineConfig({
  test: {
    inspectBrk: true,
    fileParallelism: false,
    browser: {
      provider: playwright(),
      instances: [{ browser: 'chromium' }]
    },
  },
})
```

```bash
vitest --inspect-brk=127.0.0.1:3000 --browser --no-file-parallelism
```

---

## 5. Vitest v4.0 Release Notes - Pool Rework

**Source**: https://vitest.dev/guide/migration

Vitest has undergone a significant rework of its pool architecture, moving away from [`tinypool`](https://github.com/tinylibs/tinypool) which was previously responsible for orchestrating test file execution in worker threads. This change, implemented in Vitest v4, removes the dependency on tinypool and introduces a rewritten pool system without new external dependencies.

**Key Changes**:

1. **Unified Worker Configuration**:
   - `maxThreads` and `maxForks` → `maxWorkers`
   - `singleThread` and `singleFork` → `maxWorkers: 1` + `isolate: false`

2. **Module State Reset**:
   - If tests relied on module state being reset between runs, implement a `setupFile` that calls `vi.resetModules()` within a `beforeAll` hook

3. **Simplified Configuration**:
   - `poolOptions` object removed
   - Settings now available as top-level options
   - `memoryLimit` for VM pools renamed to `vmMemoryLimit`

4. **Removed Options**:
   - `threads.useAtomics` removed (open feature request if needed)

---

## OpenClaw 项目中的 Vitest 配置

### 1. 单元测试配置 (vitest.unit.config.ts)

```typescript
import { defineConfig } from "vitest/config";
import baseConfig from "./vitest.config.ts";

const base = baseConfig as unknown as Record<string, unknown>;
const baseTest = (baseConfig as { test?: { include?: string[]; exclude?: string[] } }).test ?? {};
const include = (
  baseTest.include ?? ["src/**/*.test.ts", "extensions/**/*.test.ts", "test/format-error.test.ts"]
).filter((pattern) => !pattern.includes("extensions/"));
const exclude = baseTest.exclude ?? [];

export default defineConfig({
  ...base,
  test: {
    ...baseTest,
    include,
    exclude: [...exclude, "src/gateway/**", "extensions/**"],
  },
});
```

**特点**:
- 排除 gateway 和 extensions (有独立配置)
- 覆盖核心业务逻辑

### 2. E2E 测试配置 (vitest.e2e.config.ts)

```typescript
import os from "node:os";
import { defineConfig } from "vitest/config";
import baseConfig from "./vitest.config.ts";

const base = baseConfig as unknown as Record<string, unknown>;
const isCI = process.env.CI === "true" || process.env.GITHUB_ACTIONS === "true";
const cpuCount = os.cpus().length;
const defaultWorkers = isCI
  ? Math.min(4, Math.max(2, Math.floor(cpuCount * 0.5)))
  : Math.min(8, Math.max(4, Math.floor(cpuCount * 0.6)));
const requestedWorkers = Number.parseInt(process.env.OPENCLAW_E2E_WORKERS ?? "", 10);
const e2eWorkers =
  Number.isFinite(requestedWorkers) && requestedWorkers > 0
    ? Math.min(16, requestedWorkers)
    : defaultWorkers;
const verboseE2E = process.env.OPENCLAW_E2E_VERBOSE === "1";

const baseTest = (baseConfig as { test?: { exclude?: string[] } }).test ?? {};
const exclude = (baseTest.exclude ?? []).filter((p) => p !== "**/*.e2e.test.ts");

export default defineConfig({
  ...base,
  test: {
    ...baseTest,
    pool: "vmForks",
    maxWorkers: e2eWorkers,
    silent: !verboseE2E,
    include: ["test/**/*.e2e.test.ts", "src/**/*.e2e.test.ts"],
    exclude,
  },
});
```

**特点**:
- **隔离池**: vmForks (完全隔离的进程)
- **动态 Worker**: 根据 CPU 核心数和环境自动调整
- **CI 优化**: CI 环境使用 50% CPU,本地使用 60% CPU

### 3. Live 测试配置 (vitest.live.config.ts)

```typescript
import { defineConfig } from "vitest/config";
import baseConfig from "./vitest.config.ts";

const base = baseConfig as unknown as Record<string, unknown>;
const baseTest = (baseConfig as { test?: { exclude?: string[] } }).test ?? {};
const exclude = (baseTest.exclude ?? []).filter((p) => p !== "**/*.live.test.ts");

export default defineConfig({
  ...base,
  test: {
    ...baseTest,
    maxWorkers: 1,
    include: ["src/**/*.live.test.ts"],
    exclude,
  },
});
```

**特点**:
- **单线程**: maxWorkers: 1 (避免并发冲突)
- **用途**: 需要外部服务的集成测试

### 4. Gateway 测试配置 (vitest.gateway.config.ts)

```typescript
import { defineConfig } from "vitest/config";
import baseConfig from "./vitest.config.ts";

const base = baseConfig as unknown as Record<string, unknown>;
const baseTest = (baseConfig as { test?: { exclude?: string[] } }).test ?? {};
const exclude = baseTest.exclude ?? [];

export default defineConfig({
  ...base,
  test: {
    ...baseTest,
    include: ["src/gateway/**/*.test.ts"],
    exclude,
  },
});
```

### 5. 扩展测试配置 (vitest.extensions.config.ts)

```typescript
import { defineConfig } from "vitest/config";
import baseConfig from "./vitest.config.ts";

const base = baseConfig as unknown as Record<string, unknown>;
const baseTest = (baseConfig as { test?: { exclude?: string[] } }).test ?? {};
const exclude = baseTest.exclude ?? [];

export default defineConfig({
  ...base,
  test: {
    ...baseTest,
    include: ["extensions/**/*.test.ts"],
    exclude,
  },
});
```

---

## 关键配置选项

### pool (执行池)
- **类型**: `'threads' | 'forks' | 'vmThreads' | 'vmForks'`
- **说明**: 指定测试执行的隔离方式
- **选项**:
  - `threads`: 线程池 (默认,性能最好)
  - `forks`: 进程池 (更好的隔离)
  - `vmThreads`: VM 线程池
  - `vmForks`: VM 进程池 (最强隔离)

### maxWorkers (最大 Worker 数)
- **类型**: `number`
- **说明**: 并行执行测试的最大 worker 数
- **建议**:
  - CI: `Math.floor(cpuCount * 0.5)`
  - 本地: `Math.floor(cpuCount * 0.6)`

### include (包含模式)
- **类型**: `string[]`
- **说明**: 指定要运行的测试文件模式
- **示例**: `['src/**/*.test.ts', 'test/**/*.spec.ts']`

### exclude (排除模式)
- **类型**: `string[]`
- **说明**: 指定要排除的测试文件模式
- **默认**: `['node_modules', 'dist', '.git']`

### silent (静默模式)
- **类型**: `boolean`
- **说明**: 是否静默输出
- **默认**: `false`

### coverage (覆盖率)
- **类型**: `object`
- **说明**: 配置代码覆盖率
- **示例**:
```typescript
{
  coverage: {
    provider: 'v8',
    reporter: ['text', 'json', 'html'],
    exclude: ['**/*.test.ts', '**/node_modules/**'],
  }
}
```

---

## 测试脚本

```json
{
  "test": "node scripts/test-parallel.mjs",
  "test:fast": "vitest run --config vitest.unit.config.ts",
  "test:e2e": "vitest run --config vitest.e2e.config.ts",
  "test:live": "OPENCLAW_LIVE_TEST=1 vitest run --config vitest.live.config.ts",
  "test:coverage": "vitest run --config vitest.unit.config.ts --coverage",
  "test:watch": "vitest"
}
```

---

## 最佳实践

### 1. 动态 Worker 配置
```typescript
import os from 'node:os'

const cpuCount = os.cpus().length
const isCI = process.env.CI === 'true'

export default defineConfig({
  test: {
    maxWorkers: isCI
      ? Math.floor(cpuCount * 0.5)
      : Math.floor(cpuCount * 0.6),
  },
})
```

### 2. 分层测试配置
```typescript
// 单元测试 - 快速反馈
export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    exclude: ['**/*.e2e.test.ts', '**/*.integration.test.ts'],
  },
})

// E2E 测试 - 完全隔离
export default defineConfig({
  test: {
    pool: 'vmForks',
    maxWorkers: 2,
    include: ['**/*.e2e.test.ts'],
  },
})
```

### 3. 环境变量控制
```typescript
export default defineConfig({
  test: {
    maxWorkers: Number(process.env.TEST_WORKERS) || 4,
    silent: process.env.TEST_SILENT === '1',
  },
})
```

---

## 调试技巧

### 1. VS Code 调试配置
```json
{
  "type": "node",
  "request": "launch",
  "name": "Debug Vitest",
  "runtimeExecutable": "npm",
  "runtimeArgs": ["run", "test:watch"],
  "console": "integratedTerminal",
  "internalConsoleOptions": "neverOpen"
}
```

### 2. 命令行调试
```bash
# 调试单个测试文件
node --inspect-brk ./node_modules/vitest/vitest.mjs run src/example.test.ts

# 调试 E2E 测试
vitest --inspect-brk --pool=forks --no-file-parallelism
```

### 3. 性能分析
```bash
# 生成性能报告
vitest --reporter=verbose --reporter=json --outputFile=test-results.json

# 查看最慢的测试
vitest --reporter=verbose | grep "SLOW"
```

---

## 常见问题

### Q1: 如何解决 "Failed to Terminate Worker" 错误?
```typescript
export default defineConfig({
  test: {
    pool: 'forks', // 或 'vmForks'
  },
})
```

### Q2: 如何提高测试性能?
```typescript
export default defineConfig({
  test: {
    pool: 'threads', // 线程池性能最好
    maxWorkers: Math.floor(os.cpus().length * 0.8),
    isolate: false, // 禁用隔离(谨慎使用)
  },
})
```

### Q3: 如何运行特定测试?
```bash
# 运行特定文件
vitest run src/example.test.ts

# 运行匹配模式的测试
vitest run --grep="should handle errors"

# 运行特定配置
vitest run --config vitest.e2e.config.ts
```

---

**文档版本**: Vitest 4.0.18
**最后更新**: 2026-02-24
**来源**: Context7 官方文档
