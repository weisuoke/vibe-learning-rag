# 核心概念 3：Vitest 测试系统

> OpenClaw 使用 Vitest 实现分层测试策略

---

## 概念介绍

### 什么是 Vitest？

**Vitest**：基于 Vite 的现代化测试框架，专为 TypeScript/JavaScript 项目设计。

**核心特性**：
- **快速执行**：基于 Vite 的快速构建
- **多种隔离模式**：threads, forks, vmForks
- **并发测试**：动态 worker 配置
- **覆盖率支持**：内置 v8 coverage
- **TypeScript 原生支持**：无需额外配置

**OpenClaw 的使用**：
- 5 个独立测试配置
- 分层测试策略
- 动态 worker 优化
- 完整覆盖率报告

---

## Vitest 架构

### 1. Pool 模式

**Pool**：Vitest 执行测试的隔离方式。

**4 种 Pool 模式**：

| Pool 模式 | 隔离级别 | 性能 | 适用场景 |
|----------|---------|------|---------|
| `threads` | 线程隔离 | ⭐⭐⭐⭐⭐ | 单元测试（默认） |
| `forks` | 进程隔离 | ⭐⭐⭐⭐ | 需要进程隔离的测试 |
| `vmThreads` | VM 线程隔离 | ⭐⭐⭐ | 需要完全隔离的单元测试 |
| `vmForks` | VM 进程隔离 | ⭐⭐ | E2E 测试（最强隔离） |

---

**threads（线程池）**：
```typescript
export default defineConfig({
  test: {
    pool: 'threads',  // 默认
    maxWorkers: 4,
  },
})
```

**特点**：
- 最快的执行速度
- 共享内存空间
- 适合单元测试

---

**forks（进程池）**：
```typescript
export default defineConfig({
  test: {
    pool: 'forks',
    maxWorkers: 4,
  },
})
```

**特点**：
- 进程级隔离
- 避免内存泄漏
- 适合集成测试

---

**vmForks（VM 进程池）**：
```typescript
export default defineConfig({
  test: {
    pool: 'vmForks',
    maxWorkers: 2,
  },
})
```

**特点**：
- 最强隔离
- 完全独立的 VM
- 适合 E2E 测试

---

### 2. Worker 配置

**maxWorkers**：并行执行测试的最大 worker 数。

**动态配置**：
```typescript
import os from 'node:os'

const cpuCount = os.cpus().length
const isCI = process.env.CI === 'true'

const workers = isCI
  ? Math.floor(cpuCount * 0.5)  // CI: 50% CPU
  : Math.floor(cpuCount * 0.6)  // 本地: 60% CPU

export default defineConfig({
  test: {
    maxWorkers: workers,
  },
})
```

**为什么这样配置**：
- **CI 环境**：资源受限，避免超载
- **本地环境**：充分利用 CPU，加快测试速度
- **动态调整**：根据 CPU 核心数自动优化

---

### 3. Coverage Provider

**Coverage Provider**：代码覆盖率提供者。

**v8 Provider**：
```typescript
export default defineConfig({
  test: {
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        '**/*.test.ts',
        '**/node_modules/**',
        '**/dist/**',
      ],
    },
  },
})
```

**特点**：
- 基于 V8 引擎的原生覆盖率
- 快速准确
- 支持多种报告格式

---

## OpenClaw 的测试配置

### 1. 单元测试配置

**文件**：`vitest.unit.config.ts`

```typescript
import { defineConfig } from "vitest/config";
import baseConfig from "./vitest.config.ts";

const base = baseConfig as unknown as Record<string, unknown>;
const baseTest = (baseConfig as { test?: { include?: string[]; exclude?: string[] } }).test ?? {};
const include = (
  baseTest.include ?? ["src/**/*.test.ts", "extensions/**/*.test.ts"]
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

**特点**：
- 排除 gateway 和 extensions（有独立配置）
- 覆盖核心业务逻辑
- 使用默认 pool（threads）

**运行**：
```bash
pnpm test:fast
# 或
vitest run --config vitest.unit.config.ts
```

---

### 2. E2E 测试配置

**文件**：`vitest.e2e.config.ts`

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

**特点**：
- **隔离池**：vmForks（完全隔离的进程）
- **动态 Worker**：根据 CPU 核心数和环境自动调整
- **CI 优化**：CI 环境使用 50% CPU，本地使用 60% CPU
- **环境变量控制**：OPENCLAW_E2E_WORKERS, OPENCLAW_E2E_VERBOSE

**运行**：
```bash
pnpm test:e2e
# 或
vitest run --config vitest.e2e.config.ts
```

---

### 3. Live 测试配置

**文件**：`vitest.live.config.ts`

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

**特点**：
- **单线程**：maxWorkers: 1（避免并发冲突）
- **用途**：需要外部服务的集成测试
- **环境变量**：OPENCLAW_LIVE_TEST=1

**运行**：
```bash
pnpm test:live
# 或
OPENCLAW_LIVE_TEST=1 vitest run --config vitest.live.config.ts
```

---

### 4. Gateway 测试配置

**文件**：`vitest.gateway.config.ts`

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

**特点**：
- 专门测试 Gateway 模块
- 使用默认 pool（threads）

---

### 5. 扩展测试配置

**文件**：`vitest.extensions.config.ts`

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

**特点**：
- 专门测试扩展模块
- 使用默认 pool（threads）

---

## 测试编写规范

### 1. 基本结构

```typescript
import { describe, it, expect, beforeAll, afterAll } from 'vitest'

describe('Gateway', () => {
  beforeAll(async () => {
    // 测试前的准备工作
  })

  afterAll(async () => {
    // 测试后的清理工作
  })

  it('should start successfully', async () => {
    // 测试逻辑
    const result = await startGateway()
    expect(result).toBe(true)
  })

  it('should handle messages', async () => {
    const message = { text: 'Hello' }
    const response = await sendMessage(message)
    expect(response).toHaveProperty('id')
  })
})
```

---

### 2. 断言

```typescript
// 基本断言
expect(value).toBe(expected)
expect(value).toEqual(expected)
expect(value).toBeTruthy()
expect(value).toBeFalsy()

// 对象断言
expect(obj).toHaveProperty('key')
expect(obj).toMatchObject({ key: 'value' })

// 数组断言
expect(arr).toContain(item)
expect(arr).toHaveLength(3)

// 异步断言
await expect(promise).resolves.toBe(value)
await expect(promise).rejects.toThrow(Error)

// 函数断言
expect(fn).toHaveBeenCalled()
expect(fn).toHaveBeenCalledWith(arg1, arg2)
```

---

### 3. Mock

```typescript
import { vi } from 'vitest'

// Mock 函数
const mockFn = vi.fn()
mockFn.mockReturnValue('mocked')
mockFn.mockResolvedValue('async mocked')

// Mock 模块
vi.mock('./module', () => ({
  default: vi.fn(),
  namedExport: vi.fn(),
}))

// Spy
const spy = vi.spyOn(obj, 'method')
spy.mockImplementation(() => 'mocked')
```

---

### 4. 钩子

```typescript
import { beforeAll, afterAll, beforeEach, afterEach } from 'vitest'

// 所有测试前执行一次
beforeAll(async () => {
  await setupDatabase()
})

// 所有测试后执行一次
afterAll(async () => {
  await teardownDatabase()
})

// 每个测试前执行
beforeEach(() => {
  resetState()
})

// 每个测试后执行
afterEach(() => {
  cleanup()
})
```

---

## 覆盖率配置

### 1. 基本配置

```typescript
export default defineConfig({
  test: {
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      reportsDirectory: './coverage',
      exclude: [
        '**/*.test.ts',
        '**/*.spec.ts',
        '**/node_modules/**',
        '**/dist/**',
        '**/.vscode/**',
      ],
    },
  },
})
```

---

### 2. 覆盖率阈值

```typescript
export default defineConfig({
  test: {
    coverage: {
      provider: 'v8',
      thresholds: {
        lines: 80,
        functions: 80,
        branches: 80,
        statements: 80,
      },
    },
  },
})
```

---

### 3. 运行覆盖率

```bash
# 生成覆盖率报告
pnpm test:coverage

# 查看 HTML 报告
open coverage/index.html
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

---

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

---

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

### 4. test.projects 管理

```typescript
// 根目录 vitest.config.ts
export default defineConfig({
  test: {
    projects: ['packages/*/vitest.config.ts']
  }
})

// packages/core/vitest.config.ts
export default defineConfig({
  test: {
    include: ['src/**/*.test.ts']
  }
})
```

---

### 5. 共享配置

```typescript
// shared/vitest.config.ts
export const sharedConfig = {
  coverage: {
    provider: 'v8',
    reporter: ['text', 'json', 'html']
  }
}

// packages/core/vitest.config.ts
import { mergeConfig } from 'vitest/config'
import { sharedConfig } from '../../shared/vitest.config'

export default mergeConfig(sharedConfig, {
  test: {
    include: ['src/**/*.test.ts']
  }
})
```

---

## 与 OpenClaw 的关系

### 1. 为什么 OpenClaw 需要 5 个测试配置？

**问题**：不同类型的测试有不同的需求

**解决方案**：分层测试配置
- **单元测试**：快速反馈，覆盖核心逻辑
- **E2E 测试**：完整流程，真实环境
- **Live 测试**：外部服务集成（需要单线程）
- **Gateway 测试**：Gateway 特定逻辑
- **Extension 测试**：扩展系统测试

---

### 2. OpenClaw 的测试策略

```
┌─────────────────────────────────────┐
│  1. 单元测试 (vitest.unit.config)  │
│     - 核心业务逻辑                  │
│     - 快速反馈                      │
│     - threads pool                  │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  2. Gateway 测试 (vitest.gateway)   │
│     - Gateway 模块                  │
│     - threads pool                  │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  3. Extension 测试 (vitest.ext)     │
│     - 扩展模块                      │
│     - threads pool                  │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  4. E2E 测试 (vitest.e2e.config)    │
│     - 完整流程                      │
│     - vmForks pool                  │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  5. Live 测试 (vitest.live.config)  │
│     - 外部服务集成                  │
│     - maxWorkers: 1                 │
└─────────────────────────────────────┘
```

---

### 3. OpenClaw 的测试流程

```bash
# 1. 快速单元测试
pnpm test:fast

# 2. 完整测试（并行）
pnpm test

# 3. E2E 测试
pnpm test:e2e

# 4. Live 测试
pnpm test:live

# 5. 覆盖率
pnpm test:coverage
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

---

### 2. 命令行调试

```bash
# 调试单个测试文件
node --inspect-brk ./node_modules/vitest/vitest.mjs run src/example.test.ts

# 调试 E2E 测试
vitest --inspect-brk --pool=forks --no-file-parallelism
```

---

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

---

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

---

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

## 核心洞察

### 1. Vitest 不是简单的测试框架

**表面**：运行测试

**实际**：
- 多种隔离模式
- 动态 worker 配置
- 覆盖率分析
- 性能优化

---

### 2. 分层测试的威力

**表面**：多个配置文件

**实际**：
- 不同测试类型不同策略
- 优化执行速度
- 确保测试质量
- 避免测试冲突

---

### 3. 动态配置的重要性

**表面**：maxWorkers 配置

**实际**：
- 根据环境自动调整
- CI 和本地不同策略
- 充分利用 CPU
- 避免资源超载

---

## 一句话总结

**Vitest 通过多种 pool 模式和动态 worker 配置实现高效测试，OpenClaw 使用 5 个独立配置实现分层测试策略。**

---

[来源: reference/03_vitest_official_docs.md, reference/17_vitest_multi_config.md, reference/01_source_code_analysis.md]
