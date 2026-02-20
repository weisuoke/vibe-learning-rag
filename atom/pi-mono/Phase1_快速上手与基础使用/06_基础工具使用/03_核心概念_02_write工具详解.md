# write 工具详解

> 深入理解 Pi 的文件写入工具：原子操作、LSP 集成、安全机制

---

## 概述

**write 工具**是 Pi 的 4 个基础工具之一，负责创建新文件或完全覆盖现有文件。它采用原子操作保证数据安全，并支持 LSP 诊断集成。

**核心职责：**
- 创建新文件
- 完全覆盖现有文件
- 原子写入操作
- 触发 LSP 诊断（oh-my-pi）

---

## 设计哲学

### 1. 声明式语义

**write 工具表达"文件应该是这样"**

```typescript
// write 工具的语义
await write("src/app.ts", `
export function app() {
  return "Hello World"
}
`)

// 语义：我知道文件的最终状态
// 不关心：文件之前是什么样子
// 结果：文件内容完全替换为新内容
```

**对比 edit 工具：**
```typescript
// edit 工具的语义
await edit("src/app.ts", oldContent, newContent)

// 语义：我知道要改什么
// 关心：文件之前的内容
// 结果：只修改指定部分
```

---

### 2. 原子操作

**要么成功，要么失败，不会出现中间状态**

```typescript
// 原子写入的保证
await write("src/app.ts", newContent)

// 保证 1：写入成功 → 文件内容完全是 newContent
// 保证 2：写入失败 → 文件内容保持不变（或不存在）
// 保证 3：不会出现部分写入的情况
```

**为什么重要？**
- 避免文件损坏
- 保证数据一致性
- 支持并发操作

---

### 3. 覆盖警告

**write 会覆盖现有文件，需要谨慎使用**

```typescript
// 场景 1：创建新文件 ✅
await write("src/new-feature.ts", code)
// 安全：文件不存在，创建新文件

// 场景 2：覆盖现有文件 ⚠️
await write("src/existing.ts", code)
// 警告：文件存在，所有旧内容将丢失

// 场景 3：应该用 edit ✅
await edit("src/existing.ts", oldCode, newCode)
// 正确：只修改特定部分，保留其他内容
```

---

## 实现细节

### 1. 核心实现

**基于 Node.js fs 模块的原子写入**

```typescript
// 简化的实现（实际更复杂）
class WriteTool {
  async execute(path: string, content: string): Promise<void> {
    // 1. 创建临时文件
    const tempPath = `${path}.tmp.${Date.now()}`

    try {
      // 2. 写入临时文件
      fs.writeFileSync(tempPath, content, 'utf-8')

      // 3. 原子重命名（这是关键）
      fs.renameSync(tempPath, path)
      // renameSync 在大多数文件系统上是原子操作

    } catch (error) {
      // 4. 失败时清理临时文件
      if (fs.existsSync(tempPath)) {
        fs.unlinkSync(tempPath)
      }
      throw error
    }
  }
}
```

**原子性的保证：**
- `fs.renameSync()` 在 POSIX 系统上是原子操作
- 要么完全成功，要么完全失败
- 不会出现部分写入的情况

**来源：**
- [pi-mono write.ts implementation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/core/tools/write.ts) - 核心实现，2026

---

### 2. 目录自动创建

**自动创建父目录**

```typescript
class WriteTool {
  async execute(path: string, content: string): Promise<void> {
    // 1. 提取目录路径
    const dir = path.substring(0, path.lastIndexOf('/'))

    // 2. 如果目录不存在，创建它
    if (dir && !fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true })
    }

    // 3. 写入文件
    const tempPath = `${path}.tmp.${Date.now()}`
    fs.writeFileSync(tempPath, content, 'utf-8')
    fs.renameSync(tempPath, path)
  }
}
```

**使用示例：**
```typescript
// 自动创建 src/features/auth/ 目录
await write("src/features/auth/login.ts", code)

// 不需要先执行
// await bash("mkdir -p src/features/auth")
```

---

### 3. 编码处理

**默认使用 UTF-8 编码**

```typescript
class WriteTool {
  async execute(path: string, content: string): Promise<void> {
    // 默认 UTF-8 编码
    fs.writeFileSync(tempPath, content, 'utf-8')

    // 支持其他编码（通过 Extensions）
    // fs.writeFileSync(tempPath, content, 'latin1')
  }
}
```

---

## LSP 集成（oh-my-pi）

### 1. 写入后自动诊断

**oh-my-pi 的重要增强：写入后自动运行 LSP 诊断**

```typescript
// oh-my-pi 的 write 工具增强
class WriteTool {
  async execute(path: string, content: string): Promise<void> {
    // 1. 写入文件
    const tempPath = `${path}.tmp.${Date.now()}`
    fs.writeFileSync(tempPath, content, 'utf-8')
    fs.renameSync(tempPath, path)

    // 2. 触发 LSP 诊断（oh-my-pi 特性）
    if (this.lspEnabled && this.isCodeFile(path)) {
      const diagnostics = await this.lsp.getDiagnostics(path)

      // 3. 返回诊断结果
      if (diagnostics.length > 0) {
        return {
          success: true,
          diagnostics: diagnostics.map(d => ({
            line: d.range.start.line,
            message: d.message,
            severity: d.severity
          }))
        }
      }
    }
  }

  private isCodeFile(path: string): boolean {
    const ext = path.split('.').pop()
    return ['ts', 'tsx', 'js', 'jsx', 'py'].includes(ext || '')
  }
}
```

**实际效果：**
```typescript
// Agent 写入代码
await write("src/app.ts", `
export function add(a: number, b: number) {
  return a + b
}
`)

// oh-my-pi 自动返回 LSP 诊断
// 输出：
// ✅ No errors found
// 或
// ❌ Type error at line 2: ...
```

**来源：**
- [oh-my-pi CHANGELOG](https://github.com/can1357/oh-my-pi/blob/main/packages/coding-agent/CHANGELOG.md) - LSP diagnostics on write，2026
- [oh-my-pi GitHub](https://github.com/can1357/oh-my-pi) - LSP 集成实现，2025

---

### 2. Format-on-Write

**oh-my-pi 支持写入后自动格式化**

```typescript
// oh-my-pi 的 format-on-write 特性
class WriteTool {
  async execute(path: string, content: string): Promise<void> {
    // 1. 写入文件
    fs.writeFileSync(tempPath, content, 'utf-8')
    fs.renameSync(tempPath, path)

    // 2. 自动格式化（如果启用）
    if (this.formatOnWrite && this.isCodeFile(path)) {
      const formatted = await this.lsp.format(path)
      if (formatted !== content) {
        // 重新写入格式化后的内容
        fs.writeFileSync(path, formatted, 'utf-8')
      }
    }
  }
}
```

**配置：**
```json
// .pi/settings.json
{
  "formatOnWrite": true,
  "lsp": {
    "enabled": true
  }
}
```

---

## 高级特性

### 1. 错误处理

**清晰的错误信息**

```typescript
class WriteTool {
  async execute(path: string, content: string): Promise<void> {
    try {
      // 检查路径是否是目录
      if (fs.existsSync(path) && fs.statSync(path).isDirectory()) {
        throw new Error(`Cannot write to directory: ${path}`)
      }

      // 检查权限
      const dir = path.substring(0, path.lastIndexOf('/'))
      if (dir && fs.existsSync(dir)) {
        fs.accessSync(dir, fs.constants.W_OK)
      }

      // 写入文件
      const tempPath = `${path}.tmp.${Date.now()}`
      fs.writeFileSync(tempPath, content, 'utf-8')
      fs.renameSync(tempPath, path)

    } catch (error) {
      // 提供有用的错误信息
      if (error.code === 'EACCES') {
        throw new Error(`Permission denied: ${path}`)
      }
      if (error.code === 'ENOSPC') {
        throw new Error(`No space left on device`)
      }
      if (error.code === 'EROFS') {
        throw new Error(`Read-only file system`)
      }
      throw error
    }
  }
}
```

**错误类型：**
- 权限不足（EACCES）
- 磁盘空间不足（ENOSPC）
- 只读文件系统（EROFS）
- 路径是目录

---

### 2. 内容验证

**写入前验证内容**

```typescript
class WriteTool {
  async execute(path: string, content: string): Promise<void> {
    // 1. 验证内容不为空（可选）
    if (content.trim().length === 0) {
      console.warn(`Warning: Writing empty file: ${path}`)
    }

    // 2. 验证内容大小
    const sizeInMB = Buffer.byteLength(content, 'utf-8') / (1024 * 1024)
    if (sizeInMB > 10) {
      throw new Error(`Content too large: ${sizeInMB.toFixed(2)}MB (max 10MB)`)
    }

    // 3. 写入文件
    const tempPath = `${path}.tmp.${Date.now()}`
    fs.writeFileSync(tempPath, content, 'utf-8')
    fs.renameSync(tempPath, path)
  }
}
```

---

### 3. 备份机制

**写入前备份现有文件（可选）**

```typescript
class WriteTool {
  async execute(path: string, content: string, options?: { backup?: boolean }): Promise<void> {
    // 1. 如果启用备份且文件存在
    if (options?.backup && fs.existsSync(path)) {
      const backupPath = `${path}.backup.${Date.now()}`
      fs.copyFileSync(path, backupPath)
    }

    // 2. 写入文件
    const tempPath = `${path}.tmp.${Date.now()}`
    fs.writeFileSync(tempPath, content, 'utf-8')
    fs.renameSync(tempPath, path)
  }
}
```

---

## 与 Extensions 集成

### 1. 覆盖 write 工具

**通过 Extensions 自定义 write 行为**

```typescript
// 示例：添加自动格式化
extensions.register({
  name: "auto-format-write",
  tool: {
    name: "write",
    override: true,
    execute: async (path: string, content: string) => {
      // 1. 写入文件
      const tempPath = `${path}.tmp.${Date.now()}`
      fs.writeFileSync(tempPath, content, 'utf-8')
      fs.renameSync(tempPath, path)

      // 2. 自动格式化
      if (path.endsWith('.ts') || path.endsWith('.js')) {
        execSync(`prettier --write ${path}`)
      }
    }
  }
})
```

---

### 2. 添加 write hooks

**在写入前后插入逻辑**

```typescript
// 示例：写入日志
extensions.register({
  name: "write-logger",
  hooks: {
    beforeWrite: async (path: string, content: string) => {
      console.log(`Writing ${content.length} bytes to ${path}`)
      return { path, content }
    },
    afterWrite: async (path: string) => {
      console.log(`Successfully wrote ${path}`)
    }
  }
})
```

---

### 3. 远程文件写入

**通过 Extensions 支持远程文件**

```typescript
// 示例：SSH 远程写入
extensions.register({
  name: "ssh-write",
  tool: {
    name: "write",
    override: true,
    execute: async (path: string, content: string) => {
      // 如果路径以 ssh:// 开头
      if (path.startsWith('ssh://')) {
        const { host, remotePath } = parseSSHUrl(path)
        return await ssh.writeFile(host, remotePath, content)
      }

      // 否则使用默认实现
      const tempPath = `${path}.tmp.${Date.now()}`
      fs.writeFileSync(tempPath, content, 'utf-8')
      fs.renameSync(tempPath, path)
    }
  }
})

// 使用
await write("ssh://server.com/path/to/file.ts", code)
```

---

## 实际应用场景

### 场景 1：创建新功能

```typescript
// 1. 创建新模块
await write("src/features/payment.ts", `
export class PaymentService {
  async processPayment(amount: number) {
    // implementation
  }
}
`)

// 2. 创建测试文件
await write("src/features/payment.test.ts", `
import { PaymentService } from './payment'

describe('PaymentService', () => {
  it('should process payment', async () => {
    // test implementation
  })
})
`)

// 3. 创建导出文件
await write("src/features/index.ts", `
export { PaymentService } from './payment'
`)
```

---

### 场景 2：生成配置文件

```typescript
// 1. 生成 TypeScript 配置
await write("tsconfig.json", JSON.stringify({
  compilerOptions: {
    target: "ES2020",
    module: "commonjs",
    strict: true
  }
}, null, 2))

// 2. 生成 ESLint 配置
await write(".eslintrc.json", JSON.stringify({
  extends: ["eslint:recommended"],
  rules: {
    "no-console": "warn"
  }
}, null, 2))

// 3. 生成 .gitignore
await write(".gitignore", `
node_modules/
dist/
.env
*.log
`)
```

---

### 场景 3：代码生成

```typescript
// 1. 读取模板
const template = await read("templates/component.tsx")

// 2. 生成代码
const code = template
  .replace(/{{ComponentName}}/g, "UserProfile")
  .replace(/{{props}}/g, "{ userId: string }")

// 3. 写入新文件
await write("src/components/UserProfile.tsx", code)
```

---

### 场景 4：完全重写文件

```typescript
// 场景：文件内容完全不同，用 write 而非 edit

// 1. 读取旧文件（了解结构）
const oldCode = await read("src/legacy.ts")

// 2. 完全重写（新架构）
await write("src/legacy.ts", `
// 完全重写的新代码
export class NewArchitecture {
  // 全新实现
}
`)

// 3. 验证
await bash("npm test")
```

---

## 性能优化

### 1. 批量写入

**一次写入多个文件**

```typescript
// 不好的做法：顺序写入
await write("file1.ts", content1)
await write("file2.ts", content2)
await write("file3.ts", content3)

// 好的做法：并行写入（如果文件独立）
await Promise.all([
  write("file1.ts", content1),
  write("file2.ts", content2),
  write("file3.ts", content3)
])
```

---

### 2. 避免不必要的写入

**检查内容是否改变**

```typescript
// 优化：只在内容改变时写入
async function writeIfChanged(path: string, newContent: string) {
  // 1. 检查文件是否存在
  if (fs.existsSync(path)) {
    // 2. 读取现有内容
    const oldContent = fs.readFileSync(path, 'utf-8')

    // 3. 如果内容相同，跳过写入
    if (oldContent === newContent) {
      console.log(`Skipping ${path} (no changes)`)
      return
    }
  }

  // 4. 内容不同，执行写入
  await write(path, newContent)
}
```

---

### 3. 大文件处理

**分块写入大文件**

```typescript
// 对于非常大的文件，使用流式写入
import { createWriteStream } from 'fs'

async function writeLargeFile(path: string, content: string) {
  const stream = createWriteStream(path)

  // 分块写入
  const chunkSize = 1024 * 1024  // 1MB
  for (let i = 0; i < content.length; i += chunkSize) {
    const chunk = content.slice(i, i + chunkSize)
    stream.write(chunk)
  }

  stream.end()
}
```

---

## 与其他工具的协作

### write + read

**创建后验证**

```typescript
// 1. 写入文件
await write("src/config.ts", configCode)

// 2. 读取验证
const written = await read("src/config.ts")

// 3. Agent 验证内容是否正确
```

---

### write + bash

**写入后测试**

```typescript
// 1. 写入新代码
await write("src/feature.ts", newCode)

// 2. 运行测试
await bash("npm test")

// 3. 如果测试失败，修复
await edit("src/feature.ts", buggyCode, fixedCode)
```

---

### write + edit

**创建后微调**

```typescript
// 1. 创建基础文件
await write("src/api.ts", baseCode)

// 2. 微调特定部分
await edit("src/api.ts", "localhost", "production.com")
```

---

## 常见问题

### Q1: write 会覆盖现有文件吗？

**A:** 是的，write 会完全覆盖现有文件。

```typescript
// 文件存在
// src/app.ts: "old content"

// 执行 write
await write("src/app.ts", "new content")

// 结果
// src/app.ts: "new content"  (旧内容丢失)
```

**建议：** 修改现有文件应该用 edit，而非 write。

---

### Q2: write 是原子操作吗？

**A:** 是的，write 使用原子重命名保证数据安全。

```typescript
// 原子性保证
await write("src/app.ts", newContent)

// 要么：文件内容完全是 newContent
// 要么：写入失败，文件保持不变
// 不会：出现部分写入的情况
```

---

### Q3: write 会自动创建目录吗？

**A:** 是的，write 会自动创建父目录。

```typescript
// 目录不存在
await write("src/features/auth/login.ts", code)

// 自动创建 src/features/auth/ 目录
```

---

### Q4: write 支持哪些编码？

**A:** 默认 UTF-8，可以通过 Extensions 支持其他编码。

```typescript
// 默认 UTF-8
await write("file.ts", content)

// 通过 Extension 支持其他编码
extensions.register({
  name: "latin1-write",
  tool: {
    name: "write",
    override: true,
    execute: async (path, content) => {
      fs.writeFileSync(path, content, 'latin1')
    }
  }
})
```

---

### Q5: write 有文件大小限制吗？

**A:** 通常限制在 10MB 左右，具体取决于实现。

```typescript
// 如果内容太大
await write("huge-file.ts", hugeContent)  // 可能抛出错误

// 解决方案：使用流式写入或分批写入
```

---

## 2025-2026 最新发展

### LSP Diagnostics on Write（oh-my-pi）

2026 年 oh-my-pi 引入的重要特性：写入后自动运行 LSP 诊断。

**优势：**
- 即时反馈类型错误
- 即时反馈语法错误
- 提高代码质量
- 减少调试时间

**示例：**
```typescript
// Agent 写入代码
await write("src/app.ts", `
export function add(a: number, b: number) {
  return a + b
}
`)

// oh-my-pi 自动返回诊断
// ✅ No errors found

// 如果有错误
await write("src/app.ts", `
export function add(a: number, b: number) {
  return a + c  // 错误：c 未定义
}
`)

// ❌ Error at line 3: Cannot find name 'c'
```

**来源：**
- [oh-my-pi CHANGELOG](https://github.com/can1357/oh-my-pi/blob/main/packages/coding-agent/CHANGELOG.md) - LSP diagnostics on write，2026

---

### Atomic Abort-and-Reprompt

oh-my-pi 引入的原子中止和重新提示机制：

**用途：**
- 写入失败时自动回滚
- 提示 Agent 重新尝试
- 保证数据一致性

**来源：**
- [oh-my-pi CHANGELOG](https://github.com/can1357/oh-my-pi/blob/main/packages/coding-agent/CHANGELOG.md) - Atomic abort-and-reprompt，2026

---

## 设计权衡

### 优势

1. **原子操作** - 保证数据安全
2. **简单接口** - 只需要路径和内容
3. **自动创建目录** - 减少手动操作
4. **LSP 集成** - 即时反馈（oh-my-pi）

### 劣势

1. **覆盖风险** - 会丢失旧内容
2. **大文件限制** - 不适合非常大的文件
3. **无 diff** - 看不到修改了什么（应该用 edit）

### Pi 的选择

**宁可语义清晰，也不要功能混杂。**

write 专注于"创建/覆盖"，edit 专注于"修改"，职责分明。

---

## 最佳实践

### 1. 创建新文件用 write

```typescript
// ✅ 好的做法
await write("src/new-feature.ts", code)

// ❌ 不好的做法
await bash("touch src/new-feature.ts")
await edit("src/new-feature.ts", "", code)
```

---

### 2. 修改现有文件用 edit

```typescript
// ✅ 好的做法
await edit("src/app.ts", oldCode, newCode)

// ❌ 不好的做法
await read("src/app.ts")
// 在内存中修改
await write("src/app.ts", modifiedCode)  // 可能出错
```

---

### 3. 完全重写用 write

```typescript
// ✅ 好的做法：内容完全不同
await write("src/legacy.ts", completelyNewCode)

// ❌ 不好的做法：只修改一行
await write("src/app.ts", entireFileWithOneLineChanged)
// 应该用 edit
```

---

### 4. 写入后验证

```typescript
// ✅ 好的做法
await write("src/feature.ts", code)
await bash("npm test")  // 验证

// ❌ 不好的做法
await write("src/feature.ts", code)
// 没有验证，可能有错误
```

---

## 延伸阅读

### 官方文档
- [pi-mono write.ts](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/core/tools/write.ts) - 核心实现
- [pi-mono README](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/README.md) - 官方文档

### 社区资源
- [oh-my-pi](https://github.com/can1357/oh-my-pi) - LSP 集成增强
- [awesome-pi-agent](https://github.com/qualisero/awesome-pi-agent) - 社区资源列表

---

## 下一步学习

完成 write 工具学习后，建议学习：

### 其他工具
- [03_核心概念_01_read工具详解](./03_核心概念_01_read工具详解.md) - read 工具深入
- [03_核心概念_03_edit工具详解](./03_核心概念_03_edit工具详解.md) - edit 工具深入
- [03_核心概念_04_bash工具详解](./03_核心概念_04_bash工具详解.md) - bash 工具深入

### 实战应用
- [07_实战代码_02_write工具实战](./07_实战代码_02_write工具实战.md) - write 工具实战
- [07_实战代码_05_工具组合模式](./07_实战代码_05_工具组合模式.md) - 工具组合

---

**版本：** v1.0
**最后更新：** 2026-02-19
**维护者：** Claude Code
