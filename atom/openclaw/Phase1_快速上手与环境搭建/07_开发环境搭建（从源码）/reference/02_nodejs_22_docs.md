# Node.js 22 官方文档

**来源**: Context7 - /nodejs/node/v22.17.0
**查询时间**: 2026-02-23
**用途**: 源码结构层 - 模块系统、运行时特性、开发要求

---

## 在 ESM 中创建 CommonJS Require 函数

**来源**: https://github.com/nodejs/node/blob/v22.17.0/doc/api/module.md

演示如何在 ECMAScript 模块 (ESM) 上下文中使用 `module.createRequire(filename)` 创建 `require` 函数。这允许从 ESM 中加载 CommonJS 模块，桥接模块系统之间的兼容性。`filename` 参数指定新 `require` 函数的上下文。

```javascript
// ESM 模块
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);

// sibling-module.js is a CommonJS module.
const siblingModule = require('./sibling-module');
```

---

## CommonJS 与 require 的互操作性

**来源**: https://github.com/nodejs/node/blob/v22.17.0/doc/api/esm.md

描述 CommonJS `require` 函数与 ES 模块交互时的行为。指定 `require` 目前仅支持加载同步 ES 模块（不使用顶层 `await` 的模块）。

```
互操作性: require
  行为: CommonJS 模块的 require 目前仅支持加载同步 ES 模块（即不使用顶层 await 的 ES 模块）。
```

---

## 从 ES 模块访问 CommonJS 命名导出

**来源**: https://github.com/nodejs/node/blob/v22.17.0/doc/api/esm.md

解释 Node.js 如何尝试静态分析并将 CommonJS 命名导出作为单独的 ES 模块导出公开。提供了一个带有 `exports.name` 的 CommonJS 模块示例，并演示如何在 ES 模块中将其作为命名导出、默认导出或模块命名空间导入。

```javascript
// cjs.cjs
exports.name = 'exported';
```

```javascript
// ESM 中导入
import { name } from './cjs.cjs';
console.log(name);
// Prints: 'exported'

import cjs from './cjs.cjs';
console.log(cjs);
// Prints: { name: 'exported' }

import * as m from './cjs.cjs';
console.log(m);
// Prints: [Module] { default: { name: 'exported' }, name: 'exported' }
```

---

## 在 CommonJS 中使用 require() 加载基本 ECMAScript 模块

**来源**: https://github.com/nodejs/node/blob/v22.17.0/doc/api/modules.md

此示例展示 CommonJS 模块如何使用 `require()` 导入先前定义的 `distance.mjs` 和 `point.mjs` 模块。它说明了返回的模块命名空间对象的结构，包括默认导出的 `__esModule` 属性。

```javascript
// CommonJS 模块
const distance = require('./distance.mjs');
console.log(distance);
// [Module: null prototype] {
//   distance: [Function: distance]
// }

const point = require('./point.mjs');
console.log(point);
// [Module: null prototype] {
//   default: [class Point],
//   __esModule: true,
// }
```

---

## Node.js 内置模块加载和 API

**来源**: https://github.com/nodejs/node/blob/v22.17.0/doc/api/modules.md

解释如何使用 `require()` 加载 Node.js 内置模块，区分可以直接加载的模块（如 `http`）和需要 `node:` 前缀的模块（如 `node:sea`）。还引用了 `module.builtinModules` 来列出内置模块。

```
模块加载模式:
  require(module_name: string): Module
    module_name: 要加载的模块标识符。
    描述: 从 Node.js 环境加载模块。
    行为:
      - 如果 module_name 是内置模块（如 'http'），则优先加载。
      - 如果 module_name 以 'node:' 开头（如 'node:http'），则绕过 require 缓存并始终返回内置模块。

需要 'node:' 前缀的模块:
  node:sea
  node:sqlite
  node:test
  node:test/reporters

全局属性:
  module.builtinModules: string[]
    描述: 包含可以不使用 'node:' 前缀加载的内置模块名称的数组。
```

---

## OpenClaw 项目中的应用

### Node.js 版本要求

OpenClaw 要求 Node.js 22.12.0 或更高版本：

```json
{
  "engines": {
    "node": ">=22.12.0"
  }
}
```

### 模块系统配置

OpenClaw 使用 ES 模块系统，在 package.json 中配置：

```json
{
  "type": "module"
}
```

### TypeScript 配置中的模块设置

```json
{
  "compilerOptions": {
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "target": "ES2023"
  }
}
```

---

## Node.js 22 关键特性

### 1. 模块系统增强

- **ESM 优先**: Node.js 22 进一步优化了 ES 模块支持
- **CommonJS 互操作**: 改进了 ESM 和 CommonJS 之间的互操作性
- **node: 协议**: 推荐使用 `node:` 前缀导入内置模块

### 2. 性能改进

- **V8 引擎更新**: 使用最新的 V8 引擎，性能显著提升
- **启动速度**: 应用启动速度更快
- **内存优化**: 更好的内存管理

### 3. 开发体验

- **更好的错误消息**: 更清晰的错误堆栈和提示
- **调试支持**: 改进的调试工具支持
- **TypeScript 支持**: 更好的 TypeScript 集成

---

## 最佳实践

1. **使用 node: 协议**: 导入内置模块时使用 `node:` 前缀
   ```javascript
   import { readFile } from 'node:fs/promises';
   ```

2. **ESM 优先**: 新项目优先使用 ES 模块
   ```json
   {
     "type": "module"
   }
   ```

3. **版本管理**: 使用 nvm 或 asdf 管理 Node.js 版本
   ```bash
   asdf install nodejs 22.12.0
   asdf global nodejs 22.12.0
   ```

4. **模块解析**: 配置 TypeScript 使用 NodeNext 模块解析
   ```json
   {
     "compilerOptions": {
       "module": "NodeNext",
       "moduleResolution": "NodeNext"
     }
   }
   ```

---

## 常见问题

### Q: 如何在 ESM 中使用 __dirname 和 __filename?

```javascript
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
```

### Q: 如何在 ESM 中加载 JSON 文件?

```javascript
// Node.js 22 支持 import assertions
import data from './data.json' with { type: 'json' };

// 或使用动态导入
const data = await import('./data.json', { with: { type: 'json' } });
```

### Q: 如何处理 CommonJS 和 ESM 混合项目?

使用 `createRequire` 在 ESM 中加载 CommonJS 模块：

```javascript
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);

const cjsModule = require('./cjs-module');
```

---

**参考资料**:
- Node.js 官方文档: https://nodejs.org/docs/latest-v22.x/api/
- Context7 Node.js 文档: https://context7.com/nodejs/node/
- Node.js 22 发布说明: https://nodejs.org/en/blog/release/v22.0.0
