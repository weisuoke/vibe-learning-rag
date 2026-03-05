# VS Code 调试 TypeScript Node.js 项目配置

> 来源: Grok Web Search
> 查询时间: 2026-02-24
> 查询内容: VS Code debugging TypeScript Node.js project configuration launch.json 2026

---

## 搜索结果

### 1. VS Code TypeScript调试官方文档

**URL**: https://code.visualstudio.com/docs/typescript/typescript-debugging

**描述**: 官方指南详解TypeScript Node.js项目launch.json配置，使用source maps与preLaunchTask实现源码调试

---

### 2. VS Code Node.js调试官方文档

**URL**: https://code.visualstudio.com/docs/nodejs/nodejs-debugging

**描述**: Node.js运行时调试完整说明，支持TypeScript的launch.json示例、配置属性与sourceMap设置

---

### 3. Stack Overflow调试TypeScript Node.js

**URL**: https://stackoverflow.com/questions/35606423/visual-studio-code-debug-node-js-through-typescript

**描述**: 社区分享VS Code直接调试TypeScript编写Node.js应用的launch.json配置方案与常见问题解决

---

### 4. NodeJS+TypeScript VSCode调试教程

**URL**: https://medium.com/@matttom/setup-the-vscode-debugger-for-a-nodejs-typescript-project-a9c0a5042687

**描述**: 实战教程：创建launch.json指向build输出目录，实现TypeScript Node项目的断点调试

---

### 5. VS Code调试配置launch.json指南

**URL**: https://code.visualstudio.com/docs/debugtest/debugging-configuration

**描述**: 官方文档解释launch.json结构、属性及变量替换，适用于Node.js与TypeScript调试场景

---

### 6. 调试TypeScript Node.js应用教程

**URL**: https://tsh.io/blog/visual-studio-code-typescript-debugging

**描述**: 逐步指导修改launch.json支持TypeScript，包括outFiles、sourceMaps等关键设置

---

## launch.json 配置详解

### 基础配置

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "node",
      "request": "launch",
      "name": "Debug TypeScript",
      "program": "${workspaceFolder}/src/index.ts",
      "preLaunchTask": "tsc: build - tsconfig.json",
      "outFiles": ["${workspaceFolder}/dist/**/*.js"],
      "sourceMaps": true,
      "console": "integratedTerminal",
      "internalConsoleOptions": "neverOpen"
    }
  ]
}
```

**关键配置项**:
- `type`: "node" (Node.js 调试器)
- `request`: "launch" (启动模式) 或 "attach" (附加模式)
- `program`: 入口文件路径
- `preLaunchTask`: 构建任务 (编译 TypeScript)
- `outFiles`: 编译后的 JS 文件路径
- `sourceMaps`: 启用 source map

---

### 使用 tsx 运行器

```json
{
  "type": "node",
  "request": "launch",
  "name": "Debug with tsx",
  "runtimeArgs": ["--import", "tsx"],
  "args": ["${file}"],
  "console": "integratedTerminal",
  "internalConsoleOptions": "neverOpen"
}
```

**优势**:
- 无需预编译
- 直接调试 .ts 文件
- 更快的开发体验

---

### 使用 ts-node

```json
{
  "type": "node",
  "request": "launch",
  "name": "Debug with ts-node",
  "runtimeArgs": ["-r", "ts-node/register"],
  "args": ["${file}"],
  "console": "integratedTerminal"
}
```

---

### 调试测试

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

### 附加到运行中的进程

```json
{
  "type": "node",
  "request": "attach",
  "name": "Attach to Process",
  "port": 9229,
  "restart": true,
  "sourceMaps": true,
  "outFiles": ["${workspaceFolder}/dist/**/*.js"]
}
```

**使用方式**:
```bash
# 启动 Node.js 进程并开启调试
node --inspect dist/index.js

# 或使用 tsx
node --inspect --import tsx src/index.ts
```

---

## tasks.json 配置

### TypeScript 构建任务

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "type": "typescript",
      "tsconfig": "tsconfig.json",
      "problemMatcher": ["$tsc"],
      "group": {
        "kind": "build",
        "isDefault": true
      },
      "label": "tsc: build - tsconfig.json"
    }
  ]
}
```

---

## tsconfig.json 配置

### 调试必需配置

```json
{
  "compilerOptions": {
    "sourceMap": true,
    "inlineSourceMap": false,
    "inlineSources": false,
    "outDir": "dist",
    "rootDir": "src"
  }
}
```

**关键选项**:
- `sourceMap: true` - 生成 .map 文件
- `inlineSourceMap: false` - 不内联 source map
- `outDir` - 输出目录
- `rootDir` - 源码目录

---

## OpenClaw 项目配置

### 推荐配置

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "node",
      "request": "launch",
      "name": "Debug OpenClaw",
      "runtimeArgs": ["--import", "tsx"],
      "args": ["src/index.ts"],
      "console": "integratedTerminal",
      "internalConsoleOptions": "neverOpen",
      "env": {
        "NODE_ENV": "development"
      }
    },
    {
      "type": "node",
      "request": "launch",
      "name": "Debug Gateway",
      "runtimeArgs": ["--import", "tsx"],
      "args": ["src/entry.ts", "gateway"],
      "console": "integratedTerminal",
      "env": {
        "OPENCLAW_SKIP_CHANNELS": "1",
        "CLAWDBOT_SKIP_CHANNELS": "1"
      }
    },
    {
      "type": "node",
      "request": "launch",
      "name": "Debug Tests",
      "runtimeExecutable": "npm",
      "runtimeArgs": ["run", "test:watch"],
      "console": "integratedTerminal"
    }
  ]
}
```

---

## 调试技巧

### 1. 断点类型

**普通断点**:
- 点击行号左侧设置
- F9 快捷键

**条件断点**:
- 右键断点 → 编辑断点
- 添加条件表达式

**日志断点**:
- 右键断点 → 编辑断点
- 选择"日志消息"

### 2. 调试控制台

**执行表达式**:
```javascript
// 在调试控制台中
> user.name
> JSON.stringify(data, null, 2)
> process.env.NODE_ENV
```

### 3. 监视表达式

添加监视:
- 调试侧边栏 → 监视
- 添加表达式

### 4. 调用堆栈

查看调用链:
- 调试侧边栏 → 调用堆栈
- 点击跳转到对应代码

---

## 常见问题

### Q1: 断点未命中?

**检查 source map**:
```json
{
  "compilerOptions": {
    "sourceMap": true
  }
}
```

**检查 outFiles**:
```json
{
  "outFiles": ["${workspaceFolder}/dist/**/*.js"]
}
```

### Q2: 无法调试 TypeScript 文件?

**使用 tsx**:
```json
{
  "runtimeArgs": ["--import", "tsx"],
  "args": ["${file}"]
}
```

### Q3: 调试时找不到模块?

**设置工作目录**:
```json
{
  "cwd": "${workspaceFolder}"
}
```

### Q4: 环境变量未生效?

**在 launch.json 中设置**:
```json
{
  "env": {
    "NODE_ENV": "development",
    "DEBUG": "*"
  }
}
```

---

## 快捷键

| 操作 | Windows/Linux | macOS |
|------|---------------|-------|
| 开始调试 | F5 | F5 |
| 停止调试 | Shift+F5 | Shift+F5 |
| 重启调试 | Ctrl+Shift+F5 | Cmd+Shift+F5 |
| 继续 | F5 | F5 |
| 单步跳过 | F10 | F10 |
| 单步进入 | F11 | F11 |
| 单步跳出 | Shift+F11 | Shift+F11 |
| 切换断点 | F9 | F9 |

---

**文档版本**: 基于 2026-02-24 搜索结果
**来源**: VS Code 官方文档 + 社区最佳实践
