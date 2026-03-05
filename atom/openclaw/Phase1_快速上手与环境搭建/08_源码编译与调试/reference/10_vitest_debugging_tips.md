# Vitest 调试技巧

> 来源: Grok Web Search
> 查询时间: 2026-02-24

## 搜索结果

### 1. Vitest 官方调试指南
**URL**: https://vitest.dev/guide/debugging
**描述**: Vitest在VS Code中使用JavaScript Debug Terminal和自定义launch配置调试测试并设置断点。

### 2. Vitest Explorer VS Code扩展
**URL**: https://marketplace.visualstudio.com/items?itemName=vitest.explorer
**描述**: 官方Vitest VS Code扩展，支持测试运行、调试、断点设置和测试视图集成。

### 3. Vitest VS Code扩展 GitHub
**URL**: https://github.com/vitest-dev/vscode
**描述**: 官方扩展源代码仓库，包含断点调试功能、测试视图和2026最新更新。

### 4. Vitest IDE集成文档
**URL**: https://vitest.dev/guide/ide
**描述**: Vitest与VS Code官方集成指南，涵盖调试断点支持和扩展推荐。

### 5. Vitest断点未绑定问题
**URL**: https://stackoverflow.com/questions/71591463/how-to-debug-vitest-in-visual-studio-code-currently-have-unbound-breakpoints
**描述**: VS Code调试Vitest时断点未绑定问题的配置解决方案和社区讨论。

### 6. Vitest断点位置错误Issue
**URL**: https://github.com/vitest-dev/vitest/issues/5380
**描述**: VS Code中Vitest调试Vue组件时断点停止位置错误的已知问题报告。

## 调试技巧总结

### JavaScript Debug Terminal
```bash
# 最简单的调试方式
# 1. 打开 JavaScript Debug Terminal
# 2. 运行测试命令
npm run test
```

### launch.json 配置
```json
{
  "type": "node",
  "request": "launch",
  "name": "Debug Vitest",
  "runtimeExecutable": "npm",
  "runtimeArgs": ["run", "test"],
  "console": "integratedTerminal"
}
```

### 常见问题解决
- 断点未绑定: 检查 source map 配置
- 断点位置错误: 更新 Vitest 到最新版本
- 无法调试: 使用 --no-file-parallelism 标志

**文档版本**: 基于 2026-02-24 搜索结果
