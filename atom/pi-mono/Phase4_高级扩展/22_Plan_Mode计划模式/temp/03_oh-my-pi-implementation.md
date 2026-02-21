# oh-my-pi - Plan Mode 实现

**来源**: https://github.com/can1357/oh-my-pi
**获取时间**: 2026-02-21
**类型**: pi-mono 的 fork，添加了 Plan Mode 功能

## 项目概述

oh-my-pi 是 pi-mono 的一个 fork，由 @can1357 维护，添加了多项增强功能。

**Stars**: 976
**Forks**: 73
**主要语言**: TypeScript (85.5%)
**License**: MIT
**最新提交**: 2026-02-20

## 核心特性

oh-my-pi 添加的功能包括：
- **Hash-anchored edits** (基于 hash 的精准代码编辑)
- Optimized tool harness
- LSP 支持
- Python 工具集成
- Browser 工具
- **Subagents (子代理) 机制**
- 终端 AI 编码代理

## Plan Mode 实现

虽然具体实现细节在 README 中未完全展开，但从项目描述可以推断：

### 可能的实现方式

1. **斜杠命令**:
   - `/plan` - 切换到计划模式
   - `/chat` - 纯聊天模式
   - `/edit` - 编辑模式

2. **模式切换**:
   - 计划阶段使用专用模型进行架构和执行序列规划
   - 执行阶段切换到实现模式

3. **与 Extensions 集成**:
   - 作为 Extension 实现
   - 可以注册自定义命令
   - 可以修改 UI 显示

### 安装方式

```bash
# Via Bun
bun install -g @oh-my-pi/pi-coding-agent

# Via mise
mise use -g github:can1357/oh-my-pi
```

## 参考价值

oh-my-pi 证明了：
1. **Plan Mode 可以作为 Extension 实现**
2. **不需要修改 pi-mono 核心代码**
3. **可以通过斜杠命令切换模式**
4. **社区可以构建自己的增强版本**

## 相关 Issue

GitHub Issue #97: "Add plan mode support"
- 讨论在 pi-mono 中添加 plan mode 支持
- 参考 Claude Code 的 plan 到 execution 切换机制
- 核心项目未内置，但社区实现了

## 学习要点

对于想要实现 Plan Mode 的开发者：
1. 研究 oh-my-pi 的源码
2. 理解如何通过 Extension API 注册命令
3. 学习如何管理不同的执行模式
4. 了解如何与 pi 的 session 系统集成
