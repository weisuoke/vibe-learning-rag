# Mario's Blog - Why No Plan Mode

**来源**: https://mariozechner.at/posts/2025-11-30-pi-coding-agent
**获取时间**: 2026-02-21
**作者**: Mario Zechner (pi-mono 作者)

## 核心观点

文章标题: "What I learned building an opinionated and minimal coding agent"

### 设计哲学

作者在过去三年使用 LLM 辅助编码，从 ChatGPT 复制粘贴，到 Copilot，到 Cursor，最后到 Claude Code 等新一代 coding agent。

**为什么构建 pi-mono:**
- Claude Code 变得过于复杂，80% 的功能用不上
- 系统提示词和工具在每次发布时都会改变，破坏工作流
- 现有工具难以精确控制上下文
- 想要完全透明的交互检查
- 需要简洁的 session 格式用于后处理
- 想要简单的 API 构建替代 UI

### "No plan mode" 章节

文章中有专门的 "No plan mode" 章节（虽然在提取的内容中被截断），但从整体哲学可以推断：

**核心理念:**
> "My philosophy in all of this was: if I don't need it, it won't be built. And I don't need a lot of things."

**推荐方案:**
- 将计划写入文件以获得完整的可观察性
- 通过 Extensions 实现自定义功能
- 避免黑盒编排

### 构建的组件

为了实现 pi-mono，作者构建了：
- **pi-ai**: 统一的 LLM API，支持多提供商
- **pi-agent-core**: Agent 循环，处理工具执行和验证
- **pi-tui**: 最小化终端 UI 框架
- **pi-coding-agent**: CLI，整合 session 管理、自定义工具、主题

## 关键引用

> "No black-box orchestration. No forced sub-agents or planners. Build what you need — or install someone else's package."

## 实现建议

基于作者的哲学，Plan Mode 应该：
1. **文件优先**: 将计划写入文件（如 `plan.md`）
2. **可观察**: 所有步骤都应该透明可见
3. **可扩展**: 通过 Extensions 实现，而非内置
4. **用户控制**: 用户决定何时进入/退出计划模式
