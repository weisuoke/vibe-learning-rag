# Pi-mono README - Plan Mode 相关内容

**来源**: https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/README.md
**获取时间**: 2026-02-21

## 核心哲学

Pi is a minimal terminal coding harness. Adapt pi to your workflows, not the other way around, without having to fork and modify pi internals.

**Pi ships with powerful defaults but skips features like sub agents and plan mode.** Instead, you can ask pi to build what you want or install a third party pi package that matches your workflow.

## Extensions 可以实现的功能

Extensions 是 TypeScript 模块，可以扩展 pi 的功能：

```typescript
export default function (pi: ExtensionAPI) {
  pi.registerTool({ name: "deploy", ... });
  pi.registerCommand("stats", { ... });
  pi.on("tool_call", async (event, ctx) => { ... });
}
```

**Extensions 可以实现的功能包括：**
- Custom tools (or replace built-in tools entirely)
- **Sub-agents and plan mode** ⭐
- Custom compaction and summarization
- Permission gates and path protection
- Custom editors and UI components
- Status lines, headers, footers
- Overlays, modals, notifications
- Themes (CSS-in-JS)
- Hot-reloading during development

## Philosophy

Pi aims to be:

- **Minimal** — Small core, few built-in features
- **Composable** — Everything is an extension or skill
- **Hackable** — Full TypeScript API, hot-reloading
- **Workflow-first** — Adapt to your flow, not the other way around
- **Transparent** — See exactly what the model sees and does

**No black-box orchestration. No forced sub-agents or planners. Build what you need — or install someone else's package.**

## 文件位置

Extensions 放置位置：
- `~/.pi/agent/extensions/`
- `.pi/extensions/`
- Pi packages (npm 或 git)

## 参考资源

- Extensions 文档: [docs/extensions.md](https://github.com/badlogic/pi-mono/blob/main/docs/extensions.md)
- Pi Packages: [docs/pi-packages.md](https://github.com/badlogic/pi-mono/blob/main/docs/pi-packages.md)
