# Phase 3: 定制化开发

> 第 6-8 周 | 6个知识点 | 目标：掌握 Prompt Templates、Skills、Extensions 的开发

---

## 知识点清单

| 编号 | 知识点名称 | 核心概念 | 应用场景 | 学习时长 |
|------|-----------|---------|---------|---------|
| 13 | Prompt Templates 模板系统 | Markdown 模板、变量插值、模板加载 | 提示词复用、团队协作 | 1.5小时 |
| 14 | Skills 技能包开发 | Agent Skills 标准、SKILL.md 格式、技能调用 | 功能封装、技能复用 | 2小时 |
| 15 | Extensions 扩展开发基础 | TypeScript 扩展、ExtensionAPI、扩展加载 | 功能扩展、插件开发 | 2.5小时 |
| 16 | 自定义工具注册 | registerTool API、工具 schema、执行函数 | 工具开发、功能增强 | 2小时 |
| 17 | UI 组件定制 | pi-tui 差分渲染、自定义编辑器、Widget 系统 | 界面定制、交互优化 | 2小时 |
| 18 | 事件系统与钩子 | 事件监听、生命周期钩子、自定义事件 | 流程控制、行为定制 | 1.5小时 |

**总学习时长：** 约 11.5 小时

---

## 学习顺序建议

### 定制化开发路径（核心）
按编号顺序学习，掌握 pi-mono 的定制化能力：
1. **13 - Prompt Templates 模板系统**（最简单的定制）
2. **14 - Skills 技能包开发**（功能封装）
3. **15 - Extensions 扩展开发基础**（核心扩展机制）
4. **16 - 自定义工具注册**（工具开发）
5. **17 - UI 组件定制**（界面定制）
6. **18 - 事件系统与钩子**（流程控制）

### 快速定制路径（3小时）
适合想快速定制 pi 的开发者：
1. **13 - Prompt Templates 模板系统**（最简单）
2. **14 - Skills 技能包开发**（最实用）
3. **16 - 自定义工具注册**（最强大）

### 实战项目驱动路径
边学边用，推荐顺序：
1. 13 - 创建项目专用的 Prompt Templates
2. 14 - 封装常用操作为 Skills
3. 16 - 开发项目特定的工具
4. 15 - 构建完整的 Extension
5. 18 - 添加自定义钩子
6. 17 - 定制 UI 组件（可选）

---

## 与 AI Agent 开发的关系

### 定制化能力
- **13**：复用提示词，提升效率
- **14**：封装功能，构建技能库
- **15**：扩展 pi，实现自定义功能
- **16**：开发工具，增强 Agent 能力

### 为后续学习打基础
- **Phase 4**：基于 Extensions 实现高级功能
- **Phase 5**：基于定制化能力构建实战项目
- **Phase 6**：为 pi-mono 贡献 Extensions 和 Skills

### 构建者视角
- 理解 pi-mono 的扩展机制
- 掌握定制化开发的最佳实践
- 为构建自定义 Agent 打基础

---

## 学习检查清单

完成本 Phase 后，你应该能够：

### Prompt Templates
- [ ] 创建 Markdown 格式的 Prompt Template
- [ ] 使用变量插值（{{variable}}）
- [ ] 配置模板加载路径
- [ ] 在项目中使用自定义模板
- [ ] 理解模板的优先级和覆盖规则

### Skills 技能包
- [ ] 理解 Agent Skills 标准
- [ ] 创建 SKILL.md 文件
- [ ] 定义技能的输入和输出
- [ ] 使用 /skill:name 调用技能
- [ ] 发布和分享技能包

### Extensions 扩展
- [ ] 创建 TypeScript 扩展模块
- [ ] 使用 ExtensionAPI 接口
- [ ] 理解扩展加载机制
- [ ] 使用 /reload 热重载扩展
- [ ] 调试扩展代码

### 自定义工具
- [ ] 使用 registerTool API 注册工具
- [ ] 定义工具 schema（Zod）
- [ ] 实现工具执行函数
- [ ] 处理工具错误
- [ ] 测试自定义工具

### UI 组件
- [ ] 理解 pi-tui 差分渲染机制
- [ ] 创建自定义编辑器
- [ ] 使用 Widget 系统
- [ ] 定制界面布局
- [ ] 优化渲染性能

### 事件系统
- [ ] 使用 on/off 监听事件
- [ ] 理解生命周期钩子
- [ ] 创建自定义事件
- [ ] 实现事件驱动的流程
- [ ] 调试事件流

### 实战能力
- [ ] 为项目创建专用的 Prompt Templates
- [ ] 开发至少一个 Skill
- [ ] 开发至少一个 Extension
- [ ] 注册至少一个自定义工具
- [ ] 定制 UI 组件（可选）
- [ ] 使用事件系统控制流程

---

## 双重类比速查表

| Pi-mono 概念 | TypeScript/Node.js 类比 | 日常生活类比 |
|-------------|------------------------|--------------|
| **Prompt Templates** |
| Markdown 模板 | Handlebars/EJS 模板 | 邮件模板 |
| 变量插值 | 模板字符串 `${var}` | 填空题 |
| 模板加载 | require/import | 读取文件 |
| **Skills** |
| Skill | npm 包 | 技能卡片 |
| SKILL.md | package.json | 说明书 |
| 技能调用 | 函数调用 | 使用技能 |
| **Extensions** |
| Extension | Express 插件 | 浏览器扩展 |
| ExtensionAPI | Plugin API | 插件接口 |
| 扩展加载 | 动态 import | 安装插件 |
| /reload | 热重载 | 刷新页面 |
| **自定义工具** |
| registerTool | 注册路由 | 注册服务 |
| Tool schema | Zod schema | 函数签名 |
| 执行函数 | 路由处理器 | 执行操作 |
| **UI 组件** |
| pi-tui | Ink (React for CLI) | 终端界面 |
| 差分渲染 | Virtual DOM | 只更新变化 |
| Widget | React Component | UI 组件 |
| **事件系统** |
| on/off | EventEmitter | 事件监听 |
| 生命周期钩子 | React Hooks | 生命周期 |
| 自定义事件 | CustomEvent | 自定义信号 |

---

## 定制化开发最佳实践

### Prompt Templates
- **命名规范**：使用描述性名称（如 `code-review.md`）
- **变量命名**：使用 camelCase（如 `{{fileName}}`）
- **模板组织**：按功能分类（如 `templates/code/`, `templates/docs/`）
- **版本控制**：将模板提交到 Git

### Skills
- **单一职责**：每个 Skill 只做一件事
- **清晰文档**：SKILL.md 要详细说明用法
- **错误处理**：提供清晰的错误信息
- **测试**：编写测试用例验证 Skill

### Extensions
- **TypeScript 优先**：使用 TypeScript 开发
- **类型安全**：充分利用 ExtensionAPI 的类型定义
- **错误处理**：捕获并处理所有错误
- **性能优化**：避免阻塞主线程

### 自定义工具
- **Schema 定义**：使用 Zod 定义清晰的 schema
- **参数验证**：验证所有输入参数
- **异步处理**：使用 async/await 处理异步操作
- **错误信息**：提供有用的错误信息

---

## 常见问题

### Q1: Prompt Templates 和 Skills 有什么区别？
**A:** Prompt Templates 是静态的提示词模板，Skills 是可执行的功能封装。Templates 用于复用提示词，Skills 用于封装复杂逻辑。

### Q2: Extensions 需要重启 pi 吗？
**A:** 不需要。使用 `/reload` 命令可以热重载 Extensions，无需重启。

### Q3: 自定义工具和内置工具有什么区别？
**A:** 自定义工具通过 registerTool API 注册，内置工具（read、write、edit、bash）是 pi 核心功能。自定义工具可以扩展 Agent 的能力。

### Q4: 如何调试 Extension？
**A:** 使用 VS Code Debugger 附加到 pi 进程，或在 Extension 代码中添加 console.log。

### Q5: Skills 可以调用其他 Skills 吗？
**A:** 可以。在 Skill 的实现中可以调用其他 Skills，但要注意避免循环依赖。

### Q6: UI 组件定制需要重新构建 pi 吗？
**A:** 如果使用 Extension 方式定制，不需要重新构建。如果修改 pi-tui 源码，需要重新构建。

---

## 下一步学习

完成 Phase 3 后，你已经掌握了 pi-mono 的定制化开发。接下来：

- **Phase 4: 高级扩展** - 实现自定义 Provider、MCP 集成、Sub-Agents
- **Phase 5: 实战项目** - 基于 pi-mono 构建实际应用
- **Phase 6: 贡献与优化** - 为 pi-mono 贡献代码

---

## 参考资源

- **Extension 示例**: https://github.com/badlogic/pi-mono/tree/main/examples/extensions
- **Skills 示例**: https://github.com/badlogic/pi-mono/tree/main/examples/skills
- **ExtensionAPI 文档**: https://github.com/badlogic/pi-mono/tree/main/packages/pi-agent-core
- **Discord 社区**: https://discord.gg/pi-mono

---

**版本：** v1.0
**最后更新：** 2026-02-17
**维护者：** Claude Code
