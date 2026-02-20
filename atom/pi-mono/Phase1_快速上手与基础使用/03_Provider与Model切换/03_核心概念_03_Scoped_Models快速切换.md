# 核心概念 03：Scoped Models 快速切换

> **掌握 Ctrl+P 快捷键，实现零中断的模型切换**

---

## 什么是 Scoped Models？

**Scoped Models** 是预配置的 3-5 个常用模型，通过键盘快捷键快速循环切换。

**核心特点**：
- `Ctrl+P` 向前循环，`Shift+Ctrl+P` 向后循环
- 零中断切换，不影响工作流
- 一次配置，长期使用
- 按成本排序，默认使用最便宜模型

---

## 配置方法

### 配置文件

**位置**：`.pi/settings.json` 或 `~/.pi/agent/settings.json`

```json
{
  "scopedModels": [
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "claude-opus-4-20250514"
  ]
}
```

### 推荐配置（按成本递增）

```json
{
  "scopedModels": [
    "claude-3-5-haiku-20241022",    // $0.8/MTok - 快速任务
    "claude-3-5-sonnet-20241022",   // $3/MTok - 日常开发
    "claude-opus-4-20250514"        // $15/MTok - 复杂问题
  ]
}
```

---

## 快捷键操作

### 向前循环：Ctrl+P

```
Haiku → Sonnet → Opus → Haiku → ...
```

### 向后循环：Shift+Ctrl+P

```
Haiku → Opus → Sonnet → Haiku → ...
```

### 视觉反馈

```bash
# 按 Ctrl+P
✓ Switched to claude-3-5-sonnet-20241022 (2/3)

# 再按 Ctrl+P
✓ Switched to claude-opus-4-20250514 (3/3)
```

---

## 交互式配置

### 使用 /scoped-models 命令

```bash
> /scoped-models
```

**显示当前配置**：

```
Scoped Models (3):
  1. claude-3-5-haiku-20241022
  2. claude-3-5-sonnet-20241022
  3. claude-opus-4-20250514

Options:
  [a] Add model
  [r] Remove model
  [c] Clear all
  [q] Quit
```

---

## 最佳实践

### 1. 数量控制：3-5 个最优

```json
// ✅ 推荐：3 个模型
{
  "scopedModels": [
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "claude-opus-4-20250514"
  ]
}
```

### 2. 按成本排序

从便宜到昂贵排列，默认使用第一个（最便宜）。

### 3. 场景分组

**前端项目**：

```json
{
  "scopedModels": [
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "gpt-4o"
  ]
}
```

**后端项目**：

```json
{
  "scopedModels": [
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "claude-opus-4-20250514"
  ]
}
```

---

## 工作流示例

```bash
# 1. 启动 Pi（默认使用 Haiku）
pi

# 2. 简单任务用 Haiku
> Format this code

# 3. 遇到复杂问题，按 Ctrl+P 切换到 Sonnet
> Refactor this component

# 4. 极端复杂问题，再按 Ctrl+P 切换到 Opus
> Design the architecture

# 5. 问题解决，按 Shift+Ctrl+P 两次切回 Haiku
```

---

## 双重类比

### TypeScript/Node.js 类比

**Scoped Models** 就像 VS Code 的 "最近打开的文件" (Ctrl+Tab)：

```typescript
const recentFiles = ['index.ts', 'utils.ts', 'types.ts'];
// 按 Ctrl+Tab 快速切换
```

### 日常生活类比

- **电视遥控器的频道收藏**：预设 3-5 个常看频道，一键切换
- **汽车座椅记忆**：保存 3 个常用位置，按钮快速调整

---

## 常见问题

### Q1: 最多配置几个 Scoped Models？

**答**：建议 3-5 个。太多会导致切换困难。

### Q2: 如何查看当前是第几个模型？

**答**：切换时会显示 `(2/3)` 表示第 2 个，共 3 个。

### Q3: 可以在不同项目使用不同配置吗？

**答**：可以。项目级配置 (`.pi/settings.json`) 会覆盖全局配置。

---

## 下一步

- **Provider 切换命令**：阅读 [03_核心概念_04_Provider切换命令.md](./03_核心概念_04_Provider切换命令.md)
- **实战配置**：阅读 [07_实战代码_02_Scoped_Models配置.md](./07_实战代码_02_Scoped_Models配置.md)

---

**记住**：Scoped Models 是效率工具，3-5 个模型 + Ctrl+P 快捷键 = 零中断的模型切换体验。
