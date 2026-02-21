# 核心概念 03: Widget 系统

> **学习目标**: 深入理解 Widget 系统和 Overlay 定位机制
> **阅读时间**: 75 分钟
> **难度级别**: ⭐⭐⭐⭐ (进阶)

---

## 概述

**Widget 系统**是 pi-tui 中构建可复用 UI 组件的核心架构。通过组件化设计和灵活的定位系统，开发者可以创建丰富的交互组件，如 Tooltip、Context Menu、Progress Bar 等。

### 核心价值

**模块化**: 封装复用，减少代码重复
- 一次实现，到处使用
- 清晰的组件边界
- 易于维护和测试

**灵活性**: 多种定位策略，适应不同场景
- Absolute: 绝对坐标定位
- Relative: 相对父容器定位
- Anchor: 锚点定位（类似 CSS）

**可组合性**: 组件可以嵌套组合
- Container 包含多个 Component
- 树形结构，层次清晰
- 支持复杂的 UI 布局

### 本文内容

1. **Widget 架构**: Component、Container、Focusable 接口
2. **Overlay 定位系统**: 三种定位策略和边界处理
3. **实际应用**: Tooltip、Context Menu、Progress Bar 实现

---

## Widget 架构

### 组件层次

**基础层次**:
```
Component (基础组件)
├─ render(ctx): string
└─ getSize?(): { width, height }

Container (容器组件)
├─ extends Component
├─ children: Component[]
└─ layout(): void

Focusable (可聚焦组件)
├─ extends Component
├─ focus(): void
├─ blur(): void
└─ isFocused(): boolean
```

### Component 接口

**定义**:
```typescript
interface Component {
  render(ctx: RenderContext): string;
  getSize?(): { width: number; height: number };
}

interface RenderContext {
  width: number;
  height: number;
  focused: boolean;
}
```

**职责**:
- `render()`: 将组件渲染为字符串
- `getSize()`: 返回组件的尺寸（可选）

**示例**:
```typescript
class TextComponent implements Component {
  constructor(private content: string) {}

  render(ctx: RenderContext): string {
    return this.content;
  }

  getSize(): { width: number; height: number } {
    return {
      width: this.content.length,
      height: 1
    };
  }
}
```

### Container 接口

**定义**:
```typescript
interface Container extends Component {
  children: Component[];
  layout(): void;
}
```

**职责**:
- 管理子组件
- 计算子组件的位置
- 渲染所有子组件

**示例**:
```typescript
class BoxContainer implements Container {
  children: Component[] = [];

  constructor(options: { children: Component[] }) {
    this.children = options.children;
  }

  layout(): void {
    // 计算每个子组件的位置
    let y = 0;
    for (const child of this.children) {
      const size = child.getSize?.() || { width: 0, height: 1 };
      // 设置子组件位置 (假设有 setPosition 方法)
      y += size.height;
    }
  }

  render(ctx: RenderContext): string {
    this.layout();

    let output = '';
    for (const child of this.children) {
      output += child.render(ctx) + '\n';
    }
    return output;
  }

  getSize(): { width: number; height: number } {
    let totalHeight = 0;
    let maxWidth = 0;

    for (const child of this.children) {
      const size = child.getSize?.() || { width: 0, height: 1 };
      totalHeight += size.height;
      maxWidth = Math.max(maxWidth, size.width);
    }

    return { width: maxWidth, height: totalHeight };
  }
}
```

### Focusable 接口

**定义**:
```typescript
interface Focusable extends Component {
  focus(): void;
  blur(): void;
  isFocused(): boolean;
}
```

**职责**:
- 管理焦点状态
- 响应焦点变化
- 渲染焦点指示器

**示例**:
```typescript
class ButtonComponent implements Focusable {
  private focused: boolean = false;

  constructor(private label: string) {}

  focus(): void {
    this.focused = true;
  }

  blur(): void {
    this.focused = false;
  }

  isFocused(): boolean {
    return this.focused;
  }

  render(ctx: RenderContext): string {
    const prefix = this.focused ? '> ' : '  ';
    const style = this.focused ? '\x1b[1m' : '';  // 粗体
    const reset = '\x1b[0m';

    return `${prefix}${style}${this.label}${reset}`;
  }

  getSize(): { width: number; height: number } {
    return {
      width: this.label.length + 2,  // 包含前缀
      height: 1
    };
  }
}
```

### 组合模式

**核心思想**: 组件可以包含其他组件，形成树形结构

**示例**:
```typescript
const ui = new BoxContainer({
  children: [
    new TextComponent('Title'),
    new BoxContainer({
      children: [
        new ButtonComponent('OK'),
        new ButtonComponent('Cancel')
      ]
    }),
    new TextComponent('Footer')
  ]
});
```

**树形结构**:
```
BoxContainer
├─ TextComponent ('Title')
├─ BoxContainer
│  ├─ ButtonComponent ('OK')
│  └─ ButtonComponent ('Cancel')
└─ TextComponent ('Footer')
```

---

## Overlay 定位系统

### 定位策略

**三种策略**:

**1. Absolute (绝对定位)**
```typescript
interface AbsolutePosition {
  type: 'absolute';
  x: number;  // 绝对 X 坐标
  y: number;  // 绝对 Y 坐标
}
```

**用途**: 固定位置的元素
**示例**: 状态栏、标题栏

**2. Relative (相对定位)**
```typescript
interface RelativePosition {
  type: 'relative';
  x: number;  // 相对父容器的 X 偏移
  y: number;  // 相对父容器的 Y 偏移
}
```

**用途**: 相对父容器定位的元素
**示例**: 布局中的元素

**3. Anchor (锚点定位)**
```typescript
interface AnchorPosition {
  type: 'anchor';
  anchor: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right' | 'center';
  offset?: { x: number; y: number };
}
```

**用途**: 锚定到特定位置的元素
**示例**: Tooltip、Notification

**完整类型**:
```typescript
type OverlayPosition = AbsolutePosition | RelativePosition | AnchorPosition;
```

### OverlayContainer 实现

**核心类**:
```typescript
class OverlayContainer implements Container {
  children: Component[] = [];
  private position: OverlayPosition;

  constructor(options: {
    position: OverlayPosition;
    child: Component;
  }) {
    this.position = options.position;
    this.children = [options.child];
  }

  layout(): void {
    const { x, y } = this.calculatePosition();
    // 设置子组件位置
    this.setChildPosition(this.children[0], x, y);
  }

  private calculatePosition(): { x: number; y: number } {
    switch (this.position.type) {
      case 'absolute':
        return this.calculateAbsolutePosition(this.position);

      case 'relative':
        return this.calculateRelativePosition(this.position);

      case 'anchor':
        return this.calculateAnchorPosition(this.position);
    }
  }

  private calculateAbsolutePosition(pos: AbsolutePosition): { x: number; y: number } {
    return { x: pos.x, y: pos.y };
  }

  private calculateRelativePosition(pos: RelativePosition): { x: number; y: number } {
    const parent = this.getParent();
    return {
      x: parent.x + pos.x,
      y: parent.y + pos.y
    };
  }

  private calculateAnchorPosition(pos: AnchorPosition): { x: number; y: number } {
    const terminal = this.getTerminal();
    const widget = this.children[0].getSize?.() || { width: 0, height: 0 };

    let x = 0, y = 0;

    switch (pos.anchor) {
      case 'top-left':
        x = 0;
        y = 0;
        break;

      case 'top-right':
        x = terminal.width - widget.width;
        y = 0;
        break;

      case 'bottom-left':
        x = 0;
        y = terminal.height - widget.height;
        break;

      case 'bottom-right':
        x = terminal.width - widget.width;
        y = terminal.height - widget.height;
        break;

      case 'center':
        x = Math.floor((terminal.width - widget.width) / 2);
        y = Math.floor((terminal.height - widget.height) / 2);
        break;
    }

    // 应用偏移
    if (pos.offset) {
      x += pos.offset.x;
      y += pos.offset.y;
    }

    return { x, y };
  }

  render(ctx: RenderContext): string {
    this.layout();
    return this.children[0].render(ctx);
  }
}
```

### 边界处理

**问题**: Widget 可能超出终端边界

**解决方案 1: Clamp (限制)**
```typescript
function clampPosition(
  widget: { width: number; height: number },
  terminal: { width: number; height: number },
  position: { x: number; y: number }
): { x: number; y: number } {
  return {
    x: Math.max(0, Math.min(position.x, terminal.width - widget.width)),
    y: Math.max(0, Math.min(position.y, terminal.height - widget.height))
  };
}
```

**解决方案 2: Clip (裁剪)**
```typescript
function clipWidget(
  widget: Component,
  terminal: { width: number; height: number },
  position: { x: number; y: number }
): string {
  const content = widget.render({ width: terminal.width, height: terminal.height, focused: false });
  const lines = content.split('\n');

  // 裁剪超出部分
  const visibleLines = lines.slice(
    Math.max(0, -position.y),
    Math.max(0, terminal.height - position.y)
  );

  return visibleLines.map(line => {
    if (position.x < 0) {
      line = line.slice(-position.x);
    }
    if (position.x + line.length > terminal.width) {
      line = line.slice(0, terminal.width - position.x);
    }
    return line;
  }).join('\n');
}
```

**解决方案 3: Reposition (智能重定位)**
```typescript
function smartPosition(
  widget: { width: number; height: number },
  anchor: { x: number; y: number },
  terminal: { width: number; height: number }
): { x: number; y: number } {
  // 默认在锚点下方
  let x = anchor.x;
  let y = anchor.y + 1;

  // 如果超出右边界，左对齐
  if (x + widget.width > terminal.width) {
    x = terminal.width - widget.width;
  }

  // 如果超出下边界，显示在锚点上方
  if (y + widget.height > terminal.height) {
    y = anchor.y - widget.height;
  }

  // 如果还是超出，使用 clamp
  return clampPosition(widget, terminal, { x, y });
}
```

### Z-Index 管理

**问题**: 终端没有真正的 z-index，后渲染的覆盖先渲染的

**解决方案**: 维护 overlay 列表，按 z-index 排序

**实现**:
```typescript
interface OverlayItem {
  widget: Component;
  position: OverlayPosition;
  zIndex: number;
}

class OverlayManager {
  private overlays: OverlayItem[] = [];

  add(widget: Component, position: OverlayPosition, zIndex: number = 0): void {
    this.overlays.push({ widget, position, zIndex });
    // 按 zIndex 排序（升序）
    this.overlays.sort((a, b) => a.zIndex - b.zIndex);
  }

  remove(widget: Component): void {
    this.overlays = this.overlays.filter(o => o.widget !== widget);
  }

  render(ctx: RenderContext): string {
    let output = '';

    // 按顺序渲染，后渲染的覆盖先渲染的
    for (const overlay of this.overlays) {
      const container = new OverlayContainer({
        position: overlay.position,
        child: overlay.widget
      });
      output += container.render(ctx);
    }

    return output;
  }
}
```

**使用示例**:
```typescript
const manager = new OverlayManager();

// 添加背景 (z-index: 0)
manager.add(backgroundWidget, { type: 'absolute', x: 0, y: 0 }, 0);

// 添加对话框 (z-index: 10)
manager.add(dialogWidget, { type: 'anchor', anchor: 'center' }, 10);

// 添加 Tooltip (z-index: 20)
manager.add(tooltipWidget, { type: 'absolute', x: 10, y: 5 }, 20);

// 渲染顺序: background → dialog → tooltip
```

---

## 实际应用

### 应用 1: Tooltip Widget

**需求**: 鼠标悬停时显示提示信息

**实现**:
```typescript
class TooltipWidget implements Component {
  constructor(
    private text: string,
    private anchor: { x: number; y: number }
  ) {}

  render(ctx: RenderContext): string {
    const border = '─'.repeat(this.text.length + 2);
    return `┌${border}┐\n│ ${this.text} │\n└${border}┘`;
  }

  getSize(): { width: number; height: number } {
    return {
      width: this.text.length + 4,  // 包含边框
      height: 3
    };
  }
}

// 使用
function showTooltip(text: string, anchor: { x: number; y: number }): void {
  const tooltip = new TooltipWidget(text, anchor);
  const position = smartPosition(
    tooltip.getSize()!,
    anchor,
    terminal.getSize()
  );

  const overlay = new OverlayContainer({
    position: { type: 'absolute', x: position.x, y: position.y },
    child: tooltip
  });

  overlayManager.add(overlay, { type: 'absolute', ...position }, 20);
}
```

### 应用 2: Context Menu Widget

**需求**: 右键显示菜单

**实现**:
```typescript
interface MenuItem {
  label: string;
  action: () => void;
}

class ContextMenuWidget implements Component, Focusable {
  private selectedIndex: number = 0;
  private focused: boolean = false;

  constructor(private items: MenuItem[]) {}

  focus(): void {
    this.focused = true;
  }

  blur(): void {
    this.focused = false;
  }

  isFocused(): boolean {
    return this.focused;
  }

  handleKey(key: KeyEvent): boolean {
    if (!this.focused) return false;

    switch (key.name) {
      case 'up':
        this.selectedIndex = Math.max(0, this.selectedIndex - 1);
        return true;

      case 'down':
        this.selectedIndex = Math.min(this.items.length - 1, this.selectedIndex + 1);
        return true;

      case 'return':
        this.items[this.selectedIndex].action();
        return true;

      case 'escape':
        // 关闭菜单
        return false;
    }

    return false;
  }

  render(ctx: RenderContext): string {
    const maxWidth = Math.max(...this.items.map(item => item.label.length));
    const border = '─'.repeat(maxWidth + 2);

    let output = `┌${border}┐\n`;

    for (let i = 0; i < this.items.length; i++) {
      const item = this.items[i];
      const selected = i === this.selectedIndex;
      const prefix = selected ? '> ' : '  ';
      const style = selected ? '\x1b[1m' : '';
      const reset = '\x1b[0m';

      output += `│${prefix}${style}${item.label.padEnd(maxWidth)}${reset}│\n`;
    }

    output += `└${border}┘`;
    return output;
  }

  getSize(): { width: number; height: number } {
    const maxWidth = Math.max(...this.items.map(item => item.label.length));
    return {
      width: maxWidth + 6,  // 包含边框和前缀
      height: this.items.length + 2  // 包含边框
    };
  }
}

// 使用
function showContextMenu(items: MenuItem[], position: { x: number; y: number }): void {
  const menu = new ContextMenuWidget(items);
  const smartPos = smartPosition(
    menu.getSize()!,
    position,
    terminal.getSize()
  );

  const overlay = new OverlayContainer({
    position: { type: 'absolute', x: smartPos.x, y: smartPos.y },
    child: menu
  });

  overlayManager.add(overlay, { type: 'absolute', ...smartPos }, 15);
  menu.focus();
}
```

### 应用 3: Progress Bar Widget

**需求**: 显示进度指示器

**实现**:
```typescript
class ProgressBarWidget implements Component {
  constructor(
    private progress: number,  // 0-100
    private width: number = 20
  ) {}

  setProgress(progress: number): void {
    this.progress = Math.max(0, Math.min(100, progress));
  }

  render(ctx: RenderContext): string {
    const filledWidth = Math.floor(this.width * this.progress / 100);
    const emptyWidth = this.width - filledWidth;

    const filled = '█'.repeat(filledWidth);
    const empty = '░'.repeat(emptyWidth);
    const percentage = `${this.progress}%`.padStart(4);

    return `[${filled}${empty}] ${percentage}`;
  }

  getSize(): { width: number; height: number } {
    return {
      width: this.width + 8,  // 包含括号和百分比
      height: 1
    };
  }
}

// 使用
const progressBar = new ProgressBarWidget(0, 30);

const overlay = new OverlayContainer({
  position: { type: 'anchor', anchor: 'bottom-left', offset: { x: 1, y: -1 } },
  child: progressBar
});

overlayManager.add(overlay, { type: 'anchor', anchor: 'bottom-left' }, 10);

// 更新进度
setInterval(() => {
  progressBar.setProgress(progressBar.progress + 10);
  if (progressBar.progress >= 100) {
    overlayManager.remove(overlay);
  }
}, 1000);
```

### 应用 4: Notification Widget

**需求**: 显示通知消息，自动消失

**实现**:
```typescript
class NotificationWidget implements Component {
  constructor(
    private message: string,
    private type: 'info' | 'success' | 'warning' | 'error' = 'info'
  ) {}

  render(ctx: RenderContext): string {
    const colors = {
      info: '\x1b[34m',     // 蓝色
      success: '\x1b[32m',  // 绿色
      warning: '\x1b[33m',  // 黄色
      error: '\x1b[31m'     // 红色
    };

    const icons = {
      info: 'ℹ',
      success: '✓',
      warning: '⚠',
      error: '✗'
    };

    const color = colors[this.type];
    const icon = icons[this.type];
    const reset = '\x1b[0m';

    const border = '─'.repeat(this.message.length + 4);
    return `${color}┌${border}┐\n│ ${icon} ${this.message} │\n└${border}┘${reset}`;
  }

  getSize(): { width: number; height: number } {
    return {
      width: this.message.length + 6,
      height: 3
    };
  }
}

// 使用
function showNotification(
  message: string,
  type: 'info' | 'success' | 'warning' | 'error' = 'info',
  duration: number = 3000
): void {
  const notification = new NotificationWidget(message, type);

  const overlay = new OverlayContainer({
    position: { type: 'anchor', anchor: 'top-right', offset: { x: -1, y: 1 } },
    child: notification
  });

  overlayManager.add(overlay, { type: 'anchor', anchor: 'top-right' }, 25);

  // 自动移除
  setTimeout(() => {
    overlayManager.remove(overlay);
  }, duration);
}

// 示例
showNotification('File saved successfully', 'success');
showNotification('Connection lost', 'error');
```

---

## 总结

### 核心要点

**Widget 架构**:
- **Component**: 基础组件接口，render() + getSize()
- **Container**: 容器组件，管理子组件
- **Focusable**: 可聚焦组件，管理焦点状态

**Overlay 定位**:
- **Absolute**: 绝对坐标定位
- **Relative**: 相对父容器定位
- **Anchor**: 锚点定位（top-left, center, etc.）

**边界处理**:
- **Clamp**: 限制在边界内
- **Clip**: 裁剪超出部分
- **Reposition**: 智能重新定位

**Z-Index 管理**:
- 维护 overlay 列表
- 按 z-index 排序
- 按顺序渲染

### 关键洞察

1. **组件化是关键**: 封装、复用、组合
2. **定位需要灵活**: 三种策略覆盖不同场景
3. **边界处理很重要**: 终端尺寸固定，必须处理超出情况
4. **Z-Index 需要管理**: 终端没有真正的层叠，需要手动管理

### 实际应用

**Tooltip**: 智能定位，自动翻转
**Context Menu**: 键盘导航，焦点管理
**Progress Bar**: 动态更新，锚点定位
**Notification**: 自动消失，层叠显示

### 下一步

- **深入学习**: 阅读 **04_最小可用.md**
- **实战练习**: 阅读 **07_实战代码_03_Widget组件创建.md**
- **源码参考**: `sourcecode/pi-mono/packages/tui/src/components/`

---

**版本**: v1.0
**最后更新**: 2026-02-21
**维护者**: Claude Code
