# 实战代码 03: Widget 组件创建

> **学习目标**: 创建可复用的 Widget 组件
> **阅读时间**: 60 分钟
> **难度级别**: ⭐⭐⭐⭐ (进阶)
> **代码语言**: TypeScript

---

## 场景概述

**目标**: 创建一套可复用的 Widget 组件库

**组件列表**:
- Tooltip: 提示信息
- Progress Bar: 进度指示器
- Context Menu: 右键菜单
- Notification: 通知消息

**技术要点**:
- Component 接口实现
- Overlay 定位系统
- Widget 管理器
- 边界处理

---

## 完整实现

### 1. 基础类型定义

```typescript
// types.ts
export interface Component {
  render(ctx: RenderContext): string;
  getSize?(): { width: number; height: number };
}

export interface RenderContext {
  width: number;
  height: number;
  focused: boolean;
}

export type OverlayPosition =
  | { type: 'absolute'; x: number; y: number }
  | { type: 'relative'; x: number; y: number }
  | { type: 'anchor'; anchor: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right' | 'center'; offset?: { x: number; y: number } };
```

### 2. Widget 基类

```typescript
// base-widget.ts
import { Component, RenderContext, OverlayPosition } from './types';

export abstract class BaseWidget implements Component {
  protected position: OverlayPosition;
  protected visible: boolean = true;

  constructor(position: OverlayPosition) {
    this.position = position;
  }

  abstract render(ctx: RenderContext): string;
  abstract getSize(): { width: number; height: number };

  show(): void {
    this.visible = true;
  }

  hide(): void {
    this.visible = false;
  }

  isVisible(): boolean {
    return this.visible;
  }

  setPosition(position: OverlayPosition): void {
    this.position = position;
  }

  getPosition(): OverlayPosition {
    return this.position;
  }

  protected calculatePosition(terminal: { width: number; height: number }): { x: number; y: number } {
    const size = this.getSize();

    switch (this.position.type) {
      case 'absolute':
        return { x: this.position.x, y: this.position.y };

      case 'relative':
        // 需要父容器信息，这里简化处理
        return { x: this.position.x, y: this.position.y };

      case 'anchor':
        return this.calculateAnchorPosition(this.position.anchor, size, terminal, this.position.offset);
    }
  }

  private calculateAnchorPosition(
    anchor: string,
    size: { width: number; height: number },
    terminal: { width: number; height: number },
    offset?: { x: number; y: number }
  ): { x: number; y: number } {
    let x = 0, y = 0;

    switch (anchor) {
      case 'top-left':
        x = 0;
        y = 0;
        break;
      case 'top-right':
        x = terminal.width - size.width;
        y = 0;
        break;
      case 'bottom-left':
        x = 0;
        y = terminal.height - size.height;
        break;
      case 'bottom-right':
        x = terminal.width - size.width;
        y = terminal.height - size.height;
        break;
      case 'center':
        x = Math.floor((terminal.width - size.width) / 2);
        y = Math.floor((terminal.height - size.height) / 2);
        break;
    }

    if (offset) {
      x += offset.x;
      y += offset.y;
    }

    // Clamp to terminal bounds
    x = Math.max(0, Math.min(x, terminal.width - size.width));
    y = Math.max(0, Math.min(y, terminal.height - size.height));

    return { x, y };
  }
}
```

### 3. Tooltip Widget

```typescript
// tooltip-widget.ts
import { BaseWidget } from './base-widget';
import { RenderContext, OverlayPosition } from './types';

export class TooltipWidget extends BaseWidget {
  private text: string;
  private style: 'simple' | 'bordered' = 'bordered';

  constructor(text: string, position: OverlayPosition, style: 'simple' | 'bordered' = 'bordered') {
    super(position);
    this.text = text;
    this.style = style;
  }

  setText(text: string): void {
    this.text = text;
  }

  render(ctx: RenderContext): string {
    if (!this.visible) return '';

    if (this.style === 'simple') {
      return this.renderSimple();
    } else {
      return this.renderBordered();
    }
  }

  private renderSimple(): string {
    return this.text;
  }

  private renderBordered(): string {
    const border = '─'.repeat(this.text.length + 2);
    return `┌${border}┐\n│ ${this.text} │\n└${border}┘`;
  }

  getSize(): { width: number; height: number } {
    if (this.style === 'simple') {
      return { width: this.text.length, height: 1 };
    } else {
      return { width: this.text.length + 4, height: 3 };
    }
  }
}
```

### 4. Progress Bar Widget

```typescript
// progress-bar-widget.ts
import { BaseWidget } from './base-widget';
import { RenderContext, OverlayPosition } from './types';

export class ProgressBarWidget extends BaseWidget {
  private progress: number = 0;
  private width: number;
  private showPercentage: boolean;
  private style: 'blocks' | 'smooth' = 'blocks';

  constructor(
    width: number,
    position: OverlayPosition,
    options: { showPercentage?: boolean; style?: 'blocks' | 'smooth' } = {}
  ) {
    super(position);
    this.width = width;
    this.showPercentage = options.showPercentage ?? true;
    this.style = options.style ?? 'blocks';
  }

  setProgress(progress: number): void {
    this.progress = Math.max(0, Math.min(100, progress));
  }

  getProgress(): number {
    return this.progress;
  }

  render(ctx: RenderContext): string {
    if (!this.visible) return '';

    const filled = Math.floor(this.width * this.progress / 100);
    const empty = this.width - filled;

    let bar: string;
    if (this.style === 'blocks') {
      bar = '█'.repeat(filled) + '░'.repeat(empty);
    } else {
      bar = '━'.repeat(filled) + '─'.repeat(empty);
    }

    if (this.showPercentage) {
      const percentage = `${this.progress}%`.padStart(4);
      return `[${bar}] ${percentage}`;
    } else {
      return `[${bar}]`;
    }
  }

  getSize(): { width: number; height: number } {
    const width = this.width + 2 + (this.showPercentage ? 5 : 0);
    return { width, height: 1 };
  }
}
```

### 5. Context Menu Widget

```typescript
// context-menu-widget.ts
import { BaseWidget } from './base-widget';
import { RenderContext, OverlayPosition } from './types';

export interface MenuItem {
  label: string;
  action: () => void;
  disabled?: boolean;
}

export class ContextMenuWidget extends BaseWidget {
  private items: MenuItem[];
  private selectedIndex: number = 0;

  constructor(items: MenuItem[], position: OverlayPosition) {
    super(position);
    this.items = items;
  }

  setItems(items: MenuItem[]): void {
    this.items = items;
    this.selectedIndex = Math.min(this.selectedIndex, items.length - 1);
  }

  selectNext(): void {
    this.selectedIndex = Math.min(this.items.length - 1, this.selectedIndex + 1);
  }

  selectPrevious(): void {
    this.selectedIndex = Math.max(0, this.selectedIndex - 1);
  }

  executeSelected(): void {
    const item = this.items[this.selectedIndex];
    if (item && !item.disabled) {
      item.action();
    }
  }

  render(ctx: RenderContext): string {
    if (!this.visible) return '';

    const maxWidth = Math.max(...this.items.map(item => item.label.length));
    const border = '─'.repeat(maxWidth + 2);

    let output = `┌${border}┐\n`;

    for (let i = 0; i < this.items.length; i++) {
      const item = this.items[i];
      const selected = i === this.selectedIndex;
      const prefix = selected ? '> ' : '  ';
      const style = selected ? '\x1b[1m' : item.disabled ? '\x1b[90m' : '';
      const reset = '\x1b[0m';

      output += `│${prefix}${style}${item.label.padEnd(maxWidth)}${reset}│\n`;
    }

    output += `└${border}┘`;
    return output;
  }

  getSize(): { width: number; height: number } {
    const maxWidth = Math.max(...this.items.map(item => item.label.length));
    return {
      width: maxWidth + 6,
      height: this.items.length + 2
    };
  }
}
```

### 6. Notification Widget

```typescript
// notification-widget.ts
import { BaseWidget } from './base-widget';
import { RenderContext, OverlayPosition } from './types';

export type NotificationType = 'info' | 'success' | 'warning' | 'error';

export class NotificationWidget extends BaseWidget {
  private message: string;
  private type: NotificationType;
  private autoHideTimer?: NodeJS.Timeout;

  constructor(message: string, type: NotificationType, position: OverlayPosition) {
    super(position);
    this.message = message;
    this.type = type;
  }

  setMessage(message: string, type?: NotificationType): void {
    this.message = message;
    if (type) this.type = type;
  }

  autoHide(duration: number): void {
    if (this.autoHideTimer) {
      clearTimeout(this.autoHideTimer);
    }

    this.autoHideTimer = setTimeout(() => {
      this.hide();
    }, duration);
  }

  render(ctx: RenderContext): string {
    if (!this.visible) return '';

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
```

### 7. Widget Manager

```typescript
// widget-manager.ts
import { BaseWidget } from './base-widget';
import { RenderContext } from './types';

interface ManagedWidget {
  widget: BaseWidget;
  zIndex: number;
}

export class WidgetManager {
  private widgets: ManagedWidget[] = [];
  private terminal: { width: number; height: number };

  constructor(terminal: { width: number; height: number }) {
    this.terminal = terminal;
  }

  add(widget: BaseWidget, zIndex: number = 0): void {
    this.widgets.push({ widget, zIndex });
    this.sortWidgets();
  }

  remove(widget: BaseWidget): void {
    this.widgets = this.widgets.filter(w => w.widget !== widget);
  }

  clear(): void {
    this.widgets = [];
  }

  private sortWidgets(): void {
    this.widgets.sort((a, b) => a.zIndex - b.zIndex);
  }

  render(ctx: RenderContext): string {
    let output = '';

    for (const { widget } of this.widgets) {
      if (!widget.isVisible()) continue;

      const position = widget['calculatePosition'](this.terminal);
      const content = widget.render(ctx);
      const lines = content.split('\n');

      lines.forEach((line, i) => {
        const row = position.y + i;
        const col = position.x;

        if (row >= 0 && row < this.terminal.height) {
          output += `\x1b[${row + 1};${col + 1}H${line}`;
        }
      });
    }

    return output;
  }

  updateTerminalSize(width: number, height: number): void {
    this.terminal = { width, height };
  }
}
```

### 8. 完整示例

```typescript
// example.ts
import { WidgetManager } from './widget-manager';
import { TooltipWidget } from './tooltip-widget';
import { ProgressBarWidget } from './progress-bar-widget';
import { ContextMenuWidget, MenuItem } from './context-menu-widget';
import { NotificationWidget } from './notification-widget';

async function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function demo() {
  const terminal = { width: 80, height: 24 };
  const manager = new WidgetManager(terminal);

  // 清屏
  process.stdout.write('\x1b[2J\x1b[H');
  process.stdout.write('\x1b[?25l');  // 隐藏光标

  // 示例 1: Tooltip
  const tooltip = new TooltipWidget(
    'Press Enter to continue',
    { type: 'anchor', anchor: 'bottom-right', offset: { x: -1, y: -1 } }
  );
  manager.add(tooltip, 10);

  // 示例 2: Progress Bar
  const progressBar = new ProgressBarWidget(
    30,
    { type: 'anchor', anchor: 'center' },
    { showPercentage: true, style: 'blocks' }
  );
  manager.add(progressBar, 5);

  // 动画进度条
  for (let progress = 0; progress <= 100; progress += 5) {
    progressBar.setProgress(progress);

    const output = manager.render({ width: 80, height: 24, focused: false });
    process.stdout.write('\x1b[2J\x1b[H');  // 清屏
    process.stdout.write(output);

    await sleep(100);
  }

  await sleep(1000);

  // 示例 3: Notification
  const notification = new NotificationWidget(
    'Task completed successfully!',
    'success',
    { type: 'anchor', anchor: 'top-right', offset: { x: -1, y: 1 } }
  );
  manager.add(notification, 20);
  notification.autoHide(3000);

  const output = manager.render({ width: 80, height: 24, focused: false });
  process.stdout.write('\x1b[2J\x1b[H');
  process.stdout.write(output);

  await sleep(3000);

  // 示例 4: Context Menu
  const menuItems: MenuItem[] = [
    { label: 'New File', action: () => console.log('New File') },
    { label: 'Open File', action: () => console.log('Open File') },
    { label: 'Save', action: () => console.log('Save'), disabled: true },
    { label: 'Exit', action: () => process.exit(0) }
  ];

  const contextMenu = new ContextMenuWidget(
    menuItems,
    { type: 'anchor', anchor: 'center' }
  );
  manager.add(contextMenu, 15);

  const menuOutput = manager.render({ width: 80, height: 24, focused: false });
  process.stdout.write('\x1b[2J\x1b[H');
  process.stdout.write(menuOutput);

  // 清理
  await sleep(3000);
  process.stdout.write('\x1b[?25h');  // 显示光标
}

demo().catch(console.error);
```

---

## 运行代码

### 项目设置

```bash
mkdir widget-demo
cd widget-demo
npm init -y
npm install --save-dev typescript @types/node

cat > tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true
  }
}
EOF
```

### 文件组织

```
widget-demo/
├── src/
│   ├── types.ts
│   ├── base-widget.ts
│   ├── tooltip-widget.ts
│   ├── progress-bar-widget.ts
│   ├── context-menu-widget.ts
│   ├── notification-widget.ts
│   ├── widget-manager.ts
│   └── example.ts
├── package.json
└── tsconfig.json
```

### 编译和运行

```bash
npx tsc
node dist/example.js
```

---

## 定制指南

### 1. 自定义 Widget 样式

```typescript
class CustomTooltip extends TooltipWidget {
  render(ctx: RenderContext): string {
    // 自定义样式
    return `\x1b[44m\x1b[37m ${this.text} \x1b[0m`;  // 蓝底白字
  }
}
```

### 2. 添加动画效果

```typescript
class AnimatedProgressBar extends ProgressBarWidget {
  private animationFrame = 0;

  render(ctx: RenderContext): string {
    const base = super.render(ctx);
    const spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    const frame = spinner[this.animationFrame % spinner.length];
    this.animationFrame++;

    return `${frame} ${base}`;
  }
}
```

### 3. 响应式 Widget

```typescript
class ResponsiveWidget extends BaseWidget {
  render(ctx: RenderContext): string {
    const size = this.getSize();

    // 根据可用空间调整
    if (ctx.width < size.width) {
      return this.renderCompact(ctx);
    } else {
      return this.renderFull(ctx);
    }
  }

  private renderCompact(ctx: RenderContext): string {
    // 紧凑模式
    return '...';
  }

  private renderFull(ctx: RenderContext): string {
    // 完整模式
    return 'Full content';
  }
}
```

---

## 总结

### 核心实现

**1. BaseWidget**: 提供通用功能
**2. 具体 Widget**: Tooltip, ProgressBar, ContextMenu, Notification
**3. WidgetManager**: 统一管理和渲染
**4. 定位系统**: Absolute, Relative, Anchor

### 关键技术

**1. 组件化**: 每个 Widget 独立封装
**2. 定位灵活**: 多种定位策略
**3. Z-Index 管理**: 控制层叠顺序
**4. 边界处理**: Clamp 到终端范围

### 扩展方向

**1. 更多 Widget**: Modal, Dropdown, Tabs
**2. 主题系统**: 统一样式管理
**3. 动画系统**: 过渡和动画效果
**4. 事件系统**: 鼠标和键盘事件

### 下一步

- **实战练习**: 创建自己的 Widget
- **深入学习**: 阅读 **07_实战代码_04_复杂UI定制案例.md**
- **源码参考**: `sourcecode/pi-mono/packages/tui/src/components/`

---

**记住**: Widget 的核心是封装和复用，保持接口简单，功能专注！

---

**版本**: v1.0
**最后更新**: 2026-02-21
**维护者**: Claude Code
