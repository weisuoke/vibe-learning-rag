# 核心概念 04：Provider 切换命令

> **掌握交互式命令与 CLI 参数，实现灵活的 Provider 切换**

---

## 核心命令概览

| 命令 | 功能 | 使用场景 |
|------|------|----------|
| `/model` | 选择模型 | 临时切换、探索新模型 |
| `/provider` | 选择 Provider | 切换服务商 |
| `/reload` | 重载配置 | 配置文件修改后 |
| `/session` | 查看会话信息 | 监控使用情况 |
| `/scoped-models` | 配置快捷模型 | 管理常用模型列表 |

---

## 交互式命令

### 1. /model 命令

**功能**：交互式选择模型

```bash
> /model
```

**显示效果**：

```
┌─────────────────────────────────────────────────────────────┐
│ Select a model                                              │
├─────────────────────────────────────────────────────────────┤
│ Search: _                                                   │
├─────────────────────────────────────────────────────────────┤
│ ● claude-3-5-sonnet-20241022 (Anthropic)                   │
│   Claude 3.5 Sonnet - 200K context - $3/$15 per MTok       │
│                                                             │
│   claude-3-5-haiku-20241022 (Anthropic)                    │
│   Claude 3.5 Haiku - 200K context - $0.8/$4 per MTok       │
└─────────────────────────────────────────────────────────────┘
```

**搜索功能**：

```bash
# 按名称搜索
Search: sonnet

# 按 Provider 搜索
Search: openai

# 按标签搜索
Search: coding
```

### 2. /provider 命令

**功能**：切换 Provider（保持当前模型类型）

```bash
> /provider
```

**示例**：

```bash
# 当前使用：claude-3-5-sonnet-20241022 (Anthropic)
> /provider
# 选择 OpenAI
✓ Switched to gpt-4o (OpenAI)
```

### 3. /reload 命令

**功能**：热重载配置文件

```bash
> /reload
```

**使用场景**：

```bash
# 1. 修改配置文件
vim ~/.pi/agent/models.json

# 2. 重载配置（无需重启）
> /reload
✓ Configuration reloaded

# 3. 验证新配置
> /model
# 应该显示新添加的模型
```

### 4. /session 命令

**功能**：查看当前会话信息

```bash
> /session
```

**输出示例**：

```
Session Information:
  Model: claude-3-5-sonnet-20241022
  Provider: anthropic
  Context Window: 200000 tokens
  Tokens Used: 12345 input, 5678 output
  Cost: $0.12
  Duration: 15m 32s
```

### 5. /scoped-models 命令

**功能**：管理 Scoped Models 配置

```bash
> /scoped-models
```

**交互界面**：

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

## CLI 启动参数

### 基本语法

```bash
pi [options]
```

### 常用参数

#### --provider

指定 Provider：

```bash
pi --provider anthropic
pi --provider openai
pi --provider ollama
```

#### --model

指定模型：

```bash
pi --model claude-3-5-sonnet-20241022
pi --model gpt-4o
pi --model llama3.1:8b
```

#### 组合使用

```bash
pi --provider anthropic --model claude-3-5-sonnet-20241022
```

---

## 会话延续

### 跨模型对话

**场景**：在不同模型间继续同一对话

```bash
# 1. 使用 Haiku 开始对话
pi --model claude-3-5-haiku-20241022
> Explain async/await

# 2. 切换到 Sonnet 继续深入
> /model
# 选择 claude-3-5-sonnet-20241022
> Can you provide a more detailed example?

# 3. 切换到 Opus 进行架构设计
> /model
# 选择 claude-opus-4-20250514
> How would you design this for production?
```

**上下文保持**：
- 对话历史在模型切换后保留
- 新模型可以看到之前的对话内容
- 适合渐进式深入的场景

---

## 热重载机制

### 工作原理

```typescript
// 配置文件监听
watchFiles([
  '~/.pi/agent/models.json',
  '~/.pi/agent/auth.json',
  '.pi/settings.json'
]);

// 文件变化时
onFileChange(() => {
  reloadConfiguration();
  updateModelList();
  notifyUser('Configuration reloaded');
});
```

### 支持的配置

**可热重载**：
- models.json（模型配置）
- auth.json（认证信息）
- settings.json（用户设置）

**需要重启**：
- 环境变量
- CLI 参数

### 使用示例

```bash
# 终端 1：运行 Pi
pi

# 终端 2：添加新 Provider
cat >> ~/.pi/agent/models.json <<EOF
{
  "providers": {
    "ollama": {
      "apiType": "openai-compatible",
      "baseUrl": "http://localhost:11434",
      "models": {
        "llama3.1:8b": {
          "id": "llama3.1:8b",
          "name": "Llama 3.1 8B"
        }
      }
    }
  }
}
EOF

# 终端 1：重载配置
> /reload
✓ Configuration reloaded

# 验证新模型
> /model
# 应该显示 llama3.1:8b
```

---

## 命令优先级

### 优先级顺序

```
1. 运行时命令 (/model, /provider)
2. CLI 参数 (--model, --provider)
3. Scoped Models (Ctrl+P)
4. 项目配置 (.pi/settings.json)
5. 全局配置 (~/.pi/agent/settings.json)
6. 环境变量
7. 默认值
```

### 示例

```bash
# 全局配置：Haiku
# 项目配置：Sonnet
# CLI 参数：Opus
pi --model claude-opus-4-20250514

# 实际使用：Opus（CLI 参数优先）

# 运行时切换
> /model
# 选择 Haiku

# 实际使用：Haiku（运行时命令最高优先级）
```

---

## 实战场景

### 场景 1：快速测试不同模型

```bash
# 测试 Haiku
pi --model claude-3-5-haiku-20241022
> Generate a hello world function

# 测试 Sonnet
pi --model claude-3-5-sonnet-20241022
> Generate a hello world function

# 对比结果
```

### 场景 2：成本控制

```bash
# 日常开发用 Haiku
pi --model claude-3-5-haiku-20241022

# 遇到复杂问题时切换
> /model
# 选择 claude-3-5-sonnet-20241022

# 问题解决后切回
> /model
# 选择 claude-3-5-haiku-20241022
```

### 场景 3：多环境配置

```bash
# 开发环境：本地模型
if [ "$ENV" = "dev" ]; then
  pi --provider ollama --model llama3.1:8b
else
  # 生产环境：云端模型
  pi --provider anthropic --model claude-3-5-sonnet-20241022
fi
```

### 场景 4：自动化脚本

```bash
#!/bin/bash
# auto-review.sh

# 使用 Sonnet 进行代码审查
pi --model claude-3-5-sonnet-20241022 <<EOF
Review the code in src/
EOF
```

---

## 常见问题

### Q1: /reload 会中断当前对话吗？

**答**：不会。对话历史保留，只是重新加载配置。

### Q2: 如何查看所有可用命令？

**答**：输入 `/help` 或 `/?`

### Q3: CLI 参数和运行时命令冲突怎么办？

**答**：运行时命令优先级更高，会覆盖 CLI 参数。

### Q4: 如何恢复到默认模型？

**答**：使用 `/model` 命令选择默认模型，或重启 Pi。

---

## 下一步

- **配置文件层级**：阅读 [03_核心概念_05_配置文件层级.md](./03_核心概念_05_配置文件层级.md)
- **实战命令**：阅读 [07_实战代码_05_多Provider切换脚本.md](./07_实战代码_05_多Provider切换脚本.md)

---

**记住**：命令是工具，熟练掌握它们能让你在不同场景下灵活切换 Provider 和模型。
