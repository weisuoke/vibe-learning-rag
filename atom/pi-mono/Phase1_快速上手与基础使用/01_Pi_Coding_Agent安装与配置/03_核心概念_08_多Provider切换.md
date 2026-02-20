# 核心概念 08：多 Provider 切换

> **知识点定位**：掌握 Pi Coding Agent 的多 Provider 切换机制，理解 /model、/provider 命令和 Scoped Models 的使用

---

## 一、多 Provider 切换概述

### 1.1 为什么需要多 Provider 切换

**场景示例：**

```
场景 1：成本优化
- 简单任务用 Claude Haiku 4（便宜）
- 复杂任务用 Claude Opus 4（强大）
- 节省 80% API 成本

场景 2：能力互补
- 代码生成用 Claude Sonnet 4（代码能力强）
- 数学推理用 o1（推理能力强）
- 快速响应用 GPT-4o（速度快）

场景 3：可用性保障
- 主 Provider 限流时切换到备用
- 自动 Fallback 策略
- 提高服务可用性
```

### 1.2 切换方式对比

| 切换方式 | 快捷键 | 适用场景 | 持久化 |
|---------|-------|---------|--------|
| `/model` 命令 | `Ctrl+L` | 手动选择任意模型 | 当前会话 |
| Scoped Models | `Ctrl+P` / `Shift+Ctrl+P` | 快速循环常用模型 | 配置文件 |
| CLI 参数 | — | 启动时指定模型 | 单次启动 |
| 配置文件 | — | 设置默认模型 | 永久 |

---

## 二、/model 命令详解

### 2.1 基本用法

```bash
# 在 Pi 交互模式中
> /model

┌─────────────────────────────────────┐
│ Select model:                       │
│                                     │
│ Anthropic                           │
│ ✓ claude-opus-4                     │
│   claude-sonnet-4                   │
│   claude-haiku-4                    │
│                                     │
│ OpenAI                              │
│   gpt-4o                            │
│   o1                                │
│   o3-mini                           │
│                                     │
│ xAI                                 │
│   grok-2-latest                     │
│   grok-vision-beta                  │
│                                     │
│ Google                              │
│   gemini-2.0-flash-exp              │
│   gemini-exp-1206                   │
│                                     │
│ Groq                                │
│   llama-3.3-70b-versatile           │
│   deepseek-r1-distill-llama-70b     │
└─────────────────────────────────────┘
```

### 2.2 快捷键

```bash
# 打开模型选择器
Ctrl+L

# 或输入命令
/model
```

### 2.3 模型分组

模型按 Provider 分组显示：

```
Anthropic
├── claude-opus-4
├── claude-sonnet-4
└── claude-haiku-4

OpenAI
├── gpt-4o
├── o1
└── o3-mini

xAI
├── grok-2-latest
└── grok-vision-beta
```

### 2.4 模型信息显示

```
claude-opus-4
├── 名称: Claude Opus 4
├── 上下文: 200K tokens
├── 成本: $15/1M input, $75/1M output
└── 能力: 文本 + 图片
```

### 2.5 搜索过滤

```
> /model
# 输入搜索词
> opus

┌─────────────────────────────────────┐
│ Search: opus                        │
│                                     │
│ Anthropic                           │
│ ✓ claude-opus-4                     │
└─────────────────────────────────────┘
```

---

## 三、Scoped Models 快速切换

### 3.1 什么是 Scoped Models

Scoped Models 是预定义的常用模型列表，支持快捷键快速循环切换。

**优势：**
- ✅ 快捷键切换（`Ctrl+P` / `Shift+Ctrl+P`）
- ✅ 只显示常用模型
- ✅ 提高切换效率
- ✅ 减少选择疲劳

### 3.2 配置 Scoped Models

**方式 1：通过 /scoped-models 命令**

```bash
> /scoped-models

┌─────────────────────────────────────┐
│ Enable models for Ctrl+P cycling:  │
│                                     │
│ Anthropic                           │
│ ☑ claude-opus-4                     │
│ ☑ claude-sonnet-4                   │
│ ☐ claude-haiku-4                    │
│                                     │
│ OpenAI                              │
│ ☑ gpt-4o                            │
│ ☐ o1                                │
│ ☐ o3-mini                           │
│                                     │
│ xAI                                 │
│ ☐ grok-2-latest                     │
└─────────────────────────────────────┘
```

**方式 2：通过 settings.json**

```json
{
  "scopedModels": [
    "claude-opus-4",
    "claude-sonnet-4",
    "gpt-4o"
  ]
}
```

### 3.3 快捷键切换

```bash
# 向前循环（下一个模型）
Ctrl+P

# 向后循环（上一个模型）
Shift+Ctrl+P
```

**切换顺序：**
```
claude-opus-4 → claude-sonnet-4 → gpt-4o → claude-opus-4 → ...
```

### 3.4 实时切换示例

```bash
# 当前模型：claude-opus-4
> 帮我重构这个复杂的函数
# 使用 Opus 4 处理复杂任务

# 按 Ctrl+P 切换到 claude-sonnet-4
> 修复这个小 bug
# 使用 Sonnet 4 处理简单任务

# 按 Ctrl+P 切换到 gpt-4o
> 快速生成一个测试
# 使用 GPT-4o 快速响应
```

---

## 四、CLI 参数切换

### 4.1 启动时指定模型

```bash
# 指定 Provider 和 Model
pi --provider anthropic --model claude-opus-4

# 只指定 Model（自动推断 Provider）
pi --model claude-sonnet-4

# 简写
pi --model gpt-4o
```

### 4.2 继续会话时切换

```bash
# 继续上次会话，但使用不同模型
pi -c --model claude-haiku-4

# 恢复特定会话，使用不同模型
pi -r --model gpt-4o
```

### 4.3 一次性任务

```bash
# 使用特定模型执行一次性任务
pi --model o1 "解决这个数学问题：..."

# Print 模式
pi --print --model claude-opus-4 "分析这段代码"
```

---

## 五、配置文件默认模型

### 5.1 全局默认模型

```json
// ~/.pi/agent/settings.json
{
  "provider": "anthropic",
  "model": "claude-sonnet-4"
}
```

**效果：**
- 所有项目默认使用 Claude Sonnet 4
- 除非项目配置或 CLI 参数覆盖

### 5.2 项目默认模型

```json
// .pi/settings.json
{
  "provider": "openai",
  "model": "gpt-4o"
}
```

**效果：**
- 当前项目默认使用 GPT-4o
- 覆盖全局配置

### 5.3 优先级

```
CLI 参数（最高）
    ↓
项目配置 (.pi/settings.json)
    ↓
全局配置 (~/.pi/agent/settings.json)
    ↓
环境变量
    ↓
默认值（最低）
```

---

## 六、多 Provider 策略

### 6.1 成本优化策略

**策略：根据任务复杂度选择模型**

```json
// .pi/settings.json
{
  "scopedModels": [
    "claude-haiku-4",      // 简单任务（$0.25/1M input）
    "claude-sonnet-4",     // 中等任务（$3/1M input）
    "claude-opus-4"        // 复杂任务（$15/1M input）
  ]
}
```

**使用方式：**
```bash
# 简单任务：文件读取、格式化
Ctrl+P → claude-haiku-4

# 中等任务：代码重构、bug 修复
Ctrl+P → claude-sonnet-4

# 复杂任务：架构设计、算法优化
Ctrl+P → claude-opus-4
```

**成本对比：**
```
100K tokens 输入 + 50K tokens 输出

Haiku:  $0.025 + $0.063 = $0.088
Sonnet: $0.30 + $0.75 = $1.05
Opus:   $1.50 + $3.75 = $5.25

节省: 使用 Haiku 比 Opus 节省 98%
```

### 6.2 能力互补策略

**策略：根据任务类型选择最佳模型**

```json
{
  "scopedModels": [
    "claude-sonnet-4",     // 代码生成
    "o1",                  // 数学推理
    "gpt-4o",              // 快速响应
    "grok-vision-beta"     // 图片分析
  ]
}
```

**使用场景：**
```
代码生成 → Claude Sonnet 4
- 编写函数
- 重构代码
- 生成测试

数学推理 → o1
- 算法设计
- 复杂计算
- 逻辑推理

快速响应 → GPT-4o
- 简单问答
- 文档查询
- 快速修复

图片分析 → Grok Vision
- UI 截图分析
- 设计稿转代码
- 图表解读
```

### 6.3 可用性保障策略

**策略：设置主备 Provider**

```json
{
  "scopedModels": [
    "claude-sonnet-4",           // 主 Provider
    "gpt-4o",                    // 备用 Provider 1
    "groq/llama-3.3-70b"         // 备用 Provider 2
  ]
}
```

**切换场景：**
```
1. Claude 限流 → 切换到 GPT-4o
2. OpenAI 故障 → 切换到 Groq
3. 成本控制 → 切换到本地模型
```

### 6.4 开发/生产分离策略

**开发环境：**
```json
// .pi/settings.json (开发)
{
  "provider": "ollama",
  "model": "llama3.1:8b",
  "scopedModels": [
    "llama3.1:8b",
    "qwen2.5-coder:7b"
  ]
}
```

**生产环境：**
```json
// .pi/settings.json (生产)
{
  "provider": "anthropic",
  "model": "claude-opus-4",
  "scopedModels": [
    "claude-opus-4",
    "claude-sonnet-4"
  ]
}
```

---

## 七、实战场景

### 7.1 场景 1：日常开发工作流

```bash
# 启动 Pi，默认使用 Sonnet 4
pi

# 简单任务：格式化代码
Ctrl+P → claude-haiku-4
> 格式化这个文件

# 中等任务：重构函数
Ctrl+P → claude-sonnet-4
> 重构 getUserData 函数

# 复杂任务：架构设计
Ctrl+P → claude-opus-4
> 设计一个可扩展的插件系统

# 快速响应：查询文档
Ctrl+P → gpt-4o
> React useEffect 的清理函数怎么用？
```

### 7.2 场景 2：成本敏感项目

```bash
# 配置 Scoped Models
> /scoped-models
# 启用：haiku-4, sonnet-4

# 默认使用 Haiku（最便宜）
> 读取 package.json 并列出依赖

# 需要更强能力时切换到 Sonnet
Ctrl+P
> 分析这个复杂的 bug

# 完成后切回 Haiku
Ctrl+P
> 继续简单任务
```

### 7.3 场景 3：多模态任务

```bash
# 配置支持图片的模型
> /scoped-models
# 启用：claude-sonnet-4, gpt-4o, grok-vision-beta

# 分析 UI 截图
Ctrl+P → grok-vision-beta
> 分析这个 UI 设计（粘贴截图）

# 生成代码
Ctrl+P → claude-sonnet-4
> 根据设计生成 React 组件

# 快速调整
Ctrl+P → gpt-4o
> 修改按钮颜色
```

### 7.4 场景 4：离线开发

```bash
# 配置本地模型
> /scoped-models
# 启用：ollama/llama3.1:8b, ollama/qwen2.5-coder:7b

# 使用本地模型开发
pi --provider ollama --model llama3.1:8b

# 需要更强能力时切换到云端
Ctrl+L
# 选择 claude-sonnet-4

# 完成后切回本地
Ctrl+L
# 选择 ollama/llama3.1:8b
```

---

## 八、高级技巧

### 8.1 模型别名

```json
// ~/.pi/agent/models.json
{
  "providers": {
    "anthropic": {
      "models": [
        {
          "id": "claude-opus-4",
          "name": "Opus (Complex)"
        },
        {
          "id": "claude-sonnet-4",
          "name": "Sonnet (Daily)"
        },
        {
          "id": "claude-haiku-4",
          "name": "Haiku (Fast)"
        }
      ]
    }
  }
}
```

**效果：**
```
模型选择器显示：
- Opus (Complex)
- Sonnet (Daily)
- Haiku (Fast)
```

### 8.2 自动切换脚本

```bash
#!/bin/bash
# auto-switch-model.sh

# 根据任务类型自动选择模型
task_type=$1

case $task_type in
  "simple")
    pi --model claude-haiku-4
    ;;
  "medium")
    pi --model claude-sonnet-4
    ;;
  "complex")
    pi --model claude-opus-4
    ;;
  "math")
    pi --model o1
    ;;
  *)
    echo "Usage: $0 {simple|medium|complex|math}"
    exit 1
    ;;
esac
```

**使用方式：**
```bash
./auto-switch-model.sh simple
./auto-switch-model.sh complex
```

### 8.3 成本追踪

```bash
# 查看当前会话成本
> /session

Session: my-session
Tokens: 15.2K (input: 8.1K, output: 7.1K)
Cost: $0.45
Model: claude-opus-4

# 切换到更便宜的模型
Ctrl+P → claude-haiku-4

# 继续工作，成本降低
```

### 8.4 模型性能对比

```bash
# 测试不同模型的响应速度
time pi --model claude-haiku-4 --print "生成一个函数"
# 2.3 秒

time pi --model claude-sonnet-4 --print "生成一个函数"
# 3.1 秒

time pi --model claude-opus-4 --print "生成一个函数"
# 4.5 秒
```

---

## 九、故障排查

### 9.1 模型切换失败

**问题：** 切换模型后仍使用旧模型

**排查步骤：**

```bash
# 1. 检查当前模型
> /session
# 查看 Model 字段

# 2. 重新切换
Ctrl+L
# 选择新模型

# 3. 验证切换
# 查看底部状态栏的模型名称

# 4. 重启 Pi
/quit
pi
```

### 9.2 Scoped Models 未生效

**问题：** `Ctrl+P` 无法切换模型

**解决方案：**

```bash
# 1. 检查配置
cat .pi/settings.json | jq '.scopedModels'

# 2. 重新配置
> /scoped-models
# 启用至少 2 个模型

# 3. 重新加载
> /reload

# 4. 测试切换
Ctrl+P
```

### 9.3 模型不可用

**问题：** 选择的模型无法使用

**排查步骤：**

```bash
# 1. 检查 API Key
echo $ANTHROPIC_API_KEY

# 2. 检查 Provider 状态
pi --provider anthropic --model claude-opus-4

# 3. 查看错误日志
PI_DEBUG=1 pi

# 4. 切换到备用模型
Ctrl+L
# 选择其他可用模型
```

---

## 十、最佳实践

### 10.1 Scoped Models 配置建议

**推荐配置（3-5 个模型）：**

```json
{
  "scopedModels": [
    "claude-haiku-4",      // 快速任务
    "claude-sonnet-4",     // 日常任务
    "claude-opus-4",       // 复杂任务
    "gpt-4o"               // 备用
  ]
}
```

**不推荐：**
```json
{
  "scopedModels": [
    // ❌ 太多模型，难以选择
    "claude-opus-4",
    "claude-sonnet-4",
    "claude-haiku-4",
    "gpt-4o",
    "o1",
    "o3-mini",
    "grok-2-latest",
    "gemini-2.0-flash-exp"
  ]
}
```

### 10.2 切换时机建议

**何时切换到更强模型：**
- ✅ 当前模型无法理解任务
- ✅ 需要更深入的分析
- ✅ 代码质量不满意
- ✅ 需要更复杂的推理

**何时切换到更快模型：**
- ✅ 简单的格式化任务
- ✅ 快速查询文档
- ✅ 简单的 bug 修复
- ✅ 成本控制

### 10.3 成本控制建议

**策略 1：默认使用中等模型**
```json
{
  "model": "claude-sonnet-4",
  "scopedModels": [
    "claude-haiku-4",
    "claude-sonnet-4",
    "claude-opus-4"
  ]
}
```

**策略 2：任务开始用便宜模型**
```bash
# 先用 Haiku 尝试
pi --model claude-haiku-4

# 如果不行，切换到 Sonnet
Ctrl+P

# 最后才用 Opus
Ctrl+P
```

**策略 3：定期检查成本**
```bash
# 每天检查成本
> /session
# 查看 Cost 字段

# 如果超预算，切换到更便宜的模型
```

---

## 十一、总结

### 11.1 核心要点

1. **切换方式**：`/model`（手动）、`Ctrl+P`（快速）、CLI 参数、配置文件
2. **Scoped Models**：预定义常用模型，快捷键快速切换
3. **优先级**：CLI > 项目配置 > 全局配置 > 环境变量
4. **策略**：成本优化、能力互补、可用性保障
5. **最佳实践**：3-5 个 Scoped Models，根据任务复杂度切换

### 11.2 快捷键总结

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+L` | 打开模型选择器 |
| `Ctrl+P` | 向前循环 Scoped Models |
| `Shift+Ctrl+P` | 向后循环 Scoped Models |

### 11.3 下一步

- 学习 **实战代码示例**（实战代码 01-08）
- 掌握 **Extensions 开发**（Phase 3）
- 了解 **高级功能**（Phase 4）

---

**参考资料：**
- [Pi Coding Agent README - Interactive Mode](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/README.md#interactive-mode)
- [Pi Settings Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/settings.md)
- [Pi Models Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/models.md)

**文档版本**：v1.0 (2026-02-18)
