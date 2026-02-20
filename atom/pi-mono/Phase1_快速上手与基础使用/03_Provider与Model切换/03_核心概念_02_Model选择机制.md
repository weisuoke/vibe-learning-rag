# 核心概念 02：Model 选择机制

> **掌握 Pi 的 3 种模型选择方法：命令、快捷键、CLI**

---

## 为什么需要模型选择？

### 不同任务需要不同模型

```typescript
// 简单任务：代码格式化
const task1 = "Format this JSON";
// 最佳选择：Haiku ($0.25/MTok) - 快速且便宜

// 中等任务：代码重构
const task2 = "Refactor this component to use hooks";
// 最佳选择：Sonnet ($3/MTok) - 平衡性能与成本

// 复杂任务：架构设计
const task3 = "Design a scalable microservices architecture";
// 最佳选择：Opus ($15/MTok) - 最强推理能力
```

### 模型选择的三个维度

1. **性能**：推理能力、代码质量、准确度
2. **成本**：每百万 tokens 的价格
3. **速度**：响应时间、吞吐量

**核心原则**：根据任务复杂度选择合适的模型，避免"大炮打蚊子"或"小刀砍大树"。

---

## Pi 的 3 种模型选择方法

### 方法对比

| 方法 | 使用场景 | 优点 | 缺点 |
|------|----------|------|------|
| **`/model` 命令** | 临时切换、探索新模型 | 交互式、支持搜索 | 需要输入命令 |
| **Scoped Models** | 日常工作、频繁切换 | 快捷键、零中断 | 需要预配置 |
| **CLI 参数** | 脚本自动化、CI/CD | 可编程、可重复 | 不灵活 |

---

## 方法 1：`/model` 命令（交互式选择）

### 基本用法

```bash
# 在 Pi 中输入
> /model
```

**效果**：显示交互式模型选择器

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
│                                                             │
│   claude-opus-4-20250514 (Anthropic)                       │
│   Claude Opus 4 - 200K context - $15/$75 per MTok          │
│                                                             │
│   gpt-4-turbo (OpenAI)                                     │
│   GPT-4 Turbo - 128K context - $10/$30 per MTok            │
│                                                             │
│   gpt-4o (OpenAI)                                          │
│   GPT-4o - 128K context - $2.5/$10 per MTok                │
└─────────────────────────────────────────────────────────────┘
```

### 搜索功能

**按名称搜索**：

```bash
> /model
Search: sonnet
```

**结果**：只显示包含 "sonnet" 的模型

```
┌─────────────────────────────────────────────────────────────┐
│ ● claude-3-5-sonnet-20241022 (Anthropic)                   │
│   Claude 3.5 Sonnet - 200K context - $3/$15 per MTok       │
└─────────────────────────────────────────────────────────────┘
```

**按 Provider 搜索**：

```bash
> /model
Search: openai
```

**结果**：只显示 OpenAI 的模型

**按标签搜索**：

```bash
> /model
Search: coding
```

**结果**：显示标记为 "coding" 的模型

### 模型信息显示

每个模型显示以下信息：

```
claude-3-5-sonnet-20241022 (Anthropic)
Claude 3.5 Sonnet - 200K context - $3/$15 per MTok
```

**字段解析**：
- **模型 ID**：`claude-3-5-sonnet-20241022`
- **Provider**：`Anthropic`
- **显示名称**：`Claude 3.5 Sonnet`
- **上下文窗口**：`200K` tokens
- **成本**：`$3` (input) / `$15` (output) per MTok

### 选择确认

```bash
# 选择模型后
✓ Switched to claude-3-5-sonnet-20241022

# 验证当前模型
> /session
```

**输出**：

```
Session Information:
  Model: claude-3-5-sonnet-20241022
  Provider: anthropic
  Context Window: 200000 tokens
  Tokens Used: 1234 input, 567 output
  Cost: $0.012
```

---

## 方法 2：Scoped Models（快捷键切换）

### 什么是 Scoped Models？

**定义**：预配置的 3-5 个常用模型，通过键盘快捷键快速循环切换。

**类比**：
- **TypeScript 类比**：就像 VS Code 的 "最近打开的文件" (Ctrl+Tab)
- **日常类比**：就像电视遥控器的 "频道收藏" 按钮

### 配置 Scoped Models

**配置文件**：`.pi/settings.json` 或 `~/.pi/agent/settings.json`

```json
{
  "scopedModels": [
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "claude-opus-4-20250514"
  ]
}
```

**推荐配置**（按成本递增）：

```json
{
  "scopedModels": [
    "claude-3-5-haiku-20241022",    // 快速任务
    "claude-3-5-sonnet-20241022",   // 日常开发
    "claude-opus-4-20250514",       // 复杂问题
    "gpt-4o",                       // 多模态任务
    "llama3.1:8b"                   // 本地测试
  ]
}
```

### 快捷键操作

**向前循环**：`Ctrl+P`

```
Haiku → Sonnet → Opus → GPT-4o → Llama → Haiku → ...
```

**向后循环**：`Shift+Ctrl+P`

```
Haiku → Llama → GPT-4o → Opus → Sonnet → Haiku → ...
```

**视觉反馈**：

```bash
# 按 Ctrl+P
✓ Switched to claude-3-5-sonnet-20241022 (2/5)

# 再按 Ctrl+P
✓ Switched to claude-opus-4-20250514 (3/5)
```

### 交互式配置

**使用 `/scoped-models` 命令**：

```bash
> /scoped-models
```

**效果**：显示当前配置并允许修改

```
┌─────────────────────────────────────────────────────────────┐
│ Scoped Models Configuration                                 │
├─────────────────────────────────────────────────────────────┤
│ Current scoped models (3):                                  │
│   1. claude-3-5-haiku-20241022                              │
│   2. claude-3-5-sonnet-20241022                             │
│   3. claude-opus-4-20250514                                 │
│                                                             │
│ Options:                                                    │
│   [a] Add model                                             │
│   [r] Remove model                                          │
│   [c] Clear all                                             │
│   [q] Quit                                                  │
└─────────────────────────────────────────────────────────────┘
```

**添加模型**：

```bash
> a
Search for model: gpt-4o
✓ Added gpt-4o to scoped models
```

**删除模型**：

```bash
> r
Select model to remove: 3
✓ Removed claude-opus-4-20250514 from scoped models
```

### 最佳实践

**1. 数量控制**：3-5 个模型最优

```json
// ✅ 推荐：3 个模型（快速切换）
{
  "scopedModels": [
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "claude-opus-4-20250514"
  ]
}

// ❌ 不推荐：10 个模型（切换困难）
{
  "scopedModels": [
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "claude-opus-4-20250514",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    "grok-2-1212",
    "gemini-2.0-flash-exp",
    "mistral-large-latest",
    "llama3.1:8b"
  ]
}
```

**2. 按成本排序**：从便宜到昂贵

```json
{
  "scopedModels": [
    "claude-3-5-haiku-20241022",    // $0.25/MTok
    "claude-3-5-sonnet-20241022",   // $3/MTok
    "claude-opus-4-20250514"        // $15/MTok
  ]
}
```

**原因**：默认使用第一个模型（最便宜），需要时向后切换。

**3. 场景分组**：不同项目不同配置

```json
// 前端项目：.pi/settings.json
{
  "scopedModels": [
    "claude-3-5-haiku-20241022",    // 快速代码生成
    "claude-3-5-sonnet-20241022",   // 组件开发
    "gpt-4o"                        // UI 设计（多模态）
  ]
}

// 后端项目：.pi/settings.json
{
  "scopedModels": [
    "claude-3-5-haiku-20241022",    // API 开发
    "claude-3-5-sonnet-20241022",   // 业务逻辑
    "claude-opus-4-20250514"        // 架构设计
  ]
}

// 本地开发：.pi/settings.json
{
  "scopedModels": [
    "llama3.1:8b",                  // 离线测试
    "claude-3-5-haiku-20241022",    // 在线验证
    "claude-3-5-sonnet-20241022"    // 复杂任务
  ]
}
```

---

## 方法 3：CLI 参数（启动时指定）

### 基本用法

**指定 Provider**：

```bash
pi --provider anthropic
```

**指定 Model**：

```bash
pi --model claude-3-5-sonnet-20241022
```

**同时指定**：

```bash
pi --provider anthropic --model claude-3-5-sonnet-20241022
```

### 使用场景

#### 1. 脚本自动化

```bash
#!/bin/bash
# generate-docs.sh

# 使用便宜的模型生成文档
pi --model claude-3-5-haiku-20241022 <<EOF
Generate API documentation for src/api/
EOF
```

#### 2. CI/CD 集成

```yaml
# .github/workflows/code-review.yml
name: AI Code Review

on: [pull_request]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run AI Review
        run: |
          pi --provider anthropic \
             --model claude-3-5-sonnet-20241022 \
             "Review the changes in this PR"
```

#### 3. 成本控制

```bash
#!/bin/bash
# cost-aware-pi.sh

TASK_COMPLEXITY=$1

if [ "$TASK_COMPLEXITY" = "simple" ]; then
  MODEL="claude-3-5-haiku-20241022"
elif [ "$TASK_COMPLEXITY" = "medium" ]; then
  MODEL="claude-3-5-sonnet-20241022"
else
  MODEL="claude-opus-4-20250514"
fi

pi --model "$MODEL"
```

#### 4. 多环境配置

```bash
# 开发环境：使用本地模型
if [ "$ENV" = "development" ]; then
  pi --provider ollama --model llama3.1:8b
else
  # 生产环境：使用云端模型
  pi --provider anthropic --model claude-3-5-sonnet-20241022
fi
```

### CLI 参数优先级

```
CLI 参数 > 项目配置 > 全局配置 > 环境变量 > 默认值
```

**示例**：

```bash
# 全局配置：~/.pi/agent/settings.json
{
  "defaultModel": "claude-3-5-haiku-20241022"
}

# 项目配置：.pi/settings.json
{
  "defaultModel": "claude-3-5-sonnet-20241022"
}

# 启动命令
pi --model claude-opus-4-20250514

# 实际使用：claude-opus-4-20250514 (CLI 参数优先)
```

---

## 模型元数据

### 元数据字段

Pi 为每个模型提供丰富的元数据，用于选择和过滤。

```json
{
  "id": "claude-3-5-sonnet-20241022",
  "name": "Claude 3.5 Sonnet",
  "provider": "anthropic",
  "contextWindow": 200000,
  "maxOutput": 8192,
  "reasoning": true,
  "cost": {
    "input": 3.0,
    "output": 15.0
  },
  "tags": ["coding", "analysis", "reasoning"],
  "capabilities": ["text", "code", "analysis"],
  "deprecated": false,
  "releaseDate": "2024-10-22"
}
```

### 元数据用途

#### 1. 智能过滤

```typescript
// 按上下文窗口过滤
const longContextModels = models.filter(m => m.contextWindow >= 100000);

// 按成本过滤
const cheapModels = models.filter(m => m.cost.input < 1.0);

// 按能力过滤
const reasoningModels = models.filter(m => m.reasoning === true);
```

#### 2. 成本估算

```typescript
function estimateCost(model: Model, inputTokens: number, outputTokens: number): number {
  const inputCost = (inputTokens / 1_000_000) * model.cost.input;
  const outputCost = (outputTokens / 1_000_000) * model.cost.output;
  return inputCost + outputCost;
}

// 示例
const cost = estimateCost(
  models['claude-3-5-sonnet-20241022'],
  10000,  // 10K input tokens
  2000    // 2K output tokens
);
console.log(`Estimated cost: $${cost.toFixed(4)}`);
// 输出：Estimated cost: $0.0600
```

#### 3. 自动选择

```typescript
function selectModel(taskComplexity: 'simple' | 'medium' | 'complex'): string {
  const models = {
    simple: 'claude-3-5-haiku-20241022',
    medium: 'claude-3-5-sonnet-20241022',
    complex: 'claude-opus-4-20250514'
  };
  return models[taskComplexity];
}
```

---

## 模型选择策略

### 策略 1：基于任务复杂度

```typescript
interface TaskComplexityStrategy {
  simple: string[];    // 简单任务：格式化、简单查询
  medium: string[];    // 中等任务：代码重构、功能开发
  complex: string[];   // 复杂任务：架构设计、算法优化
}

const strategy: TaskComplexityStrategy = {
  simple: [
    'claude-3-5-haiku-20241022',
    'gpt-4o-mini'
  ],
  medium: [
    'claude-3-5-sonnet-20241022',
    'gpt-4o'
  ],
  complex: [
    'claude-opus-4-20250514',
    'gpt-4-turbo'
  ]
};
```

### 策略 2：基于成本预算

```typescript
interface CostBudgetStrategy {
  budget: number;      // 每月预算（美元）
  allocation: {
    cheap: number;     // 便宜模型占比
    medium: number;    // 中等模型占比
    expensive: number; // 昂贵模型占比
  };
}

const strategy: CostBudgetStrategy = {
  budget: 100,
  allocation: {
    cheap: 0.7,      // 70% 使用 Haiku
    medium: 0.25,    // 25% 使用 Sonnet
    expensive: 0.05  // 5% 使用 Opus
  }
};
```

### 策略 3：基于响应时间

```typescript
interface LatencyStrategy {
  maxLatency: number;  // 最大延迟（秒）
  models: string[];    // 按速度排序的模型列表
}

const strategy: LatencyStrategy = {
  maxLatency: 5,
  models: [
    'claude-3-5-haiku-20241022',    // 最快
    'gpt-4o-mini',
    'claude-3-5-sonnet-20241022',
    'gpt-4o',
    'claude-opus-4-20250514'        // 最慢
  ]
};
```

### 策略 4：Fallback 链

```typescript
interface FallbackStrategy {
  primary: string;
  fallbacks: string[];
}

const strategy: FallbackStrategy = {
  primary: 'claude-3-5-sonnet-20241022',
  fallbacks: [
    'gpt-4o',                       // 第一备选
    'claude-3-5-haiku-20241022',    // 第二备选
    'llama3.1:8b'                   // 本地备选
  ]
};

async function executeWithFallback(prompt: string): Promise<string> {
  const models = [strategy.primary, ...strategy.fallbacks];

  for (const model of models) {
    try {
      return await callModel(model, prompt);
    } catch (error) {
      console.log(`${model} failed, trying next...`);
    }
  }

  throw new Error('All models failed');
}
```

---

## 选择优先级与覆盖

### 优先级顺序

```
1. CLI 参数 (--model)
2. /model 命令
3. Scoped Models (Ctrl+P)
4. 项目配置 (.pi/settings.json)
5. 全局配置 (~/.pi/agent/settings.json)
6. 环境变量 (ANTHROPIC_MODEL)
7. 默认值
```

### 覆盖示例

**场景**：全局配置使用 Haiku，项目配置使用 Sonnet，CLI 指定 Opus

```bash
# 全局配置：~/.pi/agent/settings.json
{
  "defaultModel": "claude-3-5-haiku-20241022"
}

# 项目配置：.pi/settings.json
{
  "defaultModel": "claude-3-5-sonnet-20241022"
}

# 启动命令
pi --model claude-opus-4-20250514

# 实际使用：claude-opus-4-20250514
```

**场景**：启动后使用 `/model` 命令切换

```bash
# 启动时使用 Sonnet
pi --model claude-3-5-sonnet-20241022

# 运行时切换到 Opus
> /model
# 选择 claude-opus-4-20250514

# 当前使用：claude-opus-4-20250514
```

---

## 实战技巧

### 技巧 1：快速切换工作流

```bash
# 1. 启动时使用默认模型（Haiku）
pi

# 2. 遇到复杂问题，按 Ctrl+P 切换到 Sonnet
# 3. 问题解决后，按 Shift+Ctrl+P 切回 Haiku
```

### 技巧 2：成本监控

```bash
# 查看当前会话成本
> /session

# 输出
Session Information:
  Model: claude-3-5-sonnet-20241022
  Tokens Used: 10000 input, 2000 output
  Cost: $0.06
```

### 技巧 3：模型对比

```bash
# 使用不同模型回答同一问题
> /model
# 选择 claude-3-5-sonnet-20241022
> Explain async/await in JavaScript

> /model
# 选择 gpt-4o
> Explain async/await in JavaScript

# 对比两个回答的质量
```

### 技巧 4：项目模板

```bash
# 创建项目模板
mkdir -p project-template/.pi

cat > project-template/.pi/settings.json <<EOF
{
  "defaultModel": "claude-3-5-sonnet-20241022",
  "scopedModels": [
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "claude-opus-4-20250514"
  ]
}
EOF

# 新项目复制模板
cp -r project-template/.pi new-project/
```

---

## 常见问题

### Q1: 如何查看当前使用的模型？

**答**：使用 `/session` 命令

```bash
> /session
```

### Q2: Scoped Models 最多配置几个？

**答**：建议 3-5 个。太多会导致切换困难，太少会失去灵活性。

### Q3: 如何重置到默认模型？

**答**：使用 `/model` 命令选择默认模型，或重启 Pi。

### Q4: CLI 参数和 Scoped Models 冲突怎么办？

**答**：CLI 参数优先级更高，会覆盖 Scoped Models 的第一个模型。

### Q5: 如何在脚本中动态选择模型？

**答**：使用环境变量或 CLI 参数

```bash
#!/bin/bash
MODEL=${1:-claude-3-5-haiku-20241022}
pi --model "$MODEL"
```

---

## 下一步

- **Scoped Models 详解**：阅读 [03_核心概念_03_Scoped_Models快速切换.md](./03_核心概念_03_Scoped_Models快速切换.md)
- **实战配置**：阅读 [07_实战代码_02_Scoped_Models配置.md](./07_实战代码_02_Scoped_Models配置.md)

---

**记住**：选择合适的模型是成本优化的第一步，掌握 3 种选择方法让你在不同场景下游刃有余。
