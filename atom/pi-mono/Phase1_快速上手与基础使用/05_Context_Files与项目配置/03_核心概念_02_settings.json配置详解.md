# 核心概念：settings.json 配置详解

> 深入理解 settings.json 的配置选项、覆盖规则和最佳实践

---

## 一、settings.json 是什么？

**定义：** settings.json 是 Pi Agent 的配置中心，用于控制模型选择、UI 行为、资源加载等运行时配置。

**核心特点：**
- 📝 JSON 格式
- 🔄 支持全局和项目级配置
- 🎯 项目级覆盖全局级
- 🧩 嵌套对象深度合并
- 🔥 支持热重载（/reload）

**类比：** settings.json 就像 VS Code 的 settings.json，控制编辑器的行为和外观。

---

## 二、文件位置和优先级

### 2.1 两个层级

```bash
# 全局配置（所有项目生效）
~/.pi/agent/settings.json

# 项目配置（当前项目生效，覆盖全局）
/project/.pi/settings.json
```

### 2.2 优先级规则

**项目配置 > 全局配置**

```json
// 全局配置：~/.pi/agent/settings.json
{
  "defaultModel": "claude-sonnet-4",
  "theme": "dark"
}

// 项目配置：.pi/settings.json
{
  "defaultModel": "claude-opus-4"
}

// 最终生效
{
  "defaultModel": "claude-opus-4",  // 项目覆盖全局
  "theme": "dark"                    // 继承全局
}
```

---

## 三、配置合并规则

### 3.1 核心规则

**嵌套对象：深度合并（Merge）**
**数组：完全替换（Replace）**

### 3.2 对象合并示例

```json
// 全局配置
{
  "compaction": {
    "enabled": true,
    "reserveTokens": 50000,
    "keepRecentTokens": 10000
  }
}

// 项目配置
{
  "compaction": {
    "reserveTokens": 80000
  }
}

// 最终结果（对象深度合并）
{
  "compaction": {
    "enabled": true,           // 继承全局
    "reserveTokens": 80000,    // 项目覆盖
    "keepRecentTokens": 10000  // 继承全局
  }
}
```

### 3.3 数组替换示例

```json
// 全局配置
{
  "packages": ["@pi/core", "@pi/utils"]
}

// 项目配置
{
  "packages": ["@pi/custom"]
}

// 最终结果（数组完全替换）
{
  "packages": ["@pi/custom"]  // 全局的 @pi/core 和 @pi/utils 丢失
}
```

**注意：** 如果需要扩展数组，必须显式包含全局的值！

```json
// ✅ 正确做法：显式包含全局值
{
  "packages": [
    "@pi/core",      // 手动包含全局的
    "@pi/utils",     // 手动包含全局的
    "@pi/custom"     // 添加项目特定的
  ]
}
```

**来源：** 官方文档 settings.md 的 override rules

---

## 四、完整配置选项

### 4.1 模型与思考配置

```json
{
  // 默认模型
  "defaultModel": "claude-opus-4",
  // 可选值：
  // - "claude-opus-4" - 最强大的模型
  // - "claude-sonnet-4" - 平衡性能和成本
  // - "claude-haiku-4" - 快速响应

  // 默认 Provider
  "defaultProvider": "anthropic",
  // 可选值：
  // - "anthropic" - Anthropic 官方
  // - "openrouter" - OpenRouter 代理
  // - "custom" - 自定义 Provider

  // 思考级别
  "defaultThinkingLevel": "medium",
  // 可选值：
  // - "low" - 快速响应
  // - "medium" - 平衡（推荐）
  // - "high" - 深度思考
  // - "disabled" - 禁用思考模式
}
```

### 4.2 UI 与显示配置

```json
{
  // 主题
  "theme": "dark",
  // 可选值：
  // - "dark" - 深色主题
  // - "light" - 浅色主题
  // - "system" - 跟随系统

  // 安静启动（不显示欢迎信息）
  "quietStartup": true,

  // 折叠更新日志
  "collapseChangelog": true,

  // 显示 token 使用情况
  "showTokenUsage": true,

  // 显示思考过程
  "showThinking": false
}
```

### 4.3 对话压缩配置

```json
{
  "compaction": {
    // 启用自动压缩
    "enabled": true,

    // 保留的 token 数量（压缩阈值）
    "reserveTokens": 50000,
    // 当对话超过此值时触发压缩

    // 保留最近的 token 数量
    "keepRecentTokens": 10000,
    // 压缩时保留最近的对话内容

    // 压缩策略
    "strategy": "smart",
    // 可选值：
    // - "smart" - 智能压缩（保留重要内容）
    // - "simple" - 简单压缩（按时间顺序）
  }
}
```

**压缩机制说明：**
- 当对话 token 数超过 `reserveTokens` 时，自动触发压缩
- 保留最近 `keepRecentTokens` 的对话内容
- 其余内容使用 AI 总结压缩

### 4.4 资源路径配置

```json
{
  // 包路径（npm 包或本地路径）
  "packages": [
    "@pi/core",                    // npm 包
    "~/.pi/packages/custom"        // 本地路径
  ],

  // 扩展路径
  "extensions": [
    "~/.pi/extensions/custom",
    "./extensions/project-specific"
  ],

  // 技能路径
  "skills": [
    "~/.pi/skills/common",
    "./.pi/skills/project"
  ],

  // 提示词路径
  "prompts": [
    "~/.pi/prompts/templates"
  ],

  // 主题路径
  "themes": [
    "~/.pi/themes/custom"
  ]
}
```

**路径规则：**
- 支持绝对路径和相对路径
- `~` 表示用户主目录
- `./` 表示相对于 settings.json 的目录
- 支持 glob 模式（如 `~/.pi/skills/*`）

### 4.5 Shell 配置

```json
{
  // Shell 路径
  "shellPath": "/bin/zsh",
  // 默认使用系统 shell

  // Shell 命令前缀
  "shellCommandPrefix": "",
  // 在所有命令前添加的前缀

  // Shell 环境变量
  "shellEnv": {
    "NODE_ENV": "development",
    "API_BASE_URL": "https://api.example.com"
  }
}
```

### 4.6 重试配置

```json
{
  "retry": {
    // 启用自动重试
    "enabled": true,

    // 最大重试次数
    "maxRetries": 3,

    // 基础延迟（毫秒）
    "baseDelayMs": 1000,

    // 延迟倍数（指数退避）
    "delayMultiplier": 2
  }
}
```

**重试机制：**
- 第 1 次重试：延迟 1000ms
- 第 2 次重试：延迟 2000ms
- 第 3 次重试：延迟 4000ms

### 4.7 其他配置

```json
{
  // 自动保存对话历史
  "autoSave": true,

  // 对话历史保存路径
  "historyPath": "~/.pi/history",

  // 日志级别
  "logLevel": "info",
  // 可选值：debug, info, warn, error

  // 日志路径
  "logPath": "~/.pi/logs",

  // 启用遥测（匿名使用统计）
  "telemetry": false
}
```

---

## 五、常用配置场景

### 5.1 场景 1：切换模型

```json
// 全局使用 Sonnet（省钱）
// ~/.pi/agent/settings.json
{
  "defaultModel": "claude-sonnet-4"
}

// 重要项目使用 Opus（高质量）
// /important-project/.pi/settings.json
{
  "defaultModel": "claude-opus-4"
}
```

### 5.2 场景 2：个性化 UI

```json
// ~/.pi/agent/settings.json
{
  "theme": "dark",
  "quietStartup": true,
  "collapseChangelog": true,
  "showTokenUsage": true
}
```

### 5.3 场景 3：性能优化

```json
// 大型项目配置
// .pi/settings.json
{
  "compaction": {
    "enabled": true,
    "reserveTokens": 80000,    // 提高压缩阈值
    "keepRecentTokens": 20000  // 保留更多最近内容
  },
  "defaultThinkingLevel": "low"  // 快速响应
}
```

### 5.4 场景 4：加载自定义资源

```json
// .pi/settings.json
{
  "skills": [
    "~/.pi/skills/common",       // 全局技能
    "./.pi/skills/project"       // 项目特定技能
  ],
  "extensions": [
    "./extensions/custom-linter"  // 项目特定扩展
  ]
}
```

### 5.5 场景 5：团队协作

```json
// 不提交到 Git（个人配置）
// .pi/settings.json
{
  "defaultModel": "claude-opus-4",  // 个人偏好
  "theme": "dark"                   // 个人偏好
}

// .gitignore
.pi/settings.json
```

---

## 六、配置模板

### 6.1 最小配置

```json
{
  "defaultModel": "claude-opus-4"
}
```

### 6.2 推荐配置（个人）

```json
{
  "defaultModel": "claude-sonnet-4",
  "theme": "dark",
  "quietStartup": true,
  "collapseChangelog": true,
  "compaction": {
    "enabled": true,
    "reserveTokens": 50000
  }
}
```

### 6.3 推荐配置（团队项目）

```json
{
  "compaction": {
    "enabled": true,
    "reserveTokens": 80000,
    "keepRecentTokens": 20000
  },
  "skills": [
    "./.pi/skills/project"
  ]
}
```

### 6.4 完整配置示例

```json
{
  // 模型配置
  "defaultModel": "claude-opus-4",
  "defaultProvider": "anthropic",
  "defaultThinkingLevel": "medium",

  // UI 配置
  "theme": "dark",
  "quietStartup": true,
  "collapseChangelog": true,
  "showTokenUsage": true,

  // 压缩配置
  "compaction": {
    "enabled": true,
    "reserveTokens": 50000,
    "keepRecentTokens": 10000,
    "strategy": "smart"
  },

  // 资源配置
  "packages": ["@pi/core"],
  "skills": ["~/.pi/skills/common"],
  "extensions": ["~/.pi/extensions/custom"],

  // Shell 配置
  "shellPath": "/bin/zsh",
  "shellEnv": {
    "NODE_ENV": "development"
  },

  // 重试配置
  "retry": {
    "enabled": true,
    "maxRetries": 3,
    "baseDelayMs": 1000
  },

  // 其他配置
  "autoSave": true,
  "logLevel": "info",
  "telemetry": false
}
```

---

## 七、高级技巧

### 7.1 使用环境变量

虽然 settings.json 本身不支持环境变量，但可以通过 `shellEnv` 传递：

```json
{
  "shellEnv": {
    "OPENAI_API_KEY": "sk-...",
    "API_BASE_URL": "https://api.example.com"
  }
}
```

**注意：** 敏感信息应该使用 .env 文件，不要写在 settings.json 中！

### 7.2 使用相对路径

```json
{
  // 相对于 settings.json 的路径
  "skills": [
    "./.pi/skills/project",      // 项目根目录的 .pi/skills/project
    "../shared/skills"           // 父目录的 shared/skills
  ]
}
```

### 7.3 使用 Glob 模式

```json
{
  "skills": [
    "~/.pi/skills/*",            // 加载所有子目录
    "./.pi/skills/**/*.js"       // 加载所有 JS 文件
  ]
}
```

### 7.4 条件配置（通过多个文件）

```bash
# 开发环境
.pi/settings.dev.json

# 生产环境
.pi/settings.prod.json

# 使用时手动切换
cp .pi/settings.dev.json .pi/settings.json
```

---

## 八、常见问题

### Q1: settings.json 的配置会立即生效吗？

**A:** 不会。需要执行 `/reload` 命令或重启 Pi。

```bash
# 修改 settings.json 后
/reload
```

### Q2: 如何查看当前生效的配置？

**A:** Pi 启动时会显示加载的配置文件：

```bash
$ pi

Loaded settings:
- ~/.pi/agent/settings.json
- /project/.pi/settings.json

Ready to assist!
```

### Q3: 为什么我的数组配置没有生效？

**A:** 数组是完全替换，不是合并。需要显式包含全局的值：

```json
// ❌ 错误：丢失全局的 packages
{
  "packages": ["@pi/custom"]
}

// ✅ 正确：显式包含全局的 packages
{
  "packages": [
    "@pi/core",      // 全局的
    "@pi/utils",     // 全局的
    "@pi/custom"     // 项目的
  ]
}
```

### Q4: settings.json 可以有注释吗？

**A:** 标准 JSON 不支持注释，但 Pi Agent 支持 JSON5 格式（带注释）：

```json5
{
  // 这是注释
  "defaultModel": "claude-opus-4",  // 行尾注释
  /* 多行注释
     也支持 */
  "theme": "dark"
}
```

### Q5: 如何重置配置？

**A:** 删除配置文件即可恢复默认：

```bash
# 删除项目配置
rm .pi/settings.json

# 删除全局配置
rm ~/.pi/agent/settings.json
```

---

## 九、配置验证

### 9.1 检查 JSON 语法

```bash
# 使用 jq 验证 JSON 语法
cat .pi/settings.json | jq .

# 如果有语法错误，jq 会报错
```

### 9.2 检查配置是否生效

```bash
# 1. 修改配置
echo '{"defaultModel": "claude-opus-4"}' > .pi/settings.json

# 2. 重新加载
pi
/reload

# 3. 测试：问 Pi "你使用的是什么模型？"
# Pi 应该回答 "claude-opus-4"
```

### 9.3 调试配置问题

```bash
# 查看配置文件
cat ~/.pi/agent/settings.json
cat .pi/settings.json

# 检查文件权限
ls -la .pi/settings.json

# 查看 Pi 日志
tail -f ~/.pi/logs/pi.log
```

---

## 十、最佳实践

### 10.1 全局 vs 项目配置

**全局配置（~/.pi/agent/settings.json）：**
- ✅ 个人偏好（主题、模型）
- ✅ 通用资源（全局 skills、extensions）
- ✅ UI 设置（quietStartup、collapseChangelog）

**项目配置（.pi/settings.json）：**
- ✅ 项目特定模型（重要项目用 Opus）
- ✅ 项目特定资源（项目 skills、extensions）
- ✅ 性能优化（compaction 设置）

### 10.2 Git 管理策略

```bash
# .gitignore
.pi/settings.json          # 个人配置不提交

# 可选：提供配置模板
.pi/settings.example.json  # 提交模板供团队参考
```

**settings.example.json 示例：**
```json
{
  "defaultModel": "claude-sonnet-4",
  "compaction": {
    "enabled": true,
    "reserveTokens": 50000
  }
}
```

### 10.3 敏感信息管理

```json
// ❌ 不要在 settings.json 中存储敏感信息
{
  "shellEnv": {
    "API_KEY": "sk-1234567890"  // 危险！
  }
}

// ✅ 使用 .env 文件
// .env
API_KEY=sk-1234567890

// settings.json
{
  "shellEnv": {
    "API_KEY": "${API_KEY}"  // 引用环境变量
  }
}
```

### 10.4 配置分层

```
全局配置（个人偏好）
↓
项目配置（项目特定）
↓
最终生效配置
```

**示例：**
```json
// 全局：~/.pi/agent/settings.json
{
  "defaultModel": "claude-sonnet-4",
  "theme": "dark",
  "compaction": {
    "enabled": true,
    "reserveTokens": 50000
  }
}

// 项目：.pi/settings.json
{
  "defaultModel": "claude-opus-4",  // 覆盖全局
  "compaction": {
    "reserveTokens": 80000          // 覆盖全局，但继承 enabled
  }
}

// 最终生效
{
  "defaultModel": "claude-opus-4",
  "theme": "dark",
  "compaction": {
    "enabled": true,
    "reserveTokens": 80000
  }
}
```

---

## 十一、配置速查表

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `defaultModel` | string | `"claude-sonnet-4"` | 默认模型 |
| `defaultProvider` | string | `"anthropic"` | 默认 Provider |
| `defaultThinkingLevel` | string | `"medium"` | 思考级别 |
| `theme` | string | `"system"` | 主题 |
| `quietStartup` | boolean | `false` | 安静启动 |
| `collapseChangelog` | boolean | `false` | 折叠更新日志 |
| `compaction.enabled` | boolean | `true` | 启用压缩 |
| `compaction.reserveTokens` | number | `50000` | 压缩阈值 |
| `compaction.keepRecentTokens` | number | `10000` | 保留最近 token |
| `packages` | array | `[]` | 包路径 |
| `skills` | array | `[]` | 技能路径 |
| `extensions` | array | `[]` | 扩展路径 |
| `shellPath` | string | 系统 shell | Shell 路径 |
| `retry.enabled` | boolean | `true` | 启用重试 |
| `retry.maxRetries` | number | `3` | 最大重试次数 |
| `autoSave` | boolean | `true` | 自动保存历史 |
| `logLevel` | string | `"info"` | 日志级别 |

---

## 十二、总结

**settings.json 的核心要点：**

1. **文件位置** - 全局（~/.pi/agent/）和项目级（.pi/）
2. **合并规则** - 对象深度合并，数组完全替换
3. **配置选项** - 模型、UI、压缩、资源、Shell、重试
4. **最佳实践** - 全局个人偏好，项目特定配置，敏感信息用 .env
5. **热重载** - 使用 /reload 快速更新配置

**记住：** settings.json 是 Pi Agent 的"控制面板"，合理配置能显著提升使用体验！

**参考资源：**
- 官方文档：https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/settings.md
- 配置示例：https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/settings.example.json
