# 社区工具更新日志

> 追踪 Codex CLI 社区工具的发布、更新和最佳实践

---

## 工具索引

| 工具名称 | 类型 | 主要功能 | 维护者 | 状态 |
|---------|------|---------|--------|------|
| TSK | 并行化 | Agent sandbox + 多 worktree 并行 | 社区 | ✅ 活跃 |
| Emdash 2.0 | 并行化 | 多 worktree 并行运行 | 社区 | ✅ 活跃 |
| ralph CLI | 自动化 | PRD 生成 + Ralph Loop | Ian Nuttall | ✅ 活跃 |
| codex-settings | 配置 | 配置集合 + 提示模板 | feiskyer | ✅ 活跃 |
| codex-monitor | 监控 | API 调用监控 + 配额管理 | 社区 | 🔄 开发中 |

---

## TSK - Agent Sandbox 与并行化工具

### 概述
TSK 是开源的 agent sandbox 工具，提供多 worktree 并行执行能力，避免文件冲突。

### 核心功能
- ✅ 自动创建和管理 Git Worktrees
- ✅ 并行运行多个 Codex 实例
- ✅ 任务隔离与结果合并
- ✅ 资源监控与配额管理

### 安装
```bash
git clone https://github.com/community/tsk.git
cd tsk
npm install -g .
```

### 基础使用
```bash
# 初始化 TSK 环境
tsk init

# 创建任务
tsk create task1 "Refactor authentication module"
tsk create task2 "Add new API endpoints"

# 并行运行（最多3个worker）
tsk run --parallel --max-workers 3

# 查看任务状态
tsk status

# 合并结果
tsk merge --auto
```

### 高级配置
```yaml
# tsk.config.yml
worktrees:
  max_count: 6
  base_dir: ../tsk-worktrees
  auto_cleanup: true

parallel:
  max_workers: 3
  spawn_delay_ms: 1000

codex:
  model: gpt-5.2-codex
  temperature: 0.7
```

### 适用场景
- 大型重构项目（多模块并行）
- 独立功能开发（无文件冲突）
- 测试套件优化（分模块执行）

### 更新历史
- **v1.3** (2026-02): 增加资源监控
- **v1.2** (2026-01): 优化 worktree 管理
- **v1.1** (2025-12): 支持自动合并
- **v1.0** (2025-09): 首次发布

### 相关资源
- GitHub: https://github.com/community/tsk
- 文档: https://tsk.dev
- Reddit: r/CodexAutomation

---

## Emdash 2.0 - 多 Worktree 并行运行

### 概述
Emdash 2.0 专注于多 worktree 并行执行，提供简洁的 CLI 和配置文件支持。

### 核心功能
- ✅ 声明式任务配置（JSON/YAML）
- ✅ 自动 worktree 创建与清理
- ✅ 实时进度监控
- ✅ 失败重试机制

### 安装
```bash
npm install -g emdash-cli
```

### 基础使用
```bash
# 配置 Emdash
emdash config --worktrees 4

# 从配置文件运行
emdash run --tasks tasks.json

# 实时监控
emdash watch
```

### 任务配置示例
```json
{
  "tasks": [
    {
      "id": "auth-refactor",
      "description": "Refactor authentication module",
      "branch": "feature/auth-refactor",
      "priority": "high",
      "dependencies": []
    },
    {
      "id": "api-endpoints",
      "description": "Add new API endpoints",
      "branch": "feature/new-api",
      "priority": "medium",
      "dependencies": ["auth-refactor"]
    }
  ],
  "config": {
    "max_parallel": 3,
    "retry_on_failure": true,
    "auto_merge": false
  }
}
```

### 高级特性
```bash
# 依赖管理（任务按依赖顺序执行）
emdash run --respect-dependencies

# 失败重试
emdash run --retry 3

# 自动合并到主分支
emdash run --auto-merge
```

### 适用场景
- 复杂任务依赖管理
- 需要重试机制的任务
- 团队协作（共享任务配置）

### 更新历史
- **v2.1** (2026-02): 增加依赖管理
- **v2.0** (2025-07): 重大重构，支持配置文件
- **v1.x** (2025-05): 早期版本

### 相关资源
- GitHub: https://github.com/community/emdash
- 文档: https://emdash.dev
- Twitter: @emdash_cli

---

## ralph CLI - PRD 生成与 Ralph Loop

### 概述
ralph CLI 由 Ian Nuttall 开发，专注于自动化 PRD 生成和 Ralph Loop 执行。

### 核心功能
- ✅ 自动 PRD 生成（从需求文档）
- ✅ PRD 完成度验证
- ✅ Ralph Loop 循环执行
- ✅ 任务进度追踪

### 安装
```bash
npm install -g ralph-cli
```

### PRD 生成
```bash
# 从需求文档生成 PRD
ralph generate-prd --input requirements.md --output prd.md

# 生成的 PRD 包含：
# - 任务清单（带编号）
# - 验收标准
# - 依赖关系
# - 预估复杂度
```

### PRD 验证
```bash
# 验证 PRD 完成度
ralph verify-prd --prd prd.md --check-files

# 输出示例：
# ✅ Task 1: Completed (3 files changed)
# ✅ Task 2: Completed (5 files changed)
# ⏳ Task 3: In Progress (2/4 subtasks)
# ❌ Task 4: Not Started
```

### Ralph Loop 执行
```bash
# 启动 Ralph Loop
ralph loop --prd prd.md --max-iterations 10

# 每次循环：
# 1. 运行 Codex 会话
# 2. 验证 PRD 完成度
# 3. 如果未完成，继续下一轮
# 4. 如果完成，退出循环
```

### 配置示例
```yaml
# ralph.config.yml
prd:
  path: ./prd.md
  auto_verify: true
  verify_interval: 5m

loop:
  max_iterations: 20
  session_timeout: 30m
  stop_on_error: false

codex:
  model: gpt-5.2-codex
  temperature: 0.7
  prompt_template: |
    You have a PRD with tasks. Don't stop until all tasks are completed.
    Current progress: {progress}
    Remaining tasks: {remaining_tasks}
```

### 适用场景
- 20+ 小时长时间任务
- 大型重构项目
- 需要严格验收的项目
- 多阶段开发任务

### 更新历史
- **v1.2** (2026-02): 增加自动 PRD 生成
- **v1.1** (2025-12): 优化验证逻辑
- **v1.0** (2025-08): 首次发布

### 相关资源
- GitHub: https://github.com/iannuttall/ralph-cli
- 文档: https://ralph-cli.dev
- Twitter: @iannuttall

---

## codex-settings - 配置集合与提示模板

### 概述
feiskyer 维护的 Codex CLI 配置集合和提示模板库。

### 核心功能
- ✅ 预配置的 config.json 模板
- ✅ 常用提示模板库
- ✅ AGENTS.md 示例
- ✅ SKILL.md 示例

### 安装
```bash
git clone https://github.com/feiskyer/codex-settings.git
cd codex-settings
```

### 配置模板
```bash
# 复制基础配置
cp templates/config.basic.json ~/.config/codex/config.json

# 复制 Multi-Agent 配置
cp templates/config.multi-agent.json ~/.config/codex/config.json

# 复制长时间任务配置
cp templates/config.long-running.json ~/.config/codex/config.json
```

### 提示模板库
```bash
# 查看可用模板
ls prompts/

# 使用模板
cat prompts/ralph-loop.txt | codex
cat prompts/multi-agent-refactor.txt | codex
```

### 常用模板
- `ralph-loop.txt`: Ralph Loop 提示模板
- `multi-agent-refactor.txt`: Multi-Agent 重构模板
- `long-running-task.txt`: 长时间任务模板
- `test-optimization.txt`: 测试优化模板

### 适用场景
- 快速配置 Codex CLI
- 学习最佳实践
- 团队统一配置

### 更新历史
- **v1.5** (2026-02): 增加 Ralph Loop 模板
- **v1.4** (2026-01): 增加 Multi-Agent 配置
- **v1.3** (2025-11): 增加提示模板库
- **v1.0** (2025-06): 首次发布

### 相关资源
- GitHub: https://github.com/feiskyer/codex-settings
- Reddit: r/codex

---

## codex-monitor - API 调用监控（开发中）

### 概述
社区开发的 API 调用监控和配额管理工具（目前处于开发阶段）。

### 计划功能
- 🔄 实时 API 调用监控
- 🔄 配额使用统计
- 🔄 成本估算
- 🔄 告警通知

### 预期使用
```bash
# 安装（开发版）
npm install -g codex-monitor@beta

# 启动监控
codex-monitor --watch

# 查看统计
codex-monitor stats --today
```

### 开发状态
- **当前版本**: v0.3-beta
- **预计正式发布**: 2026 Q2
- **GitHub**: https://github.com/community/codex-monitor

---

## 工具对比

| 特性 | TSK | Emdash 2.0 | ralph CLI |
|------|-----|-----------|-----------|
| 并行执行 | ✅ | ✅ | ❌ |
| PRD 管理 | ❌ | ❌ | ✅ |
| 依赖管理 | ⚠️ 基础 | ✅ 完整 | ✅ 完整 |
| 配置文件 | ✅ YAML | ✅ JSON/YAML | ✅ YAML |
| 自动合并 | ✅ | ⚠️ 可选 | ❌ |
| 失败重试 | ⚠️ 手动 | ✅ 自动 | ✅ 自动 |
| 学习曲线 | 中等 | 简单 | 简单 |

---

## 推荐组合

### 组合1：大型重构项目
```bash
# 使用 TSK 并行执行 + ralph CLI 验证
tsk init
ralph generate-prd --input requirements.md
tsk run --parallel --max-workers 3
ralph verify-prd --prd prd.md
```

### 组合2：长时间任务
```bash
# 使用 ralph CLI + codex-settings 模板
ralph generate-prd --input requirements.md
cat ~/.codex-settings/prompts/ralph-loop.txt | codex
ralph loop --prd prd.md --max-iterations 20
```

### 组合3：团队协作
```bash
# 使用 Emdash 2.0 + codex-settings 配置
cp ~/.codex-settings/templates/config.multi-agent.json ~/.config/codex/
emdash run --tasks team-tasks.json --auto-merge
```

---

## 贡献新工具

如果你开发了 Codex CLI 相关工具，欢迎提交到本列表：

1. 在 GitHub 创建 Issue
2. 提供工具信息：
   - 名称、功能、安装方式
   - 使用示例
   - 适用场景
3. 等待社区审核

---

**返回：** [CHANGELOG 主页](../README.md)
