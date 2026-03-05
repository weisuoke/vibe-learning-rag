# 实战代码 场景4：多 Agent 环境搭建

**本文档演示如何搭建多 Agent 环境，包括 Agent 创建、Workspace 管理和路由配置。**

---

## 场景概述

**目标**：搭建多 Agent 环境，实现不同场景使用不同 Agent，提高灵活性和隔离性。

**适用场景**：
- 工作与个人分离（work agent、personal agent）
- 不同项目使用不同 Agent（project-a agent、project-b agent）
- 不同模型测试（gpt-4 agent、claude agent）
- 多用户环境（user1 agent、user2 agent）

**核心优势**：
- **隔离性**：每个 Agent 有独立的 Workspace 和配置
- **灵活性**：不同 Agent 可使用不同模型和工具
- **可扩展性**：轻松添加新 Agent
- **安全性**：敏感数据隔离

---

## Agent 基础概念

### Agent 是什么？

Agent 是 OpenClaw 的核心概念，代表一个独立的 AI 助手实例，包含：

- **Workspace**：Agent 的工作目录
- **Bootstrap 文件**：Agent 的初始化指令
- **模型配置**：使用的 LLM 模型
- **工具配置**：可用的工具和 Skills
- **会话管理**：对话历史和上下文

### Agent 层级结构

```
OpenClaw
    ↓
Gateway（网关）
    ↓
┌─────────────────────────────────────┐
│  Agent 1 (main)                     │
│  - Workspace: ~/.openclaw/workspace │
│  - Model: claude-sonnet-4-5         │
│  - Bootstrap: bootstrap.md          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Agent 2 (work)                     │
│  - Workspace: ~/.openclaw/work      │
│  - Model: gpt-4                     │
│  - Bootstrap: work-bootstrap.md     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Agent 3 (personal)                 │
│  - Workspace: ~/.openclaw/personal  │
│  - Model: claude-opus-4-6           │
│  - Bootstrap: personal-bootstrap.md │
└─────────────────────────────────────┘
```

---

## 创建 Agent

### 方式 1：使用 CLI 创建

```bash
# 创建新 Agent
openclaw agents add work

# 指定 Workspace
openclaw agents add work --workspace ~/.openclaw/work

# 指定模型
openclaw agents add work \
  --workspace ~/.openclaw/work \
  --model anthropic/claude-sonnet-4-5
```

### 方式 2：交互式创建

```bash
# 运行交互式创建
openclaw agents add work

# 向导会提示：
# ◆  Agent name: work
# ◆  Workspace directory: ~/.openclaw/work
# ◆  Model: anthropic/claude-sonnet-4-5
# ◆  Bootstrap file: (optional)
```

### 方式 3：非交互式创建

```bash
#!/bin/bash
# create-agent.sh

set -e

AGENT_NAME="${1:-work}"
WORKSPACE="${2:-$HOME/.openclaw/$AGENT_NAME}"
MODEL="${3:-anthropic/claude-sonnet-4-5}"

openclaw agents add "$AGENT_NAME" \
  --workspace "$WORKSPACE" \
  --model "$MODEL" \
  --non-interactive

echo "Agent '$AGENT_NAME' created successfully!"
```

**使用方法**：

```bash
# 创建 work agent
./create-agent.sh work ~/.openclaw/work anthropic/claude-sonnet-4-5

# 创建 personal agent
./create-agent.sh personal ~/.openclaw/personal anthropic/claude-opus-4-6
```

---

## 完整多 Agent 配置示例

### 示例 1：工作与个人分离

```bash
#!/bin/bash
# setup-work-personal-agents.sh

set -e

echo "Setting up work and personal agents..."

# 1. 创建 work agent
openclaw agents add work \
  --workspace ~/.openclaw/work \
  --model anthropic/claude-sonnet-4-5

# 2. 创建 personal agent
openclaw agents add personal \
  --workspace ~/.openclaw/personal \
  --model anthropic/claude-opus-4-6

# 3. 配置 work agent Bootstrap
cat > ~/.openclaw/work/.openclaw/bootstrap.md <<EOF
# Work Agent

You are a professional work assistant. Focus on:
- Code development and debugging
- Technical documentation
- Project management
- Professional communication

Workspace: ~/.openclaw/work
EOF

# 4. 配置 personal agent Bootstrap
cat > ~/.openclaw/personal/.openclaw/bootstrap.md <<EOF
# Personal Agent

You are a personal assistant. Focus on:
- Personal task management
- Creative writing
- Learning and research
- Casual conversation

Workspace: ~/.openclaw/personal
EOF

# 5. 验证
openclaw agents list

echo "Work and personal agents configured successfully!"
```

### 示例 2：多项目环境

```bash
#!/bin/bash
# setup-multi-project-agents.sh

set -e

# 项目列表
PROJECTS=(
  "project-a"
  "project-b"
  "project-c"
)

# 为每个项目创建 Agent
for project in "${PROJECTS[@]}"; do
  echo "Creating agent for $project..."

  # 创建 Agent
  openclaw agents add "$project" \
    --workspace "$HOME/.openclaw/$project" \
    --model anthropic/claude-sonnet-4-5

  # 创建 Bootstrap 文件
  cat > "$HOME/.openclaw/$project/.openclaw/bootstrap.md" <<EOF
# $project Agent

Project: $project
Workspace: ~/.openclaw/$project

## Project Context
- Project name: $project
- Focus: [Add project-specific context]
- Tech stack: [Add tech stack]

## Guidelines
- Follow project coding standards
- Use project-specific tools
- Maintain project documentation
EOF

  echo "Agent for $project created successfully!"
done

# 验证
openclaw agents list
```

### 示例 3：多模型测试环境

```bash
#!/bin/bash
# setup-multi-model-agents.sh

set -e

# 模型配置
declare -A MODELS=(
  ["claude-sonnet"]="anthropic/claude-sonnet-4-5"
  ["claude-opus"]="anthropic/claude-opus-4-6"
  ["gpt-4"]="openai/gpt-4"
  ["gpt-4-turbo"]="openai/gpt-4-turbo"
)

# 为每个模型创建 Agent
for agent_name in "${!MODELS[@]}"; do
  model="${MODELS[$agent_name]}"
  echo "Creating agent for $model..."

  openclaw agents add "$agent_name" \
    --workspace "$HOME/.openclaw/$agent_name" \
    --model "$model"

  echo "Agent '$agent_name' with model '$model' created!"
done

# 验证
openclaw agents list
```

---

## Agent 管理

### 列出所有 Agent

```bash
# 列出所有 Agent
openclaw agents list

# 输出示例：
# Agents:
# - main (default)
#   Workspace: ~/.openclaw/workspace
#   Model: anthropic/claude-sonnet-4-5
# - work
#   Workspace: ~/.openclaw/work
#   Model: anthropic/claude-sonnet-4-5
# - personal
#   Workspace: ~/.openclaw/personal
#   Model: anthropic/claude-opus-4-6
```

### 切换 Agent

```bash
# 切换到 work agent
openclaw agents switch work

# 切换到 personal agent
openclaw agents switch personal

# 切换回默认 agent
openclaw agents switch main
```

### 查看当前 Agent

```bash
# 查看当前 Agent
openclaw agents current

# 输出示例：
# Current agent: work
# Workspace: ~/.openclaw/work
# Model: anthropic/claude-sonnet-4-5
```

### 删除 Agent

```bash
# 删除 Agent
openclaw agents remove work

# 删除 Agent 并删除 Workspace
openclaw agents remove work --delete-workspace
```

---

## Agent 路由配置

### 配置文件格式

```json5
// ~/.openclaw/openclaw.json
{
  agents: {
    defaults: {
      workspace: "~/.openclaw/workspace",
      model: {
        primary: "anthropic/claude-sonnet-4-5",
      },
    },
    work: {
      workspace: "~/.openclaw/work",
      model: {
        primary: "anthropic/claude-sonnet-4-5",
      },
      bootstrap: "~/.openclaw/work/.openclaw/bootstrap.md",
    },
    personal: {
      workspace: "~/.openclaw/personal",
      model: {
        primary: "anthropic/claude-opus-4-6",
      },
      bootstrap: "~/.openclaw/personal/.openclaw/bootstrap.md",
    },
  },
  routing: {
    rules: [
      {
        channel: "whatsapp",
        from: "+1234567890",
        agent: "work",
      },
      {
        channel: "telegram",
        from: "tg:123456789",
        agent: "personal",
      },
    ],
  },
}
```

### 路由规则

**按 Channel 路由**：

```json5
{
  routing: {
    rules: [
      {
        channel: "whatsapp",
        agent: "work",  // 所有 WhatsApp 消息路由到 work agent
      },
      {
        channel: "telegram",
        agent: "personal",  // 所有 Telegram 消息路由到 personal agent
      },
    ],
  },
}
```

**按用户路由**：

```json5
{
  routing: {
    rules: [
      {
        channel: "whatsapp",
        from: "+1234567890",
        agent: "work",  // 特定用户路由到 work agent
      },
      {
        channel: "whatsapp",
        from: "+9876543210",
        agent: "personal",  // 另一个用户路由到 personal agent
      },
    ],
  },
}
```

**按关键词路由**：

```json5
{
  routing: {
    rules: [
      {
        channel: "*",
        keywords: ["work", "project", "code"],
        agent: "work",  // 包含关键词的消息路由到 work agent
      },
      {
        channel: "*",
        keywords: ["personal", "hobby", "fun"],
        agent: "personal",  // 包含关键词的消息路由到 personal agent
      },
    ],
  },
}
```

---

## Workspace 管理

### Workspace 目录结构

```
~/.openclaw/
├── workspace/                      # main agent workspace
│   ├── .openclaw/
│   │   └── bootstrap.md
│   ├── projects/
│   └── files/
├── work/                           # work agent workspace
│   ├── .openclaw/
│   │   └── bootstrap.md
│   ├── code/
│   └── docs/
└── personal/                       # personal agent workspace
    ├── .openclaw/
    │   └── bootstrap.md
    ├── notes/
    └── ideas/
```

### 初始化 Workspace

```bash
#!/bin/bash
# init-workspace.sh

set -e

AGENT_NAME="${1:-work}"
WORKSPACE="${2:-$HOME/.openclaw/$AGENT_NAME}"

# 创建 Workspace 目录
mkdir -p "$WORKSPACE/.openclaw"
mkdir -p "$WORKSPACE/projects"
mkdir -p "$WORKSPACE/files"

# 创建 Bootstrap 文件
cat > "$WORKSPACE/.openclaw/bootstrap.md" <<EOF
# $AGENT_NAME Agent

Workspace: $WORKSPACE

## Context
- Agent: $AGENT_NAME
- Purpose: [Add purpose]

## Guidelines
- [Add guidelines]
EOF

echo "Workspace initialized at $WORKSPACE"
```

### 备份 Workspace

```bash
#!/bin/bash
# backup-workspace.sh

set -e

AGENT_NAME="${1:-work}"
WORKSPACE="$HOME/.openclaw/$AGENT_NAME"
BACKUP_DIR="$HOME/backups/openclaw"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 创建备份目录
mkdir -p "$BACKUP_DIR"

# 备份 Workspace
tar -czf "$BACKUP_DIR/${AGENT_NAME}_${TIMESTAMP}.tar.gz" -C "$HOME/.openclaw" "$AGENT_NAME"

echo "Workspace backed up to $BACKUP_DIR/${AGENT_NAME}_${TIMESTAMP}.tar.gz"
```

### 恢复 Workspace

```bash
#!/bin/bash
# restore-workspace.sh

set -e

BACKUP_FILE="${1}"
AGENT_NAME="${2:-work}"

if [ ! -f "$BACKUP_FILE" ]; then
  echo "Error: Backup file not found: $BACKUP_FILE"
  exit 1
fi

# 恢复 Workspace
tar -xzf "$BACKUP_FILE" -C "$HOME/.openclaw"

echo "Workspace restored from $BACKUP_FILE"
```

---

## 高级配置

### 配置 Agent 特定工具

```json5
{
  agents: {
    work: {
      workspace: "~/.openclaw/work",
      model: {
        primary: "anthropic/claude-sonnet-4-5",
      },
      tools: {
        bash: {
          enabled: true,
          requireApproval: false,  // work agent 不需要批准
        },
        filesystem: {
          enabled: true,
          allowedPaths: ["~/.openclaw/work", "~/projects"],
        },
      },
    },
    personal: {
      workspace: "~/.openclaw/personal",
      model: {
        primary: "anthropic/claude-opus-4-6",
      },
      tools: {
        bash: {
          enabled: true,
          requireApproval: true,  // personal agent 需要批准
        },
        filesystem: {
          enabled: true,
          allowedPaths: ["~/.openclaw/personal"],
        },
      },
    },
  },
}
```

### 配置 Agent 特定 Skills

```bash
# 为 work agent 安装 Skills
openclaw agents switch work
openclaw skills install code-review github

# 为 personal agent 安装 Skills
openclaw agents switch personal
openclaw skills install web-search notion
```

### 配置 Agent 环境变量

```json5
{
  agents: {
    work: {
      workspace: "~/.openclaw/work",
      env: {
        WORK_API_KEY: "${WORK_API_KEY}",
        PROJECT_NAME: "work-project",
      },
    },
    personal: {
      workspace: "~/.openclaw/personal",
      env: {
        PERSONAL_API_KEY: "${PERSONAL_API_KEY}",
        HOBBY: "coding",
      },
    },
  },
}
```

---

## 故障排查

### 问题 1：Agent 创建失败

**症状**：

```
Error: Failed to create agent 'work'
```

**排查步骤**：

```bash
# 1. 检查 Workspace 是否已存在
ls -la ~/.openclaw/work

# 2. 检查权限
ls -ld ~/.openclaw

# 3. 手动创建 Workspace
mkdir -p ~/.openclaw/work/.openclaw

# 4. 重新创建 Agent
openclaw agents add work --workspace ~/.openclaw/work
```

### 问题 2：Agent 切换失败

**症状**：

```
Error: Agent 'work' not found
```

**解决方案**：

```bash
# 1. 列出所有 Agent
openclaw agents list

# 2. 检查配置文件
cat ~/.openclaw/openclaw.json | grep -A 5 "agents"

# 3. 重新创建 Agent
openclaw agents add work --workspace ~/.openclaw/work
```

### 问题 3：路由规则不生效

**症状**：

消息没有路由到正确的 Agent

**解决方案**：

```bash
# 1. 检查路由配置
openclaw config get routing.rules

# 2. 验证规则格式
cat ~/.openclaw/openclaw.json | jq '.routing'

# 3. 重启 Gateway
openclaw gateway restart

# 4. 测试路由
openclaw test-routing --channel whatsapp --from "+1234567890"
```

### 问题 4：Workspace 权限问题

**症状**：

```
Error: Permission denied: ~/.openclaw/work
```

**解决方案**：

```bash
# 1. 检查权限
ls -ld ~/.openclaw/work

# 2. 修复权限
chmod -R 755 ~/.openclaw/work

# 3. 检查所有者
ls -l ~/.openclaw/work

# 4. 修复所有者
chown -R $USER ~/.openclaw/work
```

---

## 最佳实践

### 1. 命名规范

```bash
# ✅ 推荐：使用描述性名称
openclaw agents add work
openclaw agents add personal
openclaw agents add project-a

# ❌ 不推荐：使用模糊名称
openclaw agents add agent1
openclaw agents add test
openclaw agents add temp
```

### 2. Workspace 组织

```
~/.openclaw/
├── work/
│   ├── .openclaw/
│   │   └── bootstrap.md
│   ├── projects/
│   │   ├── project-a/
│   │   └── project-b/
│   └── docs/
└── personal/
    ├── .openclaw/
    │   └── bootstrap.md
    ├── notes/
    └── ideas/
```

### 3. Bootstrap 文件模板

```markdown
# {Agent Name} Agent

## Context
- Agent: {agent_name}
- Purpose: {purpose}
- Workspace: {workspace_path}

## Guidelines
- {guideline_1}
- {guideline_2}
- {guideline_3}

## Tools
- {tool_1}
- {tool_2}

## Skills
- {skill_1}
- {skill_2}

## Notes
- {note_1}
- {note_2}
```

### 4. 定期备份

```bash
#!/bin/bash
# backup-all-agents.sh

set -e

BACKUP_DIR="$HOME/backups/openclaw"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 获取所有 Agent
AGENTS=$(openclaw agents list --json | jq -r '.[].name')

# 备份每个 Agent
for agent in $AGENTS; do
  echo "Backing up agent: $agent"
  tar -czf "$BACKUP_DIR/${agent}_${TIMESTAMP}.tar.gz" \
    -C "$HOME/.openclaw" "$agent"
done

echo "All agents backed up to $BACKUP_DIR"
```

---

## 完整部署示例

### 生产环境多 Agent 部署

```bash
#!/bin/bash
# production-multi-agent-deploy.sh

set -e

echo "Deploying multi-agent environment..."

# 1. 配置 main agent
openclaw onboard --non-interactive \
  --auth-choice anthropic-api-key \
  --anthropic-api-key "$ANTHROPIC_API_KEY" \
  --workspace "$HOME/.openclaw/workspace"

# 2. 创建 work agent
openclaw agents add work \
  --workspace "$HOME/.openclaw/work" \
  --model anthropic/claude-sonnet-4-5

# 3. 创建 personal agent
openclaw agents add personal \
  --workspace "$HOME/.openclaw/personal" \
  --model anthropic/claude-opus-4-6

# 4. 配置路由规则
cat > /tmp/routing-config.json <<EOF
{
  "routing": {
    "rules": [
      {
        "channel": "whatsapp",
        "from": "+1234567890",
        "agent": "work"
      },
      {
        "channel": "telegram",
        "from": "tg:123456789",
        "agent": "personal"
      }
    ]
  }
}
EOF

# 5. 应用路由配置
openclaw config merge /tmp/routing-config.json

# 6. 重启 Gateway
openclaw gateway restart

# 7. 验证
openclaw agents list
openclaw config get routing.rules

echo "Multi-agent environment deployed successfully!"
```

---

## 总结

多 Agent 环境搭建的核心要点：

1. **Agent 创建**：使用 `openclaw agents add` 创建新 Agent
2. **Workspace 管理**：每个 Agent 有独立的 Workspace
3. **路由配置**：配置路由规则将消息路由到正确的 Agent
4. **工具隔离**：每个 Agent 可配置不同的工具和 Skills
5. **备份恢复**：定期备份 Workspace 和配置

完成这些步骤后，你就可以搭建灵活的多 Agent 环境了！🎉
