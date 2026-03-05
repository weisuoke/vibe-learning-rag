# 核心概念 4：Workspace 初始化

**本文档深入讲解 Workspace 的初始化过程、目录结构、Bootstrap 文件和最佳实践。**

---

## 概述

Workspace 是 Agent 的工作目录，包含：
1. **Bootstrap 文件**：Agent 的初始指令和上下文
2. **会话数据**：对话历史和状态
3. **文档存储**：Agent 可访问的文件
4. **脚本工具**：自定义脚本和工具

Workspace 的正确初始化是 Agent 正常工作的基础。

---

## Workspace 目录结构

### 标准结构

```
~/.openclaw/workspace/
├── .openclaw/                    # Agent 元数据目录
│   ├── bootstrap.md              # Bootstrap 文件（Agent 初始指令）
│   ├── agent.json                # Agent 配置
│   ├── sessions/                 # 会话数据
│   │   ├── main:whatsapp:dm:+1234567890/
│   │   │   ├── transcript.jsonl  # 对话历史
│   │   │   └── state.json        # 会话状态
│   │   └── main:telegram:dm:123456789/
│   │       ├── transcript.jsonl
│   │       └── state.json
│   └── cache/                    # 缓存数据
│       ├── embeddings/           # Embedding 缓存
│       └── tools/                # 工具缓存
├── documents/                    # 文档存储
│   ├── notes/                    # 笔记
│   ├── references/               # 参考资料
│   └── projects/                 # 项目文件
├── scripts/                      # 脚本工具
│   ├── automation/               # 自动化脚本
│   └── utilities/                # 工具脚本
└── temp/                         # 临时文件
    └── downloads/                # 下载文件
```

### 目录说明

| 目录 | 用途 | 权限 | 备份建议 |
|------|------|------|---------|
| `.openclaw/` | Agent 元数据 | 仅用户可读写 | 必须备份 |
| `.openclaw/bootstrap.md` | Agent 初始指令 | 仅用户可读写 | 必须备份 |
| `.openclaw/sessions/` | 会话数据 | 仅用户可读写 | 建议备份 |
| `documents/` | 文档存储 | 用户可读写 | 必须备份 |
| `scripts/` | 脚本工具 | 用户可读写执行 | 建议备份 |
| `temp/` | 临时文件 | 用户可读写 | 无需备份 |

---

## Workspace 初始化流程

### 1. 创建 Workspace 目录

```typescript
async function ensureWorkspaceExists(workspaceDir: string): Promise<void> {
  const resolvedPath = resolveUserPath(workspaceDir);

  // 检查目录是否存在
  if (await pathExists(resolvedPath)) {
    // 目录已存在，验证权限
    const stats = await fs.stat(resolvedPath);
    if (!stats.isDirectory()) {
      throw new Error(`Workspace path exists but is not a directory: ${resolvedPath}`);
    }
    return;
  }

  // 创建目录（递归创建父目录）
  await fs.mkdir(resolvedPath, { recursive: true, mode: 0o700 });

  // 验证创建成功
  if (!(await pathExists(resolvedPath))) {
    throw new Error(`Failed to create workspace directory: ${resolvedPath}`);
  }
}
```

**权限设置**：
- `0o700`：仅用户可读写执行（`rwx------`）
- 防止其他用户访问敏感数据

### 2. 创建 `.openclaw/` 元数据目录

```typescript
async function ensureOpenClawMetadataDir(workspaceDir: string): Promise<void> {
  const metadataDir = path.join(workspaceDir, ".openclaw");

  await fs.mkdir(metadataDir, { recursive: true, mode: 0o700 });

  // 创建子目录
  await fs.mkdir(path.join(metadataDir, "sessions"), { recursive: true, mode: 0o700 });
  await fs.mkdir(path.join(metadataDir, "cache"), { recursive: true, mode: 0o700 });
  await fs.mkdir(path.join(metadataDir, "cache", "embeddings"), { recursive: true, mode: 0o700 });
  await fs.mkdir(path.join(metadataDir, "cache", "tools"), { recursive: true, mode: 0o700 });
}
```

### 3. 生成 Bootstrap 文件

```typescript
async function seedBootstrapFile(workspaceDir: string): Promise<void> {
  const bootstrapPath = path.join(workspaceDir, ".openclaw", "bootstrap.md");

  // 检查是否已存在
  if (await pathExists(bootstrapPath)) {
    // 已存在，不覆盖
    return;
  }

  // 生成默认 Bootstrap 内容
  const bootstrapContent = generateDefaultBootstrap();

  // 写入文件
  await fs.writeFile(bootstrapPath, bootstrapContent, { mode: 0o600 });
}
```

**Bootstrap 文件权限**：
- `0o600`：仅用户可读写（`rw-------`）
- 防止其他用户读取敏感指令

### 4. 创建标准子目录

```typescript
async function createStandardSubdirectories(workspaceDir: string): Promise<void> {
  const subdirs = [
    "documents",
    "documents/notes",
    "documents/references",
    "documents/projects",
    "scripts",
    "scripts/automation",
    "scripts/utilities",
    "temp",
    "temp/downloads",
  ];

  for (const subdir of subdirs) {
    const subdirPath = path.join(workspaceDir, subdir);
    await fs.mkdir(subdirPath, { recursive: true, mode: 0o755 });
  }
}
```

**子目录权限**：
- `0o755`：用户可读写执行，其他用户可读执行（`rwxr-xr-x`）
- 允许 Agent 访问和执行脚本

---

## Bootstrap 文件详解

### Bootstrap 文件作用

Bootstrap 文件（`.openclaw/bootstrap.md`）是 Agent 的"新员工手册"，包含：

1. **身份定义**：Agent 的角色和职责
2. **工作指南**：如何处理任务和请求
3. **工具说明**：可用工具和使用方法
4. **约束规则**：不应该做什么
5. **上下文信息**：用户偏好和环境信息

### 默认 Bootstrap 内容

```markdown
# Agent Bootstrap

You are an AI assistant powered by OpenClaw. Your role is to help the user with various tasks while respecting their preferences and maintaining security.

## Your Capabilities

- **File Operations**: Read, write, and manage files in the workspace
- **Code Execution**: Run scripts and commands (with user approval)
- **Web Search**: Search the web for information
- **Task Automation**: Automate repetitive tasks

## Workspace Structure

Your workspace is located at: `~/.openclaw/workspace`

- `documents/`: Store and organize documents
- `scripts/`: Custom scripts and automation
- `temp/`: Temporary files (auto-cleaned)

## Guidelines

1. **Always ask before**:
   - Executing commands that modify the system
   - Accessing files outside the workspace
   - Making network requests

2. **Security**:
   - Never expose sensitive information (API keys, passwords)
   - Validate user input before executing commands
   - Use least-privilege access

3. **Communication**:
   - Be concise and clear
   - Explain your reasoning when making decisions
   - Ask for clarification when uncertain

## User Preferences

(This section will be customized based on user configuration)

- Preferred language: English
- Timezone: UTC
- Communication style: Professional

## Notes

This bootstrap file can be customized to fit your specific needs. Edit it directly or use `openclaw configure` to update settings.
```

### Bootstrap 文件生成逻辑

```typescript
function generateDefaultBootstrap(): string {
  const template = `# Agent Bootstrap

You are an AI assistant powered by OpenClaw. Your role is to help the user with various tasks while respecting their preferences and maintaining security.

## Your Capabilities

- **File Operations**: Read, write, and manage files in the workspace
- **Code Execution**: Run scripts and commands (with user approval)
- **Web Search**: Search the web for information
- **Task Automation**: Automate repetitive tasks

## Workspace Structure

Your workspace is located at: \`~/.openclaw/workspace\`

- \`documents/\`: Store and organize documents
- \`scripts/\`: Custom scripts and automation
- \`temp/\`: Temporary files (auto-cleaned)

## Guidelines

1. **Always ask before**:
   - Executing commands that modify the system
   - Accessing files outside the workspace
   - Making network requests

2. **Security**:
   - Never expose sensitive information (API keys, passwords)
   - Validate user input before executing commands
   - Use least-privilege access

3. **Communication**:
   - Be concise and clear
   - Explain your reasoning when making decisions
   - Ask for clarification when uncertain

## User Preferences

(This section will be customized based on user configuration)

- Preferred language: English
- Timezone: UTC
- Communication style: Professional

## Notes

This bootstrap file can be customized to fit your specific needs. Edit it directly or use \`openclaw configure\` to update settings.
`;

  return template;
}
```

### 自定义 Bootstrap 文件

**方式 1：直接编辑**

```bash
# 编辑 Bootstrap 文件
vim ~/.openclaw/workspace/.openclaw/bootstrap.md

# 或使用 OpenClaw 命令
openclaw workspace edit-bootstrap
```

**方式 2：使用模板**

```bash
# 使用预定义模板
openclaw workspace bootstrap --template developer
openclaw workspace bootstrap --template researcher
openclaw workspace bootstrap --template assistant
```

**方式 3：从文件导入**

```bash
# 从文件导入 Bootstrap
openclaw workspace bootstrap --from-file ~/my-bootstrap.md
```

---

## 会话数据管理

### 会话目录结构

```
.openclaw/sessions/
├── main:whatsapp:dm:+1234567890/
│   ├── transcript.jsonl          # 对话历史（JSONL 格式）
│   ├── state.json                # 会话状态
│   └── metadata.json             # 会话元数据
└── main:telegram:dm:123456789/
    ├── transcript.jsonl
    ├── state.json
    └── metadata.json
```

### 会话 Key 格式

```
<agentId>:<channel>:<scope>:<identifier>
```

**示例**：

| 会话 Key | 说明 |
|---------|------|
| `main:whatsapp:dm:+1234567890` | 主 Agent，WhatsApp DM，电话号码 |
| `main:telegram:dm:123456789` | 主 Agent，Telegram DM，用户 ID |
| `work:discord:guild:987654321` | 工作 Agent，Discord 服务器 |

### Transcript 格式

```jsonl
{"role":"user","content":"你好","timestamp":"2026-02-22T00:00:00Z"}
{"role":"assistant","content":"你好！有什么可以帮助你的吗？","timestamp":"2026-02-22T00:00:05Z"}
{"role":"user","content":"今天天气怎么样？","timestamp":"2026-02-22T00:00:10Z"}
{"role":"assistant","content":"我需要你的位置信息才能查询天气。","timestamp":"2026-02-22T00:00:15Z"}
```

**JSONL 格式优点**：
- 每行一个 JSON 对象
- 易于追加（不需要重写整个文件）
- 易于流式处理
- 易于备份和恢复

### 会话状态管理

```json
{
  "sessionKey": "main:whatsapp:dm:+1234567890",
  "agentId": "main",
  "channel": "whatsapp",
  "scope": "dm",
  "identifier": "+1234567890",
  "createdAt": "2026-02-22T00:00:00Z",
  "lastActiveAt": "2026-02-22T00:00:15Z",
  "messageCount": 4,
  "context": {
    "userPreferences": {
      "language": "zh-CN",
      "timezone": "Asia/Shanghai"
    },
    "conversationSummary": "用户询问天气信息"
  }
}
```

---

## Workspace 配置

### Agent 配置文件

```json
// .openclaw/agent.json
{
  "agentId": "main",
  "workspace": "~/.openclaw/workspace",
  "model": {
    "primary": "anthropic/claude-sonnet-4-5",
    "fallbacks": ["openai/gpt-4"]
  },
  "tools": {
    "filesystem": {
      "enabled": true,
      "allowedPaths": ["~/.openclaw/workspace"]
    },
    "bash": {
      "enabled": true,
      "requireApproval": true
    },
    "web": {
      "enabled": true,
      "search": {
        "provider": "brave",
        "apiKey": "${BRAVE_API_KEY}"
      }
    }
  },
  "security": {
    "dangerousNodeDenyCommands": [
      "camera.snap",
      "camera.clip",
      "screen.record"
    ]
  }
}
```

### Workspace 元数据

```json
// .openclaw/workspace.json
{
  "version": "1.0",
  "createdAt": "2026-02-22T00:00:00Z",
  "lastModifiedAt": "2026-02-22T00:00:15Z",
  "owner": "user@example.com",
  "description": "Main workspace for personal assistant",
  "tags": ["personal", "assistant"],
  "statistics": {
    "totalSessions": 10,
    "totalMessages": 1000,
    "storageUsed": "50MB"
  }
}
```

---

## 多 Workspace 管理

### 创建多个 Workspace

```bash
# 创建工作 Workspace
openclaw agents add work \
  --workspace ~/.openclaw/workspace-work

# 创建个人 Workspace
openclaw agents add personal \
  --workspace ~/.openclaw/workspace-personal

# 创建实验 Workspace
openclaw agents add experimental \
  --workspace ~/.openclaw/workspace-experimental
```

### Workspace 隔离

```
~/.openclaw/
├── workspace/                    # 主 Workspace
│   ├── .openclaw/
│   ├── documents/
│   └── scripts/
├── workspace-work/               # 工作 Workspace
│   ├── .openclaw/
│   ├── documents/
│   │   ├── projects/
│   │   └── meetings/
│   └── scripts/
└── workspace-personal/           # 个人 Workspace
    ├── .openclaw/
    ├── documents/
    │   ├── notes/
    │   └── photos/
    └── scripts/
```

**隔离优点**：
- 数据分离（工作/个人）
- 权限隔离（不同 Agent 不同权限）
- 配置独立（不同模型、工具）

---

## Workspace 维护

### 清理临时文件

```bash
# 清理 temp 目录
openclaw workspace clean --temp

# 清理缓存
openclaw workspace clean --cache

# 清理旧会话（超过 30 天）
openclaw workspace clean --sessions --older-than 30d
```

**实现逻辑**：

```typescript
async function cleanWorkspace(options: {
  temp?: boolean;
  cache?: boolean;
  sessions?: boolean;
  olderThan?: string;
}): Promise<void> {
  const workspaceDir = resolveWorkspaceDir();

  if (options.temp) {
    const tempDir = path.join(workspaceDir, "temp");
    await fs.rm(tempDir, { recursive: true, force: true });
    await fs.mkdir(tempDir, { recursive: true, mode: 0o755 });
  }

  if (options.cache) {
    const cacheDir = path.join(workspaceDir, ".openclaw", "cache");
    await fs.rm(cacheDir, { recursive: true, force: true });
    await fs.mkdir(cacheDir, { recursive: true, mode: 0o700 });
  }

  if (options.sessions && options.olderThan) {
    const sessionsDir = path.join(workspaceDir, ".openclaw", "sessions");
    const cutoffDate = parseDuration(options.olderThan);

    const sessionDirs = await fs.readdir(sessionsDir);
    for (const sessionDir of sessionDirs) {
      const sessionPath = path.join(sessionsDir, sessionDir);
      const stats = await fs.stat(sessionPath);

      if (stats.mtime < cutoffDate) {
        await fs.rm(sessionPath, { recursive: true, force: true });
      }
    }
  }
}
```

### 备份 Workspace

```bash
# 备份整个 Workspace
openclaw workspace backup --output ~/backups/workspace-2026-02-22.tar.gz

# 仅备份重要数据（排除 temp 和 cache）
openclaw workspace backup --exclude-temp --exclude-cache \
  --output ~/backups/workspace-essential-2026-02-22.tar.gz

# 增量备份（仅备份变更）
openclaw workspace backup --incremental \
  --base ~/backups/workspace-2026-02-21.tar.gz \
  --output ~/backups/workspace-2026-02-22-incremental.tar.gz
```

**实现逻辑**：

```typescript
async function backupWorkspace(options: {
  output: string;
  excludeTemp?: boolean;
  excludeCache?: boolean;
  incremental?: boolean;
  base?: string;
}): Promise<void> {
  const workspaceDir = resolveWorkspaceDir();

  const excludePatterns: string[] = [];
  if (options.excludeTemp) {
    excludePatterns.push("temp/**");
  }
  if (options.excludeCache) {
    excludePatterns.push(".openclaw/cache/**");
  }

  if (options.incremental && options.base) {
    // 增量备份：仅备份自上次备份以来的变更
    const baseTimestamp = await getBackupTimestamp(options.base);
    await createIncrementalBackup(workspaceDir, options.output, baseTimestamp, excludePatterns);
  } else {
    // 完整备份
    await createFullBackup(workspaceDir, options.output, excludePatterns);
  }
}
```

### 恢复 Workspace

```bash
# 恢复完整备份
openclaw workspace restore --from ~/backups/workspace-2026-02-22.tar.gz

# 恢复到指定目录
openclaw workspace restore --from ~/backups/workspace-2026-02-22.tar.gz \
  --to ~/.openclaw/workspace-restored

# 仅恢复特定文件
openclaw workspace restore --from ~/backups/workspace-2026-02-22.tar.gz \
  --files ".openclaw/bootstrap.md" "documents/notes/**"
```

---

## Workspace 安全

### 权限管理

```bash
# 检查 Workspace 权限
openclaw workspace check-permissions

# 修复权限
openclaw workspace fix-permissions
```

**实现逻辑**：

```typescript
async function fixWorkspacePermissions(workspaceDir: string): Promise<void> {
  // .openclaw/ 目录：仅用户可读写执行
  await fs.chmod(path.join(workspaceDir, ".openclaw"), 0o700);

  // Bootstrap 文件：仅用户可读写
  await fs.chmod(path.join(workspaceDir, ".openclaw", "bootstrap.md"), 0o600);

  // 会话目录：仅用户可读写执行
  await fs.chmod(path.join(workspaceDir, ".openclaw", "sessions"), 0o700);

  // 文档目录：用户可读写执行
  await fs.chmod(path.join(workspaceDir, "documents"), 0o755);

  // 脚本目录：用户可读写执行
  await fs.chmod(path.join(workspaceDir, "scripts"), 0o755);
}
```

### 敏感数据保护

```bash
# 扫描敏感数据
openclaw workspace scan-secrets

# 加密敏感文件
openclaw workspace encrypt --files "documents/secrets/**"

# 解密文件
openclaw workspace decrypt --files "documents/secrets/**"
```

**敏感数据检测**：

```typescript
const SENSITIVE_PATTERNS = [
  /sk-ant-[a-zA-Z0-9]{32,}/,  // Anthropic API Key
  /sk-[a-zA-Z0-9]{32,}/,       // OpenAI API Key
  /ghp_[a-zA-Z0-9]{36}/,       // GitHub Personal Access Token
  /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/,  // Email
  /\b\d{3}-\d{2}-\d{4}\b/,     // SSN
];

async function scanForSecrets(workspaceDir: string): Promise<string[]> {
  const findings: string[] = [];

  const files = await glob("**/*", { cwd: workspaceDir, ignore: [".openclaw/cache/**", "temp/**"] });

  for (const file of files) {
    const content = await fs.readFile(path.join(workspaceDir, file), "utf-8");

    for (const pattern of SENSITIVE_PATTERNS) {
      if (pattern.test(content)) {
        findings.push(`${file}: Potential sensitive data detected`);
      }
    }
  }

  return findings;
}
```

---

## 最佳实践

### 1. Workspace 组织

```
~/.openclaw/workspace/
├── .openclaw/                    # 元数据（不要手动修改）
├── documents/
│   ├── inbox/                    # 待处理文档
│   ├── archive/                  # 归档文档
│   ├── projects/
│   │   ├── project-a/
│   │   └── project-b/
│   └── references/
│       ├── apis/
│       └── docs/
├── scripts/
│   ├── daily/                    # 每日脚本
│   ├── weekly/                   # 每周脚本
│   └── utilities/                # 工具脚本
└── temp/                         # 临时文件（定期清理）
```

### 2. Bootstrap 文件维护

```markdown
# Agent Bootstrap

## Identity
You are a personal AI assistant specialized in software development.

## Capabilities
- Code review and refactoring
- Documentation generation
- Test automation
- Deployment assistance

## User Context
- Name: John Doe
- Role: Senior Software Engineer
- Preferred languages: TypeScript, Python
- Timezone: America/New_York
- Working hours: 9 AM - 6 PM EST

## Project Context
- Current project: E-commerce platform
- Tech stack: Next.js, PostgreSQL, Redis
- Repository: https://github.com/company/ecommerce

## Guidelines
1. Always write tests for new features
2. Follow company coding standards
3. Document all API changes
4. Ask before making breaking changes

## Tools
- Filesystem: Full access to workspace
- Bash: Requires approval for system commands
- Web: Enabled for documentation lookup

## Notes
Last updated: 2026-02-22
```

### 3. 会话管理

```bash
# 定期清理旧会话
0 2 * * * openclaw workspace clean --sessions --older-than 90d

# 备份重要会话
openclaw workspace export-session main:whatsapp:dm:+1234567890 \
  --output ~/backups/important-conversation.jsonl
```

### 4. 安全检查

```bash
# 每周安全扫描
0 0 * * 0 openclaw workspace scan-secrets >> ~/logs/security-scan.log

# 每月权限检查
0 0 1 * * openclaw workspace check-permissions >> ~/logs/permissions-check.log
```

---

## 故障排查

### 问题 1：Workspace 初始化失败

```bash
# 检查目录权限
ls -la ~/.openclaw/

# 手动创建目录
mkdir -p ~/.openclaw/workspace/.openclaw
chmod 700 ~/.openclaw/workspace/.openclaw

# 重新初始化
openclaw workspace init
```

### 问题 2：Bootstrap 文件丢失

```bash
# 重新生成 Bootstrap 文件
openclaw workspace bootstrap --regenerate

# 从备份恢复
cp ~/backups/bootstrap.md ~/.openclaw/workspace/.openclaw/bootstrap.md
chmod 600 ~/.openclaw/workspace/.openclaw/bootstrap.md
```

### 问题 3：会话数据损坏

```bash
# 验证会话数据
openclaw workspace verify-sessions

# 修复损坏的会话
openclaw workspace repair-sessions

# 从备份恢复
openclaw workspace restore --from ~/backups/workspace-2026-02-22.tar.gz \
  --files ".openclaw/sessions/**"
```

---

## 总结

Workspace 初始化的核心要点：

1. **目录结构**：标准化的目录结构，易于管理和备份
2. **Bootstrap 文件**：Agent 的"新员工手册"，定义身份和行为
3. **会话管理**：JSONL 格式的对话历史，易于追加和处理
4. **权限控制**：严格的权限设置，保护敏感数据
5. **多 Workspace**：支持多个隔离的 Workspace，适合不同场景

理解 Workspace 的初始化和管理，可以帮助你更好地组织和保护 Agent 的数据。
