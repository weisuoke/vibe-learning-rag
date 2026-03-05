# 核心概念 6：Skills 安装流程

**本文档深入讲解 Skills 的安装流程、依赖管理和配置方法。**

---

## 概述

Skills 是 OpenClaw 的扩展系统，为 Agent 提供额外的能力。Skills 可以：

1. **扩展工具集**：添加新的工具和命令
2. **集成服务**：连接第三方服务（Gmail、Notion、GitHub 等）
3. **自动化任务**：执行定时任务和自动化流程
4. **增强能力**：提供专业领域的知识和能力

---

## Skills 类型

### 按功能分类

| 类型 | 示例 | 用途 |
|------|------|------|
| **工具类** | web-search, file-operations | 基础工具能力 |
| **集成类** | gmail, notion, github | 第三方服务集成 |
| **自动化类** | cron-jobs, webhooks | 任务自动化 |
| **专业类** | code-review, data-analysis | 专业领域能力 |

### 按安装方式分类

| 类型 | 安装方式 | 示例 |
|------|---------|------|
| **内置 Skills** | 随 OpenClaw 安装 | web-search, file-operations |
| **官方 Skills** | 从官方仓库安装 | gmail, notion, github |
| **社区 Skills** | 从社区仓库安装 | custom-tools, third-party-integrations |
| **本地 Skills** | 从本地目录安装 | ~/my-skills/custom-skill |

---

## Skills 安装流程

### 1. Onboarding 时安装

```
用户运行 openclaw onboard
    ↓
完成基础配置（Model/Auth、Workspace、Gateway、Channels、Daemon）
    ↓
┌─────────────────────────────────────┐
│  Skills 安装提示                    │
│                                     │
│  ◆  Install recommended skills?     │
│     ● Yes / ○ No                    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  推荐 Skills 列表                   │
│  - web-search (网络搜索)           │
│  - file-operations (文件操作)      │
│  - code-execution (代码执行)       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  依赖检查                           │
│  - 检查 Homebrew（macOS）          │
│  - 检查 Python 环境                │
│  - 检查必需工具                    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  安装 Skills                        │
│  - 下载 Skill 包                   │
│  - 安装依赖                        │
│  - 配置 Skill                      │
│  - 启用 Skill                      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  验证安装                           │
│  - 测试 Skill 功能                 │
│  - 显示安装结果                    │
└─────────────────────────────────────┘
```

### 2. 手动安装

```bash
# 安装单个 Skill
openclaw skills install web-search

# 安装多个 Skills
openclaw skills install web-search file-operations code-execution

# 从 URL 安装
openclaw skills install https://github.com/openclaw/skill-gmail

# 从本地目录安装
openclaw skills install ~/my-skills/custom-skill
```

---

## 推荐 Skills

### Web Search（网络搜索）

**功能**：
- 使用 Brave Search API 搜索网络
- 获取实时信息
- 支持多种搜索类型（网页、新闻、图片）

**依赖**：
- Brave Search API Key

**配置**：

```json5
{
  tools: {
    web: {
      search: {
        provider: "brave",
        apiKey: "${BRAVE_API_KEY}",
      },
    },
  },
}
```

**使用示例**：

```
用户：今天的天气怎么样？
Agent：[使用 web_search 工具搜索天气信息]
Agent：根据搜索结果，今天北京天气晴朗，温度 15-25°C。
```

### File Operations（文件操作）

**功能**：
- 读取、写入、编辑文件
- 创建、删除、移动文件
- 搜索文件内容

**依赖**：
- 无（内置）

**配置**：

```json5
{
  tools: {
    filesystem: {
      enabled: true,
      allowedPaths: ["~/.openclaw/workspace"],
    },
  },
}
```

**使用示例**：

```
用户：创建一个 TODO 列表文件
Agent：[使用 file_write 工具创建文件]
Agent：已创建 TODO.md 文件，内容如下：
# TODO List
- [ ] Task 1
- [ ] Task 2
```

### Code Execution（代码执行）

**功能**：
- 执行 Python、JavaScript、Bash 脚本
- 运行代码片段
- 测试代码功能

**依赖**：
- Python 3.x
- Node.js（可选）

**配置**：

```json5
{
  tools: {
    bash: {
      enabled: true,
      requireApproval: true,  // 需要用户批准
    },
    python: {
      enabled: true,
      requireApproval: true,
    },
  },
}
```

**使用示例**：

```
用户：计算 1 到 100 的和
Agent：[使用 python_execute 工具执行代码]
Agent：计算结果：5050
```

---

## Skills 安装实现

### 安装流程代码

```typescript
// 伪代码示例
async function installSkills(options: {
  skills: string[];
  flow: WizardFlow;
  prompter: WizardPrompter;
}): Promise<void> {
  const { skills, flow, prompter } = options;

  // 1. 检查依赖
  const progress = prompter.progress("Checking dependencies");
  try {
    progress.update("Checking Homebrew...");
    const hasHomebrew = await checkHomebrew();

    if (!hasHomebrew && process.platform === "darwin") {
      await prompter.note(
        "Homebrew not found. Some skills may require manual dependency installation.",
        "Dependencies"
      );
    }

    progress.update("Checking Python...");
    const hasPython = await checkPython();

    if (!hasPython) {
      await prompter.note(
        "Python not found. Some skills may not work correctly.",
        "Dependencies"
      );
    }
  } finally {
    progress.stop("Dependency check complete.");
  }

  // 2. 安装 Skills
  for (const skill of skills) {
    const skillProgress = prompter.progress(`Installing ${skill}`);

    try {
      // 2.1 下载 Skill
      skillProgress.update(`Downloading ${skill}...`);
      await downloadSkill(skill);

      // 2.2 安装依赖
      skillProgress.update(`Installing dependencies for ${skill}...`);
      await installSkillDependencies(skill);

      // 2.3 配置 Skill
      skillProgress.update(`Configuring ${skill}...`);
      await configureSkill(skill);

      // 2.4 启用 Skill
      skillProgress.update(`Enabling ${skill}...`);
      await enableSkill(skill);

      skillProgress.stop(`✓ ${skill} installed successfully.`);
    } catch (err) {
      skillProgress.stop(`✗ ${skill} installation failed.`);
      await prompter.note(
        `Failed to install ${skill}: ${err instanceof Error ? err.message : String(err)}`,
        "Error"
      );
    }
  }

  // 3. 验证安装
  await prompter.note(
    "Skills installed. Run `openclaw skills list` to see all installed skills.",
    "Skills"
  );
}
```

### 依赖检查

```typescript
async function checkHomebrew(): Promise<boolean> {
  try {
    const result = await exec("which brew");
    return result.exitCode === 0;
  } catch {
    return false;
  }
}

async function checkPython(): Promise<boolean> {
  try {
    const result = await exec("python3 --version");
    return result.exitCode === 0;
  } catch {
    return false;
  }
}

async function checkNode(): Promise<boolean> {
  try {
    const result = await exec("node --version");
    return result.exitCode === 0;
  } catch {
    return false;
  }
}
```

### Skill 下载

```typescript
async function downloadSkill(skillName: string): Promise<void> {
  // 1. 解析 Skill 来源
  const source = resolveSkillSource(skillName);

  if (source.type === "official") {
    // 从官方仓库下载
    await downloadFromGitHub(`openclaw/skill-${skillName}`);
  } else if (source.type === "url") {
    // 从 URL 下载
    await downloadFromUrl(source.url);
  } else if (source.type === "local") {
    // 从本地目录复制
    await copyFromLocal(source.path);
  }

  // 2. 解压到 Skills 目录
  const skillDir = path.join(os.homedir(), ".openclaw", "skills", skillName);
  await extractSkill(skillName, skillDir);
}
```

### 依赖安装

```typescript
async function installSkillDependencies(skillName: string): Promise<void> {
  const skillDir = path.join(os.homedir(), ".openclaw", "skills", skillName);
  const manifestPath = path.join(skillDir, "skill.json");

  // 1. 读取 Skill 清单
  const manifest = JSON.parse(await fs.readFile(manifestPath, "utf-8"));

  // 2. 安装系统依赖（通过 Homebrew）
  if (manifest.dependencies?.system) {
    for (const dep of manifest.dependencies.system) {
      await exec(`brew install ${dep}`);
    }
  }

  // 3. 安装 Python 依赖
  if (manifest.dependencies?.python) {
    const requirementsPath = path.join(skillDir, "requirements.txt");
    await exec(`pip3 install -r ${requirementsPath}`);
  }

  // 4. 安装 Node.js 依赖
  if (manifest.dependencies?.node) {
    await exec(`npm install`, { cwd: skillDir });
  }
}
```

---

## Skill 清单格式

### skill.json 示例

```json
{
  "name": "web-search",
  "version": "1.0.0",
  "description": "Web search using Brave Search API",
  "author": "OpenClaw Team",
  "license": "MIT",
  "dependencies": {
    "system": [],
    "python": ["requests"],
    "node": []
  },
  "config": {
    "apiKey": {
      "type": "string",
      "required": true,
      "env": "BRAVE_API_KEY",
      "description": "Brave Search API Key"
    }
  },
  "tools": [
    {
      "name": "web_search",
      "description": "Search the web using Brave Search",
      "parameters": {
        "query": {
          "type": "string",
          "required": true,
          "description": "Search query"
        },
        "count": {
          "type": "number",
          "default": 10,
          "description": "Number of results"
        }
      }
    }
  ]
}
```

---

## Skills 管理

### 列出已安装的 Skills

```bash
# 列出所有 Skills
openclaw skills list

# 输出示例：
# Installed Skills:
# ✓ web-search (v1.0.0) - Web search using Brave Search API
# ✓ file-operations (v1.0.0) - File operations toolkit
# ✓ code-execution (v1.0.0) - Execute code snippets
```

### 启用/禁用 Skills

```bash
# 启用 Skill
openclaw skills enable web-search

# 禁用 Skill
openclaw skills disable web-search

# 查看 Skill 状态
openclaw skills status web-search
```

### 更新 Skills

```bash
# 更新单个 Skill
openclaw skills update web-search

# 更新所有 Skills
openclaw skills update --all

# 检查可用更新
openclaw skills check-updates
```

### 卸载 Skills

```bash
# 卸载 Skill
openclaw skills uninstall web-search

# 卸载并删除配置
openclaw skills uninstall web-search --purge
```

---

## Skills 配置

### 配置文件位置

```
~/.openclaw/skills/
├── web-search/
│   ├── skill.json              # Skill 清单
│   ├── config.json             # Skill 配置
│   ├── main.py                 # Skill 主程序
│   └── requirements.txt        # Python 依赖
├── file-operations/
│   ├── skill.json
│   ├── config.json
│   └── main.js
└── code-execution/
    ├── skill.json
    ├── config.json
    └── main.py
```

### 配置 Skill

```bash
# 交互式配置
openclaw skills configure web-search

# 设置配置项
openclaw skills config set web-search apiKey "your-api-key"

# 查看配置
openclaw skills config get web-search apiKey

# 删除配置项
openclaw skills config unset web-search apiKey
```

### 配置示例

```json
// ~/.openclaw/skills/web-search/config.json
{
  "apiKey": "your-brave-api-key",
  "defaultCount": 10,
  "safeSearch": true,
  "language": "en"
}
```

---

## 常见 Skills 详解

### Gmail Skill

**功能**：
- 读取邮件
- 发送邮件
- 搜索邮件
- 管理标签

**安装**：

```bash
openclaw skills install gmail
```

**配置**：

```bash
# 设置 Gmail OAuth 凭证
openclaw skills configure gmail

# 授权 Gmail 访问
openclaw skills auth gmail
```

**使用示例**：

```
用户：查看最近的 5 封邮件
Agent：[使用 gmail_list 工具]
Agent：最近的 5 封邮件：
1. [重要] 项目进度更新 - from: boss@company.com
2. 会议邀请 - from: colleague@company.com
...
```

### Notion Skill

**功能**：
- 读取 Notion 页面
- 创建 Notion 页面
- 更新 Notion 数据库
- 搜索 Notion 内容

**安装**：

```bash
openclaw skills install notion
```

**配置**：

```bash
# 设置 Notion API Token
openclaw skills config set notion apiToken "your-notion-token"
```

**使用示例**：

```
用户：在 Notion 中创建一个新的任务
Agent：[使用 notion_create_page 工具]
Agent：已在 Notion 中创建任务：
标题：新任务
状态：待办
截止日期：2026-02-25
```

### GitHub Skill

**功能**：
- 查看仓库信息
- 创建 Issue
- 创建 Pull Request
- 查看 Commit 历史

**安装**：

```bash
openclaw skills install github
```

**配置**：

```bash
# 设置 GitHub Personal Access Token
openclaw skills config set github token "your-github-token"
```

**使用示例**：

```
用户：在 GitHub 上创建一个 Issue
Agent：[使用 github_create_issue 工具]
Agent：已创建 Issue #123：
标题：修复登录 Bug
描述：用户无法登录...
链接：https://github.com/user/repo/issues/123
```

---

## 自定义 Skills

### 创建自定义 Skill

```bash
# 创建 Skill 模板
openclaw skills create my-custom-skill

# 目录结构
~/.openclaw/skills/my-custom-skill/
├── skill.json              # Skill 清单
├── main.py                 # 主程序
├── requirements.txt        # Python 依赖
└── README.md               # 说明文档
```

### skill.json 模板

```json
{
  "name": "my-custom-skill",
  "version": "1.0.0",
  "description": "My custom skill",
  "author": "Your Name",
  "license": "MIT",
  "dependencies": {
    "system": [],
    "python": [],
    "node": []
  },
  "config": {},
  "tools": [
    {
      "name": "my_tool",
      "description": "My custom tool",
      "parameters": {
        "input": {
          "type": "string",
          "required": true,
          "description": "Input parameter"
        }
      }
    }
  ]
}
```

### main.py 模板

```python
#!/usr/bin/env python3
import json
import sys

def my_tool(input: str) -> dict:
    """My custom tool implementation"""
    # 实现工具逻辑
    result = f"Processed: {input}"
    return {"result": result}

if __name__ == "__main__":
    # 读取输入
    input_data = json.loads(sys.stdin.read())
    tool_name = input_data["tool"]
    params = input_data["params"]

    # 调用工具
    if tool_name == "my_tool":
        result = my_tool(params["input"])
    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    # 输出结果
    print(json.dumps(result))
```

### 测试自定义 Skill

```bash
# 测试 Skill
openclaw skills test my-custom-skill

# 调试 Skill
openclaw skills debug my-custom-skill --tool my_tool --params '{"input":"test"}'
```

---

## Skills 安全

### 权限控制

```json5
{
  skills: {
    "web-search": {
      enabled: true,
      permissions: {
        network: true,  // 允许网络访问
        filesystem: false,  // 禁止文件系统访问
      },
    },
    "file-operations": {
      enabled: true,
      permissions: {
        network: false,
        filesystem: true,
        allowedPaths: ["~/.openclaw/workspace"],  // 限制访问路径
      },
    },
  },
}
```

### 审计日志

```bash
# 查看 Skill 使用日志
openclaw skills logs web-search

# 查看所有 Skills 日志
openclaw skills logs --all

# 导出日志
openclaw skills logs --export ~/skill-logs.json
```

### 安全扫描

```bash
# 扫描 Skill 安全问题
openclaw skills scan web-search

# 扫描所有 Skills
openclaw skills scan --all

# 自动修复安全问题
openclaw skills scan --fix
```

---

## 故障排查

### 问题 1：Skill 安装失败

```bash
# 检查依赖
openclaw skills check-dependencies

# 手动安装依赖
brew install python3
pip3 install requests

# 重新安装 Skill
openclaw skills uninstall web-search
openclaw skills install web-search
```

### 问题 2：Skill 无法启用

```bash
# 检查 Skill 状态
openclaw skills status web-search

# 查看错误日志
openclaw skills logs web-search --level error

# 验证配置
openclaw skills validate web-search
```

### 问题 3：Skill 配置错误

```bash
# 重置配置
openclaw skills config reset web-search

# 重新配置
openclaw skills configure web-search

# 验证配置
openclaw skills config validate web-search
```

### 问题 4：Skill 版本冲突

```bash
# 检查版本
openclaw skills version web-search

# 降级到特定版本
openclaw skills install web-search@1.0.0

# 升级到最新版本
openclaw skills update web-search --latest
```

---

## 最佳实践

### 1. 选择必需的 Skills

```bash
# 最小化安装（仅基础 Skills）
openclaw skills install web-search file-operations

# 完整安装（所有推荐 Skills）
openclaw skills install --recommended

# 按需安装（根据使用场景）
openclaw skills install gmail notion github  # 办公场景
openclaw skills install code-review data-analysis  # 开发场景
```

### 2. 定期更新 Skills

```bash
# 每周检查更新
openclaw skills check-updates

# 自动更新（谨慎使用）
openclaw skills update --all --auto

# 手动更新（推荐）
openclaw skills update web-search
```

### 3. 备份 Skills 配置

```bash
# 导出配置
openclaw skills export --output ~/skills-backup.json

# 导入配置
openclaw skills import --from ~/skills-backup.json
```

### 4. 监控 Skills 使用

```bash
# 查看使用统计
openclaw skills stats

# 查看最常用的 Skills
openclaw skills stats --top 10

# 清理未使用的 Skills
openclaw skills prune --unused-days 90
```

---

## 总结

Skills 安装流程的核心要点：

1. **灵活安装**：支持 Onboarding 时安装或后续手动安装
2. **依赖管理**：自动检查和安装依赖（Homebrew、Python、Node.js）
3. **配置简单**：交互式配置或命令行配置
4. **扩展性强**：支持官方、社区和自定义 Skills
5. **安全可控**：权限控制、审计日志、安全扫描

理解 Skills 安装流程，可以帮助你充分利用 OpenClaw 的扩展能力。
