# OpenClaw 基础故障排查 - 核心概念 1：openclaw doctor 诊断工具

## 一、概念定义

**openclaw doctor 是 OpenClaw 的自动化健康检查和修复工具，通过系统化的诊断流程和自动修复机制，快速识别并解决配置、服务和环境问题。**

**来源**: `reference/source_doctor_command.md`

---

## 二、核心价值

### 2.1 为什么需要 doctor 命令？

**问题场景**：
- 升级后配置不兼容
- 手动编辑配置出错
- 环境变量覆盖配置
- 服务状态不一致
- 权限配置错误

**传统解决方式的问题**：
- ❌ 手动检查配置文件 → 容易遗漏
- ❌ 手动修复配置 → 容易出错
- ❌ 不知道从哪里开始 → 浪费时间
- ❌ 修复后无法验证 → 可能引入新问题

**doctor 命令的价值**：
- ✅ 自动化健康检查
- ✅ 一键修复常见问题
- ✅ 配置迁移和清理
- ✅ 修复前自动备份
- ✅ 提供详细的修复报告

**来源**: `reference/source_doctor_command.md`

### 2.2 doctor 命令的三大功能

#### 功能 1：健康检查

**检查内容**：
```typescript
// doctor 检查的内容
interface HealthCheck {
  config: {
    syntax: boolean;        // 配置文件语法
    validity: boolean;      // 配置值有效性
    compatibility: boolean; // 版本兼容性
  };
  services: {
    gateway: boolean;       // Gateway 状态
    channels: boolean;      // 通道状态
    permissions: boolean;   // 权限状态
  };
  environment: {
    nodeVersion: boolean;   // Node.js 版本
    dependencies: boolean;  // 依赖完整性
    envVars: boolean;       // 环境变量冲突
  };
}
```

**来源**: `reference/source_doctor_command.md`

#### 功能 2：自动修复

**修复能力**：
```bash
# doctor 可以自动修复的问题
openclaw doctor --repair

# 修复内容：
# 1. 配置迁移（v1 → v2）
# 2. 清理无效配置键
# 3. 修复配置值错误
# 4. 解决环境变量冲突
# 5. 修复权限问题
```

**来源**: `reference/source_doctor_command.md`

#### 功能 3：配置备份

**备份机制**：
```bash
# 修复前自动备份
~/.openclaw/openclaw.json.bak

# 备份内容：
# - 原始配置文件
# - 时间戳
# - 版本信息
```

**来源**: `reference/source_doctor_command.md`

---

## 三、使用方法

### 3.1 基础用法

```bash
# 1. 基础健康检查
openclaw doctor

# 输出示例：
# ✓ Config syntax valid
# ✓ Gateway mode configured
# ✓ Node.js version >= 22
# ✗ gateway.auth.token not set
#
# Issues found: 1
# Run 'openclaw doctor --repair' to fix

# 2. 自动修复
openclaw doctor --repair

# 输出示例：
# ✓ Backup saved to ~/.openclaw/openclaw.json.bak
# ✓ Config migrated from v1 to v2
# ✓ Removed 3 unknown config keys
# ✓ Fixed gateway.mode setting
# ✓ Generated gateway.auth.token
#
# All issues fixed!

# 3. 深度诊断
openclaw doctor --deep

# 输出示例：
# Running deep diagnostics...
# ✓ Config validation
# ✓ Service health check
# ✓ Channel connectivity
# ✓ Provider authentication
# ✓ Dependency check
#
# All checks passed!
```

**来源**: `reference/source_doctor_command.md`

### 3.2 常用选项

| 选项 | 说明 | 使用场景 |
|------|------|---------|
| `openclaw doctor` | 基础健康检查 | 日常检查、快速诊断 |
| `openclaw doctor --repair` | 自动修复 | 升级后、配置错误 |
| `openclaw doctor --fix` | 修复别名 | 同 --repair |
| `openclaw doctor --deep` | 深度诊断 | 复杂问题、全面检查 |
| `openclaw doctor --non-interactive` | 非交互模式 | 自动化脚本、CI/CD |

**来源**: `reference/source_doctor_command.md`

### 3.3 实战示例

**场景 1：升级后配置不兼容**

```bash
# 问题：升级到新版本后 Gateway 无法启动
# 错误：Config validation failed

# 解决步骤：
# 1. 运行 doctor
openclaw doctor

# 输出：
# ✗ Config version mismatch (v1 found, v2 required)
# ✗ Unknown config key: gateway.token
# ✗ Missing config key: gateway.auth.token

# 2. 自动修复
openclaw doctor --repair

# 输出：
# ✓ Backup saved
# ✓ Config migrated v1 → v2
# ✓ Removed gateway.token
# ✓ Added gateway.auth.token
# ✓ All issues fixed!

# 3. 验证
openclaw gateway status
# Runtime: running ✓
```

**来源**: `reference/search_doctor_command.md`

**场景 2：环境变量覆盖配置**

```bash
# 问题：macOS 上 launchctl 环境变量覆盖配置
# 症状：持续出现 "unauthorized" 错误

# 解决步骤：
# 1. 检查环境变量
launchctl getenv OPENCLAW_GATEWAY_TOKEN
# sk-ant-old-token-123

# 2. 运行 doctor
openclaw doctor

# 输出：
# ⚠ Environment variable OPENCLAW_GATEWAY_TOKEN overrides config
# ⚠ Token mismatch detected

# 3. 清除环境变量
launchctl unsetenv OPENCLAW_GATEWAY_TOKEN
launchctl unsetenv OPENCLAW_GATEWAY_PASSWORD

# 4. 验证
openclaw doctor
# ✓ All checks passed!
```

**来源**: `reference/source_doctor_command.md`

**场景 3：配置文件损坏**

```bash
# 问题：手动编辑配置后 Gateway 无法启动
# 错误：JSON parse error

# 解决步骤：
# 1. 运行 doctor
openclaw doctor

# 输出：
# ✗ Config syntax error at line 42
# ✗ Invalid JSON: unexpected token

# 2. 恢复备份（如果有）
cp ~/.openclaw/openclaw.json.bak ~/.openclaw/openclaw.json

# 3. 或者运行 repair
openclaw doctor --repair

# 输出：
# ✓ Config syntax fixed
# ✓ Invalid keys removed
# ✓ All issues fixed!
```

**来源**: `reference/search_doctor_command.md`

---

## 四、工作原理

### 4.1 检查流程

```typescript
// doctor 的检查流程
async function runDoctor(options: DoctorOptions): Promise<DoctorResult> {
  const checks: Check[] = [];

  // 1. 配置文件检查
  checks.push(await checkConfigSyntax());
  checks.push(await checkConfigValidity());
  checks.push(await checkConfigVersion());

  // 2. 服务状态检查
  checks.push(await checkGatewayStatus());
  checks.push(await checkChannelStatus());

  // 3. 环境检查
  checks.push(await checkNodeVersion());
  checks.push(await checkDependencies());
  checks.push(await checkEnvVars());

  // 4. 深度检查（如果启用）
  if (options.deep) {
    checks.push(await checkProviderAuth());
    checks.push(await checkChannelConnectivity());
  }

  // 5. 生成报告
  return generateReport(checks);
}
```

**来源**: `reference/source_doctor_command.md`

### 4.2 修复流程

```typescript
// doctor 的修复流程
async function runRepair(): Promise<RepairResult> {
  // 1. 备份配置
  await backupConfig('~/.openclaw/openclaw.json.bak');

  // 2. 运行检查
  const issues = await runDoctor({ repair: false });

  // 3. 修复问题
  const fixes: Fix[] = [];

  for (const issue of issues) {
    switch (issue.type) {
      case 'config_version':
        fixes.push(await migrateConfig());
        break;
      case 'unknown_key':
        fixes.push(await removeUnknownKeys());
        break;
      case 'missing_key':
        fixes.push(await addMissingKeys());
        break;
      case 'invalid_value':
        fixes.push(await fixInvalidValues());
        break;
      case 'env_conflict':
        fixes.push(await resolveEnvConflicts());
        break;
    }
  }

  // 4. 验证修复
  const verification = await runDoctor({ repair: false });

  // 5. 生成报告
  return generateRepairReport(fixes, verification);
}
```

**来源**: `reference/source_doctor_command.md`

### 4.3 配置迁移

```typescript
// 配置迁移示例
interface ConfigMigration {
  from: string;  // 源版本
  to: string;    // 目标版本
  changes: ConfigChange[];
}

interface ConfigChange {
  type: 'rename' | 'remove' | 'add' | 'transform';
  path: string;
  oldValue?: any;
  newValue?: any;
}

// 示例：v1 → v2 迁移
const v1ToV2Migration: ConfigMigration = {
  from: 'v1',
  to: 'v2',
  changes: [
    {
      type: 'rename',
      path: 'gateway.token',
      newValue: 'gateway.auth.token'
    },
    {
      type: 'add',
      path: 'gateway.auth.mode',
      newValue: 'token'
    },
    {
      type: 'remove',
      path: 'gateway.password'
    }
  ]
};
```

**来源**: `reference/source_doctor_command.md`

---

## 五、类比理解

### 5.1 后端开发类比

**类比 1：数据库健康检查**

```typescript
// OpenClaw doctor ≈ MySQL mysqlcheck
openclaw doctor          // 类似 mysqlcheck --check-only
openclaw doctor --repair // 类似 mysqlcheck --auto-repair

// 相似点：
// 1. 自动检测问题
// 2. 一键修复
// 3. 修复前备份
// 4. 提供详细报告
```

**类比 2：Kubernetes 健康检查**

```typescript
// OpenClaw doctor ≈ kubectl describe + kubectl doctor
openclaw doctor        // 类似 kubectl describe
openclaw doctor --deep // 类似 kubectl doctor

// 相似点：
// 1. 多层次检查（配置、服务、环境）
// 2. 自动化诊断
// 3. 提供修复建议
```

**类比 3：Nginx 配置验证**

```bash
# OpenClaw doctor ≈ nginx -t
openclaw doctor        # 类似 nginx -t（测试配置）
openclaw doctor --repair # 类似 nginx -t && nginx -s reload

# 相似点：
# 1. 配置语法检查
# 2. 配置有效性验证
# 3. 提供错误位置
```

### 5.2 日常生活类比

**类比 1：汽车诊断系统**

```
OpenClaw doctor = 汽车诊断仪

相似点：
1. 插上诊断仪 = 运行 doctor
2. 自动扫描故障码 = 自动检查配置
3. 显示故障原因 = 显示检查结果
4. 一键修复 = doctor --repair
5. 修复后验证 = 再次运行 doctor
```

**类比 2：体检中心**

```
OpenClaw doctor = 体检套餐

相似点：
1. 基础体检 = openclaw doctor
2. 全面体检 = openclaw doctor --deep
3. 发现问题 = 检查报告
4. 治疗方案 = 修复建议
5. 复查验证 = 修复后验证
```

**类比 3：电脑管家**

```
OpenClaw doctor = 电脑管家一键修复

相似点：
1. 系统扫描 = 健康检查
2. 发现问题 = 识别配置错误
3. 一键修复 = doctor --repair
4. 修复报告 = 详细的修复日志
5. 定期检查 = 升级后运行 doctor
```

---

## 六、常见问题

### 6.1 doctor 检查哪些内容？

**检查清单**：

1. **配置文件**：
   - 语法正确性
   - 配置值有效性
   - 版本兼容性
   - 必需字段完整性

2. **服务状态**：
   - Gateway 运行状态
   - 通道连接状态
   - 权限配置

3. **环境**：
   - Node.js 版本
   - 依赖完整性
   - 环境变量冲突

4. **深度检查**（--deep）：
   - Provider 认证
   - 通道连接性
   - 数据库状态

**来源**: `reference/source_doctor_command.md`

### 6.2 doctor 可以修复哪些问题？

**可修复问题**：

| 问题类型 | 修复方式 | 示例 |
|---------|---------|------|
| 配置版本不兼容 | 自动迁移 | v1 → v2 |
| 无效配置键 | 自动删除 | 删除 gateway.token |
| 缺失配置键 | 自动添加 | 添加 gateway.auth.token |
| 配置值错误 | 自动修正 | 修正 gateway.mode |
| 环境变量冲突 | 提示清除 | 提示清除 launchctl 变量 |

**不可修复问题**：
- 网络连接问题
- API 密钥无效
- 权限不足（需要手动授权）
- 端口被占用（需要手动停止其他进程）

**来源**: `reference/source_doctor_command.md`

### 6.3 doctor 会修改什么？

**修改内容**：

1. **配置文件**：
   - 迁移配置版本
   - 删除无效键
   - 添加缺失键
   - 修正配置值

2. **备份文件**：
   - 创建 `~/.openclaw/openclaw.json.bak`

3. **不会修改**：
   - 环境变量（只提示）
   - 系统权限（只检查）
   - 网络设置（只检查）
   - 其他文件（只读取）

**来源**: `reference/source_doctor_command.md`

### 6.4 什么时候应该运行 doctor？

**推荐时机**：

1. **升级后**：
   ```bash
   npm install -g openclaw@latest
   openclaw doctor --repair
   ```

2. **安装后**：
   ```bash
   openclaw onboard
   openclaw doctor
   ```

3. **遇到问题时**：
   ```bash
   # Gateway 无法启动
   openclaw doctor --repair
   ```

4. **定期检查**：
   ```bash
   # 每周或每月
   openclaw doctor
   ```

5. **修改配置后**：
   ```bash
   # 手动编辑配置后
   openclaw doctor
   ```

**来源**: `reference/search_doctor_command.md`

---

## 七、最佳实践

### 7.1 升级工作流

```bash
# 1. 备份当前配置
cp ~/.openclaw/openclaw.json ~/.openclaw/openclaw.json.backup

# 2. 升级 OpenClaw
npm install -g openclaw@latest

# 3. 运行 doctor
openclaw doctor --repair

# 4. 验证
openclaw gateway status
openclaw status

# 5. 如果有问题，恢复备份
# cp ~/.openclaw/openclaw.json.backup ~/.openclaw/openclaw.json
```

**来源**: `reference/search_doctor_command.md`

### 7.2 故障排查工作流

```bash
# 1. 快速诊断
openclaw status
openclaw doctor

# 2. 如果发现问题，自动修复
openclaw doctor --repair

# 3. 如果问题仍然存在，深度诊断
openclaw doctor --deep

# 4. 查看日志
openclaw logs --follow

# 5. 如果仍然无法解决，查看备份
ls -la ~/.openclaw/*.bak
```

**来源**: `reference/source_troubleshooting_main.md`

### 7.3 自动化脚本

```bash
#!/bin/bash
# openclaw-health-check.sh

# 定期健康检查脚本
echo "Running OpenClaw health check..."

# 运行 doctor
openclaw doctor --non-interactive

# 检查退出码
if [ $? -ne 0 ]; then
  echo "Issues found! Running repair..."
  openclaw doctor --repair --non-interactive

  # 发送通知
  echo "OpenClaw health check failed and repaired" | mail -s "OpenClaw Alert" admin@example.com
fi

echo "Health check complete!"
```

**来源**: `reference/search_doctor_command.md`

---

## 八、故障排查

### 8.1 doctor 本身失败

**问题**：运行 `openclaw doctor` 失败

**可能原因**：
1. Node.js 版本过低
2. 配置文件严重损坏
3. 权限不足

**解决方法**：
```bash
# 1. 检查 Node.js 版本
node --version
# 应该 >= 22.12.0

# 2. 检查配置文件
cat ~/.openclaw/openclaw.json
# 如果无法读取，恢复备份或重新初始化

# 3. 检查权限
ls -la ~/.openclaw/
# 确保有读写权限
```

**来源**: `reference/source_doctor_command.md`

### 8.2 repair 后问题仍然存在

**问题**：运行 `openclaw doctor --repair` 后问题仍然存在

**可能原因**：
1. 问题不在配置文件中
2. 环境变量覆盖
3. 服务未重启

**解决方法**：
```bash
# 1. 检查环境变量
env | grep OPENCLAW

# 2. 重启 Gateway
openclaw gateway restart

# 3. 深度诊断
openclaw doctor --deep

# 4. 查看日志
openclaw logs --follow
```

**来源**: `reference/source_gateway_troubleshooting.md`

---

## 九、进阶使用

### 9.1 CI/CD 集成

```yaml
# .github/workflows/openclaw-health.yml
name: OpenClaw Health Check

on:
  schedule:
    - cron: '0 0 * * *'  # 每天运行

jobs:
  health-check:
    runs-on: ubuntu-latest
    steps:
      - name: Install OpenClaw
        run: npm install -g openclaw@latest

      - name: Run Doctor
        run: openclaw doctor --non-interactive

      - name: Repair if needed
        if: failure()
        run: openclaw doctor --repair --non-interactive

      - name: Notify on failure
        if: failure()
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.name,
              title: 'OpenClaw Health Check Failed',
              body: 'Automated health check detected issues'
            })
```

**来源**: `reference/search_doctor_command.md`

### 9.2 监控集成

```typescript
// 集成到监控系统
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

async function checkOpenClawHealth(): Promise<HealthStatus> {
  try {
    const { stdout, stderr } = await execAsync('openclaw doctor --non-interactive');

    if (stderr) {
      // 发送告警
      await sendAlert({
        level: 'warning',
        message: 'OpenClaw health check found issues',
        details: stderr
      });

      // 尝试自动修复
      await execAsync('openclaw doctor --repair --non-interactive');
    }

    return { status: 'healthy', output: stdout };
  } catch (error) {
    // 发送严重告警
    await sendAlert({
      level: 'critical',
      message: 'OpenClaw health check failed',
      details: error.message
    });

    return { status: 'unhealthy', error: error.message };
  }
}

// 定期运行
setInterval(checkOpenClawHealth, 60 * 60 * 1000); // 每小时
```

**来源**: `reference/search_doctor_command.md`

---

## 十、总结

### 10.1 核心要点

1. **doctor 是自动化工具**：不需要手动检查和修复
2. **升级后必须运行**：确保配置兼容性
3. **修复前自动备份**：安全可靠
4. **提供详细报告**：清楚知道修复了什么
5. **支持深度诊断**：全面检查系统健康

### 10.2 使用建议

- ✅ 升级后立即运行 `openclaw doctor --repair`
- ✅ 遇到问题先运行 `openclaw doctor`
- ✅ 定期运行健康检查
- ✅ 集成到 CI/CD 流程
- ❌ 不要跳过 doctor 直接修改配置

### 10.3 下一步学习

- **核心概念 2**：日志系统与查看
- **核心概念 3**：网关状态检查
- **实战场景 5**：配置验证与修复

---

**文档版本**: v1.0
**最后更新**: 2026-02-24
**维护者**: Claude Code
**基于**: OpenClaw 2026.2.22

**数据来源**：
- `reference/source_doctor_command.md`
- `reference/search_doctor_command.md`
- `reference/source_troubleshooting_main.md`
- `reference/source_gateway_troubleshooting.md`
