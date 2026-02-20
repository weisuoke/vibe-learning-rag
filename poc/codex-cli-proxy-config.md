# Codex CLI 自定义代理服务配置指南

> 完整的配置参考：如何为 Codex CLI 配置自定义代理/中转服务

**调研日期**: 2026-02-18
**版本**: v1.0
**状态**: 完成

---

## 概述

### 适用场景

本指南适用于以下需求：

- **国内中转服务**：使用国内 API 代理服务加速访问
- **企业内部代理**：通过企业内部网关访问 OpenAI API
- **自定义端点**：使用兼容 OpenAI API 的第三方服务
- **多代理切换**：在不同项目中使用不同的代理服务

### 核心价值

- ✅ 灵活配置：支持全局和项目级两种配置方式
- ✅ 安全管理：通过环境变量管理 API key
- ✅ 多代理支持：可配置多个代理服务并快速切换
- ✅ 项目隔离：不同项目可使用不同的代理配置

### 官方文档参考

- [配置参考](https://developers.openai.com/codex/config-reference) - 完整的配置选项说明
- [认证方式](https://developers.openai.com/codex/auth) - API key 认证配置

---

## 目录

1. [核心配置原理](#核心配置原理)
2. [方案一：全局配置](#方案一全局配置)
3. [方案二：项目级配置](#方案二项目级配置)
4. [方案三：混合配置](#方案三混合配置)
5. [常见配置场景](#常见配置场景)
6. [环境变量管理最佳实践](#环境变量管理最佳实践)
7. [验证配置](#验证配置)
8. [故障排查](#故障排查)
9. [关键文件路径总结](#关键文件路径总结)

---

## 核心配置原理

### config.toml 工作机制

Codex CLI 通过 `config.toml` 文件管理模型提供商配置：

```
配置加载优先级（从高到低）：
1. 项目级配置：<project>/.codex/config.toml
2. 全局配置：~/.codex/config.toml
3. 默认配置：内置的 OpenAI 配置
```

### 配置文件结构

```toml
# 指定默认使用的模型提供商
model_provider = "custom_proxy"

# 指定默认使用的模型
model = "gpt-4"

# 定义模型提供商
[model_providers.custom_proxy]
name = "Custom Proxy Service"
base_url = "https://your-proxy-service.com/v1"
env_key = "CUSTOM_PROXY_API_KEY"
requires_openai_auth = false
```

### 关键配置项说明

| 配置项 | 说明 | 示例 |
|--------|------|------|
| `model_provider` | 默认使用的提供商 ID | `"custom_proxy"` |
| `model` | 默认使用的模型名称 | `"gpt-4"` |
| `[model_providers.<id>]` | 定义一个提供商 | `[model_providers.my_proxy]` |
| `name` | 提供商显示名称 | `"My Proxy Service"` |
| `base_url` | API 端点 URL | `"https://api.example.com/v1"` |
| `env_key` | 环境变量名（存储 API key） | `"MY_PROXY_API_KEY"` |
| `requires_openai_auth` | 是否使用 OpenAI 认证 | `false`（自定义代理设为 false） |

---

## 方案一：全局配置

### 适用场景

- 个人开发环境
- 所有项目使用同一个代理服务
- 快速配置，立即生效

### 配置步骤

#### 1. 创建全局配置文件

```bash
# 创建配置目录（如果不存在）
mkdir -p ~/.codex

# 创建配置文件
touch ~/.codex/config.toml
```

#### 2. 编辑配置文件

```bash
# 使用你喜欢的编辑器
vim ~/.codex/config.toml
# 或
code ~/.codex/config.toml
```

#### 3. 添加配置内容

```toml
# ~/.codex/config.toml

# 默认使用自定义代理
model_provider = "custom_proxy"
model = "gpt-4"

# 定义自定义代理提供商
[model_providers.custom_proxy]
name = "Custom Proxy Service"
base_url = "https://your-proxy-service.com/v1"
env_key = "CUSTOM_PROXY_API_KEY"
requires_openai_auth = false
```

#### 4. 设置环境变量

**方法 1：添加到 shell 配置文件（推荐）**

```bash
# 对于 bash 用户
echo 'export CUSTOM_PROXY_API_KEY="your_api_key_here"' >> ~/.bashrc
source ~/.bashrc

# 对于 zsh 用户
echo 'export CUSTOM_PROXY_API_KEY="your_api_key_here"' >> ~/.zshrc
source ~/.zshrc
```

**方法 2：临时设置（仅当前会话）**

```bash
export CUSTOM_PROXY_API_KEY="your_api_key_here"
codex exec "your command"
```

#### 5. 验证配置

```bash
# 测试配置是否生效
codex exec "Hello, test my proxy configuration"
```

### 完整示例

```toml
# ~/.codex/config.toml

model_provider = "openai_proxy"
model = "gpt-4"

[model_providers.openai_proxy]
name = "OpenAI Proxy Service"
base_url = "https://api.openai-proxy.com/v1"
env_key = "OPENAI_PROXY_KEY"
requires_openai_auth = false
```

```bash
# 环境变量设置
export OPENAI_PROXY_KEY="sk-proj-xxxxxxxxxxxxx"
```

---

## 方案二：项目级配置

### 适用场景

- 不同项目使用不同的代理服务
- 团队协作，需要项目级配置
- 需要项目隔离的 API key

### 配置步骤

#### 1. 创建项目配置目录

```bash
# 在项目根目录下
cd /path/to/your/project

# 创建 .codex 目录
mkdir -p .codex
```

#### 2. 创建项目配置文件

```bash
# 创建配置文件
touch .codex/config.toml
```

#### 3. 添加项目配置

```toml
# <project>/.codex/config.toml

model_provider = "project_proxy"
model = "gpt-4"

[model_providers.project_proxy]
name = "Project Specific Proxy"
base_url = "https://project-proxy.example.com/v1"
env_key = "PROJECT_PROXY_API_KEY"
requires_openai_auth = false
```

#### 4. 信任项目配置

**重要**：Codex CLI 需要用户明确信任项目配置才会加载。

```bash
# 在项目目录下运行
codex trust

# 或者在首次运行时，Codex 会提示是否信任
codex exec "test command"
# 提示：Do you trust this project's .codex/config.toml? [y/N]
```

#### 5. 设置项目环境变量

**方法 1：使用 .env 文件（推荐）**

```bash
# 创建 .env 文件
echo "PROJECT_PROXY_API_KEY=your_project_api_key" > .env

# 使用 direnv 自动加载（需要先安装 direnv）
echo "dotenv" > .envrc
direnv allow
```

**方法 2：项目级 shell 脚本**

```bash
# 创建 setup-env.sh
cat > setup-env.sh << 'EOF'
#!/bin/bash
export PROJECT_PROXY_API_KEY="your_project_api_key"
EOF

# 使用前先 source
source setup-env.sh
codex exec "command"
```

### 项目配置覆盖规则

项目配置会**完全覆盖**全局配置中的相同字段：

```toml
# 全局配置 ~/.codex/config.toml
model_provider = "global_proxy"
model = "gpt-4"

# 项目配置 <project>/.codex/config.toml
model_provider = "project_proxy"
# model 字段未指定，会继承全局的 "gpt-4"
```

### 安全注意事项

```bash
# 将 .env 添加到 .gitignore
echo ".env" >> .gitignore
echo ".envrc" >> .gitignore

# .codex/config.toml 可以提交（不包含敏感信息）
# API key 通过环境变量管理，不提交到版本控制
```

---

## 方案三：混合配置

### 适用场景

- 全局配置作为默认
- 特定项目使用特定代理
- 灵活切换不同配置

### 配置策略

#### 策略 1：全局默认 + 项目覆盖

```toml
# 全局配置 ~/.codex/config.toml
model_provider = "default_proxy"
model = "gpt-4"

[model_providers.default_proxy]
name = "Default Proxy"
base_url = "https://default-proxy.com/v1"
env_key = "DEFAULT_PROXY_KEY"
requires_openai_auth = false

[model_providers.special_proxy]
name = "Special Proxy"
base_url = "https://special-proxy.com/v1"
env_key = "SPECIAL_PROXY_KEY"
requires_openai_auth = false
```

```toml
# 项目配置 <project>/.codex/config.toml
# 仅覆盖 model_provider，使用全局定义的 special_proxy
model_provider = "special_proxy"
```

#### 策略 2：使用 Profiles 切换

```toml
# ~/.codex/config.toml

# 默认配置
model_provider = "proxy_a"
model = "gpt-4"

# 定义多个代理
[model_providers.proxy_a]
name = "Proxy A"
base_url = "https://proxy-a.com/v1"
env_key = "PROXY_A_KEY"
requires_openai_auth = false

[model_providers.proxy_b]
name = "Proxy B"
base_url = "https://proxy-b.com/v1"
env_key = "PROXY_B_KEY"
requires_openai_auth = false

# 定义 profile 用于快速切换
[profiles.use_proxy_b]
model_provider = "proxy_b"

[profiles.use_proxy_a]
model_provider = "proxy_a"
```

**使用 profile**：

```bash
# 使用 proxy_b
codex --profile use_proxy_b exec "command"

# 使用 proxy_a（默认）
codex exec "command"
```

---

## 常见配置场景

### 场景 1：国内中转服务

```toml
# ~/.codex/config.toml

model_provider = "china_proxy"
model = "gpt-4"

[model_providers.china_proxy]
name = "China OpenAI Proxy"
base_url = "https://api.openai-proxy.cn/v1"
env_key = "CHINA_PROXY_API_KEY"
requires_openai_auth = false
```

```bash
# 环境变量
export CHINA_PROXY_API_KEY="your_china_proxy_key"
```

### 场景 2：企业内部代理

```toml
# <project>/.codex/config.toml

model_provider = "corporate_gateway"
model = "gpt-4-turbo"

[model_providers.corporate_gateway]
name = "Corporate API Gateway"
base_url = "https://api-gateway.company.internal/openai/v1"
env_key = "CORPORATE_API_KEY"
requires_openai_auth = false
```

```bash
# 企业 SSO 认证后获取的 token
export CORPORATE_API_KEY="corp-token-xxxxx"
```

### 场景 3：多代理切换（开发/生产）

```toml
# ~/.codex/config.toml

# 默认使用开发代理
model_provider = "dev_proxy"
model = "gpt-4"

# 开发环境代理
[model_providers.dev_proxy]
name = "Development Proxy"
base_url = "https://dev-api.example.com/v1"
env_key = "DEV_PROXY_KEY"
requires_openai_auth = false

# 生产环境代理
[model_providers.prod_proxy]
name = "Production Proxy"
base_url = "https://api.example.com/v1"
env_key = "PROD_PROXY_KEY"
requires_openai_auth = false

# Profile：切换到生产环境
[profiles.production]
model_provider = "prod_proxy"
model = "gpt-4-turbo"
```

**使用方式**：

```bash
# 开发环境（默认）
codex exec "test feature"

# 生产环境
codex --profile production exec "deploy task"
```

---

## 环境变量管理最佳实践

### 方法 1：使用 direnv（推荐）

**安装 direnv**：

```bash
# macOS
brew install direnv

# Linux
sudo apt install direnv  # Debian/Ubuntu
sudo yum install direnv  # CentOS/RHEL

# 添加到 shell 配置
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc  # bash
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc    # zsh
```

**项目配置**：

```bash
# 在项目根目录
cd /path/to/project

# 创建 .envrc
cat > .envrc << 'EOF'
# 加载 .env 文件
dotenv

# 或直接定义环境变量
export PROJECT_PROXY_API_KEY="your_key_here"
EOF

# 允许 direnv 加载
direnv allow
```

**优势**：
- 自动加载/卸载环境变量
- 进入项目目录自动生效
- 离开项目目录自动清理

### 方法 2：使用 .env 文件

```bash
# 创建 .env 文件
cat > .env << 'EOF'
CUSTOM_PROXY_API_KEY=your_api_key_here
PROJECT_PROXY_API_KEY=project_specific_key
EOF

# 使用 dotenv 工具加载
# 方式 1：使用 dotenv CLI
npm install -g dotenv-cli
dotenv codex exec "command"

# 方式 2：在脚本中加载
cat > run-codex.sh << 'EOF'
#!/bin/bash
set -a
source .env
set +a
codex exec "$@"
EOF
chmod +x run-codex.sh
./run-codex.sh "your command"
```

### 方法 3：密钥管理工具

**使用 1Password CLI**：

```bash
# 安装 1Password CLI
brew install --cask 1password-cli

# 从 1Password 读取密钥
export CUSTOM_PROXY_API_KEY=$(op read "op://vault/item/field")

# 或在命令中直接使用
op run -- codex exec "command"
```

**使用 AWS Secrets Manager**：

```bash
# 从 AWS Secrets Manager 获取密钥
export CUSTOM_PROXY_API_KEY=$(aws secretsmanager get-secret-value \
  --secret-id codex-proxy-key \
  --query SecretString \
  --output text)
```

### 安全注意事项

**必须做**：
- ✅ 将 `.env` 添加到 `.gitignore`
- ✅ 使用环境变量而非硬编码 API key
- ✅ 定期轮换 API key
- ✅ 使用最小权限原则

**禁止做**：
- ❌ 将 API key 提交到版本控制
- ❌ 在配置文件中硬编码密钥
- ❌ 在日志中打印 API key
- ❌ 共享个人 API key

---

## 验证配置

### 检查配置文件

```bash
# 查看当前使用的配置
codex config show

# 查看配置文件路径
codex config path

# 验证配置语法
codex config validate
```

### 测试代理连接

```bash
# 简单测试
codex exec "Hello, world"

# 详细输出（查看请求详情）
codex exec --verbose "Test proxy connection"

# JSON 输出（便于调试）
codex exec --json "Test" | jq
```

### 查看日志

```bash
# 查看 Codex 日志
tail -f ~/.codex/logs/codex.log

# 查看最近的错误
grep ERROR ~/.codex/logs/codex.log | tail -20
```

### 验证环境变量

```bash
# 检查环境变量是否设置
echo $CUSTOM_PROXY_API_KEY

# 验证环境变量是否被 Codex 读取
codex exec --verbose "test" 2>&1 | grep -i "api.*key"
```

---

## 故障排查

### 问题 1：配置不生效

**症状**：
- 修改了配置文件，但 Codex 仍使用旧配置
- 提示找不到 API key

**排查步骤**：

```bash
# 1. 确认配置文件路径
codex config path

# 2. 验证配置文件语法
codex config validate

# 3. 检查项目是否已信任
cd /path/to/project
codex trust --status

# 4. 重新信任项目
codex trust

# 5. 检查环境变量
echo $CUSTOM_PROXY_API_KEY
env | grep PROXY
```

**解决方案**：

```bash
# 清除缓存
rm -rf ~/.codex/cache

# 重启 Codex
codex restart

# 重新加载配置
codex config reload
```

### 问题 2：API 请求失败

**症状**：
- 提示连接超时
- 返回 401/403 错误
- 提示 API key 无效

**排查步骤**：

```bash
# 1. 测试代理 URL 是否可访问
curl -I https://your-proxy-service.com/v1/models

# 2. 验证 API key 格式
echo $CUSTOM_PROXY_API_KEY | wc -c  # 检查长度

# 3. 使用 curl 测试完整请求
curl https://your-proxy-service.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $CUSTOM_PROXY_API_KEY" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# 4. 查看详细错误日志
codex exec --verbose --debug "test" 2>&1 | tee debug.log
```

**解决方案**：

```toml
# 检查 base_url 是否正确（必须以 /v1 结尾）
[model_providers.custom_proxy]
base_url = "https://api.example.com/v1"  # ✅ 正确
# base_url = "https://api.example.com"   # ❌ 错误

# 检查 env_key 是否匹配
env_key = "CUSTOM_PROXY_API_KEY"  # 必须与环境变量名完全一致
```

### 问题 3：项目配置未加载

**症状**：
- 项目配置存在，但使用的是全局配置
- 提示 "Project config not trusted"

**排查步骤**：

```bash
# 1. 检查项目配置文件是否存在
ls -la .codex/config.toml

# 2. 检查信任状态
codex trust --status

# 3. 查看配置加载顺序
codex config show --verbose
```

**解决方案**：

```bash
# 信任项目配置
codex trust

# 或在命令中强制使用项目配置
codex exec --use-project-config "command"

# 检查 .codex 目录权限
chmod 755 .codex
chmod 644 .codex/config.toml
```

---

## 关键文件路径总结

### 配置文件

| 文件路径 | 说明 | 优先级 |
|---------|------|--------|
| `~/.codex/config.toml` | 全局配置 | 低 |
| `<project>/.codex/config.toml` | 项目配置 | 高 |
| `~/.codex/trusted-projects.json` | 受信任的项目列表 | - |

### 环境变量文件

| 文件路径 | 说明 | 工具 |
|---------|------|------|
| `~/.bashrc` / `~/.zshrc` | Shell 配置文件 | bash/zsh |
| `.env` | 项目环境变量 | dotenv |
| `.envrc` | direnv 配置 | direnv |

### 日志文件

| 文件路径 | 说明 |
|---------|------|
| `~/.codex/logs/codex.log` | Codex 运行日志 |
| `~/.codex/logs/api.log` | API 请求日志 |
| `~/.codex/cache/` | 缓存目录 |

### 快速参考命令

```bash
# 查看全局配置
cat ~/.codex/config.toml

# 查看项目配置
cat .codex/config.toml

# 查看环境变量
env | grep -i proxy | grep -i key

# 查看信任的项目
cat ~/.codex/trusted-projects.json

# 查看最近日志
tail -50 ~/.codex/logs/codex.log
```

---

## 总结

### 配置方案选择

| 场景 | 推荐方案 | 配置位置 |
|------|---------|---------|
| 个人开发，单一代理 | 全局配置 | `~/.codex/config.toml` |
| 多项目，不同代理 | 项目级配置 | `<project>/.codex/config.toml` |
| 需要快速切换 | Profiles | 全局配置 + profiles |
| 团队协作 | 项目级配置 + .env | 项目配置 + 环境变量 |

### 最佳实践清单

- ✅ 使用环境变量管理 API key
- ✅ 将 `.env` 添加到 `.gitignore`
- ✅ 使用 direnv 自动管理环境变量
- ✅ 定期验证配置和连接
- ✅ 为不同环境配置不同的 profiles
- ✅ 记录代理服务的文档和联系方式

### 下一步

1. ✅ 选择适合的配置方案
2. ✅ 创建配置文件
3. ✅ 设置环境变量
4. ✅ 验证配置
5. ✅ 测试实际使用

---

**文档结束**
