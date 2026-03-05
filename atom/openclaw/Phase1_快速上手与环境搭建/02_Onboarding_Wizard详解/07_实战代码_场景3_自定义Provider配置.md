# 实战代码 场景3：自定义 Provider 配置

**本文档演示如何配置自定义 LLM Provider，包括 OpenAI 兼容端点、Anthropic 兼容端点和自动检测。**

---

## 场景概述

**目标**：配置自定义 LLM Provider，连接到非官方端点或自托管模型服务。

**适用场景**：
- 使用代理服务（如 OpenAI 代理、Cloudflare AI Gateway）
- 连接自托管模型（Ollama、vLLM、LocalAI）
- 使用第三方 API 服务（Azure OpenAI、AWS Bedrock）
- 企业内部 LLM 服务

**支持的兼容性模式**：
- **OpenAI 兼容**：支持 OpenAI API 格式的端点
- **Anthropic 兼容**：支持 Anthropic API 格式的端点
- **Unknown（自动检测）**：自动检测端点兼容性

---

## 自定义 Provider 基础

### 配置参数

| 参数 | 说明 | 必需 |
|------|------|------|
| `--auth-choice` | 设置为 `custom-api-key` | 是 |
| `--custom-base-url` | 自定义 API 端点 URL | 是 |
| `--custom-model-id` | 自定义模型 ID | 是 |
| `--custom-api-key` | 自定义 API 密钥 | 条件 |
| `--custom-compatibility` | 兼容性模式（`openai`/`anthropic`/`unknown`） | 是 |

### 基本语法

```bash
openclaw onboard --non-interactive \
  --auth-choice custom-api-key \
  --custom-base-url "https://api.example.com/v1" \
  --custom-model-id "model-name" \
  --custom-api-key "$CUSTOM_API_KEY" \
  --custom-compatibility openai
```

---

## OpenAI 兼容端点配置

### 场景 1：OpenAI 代理服务

```bash
#!/bin/bash
# setup-openai-proxy.sh

set -e

# 配置参数
PROXY_BASE_URL="https://api.openai-proxy.com/v1"
PROXY_API_KEY="${OPENAI_PROXY_KEY}"
MODEL_ID="gpt-4"

# 配置 OpenClaw
openclaw onboard --non-interactive \
  --auth-choice custom-api-key \
  --custom-base-url "$PROXY_BASE_URL" \
  --custom-model-id "$MODEL_ID" \
  --custom-api-key "$PROXY_API_KEY" \
  --custom-compatibility openai

# 验证配置
openclaw config get agents.defaults.model.primary
openclaw status
```

**使用场景**：
- 使用国内 OpenAI 代理服务
- 通过企业代理访问 OpenAI
- 使用 Cloudflare AI Gateway

### 场景 2：Azure OpenAI

```bash
#!/bin/bash
# setup-azure-openai.sh

set -e

# Azure OpenAI 配置
AZURE_ENDPOINT="https://your-resource.openai.azure.com"
AZURE_API_KEY="${AZURE_OPENAI_KEY}"
AZURE_DEPLOYMENT="gpt-4"
AZURE_API_VERSION="2024-02-15-preview"

# 构建完整 URL
CUSTOM_BASE_URL="${AZURE_ENDPOINT}/openai/deployments/${AZURE_DEPLOYMENT}"

# 配置 OpenClaw
openclaw onboard --non-interactive \
  --auth-choice custom-api-key \
  --custom-base-url "$CUSTOM_BASE_URL" \
  --custom-model-id "$AZURE_DEPLOYMENT" \
  --custom-api-key "$AZURE_API_KEY" \
  --custom-compatibility openai

# 验证
openclaw status
```

**说明**：
- Azure OpenAI 使用不同的 URL 格式
- 需要指定 API 版本
- Deployment 名称作为模型 ID

### 场景 3：Ollama 本地模型

```bash
#!/bin/bash
# setup-ollama.sh

set -e

# Ollama 配置
OLLAMA_BASE_URL="http://localhost:11434/v1"
OLLAMA_MODEL="llama3:8b"

# 配置 OpenClaw（Ollama 不需要 API Key）
openclaw onboard --non-interactive \
  --auth-choice custom-api-key \
  --custom-base-url "$OLLAMA_BASE_URL" \
  --custom-model-id "$OLLAMA_MODEL" \
  --custom-compatibility openai

# 验证
openclaw status
```

**前置条件**：

```bash
# 安装 Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 拉取模型
ollama pull llama3:8b

# 启动 Ollama 服务
ollama serve
```

### 场景 4：vLLM 自托管

```bash
#!/bin/bash
# setup-vllm.sh

set -e

# vLLM 配置
VLLM_BASE_URL="http://localhost:8000/v1"
VLLM_MODEL="meta-llama/Llama-3-8b-chat-hf"

# 配置 OpenClaw
openclaw onboard --non-interactive \
  --auth-choice custom-api-key \
  --custom-base-url "$VLLM_BASE_URL" \
  --custom-model-id "$VLLM_MODEL" \
  --custom-compatibility openai

# 验证
openclaw status
```

**启动 vLLM 服务**：

```bash
# 安装 vLLM
pip install vllm

# 启动 OpenAI 兼容服务器
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3-8b-chat-hf \
  --port 8000
```

---

## Anthropic 兼容端点配置

### 场景 5：Anthropic 代理服务

```bash
#!/bin/bash
# setup-anthropic-proxy.sh

set -e

# 配置参数
PROXY_BASE_URL="https://api.anthropic-proxy.com"
PROXY_API_KEY="${ANTHROPIC_PROXY_KEY}"
MODEL_ID="claude-sonnet-4-5"

# 配置 OpenClaw
openclaw onboard --non-interactive \
  --auth-choice custom-api-key \
  --custom-base-url "$PROXY_BASE_URL" \
  --custom-model-id "$MODEL_ID" \
  --custom-api-key "$PROXY_API_KEY" \
  --custom-compatibility anthropic

# 验证
openclaw status
```

### 场景 6：AWS Bedrock (Anthropic)

```bash
#!/bin/bash
# setup-bedrock-anthropic.sh

set -e

# AWS Bedrock 配置
BEDROCK_ENDPOINT="https://bedrock-runtime.us-east-1.amazonaws.com"
AWS_ACCESS_KEY="${AWS_ACCESS_KEY_ID}"
AWS_SECRET_KEY="${AWS_SECRET_ACCESS_KEY}"
MODEL_ID="anthropic.claude-3-sonnet-20240229-v1:0"

# 配置 OpenClaw
openclaw onboard --non-interactive \
  --auth-choice custom-api-key \
  --custom-base-url "$BEDROCK_ENDPOINT" \
  --custom-model-id "$MODEL_ID" \
  --custom-api-key "$AWS_ACCESS_KEY:$AWS_SECRET_KEY" \
  --custom-compatibility anthropic

# 验证
openclaw status
```

**说明**：
- AWS Bedrock 需要 AWS 凭证
- 模型 ID 格式：`anthropic.claude-3-sonnet-20240229-v1:0`
- API Key 格式：`ACCESS_KEY:SECRET_KEY`

---

## Unknown Provider（自动检测）

### 场景 7：自动检测兼容性

```bash
#!/bin/bash
# setup-unknown-provider.sh

set -e

# 未知 Provider 配置
CUSTOM_BASE_URL="https://api.unknown-provider.com/v1"
CUSTOM_API_KEY="${UNKNOWN_API_KEY}"
CUSTOM_MODEL="custom-model-v1"

# 使用 unknown 模式自动检测
openclaw onboard --non-interactive \
  --auth-choice custom-api-key \
  --custom-base-url "$CUSTOM_BASE_URL" \
  --custom-model-id "$CUSTOM_MODEL" \
  --custom-api-key "$CUSTOM_API_KEY" \
  --custom-compatibility unknown

# 验证
openclaw status
```

**自动检测逻辑**：

```
1. 尝试 OpenAI 格式请求
   ↓
2. 如果失败，尝试 Anthropic 格式
   ↓
3. 如果都失败，报错
```

---

## 完整配置示例

### 示例 1：多 Provider 环境

```bash
#!/bin/bash
# setup-multi-provider.sh

set -e

# 环境选择
ENVIRONMENT="${1:-development}"

case "$ENVIRONMENT" in
  development)
    # 开发环境：使用 Ollama 本地模型
    BASE_URL="http://localhost:11434/v1"
    MODEL_ID="llama3:8b"
    API_KEY=""
    COMPATIBILITY="openai"
    ;;
  staging)
    # 测试环境：使用 OpenAI 代理
    BASE_URL="https://api.openai-proxy.com/v1"
    MODEL_ID="gpt-4"
    API_KEY="$STAGING_OPENAI_KEY"
    COMPATIBILITY="openai"
    ;;
  production)
    # 生产环境：使用官方 Anthropic
    BASE_URL="https://api.anthropic.com"
    MODEL_ID="claude-sonnet-4-5"
    API_KEY="$PROD_ANTHROPIC_KEY"
    COMPATIBILITY="anthropic"
    ;;
  *)
    echo "Unknown environment: $ENVIRONMENT"
    exit 1
    ;;
esac

echo "Configuring OpenClaw for $ENVIRONMENT..."

# 配置 OpenClaw
if [ -z "$API_KEY" ]; then
  # 无需 API Key（如 Ollama）
  openclaw onboard --non-interactive \
    --auth-choice custom-api-key \
    --custom-base-url "$BASE_URL" \
    --custom-model-id "$MODEL_ID" \
    --custom-compatibility "$COMPATIBILITY"
else
  # 需要 API Key
  openclaw onboard --non-interactive \
    --auth-choice custom-api-key \
    --custom-base-url "$BASE_URL" \
    --custom-model-id "$MODEL_ID" \
    --custom-api-key "$API_KEY" \
    --custom-compatibility "$COMPATIBILITY"
fi

# 验证
openclaw status
echo "OpenClaw configured for $ENVIRONMENT successfully!"
```

**使用方法**：

```bash
# 配置开发环境
./setup-multi-provider.sh development

# 配置生产环境
./setup-multi-provider.sh production
```

### 示例 2：带健康检查的配置

```bash
#!/bin/bash
# setup-with-health-check.sh

set -e

# 配置参数
CUSTOM_BASE_URL="${CUSTOM_BASE_URL:-https://api.example.com/v1}"
CUSTOM_MODEL_ID="${CUSTOM_MODEL_ID:-model-name}"
CUSTOM_API_KEY="${CUSTOM_API_KEY}"
CUSTOM_COMPATIBILITY="${CUSTOM_COMPATIBILITY:-openai}"

# 健康检查函数
check_endpoint() {
  local url=$1
  echo "Checking endpoint: $url"

  if curl -s -f -o /dev/null "$url/models"; then
    echo "✓ Endpoint is reachable"
    return 0
  else
    echo "✗ Endpoint is not reachable"
    return 1
  fi
}

# 检查端点可达性
if ! check_endpoint "$CUSTOM_BASE_URL"; then
  echo "Error: Custom endpoint is not reachable"
  exit 1
fi

# 配置 OpenClaw
openclaw onboard --non-interactive \
  --auth-choice custom-api-key \
  --custom-base-url "$CUSTOM_BASE_URL" \
  --custom-model-id "$CUSTOM_MODEL_ID" \
  --custom-api-key "$CUSTOM_API_KEY" \
  --custom-compatibility "$CUSTOM_COMPATIBILITY"

# 验证配置
openclaw status
openclaw health

echo "Configuration completed successfully!"
```

---

## 配置文件格式

### 手动编辑配置文件

```json5
// ~/.openclaw/openclaw.json
{
  agents: {
    defaults: {
      workspace: "~/.openclaw/workspace",
      model: {
        primary: "custom/model-name",  // 格式：custom/<model-id>
        provider: {
          type: "custom",
          baseUrl: "https://api.example.com/v1",
          apiKey: "your-api-key",
          compatibility: "openai",  // openai | anthropic
        },
      },
    },
  },
  env: {
    CUSTOM_API_KEY: "your-api-key",
  },
}
```

### 使用 CLI 修改配置

```bash
# 修改 Base URL
openclaw config set agents.defaults.model.provider.baseUrl "https://new-api.com/v1"

# 修改模型 ID
openclaw config set agents.defaults.model.primary "custom/new-model"

# 修改兼容性模式
openclaw config set agents.defaults.model.provider.compatibility "anthropic"

# 查看配置
openclaw config get agents.defaults.model
```

---

## 故障排查

### 问题 1：端点不可达

**症状**：

```
Error: Failed to connect to custom endpoint
```

**排查步骤**：

```bash
# 1. 检查端点可达性
curl -v https://api.example.com/v1/models

# 2. 检查 DNS 解析
nslookup api.example.com

# 3. 检查防火墙
telnet api.example.com 443

# 4. 检查代理设置
echo $HTTP_PROXY
echo $HTTPS_PROXY
```

### 问题 2：API Key 无效

**症状**：

```
Error: Invalid API key
```

**解决方案**：

```bash
# 1. 验证 API Key
curl -H "Authorization: Bearer $CUSTOM_API_KEY" \
  https://api.example.com/v1/models

# 2. 检查环境变量
echo $CUSTOM_API_KEY

# 3. 重新配置
openclaw onboard --non-interactive \
  --auth-choice custom-api-key \
  --custom-base-url "https://api.example.com/v1" \
  --custom-model-id "model-name" \
  --custom-api-key "new-api-key" \
  --custom-compatibility openai
```

### 问题 3：兼容性模式错误

**症状**：

```
Error: Unsupported API format
```

**解决方案**：

```bash
# 尝试不同的兼容性模式

# 1. 尝试 OpenAI 兼容
openclaw config set agents.defaults.model.provider.compatibility "openai"

# 2. 尝试 Anthropic 兼容
openclaw config set agents.defaults.model.provider.compatibility "anthropic"

# 3. 使用自动检测
openclaw onboard --non-interactive \
  --auth-choice custom-api-key \
  --custom-base-url "https://api.example.com/v1" \
  --custom-model-id "model-name" \
  --custom-api-key "$CUSTOM_API_KEY" \
  --custom-compatibility unknown
```

### 问题 4：模型 ID 不存在

**症状**：

```
Error: Model not found: model-name
```

**解决方案**：

```bash
# 1. 列出可用模型
curl -H "Authorization: Bearer $CUSTOM_API_KEY" \
  https://api.example.com/v1/models

# 2. 使用正确的模型 ID
openclaw config set agents.defaults.model.primary "custom/correct-model-id"

# 3. 重启 Gateway
openclaw gateway restart
```

---

## 高级配置

### 配置多个自定义 Provider

```json5
// ~/.openclaw/openclaw.json
{
  agents: {
    defaults: {
      model: {
        primary: "custom/model-a",
      },
    },
    "agent-b": {
      model: {
        primary: "custom/model-b",
        provider: {
          type: "custom",
          baseUrl: "https://api-b.example.com/v1",
          apiKey: "${CUSTOM_API_KEY_B}",
          compatibility: "openai",
        },
      },
    },
  },
}
```

### 使用环境变量

```bash
# .env 文件
CUSTOM_BASE_URL=https://api.example.com/v1
CUSTOM_MODEL_ID=model-name
CUSTOM_API_KEY=your-api-key
CUSTOM_COMPATIBILITY=openai

# 加载环境变量
source .env

# 配置 OpenClaw
openclaw onboard --non-interactive \
  --auth-choice custom-api-key \
  --custom-base-url "$CUSTOM_BASE_URL" \
  --custom-model-id "$CUSTOM_MODEL_ID" \
  --custom-api-key "$CUSTOM_API_KEY" \
  --custom-compatibility "$CUSTOM_COMPATIBILITY"
```

### 配置请求超时

```json5
{
  agents: {
    defaults: {
      model: {
        provider: {
          type: "custom",
          baseUrl: "https://api.example.com/v1",
          timeout: 60000,  // 60 秒
        },
      },
    },
  },
}
```

---

## 最佳实践

### 1. 使用环境变量管理凭证

```bash
# ✅ 推荐
export CUSTOM_API_KEY="your-key"
openclaw onboard --non-interactive \
  --auth-choice custom-api-key \
  --custom-base-url "https://api.example.com/v1" \
  --custom-model-id "model-name" \
  --custom-compatibility openai

# ❌ 不推荐
openclaw onboard --non-interactive \
  --auth-choice custom-api-key \
  --custom-base-url "https://api.example.com/v1" \
  --custom-model-id "model-name" \
  --custom-api-key "hardcoded-key" \
  --custom-compatibility openai
```

### 2. 验证端点可达性

```bash
#!/bin/bash
# validate-endpoint.sh

CUSTOM_BASE_URL="https://api.example.com/v1"

# 检查端点
if curl -s -f -o /dev/null "$CUSTOM_BASE_URL/models"; then
  echo "✓ Endpoint is valid"
  # 继续配置
  openclaw onboard --non-interactive \
    --auth-choice custom-api-key \
    --custom-base-url "$CUSTOM_BASE_URL" \
    --custom-model-id "model-name" \
    --custom-compatibility openai
else
  echo "✗ Endpoint is invalid"
  exit 1
fi
```

### 3. 使用配置模板

```bash
# config-template.json
{
  "baseUrl": "https://api.example.com/v1",
  "modelId": "model-name",
  "compatibility": "openai"
}

# 从模板加载
BASE_URL=$(jq -r '.baseUrl' config-template.json)
MODEL_ID=$(jq -r '.modelId' config-template.json)
COMPATIBILITY=$(jq -r '.compatibility' config-template.json)

openclaw onboard --non-interactive \
  --auth-choice custom-api-key \
  --custom-base-url "$BASE_URL" \
  --custom-model-id "$MODEL_ID" \
  --custom-api-key "$CUSTOM_API_KEY" \
  --custom-compatibility "$COMPATIBILITY"
```

### 4. 日志记录

```bash
#!/bin/bash
# setup-with-logging.sh

LOG_FILE="/var/log/openclaw-custom-setup.log"

exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "[$(date)] Configuring custom provider..."

openclaw onboard --non-interactive \
  --auth-choice custom-api-key \
  --custom-base-url "$CUSTOM_BASE_URL" \
  --custom-model-id "$CUSTOM_MODEL_ID" \
  --custom-api-key "$CUSTOM_API_KEY" \
  --custom-compatibility "$CUSTOM_COMPATIBILITY"

echo "[$(date)] Configuration completed"
```

---

## 总结

自定义 Provider 配置的核心要点：

1. **兼容性模式**：选择正确的兼容性模式（OpenAI/Anthropic/Unknown）
2. **端点验证**：配置前验证端点可达性
3. **API Key 管理**：使用环境变量管理敏感信息
4. **错误处理**：实现健康检查和错误处理
5. **多环境支持**：支持开发、测试、生产环境切换

完成这些步骤后，你就可以连接任何 OpenAI 或 Anthropic 兼容的 LLM 服务了！🎉
