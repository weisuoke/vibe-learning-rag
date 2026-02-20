# 实战代码 02：Anthropic 配置

> **实战目标**：完整配置 Anthropic Claude API，掌握 API Key 和 OAuth 两种认证方式，实现模型切换和成本优化

---

## 一、API Key 配置

### 1.1 获取 API Key

```bash
#!/bin/bash
# get-anthropic-api-key.sh - 获取 Anthropic API Key 指南

echo "📝 获取 Anthropic API Key"
echo ""
echo "步骤："
echo "1. 访问 https://console.anthropic.com/"
echo "2. 登录或注册账户"
echo "3. 进入 API Keys 页面"
echo "4. 点击 'Create Key'"
echo "5. 复制生成的 API Key (sk-ant-api03-...)"
echo ""
echo "⚠️  注意："
echo "- API Key 只显示一次，请妥善保存"
echo "- 不要将 API Key 提交到 Git"
echo "- 定期轮换 API Key（建议 90 天）"
```

### 1.2 环境变量配置

```bash
#!/bin/bash
# setup-anthropic-env.sh - 配置 Anthropic 环境变量

# 临时设置（当前会话）
export ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# 永久设置（Bash）
echo 'export ANTHROPIC_API_KEY=sk-ant-api03-your-key-here' >> ~/.bashrc
source ~/.bashrc

# 永久设置（Zsh）
echo 'export ANTHROPIC_API_KEY=sk-ant-api03-your-key-here' >> ~/.zshrc
source ~/.zshrc

# 验证设置
echo "✅ ANTHROPIC_API_KEY 已设置"
echo "Key 前缀: ${ANTHROPIC_API_KEY:0:15}..."
```

### 1.3 使用密钥管理工具

```bash
#!/bin/bash
# setup-anthropic-keychain.sh - 使用 macOS Keychain 存储 API Key

# 存储到 Keychain
security add-generic-password \
  -a "$USER" \
  -s "anthropic-api-key" \
  -w "sk-ant-api03-your-key-here"

echo "✅ API Key 已存储到 Keychain"

# 配置 auth.json
mkdir -p ~/.pi/agent
cat > ~/.pi/agent/auth.json << 'EOF'
{
  "anthropic": {
    "type": "api_key",
    "key": "!security find-generic-password -ws 'anthropic-api-key'"
  }
}
EOF

chmod 600 ~/.pi/agent/auth.json
echo "✅ auth.json 已配置"
```

### 1.4 测试 API Key

```bash
#!/bin/bash
# test-anthropic-api-key.sh - 测试 Anthropic API Key

echo "🔍 测试 Anthropic API Key..."

response=$(curl -s https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{
    "model": "claude-opus-4",
    "max_tokens": 10,
    "messages": [{"role": "user", "content": "Hi"}]
  }')

if echo "$response" | grep -q "content"; then
    echo "✅ API Key 有效"
    echo "响应: $(echo $response | jq -r '.content[0].text')"
else
    echo "❌ API Key 无效"
    echo "错误: $(echo $response | jq -r '.error.message')"
fi
```

---

## 二、OAuth 配置

### 2.1 OAuth 登录脚本

```bash
#!/bin/bash
# anthropic-oauth-login.sh - Anthropic OAuth 登录

echo "🔐 Anthropic OAuth 登录"
echo ""
echo "要求："
echo "- Claude Pro 或 Claude Max 订阅"
echo "- 浏览器访问权限"
echo ""

read -p "按 Enter 键启动 Pi 并登录..." -r

# 启动 Pi
pi << 'EOF'
/login
EOF

# 验证登录
if [ -f ~/.pi/agent/auth.json ]; then
    if grep -q '"anthropic"' ~/.pi/agent/auth.json; then
        echo "✅ OAuth 登录成功"
        echo "Token 已保存到 ~/.pi/agent/auth.json"
    else
        echo "❌ OAuth 登录失败"
    fi
else
    echo "❌ auth.json 文件不存在"
fi
```

### 2.2 检查 OAuth Token

```bash
#!/bin/bash
# check-anthropic-oauth.sh - 检查 Anthropic OAuth Token

echo "🔍 检查 Anthropic OAuth Token"
echo ""

if [ ! -f ~/.pi/agent/auth.json ]; then
    echo "❌ auth.json 文件不存在"
    exit 1
fi

# 检查 Anthropic OAuth 配置
if grep -q '"anthropic"' ~/.pi/agent/auth.json; then
    echo "✅ Anthropic OAuth 配置存在"

    # 提取 Token 信息（不显示完整 Token）
    token_type=$(jq -r '.anthropic.type' ~/.pi/agent/auth.json)
    echo "认证类型: $token_type"

    if [ "$token_type" = "oauth" ]; then
        expires_at=$(jq -r '.anthropic.expiresAt' ~/.pi/agent/auth.json)
        current_time=$(date +%s)000

        if [ "$expires_at" -gt "$current_time" ]; then
            echo "✅ Token 有效"
            expires_date=$(date -r $((expires_at / 1000)) '+%Y-%m-%d %H:%M:%S')
            echo "过期时间: $expires_date"
        else
            echo "⚠️  Token 已过期"
            echo "请重新登录: pi -> /login"
        fi
    fi
else
    echo "❌ 未找到 Anthropic OAuth 配置"
fi
```

---

## 三、模型配置

### 3.1 可用模型列表

```typescript
// anthropic-models.ts - Anthropic 可用模型配置

interface AnthropicModel {
  id: string;
  name: string;
  contextWindow: number;
  maxTokens: number;
  cost: {
    input: number;    // per 1M tokens
    output: number;   // per 1M tokens
    cacheRead: number;
    cacheWrite: number;
  };
  capabilities: string[];
}

const anthropicModels: AnthropicModel[] = [
  {
    id: 'claude-opus-4',
    name: 'Claude Opus 4',
    contextWindow: 200000,
    maxTokens: 16384,
    cost: {
      input: 15,
      output: 75,
      cacheRead: 1.5,
      cacheWrite: 18.75
    },
    capabilities: ['text', 'image', 'extended-thinking']
  },
  {
    id: 'claude-sonnet-4',
    name: 'Claude Sonnet 4',
    contextWindow: 200000,
    maxTokens: 16384,
    cost: {
      input: 3,
      output: 15,
      cacheRead: 0.3,
      cacheWrite: 3.75
    },
    capabilities: ['text', 'image']
  },
  {
    id: 'claude-haiku-4',
    name: 'Claude Haiku 4',
    contextWindow: 200000,
    maxTokens: 16384,
    cost: {
      input: 0.25,
      output: 1.25,
      cacheRead: 0.025,
      cacheWrite: 0.3125
    },
    capabilities: ['text', 'image']
  }
];

// 导出模型信息
export { anthropicModels, type AnthropicModel };
```

### 3.2 模型选择脚本

```bash
#!/bin/bash
# select-anthropic-model.sh - 选择 Anthropic 模型

echo "🤖 选择 Anthropic 模型"
echo ""
echo "可用模型："
echo "1. claude-opus-4   - 最强能力，适合复杂任务"
echo "2. claude-sonnet-4 - 平衡性能，适合日常开发"
echo "3. claude-haiku-4  - 快速响应，适合简单任务"
echo ""

read -p "选择模型 (1-3): " choice

case $choice in
    1)
        model="claude-opus-4"
        ;;
    2)
        model="claude-sonnet-4"
        ;;
    3)
        model="claude-haiku-4"
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo "✅ 已选择: $model"
echo ""
echo "启动 Pi:"
pi --provider anthropic --model $model
```

---

## 四、成本优化

### 4.1 成本计算器

```typescript
// anthropic-cost-calculator.ts - Anthropic 成本计算器

interface TokenUsage {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
}

interface CostBreakdown {
  inputCost: number;
  outputCost: number;
  cacheReadCost: number;
  cacheWriteCost: number;
  total: number;
}

function calculateCost(
  model: string,
  usage: TokenUsage
): CostBreakdown {
  const rates = {
    'claude-opus-4': {
      input: 15,
      output: 75,
      cacheRead: 1.5,
      cacheWrite: 18.75
    },
    'claude-sonnet-4': {
      input: 3,
      output: 15,
      cacheRead: 0.3,
      cacheWrite: 3.75
    },
    'claude-haiku-4': {
      input: 0.25,
      output: 1.25,
      cacheRead: 0.025,
      cacheWrite: 0.3125
    }
  };

  const rate = rates[model];
  if (!rate) {
    throw new Error(`Unknown model: ${model}`);
  }

  const inputCost = (usage.input / 1_000_000) * rate.input;
  const outputCost = (usage.output / 1_000_000) * rate.output;
  const cacheReadCost = (usage.cacheRead / 1_000_000) * rate.cacheRead;
  const cacheWriteCost = (usage.cacheWrite / 1_000_000) * rate.cacheWrite;

  return {
    inputCost,
    outputCost,
    cacheReadCost,
    cacheWriteCost,
    total: inputCost + outputCost + cacheReadCost + cacheWriteCost
  };
}

// 示例使用
const usage: TokenUsage = {
  input: 10000,
  output: 5000,
  cacheRead: 2000,
  cacheWrite: 1000
};

const cost = calculateCost('claude-sonnet-4', usage);
console.log('成本明细:');
console.log(`输入: $${cost.inputCost.toFixed(4)}`);
console.log(`输出: $${cost.outputCost.toFixed(4)}`);
console.log(`缓存读取: $${cost.cacheReadCost.toFixed(4)}`);
console.log(`缓存写入: $${cost.cacheWriteCost.toFixed(4)}`);
console.log(`总计: $${cost.total.toFixed(4)}`);
```

### 4.2 成本优化策略

```bash
#!/bin/bash
# optimize-anthropic-cost.sh - Anthropic 成本优化策略

cat << 'EOF'
💰 Anthropic 成本优化策略

1. 模型选择策略
   - 简单任务: Haiku 4 ($0.25/1M input)
   - 日常开发: Sonnet 4 ($3/1M input)
   - 复杂任务: Opus 4 ($15/1M input)

2. Prompt 缓存
   - 启用 Prompt Caching 可节省 90% 成本
   - 缓存读取: 仅 10% 的输入成本
   - 适合重复使用的上下文

3. 上下文管理
   - 使用 Compaction 压缩长会话
   - 定期创建新会话
   - 避免不必要的文件引用

4. 批量处理
   - 合并多个小任务
   - 减少 API 调用次数
   - 使用 --print 模式处理批量任务

5. 监控成本
   - 使用 /session 查看当前成本
   - 设置每日预算提醒
   - 定期审查使用情况

示例：
# 简单任务用 Haiku
pi --model claude-haiku-4 "格式化这个文件"

# 复杂任务用 Opus
pi --model claude-opus-4 "设计系统架构"

# 启用 Prompt Caching
# Pi 自动启用，无需额外配置
EOF
```

---

## 五、项目配置

### 5.1 项目级 Anthropic 配置

```json
// .pi/settings.json - 项目级 Anthropic 配置
{
  "provider": "anthropic",
  "model": "claude-sonnet-4",
  "thinkingLevel": "normal",
  "scopedModels": [
    "claude-haiku-4",
    "claude-sonnet-4",
    "claude-opus-4"
  ],
  "compaction": {
    "enabled": true,
    "strategy": "auto",
    "threshold": 0.8
  }
}
```

### 5.2 AGENTS.md 配置

```markdown
<!-- AGENTS.md - 项目上下文配置 -->
# 项目上下文

## Anthropic 模型使用指南

### 模型选择
- **Haiku 4**: 代码格式化、简单重构、文档查询
- **Sonnet 4**: 功能开发、Bug 修复、代码审查
- **Opus 4**: 架构设计、复杂算法、深度重构

### 成本控制
- 默认使用 Sonnet 4
- 简单任务切换到 Haiku 4
- 复杂任务才使用 Opus 4

### 快捷键
- `Ctrl+P`: 循环切换模型
- `Ctrl+L`: 打开模型选择器
```

---

## 六、实战示例

### 6.1 完整工作流示例

```bash
#!/bin/bash
# anthropic-workflow-example.sh - Anthropic 完整工作流示例

echo "🚀 Anthropic 工作流示例"
echo ""

# 1. 配置 API Key
export ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
echo "✅ API Key 已配置"

# 2. 启动 Pi（默认 Sonnet 4）
echo "📝 启动 Pi (Sonnet 4)..."
pi --model claude-sonnet-4 << 'EOF'
# 创建一个简单的 TypeScript 函数
创建一个 calculateSum 函数，接受数字数组，返回总和
EOF

# 3. 切换到 Haiku 4 处理简单任务
echo "📝 切换到 Haiku 4..."
pi --model claude-haiku-4 << 'EOF'
# 格式化代码
格式化 src/utils.ts
EOF

# 4. 切换到 Opus 4 处理复杂任务
echo "📝 切换到 Opus 4..."
pi --model claude-opus-4 << 'EOF'
# 设计系统架构
设计一个可扩展的插件系统架构
EOF

echo "✅ 工作流完成"
```

### 6.2 成本对比示例

```bash
#!/bin/bash
# anthropic-cost-comparison.sh - Anthropic 成本对比

cat << 'EOF'
💰 成本对比示例

任务: 生成 1000 行代码
输入: 5K tokens
输出: 20K tokens

Haiku 4:
  输入: $0.00125
  输出: $0.025
  总计: $0.02625

Sonnet 4:
  输入: $0.015
  输出: $0.30
  总计: $0.315

Opus 4:
  输入: $0.075
  输出: $1.50
  总计: $1.575

节省: 使用 Haiku 比 Opus 节省 98.3%
EOF
```

---

## 七、故障排查

### 7.1 常见问题诊断

```bash
#!/bin/bash
# troubleshoot-anthropic.sh - Anthropic 故障排查

echo "🔧 Anthropic 故障排查"
echo ""

# 问题 1: API Key 无效
echo "1️⃣ 检查 API Key:"
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌ ANTHROPIC_API_KEY 未设置"
else
    echo "✅ ANTHROPIC_API_KEY 已设置"
    echo "Key 前缀: ${ANTHROPIC_API_KEY:0:15}..."
fi
echo ""

# 问题 2: 速率限制
echo "2️⃣ 检查速率限制:"
echo "Tier 1 限制:"
echo "- 50 RPM (每分钟请求数)"
echo "- 40K TPM (每分钟 Token 数)"
echo "- 200K TPD (每天 Token 数)"
echo ""

# 问题 3: 模型不可用
echo "3️⃣ 检查模型可用性:"
available_models=("claude-opus-4" "claude-sonnet-4" "claude-haiku-4")
for model in "${available_models[@]}"; do
    echo "- $model: ✅"
done
echo ""

# 问题 4: OAuth Token 过期
echo "4️⃣ 检查 OAuth Token:"
if [ -f ~/.pi/agent/auth.json ]; then
    if grep -q '"anthropic"' ~/.pi/agent/auth.json; then
        token_type=$(jq -r '.anthropic.type' ~/.pi/agent/auth.json)
        if [ "$token_type" = "oauth" ]; then
            echo "OAuth Token 存在"
            echo "如果遇到认证错误，请重新登录:"
            echo "  pi"
            echo "  /logout"
            echo "  /login"
        fi
    fi
else
    echo "⚠️  auth.json 不存在"
fi
```

---

## 八、总结

### 8.1 配置检查清单

- [ ] API Key 已获取并配置
- [ ] 环境变量已设置或 OAuth 已登录
- [ ] API Key 已测试验证
- [ ] 模型选择策略已确定
- [ ] 成本优化策略已实施
- [ ] 项目配置已完成

### 8.2 快速参考

```bash
# 配置 API Key
export ANTHROPIC_API_KEY=sk-ant-api03-...

# 测试 API Key
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-opus-4","max_tokens":10,"messages":[{"role":"user","content":"Hi"}]}'

# 使用特定模型
pi --model claude-sonnet-4

# OAuth 登录
pi
/login

# 查看成本
pi
/session
```

---

**参考资料:**
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Anthropic Console](https://console.anthropic.com/)
- [Pi Providers Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/providers.md)

**文档版本:** v1.0 (2026-02-18)
