# 实战代码 03：OpenAI 配置

> **实战目标**：完整配置 OpenAI GPT API，掌握 API Key 和 OAuth (Codex) 认证，实现多模型切换和成本优化

---

## 一、API Key 配置

### 1.1 获取 OpenAI API Key

```bash
#!/bin/bash
# get-openai-api-key.sh - 获取 OpenAI API Key 指南

cat << 'EOF'
📝 获取 OpenAI API Key

步骤：
1. 访问 https://platform.openai.com/api-keys
2. 登录 OpenAI 账户
3. 点击 "Create new secret key"
4. 命名密钥（如 "pi-coding-agent"）
5. 复制生成的 API Key (sk-proj-...)

⚠️  注意：
- API Key 只显示一次，请妥善保存
- 不要将 API Key 提交到 Git
- 设置使用限额避免超支
- 定期轮换 API Key（建议 90 天）

💰 计费说明：
- 按使用量付费（Pay-as-you-go）
- 可设置月度预算限制
- 查看使用情况：https://platform.openai.com/usage
EOF
```

### 1.2 环境变量配置

```bash
#!/bin/bash
# setup-openai-env.sh - 配置 OpenAI 环境变量

# 临时设置
export OPENAI_API_KEY=sk-proj-your-key-here

# 永久设置（Bash）
echo 'export OPENAI_API_KEY=sk-proj-your-key-here' >> ~/.bashrc
source ~/.bashrc

# 永久设置（Zsh）
echo 'export OPENAI_API_KEY=sk-proj-your-key-here' >> ~/.zshrc
source ~/.zshrc

# 验证设置
if [ -n "$OPENAI_API_KEY" ]; then
    echo "✅ OPENAI_API_KEY 已设置"
    echo "Key 前缀: ${OPENAI_API_KEY:0:15}..."
else
    echo "❌ OPENAI_API_KEY 未设置"
fi
```

### 1.3 测试 API Key

```bash
#!/bin/bash
# test-openai-api-key.sh - 测试 OpenAI API Key

echo "🔍 测试 OpenAI API Key..."

response=$(curl -s https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 10
  }')

if echo "$response" | grep -q "choices"; then
    echo "✅ API Key 有效"
    echo "响应: $(echo $response | jq -r '.choices[0].message.content')"
    echo "使用 Tokens: $(echo $response | jq -r '.usage.total_tokens')"
else
    echo "❌ API Key 无效"
    echo "错误: $(echo $response | jq -r '.error.message')"
fi
```

---

## 二、OAuth (Codex) 配置

### 2.1 ChatGPT Plus/Pro 订阅登录

```bash
#!/bin/bash
# openai-codex-login.sh - OpenAI Codex OAuth 登录

cat << 'EOF'
🔐 OpenAI Codex OAuth 登录

要求：
- ChatGPT Plus ($20/月) 或 ChatGPT Pro ($200/月) 订阅
- 仅供个人使用（非商业用途）

步骤：
1. 启动 Pi
2. 输入 /login
3. 选择 "OpenAI ChatGPT Plus/Pro (Codex)"
4. 浏览器打开授权页面
5. 使用 ChatGPT 账户登录
6. 授权访问
7. 返回终端，认证完成

⚠️  注意：
- Codex 仅供个人使用
- 生产环境请使用 OpenAI Platform API
- Pro 订阅提供无限 o1 访问
EOF

read -p "按 Enter 键启动 Pi 并登录..." -r
pi
```

### 2.2 检查 OAuth Token

```bash
#!/bin/bash
# check-openai-oauth.sh - 检查 OpenAI OAuth Token

echo "🔍 检查 OpenAI OAuth Token"

if [ ! -f ~/.pi/agent/auth.json ]; then
    echo "❌ auth.json 文件不存在"
    exit 1
fi

if grep -q '"openai"' ~/.pi/agent/auth.json; then
    echo "✅ OpenAI OAuth 配置存在"

    token_type=$(jq -r '.openai.type' ~/.pi/agent/auth.json)
    echo "认证类型: $token_type"

    if [ "$token_type" = "oauth" ]; then
        expires_at=$(jq -r '.openai.expiresAt' ~/.pi/agent/auth.json)
        current_time=$(date +%s)000

        if [ "$expires_at" -gt "$current_time" ]; then
            echo "✅ Token 有效"
        else
            echo "⚠️  Token 已过期，请重新登录"
        fi
    fi
else
    echo "❌ 未找到 OpenAI OAuth 配置"
fi
```

---

## 三、模型配置

### 3.1 可用模型列表

```typescript
// openai-models.ts - OpenAI 可用模型配置

interface OpenAIModel {
  id: string;
  name: string;
  contextWindow: number;
  maxTokens: number;
  cost: {
    input: number;    // per 1M tokens
    output: number;   // per 1M tokens
  };
  capabilities: string[];
}

const openaiModels: OpenAIModel[] = [
  {
    id: 'gpt-4o',
    name: 'GPT-4o',
    contextWindow: 128000,
    maxTokens: 16384,
    cost: {
      input: 2.5,
      output: 10
    },
    capabilities: ['text', 'image', 'fast']
  },
  {
    id: 'o1',
    name: 'o1',
    contextWindow: 200000,
    maxTokens: 100000,
    cost: {
      input: 15,
      output: 60
    },
    capabilities: ['text', 'reasoning']
  },
  {
    id: 'o3-mini',
    name: 'o3-mini',
    contextWindow: 200000,
    maxTokens: 100000,
    cost: {
      input: 1.1,
      output: 4.4
    },
    capabilities: ['text', 'reasoning', 'fast']
  },
  {
    id: 'gpt-4-turbo',
    name: 'GPT-4 Turbo',
    contextWindow: 128000,
    maxTokens: 4096,
    cost: {
      input: 10,
      output: 30
    },
    capabilities: ['text', 'image']
  }
];

export { openaiModels, type OpenAIModel };
```

### 3.2 模型选择脚本

```bash
#!/bin/bash
# select-openai-model.sh - 选择 OpenAI 模型

cat << 'EOF'
🤖 选择 OpenAI 模型

可用模型：
1. gpt-4o      - 快速多模态，适合日常开发
2. o1          - 深度推理，适合复杂问题
3. o3-mini     - 快速推理，适合中等任务
4. gpt-4-turbo - 平衡性能，适合通用任务
EOF

read -p "选择模型 (1-4): " choice

case $choice in
    1) model="gpt-4o" ;;
    2) model="o1" ;;
    3) model="o3-mini" ;;
    4) model="gpt-4-turbo" ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo "✅ 已选择: $model"
pi --provider openai --model $model
```

---

## 四、成本优化

### 4.1 成本计算器

```typescript
// openai-cost-calculator.ts - OpenAI 成本计算器

interface TokenUsage {
  input: number;
  output: number;
}

interface CostBreakdown {
  inputCost: number;
  outputCost: number;
  total: number;
}

function calculateOpenAICost(
  model: string,
  usage: TokenUsage
): CostBreakdown {
  const rates = {
    'gpt-4o': { input: 2.5, output: 10 },
    'o1': { input: 15, output: 60 },
    'o3-mini': { input: 1.1, output: 4.4 },
    'gpt-4-turbo': { input: 10, output: 30 }
  };

  const rate = rates[model];
  if (!rate) {
    throw new Error(`Unknown model: ${model}`);
  }

  const inputCost = (usage.input / 1_000_000) * rate.input;
  const outputCost = (usage.output / 1_000_000) * rate.output;

  return {
    inputCost,
    outputCost,
    total: inputCost + outputCost
  };
}

// 示例
const usage: TokenUsage = { input: 10000, output: 5000 };
const cost = calculateOpenAICost('gpt-4o', usage);
console.log(`总成本: $${cost.total.toFixed(4)}`);
```

### 4.2 成本优化策略

```bash
#!/bin/bash
# optimize-openai-cost.sh - OpenAI 成本优化策略

cat << 'EOF'
💰 OpenAI 成本优化策略

1. 模型选择
   - 日常开发: gpt-4o ($2.5/1M input)
   - 快速推理: o3-mini ($1.1/1M input)
   - 深度推理: o1 ($15/1M input)

2. 上下文管理
   - 启用 Compaction 压缩长会话
   - 避免重复发送大文件
   - 使用文件引用而非完整内容

3. 批量处理
   - 合并多个小任务
   - 使用 --print 模式批量处理

4. 监控使用
   - 查看 /session 了解当前成本
   - 设置月度预算限制
   - 定期审查使用情况

5. 缓存策略
   - OpenAI 自动缓存重复内容
   - 利用 Prompt Caching 节省成本

示例：
# 日常任务用 gpt-4o
pi --model gpt-4o "重构这个函数"

# 复杂推理用 o1
pi --model o1 "设计算法解决这个问题"

# 快速任务用 o3-mini
pi --model o3-mini "解释这段代码"
EOF
```

---

## 五、项目配置

### 5.1 项目级 OpenAI 配置

```json
// .pi/settings.json - 项目级 OpenAI 配置
{
  "provider": "openai",
  "model": "gpt-4o",
  "thinkingLevel": "normal",
  "scopedModels": [
    "gpt-4o",
    "o3-mini",
    "o1"
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
<!-- AGENTS.md - OpenAI 使用指南 -->
# 项目上下文

## OpenAI 模型使用指南

### 模型选择
- **gpt-4o**: 日常开发、代码生成、快速响应
- **o3-mini**: 中等复杂度推理、成本敏感场景
- **o1**: 复杂算法、深度推理、数学问题

### 成本控制
- 默认使用 gpt-4o
- 推理任务使用 o3-mini
- 复杂问题才使用 o1

### 快捷键
- `Ctrl+P`: 循环切换模型
- `Ctrl+L`: 打开模型选择器
```

---

## 六、实战示例

### 6.1 完整工作流示例

```bash
#!/bin/bash
# openai-workflow-example.sh - OpenAI 完整工作流

echo "🚀 OpenAI 工作流示例"

# 1. 配置 API Key
export OPENAI_API_KEY=sk-proj-your-key-here
echo "✅ API Key 已配置"

# 2. 日常开发（gpt-4o）
echo "📝 使用 gpt-4o 进行日常开发..."
pi --model gpt-4o << 'EOF'
创建一个 React 组件，显示用户列表
EOF

# 3. 快速推理（o3-mini）
echo "📝 使用 o3-mini 进行快速推理..."
pi --model o3-mini << 'EOF'
优化这个排序算法的时间复杂度
EOF

# 4. 深度推理（o1）
echo "📝 使用 o1 进行深度推理..."
pi --model o1 << 'EOF'
设计一个分布式系统的一致性协议
EOF

echo "✅ 工作流完成"
```

### 6.2 成本对比示例

```bash
#!/bin/bash
# openai-cost-comparison.sh - OpenAI 成本对比

cat << 'EOF'
💰 成本对比示例

任务: 生成 1000 行代码
输入: 5K tokens
输出: 20K tokens

gpt-4o:
  输入: $0.0125
  输出: $0.20
  总计: $0.2125

o3-mini:
  输入: $0.0055
  输出: $0.088
  总计: $0.0935

o1:
  输入: $0.075
  输出: $1.20
  总计: $1.275

节省: 使用 o3-mini 比 o1 节省 92.7%
EOF
```

---

## 七、故障排查

### 7.1 常见问题诊断

```bash
#!/bin/bash
# troubleshoot-openai.sh - OpenAI 故障排查

echo "🔧 OpenAI 故障排查"

# 问题 1: API Key 无效
echo "1️⃣ 检查 API Key:"
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY 未设置"
else
    echo "✅ OPENAI_API_KEY 已设置"
    echo "Key 前缀: ${OPENAI_API_KEY:0:15}..."
fi

# 问题 2: 速率限制
echo ""
echo "2️⃣ 速率限制:"
echo "Tier 1 限制:"
echo "- 500 RPM (每分钟请求数)"
echo "- 30K TPM (每分钟 Token 数)"
echo "- 200 RPD (每天请求数)"

# 问题 3: 预算限制
echo ""
echo "3️⃣ 检查预算:"
echo "访问 https://platform.openai.com/settings/organization/billing/limits"
echo "设置月度预算限制避免超支"

# 问题 4: 模型不可用
echo ""
echo "4️⃣ 检查模型可用性:"
available_models=("gpt-4o" "o1" "o3-mini" "gpt-4-turbo")
for model in "${available_models[@]}"; do
    echo "- $model: ✅"
done
```

---

## 八、总结

### 8.1 配置检查清单

- [ ] API Key 已获取并配置
- [ ] 环境变量已设置或 OAuth 已登录
- [ ] API Key 已测试验证
- [ ] 模型选择策略已确定
- [ ] 成本优化策略已实施
- [ ] 预算限制已设置
- [ ] 项目配置已完成

### 8.2 快速参考

```bash
# 配置 API Key
export OPENAI_API_KEY=sk-proj-...

# 测试 API Key
curl https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hi"}],"max_tokens":10}'

# 使用特定模型
pi --model gpt-4o

# OAuth 登录
pi
/login

# 查看成本
pi
/session
```

---

**参考资料:**
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [OpenAI Platform](https://platform.openai.com/)
- [Pi Providers Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/providers.md)

**文档版本:** v1.0 (2026-02-18)
