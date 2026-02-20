# 实战代码 04：其他 Provider 配置

> **实战目标**：配置 xAI (Grok)、Google Gemini、Groq、DeepSeek 等其他主流 Provider，实现多 Provider 协作

---

## 一、xAI (Grok) 配置

### 1.1 获取 xAI API Key

```bash
#!/bin/bash
# setup-xai.sh - 配置 xAI Grok

cat << 'EOF'
📝 获取 xAI API Key

步骤:
1. 访问 https://console.x.ai/
2. 登录 X (Twitter) 账户
3. 创建 API Key
4. 复制 API Key (xai-...)

模型:
- grok-2-latest: 最新 Grok 2 模型
- grok-vision-beta: 支持图片的 Grok Vision
EOF

# 配置环境变量
export XAI_API_KEY=xai-your-key-here
echo 'export XAI_API_KEY=xai-your-key-here' >> ~/.bashrc

# 测试
curl https://api.x.ai/v1/chat/completions \
  -H "Authorization: Bearer $XAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"grok-2-latest","messages":[{"role":"user","content":"Hi"}],"max_tokens":10}'
```

### 1.2 使用 xAI

```bash
#!/bin/bash
# use-xai.sh - 使用 xAI Grok

# 启动 Pi with Grok
pi --provider xai --model grok-2-latest

# 或在交互模式中切换
pi
/model
# 选择 xAI -> grok-2-latest
```

---

## 二、Google Gemini 配置

### 2.1 API Key 方式

```bash
#!/bin/bash
# setup-gemini.sh - 配置 Google Gemini

cat << 'EOF'
📝 获取 Gemini API Key

步骤:
1. 访问 https://makersuite.google.com/app/apikey
2. 创建 API Key
3. 复制 API Key

模型:
- gemini-2.0-flash-exp: 最新 Gemini 2.0 Flash
- gemini-exp-1206: 实验性模型
EOF

# 配置
export GEMINI_API_KEY=your-gemini-key
echo 'export GEMINI_API_KEY=your-gemini-key' >> ~/.bashrc

# 使用
pi --provider google --model gemini-2.0-flash-exp
```

### 2.2 OAuth 方式 (免费)

```bash
#!/bin/bash
# gemini-oauth.sh - Gemini CLI OAuth 登录

cat << 'EOF'
🔐 Gemini CLI OAuth 登录 (免费)

特点:
- 完全免费
- 使用任何 Google 账户
- 有速率限制

步骤:
1. pi
2. /login
3. 选择 "Google Gemini CLI"
4. 浏览器授权
EOF

pi
```

---

## 三、Groq 配置

### 3.1 获取 Groq API Key

```bash
#!/bin/bash
# setup-groq.sh - 配置 Groq

cat << 'EOF'
📝 获取 Groq API Key

步骤:
1. 访问 https://console.groq.com/
2. 注册账户
3. 创建 API Key
4. 复制 API Key (gsk_...)

特点:
- 超快推理速度
- 免费额度
- 支持多种开源模型

模型:
- llama-3.3-70b-versatile: Llama 3.3 70B
- deepseek-r1-distill-llama-70b: DeepSeek R1
- mixtral-8x7b-32768: Mixtral 8x7B
EOF

# 配置
export GROQ_API_KEY=gsk_your-key-here
echo 'export GROQ_API_KEY=gsk_your-key-here' >> ~/.bashrc

# 测试
curl https://api.groq.com/openai/v1/chat/completions \
  -H "Authorization: Bearer $GROQ_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.3-70b-versatile","messages":[{"role":"user","content":"Hi"}],"max_tokens":10}'
```

### 3.2 使用 Groq

```bash
#!/bin/bash
# use-groq.sh - 使用 Groq

# 快速响应场景
pi --provider groq --model llama-3.3-70b-versatile

# 推理场景
pi --provider groq --model deepseek-r1-distill-llama-70b
```

---

## 四、DeepSeek 配置

### 4.1 DeepSeek API

```bash
#!/bin/bash
# setup-deepseek.sh - 配置 DeepSeek

cat << 'EOF'
📝 DeepSeek API 配置

获取 API Key:
1. 访问 https://platform.deepseek.com/
2. 注册账户
3. 创建 API Key

模型:
- deepseek-chat: DeepSeek Chat
- deepseek-coder: DeepSeek Coder (代码专用)
EOF

# 通过 models.json 配置
mkdir -p ~/.pi/agent
cat > ~/.pi/agent/models.json << 'EOF'
{
  "providers": {
    "deepseek": {
      "baseUrl": "https://api.deepseek.com/v1",
      "api": "openai-completions",
      "apiKey": "DEEPSEEK_API_KEY",
      "models": [
        {
          "id": "deepseek-chat",
          "name": "DeepSeek Chat"
        },
        {
          "id": "deepseek-coder",
          "name": "DeepSeek Coder"
        }
      ]
    }
  }
}
EOF

export DEEPSEEK_API_KEY=your-key-here
pi --provider deepseek --model deepseek-coder
```

---

## 五、GitHub Copilot 配置

### 5.1 OAuth 登录

```bash
#!/bin/bash
# setup-github-copilot.sh - 配置 GitHub Copilot

cat << 'EOF'
🔐 GitHub Copilot OAuth 登录

要求:
- GitHub Copilot 订阅 ($10/月)
- GitHub 账户

步骤:
1. pi
2. /login
3. 选择 "GitHub Copilot"
4. 输入 GitHub 域名 (默认 github.com)
5. 浏览器授权

模型:
- gpt-4o: GPT-4o
- claude-sonnet-4: Claude Sonnet 4
- o1-mini: OpenAI o1-mini
EOF

pi
```

### 5.2 启用模型

```bash
#!/bin/bash
# enable-copilot-models.sh - 启用 Copilot 模型

cat << 'EOF'
📝 启用 Copilot 模型

如果遇到 "model not supported" 错误:

1. 打开 VS Code
2. 打开 Copilot Chat
3. 点击模型选择器
4. 选择要使用的模型 (如 GPT-4o)
5. 点击 "Enable"
6. 返回 Pi 重试
EOF
```

---

## 六、多 Provider 配置示例

### 6.1 完整 models.json 配置

```json
// ~/.pi/agent/models.json - 多 Provider 配置
{
  "providers": {
    "groq": {
      "baseUrl": "https://api.groq.com/openai/v1",
      "api": "openai-completions",
      "apiKey": "GROQ_API_KEY",
      "models": [
        {
          "id": "llama-3.3-70b-versatile",
          "name": "Llama 3.3 70B (Groq)",
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        },
        {
          "id": "deepseek-r1-distill-llama-70b",
          "name": "DeepSeek R1 (Groq)",
          "reasoning": true,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    },
    "deepseek": {
      "baseUrl": "https://api.deepseek.com/v1",
      "api": "openai-completions",
      "apiKey": "DEEPSEEK_API_KEY",
      "models": [
        {
          "id": "deepseek-chat",
          "name": "DeepSeek Chat"
        },
        {
          "id": "deepseek-coder",
          "name": "DeepSeek Coder"
        }
      ]
    }
  }
}
```

### 6.2 环境变量配置

```bash
#!/bin/bash
# setup-all-providers.sh - 配置所有 Provider

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
export OPENAI_API_KEY=sk-proj-...

# xAI
export XAI_API_KEY=xai-...

# Google Gemini
export GEMINI_API_KEY=...

# Groq
export GROQ_API_KEY=gsk_...

# DeepSeek
export DEEPSEEK_API_KEY=...

# 保存到 .bashrc
cat >> ~/.bashrc << 'EOF'
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-proj-...
export XAI_API_KEY=xai-...
export GEMINI_API_KEY=...
export GROQ_API_KEY=gsk_...
export DEEPSEEK_API_KEY=...
EOF

source ~/.bashrc
echo "✅ 所有 Provider 已配置"
```

---

## 七、Provider 选择策略

### 7.1 按场景选择

```typescript
// provider-selector.ts - Provider 选择策略

interface ProviderStrategy {
  scenario: string;
  provider: string;
  model: string;
  reason: string;
}

const strategies: ProviderStrategy[] = [
  {
    scenario: '日常开发',
    provider: 'anthropic',
    model: 'claude-sonnet-4',
    reason: '平衡性能和成本'
  },
  {
    scenario: '快速响应',
    provider: 'groq',
    model: 'llama-3.3-70b-versatile',
    reason: '超快推理速度，免费'
  },
  {
    scenario: '代码生成',
    provider: 'deepseek',
    model: 'deepseek-coder',
    reason: '代码专用模型'
  },
  {
    scenario: '复杂推理',
    provider: 'openai',
    model: 'o1',
    reason: '最强推理能力'
  },
  {
    scenario: '图片分析',
    provider: 'xai',
    model: 'grok-vision-beta',
    reason: '支持图片输入'
  },
  {
    scenario: '成本优化',
    provider: 'groq',
    model: 'llama-3.3-70b-versatile',
    reason: '免费使用'
  }
];

// 使用示例
function selectProvider(scenario: string): ProviderStrategy | undefined {
  return strategies.find(s => s.scenario === scenario);
}

const strategy = selectProvider('快速响应');
console.log(`使用 ${strategy?.provider} - ${strategy?.model}`);
```

### 7.2 成本对比

```bash
#!/bin/bash
# compare-providers.sh - Provider 成本对比

cat << 'EOF'
💰 Provider 成本对比 (每 1M tokens)

输入成本:
- Groq (免费):        $0
- Gemini CLI (免费):  $0
- DeepSeek:           $0.14
- Anthropic Haiku:    $0.25
- OpenAI gpt-4o:      $2.5
- Anthropic Sonnet:   $3
- xAI Grok:           $5
- OpenAI o1:          $15
- Anthropic Opus:     $15

输出成本:
- Groq (免费):        $0
- Gemini CLI (免费):  $0
- DeepSeek:           $0.28
- Anthropic Haiku:    $1.25
- OpenAI gpt-4o:      $10
- Anthropic Sonnet:   $15
- xAI Grok:           $15
- OpenAI o1:          $60
- Anthropic Opus:     $75

推荐策略:
1. 开发/测试: Groq (免费)
2. 日常使用: Anthropic Sonnet ($3)
3. 代码专用: DeepSeek Coder ($0.14)
4. 复杂任务: Anthropic Opus ($15)
EOF
```

---

## 八、实战示例

### 8.1 多 Provider 工作流

```bash
#!/bin/bash
# multi-provider-workflow.sh - 多 Provider 工作流

echo "🚀 多 Provider 工作流示例"

# 1. 快速原型 (Groq - 免费)
echo "1️⃣ 使用 Groq 快速原型..."
pi --provider groq --model llama-3.3-70b-versatile << 'EOF'
创建一个简单的 TODO 应用
EOF

# 2. 代码优化 (DeepSeek Coder)
echo "2️⃣ 使用 DeepSeek Coder 优化代码..."
pi --provider deepseek --model deepseek-coder << 'EOF'
优化 TODO 应用的性能
EOF

# 3. 架构设计 (Claude Opus)
echo "3️⃣ 使用 Claude Opus 设计架构..."
pi --provider anthropic --model claude-opus-4 << 'EOF'
设计 TODO 应用的可扩展架构
EOF

# 4. 图片分析 (Grok Vision)
echo "4️⃣ 使用 Grok Vision 分析 UI..."
pi --provider xai --model grok-vision-beta << 'EOF'
分析这个 UI 设计（粘贴截图）
EOF

echo "✅ 工作流完成"
```

### 8.2 Fallback 策略

```bash
#!/bin/bash
# provider-fallback.sh - Provider Fallback 策略

try_providers() {
    local prompt="$1"
    local providers=("groq" "anthropic" "openai")

    for provider in "${providers[@]}"; do
        echo "尝试 $provider..."

        if pi --provider "$provider" --print "$prompt" 2>/dev/null; then
            echo "✅ $provider 成功"
            return 0
        else
            echo "❌ $provider 失败，尝试下一个..."
        fi
    done

    echo "❌ 所有 Provider 都失败"
    return 1
}

# 使用
try_providers "生成一个函数"
```

---

## 九、故障排查

### 9.1 Provider 诊断脚本

```bash
#!/bin/bash
# diagnose-providers.sh - Provider 诊断

echo "🔧 Provider 诊断"

# 检查所有 Provider 的 API Key
providers=(
    "ANTHROPIC_API_KEY:Anthropic"
    "OPENAI_API_KEY:OpenAI"
    "XAI_API_KEY:xAI"
    "GEMINI_API_KEY:Google Gemini"
    "GROQ_API_KEY:Groq"
    "DEEPSEEK_API_KEY:DeepSeek"
)

for provider in "${providers[@]}"; do
    key="${provider%%:*}"
    name="${provider##*:}"

    if [ -n "${!key}" ]; then
        echo "✅ $name: 已配置"
    else
        echo "❌ $name: 未配置"
    fi
done

# 检查 models.json
if [ -f ~/.pi/agent/models.json ]; then
    echo ""
    echo "📝 自定义 Provider:"
    jq -r '.providers | keys[]' ~/.pi/agent/models.json
else
    echo ""
    echo "⚠️  未配置自定义 Provider"
fi
```

---

## 十、总结

### 10.1 Provider 对比表

| Provider | 成本 | 速度 | 能力 | 适用场景 |
|----------|------|------|------|---------|
| Groq | 免费 | 极快 | 中等 | 快速原型、测试 |
| DeepSeek | 极低 | 快 | 代码强 | 代码生成、优化 |
| Gemini CLI | 免费 | 快 | 中等 | 学习、实验 |
| Anthropic | 中等 | 中等 | 强 | 日常开发 |
| OpenAI | 中等 | 快 | 强 | 通用任务 |
| xAI | 中等 | 中等 | 强 | 图片分析 |

### 10.2 配置检查清单

- [ ] 主要 Provider (Anthropic/OpenAI) 已配置
- [ ] 免费 Provider (Groq/Gemini) 已配置
- [ ] 专用 Provider (DeepSeek Coder) 已配置
- [ ] models.json 已创建
- [ ] 所有 API Key 已测试
- [ ] Fallback 策略已设置

### 10.3 快速参考

```bash
# 配置所有 Provider
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-proj-...
export XAI_API_KEY=xai-...
export GEMINI_API_KEY=...
export GROQ_API_KEY=gsk_...

# 使用特定 Provider
pi --provider groq --model llama-3.3-70b-versatile
pi --provider xai --model grok-2-latest
pi --provider deepseek --model deepseek-coder

# OAuth 登录
pi
/login
```

---

**参考资料:**
- [xAI Console](https://console.x.ai/)
- [Google AI Studio](https://makersuite.google.com/)
- [Groq Console](https://console.groq.com/)
- [DeepSeek Platform](https://platform.deepseek.com/)

**文档版本:** v1.0 (2026-02-18)
