# 实战代码：基础Provider注册

## 场景描述

**需求：** 注册一个简单的自定义Provider（本地Ollama），支持多个开源模型。

**目标：**
- 零OAuth认证（本地服务）
- OpenAI兼容API
- 支持多个模型
- 完全本地运行

**适用场景：**
- 本地开发和测试
- 数据隐私敏感场景
- 无网络环境
- 成本控制

## 方案选择

### 方案1：models.json配置（推荐）

**优点：**
- 零代码，仅需配置
- 热重载，编辑后立即生效
- 易于维护

**缺点：**
- 功能受限于配置选项
- 无法自定义复杂逻辑

### 方案2：Extension开发

**优点：**
- 完全控制
- 可以添加自定义逻辑
- 可以打包分享

**缺点：**
- 需要编写TypeScript代码
- 需要理解Extension API

## 实现方式1：models.json配置

### 步骤1：创建配置文件

```bash
# 创建用户级配置目录
mkdir -p ~/.pi/agent

# 创建models.json
touch ~/.pi/agent/models.json
```

### 步骤2：编写配置

```json
{
  "providers": {
    "ollama": {
      "baseUrl": "http://localhost:11434/v1",
      "apiKey": "ollama",
      "api": "openai-completions",
      "models": [
        {
          "id": "llama3.1:8b",
          "name": "Llama 3.1 8B (Local)",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 128000,
          "maxTokens": 32000,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        },
        {
          "id": "qwen2.5-coder:7b",
          "name": "Qwen 2.5 Coder 7B (Local)",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 32768,
          "maxTokens": 8192,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        },
        {
          "id": "deepseek-coder:6.7b",
          "name": "DeepSeek Coder 6.7B (Local)",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 16384,
          "maxTokens": 4096,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
```

### 步骤3：启动Ollama

```bash
# 安装Ollama（如果未安装）
# macOS
brew install ollama

# 启动Ollama服务
ollama serve

# 拉取模型（在另一个终端）
ollama pull llama3.1:8b
ollama pull qwen2.5-coder:7b
ollama pull deepseek-coder:6.7b
```

### 步骤4：测试配置

```bash
# 启动pi
pi

# 打开模型选择器
/model

# 选择Ollama模型（应该看到3个模型）
# 选择 "Llama 3.1 8B (Local)"

# 测试对话
Hello, can you introduce yourself?
```

### 步骤5：验证

**检查点：**
- [ ] models.json文件创建成功
- [ ] Ollama服务运行中
- [ ] 模型已拉取
- [ ] pi中可以看到3个Ollama模型
- [ ] 可以正常对话

## 实现方式2：Extension开发

### 步骤1：创建Extension目录

```bash
# 创建Extension目录
mkdir -p ~/.pi/agent/extensions/ollama-provider

# 进入目录
cd ~/.pi/agent/extensions/ollama-provider
```

### 步骤2：初始化项目

```bash
# 创建package.json
cat > package.json << 'EOF'
{
  "name": "ollama-provider",
  "version": "1.0.0",
  "type": "module",
  "main": "index.ts",
  "dependencies": {
    "@mariozechner/pi-coding-agent": "latest"
  }
}
EOF

# 安装依赖
npm install
```

### 步骤3：编写Extension代码

```typescript
// index.ts
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  pi.registerProvider("ollama", {
    baseUrl: "http://localhost:11434/v1",
    apiKey: "ollama",
    api: "openai-completions",

    models: [
      {
        id: "llama3.1:8b",
        name: "Llama 3.1 8B (Local)",
        reasoning: false,
        input: ["text"],
        contextWindow: 128000,
        maxTokens: 32000,
        cost: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
        },
      },
      {
        id: "qwen2.5-coder:7b",
        name: "Qwen 2.5 Coder 7B (Local)",
        reasoning: false,
        input: ["text"],
        contextWindow: 32768,
        maxTokens: 8192,
        cost: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
        },
      },
      {
        id: "deepseek-coder:6.7b",
        name: "DeepSeek Coder 6.7B (Local)",
        reasoning: false,
        input: ["text"],
        contextWindow: 16384,
        maxTokens: 4096,
        cost: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
        },
      },
    ],
  });

  console.log("Ollama Provider registered with 3 models");
}
```

### 步骤4：加载Extension

```bash
# 方式1：全局加载（推荐）
# 在~/.pi/agent/settings.json中添加
{
  "extensions": [
    "~/.pi/agent/extensions/ollama-provider"
  ]
}

# 方式2：命令行加载
pi -e ~/.pi/agent/extensions/ollama-provider

# 方式3：项目级加载
# 在项目的.pi/agent/settings.json中添加
{
  "extensions": [
    "~/.pi/agent/extensions/ollama-provider"
  ]
}
```

### 步骤5：测试Extension

```bash
# 启动pi
pi

# 应该看到Extension加载日志
# "Ollama Provider registered with 3 models"

# 打开模型选择器
/model

# 选择Ollama模型
# 测试对话
```

## 进阶配置

### 1. 添加更多模型

```json
{
  "models": [
    {
      "id": "mistral:7b",
      "name": "Mistral 7B (Local)"
    },
    {
      "id": "codellama:13b",
      "name": "Code Llama 13B (Local)"
    },
    {
      "id": "phi3:mini",
      "name": "Phi-3 Mini (Local)"
    }
  ]
}
```

### 2. 配置远程Ollama服务器

```json
{
  "providers": {
    "ollama-remote": {
      "baseUrl": "http://gpu-server.local:11434/v1",
      "apiKey": "ollama",
      "api": "openai-completions",
      "models": [...]
    }
  }
}
```

### 3. 添加自定义请求头

```typescript
pi.registerProvider("ollama", {
  baseUrl: "http://localhost:11434/v1",
  apiKey: "ollama",
  api: "openai-completions",
  headers: {
    "X-Custom-Header": "value",
  },
  models: [...],
});
```

### 4. 动态模型列表

```typescript
export default async function (pi: ExtensionAPI) {
  // 从Ollama API获取模型列表
  const response = await fetch("http://localhost:11434/api/tags");
  const data = await response.json();

  const models = data.models.map((model: any) => ({
    id: model.name,
    name: `${model.name} (Local)`,
    reasoning: false,
    input: ["text"],
    contextWindow: 128000,
    maxTokens: 32000,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  }));

  pi.registerProvider("ollama", {
    baseUrl: "http://localhost:11434/v1",
    apiKey: "ollama",
    api: "openai-completions",
    models,
  });

  console.log(`Ollama Provider registered with ${models.length} models`);
}
```

## 故障排查

### 问题1：模型不显示

**检查：**
```bash
# 检查Ollama服务
curl http://localhost:11434/api/tags

# 检查models.json语法
jq . ~/.pi/agent/models.json

# 检查pi日志
pi --verbose
```

### 问题2：连接失败

**检查：**
```bash
# 测试Ollama API
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### 问题3：响应慢

**优化：**
- 使用更小的模型（如7B而非70B）
- 增加GPU内存
- 调整Ollama配置（`OLLAMA_NUM_PARALLEL`）

## 最佳实践

### 1. 模型命名

**清晰的显示名称：**
```json
{
  "id": "llama3.1:8b",
  "name": "Llama 3.1 8B (Local, Fast)"
}
```

### 2. 成本追踪

**即使是本地模型，也可以设置虚拟成本：**
```json
{
  "cost": {
    "input": 0.001,  // 虚拟成本，用于追踪使用量
    "output": 0.001,
    "cacheRead": 0,
    "cacheWrite": 0
  }
}
```

### 3. 上下文窗口

**根据实际模型设置：**
```json
{
  "id": "llama3.1:8b",
  "contextWindow": 128000  // Llama 3.1支持128K
}
```

### 4. 热重载

**编辑models.json后：**
```bash
# 在pi中重新打开/model即可
/model
```

## 参考资源

**Ollama文档：**
- https://ollama.ai/
- https://github.com/ollama/ollama

**Pi-mono文档：**
- https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/models.md
- https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/custom-provider.md

**模型资源：**
- Ollama模型库：https://ollama.ai/library
- Hugging Face：https://huggingface.co/models

## 总结

**models.json方式：**
- ✅ 零代码，快速配置
- ✅ 热重载，编辑即生效
- ✅ 适合80%的场景

**Extension方式：**
- ✅ 完全控制
- ✅ 可以添加自定义逻辑
- ✅ 可以打包分享

**推荐：**
- 简单场景：使用models.json
- 复杂场景：使用Extension
- 动态场景：使用Extension + API查询
