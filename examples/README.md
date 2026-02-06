# 示例脚本

本目录包含可运行的 RAG 开发示例代码。

## 运行前准备

### 1. 配置环境变量

复制模板文件：
```bash
cp .env.example .env
```

编辑 `.env` 文件，添加你的 API keys：
```bash
OPENAI_API_KEY=sk-...
```

#### 使用自定义 API 端点（可选）

如果你使用代理或第三方兼容服务，可以配置自定义 BASE_URL：

```bash
# OpenAI 兼容服务
OPENAI_BASE_URL=https://your-proxy.com/v1

# Anthropic 兼容服务
ANTHROPIC_BASE_URL=https://your-proxy.com
```

如果不设置，将使用官方 API 端点。

### 2. 激活虚拟环境

```bash
source .venv/bin/activate
```

## 示例列表

- `basic_rag.py` - 基础 RAG 流程演示，验证环境配置

## 目录结构

```
examples/
├── README.md           # 本文件
├── basic_rag.py        # 入门示例
├── l1_nlp/             # L1 NLP 基础示例
├── l2_llm/             # L2 LLM 核心示例
└── l3_rag/             # L3 RAG 核心流程示例
```

每个子目录对应 `atom/` 中的学习层级。
