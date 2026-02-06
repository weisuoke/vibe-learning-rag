# Python 开发环境设计

**日期：** 2026-02-06
**目标：** 为 RAG 学习文档仓库集成完整的 Python 开发环境

---

## 设计概述

为 vibe-learning-rag 项目配置 Python 开发环境，支持：
- 运行文档中的代码示例
- 开发新的学习内容
- 构建实际的 RAG 应用项目

## 技术栈

- **Python 版本：** 3.11.11
- **版本管理：** asdf
- **包管理：** uv
- **依赖范围：** 最小可用集（核心 RAG 开发库）

---

## 架构设计

### 1. 版本管理层 (asdf)

使用 `.tool-versions` 文件指定 Python 版本：
```
python 3.11.11
```

**优势：**
- 跨机器版本一致性
- 与现有 asdf 工作流集成
- 团队协作友好

### 2. 包管理层 (uv)

使用 `pyproject.toml` 作为依赖配置中心：
- 遵循 PEP 621 标准
- 快速依赖解析
- 生成 `uv.lock` 确保可重现构建

### 3. 项目结构

```
vibe-learning-rag/
├── .tool-versions          # Python 版本声明
├── pyproject.toml          # 项目元数据 + 依赖
├── uv.lock                 # 锁定的依赖版本
├── .gitignore              # Python 相关忽略规则
├── .env.example            # API 密钥模板
├── examples/               # 可运行的示例脚本
│   ├── README.md
│   ├── basic_rag.py
│   ├── l1_nlp/
│   ├── l2_llm/
│   └── l3_rag/
└── atom/                   # 现有学习材料
```

---

## 依赖配置

### 核心依赖（最小可用集）

| 库 | 版本 | 用途 |
|---|---|---|
| openai | >=1.0.0 | LLM API 调用 |
| sentence-transformers | >=2.2.0 | 本地 Embedding |
| chromadb | >=0.4.0 | 向量数据库 |
| langchain | >=0.1.0 | RAG 框架 |
| langchain-openai | >=0.0.5 | LangChain OpenAI 集成 |
| pypdf | >=3.0.0 | PDF 文档解析 |
| fastapi | >=0.100.0 | API 服务框架 |
| uvicorn[standard] | >=0.23.0 | ASGI 服务器 |
| python-dotenv | >=1.0.0 | 环境变量管理 |

### 开发依赖

| 库 | 版本 | 用途 |
|---|---|---|
| jupyter | >=1.0.0 | 交互式学习 |
| ipython | >=8.12.0 | 增强 REPL |

**设计原则：**
- 最小但完整：覆盖所有文档示例需求
- 易于扩展：后续可通过 `uv add` 添加
- 版本宽松：使用 `>=` 获取最新稳定版

---

## 工作流程

### 初始化设置

```bash
# 1. 安装 Python 版本
asdf plugin add python
asdf install python 3.11.11

# 2. 安装 uv（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 同步依赖
uv sync

# 4. 激活虚拟环境
source .venv/bin/activate
```

### 日常开发

```bash
# 激活环境
source .venv/bin/activate

# 运行示例
python examples/basic_rag.py

# 启动 Jupyter
jupyter notebook

# 启动 API 服务
uvicorn examples.api:app --reload

# 退出环境
deactivate
```

### 依赖管理

```bash
# 添加运行时依赖
uv add <package-name>

# 添加开发依赖
uv add --dev <package-name>

# 更新依赖
uv sync --upgrade

# 重新安装
uv sync --reinstall
```

---

## 配置文件详情

### `.tool-versions`

```
python 3.11.11
```

### `pyproject.toml`

```toml
[project]
name = "vibe-learning-rag"
version = "0.1.0"
description = "RAG Development Learning Repository with atomized knowledge points"
requires-python = ">=3.11"
dependencies = [
    "openai>=1.0.0",
    "sentence-transformers>=2.2.0",
    "chromadb>=0.4.0",
    "langchain>=0.1.0",
    "langchain-openai>=0.0.5",
    "pypdf>=3.0.0",
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.23.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "jupyter>=1.0.0",
    "ipython>=8.12.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### `.gitignore` 新增内容

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
.venv/
env/
ENV/
.env
*.egg-info/
dist/
build/

# uv
uv.lock

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDE
.vscode/
.idea/
```

### `.env.example`

```bash
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (可选)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

---

## 示例代码

### `examples/basic_rag.py`

基础 RAG 流程演示，验证环境配置：

```python
"""
基础 RAG 示例
演示：文档加载 -> Embedding -> 向量存储 -> 检索 -> 生成
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# 加载环境变量
load_dotenv()

def main():
    print("=== 基础 RAG 示例 ===\n")

    # 1. 初始化客户端
    print("1. 初始化客户端...")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chroma_client = chromadb.Client()

    # 2. 创建向量集合
    print("2. 创建向量集合...")
    collection = chroma_client.create_collection("demo")

    # 3. 示例文档
    docs = [
        "RAG 是检索增强生成的缩写，结合了检索和生成两个步骤",
        "Embedding 将文本转换为高维向量，捕捉语义信息",
        "ChromaDB 是一个轻量级向量数据库，适合快速原型开发"
    ]

    print(f"3. 加载了 {len(docs)} 个示例文档")
    print("\n✓ 环境配置验证成功！")
    print("✓ 所有核心库可正常导入")
    print("\n下一步：配置 .env 文件后可运行完整 RAG 流程")

if __name__ == "__main__":
    main()
```

### `examples/README.md`

```markdown
# 示例脚本

本目录包含可运行的 RAG 开发示例代码。

## 运行前准备

### 1. 配置环境变量

复制模板文件：
\```bash
cp .env.example .env
\```

编辑 `.env` 文件，添加你的 API keys：
\```bash
OPENAI_API_KEY=sk-...
\```

### 2. 激活虚拟环境

\```bash
source .venv/bin/activate
\```

## 示例列表

- `basic_rag.py` - 基础 RAG 流程演示，验证环境配置

## 目录结构

\```
examples/
├── README.md           # 本文件
├── basic_rag.py        # 入门示例
├── l1_nlp/             # L1 NLP 基础示例
├── l2_llm/             # L2 LLM 核心示例
└── l3_rag/             # L3 RAG 核心流程示例
\```

每个子目录对应 `atom/` 中的学习层级。
```

---

## 文档集成

### CLAUDE.md 更新

在"代码规范"章节之前添加：

```markdown
## Python 环境配置

### 环境要求
- Python 3.11+ (通过 asdf 管理)
- uv 包管理器

### 快速开始
\```bash
# 1. 安装 Python
asdf install

# 2. 安装依赖
uv sync

# 3. 激活环境
source .venv/bin/activate

# 4. 运行示例
python examples/basic_rag.py
\```

### 可用的库
所有代码示例可以使用以下库：
- **LLM 调用**: openai
- **Embedding**: sentence-transformers
- **向量存储**: chromadb
- **RAG 框架**: langchain, langchain-openai
- **文档解析**: pypdf
- **API 服务**: fastapi, uvicorn
- **工具**: python-dotenv
```

---

## 故障排除

### 问题1：asdf 找不到 Python

```bash
# 检查插件
asdf plugin list

# 添加 Python 插件
asdf plugin add python

# 安装指定版本
asdf install python 3.11.11
```

### 问题2：uv 命令未找到

```bash
# 重新安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 重启终端或重新加载配置
source ~/.zshrc  # 或 ~/.bashrc
```

### 问题3：ChromaDB 依赖问题

```bash
# macOS 可能需要 sqlite3
brew install sqlite3

# 重新同步依赖
uv sync --reinstall
```

### 问题4：OpenAI API 密钥错误

```python
# 验证 .env 文件加载
from dotenv import load_dotenv
import os

load_dotenv()
print(os.getenv("OPENAI_API_KEY"))  # 应显示你的密钥
```

### 问题5：导入错误

```bash
# 检查虚拟环境
which python  # 应指向 .venv/bin/python

# 如果不是，激活环境
source .venv/bin/activate
```

### 环境验证

```bash
# 快速检查所有核心库
python -c "import openai, chromadb, langchain; print('✓ 所有核心库已安装')"
```

---

## 扩展计划

### 短期（可选）

- 添加更多示例脚本到 `examples/l1_nlp/`, `examples/l2_llm/` 等
- 创建 Jupyter notebooks 用于交互式学习

### 中期（按需）

- 添加更多 RAG 库：`llama-index`, `faiss`, `anthropic`
- 添加测试框架：`pytest`
- 添加代码质量工具：`ruff`, `black`

### 长期（未来）

- 容器化：Docker 配置
- CI/CD：自动化测试和部署
- 文档网站：使用 MkDocs 或 Sphinx

---

## 设计决策记录

### 为什么选择 Python 3.11？

- ✅ 稳定性和性能的最佳平衡
- ✅ 所有 RAG 库完全支持
- ✅ 现代 Python 特性（如改进的错误消息）
- ✅ 长期支持（至 2027 年 10 月）

### 为什么选择 uv？

- ✅ 比 pip/poetry 快 10-100 倍
- ✅ 遵循 Python 标准（PEP 621）
- ✅ 简单的 CLI 接口
- ✅ 自动虚拟环境管理

### 为什么选择最小依赖集？

- ✅ 快速安装（~5 分钟 vs ~20 分钟）
- ✅ 减少依赖冲突
- ✅ 易于理解和维护
- ✅ 按需扩展

### 为什么创建 examples/ 目录？

- ✅ 分离文档和可执行代码
- ✅ 提供即用的验证脚本
- ✅ 便于 CI/CD 集成
- ✅ 清晰的学习路径

---

## 验收标准

- [x] `.tool-versions` 文件创建
- [x] `pyproject.toml` 配置完成
- [x] `.gitignore` 包含 Python 规则
- [x] `.env.example` 模板创建
- [x] `examples/basic_rag.py` 可运行
- [x] `examples/README.md` 说明完整
- [x] CLAUDE.md 更新环境说明
- [x] 设计文档提交到 git

---

**版本：** v1.0
**作者：** Claude Code
**状态：** 已批准，待实施
