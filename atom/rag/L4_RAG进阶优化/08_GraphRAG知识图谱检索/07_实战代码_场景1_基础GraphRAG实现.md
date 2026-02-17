# GraphRAG知识图谱检索 - 实战代码场景1: 基础GraphRAG实现

> 使用 Microsoft GraphRAG 快速搭建系统

---

## 场景描述

使用 Microsoft GraphRAG 官方框架,从零搭建一个完整的 GraphRAG 系统,包含文档索引和查询功能。

**学习目标**:
- 掌握 Microsoft GraphRAG 的安装和配置
- 理解索引构建流程
- 实现 Local 和 Global 搜索

---

## 完整代码

```bash
# ===== 1. 安装 GraphRAG =====
pip install graphrag

# ===== 2. 初始化项目 =====
mkdir my-graphrag
cd my-graphrag
graphrag init --root .

# ===== 3. 配置 API Key =====
# 编辑 .env 文件
cat > .env << EOF
OPENAI_API_KEY=your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
EOF

# ===== 4. 准备文档 =====
mkdir -p input
cat > input/sample.txt << EOF
Alice 在 Google 工作,负责搜索引擎开发。她毕业于斯坦福大学。
Bob 是 Google 的 CEO,管理整个公司。他之前在 Oracle 工作。
Charlie 在 Microsoft 工作,与 Alice 是大学同学。
EOF

# ===== 5. 索引文档 =====
graphrag index --root .

# ===== 6. Local Search 查询 =====
graphrag query \
  --root . \
  --method local \
  --query "Alice 在哪里工作?"

# ===== 7. Global Search 查询 =====
graphrag query \
  --root . \
  --method global \
  --query "公司的主要人员有哪些?"
```

---

## Python API 使用

```python
"""
使用 Python API 进行 GraphRAG 查询
"""

import os
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_reports
)
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext
)
from graphrag.query.structured_search.local_search.search import LocalSearch

# ===== 1. 配置 =====
os.environ["OPENAI_API_KEY"] = "your-key"

INPUT_DIR = "./output"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
COMMUNITY_REPORT_TABLE = "create_final_community_reports"

# ===== 2. 加载索引 =====
entity_df = read_indexer_entities(
    INPUT_DIR,
    ENTITY_TABLE,
    ENTITY_EMBEDDING_TABLE
)

report_df = read_indexer_reports(
    INPUT_DIR,
    COMMUNITY_REPORT_TABLE
)

# ===== 3. 创建 LLM =====
llm = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4",
    api_type=OpenaiApiType.OpenAI,
    max_retries=20,
)

# ===== 4. 创建搜索引擎 =====
context_builder = LocalSearchMixedContext(
    community_reports=report_df,
    entities=entity_df,
    token_encoder=llm.get_token_encoder(),
)

search_engine = LocalSearch(
    llm=llm,
    context_builder=context_builder,
    token_encoder=llm.get_token_encoder(),
    response_type="multiple paragraphs",
)

# ===== 5. 执行查询 =====
query = "Alice 在哪里工作?"
result = search_engine.search(query)

print(f"查询: {query}")
print(f"答案: {result.response}")
print(f"上下文数据: {len(result.context_data)} 条")
```

---

## 运行输出

```
=== 索引阶段 ===
⠋ 正在加载文档...
✓ 加载了 1 个文档
⠋ 正在提取实体...
✓ 提取了 7 个实体
⠋ 正在提取关系...
✓ 提取了 6 个关系
⠋ 正在检测社区...
✓ 检测到 2 个社区
⠋ 正在生成摘要...
✓ 生成了 2 个社区摘要
✓ 索引完成!

=== Local Search ===
查询: Alice 在哪里工作?
答案: Alice 在 Google 工作,负责搜索引擎开发。
上下文数据: 3 条

=== Global Search ===
查询: 公司的主要人员有哪些?
答案: 主要人员包括:
1. Google: Alice (工程师), Bob (CEO)
2. Microsoft: Charlie (员工)
```

---

## 关键要点

1. **Microsoft GraphRAG 是 2025-2026 最成熟的开源方案**
2. **CLI 工具简化了索引和查询流程**
3. **支持 Local 和 Global 两种搜索模式**
4. **Python API 提供更灵活的集成方式**
5. **索引过程包含: 文档加载 → 实体提取 → 关系抽取 → 社区检测 → 摘要生成**

---

**版本**: v1.0 (基于 Microsoft GraphRAG v3.0.2)
**最后更新**: 2026-02-17
**维护者**: Claude Code
