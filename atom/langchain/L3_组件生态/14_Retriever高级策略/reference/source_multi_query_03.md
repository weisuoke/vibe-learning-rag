---
type: source_code_analysis
source: sourcecode/langchain
analyzed_files:
  - libs/langchain/langchain_classic/retrievers/multi_query.py
analyzed_at: 2026-02-27
knowledge_point: 14_Retriever高级策略
---

# 源码分析：MultiQueryRetriever

## 分析的文件
- `libs/langchain/langchain_classic/retrievers/multi_query.py` - 多查询检索器

## 关键发现

### 类：MultiQueryRetriever(BaseRetriever)
使用 LLM 从单个用户查询生成多个查询变体，为每个变体检索文档，返回唯一并集。

### 核心属性
| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `retriever` | `BaseRetriever` | (必需) | 底层检索器 |
| `llm_chain` | `Runnable` | (必需) | 生成查询变体的链 |
| `verbose` | `bool` | `True` | 是否记录生成的查询 |
| `include_original` | `bool` | `False` | 是否包含原始查询 |

### 辅助类：LineListOutputParser
按换行符分割 LLM 输出，过滤空行。

### 默认提示词
指示 LLM 生成 3 个原始问题的替代版本，以克服基于距离的相似性搜索的局限性。

### 核心流程
1. `generate_queries` → 调用 llm_chain 生成查询变体
2. 可选追加原始查询（include_original=True）
3. `retrieve_documents` → 为每个查询运行基础检索器（async 版本并行执行）
4. `unique_union` → 通过 `_unique_documents` 去重

### 设计模式
- **查询扩展**：经典 IR 技术，生成多个查询重述以提高召回率
- **工厂方法**：`from_llm` 封装内部 LCEL 链的构建
- **模板方法**：检索管道分解为可覆盖的步骤
- **向后兼容**：支持旧版 LLMChain 和现代 LCEL Runnable
