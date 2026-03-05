---
type: source_code_analysis
source: sourcecode/langchain
analyzed_files:
  - libs/langchain/langchain_classic/retrievers/ensemble.py
analyzed_at: 2026-02-27
knowledge_point: 14_Retriever高级策略
---

# 源码分析：EnsembleRetriever

## 分析的文件
- `libs/langchain/langchain_classic/retrievers/ensemble.py` - 混合检索核心实现

## 关键发现

### 类：EnsembleRetriever(BaseRetriever)
组合多个检索器的结果，使用加权倒数排名融合（Weighted Reciprocal Rank Fusion, RRF）。

### 核心属性
| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `retrievers` | `list[RetrieverLike]` | (必需) | 要组合的检索器列表 |
| `weights` | `list[float]` | 等权重 `1/n` | 每个检索器的 RRF 权重 |
| `c` | `int` | `60` | RRF 常数，控制高排名和低排名项之间的平衡 |
| `id_key` | `str \| None` | `None` | 去重用的元数据键，回退到 `page_content` |

### 核心算法：weighted_reciprocal_rank
```python
score(doc) = SUM over each retriever i: weight_i / (rank_i + c)
```
- 使用 `defaultdict(float)` 累积分数
- 通过 `unique_by_key` 去重（保持首次出现顺序）
- 按 RRF 分数降序排列返回

### 设计模式
- **策略模式**：检索器列表可注入，每个作为可互换的检索策略
- **加权集成/排名融合**：经典 IR 技术（Cormack et al., SIGIR 2009），c=60 是原始 RRF 论文的标准默认值
- **装饰器/包装器**：将 N 个检索器包装在单个 BaseRetriever 接口后面
- **异步并行**：async 版本使用 `asyncio.gather` 并行执行所有子检索器

### 特殊实现
- 直接覆盖 `invoke`/`ainvoke`（而非 `_get_relevant_documents`），因为需要将完整 `RunnableConfig` 传播到子检索器
- `config_specs` 属性聚合所有子检索器的配置规格
