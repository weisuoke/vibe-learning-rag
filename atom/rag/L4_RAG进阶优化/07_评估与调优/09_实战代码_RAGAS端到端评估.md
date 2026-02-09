# 实战代码：RAGAS 端到端评估

> 使用 RAGAS 框架对 RAG 系统进行完整评估

---

## 环境准备

```bash
pip install ragas datasets langchain-openai python-dotenv
```

**依赖说明：**

| 库 | 用途 |
|------|------|
| `ragas` | RAG 评估框架，提供 4 大核心指标 |
| `datasets` | HuggingFace 数据集库，RAGAS 要求的数据格式 |
| `langchain-openai` | RAGAS 底层使用 LLM 作为评估器 |
| `python-dotenv` | 从 `.env` 文件加载 API 密钥 |

---

## 完整代码

```python
"""
RAGAS 端到端评估实战
演示：如何使用 RAGAS 框架评估 RAG 系统的检索和生成质量

运行前准备：
1. 在项目根目录创建 .env 文件
2. 添加 OPENAI_API_KEY=your_key_here
3. 可选：添加 OPENAI_BASE_URL=https://your-proxy.com/v1
"""

import os
from dotenv import load_dotenv

# 加载环境变量（API 密钥）
load_dotenv()

# 验证 API 密钥已设置
assert os.getenv("OPENAI_API_KEY"), "请在 .env 文件中设置 OPENAI_API_KEY"

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,        # 忠实度：答案是否忠于检索到的上下文
    answer_relevancy,    # 答案相关性：答案是否切题
    context_precision,   # 上下文精确度：检索结果中相关内容的排序质量
    context_recall,      # 上下文召回率：检索结果是否覆盖了标准答案的要点
)


# ===== 1. 构建评估数据集 =====
print("=" * 50)
print("1. 构建评估数据集")
print("=" * 50)

# 模拟 RAG 系统的输入输出
# 实际使用时，这些数据来自你的 RAG 系统的真实运行结果
eval_data = {
    # question: 用户提出的问题
    "question": [
        "什么是向量数据库？",
        "RAG 和 Fine-tuning 有什么区别？",
        "如何选择合适的 Embedding 模型？",
        "什么是 Chunk 策略？",
        "ReRank 的作用是什么？",
    ],
    # answer: RAG 系统生成的答案
    "answer": [
        "向量数据库是专门用于存储和检索高维向量数据的数据库系统，"
        "支持基于相似度的快速检索，常用于语义搜索和推荐系统。",

        "RAG 通过检索外部知识来增强生成，不需要重新训练模型；"
        "Fine-tuning 则是通过在特定数据上训练来调整模型参数。"
        "RAG 更灵活，知识可以实时更新。",

        "选择 Embedding 模型需要考虑：语言支持（中文/英文）、"
        "维度大小、性能基准（MTEB排名）、推理速度和成本。"
        "对于中文场景推荐 BGE 系列。",

        "Chunk 策略是将长文档分割成小片段的方法，常见策略包括"
        "固定大小分块、递归分块、语义分块等。"
        "选择策略需要考虑文档类型和检索需求。",

        "ReRank 是对初步检索结果进行精细重排序的技术，"
        "使用交叉编码器等模型对 query-document 对进行相关性打分，"
        "提升最终检索质量。",
    ],
    # contexts: RAG 系统检索到的上下文（注意：每条是一个列表）
    "contexts": [
        [
            "向量数据库是一种专门设计用于存储、索引和查询高维向量数据的数据库。"
            "它支持近似最近邻搜索（ANN），常用于语义搜索、推荐系统等场景。"
        ],
        [
            "RAG（检索增强生成）通过检索外部知识库来增强 LLM 的生成能力，"
            "无需重新训练模型。Fine-tuning 则需要在特定数据集上微调模型参数。"
            "RAG 的优势在于知识可以实时更新，而 Fine-tuning 需要重新训练。"
        ],
        [
            "选择 Embedding 模型时需要考虑多个因素：目标语言支持、向量维度、"
            "在 MTEB 等基准上的表现、推理速度和部署成本。"
            "中文场景推荐使用 BGE、M3E 等模型。"
        ],
        [
            "文本分块（Chunking）是 RAG 系统中的关键步骤，"
            "将长文档分割成适合检索的小片段。常见策略包括：固定大小分块、"
            "基于分隔符的递归分块、基于语义的智能分块。"
        ],
        [
            "ReRank（重排序）是在初步检索之后，使用更精确的模型对候选文档"
            "进行重新排序。通常使用交叉编码器（Cross-Encoder）对 query 和 "
            "document 进行联合编码，计算更准确的相关性分数。"
        ],
    ],
    # ground_truth: 人工标注的标准答案（Context Recall 指标需要）
    "ground_truth": [
        "向量数据库是存储高维向量并支持相似性搜索的专用数据库，"
        "用于语义搜索和推荐系统。",

        "RAG 通过检索外部知识增强生成，不需要重新训练，知识可实时更新；"
        "Fine-tuning 需要在特定数据上训练模型参数。",

        "选择 Embedding 模型要考虑语言支持、维度、MTEB 基准表现、"
        "速度和成本，中文推荐 BGE 系列。",

        "Chunk 策略是将长文档分割成小片段的方法，"
        "包括固定大小、递归分块、语义分块等。",

        "ReRank 使用交叉编码器对初步检索结果进行精细重排序，提升检索质量。",
    ],
}

# 转换为 HuggingFace Dataset 格式（RAGAS 要求的输入格式）
dataset = Dataset.from_dict(eval_data)
print(f"评估数据集大小: {len(dataset)} 条")
print(f"数据字段: {list(eval_data.keys())}")
print(f"示例问题: {eval_data['question'][0]}")


# ===== 2. 配置评估指标并运行 =====
print("\n" + "=" * 50)
print("2. 运行 RAGAS 评估（需要调用 LLM，请耐心等待）")
print("=" * 50)

# 定义要评估的指标
metrics = [
    faithfulness,        # 忠实度（生成侧）
    answer_relevancy,    # 答案相关性（生成侧）
    context_precision,   # 上下文精确度（检索侧）
    context_recall,      # 上下文召回率（检索侧）
]

# 运行评估（RAGAS 会自动调用 LLM 进行评估打分）
result = evaluate(
    dataset=dataset,
    metrics=metrics,
)


# ===== 3. 查看整体评估结果 =====
print("\n" + "=" * 50)
print("3. 整体评估结果")
print("=" * 50)

print(f"\n{'指标':<25} {'分数':<10} {'含义'}")
print("-" * 65)
metric_descriptions = {
    "faithfulness": "答案是否忠于检索到的上下文（不编造）",
    "answer_relevancy": "答案是否与问题相关（切题程度）",
    "context_precision": "检索结果中相关内容是否排在前面",
    "context_recall": "检索结果是否覆盖了标准答案的要点",
}
for metric_name, score in result.items():
    desc = metric_descriptions.get(metric_name, "")
    print(f"{metric_name:<25} {score:.4f}    {desc}")


# ===== 4. 查看每条数据的详细结果 =====
print("\n" + "=" * 50)
print("4. 逐条详细结果")
print("=" * 50)

# 转换为 Pandas DataFrame，方便逐条查看
df = result.to_pandas()
for i, row in df.iterrows():
    question_preview = row["question"][:30]
    print(f"\n--- 问题 {i+1}: {question_preview}... ---")
    print(f"  Faithfulness（忠实度）:        {row.get('faithfulness', 'N/A')}")
    print(f"  Answer Relevancy（答案相关性）: {row.get('answer_relevancy', 'N/A')}")
    print(f"  Context Precision（上下文精确度）: {row.get('context_precision', 'N/A')}")
    print(f"  Context Recall（上下文召回率）:   {row.get('context_recall', 'N/A')}")


# ===== 5. 自动诊断与优化建议 =====
print("\n" + "=" * 50)
print("5. 自动诊断与优化建议")
print("=" * 50)

# 设定阈值：低于 0.7 视为需要优化
THRESHOLD = 0.7

optimization_tips = {
    "faithfulness": [
        "在 Prompt 中明确要求「仅根据提供的上下文回答」",
        "降低 temperature 参数（如设为 0.1）减少创造性",
        "添加引用要求，让模型标注答案来源",
    ],
    "answer_relevancy": [
        "优化 Prompt 模板，明确要求「直接回答问题」",
        "检查是否有无关上下文干扰了生成",
        "添加「不要回答问题以外的内容」约束",
    ],
    "context_precision": [
        "添加 ReRank 重排序步骤，提升相关文档排名",
        "优化 Query 改写策略，让检索更精准",
        "调整 Embedding 模型，选择更适合业务场景的模型",
    ],
    "context_recall": [
        "增大 top_k 检索数量，召回更多候选文档",
        "优化 Chunk 策略，避免关键信息被切断",
        "尝试混合检索（向量 + 关键词），提升召回覆盖面",
    ],
}

has_issue = False
for metric_name, score in result.items():
    if score < THRESHOLD:
        has_issue = True
        print(f"\n[需优化] {metric_name} = {score:.2f} (低于阈值 {THRESHOLD})")
        tips = optimization_tips.get(metric_name, [])
        for j, tip in enumerate(tips, 1):
            print(f"  建议{j}: {tip}")
    else:
        print(f"\n[良好]   {metric_name} = {score:.2f}")

if not has_issue:
    print("\n所有指标均达标，RAG 系统整体质量良好！")


# ===== 6. 导出评估报告 =====
print("\n" + "=" * 50)
print("6. 导出评估报告")
print("=" * 50)

# 保存为 CSV 文件，方便后续分析和对比
output_path = "ragas_evaluation_report.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"详细报告已保存到: {output_path}")
print("可以用 Excel 或 Pandas 打开查看每条数据的评分详情")
```

---

## 运行输出示例

```
==================================================
1. 构建评估数据集
==================================================
评估数据集大小: 5 条
数据字段: ['question', 'answer', 'contexts', 'ground_truth']
示例问题: 什么是向量数据库？

==================================================
2. 运行 RAGAS 评估（需要调用 LLM，请耐心等待）
==================================================

==================================================
3. 整体评估结果
==================================================

指标                       分数       含义
-----------------------------------------------------------------
faithfulness               0.8500    答案是否忠于检索到的上下文（不编造）
answer_relevancy           0.9200    答案是否与问题相关（切题程度）
context_precision          0.9000    检索结果中相关内容是否排在前面
context_recall             0.8800    检索结果是否覆盖了标准答案的要点

==================================================
4. 逐条详细结果
==================================================

--- 问题 1: 什么是向量数据库？... ---
  Faithfulness（忠实度）:        1.0
  Answer Relevancy（答案相关性）: 0.95
  Context Precision（上下文精确度）: 1.0
  Context Recall（上下文召回率）:   0.9

--- 问题 2: RAG 和 Fine-tuning 有什么区别？... ---
  Faithfulness（忠实度）:        0.85
  Answer Relevancy（答案相关性）: 0.92
  Context Precision（上下文精确度）: 0.9
  Context Recall（上下文召回率）:   0.88

==================================================
5. 自动诊断与优化建议
==================================================

[良好]   faithfulness = 0.85
[良好]   answer_relevancy = 0.92
[良好]   context_precision = 0.90
[良好]   context_recall = 0.88

所有指标均达标，RAG 系统整体质量良好！

==================================================
6. 导出评估报告
==================================================
详细报告已保存到: ragas_evaluation_report.csv
可以用 Excel 或 Pandas 打开查看每条数据的评分详情
```

---

## 常见问题排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `OPENAI_API_KEY` 报错 | `.env` 文件未创建或密钥未填写 | 在项目根目录创建 `.env`，写入 `OPENAI_API_KEY=sk-xxx` |
| 评估速度很慢 | RAGAS 需要多次调用 LLM 进行评估 | 减少评估数据量，或使用 `gpt-3.5-turbo` 降低成本 |
| 某个指标分数为 0 | 数据格式不正确 | 检查 `contexts` 是否为 `List[List[str]]` 格式 |
| `context_recall` 报错 | 缺少 `ground_truth` 字段 | 该指标必须提供人工标注的标准答案 |
| 分数异常偏低 | 评估数据质量差或 answer 与 contexts 不匹配 | 确保 answer 确实是基于 contexts 生成的 |

---

## 一句话记住

**RAGAS 评估只需要三步：准备数据集 -> 调用 evaluate -> 解读结果，5 行核心代码就能完成 RAG 系统的全面体检。**
