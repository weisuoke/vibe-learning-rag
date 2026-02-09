# 实战代码：自动化评估 Pipeline

> 构建端到端的 RAG 评估流水线，支持多维度评估和 A/B 测试

---

## 环境准备

```bash
pip install openai python-dotenv
```

**依赖说明：**

| 库 | 用途 |
|------|------|
| `openai` | 调用 LLM 作为评估器（LLM-as-Judge） |
| `python-dotenv` | 从 `.env` 文件加载 API 密钥 |

---

## 完整代码

```python
"""
自动化评估 Pipeline 实战
演示：构建一个完整的 RAG 评估流水线
功能：多维度评估 + 结果存储 + A/B 测试对比 + 评估报告生成

运行前准备：
1. 在项目根目录创建 .env 文件
2. 添加 OPENAI_API_KEY=your_key_here
3. 可选：添加 OPENAI_BASE_URL=https://your-proxy.com/v1
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量（API 密钥）
load_dotenv()

# 验证 API 密钥已设置
assert os.getenv("OPENAI_API_KEY"), "请在 .env 文件中设置 OPENAI_API_KEY"

client = OpenAI()


# ===== 1. 评估数据集管理 =====
print("=" * 60)
print("1. 评估数据集管理")
print("=" * 60)


class EvalDataset:
    """评估数据集管理器

    管理评估所需的四元组数据：问题、检索上下文、生成答案、标准答案
    支持序列化存储和加载，方便复用和版本管理
    """

    def __init__(self, name: str):
        self.name = name
        self.items: List[Dict[str, Any]] = []

    def add_item(self, question: str, contexts: List[str],
                 answer: str, ground_truth: str):
        """添加一条评估数据（四元组）"""
        self.items.append({
            "question": question,       # 用户问题
            "contexts": contexts,       # 检索到的上下文列表
            "answer": answer,           # RAG 系统生成的答案
            "ground_truth": ground_truth,  # 人工标注的标准答案
        })

    def __len__(self):
        return len(self.items)

    def save(self, path: str):
        """保存数据集到 JSON 文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"name": self.name, "items": self.items}, f,
                      ensure_ascii=False, indent=2)
        print(f"  数据集已保存: {path} ({len(self.items)} 条)")

    @classmethod
    def load(cls, path: str) -> "EvalDataset":
        """从 JSON 文件加载数据集"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ds = cls(data["name"])
        ds.items = data["items"]
        print(f"  数据集已加载: {ds.name} ({len(ds.items)} 条)")
        return ds


# 构建示例评估数据集（实际项目中应从标注数据加载）
dataset = EvalDataset("RAG基础知识评估集")

dataset.add_item(
    question="什么是向量数据库？",
    contexts=["向量数据库是专门存储和检索高维向量数据的数据库系统，支持近似最近邻（ANN）搜索。"],
    answer="向量数据库是存储高维向量并支持相似性搜索的数据库，常用于语义检索场景。",
    ground_truth="向量数据库是存储高维向量并支持相似性搜索的专用数据库。"
)

dataset.add_item(
    question="RAG 的核心流程是什么？",
    contexts=["RAG 的核心流程包括：文档加载、文本分块、向量化、检索、上下文注入、LLM 生成。"],
    answer="RAG 的核心流程是：加载文档、分块、向量化存储、检索相关文档、注入上下文、LLM 生成答案。",
    ground_truth="RAG 核心流程：文档加载 -> 分块 -> 向量化 -> 检索 -> 上下文注入 -> 生成。"
)

dataset.add_item(
    question="什么是 ReRank？",
    contexts=["ReRank 是对初步检索结果进行精细重排序的技术，使用交叉编码器提升排序质量。"],
    answer="ReRank 是重新排序检索结果的技术。Python 是一种编程语言。",
    ground_truth="ReRank 使用交叉编码器对检索结果重排序，提升检索质量。"
)

print(f"  数据集: {dataset.name}, 共 {len(dataset)} 条")


# ===== 2. 多维度评估器 =====
print("\n" + "=" * 60)
print("2. 多维度评估器（LLM-as-Judge）")
print("=" * 60)


class RAGEvaluator:
    """RAG 系统多维度评估器

    使用 LLM 作为评判者，从三个维度评估 RAG 系统：
    - 忠实度（Faithfulness）：答案是否忠于检索到的上下文
    - 相关性（Relevancy）：答案是否切题回答了问题
    - 正确性（Correctness）：答案与标准答案的一致程度
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def _llm_judge(self, prompt: str) -> dict:
        """通用 LLM 评估调用，返回 JSON 格式的评分结果"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是专业的 RAG 系统评估专家。请严格按 JSON 格式输出。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,  # 评估时使用 temperature=0 保证一致性
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """评估忠实度：答案中的信息是否都来自上下文"""
        ctx = "\n".join(contexts)
        result = self._llm_judge(
            f"评估答案对上下文的忠实度。答案中的每个声明是否都能在上下文中找到依据？\n"
            f"上下文: {ctx}\n答案: {answer}\n"
            f'输出 JSON: {{"score": 0.0到1.0的浮点数, "reason": "简要原因"}}'
        )
        return float(result.get("score", 0))

    def evaluate_relevancy(self, question: str, answer: str) -> float:
        """评估答案相关性：答案是否针对问题进行了回答"""
        result = self._llm_judge(
            f"评估答案与问题的相关性。答案是否直接回答了问题？是否包含无关信息？\n"
            f"问题: {question}\n答案: {answer}\n"
            f'输出 JSON: {{"score": 0.0到1.0的浮点数, "reason": "简要原因"}}'
        )
        return float(result.get("score", 0))

    def evaluate_correctness(self, answer: str, ground_truth: str) -> float:
        """评估答案正确性：生成答案与标准答案的语义一致程度"""
        result = self._llm_judge(
            f"评估生成答案与标准答案的语义一致性。核心信息是否匹配？\n"
            f"标准答案: {ground_truth}\n生成答案: {answer}\n"
            f'输出 JSON: {{"score": 0.0到1.0的浮点数, "reason": "简要原因"}}'
        )
        return float(result.get("score", 0))

    def evaluate_item(self, item: dict) -> dict:
        """对单条数据进行全维度评估"""
        return {
            "question": item["question"],
            "faithfulness": self.evaluate_faithfulness(item["answer"], item["contexts"]),
            "relevancy": self.evaluate_relevancy(item["question"], item["answer"]),
            "correctness": self.evaluate_correctness(item["answer"], item["ground_truth"]),
        }

    def evaluate_dataset(self, dataset: EvalDataset) -> dict:
        """对整个数据集进行批量评估，返回汇总结果"""
        results = []
        for i, item in enumerate(dataset.items):
            print(f"  评估第 {i + 1}/{len(dataset)} 条...")
            result = self.evaluate_item(item)
            results.append(result)

        # 计算各维度平均分
        avg_scores = {
            "faithfulness": sum(r["faithfulness"] for r in results) / len(results),
            "relevancy": sum(r["relevancy"] for r in results) / len(results),
            "correctness": sum(r["correctness"] for r in results) / len(results),
        }

        return {
            "dataset_name": dataset.name,
            "num_items": len(dataset),
            "timestamp": datetime.now().isoformat(),
            "avg_scores": avg_scores,
            "details": results,
        }


evaluator = RAGEvaluator()
print("  评估器已初始化（模型: gpt-4o-mini）")


# ===== 3. 运行评估 =====
print("\n" + "=" * 60)
print("3. 运行评估")
print("=" * 60)

eval_result = evaluator.evaluate_dataset(dataset)

print(f"\n  评估结果 ({eval_result['dataset_name']})")
print(f"  {'指标':<15} {'平均分':<10}")
print("  " + "-" * 25)
for metric, score in eval_result["avg_scores"].items():
    status = "PASS" if score >= 0.7 else "WARN"
    print(f"  {metric:<15} {score:<10.3f} {status}")


# ===== 4. A/B 测试对比 =====
print("\n" + "=" * 60)
print("4. A/B 测试对比")
print("=" * 60)


def ab_test_compare(result_a: dict, result_b: dict,
                    label_a: str = "配置A", label_b: str = "配置B") -> None:
    """对比两组评估结果，找出哪个配置更优

    实际使用时，配置A和配置B可以是不同的：
    - chunk_size（如 512 vs 256）
    - 检索策略（如 向量检索 vs 混合检索）
    - Prompt 模板
    - LLM 模型
    """
    print(f"\n  {'指标':<15} {label_a:<10} {label_b:<10} {'差异':<10} {'胜出':<6}")
    print("  " + "-" * 55)

    for metric in result_a["avg_scores"]:
        score_a = result_a["avg_scores"][metric]
        score_b = result_b["avg_scores"][metric]
        diff = score_b - score_a
        # 差异超过 0.01 才判定胜出，避免噪声干扰
        winner = f"{label_b}" if diff > 0.01 else (f"{label_a}" if diff < -0.01 else "平局")
        print(f"  {metric:<15} {score_a:<10.3f} {score_b:<10.3f} {diff:+<10.3f} {winner}")


# 模拟配置B的评估结果
# 实际项目中，应该用不同的 RAG 配置生成答案，再分别评估
result_b = {
    "avg_scores": {
        "faithfulness": eval_result["avg_scores"]["faithfulness"] + 0.05,
        "relevancy": eval_result["avg_scores"]["relevancy"] + 0.03,
        "correctness": eval_result["avg_scores"]["correctness"] - 0.02,
    }
}

print("  对比: 配置A (chunk_size=512) vs 配置B (chunk_size=256)")
ab_test_compare(eval_result, result_b, "chunk=512", "chunk=256")


# ===== 5. 生成评估报告 =====
print("\n" + "=" * 60)
print("5. 评估报告")
print("=" * 60)


def generate_report(result: dict) -> str:
    """生成文本格式的评估报告

    包含：整体评分、逐条详情、优化建议
    实际项目中可扩展为 HTML 或 PDF 格式
    """
    lines = []
    lines.append(f"# RAG 评估报告")
    lines.append(f"数据集: {result['dataset_name']}")
    lines.append(f"评估时间: {result['timestamp']}")
    lines.append(f"样本数量: {result['num_items']}")
    lines.append("")

    # 整体评分
    lines.append("## 整体评分")
    for metric, score in result["avg_scores"].items():
        status = "良好" if score >= 0.8 else ("一般" if score >= 0.6 else "需改进")
        lines.append(f"  {metric}: {score:.3f} ({status})")
    lines.append("")

    # 逐条详情
    lines.append("## 逐条详情")
    for i, detail in enumerate(result["details"]):
        lines.append(f"  [{i + 1}] {detail['question']}")
        lines.append(f"      忠实度={detail['faithfulness']:.2f}  "
                     f"相关性={detail['relevancy']:.2f}  "
                     f"正确性={detail['correctness']:.2f}")
    lines.append("")

    # 优化建议
    lines.append("## 优化建议")
    for metric, score in result["avg_scores"].items():
        if score < 0.8:
            if "faith" in metric:
                lines.append(f"  - {metric} 偏低: 建议加强 Prompt 约束，降低 temperature")
            elif "relev" in metric:
                lines.append(f"  - {metric} 偏低: 建议优化 Prompt 模板，确保答案聚焦问题")
            elif "correct" in metric:
                lines.append(f"  - {metric} 偏低: 建议优化检索策略或增加 ReRank")

    if all(s >= 0.8 for s in result["avg_scores"].values()):
        lines.append("  各项指标均达标，系统表现良好！")

    return "\n".join(lines)


report = generate_report(eval_result)
print(report)

print("\n" + "=" * 60)
print("评估 Pipeline 执行完毕！")
print("=" * 60)
```

---

## 运行输出示例

```
============================================================
1. 评估数据集管理
============================================================
  数据集: RAG基础知识评估集, 共 3 条

============================================================
2. 多维度评估器（LLM-as-Judge）
============================================================
  评估器已初始化（模型: gpt-4o-mini）

============================================================
3. 运行评估
============================================================
  评估第 1/3 条...
  评估第 2/3 条...
  评估第 3/3 条...

  评估结果 (RAG基础知识评估集)
  指标             平均分
  -------------------------
  faithfulness     0.833      PASS
  relevancy        0.800      PASS
  correctness      0.767      PASS

============================================================
4. A/B 测试对比
============================================================
  对比: 配置A (chunk_size=512) vs 配置B (chunk_size=256)

  指标             chunk=512  chunk=256  差异         胜出
  -------------------------------------------------------
  faithfulness     0.833      0.883      +0.050     chunk=256
  relevancy        0.800      0.830      +0.030     chunk=256
  correctness      0.767      0.747      -0.020     chunk=512

============================================================
5. 评估报告
============================================================
# RAG 评估报告
数据集: RAG基础知识评估集
评估时间: 2026-02-08T14:30:00.000000
样本数量: 3

## 整体评分
  faithfulness: 0.833 (良好)
  relevancy: 0.800 (良好)
  correctness: 0.767 (一般)

## 逐条详情
  [1] 什么是向量数据库？
      忠实度=0.90  相关性=0.90  正确性=0.85
  [2] RAG 的核心流程是什么？
      忠实度=0.95  相关性=0.90  正确性=0.90
  [3] 什么是 ReRank？
      忠实度=0.65  相关性=0.60  正确性=0.55

## 优化建议
  - correctness 偏低: 建议优化检索策略或增加 ReRank

============================================================
评估 Pipeline 执行完毕！
============================================================
```

---

## 关键设计说明

| 模块 | 作用 | 扩展方向 |
|------|------|----------|
| `EvalDataset` | 管理评估四元组数据 | 支持从 CSV/数据库加载 |
| `RAGEvaluator` | LLM-as-Judge 多维度评估 | 增加上下文相关性、噪声鲁棒性等指标 |
| `ab_test_compare` | 对比不同配置的评估结果 | 增加统计显著性检验 |
| `generate_report` | 生成文本评估报告 | 扩展为 HTML/PDF 格式 |

---

## 一句话记住

**自动化评估 Pipeline = 数据集管理 + 多维度评估器 + A/B 测试框架 + 报告生成——让 RAG 优化从"拍脑袋"变成"看数据"。**
