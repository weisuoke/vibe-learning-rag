# 核心概念 2: Chain-of-Thought (CoT)

## 一句话定义

**通过引导大模型逐步展示推理过程,而非直接给出答案,显著提升复杂推理任务的准确性和可验证性。**

**RAG应用:** 在RAG系统中,CoT让模型解释为什么从检索文档中得出某个答案,提供推理依据,增强答案的可信度和可解释性。

---

## 为什么重要?

### 问题场景

```python
# 场景:数学推理任务
from openai import OpenAI

client = OpenAI()

# ❌ 没有CoT:直接给答案
prompt = "计算: 23 * 47 = ?"

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
# 可能输出: "1081"
# 问题:
# 1. 如果答案错了,不知道哪里错
# 2. 无法验证推理过程
# 3. 难以调试和改进
```

### 解决方案

```python
# ✅ 使用CoT:展示推理过程
prompt = """
计算: 23 * 47

让我们一步步计算:
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
# 输出:
# 让我们一步步计算:
# 1. 将47分解为40 + 7
# 2. 计算23 * 40 = 920
# 3. 计算23 * 7 = 161
# 4. 相加: 920 + 161 = 1081
#
# 答案: 1081

# 优势:
# 1. 每一步都可验证
# 2. 如果错了,能定位错误步骤
# 3. 推理过程透明
```

**性能提升:**

| 任务类型 | Zero-shot | CoT | 提升 |
|---------|-----------|-----|------|
| 数学推理 | 17.7% | 40.7% | +130% |
| 常识推理 | 69.0% | 79.0% | +14% |
| 符号推理 | 5.0% | 25.0% | +400% |

**来源:** [Chain-of-Thought Prompting (2022)](https://arxiv.org/abs/2201.11903)

---

## 核心原理

### 原理1:推理链外化

**定义:** 将模型内部的隐式推理过程显式化为可观察的步骤序列。

**机制:**

```
传统方法:
输入 → [黑盒推理] → 输出
      ↑ 不可见

CoT方法:
输入 → 步骤1 → 步骤2 → 步骤3 → 输出
      ↓       ↓       ↓
      可见    可验证  可优化
```

**为什么有效?**

大模型在预训练时见过大量"逐步解题"的文本:

```python
# 预训练数据中的模式
"""
问题: 求解方程 2x + 5 = 13

解答:
步骤1: 两边减5
2x + 5 - 5 = 13 - 5
2x = 8

步骤2: 两边除以2
2x / 2 = 8 / 2
x = 4

答案: x = 4
"""
# 模型学会了:看到"步骤"提示,就按步骤推理
```

**实验验证:**

```python
# 实验:不同推理方式的效果
# 任务:GSM8K数学题(小学数学应用题)

# 直接回答
# 准确率: 17.7%

# CoT推理
# 准确率: 40.7%

# 提升: +130%
```

**来源:** [Chain-of-Thought Prompting Elicits Reasoning (2022)](https://arxiv.org/abs/2201.11903)

---

### 原理2:中间步骤作为锚点

**核心发现:** 正确的中间步骤引导模型走向正确答案。

**实验:**

```python
# 实验:中间步骤的影响
# 任务:多步推理

# 场景1:没有中间步骤
prompt = "如果一个数的3倍加5等于20,这个数是多少?"
# 模型可能直接猜: "5" (错误)

# 场景2:有中间步骤
prompt = """
如果一个数的3倍加5等于20,这个数是多少?

让我们一步步思考:
1. 设这个数为x
2. 根据题意: 3x + 5 = 20
3. 两边减5: 3x = 15
4. 两边除以3: x = 5

答案: 5
"""
# 模型按步骤推理,得到正确答案
```

**锚点效应:**

```
步骤1(正确) → 步骤2(正确) → 步骤3(正确) → 答案(正确)
   ↓              ↓              ↓
  锚点           锚点           锚点

步骤1(错误) → 步骤2(错误) → 步骤3(错误) → 答案(错误)
   ↓
  偏离轨道
```

**来源:** [Large Language Models are Zero-Shot Reasoners (2022)](https://arxiv.org/abs/2205.11916)

---

### 原理3:简洁引导优于详细指令

**核心发现:** "让我们一步步思考"比详细列出步骤更有效。

**对比实验:**

```python
# 方法1:详细指令(效果较差)
prompt = """
请按以下步骤解题:
1. 首先理解问题
2. 然后列出已知条件
3. 接着设定未知数
4. 建立方程
5. 求解方程
6. 验证答案

问题: ...
"""
# 准确率: 35%

# 方法2:简洁引导(效果更好)
prompt = """
问题: ...

让我们一步步思考:
"""
# 准确率: 41%
```

**原因:**
1. **灵活性:** 模型可以根据问题选择合适的步骤
2. **自然性:** 更接近人类思考方式
3. **避免过度约束:** 详细指令可能限制模型的推理能力

**魔法提示词:**

```python
# 英文
"Let's think step by step:"

# 中文
"让我们一步步思考:"
"让我们逐步分析:"
"让我们仔细推理:"
```

**来源:** [Large Language Models are Zero-Shot Reasoners (2022)](https://arxiv.org/abs/2205.11916)

---

## 手写实现

### 从零实现 CoT Prompt Builder

```python
"""
Chain-of-Thought Prompt Builder
功能:自动构建CoT提示词
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class CoTStep:
    """推理步骤"""
    step_number: int
    description: str
    result: Optional[str] = None

class CoTBuilder:
    """CoT提示词构建器"""

    def __init__(self, client: OpenAI):
        self.client = client

    def build_zero_shot_cot(
        self,
        query: str,
        language: str = "zh"
    ) -> str:
        """
        构建Zero-shot CoT提示词

        Args:
            query: 查询问题
            language: 语言("zh"或"en")
        """
        if language == "zh":
            trigger = "让我们一步步思考:"
        else:
            trigger = "Let's think step by step:"

        return f"{query}\n\n{trigger}"

    def build_few_shot_cot(
        self,
        query: str,
        examples: List[Dict[str, str]],
        language: str = "zh"
    ) -> str:
        """
        构建Few-shot CoT提示词

        Args:
            query: 查询问题
            examples: 示例列表,每个示例包含"question"和"reasoning"
            language: 语言
        """
        prompt_parts = []

        # 添加示例
        for i, example in enumerate(examples, 1):
            prompt_parts.append(f"示例{i}:")
            prompt_parts.append(f"问题: {example['question']}")
            prompt_parts.append(f"推理: {example['reasoning']}")
            prompt_parts.append("")

        # 添加查询
        prompt_parts.append("现在解答:")
        prompt_parts.append(f"问题: {query}")

        if language == "zh":
            prompt_parts.append("推理: 让我们一步步思考:")
        else:
            prompt_parts.append("Reasoning: Let's think step by step:")

        return "\n".join(prompt_parts)

    def generate_with_cot(
        self,
        query: str,
        method: str = "zero-shot",
        examples: Optional[List[Dict[str, str]]] = None,
        model: str = "gpt-4o-mini",
        extract_answer: bool = True
    ) -> Dict[str, str]:
        """
        使用CoT生成响应

        Args:
            query: 查询问题
            method: 方法("zero-shot"或"few-shot")
            examples: Few-shot示例
            model: 使用的模型
            extract_answer: 是否提取最终答案
        """
        # 构建提示词
        if method == "zero-shot":
            prompt = self.build_zero_shot_cot(query)
        elif method == "few-shot":
            if not examples:
                raise ValueError("Few-shot method requires examples")
            prompt = self.build_few_shot_cot(query, examples)
        else:
            raise ValueError(f"Unknown method: {method}")

        # 调用LLM
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )

        reasoning = response.choices[0].message.content

        result = {
            "query": query,
            "reasoning": reasoning
        }

        # 提取最终答案
        if extract_answer:
            answer = self._extract_answer(reasoning)
            result["answer"] = answer

        return result

    def _extract_answer(self, reasoning: str) -> str:
        """从推理过程中提取最终答案"""
        # 简单实现:查找"答案:"或"因此"等关键词
        keywords = ["答案:", "因此", "所以", "最终", "Answer:", "Therefore"]

        lines = reasoning.split("\n")
        for i, line in enumerate(lines):
            for keyword in keywords:
                if keyword in line:
                    # 返回该行及后续内容
                    return "\n".join(lines[i:])

        # 如果没找到,返回最后一行
        return lines[-1] if lines else ""

    def verify_reasoning(
        self,
        reasoning: str,
        model: str = "gpt-4o-mini"
    ) -> Dict[str, any]:
        """
        验证推理过程的正确性

        Args:
            reasoning: 推理过程
            model: 使用的模型
        """
        verification_prompt = f"""
请验证以下推理过程是否正确:

{reasoning}

请分析:
1. 每个步骤的逻辑是否正确
2. 步骤之间的连接是否合理
3. 最终答案是否正确

验证结果:
"""

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": verification_prompt}]
        )

        return {
            "reasoning": reasoning,
            "verification": response.choices[0].message.content
        }


# 使用示例
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    client = OpenAI()
    builder = CoTBuilder(client)

    # 测试1: Zero-shot CoT
    print("=== Zero-shot CoT ===")
    result1 = builder.generate_with_cot(
        query="如果一个数的3倍加5等于20,这个数是多少?",
        method="zero-shot"
    )
    print(f"问题: {result1['query']}")
    print(f"推理:\n{result1['reasoning']}")
    print(f"答案: {result1['answer']}")
    print()

    # 测试2: Few-shot CoT
    print("=== Few-shot CoT ===")
    examples = [
        {
            "question": "一个数的2倍是10,这个数是多少?",
            "reasoning": """
让我们一步步思考:
1. 设这个数为x
2. 根据题意: 2x = 10
3. 两边除以2: x = 5
答案: 5
"""
        }
    ]

    result2 = builder.generate_with_cot(
        query="一个数的4倍减3等于13,这个数是多少?",
        method="few-shot",
        examples=examples
    )
    print(f"问题: {result2['query']}")
    print(f"推理:\n{result2['reasoning']}")
    print(f"答案: {result2['answer']}")
    print()

    # 测试3: 验证推理
    print("=== 验证推理 ===")
    verification = builder.verify_reasoning(result2['reasoning'])
    print(f"验证结果:\n{verification['verification']}")
```

### 实现原理解析

**1. Zero-shot CoT**
- 最简单的实现:只需添加"让我们一步步思考"
- 适用于大多数推理任务
- 无需准备示例

**2. Few-shot CoT**
- 提供示例推理过程
- 引导模型按特定格式推理
- 适用于需要特定推理风格的任务

**3. 答案提取**
- 从推理过程中提取最终答案
- 使用关键词匹配
- 可以进一步优化(如使用正则表达式)

**4. 推理验证**
- 使用另一次LLM调用验证推理过程
- 提供额外的质量保证
- 适用于关键任务

---

## RAG 应用场景

### 场景1: RAG答案解释

**问题:** RAG系统给出答案,但用户不知道为什么

**解决方案:** 使用CoT解释答案来源

```python
from openai import OpenAI
import chromadb

client = OpenAI()
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("docs")

# 添加文档
docs = [
    "Python由Guido van Rossum于1991年创建,是一种解释型语言。",
    "JavaScript由Brendan Eich于1995年创建,主要用于Web开发。",
    "Java由James Gosling于1995年创建,是一种编译型语言。"
]
collection.add(
    documents=docs,
    ids=[f"doc{i}" for i in range(len(docs))]
)

def rag_with_cot_explanation(query: str) -> Dict:
    """RAG + CoT解释"""

    # 1. 检索
    results = collection.query(
        query_texts=[query],
        n_results=2
    )
    retrieved_docs = results['documents'][0]

    # 2. CoT生成答案
    prompt = f"""
文档:
{chr(10).join(f"{i+1}. {doc}" for i, doc in enumerate(retrieved_docs))}

问题: {query}

让我们一步步分析:
1. 首先,理解问题的核心意图
2. 然后,检查哪些文档片段相关
3. 接着,从相关片段中提取信息
4. 最后,综合得出答案

分析:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "query": query,
        "retrieved_docs": retrieved_docs,
        "reasoning": response.choices[0].message.content
    }

# 测试
result = rag_with_cot_explanation("Python是什么时候创建的?")
print(result['reasoning'])
```

**输出示例:**
```
分析:
1. 首先,理解问题的核心意图
   - 问题询问Python的创建时间

2. 然后,检查哪些文档片段相关
   - 文档1提到"Python由Guido van Rossum于1991年创建"
   - 这直接回答了问题

3. 接着,从相关片段中提取信息
   - 创建时间: 1991年
   - 创建者: Guido van Rossum

4. 最后,综合得出答案
   - Python是在1991年创建的

答案: 1991年
依据: 文档1明确说明"Python由Guido van Rossum于1991年创建"
```

---

### 场景2: 复杂查询推理

**问题:** 用户查询需要多步推理才能回答

**解决方案:** 使用CoT分解推理步骤

```python
def complex_query_with_cot(query: str) -> Dict:
    """复杂查询 + CoT推理"""

    # 示例查询: "比较Python和JavaScript的创建时间,哪个更早?"

    # 1. 检索相关文档
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    retrieved_docs = results['documents'][0]

    # 2. CoT推理
    prompt = f"""
文档:
{chr(10).join(f"{i+1}. {doc}" for i, doc in enumerate(retrieved_docs))}

问题: {query}

这是一个比较类问题,需要多步推理。让我们一步步分析:

步骤1: 找出Python的创建时间
步骤2: 找出JavaScript的创建时间
步骤3: 比较两个时间
步骤4: 得出结论

推理过程:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "query": query,
        "retrieved_docs": retrieved_docs,
        "reasoning": response.choices[0].message.content
    }

# 测试
result = complex_query_with_cot(
    "比较Python和JavaScript的创建时间,哪个更早?"
)
print(result['reasoning'])
```

**输出示例:**
```
推理过程:

步骤1: 找出Python的创建时间
- 从文档1: "Python由Guido van Rossum于1991年创建"
- Python创建于1991年

步骤2: 找出JavaScript的创建时间
- 从文档2: "JavaScript由Brendan Eich于1995年创建"
- JavaScript创建于1995年

步骤3: 比较两个时间
- 1991年 < 1995年
- Python比JavaScript早4年

步骤4: 得出结论
- Python(1991年)比JavaScript(1995年)更早创建

答案: Python更早,比JavaScript早4年
```

---

### 场景3: RAG答案验证

**问题:** 需要验证RAG生成的答案是否合理

**解决方案:** 使用CoT验证推理链

```python
def verify_rag_answer_with_cot(
    query: str,
    answer: str,
    docs: List[str]
) -> Dict:
    """使用CoT验证RAG答案"""

    prompt = f"""
问题: {query}
答案: {answer}

文档:
{chr(10).join(f"{i+1}. {doc}" for i, doc in enumerate(docs))}

请验证这个答案是否正确。让我们一步步分析:

1. 检查答案是否直接来自文档
2. 检查答案是否完整回答了问题
3. 检查是否有矛盾或错误
4. 给出验证结论

验证过程:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "query": query,
        "answer": answer,
        "verification": response.choices[0].message.content
    }

# 测试
verification = verify_rag_answer_with_cot(
    query="Python是什么时候创建的?",
    answer="1991年",
    docs=["Python由Guido van Rossum于1991年创建,是一种解释型语言。"]
)
print(verification['verification'])
```

---

## 最佳实践

### 1. 选择合适的CoT类型

```python
# Zero-shot CoT: 简单任务
query = "23 * 47 = ?"
prompt = f"{query}\n\n让我们一步步计算:"

# Few-shot CoT: 需要特定格式
examples = [...]
prompt = build_few_shot_cot(query, examples)

# Manual CoT: 需要精确控制
prompt = f"""
{query}

请按以下步骤:
1. 步骤1
2. 步骤2
3. 步骤3
"""
```

### 2. 推理步骤的粒度

```python
# ✅ 好: 适中的粒度
prompt = """
计算 23 * 47

让我们一步步计算:
"""
# 模型会自己决定合适的步骤

# ❌ 坏: 过于详细
prompt = """
计算 23 * 47

步骤1: 理解乘法定义
步骤2: 分解第一个数
步骤3: 分解第二个数
...
步骤10: 得出答案
"""
# 过度约束,限制模型能力
```

### 3. 结合Few-shot提升效果

```python
# 对于特定领域,Few-shot CoT效果更好
examples = [
    {
        "question": "代码审查: 这段代码有什么问题?",
        "reasoning": """
让我们逐步分析:
1. 检查语法错误
2. 检查逻辑错误
3. 检查性能问题
4. 给出改进建议
"""
    }
]
```

### 4. 验证推理过程

```python
# 对关键任务,使用第二次LLM调用验证
reasoning = generate_with_cot(query)
verification = verify_reasoning(reasoning)
if "错误" in verification:
    # 重新生成
    reasoning = generate_with_cot(query, temperature=0.3)
```

---

## 参考资源

- [Chain-of-Thought Prompting (2022)](https://arxiv.org/abs/2201.11903)
- [Large Language Models are Zero-Shot Reasoners (2022)](https://arxiv.org/abs/2205.11916)
- [Automatic Chain of Thought Prompting (2022)](https://arxiv.org/abs/2210.03493)
- [Prompt Engineering Guide - CoT](https://www.promptingguide.ai/techniques/cot)
- [OpenAI Prompt Engineering - CoT](https://platform.openai.com/docs/guides/prompt-engineering)
