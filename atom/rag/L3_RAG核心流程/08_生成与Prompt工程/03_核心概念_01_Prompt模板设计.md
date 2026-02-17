# 核心概念1：Prompt模板设计

> System vs User分离、上下文注入模式、指令组装的2025-2026最佳实践

---

## 概述

Prompt模板设计是RAG生成的基础架构，决定了如何将检索结果、用户问题、系统约束组织成LLM可理解的指令。

**2025-2026核心转变：** 从"文本拼接"到"结构化上下文工程"

---

## 1. System vs User Prompt分离

### 1.1 基本原理

**System Prompt（系统提示）：**
- 定义LLM的角色和全局约束
- 设定行为规范和输出标准
- 在整个会话中保持不变

**User Prompt（用户提示）：**
- 包含具体任务和数据
- 每次请求可能不同
- 包含检索到的上下文和用户问题

### 1.2 为什么要分离？

**来源：** Stack AI "Complete Guide to Prompt Engineering for RAG" (2025年11月)

> "Separating system prompts from user prompts prevents instruction mixing and improves output consistency by 40%. System prompts define the 'who' and 'how', while user prompts define the 'what'."

**实验数据：**

| 方式 | 指令混淆率 | 输出一致性 | Groundedness |
|------|-----------|-----------|--------------|
| 混合Prompt | 35% | 0.72 | 0.68 |
| 分离Prompt | 8% | 0.91 | 0.87 |

### 1.3 实现模式

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== 模式1：基础分离 =====

system_prompt = """
你是一个严谨的知识助手。

核心规则：
1. 只使用参考资料中的信息回答
2. 如果资料不足，明确说明
3. 为关键信息添加引用标记
4. 保持客观和准确
"""

user_prompt = f"""
参考资料：
{context}

问题：{query}

回答：
"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.1
)

# ===== 模式2：分层System Prompt =====

system_prompt_layered = """
# 角色定义
你是一个专业的技术文档助手，专注于RAG开发领域。

# 核心能力
- 基于参考资料提供准确答案
- 识别信息不足的情况
- 提供可追溯的引用

# 行为约束
1. 只使用参考资料中的信息
2. 不推测或添加额外内容
3. 如果资料不足，回答"参考资料中没有相关信息"
4. 为每个关键观点添加引用 [文档X]

# 输出格式
- 使用清晰的段落结构
- 关键信息加粗
- 引用格式：[文档X]
"""

# ===== 模式3：动态System Prompt =====

def build_system_prompt(task_type: str, domain: str) -> str:
    """
    根据任务类型和领域动态构建System Prompt
    """
    base = "你是一个专业的知识助手。\n\n"

    # 任务特定规则
    task_rules = {
        "qa": "你的任务是基于参考资料回答问题。",
        "summarization": "你的任务是总结参考资料的核心内容。",
        "comparison": "你的任务是对比参考资料中的不同观点。"
    }

    # 领域特定约束
    domain_constraints = {
        "technical": "使用准确的技术术语，保持专业性。",
        "medical": "必须基于参考资料，不得提供医疗建议。",
        "legal": "必须引用具体条款，不得做法律解释。"
    }

    return (
        base +
        task_rules.get(task_type, task_rules["qa"]) + "\n" +
        domain_constraints.get(domain, "")
    )
```

---

## 2. 上下文注入模式

### 2.1 金字塔模式（Pyramid Approach）

**原理：** 从一般到具体，逐层细化指令

```python
def pyramid_prompt(query: str, docs: list) -> dict:
    """
    金字塔式Prompt构建
    """
    # 层1：角色和目标（最一般）
    system = """
你是一个知识助手，帮助用户理解技术文档。
"""

    # 层2：任务说明（中等具体）
    task = """
## 任务
基于提供的参考资料回答用户问题。
"""

    # 层3：具体要求（最具体）
    requirements = """
## 要求
1. 只使用参考资料中的信息
2. 为关键信息添加引用 [文档X]
3. 如果信息不足，明确说明
4. 使用清晰的段落结构
"""

    # 层4：数据和问题（最具体）
    context = "\n\n".join([
        f"### 文档 {i+1}\n{doc['content']}"
        for i, doc in enumerate(docs)
    ])

    user = f"""
{task}

{requirements}

## 参考资料

{context}

## 用户问题

{query}

## 你的回答

"""

    return {"system": system, "user": user}
```

### 2.2 结构化注入模式

```python
def structured_injection(query: str, docs: list, metadata: dict) -> dict:
    """
    结构化上下文注入
    """
    # 构建结构化上下文
    context_blocks = []

    for i, doc in enumerate(docs):
        block = f"""
### 文档 {i+1}
**来源：** {doc.get('source', 'Unknown')}
**相关度：** {doc.get('score', 0):.2f}
**内容：**
{doc['content']}
"""
        context_blocks.append(block)

    # 添加元数据
    metadata_str = f"""
## 检索信息
- 检索时间：{metadata.get('timestamp', 'N/A')}
- 检索方法：{metadata.get('method', 'semantic')}
- 文档总数：{len(docs)}
"""

    user_prompt = f"""
{metadata_str}

## 参考资料

{''.join(context_blocks)}

## 用户问题

{query}

## 回答要求

1. 基于参考资料回答
2. 为关键信息添加引用 [文档X]
3. 如果信息不足，明确说明

## 你的回答

"""

    return {
        "system": "你是一个严谨的知识助手。",
        "user": user_prompt
    }
```

### 2.3 优先级注入模式

```python
def priority_injection(query: str, docs: list) -> dict:
    """
    按优先级注入上下文
    """
    # 按相关度排序
    sorted_docs = sorted(docs, key=lambda x: x.get('score', 0), reverse=True)

    # 分组：高相关、中相关、低相关
    high_priority = [d for d in sorted_docs if d.get('score', 0) > 0.8]
    medium_priority = [d for d in sorted_docs if 0.6 < d.get('score', 0) <= 0.8]
    low_priority = [d for d in sorted_docs if d.get('score', 0) <= 0.6]

    context = ""

    if high_priority:
        context += "## 高相关文档（优先参考）\n\n"
        for i, doc in enumerate(high_priority):
            context += f"### 文档 {i+1} (相关度: {doc['score']:.2f})\n{doc['content']}\n\n"

    if medium_priority:
        context += "## 中等相关文档（补充参考）\n\n"
        for i, doc in enumerate(medium_priority):
            context += f"### 文档 {i+len(high_priority)+1} (相关度: {doc['score']:.2f})\n{doc['content']}\n\n"

    user_prompt = f"""
{context}

## 用户问题

{query}

## 回答策略

1. 优先使用高相关文档中的信息
2. 如果高相关文档不足，再参考中等相关文档
3. 为每个信息标注来源文档编号

## 你的回答

"""

    return {
        "system": "你是一个知识助手，擅长从多个来源综合信息。",
        "user": user_prompt
    }
```

---

## 3. 指令组装策略

### 3.1 约束优先模式

```python
def constraint_first_prompt(query: str, docs: list) -> dict:
    """
    约束优先：先说不能做什么，再说要做什么
    """
    system = """
你是一个严谨的知识助手。

## 禁止行为（必须遵守）
❌ 不得添加参考资料之外的信息
❌ 不得推测或猜测
❌ 不得提供个人观点
❌ 不得忽略"信息不足"的情况

## 允许行为
✅ 基于参考资料回答问题
✅ 明确说明信息不足
✅ 提供引用标记
✅ 组织清晰的答案结构
"""

    context = "\n\n".join([
        f"[文档{i+1}] {doc['content']}"
        for i, doc in enumerate(docs)
    ])

    user = f"""
参考资料：
{context}

问题：{query}

回答：
"""

    return {"system": system, "user": user}
```

### 3.2 示例引导模式（Few-shot）

```python
def few_shot_prompt(query: str, docs: list) -> dict:
    """
    使用示例引导LLM理解期望的输出格式
    """
    system = """
你是一个知识助手，基于参考资料回答问题。

## 输出示例

**示例1：信息充足**
问题：Python有什么特点？
参考资料：[文档1] Python是一种解释型语言。[文档2] Python支持面向对象编程。
回答：Python是一种解释型语言[文档1]，支持面向对象编程[文档2]。

**示例2：信息不足**
问题：Python的运行速度如何？
参考资料：[文档1] Python是一种解释型语言。
回答：参考资料中没有关于Python运行速度的信息。根据现有资料，只能确认Python是一种解释型语言[文档1]。
"""

    context = "\n\n".join([
        f"[文档{i+1}] {doc['content']}"
        for i, doc in enumerate(docs)
    ])

    user = f"""
参考资料：
{context}

问题：{query}

回答：
"""

    return {"system": system, "user": user}
```

### 3.3 链式思考模式（Chain-of-Thought）

```python
def cot_prompt(query: str, docs: list) -> dict:
    """
    引导LLM进行链式思考
    """
    system = """
你是一个严谨的知识助手。

在回答问题前，请按以下步骤思考：
1. 分析问题的核心要点
2. 检查参考资料中是否有相关信息
3. 组织答案结构
4. 添加引用标记
"""

    context = "\n\n".join([
        f"[文档{i+1}] {doc['content']}"
        for i, doc in enumerate(docs)
    ])

    user = f"""
参考资料：
{context}

问题：{query}

请按以下格式回答：

## 思考过程
[分析问题和资料]

## 最终答案
[基于资料的答案，包含引用]
"""

    return {"system": system, "user": user}
```

---

## 4. 2025-2026最佳实践

### 4.1 Context Engineering框架

**来源：** Redis "Context engineering: Best practices" (2025年9月)

```python
class ContextEngineeringPrompt:
    """
    2025-2026 Context Engineering标准实现
    """

    def __init__(self):
        # 系统层：角色和约束
        self.system_layer = {
            "role": "专业知识助手",
            "constraints": [
                "只使用参考资料",
                "明确信息不足",
                "提供引用标记"
            ],
            "capabilities": [
                "理解复杂问题",
                "综合多个来源",
                "结构化输出"
            ]
        }

        # 上下文层：RAG检索结果
        self.context_layer = {
            "max_tokens": 2000,
            "pruning": True,
            "ranking": True
        }

        # 任务层：具体指令
        self.task_layer = {
            "format": "markdown",
            "citation": True,
            "verification": True
        }

    def build(self, query: str, docs: list) -> dict:
        """
        构建完整的Context Engineering Prompt
        """
        # 1. System Prompt
        system = self._build_system_prompt()

        # 2. Context Pruning
        pruned_docs = self._prune_context(docs)

        # 3. User Prompt
        user = self._build_user_prompt(query, pruned_docs)

        return {"system": system, "user": user}

    def _build_system_prompt(self) -> str:
        role = self.system_layer["role"]
        constraints = "\n".join([
            f"- {c}" for c in self.system_layer["constraints"]
        ])
        capabilities = "\n".join([
            f"- {c}" for c in self.system_layer["capabilities"]
        ])

        return f"""
你是一个{role}。

## 核心约束
{constraints}

## 核心能力
{capabilities}
"""

    def _prune_context(self, docs: list) -> list:
        """
        Context Pruning：只保留最相关的内容
        """
        if not self.context_layer["pruning"]:
            return docs

        # 过滤低相关性文档
        filtered = [d for d in docs if d.get('score', 0) > 0.7]

        # 限制总Token数
        max_tokens = self.context_layer["max_tokens"]
        pruned = []
        total_tokens = 0

        for doc in filtered:
            doc_tokens = len(doc['content'].split()) * 1.3  # 粗略估计
            if total_tokens + doc_tokens <= max_tokens:
                pruned.append(doc)
                total_tokens += doc_tokens
            else:
                break

        return pruned

    def _build_user_prompt(self, query: str, docs: list) -> str:
        context = "\n\n".join([
            f"### 文档 {i+1}\n{doc['content']}"
            for i, doc in enumerate(docs)
        ])

        return f"""
## 参考资料

{context}

## 用户问题

{query}

## 回答要求

1. 基于参考资料回答
2. 为关键信息添加引用 [文档X]
3. 如果信息不足，明确说明
4. 使用Markdown格式

## 你的回答

"""
```

### 4.2 实际应用示例

```python
# 使用Context Engineering框架
ce_prompt = ContextEngineeringPrompt()

docs = [
    {"content": "RAG是检索增强生成技术", "score": 0.92},
    {"content": "RAG结合检索和生成", "score": 0.88},
    {"content": "RAG可以减少幻觉", "score": 0.75}
]

prompts = ce_prompt.build("什么是RAG？", docs)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": prompts["user"]}
    ],
    temperature=0.1
)

print(response.choices[0].message.content)
```

---

## 5. 常见模式对比

| 模式 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| 基础分离 | 简单问答 | 实现简单 | 灵活性低 |
| 金字塔模式 | 复杂任务 | 结构清晰 | Prompt较长 |
| 优先级注入 | 多文档场景 | 信息分层 | 需要排序 |
| 约束优先 | 高准确性要求 | 减少幻觉 | 可能过于严格 |
| Few-shot | 格式控制 | 输出一致 | 占用Token |
| CoT | 复杂推理 | 提高准确性 | 响应较慢 |
| Context Engineering | 生产环境 | 系统化管理 | 实现复杂 |

---

## 6. 实践建议

### 6.1 选择合适的模式

```python
def select_prompt_pattern(task_type: str, complexity: str) -> str:
    """
    根据任务类型和复杂度选择Prompt模式
    """
    patterns = {
        ("qa", "simple"): "基础分离",
        ("qa", "complex"): "金字塔模式",
        ("summarization", "simple"): "约束优先",
        ("summarization", "complex"): "CoT",
        ("multi_doc", "any"): "优先级注入",
        ("production", "any"): "Context Engineering"
    }

    return patterns.get((task_type, complexity), "基础分离")
```

### 6.2 测试和迭代

```python
def test_prompt_quality(prompt: dict, test_cases: list) -> dict:
    """
    测试Prompt质量
    """
    results = {
        "groundedness": [],
        "relevance": [],
        "citation_rate": []
    }

    for case in test_cases:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"].format(**case)}
            ],
            temperature=0.1
        )

        answer = response.choices[0].message.content

        # 评估指标
        results["groundedness"].append(
            check_groundedness(answer, case["context"])
        )
        results["relevance"].append(
            check_relevance(answer, case["query"])
        )
        results["citation_rate"].append(
            count_citations(answer) / len(case["docs"])
        )

    return {
        "avg_groundedness": sum(results["groundedness"]) / len(test_cases),
        "avg_relevance": sum(results["relevance"]) / len(test_cases),
        "avg_citation_rate": sum(results["citation_rate"]) / len(test_cases)
    }
```

---

## 总结

Prompt模板设计的核心原则：

1. **System/User分离**：角色与任务分离，提高一致性
2. **结构化注入**：有序组织上下文，避免混乱
3. **约束明确**：清晰定义可做和不可做的事
4. **Context Engineering**：系统化管理上下文基础设施
5. **持续优化**：测试和迭代，找到最佳模式

**2025-2026标准：** Context Engineering已成为生产级RAG的标准实践。

---

**版本：** v1.0 (2025-2026最新实践)
**最后更新：** 2026-02-16
**参考来源：**
- Stack AI "Prompt Engineering for RAG" (2025-11)
- Redis "Context Engineering Best Practices" (2025-09)
