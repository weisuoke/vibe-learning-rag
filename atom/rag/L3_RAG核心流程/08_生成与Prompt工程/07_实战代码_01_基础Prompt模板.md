# 实战代码1：基础Prompt模板

> 可运行的基础Prompt模板实现

---

## 代码示例

```python
"""
基础Prompt模板实战
演示：System/User分离、变量替换、格式控制
"""

from openai import OpenAI
import os
from typing import Dict, List

# 初始化客户端
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== 1. 基础模板类 =====

class BasicPromptTemplate:
    """
    基础Prompt模板
    """
    
    def __init__(self, system_template: str, user_template: str):
        self.system_template = system_template
        self.user_template = user_template
    
    def format(self, **kwargs) -> Dict[str, str]:
        """
        格式化模板
        """
        return {
            "system": self.system_template.format(**kwargs),
            "user": self.user_template.format(**kwargs)
        }

# ===== 2. RAG专用模板 =====

# System Prompt模板
SYSTEM_TEMPLATE = """
你是一个{role}。

核心能力：
{capabilities}

行为约束：
{constraints}
"""

# User Prompt模板
USER_TEMPLATE = """
## 参考资料

{context}

## 用户问题

{query}

## 回答要求

{requirements}

## 你的回答

"""

# ===== 3. 使用示例 =====

def example_basic_template():
    """
    基础模板使用示例
    """
    print("=== 示例1：基础模板 ===\n")
    
    # 创建模板
    template = BasicPromptTemplate(
        system_template=SYSTEM_TEMPLATE,
        user_template=USER_TEMPLATE
    )
    
    # 格式化
    prompts = template.format(
        role="严谨的知识助手",
        capabilities="- 基于参考资料提供准确答案\n- 识别信息不足的情况\n- 提供可追溯的引用",
        constraints="- 只使用参考资料中的信息\n- 不推测或添加额外内容\n- 如果资料不足，明确说明",
        context="Python是一种解释型、面向对象的编程语言。\nPython支持多种编程范式。",
        query="Python有什么特点？",
        requirements="1. 基于参考资料回答\n2. 为关键信息添加引用\n3. 如果信息不足，明确说明"
    )
    
    print("System Prompt:")
    print(prompts["system"])
    print("\nUser Prompt:")
    print(prompts["user"])
    
    # 调用LLM
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]}
        ],
        temperature=0.1
    )
    
    print("\n生成的答案:")
    print(response.choices[0].message.content)

# ===== 4. 动态模板 =====

class DynamicPromptTemplate:
    """
    动态Prompt模板（根据任务类型调整）
    """
    
    def __init__(self):
        self.templates = {
            "qa": {
                "system": "你是知识助手，基于参考资料回答问题。",
                "user": "参考资料：\n{context}\n\n问题：{query}\n\n回答："
            },
            "summarization": {
                "system": "你是总结专家，提炼参考资料的核心内容。",
                "user": "参考资料：\n{context}\n\n请总结核心内容："
            },
            "comparison": {
                "system": "你是分析专家，对比参考资料中的不同观点。",
                "user": "参考资料：\n{context}\n\n请对比分析："
            }
        }
    
    def get_template(self, task_type: str) -> Dict[str, str]:
        """
        获取任务类型对应的模板
        """
        return self.templates.get(task_type, self.templates["qa"])
    
    def format(self, task_type: str, **kwargs) -> Dict[str, str]:
        """
        格式化模板
        """
        template = self.get_template(task_type)
        return {
            "system": template["system"],
            "user": template["user"].format(**kwargs)
        }

def example_dynamic_template():
    """
    动态模板使用示例
    """
    print("\n=== 示例2：动态模板 ===\n")
    
    template = DynamicPromptTemplate()
    
    context = "Python是解释型语言。Java是编译型语言。"
    
    # QA任务
    prompts_qa = template.format(
        task_type="qa",
        context=context,
        query="Python和Java有什么区别？"
    )
    
    print("QA任务 System Prompt:")
    print(prompts_qa["system"])
    
    # 总结任务
    prompts_summary = template.format(
        task_type="summarization",
        context=context
    )
    
    print("\n总结任务 System Prompt:")
    print(prompts_summary["system"])

# ===== 5. 带验证的模板 =====

class ValidatedPromptTemplate:
    """
    带验证的Prompt模板
    """
    
    def __init__(self, system_template: str, user_template: str):
        self.system_template = system_template
        self.user_template = user_template
    
    def validate_inputs(self, **kwargs) -> bool:
        """
        验证输入参数
        """
        # 检查必需参数
        required = ["context", "query"]
        for param in required:
            if param not in kwargs:
                raise ValueError(f"缺少必需参数: {param}")
            if not kwargs[param]:
                raise ValueError(f"参数不能为空: {param}")
        
        # 检查上下文长度
        if len(kwargs["context"]) > 10000:
            raise ValueError("上下文过长（>10000字符）")
        
        return True
    
    def format(self, **kwargs) -> Dict[str, str]:
        """
        格式化模板（带验证）
        """
        self.validate_inputs(**kwargs)
        
        return {
            "system": self.system_template.format(**kwargs),
            "user": self.user_template.format(**kwargs)
        }

def example_validated_template():
    """
    带验证的模板示例
    """
    print("\n=== 示例3：带验证的模板 ===\n")
    
    template = ValidatedPromptTemplate(
        system_template="你是{role}",
        user_template="上下文：{context}\n问题：{query}"
    )
    
    try:
        # 正常情况
        prompts = template.format(
            role="助手",
            context="Python是编程语言",
            query="什么是Python？"
        )
        print("验证通过，模板格式化成功")
        
        # 缺少参数
        prompts = template.format(
            role="助手",
            context="Python是编程语言"
            # 缺少query
        )
    except ValueError as e:
        print(f"验证失败：{e}")

# ===== 6. 完整RAG模板 =====

class RAGPromptTemplate:
    """
    完整RAG Prompt模板
    """
    
    def __init__(self):
        self.system_base = """
你是一个专业的知识助手。

核心规则：
1. 只使用参考资料中的信息回答
2. 如果资料不足，明确说明
3. 为关键信息添加引用标记 [文档X]
4. 保持客观和准确
"""
    
    def build_context(self, docs: List[Dict]) -> str:
        """
        构建上下文
        """
        context_parts = []
        for i, doc in enumerate(docs):
            context_parts.append(f"[文档{i+1}] {doc['content']}")
        return "\n\n".join(context_parts)
    
    def build_prompt(
        self,
        query: str,
        docs: List[Dict],
        requirements: List[str] = None
    ) -> Dict[str, str]:
        """
        构建完整Prompt
        """
        # 构建上下文
        context = self.build_context(docs)
        
        # 构建要求
        if requirements is None:
            requirements = [
                "基于参考资料回答",
                "为关键信息添加引用 [文档X]",
                "如果信息不足，明确说明"
            ]
        
        requirements_str = "\n".join([
            f"{i+1}. {req}" for i, req in enumerate(requirements)
        ])
        
        # 构建User Prompt
        user_prompt = f"""
## 参考资料

{context}

## 用户问题

{query}

## 回答要求

{requirements_str}

## 你的回答

"""
        
        return {
            "system": self.system_base,
            "user": user_prompt
        }

def example_rag_template():
    """
    完整RAG模板示例
    """
    print("\n=== 示例4：完整RAG模板 ===\n")
    
    template = RAGPromptTemplate()
    
    # 准备文档
    docs = [
        {"content": "RAG是检索增强生成技术"},
        {"content": "RAG结合检索和生成两个步骤"},
        {"content": "RAG可以减少幻觉"}
    ]
    
    # 构建Prompt
    prompts = template.build_prompt(
        query="什么是RAG？",
        docs=docs
    )
    
    print("System Prompt:")
    print(prompts["system"])
    print("\nUser Prompt:")
    print(prompts["user"])
    
    # 调用LLM
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]}
        ],
        temperature=0.1
    )
    
    print("\n生成的答案:")
    print(response.choices[0].message.content)

# ===== 运行所有示例 =====

if __name__ == "__main__":
    example_basic_template()
    example_dynamic_template()
    example_validated_template()
    example_rag_template()
```

---

## 运行输出示例

```
=== 示例1：基础模板 ===

System Prompt:
你是一个严谨的知识助手。

核心能力：
- 基于参考资料提供准确答案
- 识别信息不足的情况
- 提供可追溯的引用

行为约束：
- 只使用参考资料中的信息
- 不推测或添加额外内容
- 如果资料不足，明确说明

User Prompt:
## 参考资料

Python是一种解释型、面向对象的编程语言。
Python支持多种编程范式。

## 用户问题

Python有什么特点？

## 回答要求

1. 基于参考资料回答
2. 为关键信息添加引用
3. 如果信息不足，明确说明

## 你的回答

生成的答案:
Python是一种解释型、面向对象的编程语言[文档1]，支持多种编程范式[文档1]。

=== 示例2：动态模板 ===

QA任务 System Prompt:
你是知识助手，基于参考资料回答问题。

总结任务 System Prompt:
你是总结专家，提炼参考资料的核心内容。

=== 示例3：带验证的模板 ===

验证通过，模板格式化成功
验证失败：缺少必需参数: query

=== 示例4：完整RAG模板 ===

System Prompt:
你是一个专业的知识助手。

核心规则：
1. 只使用参考资料中的信息回答
2. 如果资料不足，明确说明
3. 为关键信息添加引用标记 [文档X]
4. 保持客观和准确

User Prompt:
## 参考资料

[文档1] RAG是检索增强生成技术
[文档2] RAG结合检索和生成两个步骤
[文档3] RAG可以减少幻觉

## 用户问题

什么是RAG？

## 回答要求

1. 基于参考资料回答
2. 为关键信息添加引用 [文档X]
3. 如果信息不足，明确说明

## 你的回答

生成的答案:
RAG是检索增强生成技术[文档1]，它结合检索和生成两个步骤[文档2]，可以减少幻觉[文档3]。
```

---

## 关键要点

1. **System/User分离**：角色定义与任务指令分离
2. **变量替换**：使用`{}`占位符实现动态内容
3. **格式控制**：使用Markdown结构化组织内容
4. **输入验证**：检查必需参数和数据有效性
5. **模板复用**：根据任务类型选择合适模板

---

**版本：** v1.0
**最后更新：** 2026-02-16
