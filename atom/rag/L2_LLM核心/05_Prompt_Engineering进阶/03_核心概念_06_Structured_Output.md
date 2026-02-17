# 核心概念 6: Structured Output

## 一句话定义

**通过JSON Schema强制约束大模型输出格式,确保生成的内容100%可被程序解析,消除格式不一致和解析错误。**

**RAG应用:** 在RAG系统中,Structured Output确保检索结果、答案评分、元数据提取等关键信息以统一的JSON格式输出,便于后续处理和展示。

---

## 为什么重要?

### 问题场景

```python
# 场景:从文档中提取结构化信息
from openai import OpenAI

client = OpenAI()

document = """
张三,30岁,软件工程师,擅长Python和JavaScript,5年经验。
李四,25岁,数据科学家,擅长机器学习,3年经验。
"""

# ❌ 自然语言输出:格式不可控
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": f"提取人员信息:\n{document}"}
    ]
)

print(response.choices[0].message.content)
# 可能输出:
# "张三是30岁的软件工程师,擅长Python和JavaScript..."
# 或:"姓名:张三\n年龄:30\n..."
# 或:其他格式
# 问题:每次格式不同,难以解析
```

### 解决方案

```python
# ✅ Structured Output:强制JSON格式
response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",  # 需要支持structured output的模型
    messages=[
        {"role": "user", "content": f"提取人员信息:\n{document}"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "people_info",
            "schema": {
                "type": "object",
                "properties": {
                    "people": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "age": {"type": "integer"},
                                "job": {"type": "string"},
                                "skills": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "experience_years": {"type": "integer"}
                            },
                            "required": ["name", "age", "job"]
                        }
                    }
                },
                "required": ["people"]
            }
        }
    }
)

import json
data = json.loads(response.choices[0].message.content)
print(json.dumps(data, indent=2, ensure_ascii=False))
# 输出:
# {
#   "people": [
#     {
#       "name": "张三",
#       "age": 30,
#       "job": "软件工程师",
#       "skills": ["Python", "JavaScript"],
#       "experience_years": 5
#     },
#     {
#       "name": "李四",
#       "age": 25,
#       "job": "数据科学家",
#       "skills": ["机器学习"],
#       "experience_years": 3
#     }
#   ]
# }
# 优势:格式100%一致,直接json.loads()解析
```

**性能提升:**

| 指标 | 自然语言 | Structured Output | 提升 |
|------|---------|------------------|------|
| 格式准确率 | 70% | 100% | +43% |
| 解析成功率 | 75% | 100% | +33% |
| 后处理时间 | 100ms | 10ms | -90% |

**来源:** [OpenAI Structured Outputs (2024)](https://openai.com/index/introducing-structured-outputs-in-the-api/)

---

## 核心原理

### 原理1:JSON Schema约束

**定义:** 使用JSON Schema定义输出结构,模型必须严格遵守。

**Schema结构:**

```python
schema = {
    "type": "object",  # 根类型
    "properties": {    # 属性定义
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "tags": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["name"]  # 必需字段
}
```

**支持的类型:**

```python
# 基础类型
{"type": "string"}   # 字符串
{"type": "integer"}  # 整数
{"type": "number"}   # 数字(含小数)
{"type": "boolean"}  # 布尔值
{"type": "null"}     # null

# 复合类型
{"type": "array", "items": {...}}  # 数组
{"type": "object", "properties": {...}}  # 对象

# 枚举
{"enum": ["A", "B", "C"]}  # 限定值

# 组合
{"anyOf": [{...}, {...}]}  # 任一满足
{"allOf": [{...}, {...}]}  # 全部满足
```

**来源:** [JSON Schema Specification](https://json-schema.org/)

---

### 原理2:约束解码

**定义:** 模型在生成时实时检查是否符合Schema,不符合的token不会被采样。

**工作机制:**

```
传统生成:
模型 → 采样token → 输出
      (可能不符合格式)

约束解码:
模型 → 采样token → 检查Schema → 输出
                  ↓
              不符合则重新采样
```

**示例:**

```python
# Schema要求age是integer
schema = {"type": "object", "properties": {"age": {"type": "integer"}}}

# 生成过程:
# 模型想输出: {"age": "30"}  ← 字符串,不符合
# 约束解码: 拒绝,重新采样
# 最终输出: {"age": 30}     ← 整数,符合
```

**来源:** [Constrained Decoding (2024)](https://arxiv.org/abs/2401.12345)

---

### 原理3:类型强制转换

**定义:** 模型会自动将内容转换为Schema要求的类型。

**转换规则:**

```python
# 字符串 → 整数
输入: "年龄是30岁"
Schema: {"age": {"type": "integer"}}
输出: {"age": 30}

# 文本 → 数组
输入: "擅长Python、JavaScript和Go"
Schema: {"skills": {"type": "array", "items": {"type": "string"}}}
输出: {"skills": ["Python", "JavaScript", "Go"]}

# 描述 → 布尔值
输入: "这个产品很好"
Schema: {"is_positive": {"type": "boolean"}}
输出: {"is_positive": true}
```

---

## 手写实现

### 从零实现 Structured Output Helper

```python
"""
Structured Output Helper
功能:简化Structured Output的使用
"""

from typing import Dict, List, Any, Optional, Type
from pydantic import BaseModel
from openai import OpenAI
import json

class StructuredOutputHelper:
    """Structured Output辅助类"""
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def pydantic_to_schema(self, model: Type[BaseModel]) -> Dict:
        """将Pydantic模型转换为JSON Schema"""
        return model.model_json_schema()
    
    def generate(
        self,
        prompt: str,
        schema: Dict,
        schema_name: str = "output",
        model: str = "gpt-4o-2024-08-06"
    ) -> Dict:
        """生成结构化输出"""
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema
                }
            }
        )
        
        return json.loads(response.choices[0].message.content)
    
    def generate_with_pydantic(
        self,
        prompt: str,
        model_class: Type[BaseModel],
        llm_model: str = "gpt-4o-2024-08-06"
    ) -> BaseModel:
        """使用Pydantic模型生成"""
        schema = self.pydantic_to_schema(model_class)
        result = self.generate(prompt, schema, model_class.__name__, llm_model)
        return model_class(**result)
    
    def batch_generate(
        self,
        prompts: List[str],
        schema: Dict,
        schema_name: str = "output",
        model: str = "gpt-4o-2024-08-06"
    ) -> List[Dict]:
        """批量生成"""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, schema, schema_name, model)
            results.append(result)
        return results


# 使用示例
if __name__ == "__main__":
    from dotenv import load_dotenv
    from pydantic import BaseModel, Field
    
    load_dotenv()
    
    client = OpenAI()
    helper = StructuredOutputHelper(client)
    
    # 方法1:直接使用Schema
    print("=== 方法1:直接使用Schema ===")
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "skills": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["name", "age"]
    }
    
    result1 = helper.generate(
        "提取:张三,30岁,擅长Python和JavaScript",
        schema
    )
    print(json.dumps(result1, indent=2, ensure_ascii=False))
    
    # 方法2:使用Pydantic模型
    print("\n=== 方法2:使用Pydantic模型 ===")
    
    class Person(BaseModel):
        name: str = Field(description="姓名")
        age: int = Field(description="年龄")
        job: str = Field(description="职业")
        skills: List[str] = Field(description="技能列表")
        experience_years: Optional[int] = Field(None, description="工作年限")
    
    result2 = helper.generate_with_pydantic(
        "提取:李四,25岁,数据科学家,擅长机器学习,3年经验",
        Person
    )
    print(result2.model_dump_json(indent=2, ensure_ascii=False))
    
    # 方法3:批量处理
    print("\n=== 方法3:批量处理 ===")
    prompts = [
        "提取:王五,35岁,产品经理",
        "提取:赵六,28岁,设计师"
    ]
    
    results3 = helper.batch_generate(prompts, schema)
    for i, result in enumerate(results3, 1):
        print(f"结果{i}:", json.dumps(result, ensure_ascii=False))
```

---

## RAG 应用场景

### 场景1:文档元数据提取

```python
from pydantic import BaseModel, Field
from typing import List

class DocumentMetadata(BaseModel):
    title: str = Field(description="文档标题")
    summary: str = Field(description="文档摘要")
    keywords: List[str] = Field(description="关键词列表")
    category: str = Field(description="文档分类")
    language: str = Field(description="语言")

def extract_metadata(document: str) -> DocumentMetadata:
    """提取文档元数据"""
    helper = StructuredOutputHelper(client)
    
    prompt = f"""
请分析以下文档并提取元数据:

{document}

提取:标题、摘要、关键词、分类、语言
"""
    
    return helper.generate_with_pydantic(prompt, DocumentMetadata)

# 测试
doc = """
Python是一种解释型、面向对象的编程语言。
它由Guido van Rossum于1991年创建。
Python以其简洁的语法和强大的功能而闻名。
"""

metadata = extract_metadata(doc)
print(metadata.model_dump_json(indent=2, ensure_ascii=False))
```

---

### 场景2:RAG答案评分

```python
class AnswerEvaluation(BaseModel):
    accuracy_score: int = Field(ge=0, le=10, description="准确性评分(0-10)")
    relevance_score: int = Field(ge=0, le=10, description="相关性评分(0-10)")
    completeness_score: int = Field(ge=0, le=10, description="完整性评分(0-10)")
    confidence_score: int = Field(ge=0, le=10, description="置信度评分(0-10)")
    reasoning: str = Field(description="评分理由")
    suggestions: List[str] = Field(description="改进建议")

def evaluate_rag_answer(
    query: str,
    answer: str,
    docs: List[str]
) -> AnswerEvaluation:
    """评估RAG答案质量"""
    helper = StructuredOutputHelper(client)
    
    prompt = f"""
评估以下RAG系统的答案质量:

问题:{query}
答案:{answer}
文档:{' | '.join(docs)}

请从准确性、相关性、完整性、置信度四个维度评分(0-10分),
并给出评分理由和改进建议。
"""
    
    return helper.generate_with_pydantic(prompt, AnswerEvaluation)
```

---

### 场景3:多文档信息聚合

```python
class AggregatedInfo(BaseModel):
    topic: str = Field(description="主题")
    key_points: List[str] = Field(description="关键要点")
    sources: List[str] = Field(description="信息来源")
    confidence: str = Field(description="置信度", enum=["高", "中", "低"])
    contradictions: Optional[List[str]] = Field(None, description="矛盾信息")

def aggregate_documents(query: str, docs: List[str]) -> AggregatedInfo:
    """聚合多个文档的信息"""
    helper = StructuredOutputHelper(client)
    
    prompt = f"""
基于以下文档回答问题:

问题:{query}

文档:
{chr(10).join(f"{i+1}. {doc}" for i, doc in enumerate(docs))}

请聚合信息,提取关键要点,标注来源,评估置信度,
并指出文档间的矛盾(如果有)。
"""
    
    return helper.generate_with_pydantic(prompt, AggregatedInfo)
```

---

## 最佳实践

### 1. Schema设计原则

```python
# ✅ 好:清晰的字段描述
schema = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "人员姓名"
        },
        "age": {
            "type": "integer",
            "description": "年龄(整数)",
            "minimum": 0,
            "maximum": 150
        }
    }
}

# ❌ 坏:缺少描述和约束
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    }
}
```

### 2. 使用Pydantic简化

```python
# ✅ 推荐:使用Pydantic
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(ge=0, le=150, description="年龄")

# 自动生成Schema
schema = Person.model_json_schema()
```

### 3. 错误处理

```python
def safe_generate(prompt: str, schema: Dict) -> Optional[Dict]:
    """安全生成,带错误处理"""
    try:
        result = helper.generate(prompt, schema)
        return result
    except json.JSONDecodeError:
        print("JSON解析失败")
        return None
    except Exception as e:
        print(f"生成失败:{e}")
        return None
```

### 4. 模型选择

```python
# Structured Output支持的模型
supported_models = [
    "gpt-4o-2024-08-06",
    "gpt-4o-mini-2024-07-18",
    # 其他支持的模型
]

# 不支持的模型会报错
unsupported = [
    "gpt-3.5-turbo",
    "gpt-4",
    # 旧版本模型
]
```

---

## 参考资源

- [OpenAI Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/)
- [JSON Schema Specification](https://json-schema.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [LLM Structured Output 2026](https://dev.to/pockit_tools/llm-structured-output-in-2026-stop-parsing-json-with-regex-and-do-it-right-34pk)
