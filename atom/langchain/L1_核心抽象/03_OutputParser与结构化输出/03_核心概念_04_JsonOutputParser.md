# 核心概念 4：JsonOutputParser（简单解析）

## 什么是 JsonOutputParser？

**JsonOutputParser 是 LangChain 中最简单的输出解析器，只做基本的 JSON 解析，不进行 Pydantic 验证，适合不需要严格类型检查的场景。**

---

## 1. 基础用法

```python
from langchain.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 创建解析器（无需 Pydantic 模型）
parser = JsonOutputParser()

# 创建 Prompt
prompt = PromptTemplate(
    template="提取人物信息并返回 JSON：{text}",
    input_variables=["text"]
)

# 构建链
llm = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | llm | parser

# 调用
result = chain.invoke({"text": "Alice is 25 years old"})
print(result)  # {'name': 'Alice', 'age': 25}
print(type(result))  # <class 'dict'>
```

---

## 2. 与 PydanticOutputParser 对比

| 特性 | JsonOutputParser | PydanticOutputParser |
|------|------------------|----------------------|
| **类型检查** | ❌ 无 | ✅ 有 |
| **数据验证** | ❌ 无 | ✅ 有 |
| **返回类型** | dict | Pydantic 对象 |
| **性能** | ⭐⭐⭐⭐⭐ 最快 | ⭐⭐⭐⭐ 较快 |
| **代码复杂度** | ⭐⭐⭐⭐⭐ 最简单 | ⭐⭐⭐ 中等 |
| **适用场景** | 简单场景、原型开发 | 生产环境、需要验证 |

---

## 3. 何时使用

### 3.1 适合使用的场景

✅ **原型开发**：快速验证想法
✅ **简单结构**：只有几个字段，不需要验证
✅ **灵活格式**：不确定返回的字段
✅ **性能敏感**：需要最快的解析速度

### 3.2 不适合使用的场景

❌ **生产环境**：需要类型安全和验证
❌ **复杂结构**：嵌套对象、列表等
❌ **严格验证**：需要字段约束（如年龄 0-150）
❌ **类型提示**：需要 IDE 自动补全

---

## 4. 实际示例

### 4.1 简单信息提取

```python
parser = JsonOutputParser()
prompt = PromptTemplate(
    template="从以下文本提取关键信息并返回 JSON：{text}",
    input_variables=["text"]
)
chain = prompt | llm | parser

result = chain.invoke({"text": "北京今天晴天，温度 15 度"})
print(result)  # {'city': '北京', 'weather': '晴天', 'temperature': 15}
```

### 4.2 灵活字段提取

```python
# 不确定会返回哪些字段
texts = [
    "Alice, 25 years old, engineer",
    "Bob, teacher",  # 没有年龄
    "Charlie, 30, doctor, married"  # 额外字段
]

results = chain.batch([{"text": t} for t in texts])
for result in results:
    print(result)
# {'name': 'Alice', 'age': 25, 'job': 'engineer'}
# {'name': 'Bob', 'job': 'teacher'}
# {'name': 'Charlie', 'age': 30, 'job': 'doctor', 'status': 'married'}
```

---

## 5. 错误处理

```python
from langchain.schema import OutputParserException

try:
    result = parser.parse("This is not JSON")
except OutputParserException as e:
    print(f"解析失败: {e}")
    result = {}
```

---

## 6. 升级到 PydanticOutputParser

```python
# 当需要验证时，升级到 PydanticOutputParser
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

# 从 JsonOutputParser 升级
parser = PydanticOutputParser(pydantic_object=Person)
```

---

## 7. 总结

### 核心要点

1. **最简单**：只做 JSON 解析，无验证
2. **返回 dict**：不是 Pydantic 对象
3. **快速原型**：适合快速开发和测试
4. **生产慎用**：缺少类型安全和验证

### 何时使用

- ✅ 原型开发、简单场景
- ❌ 生产环境、需要验证

---

**记住**：JsonOutputParser 适合快速原型，生产环境应使用 PydanticOutputParser 或 with_structured_output()。
