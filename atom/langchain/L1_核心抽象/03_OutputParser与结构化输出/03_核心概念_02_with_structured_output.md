# 核心概念 2：with_structured_output()（2025+ 推荐方法）

## 什么是 with_structured_output()？

**with_structured_output() 是 LangChain 1.0+ 提供的现代结构化输出方法，利用模型原生工具调用能力直接返回类型化对象，无需手动注入格式指令。**

这是 2025-2026 年的**最佳实践**，优先于传统 OutputParser。

---

## 1. 工作原理

### 1.1 技术架构

```
传统方法（PydanticOutputParser）：
用户输入 → Prompt + 格式指令 → LLM → 字符串 → 手动解析 → 对象

现代方法（with_structured_output）：
用户输入 → Prompt → LLM（原生工具调用）→ 对象
```

**核心差异**：
- **传统方法**：在 Prompt 中注入格式指令，LLM 返回字符串，应用层解析
- **现代方法**：利用模型原生能力（Function Calling / Tool Use），模型直接返回结构化数据

### 1.2 底层实现

```python
# with_structured_output() 内部实现（简化版）
def with_structured_output(self, schema: type[BaseModel]):
    """
    创建支持结构化输出的 LLM

    内部流程：
    1. 从 Pydantic 模型生成 JSON Schema
    2. 使用模型的原生工具调用 API
    3. 模型返回结构化数据（不是字符串）
    4. 自动转换为 Pydantic 对象
    """
    # 1. 生成 JSON Schema
    json_schema = schema.schema()

    # 2. 配置模型使用工具调用
    if self.supports_native_tool_calling:
        # OpenAI: 使用 response_format 参数
        # Anthropic: 使用 tools 参数
        return StructuredOutputRunnable(
            llm=self,
            schema=json_schema,
            output_type=schema
        )
    else:
        # 降级到传统 OutputParser
        return self | PydanticOutputParser(pydantic_object=schema)
```

---

## 2. 基础用法

### 2.1 最简单示例

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 1. 定义 Pydantic 模型
class Person(BaseModel):
    name: str = Field(description="人名")
    age: int = Field(description="年龄")

# 2. 创建结构化 LLM
llm = ChatOpenAI(model="gpt-4o")
structured_llm = llm.with_structured_output(Person)

# 3. 调用（直接返回 Person 对象）
result = structured_llm.invoke("Alice is 25 years old")
print(result)  # Person(name='Alice', age=25)
print(type(result))  # <class '__main__.Person'>
```

### 2.2 与传统方法对比

```python
# ❌ 传统方法：需要手动注入格式指令
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

parser = PydanticOutputParser(pydantic_object=Person)
prompt = PromptTemplate(
    template="Extract: {text}\n\n{format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
chain = prompt | llm | parser
result = chain.invoke({"text": "Alice is 25"})

# ✅ 现代方法：一步到位
structured_llm = llm.with_structured_output(Person)
result = structured_llm.invoke("Alice is 25")
```

---

## 3. 支持的模型（2025-2026）

### 3.1 原生支持列表

| 提供商 | 模型 | 支持方式 | 推荐度 |
|--------|------|----------|--------|
| **OpenAI** | GPT-4o, GPT-4o-mini | `response_format` | ⭐⭐⭐⭐⭐ |
| **OpenAI** | GPT-4-turbo | `response_format` | ⭐⭐⭐⭐ |
| **Anthropic** | Claude 3.5 Sonnet | Tool Use | ⭐⭐⭐⭐⭐ |
| **Anthropic** | Claude 3 Opus/Sonnet | Tool Use | ⭐⭐⭐⭐ |
| **Google** | Gemini 2.5 Pro | Function Calling | ⭐⭐⭐⭐ |
| **Google** | Gemini 1.5 Pro | Function Calling | ⭐⭐⭐ |

### 3.2 不支持的模型

- GPT-3.5-turbo（旧版本）
- 大部分开源模型（Llama 2, Mistral 等）
- 自部署模型（除非实现了工具调用）

**降级策略**：不支持的模型会自动降级到 PydanticOutputParser

---

## 4. 高级用法

### 4.1 response_format 参数（OpenAI）

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

# 方式 1：使用 with_structured_output()（推荐）
llm = ChatOpenAI(model="gpt-4o")
structured_llm = llm.with_structured_output(Person)

# 方式 2：直接配置 response_format（底层 API）
llm = ChatOpenAI(
    model="gpt-4o",
    model_kwargs={
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "Person",
                "schema": Person.schema()
            }
        }
    }
)
```

### 4.2 method 参数（选择策略）

```python
# 策略 1：JSON Mode（默认，OpenAI）
structured_llm = llm.with_structured_output(
    Person,
    method="json_mode"  # 使用 response_format
)

# 策略 2：Function Calling
structured_llm = llm.with_structured_output(
    Person,
    method="function_calling"  # 使用 tools 参数
)
```

**选择建议**：
- **json_mode**：更可靠，OpenAI 推荐（2025+ 默认）
- **function_calling**：兼容性更好，适合旧版本

### 4.3 include_raw 参数（保留原始响应）

```python
structured_llm = llm.with_structured_output(
    Person,
    include_raw=True  # 返回原始响应 + 解析结果
)

result = structured_llm.invoke("Alice is 25")
print(result)
# {
#     "raw": AIMessage(...),  # 原始 LLM 响应
#     "parsed": Person(name='Alice', age=25)  # 解析后的对象
# }
```

**使用场景**：
- 调试和日志记录
- 需要访问原始 token 使用量
- 需要查看模型的原始输出

---

## 5. 多提供商示例

### 5.1 OpenAI

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
structured_llm = llm.with_structured_output(Person)
result = structured_llm.invoke("Extract: Bob, 30")
```

### 5.2 Anthropic

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
structured_llm = llm.with_structured_output(Person)
result = structured_llm.invoke("Extract: Charlie, 35")
```

### 5.3 Google Gemini

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
structured_llm = llm.with_structured_output(Person)
result = structured_llm.invoke("Extract: David, 40")
```

---

## 6. 与 PydanticOutputParser 的对比

### 6.1 功能对比

| 特性 | with_structured_output | PydanticOutputParser |
|------|------------------------|----------------------|
| **模型支持** | 需要原生工具调用 | 所有模型 |
| **可靠性** | ⭐⭐⭐⭐⭐ 最高 | ⭐⭐⭐ 中等 |
| **性能** | ⭐⭐⭐⭐⭐ 最快 | ⭐⭐⭐⭐ 较快 |
| **Token 消耗** | ⭐⭐⭐⭐⭐ 最少 | ⭐⭐⭐ 中等（格式指令） |
| **代码复杂度** | ⭐⭐⭐⭐⭐ 最简单 | ⭐⭐⭐ 中等 |
| **错误率** | ⭐⭐⭐⭐⭐ 最低 | ⭐⭐⭐ 中等 |

### 6.2 代码对比

```python
# with_structured_output：3 行代码
llm = ChatOpenAI(model="gpt-4o")
structured_llm = llm.with_structured_output(Person)
result = structured_llm.invoke("Alice, 25")

# PydanticOutputParser：10+ 行代码
parser = PydanticOutputParser(pydantic_object=Person)
prompt = PromptTemplate(
    template="Extract: {text}\n\n{format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
chain = prompt | llm | parser
result = chain.invoke({"text": "Alice, 25"})
```

### 6.3 成本对比

```python
# 假设：提取 100 个人物信息

# with_structured_output
# - 无格式指令
# - Token 消耗：~100 tokens/次
# - 总成本：$0.10

# PydanticOutputParser
# - 格式指令：~200 tokens
# - Token 消耗：~300 tokens/次
# - 总成本：$0.30

# 节省：66% 成本
```

---

## 7. 最佳实践

### 7.1 选择合适的模型

```python
# ✅ 推荐：使用支持原生结构化输出的模型
llm = ChatOpenAI(model="gpt-4o")  # 原生支持

# ❌ 不推荐：使用不支持的模型
llm = ChatOpenAI(model="gpt-3.5-turbo")  # 会降级到 PydanticOutputParser
```

### 7.2 错误处理

```python
from pydantic import ValidationError

try:
    result = structured_llm.invoke("Extract person info")
except ValidationError as e:
    print(f"验证失败: {e}")
    # 处理错误
```

### 7.3 批量处理

```python
# 批量提取
texts = ["Alice, 25", "Bob, 30", "Charlie, 35"]
results = structured_llm.batch(texts)

for person in results:
    print(person)
```

### 7.4 流式输出（如果支持）

```python
# 某些模型支持流式结构化输出
for chunk in structured_llm.stream("Alice, 25"):
    print(chunk)
```

---

## 8. 实际应用场景

### 8.1 RAG 元数据提取

```python
from typing import List

class DocumentMetadata(BaseModel):
    title: str
    author: str
    keywords: List[str]
    summary: str

llm = ChatOpenAI(model="gpt-4o")
extractor = llm.with_structured_output(DocumentMetadata)

metadata = extractor.invoke(f"Extract metadata from: {document_text}")
vector_store.add_documents(
    documents=[document_text],
    metadatas=[metadata.dict()]
)
```

### 8.2 Agent 工具响应

```python
class WeatherResponse(BaseModel):
    city: str
    temperature: float
    condition: str

weather_formatter = llm.with_structured_output(WeatherResponse)
formatted = weather_formatter.invoke(f"Format: {raw_weather_data}")
```

### 8.3 数据验证和清洗

```python
class CleanedData(BaseModel):
    email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    phone: str = Field(pattern=r"^\d{11}$")
    age: int = Field(ge=0, le=150)

cleaner = llm.with_structured_output(CleanedData)
cleaned = cleaner.invoke(f"Clean and validate: {raw_data}")
```

---

## 9. 2025-2026 更新

### 9.1 LangChain 1.0 改进

- **默认方法变更**：`json_mode` 成为 OpenAI 的默认方法
- **性能优化**：减少了内部转换开销
- **错误提示**：更清晰的验证错误信息

### 9.2 模型支持扩展

- **Gemini 2.5**：新增原生支持（2026）
- **Claude 3.5**：改进的工具调用精度
- **GPT-4o**：更快的响应速度

### 9.3 新增功能

```python
# 2026 新增：strict 模式（OpenAI）
structured_llm = llm.with_structured_output(
    Person,
    strict=True  # 严格模式，100% 遵守 schema
)
```

---

## 10. 何时使用 / 何时不使用

### 10.1 适合使用的场景

✅ **模型支持原生结构化输出**（OpenAI GPT-4o、Anthropic Claude 3.5+）
✅ **需要高可靠性**（生产环境）
✅ **成本敏感**（减少 token 消耗）
✅ **代码简洁性**（减少样板代码）
✅ **实时应用**（低延迟要求）

### 10.2 不适合使用的场景

❌ **模型不支持原生结构化输出**（会自动降级，不如直接用 PydanticOutputParser）
❌ **需要兼容旧代码**（已有 PydanticOutputParser 实现）
❌ **特殊格式解析**（非标准 JSON，需要自定义 Parser）

---

## 11. 迁移指南

### 11.1 从 PydanticOutputParser 迁移

```python
# 旧代码
parser = PydanticOutputParser(pydantic_object=Person)
prompt = PromptTemplate(
    template="Extract: {text}\n\n{format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
chain = prompt | llm | parser

# 新代码
structured_llm = llm.with_structured_output(Person)
# 移除格式指令，简化 Prompt
prompt = PromptTemplate(
    template="Extract: {text}",
    input_variables=["text"]
)
chain = prompt | structured_llm
```

### 11.2 迁移检查清单

- [ ] 确认模型支持原生结构化输出
- [ ] 移除 Prompt 中的格式指令
- [ ] 替换 `| parser` 为 `with_structured_output()`
- [ ] 测试输出质量
- [ ] 监控成本变化（应该降低）

---

## 12. 总结

### 核心要点

1. **现代最佳实践**：2025-2026 年优先使用 with_structured_output()
2. **原生能力**：利用模型的工具调用能力，不需要格式指令
3. **更可靠**：模型训练时优化了结构化输出
4. **更简单**：3 行代码完成，无需手动注入格式指令
5. **更便宜**：减少 token 消耗，降低成本

### 决策树

```
需要结构化输出？
└─ 是 → 模型支持原生工具调用？
    ├─ 是 → 使用 with_structured_output() ✅
    └─ 否 → 使用 PydanticOutputParser
```

---

**记住**：with_structured_output() 是 2025-2026 年的标准方法，优先使用它而不是传统 OutputParser。

**来源**：
- https://python.langchain.com/docs/how_to/structured_output/
- https://blog.langchain.com/langchain-langgraph-1dot0
- https://mirascope.com/blog/langchain-structured-output
