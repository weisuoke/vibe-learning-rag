# 核心概念 5：OutputFixingParser（自动修复）

## 什么是 OutputFixingParser？

**OutputFixingParser 是一个包装器解析器，当基础解析器失败时，自动调用 LLM 修复错误的输出，适合高价值任务但会增加额外的 LLM 调用成本。**

---

## 1. 工作原理

```
LLM 输出 → 基础 Parser 解析
    ↓ 失败
调用 LLM 修复（传入错误信息 + 原始输出）
    ↓
修复后的输出 → 基础 Parser 再次解析
    ↓ 成功
返回结果
```

---

## 2. 基础用法

```python
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

class Person(BaseModel):
    name: str
    age: int

# 1. 创建基础解析器
base_parser = PydanticOutputParser(pydantic_object=Person)

# 2. 包装为修复解析器
llm = ChatOpenAI(model="gpt-4o-mini")
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=llm
)

# 3. 解析错误的输出（会自动修复）
bad_output = '{"name": "Alice", "age": "twenty-five"}'  # age 应该是数字
result = fixing_parser.parse(bad_output)
print(result)  # Person(name='Alice', age=25)
```

---

## 3. 修复过程详解

### 3.1 内部实现

```python
class OutputFixingParser(BaseOutputParser):
    def parse(self, completion: str) -> T:
        try:
            # 尝试用基础 parser 解析
            return self.parser.parse(completion)
        except Exception as e:
            # 解析失败，调用 LLM 修复
            fix_prompt = f"""
原始输出有错误：
{completion}

错误信息：
{str(e)}

请修复输出，使其符合以下格式：
{self.parser.get_format_instructions()}
"""
            fixed_output = self.llm.invoke(fix_prompt)
            # 再次解析修复后的输出
            return self.parser.parse(fixed_output.content)
```

### 3.2 修复示例

```python
# 错误 1：类型错误
bad_output = '{"name": "Alice", "age": "twenty-five"}'
# LLM 修复为: '{"name": "Alice", "age": 25}'

# 错误 2：字段缺失
bad_output = '{"name": "Bob"}'
# LLM 修复为: '{"name": "Bob", "age": 30}'  # 推测年龄

# 错误 3：格式错误
bad_output = 'The person is Alice, 25 years old'
# LLM 修复为: '{"name": "Alice", "age": 25}'
```

---

## 4. 成本分析

### 4.1 额外 LLM 调用

```python
# 场景 1：输出正确（0 次额外调用）
good_output = '{"name": "Alice", "age": 25}'
result = fixing_parser.parse(good_output)
# 成本：$0（无额外调用）

# 场景 2：输出错误（1 次额外调用）
bad_output = '{"name": "Alice", "age": "twenty-five"}'
result = fixing_parser.parse(bad_output)
# 成本：$0.01（1 次修复调用）

# 场景 3：批量处理（假设 50% 错误率）
outputs = [good1, bad1, good2, bad2, ...]  # 100 个输出
results = [fixing_parser.parse(o) for o in outputs]
# 成本：$0.50（50 次修复调用）
```

### 4.2 成本对比

| 方案 | 基础成本 | 修复成本 | 总成本 | 适用场景 |
|------|----------|----------|--------|----------|
| **无修复** | $1.00 | $0 | $1.00 | 低价值任务 |
| **OutputFixingParser** | $1.00 | $0.50 | $1.50 | 高价值任务 |
| **优化 Prompt** | $1.00 | $0 | $1.00 | 最佳方案 |

---

## 5. 何时使用

### 5.1 适合使用的场景

✅ **高价值任务**：数据准确性比成本更重要
✅ **复杂结构**：嵌套对象、多字段验证
✅ **不稳定的 LLM**：输出质量不稳定
✅ **原型开发**：快速验证想法

### 5.2 不适合使用的场景

❌ **成本敏感**：额外 LLM 调用成本不可接受
❌ **实时应用**：延迟敏感（修复增加延迟）
❌ **大批量处理**：修复成本会累积
❌ **可优化 Prompt**：应该优化 Prompt 而非依赖修复

---

## 6. 最佳实践

### 6.1 监控修复率

```python
class MonitoredFixingParser(OutputFixingParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_calls = 0
        self.fix_calls = 0

    def parse(self, completion: str):
        self.total_calls += 1
        try:
            return self.parser.parse(completion)
        except Exception as e:
            self.fix_calls += 1
            # 修复逻辑
            return super().parse(completion)

    @property
    def fix_rate(self):
        return self.fix_calls / self.total_calls if self.total_calls > 0 else 0

# 使用
parser = MonitoredFixingParser.from_llm(base_parser, llm)
# ... 处理数据
print(f"修复率: {parser.fix_rate:.2%}")  # 如果 > 20%，应优化 Prompt
```

### 6.2 设置修复预算

```python
class BudgetedFixingParser(OutputFixingParser):
    def __init__(self, *args, max_fixes=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_fixes = max_fixes
        self.fix_count = 0

    def parse(self, completion: str):
        try:
            return self.parser.parse(completion)
        except Exception as e:
            if self.fix_count >= self.max_fixes:
                raise ValueError(f"超过修复预算 ({self.max_fixes})")
            self.fix_count += 1
            return super().parse(completion)
```

---

## 7. 与其他方案对比

### 7.1 vs RetryOutputParser

| 特性 | OutputFixingParser | RetryOutputParser |
|------|-------------------|-------------------|
| **策略** | 修复错误输出 | 重新生成输出 |
| **需要原始 Prompt** | ❌ 不需要 | ✅ 需要 |
| **适用错误** | 格式错误、类型错误 | 临时性错误 |
| **成本** | 中等（修复调用） | 高（完整重试） |

### 7.2 vs 优化 Prompt

| 方案 | 成本 | 可靠性 | 开发时间 |
|------|------|--------|----------|
| **OutputFixingParser** | 高 | 高 | 短 |
| **优化 Prompt** | 低 | 高 | 长 |

**建议**：先用 OutputFixingParser 快速验证，然后优化 Prompt 降低修复率。

---

## 8. 实际示例

### 8.1 复杂嵌套结构

```python
from typing import List

class Address(BaseModel):
    street: str
    city: str

class Person(BaseModel):
    name: str
    age: int
    addresses: List[Address]

base_parser = PydanticOutputParser(pydantic_object=Person)
fixing_parser = OutputFixingParser.from_llm(base_parser, llm)

# 复杂输出容易出错，修复器可以自动修正
bad_output = '''
{
    "name": "Alice",
    "age": "25",
    "addresses": [
        {"street": "123 Main St", "city": "Beijing"},
        {"street": "456 Park Ave"}  // 缺少 city
    ]
}
'''
result = fixing_parser.parse(bad_output)
```

---

## 9. 总结

### 核心要点

1. **自动修复**：解析失败时调用 LLM 修复
2. **额外成本**：每次修复增加 1 次 LLM 调用
3. **高价值任务**：适合准确性比成本更重要的场景
4. **监控修复率**：如果 > 20%，应优化 Prompt

### 何时使用

- ✅ 高价值任务、复杂结构
- ❌ 成本敏感、实时应用

---

**记住**：OutputFixingParser 是"安全网"，不是长期方案。应该优化 Prompt 降低修复率。
