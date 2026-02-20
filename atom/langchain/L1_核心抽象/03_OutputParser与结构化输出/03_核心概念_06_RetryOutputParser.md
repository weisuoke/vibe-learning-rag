# 核心概念 6：RetryOutputParser（重试机制）

## 什么是 RetryOutputParser？

**RetryOutputParser 是一个包装器解析器，当解析失败时，使用原始 Prompt 重新调用 LLM 生成输出，适合处理临时性错误但会增加延迟和成本。**

---

## 1. 工作原理

```
LLM 输出 → Parser 解析
    ↓ 失败
使用原始 Prompt 重新调用 LLM
    ↓
新输出 → Parser 解析
    ↓ 失败
再次重试（最多 max_retries 次）
    ↓
成功或抛出异常
```

---

## 2. 基础用法

```python
from langchain.output_parsers import RetryWithErrorOutputParser, PydanticOutputParser
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

class Person(BaseModel):
    name: str
    age: int

# 1. 创建基础解析器
base_parser = PydanticOutputParser(pydantic_object=Person)

# 2. 创建重试解析器
llm = ChatOpenAI(model="gpt-4o-mini")
retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=base_parser,
    llm=llm,
    max_retries=3  # 最多重试 3 次
)

# 3. 需要提供原始 Prompt
prompt = PromptTemplate(
    template="提取人物信息：{text}\n\n{format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": base_parser.get_format_instructions()}
)

# 4. 使用 parse_with_prompt（需要 Prompt）
prompt_value = prompt.format_prompt(text="Alice is 25")
bad_output = '{"name": "Alice", "age": "invalid"}'
result = retry_parser.parse_with_prompt(bad_output, prompt_value)
print(result)  # Person(name='Alice', age=25)
```

---

## 3. 与 OutputFixingParser 的区别

### 3.1 策略对比

| 特性 | RetryOutputParser | OutputFixingParser |
|------|-------------------|-------------------|
| **策略** | 重新生成输出 | 修复错误输出 |
| **需要原始 Prompt** | ✅ 必需 | ❌ 不需要 |
| **LLM 调用** | 完整调用（重新生成） | 轻量调用（只修复） |
| **成本** | 高（完整 token） | 中（修复 token） |
| **适用错误** | 临时性错误、随机错误 | 格式错误、类型错误 |

### 3.2 代码对比

```python
# OutputFixingParser：不需要 Prompt
fixing_parser = OutputFixingParser.from_llm(base_parser, llm)
result = fixing_parser.parse(bad_output)  # 直接解析

# RetryOutputParser：需要 Prompt
retry_parser = RetryWithErrorOutputParser.from_llm(base_parser, llm)
result = retry_parser.parse_with_prompt(bad_output, prompt_value)  # 需要 Prompt
```

---

## 4. 重试策略

### 4.1 最大重试次数

```python
# 配置重试次数
retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=base_parser,
    llm=llm,
    max_retries=3  # 最多重试 3 次
)

# 重试流程：
# 尝试 1：原始输出 → 失败
# 尝试 2：重试 1 → 失败
# 尝试 3：重试 2 → 失败
# 尝试 4：重试 3 → 失败
# 抛出异常
```

### 4.2 指数退避（自定义）

```python
import time

class ExponentialRetryParser(RetryWithErrorOutputParser):
    def parse_with_prompt(self, completion, prompt_value):
        for attempt in range(self.max_retries + 1):
            try:
                if attempt == 0:
                    return self.parser.parse(completion)
                else:
                    # 指数退避
                    wait_time = 2 ** (attempt - 1)
                    time.sleep(wait_time)
                    # 重新调用 LLM
                    new_completion = self.llm.invoke(prompt_value)
                    return self.parser.parse(new_completion.content)
            except Exception as e:
                if attempt == self.max_retries:
                    raise
                continue
```

---

## 5. 成本分析

### 5.1 成本计算

```python
# 假设：1 次 LLM 调用 = $0.01

# 场景 1：第一次成功（0 次重试）
# 成本：$0.01

# 场景 2：第二次成功（1 次重试）
# 成本：$0.02

# 场景 3：第三次成功（2 次重试）
# 成本：$0.03

# 场景 4：全部失败（3 次重试）
# 成本：$0.04（原始 + 3 次重试）

# 平均成本（假设 50% 第一次成功，30% 第二次，20% 第三次）
# 平均成本：0.5 * $0.01 + 0.3 * $0.02 + 0.2 * $0.03 = $0.017
```

### 5.2 与其他方案对比

| 方案 | 平均成本 | 最坏成本 | 延迟 |
|------|----------|----------|------|
| **无重试** | $0.01 | $0.01 | 1x |
| **OutputFixingParser** | $0.015 | $0.02 | 1.5x |
| **RetryOutputParser** | $0.017 | $0.04 | 2x |

---

## 6. 何时使用

### 6.1 适合使用的场景

✅ **临时性错误**：网络波动、API 限流
✅ **随机错误**：LLM 偶尔返回错误格式
✅ **高可靠性要求**：必须成功解析
✅ **可接受延迟**：不是实时应用

### 6.2 不适合使用的场景

❌ **系统性错误**：Prompt 设计问题（应优化 Prompt）
❌ **实时应用**：延迟敏感
❌ **成本敏感**：重试成本累积
❌ **无 Prompt 上下文**：RetryOutputParser 需要原始 Prompt

---

## 7. 最佳实践

### 7.1 监控重试率

```python
class MonitoredRetryParser(RetryWithErrorOutputParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_calls = 0
        self.retry_counts = []

    def parse_with_prompt(self, completion, prompt_value):
        self.total_calls += 1
        retries = 0
        for attempt in range(self.max_retries + 1):
            try:
                if attempt == 0:
                    result = self.parser.parse(completion)
                else:
                    retries += 1
                    new_completion = self.llm.invoke(prompt_value)
                    result = self.parser.parse(new_completion.content)
                self.retry_counts.append(retries)
                return result
            except Exception as e:
                if attempt == self.max_retries:
                    self.retry_counts.append(self.max_retries + 1)
                    raise
                continue

    @property
    def avg_retries(self):
        return sum(self.retry_counts) / len(self.retry_counts) if self.retry_counts else 0

# 使用
parser = MonitoredRetryParser.from_llm(base_parser, llm, max_retries=3)
# ... 处理数据
print(f"平均重试次数: {parser.avg_retries:.2f}")
```

### 7.2 结合 OutputFixingParser

```python
# 策略：先修复，修复失败再重试
fixing_parser = OutputFixingParser.from_llm(base_parser, llm)
retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=fixing_parser,  # 包装修复解析器
    llm=llm,
    max_retries=2
)

# 流程：
# 1. 尝试解析
# 2. 失败 → 修复
# 3. 修复失败 → 重试
# 4. 重试失败 → 再次修复
# ...
```

---

## 8. 实际示例

### 8.1 处理网络波动

```python
from langchain.schema import OutputParserException

def robust_parse(text, max_retries=3):
    """容错解析（处理网络波动）"""
    retry_parser = RetryWithErrorOutputParser.from_llm(
        parser=base_parser,
        llm=llm,
        max_retries=max_retries
    )

    prompt_value = prompt.format_prompt(text=text)

    try:
        # 第一次尝试
        completion = llm.invoke(prompt_value)
        return retry_parser.parse_with_prompt(completion.content, prompt_value)
    except OutputParserException as e:
        logger.error(f"解析失败（已重试 {max_retries} 次）: {e}")
        return None
```

### 8.2 批量处理with重试

```python
def batch_parse_with_retry(texts, max_retries=2):
    """批量解析（带重试）"""
    retry_parser = RetryWithErrorOutputParser.from_llm(
        parser=base_parser,
        llm=llm,
        max_retries=max_retries
    )

    results = []
    for text in texts:
        prompt_value = prompt.format_prompt(text=text)
        try:
            completion = llm.invoke(prompt_value)
            result = retry_parser.parse_with_prompt(completion.content, prompt_value)
            results.append(result)
        except Exception as e:
            logger.error(f"解析失败: {text[:50]}... - {e}")
            results.append(None)

    return results
```

---

## 9. 调试和日志

### 9.1 记录重试过程

```python
class LoggingRetryParser(RetryWithErrorOutputParser):
    def parse_with_prompt(self, completion, prompt_value):
        logger.info(f"开始解析: {completion[:100]}...")

        for attempt in range(self.max_retries + 1):
            try:
                if attempt == 0:
                    result = self.parser.parse(completion)
                    logger.info("第一次尝试成功")
                    return result
                else:
                    logger.warning(f"重试 {attempt}/{self.max_retries}")
                    new_completion = self.llm.invoke(prompt_value)
                    result = self.parser.parse(new_completion.content)
                    logger.info(f"重试 {attempt} 成功")
                    return result
            except Exception as e:
                logger.error(f"尝试 {attempt + 1} 失败: {e}")
                if attempt == self.max_retries:
                    logger.error("所有重试都失败")
                    raise
                continue
```

---

## 10. 总结

### 核心要点

1. **重新生成**：失败时使用原始 Prompt 重新调用 LLM
2. **需要 Prompt**：必须提供原始 Prompt 上下文
3. **高成本**：每次重试都是完整的 LLM 调用
4. **适合临时错误**：网络波动、随机错误
5. **监控重试率**：如果 > 30%，应优化 Prompt

### 何时使用

- ✅ 临时性错误、高可靠性要求
- ❌ 系统性错误、实时应用、成本敏感

### 决策树

```
解析失败？
└─ 是 → 错误类型？
    ├─ 格式错误 → OutputFixingParser
    ├─ 临时错误 → RetryOutputParser
    └─ 系统性错误 → 优化 Prompt
```

---

**记住**：RetryOutputParser 适合处理临时性错误，但不是解决系统性问题的方案。应该优化 Prompt 降低错误率。
