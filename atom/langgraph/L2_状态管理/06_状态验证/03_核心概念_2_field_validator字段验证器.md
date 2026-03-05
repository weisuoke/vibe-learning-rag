# 核心概念 2：field_validator 字段验证器

## 概述

`field_validator` 是 Pydantic v2 提供的字段级验证装饰器，用于对单个字段的值进行自定义校验和转换。在 LangGraph 中，当你使用 `BaseModel` 定义状态时，`field_validator` 可以在每次状态更新时自动验证数据的合法性，比如清洗用户输入、限制 token 范围、规范化文本格式等。

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

## 第一性原理

### 为什么需要字段级验证?

LangGraph 的节点在运行过程中会不断更新状态。如果某个节点写入了非法数据（空字符串、超范围数值、格式错误的输出），后续节点可能会静默出错或产生不可预期的行为。

核心矛盾在于：**节点是独立编写的，但状态是共享的。**

没有验证机制时，你只能在每个节点内部手动检查输入，这会导致：
1. 验证逻辑散落在各个节点中，难以维护
2. 不同节点对同一字段的校验标准可能不一致
3. 遗漏校验时 bug 难以追踪

`field_validator` 把验证逻辑集中到状态定义中，无论哪个节点更新状态，都会自动触发验证。这就像在数据库层面加约束，而不是在每个 SQL 语句里手动检查。

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

## 基础语法

### 最简形式

```python
from pydantic import BaseModel, field_validator

class AgentState(BaseModel):
    query: str
    max_tokens: int = 1000

    @field_validator('query')
    @classmethod
    def query_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('查询不能为空')
        return v.strip()

    @field_validator('max_tokens')
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        if v < 1 or v > 4096:
            raise ValueError('max_tokens 必须在 1-4096 之间')
        return v
```

**关键语法要素**：

| 要素 | 说明 |
|------|------|
| `@field_validator('字段名')` | 指定要验证的字段 |
| `@classmethod` | 必须紧跟在 `@field_validator` 下方 |
| `cls` | 第一个参数，类本身（classmethod 要求） |
| `v` | 第二个参数，字段的值 |
| `return v` | 必须返回值，不返回等于返回 `None` |
| `raise ValueError(...)` | 验证失败时抛出异常 |

### 验证效果

```python
state = AgentState(query="什么是 RAG?")       # ✅ 正常
state = AgentState(query="  什么是 RAG?  ")   # ✅ 自动清洗为 "什么是 RAG?"

try:
    AgentState(query="   ")                    # ❌ ValueError: 查询不能为空
except Exception as e:
    print(e)

try:
    AgentState(query="test", max_tokens=9999)  # ❌ ValueError: max_tokens 超出范围
except Exception as e:
    print(e)
```

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

## before vs after 模式

`field_validator` 有两种执行模式，决定了验证器在 Pydantic 类型转换流程中的位置。

### mode='after'（默认）

在 Pydantic 完成类型转换之后执行。此时 `v` 已经是目标类型。

```python
from pydantic import BaseModel, field_validator

class AgentState(BaseModel):
    temperature: float = 0.7

    @field_validator('temperature', mode='after')
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        # v 已经是 float 类型（Pydantic 已完成转换）
        if v < 0.0 or v > 2.0:
            raise ValueError('temperature 必须在 0.0-2.0 之间')
        return v

# Pydantic 先把字符串 "0.8" 转成 float 0.8，再交给验证器
state = AgentState(temperature="0.8")  # ✅ 正常
print(state.temperature)  # 0.8
print(type(state.temperature))  # <class 'float'>
```

### mode='before'

在 Pydantic 类型转换之前执行。此时 `v` 是原始输入值，类型不确定。

```python
from pydantic import BaseModel, field_validator

class AgentState(BaseModel):
    temperature: float = 0.7

    @field_validator('temperature', mode='before')
    @classmethod
    def coerce_temperature(cls, v) -> float:
        # v 是原始输入，可能是 str、int、float 等任何类型
        if isinstance(v, str):
            v = v.strip()
            if v.endswith('%'):
                # 支持百分比格式："70%" -> 0.7
                return float(v[:-1]) / 100
        return v  # 返回后 Pydantic 继续做类型转换

state = AgentState(temperature="70%")  # ✅ 百分比格式
print(state.temperature)  # 0.7
```

### 两者区别对比

```
输入数据 → [before 验证器] → [Pydantic 类型转换] → [after 验证器] → 最终值
```

| 特性 | mode='before' | mode='after'（默认） |
|------|---------------|---------------------|
| 执行时机 | 类型转换之前 | 类型转换之后 |
| 接收的 `v` 类型 | 原始输入（Any） | 已转换的目标类型 |
| 典型用途 | 格式预处理、类型兼容 | 值范围校验、业务规则 |
| 类型安全 | 低（需自行判断类型） | 高（类型已确定） |

### 组合使用 before + after

```python
from pydantic import BaseModel, field_validator

class AgentState(BaseModel):
    score: float

    @field_validator('score', mode='before')
    @classmethod
    def preprocess_score(cls, v):
        """before: 处理百分比格式输入"""
        if isinstance(v, str) and v.strip().endswith('%'):
            return float(v.strip().rstrip('%')) / 100
        return v

    @field_validator('score', mode='after')
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """after: 验证最终值的范围"""
        if v < 0.0 or v > 1.0:
            raise ValueError('score 必须在 0.0-1.0 之间')
        return round(v, 4)

print(AgentState(score="85%").score)   # 0.85
print(AgentState(score=0.9).score)     # 0.9
```

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

## 多字段验证

### 同一验证器应用于多个字段

当多个字段需要相同的验证逻辑时，可以在装饰器中列出所有字段名：

```python
from pydantic import BaseModel, field_validator

class AgentState(BaseModel):
    query: str
    context: str
    response: str = ""

    @field_validator('query', 'context')
    @classmethod
    def validate_text_fields(cls, v: str) -> str:
        if len(v) > 10000:
            raise ValueError('文本长度不能超过 10000 字符')
        return v
```

这比为每个字段写一个验证器要简洁得多。验证器会对列出的每个字段独立执行。

### 使用 '*' 验证所有字段

`'*'` 表示对模型中的所有字段都执行该验证器：

```python
from pydantic import BaseModel, field_validator

class AgentState(BaseModel):
    query: str
    context: str
    system_prompt: str

    @field_validator('*')
    @classmethod
    def no_field_is_none_string(cls, v):
        """确保没有字段的值是字符串 'None'"""
        if isinstance(v, str) and v.strip().lower() == 'none':
            raise ValueError("字段值不能是字符串 'None'")
        return v
```

**注意**：`'*'` 验证器会对每个字段都执行。如果字段类型不同（`str` 和 `int` 混合），验证器内部需要做类型判断：

```python
@field_validator('*', mode='before')
@classmethod
def strip_strings(cls, v):
    if isinstance(v, str):
        return v.strip()
    return v
```

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

## ValidationInfo 上下文

验证器可以接收第三个参数 `info`，类型为 `ValidationInfo`，提供当前验证的上下文信息。

### 基础用法

```python
from pydantic import BaseModel, field_validator, ValidationInfo

class AgentState(BaseModel):
    query: str
    max_tokens: int = 1000
    response: str = ""

    @field_validator('response')
    @classmethod
    def validate_response(cls, v: str, info: ValidationInfo) -> str:
        # info.field_name: 当前正在验证的字段名
        print(f"正在验证字段: {info.field_name}")

        # info.data: 已经通过验证的字段数据（dict）
        # 注意：只包含在当前字段之前定义的字段
        if 'query' in info.data:
            query = info.data['query']
            print(f"对应的查询: {query}")

        return v
```

### 跨字段依赖验证

`info.data` 只包含在当前字段之前已验证通过的字段（按定义顺序）。

```python
from pydantic import BaseModel, field_validator, ValidationInfo

class AgentState(BaseModel):
    model_name: str = "gpt-4"       # 先定义
    max_tokens: int = 1000          # 后定义，可以访问 model_name

    @field_validator('max_tokens')
    @classmethod
    def validate_max_tokens(cls, v: int, info: ValidationInfo) -> int:
        """根据模型名称限制 max_tokens 范围"""
        model = info.data.get('model_name', 'gpt-4')
        limits = {'gpt-4': 8192, 'gpt-3.5-turbo': 4096}
        max_limit = limits.get(model, 4096)
        if v > max_limit:
            raise ValueError(f'{model} 的 max_tokens 上限为 {max_limit}')
        return v

state = AgentState(model_name="gpt-3.5-turbo", max_tokens=2000)  # ✅
try:
    AgentState(model_name="gpt-3.5-turbo", max_tokens=5000)      # ❌
except Exception as e:
    print(e)  # gpt-3.5-turbo 的 max_tokens 上限为 4096
```

**字段顺序很重要**：`max_tokens` 的验证器能访问 `model_name`，是因为 `model_name` 在前面定义。顺序反过来，`info.data` 中就不会有 `model_name`。

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

## 在 LangGraph 中的实际应用

### 场景 1：清洗用户输入

```python
from pydantic import BaseModel, field_validator
import re

class AgentState(BaseModel):
    query: str

    @field_validator('query')
    @classmethod
    def clean_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError('查询不能为空')
        if len(v) > 2000:
            raise ValueError('查询长度不能超过 2000 字符')
        return re.sub(r'\s+', ' ', v)
```

### 场景 2：验证 LLM 输出格式

```python
import json
from pydantic import BaseModel, field_validator

class AgentState(BaseModel):
    query: str
    raw_response: str = ""
    structured_output: dict = {}

    @field_validator('structured_output', mode='before')
    @classmethod
    def parse_structured_output(cls, v):
        """支持 JSON 字符串自动解析为 dict"""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                raise ValueError(
                    f'structured_output 不是有效的 JSON: {v[:100]}...'
                )
        return v
```

### 场景 3：规范化数据格式

```python
from pydantic import BaseModel, field_validator

class AgentState(BaseModel):
    query: str
    sources: list[str] = []

    @field_validator('sources')
    @classmethod
    def deduplicate_sources(cls, v: list[str]) -> list[str]:
        """去重并保持顺序"""
        seen = set()
        result = []
        for s in v:
            if s not in seen:
                seen.add(s)
                result.append(s)
        return result
```

### 场景 4：限制字段值范围

```python
from pydantic import BaseModel, field_validator

ALLOWED_MODELS = ['gpt-4', 'gpt-3.5-turbo', 'gpt-4-turbo', 'claude-3-opus']

class AgentState(BaseModel):
    model_name: str = "gpt-4"
    temperature: float = 0.7

    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if v not in ALLOWED_MODELS:
            raise ValueError(f'不支持的模型: {v}')
        return v

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if v < 0.0 or v > 2.0:
            raise ValueError('temperature 必须在 0.0-2.0 之间')
        return v
```

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

## 常见陷阱

### 陷阱 1：忘记 @classmethod

```python
# ❌ 错误：缺少 @classmethod，Pydantic v2 会报错
@field_validator('query')
def validate_query(cls, v: str) -> str:
    return v.strip()

# ✅ 正确：@classmethod 紧跟 @field_validator
@field_validator('query')
@classmethod
def validate_query(cls, v: str) -> str:
    return v.strip()
```

### 陷阱 2：验证器没有返回值

```python
# ❌ 错误：忘记 return，Python 默认返回 None
@field_validator('query')
@classmethod
def validate_query(cls, v: str) -> str:
    v.strip()  # 只调用了 strip()，没有 return → 字段值变成 None

# ✅ 正确：始终返回处理后的值
@field_validator('query')
@classmethod
def validate_query(cls, v: str) -> str:
    return v.strip()
```

### 陷阱 3：before 模式下假设类型

```python
# ❌ 错误：before 模式下 v 不一定是 str
@field_validator('query', mode='before')
@classmethod
def validate_query(cls, v: str) -> str:
    return v.strip()  # 如果 v 是 int，AttributeError

# ✅ 正确：before 模式下先检查类型
@field_validator('query', mode='before')
@classmethod
def validate_query(cls, v) -> str:
    if isinstance(v, str):
        return v.strip()
    return v  # 非字符串交给 Pydantic 的类型转换处理
```

### 陷阱 4：info.data 中字段不存在

```python
# ❌ 危险：直接访问可能不存在的字段
query = info.data['query']  # 如果 query 验证失败，KeyError

# ✅ 安全：使用 .get() 并提供默认值
query = info.data.get('query', '')
```

### 陷阱 5：装饰器顺序错误

`@field_validator` 必须在 `@classmethod` 外层，顺序反过来会导致错误：

```python
# ❌ @classmethod 在外层
@classmethod
@field_validator('query')
def validate_query(cls, v): ...

# ✅ @field_validator 在外层
@field_validator('query')
@classmethod
def validate_query(cls, v): ...
```

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

## 总结

### 核心要点

1. `field_validator` 是字段级验证装饰器，绑定到特定字段
2. 必须搭配 `@classmethod` 使用，且 `@field_validator` 在外层
3. `mode='before'` 在类型转换前执行，接收原始输入；`mode='after'`（默认）在转换后执行
4. 支持多字段 `@field_validator('a', 'b')` 和全字段 `@field_validator('*')`
5. `ValidationInfo` 提供字段名和已验证字段数据，支持跨字段依赖验证
6. 验证器必须返回值，不返回等于返回 `None`

### 与 model_validator 的分工

| 场景 | 用哪个 |
|------|--------|
| 单字段的格式、范围校验 | `field_validator` |
| 跨字段的业务规则（如 A 和 B 不能同时为空） | `model_validator` |
| 输入预处理（类型兼容） | `field_validator(mode='before')` |
| 全局一致性检查 | `model_validator(mode='after')` |

### 下一步

理解了 `field_validator` 之后，下一个核心概念将讲解 `model_validator` 模型验证器，它用于跨字段的整体验证，比如"如果 `use_rag` 为 True，则 `retrieved_docs` 不能为空"这类业务规则。

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

---

**参考资料**：
- [Pydantic v2 验证机制文档](reference/context7_pydantic_01.md)
- [LangGraph 状态验证官方文档](reference/context7_langgraph_01.md)
- [社区讨论与最佳实践](reference/search_状态验证_01.md)
