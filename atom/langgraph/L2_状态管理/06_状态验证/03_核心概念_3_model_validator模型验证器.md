# 核心概念 3：model_validator 模型验证器

## 概述

`model_validator` 是 Pydantic 提供的**模型级别验证装饰器**，用于对整个模型实例进行跨字段验证。与 `field_validator` 只能验证单个字段不同，`model_validator` 可以同时访问多个字段的值，验证它们之间的逻辑关系。

在 LangGraph 状态管理中，这意味着你可以确保状态中多个字段之间始终保持一致性——比如"开始日期不能晚于结束日期"、"高级模式必须提供 API Key"等业务规则。

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

---

## 与 field_validator 的区别

在深入 `model_validator` 之前，先搞清楚它和 `field_validator` 的本质区别：

| 维度 | field_validator | model_validator |
|------|----------------|-----------------|
| 验证范围 | 单个字段 | 整个模型（所有字段） |
| 访问能力 | 只能看到当前字段的值 | 可以看到所有字段的值 |
| 典型用途 | 格式校验、范围检查 | 跨字段逻辑、业务规则 |
| 装饰器写法 | `@field_validator('字段名')` | `@model_validator(mode='after')` |
| 参数类型 | 字段值 `v` | 模型实例 `self` 或原始字典 `data` |

**一句话区分**：`field_validator` 是"检查单个零件"，`model_validator` 是"检查零件之间的配合"。

```python
from pydantic import BaseModel, field_validator, model_validator
from typing_extensions import Self

class ExampleState(BaseModel):
    name: str
    age: int

    # field_validator：只看 name 这一个字段
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('名字不能为空')
        return v.strip()

    # model_validator：可以同时看 name 和 age
    @model_validator(mode='after')
    def validate_name_age_consistency(self) -> Self:
        if self.age < 0:
            raise ValueError('年龄不能为负数')
        if self.age > 150 and self.name != 'test':
            raise ValueError('非测试数据的年龄不能超过 150')
        return self
```

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

---

## after 模式（推荐）

`mode='after'` 是最常用的模式。它在所有字段验证完成之后执行，此时你拿到的是一个**已经通过字段级验证的模型实例**，可以安全地访问所有字段。

### 基本语法

```python
from pydantic import BaseModel, model_validator
from typing_extensions import Self

class SearchState(BaseModel):
    query: str
    start_date: str = ""
    end_date: str = ""

    @model_validator(mode='after')
    def validate_date_range(self) -> Self:
        if self.start_date and self.end_date:
            if self.start_date > self.end_date:
                raise ValueError('开始日期不能晚于结束日期')
        return self
```

### 关键点

- **参数是 `self`**：因为此时模型已经实例化，你拿到的就是模型实例本身
- **返回类型是 `Self`**：必须返回 `self`（或修改后的实例）
- **不需要 `@classmethod`**：这是一个实例方法
- **字段值已验证**：所有 `field_validator` 已经执行完毕，字段类型也已转换

### 使用场景

```python
# 场景：RAG 检索参数验证
class RetrievalState(BaseModel):
    query: str
    top_k: int = 5
    threshold: float = 0.7
    max_tokens: int = 4096

    @model_validator(mode='after')
    def validate_retrieval_params(self) -> Self:
        # 跨字段验证：top_k 和 threshold 的组合逻辑
        if self.top_k > 20 and self.threshold < 0.5:
            raise ValueError(
                f'top_k={self.top_k} 较大时，threshold 不能低于 0.5，'
                f'当前 threshold={self.threshold}，否则检索结果质量差'
            )
        return self
```

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

---

## before 模式

`mode='before'` 在所有字段验证**之前**执行。此时你拿到的是**原始输入数据**（通常是一个字典），字段还没有经过类型转换和验证。

### 基本语法

```python
from typing import Any
from pydantic import BaseModel, model_validator

class AgentConfig(BaseModel):
    mode: str
    api_key: str = ""

    @model_validator(mode='before')
    @classmethod
    def check_api_key_for_advanced_mode(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get('mode') == 'advanced' and not data.get('api_key'):
                raise ValueError('高级模式需要提供 api_key')
        return data
```

### 关键点

- **参数是 `data`**：原始输入数据，通常是 `dict`
- **需要 `@classmethod`**：因为此时模型还没有实例化
- **返回类型是 `Any`**：返回处理后的原始数据
- **需要类型检查**：`data` 可能不是字典（比如传入另一个模型实例），所以要先 `isinstance(data, dict)`

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

---

## before vs after 模式对比

| 维度 | before 模式 | after 模式 |
|------|------------|------------|
| 执行时机 | 字段验证之前 | 字段验证之后 |
| 接收参数 | 原始字典 `data: Any` | 模型实例 `self` |
| 是否需要 `@classmethod` | 是 | 否 |
| 返回类型 | `Any`（原始数据） | `Self`（模型实例） |
| 字段类型是否已转换 | 否 | 是 |
| 能否修改输入数据 | 能（修改字典） | 能（修改实例属性） |
| 典型用途 | 预处理、自动填充、格式转换 | 跨字段逻辑验证 |
| 安全性 | 需要手动类型检查 | 字段值已经过验证 |

### 选择建议

- **优先用 after**：大多数跨字段验证场景都应该用 `after`，因为字段值已经过验证，更安全
- **用 before 的场景**：需要在验证前预处理数据、自动填充字段、或者根据某些字段动态调整其他字段的默认值

---

## 执行顺序

理解验证器的执行顺序非常重要，尤其是同时使用 `field_validator` 和 `model_validator` 时：

```
model_validator(mode='before')     ← 最先执行，拿到原始数据
        ↓
field_validator(mode='before')     ← 对每个字段的原始值验证
        ↓
    类型转换（Pydantic 自动）        ← 把 "5" 转成 5 等
        ↓
field_validator(mode='after')      ← 对每个字段的转换后值验证
        ↓
model_validator(mode='after')      ← 最后执行，拿到完整模型实例
```

**记忆口诀**：`model_before → field_before → 类型转换 → field_after → model_after`

```python
from pydantic import BaseModel, field_validator, model_validator
from typing import Any
from typing_extensions import Self

class DebugState(BaseModel):
    value: int

    @model_validator(mode='before')
    @classmethod
    def step1_model_before(cls, data: Any) -> Any:
        print(f"1. model_validator(before): data={data}")
        return data

    @field_validator('value', mode='before')
    @classmethod
    def step2_field_before(cls, v):
        print(f"2. field_validator(before): v={v}, type={type(v)}")
        return v

    @field_validator('value', mode='after')
    @classmethod
    def step3_field_after(cls, v: int) -> int:
        print(f"3. field_validator(after): v={v}, type={type(v)}")
        return v

    @model_validator(mode='after')
    def step4_model_after(self) -> Self:
        print(f"4. model_validator(after): self.value={self.value}")
        return self

# 测试
state = DebugState(value="42")
# 输出：
# 1. model_validator(before): data={'value': '42'}
# 2. field_validator(before): v=42, type=<class 'str'>
# 3. field_validator(after): v=42, type=<class 'int'>
# 4. model_validator(after): self.value=42
```

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

---

## 组合使用 field_validator 和 model_validator

实际项目中，最常见的模式是**两者配合使用**：`field_validator` 负责单字段的格式和范围校验，`model_validator` 负责跨字段的业务逻辑。

```python
from pydantic import BaseModel, field_validator, model_validator
from typing_extensions import Self

class RAGState(BaseModel):
    query: str
    top_k: int = 5
    threshold: float = 0.7
    results: list = []

    # 第一层：单字段验证
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('查询不能为空')
        return v.strip()

    @field_validator('top_k')
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError('top_k 必须在 1-100 之间')
        return v

    @field_validator('threshold')
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError('threshold 必须在 0.0-1.0 之间')
        return v

    # 第二层：跨字段验证
    @model_validator(mode='after')
    def validate_search_params(self) -> Self:
        if self.top_k > 20 and self.threshold < 0.5:
            raise ValueError(
                'top_k 较大时，threshold 不能太低，否则结果质量差'
            )
        return self
```

**验证流程**：
1. 先验证 `query` 不为空
2. 再验证 `top_k` 在合法范围
3. 再验证 `threshold` 在合法范围
4. 最后验证 `top_k` 和 `threshold` 的组合是否合理

如果第 1-3 步任何一个失败，第 4 步不会执行（因为 `model_validator(after)` 依赖字段验证通过）。

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

---

## 在 LangGraph 中的实际应用

### 场景 1：RAG 系统 — 验证检索参数一致性

```python
from pydantic import BaseModel, field_validator, model_validator
from typing_extensions import Self

class RAGRetrievalState(BaseModel):
    """RAG 检索阶段的状态"""
    query: str
    search_type: str = "semantic"       # semantic | keyword | hybrid
    top_k: int = 5
    keyword_weight: float = 0.0         # 仅 hybrid 模式使用
    semantic_weight: float = 1.0        # 仅 hybrid 模式使用

    @model_validator(mode='after')
    def validate_search_config(self) -> Self:
        # 混合检索模式下，权重之和必须为 1
        if self.search_type == 'hybrid':
            total = self.keyword_weight + self.semantic_weight
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f'hybrid 模式下权重之和必须为 1.0，'
                    f'当前: keyword={self.keyword_weight} + '
                    f'semantic={self.semantic_weight} = {total}'
                )
        # 纯语义模式下不应设置 keyword_weight
        if self.search_type == 'semantic' and self.keyword_weight > 0:
            raise ValueError(
                'semantic 模式下不应设置 keyword_weight'
            )
        return self

# 正常使用
state = RAGRetrievalState(
    query="什么是 RAG？",
    search_type="hybrid",
    keyword_weight=0.3,
    semantic_weight=0.7
)

# 触发验证错误
try:
    bad_state = RAGRetrievalState(
        query="什么是 RAG？",
        search_type="hybrid",
        keyword_weight=0.5,
        semantic_weight=0.8  # 0.5 + 0.8 = 1.3，不等于 1
    )
except Exception as e:
    print(f"验证失败: {e}")
```

### 场景 2：Agent 系统 — 验证工具选择与参数匹配

```python
class AgentActionState(BaseModel):
    """Agent 执行动作的状态"""
    tool_name: str
    tool_args: dict = {}
    requires_confirmation: bool = False

    @model_validator(mode='after')
    def validate_tool_args(self) -> Self:
        # 不同工具需要不同的必填参数
        required_args = {
            'web_search': ['query'],
            'calculator': ['expression'],
            'file_read': ['file_path'],
            'database_query': ['sql', 'database'],
        }

        if self.tool_name in required_args:
            missing = [
                arg for arg in required_args[self.tool_name]
                if arg not in self.tool_args
            ]
            if missing:
                raise ValueError(
                    f'工具 {self.tool_name} 缺少必填参数: {missing}'
                )

        # 危险操作需要确认
        dangerous_tools = ['database_query', 'file_write', 'shell_exec']
        if self.tool_name in dangerous_tools and not self.requires_confirmation:
            raise ValueError(
                f'工具 {self.tool_name} 是危险操作，'
                f'requires_confirmation 必须为 True'
            )

        return self
```

### 场景 3：多步推理 — 验证步骤之间的逻辑关系

```python
class ReasoningState(BaseModel):
    """多步推理的状态"""
    current_step: int = 0
    max_steps: int = 10
    steps_completed: list[str] = []
    final_answer: str = ""
    is_finished: bool = False

    @model_validator(mode='after')
    def validate_reasoning_logic(self) -> Self:
        # 已完成的步骤数不能超过当前步骤
        if len(self.steps_completed) > self.current_step:
            raise ValueError(
                f'已完成步骤数({len(self.steps_completed)}) '
                f'不能超过当前步骤({self.current_step})'
            )

        # 标记完成时必须有最终答案
        if self.is_finished and not self.final_answer:
            raise ValueError('标记为已完成时必须提供 final_answer')

        # 不能超过最大步骤数
        if self.current_step > self.max_steps:
            raise ValueError(
                f'当前步骤({self.current_step}) '
                f'超过最大步骤数({self.max_steps})'
            )

        return self
```

---

## 常见误区

### 误区 1：after 模式忘记返回 self

```python
# ❌ 错误：没有返回 self，会导致模型变成 None
@model_validator(mode='after')
def validate(self) -> Self:
    if self.a > self.b:
        raise ValueError('a 不能大于 b')
    # 忘记 return self

# ✅ 正确：必须返回 self
@model_validator(mode='after')
def validate(self) -> Self:
    if self.a > self.b:
        raise ValueError('a 不能大于 b')
    return self
```

### 误区 2：before 模式忘记 @classmethod

```python
# ❌ 错误：before 模式必须是类方法
@model_validator(mode='before')
def validate(cls, data):
    return data

# ✅ 正确：加上 @classmethod
@model_validator(mode='before')
@classmethod
def validate(cls, data: Any) -> Any:
    return data
```

### 误区 3：before 模式不检查 data 类型

```python
# ❌ 危险：data 不一定是 dict
@model_validator(mode='before')
@classmethod
def validate(cls, data: Any) -> Any:
    if data['mode'] == 'advanced':  # 如果 data 不是 dict 会报错
        pass
    return data

# ✅ 安全：先检查类型
@model_validator(mode='before')
@classmethod
def validate(cls, data: Any) -> Any:
    if isinstance(data, dict):
        if data.get('mode') == 'advanced':
            pass
    return data
```

---

## 总结

### 核心要点

1. `model_validator` 用于**跨字段验证**，能同时访问模型的所有字段
2. `after` 模式最常用，接收已验证的模型实例，安全可靠
3. `before` 模式用于预处理原始数据，需要手动做类型检查
4. 执行顺序：`model_before → field_before → 类型转换 → field_after → model_after`
5. 在 LangGraph 中，`model_validator` 确保状态在每次更新时都满足业务规则

### 最佳实践

1. **优先使用 after 模式**：更安全，字段值已经过验证
2. **field_validator + model_validator 配合**：单字段校验用前者，跨字段逻辑用后者
3. **after 模式必须 return self**：否则模型实例会变成 None
4. **before 模式必须检查 data 类型**：用 `isinstance(data, dict)` 保护
5. **错误信息要清晰**：在 `ValueError` 中说明哪些字段冲突、期望值是什么

### 下一步

理解了 `model_validator` 之后，下一个核心概念将讲解**运行时类型强制转换**——Pydantic 如何在 LangGraph 状态更新时自动将输入数据转换为正确的类型。

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]