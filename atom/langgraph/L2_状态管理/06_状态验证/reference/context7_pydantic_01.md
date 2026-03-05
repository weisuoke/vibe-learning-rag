---
type: context7_documentation
library: pydantic
version: v2
fetched_at: 2026-02-27
knowledge_point: 06_状态验证
context7_query: BaseModel field_validator model_validator validation error handling
---

# Context7 文档：Pydantic v2 验证机制

## 文档来源
- 库名称：pydantic
- Context7 ID: /llmstxt/pydantic_dev_llms-full_txt
- 官方文档链接：https://docs.pydantic.dev/latest/

## 关键信息提取

### 1. field_validator 装饰器

对特定字段进行自定义验证：

```python
from pydantic import BaseModel, ValidationError, field_validator

class UserModel(BaseModel):
    name: str
    id: int

    @field_validator('name')
    @classmethod
    def name_must_contain_space(cls, v: str) -> str:
        if ' ' not in v:
            raise ValueError('must contain a space')
        return v.title()
```

- 支持选择多个字段：`@field_validator('id', 'name')`
- 支持 `'*'` 选择所有字段
- `mode='before'` 在类型转换前验证
- `mode='after'` 在类型转换后验证（默认）

### 2. model_validator 装饰器

对整个模型进行跨字段验证：

```python
from pydantic import BaseModel, model_validator
from typing_extensions import Self

class Square(BaseModel):
    width: float
    height: float

    @model_validator(mode='after')
    def verify_square(self) -> Self:
        if self.width != self.height:
            raise ValueError('width and height do not match')
        return self
```

- `mode='before'`：接收原始输入数据（dict）
- `mode='after'`：接收已验证的模型实例

### 3. Field 约束

```python
from pydantic import BaseModel, Field

class Model(BaseModel):
    gt_int: int = Field(gt=42)        # 大于 42
    list_of_ints: list[int]           # 列表元素类型检查
    a_float: float                     # 类型强制转换
```

### 4. ValidationError 处理

```python
try:
    Model(**invalid_data)
except ValidationError as e:
    print(e)           # 人类可读格式
    print(e.errors())  # 结构化错误列表
    # 每个错误包含: type, loc, msg, input, url
```

### 5. 嵌套模型验证

Pydantic 自动递归验证嵌套模型：

```python
class Location(BaseModel):
    lat: float
    lng: float

class Model(BaseModel):
    recursive_model: Location
    # Location 的字段也会被验证
```
