# Context7 查询结果：Pydantic 文档

> 库：`/pydantic/pydantic`

## 查询主题：BaseModel 与 Field

### 1. BaseModel 基础

**官方文档说明**：

Pydantic 的 `BaseModel` 是创建数据模型的基类。

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    email: str
```

**核心特性**：
- 自动类型验证
- 数据序列化/反序列化
- JSON Schema 生成
- 默认值支持

### 2. Field 函数

**官方文档说明**：

`Field` 用于为模型字段添加额外的元数据和验证规则。

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    id: int = Field(description="User ID")
    name: str = Field(min_length=1, max_length=100)
    age: int = Field(ge=0, le=150)
    email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
```

**Field 参数**：
- `default`: 默认值
- `default_factory`: 默认值工厂函数
- `description`: 字段描述
- `min_length`/`max_length`: 字符串/列表长度限制
- `ge`/`le`/`gt`/`lt`: 数值范围限制
- `pattern`: 正则表达式验证
- `alias`: 字段别名

### 3. 默认值处理

**官方文档示例**：

```python
from pydantic import BaseModel, Field

class State(BaseModel):
    # 简单默认值
    count: int = 0

    # 使用 Field
    messages: list = Field(default_factory=list)

    # 可选字段
    metadata: dict | None = None
```

**默认值规则**：
- 不可变类型（int、str）可以直接赋值
- 可变类型（list、dict）应使用 `default_factory`
- `None` 表示可选字段

### 4. 类型验证

**官方文档说明**：

Pydantic 在实例化时自动验证类型：

```python
from pydantic import BaseModel, ValidationError

class User(BaseModel):
    id: int
    name: str

# 正确
user = User(id=1, name="Alice")

# 错误：类型不匹配
try:
    user = User(id="not an int", name="Alice")
except ValidationError as e:
    print(e)
```

**验证时机**：
- 实例化时
- 字段赋值时（如果启用 `validate_assignment`）
- 调用 `model_validate()` 时

### 5. 序列化与反序列化

**官方文档示例**：

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str

# 反序列化（从 dict）
user = User(**{"id": 1, "name": "Alice"})

# 序列化（到 dict）
user_dict = user.model_dump()

# 序列化（到 JSON）
user_json = user.model_dump_json()
```

**序列化选项**：
- `model_dump()`: 转换为 dict
- `model_dump_json()`: 转换为 JSON 字符串
- `exclude`: 排除字段
- `include`: 包含字段
- `by_alias`: 使用别名

### 6. 与 TypedDict 对比

**官方文档说明**：

| 特性 | TypedDict | Pydantic BaseModel |
|------|-----------|-------------------|
| 类型检查 | 静态（mypy） | 运行时 |
| 验证 | 无 | 自动验证 |
| 默认值 | 不支持 | 支持 |
| 序列化 | 手动 | 自动 |
| 性能 | 更快 | 稍慢 |
| 依赖 | 无 | pydantic |

**选择建议**：
- **TypedDict**：简单状态、性能敏感、无需验证
- **Pydantic**：需要验证、复杂默认值、序列化需求

### 7. 在 LangGraph 中使用

**官方文档示例**：

```python
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from typing import Annotated
import operator

class State(BaseModel):
    messages: Annotated[list, operator.add] = Field(default_factory=list)
    user_id: str = Field(description="User identifier")
    count: int = Field(default=0, ge=0)

graph = StateGraph(State)
```

**注意事项**：
- Pydantic 模型可以直接用作 StateGraph 的 state_schema
- `Annotated` 与 `Field` 可以组合使用
- 验证在状态更新时自动执行

### 8. Config 配置

**官方文档说明**：

使用 `model_config` 配置模型行为：

```python
from pydantic import BaseModel, ConfigDict

class State(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,  # 赋值时验证
        frozen=False,  # 允许修改
        extra='forbid',  # 禁止额外字段
        str_strip_whitespace=True,  # 去除空格
    )

    messages: list
    user_id: str
```

**常用配置**：
- `validate_assignment`: 赋值时验证
- `frozen`: 不可变模型
- `extra`: 额外字段处理（'allow', 'forbid', 'ignore'）
- `str_strip_whitespace`: 去除字符串空格
- `use_enum_values`: 使用枚举值

### 9. 自定义验证器

**官方文档示例**：

```python
from pydantic import BaseModel, field_validator

class State(BaseModel):
    messages: list
    count: int

    @field_validator('count')
    @classmethod
    def validate_count(cls, v):
        if v < 0:
            raise ValueError('count must be non-negative')
        return v

    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        if len(v) > 100:
            raise ValueError('too many messages')
        return v
```

**验证器类型**：
- `@field_validator`: 字段验证器
- `@model_validator`: 模型验证器
- `mode='before'`: 验证前执行
- `mode='after'`: 验证后执行

### 10. 最佳实践

**官方推荐**：

1. **使用 Field 添加元数据**：提高可读性
2. **default_factory 用于可变类型**：避免共享引用
3. **启用 validate_assignment**：确保数据一致性
4. **合理使用验证器**：避免过度验证
5. **文档化字段**：使用 `description` 参数
6. **类型注解完整**：确保类型安全
7. **测试验证逻辑**：确保验证器正确
8. **性能考虑**：验证有开销，权衡使用

## 参考链接

- Pydantic 官方文档：https://docs.pydantic.dev/
- BaseModel 指南：https://docs.pydantic.dev/latest/concepts/models/
- Field 文档：https://docs.pydantic.dev/latest/concepts/fields/
- 验证器文档：https://docs.pydantic.dev/latest/concepts/validators/
