---
type: context7_documentation
library: pydantic
version: main
fetched_at: 2026-02-26
knowledge_point: 04_状态类型系统
context7_query: BaseModel type validation field types model_fields model_fields_set
---

# Context7 文档：Pydantic

## 文档来源
- 库名称：pydantic
- 版本：main
- 官方文档链接：https://github.com/pydantic/pydantic
- Context7 Library ID：/pydantic/pydantic

## 关键信息提取

### 1. BaseModel 基础用法

**定义**：
- Pydantic 的核心类，用于定义数据模式
- 使用类型注解定义字段
- 自动验证和类型转换

**基础示例**：
```python
from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str = 'John Doe'
    signup_ts: Optional[datetime] = None
    friends: list[int] = []

external_data = {'id': '123', 'signup_ts': '2017-06-01 12:22', 'friends': [1, '2', b'3']}
user = User(**external_data)
print(user)
#> User id=123 name='John Doe' signup_ts=datetime.datetime(2017, 6, 1, 12, 22) friends=[1, 2, 3]
print(user.id)
#> 123
```

**关键特性**：
- 自动类型转换：字符串 '123' 转换为整数 123
- 日期时间解析：字符串 '2017-06-01 12:22' 转换为 datetime 对象
- 列表元素转换：字符串 '2' 和字节 b'3' 转换为整数
- 默认值应用：未提供的字段使用默认值

### 2. 字段访问与类型验证

**字段访问**：
```python
assert user.name == 'Jane Doe'
assert user.id == 123
assert isinstance(user.id, int)
```

**关键特性**：
- 字段作为普通 Python 属性访问
- 类型转换后的值保持正确的类型
- 默认值在未显式设置时应用

### 3. model_fields_set 属性

**定义**：
- 跟踪哪些字段在初始化时被显式提供
- 用于区分显式设置的字段和使用默认值的字段

**使用场景**：
- 部分更新：只更新显式提供的字段
- 状态管理：LangGraph 使用此属性判断哪些字段需要更新
- 序列化控制：决定哪些字段应该包含在输出中

### 4. 高级类型支持

**复杂类型示例**：
```python
from datetime import datetime
from pydantic import BaseModel, PositiveInt

class User(BaseModel):
    id: int  # 必需字段
    name: str = 'John Doe'  # 带默认值的字段
    signup_ts: datetime | None  # 可选字段（Python 3.10+ 语法）
    tastes: dict[str, PositiveInt]  # 字典类型，值必须是正整数

external_data = {
    'id': 123,
    'signup_ts': '2019-06-01 12:22',  # 字符串自动转换为 datetime
    'tastes': {
        'wine': 9,
        b'cheese': 7,  # 字节键自动转换为字符串
        'cabbage': '1',  # 字符串值自动转换为整数
    },
}

user = User(**external_data)
print(user.id)
#> 123
print(user.model_dump())
"""
{
    'id': 123,
    'name': 'John Doe',
    'signup_ts': datetime.datetime(2019, 6, 1, 12, 22),
    'tastes': {'wine': 9, 'cheese': 7, 'cabbage': 1},
}
"""
```

**关键特性**：
- `PositiveInt`：约束类型，确保值为正整数
- `dict[str, PositiveInt]`：泛型字典类型，键为字符串，值为正整数
- `datetime | None`：联合类型，支持 None
- `model_dump()`：将模型转换为字典

### 5. 严格模式（Strict Mode）

**定义**：
- 禁用类型转换，要求值精确匹配声明的类型
- 使用 `Annotated` 和 `Strict()` 或便捷别名（如 `StrictInt`）

**示例**：
```python
from typing import Annotated
from uuid import UUID
from pydantic import BaseModel, Strict, StrictInt

class User(BaseModel):
    id: Annotated[UUID, Strict()]  # 必须是 UUID 对象，不接受字符串
    age: StrictInt  # 必须是整数，不接受字符串 '123'
```

**使用场景**：
- API 输入验证：确保客户端发送正确的类型
- 类型安全：避免意外的类型转换
- 性能优化：跳过类型转换逻辑

### 6. 与 LangGraph 的集成

**LangGraph 中的特殊处理**（来自源码分析）：

```python
# 从 _fields.py 中的 get_update_as_tuples 函数
def get_update_as_tuples(input: Any, keys: Sequence[str]) -> list[tuple[str, Any]]:
    """Get Pydantic state update as a list of (key, value) tuples."""
    if isinstance(input, BaseModel):
        keep = input.model_fields_set  # 跟踪显式设置的字段
        defaults = {k: v.default for k, v in type(input).model_fields.items()}
    else:
        keep = None
        defaults = {}

    # 只更新与默认值不同的字段或在 model_fields_set 中的字段
    return [
        (k, value)
        for k in keys
        if (value := getattr(input, k, MISSING)) is not MISSING
        and (
            value is not None
            or defaults.get(k, MISSING) is not None
            or (keep is not None and k in keep)
        )
    ]
```

**关键特性**：
- 使用 `model_fields_set` 跟踪显式设置的字段
- 只更新与默认值不同的字段
- 特殊处理 `None` 值：如果默认值不是 `None`，则更新
- 向后兼容性考虑

### 7. model_fields 属性

**定义**：
- 包含模型所有字段的元数据
- 每个字段包含类型、默认值、验证器等信息

**使用场景**：
- 动态字段访问
- 字段元数据提取
- 默认值获取

### 8. 类型转换规则

**自动转换示例**：
- 字符串 → 整数：`'123'` → `123`
- 字符串 → 日期时间：`'2019-06-01 12:22'` → `datetime(2019, 6, 1, 12, 22)`
- 字节 → 字符串：`b'cheese'` → `'cheese'`
- 字符串 → 整数（列表元素）：`['1', '2']` → `[1, 2]`

**转换失败**：
- 如果转换失败，Pydantic 会抛出 `ValidationError`
- 错误消息包含详细的字段路径和错误原因

## 实践建议

### 1. 在 LangGraph 中使用 Pydantic

**优势**：
- 自动类型验证
- 丰富的类型支持（PositiveInt、EmailStr、HttpUrl 等）
- 清晰的错误消息
- IDE 支持（类型提示、自动补全）

**注意事项**：
- 只有显式设置的字段会被更新（通过 `model_fields_set`）
- 默认值的处理需要特别注意
- 性能开销：Pydantic 的验证有一定的性能成本

### 2. 类型定义最佳实践

**推荐**：
```python
from pydantic import BaseModel, Field

class State(BaseModel):
    # 使用 Field 添加验证和元数据
    id: int = Field(..., gt=0, description="User ID")
    name: str = Field(default="", min_length=1, max_length=100)
    age: int | None = Field(default=None, ge=0, le=150)
```

**避免**：
```python
# 不推荐：缺少验证和约束
class State(BaseModel):
    id: int
    name: str
    age: int | None = None
```

### 3. 与 TypedDict 的对比

| 特性 | Pydantic BaseModel | TypedDict |
|------|-------------------|-----------|
| 运行时验证 | ✓ | ✗ |
| 类型转换 | ✓ | ✗ |
| 默认值 | ✓ | ✗ |
| 字段约束 | ✓ | ✗ |
| 性能 | 较慢 | 快 |
| IDE 支持 | ✓ | ✓ |
| 序列化 | ✓ (model_dump) | 手动 |

**选择建议**：
- 需要验证和转换：使用 Pydantic
- 追求性能：使用 TypedDict
- 简单状态：使用 TypedDict
- 复杂状态：使用 Pydantic

### 4. 性能优化

**技巧**：
- 使用 `model_construct()` 跳过验证（信任数据源时）
- 使用 `model_validate()` 而非 `__init__`（更快）
- 避免嵌套过深的模型
- 考虑使用 `TypedDict` 替代简单场景

## 版本兼容性

- Pydantic v2：性能大幅提升，API 有变化
- Pydantic v1：广泛使用，但性能较慢
- LangGraph 支持 Pydantic v1 和 v2

## 相关资源

- 官方文档：https://docs.pydantic.dev/
- 迁移指南：https://docs.pydantic.dev/latest/migration/
- 性能对比：https://docs.pydantic.dev/latest/concepts/performance/
