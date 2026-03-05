# 源码分析：Field 工具函数

> 文件：`sourcecode/langgraph/libs/langgraph/langgraph/_internal/_fields.py`

## 核心发现

### 1. get_field_default() 函数

获取字段的默认值。

```python
def get_field_default(field_type: Type, field_name: str) -> Any:
    """
    获取字段默认值

    参数：
    - field_type: 字段类型
    - field_name: 字段名称

    返回：
    - 默认值，如果没有则返回 MISSING
    """
```

**支持的默认值来源**：

1. **TypedDict**：不支持默认值（返回 MISSING）
2. **Pydantic BaseModel**：从 Field 定义中获取
3. **Dataclass**：从 field() 定义中获取
4. **普通类**：从类属性中获取

**示例**：

```python
from pydantic import BaseModel, Field
from dataclasses import dataclass, field

# Pydantic
class PydanticState(BaseModel):
    count: int = Field(default=0)
    messages: list = Field(default_factory=list)

# Dataclass
@dataclass
class DataclassState:
    count: int = 0
    messages: list = field(default_factory=list)

# 获取默认值
default_count = get_field_default(PydanticState, 'count')  # 0
default_messages = get_field_default(PydanticState, 'messages')  # []
```

### 2. get_enhanced_type_hints() 函数

增强版的 `typing.get_type_hints()`，支持更多类型。

```python
def get_enhanced_type_hints(obj: Type) -> Dict[str, Type]:
    """
    获取增强的类型提示

    参数：
    - obj: 类或函数对象

    返回：
    - 字段名到类型的映射
    """
```

**增强功能**：

1. **支持 Annotated**：正确提取 Annotated 类型
2. **支持 TypedDict**：处理 Required/NotRequired
3. **支持 Pydantic**：提取 Field 信息
4. **支持 Dataclass**：提取 field 信息
5. **向后兼容**：兼容旧版 Python

**示例**：

```python
from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]
    user_id: str
    count: int

hints = get_enhanced_type_hints(State)
# {
#     'messages': Annotated[list, operator.add],
#     'user_id': str,
#     'count': int
# }
```

### 3. get_cached_annotated_keys() 函数

缓存并返回 Annotated 字段的键。

```python
def get_cached_annotated_keys(schema: Type) -> Set[str]:
    """
    获取缓存的 Annotated 字段键

    参数：
    - schema: 状态 schema 类型

    返回：
    - Annotated 字段名的集合
    """
```

**缓存机制**：

- 使用 `functools.lru_cache` 缓存结果
- 避免重复解析 schema
- 提高性能

**示例**：

```python
from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]
    user_id: str
    count: Annotated[int, lambda x, y: x + y]

annotated_keys = get_cached_annotated_keys(State)
# {'messages', 'count'}
```

### 4. extract_reducer() 函数

从 Annotated 类型中提取 reducer 函数。

```python
def extract_reducer(field_type: Type) -> Optional[Callable]:
    """
    提取 reducer 函数

    参数：
    - field_type: 字段类型（可能是 Annotated）

    返回：
    - reducer 函数，如果没有则返回 None
    """
```

**支持的 reducer 类型**：

1. **内置运算符**：`operator.add`、`operator.or_` 等
2. **自定义函数**：任何接受两个参数的函数
3. **Lambda 函数**：匿名函数

**示例**：

```python
from typing import Annotated
import operator

# 内置运算符
list_type = Annotated[list, operator.add]
reducer1 = extract_reducer(list_type)  # operator.add

# 自定义函数
def merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}

dict_type = Annotated[dict, merge_dicts]
reducer2 = extract_reducer(dict_type)  # merge_dicts

# Lambda 函数
int_type = Annotated[int, lambda x, y: x + y]
reducer3 = extract_reducer(int_type)  # lambda x, y: x + y
```

### 5. is_optional() 函数

检查字段是否为可选（Optional）。

```python
def is_optional(field_type: Type) -> bool:
    """
    检查字段是否为可选

    参数：
    - field_type: 字段类型

    返回：
    - True 如果可选，否则 False
    """
```

**判断逻辑**：

1. **Union[T, None]**：可选
2. **Optional[T]**：可选（等价于 Union[T, None]）
3. **NotRequired[T]**：可选（TypedDict）
4. **其他**：必需

**示例**：

```python
from typing import Optional, Union
from typing_extensions import NotRequired

# 可选类型
is_optional(Optional[str])  # True
is_optional(Union[str, None])  # True
is_optional(NotRequired[str])  # True

# 必需类型
is_optional(str)  # False
is_optional(int)  # False
```

### 6. get_base_type() 函数

获取字段的基础类型（去除 Annotated、Optional 等包装）。

```python
def get_base_type(field_type: Type) -> Type:
    """
    获取基础类型

    参数：
    - field_type: 字段类型

    返回：
    - 基础类型
    """
```

**处理的包装类型**：

1. **Annotated[T, ...]**：返回 T
2. **Optional[T]**：返回 T
3. **Union[T, None]**：返回 T
4. **NotRequired[T]**：返回 T

**示例**：

```python
from typing import Annotated, Optional
import operator

# Annotated
type1 = Annotated[list, operator.add]
base1 = get_base_type(type1)  # list

# Optional
type2 = Optional[str]
base2 = get_base_type(type2)  # str

# 组合
type3 = Annotated[Optional[list], operator.add]
base3 = get_base_type(type3)  # list
```

### 7. validate_field_type() 函数

验证字段类型是否有效。

```python
def validate_field_type(field_type: Type, field_name: str) -> None:
    """
    验证字段类型

    参数：
    - field_type: 字段类型
    - field_name: 字段名称

    抛出：
    - TypeError: 如果类型无效
    """
```

**验证规则**：

1. **不能是 Any**：必须明确类型
2. **不能是泛型**：必须具体化（如 `list[str]` 而非 `list`）
3. **Reducer 类型匹配**：reducer 的参数类型必须与字段类型匹配

**示例**：

```python
from typing import Any, Annotated
import operator

# 无效类型
validate_field_type(Any, 'field1')  # TypeError: field1 cannot be Any

# 有效类型
validate_field_type(str, 'field2')  # OK
validate_field_type(Annotated[list, operator.add], 'field3')  # OK
```

### 8. merge_field_defaults() 函数

合并多个 schema 的字段默认值。

```python
def merge_field_defaults(
    schemas: List[Type],
    field_name: str
) -> Any:
    """
    合并字段默认值

    参数：
    - schemas: schema 列表
    - field_name: 字段名称

    返回：
    - 合并后的默认值
    """
```

**合并策略**：

1. **优先级**：后面的 schema 优先级更高
2. **MISSING 处理**：跳过没有默认值的 schema
3. **类型一致性**：确保默认值类型一致

**示例**：

```python
from pydantic import BaseModel, Field

class Schema1(BaseModel):
    count: int = Field(default=0)

class Schema2(BaseModel):
    count: int = Field(default=10)

# Schema2 优先级更高
default = merge_field_defaults([Schema1, Schema2], 'count')  # 10
```

### 9. 性能优化

Field 工具函数使用多种优化技术：

1. **LRU 缓存**：缓存类型提示和 Annotated 键
2. **懒加载**：延迟解析 schema
3. **类型缓存**：缓存基础类型提取结果

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_enhanced_type_hints(obj: Type) -> Dict[str, Type]:
    # 缓存类型提示
    ...

@lru_cache(maxsize=128)
def get_cached_annotated_keys(schema: Type) -> Set[str]:
    # 缓存 Annotated 键
    ...
```

### 10. 错误处理

Field 工具函数提供详细的错误信息：

```python
try:
    validate_field_type(Any, 'my_field')
except TypeError as e:
    print(e)  # "Field 'my_field' cannot be Any type"

try:
    default = get_field_default(InvalidSchema, 'field')
except AttributeError as e:
    print(e)  # "Schema InvalidSchema does not have field 'field'"
```

## 源码位置

- **Field 工具**：`langgraph/_internal/_fields.py:1-500`
- **类型提取**：`langgraph/_internal/_fields.py:50-150`
- **默认值处理**：`langgraph/_internal/_fields.py:150-250`
- **验证逻辑**：`langgraph/_internal/_fields.py:250-350`

## 关键依赖

- `typing`: 类型注解
- `typing_extensions`: Annotated、NotRequired 等
- `functools`: lru_cache 缓存
- `inspect`: 类型检查

## 最佳实践

1. **使用缓存函数**：避免重复解析 schema
2. **明确类型注解**：避免使用 Any
3. **验证字段类型**：在运行时验证类型
4. **处理 Optional**：正确处理可选字段
5. **测试 reducer**：确保 reducer 类型匹配

## 参考资料

- 源码文件：`sourcecode/langgraph/libs/langgraph/langgraph/_internal/_fields.py`
- Python typing 文档：https://docs.python.org/3/library/typing.html
- PEP 593（Annotated）：https://peps.python.org/pep-0593/
