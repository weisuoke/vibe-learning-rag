# Context7 查询结果：typing_extensions 文档

> 库：`/python/typing_extensions`

## 查询主题：Annotated、Required、NotRequired

### 1. Annotated 类型

**官方文档说明**：

`Annotated` 允许为类型添加元数据，而不影响类型检查。

```python
from typing import Annotated

# 基础用法
UserId = Annotated[int, "User identifier"]

# 多个元数据
Count = Annotated[int, "Count value", range(0, 100)]

# 函数元数据
def validator(x): return x > 0
PositiveInt = Annotated[int, validator]
```

**PEP 593**：
- Python 3.9+ 内置支持
- Python 3.7-3.8 需要 `typing_extensions`
- 元数据存储在 `__metadata__` 属性中

**访问元数据**：

```python
from typing import get_args, get_origin

type_hint = Annotated[list, operator.add]
origin = get_origin(type_hint)  # list
args = get_args(type_hint)  # (operator.add,)
metadata = type_hint.__metadata__  # (operator.add,)
```

### 2. Required 类型

**官方文档说明**：

`Required` 标记 TypedDict 中的必需字段。

```python
from typing import TypedDict
from typing_extensions import Required

class User(TypedDict, total=False):
    # 默认所有字段可选
    name: str
    age: int

    # 但 id 是必需的
    id: Required[str]
```

**使用场景**：
- `total=False` 时标记必需字段
- 与 `NotRequired` 配合使用
- 提高类型安全性

**类型检查**：

```python
# 正确
user: User = {"id": "123"}

# 错误：缺少必需字段 id
user: User = {"name": "Alice"}  # Type error
```

### 3. NotRequired 类型

**官方文档说明**：

`NotRequired` 标记 TypedDict 中的可选字段。

```python
from typing import TypedDict
from typing_extensions import NotRequired

class User(TypedDict):
    # 默认所有字段必需
    id: str
    name: str

    # 但 age 是可选的
    age: NotRequired[int]
```

**使用场景**：
- `total=True`（默认）时标记可选字段
- 比 `Optional` 语义更清晰
- 推荐用于 TypedDict

**NotRequired vs Optional**：

```python
from typing import Optional

class State1(TypedDict):
    # Optional: 字段必须存在，但值可以是 None
    count: Optional[int]

class State2(TypedDict):
    # NotRequired: 字段可以不存在
    count: NotRequired[int]

# State1 用法
state1: State1 = {"count": None}  # OK
state1: State1 = {}  # Error: 缺少 count

# State2 用法
state2: State2 = {"count": 10}  # OK
state2: State2 = {}  # OK: count 可以不存在
```

### 4. ReadOnly 类型（PEP 705）

**官方文档说明**：

`ReadOnly` 标记 TypedDict 中的只读字段。

```python
from typing import TypedDict
from typing_extensions import ReadOnly

class User(TypedDict):
    id: ReadOnly[str]  # 只读
    name: str  # 可写
```

**使用场景**：
- 防止意外修改
- 文档化不可变字段
- 类型检查器支持

**类型检查**：

```python
user: User = {"id": "123", "name": "Alice"}

user["name"] = "Bob"  # OK
user["id"] = "456"  # Type error: id is read-only
```

### 5. Annotated 与 TypedDict 结合

**官方文档示例**：

```python
from typing import TypedDict, Annotated
from typing_extensions import NotRequired
import operator

class State(TypedDict):
    # 必需字段 + reducer
    messages: Annotated[list, operator.add]

    # 可选字段 + reducer
    metadata: Annotated[NotRequired[dict], operator.or_]

    # 只读字段
    user_id: ReadOnly[str]
```

**最佳实践**：
- 使用 `Annotated` 添加 reducer
- 使用 `NotRequired` 标记可选字段
- 使用 `ReadOnly` 标记不可变字段
- 组合使用提高类型安全性

### 6. get_type_hints() 函数

**官方文档说明**：

`get_type_hints()` 提取类型提示，支持 `Annotated`。

```python
from typing import get_type_hints, Annotated

class State(TypedDict):
    messages: Annotated[list, operator.add]
    count: int

# 默认：去除 Annotated
hints = get_type_hints(State)
# {'messages': list, 'count': int}

# 保留 Annotated
hints = get_type_hints(State, include_extras=True)
# {'messages': Annotated[list, operator.add], 'count': int}
```

**参数**：
- `include_extras=True`：保留 `Annotated` 元数据
- `localns`：本地命名空间
- `globalns`：全局命名空间

### 7. 类型兼容性

**官方文档说明**：

`Annotated` 不影响类型兼容性：

```python
from typing import Annotated

def process(items: list) -> None:
    pass

# Annotated[list, ...] 兼容 list
annotated_list: Annotated[list, operator.add] = []
process(annotated_list)  # OK
```

**子类型关系**：
- `Annotated[T, ...]` 是 `T` 的子类型
- 元数据不影响类型检查
- 运行时行为一致

### 8. 运行时访问元数据

**官方文档示例**：

```python
from typing import Annotated, get_args

def extract_metadata(type_hint):
    """提取 Annotated 元数据"""
    if hasattr(type_hint, '__metadata__'):
        return type_hint.__metadata__
    return ()

# 示例
type_hint = Annotated[list, operator.add, "messages"]
metadata = extract_metadata(type_hint)
# (operator.add, "messages")
```

### 9. 类型检查器支持

**官方文档说明**：

主流类型检查器支持 `Annotated`：
- **mypy**：完全支持
- **pyright**：完全支持
- **pyre**：部分支持

**示例**：

```python
from typing import Annotated

def validate_positive(x: int) -> bool:
    return x > 0

PositiveInt = Annotated[int, validate_positive]

def process(value: PositiveInt) -> None:
    # 类型检查器知道 value 是 int
    print(value + 1)
```

### 10. 最佳实践

**官方推荐**：

1. **使用 Annotated 添加元数据**：而非注释
2. **NotRequired 优于 Optional**：语义更清晰
3. **ReadOnly 标记不可变字段**：提高安全性
4. **include_extras=True**：提取完整类型信息
5. **文档化元数据**：说明元数据含义
6. **避免过度使用**：只在必要时使用
7. **类型检查器兼容**：确保工具支持
8. **运行时验证**：元数据可用于验证

## 参考链接

- PEP 593（Annotated）：https://peps.python.org/pep-0593/
- PEP 655（Required/NotRequired）：https://peps.python.org/pep-0655/
- PEP 705（ReadOnly）：https://peps.python.org/pep-0705/
- typing_extensions 文档：https://typing-extensions.readthedocs.io/
