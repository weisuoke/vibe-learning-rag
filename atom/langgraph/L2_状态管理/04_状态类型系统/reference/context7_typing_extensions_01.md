---
type: context7_documentation
library: typing_extensions
version: main
fetched_at: 2026-02-26
knowledge_point: 04_状态类型系统
context7_query: TypedDict Annotated Required NotRequired ReadOnly type annotations
---

# Context7 文档：typing_extensions

## 文档来源
- 库名称：typing_extensions
- 版本：main
- 官方文档链接：https://github.com/python/typing_extensions
- Context7 Library ID：/python/typing_extensions

## 关键信息提取

### 1. NotRequired

**定义**：
- 表示 TypedDict 中的键不是必需的
- 等同于 `typing.NotRequired`
- 关联 PEP 655
- 在 Python 3.11+ 的 `typing` 模块中可用

**API 文档**：
```python
data NotRequired
  See :py:data:`typing.NotRequired` and :pep:`655`. In ``typing`` since 3.11.

  .. versionadded:: 4.0.0
```

**使用场景**：
- 在 TypedDict 中标记可选字段
- 覆盖 `total=True` 的默认行为
- 与 `Required` 配合使用实现精细的字段控制

### 2. Required

**定义**：
- 表示 TypedDict 中的键是必需的
- 等同于 `typing.Required`
- 关联 PEP 655
- 在 Python 3.11+ 的 `typing` 模块中可用

**API 文档**：
```python
data Required
  See :py:data:`typing.Required` and :pep:`655`. In ``typing`` since 3.11.

  .. versionadded:: 4.0.0
```

**使用场景**：
- 在 TypedDict 中显式标记必需字段
- 覆盖 `total=False` 的默认行为
- 提高代码可读性和类型安全性

### 3. ReadOnly

**定义**：
- 表示 TypedDict 中的项是只读的，初始化后不能修改
- 关联 PEP 705
- 等同于 `typing.ReadOnly`
- 在 Python 3.13+ 的 `typing` 模块中可用

**API 文档**：
```python
data ReadOnly
  See :py:data:`typing.ReadOnly` and :pep:`705`. In ``typing`` since 3.13.

  Indicates that a :class:`TypedDict` item may not be modified.

  .. versionadded:: 4.9.0
```

**TypedDict 与 ReadOnly 限定符**：
```python
from typing_extensions import TypedDict, ReadOnly

class MyTypedDict(TypedDict):
    __readonly_keys__ = frozenset[str]
    __mutable_keys__ = frozenset[str]

class Example(MyTypedDict):
    name: ReadOnly[str]
    age: int

# __readonly_keys__ 将包含 'name'
# __mutable_keys__ 将包含 'age'
```

**关键特性**：
- 引入 `__readonly_keys__` 和 `__mutable_keys__` 属性
- `__readonly_keys__` 包含所有只读字段
- `__mutable_keys__` 包含所有可变字段
- 实验性功能（PEP 705）

### 4. Annotated

**定义**：
- 表示带有元数据的注解类型
- 在类型提示旁边提供元数据
- 在 Python 3.9+ 的 `typing` 模块中可用

**API 文档**：
```python
Annotated
  See :py:data:`typing.Annotated` and :pep:`593`. In ``typing`` since 3.9.

  .. versionchanged:: 4.1.0

     ``Annotated`` can now wrap :data:`ClassVar` and :data:`Final`.
```

**关键特性**：
- 支持包装 `ClassVar` 和 `Final`（自 4.1.0 版本）
- 元数据存储在 `__metadata__` 属性中
- 不影响类型检查，仅提供额外信息

**使用场景**：
- 绑定 Reducer 函数到状态字段
- 添加验证规则
- 提供字段描述和约束
- 框架特定的元数据（如 Pydantic、FastAPI）

### 5. TypedDict（隐含）

虽然文档中没有直接提到 TypedDict，但从 Required、NotRequired、ReadOnly 的描述可以推断：

**TypedDict 特性**：
- 支持 `total=True/False` 参数控制默认字段必需性
- 支持 `__required_keys__` 和 `__optional_keys__` 属性
- 支持 `__readonly_keys__` 和 `__mutable_keys__` 属性（PEP 705）
- 可以与 Required、NotRequired、ReadOnly 组合使用

## 版本兼容性

| 特性 | typing_extensions 版本 | typing 模块版本 | Python 版本 |
|------|------------------------|-----------------|-------------|
| Required | 4.0.0+ | 3.11+ | 3.11+ |
| NotRequired | 4.0.0+ | 3.11+ | 3.11+ |
| ReadOnly | 4.9.0+ | 3.13+ | 3.13+ |
| Annotated | - | 3.9+ | 3.9+ |
| Annotated 包装 ClassVar/Final | 4.1.0+ | - | - |

## 相关 PEP

- **PEP 593**：Flexible function and variable annotations（Annotated）
- **PEP 655**：Marking individual TypedDict items as required or not-required
- **PEP 705**：TypedDict: Read-only items

## 实践建议

1. **使用 Required/NotRequired 而非 total 参数**：
   - 更精细的字段控制
   - 更好的可读性
   - 避免 total 参数的全局影响

2. **ReadOnly 的使用场景**：
   - 配置对象（初始化后不应修改）
   - 不可变状态字段
   - 防止意外修改的关键字段

3. **Annotated 的最佳实践**：
   - 用于框架特定的元数据
   - 不要过度使用，保持类型提示简洁
   - 确保元数据有明确的用途

4. **向后兼容性**：
   - 使用 `typing_extensions` 支持旧版本 Python
   - 在 Python 3.11+ 可以直接使用 `typing` 模块
   - 注意 ReadOnly 需要 Python 3.13+ 或 typing_extensions 4.9.0+
