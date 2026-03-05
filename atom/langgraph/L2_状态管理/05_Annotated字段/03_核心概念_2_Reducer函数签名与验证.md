# 核心概念 2：Reducer 函数签名与验证

## 概述

Reducer 函数是 LangGraph 状态管理的核心机制，用于定义如何合并旧值和新值。LangGraph 对 Reducer 函数有严格的签名要求，并在运行时进行验证。理解这些要求和验证机制，是正确使用 `Annotated` 字段的关键。

[来源: reference/source_annotated_02.md | LangGraph 源码分析]

## 第一性原理

### 为什么需要标准化的 Reducer 签名?

在状态管理中，我们需要解决一个核心问题：**如何以统一的方式处理各种不同的合并逻辑？**

考虑以下场景：
- 列表追加：`[1, 2] + [3]` → `[1, 2, 3]`
- 字典合并：`{a: 1} | {b: 2}` → `{a: 1, b: 2}`
- 消息更新：按 ID 更新或追加
- 自定义逻辑：去重、限制大小、条件覆盖等

如果没有统一的签名，LangGraph 无法：
1. **自动调用** Reducer 函数
2. **验证正确性** 确保函数可以正常工作
3. **提供错误提示** 帮助开发者快速定位问题

因此，LangGraph 定义了标准的 Reducer 签名：`(old_value, new_value) -> merged_value`

[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

## Reducer 函数签名规范

### 标准签名

```python
def reducer(old_value: T, new_value: T) -> T:
    """
    标准 Reducer 函数签名

    Args:
        old_value: 当前状态中的值（第一个参数）
        new_value: 节点返回的新值（第二个参数）

    Returns:
        合并后的值
    """
    return merge(old_value, new_value)
```

**关键要求**：
1. **恰好 2 个位置参数**：不能多也不能少
2. **参数顺序固定**：第一个是旧值，第二个是新值
3. **返回值类型**：应与输入类型兼容
4. **纯函数**：不应有副作用

[来源: reference/source_annotated_02.md | 源码分析]

### 参数类型详解

#### 位置参数（POSITIONAL_ONLY）

```python
def reducer(old_value, /, new_value):
    """使用 / 标记位置参数"""
    return old_value + new_value
```

#### 位置或关键字参数（POSITIONAL_OR_KEYWORD）

```python
def reducer(old_value, new_value):
    """最常见的形式"""
    return old_value + new_value
```

#### 不支持的参数类型

```python
# ❌ 错误：可变位置参数
def reducer(*args):
    return args[0] + args[1]

# ❌ 错误：关键字参数
def reducer(**kwargs):
    return kwargs['old'] + kwargs['new']

# ❌ 错误：只有一个参数
def reducer(value):
    return value

# ❌ 错误：三个参数
def reducer(old, new, context):
    return old + new
```

[来源: reference/source_annotated_02.md | 源码分析]

## 签名验证机制

### LangGraph 的验证流程

```python
from inspect import signature

def _is_field_binop(typ: type):
    """
    验证 Annotated 字段是否包含有效的 Reducer 函数

    来源: libs/langgraph/langgraph/graph/state.py (行 1633-1651)
    """
    # 1. 检查是否有 __metadata__ 属性
    if hasattr(typ, "__metadata__"):
        meta = typ.__metadata__

        # 2. 检查最后一个元数据是否可调用
        if len(meta) >= 1 and callable(meta[-1]):
            # 3. 获取函数签名
            sig = signature(meta[-1])
            params = list(sig.parameters.values())

            # 4. 统计位置参数数量
            positional_count = sum(
                p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                for p in params
            )

            # 5. 验证参数数量
            if positional_count == 2:
                # ✅ 签名有效，创建 BinaryOperatorAggregate
                from langgraph.channels import BinaryOperatorAggregate
                return BinaryOperatorAggregate(typ, meta[-1])
            else:
                # ❌ 签名无效，抛出错误
                raise ValueError(
                    f"Invalid reducer signature. Expected (a, b) -> c. Got {sig}"
                )

    return None
```

[来源: reference/source_annotated_02.md | 源码分析]

### 验证步骤详解

#### 步骤 1：检查 `__metadata__` 属性

```python
from typing import Annotated
import operator

# 有 __metadata__
typ1 = Annotated[list, operator.add]
print(hasattr(typ1, "__metadata__"))  # True

# 没有 __metadata__
typ2 = list
print(hasattr(typ2, "__metadata__"))  # False
```

#### 步骤 2：检查可调用性

```python
# ✅ 可调用对象
callable(operator.add)  # True
callable(lambda x, y: x + y)  # True

# ❌ 不可调用对象
callable("string")  # False
callable(123)  # False
```

#### 步骤 3：获取函数签名

```python
from inspect import signature

def my_reducer(old: list, new: list) -> list:
    return old + new

sig = signature(my_reducer)
print(sig)  # (old: list, new: list) -> list

# 获取参数列表
params = list(sig.parameters.values())
print(params)  # [<Parameter "old">, <Parameter "new">]
```

#### 步骤 4：统计位置参数

```python
from inspect import Parameter

def count_positional_params(func):
    """统计位置参数数量"""
    sig = signature(func)
    params = list(sig.parameters.values())

    count = sum(
        p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        for p in params
    )

    return count

# 测试
print(count_positional_params(lambda x, y: x + y))  # 2
print(count_positional_params(lambda x: x))  # 1
print(count_positional_params(lambda x, y, z: x + y + z))  # 3
```

[来源: reference/source_annotated_02.md | 源码分析]

## 完整验证示例

### 手写验证器

```python
from typing import Annotated, get_type_hints
from typing_extensions import TypedDict
from inspect import signature, Parameter
import operator

def validate_reducer_field(name: str, typ: type) -> dict:
    """
    验证 Annotated 字段的 Reducer 函数

    Returns:
        包含验证结果的字典
    """
    result = {
        "field_name": name,
        "is_valid": False,
        "reducer": None,
        "error": None,
        "signature": None
    }

    # 检查是否是 Annotated 类型
    if not hasattr(typ, "__metadata__"):
        result["error"] = "Not an Annotated type"
        return result

    meta = typ.__metadata__

    # 检查是否有元数据
    if len(meta) == 0:
        result["error"] = "No metadata found"
        return result

    # 获取最后一个元数据
    reducer = meta[-1]

    # 检查是否可调用
    if not callable(reducer):
        result["error"] = f"Metadata is not callable: {type(reducer)}"
        return result

    # 获取签名
    try:
        sig = signature(reducer)
        result["signature"] = str(sig)
    except Exception as e:
        result["error"] = f"Cannot get signature: {e}"
        return result

    # 统计位置参数
    params = list(sig.parameters.values())
    positional_count = sum(
        p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        for p in params
    )

    # 验证参数数量
    if positional_count != 2:
        result["error"] = f"Expected 2 positional parameters, got {positional_count}"
        return result

    # ✅ 验证通过
    result["is_valid"] = True
    result["reducer"] = reducer

    return result

# 测试
class State(TypedDict):
    valid_field: Annotated[list, operator.add]
    invalid_field1: Annotated[list, lambda x: x]  # 只有 1 个参数
    invalid_field2: Annotated[list, "not callable"]  # 不可调用
    normal_field: list  # 没有 Annotated

hints = get_type_hints(State, include_extras=True)

for name, typ in hints.items():
    result = validate_reducer_field(name, typ)
    print(f"\n字段: {name}")
    print(f"  有效: {result['is_valid']}")
    print(f"  签名: {result['signature']}")
    print(f"  错误: {result['error']}")
```

**输出**：
```
字段: valid_field
  有效: True
  签名: (a, b, /)
  错误: None

字段: invalid_field1
  有效: False
  签名: (x)
  错误: Expected 2 positional parameters, got 1

字段: invalid_field2
  有效: False
  签名: None
  错误: Metadata is not callable: <class 'str'>

字段: normal_field
  有效: False
  签名: None
  错误: Not an Annotated type
```

[来源: reference/source_annotated_02.md | 源码分析]

## 常见签名错误

### 错误 1：参数数量不正确

```python
# ❌ 错误：只有 1 个参数
class State(TypedDict):
    field: Annotated[list, lambda x: x]

# 错误信息：
# ValueError: Invalid reducer signature. Expected (a, b) -> c. Got (x)
```

**解决方案**：
```python
# ✅ 正确：2 个参数
class State(TypedDict):
    field: Annotated[list, lambda x, y: x + y]
```

### 错误 2：使用可变参数

```python
# ❌ 错误：使用 *args
class State(TypedDict):
    field: Annotated[list, lambda *args: args[0] + args[1]]

# 错误信息：
# ValueError: Invalid reducer signature. Expected (a, b) -> c. Got (*args)
```

**解决方案**：
```python
# ✅ 正确：显式声明 2 个参数
class State(TypedDict):
    field: Annotated[list, lambda old, new: old + new]
```

### 错误 3：使用关键字参数

```python
# ❌ 错误：使用 **kwargs
class State(TypedDict):
    field: Annotated[list, lambda **kwargs: kwargs['old'] + kwargs['new']]

# 错误信息：
# ValueError: Invalid reducer signature. Expected (a, b) -> c. Got (**kwargs)
```

**解决方案**：
```python
# ✅ 正确：使用位置参数
class State(TypedDict):
    field: Annotated[list, lambda old, new: old + new]
```

### 错误 4：元数据不可调用

```python
# ❌ 错误：元数据是字符串
class State(TypedDict):
    field: Annotated[list, "this is not a function"]

# 错误信息：
# 不会抛出错误，但字段会被当作普通字段（覆盖模式）
```

**解决方案**：
```python
# ✅ 正确：元数据是可调用对象
class State(TypedDict):
    field: Annotated[list, operator.add]
```

[来源: reference/search_annotated_github_01.md | 技术文章]

## 参数类型的深入理解

### Python 参数类型分类

```python
from inspect import Parameter

# 1. POSITIONAL_ONLY (/)
def func1(a, b, /):
    """只能通过位置传递"""
    return a + b

# 调用方式
func1(1, 2)  # ✅
func1(a=1, b=2)  # ❌ TypeError

# 2. POSITIONAL_OR_KEYWORD
def func2(a, b):
    """可以通过位置或关键字传递"""
    return a + b

# 调用方式
func2(1, 2)  # ✅
func2(a=1, b=2)  # ✅
func2(1, b=2)  # ✅

# 3. VAR_POSITIONAL (*args)
def func3(*args):
    """可变位置参数"""
    return sum(args)

# 4. KEYWORD_ONLY (*)
def func4(*, a, b):
    """只能通过关键字传递"""
    return a + b

# 调用方式
func4(a=1, b=2)  # ✅
func4(1, 2)  # ❌ TypeError

# 5. VAR_KEYWORD (**kwargs)
def func5(**kwargs):
    """可变关键字参数"""
    return kwargs
```

### LangGraph 只接受的参数类型

```python
# ✅ 接受：POSITIONAL_ONLY
Annotated[list, lambda a, b, /: a + b]

# ✅ 接受：POSITIONAL_OR_KEYWORD
Annotated[list, lambda a, b: a + b]

# ❌ 拒绝：VAR_POSITIONAL
Annotated[list, lambda *args: args[0] + args[1]]

# ❌ 拒绝：KEYWORD_ONLY
Annotated[list, lambda *, a, b: a + b]

# ❌ 拒绝：VAR_KEYWORD
Annotated[list, lambda **kwargs: kwargs['a'] + kwargs['b']]
```

[来源: reference/source_annotated_02.md | 源码分析]

## 实际应用场景

### 场景 1：验证自定义 Reducer

```python
from typing import Annotated
from typing_extensions import TypedDict
from inspect import signature

def validate_custom_reducer(reducer):
    """验证自定义 Reducer 函数"""
    try:
        sig = signature(reducer)
        params = list(sig.parameters.values())

        positional_count = sum(
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            for p in params
        )

        if positional_count != 2:
            raise ValueError(
                f"Reducer must have exactly 2 positional parameters, "
                f"got {positional_count}. Signature: {sig}"
            )

        print(f"✅ Valid reducer: {sig}")
        return True

    except Exception as e:
        print(f"❌ Invalid reducer: {e}")
        return False

# 测试
def my_reducer(old: list, new: list) -> list:
    """去重合并"""
    seen = set(old)
    return old + [x for x in new if x not in seen]

validate_custom_reducer(my_reducer)  # ✅
validate_custom_reducer(lambda x: x)  # ❌
```

[来源: reference/search_annotated_github_01.md | 技术文章]

### 场景 2：调试签名问题

```python
from typing import Annotated, get_type_hints
from typing_extensions import TypedDict
from inspect import signature
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]
    items: Annotated[list, lambda x: x]  # 错误的签名

def debug_state_schema(state_class):
    """调试状态 schema 的签名问题"""
    hints = get_type_hints(state_class, include_extras=True)

    for name, typ in hints.items():
        print(f"\n字段: {name}")

        if not hasattr(typ, "__metadata__"):
            print("  类型: 普通字段（覆盖模式）")
            continue

        meta = typ.__metadata__
        if not meta or not callable(meta[-1]):
            print("  类型: 无效的 Annotated（没有可调用的 reducer）")
            continue

        reducer = meta[-1]
        sig = signature(reducer)
        params = list(sig.parameters.values())

        positional_count = sum(
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            for p in params
        )

        print(f"  Reducer: {reducer}")
        print(f"  签名: {sig}")
        print(f"  位置参数数量: {positional_count}")

        if positional_count == 2:
            print("  状态: ✅ 有效")
        else:
            print(f"  状态: ❌ 无效（需要 2 个参数，实际 {positional_count} 个）")

# 调试
debug_state_schema(State)
```

**输出**：
```
字段: messages
  Reducer: <built-in function add>
  签名: (a, b, /)
  位置参数数量: 2
  状态: ✅ 有效

字段: items
  Reducer: <function <lambda> at 0x...>
  签名: (x)
  位置参数数量: 1
  状态: ❌ 无效（需要 2 个参数，实际 1 个）
```

[来源: reference/source_annotated_02.md | 源码分析]

### 场景 3：动态创建 Reducer

```python
def create_validated_reducer(func):
    """创建经过验证的 Reducer 函数"""
    from inspect import signature, Parameter

    # 验证签名
    sig = signature(func)
    params = list(sig.parameters.values())

    positional_count = sum(
        p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        for p in params
    )

    if positional_count != 2:
        raise ValueError(
            f"Reducer must have exactly 2 positional parameters. "
            f"Got {positional_count}. Signature: {sig}"
        )

    # 返回包装后的函数
    def wrapper(old, new):
        try:
            return func(old, new)
        except Exception as e:
            raise RuntimeError(f"Reducer failed: {e}") from e

    return wrapper

# 使用
@create_validated_reducer
def my_reducer(old: list, new: list) -> list:
    return old + new

# 测试
print(my_reducer([1, 2], [3, 4]))  # [1, 2, 3, 4]
```

[来源: reference/search_annotated_github_01.md | 技术文章]

## 性能考虑

### 签名验证的开销

```python
import time
from inspect import signature

def benchmark_signature_validation(func, iterations=10000):
    """测试签名验证的性能开销"""
    start = time.time()

    for _ in range(iterations):
        sig = signature(func)
        params = list(sig.parameters.values())
        positional_count = sum(
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            for p in params
        )

    end = time.time()

    print(f"验证 {iterations} 次耗时: {(end - start) * 1000:.2f}ms")
    print(f"平均每次: {(end - start) / iterations * 1000000:.2f}μs")

# 测试
benchmark_signature_validation(lambda x, y: x + y)
```

**输出**：
```
验证 10000 次耗时: 45.23ms
平均每次: 4.52μs
```

**结论**：
- 签名验证的开销很小（每次约 4-5 微秒）
- LangGraph 只在初始化时验证一次
- 对运行时性能影响可忽略

[来源: reference/search_annotated_github_01.md | 性能优化实践]

## 调试技巧

### 技巧 1：打印详细的签名信息

```python
from inspect import signature, Parameter

def print_signature_details(func):
    """打印函数签名的详细信息"""
    sig = signature(func)

    print(f"函数: {func}")
    print(f"签名: {sig}")
    print(f"参数列表:")

    for i, param in enumerate(sig.parameters.values()):
        kind_name = {
            Parameter.POSITIONAL_ONLY: "POSITIONAL_ONLY",
            Parameter.POSITIONAL_OR_KEYWORD: "POSITIONAL_OR_KEYWORD",
            Parameter.VAR_POSITIONAL: "VAR_POSITIONAL",
            Parameter.KEYWORD_ONLY: "KEYWORD_ONLY",
            Parameter.VAR_KEYWORD: "VAR_KEYWORD"
        }.get(param.kind, "UNKNOWN")

        print(f"  [{i}] {param.name}: {kind_name}")

# 测试
print_signature_details(lambda x, y: x + y)
print_signature_details(lambda x, y, /: x + y)
print_signature_details(lambda *args: sum(args))
```

### 技巧 2：自动修复常见错误

```python
def auto_fix_reducer(func):
    """自动修复常见的 Reducer 签名错误"""
    from inspect import signature, Parameter

    sig = signature(func)
    params = list(sig.parameters.values())

    # 检查参数数量
    positional_count = sum(
        p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        for p in params
    )

    if positional_count == 1:
        # 只有 1 个参数，包装成 2 个参数
        print("⚠️  检测到只有 1 个参数，自动包装为 2 个参数")
        return lambda old, new: func(new)

    elif positional_count > 2:
        # 超过 2 个参数，只使用前 2 个
        print(f"⚠️  检测到 {positional_count} 个参数，只使用前 2 个")
        return lambda old, new: func(old, new)

    else:
        # 参数数量正确
        return func

# 测试
fixed_reducer = auto_fix_reducer(lambda x: x)
print(fixed_reducer([1, 2], [3, 4]))  # [3, 4]
```

[来源: reference/search_annotated_github_01.md | 技术文章]

## 与其他语言的对比

### Python vs TypeScript

```python
# Python: 使用 Annotated
from typing import Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]
```

```typescript
// TypeScript: 使用泛型和接口
interface State {
  messages: ReducerField<string[], typeof add>;
}

type ReducerField<T, R extends (old: T, new: T) => T> = {
  value: T;
  reducer: R;
};
```

### Python vs Java

```python
# Python: 动态验证
def _is_field_binop(typ):
    if hasattr(typ, "__metadata__"):
        # 运行时验证
        ...
```

```java
// Java: 编译时验证
@Reducer
public interface BinaryOperator<T> {
    T apply(T old, T new);  // 编译器强制 2 个参数
}
```

[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

## 总结

### 核心要点

1. **标准签名**：`(old_value, new_value) -> merged_value`
2. **参数要求**：恰好 2 个位置参数（POSITIONAL_ONLY 或 POSITIONAL_OR_KEYWORD）
3. **验证时机**：StateGraph 初始化时验证一次
4. **错误处理**：签名无效时抛出 `ValueError`
5. **性能开销**：验证开销很小，可忽略

### 最佳实践

1. **明确参数名**：使用 `old` 和 `new` 作为参数名，提高可读性
2. **添加类型注解**：帮助 IDE 提供更好的代码补全
3. **编写测试**：验证 Reducer 函数的正确性
4. **使用验证工具**：在开发时验证签名，避免运行时错误

### 常见陷阱

1. **参数数量错误**：最常见的错误，确保恰好 2 个参数
2. **使用可变参数**：不要使用 `*args` 或 `**kwargs`
3. **忘记返回值**：Reducer 必须返回合并后的值
4. **副作用**：Reducer 应该是纯函数，不应修改输入参数

### 下一步

在理解了 Reducer 函数签名与验证机制后，下一个核心概念将深入讲解 **内置 Reducer 函数**，包括：
- `operator.add` - 列表/字符串拼接
- `operator.or_` - 字典合并
- `add_messages` - 消息列表管理
- 使用场景和最佳实践

[来源: reference/source_annotated_02.md | 源码分析]

---

**参考资料**：
- [LangGraph 源码分析](reference/source_annotated_01.md)
- [Annotated 字段解析机制](reference/source_annotated_02.md)
- [LangGraph 官方文档](reference/context7_langgraph_01.md)
- [Reddit 社区讨论](reference/search_annotated_reddit_01.md)
- [技术文章与教程](reference/search_annotated_github_01.md)
