# 核心概念 9：mypy 与 pyright

## 一句话定义

**mypy 和 pyright 是 Python 的静态类型检查器，通过分析代码中的类型注解在编译时发现类型错误，mypy 是官方推荐的类型检查器，pyright 是微软开发的高性能替代方案，两者都是构建类型安全 Python 项目的核心工具。**

---

## 为什么需要类型检查器？

### 问题：类型注解不会自动检查

```python
# 有类型注解，但运行时不检查
def add(a: int, b: int) -> int:
    return a + b

# ❌ 运行时不会报错
result = add("hello", "world")  # 返回 "helloworld"
print(result.upper())  # AttributeError: 'str' object has no attribute 'upper'
```

**类型检查器的解决方案**：

```bash
# 使用 mypy 检查
$ mypy example.py
example.py:5: error: Argument 1 to "add" has incompatible type "str"; expected "int"
example.py:5: error: Argument 2 to "add" has incompatible type "str"; expected "int"

# 使用 pyright 检查
$ pyright example.py
example.py:5:14 - error: Argument of type "Literal['hello']" cannot be assigned to parameter "a" of type "int"
example.py:5:23 - error: Argument of type "Literal['world']" cannot be assigned to parameter "b" of type "int"
```

---

## mypy 基础

### 1. 安装和基本使用

```bash
# 安装 mypy
uv add --dev mypy

# 检查单个文件
mypy example.py

# 检查整个目录
mypy src/

# 检查特定模块
mypy -m mymodule

# 检查包
mypy -p mypackage
```

### 2. 基本配置

```ini
# mypy.ini
[mypy]
python_version = 3.13
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_any_unimported = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True
```

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.13"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

### 3. 严格模式

```ini
# mypy.ini
[mypy]
strict = True  # 启用所有严格检查

# 等价于：
# disallow_any_unimported = True
# disallow_any_expr = True
# disallow_any_decorated = True
# disallow_any_explicit = True
# disallow_any_generics = True
# disallow_subclassing_any = True
# disallow_untyped_calls = True
# disallow_untyped_defs = True
# disallow_incomplete_defs = True
# check_untyped_defs = True
# disallow_untyped_decorators = True
# warn_redundant_casts = True
# warn_unused_ignores = True
# warn_return_any = True
# warn_unreachable = True
# strict_equality = True
# strict_concatenate = True
```

### 4. 忽略错误

```python
# 忽略单行错误
result = some_function()  # type: ignore

# 忽略特定错误类型
result = some_function()  # type: ignore[arg-type]

# 忽略整个文件
# mypy: ignore-errors

# 配置文件中忽略特定模块
# mypy.ini
[mypy-third_party_module.*]
ignore_missing_imports = True
```

---

## pyright 基础

### 1. 安装和基本使用

```bash
# 安装 pyright
uv add --dev pyright

# 检查当前目录
pyright

# 检查特定文件
pyright example.py

# 检查特定目录
pyright src/

# 监视模式（自动重新检查）
pyright --watch
```

### 2. 基本配置

```json
// pyrightconfig.json
{
  "pythonVersion": "3.13",
  "typeCheckingMode": "strict",
  "reportMissingImports": true,
  "reportMissingTypeStubs": false,
  "reportUnusedImport": true,
  "reportUnusedClass": true,
  "reportUnusedFunction": true,
  "reportUnusedVariable": true,
  "reportDuplicateImport": true,
  "reportOptionalSubscript": true,
  "reportOptionalMemberAccess": true,
  "reportOptionalCall": true,
  "reportOptionalIterable": true,
  "reportOptionalContextManager": true,
  "reportOptionalOperand": true,
  "reportUntypedFunctionDecorator": true,
  "reportUntypedClassDecorator": true,
  "reportUntypedBaseClass": true,
  "reportUntypedNamedTuple": true,
  "reportPrivateUsage": true,
  "reportConstantRedefinition": true,
  "reportIncompatibleMethodOverride": true,
  "reportIncompatibleVariableOverride": true,
  "reportInconsistentConstructor": true,
  "reportOverlappingOverload": true,
  "reportMissingSuperCall": false,
  "reportUninitializedInstanceVariable": true,
  "reportInvalidStringEscapeSequence": true,
  "reportUnknownParameterType": true,
  "reportUnknownArgumentType": true,
  "reportUnknownLambdaType": true,
  "reportUnknownVariableType": true,
  "reportUnknownMemberType": true,
  "reportMissingParameterType": true,
  "reportMissingTypeArgument": true,
  "reportInvalidTypeVarUse": true,
  "reportCallInDefaultInitializer": true,
  "reportUnnecessaryIsInstance": true,
  "reportUnnecessaryCast": true,
  "reportUnnecessaryComparison": true,
  "reportAssertAlwaysTrue": true,
  "reportSelfClsParameterName": true,
  "reportImplicitStringConcatenation": false,
  "reportUndefinedVariable": true,
  "reportUnboundVariable": true,
  "reportInvalidStubStatement": true,
  "reportIncompleteStub": true,
  "reportUnsupportedDunderAll": true,
  "reportUnusedCoroutine": true
}
```

```json
// pyproject.toml
[tool.pyright]
pythonVersion = "3.13"
typeCheckingMode = "strict"
```

### 3. 类型检查模式

```json
// pyrightconfig.json
{
  "typeCheckingMode": "off",      // 关闭类型检查
  "typeCheckingMode": "basic",    // 基础检查
  "typeCheckingMode": "standard", // 标准检查（默认）
  "typeCheckingMode": "strict"    // 严格检查
}
```

### 4. 忽略错误

```python
# 忽略单行错误
result = some_function()  # type: ignore

# 忽略特定错误类型
result = some_function()  # pyright: ignore[reportGeneralTypeIssues]

# 忽略整个文件
# pyright: reportGeneralTypeIssues=false

# 配置文件中忽略特定文件
// pyrightconfig.json
{
  "ignore": ["**/node_modules", "**/__pycache__", "build"]
}
```

---

## mypy vs pyright 对比

### 性能对比

| 特性 | mypy | pyright |
|------|------|---------|
| 检查速度 | 中等 | 快（3-5倍） |
| 冷启动 | 慢 | 快 |
| 增量检查 | 支持 | 支持 |
| 内存占用 | 中等 | 低 |
| 并行检查 | 有限 | 优秀 |

**实际测试**（2025 年数据）：

```bash
# 大型项目（10000+ 文件）
$ time mypy src/
real    0m45.123s

$ time pyright src/
real    0m12.456s  # 快 3.6 倍
```

### 类型推断对比

| 特性 | mypy | pyright |
|------|------|---------|
| 基础推断 | ✅ 良好 | ✅ 优秀 |
| 泛型推断 | ✅ 良好 | ✅ 更强 |
| 类型收窄 | ✅ 支持 | ✅ 更准确 |
| 联合类型 | ✅ 支持 | ✅ 更智能 |
| 条件类型 | ✅ 基础 | ✅ 高级 |

**示例**：

```python
from typing import Union

def process(value: Union[int, str, None]) -> str:
    if value is None:
        return "None"
    elif isinstance(value, int):
        return str(value * 2)
    else:
        return value.upper()

# mypy 推断
# - 第一个分支：value 是 None
# - 第二个分支：value 是 int
# - 第三个分支：value 是 str

# pyright 推断（更准确）
# - 第一个分支：value 是 None
# - 第二个分支：value 是 int
# - 第三个分支：value 是 str（排除了 None 和 int）
```

### 错误提示对比

**mypy**：
```
example.py:5: error: Argument 1 to "add" has incompatible type "str"; expected "int"
```

**pyright**：
```
example.py:5:14 - error: Argument of type "Literal['hello']" cannot be assigned to parameter "a" of type "int"
  Type "Literal['hello']" cannot be assigned to type "int"
    "str" is incompatible with "int"
```

pyright 的错误提示通常更详细，包含更多上下文信息。

### 生态系统对比

| 特性 | mypy | pyright |
|------|------|---------|
| 官方支持 | ✅ Python 官方 | ✅ 微软 |
| IDE 集成 | ✅ 广泛 | ✅ VS Code 原生 |
| 插件生态 | ✅ 丰富 | ❌ 较少 |
| 社区规模 | ✅ 大 | ✅ 增长中 |
| 文档质量 | ✅ 优秀 | ✅ 优秀 |

---

## 2025-2026 新特性

### mypy 1.19（2025 年 11 月）

**主要新特性**：

1. **PEP 747 支持（TypeForm）**：
```python
from typing import TypeForm

def create_instance(cls: TypeForm[T]) -> T:
    return cls()
```

2. **改进的泛型推断**：
```python
from typing import TypeVar

T = TypeVar('T')

def identity(x: T) -> T:
    return x

# mypy 1.19 更准确地推断类型
result = identity([1, 2, 3])  # 推断为 list[int]
```

3. **更好的错误消息**：
```python
# 更清晰的错误提示
def add(a: int, b: int) -> int:
    return a + b

add("hello", "world")
# mypy 1.19: error: Argument 1 to "add" has incompatible type "str"; expected "int"
#   Note: "str" is not compatible with "int"
```

### pyright 1.1.400+（2025-2026）

**主要新特性**：

1. **更快的类型检查**：
   - 优化的增量检查
   - 更好的并行处理
   - 减少内存占用

2. **改进的类型推断**：
```python
# pyright 更准确地推断复杂类型
def get_value():
    return {"name": "Alice", "age": 30}

data = get_value()
# pyright 推断为 dict[str, str | int]
# mypy 可能推断为 dict[str, Any]
```

3. **更好的 TypedDict 支持**：
```python
from typing import TypedDict, NotRequired

class User(TypedDict):
    name: str
    age: int
    email: NotRequired[str]

# pyright 更准确地检查 NotRequired 字段
```

### ty：超快类型检查器（Astral, 2025）

**新的竞争者**：

```bash
# 安装 ty
uv add --dev ty

# 使用 ty
ty check src/

# 性能对比（2025 年数据）
# mypy:    45s
# pyright: 12s
# ty:      3s  # 快 15 倍！
```

**特点**：
- ✅ 极快的检查速度（Rust 实现）
- ✅ 兼容 mypy 配置
- ✅ 渐进式采用
- ⚠️ 功能还在完善中

---

## 实战配置

### 1. 项目初始化

```bash
# 创建配置文件
touch mypy.ini
touch pyrightconfig.json

# 或使用 pyproject.toml
touch pyproject.toml
```

### 2. 渐进式类型检查

```ini
# mypy.ini - 渐进式配置
[mypy]
python_version = 3.13

# 第一阶段：基础检查
warn_return_any = True
warn_unused_configs = True

# 第二阶段：禁止未类型化的定义
# disallow_untyped_defs = True

# 第三阶段：严格模式
# strict = True

# 忽略第三方库
[mypy-third_party.*]
ignore_missing_imports = True
```

### 3. CI/CD 集成

```yaml
# .github/workflows/type-check.yml
name: Type Check

on: [push, pull_request]

jobs:
  type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'

      - name: Install dependencies
        run: |
          pip install uv
          uv sync

      - name: Run mypy
        run: uv run mypy src/

      - name: Run pyright
        run: uv run pyright src/
```

### 4. pre-commit 集成

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.19.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]

  - repo: https://github.com/RobertCraigie/pyright-python
    rev: v1.1.400
    hooks:
      - id: pyright
```

---

## 在 LangChain 项目中使用

### 1. 配置文件

```ini
# mypy.ini
[mypy]
python_version = 3.13
strict = True

# LangChain 相关配置
[mypy-langchain.*]
ignore_missing_imports = False

[mypy-langchain_core.*]
ignore_missing_imports = False

[mypy-langchain_openai.*]
ignore_missing_imports = False

# 第三方库
[mypy-chromadb.*]
ignore_missing_imports = True

[mypy-sentence_transformers.*]
ignore_missing_imports = True
```

```json
// pyrightconfig.json
{
  "pythonVersion": "3.13",
  "typeCheckingMode": "strict",
  "include": ["src"],
  "exclude": [
    "**/node_modules",
    "**/__pycache__",
    ".venv"
  ],
  "reportMissingImports": true,
  "reportMissingTypeStubs": false
}
```

### 2. 类型检查 LangChain 代码

```python
"""
类型安全的 LangChain 代码
"""

from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# ===== 1. 定义类型明确的函数 =====

def format_prompt(topic: str) -> dict[str, str]:
    """格式化 prompt"""
    return {"topic": topic}

def extract_text(response: str) -> str:
    """提取文本"""
    return response.strip()

# ===== 2. 构建类型安全的链 =====

# 每个组件都有明确的类型
formatter: Runnable[str, dict[str, str]] = RunnableLambda(format_prompt)
prompt: Runnable[dict[str, str], Any] = ChatPromptTemplate.from_template(
    "Tell me about {topic}"
)
model: Runnable[Any, Any] = ChatOpenAI(model="gpt-4o-mini")
parser: Runnable[Any, str] = StrOutputParser()
extractor: Runnable[str, str] = RunnableLambda(extract_text)

# 组合链
chain: Runnable[str, str] = formatter | prompt | model | parser | extractor

# ===== 3. 使用链 =====

# mypy 和 pyright 都会验证类型
result: str = chain.invoke("Python")
print(result)

# ❌ 类型错误（编译时发现）
# result: int = chain.invoke("Python")
# mypy: error: Incompatible types in assignment
# pyright: error: Type "str" cannot be assigned to type "int"
```

### 3. 运行类型检查

```bash
# 检查整个项目
uv run mypy src/
uv run pyright src/

# 检查特定文件
uv run mypy src/chains/rag_chain.py
uv run pyright src/chains/rag_chain.py

# 生成报告
uv run mypy src/ --html-report mypy-report/
uv run pyright src/ --outputjson > pyright-report.json
```

---

## 常见问题和解决方案

### 1. 第三方库缺少类型注解

**问题**：
```python
import some_library

result = some_library.process(data)
# mypy: error: Skipping analyzing "some_library": module is installed, but missing library stubs or py.typed marker
```

**解决方案**：

```bash
# 方案1：安装 type stubs
uv add --dev types-some-library

# 方案2：忽略该库
# mypy.ini
[mypy-some_library.*]
ignore_missing_imports = True

# 方案3：创建自己的 stub 文件
# stubs/some_library.pyi
def process(data: str) -> str: ...
```

### 2. 动态代码类型检查

**问题**：
```python
# 动态属性访问
def get_attr(obj, name):
    return getattr(obj, name)

# mypy: error: Missing type annotation for function
```

**解决方案**：

```python
from typing import Any

# 方案1：使用 Any
def get_attr(obj: Any, name: str) -> Any:
    return getattr(obj, name)

# 方案2：使用泛型
from typing import TypeVar

T = TypeVar('T')

def get_attr(obj: T, name: str) -> Any:
    return getattr(obj, name)

# 方案3：使用 Protocol
from typing import Protocol

class HasAttr(Protocol):
    def __getattr__(self, name: str) -> Any: ...

def get_attr(obj: HasAttr, name: str) -> Any:
    return getattr(obj, name)
```

### 3. 循环导入

**问题**：
```python
# a.py
from b import B

class A:
    def get_b(self) -> B:
        ...

# b.py
from a import A

class B:
    def get_a(self) -> A:
        ...

# mypy: error: Cannot find implementation or library stub for module named "b"
```

**解决方案**：

```python
# 方案1：使用字符串注解
# a.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from b import B

class A:
    def get_b(self) -> "B":
        ...

# 方案2：使用 from __future__ import annotations
# a.py
from __future__ import annotations

from b import B

class A:
    def get_b(self) -> B:
        ...
```

---

## 2025-2026 最佳实践

### 1. 同时使用 mypy 和 pyright

```bash
# 在 CI 中同时运行两个检查器
uv run mypy src/
uv run pyright src/

# 原因：
# - mypy 是官方标准，社区广泛使用
# - pyright 更快，错误提示更好
# - 两者互补，提高代码质量
```

### 2. 渐进式采用严格模式

```ini
# mypy.ini - 第一阶段
[mypy]
python_version = 3.13
warn_return_any = True

# 第二阶段（几周后）
# disallow_untyped_defs = True

# 第三阶段（几个月后）
# strict = True
```

### 3. 使用 type stubs

```bash
# 安装常用库的 type stubs
uv add --dev types-requests
uv add --dev types-redis
uv add --dev types-pyyaml
```

### 4. 配置 IDE 集成

**VS Code**：
```json
// .vscode/settings.json
{
  "python.linting.mypyEnabled": true,
  "python.linting.pylintEnabled": false,
  "python.analysis.typeCheckingMode": "strict",
  "python.analysis.diagnosticMode": "workspace"
}
```

**PyCharm**：
- Settings → Tools → Python Integrated Tools → Type Checker: mypy

---

## 实战示例：完整的类型检查工作流

```bash
#!/bin/bash
# scripts/type-check.sh

echo "=== 类型检查工作流 ==="

# 1. 运行 mypy
echo "\n1. Running mypy..."
uv run mypy src/ || exit 1

# 2. 运行 pyright
echo "\n2. Running pyright..."
uv run pyright src/ || exit 1

# 3. 生成报告
echo "\n3. Generating reports..."
uv run mypy src/ --html-report mypy-report/
uv run pyright src/ --outputjson > pyright-report.json

# 4. 统计
echo "\n4. Statistics..."
echo "mypy report: mypy-report/index.html"
echo "pyright report: pyright-report.json"

echo "\n✅ Type checking completed successfully!"
```

```bash
# 使用
chmod +x scripts/type-check.sh
./scripts/type-check.sh
```

---

## 学习检查清单

- [ ] 理解类型检查器的作用
- [ ] 掌握 mypy 的安装和配置
- [ ] 掌握 pyright 的安装和配置
- [ ] 了解 mypy vs pyright 的区别
- [ ] 知道如何忽略错误
- [ ] 掌握渐进式类型检查策略
- [ ] 了解 CI/CD 集成
- [ ] 能够解决常见类型检查问题
- [ ] 遵循 2025-2026 最佳实践

---

## 下一步学习

- **核心概念 10**：高级类型技巧 - 学习更多高级特性
- **核心概念 5**：类型推断机制 - 复习类型推断
- **核心概念 8**：类型守卫 - 复习类型收窄

---

## 参考资源

1. [mypy 官方文档](https://mypy.readthedocs.io/) - 完整文档
2. [Mypy 1.19 released](https://mypy-lang.blogspot.com/2025/11/mypy-119-released.html) - 2025 新特性
3. [pyright 官方文档](https://github.com/microsoft/pyright) - GitHub 仓库
4. [Differences Between Pyright and Mypy](https://github.com/microsoft/pyright/blob/main/docs/mypy-comparison.md) - 官方对比
5. [Mypy vs pyright in practice](https://discuss.python.org/t/mypy-vs-pyright-in-practice/75984) - 2025 社区讨论
6. [ty: An extremely fast Python type checker](https://astral.sh/blog/ty) - Astral 新工具
7. [Python Type Checking: mypy vs Pyright Performance Battle](https://medium.com/@ashusk_1790/python-type-checking-mypy-vs-pyright-performance-battle-fce38c8cb874) - 2025 性能对比
8. [How Well Do New Python Type Checkers Conform?](https://sinon.github.io/future-python-type-checkers) - 2025 符合度测试
