# 核心概念 01 - 单元测试原理与 pytest 基础

## 什么是单元测试？

### 定义

**单元测试（Unit Test）**：测试程序中最小的可测试单元（函数、方法、类）

**核心特征**：
- **独立性**：不依赖外部系统（数据库、API、文件系统）
- **快速性**：毫秒级执行
- **可重复性**：每次运行结果相同
- **单一职责**：只测试一个功能点

### 单元测试的价值

```python
# 没有单元测试
def chunk_text(text: str, chunk_size: int) -> list[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# 问题：
# - 空字符串会怎样？
# - chunk_size 为 0 会怎样？
# - chunk_size 为负数会怎样？
# - 需要手动测试所有情况
```

```python
# 有单元测试
def test_chunk_text_empty():
    assert chunk_text("", 10) == []

def test_chunk_text_zero_size():
    with pytest.raises(ValueError):
        chunk_text("test", 0)

def test_chunk_text_negative_size():
    with pytest.raises(ValueError):
        chunk_text("test", -1)

# 好处：
# - 自动验证所有边界条件
# - 修改代码后立即知道是否破坏功能
# - 文档化预期行为
```

---

## pytest 框架核心概念

### 1. 测试发现机制

**pytest 如何找到测试？**

```
规则1: 文件名以 test_ 开头或 _test.py 结尾
  ✓ test_user.py
  ✓ user_test.py
  ✗ user.py

规则2: 函数名以 test_ 开头
  ✓ def test_create_user():
  ✗ def create_user():

规则3: 类名以 Test 开头（且不能有 __init__）
  ✓ class TestUser:
  ✗ class User:
```

**示例**：

```python
# tests/test_user.py

def test_user_creation():
    """pytest 会自动发现这个测试"""
    user = User(username="alice")
    assert user.username == "alice"

class TestUser:
    """pytest 会自动发现这个测试类"""

    def test_username(self):
        user = User(username="alice")
        assert user.username == "alice"

    def test_email(self):
        user = User(email="alice@example.com")
        assert user.email == "alice@example.com"
```

### 2. 断言（Assertion）

**pytest 的断言增强**

```python
# 普通断言
assert 1 + 1 == 2

# pytest 会显示详细的失败信息
assert add(2, 3) == 6
# AssertionError: assert 5 == 6
#  +  where 5 = add(2, 3)
```

**常用断言模式**：

```python
# 相等性断言
assert result == expected
assert result != unexpected

# 真值断言
assert is_valid
assert not is_invalid

# 包含断言
assert "RAG" in response
assert item in collection

# 类型断言
assert isinstance(result, str)
assert isinstance(user, User)

# 比较断言
assert len(results) > 0
assert score >= 0.8
```

### 3. 异常测试

**测试函数是否抛出预期的异常**

```python
import pytest

def divide(a: int, b: int) -> float:
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b

# 测试异常
def test_divide_by_zero():
    """测试：除以零应该抛出 ValueError"""
    with pytest.raises(ValueError):
        divide(10, 0)

# 测试异常消息
def test_divide_by_zero_message():
    """测试：异常消息正确"""
    with pytest.raises(ValueError, match="除数不能为零"):
        divide(10, 0)

# 捕获异常对象
def test_divide_by_zero_details():
    """测试：检查异常详情"""
    with pytest.raises(ValueError) as exc_info:
        divide(10, 0)

    assert "除数" in str(exc_info.value)
```

### 4. 参数化测试

**使用 @pytest.mark.parametrize 批量测试**

```python
import pytest

@pytest.mark.parametrize("a,b,expected", [
    (2, 3, 5),
    (0, 0, 0),
    (-1, 1, 0),
    (100, 200, 300),
])
def test_add(a, b, expected):
    """测试：加法（参数化）"""
    result = add(a, b)
    assert result == expected
```

**运行结果**：

```
test_add[2-3-5] PASSED
test_add[0-0-0] PASSED
test_add[-1-1-0] PASSED
test_add[100-200-300] PASSED
```

**多参数组合**：

```python
@pytest.mark.parametrize("text", ["Hello", "世界", ""])
@pytest.mark.parametrize("chunk_size", [5, 10, 20])
def test_chunk_text_combinations(text, chunk_size):
    """测试：文本分块（多参数组合）"""
    result = chunk_text(text, chunk_size)
    assert isinstance(result, list)
    assert all(len(chunk) <= chunk_size for chunk in result)

# 会生成 3 × 3 = 9 个测试用例
```

---

## fixture 机制

### 什么是 fixture？

**fixture = 测试的前置准备和后置清理**

```python
# 不使用 fixture（重复代码）
def test_user_creation():
    db = create_database()
    user = User(username="alice")
    db.add(user)
    db.commit()
    db.close()

def test_user_query():
    db = create_database()
    user = User(username="alice")
    db.add(user)
    db.commit()
    # ... 测试逻辑
    db.close()

# 使用 fixture（复用代码）
@pytest.fixture
def db():
    database = create_database()
    yield database
    database.close()

@pytest.fixture
def sample_user():
    return User(username="alice")

def test_user_creation(db, sample_user):
    db.add(sample_user)
    db.commit()

def test_user_query(db, sample_user):
    db.add(sample_user)
    db.commit()
    # ... 测试逻辑
```

### fixture 的作用域

**scope 参数控制 fixture 的生命周期**

```python
# function 作用域（默认）：每个测试函数都创建新的 fixture
@pytest.fixture(scope="function")
def user():
    return User(username="alice")

# class 作用域：每个测试类创建一次
@pytest.fixture(scope="class")
def db_connection():
    conn = create_connection()
    yield conn
    conn.close()

# module 作用域：每个测试模块创建一次
@pytest.fixture(scope="module")
def app():
    app = create_app()
    yield app
    app.shutdown()

# session 作用域：整个测试会话创建一次
@pytest.fixture(scope="session")
def test_database():
    db = create_test_database()
    yield db
    db.drop_all()
```

### fixture 的依赖

**fixture 可以依赖其他 fixture**

```python
@pytest.fixture
def db_engine():
    """数据库引擎"""
    engine = create_engine("sqlite:///:memory:")
    return engine

@pytest.fixture
def db_session(db_engine):
    """数据库会话（依赖 db_engine）"""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()

@pytest.fixture
def sample_user(db_session):
    """测试用户（依赖 db_session）"""
    user = User(username="alice")
    db_session.add(user)
    db_session.commit()
    return user

def test_user_query(sample_user, db_session):
    """测试：查询用户"""
    found = db_session.query(User).filter_by(username="alice").first()
    assert found.id == sample_user.id
```

### conftest.py - 共享 fixture

**conftest.py 中的 fixture 可以被所有测试文件使用**

```python
# tests/conftest.py
import pytest
from app.database import create_engine, Session

@pytest.fixture(scope="session")
def db_engine():
    """全局数据库引擎"""
    engine = create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()

@pytest.fixture
def db_session(db_engine):
    """数据库会话"""
    session = Session(bind=db_engine)
    yield session
    session.rollback()
    session.close()

# tests/test_user.py
def test_create_user(db_session):
    """可以直接使用 conftest.py 中的 fixture"""
    user = User(username="alice")
    db_session.add(user)
    db_session.commit()
    assert user.id is not None
```

---

## 测试覆盖率

### 什么是测试覆盖率？

**覆盖率 = 测试执行的代码行数 / 总代码行数**

```python
# app/math.py
def divide(a: int, b: int) -> float:
    if b == 0:  # 第1行
        raise ValueError("除数不能为零")  # 第2行
    return a / b  # 第3行

# tests/test_math.py
def test_divide_normal():
    assert divide(10, 2) == 5  # 只执行了第3行

# 覆盖率：1/3 = 33%
```

### 使用 pytest-cov 测量覆盖率

**安装**：

```bash
uv add pytest-cov
```

**运行测试并生成覆盖率报告**：

```bash
# 基本用法
pytest --cov=app tests/

# 显示未覆盖的行
pytest --cov=app --cov-report=term-missing tests/

# 生成 HTML 报告
pytest --cov=app --cov-report=html tests/
```

**输出示例**：

```
---------- coverage: platform darwin, python 3.13.1 -----------
Name                Stmts   Miss  Cover   Missing
-------------------------------------------------
app/__init__.py         0      0   100%
app/math.py            10      2    80%   15-16
app/user.py            25      5    80%   30-34
-------------------------------------------------
TOTAL                  35      7    80%
```

### 覆盖率的误区

**高覆盖率 ≠ 好测试**

```python
# 100% 覆盖率但无意义的测试
def test_divide():
    divide(10, 2)  # 没有断言
    divide(10, 0)  # 没有捕获异常

# 80% 覆盖率但有意义的测试
def test_divide_normal():
    assert divide(10, 2) == 5

def test_divide_by_zero():
    with pytest.raises(ValueError):
        divide(10, 0)
```

**合理的覆盖率目标**：
- 核心业务逻辑：90%+
- 工具函数：80%+
- 配置和常量：不需要测试

---

## AI Agent 中的单元测试应用

### 场景1：测试文本分块函数

```python
# app/rag/chunking.py
def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int = 0
) -> list[str]:
    """将文本分块"""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return chunks

# tests/test_chunking.py
import pytest

def test_chunk_text_basic():
    """测试：基本分块"""
    text = "Hello world. This is a test."
    chunks = chunk_text(text, chunk_size=10)

    assert len(chunks) > 0
    assert all(len(chunk) <= 10 for chunk in chunks)

def test_chunk_text_with_overlap():
    """测试：带重叠的分块"""
    text = "ABCDEFGHIJ"
    chunks = chunk_text(text, chunk_size=5, overlap=2)

    assert chunks[0] == "ABCDE"
    assert chunks[1] == "DEFGH"  # 重叠了 DE

def test_chunk_text_empty():
    """测试：空字符串"""
    assert chunk_text("", chunk_size=10) == []

def test_chunk_text_invalid_size():
    """测试：无效的 chunk_size"""
    with pytest.raises(ValueError, match="must be positive"):
        chunk_text("test", chunk_size=0)

def test_chunk_text_invalid_overlap():
    """测试：无效的 overlap"""
    with pytest.raises(ValueError, match="must be in"):
        chunk_text("test", chunk_size=10, overlap=20)

@pytest.mark.parametrize("text,chunk_size,expected_count", [
    ("A" * 100, 10, 10),
    ("A" * 100, 20, 5),
    ("A" * 100, 50, 2),
])
def test_chunk_text_counts(text, chunk_size, expected_count):
    """测试：分块数量（参数化）"""
    chunks = chunk_text(text, chunk_size)
    assert len(chunks) == expected_count
```

### 场景2：测试 Embedding 相似度计算

```python
# app/rag/similarity.py
import numpy as np

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """计算余弦相似度"""
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same length")

    v1 = np.array(vec1)
    v2 = np.array(vec2)

    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))

# tests/test_similarity.py
import pytest

def test_cosine_similarity_identical():
    """测试：相同向量的相似度为1"""
    vec = [1, 2, 3]
    assert cosine_similarity(vec, vec) == pytest.approx(1.0)

def test_cosine_similarity_orthogonal():
    """测试：正交向量的相似度为0"""
    vec1 = [1, 0, 0]
    vec2 = [0, 1, 0]
    assert cosine_similarity(vec1, vec2) == pytest.approx(0.0)

def test_cosine_similarity_opposite():
    """测试：相反向量的相似度为-1"""
    vec1 = [1, 2, 3]
    vec2 = [-1, -2, -3]
    assert cosine_similarity(vec1, vec2) == pytest.approx(-1.0)

def test_cosine_similarity_zero_vector():
    """测试：零向量的相似度为0"""
    vec1 = [0, 0, 0]
    vec2 = [1, 2, 3]
    assert cosine_similarity(vec1, vec2) == 0.0

def test_cosine_similarity_different_length():
    """测试：不同长度的向量应该报错"""
    vec1 = [1, 2, 3]
    vec2 = [1, 2]
    with pytest.raises(ValueError, match="same length"):
        cosine_similarity(vec1, vec2)
```

### 场景3：测试 Prompt 模板渲染

```python
# app/agent/prompt.py
from string import Template

def render_prompt(template: str, **kwargs) -> str:
    """渲染 Prompt 模板"""
    try:
        t = Template(template)
        return t.safe_substitute(**kwargs)
    except Exception as e:
        raise ValueError(f"Failed to render template: {e}")

# tests/test_prompt.py
def test_render_prompt_basic():
    """测试：基本模板渲染"""
    template = "Hello, $name!"
    result = render_prompt(template, name="Alice")
    assert result == "Hello, Alice!"

def test_render_prompt_multiple_vars():
    """测试：多个变量"""
    template = "Context: $context\n\nQuestion: $question"
    result = render_prompt(
        template,
        context="RAG is retrieval augmented generation",
        question="What is RAG?"
    )
    assert "RAG is retrieval" in result
    assert "What is RAG?" in result

def test_render_prompt_missing_var():
    """测试：缺少变量（safe_substitute 不会报错）"""
    template = "Hello, $name! Your age is $age."
    result = render_prompt(template, name="Alice")
    assert "Alice" in result
    assert "$age" in result  # 未提供的变量保持原样

def test_render_prompt_empty_template():
    """测试：空模板"""
    assert render_prompt("") == ""

@pytest.mark.parametrize("template,vars,expected", [
    ("$x + $y", {"x": "2", "y": "3"}, "2 + 3"),
    ("$name is $age years old", {"name": "Bob", "age": "25"}, "Bob is 25 years old"),
])
def test_render_prompt_parametrized(template, vars, expected):
    """测试：参数化模板渲染"""
    result = render_prompt(template, **vars)
    assert result == expected
```

### 场景4：测试数据验证（Pydantic）

```python
# app/models/message.py
from pydantic import BaseModel, Field, validator

class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)

    @validator("content")
    def content_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Content cannot be empty or whitespace")
        return v

# tests/test_message.py
import pytest
from pydantic import ValidationError

def test_message_valid():
    """测试：有效消息"""
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"

def test_message_invalid_role():
    """测试：无效角色"""
    with pytest.raises(ValidationError) as exc_info:
        Message(role="invalid", content="Hello")

    errors = exc_info.value.errors()
    assert any("role" in str(e) for e in errors)

def test_message_empty_content():
    """测试：空内容"""
    with pytest.raises(ValidationError) as exc_info:
        Message(role="user", content="")

    errors = exc_info.value.errors()
    assert any("content" in str(e) for e in errors)

def test_message_whitespace_content():
    """测试：纯空白内容"""
    with pytest.raises(ValidationError, match="cannot be empty"):
        Message(role="user", content="   ")

@pytest.mark.parametrize("role", ["user", "assistant", "system"])
def test_message_all_roles(role):
    """测试：所有有效角色"""
    msg = Message(role=role, content="Test")
    assert msg.role == role
```

---

## pytest 常用命令

```bash
# 运行所有测试
pytest

# 运行指定文件
pytest tests/test_user.py

# 运行指定测试函数
pytest tests/test_user.py::test_create_user

# 运行指定测试类
pytest tests/test_user.py::TestUser

# 详细输出
pytest -v

# 显示 print 输出
pytest -s

# 失败时停止
pytest -x

# 只运行失败的测试
pytest --lf

# 运行最后失败的测试，然后运行其他测试
pytest --ff

# 并行运行（需要 pytest-xdist）
pytest -n auto

# 生成覆盖率报告
pytest --cov=app --cov-report=html

# 只运行标记的测试
pytest -m slow

# 排除标记的测试
pytest -m "not slow"
```

---

## pytest 配置文件

### pytest.ini

```ini
# pytest.ini
[pytest]
# 测试目录
testpaths = tests

# 测试文件模式
python_files = test_*.py *_test.py

# 测试类模式
python_classes = Test*

# 测试函数模式
python_functions = test_*

# 默认选项
addopts =
    -v
    --tb=short
    --strict-markers
    --cov=app
    --cov-report=term-missing

# 异步测试模式
asyncio_mode = auto

# 自定义标记
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

### 使用标记

```python
import pytest

@pytest.mark.unit
def test_add():
    """单元测试"""
    assert add(2, 3) == 5

@pytest.mark.integration
def test_database():
    """集成测试"""
    pass

@pytest.mark.slow
def test_large_dataset():
    """慢速测试"""
    pass

# 运行特定标记的测试
# pytest -m unit
# pytest -m "integration and not slow"
```

---

## 最佳实践

### 1. 测试命名

**好的命名**：

```python
def test_user_creation_with_valid_data_should_succeed():
    """测试：使用有效数据创建用户应该成功"""
    pass

def test_user_creation_with_duplicate_username_should_raise_error():
    """测试：使用重复用户名创建用户应该报错"""
    pass
```

**不好的命名**：

```python
def test1():
    pass

def test_user():
    pass
```

### 2. 测试独立性

**好的做法**：

```python
@pytest.fixture
def clean_db():
    """每个测试都有干净的数据库"""
    db = create_test_db()
    yield db
    db.drop_all()

def test_create_user(clean_db):
    user = User(username="alice")
    clean_db.add(user)
    clean_db.commit()
```

**不好的做法**：

```python
# 依赖全局状态
global_user = None

def test_create_user():
    global global_user
    global_user = User(username="alice")

def test_update_user():
    # 依赖上一个测试
    global_user.email = "new@example.com"
```

### 3. 一个测试一个断言（建议）

**好的做法**：

```python
def test_user_username():
    user = User(username="alice")
    assert user.username == "alice"

def test_user_email():
    user = User(email="alice@example.com")
    assert user.email == "alice@example.com"
```

**可接受的做法**：

```python
def test_user_creation():
    """相关的断言可以放在一起"""
    user = User(username="alice", email="alice@example.com")
    assert user.username == "alice"
    assert user.email == "alice@example.com"
    assert user.id is None  # 未保存到数据库
```

---

## 总结

### 核心要点

1. **单元测试**：测试最小的可测试单元，快速、独立、可重复
2. **pytest**：Python 最流行的测试框架，简单易用
3. **fixture**：准备测试数据，复用代码，自动清理
4. **参数化**：批量测试多个输入组合
5. **覆盖率**：衡量测试完整性，但不是唯一指标

### AI Agent 应用

- 测试文本处理函数（分块、清洗）
- 测试相似度计算
- 测试 Prompt 模板渲染
- 测试数据验证（Pydantic）
- 测试工具函数

### 下一步

掌握了单元测试和 pytest 基础后，继续学习：
- 集成测试与数据库测试
- API 测试与端到端测试
- Mock 技巧与测试隔离
