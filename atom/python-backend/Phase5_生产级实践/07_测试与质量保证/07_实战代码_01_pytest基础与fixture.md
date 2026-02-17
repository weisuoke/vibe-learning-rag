# 实战代码 01 - pytest 基础与 fixture

## 项目结构

```
project/
├── app/
│   ├── __init__.py
│   ├── math_utils.py
│   └── user.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_math_utils.py
│   └── test_user.py
└── pytest.ini
```

---

## 示例1：基础测试

### 被测试代码

```python
# app/math_utils.py
"""数学工具函数"""

def add(a: int, b: int) -> int:
    """加法"""
    return a + b

def subtract(a: int, b: int) -> int:
    """减法"""
    return a - b

def multiply(a: int, b: int) -> int:
    """乘法"""
    return a * b

def divide(a: float, b: float) -> float:
    """除法"""
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b

def is_even(n: int) -> bool:
    """判断是否为偶数"""
    return n % 2 == 0

def factorial(n: int) -> int:
    """计算阶乘"""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)
```

### 测试代码

```python
# tests/test_math_utils.py
"""数学工具函数测试"""
import pytest
from app.math_utils import add, subtract, multiply, divide, is_even, factorial


class TestBasicOperations:
    """基本运算测试"""

    def test_add_positive_numbers(self):
        """测试：两个正数相加"""
        result = add(2, 3)
        assert result == 5

    def test_add_negative_numbers(self):
        """测试：两个负数相加"""
        result = add(-2, -3)
        assert result == -5

    def test_add_mixed_numbers(self):
        """测试：正负数相加"""
        result = add(-2, 3)
        assert result == 1

    def test_add_zero(self):
        """测试：加零"""
        result = add(5, 0)
        assert result == 5

    def test_subtract(self):
        """测试：减法"""
        assert subtract(5, 3) == 2
        assert subtract(3, 5) == -2
        assert subtract(0, 0) == 0

    def test_multiply(self):
        """测试：乘法"""
        assert multiply(2, 3) == 6
        assert multiply(-2, 3) == -6
        assert multiply(0, 100) == 0


class TestDivision:
    """除法测试"""

    def test_divide_normal(self):
        """测试：正常除法"""
        result = divide(10, 2)
        assert result == 5.0

    def test_divide_float(self):
        """测试：浮点数除法"""
        result = divide(7, 2)
        assert result == 3.5

    def test_divide_by_zero(self):
        """测试：除以零"""
        with pytest.raises(ValueError, match="除数不能为零"):
            divide(10, 0)

    def test_divide_negative(self):
        """测试：负数除法"""
        result = divide(-10, 2)
        assert result == -5.0


class TestIsEven:
    """偶数判断测试"""

    @pytest.mark.parametrize("n,expected", [
        (0, True),
        (2, True),
        (4, True),
        (100, True),
        (1, False),
        (3, False),
        (99, False),
        (-2, True),
        (-3, False),
    ])
    def test_is_even(self, n, expected):
        """测试：偶数判断（参数化）"""
        result = is_even(n)
        assert result == expected


class TestFactorial:
    """阶乘测试"""

    def test_factorial_zero(self):
        """测试：0的阶乘"""
        assert factorial(0) == 1

    def test_factorial_one(self):
        """测试：1的阶乘"""
        assert factorial(1) == 1

    def test_factorial_positive(self):
        """测试：正数阶乘"""
        assert factorial(5) == 120
        assert factorial(3) == 6

    def test_factorial_negative(self):
        """测试：负数阶乘"""
        with pytest.raises(ValueError, match="non-negative"):
            factorial(-1)

    @pytest.mark.parametrize("n,expected", [
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 6),
        (4, 24),
        (5, 120),
    ])
    def test_factorial_parametrized(self, n, expected):
        """测试：阶乘（参数化）"""
        assert factorial(n) == expected
```

**运行测试**：

```bash
pytest tests/test_math_utils.py -v
```

**输出**：

```
tests/test_math_utils.py::TestBasicOperations::test_add_positive_numbers PASSED
tests/test_math_utils.py::TestBasicOperations::test_add_negative_numbers PASSED
tests/test_math_utils.py::TestBasicOperations::test_add_mixed_numbers PASSED
tests/test_math_utils.py::TestBasicOperations::test_add_zero PASSED
tests/test_math_utils.py::TestBasicOperations::test_subtract PASSED
tests/test_math_utils.py::TestBasicOperations::test_multiply PASSED
tests/test_math_utils.py::TestDivision::test_divide_normal PASSED
tests/test_math_utils.py::TestDivision::test_divide_float PASSED
tests/test_math_utils.py::TestDivision::test_divide_by_zero PASSED
tests/test_math_utils.py::TestDivision::test_divide_negative PASSED
tests/test_math_utils.py::TestIsEven::test_is_even[0-True] PASSED
tests/test_math_utils.py::TestIsEven::test_is_even[2-True] PASSED
tests/test_math_utils.py::TestIsEven::test_is_even[4-True] PASSED
tests/test_math_utils.py::TestIsEven::test_is_even[100-True] PASSED
tests/test_math_utils.py::TestIsEven::test_is_even[1-False] PASSED
tests/test_math_utils.py::TestIsEven::test_is_even[3-False] PASSED
tests/test_math_utils.py::TestIsEven::test_is_even[99-False] PASSED
tests/test_math_utils.py::TestIsEven::test_is_even[-2-True] PASSED
tests/test_math_utils.py::TestIsEven::test_is_even[-3-False] PASSED
tests/test_math_utils.py::TestFactorial::test_factorial_zero PASSED
tests/test_math_utils.py::TestFactorial::test_factorial_one PASSED
tests/test_math_utils.py::TestFactorial::test_factorial_positive PASSED
tests/test_math_utils.py::TestFactorial::test_factorial_negative PASSED
tests/test_math_utils.py::TestFactorial::test_factorial_parametrized[0-1] PASSED
tests/test_math_utils.py::TestFactorial::test_factorial_parametrized[1-1] PASSED
tests/test_math_utils.py::TestFactorial::test_factorial_parametrized[2-2] PASSED
tests/test_math_utils.py::TestFactorial::test_factorial_parametrized[3-6] PASSED
tests/test_math_utils.py::TestFactorial::test_factorial_parametrized[4-24] PASSED
tests/test_math_utils.py::TestFactorial::test_factorial_parametrized[5-120] PASSED

======================== 29 passed in 0.05s ========================
```

---

## 示例2：fixture 基础

### 被测试代码

```python
# app/user.py
"""用户模型"""
from dataclasses import dataclass
from datetime import datetime

@dataclass
class User:
    """用户类"""
    username: str
    email: str
    created_at: datetime | None = None
    is_active: bool = True

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    def deactivate(self):
        """停用用户"""
        self.is_active = False

    def activate(self):
        """激活用户"""
        self.is_active = True

    def get_display_name(self) -> str:
        """获取显示名称"""
        return f"{self.username} <{self.email}>"
```

### 测试代码（使用 fixture）

```python
# tests/test_user.py
"""用户模型测试"""
import pytest
from datetime import datetime
from app.user import User


@pytest.fixture
def sample_user():
    """创建示例用户"""
    return User(
        username="alice",
        email="alice@example.com"
    )


@pytest.fixture
def inactive_user():
    """创建停用用户"""
    user = User(
        username="bob",
        email="bob@example.com"
    )
    user.deactivate()
    return user


@pytest.fixture
def user_with_custom_time():
    """创建指定时间的用户"""
    return User(
        username="charlie",
        email="charlie@example.com",
        created_at=datetime(2024, 1, 1, 12, 0, 0)
    )


class TestUserCreation:
    """用户创建测试"""

    def test_create_user_basic(self, sample_user):
        """测试：创建基本用户"""
        assert sample_user.username == "alice"
        assert sample_user.email == "alice@example.com"
        assert sample_user.is_active is True

    def test_create_user_with_defaults(self):
        """测试：使用默认值创建用户"""
        user = User(username="test", email="test@example.com")
        assert user.created_at is not None
        assert user.is_active is True

    def test_create_user_with_custom_time(self, user_with_custom_time):
        """测试：使用自定义时间创建用户"""
        assert user_with_custom_time.created_at == datetime(2024, 1, 1, 12, 0, 0)


class TestUserMethods:
    """用户方法测试"""

    def test_deactivate_user(self, sample_user):
        """测试：停用用户"""
        sample_user.deactivate()
        assert sample_user.is_active is False

    def test_activate_user(self, inactive_user):
        """测试：激活用户"""
        assert inactive_user.is_active is False
        inactive_user.activate()
        assert inactive_user.is_active is True

    def test_get_display_name(self, sample_user):
        """测试：获取显示名称"""
        display_name = sample_user.get_display_name()
        assert display_name == "alice <alice@example.com>"


class TestUserFixtures:
    """测试 fixture 的独立性"""

    def test_fixture_independence_1(self, sample_user):
        """测试：fixture 独立性（测试1）"""
        sample_user.username = "modified"
        assert sample_user.username == "modified"

    def test_fixture_independence_2(self, sample_user):
        """测试：fixture 独立性（测试2）"""
        # 每个测试都获得新的 fixture 实例
        assert sample_user.username == "alice"
```

**运行测试**：

```bash
pytest tests/test_user.py -v
```

---

## 示例3：fixture 作用域

```python
# tests/conftest.py
"""共享 fixture 配置"""
import pytest
from datetime import datetime


@pytest.fixture(scope="function")
def function_scoped_user():
    """函数作用域 fixture（每个测试函数创建一次）"""
    print("\n创建 function_scoped_user")
    user = {"username": "function_user", "count": 0}
    yield user
    print("\n清理 function_scoped_user")


@pytest.fixture(scope="class")
def class_scoped_user():
    """类作用域 fixture（每个测试类创建一次）"""
    print("\n创建 class_scoped_user")
    user = {"username": "class_user", "count": 0}
    yield user
    print("\n清理 class_scoped_user")


@pytest.fixture(scope="module")
def module_scoped_user():
    """模块作用域 fixture（每个测试模块创建一次）"""
    print("\n创建 module_scoped_user")
    user = {"username": "module_user", "count": 0}
    yield user
    print("\n清理 module_scoped_user")


@pytest.fixture(scope="session")
def session_scoped_config():
    """会话作用域 fixture（整个测试会话创建一次）"""
    print("\n创建 session_scoped_config")
    config = {"api_url": "http://test.com", "timeout": 30}
    yield config
    print("\n清理 session_scoped_config")


# tests/test_fixture_scope.py
"""fixture 作用域测试"""
import pytest


class TestFunctionScope:
    """函数作用域测试"""

    def test_function_1(self, function_scoped_user):
        """测试1"""
        function_scoped_user["count"] += 1
        assert function_scoped_user["count"] == 1

    def test_function_2(self, function_scoped_user):
        """测试2"""
        function_scoped_user["count"] += 1
        # 每个测试都获得新的 fixture
        assert function_scoped_user["count"] == 1


class TestClassScope:
    """类作用域测试"""

    def test_class_1(self, class_scoped_user):
        """测试1"""
        class_scoped_user["count"] += 1
        assert class_scoped_user["count"] == 1

    def test_class_2(self, class_scoped_user):
        """测试2"""
        class_scoped_user["count"] += 1
        # 同一个类中的测试共享 fixture
        assert class_scoped_user["count"] == 2


def test_module_1(module_scoped_user):
    """模块测试1"""
    module_scoped_user["count"] += 1
    assert module_scoped_user["count"] == 1


def test_module_2(module_scoped_user):
    """模块测试2"""
    module_scoped_user["count"] += 1
    # 同一个模块中的测试共享 fixture
    assert module_scoped_user["count"] == 2


def test_session_config(session_scoped_config):
    """会话配置测试"""
    assert session_scoped_config["api_url"] == "http://test.com"
    assert session_scoped_config["timeout"] == 30
```

**运行测试（显示 print 输出）**：

```bash
pytest tests/test_fixture_scope.py -v -s
```

---

## 示例4：fixture 依赖

```python
# tests/conftest.py
"""fixture 依赖示例"""
import pytest
from datetime import datetime


@pytest.fixture
def timestamp():
    """时间戳 fixture"""
    return datetime.now()


@pytest.fixture
def user_data(timestamp):
    """用户数据 fixture（依赖 timestamp）"""
    return {
        "username": "alice",
        "email": "alice@example.com",
        "created_at": timestamp
    }


@pytest.fixture
def user_with_profile(user_data):
    """带个人资料的用户 fixture（依赖 user_data）"""
    return {
        **user_data,
        "profile": {
            "bio": "Hello, I'm Alice",
            "avatar": "avatar.jpg"
        }
    }


# tests/test_fixture_dependency.py
"""fixture 依赖测试"""


def test_timestamp(timestamp):
    """测试：时间戳"""
    assert isinstance(timestamp, datetime)


def test_user_data(user_data):
    """测试：用户数据"""
    assert user_data["username"] == "alice"
    assert "created_at" in user_data


def test_user_with_profile(user_with_profile):
    """测试：带个人资料的用户"""
    assert user_with_profile["username"] == "alice"
    assert "profile" in user_with_profile
    assert user_with_profile["profile"]["bio"] == "Hello, I'm Alice"
```

---

## 示例5：fixture 参数化

```python
# tests/conftest.py
"""fixture 参数化示例"""
import pytest


@pytest.fixture(params=["alice", "bob", "charlie"])
def username(request):
    """参数化用户名 fixture"""
    return request.param


@pytest.fixture(params=[
    {"username": "alice", "email": "alice@example.com"},
    {"username": "bob", "email": "bob@example.com"},
    {"username": "charlie", "email": "charlie@example.com"},
])
def user_dict(request):
    """参数化用户字典 fixture"""
    return request.param


# tests/test_fixture_parametrize.py
"""fixture 参数化测试"""


def test_username_length(username):
    """测试：用户名长度"""
    # 这个测试会运行3次，每次使用不同的用户名
    assert len(username) >= 3


def test_user_dict_structure(user_dict):
    """测试：用户字典结构"""
    # 这个测试会运行3次，每次使用不同的用户字典
    assert "username" in user_dict
    assert "email" in user_dict
    assert "@" in user_dict["email"]
```

**运行测试**：

```bash
pytest tests/test_fixture_parametrize.py -v
```

**输出**：

```
tests/test_fixture_parametrize.py::test_username_length[alice] PASSED
tests/test_fixture_parametrize.py::test_username_length[bob] PASSED
tests/test_fixture_parametrize.py::test_username_length[charlie] PASSED
tests/test_fixture_parametrize.py::test_user_dict_structure[user_dict0] PASSED
tests/test_fixture_parametrize.py::test_user_dict_structure[user_dict1] PASSED
tests/test_fixture_parametrize.py::test_user_dict_structure[user_dict2] PASSED
```

---

## 示例6：fixture 清理（yield）

```python
# tests/conftest.py
"""fixture 清理示例"""
import pytest
import tempfile
import os


@pytest.fixture
def temp_file():
    """临时文件 fixture"""
    # Setup: 创建临时文件
    fd, path = tempfile.mkstemp()
    print(f"\n创建临时文件: {path}")

    # 写入测试数据
    with open(path, "w") as f:
        f.write("test data")

    yield path

    # Teardown: 清理临时文件
    os.close(fd)
    os.remove(path)
    print(f"\n删除临时文件: {path}")


@pytest.fixture
def temp_dir():
    """临时目录 fixture"""
    # Setup: 创建临时目录
    path = tempfile.mkdtemp()
    print(f"\n创建临时目录: {path}")

    yield path

    # Teardown: 清理临时目录
    import shutil
    shutil.rmtree(path)
    print(f"\n删除临时目录: {path}")


# tests/test_fixture_cleanup.py
"""fixture 清理测试"""
import os


def test_temp_file_exists(temp_file):
    """测试：临时文件存在"""
    assert os.path.exists(temp_file)

    # 读取文件内容
    with open(temp_file, "r") as f:
        content = f.read()
    assert content == "test data"


def test_temp_file_write(temp_file):
    """测试：写入临时文件"""
    # 追加内容
    with open(temp_file, "a") as f:
        f.write("\nmore data")

    # 验证内容
    with open(temp_file, "r") as f:
        content = f.read()
    assert "more data" in content


def test_temp_dir_exists(temp_dir):
    """测试：临时目录存在"""
    assert os.path.exists(temp_dir)
    assert os.path.isdir(temp_dir)

    # 在临时目录中创建文件
    file_path = os.path.join(temp_dir, "test.txt")
    with open(file_path, "w") as f:
        f.write("test")

    assert os.path.exists(file_path)
```

**运行测试（显示 print 输出）**：

```bash
pytest tests/test_fixture_cleanup.py -v -s
```

---

## 示例7：autouse fixture

```python
# tests/conftest.py
"""autouse fixture 示例"""
import pytest
import time


@pytest.fixture(autouse=True)
def log_test_time():
    """自动记录测试时间"""
    start = time.time()
    print(f"\n测试开始")

    yield

    end = time.time()
    duration = end - start
    print(f"\n测试耗时: {duration:.4f}秒")


@pytest.fixture(autouse=True, scope="session")
def setup_test_environment():
    """自动设置测试环境"""
    print("\n=== 测试环境初始化 ===")
    # 设置环境变量、初始化配置等

    yield

    print("\n=== 测试环境清理 ===")
    # 清理环境


# tests/test_autouse.py
"""autouse fixture 测试"""
import time


def test_fast_operation():
    """测试：快速操作"""
    result = 1 + 1
    assert result == 2


def test_slow_operation():
    """测试：慢速操作"""
    time.sleep(0.1)
    result = 2 * 2
    assert result == 4
```

**运行测试**：

```bash
pytest tests/test_autouse.py -v -s
```

---

## 示例8：fixture 工厂模式

```python
# tests/conftest.py
"""fixture 工厂模式示例"""
import pytest
from app.user import User


@pytest.fixture
def user_factory():
    """用户工厂 fixture"""
    created_users = []

    def _create_user(username: str, email: str | None = None):
        """创建用户"""
        if email is None:
            email = f"{username}@example.com"
        user = User(username=username, email=email)
        created_users.append(user)
        return user

    yield _create_user

    # 清理所有创建的用户
    print(f"\n清理 {len(created_users)} 个用户")


# tests/test_user_factory.py
"""用户工厂测试"""


def test_create_single_user(user_factory):
    """测试：创建单个用户"""
    user = user_factory("alice")
    assert user.username == "alice"
    assert user.email == "alice@example.com"


def test_create_multiple_users(user_factory):
    """测试：创建多个用户"""
    users = [
        user_factory("alice"),
        user_factory("bob"),
        user_factory("charlie")
    ]

    assert len(users) == 3
    assert users[0].username == "alice"
    assert users[1].username == "bob"
    assert users[2].username == "charlie"


def test_create_user_with_custom_email(user_factory):
    """测试：创建自定义邮箱的用户"""
    user = user_factory("alice", "custom@example.com")
    assert user.email == "custom@example.com"
```

---

## pytest.ini 配置

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

# 自定义标记
markers =
    slow: marks tests as slow
    fast: marks tests as fast
    unit: marks tests as unit tests
```

---

## 运行测试的常用命令

```bash
# 运行所有测试
pytest

# 运行指定文件
pytest tests/test_math_utils.py

# 运行指定测试类
pytest tests/test_user.py::TestUserCreation

# 运行指定测试函数
pytest tests/test_user.py::TestUserCreation::test_create_user_basic

# 详细输出
pytest -v

# 显示 print 输出
pytest -s

# 失败时停止
pytest -x

# 只运行失败的测试
pytest --lf

# 显示最慢的10个测试
pytest --durations=10

# 并行运行（需要 pytest-xdist）
pytest -n auto
```

---

## 总结

### 核心要点

1. **基础测试**：使用 `assert` 断言，测试函数以 `test_` 开头
2. **fixture**：使用 `@pytest.fixture` 准备测试数据
3. **作用域**：`function`、`class`、`module`、`session`
4. **依赖**：fixture 可以依赖其他 fixture
5. **参数化**：使用 `params` 参数化 fixture
6. **清理**：使用 `yield` 语法清理资源
7. **autouse**：自动应用的 fixture
8. **工厂模式**：返回函数的 fixture

### 最佳实践

- 将共享 fixture 放在 `conftest.py`
- 使用合适的作用域避免重复创建
- 使用 `yield` 确保资源清理
- 使用工厂模式创建多个实例
- 使用参数化减少重复代码
