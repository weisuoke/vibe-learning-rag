# 核心概念04：Field默认值与验证

> 状态字段的默认值设置和数据验证

## 概念定义

**Field 是 Pydantic 提供的字段配置工具,用于设置默认值、验证规则和元数据,在 LangGraph 中用于增强状态定义的健壮性。**

## TypedDict 的限制

### 1. 无默认值支持

```python
from typing import TypedDict

class State(TypedDict):
    messages: list
    count: int
    # ❌ TypedDict 不支持默认值
    # count: int = 0  # SyntaxError
```

**解决方案**：在初始化时提供默认值

```python
# 方式1：invoke 时提供
result = graph.invoke({
    "messages": [],
    "count": 0
})

# 方式2：节点内部处理
def node(state: State) -> dict:
    count = state.get("count", 0)  # 提供默认值
    return {"count": count + 1}
```

### 2. 无运行时验证

```python
class State(TypedDict):
    count: int

# TypedDict 不会验证类型
state = {"count": "not an int"}  # ✓ 运行时不报错
# 只有 mypy 会检查
```

## Pydantic Field 基础

### 1. 基本用法

```python
from pydantic import BaseModel, Field

class State(BaseModel):
    # 默认值
    count: int = 0
    name: str = "default"

    # 使用 Field 设置默认值
    messages: list = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
```

### 2. Field 参数

```python
class State(BaseModel):
    # 默认值
    count: int = Field(default=0)

    # 默认工厂函数
    messages: list = Field(default_factory=list)

    # 描述
    user_id: str = Field(description="User identifier")

    # 验证规则
    age: int = Field(ge=0, le=150)  # 0 <= age <= 150
    email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")

    # 别名
    user_name: str = Field(alias="userName")
```

### 3. 常用验证规则

```python
from pydantic import BaseModel, Field

class State(BaseModel):
    # 数值验证
    count: int = Field(ge=0)  # >= 0
    score: float = Field(gt=0, lt=100)  # 0 < score < 100

    # 字符串验证
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")

    # 列表验证
    tags: list = Field(min_length=1, max_length=10)

    # 自定义验证
    password: str = Field(min_length=8)
```

## 默认值策略

### 1. 简单默认值

```python
from pydantic import BaseModel, Field

class State(BaseModel):
    # 直接赋值
    count: int = 0
    status: str = "pending"
    is_active: bool = True

    # 使用 Field
    retry_count: int = Field(default=0)
```

### 2. 工厂函数默认值

```python
from pydantic import BaseModel, Field
from typing import Annotated
import operator
from datetime import datetime

class State(BaseModel):
    # 可变类型必须使用 default_factory
    messages: Annotated[list, operator.add] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    tags: set = Field(default_factory=set)

    # 动态默认值
    created_at: float = Field(default_factory=lambda: datetime.now().timestamp())

    # 自定义工厂函数
    config: dict = Field(default_factory=lambda: {"timeout": 30, "retries": 3})
```

**为什么使用 default_factory**：

```python
# ❌ 错误：所有实例共享同一个列表
class State(BaseModel):
    messages: list = []

# ✓ 正确：每个实例有独立的列表
class State(BaseModel):
    messages: list = Field(default_factory=list)
```

### 3. Optional 字段

```python
from typing import Optional
from pydantic import BaseModel, Field

class State(BaseModel):
    # Optional 字段（可以是 None）
    user_id: Optional[str] = None
    error: Optional[str] = None

    # 使用 Field
    result: Optional[dict] = Field(default=None)
```

### 4. NotRequired 字段（TypedDict）

```python
from typing import TypedDict
from typing_extensions import NotRequired

class State(TypedDict):
    # 必需字段
    user_id: str

    # 可选字段
    count: NotRequired[int]
    metadata: NotRequired[dict]
```

## 数据验证

### 1. 内置验证器

```python
from pydantic import BaseModel, Field, validator

class State(BaseModel):
    count: int = Field(ge=0, description="Must be non-negative")
    email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    age: int = Field(ge=0, le=150)

    # 自动验证
    # state = State(count=-1)  # ValidationError
```

### 2. 自定义验证器

```python
from pydantic import BaseModel, Field, validator

class State(BaseModel):
    messages: list = Field(default_factory=list)
    user_id: str

    @validator("messages")
    def validate_messages(cls, v):
        """验证消息格式"""
        if not isinstance(v, list):
            raise ValueError("messages must be a list")
        for msg in v:
            if not isinstance(msg, dict):
                raise ValueError("each message must be a dict")
            if "role" not in msg or "content" not in msg:
                raise ValueError("message must have 'role' and 'content'")
        return v

    @validator("user_id")
    def validate_user_id(cls, v):
        """验证用户ID格式"""
        if not v.startswith("user_"):
            raise ValueError("user_id must start with 'user_'")
        return v
```

### 3. 根验证器

```python
from pydantic import BaseModel, Field, root_validator

class State(BaseModel):
    start_time: float
    end_time: float

    @root_validator
    def validate_time_range(cls, values):
        """验证时间范围"""
        start = values.get("start_time")
        end = values.get("end_time")
        if start and end and start >= end:
            raise ValueError("start_time must be before end_time")
        return values
```

### 4. 字段依赖验证

```python
from pydantic import BaseModel, Field, validator

class State(BaseModel):
    status: str
    error: Optional[str] = None

    @validator("error", always=True)
    def validate_error(cls, v, values):
        """当状态为 failed 时，error 必须存在"""
        status = values.get("status")
        if status == "failed" and not v:
            raise ValueError("error is required when status is 'failed'")
        return v
```

## 在 LangGraph 中的应用

### 1. 聊天机器人状态

```python
from pydantic import BaseModel, Field, validator
from typing import Annotated, Optional
import operator
from datetime import datetime

class ChatState(BaseModel):
    # 必需字段
    user_id: str = Field(description="User identifier")

    # 带默认值的字段
    messages: Annotated[list, operator.add] = Field(
        default_factory=list,
        description="Conversation history"
    )

    # Optional 字段
    context: Optional[str] = Field(
        default=None,
        description="Current conversation context"
    )

    # 带验证的字段
    max_tokens: int = Field(
        default=1000,
        ge=1,
        le=4000,
        description="Maximum tokens for response"
    )

    # 动态默认值
    created_at: float = Field(
        default_factory=lambda: datetime.now().timestamp()
    )

    @validator("messages")
    def validate_messages(cls, v):
        """验证消息格式"""
        for msg in v:
            if not isinstance(msg, dict):
                raise ValueError("each message must be a dict")
            if "role" not in msg or "content" not in msg:
                raise ValueError("message must have 'role' and 'content'")
        return v
```

### 2. 工作流状态

```python
from pydantic import BaseModel, Field, validator
from typing import Annotated, Optional, Literal
import operator

class WorkflowState(BaseModel):
    # 任务ID
    task_id: str = Field(description="Task identifier")

    # 状态（限定值）
    status: Literal["pending", "running", "completed", "failed"] = Field(
        default="pending"
    )

    # 步骤列表
    steps: Annotated[list, operator.add] = Field(
        default_factory=list,
        description="Execution steps"
    )

    # 错误列表
    errors: Annotated[list, operator.add] = Field(
        default_factory=list
    )

    # 重试次数
    retry_count: Annotated[int, operator.add] = Field(
        default=0,
        ge=0,
        le=5,
        description="Number of retries"
    )

    # 结果
    result: Optional[dict] = Field(default=None)

    @validator("retry_count")
    def validate_retry_count(cls, v):
        """限制重试次数"""
        if v > 5:
            raise ValueError("Maximum 5 retries allowed")
        return v
```

### 3. 数据处理管道状态

```python
from pydantic import BaseModel, Field, validator
from typing import Annotated, Optional
import operator

class PipelineState(BaseModel):
    # 管道ID
    pipeline_id: str

    # 原始数据
    raw_data: list = Field(default_factory=list)

    # 处理后的数据
    processed_data: Annotated[list, operator.add] = Field(
        default_factory=list
    )

    # 统计信息
    stats: Annotated[dict, operator.or_] = Field(
        default_factory=dict
    )

    # 当前阶段
    current_stage: str = Field(default="init")

    # 配置
    config: dict = Field(
        default_factory=lambda: {
            "batch_size": 100,
            "timeout": 30
        }
    )

    @validator("raw_data")
    def validate_raw_data(cls, v):
        """验证原始数据不为空"""
        if not v:
            raise ValueError("raw_data cannot be empty")
        return v

    @validator("config")
    def validate_config(cls, v):
        """验证配置"""
        if "batch_size" not in v:
            raise ValueError("config must have 'batch_size'")
        if v["batch_size"] <= 0:
            raise ValueError("batch_size must be positive")
        return v
```

## TypedDict vs Pydantic 选择

### 对比表

| 特性 | TypedDict | Pydantic |
|------|-----------|----------|
| 默认值 | ❌ 不支持 | ✓ 支持 |
| 运行时验证 | ❌ 无 | ✓ 自动验证 |
| 性能 | ✓ 更快 | 稍慢 |
| 依赖 | ✓ 无需依赖 | 需要 pydantic |
| 类型检查 | ✓ mypy | ✓ mypy + 运行时 |
| 复杂验证 | ❌ 不支持 | ✓ 支持 |
| 序列化 | 手动 | ✓ 自动 |

### 选择建议

**使用 TypedDict**：
- 简单状态，无需默认值
- 性能敏感场景
- 不需要运行时验证
- 希望轻量级解决方案

**使用 Pydantic**：
- 需要默认值
- 需要数据验证
- 处理用户输入
- 复杂业务逻辑
- 需要序列化/反序列化

## 常见模式

### 1. 混合使用

```python
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
import operator

# 简单状态用 TypedDict
class SimpleState(TypedDict):
    messages: Annotated[list, operator.add]
    user_id: str

# 复杂状态用 Pydantic
class ComplexState(BaseModel):
    messages: Annotated[list, operator.add] = Field(default_factory=list)
    user_id: str = Field(pattern=r"^user_\d+$")
    config: dict = Field(default_factory=lambda: {"timeout": 30})
```

### 2. 渐进式验证

```python
from pydantic import BaseModel, Field, validator

class State(BaseModel):
    # 开发阶段：宽松验证
    data: list = Field(default_factory=list)

    # 生产阶段：严格验证
    @validator("data")
    def validate_data(cls, v):
        if not v:
            raise ValueError("data cannot be empty")
        return v
```

### 3. 条件默认值

```python
from pydantic import BaseModel, Field
import os

class State(BaseModel):
    # 从环境变量读取默认值
    api_key: str = Field(
        default_factory=lambda: os.getenv("API_KEY", "")
    )

    # 根据环境设置默认值
    debug: bool = Field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )
```

## 最佳实践

### 1. 明确默认值语义

```python
# ✓ 清晰的默认值
class State(BaseModel):
    retry_count: int = Field(default=0, description="Number of retries")
    max_retries: int = Field(default=3, description="Maximum retries allowed")

# ❌ 模糊的默认值
class State(BaseModel):
    count: int = 0  # 什么的计数？
```

### 2. 使用工厂函数避免共享

```python
# ✓ 正确：每个实例独立
class State(BaseModel):
    messages: list = Field(default_factory=list)

# ❌ 错误：所有实例共享
class State(BaseModel):
    messages: list = []
```

### 3. 验证器保持简单

```python
# ✓ 简单清晰
@validator("email")
def validate_email(cls, v):
    if "@" not in v:
        raise ValueError("invalid email")
    return v

# ❌ 过于复杂
@validator("email")
def validate_email(cls, v):
    # 100 行复杂验证逻辑...
    pass
```

### 4. 文档化验证规则

```python
class State(BaseModel):
    age: int = Field(
        ge=0,
        le=150,
        description="User age (0-150)"
    )

    email: str = Field(
        pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$",
        description="Valid email address"
    )
```

## 常见错误

### 1. 可变默认值

```python
# ❌ 错误
class State(BaseModel):
    messages: list = []

# ✓ 正确
class State(BaseModel):
    messages: list = Field(default_factory=list)
```

### 2. 忘记验证

```python
# ❌ 缺少验证
class State(BaseModel):
    count: int

# ✓ 添加验证
class State(BaseModel):
    count: int = Field(ge=0)
```

### 3. 过度验证

```python
# ❌ 过度验证，影响性能
class State(BaseModel):
    data: list

    @validator("data")
    def validate_data(cls, v):
        # 复杂的验证逻辑
        for item in v:
            # 深度验证每个元素...
        return v

# ✓ 适度验证
class State(BaseModel):
    data: list = Field(min_length=1)
```

## 总结

**Field 提供了强大的默认值和验证功能**：
- 默认值：简单值、工厂函数、动态值
- 验证：内置规则、自定义验证器、根验证器
- 选择：TypedDict（简单）vs Pydantic（复杂）

**最佳实践**：
- 明确默认值语义
- 使用工厂函数避免共享
- 验证器保持简单
- 文档化验证规则

## 参考资料

- TypedDict 基础：`03_核心概念_01_TypedDict状态定义.md`
- 实战应用：`07_实战代码_场景5_Optional字段与默认值.md`
- 生产实践：`03_核心概念_10_生产环境最佳实践.md`
